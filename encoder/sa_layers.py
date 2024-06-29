import math
import torch
from torch import nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class LocalLevelEncoding(nn.Module):
    """
        Compute Local-level Encoding for each entity in the graph.
        This module includes two feature extractors, which can be specified by the parameter extractor_type
        1. local-degree extractor
        2. local-relation extractor
    """

    def __init__(
            self, num_heads, num_entities, num_in_degree, num_out_degree, num_edges, hidden_dim, n_layers, dropout,
            extractor_type='relation'
    ):
        super(LocalLevelEncoding, self).__init__()
        self.num_heads = num_heads
        self.num_entities = num_entities

        # 1 for graph token
        self.entity_encoder = nn.Embedding(num_entities + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.relation_encoder = nn.Embedding(num_edges, hidden_dim)
        self.relation_ffn = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.extractor_type = extractor_type
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, in_degree, out_degree, link, length):

        n_graph, n_entity = x.size()[:2]

        # entity feature + graph token
        entity_feature = self.entity_encoder(x).sum(dim=-2)  # [n_graph, n_entity, n_hidden]

        # feature extractor
        if self.extractor_type == 'relation':
            relation_feature = torch.matmul(torch.matmul(link, self.relation_encoder.weight), self.relation_ffn.weight)
            entity_feature = entity_feature + relation_feature.sum(dim=-2)
            entity_feature = self.layer_norm(entity_feature)
        elif self.extractor_type == 'degree':
            entity_feature = (
                    entity_feature
                    + self.in_degree_encoder(in_degree)
                    + self.out_degree_encoder(out_degree)
            )
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_entity_feature = torch.cat([graph_token_feature, entity_feature], dim=1)
        return graph_entity_feature


class GlobalLevelEncoding(nn.Module):
    """
        Compute Global-level Encoding for each entity in the graph.
        This module includes structural encoding and relational encoding.
    """

    def __init__(
        self,
        num_heads,
        num_entities,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
        edge_padding_idx = 0
    ):
        super(GlobalLevelEncoding, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=edge_padding_idx)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.structural_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, attn_bias, spatial_pos, edge_input):

        n_graph, n_entity = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_entity+1, n_entity+1]

        # structural encoding
        # [n_graph, n_entity, n_entity, n_head] -> [n_graph, n_head, n_entity, n_entity]
        spatial_pos_bias = self.structural_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset structural pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_entity, n_entity, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias