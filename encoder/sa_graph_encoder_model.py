import torch
from fairseq import utils
from fairseq.modules import LayerNorm
from torch import nn

from encoder.sa_graph_encoder import StructureAwareGraphEncoder


class StructureAwareGraphEncoderModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.graph_encoder = StructureAwareGraphEncoder(
            num_entities=args.num_entities,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_init=args.apply_init,
            activation_fn=args.activation_fn,
            edge_padding_idx=args.edge_padding_idx
        )

    def forward(self, data_x, indeg, outdeg, spatial_pos, attn_bias, edge_input, link, length, perturb=None,
                masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            data_x, indeg, outdeg, spatial_pos, attn_bias, edge_input, link, length,
            perturb=perturb,
        )
        x = inner_states[-1].transpose(0, 1)
        return x[:, :-1, :]