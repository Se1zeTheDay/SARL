from torch import nn

from decoder.rule_decoder import RuleDecoder
from encoder.sa_graph_encoder_model import StructureAwareGraphEncoderModel


class StructureAwareGraphTransformer(nn.Module):
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj', data=None, kge_model=None, opt=None, nebor_relation=None):

        super().__init__()
        self.relationE = nn.Embedding(n_trg_vocab, d_model)
        self.relation_type_E = nn.Embedding(n_trg_vocab, d_model)
        self.entityE = nn.Embedding(n_src_vocab, d_model)
        self.n_trg_vocab = n_trg_vocab
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = StructureAwareGraphEncoderModel(opt)

        self.decoder = RuleDecoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb, relationE=self.relationE, data=data)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for n, p in self.named_parameters():
            if p.dim() > 1 and 'entityE' not in n and 'relationE' not in n:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq):
        raise NotImplementedError
