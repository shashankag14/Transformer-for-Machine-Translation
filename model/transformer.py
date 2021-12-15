"""
@author : Shashank Agarwal
@when : 04-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn

from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_mask_idx,
        tgt_mask_idx,
        device,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_mask_idx = src_mask_idx
        self.tgt_mask_idx = tgt_mask_idx
        self.device = device

        self.encoder = TransformerEncoder(
            src_vocab_size=src_vocab_size,
            device=device,
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size=tgt_vocab_size,
            device=device,
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(self.src_mask_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(self.src_mask_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    # Output : Lower triangular matrix to mask out future inputs (Only used in decoder)
    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # 1. Run Encoder
        src_mask = self.make_pad_mask(src, src)
        enc_src = self.encoder(src, src_mask)

        # 2. Run Decoder
        src_trg_mask = self.make_pad_mask(tgt, src)
        trg_mask = self.make_pad_mask(tgt, tgt) * self.make_no_peak_mask(tgt, tgt)
        out = self.decoder(tgt, enc_src, trg_mask, src_trg_mask)
        return out