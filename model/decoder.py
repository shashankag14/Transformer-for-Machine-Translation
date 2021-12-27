"""
@author : Shashank Agarwal
@when : 03-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from model.position_embedding import PositionEmbedding
from model.attention import MultiHeadAttention

# ########################################################################
# # ENCODER LAYER :
# 1. Performs Multiheaded self attention on Q,K,V from Target
# 2. Residual + Layer norm
# 3. Performs Multiheaded Enc-Dec attention on Q from target and K,V from Encoder
# 4. Residual + Layer norm
# 5. FFN
# 6. Residual + Layer norm
# ########################################################################
class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads

        self.attention1 = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        self.attention2 = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask, src_mask) -> Tensor:
        # 1. Compute self attention
        attention_1 = self.attention1(tgt, tgt, tgt, tgt_mask)
        # 2. Add and norm
        query = self.dropout1(self.norm1(attention_1 + tgt))

        # 3. Encoder-decoder self attention
        attention_2 = self.attention2(memory, memory, query, src_mask)
        # 4. Add and norm
        x = self.dropout2(self.norm2(attention_2 + query))

        # 5. FFN
        forward = self.feed_forward(x)
        # 6. Add and norm
        out = self.dropout3(self.norm3(forward + x))
        return out

# ########################################################################
# # DECODER BLOCK :
# 1. Performs word+pos embedding
# 2. Runs Decoder layers sequentially
# 3. Performs FFN + Softmax @ the end
# ########################################################################
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        device,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.dim_model = dim_model

        self.word_embedding = nn.Embedding(tgt_vocab_size, dim_model)
        self.position_embedding = PositionEmbedding()
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, tgt_vocab_size)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask, src_mask) -> Tensor:
        seq_len = tgt.size(1)

        word_emb = self.word_embedding(tgt.to(torch.long))
        pos_emb = self.position_embedding(seq_len, self.dim_model, self.device)
        tgt = self.embedding_dropout(pos_emb + word_emb)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, src_mask)

        # Softmax skipped : Using Cross entropy loss wherein softmax is already included
        return self.linear(tgt)
