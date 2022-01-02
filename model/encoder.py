"""
@author : Shashank Agarwal
@when : 03-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn

from model.position_encoding import PositionEmbedding
from model.attention import MultiHeadAttention

# ########################################################################
# # ENCODER LAYER :
# 1. Performs Multiheaded self attention
# 2. Residual + Layer norm
# 3. Performs FFN
# 4. Residual + layer norm
# ########################################################################
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads # Input dim is split into num_heads

        # Multi headed attention
        self.attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.norm1 = nn.LayerNorm(dim_model) # Layer norm instead of BatchNorm
        self.dropout1 = nn.Dropout(dropout)

        # Feed Forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_model),
        )
        self.norm2 = nn.LayerNorm(dim_model)  # Layer norm instead of BatchNorm
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, mask) -> Tensor:
        # In encoder, Key, Query and Value are all the same.
        # In decoder these will change !!
        key = query = value = src

        # 1. Compute self attention
        attention = self.attention(src, src, src, mask)
        # Attention shape = Query shape  -> (N, query_len, n_head, dim_in)

        # 2. Add and norm
        x = self.dropout1(self.norm1(attention + query))

        # 3. FFN
        forward = self.feed_forward(x)

        # 4. Add and norm
        out = self.dropout2(self.norm2(forward + x))
        return out

# ########################################################################
# # ENCODER BLOCK :
# 1. Performs word+pos embedding
# 2. Runs Encoder layers sequentially
# ########################################################################
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
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

        self.word_embedding = nn.Embedding(src_vocab_size, self.dim_model)
        self.position_embedding = PositionEmbedding()
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, mask) -> Tensor:
        seq_len = src.size(1)

        pos_emb = self.position_embedding(seq_len, self.dim_model, self.device)
        word_emb = self.word_embedding(src.to(torch.long))
        src = self.embedding_dropout(pos_emb + word_emb)

        for layer in self.layers:
            src = layer(src, mask)
        return src