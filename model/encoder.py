import torch
from torch import Tensor
from torch import nn

from model.model_utils import position_encoding
from model.attention import MultiHeadAttention

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
        self.attention = MultiHeadAttention(num_heads, dim_model, dim_k, dim_v)
        self.norm = nn.LayerNorm(dim_model) # Layer norm instead of BatchNorm
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_model),
        )

    def forward(self, src: Tensor, mask) -> Tensor:
        # In encoder, Key, Query and Value are all the same.
        # Its in the decoder that these will change
        key = query = value = src

        # 1. Compute self attention
        attention = self.attention(src, src, src, mask)
        # Attention shape = Query shape  = (N, query_len, n_head, dim_in)

        # 2. Add and norm
        x = self.dropout(self.norm(attention + query))

        # 3. FFN
        forward = self.feed_forward(x)

        # 4. Add and norm
        out = self.dropout(self.norm(forward + x))
        return out


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
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.word_embedding = nn.Embedding(src_vocab_size, self.dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, mask) -> Tensor:
        seq_len = src.size(1)
        pos_emb = position_encoding(seq_len, self.dim_model, self.device)
        word_emb = self.word_embedding(src.to(torch.long))
        src = self.dropout(pos_emb + word_emb)
        for layer in self.layers:
            src = layer(src, mask)
        return src