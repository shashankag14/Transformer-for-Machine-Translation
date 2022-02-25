"""
@author : Shashank Agarwal
@when : 03-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn

class MultiHeadAttention(nn.Module):
	def __init__(self, dim_model: int, num_heads: int, dropout: float, device):
		super().__init__()
		self.device = device
		self.dim_model = dim_model
		self.num_heads = num_heads

		self.q = nn.Linear(dim_model, dim_model)
		self.k = nn.Linear(dim_model, dim_model)
		self.v = nn.Linear(dim_model, dim_model)

		self.head_dim = dim_model // num_heads
		self.linear = nn.Linear(dim_model, dim_model)

		self.dropout = nn.Dropout(dropout)

	def scaled_dot_product_attn(self, q, k, v, mask):
		N = q.shape[0]

		# temp -> [N, num_heads, len_q, len_k]
		scale = (self.dim_model ** 0.5)
		temp = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale

		if mask is not None:
			temp = temp.masked_fill(mask == 0, -1e10)

		softmax_out = torch.softmax(temp, dim=-1)
		# matmul faster than einsum
		# Apply dropout on softmax and then multiply with value
		attention = torch.matmul(self.dropout(softmax_out), v)
		# 4d - [N, len_q, num_head, head_dim]
		attention = attention.permute(0, 2, 1, 3).contiguous()
		# 3d - [N, len_q, num_head*head_dim]
		attention = attention.view(N, -1, self.dim_model)
		return attention, softmax_out

	def forward(self, query, key, value, mask=None):
		N = query.shape[0]

		query = self.q(query)
		key = self.k(key)
		value = self.v(value)     

		# x (k/q/v) -> (N, num_heads, len_x, head_dim)
		query = query.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
		key = key.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
		value = value.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

		attention, softmax_out = self.scaled_dot_product_attn(query, key, value, mask)
		attention = self.linear(attention)

		return attention, softmax_out
