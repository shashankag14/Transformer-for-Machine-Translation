"""
@author : Shashank Agarwal
@when : 03-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as f

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask, n_head, head_dim) -> Tensor:
	N = query.shape[0]
	query_len = query.shape[1]

	# queries shape: (N, query_len, heads, heads_dim),
	# keys shape: (N, key_len, heads, heads_dim)
	# temp: (N, heads, query_len, key_len)
	temp = torch.einsum("nqhd,nkhd->nhqk", [query, key])

	if mask is not None:
		temp = temp.masked_fill(mask == 0, float("-1e20"))

	scale = (n_head*head_dim) ** 0.5
	softmax_out = f.softmax(temp / scale, dim=3)

	# softmax_out shape : (N, n_head, query_len, key_len)
	# value shape : (N, value_len, n_head, head_dim)
	# out shape : (N, query_len, n_head, head_dim)
	# out shape after reshaping : (N, query_len, n_head * head_dim)
	out = torch.einsum("nhqk, nvhd->nqhd", [softmax_out, value]).reshape(N, query_len, n_head * head_dim)
	return out

class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
		super().__init__()
		self.q = nn.Linear(dim_k, dim_k)
		self.k = nn.Linear(dim_k, dim_k)
		self.v = nn.Linear(dim_v, dim_v)
		self.num_heads = num_heads
		self.head_dim = dim_in // num_heads
		self.linear = nn.Linear(num_heads * dim_v, dim_in)

	def forward(self, query: Tensor, key: Tensor, value: Tensor, mask) -> Tensor:
		N = query.shape[0]
		query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

		# Shape of x : (N, x_len, heads, num_heads)
		query = query.reshape(N, query_len, self.num_heads, self.head_dim)
		key = key.reshape(N, key_len, self.num_heads, self.head_dim)
		value = value.reshape(N, value_len, self.num_heads, self.head_dim)

		query = self.q(query)
		key = self.k(key)
		value = self.v(value)

		attention = scaled_dot_product_attention(query, key, value, mask, self.num_heads, self.head_dim)
		out = self.linear(attention)
		return out