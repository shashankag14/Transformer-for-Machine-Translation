"""
@author : Shashank Agarwal
@when : 03-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from model.position_encoding import PositionEncoding
from model.attention import MultiHeadAttention

# ########################################################################
# # DECODER LAYER :
# 1. Performs Multiheaded self attention on Q,K,V from Target
# 2. Residual + Layer norm
# 3. Performs Multiheaded Enc-Dec attention on Q from target and K,V from Encoder
# 4. Residual + Layer norm
# 5. FFN
# 6. Residual + Layer norm
# ########################################################################
class TransformerDecoderLayer(nn.Module):
	def __init__(self,
	             dim_model : int = 512,
	             num_heads : int = 8,
	             dim_feedforward : int = 2048,
	             dropout : float = 0.1,
	             device : str = 'cpu'
	             ):
		super().__init__()

		self.attention1 = MultiHeadAttention(dim_model, num_heads, dropout, device)
		self.norm1 = nn.LayerNorm(dim_model)

		self.attention2 = MultiHeadAttention(dim_model, num_heads, dropout, device)
		self.norm2 = nn.LayerNorm(dim_model)

		self.feed_forward = FeedForwardLayer(dim_model, dim_feedforward, dropout)
		self.norm3 = nn.LayerNorm(dim_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, target, encoded_input, target_mask, input_mask):
		# 1. Compute self attention
		attention_1, _ = self.attention1(target, target, target, target_mask)
		# 2. Add and norm
		attention_1_norm = self.norm1(target + self.dropout(attention_1))
		# 3. Encoder-decoder self attention
		attention_2, attention = self.attention2(attention_1_norm, encoded_input, encoded_input, input_mask)
		# 4. Add and norm
		attention_2_norm = self.norm2(attention_1_norm + self.dropout(attention_2))
		# 5. FFN
		forward = self.feed_forward(attention_2_norm)
		# 6. Add and norm
		output = self.norm3(attention_2_norm + self.dropout(forward))

		return output, attention

# ########################################################################
# # DECODER BLOCK :
# 1. Performs word+pos embedding
# 2. Runs Decoder layers sequentially
# 3. Performs FFN 
# ########################################################################
class TransformerDecoder(nn.Module):
	def __init__(self,
	             tgt_vocab_size,
	             dim_model : int = 512,
	             num_layers : int = 6,
	             num_heads : int = 8,
	             dim_feedforward : int = 2048,
	             dropout : float = 0.1,
	             device : str = 'cpu',
	             MAX_LENGTH=100):
		super().__init__()
		self.device = device
		self.dim_model = dim_model

		self.word_embedding = nn.Embedding(tgt_vocab_size, dim_model)
		# Rescalling coefficient for word embedding
		self.coefficient = torch.sqrt(torch.FloatTensor([self.dim_model])).to(device)
		self.position_encoding = PositionEncoding()

		self.dropout = nn.Dropout(dropout)

		decoding_layers = []
		for _ in range(num_layers):
			decoding_layers.append(TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout, device))
		self.layers = nn.Sequential(*decoding_layers)

		# dim_model = 256
		self.linear = nn.Linear(dim_model, tgt_vocab_size)

	def forward(self, target, encoded_input, target_mask, input_mask):
		target_size = target.shape[1]
		target = self.dropout((self.word_embedding(target) * self.coefficient) + self.position_encoding(target_size, self.dim_model, self.device))

		for layer in self.layers:
			target, attention = layer(target, encoded_input, target_mask, input_mask)

		# Softmax skipped : Using Cross entropy loss wherein softmax is already included
		output = self.linear(target)
		return output, attention


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size)
        )

    def forward(self, input):
        output = self.ff_layer(input)
        return output