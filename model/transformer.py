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
	def __init__(self,
	             encoder,
	             decoder,
	             device,
	             mask_idx=0):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.padding_index = mask_idx
		self.device = device

	def make_input_mask(self, input):
		input_mask = (input != self.padding_index).unsqueeze(1).unsqueeze(2)
		return input_mask

	def make_target_mask(self, target):
		target_pad_mask = (target != self.padding_index).unsqueeze(1).unsqueeze(2)
		# Lower triangular matrix to mask out future inputs
		target_tril_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]), device=self.device)).bool()

		target_mask = target_pad_mask & target_tril_mask
		return target_mask

	def forward(self, src, tgt):
		# 1. Run Encoder
		src_mask = self.make_input_mask(src)
		encoded_input = self.encoder(src, src_mask)

		# 2. Run Decoder
		target_mask = self.make_target_mask(tgt)
		output, attention = self.decoder(tgt, encoded_input, target_mask, src_mask)
		return output, attention