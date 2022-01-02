import torch
from torch import nn

# ########################################################################
# # POS EMBEDDING - Using sin/cos
# Input : Sentence length, Word embedding size, Device (CPU/GPU?)
# ########################################################################
class PositionEncoding(nn.Module):
	def __init__(self):
		super(PositionEncoding, self).__init__()

	def forward(self, seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")):
		pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
		dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
		phase = pos / 1e4 ** (dim // dim_model)

		return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))