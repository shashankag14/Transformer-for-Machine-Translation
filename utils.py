"""
@author : Shashank Agarwal
@when : 06-12-2021
@homepage : https://github.com/gusdnd852
"""

import torch
import os

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## arguement parser to be added
src_ext = "cs"
tgt_ext = "en"
src_data_path = "data/PHP.cs-en." + src_ext
tgt_data_path = "data/PHP.cs-en." + tgt_ext

saved_chkpt = 'saved_chkpt/'
if not os.path.exists(saved_chkpt):
  os.mkdir(saved_chkpt) 

results = 'results/'
if not os.path.exists(results):
  os.mkdir(results) 

# model parameter setting
batch_size = 128
max_len = 256 # Where is it being used??
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
dropout = 0.4

max_sent_len = 14

# optimizer parameter setting
init_lr = 5e-5
factor = 0.9
adam_eps = 5e-9
patience = 8
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs