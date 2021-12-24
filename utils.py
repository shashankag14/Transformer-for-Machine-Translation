import argparse
import os
import torch

# For monitoring epoch time
def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directories for local files
orig_src_data_path = "data/PHP.cs-en.cs"
orig_tgt_data_path = "data/PHP.cs-en.en"

src_data_path = "data/preprocessed_src.txt"
tgt_data_path = "data/preprocessed_trg.txt"

saved_chkpt = 'saved_chkpt/'
if not os.path.exists(saved_chkpt):
  os.mkdir(saved_chkpt)

results = 'results/'
if not os.path.exists(results):
  os.mkdir(results)

# Argument parser
# For data files
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--src_data', type=str, default=src_data_path,
                    help='location of the src data')
parser.add_argument('--tgt_data', type=str, default=tgt_data_path,
                    help='location of the tgt data')

# model parameter setting
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--d_model', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of enc/dec layers in each block')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of en/dec blocks')
parser.add_argument('--ffn_hidden', type=int, default=2048,
                    help='number of hidden units in FFN')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout probability')
parser.add_argument('--max_sent_len', type=int, default=10,
                    help='Maximum length of sentence to use for train/valid/test')

# optimizer parameter setting
parser.add_argument('--init_lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--scheduler_factor', type=float, default=0.9,
                    help='Factor with which LR will decreasing using scheduler (new_lr = old_lr * optim_factor)')
parser.add_argument('--optim_adam_eps', type=float, default=5e-9,
                    help='Adam epsilon')
parser.add_argument('--optim_patience', type=int, default=8,
                    help='Number of epochs optimizer will wait before decreasing LR')
parser.add_argument('--optim_warmup', type=int, default=4000,
                    help='Optimizer warmup')
parser.add_argument('--optim_weight_decay', type=int, default=5e-4,
                    help='Weight decay factor for optimizer')

parser.add_argument('--epoch', type=int, default=50,
                    help='Number of epochs to train')
parser.add_argument('--clip', type=float, default=1.0,
                    help='Gradient clipping')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--label_smooth_eps', type=float, default=0.1,
                    help='Hyper-parameter for label smoothening')

args = parser.parse_args()

# model parameter setting
batch_size      = args.batch_size
d_model         = args.d_model
n_layers        = args.n_layers
n_heads         = args.n_heads
ffn_hidden      = args.ffn_hidden
dropout         = args.dropout
max_sent_len    = args.max_sent_len

# optimizer parameter setting
init_lr = args.init_lr
factor = args.scheduler_factor
adam_eps = args.optim_adam_eps
patience = args.optim_patience
warmup = args.optim_warmup
epoch = args.epoch
clip = args.clip
weight_decay = args.optim_weight_decay
label_smooth_eps = args.label_smooth_eps
inf = float('inf')