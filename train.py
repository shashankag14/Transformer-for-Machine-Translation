"""
@author : Shashank Agarwal
@when : 06-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
import math
import time

from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F

# Local project files
import utils
from model.transformer import Transformer
from utils import *
import dictionary as dict
import tokenizer
import dataloader
from bleu_metric import *
from optim import ScheduledOptim


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# To empty the cache for TQDM
torch.cuda.empty_cache()

MODEL_SANITY_CHECK = 0
if MODEL_SANITY_CHECK:
	src = torch.rand(25, 16)  # batch_size, seq_length
	tgt = torch.rand(25, 16)  # batch_size, seq_length
	out = Transformer(207, 183, 0, 0, "cpu")(src, tgt)
	print("MODEL OUTPUT SHAPE : ", out.shape)# torch.Size([batch_size, max_sent_len, tgt_vocab_size])

print("Device being used : ", device)

# Count the number of trainable parameters in the model 
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
	if hasattr(m, 'weight') and m.weight.dim() > 1:
		nn.init.kaiming_uniform(m.weight.data)

################################################################################
## Initialisation of Dictionary, Dataset and Model
################################################################################
# Create dictionary
corpus = tokenizer.Corpus()
src_vocab_size = corpus.dictionary_src.n_word
tgt_vocab_size = corpus.dictionary_tgt.n_word
print("Source vocab size {}, Target vocab size {}".format(src_vocab_size, tgt_vocab_size))

# Create datasets
train_dataloader, valid_dataloader, test_dataloader = dataloader.get_dataloader(corpus.tokenize_src,
                                                                                corpus.tokenize_tgt)
# Create model instance
model = Transformer(src_vocab_size,
                    tgt_vocab_size,
                    src_mask_idx=dict.PAD_token,
                    tgt_mask_idx=dict.PAD_token,
                    device=device,
                    num_encoder_layers=utils.n_layers,
                    num_decoder_layers=utils.n_layers,
                    dim_model=utils.d_model,
                    num_heads=utils.n_heads,
                    dim_feedforward=utils.ffn_hidden,
                    dropout=utils.dropout, ).to(device)

print(f'Trainable Parameters : {count_parameters(model):,}')
model.apply(initialize_weights)

##### To load saved model and continue training
# model.load_state_dict(torch.load('saved_chkpt/best_model.pt'))

################################################################################
## Loss function & Optimizer 
################################################################################
# Optimizer
# optimizer = Adam(params=model.parameters(),
#                  lr=init_lr,
#                  weight_decay=weight_decay,
#                  eps=adam_eps)

# # LR Scheduler for better gradient descent 
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                                  verbose=True,
#                                                  factor=factor,
#                                                  patience=patience)

optimizer = ScheduledOptim(optim.Adam(params=model.parameters(),
							betas=(0.9, 0.98),
	                        eps=adam_eps),
							lr_mul, d_model, warmup)
# Loss function (Cross entropy)
criterion = nn.CrossEntropyLoss(ignore_index=dict.PAD_token)

################################################################################
## Training loop for each epoch
################################################################################
def train(model, iterator, optimizer, criterion, clip, epoch_num, label_smoothening=False):
	model.train()
	epoch_loss = 0
	with tqdm(iterator, unit="batches") as tepoch:
		for batch_num, batch in enumerate(iterator.batches):
			tepoch.set_description(f"Epoch {epoch_num}")

			src, trg = tokenizer.add_padding(batch)
			src, trg = src.to(device), trg.to(device)

			optimizer.zero_grad()
			# output -> (N, seq_len, tgt_vocab_size) ; # trg -> (N, seq_len)
			output = model(src, trg) 

			# removing the first token of <SOS> and then flattening 2D to 1D tensor
			output_reshape = output[:, 1:].contiguous().view(-1, output.shape[-1])
			trg = trg[:, 1:].contiguous().view(-1)

			##########################################################################
			## Label Smoothening (Regularization technique)
			##########################################################################
			# label smoothening only being used for training and not for validation
			if label_smoothening:
				# "Smoothed" one-hot vectors for the target sequences
				# (N*seq_len, trg_vocab_size)-> one-hot
				target_vector = torch.zeros_like(output_reshape).scatter(dim=1, index=trg.unsqueeze(1),value=1.).to(device)

				# (N*seq_len, trg_vocab_size)-> "smoothed" one-hot; (-2 to ignore <pad> and target label)
				target_vector = target_vector * (1. - label_smooth_eps) + label_smooth_eps / (target_vector.size(1)-2)

				# 1D tensor : (N*seq_len)-> Compute smoothed cross-entropy loss
				loss = (-1 * target_vector * F.log_softmax(output_reshape, dim=1)).sum(dim=1)
				# Create mask for locations of <PAD token in target (N*seq_len)
				non_pad_mask = trg.ne(dict.PAD_token) 

				# Ignore <PAD> while avging loss, thus this 1D tensor might be smaller 
				# than the previous 1D loss tensor
				loss = loss.masked_select(non_pad_mask) 
				# Final avg batch loss : Sum the non pad mask losses and divide by
				# number of non pad masks
				loss = loss.sum()/loss.size(0) 

			else :
				# flattening 2D to 1D tensor
				loss = criterion(output_reshape, trg)

			# Backprop
			loss.backward()

			# Clip gradients to avoid exploding gradient issues
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			# optimizer.step()
			optimizer.step_and_update_lr()
			epoch_loss += loss.item()
			tepoch.update()
			tepoch.set_postfix(loss=loss.item())

		# To empty the cache for TQDM
		torch.cuda.empty_cache()

	return epoch_loss / len(iterator)

################################################################################
## Validation loop for each epoch
################################################################################
def evaluate(model, iterator, criterion):
	model.eval()
	epoch_loss = 0
	batch_bleu = []
	with torch.no_grad():
		for batch_num, batch in enumerate(iterator.batches):
			src, trg = tokenizer.add_padding(batch)
			src, trg = src.to(device), trg.to(device)

			output = model(src, trg)
			output_reshape = output[:,1:].contiguous().view(-1, output.shape[-1])
			trg_reshape = trg[:, 1:].contiguous().view(-1)

			loss = criterion(output_reshape, trg_reshape)
			epoch_loss += loss.item()
      
            # Compute BLEU score per batch - corpus level or sentence level??
			total_bleu = [] 
			# Note : Size of last batch might not be equal to batch_size, thus trg.size(dim=0)
			for j in range(trg.size(dim=0)):
				trg_words = tokenizer.detokenize(trg[j].tolist(), corpus.dictionary_tgt)
				output_words = output[j].max(dim=1)[1]
				output_words = tokenizer.detokenize(output_words.tolist(), corpus.dictionary_tgt)
				bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
				total_bleu.append(bleu)
				     
			total_bleu = sum(total_bleu) / len(total_bleu)
			batch_bleu.append(total_bleu)

	batch_bleu = sum(batch_bleu) / len(batch_bleu)
	torch.cuda.empty_cache()
	return epoch_loss / len(iterator), batch_bleu

################################################################################
## Run epochs - train and validation in each epoch
################################################################################
def run(total_epoch, best_loss, best_epoch):
	train_losses, test_losses, bleus = [], [], []
	early_stop_counter = 0
	for step in range(total_epoch):
		start_time = time.time()

		# Create batches - needs to be called before each loop.
		train_dataloader.create_batches()
		train_loss = train(model, train_dataloader, optimizer, criterion, clip, step, label_smoothening = True)

		valid_dataloader.create_batches()
		valid_loss, bleu = evaluate(model, valid_dataloader, criterion)

		end_time = time.time()

		# if step > warmup:
		# 	scheduler.step(valid_loss)

		train_losses.append(train_loss)
		test_losses.append(valid_loss)
		bleus.append(bleu)
		epoch_mins, epoch_secs = compute_time(start_time, end_time)

		if valid_loss <= best_loss:
			best_loss = valid_loss
			torch.save(model.state_dict(), 'saved_chkpt/best_model.pt')
			best_epoch = step
		else:
			early_stop_counter+=1

		f = open('results/train_loss.txt', 'w')
		f.write(str(train_losses))
		f.close()

		f = open('results/bleu.txt', 'w')
		f.write(str(bleus))
		f.close()

		f = open('results/valid_loss.txt', 'w')
		f.write(str(test_losses))
		f.close()

		print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f}')
		print(f'\tValid Loss: {valid_loss:.3f}')
		print(f'\tBLEU Score: {bleu:.3f}')
		print(f'\tBest epoch: {best_epoch+1}')

		if early_stop_counter == utils.early_stop_patience :
			print("Early stopping !")
			break

if __name__ == '__main__':
	run(total_epoch=epoch, best_loss=inf, best_epoch = 0)
