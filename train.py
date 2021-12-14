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

# Local project files
import utils
from model.transformer import Transformer
from utils import *
import dictionary as dict
import tokenizer
import dataloader
from bleu_metric import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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

# Create dictionary
corpus = tokenizer.Corpus()
src_vocab_size = corpus.dictionary_src.n_word
tgt_vocab_size = corpus.dictionary_tgt.n_word
print("SRC_VOCAB_SIZE {}, TGT_VOCAB_SIZE {}".format(src_vocab_size, tgt_vocab_size))

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

# Optimizer
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# LR Scheduler for better gradient descent 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

# Loss function (Cross entropy)
criterion = nn.CrossEntropyLoss(ignore_index=dict.PAD_token)

# Train loop for each epoch
def train(model, iterator, optimizer, criterion, clip):
	model.train()
	epoch_loss = 0
	for batch_num, batch in enumerate(iterator.batches):
		src, trg = tokenizer.add_padding(batch)
		src, trg = src.to(device), trg.to(device)

		optimizer.zero_grad()
		output = model(src, trg)  # trg[:,:-1] doesnt work as output is [N,seq_len] and so is src

		# Reshape output and trg before computing the loss
		output_reshape = output[:, 1:].contiguous().view(-1, output.shape[-1])
		trg = trg[:, 1:].contiguous().view(-1) # removing the first token of <SOS> and then flattening 2D to 1D tensor

		# output_reshape : (N*seq_len, model_dim), tgt : (N*seq_len)
		# Compute batch loss
		loss = criterion(output_reshape, trg)

		# Backprop
		loss.backward()

		# Clip gradients to avoid exploding gradient issues
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()

		epoch_loss += loss.item()
		print('step :', round((batch_num / len(iterator)) * 100, 2), '% , loss :', loss.item())

	return epoch_loss / len(iterator)

# Validation loop for each epoch
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
	return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
	train_losses, test_losses, bleus = [], [], []
	for step in range(total_epoch):
		start_time = time.time()

		# Create batches - needs to be called before each loop.
		train_dataloader.create_batches()
		train_loss = train(model, train_dataloader, optimizer, criterion, clip)

		valid_dataloader.create_batches()
		valid_loss, bleu = evaluate(model, valid_dataloader, criterion)

		end_time = time.time()

		if step > warmup:
			scheduler.step(valid_loss)

		train_losses.append(train_loss)
		test_losses.append(valid_loss)
		bleus.append(bleu)
		epoch_mins, epoch_secs = compute_time(start_time, end_time)

		if valid_loss < best_loss:
			best_loss = valid_loss
			torch.save(model.state_dict(), 'saved_chkpt/best_model.pt')

		f = open('results/train_loss.txt', 'w')
		f.write(str(train_losses))
		f.close()

		f = open('results/bleu.txt', 'w')
		f.write(str(bleus))
		f.close()

		f = open('results/test_loss.txt', 'w')
		f.write(str(test_losses))
		f.close()

		print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f}')
		print(f'\tVal Loss: {valid_loss:.3f}')
		print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
	run(total_epoch=epoch, best_loss=inf)
