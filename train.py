"""
@author : Shashank Agarwal
@when : 11-12-2021
@homepage : https://github.com/shashankag14
"""

import torch.nn as nn
import time
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

import utils
from utils import *
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder
from model.transformer import Transformer
import tokenizer
import dataloader
from bleu_metric import *
import dictionary as dict

print("Device being used : ", utils.device)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# To empty the cache for TQDM
torch.cuda.empty_cache()

# Count the number of trainable parameters in the model 
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)

################################################################################
## Initialisation of Dictionary, Dataset and Model
################################################################################
# Create source and target dictionary and a list of sentences
input_lang_dic, output_lang_dic, input_lang_list, output_lang_list = dict.create_dictionary('cs', 'en')

# Tokenize sentences in source and target
tokenized_input_lang = [tokenizer.tokenize(sentence, input_lang_dic, utils.max_sent_len) for sentence in input_lang_list]
tokenized_output_lang = [tokenizer.tokenize(sentence, output_lang_dic, utils.max_sent_len) for sentence in output_lang_list]

# Create and fetch train and validation dataset for source and target
train_dataloader, valid_dataloader, _ = dataloader.get_dataloader(tokenized_input_lang, tokenized_output_lang)

# Number of words in the dictionary
input_size = input_lang_dic.n_count
output_size = output_lang_dic.n_count
print("SRC Vocab size : {}, TGT Vocab size : {}".format(input_size, output_size))

# Initialize encoder and decoder blocks
encoder_part = TransformerEncoder(input_size, utils.d_model, utils.n_layers, utils.n_heads, utils.ffn_hidden, utils.dropout,
                       utils.device)
decoder_part = TransformerDecoder(output_size, utils.d_model, utils.n_layers, utils.n_heads, utils.ffn_hidden, utils.dropout,
                       utils.device)
# Initialize the transformer using encoder and decoder
model = Transformer(encoder_part, decoder_part, utils.device, dict.PAD_TOKEN).to(utils.device)
print(f'Trainable Parameters : {count_parameters(model):,}')
model.apply(initialize_weights)

################################################################################
## Loss function & Optimizer
################################################################################
criterion = nn.CrossEntropyLoss(ignore_index=dict.PAD_TOKEN)
optimizer = optim.Adam(model.parameters(), lr=utils.init_lr)

################################################################################
## Training loop for each epoch
################################################################################
def train(model, train_dataloader, clip, epoch_num,  label_smoothening=False):
    model.train()
    epoch_loss = 0
    with tqdm(train_dataloader, unit="batches") as tepoch:
        for batch_num, batch in enumerate(train_dataloader.batches):
            tepoch.set_description(f"Epoch {epoch_num+1}")

            optimizer.zero_grad()

            input = []
            target = []
            for example in batch :
                input.append(torch.unsqueeze(torch.tensor(example['src']).type(torch.int64), dim=0))
                target.append(torch.unsqueeze(torch.tensor(example['tgt']).type(torch.int64), dim=0))

            input = torch.cat(input, dim=0).to(utils.device)  # (N, seq_len)
            target = torch.cat(target, dim=0).to(utils.device)# (N, seq_len)

            # output -> (N, seq_len, tgt_vocab_size) ; # target -> (N, seq_len)
            output, _ = model(input, target[:,:-1])

            # removing the first token of <SOS> and then flattening 2D to 1D tensor
            output = output.contiguous().view(-1, output.shape[-1])
            target = target[:,1:].contiguous().view(-1)

            ##########################################################################
            ## Label Smoothening (Regularization technique)
            ##########################################################################
            # label smoothening only being used for training and not for validation
            if label_smoothening:
                # "Smoothed" one-hot vectors for the target sequences
                # (N*seq_len, trg_vocab_size)-> one-hot
                target_vector = torch.zeros_like(output).scatter(dim=1, index=target.unsqueeze(1), value=1.).to(
                    utils.device)

                # (N*seq_len, trg_vocab_size)-> "smoothed" one-hot; (-2 to ignore <pad> and target label)
                target_vector = target_vector * (1. - utils.label_smooth_eps) + utils.label_smooth_eps / (target_vector.size(1) - 2)

                # 1D tensor : (N*seq_len)-> Compute smoothed cross-entropy loss
                loss = (-1 * target_vector * F.log_softmax(output, dim=1)).sum(dim=1)
                # Create mask for locations of <PAD token in target (N*seq_len)
                non_pad_mask = target.ne(dict.PAD_TOKEN)

                # Ignore <PAD> while avging loss, thus this 1D tensor might be smaller
                # than the previous 1D loss tensor
                loss = loss.masked_select(non_pad_mask)
                # Final avg batch loss : Sum the non pad mask losses and divide by
                # number of non pad masks
                loss = loss.sum() / loss.size(0)

            else:
                # flattening 2D to 1D tensor
                loss = criterion(output, target)

            # Backprop
            loss.backward()
            # Clip gradients to avoid exploding gradient issues
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Update tqdm progress bar
            tepoch.update()
            tepoch.set_postfix(loss=loss.item())
            # Update running epoch loss
            epoch_loss += loss.item()
        # Find average of running epoch loss
        epoch_loss /= len(train_dataloader)
    return epoch_loss

################################################################################
## Validation loop for each epoch
################################################################################
def evaluate(model, valid_dataloader):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for batch_num, batch in enumerate(valid_dataloader.batches):
            input = []
            target = []
            for example in batch:
                input.append(torch.unsqueeze(torch.tensor(example['src']).type(torch.int64), dim=0))
                target.append(torch.unsqueeze(torch.tensor(example['tgt']).type(torch.int64), dim=0))

            input = torch.cat(input, dim=0).to(utils.device)   # (N, seq_len)
            target = torch.cat(target, dim=0).to(utils.device) 

            output, _ = model(input, target[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            target_reshape = target[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, target_reshape)
            epoch_loss += loss.item()

            # Compute BLEU score per batch - corpus level or sentence level??
            total_bleu = []
            # Note : Size of last batch might not be equal to batch_size, thus trg.size(dim=0)
            for j in range(target.size(dim=0)):
                target_words = tokenizer.detokenize(target[j].tolist(), output_lang_dic)                
                output_words = output[j].max(dim=1)[1]
                output_words = tokenizer.detokenize(output_words.tolist(), output_lang_dic)
                bleu = get_bleu(hypotheses=output_words.split(), reference=target_words.split())
                total_bleu.append(bleu)
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    torch.cuda.empty_cache()
    return epoch_loss / len(valid_dataloader), batch_bleu

################################################################################
## Run epochs - train and validation in each epoch
################################################################################
def run(total_epoch, best_loss, best_epoch):
    train_losses, test_losses, bleus = [], [], []
    # early_stop_counter = 0
    for step in range(total_epoch):
        start_time = time.time()

        # Create batches - needs to be called before each loop.
        train_dataloader.create_batches()
        train_loss = train(model, train_dataloader, clip, step, label_smoothening = True)

        valid_dataloader.create_batches()
        valid_loss, bleu = evaluate(model, valid_dataloader)

        end_time = time.time()

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = compute_time(start_time, end_time)

        if valid_loss <= best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved_chkpt/best_model.pt')
            best_epoch = step
        # else:
        # 	early_stop_counter+=1

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

        # if early_stop_counter == utils.early_stop_patience :
        # 	print("Early stopping !")
        # 	break

if __name__ == "__main__":
    run(total_epoch=epoch, best_loss=inf, best_epoch = 0)