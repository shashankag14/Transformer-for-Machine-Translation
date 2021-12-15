"""
@author : Shashank Agarwal
@when : 07-12-2021
@homepage : https://github.com/shashankag14
"""
from io import open

# local files in project
from utils import *
from dictionary import *

# ########################################################################
# # CORPUS CLASS
# 1. Creates SRC/TGT dictionary
# 2. Performs tokenization :
#       2.1 Performs data preprocessing using methods in dictionary.py
#       2.2 Returns tokens for each sentence as a single list
# ########################################################################
class Corpus(object):
    def __init__(self):
        self.dictionary_src = Dictionary()
        self.dictionary_tgt = Dictionary()

        self.dictionary_src.add_all_words(src_data_path)
        self.dictionary_tgt.add_all_words(tgt_data_path)

        self.tokenize_src = self.tokenize(src_data_path, self.dictionary_src, max_sent_len)
        self.tokenize_tgt = self.tokenize(tgt_data_path, self.dictionary_tgt, max_sent_len)

    def tokenize(self, path, dictionary, max_sent_len):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                normalised_line = normalizeString(line)
                words = normalised_line.split()

                ids = [SOS_token]
                for word in words:
                    ids.append(dictionary.word2idx[word])
                ids.append(EOS_token)
                idss.append(ids)
        return idss

    def get_max_sent_len(self, path):
        with open(path, 'r', encoding="utf8") as f:
            max_sent_len = 0
            for line in f:
                normalised_line = normalizeString(line)
                words = normalised_line.split()
                if len(words) > max_sent_len:
                    max_sent_len = len(words)
                    max_sent = line
        # print("Maximum length sentence : {}".format(max_sent))
        # print("Maximum length : {}".format(max_sent_len))
        return max_sent_len

# ########################################################################
# # Method to add padding as per max length of sample in each batch
# input - list of samples in each batch
# output - tensor of samples in each batch
# ########################################################################
def add_padding(batch):
    max_src_batch_len = max([len(example['src']) for example in batch])
    max_tgt_batch_len = max([len(example['tgt']) for example in batch])
    max_batch_len = max(max_src_batch_len, max_tgt_batch_len)

    padded_src_batch = []
    padded_tgt_batch = []
    for example in batch:

        if len(example['src']) < max_batch_len:
            example['src'].extend([PAD_token] * (max_batch_len - len(example['src'])))
        padded_src_batch.append(torch.unsqueeze(torch.tensor(example['src']).type(torch.int64), dim=0))  # (1, seq_len)

        if len(example['tgt']) < max_batch_len:
            example['tgt'].extend([PAD_token] * (max_batch_len - len(example['tgt'])))
        padded_tgt_batch.append(torch.unsqueeze(torch.tensor(example['tgt']).type(torch.int64), dim=0))  # (1, seq_len)

    padded_src_batch_tensor = torch.cat(padded_src_batch, dim=0)  # (N, seq_len)
    padded_tgt_batch_tensor = torch.cat(padded_tgt_batch, dim=0)  # (N, seq_len)
    return padded_src_batch_tensor, padded_tgt_batch_tensor


# ########################################################################
# # Method to detokenize i.e. convert idx to words
# input - List of idx of a sentence, Vocabulary to convert from idx2word
# output - Sentence of words
# It ignores any special tokens (EOS,SOS,PAD)
# ########################################################################
def detokenize(x, vocab):
    words = []
    for i in x:
        word = vocab.idx2word[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words

#########################################
#       ONLY FOR SANITY CHECK           #
#########################################
TOKENIZER_SANITY_CHECK = 0
if TOKENIZER_SANITY_CHECK :
    corpus = Corpus()
    print(len(corpus.dictionary_src),len(corpus.dictionary_tgt))
    print(corpus.dictionary_src.n_word, corpus.dictionary_tgt.n_word)
    print(corpus.tokenize_src[2])