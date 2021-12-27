"""
@author : Shashank Agarwal
@when : 07-12-2021
@homepage : https://github.com/shashankag14
"""

import unicodedata
import re

import utils

# ########################################################################
# # DICTIONARY DATA PRE-PROC
# ########################################################################
#
# Turn a Unicode string to plain ASCII
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Method to remove duplicate sentences from the original data
# Note : this has to be done before tokenization and train/valid/test data splitting
# Reference : In tokenizer.py -> Corpus().init()
def removeDuplicateData():
    src_inlines = []
    trg_inlines = []
    for line in open(utils.orig_src_data_path, "r"):
        src_inlines.append(line)

    for line in open(utils.orig_tgt_data_path, "r"):
        trg_inlines.append(line)
    print("{} sentences in source and target each !".format(len(trg_inlines)))
    combined_inlines = list(zip(src_inlines, trg_inlines))

    lines_seen = set()  # holds lines already seen
    src_outfile = open(utils.src_data_path, "w")
    trg_outfile = open(utils.tgt_data_path, "w")
    removed_sent_count = 0
    for src_line, trg_line in combined_inlines:
        if src_line not in lines_seen:  # not a duplicate
            src_outfile.write(src_line)
            trg_outfile.write(trg_line)
            lines_seen.add(src_line)
        else:
            removed_sent_count += 1
    src_outfile.close()
    trg_outfile.close()
    print("{} duplicate sentences have been removed from the source and target each !".format(removed_sent_count))

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# ########################################################################
# # DICTIONARY
########################################################################
SOS_token = 0
PAD_token = 1
EOS_token = 2

class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # mapping word to its idx
        self.word2count = {} # frequency of each word
        self.idx2word = {SOS_token : '<SOS>', PAD_token : '<pad>', EOS_token : '<EOS>'} # mapping idx to its word
        self.n_word = 3 # pad, SOS and EOS # total number of unique words in dictionary

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_word
            self.word2count[word] = 1
            self.idx2word[self.n_word] = word
            self.n_word += 1
        else :
            self.word2count[word] += 1
        return self.word2idx[word]

    def add_all_words(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                normalised_line = normalizeString(line)
                words = normalised_line.split()
                for word in words:
                    self.add_word(word)

    def getsentence(self, sent):
        ids = []
        words = sent.split()
        for word in words:
            ids.append(self.word2idx[word])
        return ids

    def __len__(self):
        return len(self.idx2word)