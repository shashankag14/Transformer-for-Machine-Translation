"""
@author : Shashank Agarwal
@when : 07-12-2021
@homepage : https://github.com/shashankag14
"""

from io import open

# local files in project
from utils import *
from dictionary import *

class Corpus(object):
    def __init__(self):
        self.dictionary_src = Dictionary()
        self.dictionary_tgt = Dictionary()

        self.dictionary_src.add_all_words(src_data_path)
        self.dictionary_tgt.add_all_words(tgt_data_path)

        self.tokenize_src = self.tokenize(src_data_path, self.dictionary_src)
        self.tokenize_tgt = self.tokenize(tgt_data_path, self.dictionary_tgt)

    def tokenize(self, path, dictionary):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                normalised_line = normalizeString(line)
                words = normalised_line.split()

                ids = [SOS_token]
                for word in words:
                    ids.append(dictionary.word2idx[word])
                ids.append(EOS_token)

                max_ids_len = max_sent_len + 2  # <SOS> and <EOS> tokens will always be added

               # if sentence is larger than max limit, shorten it and append <EOS> @ last
                if len(ids) > max_ids_len:
                    ids = ids[:max_ids_len-1]
                    ids.append(EOS_token)

                # if sentence is shorter than max limit, add <PAD> between <SOS> and <EOS>
                elif len(ids) < max_ids_len :
                    ids.extend([PAD_token] * (max_ids_len - len(ids)))

                    #############################################################################################
                    #            Code below is in case <PAD> needs to be added between <SOS> and <EOS>          #
                    #############################################################################################
                    # while len(ids) != max_ids_len :
                    #     PAD_pos = (len(ids) - 1) # position of <PAD> to be inserted is always behind <EOS>
                    #     ids.insert(PAD_pos, PAD_token)

                idss.append(torch.unsqueeze(torch.tensor(ids).type(torch.int64), dim=0)) # (1, seq_len)
            ids = torch.cat(idss, dim=0) # (N, seq_len)
        return ids

TOKENIZER_SANITY_CHECK = 0
if TOKENIZER_SANITY_CHECK :
    corpus = Corpus()
    print(len(corpus.dictionary_src),len(corpus.dictionary_tgt))
    print(corpus.dictionary_src.n_word, corpus.dictionary_tgt.n_word)
    print(corpus.tokenize_src[2])