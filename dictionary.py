import unicodedata
import re
import pickle

#local project file imports
import utils

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

# ########################################################################
# # DICTIONARY DATA PRE-PROC
# ########################################################################
#https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# ########################################################################
# # DICTIONARY
########################################################################
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}    # mapping word to its idx
        self.word2count = {}    # frequency of each word
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}    # mapping idx to its word
        self.n_count = 3        # pad, SOS and EOS # total number of unique words in dictionary

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word
            self.n_count += 1
        else:
            self.word2count[word] += 1

# ########################################################################
# # Preprocess sentences and create dictionaries for source and target
########################################################################
def create_dictionary(source_lang, target_lang):
    #load first language to list
    source_list = []
    source_file = open(utils.src_data_path, encoding='utf8')
    for i, line in enumerate(source_file):
        source_list.append(line)

    # load second langauge to list
    target_list = []
    target_file = open(utils.tgt_data_path, 'r', encoding='utf8')
    for i, line in enumerate(target_file):
        target_list.append(line)

    #preprocess the sentences
    source_normalized = list(map(normalizeString, source_list))
    target_normalized = list(map(normalizeString, target_list))

    source_sentences = []
    target_sentences = []

    for i in range(len(source_normalized)):
        # COnvert each sentence into list of tokens
        source_tokens = source_normalized[i].split(' ')
        target_tokens = target_normalized[i].split(' ')

        # Only add sentences < 50 words to the list of sentences to be used for train/valid/test
        if len(source_tokens) <= utils.max_sent_len and len(target_tokens) <= utils.max_sent_len:
            source_sentences.append(source_normalized[i])
            target_sentences.append(target_normalized[i])

    del source_normalized
    del target_normalized

    input_dic = Dictionary(source_lang)
    output_dic = Dictionary(target_lang)

    # Add words of each sentence into dictionary
    for sentence in source_sentences:
        input_dic.add_sentence(sentence)

    # Add words of each sentence into dictionary
    for sentence in target_sentences:
        output_dic.add_sentence(sentence)

    # Save the dictionaries of source and target in saved_chkpt. Will be helpful in translation of unseen data
    save_dictionary(input_dic, input=True)
    save_dictionary(output_dic, input=False)

    return input_dic, output_dic, source_sentences, target_sentences


def save_dictionary(dictionary, input=True):
    if input is True:
        with open('saved_chkpt/input_dic.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('saved_chkpt/output_dic.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

