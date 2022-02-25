"""
@author : Shashank Agarwal
@when : 11-12-2021
@homepage : https://github.com/shashankag14
"""

import dictionary as dict

# ########################################################################
# # Method to detokenize i.e. convert idx to words
# input - List of idx of a sentence, Vocabulary to convert from idx2word
# output - Sentence of words
# It ignores any special tokens (EOS,SOS,PAD)
# ########################################################################
#takes in a sentence and dictionary, and tokenizes based on dictionary
def tokenize(sentence, dictionary, MAX_LENGTH=50):
    split_sentence = [word for word in sentence.split(' ')]

    token = [dict.SOS_TOKEN]

    token += [dictionary.word2index[word] for word in sentence.split(' ')]
    token.append(dict.EOS_TOKEN)

    token += [dict.PAD_TOKEN]*(MAX_LENGTH - len(split_sentence))
    return token

# ########################################################################
# # Method to detokenize i.e. convert idx to words
# input - List of idx of a sentence, Vocabulary to convert from idx2word
# output - Sentence of words
# It ignores any special tokens (EOS,SOS,PAD)
# ########################################################################
def detokenize(x, vocab):
    words = []
    for i in x:
        word = vocab.index2word[i]
        if word != 'EOS' and word != 'SOS' and word != 'PAD':
            words.append(word)

    words = " ".join(words)
    return words