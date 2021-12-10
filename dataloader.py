"""
@author : Shashank Agarwal
@when : 08-12-2021
@homepage : https://github.com/shashankag14
"""

from sklearn.model_selection import train_test_split
import torch.utils.data as data

import tokenizer
import utils

# Method for splitting the tokens and creating dataloaders for train, valid and test
def get_dataloader(src_tokens, tgt_tokens) :

    # 1. Split the SRC and TGT tokens into train, valid and test sets
    train_src, remain_src = train_test_split(src_tokens, test_size=0.1,
                                             random_state=27)
    train_tgt, remain_tgt = train_test_split(tgt_tokens, test_size=0.1,
                                             random_state=27)

    valid_src, test_src = train_test_split(remain_src, test_size=0.5,
                                           random_state=27)
    valid_tgt, test_tgt = train_test_split(remain_tgt, test_size=0.5,
                                           random_state=27)

    # 2. Make tuples of SRC and TGT and create datasets
    train_dataset = data.TensorDataset(train_src, train_tgt)
    valid_dataset = data.TensorDataset(valid_src, valid_tgt)
    test_dataset = data.TensorDataset(test_src, test_tgt)

    # 3. Create batch wise data iterator (Dataloader) used while training epochs
    train_dataloader = data.DataLoader(train_dataset, batch_size=utils.batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=utils.batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=utils.batch_size, shuffle=True)
    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    return train_dataloader, valid_dataloader, test_dataloader

DATALOADER_SANITY_CHECK = 0

if DATALOADER_SANITY_CHECK :
    corpus = tokenizer.Corpus()
    src_vocab_size = corpus.dictionary_src.n_word
    tgt_vocab_size = corpus.dictionary_tgt.n_word
    print("SRC_VOCAB_SIZE {}, TGT_VOCAB_SIZE {}".format(src_vocab_size, tgt_vocab_size))

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(corpus.tokenize_src,
                                                                        corpus.tokenize_tgt)
    for i, batch in enumerate(train_dataloader):
        print(i)
        print("SRC :", batch[0])
        print("TGT :", batch[1])