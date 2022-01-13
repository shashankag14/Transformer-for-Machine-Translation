"""
@author : Shashank Agarwal
@when : 08-12-2021
@homepage : https://github.com/shashankag14
"""
from sklearn.model_selection import train_test_split
import torchtext

# Local files
import tokenizer
import utils

# ########################################################################
# # CUSTOM DATASET - Ignores the sentences longer than MAX_LEN and
# #                 returns tuples of Source and Target
# ########################################################################
class CustomDataset(object):
    def __init__(self, data):
        self.src_data = []
        self.tgt_data = []
        for src,tgt in data:
            if (len(src)<utils.max_sent_len) or (len(tgt)<utils.max_sent_len):
                self.src_data.append(src)
                self.tgt_data.append(tgt)
        self.num_examples = len(self.src_data)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item):
        return {'src': self.src_data[item], 'tgt': self.tgt_data[item]}

# ########################################################################
# # Method for splitting the tokens and creating dataloaders for train, valid and test
# ########################################################################
def get_dataloader(src_tokens, tgt_tokens) :

    # 1. Split the SRC and TGT tokens into train, valid and test sets
    train_src, remain_src = train_test_split(src_tokens, test_size=0.4,
                                             random_state=utils.args.seed)
    train_tgt, remain_tgt = train_test_split(tgt_tokens, test_size=0.4,
                                             random_state=utils.args.seed)

    valid_src, test_src = train_test_split(remain_src, test_size=0.5,
                                           random_state=utils.args.seed)
    valid_tgt, test_tgt = train_test_split(remain_tgt, test_size=0.5,
                                           random_state=utils.args.seed)

    # 2. Create lists of tran/valid/test source-target tuples
    train_data = list(zip(train_src, train_tgt))
    valid_data = list(zip(valid_src, valid_tgt))
    test_data = list(zip(test_src, test_tgt))

    # 3. Create custom dataset using the above lists for Train/Valid/Test
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    test_dataset = CustomDataset(test_data)
    print("Number of sentences in Train/Valid/Test :", len(train_dataset), len(valid_dataset), len(test_dataset))

    # 4. Create batch iterator for the Train/Valid/Test dataloaders
    train_dataloader, valid_dataloader, test_dataloader = torchtext.legacy.data.BucketIterator.splits((train_dataset, valid_dataset, test_dataset),
                                                                          batch_size=utils.batch_size,
                                                                          sort_within_batch=True,
                                                                          sort_key=lambda x: len(x['src']),
                                                                          sort=False,
                                                                          device=utils.device)

    return train_dataloader, valid_dataloader, test_dataloader


#########################################
#       ONLY FOR SANITY CHECK           #
#########################################
DATALOADER_SANITY_CHECK = 0

if DATALOADER_SANITY_CHECK :
    corpus = tokenizer.Corpus()
    src_vocab_size = corpus.dictionary_src.n_word
    tgt_vocab_size = corpus.dictionary_tgt.n_word
    print("SRC_VOCAB_SIZE {}, TGT_VOCAB_SIZE {}".format(src_vocab_size, tgt_vocab_size))

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(corpus.tokenize_src,
                                                                        corpus.tokenize_tgt)
    print("Number of batches in Train/Valid/Test: ", len(train_dataloader), len(valid_dataloader), len(test_dataloader))

    # Create batches - needs to be called before each loop.
    test_dataloader.create_batches()

    for batch_num, batch in enumerate(test_dataloader.batches):
        max_src_batch_len = max([len(example['src']) for example in batch])
        print("Max len in src batch {} : {}".format(batch_num, max_src_batch_len))

        max_tgt_batch_len = max([len(example['tgt']) for example in batch])
        print("Max len in tgt batch {} : {}".format(batch_num, max_tgt_batch_len))

        # Only print for first batch and then use break
        for example in batch:
            print('SRC example :{}\nSRC len :{}\nTGT example :{}\nTGT len :{}'.format(example['src'],
                                                                                      len(example['src']),
                                                                                      example['tgt'],
                                                                                      len(example['tgt'])))

            src_words = tokenizer.detokenize(example['src'], corpus.dictionary_src)
            tgt_words = tokenizer.detokenize(example['tgt'], corpus.dictionary_tgt)

            print('\nSRC sentence :{}\nTGT sentence :{}'.format(src_words,tgt_words))
            print('\n\n')
            break


