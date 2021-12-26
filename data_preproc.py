import unicodedata
import re

import utils

# ########################################################################
# # Turn a Unicode string to plain ASCII
# https://stackoverflow.com/a/518232/2809427
# ########################################################################
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# ########################################################################
# # Lowercase, trim, and remove non-letter characters
# ########################################################################
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# ########################################################################
# # Method to remove duplicate and long sentences from the original data
# Note : this has to be done before tokenization and train/valid/test data splitting
# Reference : In tokenizer.py -> Corpus().init()
# ########################################################################
def preprocess_data():
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
        # not a duplicate and within max sentence length limit, then write line in the file
        if src_line not in lines_seen and len(src_line) < utils.max_sent_len:
            src_outfile.write(src_line)
            trg_outfile.write(trg_line)
            lines_seen.add(src_line)
        else:
            removed_sent_count += 1

    src_outfile.close()
    trg_outfile.close()
    print("{} duplicate sentences have been removed from the source and target each !".format(removed_sent_count))