import os

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
    src_outfile = open(utils.no_dup_src_data_path, "w")
    trg_outfile = open(utils.no_dup_tgt_data_path, "w")
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


def removeLongSent():
    src_inlines = []
    trg_inlines = []
    for line in open(utils.no_dup_src_data_path, "r"):
        src_inlines.append(line)

    for line in open(utils.no_dup_tgt_data_path, "r"):
        trg_inlines.append(line)
    print("{} unique sentences in source and target each !".format(len(trg_inlines)))
    combined_inlines = list(zip(src_inlines, trg_inlines))

    src_outfile = open(utils.src_data_path, "w")
    trg_outfile = open(utils.tgt_data_path, "w")
    removed_sent_count = 0

    for src_line, trg_line in combined_inlines:
        # not a duplicate and within max sentence length limit, then write line in the file
        if len(src_line) < utils.max_sent_len:
            src_outfile.write(src_line)
            trg_outfile.write(trg_line)
        else:
            removed_sent_count += 1

    src_outfile.close()
    trg_outfile.close()
    print(f"{removed_sent_count} sentences removed having length more than {utils.max_sent_len} from the source and target each !")



de = "data/original_data/news-commentary-v8.cs-en.cs"
en = "data/original_data/news-commentary-v8.cs-en.en.txt"

new_de = "data/usable_data/news-src.txt"
new_en = "data/usable_data/news-trg.txt"

src_inlines = []
trg_inlines = []
for line in open(de, "r"):
    src_inlines.append(line)

for line in open(en, "r"):
    trg_inlines.append(line)

avg_len_src = sum(len(x.split()) for x in src_inlines) / len(src_inlines)
avg_len_trg = sum(len(x.split()) for x in trg_inlines) / len(trg_inlines)

print(f"Avg len of sent in SRC: {avg_len_src} , Avg len of sent in TRG: {avg_len_trg}")
print(f"{len(src_inlines)} {len(trg_inlines)} sentences in source and target each !")

src_outfile = open(new_de, "w")
trg_outfile = open(new_en, "w")

lines_seen = set()  # holds lines already seen
combined_inlines = list(zip(src_inlines, trg_inlines))

removed_sent_count = 0
sent_count = 0

for src_line, trg_line in combined_inlines:
    src_word_len = len(src_line.split())
    if src_line not in lines_seen and src_word_len < 50 and sent_count < 8000:  # not a duplicate
        src_outfile.write(src_line)
        trg_outfile.write(trg_line)
        lines_seen.add(src_line)
        sent_count+=1
    else:
        removed_sent_count += 1
print(f"{removed_sent_count} sentences removed!")

src_outfile.close()
trg_outfile.close()
