"""
@author : Shashank Agarwal
@when : 10-12-2021
@homepage : https://github.com/shashankag14
"""

from model.transformer import Transformer
from utils import *
import dictionary as dict
import tokenizer
import dataloader
from bleu_metric import *

print("Test phase started.")

# Create dictionary
corpus = tokenizer.Corpus()
src_vocab_size = corpus.dictionary_src.n_word
tgt_vocab_size = corpus.dictionary_tgt.n_word

# Create model instance
model = Transformer(src_vocab_size,
                    tgt_vocab_size,
                    src_mask_idx=dict.PAD_token,
                    tgt_mask_idx=dict.PAD_token,
                    device=device,
                    num_encoder_layers=n_layers,
                    num_decoder_layers=n_layers,
                    dim_model=d_model,
                    num_heads=n_heads,
                    dim_feedforward=ffn_hidden,
                    dropout=dropout).to(device)

# Create datasets
_, _, test_dataloader = dataloader.get_dataloader(corpus.tokenize_src,corpus.tokenize_tgt)
print("Size of Test datasets :", len(test_dataloader))

def test_model(dataloader):
    model.load_state_dict(torch.load("saved_chkpt/best_model.pt"))
    with torch.no_grad():
        batch_bleu = []
        for i, batch in enumerate(dataloader):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src, trg)

            total_bleu = []
            for j in range(trg.size(dim=0)):
                src_words = tokenizer.detokenize(src[j].tolist(), corpus.dictionary_src)
                trg_words = tokenizer.detokenize(trg[j].tolist(), corpus.dictionary_tgt)
                output_words = output[j].max(dim=1)[1]
                output_words = tokenizer.detokenize(output_words, corpus.dictionary_tgt)

                print('source :', src_words)
                print('target :', trg_words)
                print('predicted :', output_words)
                print()

                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu)
            print('BLEU SCORE = {}'.format(total_bleu))
            batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        print('TOTAL BLEU SCORE = {}'.format(batch_bleu))


if __name__ == '__main__':
    test_model(test_dataloader)