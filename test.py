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
# ########################################################################
# # Create instance of Dictionary
# ########################################################################
corpus = tokenizer.Corpus()
src_vocab_size = corpus.dictionary_src.n_word
tgt_vocab_size = corpus.dictionary_tgt.n_word

# ########################################################################
# # Create instance of Transformer Model
# ########################################################################
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

# ########################################################################
# # Fetch Test Dataloader
# ########################################################################
_, _, test_dataloader = dataloader.get_dataloader(corpus.tokenize_src,corpus.tokenize_tgt)
# ########################################################################
# # Run the test using best_model.pt checkpoint and compute BLEU score
# # Translations will be saved in /results/translation_results.txt
# ########################################################################
def test_model(dataloader):
    model.load_state_dict(torch.load("saved_chkpt/best_model.pt"))
    with torch.no_grad():
        batch_bleu = []
        f = open('results/translation_results.txt', 'w')        
        for i, batch in enumerate(dataloader.batches):
            src, trg = tokenizer.add_padding(batch)
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg)

            total_bleu = []
            # Print machine translated sentences
            for j in range(trg.size(dim=0)):
                src_words = tokenizer.detokenize(src[j].tolist(), corpus.dictionary_src)
                trg_words = tokenizer.detokenize(trg[j].tolist(), corpus.dictionary_tgt)
                output_words = output[j].max(dim=1)[1]
                output_words = tokenizer.detokenize(output_words.tolist(), corpus.dictionary_tgt)
                
                f.write("Source : {}\nTarget : {}\nPredicted : {}\n".format(src_words, trg_words, output_words))

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

        f.write("#"*10)
        f.write("TOTAL BLEU SCORE : {}".format(batch_bleu))
        f.close()

if __name__ == '__main__':
    # Create batches - needs to be called before each loop.
    test_dataloader.create_batches()
    test_model(test_dataloader)