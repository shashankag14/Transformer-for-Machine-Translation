"""
@author : Shashank Agarwal
@when : 10-02-2022
@homepage : https://github.com/shashankag14
"""

import utils
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder
from model.transformer import Transformer

#local project files
import dictionary as dict
import tokenizer
import dataloader
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# To empty the cache for TQDM
torch.cuda.empty_cache()

################################################################################
## Initialisation of Dictionary, Dataset and Model
################################################################################
# Create source and target dictionary and a list of sentences
input_lang_dic, output_lang_dic, input_lang_list, output_lang_list = dict.create_dictionary('cs', 'en')

# Tokenize sentences in source and target
tokenized_input_lang = [tokenizer.tokenize(sentence, input_lang_dic, utils.max_sent_len) for sentence in input_lang_list]
tokenized_output_lang = [tokenizer.tokenize(sentence, output_lang_dic, utils.max_sent_len) for sentence in output_lang_list]

# Create and fetch test dataset for source and target
_, _, test_dataset = dataloader.get_dataloader(tokenized_input_lang, tokenized_output_lang)

# Number of words in the dictionary
input_size = input_lang_dic.n_count
output_size = output_lang_dic.n_count

# Initialize encoder and decoder blocks
encoder_part = TransformerEncoder(input_size, utils.d_model, utils.n_layers, utils.n_heads, utils.ffn_hidden, utils.dropout,
                       utils.device)
decoder_part = TransformerDecoder(output_size, utils.d_model, utils.n_layers, utils.n_heads, utils.ffn_hidden, utils.dropout,
                       utils.device)
# Initialize the transformer using encoder and decoder
model = Transformer(encoder_part, decoder_part, utils.device, dict.PAD_TOKEN).to(utils.device)

# Laod best_model weights to use for translations
model.load_state_dict(torch.load('saved_chkpt/best_model.pt'))

################################################################################
## Translations on test data using greedy search
################################################################################
def translate_sentence(sentence, input_dic, output_dic, model, device, max_len):
    model.eval()
    # normalized_sentence = dict.normalizeString(sentence)
    # tokens = tokenizer.tokenize(normalized_sentence, input_dic)
    input_tensor = torch.LongTensor(sentence).unsqueeze(0).to(device)

    # Create pad mask for encoder
    input_mask = model.make_input_mask(input_tensor)

    # Pass through the encoder without backprop
    with torch.no_grad():
        encoded_input = model.encoder(input_tensor, input_mask)

    # Initialize the predicted tokens with <SOS> as the input at first time stamp to decoder
    target_tokens = [dict.SOS_TOKEN]

    # Run the decoder for max_len times which is 50 in this case but stop as soon as <EOS> is recieved as an output
    for i in range(max_len):
        target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
        # Make target mask using lower triangular matrix
        target_mask = model.make_target_mask(target_tensor)
    
        with torch.no_grad():
            output, attention = model.decoder(target_tensor, encoded_input, target_mask, input_mask)

        # fetch the prediction with highest softmax score
        pred_token = output.argmax(2)[:,-1].item()
        # Append pred_token to the prediction list and use this updated list as an input for next step to decoder
        target_tokens.append(pred_token)

        # stop the loop once <EOS> is received i.e. translation completed
        if pred_token == dict.EOS_TOKEN:
            break

    # Detokenize the translation into words
    target_results = [output_dic.index2word[i] for i in target_tokens]
    
    return ' '.join(target_results[1:-1])

if __name__ == "__main__":
    # input_text = 'O tomto manuálu'
    f = open('results/translation_results.txt', 'w')
    for sentence in test_dataset:
        input_token = sentence['src']
        target_token = sentence['tgt']

        translation = translate_sentence(input_token, input_lang_dic, output_lang_dic, model, device, utils.max_sent_len)
        
        input_text = tokenizer.detokenize(input_token, input_lang_dic)
        target_text = tokenizer.detokenize(target_token, output_lang_dic)

        f.write("Source : {}\nTarget : {}\nPredicted : {}\n\n".format(input_text, target_text, translation))

        print('\n\nCS: ' + input_text)
        print('\nEN(Target): ' + target_text)
        print('\nEN(Prediction): ' + translation)