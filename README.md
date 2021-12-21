<div id="top"></div>

# Transformer-for-Machine-Translation
A PyTorch implementation of Transformers from scratch for Machine Translation on PHP Corpus dataset[1] based on "Attention Is All You Need" by Ashish Vaswani et. al.[2]

![image](https://user-images.githubusercontent.com/74488693/146843786-d2b240cc-8aca-402d-9165-28ca8a5405b9.png)


## Getting Started : 

### Downloading
1. Download the data using `scripts/download_data.sh`.  
Note: Extract `PHP.cs-en.cs` and `PHP.cs-en.en` files from the downloaded zip file and move them inside `data/`
```
sh scripts/download_data.sh
```

3. Install the requried libraries :
```
sh scripts/requirements.sh
```

### Executing the program
1. Train the model : 
```
python3 train.py [-h] [--src_data SRC_DATA] [--tgt_data TGT_DATA]
                [--batch_size N] [--d_model D_MODEL] [--n_layers N_LAYERS]
                [--n_heads N_HEADS] [--ffn_hidden FFN_HIDDEN]
                [--dropout DROPOUT] [--max_sent_len MAX_SENT_LEN]
                [--init_lr INIT_LR] [--scheduler_factor SCHEDULER_FACTOR]
                [--optim_adam_eps OPTIM_ADAM_EPS]
                [--optim_patience OPTIM_PATIENCE]
                [--optim_warmup OPTIM_WARMUP]
                [--optim_weight_decay OPTIM_WEIGHT_DECAY] [--epoch EPOCH]
                [--clip CLIP] [--seed SEED] 
                [--label_smooth_eps LABEL_SMOOTH_EPS]
```

2. Run below code to plot Train/Valid Loss vs Epoch and save in `results/` (this step can be done anytime after executing step 1):
```
python3 plot.py
```

2. Use the saved checkpoints (`saved_ckpt/best_model.pt`) to test the model :
```
python3 test.py
```

### Arguments for train.py

| Parameters | Description | Default Value |
| --- | --- | --- |
| `--src_data` | Location of the source data | |
| `--tgt_data` | Location of the target data | |
| `--epoch` | Number of epochs to train | 40 |
| `--batch_size` | Batch size | 32 |
| `--d_model` | Size of word embedding | 512 |
| `--n_layers` | Number of enc/dec layers | 6 |
| `--n_heads` | Number of attention heads | 8 |
| `--ffn_hidden` | Number of hidden units in FFN | 2048 |
| `--dropout` | Dropout probability | 0.4 |
| `--max_sent_len` | Maximum length of sentence for train/valid/test | 10 |
| `--init_lr` | Initial Learning Rate | 5e-5 |
| `--scheduler_factor` | Factor with which LR will decreasing using scheduler | 0.9 |
| `--optim_adam_eps` | Adam epsilon | 5e-9 |
| `--optim_patience` | Number of epochs optimizer waits before decreasing LR | 8 |
| `--optim_warmup` | Optimizer warmup | 100 |
| `--optim_weight_decay` | Weight decay factor for optimizer | 5e-4 |
| `--clip` | Gradient clipping threshold to prevent exploding gradients | 1.0 |
| `--seed` | Seed for reproducibility | 1111 |
| `--label_smooth_eps` | Hyper-parameter for label smoothening | 0.1 |
<p align="right">(<a href="#top">back to top</a>)</p>

## Model :
### Transformer Architecture :
<img src="https://user-images.githubusercontent.com/74488693/146267612-aa100838-d75f-48ec-b5d5-ce3755687cb5.png" height="700" width="500">

### Multi Headed Attention Block in Encoder/Decoder:
<img src="https://user-images.githubusercontent.com/74488693/144745249-5c99709d-0446-45fc-a4cb-f0428ead371e.png" height="300" width="600">
<p align="right">(<a href="#top">back to top</a>)</p>

### Computing Attention using Key, Query and Value :
<img src="https://user-images.githubusercontent.com/74488693/146843949-2ae064f2-49da-4c99-ac25-690a8b4fd910.png" height="600" width="500">
<img src="https://user-images.githubusercontent.com/74488693/146268694-0c8517a1-5795-4efa-a51b-23bae6fab520.png" height="90" width="350">

### Positional Embedding using sin/cos : 
<img src="https://user-images.githubusercontent.com/74488693/146268889-723d15a5-2d18-48ba-85a9-936f72ce646f.png" height="90" width="340">

<p align="right">(<a href="#top">back to top</a>)</p>

## Data :
* Dataset splitting (# of sentences in each dataset)
    * Training : Validation : Test :: 25165 : 1391 : 1372
* `src_vocab_size` : 9211
* `trg_vocab_size` :  4849

## Regularization Techniques :
As mentioned in the paper "Attention is All You Need" [2], I have used two types of regularization techniques :
1. **Residual Dropout (dropout=0.4)** : Dropout has been added to embedding (positional+word) as well as to the output of each sublayer in Encoder and Decoder.
2. **Label Smoothening (eps=0.1)** : Affected the training loss but improved the BLEU score.

## Space-Time Complexity (GPU used : Tesla K80)
| Parameter    | Amount   |
|-------------|-------------|
| Number of trainable parameters | 39,865,969 |
| Time per Epoch | 4m 30s  |


## Results
| Parameter    | Value   |
|-------------|-------------|
| Minimum train loss | 2.80 |
| Minimum Validation loss | 1.95 |
| BLEU Score (on Test data) | 44.65  |

<img src="https://user-images.githubusercontent.com/74488693/146922221-f69899b0-f4c7-4b87-a42c-2f2062826203.png" height="350" width="420">

*Need to check why validation loss is lower than training loss. (Might be due to regularization techniques - dropout and label smoothening)*

`results/random_machine_translations.txt` : Some random machine translated sentences from test dataset have been generated using the saved checkpoint.

*Note : Due to limited availibilty of GPU resources, I could only train the model for 40-50 epochs.*

## References

1. [PHP Corpus Dataset](https://opus.nlpl.eu/PHP.php)
2. ["Attention is all you need."](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et. al.
3. [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

<p align="right">(<a href="#top">back to top</a>)</p>
