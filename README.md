<div id="top"></div>

# Transformer-for-Machine-Translation
A PyTorch implementation of Transformers from scratch for Machine Translation on PHP Corpus dataset[1] based on "Attention Is All You Need" by Ashish Vaswani et. al.[2]

## Getting Started : 

### Downloading
1. Download the data using **scripts/download_data.sh**.  
Note: Extract **PHP.cs-en.cs** and **PHP.cs-en.en** files from the downloaded zip file and move them inside **data/**
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
```

2. Run below code to plot Train/Valid Loss vs Epoch and save in **results/** (this step can be done anytime after executing step 1):
```
python3 plot.py
```

2. Use the saved checkpoints (**saved_ckpt/best_model.pt**) to test the model :
```
python3 test.py
```

### Arguments for train.py

| Parameters | Description | Default Value |
| --- | --- | --- |
| `--src_data` | Location of the source data | |
| `--tgt_data` | Location of the target data | |
| `--epoch` | Number of epochs to train | 200 |
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
<p align="right">(<a href="#top">back to top</a>)</p>

## Transformer Architecture :
<img src="https://user-images.githubusercontent.com/74488693/146267612-aa100838-d75f-48ec-b5d5-ce3755687cb5.png" height="700" width="500">

## Multi Headed Attention Block in Encoder/Decoder:
<img src="https://user-images.githubusercontent.com/74488693/144745249-5c99709d-0446-45fc-a4cb-f0428ead371e.png" height="300" width="600">


## Computing Attention using Key, Query and Value :
<img src="https://user-images.githubusercontent.com/74488693/146268694-0c8517a1-5795-4efa-a51b-23bae6fab520.png" height="100" width="300">


## Positional Embedding using sin/cos :
<img src="https://user-images.githubusercontent.com/74488693/146268889-723d15a5-2d18-48ba-85a9-936f72ce646f.png" height="100" width="300">

<p align="right">(<a href="#top">back to top</a>)</p>

## Space-Time Complexity (GPU used : Tesla K80)
| Parameter    | Amount   |
|-------------|-------------|
| GPU usage |  |
| Number of trainable parameters | 39,865,969 |
| Time per Epoch | 3m 40s  |

<p align="right">(<a href="#top">back to top</a>)</p>

## Results
| Parameter    | Value   |
|-------------|-------------|
| Minimum train loss | 1.91 |
| Minimum Validation loss | 1.81 |
| BLEU Score (on Test data) | 46.26  |

<p align="right">(<a href="#top">back to top</a>)</p>

## References

1. [PHP Corpus Dataset](https://opus.nlpl.eu/PHP.php)
2. ["Attention is all you need."](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et. al.
3. [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

<p align="right">(<a href="#top">back to top</a>)</p>
