<div id="top"></div>

# Transformer-for-Machine-Translation
A PyTorch implementation of Transformers from scratch for Machine Translation on PHP Corpus dataset[1] (Czech->English) based on "Attention Is All You Need" by Ashish Vaswani et. al.[2]. The motive to create this repository is not to implement a state-of-the-art model for Machine Translation, but to get a hands-on experience in implementing the Transformer architecture from scratch. 

_**Modification done over the baseline[2] :**_
- _Reduced the model depth and improving the BLEU score by a margin of **0.84** (Check Ablation Study for more details)._
- _Other minor modifications include using Early Stopping as an additional regularization technique, gradient clipping to avoid gradient explosion, maximum size of sentences to train, and played around with the hyperparameter values._


![image](https://user-images.githubusercontent.com/74488693/146843786-d2b240cc-8aca-402d-9165-28ca8a5405b9.png)


## Getting Started : 

### Downloading
1. Download the data using [scripts/download_data.sh](scripts/download_data.sh).  
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
                [--early_stop_patience EARLY_STOP_PATIENCE]
```

2. Run below code to plot Train/Valid Loss vs Epoch and save in [results/](results/) (This step can be done anytime after executing step 1 to see the progress of the graph):
```
python3 plot.py
```

2. Use the saved checkpoints ([saved_ckpt/best_model.pt](saved_ckpt/best_model.pt)) to test the model :
```
python3 test.py
```

### Arguments for train.py

| Parameters | Description | Value used in modified version | Value (in the paper) |
| --- | --- | --- | --- |
| `--src_data` | Location of the source data | -path- | -path- |
| `--tgt_data` | Location of the target data | -path- | -path- |
| `--epoch` | Number of epochs to train | 40 | N/A |
| `--batch_size` | Batch size | 32 | N\A |
| `--d_model` | Size of word embedding | 512 | 512 |
| `--n_layers` | _**Number of enc/dec layers**_ | _**4**_ | _**6**_ |
| `--n_heads` | Number of attention heads | 8 | 8 |
| `--ffn_hidden` | Number of hidden units in FFN | 2048 | 2048 |
| `--dropout` | Dropout probability | 0.1 | 0.1 |
| `--max_sent_len` | _**Maximum length of sentence for train/valid/test**_ | _**30**_ | _**N/A**_ |
| `--init_lr` | Initial Learning Rate | 5e-5 | N/A |
| `--scheduler_factor` | Factor with which LR will decreasing using scheduler | 0.9 | 0.9 |
| `--optim_adam_eps` | _**Adam epsilon** | **5e-9** | **1e-9**_ |
| `--optim_patience` | Number of epochs optimizer waits before decreasing LR | 8 | N/A |
| `--optim_warmup` | Optimizer warmup | **16000** | 4000 |
| `--optim_weight_decay` | Weight decay factor for optimizer | 5e-4 | N/A |
| `--clip` | _**Gradient clipping threshold to prevent exploding gradients** | **1.0** | **N/A**_ |
| `--seed` | Seed for reproducibility | 1111 | N/A |
| `--label_smooth_eps` | Hyper-parameter for label smoothening | 0.1 | 0.1 |
| `--early_stop_patience` | _**Patience for Early Stopping** | **20** | **0.1** _|

<p align="right">(<a href="#top">back to top</a>)</p>

## Model :
### Transformer Architecture ([model/](model/)):
<img src="https://user-images.githubusercontent.com/74488693/146267612-aa100838-d75f-48ec-b5d5-ce3755687cb5.png" height="700" width="500">

### Multi Headed Attention Block in Encoder/Decoder ([attention.py](model/attention.py)) :
<img src="https://user-images.githubusercontent.com/74488693/144745249-5c99709d-0446-45fc-a4cb-f0428ead371e.png" height="300" width="600">
<p align="right">(<a href="#top">back to top</a>)</p>

### Computing Attention using Key, Query and Value ([attention.py](model/attention.py)) :
<img src="https://user-images.githubusercontent.com/74488693/146843949-2ae064f2-49da-4c99-ac25-690a8b4fd910.png" height="600" width="500">
<img src="https://user-images.githubusercontent.com/74488693/146268694-0c8517a1-5795-4efa-a51b-23bae6fab520.png" height="90" width="350">

### Positional Encoding using sin/cos ([position_encoding.py](model/position_encoding.py)): 
<img src="https://user-images.githubusercontent.com/74488693/146268889-723d15a5-2d18-48ba-85a9-936f72ce646f.png" height="90" width="340">

<p align="right">(<a href="#top">back to top</a>)</p>

## Data :
* Original data : 32983 sentences each in source and target
* After removing duplicate sentences (from Source): 5529 sentences each in source and target
* Dataset splitting (# of sentences in each dataset)
    * >Training : Validation : Test :: 4257 : 525 : 533
* >src_vocab_size : 9211
* >trg_vocab_size :  4735

## Regularization Techniques :
As mentioned in the paper "Attention is All You Need" [2], I have used two types of regularization techniques which are active *only* during the train phase :
1. **Residual Dropout (dropout=0.1)** : Dropout has been added to embedding (positional+word) as well as to the output of each sublayer in Encoder and Decoder.
    * Note : In order to deativate Dropout during ```eval()```, I have used ```nn.Dropout()``` instead of ```nn.functional.dropout``` (Refer this [link](https://stackoverflow.com/questions/53419474/using-dropout-in-pytorch-nn-dropout-vs-f-dropout/53452827#53452827) for more info)
    * I tried increasing the dropout parameter, however didn't see any considerable improvement.
    
3. **Label Smoothening (eps=0.1)** : Affected the training loss but improved the BLEU score on test data.
4. **Early Stopping (early_stop_patience=20)** : To stop the training before the model starts to overfit. (This is just an additional technique used which has not been presented in the paper)

## Ablation Study
### 1. Comparison of Results
| Parameter    | Results from my model  | Results from baseline model |
|-------------|-------------|-------------|
| Minimum train loss | _**4.05**_ | 5.22 |
| Minimum Validation loss | _**4.85**_ | 5.28 |
| BLEU Score (on Test data) | _**6.25**_  | 5.41 |

* [results/random_machine_translations.txt](results/random_machine_translations.txt) : Some random machine translated sentences from test dataset have been generated using the saved checkpoint from my model.

### 2. Comparison of Learning Graphs
<img src="https://user-images.githubusercontent.com/74488693/148826900-5cc6ba25-0f00-45c7-86b2-ef7a00b77b05.png" height="350" width="840">

### 3. Comparison of Space-Time Complexity (GPU used : Tesla K80) 
| Parameter    | From my model   | From baseline model |
|-------------|-------------|-------------|
| Number of trainable parameters | 29,689,215 | 39,865,969 |
| Time per Epoch | 40s  | 2m 30s |

## Conclusion
As seen from the ablation study, my modified model (with reduced number of encoder/decoder layers) works particularily well on PHP Corpus dataset in terms of BLEU score, Train/Validation Loss as well as space time complexity. This might be due to the fact that there are less number of unique sentences in the dataset to train, hence shallow network performs better than deep networks. 

Future direction could be to try features like Relative Position Encoding, Syntax Aware NMT (might work particularily well as the data is small), Coverage based NMT or Beam Search.

## Additional Comment 
Due to limited availibilty of GPU resources, I could only train the model for very less training data due to which the results are not satisfactory. However, I tried to get the learning curve and BLEU score as good as possible. [Refer](https://arxiv.org/pdf/2105.13065.pdf) for information about how MT performs on low resources

## References

1. [PHP Corpus Dataset](https://opus.nlpl.eu/PHP.php)
2. ["Attention is all you need."](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et. al.
3. [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

<p align="right">(<a href="#top">back to top</a>)</p>
