# Transformer-for-Machine-Translation
A PyTorch implementation of Transformers from scratch for Machine Translation based on "Attention Is All You Need" by Ashish Vaswani et. al.

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data.

#### Transformer Architecture :

![image](https://user-images.githubusercontent.com/74488693/144745235-758eab17-cc7a-40c8-9710-3fcc5112c1be.png)

#### Multi Headed Attention Block in Encoder/Decoder:
![image](https://user-images.githubusercontent.com/74488693/144745249-5c99709d-0446-45fc-a4cb-f0428ead371e.png)


#### Computing Attention using Key, Query and Value :
![image](https://user-images.githubusercontent.com/74488693/144745258-317ce41c-5764-469e-8d9d-34fe194c5874.png)


#### Positional Embedding using sin/cos :
![image](https://user-images.githubusercontent.com/74488693/144745268-0e24760e-f871-4620-b21c-5f6d862ded7d.png)
