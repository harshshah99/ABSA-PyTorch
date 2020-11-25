
## Directory Structure ##

```bash
.
├── datasets
│   └── semeval14
├── figure_plots
│   ├── laptop_2way_acc.jpeg
│   ├── laptop_2way_loss.jpeg
│   ├── laptop_3way_acc.jpeg
│   ├── laptop_3way_loss.jpeg
│   ├── restaurant_2way_acc.jpeg
│   ├── restaurant_2way_loss.jpeg
│   ├── restaurant_3way_acc.jpeg
│   └── restaurant_3way_loss.jpeg
├── layers
│   ├── __pycache__
│   ├── attention.py
│   ├── dynamic_rnn.py
│   ├── __init__.py
│   └── squeeze_embedding.py
├── atae_lstm.py
├── attention_visualization.py
├── data_utils.py
├── glove.42B.300d.txt
├── infer_example.py
└── train.py
```



## DATASET ##

Dataset is in the folder [datasets](datasets/semeval14). It contains the training and testing data for Restaurants and Laptop reviews with respect to different aspects.


## GloVe Vectors ##

The file "glove.42B.300d.txt" contains the Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors) available on [stanfordnlp](https://github.com/stanfordnlp/GloVe) repository. The direct link is given [here](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip).

NOTE : The file size of GloVe vectors is around 5GB when extracted from ZIP.



## Code ##

[data_utils.py](data_utils.py) - Contains the code for preparing the SemEval14 dataset, loading the pretrained word vectors from GloVe and builds the embedding matrix which will be used for the aspect embedding.  


[atae_lstm.py](atae_lstm.py) - The model architecture. This file contains the code for ATAE_LSTM model using inbuilt modules available in PyTorch and a few custom layers like Attention, Dynamic RNN and Squeeze Embedding which are defined in [layers](layers/). 


[train.py](train.py) - The main python file to train Attenion based LSTM with aspect embedding. The default parameters are set to the values as given in the paper by authors.  

