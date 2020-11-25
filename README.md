
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


## GLOVE VECTORS ##

The file "glove.42B.300d.txt" contains the Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors) available on [stanfordnlp](https://github.com/stanfordnlp/GloVe) repository. The direct link is given [here](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip).

NOTE : The file size of GloVe vectors is around 5GB when extracted from ZIP.



## Code ##

[train.py](train.py) - The main trainer python file. The default parameters are set to the values as given in the paper by authors. 
