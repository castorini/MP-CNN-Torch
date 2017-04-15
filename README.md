# Multi-Perspective Convolutional Neural Networks for Modeling Textual Similarity

This repo contains the Torch implementation of multi-perspective convolutional neural networks for modeling textual similarity, described in the following paper:

+ Hua He, Kevin Gimpel, and Jimmy Lin. [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks.](http://aclweb.org/anthology/D/D15/D15-1181.pdf) *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015)*, pages 1576-1586.

This model does not require external resources such as WordNet or parsers, does not use sparse features, and achieves good accuracy on standard public datasets.

Installation and Dependencies
------------

- Please install Torch deep learning library. We recommend this local installation which includes all required packages our tool needs, simply follow the instructions here:
https://github.com/torch/distro

- Currently our tool only runs on CPUs, therefore it is recommended to use INTEL MKL library (or at least OpenBLAS lib) so Torch can run much faster on CPUs. 

- Our tool then requires Glove embeddings by Stanford. Please run fetech_and_preprocess.sh for downloading and preprocessing this data set (around 3 GBs).


Running
------------

- Command to run (training, tuning and testing all included): 
- ``th trainSIC.lua`` or ``th trainMSRVID.lua``

The tool will output pearson scores and also write the predicted similarity scores given each pair of sentences from test data into predictions directory.

Adaption to New Dataset
------------
To run our model on your own dataset, first you need to build the dataset following below format and put it under data folder:

- a.toks: sentence A, each sentence per line.
- b.toks: sentence B, each sentence per line.
- id.txt: sentence pair ID
- sim.txt: semantic relatedness gold label, can be in any scale. For binary classification, the set of labels will be {0, 1}.

Then build vocabulary for your dataset which writes the vocab-cased.txt into your data folder:
```
$ python build_vocab.py
```
The last thing is to change the training and model code slightly to process your dataset:
- change util/read_data.lua to handle your data.
- create a new piece of training code following trainSIC.lua to read in your dataset.
- change Conv.lua in Line 89-102 and 142-148 to handle your own task
- more details can refer to issue https://github.com/hohoCode/textSimilarityConvNet/issues/6

Then you should be able to run your training code.

Ackowledgement
-------------
We thank Kai Sheng Tai for providing the preprocessing codes. We also thank the public data providers and Torch developers. Thanks.
