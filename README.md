Text Similarity Measurement using Convolutional Neural Networks


Introduction
------------

This tool can be used to measure semantic similarity given any two pieces of texts. 

This repo contains the implementation of a convolutional neural network based model for comparing two sentences. Our model does not require external resources such as WordNet or parsers, and can still achieve highly competitive performance as measured on 3 public datasets (SICK, MSRVID, and MSRP).

For more details, please refer to our recent paper:
- ``Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks``
- Hua He, Kevin Gimpel, and Jimmy Lin. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015).


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
