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
- more details can refer to issue https://github.com/castorini/MP-CNN-Torch/issues/6

Then you should be able to run your training code.

Trained Model
-------------
We also porvide a model which is already trained on STS dataset. So it is easier if you just want to use the model and do not want to re-train the whole thing. 

The tarined model download link is [HERE](https://drive.google.com/file/d/0B-lu_eEMkpVxYVdPMldJX3JDVjg/view?usp=sharing). Model file size is 500MB. To use the trained model, then simply use codes below:
```
modelTrained = torch.load("download_local_location/modelSTS.trained.th", 'ascii')
modelTrained.convModel:evaluate()
modelTrained.softMaxC:evaluate()
local linputs = torch.zeros(rigth_sentence_length, emd_dimension)
linpus = XassignEmbeddingValuesX
local rinputs = torch.zeros(left_sentence_length, emd_dimension)
rinpus = XassignEmbeddingValuesX

local part2 = modelTrained.convModel:forward({linputs, rinputs})
local output = modelTrained.softMaxC:forward(part2)
local val = torch.range(0, 5, 1):dot(output:exp()) 
return val/5
```
The ouput variable 'val' contains a similarity score between [0,1]. The input linputs1/rinputs are torch tensors and you need to fill in the word embedding values for both. 

Example Deployment Script with Our Trained Model
-------------
We provide one example file for deployment: testDeployTrainedModel.lua. So it is easier for you to directly use our model. Run:
```
$ th testDeployTrainedModel.lua
```
This deployment file will use the trained model (assume you have downloaded the trained model from the above link), and it will generate scores given all test sentences of sick dataset. Please note the trained model is not trained on SICK data.


Ackowledgement
-------------
We thank Kai Sheng Tai for providing the preprocessing codes. We also thank the public data providers and Torch developers. Thanks.
