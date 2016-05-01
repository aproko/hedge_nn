#!/bin/bash

# Inputs: path to input training + dev data (format is: label \t text), path to word vectors and path to test data (for performance evaluation)
# Function: takes the training data and testing data and pickles them so that the CNN can read it in the proper format. Writes the vocabulary to all_vocab.txt (for future use, if wanted); the training data to train_dev.pkl; and test data to input_test.pkl.

python ./train/train_data_to_cnn_input.py ./train/train_dev.txt /Users/aproko/Downloads/GoogleNews-vectors-negative300.bin ./train/test.txt

# Inputs: none (reads files that were produced by previous script; outputs a model.pkl file that can be used for labeling hedges in new text.
# Function: trains a convolutional neural net on the given training data. For more information on parameters, refer to Yoon Kim's original code and paper (github.com/yoonkim/CNN_sentence).

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python ./train/cnn_train.py -non_static -word2vec

