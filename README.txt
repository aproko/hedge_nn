This code is based on Yoon Kim's implementation of neural nets (github.com/yoonkim/CNN_sentence, with additions by Debanjan Ghosh.

1. The lists of single-word hedge cues and multi-word hedge cues are contained in dictionary.csv and multiword_dict.csv respectively. The formats of these files are the following:

Hedge, Hedge Type (hProp/hRel), Definition of the Hedge Type, Hedge Use Definition, Hedge Use Example, Non-Hedge Use Definition, Non-Hedge Use Example, whether this word should be stemmed when looking for hedges, Probability (from gold labeled data, what is the probability that this word is a hedge)

2. TO RUN A SAVED MODEL ON AN INPUT SENTENCE:

bash -i /path/to/input/txt/file -t /path/to/pickled/training/file -m /path/to/saved/model

The shell script runs a saved model on an input text file. The input has to be a text file with a single sentence in it (example input file is in ./test/INPUT.txt). We provide a trained model and the pickled training data in ./test/model.pkl and ./test/train_dev.pkl. If you train a new model with different parameters, be sure to change the appropriate parameters in cnn_test.py as well.

Example: bash test.sh -i ./test/INPUT.txt -t ./test/train_dev.pkl -m ./test/model.pkl

3. TO TRAIN A NEW MODEL:

bash -i /path/to/training/txt/data -t /path/to/testing/txt/data -v /path/to/GoogleNews/word/vectors

The shell script allows you to train a new model. We've included our training data in ./train/train_dev.txt, but it can be replaced with any data in the format: Label \t Text. We've also included our test data in test.txt, which allow us to test the performance of the model on a held out dataset with gold standard labels. NOTE: Make sure you download the GoogleNews trained word vectors from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz and unzip them (so you end up with a .bin file). 

Example: bash train.sh -i ./train/train_dev.txt -t ./train/test.txt 
