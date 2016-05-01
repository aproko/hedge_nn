This code is based on Yoon Kim's implementation of neural nets (github.com/yoonkim/CNN_sentence, with additions by Debanjan Ghosh.

1. The lists of single-word hedge cues and multi-word hedge cues are contained in dictionary.csv and multiword_dict.csv respectively. The formats of these files are the following:

Hedge, Hedge Type (hProp/hRel), Definition of the Hedge Type, Hedge Use Definition, Hedge Use Example, Non-Hedge Use Definition, Non-Hedge Use Example, whether this word should be stemmed when looking for hedges, Probability (from gold labeled data, what is the probability that this word is a hedge)

2. test.sh - the shell script that runs a saved model on an input text file (a default saved model is included here, in ./test/model.pkl, but you can modify the shell script to run it on any saved model - but be sure to change the appropriate neural net  parameters in the function test in cnn_test.py). The input has to be a text file with a single sentence in it (example input file is in ./test/INPUT.txt). 

To test this function, just run bash test.sh.

3. train.sh - the shell script allows you to train a new model. We've included our training data in ./train/train_dev.txt, but it can be replaced with any data in the format: Label \t Text. We've also included our test data in test.txt, which allow us to test the performance of the model on a held out dataset with gold standard labels. 

To test this function, just run bash train.sh.