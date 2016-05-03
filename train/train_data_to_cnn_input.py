import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os.path

from collections import Counter


"""
    for input file, assume 1st column is label (int), and 2nd column is query (string)
    """
def read_data_file(data_file,max_l):
    queries = []
    change_count = 0
    
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            [label, text] = line.split('\t');
            text = text.lower()
            text = text.strip()
            words = text.split()
                        
            if len(words) > max_l:
                words = words[:max_l]
                change_count += 1
            
            datum = {"y": int(label),
                "text": " ".join(words),
                "num_words": len(words)}

            queries.append(datum)
    
    return queries


def get_W(word_vecs):
    """
        Get word matrix. W[i] is the vector for word indexed by i
        """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')
    
    W[0] = np.zeros(k) # 1st word is all zeros (for padding)
    W[1] = np.random.normal(0,0.17,k) # 2nd word is unknown word
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print 'vocab size =', vocab_size, ' k =', layer1_size
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def train(vocab, training_file, w2v_f):
    
    train_file = training_file
    output_file = './train/train_dev.pkl'
    max_l = 100
    
    np.random.seed(4321)
    
    print "loading training data...",
    train_data = read_data_file(train_file, max_l)
    
    
    max_l = np.max(pd.DataFrame(train_data)['num_words'])
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    
    w2v_file = w2v_f
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))

    W, word_idx_map = get_W(w2v)
    cPickle.dump([train_data, W, word_idx_map, max_l], open(output_file, "wb"))
    print "dataset created!"


def test(test_filename):
    test_file = test_filename
    output_test_file = './train/input_test.pkl'
    
    max_l = 100
    test_data = read_data_file(test_file, max_l)
    cPickle.dump(test_data, open(output_test_file, "wb"))



def create_vocab(training_file):
    
    cutoff = 0
    vocab = defaultdict(float)

    train_file = train_filename
    
    lines = open(train_file).readlines()
    raw = [process_line(l) for l in lines ]
    cntx = Counter( [ w for e in raw for w in e ] )
    lst = [ x for x, y in cntx.iteritems() if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
    writer = open ('./train/all_vocab.txt', 'w')
    for key in vocab:
        writer.write(key + '\t' + str(vocab.get(key)))
        writer.write('\n')

    writer.close()

    return vocab # (this is a dictionary of [word] = [position] which is fine since we are only bothered about the key of the dict.


def process_line(line):
    [label, text] = line.split('\t')
    return text.split()

if __name__=="__main__":
    train_filename = sys.argv[1]
    test_filename = ""
    w2v_file = ""
    
    if len(sys.argv) > 2:
        test_filename = sys.argv[2]
    if len(sys.argv) > 3:
        w2v_file = sys.argv[3]
    
    vocab = create_vocab(train_filename)
    train(vocab, train_filename, w2v_file)
    test(test_filename)


