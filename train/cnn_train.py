"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import pandas as pd
import math
from conv_net_classes import LeNetConvPoolLayer
from conv_net_classes import MLPDropout


#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


def build_model(U,
                img_h,
                img_w=300, 
                filter_hs=[1,2,3],
                hidden_units=[100,10],
                dropout_rate=0.5,
                batch_size=50, 
                conv_non_linear="relu",
                activation=Iden,
                sqr_norm_lim=9,
                non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w), ("filter shape",filter_shapes), ("pool size", pool_sizes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("conv_non_linear", conv_non_linear), 
                    ("sqr_norm_lim",sqr_norm_lim)]
    print parameters


    
    #define model architecture
    x = T.imatrix('x')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    #WV = Words[x.flatten()]
    layer0_input = Words[x.flatten()].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))
    
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=[activation], dropout_rates=[dropout_rate])
    
    return x, y, Words, conv_layers, classifier



def test_conv_net(output_file,
                test_data,
                U,
                params,
                img_w=300, 
                filter_hs=[1,2,3],
                hidden_units=[100,10],
                dropout_rate=0.5,
                shuffle_batch=True,
                n_epochs=10,
                batch_size=1000, 
                lr_decay = 0.95,
                conv_non_linear="relu",
                activation=Iden,
                sqr_norm_lim=9,
                non_static=True):

    
    img_h = len(test_data[0])-1
    x, y, Words, conv_layers, classifier = build_model(
                U,
                img_h,
                img_w=img_w, 
                filter_hs=filter_hs,
                hidden_units=hidden_units, 
                dropout_rate=dropout_rate,
                batch_size=batch_size, 
                conv_non_linear=conv_non_linear,
                sqr_norm_lim=sqr_norm_lim,
                non_static=non_static)
    
    
    ### 1.1 parameters for classifier
    W = params.pop(0)
    b = params.pop(0)
    classifier.params[0].set_value(W.get_value())
    classifier.params[1].set_value(b.get_value())
    
    ### 1.2 parameters for conv_layers
    for conv_layer in conv_layers:
        W = params.pop(0)
        b = params.pop(0)
        conv_layer.W.set_value(W.get_value())
        conv_layer.b.set_value(b.get_value())
    if non_static == True:
        U2 = params.pop(0)
        Words.set_value(U2.get_value())
    
    ### 2. organize data
    minibatches = []
    minibatch_start = 0
    for i in range(test_data.shape[0] // batch_size):
        minibatches.append((minibatch_start,minibatch_start+batch_size))
        minibatch_start += batch_size
    if (minibatch_start != test_data.shape[0]):
       # minibatches.append((minibatch_start,test_data.shape[0]))
        
        fake_size = batch_size + minibatch_start - test_data.shape[0]
        fake_data = test_data[0:fake_size]
        minibatches.append((minibatch_start,minibatch_start+batch_size))
        test_data = np.concatenate((test_data, fake_data), axis=0)
    
    
    test_set_x, test_set_y = shared_dataset((test_data[:,:img_h], test_data[:,-1]))

    ### 3. set up eval function and evaluate
    s_idx = T.iscalar()
    e_idx = T.iscalar()
    test_model = theano.function([s_idx,e_idx], classifier.errors(y),
                                givens={
                                       x: test_set_x[s_idx : e_idx],
                                       y: test_set_y[s_idx : e_idx]})
    
    test_result = theano.function([s_idx,e_idx], classifier.layers[-1].y_pred,
                                givens={
                                       x: test_set_x[s_idx : e_idx]#,
                                       })
    
    
    losses = [test_model(minibatches[i][0], minibatches[i][1]) for i in xrange(len(minibatches))]
    

    test_losses = 0
    for i in xrange(len(losses)):
        test_losses += losses[i] * (minibatches[i][1]-minibatches[i][0])
    test_losses /= test_data.shape[0]
    test_perf = 1- test_losses
    print('test size ' + str(test_data.shape[0]))
    print('test perf %f %%' % (test_perf * 100.))


    

    test_preds = [test_result(minibatches[i][0], minibatches[i][1]) for i in xrange(len(minibatches))]
    with open(output_file, 'wb') as fout:
        for pred in test_preds:
            for p in pred:
                fout.write(str(p) + '\n')
    
def train_conv_net(train_data,
                model_file,
                U,
                img_w=300, 
                filter_hs=[1,2,3],
                hidden_units=[100,10],
                dropout_rate=0.5,
                shuffle_batch=True,
                n_epochs=10,
                batch_size=1000, 
                lr_decay = 0.95,
                conv_non_linear="relu",
                activation=Iden,
                sqr_norm_lim=9,
                non_static=True):

    img_h = len(train_data[0])-1  

    x, y, Words, conv_layers, classifier = build_model(
                U,
                img_h,
                img_w=img_w, 
                filter_hs=filter_hs,
                hidden_units=hidden_units, 
                dropout_rate=dropout_rate,
                batch_size=batch_size, 
                conv_non_linear=conv_non_linear,
                activation=activation,
                sqr_norm_lim=sqr_norm_lim,
                non_static=non_static)
    index = T.lscalar()
        
    # this is to make sure that the first word is always zero (when non-static word embedding)
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])



    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    
    n_batches = int(math.ceil(train_data.shape[0]/float(batch_size)))
    n_train_batches = int(np.round(n_batches*0.9))
    print 'n_batches: ', n_batches
    print 'n_train_batches: ', n_train_batches
    train_set = train_data[:n_train_batches*batch_size,:]
    val_set = train_data[n_train_batches*batch_size:,:]
    
    if val_set.shape[0] % batch_size > 0:
        extra_data_num = batch_size - val_set.shape[0] % batch_size
        new_set = np.append(val_set, val_set[:extra_data_num], axis=0)
        # might be possible that we still do not have the proper batch size - 
        # in that case - for remaining - add from "training" data
        val_set = new_set
        if val_set.shape[0] % batch_size > 0:
             extra_data_num = batch_size - val_set.shape[0] % batch_size
             new_set = np.append(val_set, train_set[:extra_data_num], axis=0)
             val_set = new_set

    print 'train size =', train_set.shape, ' val size =', val_set.shape 
    
    
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h], train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h], val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
        givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]})
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    cost_epoch = 0    
    
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        if shuffle_batch:
            j = 1
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)

        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch %i, last training batch %f, val perf %f %%' % (epoch, cost_epoch, val_perf*100.))

        
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            fin = open(model_file, "wb")
            cPickle.dump(params, fin)
    
    return best_val_perf

    


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype='int32'),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
        return shared_x, shared_y
        
def sgd_updates_adadelta(params,cost, rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        #if param.name!='Words':
        gp = T.grad(cost, param)
        gparams.append(gp)
        #else:
        #    gp_WV = T.grad(cost, WV)
        #    gp = T.inc_subtensor(WV, gp_WV) - param
        #    gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
    
def get_idx_from_sent(sent, word_idx_map, max_l=20, filter_h=3):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    i.e. for "i love going to the dentist" --> "0, 0, 10, 12, 345, 767, 13, 45, 0, 0..."
    i.e. for same length (with padding when necessary)
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words :
        
        if len(x) >= max_l+2*pad :
            break
        
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1) # unknown word
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data(data, word_idx_map, max_l=20, filter_h=3):
    """
    Transforms sentences into a 2-d matrix.
    """
    idx_data = []
    index = 0
    for query in data:
        idx_query = get_idx_from_sent(query["text"], word_idx_map, max_l, filter_h)   

        idx_query.append(query["y"])
        idx_data.append(idx_query)
            
        index = index+1
        
    idx_data = np.array(idx_data,dtype="int")
    return idx_data  

def test():
    train_file = './train/train_dev.pkl'
    model_file = './train/model.pkl'
    test_file = './train/input_test.pkl'
    output_file = './train/input_results.pred'
    
    batch_size = 1
    filter_hs = [1,2,3]
    hidden_units = [100,10]
    non_static = True

    params = cPickle.load(open(model_file,'rb'))
    test_data = cPickle.load(open(test_file,'rb'))
    x = cPickle.load(open(train_file,"rb"))
    _, W, word_idx_map, max_l = x[0], x[1], x[2], x[3]
    img_w = len(W[0])

    U = W.astype(theano.config.floatX)
    idx_test_data = make_idx_data(test_data, word_idx_map, max_l, filter_h=max(filter_hs))

    
    test_conv_net(output_file,
                  idx_test_data,
                  U,
                  params,
                  img_w = img_w,
                  lr_decay=0.95,
                  filter_hs=filter_hs,
                  conv_non_linear="relu",
                  hidden_units=hidden_units,
                  shuffle_batch=True, 
                  n_epochs=10,
                  sqr_norm_lim=9,
                  non_static=non_static,
                  batch_size=batch_size,
                  dropout_rate=0.5)
    


def train():

    train_file = './train/train_dev.pkl'
    model_file = './train/model.pkl'
    
    batch_size = 100
    filter_hs = [1,2,3]
    hidden_units = [100,10]
    non_static = True
    
    
    print "loading data...",
    
    x = cPickle.load(open(train_file,"rb"))
    train_data, W, word_idx_map, max_l = x[0], x[1], x[2], x[3]
    img_w = len(W[0])
    print "data loaded!"
    U = W.astype(theano.config.floatX)
    
    print "max length = " + str(max_l)

    execfile("./train/conv_net_classes.py")
    
    idx_train_data = make_idx_data(train_data, word_idx_map, max_l, filter_h=max(filter_hs))
    
    perf = train_conv_net(idx_train_data,
                          model_file, 
                          U,
                          img_w = img_w,
                          lr_decay=0.95,
                          filter_hs=filter_hs,
                          conv_non_linear="relu",
                          hidden_units=hidden_units, 
                          shuffle_batch=True, 
                          n_epochs=10,
                          sqr_norm_lim=9,
                          non_static=non_static,
                          batch_size=batch_size,
                          dropout_rate=0.5)
    print "test perf: " + str(perf)


   
if __name__=="__main__":
    train()
    test()

