import theano
import theano.tensor as The
import lasagne
import numpy as np

class MlpClassifier(object):
    def __init__(self,story_input, end_input, MLP_UNITS):

        self.story_in = lasagne.layers.InputLayer(shape=(None, story_input))
        self.end_in = lasagne.layers.InputLayer(shape=(None, end_input))

        classif_in1 = lasagne.layers.ConcatLayer([self.story_in, self.end_in])


        l_hid1 = lasagne.layers.DenseLayer(classif_in1, num_units=MLP_UNITS[0],
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            W=lasagne.init.GlorotUniform())
        if len(MLP_UNITS) > 1:      
            l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=MLP_layer2,
                                                nonlinearity=lasagne.nonlinearities.tanh,
                                                W=lasagne.init.GlorotUniform())

            l_class = lasagne.layers.DenseLayer(l_hid2, num_units = 2,
                                                nonlinearity = lasagne.nonlinearities.softmax)
            
            l_out = l_class

            #we only record the output(shall we record each layer???)
            self.output = l_out
            self.all_params = lasagne.layers.get_all_params(l_out)
        else:
            l_class = lasagne.layers.DenseLayer(l_hid1, num_units = 2,
                                                nonlinearity = lasagne.nonlinearities.softmax)
            
            l_out = l_class

            #we only record the output(shall we record each layer???)
            self.output = l_out
            self.all_params = lasagne.layers.get_all_params(l_out)