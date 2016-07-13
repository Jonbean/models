import theano
import theano.tensor as T
import lasagne



class MLPEncoder(object):
    def __init__(self, LAYER_UNITS, classifier = False):
        self.layer1_units = LAYER_UNITS[0]
        self.layer2_units = None
        self.layer3_units = None
        if len(LAYER_UNITS) > 1:
            self.layer2_units = LAYER_UNITS[1]
        if len(LAYER_UNITS) > 2:
            self.layer3_units = LAYER_UNITS[2]
        self.l_in = None
        self.output = None
        self.classifier = False
        if classifier:
            self.classifier = True

    def build_model(self, input_dim):

        #create symbolic representation of inputs, mask and target_value

        # l_in input shape ==> (n_batch, n_time_steps, n_features)
        # The number of feature dimensions is 1(index). 
        self.story_in = lasagne.layers.InputLayer(shape=(None, input_dim))
        self.end_in = lasagne.layers.InputLayer(shape=(None, input_dim))
        l_concat = lasagne.layers.ConcatLayer([self.story_in, self.end_in], axis = 1)

        l_hid1 = lasagne.layers.DenseLayer(l_concat, num_units=self.layer1_units,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            W=lasagne.init.GlorotUniform())
        
        if self.layer3_units != None:
            l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=self.layer2_units,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            W=lasagne.init.GlorotUniform())
            l_hid3 = lasagne.layers.DenseLayer(l_hid2, num_units=self.layer3_units,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            W=lasagne.init.GlorotUniform())
            if self.classifier:
                self.output = lasagne.layers.DenseLayer(l_hid3, num_units=2,
                                            nonlinearity=lasagne.nonlinearities.softmax,
                                            W=lasagne.init.GlorotUniform())
            else:
                self.output = l_hid3

        elif self.layer3_units == None and self.layer2_units != None:
            l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=self.layer2_units,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            W=lasagne.init.GlorotUniform())
            if self.classifier:
                self.output = lasagne.layers.DenseLayer(l_hid2, num_units=2,
                                            nonlinearity=lasagne.nonlinearities.softmax,
                                            W=lasagne.init.GlorotUniform())
            else:
                self.output = l_hid2
        else:
            if self.classifier:
                self.output = lasagne.layers.DenseLayer(l_hid1, num_units=2,
                                            nonlinearity=lasagne.nonlinearities.softmax,
                                            W=lasagne.init.GlorotUniform())
            else:
                self.output = l_hid1
        self.all_params = lasagne.layers.get_all_params(self.output)

