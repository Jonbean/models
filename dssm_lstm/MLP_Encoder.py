import theano
import theano.tensor as T
import lasagne



class MLPEncoder(object):
    def __init__(self, LAYER_1_UNITS, N_FEATURES):
        self.layer1_units = LAYER_1_UNITS
        self.n_features = N_FEATURES
        self.l_in = None
        self.output = None

    def build_model(self, input_dim):

        #create symbolic representation of inputs, mask and target_value

        # l_in input shape ==> (n_batch, n_time_steps, n_features)
        # The number of feature dimensions is 1(index). 
        self.l_in = lasagne.layers.InputLayer(shape=(None, input_dim, 1))


        l_hid1 = lasagne.layers.DenseLayer(self.l_in, num_units=self.layer1_units,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            W=lasagne.init.GlorotUniform())

        l_out = lasagne.layers.DenseLayer(l_hid1, num_units=self.n_features,
                                            nonlinearity=lasagne.nonlinearities.tanh)
        #we only record the output(shall we record each layer???)
        self.output = l_out
        self.all_params = lasagne.layers.get_all_params(l_out)