import theano
import theano.tensor as T
import lasagne
import numpy as np

class DNN(object):
    def __init__(self, 
                 INPUTS_SIZE, 
                 LAYER_UNITS, 
                 INPUTS_PARTS, 
                 final_nonlin = lasagne.nonlinearities.tanh):

        self.inputs_parts = INPUTS_PARTS
        self.inputs_size = INPUTS_SIZE
        self.layer_units = LAYER_UNITS
        self.final_nonlin = final_nonlin
        self.hidden_layers = []
        self.dropout_rate = 0.0

        self.l_in = lasagne.layers.InputLayer(shape=(None, self.inputs_size * self.inputs_parts))
        self.hidden_layers.append(self.l_in)
        for i in range(len(self.layer_units)-1):
            self.hidden_layers.append(lasagne.layers.DenseLayer(self.hidden_layers[i], 
                                        num_units=self.layer_units[i],
                                        nonlinearity=lasagne.nonlinearities.tanh))


        self.output = lasagne.layers.DenseLayer(self.hidden_layers[-1], self.layer_units[-1], nonlinearity=self.final_nonlin)
        self.all_params = lasagne.layers.get_all_params(self.output)
