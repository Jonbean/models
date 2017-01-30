import theano
import theano.tensor as T
import lasagne
import numpy as np

class LstmEncoder(object):
    def __init__(self, LSTMLAYER_UNITS, wemb_trainable = True):
        self.layer_units = LSTMLAYER_UNITS
        self.wemb = None
        self.GRAD_CLIP = 10.
        self.l_in = None
        self.l_mask = None
        self.output = None
        self.dropout_rate = 0.0
        self.bias = 0.001
        self.wemb_trainable = wemb_trainable

    def build_model(self, WordEmbedding_Init = None):

        self.wemb = WordEmbedding_Init

        #create symbolic representation of inputs, mask and target_value

        # l_in input shape ==> (n_batch, n_time_steps, n_features)
        # The number of feature dimensions is 1(index). 
        self.l_in = lasagne.layers.InputLayer(shape=(None, None, 1))

        # Masks input shape ==> (n_batch, n_time_steps)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))



        #setting gates and cell parameters with specific nonlinearity functions
        gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                        W_hid=lasagne.init.Orthogonal(),
                                                        b=lasagne.init.Constant(self.bias))

        cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                        W_hid=lasagne.init.Orthogonal(),
                                                        # Setting W_cell to None denotes that no cell connection will be used. 
                                                        W_cell=None, 
                                                        b=lasagne.init.Constant(self.bias),
                                                        # By convention, the cell nonlinearity is tanh in an LSTM. 
                                                        nonlinearity=lasagne.nonlinearities.tanh)

       # The embedding layers with retieve subtensor from word embedding matrix
        # l_emb = lasagne.layers.EmbeddingLayer(self.l_in, input_size=self.wemb.get_value().shape[0], output_size=self.wemb.get_value().shape[1], W=self.wemb)

        # l_emb.params[l_emb.W].remove('trainable')
            
        # l_drop = lasagne.layers.DropoutLayer(l_emb, p = self.dropout_rate)
        # The LSTM layer should have the same mask input in order to avoid padding entries
        l_emb = lasagne.layers.EmbeddingLayer(self.l_in, input_size=self.wemb.get_value().shape[0], output_size=self.wemb.get_value().shape[1], W=self.wemb)
        if not self.wemb_trainable:
            l_emb.params[l_emb.W].remove('trainable')

        l_lstm = lasagne.layers.recurrent.LSTMLayer(l_emb, 
                                                    num_units=self.layer_units[0],
                                                    mask_input=self.l_mask,
                                                    ingate=gate_parameters,
                                                    forgetgate=gate_parameters, 
                                                    cell=cell_parameters,
                                                    outgate=gate_parameters,
                                                    learn_init=True,
                                                    grad_clipping=self.GRAD_CLIP)


        l_lstm2 = lasagne.layers.recurrent.LSTMLayer(l_lstm, 
                                                    num_units=self.layer_units[1],
                                                    mask_input=self.l_mask,
                                                    ingate=gate_parameters,
                                                    forgetgate=gate_parameters, 
                                                    cell=gate_parameters,
                                                    outgate=gate_parameters,
                                                    learn_init=True,
                                                    grad_clipping=self.GRAD_CLIP)

        self.output = l_lstm2
        self.all_params = lasagne.layers.get_all_params(self.output)

