import theano
import theano.tensor as T
import lasagne
import numpy as np

class LstmEncoder(object):
    def __init__(self, batch_size, max_len, LSTMLAYER_UNITS, wemb_trainable = True):
        self.layer_units = LSTMLAYER_UNITS
        self.wemb = None
        self.GRAD_CLIP = 10.
        self.l_in = None
        self.l_mask = None
        self.output = None
        self.dropout_rate = 0.0
        self.bias = 0.001
        self.wemb_trainable = wemb_trainable
        self.batch_size = batch_size
        self.max_len = max_len

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


        # Now, squash the n_batch and n_time_steps dimensions
        l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (-1, self.layer_units[0]))
        # Now, we can apply feed-forward layers as usual.
        # We want the network to predict a single value, the sum, so we'll use a single unit.
        l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=self.layer_units[1],
                                            nonlinearity=lasagne.nonlinearities.rectify)
        # Now, the shape will be n_batch*n_timesteps, 1. We can then reshape to
        # n_batch, n_timesteps to get a single value for each timstep from each sequence
        l_out = lasagne.layers.ReshapeLayer(l_dense, (self.batch_size, self.max_len, self.layer_units[1]))

        self.output = l_out

        self.all_params = lasagne.layers.get_all_params(self.output)

