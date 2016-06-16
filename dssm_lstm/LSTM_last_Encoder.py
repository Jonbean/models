import theano 
import theano.tensor as T
import lasagne
import numpy as np 


class LstmEncoder(object):
    def __init__(self, LAYER_1_UNITS):
        self.layer1_units = LAYER_1_UNITS
        self.wemb = None
        self.GRAD_CLIP = 100.
        self.l_in = None
        self.l_mask = None
        self.output = None

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
                                                        b=lasagne.init.Constant(0.))

        cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                        W_hid=lasagne.init.Orthogonal(),
                                                        # Setting W_cell to None denotes that no cell connection will be used. 
                                                        W_cell=None, 
                                                        b=lasagne.init.Constant(0.),
                                                        # By convention, the cell nonlinearity is tanh in an LSTM. 
                                                        nonlinearity=lasagne.nonlinearities.tanh)

        # The embedding layers with retieve subtensor from word embedding matrix
        l_emb = lasagne.layers.EmbeddingLayer(self.l_in, input_size=self.wemb.get_value().shape[0], output_size=self.wemb.get_value().shape[1], W=self.wemb)

        # The LSTM layer should have the same mask input in order to avoid padding entries
        l_lstm = lasagne.layers.recurrent.LSTMLayer(l_emb, 
                                                    num_units=self.layer1_units,
                                                    # We need to specify a separate input for masks
                                                    mask_input=self.l_mask,
                                                    # Here, we supply the gate parameters for each gate 
                                                    ingate=gate_parameters, forgetgate=gate_parameters, 
                                                    cell=cell_parameters, outgate=gate_parameters,
                                                    # We'll learn the initialization and use gradient clipping 
                                                    learn_init=True, grad_clipping=self.GRAD_CLIP
                                                    )

        # The second LSTM layer on top of the first LSTM layer


        #here we shuffle the dimension of the 3D output of matrix of l_lstm2 because
        #pooling layer's gonna collapse the trailling axes
        l_shuffle = lasagne.layers.DimshuffleLayer(l_lstm, (0,2,1))

        l_pooling = lasagne.layers.GlobalPoolLayer(l_shuffle)

        l_out = l_pooling

        #we only record the output(shall we record each layer???)
        self.output = l_out
        self.all_params = lasagne.layers.get_all_params(l_out)