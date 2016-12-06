import theano
import theano.tensor as T
import lasagne
import numpy as np

class BlstmEncoder(object):
    def __init__(self, 
                 LSTMLAYER_1_UNITS, 
                 LSTMLAYER_2_UNITS = 2,
                 wemb_trainable = True, 
                 mode = 'sequence'):
        self.layer1_units = LSTMLAYER_1_UNITS
        self.layer2_units = LSTMLAYER_2_UNITS
        self.wemb = None
        self.GRAD_CLIP = 10.
        self.bias = 0.001
        self.l_in = None
        self.l_mask = None
        self.output = None
        self.dropout_rate = 0.0
        self.wemb_trainable = wemb_trainable
        self.mode = mode


    def build_word_level(self, WordEmbedding_Init = None):

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
        l_emb = lasagne.layers.EmbeddingLayer(self.l_in, input_size=self.wemb.get_value().shape[0], output_size=self.wemb.get_value().shape[1], W=self.wemb)
        if not self.wemb_trainable:
            l_emb.params[l_emb.W].remove('trainable')
            
        # l_drop = lasagne.layers.DropoutLayer(l_emb, p = self.dropout_rate)
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


        # The back directional LSTM layers
        l_lstm_back = lasagne.layers.recurrent.LSTMLayer(l_emb,
                                                         num_units=self.layer1_units,
                                                         mask_input = self.l_mask,
                                                         ingate=gate_parameters, forgetgate=gate_parameters, 
                                                         cell=cell_parameters, outgate=gate_parameters,
                                                         # We'll learn the initialization and use gradient clipping 
                                                         learn_init=True, grad_clipping=self.GRAD_CLIP,
                                                         backwards=True
                                                        )


        # Do sum up of bidirectional LSTM results
        l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

        #here we shuffle the dimension of the 3D output of matrix of l_lstm2 because
        #pooling layer's gonna collapse the trailling axes
        #l_shuffle = lasagne.layers.DimshuffleLayer(l_sum, (0,2,1))

        #l_pooling = lasagne.layers.GlobalPoolLayer(l_shuffle)


        l_out = l_sum

        l_forward_last = lasagne.layers.SliceLayer(l_lstm, -1, 1)
        l_backward_last = lasagne.layers.SliceLayer(l_lstm_back, -1, 1)
        l_last_out = lasagne.layers.ElemwiseSumLayer([l_forward_last, l_backward_last])

        # l_out = lasagne.layers.SliceLayer(l_grurnn, -1, 1)
        #we only record the output(shall we record each layer???)
        if self.mode == "sequence":
            self.output = l_out
        else:
            self.output = l_last_out
        self.all_params = lasagne.layers.get_all_params(self.output)

    def build_sent_level(self, input_dim):
        #create symbolic representation of inputs, mask and target_value

        # l_in input shape ==> (n_batch, n_time_steps, n_features)
        # The number of feature dimensions is 1(index). 
        assert type(input_dim) == int

        self.l_in = lasagne.layers.InputLayer(shape=(None, None, input_dim))

        # Masks input shape ==> (n_batch, n_time_steps)

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

        # l_drop = lasagne.layers.DropoutLayer(l_emb, p = self.dropout_rate)
        # The LSTM layer should have the same mask input in order to avoid padding entries
        l_lstm = lasagne.layers.recurrent.LSTMLayer(self.l_in, 
                                                    num_units=self.layer1_units,
                                                    ingate=gate_parameters, 
                                                    forgetgate=gate_parameters, 
                                                    cell=cell_parameters, 
                                                    outgate=gate_parameters,
                                                    learn_init=True, 
                                                    grad_clipping=self.GRAD_CLIP
                                                    )


        # The back directional LSTM layers
        l_lstm_back = lasagne.layers.recurrent.LSTMLayer(self.l_in,
                                                         num_units=self.layer1_units,
                                                         mask_input = self.l_mask,
                                                         ingate=gate_parameters, 
                                                         forgetgate=gate_parameters, 
                                                         cell=cell_parameters, 
                                                         outgate=gate_parameters,
                                                         learn_init=True,
                                                         grad_clipping=self.GRAD_CLIP,
                                                         backwards=True
                                                        )


        # Do sum up of bidirectional LSTM results
        l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

        #here we shuffle the dimension of the 3D output of matrix of l_lstm2 because
        #pooling layer's gonna collapse the trailling axes
        #l_shuffle = lasagne.layers.DimshuffleLayer(l_sum, (0,2,1))

        #l_pooling = lasagne.layers.GlobalPoolLayer(l_shuffle)


        l_out = l_sum

        l_forward_last = lasagne.layers.SliceLayer(l_lstm, -1, 1)
        l_backward_last = lasagne.layers.SliceLayer(l_lstm_back, -1, 1)
        l_last_out = lasagne.layers.ElemwiseSumLayer([l_forward_last, l_backward_last])

        # l_out = lasagne.layers.SliceLayer(l_grurnn, -1, 1)
        #we only record the output(shall we record each layer???)
        if self.mode == "sequence":
            self.output = l_out
        else:
            self.output = l_last_out
        self.all_params = lasagne.layers.get_all_params(self.output)
    def build_score_func(self, input_dim):
        #create symbolic representation of inputs, mask and target_value

        # l_in input shape ==> (n_batch, n_time_steps, n_features)
        # The number of feature dimensions is 1(index). 
        assert type(input_dim) == int
        
        self.l_in = lasagne.layers.InputLayer(shape=(None, None, input_dim))

        # Masks input shape ==> (n_batch, n_time_steps)

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

        # l_drop = lasagne.layers.DropoutLayer(l_emb, p = self.dropout_rate)
        # The LSTM layer should have the same mask input in order to avoid padding entries
        l_lstm1 = lasagne.layers.recurrent.LSTMLayer(self.l_in, 
                                                    num_units=self.layer1_units,
                                                    ingate=gate_parameters, 
                                                    forgetgate=gate_parameters, 
                                                    cell=cell_parameters, 
                                                    outgate=gate_parameters,
                                                    learn_init=True, 
                                                    grad_clipping=self.GRAD_CLIP
                                                    )

        l_lstm = lasagne.layers.recurrent.LSTMLayer(l_lstm1, 
                                                    num_units = self.layer2_units,
                                                    ingate=gate_parameters, 
                                                    forgetgate=gate_parameters, 
                                                    cell=cell_parameters, 
                                                    outgate=gate_parameters,
                                                    learn_init=True, 
                                                    grad_clipping=self.GRAD_CLIP
                                                    )


        # The back directional LSTM layers
        l_lstm_back1 = lasagne.layers.recurrent.LSTMLayer(self.l_in,
                                                         num_units=self.layer1_units,
                                                         mask_input = self.l_mask,
                                                         ingate=gate_parameters, 
                                                         forgetgate=gate_parameters, 
                                                         cell=cell_parameters, 
                                                         outgate=gate_parameters,
                                                         learn_init=True,
                                                         grad_clipping=self.GRAD_CLIP,
                                                         backwards=True
                                                        )

        l_lstm_back = lasagne.layers.recurrent.LSTMLayer(l_lstm_back1,
                                                         num_units=self.layer2_units,
                                                         mask_input = self.l_mask,
                                                         ingate=gate_parameters, 
                                                         forgetgate=gate_parameters, 
                                                         cell=cell_parameters, 
                                                         outgate=gate_parameters,
                                                         learn_init=True,
                                                         grad_clipping=self.GRAD_CLIP,
                                                         backwards=True
                                                        )

        # Do sum up of bidirectional LSTM results
        l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])

        #here we shuffle the dimension of the 3D output of matrix of l_lstm2 because
        #pooling layer's gonna collapse the trailling axes
        #l_shuffle = lasagne.layers.DimshuffleLayer(l_sum, (0,2,1))

        #l_pooling = lasagne.layers.GlobalPoolLayer(l_shuffle)


        l_out = l_sum

        l_forward_last = lasagne.layers.SliceLayer(l_lstm, -1, 1)
        l_backward_last = lasagne.layers.SliceLayer(l_lstm_back, -1, 1)
        l_last_out = lasagne.layers.ElemwiseSumLayer([l_forward_last, l_backward_last])

        # l_out = lasagne.layers.SliceLayer(l_grurnn, -1, 1)
        #we only record the output(shall we record each layer???)
        if self.mode == "sequence":
            self.output = T.sum(l_out, axis = 1)
        else:
            self.output = l_last_out
        self.all_params = lasagne.layers.get_all_params(self.output)
