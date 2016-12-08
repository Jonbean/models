import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import BLSTM_sequence
import sys
# from theano.printing import pydotprint




class Hierachi_RNN(object):
    def __init__(self, 
                 rnn_setting, 
                 batchsize, 
                 val_split_ratio, 
                 score_func_nonlin = 'default',
                 wemb_trainable = '0',
                 wemb_size = None):
        # Initialize Theano Symbolic variable attributes
        self.story_input_variable = None
        self.story_mask = None
        self.story_nsent = 4

        self.cost = None

        self.train_func = None

        # Initialize data loading attributes
        self.wemb = None
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 

        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'

        self.rnn_units = int(rnn_setting)
        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.batchsize = int(batchsize)

        self.val_split_ratio = float(val_split_ratio)

        self.score_func_nonlin = None
        if score_func_nonlin == 'default':
            self.score_func_nonlin = lasagne.nonlinearities.tanh

        self.wemb_trainable = bool(int(wemb_trainable))
        
        self.wemb_size = 300
        if wemb_size == None:
            self.random_init_wemb = False
        else:
            self.random_init_wemb = True
            self.wemb_size = int(wemb_size)

        self.train_story = None
        self.train_ending = None

        self.val_story = None
        self.val_ending1 = None 
        self.val_ending2 = None
        self.val_answer = None
        self.n_val = None

        self.test_story = None 
        self.test_ending1 = None
        self.test_ending2 = None
        self.test_answer = None
        self.n_test = None

        self.train_encodinglayer_vecs = []


    def encoding_layer(self):


        assert len(self.reshaped_inputs_variables)==len(self.inputs_masks)
        for i in range(self.story_nsent+2):
            sent_seq = lasagne.layers.get_output(self.encoder.output,
                                                {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                 self.encoder.l_mask:self.inputs_masks[i]},
                                                 deterministic = True)
            sent_ave = (sent_seq * self.inputs_masks[i].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[i].sum(axis = 1, keepdims = True)
            self.train_encodinglayer_vecs.append(sent_ave)

    def model_constructor(self, wemb_size = None):
        self.inputs_variables = []
        self.inputs_masks = []
        self.reshaped_inputs_variables = []
        for i in range(self.story_nsent+2):
            self.inputs_variables.append(T.matrix('story'+str(i)+'_input', dtype='int64'))
            self.inputs_masks.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            batch_size, seqlen = self.inputs_variables[i].shape
            self.reshaped_inputs_variables.append(self.inputs_variables[i].reshape([batch_size, seqlen, 1]))

        #initialize neural network units
        self.encoder = BLSTM_sequence.BlstmEncoder(LSTMLAYER_1_UNITS = self.rnn_units, wemb_trainable = self.wemb_trainable)
        self.encoder.build_model(self.wemb)

        #build encoding layer
        self.encoding_layer()

        #build reasoning layers

        self.merge_ls = [T.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1])) for tensor in self.train_encodinglayer_vecs[:4]]

        encode_merge = T.concatenate(self.merge_ls, axis = 1)

        l_in = lasagne.layers.InputLayer(shape=(None, None, self.rnn_units))
        gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                        W_hid=lasagne.init.Orthogonal(),
                                                        b=lasagne.init.Constant(0.001))

        cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                        W_hid=lasagne.init.Orthogonal(),
                                                        # Setting W_cell to None denotes that no cell connection will be used. 
                                                        W_cell=None, 
                                                        b=lasagne.init.Constant(0.001),
                                                        # By convention, the cell nonlinearity is tanh in an LSTM. 
                                                        nonlinearity=lasagne.nonlinearities.tanh)

        l_lstm = lasagne.layers.recurrent.LSTMLayer(l_in, 
                                                    num_units=self.rnn_units,
                                                    # Here, we supply the gate parameters for each gate 
                                                    ingate=gate_parameters, forgetgate=gate_parameters, 
                                                    cell=cell_parameters, outgate=gate_parameters,
                                                    # We'll learn the initialization and use gradient clipping 
                                                    learn_init=True)

        # The back directional LSTM layers
        l_lstm_back = lasagne.layers.recurrent.LSTMLayer(l_in,
                                                         num_units=self.rnn_units,
                                                         ingate=gate_parameters, forgetgate=gate_parameters, 
                                                         cell=cell_parameters, outgate=gate_parameters,
                                                         # We'll learn the initialization and use gradient clipping 
                                                         learn_init=True,
                                                         backwards=True)

        # Do sum up of bidirectional LSTM results
        l_out_right = lasagne.layers.SliceLayer(l_lstm, -1, 1)
        l_out_left = lasagne.layers.SliceLayer(l_lstm_back, -1, 1)
        l_sum = lasagne.layers.ElemwiseSumLayer([l_out_right, l_out_left])

        reasoner_result = lasagne.layers.get_output(l_sum, {l_in: encode_merge}, deterministic = True)
        reasoner_params = lasagne.layers.get_all_params(l_sum)


        l_in = lasagne.layers.InputLayer(shape=(None, self.rnn_units * 2))
        
        l_hid1 = lasagne.layers.DenseLayer(l_in, num_units = 512, nonlinearity=lasagne.nonlinearities.tanh)
        l_hid = lasagne.layers.DenseLayer(l_hid1, num_units = 1, nonlinearity=self.score_func_nonlin)

        final_class_param = lasagne.layers.get_all_params(l_hid)

        input_matrix1 = T.concatenate((reasoner_result, self.train_encodinglayer_vecs[-2]), axis = 1)
        input_matrix2 = T.concatenate((reasoner_result, self.train_encodinglayer_vecs[-1]), axis = 1)

        score1 = lasagne.layers.get_output(l_hid, {l_in: input_matrix1})
        score2 = lasagne.layers.get_output(l_hid, {l_in: input_matrix2})


        # Construct symbolic cost function
        target1 = T.vector('gold_target1', dtype= theano.config.floatX)
        target2 = T.vector('gold_target2', dtype= theano.config.floatX)

        cost = target1 * score1 + target2 * score2 + 1.0
        
        dnn_penalty = [lasagne.regularization.l2(param) for param in final_class_param]

        dnn_penalty_mean = lasagne.objectives.aggregate(T.stack(dnn_penalty), mode = 'mean')

        self.cost = lasagne.objectives.aggregate(cost + 0.0001*dnn_penalty_mean, mode='mean')

        # Retrieve all parameters from the network
        all_params = self.encoder.all_params + reasoner_params + final_class_param

        all_updates = lasagne.updates.adam(self.cost, all_params, learning_rate=0.001)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        self.train_func = theano.function(self.inputs_variables + self.inputs_masks + [target1, target2], 
                                         [self.cost, score1, score2], updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function(self.inputs_variables + self.inputs_masks, [score1, score2])
        # pydotprint(self.train_func, './computational_graph.png')

    def load_data(self):
        # train_set = pickle.load(open(self.train_set_path))
        # self.train_story = train_set[0]
        # self.train_ending = train_set[1]

        val_set = pickle.load(open(self.val_set_path))

        self.val_story = val_set[0]
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.val_len = len(self.val_answer)
        self.val_train_n = int(self.val_split_ratio * self.val_len)
        self.val_test_n = self.val_len - self.val_train_n

        shuffled_indices_ls = utils.shuffle_index(len(self.val_answer))

        self.val_train_ls = shuffled_indices_ls[:self.val_train_n]
        self.val_test_ls = shuffled_indices_ls[self.val_train_n:]


        test_set = pickle.load(open(self.test_set_path))
        self.test_story = test_set[0]
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_answer)

        if self.random_init_wemb:
            wemb = np.random.rand(28820 ,self.wemb_size)
            wemb = np.concatenate((np.zeros((1, self.wemb_size)), wemb), axis = 0)
            self.wemb = theano.shared(wemb).astype(theano.config.floatX)
        else:
            self.wemb = theano.shared(pickle.load(open(self.wemb_matrix_path))).astype(theano.config.floatX)


    def val_set_test(self):

        correct = 0.

        for i in self.val_test_ls:
            story = [np.asarray(sent, dtype='int64').reshape((1,-1)) for sent in self.val_story[i]]
            story_mask = [np.ones((1,len(self.val_story[i][j]))) for j in range(4)]

            ending1 = np.asarray(self.val_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.val_ending1[i])))

            ending2 = np.asarray(self.val_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.val_ending2[i])))

            score1, score2 = self.prediction(story[0], story[1], story[2], story[3], 
                                             ending1, ending2, story_mask[0], story_mask[1], story_mask[2],
                                             story_mask[3], ending1_mask, ending2_mask)
            
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))


            if prediction == self.val_answer[i]:
                correct += 1.


        return correct/self.val_test_n

    def test_set_test(self):
        #load test set data
        correct = 0.

        for i in range(self.n_test):
            story = [np.asarray(sent, dtype='int64').reshape((1,-1)) for sent in self.test_story[i]]
            story_mask = [np.ones((1,len(self.test_story[i][j]))) for j in range(4)]

            ending1 = np.asarray(self.test_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.test_ending1[i])))

            ending2 = np.asarray(self.test_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.test_ending2[i])))

            score1, score2 = self.prediction(story[0], story[1], story[2], story[3], 
                                             ending1, ending2, story_mask[0], story_mask[1], story_mask[2],
                                             story_mask[3], ending1_mask, ending2_mask)
            
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            if prediction == self.test_answer[i]:
                correct += 1.


        return correct/self.n_test


    def saving_model(self, val_or_test, accuracy):
        reason_params_value = lasagne.layers.get_all_param_values(self.reason_layer.output)
        classif_params_value = lasagne.layers.get_all_param_values(self.classify_layer.output)

        if val_or_test == 'val':
            pickle.dump((reason_params_value, classif_params_value, accuracy), 
                        open(self.best_val_model_save_path, 'wb'))
        else:
            pickle.dump((reason_params_value, classif_params_value, accuracy), 
                        open(self.best_test_model_save_path, 'wb'))            

    def reload_model(self, val_or_test):
        if val_or_test == 'val': 

            reason_params, classif_params, accuracy = pickle.load(open(self.best_val_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
            lasagne.layers.set_all_param_values(self.classify_layer.output, classif_params)

            print "This model has ", accuracy * 100, "%  accuracy on valid set" 
        else:
            reason_params, classif_params, accuracy = pickle.load(open(self.best_test_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
            lasagne.layers.set_all_param_values(self.classify_layer.output, classif_params_value)
            print "This model has ", accuracy * 100, "%  accuracy on test set" 

    def begin_train(self):
        N_EPOCHS = 100
        N_BATCH = self.batchsize
        N_TRAIN_INS = self.val_train_n
        best_val_accuracy = 0
        best_test_accuracy = 0

        """init test"""
        print "initial test"
        val_result = self.val_set_test()
        print "valid set accuracy: ", val_result*100, "%"
        if val_result > best_val_accuracy:
            print "new best! test on test set..."
            best_val_accuracy = val_result

        test_accuracy = self.test_set_test()
        print "test set accuracy: ", test_accuracy * 100, "%"
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        """end of init test"""
        
        for epoch in range(N_EPOCHS):
            print "epoch ", epoch,":"
            shuffled_index_list = self.val_train_ls
            np.random.shuffle(shuffled_index_list)

            max_batch = N_TRAIN_INS/N_BATCH

            start_time = time.time()

            total_cost = 0.0
            total_err_count = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[self.val_story[index][i] for index in batch_index_list] for i in range(self.story_nsent)]
                end1 = [self.val_ending1[index] for index in batch_index_list]
                end2 = [self.val_ending2[index] for index in batch_index_list]
                # neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # while np.any((np.asarray(batch_index_list) - neg_end_index_list) == 0):
                #     neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # neg_end1 = [self.train_ending[index] for index in neg_end_index_list]
                # answer = np.random.randint(2, size = N_BATCH)
                # target1 = 1 - answer
                # target2 = 1 - target1
                answer = np.asarray([self.val_answer[index] for index in batch_index_list])

                target1 = 2 * answer - 1
                target2 = -2 * answer + 1


                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end1_matrix = utils.padding(end1)
                train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end1_mask = utils.mask_generator(end1)
                train_end2_mask = utils.mask_generator(end2)
                

                cost, prediction1, prediction2 = self.train_func(train_story_matrices[0],
                                                                 train_story_matrices[1],
                                                                 train_story_matrices[2],
                                                                 train_story_matrices[3],
                                                                 train_end1_matrix,
                                                                 train_end2_matrix,
                                                                 train_story_mask[0],
                                                                 train_story_mask[1],
                                                                 train_story_mask[2],
                                                                 train_story_mask[3],
                                                                 train_end1_mask,
                                                                 train_end2_mask,
                                                                 target1,
                                                                 target2)



                prediction = np.argmax(np.concatenate((prediction1, prediction2), axis = 1), axis = 1)


                total_err_count += (abs(prediction - answer)).sum()


                total_cost += cost

            print "======================================="
            print "epoch summary:"
            print "total cost in this epoch: ", total_cost
            print "accuracy on training set: ", (1.0-(total_err_count / N_TRAIN_INS)) * 100, "%"
            val_result = self.val_set_test()
            print "accuracy is: ", val_result*100, "%"
            if val_result > best_val_accuracy:
                print "new best! test on test set..."
                best_val_accuracy = val_result

            test_accuracy = self.test_set_test()
            print "test set accuracy: ", test_accuracy * 100, "%"
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
            print "======================================="


   
if __name__ == '__main__':

    model = Hierachi_RNN(*sys.argv[1:])

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
 

