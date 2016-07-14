import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import RNN_Encoder
import DNN_Encoder
import sys
# from theano.printing import pydotprint


class PoolingLayer(object):
    def __init__(self, in_dim, num, dropout_rate = 0.0):

        self.dropout_rate = dropout_rate

        self.l_in = [] 
        l_reshape = []
        for i in range(num):
            self.l_in.append(lasagne.layers.InputLayer(shape=(None, in_dim)))
            l_reshape.append(lasagne.layers.ReshapeLayer(self.l_in[i], shape=([0], [1], 1)))

        l_concate = lasagne.layers.ConcatLayer(l_reshape, axis = 2)

        l_pool = lasagne.layers.GlobalPoolLayer(l_concate)
        self.output = l_pool


class Neural_Reasoner_Model(object):
    def __init__(self, rnn_setting, mlp_setting, dropout_rate, batchsize, reasoning_depth, wemb_size = None):
        # Initialize Theano Symbolic variable attributes
        self.story_input_variable = None
        self.story_mask = None
        self.story_nsent = 4

        self.ending_input_variable = None
        self.ending_mask = None

        self.neg_ending1_input_variable = None
        self.neg_ending1_mask = None

        self.doc_encoder = None
        self.query_encoder = None 

        self.doc_encode = None 
        self.pos_end_encode = None
        self.neg_end_encode = None

        self.cost_vec = None
        self.cost = None

        self.train_func = None
        self.compute_cost = None

        # Initialize data loading attributes
        self.wemb = None
        self.train_set_path = '../../data/pickles/train_index_corpus.pkl'
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 

        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.rnn_units = int(rnn_setting)
        self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.dropout_rate = float(dropout_rate)
        self.batchsize = int(batchsize)
        self.reasoning_depth = int(reasoning_depth)
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
        self.test_encodinglayer_vecs = []
        self.reasoninglayer_vec1 = []
        self.reasoninglayer_vec2 = []
        self.reasoninglayer_vec1_test = []
        self.reasoninglayer_vec2_test = []
        self.reasoning_pool_results = []
        self.reasoning_pool_results_test = []
        self.reasoners = []


    def encoding_layer(self):


        assert len(self.reshaped_inputs_variables)==len(self.inputs_masks)
        for i in range(self.story_nsent+2):
            self.train_encodinglayer_vecs.append(lasagne.layers.get_output(self.encoder.output,
                                                        {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                         self.encoder.l_mask:self.inputs_masks[i]},
                                                         deterministic = True))

            # self.test_encodinglayer_vecs.append(lasagne.layers.get_output(self.encoder.output,{self.encoder.l_in:self.reshaped_inputs_variables[i], 
            #                                              self.encoder.l_mask:self.inputs_masks[i]},deterministic = True))
            

    def reasoning_layer(self, reasoning_depth):
        self.reasoninglayer_vec1.append([])
        self.reasoninglayer_vec2.append([])
        reasoner = DNN_Encoder.MLPEncoder(LAYER_UNITS = self.mlp_units)
        reasoner.build_model(self.rnn_units)
        self.reasoners.append(reasoner)
        # self.reasoninglayer_vec1_test.append([])
        # self.reasoninglayer_vec2_test.append([])

        if reasoning_depth == 0:        
            for i in range(self.story_nsent):
                self.reasoninglayer_vec1[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,{self.reasoners[reasoning_depth].story_in: self.train_encodinglayer_vecs[i],
                                                                             self.reasoners[reasoning_depth].end_in: self.train_encodinglayer_vecs[-2]}, deterministic = True))

                self.reasoninglayer_vec2[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,{self.reasoners[reasoning_depth].story_in: self.train_encodinglayer_vecs[i],
                                                                             self.reasoners[reasoning_depth].end_in: self.train_encodinglayer_vecs[-1]}, deterministic = True))

                # self.reasoninglayer_vec1_test[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,{self.reasoners[reasoning_depth].story_in: self.test_encodinglayer_vecs[i],
                #                                                              self.reasoners[reasoning_depth].end_in: self.test_encodinglayer_vecs[-2]}, deterministic = True))

                # self.reasoninglayer_vec2_test[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,{self.reasoners[reasoning_depth].story_in: self.test_encodinglayer_vecs[i],
                #                                                              self.reasoners[reasoning_depth].end_in: self.test_encodinglayer_vecs[-1]}, deterministic = True))
        #do mean pooling to generate new ending encode vec

           
            encode_poolresult1 = lasagne.layers.get_output(self.pooling.output, 
                                                            {self.pooling.l_in[i]:self.reasoninglayer_vec1[reasoning_depth][i] for i in range(self.story_nsent)},
                                                            deterministic = True)
            encode_poolresult2 = lasagne.layers.get_output(self.pooling.output, 
                                                            {self.pooling.l_in[i]:self.reasoninglayer_vec2[reasoning_depth][i] for i in range(self.story_nsent)},
                                                            deterministic = True)
            # encode_poolresult1_test = lasagne.layers.get_output(self.pooling.output, 
            #                                                     {self.pooling.l_in[i]:self.reasoninglayer_vec1_test[reasoning_depth][i] for i in range(self.story_nsent)},
            #                                                     deterministic = True)
            # encode_poolresult2_test = lasagne.layers.get_output(self.pooling.output, 
            #                                                     {self.pooling.l_in[i]:self.reasoninglayer_vec2_test[reasoning_depth][i] for i in range(self.story_nsent)},
            #                                                     deterministic = True)
            self.reasoning_pool_results.append([encode_poolresult1, encode_poolresult2])
            # self.reasoning_pool_results_test.append([encode_poolresult1_test, encode_poolresult2_test])
        else:
            for i in range(self.story_nsent):
                self.reasoninglayer_vec1[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,
                                                                            {self.reasoners[reasoning_depth].story_in: self.train_encodinglayer_vecs[i],
                                                                            self.reasoners[reasoning_depth].end_in: self.reasoning_pool_results[reasoning_depth - 1][-2]}, 
                                                                            deterministic = True))

                self.reasoninglayer_vec2[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,
                                                                            {self.reasoners[reasoning_depth].story_in: self.train_encodinglayer_vecs[i],
                                                                            self.reasoners[reasoning_depth].end_in: self.reasoning_pool_results[reasoning_depth - 1][-1]},
                                                                            deterministic = True))

                # self.reasoninglayer_vec1_test[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,
                #                                                             {self.reasoners[reasoning_depth].story_in: self.test_encodinglayer_vecs[i],
                #                                                             self.reasoners[reasoning_depth].end_in: self.reasoning_pool_results_test[reasoning_depth - 1][-2]}, 
                #                                                             deterministic = True))

                # self.reasoninglayer_vec2_test[reasoning_depth].append(lasagne.layers.get_output(self.reasoners[reasoning_depth].output,
                #                                                             {self.reasoners[reasoning_depth].story_in: self.test_encodinglayer_vecs[i],
                #                                                             self.reasoners[reasoning_depth].end_in: self.reasoning_pool_results_test[reasoning_depth - 1][-1]}, 
                #                                                             deterministic = True))
        #do mean pooling to generate new ending encode vec

           
            encode_poolresult1 = lasagne.layers.get_output(self.pooling.output, 
                                                            {self.pooling.l_in[i]:self.reasoninglayer_vec1[reasoning_depth][i] for i in range(self.story_nsent)},
                                                            deterministic = True)
            encode_poolresult2 = lasagne.layers.get_output(self.pooling.output, 
                                                            {self.pooling.l_in[i]:self.reasoninglayer_vec2[reasoning_depth][i] for i in range(self.story_nsent)},
                                                            deterministic = True)
            # encode_poolresult1_test = lasagne.layers.get_output(self.pooling.output, 
            #                                                     {self.pooling.l_in[i]:self.reasoninglayer_vec1_test[reasoning_depth][i] for i in range(self.story_nsent)},
            #                                                     deterministic = True)
            # encode_poolresult2_test = lasagne.layers.get_output(self.pooling.output, 
            #                                                     {self.pooling.l_in[i]:self.reasoninglayer_vec2_test[reasoning_depth][i] for i in range(self.story_nsent)},
            #                                                     deterministic = True)
            self.reasoning_pool_results.append([encode_poolresult1, encode_poolresult2])
            # self.reasoning_pool_results_test.append([encode_poolresult1_test, encode_poolresult2_test])        


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
        self.encoder = RNN_Encoder.RNNEncoder(LAYER_1_UNITS = self.rnn_units, dropout_rate = self.dropout_rate)
        self.encoder.build_model(self.wemb)
        self.decoder = RNN_Encoder.RNNEncoder(LAYER_1_UNITS = self.rnn_units, dropout_rate = self.dropout_rate)
        self.decoder.build_model(self.wemb)

        self.pooling = PoolingLayer(self.mlp_units[-1], self.story_nsent)
        #build encoding layer
        self.encoding_layer()

        #build reasoning layers
        for i in range(self.reasoning_depth):
            self.reasoning_layer(i)

        l_classin = lasagne.layers.InputLayer(shape=(None, self.rnn_units))

        l_hid = lasagne.layers.DenseLayer(l_classin, num_units=2,
                                            nonlinearity=lasagne.nonlinearities.tanh)

        final_class_param = lasagne.layers.get_all_params(l_hid)

        score1 = lasagne.layers.get_output(l_hid, {l_classin: self.reasoning_pool_results[-1][-2]})
        score2 = lasagne.layers.get_output(l_hid, {l_classin: self.reasoning_pool_results[-1][-1]})

        prob1 = lasagne.nonlinearities.softmax(score1)
        prob2 = lasagne.nonlinearities.softmax(score2)

        # Construct symbolic cost function
        target1 = T.vector('gold_target1', dtype= 'int64')
        target2 = T.vector('gold_target2', dtype= 'int64')

        cost1 = lasagne.objectives.categorical_crossentropy(prob1, target1)
        cost2 = lasagne.objectives.categorical_crossentropy(prob2, target2)

        self.cost = lasagne.objectives.aggregate(cost1+cost2, mode='sum')


        # Retrieve all parameters from the network
        reasoner_params = []
        for i in range(self.reasoning_depth):
            reasoner_params += self.reasoners[i].all_params
        all_params = self.encoder.all_params + reasoner_params + final_class_param

        all_updates = lasagne.updates.adam(self.cost, all_params)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.1, momentum=0.9)

        self.train_func = theano.function(self.inputs_variables + self.inputs_masks + [target1, target2], 
                                        [self.cost, score1, score2], updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function(self.inputs_variables + self.inputs_masks + [target1, target2], 
                                        [self.cost, score1, score2])
        # pydotprint(self.train_func, './computational_graph.png')

    def load_data(self):
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = train_set[0]
        self.train_ending = train_set[1]

        val_set = pickle.load(open(self.val_set_path))

        self.val_story = train_set[0]
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.n_val = len(self.val_answer)

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

    def fake_load_data(self):
        self.train_story = []
        self.train_story.append(np.concatenate((np.random.randint(10, size = (500, 10)), 10+np.random.randint(10, size=(500,10))),axis=0).astype('int64'))
        self.train_story.append(np.ones((1000, 10)).astype('int64'))
        self.train_story.append(np.ones((1000, 10)).astype('int64'))
        self.train_story.append(np.ones((1000, 10)).astype('int64'))
        self.train_story = np.asarray(self.train_story).reshape([1000,4,10])

        self.train_ending = np.concatenate((2 * np.ones((50, 5)), np.ones((50, 5))), axis = 0)
        self.val_story = []
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story = np.asarray(self.val_story).reshape([100, 4, 10])
        self.val_ending1 = np.ones((100, 10)).astype('int64')
        self.val_ending2 = 2*np.ones((100, 10)).astype('int64')
        self.val_answer = np.zeros(100)
        self.n_val = self.val_answer.shape[0]

        self.test_story = []
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story = np.asarray(self.test_story).reshape([100, 4, 10])
        self.test_ending1 = np.ones((100, 10)).astype('int64')
        self.test_ending2 = 2*np.ones((100, 10)).astype('int64')
        self.test_answer = np.ones(100)
        self.n_test = self.test_answer.shape[0]


        self.wemb = theano.shared(np.random.rand(30, self.rnn_units)).astype(theano.config.floatX)


    def val_set_test(self):

        correct = 0.

        for i in range(self.n_val):
            story = [np.asarray(sent, dtype='int64').reshape((1,-1)) for sent in self.val_story[i]]
            story_mask = [np.ones((1,len(self.val_story[i][j]))) for j in range(4)]

            ending1 = np.asarray(self.val_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.val_ending1[i])))

            ending2 = np.asarray(self.val_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.val_ending2[i])))

            cost, score1, score2 = self.prediction(story[0], story[1], story[2], story[3], 
                                                       ending1, ending2, story_mask[0], story_mask[1], story_mask[2],
                                                       story_mask[3], ending1_mask, ending2_mask, 1 - np.asarray([self.val_answer[i]]), 
                                                       np.asarray([self.val_answer[i]]))
            
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            predict_answer = 0
            if prediction == 0:
                predict_answer = 1
            elif prediction == 1:
                predict_answer = 0
            elif prediction == 2:
                predict_answer = 0
            else:
                predict_answer = 1

            if predict_answer == self.val_answer[i]:
                correct += 1.


        return correct/self.n_val

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

            cost, score1, score2 = self.prediction(story[0], story[1], story[2], story[3], 
                                                       ending1, ending2, story_mask[0], story_mask[1], story_mask[2],
                                                       story_mask[3], ending1_mask, ending2_mask, 1 - np.asarray([self.test_answer[i]]), 
                                                       np.asarray([self.test_answer[i]]))
            
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            predict_answer = 0
            if prediction == 0:
                predict_answer = 1
            elif prediction == 1:
                predict_answer = 0
            elif prediction == 2:
                predict_answer = 0
            else:
                predict_answer = 1

            if predict_answer == self.test_answer[i]:
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
        N_EPOCHS = 30
        N_BATCH = self.batchsize
        N_TRAIN_INS = len(self.val_answer)
        best_val_accuracy = 0
        best_test_accuracy = 0
        prev_percetage = 0.0
        speed = 0.0
        batch_count = 0.0
        start_batch = 0.0

        for epoch in range(N_EPOCHS):
            print "epoch ", epoch,":"
            shuffled_index_list = utils.shuffle_index(N_TRAIN_INS)

            max_batch = N_TRAIN_INS/N_BATCH

            start_time = time.time()

            total_cost = 0.0
            total_err_count = 0.0


            for batch in range(max_batch):
                # test 
                

                #test end

                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[self.val_story[index][i] for index in batch_index_list] for i in range(self.story_nsent)]
                train_ending = [self.val_ending1[index] for index in batch_index_list]

                # neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # while np.any((np.asarray(batch_index_list) - neg_end_index_list) == 0):
                #     neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # neg_end1 = [self.train_ending[index] for index in neg_end_index_list]
                neg_end1 = [self.val_ending2[index] for index in batch_index_list]
                # answer = np.random.randint(2, size = N_BATCH)
                # target1 = 1 - answer
                # target2 = 1 - target1
                answer = np.asarray([self.val_answer[index] for index in batch_index_list])

                target1 = 1 - answer
                target2 = answer
                # answer_vec = np.concatenate(((1 - answer).reshape(-1,1), answer.reshape(-1,1)),axis = 1)
                end1 = []
                end2 = []

                # for i in range(N_BATCH):
                #     if answer[i] == 0:
                #         end1.append(train_ending[i])
                #         end2.append(neg_end1[i])
                #     else:
                #         end1.append(neg_end1[i])
                #         end2.append(train_ending[i])

                for i in range(N_BATCH):
                    end1.append(train_ending[i])
                    end2.append(neg_end1[i])

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end1_matrix = utils.padding(end1)
                train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end1_mask = utils.mask_generator(end1)
                train_end2_mask = utils.mask_generator(end2)
                

                cost, prediction1, prediction2 = self.train_func(train_story_matrices[0], train_story_matrices[1], train_story_matrices[2],
                                                               train_story_matrices[3], train_end1_matrix, train_end2_matrix,
                                                               train_story_mask[0], train_story_mask[1], train_story_mask[2],
                                                               train_story_mask[3], train_end1_mask, train_end2_mask, target1, target2)



                prediction = np.argmax(np.concatenate((prediction1, prediction2), axis = 1), axis = 1)
                predict_answer = np.zeros((N_BATCH, ))
                for i in range(N_BATCH):
                    if prediction[i] == 0:
                        predict_answer[i] = 1
                    elif prediction[i] == 1:
                        predict_answer[i] = 0
                    elif prediction[i] == 2:
                        predict_answer[i] = 0
                    else:
                        predict_answer[i] = 1

                total_err_count += (abs(predict_answer - answer)).sum()



                total_cost += cost

            total_accuracy = 1.0 - (total_err_count/((max_batch)*N_BATCH))
            speed = max_batch * N_BATCH / (time.time() - start_time)
            print "======================================="
            print "epoch summary:"
            print "average speed: ", speed, "instances/sec"

            print ""
            print "total cost in this epoch: ", total_cost
            val_result = self.val_set_test()
            print "accuracy is: ", val_result*100, "%"
            if val_result >= best_val_accuracy:
                print "new best!"
                best_val_accuracy = val_result
            test_accuracy = self.test_set_test()
            print "test set accuracy: ", test_accuracy * 100, "%"
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
            print "======================================="




def main(argv):
    wemb_size = None
    if len(argv) > 5:
        wemb_size = argv[5]
    model = Neural_Reasoner_Model(argv[0], argv[1], argv[2], argv[3], argv[4], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])


