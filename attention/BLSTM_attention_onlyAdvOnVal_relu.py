import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import BLSTM_sequence
import BLSTM_last
import DNN_liar_relu
from theano.tensor.shared_randomstreams import RandomStreams

import sys
# from theano.printing import pydotprint




class Hierachi_RNN(object):
    def __init__(self, rnn_setting, val_split_ratio, D_batchsize, G_batchsize, liar_setting, 
                learning_rate1, learning_rate2, optimizer, delta, 
                score_func_nonlin = 'default', wemb_trainable = 1, wemb_size = None):
        # Initialize Theano Symbolic variable attributes
        self.story_input_variable = None
        self.story_mask = None
        self.story_nsent = 4

        self.cost = None
        self.learning_rate1 = float(learning_rate1)
        self.learning_rate2 = float(learning_rate2)
        self.classifier_train_func = None
        # Initialize data loading attributes
        self.wemb = None
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 
        self.train_set_path = '../../data/pickles/train_index_corpus.pkl'
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.index2word_dict_path = '../../data/pickles/ROC_train_index_dict.pkl'

        self.rnn_units = int(rnn_setting)
        self.liar_setting = [int(elem) for elem in liar_setting.split('x')]
        # self.dropout_rate = float(dropout_rate)
        self.D_batchsize = int(D_batchsize)
        self.G_batchsize = int(G_batchsize)

        self.val_split_ratio = float(val_split_ratio)

        self.classifier_hid1 = 1024
        # self.val_split_ratio = float(val_split_ratio)
        self.words_num = 28820
        self.delta = float(delta)
        if score_func_nonlin == 'default':
            self.score_func_nonlin = lasagne.nonlinearities.tanh
        else:
            self.score_func_nonlin = None

        self.wemb_trainable = bool(int(wemb_trainable))

        self.wemb_size = 300
        if wemb_size == None:
            self.random_init_wemb = False
        else:
            self.random_init_wemb = True
            self.wemb_size = int(wemb_size)

        self.train_story = None
        self.train_ending = None


        self.optimizer = optimizer

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
        self.attentioned_sent_rep1 = []
        self.attentioned_sent_rep2 = []
        self.bilinear_attention_matrix = theano.shared(0.02*np.random.rand(self.rnn_units, self.rnn_units) - 0.01)


    def encoding_layer(self):


        assert len(self.reshaped_inputs_variables)==len(self.inputs_masks)
        for i in range(self.story_nsent + 1):
            self.train_encodinglayer_vecs.append(lasagne.layers.get_output(self.encoder.output,
                                                        {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                         self.encoder.l_mask:self.inputs_masks[i]},
                                                         deterministic = True))


        end_representation = (self.train_encodinglayer_vecs[-1] * self.inputs_masks[4].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[4].sum(axis = 1, keepdims = True)

        #The last one (index-5) is the average of the non-attentioned ending sequence
        self.train_encodinglayer_vecs.append(end_representation)
        self.current_Nbatch = self.train_encodinglayer_vecs[0].shape[0]

    def attention1_layer(self):        
        for i in range(self.story_nsent):
            n_batch, n_seq, _ = self.train_encodinglayer_vecs[i].shape

            #second attention

            bili_part1 = T.dot(self.train_encodinglayer_vecs[i], self.bilinear_attention_matrix)

            attention1_score_tensor = T.batched_dot(bili_part1, self.train_encodinglayer_vecs[5])

            numerator1 = self.inputs_masks[i] * T.exp(attention1_score_tensor - attention1_score_tensor.max(axis = 1, keepdims = True))
        
            attention1_weight_matrix = numerator1 / numerator1.sum(axis = 1, keepdims = True)

            attentioned_sent_seq1 = self.train_encodinglayer_vecs[i]*(attention1_weight_matrix.reshape([n_batch, n_seq, 1]))

            attentioned_sent_rep1 = T.sum(attentioned_sent_seq1, axis = 1) / T.sum(self.inputs_masks[i], axis = 1).reshape([-1, 1])

            self.attentioned_sent_rep1.append(attentioned_sent_rep1)

    def attention2_layer(self):
        for i in range(self.story_nsent):
            n_batch, n_seq, _ = self.train_encodinglayer_vecs[i].shape

            #second attention

            bili_part1 = T.dot(self.train_encodinglayer_vecs[i], self.bilinear_attention_matrix)

            attention2_score_tensor = T.batched_dot(bili_part1, self.vt_2nd_end_repr)

            numerator2 = self.inputs_masks[i] * T.exp(attention2_score_tensor - attention2_score_tensor.max(axis = 1, keepdims = True))
        
            attention2_weight_matrix = numerator2 / numerator2.sum(axis = 1, keepdims = True)

            attentioned_sent_seq2 = self.train_encodinglayer_vecs[i]*(attention2_weight_matrix.reshape([n_batch, n_seq, 1]))

            attentioned_sent_rep2 = T.sum(attentioned_sent_seq2, axis = 1) / T.sum(self.inputs_masks[i], axis = 1).reshape([-1, 1])

            self.attentioned_sent_rep2.append(attentioned_sent_rep2)
        
    def model_constructor(self, wemb_size = None):
        self.inputs_variables = []
        self.inputs_masks = []
        self.reshaped_inputs_variables = []
        self.vt_2nd_end = T.matrix('second_end', dtype='int64')
        self.vt_2nd_end_mask = T.matrix('second_end_mask', dtype=theano.config.floatX)

        for i in range(self.story_nsent+1):
            self.inputs_variables.append(T.matrix('story'+str(i)+'_input', dtype='int64'))
            self.inputs_masks.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            batch_size, seqlen = self.inputs_variables[i].shape
            self.reshaped_inputs_variables.append(self.inputs_variables[i].reshape([batch_size, seqlen, 1]))

        #initialize neural network units
        self.encoder = BLSTM_sequence.BlstmEncoder(LSTMLAYER_1_UNITS = self.rnn_units, wemb_trainable = self.wemb_trainable)
        self.encoder.build_model(self.wemb)

        #build encoding layer
        self.encoding_layer()

        '''============================================================================'''
        #encode possible second ending
        self.encode_2nd_end_seq = lasagne.layers.get_output(self.encoder.output,
                                                    {self.encoder.l_in:self.vt_2nd_end, 
                                                     self.encoder.l_mask:self.vt_2nd_end_mask},
                                                     deterministic = True)

        self.vt_2nd_end_repr = (self.encode_2nd_end_seq * self.vt_2nd_end_mask.dimshuffle(0,1,'x')).sum(axis = 1) / self.vt_2nd_end_mask.sum(axis = 1, keepdims = True)
        '''============================================================================'''

        #build attention layer
        self.attention1_layer()
        self.attention2_layer()
        #build reasoning layers

        # merge tensors to fit in BLSTM models as input tensor
        # merge_ls1.shape = (n_batch, m_seq, self.rnn_units)
        self.merge_ls1 = [T.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1])) for tensor in self.attentioned_sent_rep1]
        self.merge_ls2 = [T.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1])) for tensor in self.attentioned_sent_rep2]
        encode_merge1 = T.concatenate(self.merge_ls1, axis = 1)
        encode_merge2 = T.concatenate(self.merge_ls2, axis = 1)

        # create sentence level encoder
        # using only last encoding result of both direction of the LSTM
        self.sent_encoder = BLSTM_last.BlstmEncoder(INPUT_SIZE = self.rnn_units, LSTMLAYER_1_UNITS = self.rnn_units)
        self.sent_encoder.build_model()

        # reasoner_result1.shape = (n_batch, self.rnn_units)
        reasoner_result1 = lasagne.layers.get_output(self.sent_encoder.output, {self.sent_encoder.l_in: encode_merge1}, 
                                                    deterministic = True)

        reasoner_result2 = lasagne.layers.get_output(self.sent_encoder.output, {self.sent_encoder.l_in: encode_merge2}, 
                                                    deterministic = True)
        '''alternative ending construction part'''
        '''========================================================'''

        # self.DNN_liar.l_in.shape = (n_batch, self.rnn_units * 2)
        # self.DNN_liar.output = (n_batch, self.rnn_units)
        self.DNN_liar = DNN_liar.DNNLiar(INPUTS_SIZE = self.rnn_units, LAYER_UNITS = self.liar_setting, INPUTS_PARTS = 1)
        original_storyNend = reasoner_result1

        # self.alternative_end.shape = self.DNN_liar.output
        self.alternative_end = lasagne.layers.get_output(self.DNN_liar.output, {self.DNN_liar.l_in: original_storyNend})

        '''========================================================'''

        '''discriminator'''
        '''========================================================'''

        l_story_in = lasagne.layers.InputLayer(shape=(None, self.rnn_units))
        l_end_in = lasagne.layers.InputLayer(shape = (None, self.rnn_units))
        l_concate = lasagne.layers.ConcatLayer([l_story_in, l_end_in], axis = 1)
        l_hid1 = lasagne.layers.DenseLayer(l_concate, num_units = self.classifier_hid1, nonlinearity=lasagne.nonlinearities.tanh)
        l_hid = lasagne.layers.DenseLayer(l_hid1, num_units=1,
                                          nonlinearity=self.score_func_nonlin)

        final_class_param = lasagne.layers.get_all_params(l_hid)

        '''========================================================'''
        '''               generator training graph                 '''
        '''========================================================'''
        euclidien_distance = T.sqrt(T.sum(T.sqr(self.alternative_end - reasoner_result1), axis = 1))

        '''========================================================'''
        '''                    generating score                    '''
        '''========================================================'''

        # answer = (2 * srng.uniform((n_batch,))).astype('int64')

        # ending_pair_tensor1 = self.train_encodinglayer_vecs[-1] * (1-answer).dimshuffle(0,'x') + self.alternative_end * answer.dimshuffle(0,'x')
        # ending_pair_tensor2 = self.train_encodinglayer_vecs[-1] * (answer).dimshuffle(0,'x') + self.alternative_end * (1-answer).dimshuffle(0,'x')
        

        origi_score = lasagne.layers.get_output(l_hid, {l_story_in: reasoner_result1, 
                                                   l_end_in: self.train_encodinglayer_vecs[-1]})
        alter_score = lasagne.layers.get_output(l_hid, {l_story_in: reasoner_result1, 
                                                   l_end_in: self.alternative_end})
        # srng = RandomStreams(seed=234)

        # noise_story = srng.normal((self.batchsize, self.rnn_units))
        # noise_score = lasagne.layers.get_output(l_hid, {l_story_in: noise_story})

        '''========================================================'''

        vt_2nd_score = lasagne.layers.get_output(l_hid, {l_story_in: reasoner_result2, 
                                                   l_end_in: self.vt_2nd_end_repr})

        # prob1 = lasagne.nonlinearities.softmax(origi_score)
        # prob2 = lasagne.nonlinearities.softmax(alter_score)


        # Construct symbolic cost function
        

        cost1 = T.max(T.concatenate([T.zeros_like(origi_score), - origi_score + alter_score + self.delta * T.ones_like(origi_score)], axis = 1), axis = 1)
        # cost2 = 
        liar_cost = euclidien_distance 

        self.main_cost = lasagne.objectives.aggregate(cost1, mode = 'mean')
        self.liar_cost = lasagne.objectives.aggregate(liar_cost, mode = 'mean')

        # Retrieve all parameters from the network
        main_params = self.encoder.all_params + self.sent_encoder.all_params + final_class_param + [self.bilinear_attention_matrix]


        liar_params = self.DNN_liar.all_params

        main_updates = None
        liar_updates = None

        if self.optimizer == 'adam':
            main_updates = lasagne.updates.adam(self.main_cost, main_params, learning_rate=self.learning_rate1)

            liar_updates = lasagne.updates.adam(self.liar_cost, liar_params, learning_rate=self.learning_rate2)
        else:
            main_updates = lasagne.updates.momentum(self.main_cost, main_params, learning_rate=self.learning_rate1, momentum=0.9)
            liar_updates = lasagne.updates.momentum(self.liar_cost, liar_params, learning_rate=self.learning_rate2, momentum=0.9)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        # all_updates = []
        # for k,v in main_updates.items()+liar_updates.items():
        #     all_updates.append((k,v))

        # combine two sets of parameters update into a single OrderedDict 
        self.classifier_train_func = theano.function(self.inputs_variables + self.inputs_masks, 
                                        [self.main_cost, origi_score, alter_score], updates = main_updates)
        self.generator_train_func = theano.function(self.inputs_variables + self.inputs_masks, self.liar_cost, updates = liar_updates)
        # Compute adam updates for training

        self.prediction = theano.function(self.inputs_variables + [self.vt_2nd_end] + self.inputs_masks + [self.vt_2nd_end_mask], [origi_score, vt_2nd_score])

        self.adv_monitor = theano.function(self.inputs_variables + self.inputs_masks, self.alternative_end)

        self.test_end_matrix = T.matrix('test_end', dtype='int64')
        self.test_end_mask = T.matrix('test_end_mask', dtype=theano.config.floatX)


        self.test_end_rep = lasagne.layers.get_output(self.encoder.output,
                                                    {self.encoder.l_in:self.test_end_matrix, 
                                                    self.encoder.l_mask:self.test_end_mask},
                                                    deterministic = True)
        check_end_representation = (self.test_end_rep * self.test_end_mask.dimshuffle(0,1,'x')).sum(axis = 1) / self.test_end_mask.sum(axis = 1, keepdims = True)

        self.end_rep_check = theano.function([self.test_end_matrix, self.test_end_mask], check_end_representation)

    def load_data(self):
        '''======Train Set====='''
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = train_set[0]
        self.train_ending = train_set[1]
        self.n_train = len(self.train_ending)
        
        '''=====Val Set====='''
        val_set = pickle.load(open(self.val_set_path))

        self.val_story = val_set[0]
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.n_val = len(self.val_answer)
        self.val_train_n = int(self.val_split_ratio * self.n_val)
        self.val_test_n = self.n_val - self.val_train_n

        shuffled_indices_ls = utils.shuffle_index(self.n_val)

        self.val_train_ls = shuffled_indices_ls[:self.val_train_n]
        self.val_test_ls = shuffled_indices_ls[self.val_train_n:]

        '''=====Test Set====='''
        test_set = pickle.load(open(self.test_set_path))
        self.test_story = test_set[0]
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_answer)


        ''''=====Wemb====='''
        if self.random_init_wemb:
            wemb = np.random.rand(self.words_num ,self.wemb_size)
            wemb = np.concatenate((np.zeros((1, self.wemb_size)), wemb), axis = 0)
            self.wemb = theano.shared(wemb).astype(theano.config.floatX)
        else:
            self.wemb = theano.shared(pickle.load(open(self.wemb_matrix_path))).astype(theano.config.floatX)

        '''=====Peeping Preparation====='''
        self.peeked_ends_ls = np.random.randint(self.n_train, size=(5,))
        self.ends_pool_ls = np.random.choice(range(self.n_train), 2000, replace = False)
        self.index2word_dict = pickle.load(open(self.index2word_dict_path))
        
    def fake_load_data(self):
        self.train_story = []
        self.train_story.append(np.concatenate((np.random.randint(10, size = (50, 10)), 10+np.random.randint(10, size=(50,10))),axis=0).astype('int64'))
        self.train_story.append(np.ones((100, 10)).astype('int64'))
        self.train_story.append(np.ones((100, 10)).astype('int64'))
        self.train_story.append(np.ones((100, 10)).astype('int64'))
        self.train_story = np.asarray(self.train_story).reshape([100,4,10])

        self.train_ending = np.concatenate((2 * np.ones((50, 5)), np.ones((50, 5))), axis = 0)
        self.val_story = []
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))

        self.val_ending1 = np.ones((100, 10)).astype('int64')
        self.val_ending2 = 2*np.ones((100, 10)).astype('int64')
        self.val_answer = np.zeros(100)
        self.n_val = self.val_answer.shape[0]

        self.test_story = []
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))

        self.test_ending1 = np.ones((100, 10)).astype('int64')
        self.test_ending2 = 2*np.ones((100, 10)).astype('int64')
        self.test_answer = np.ones(100)
        self.n_test = self.test_answer.shape[0]


        self.wemb = theano.shared(np.random.rand(30, self.rnn_units)).astype(theano.config.floatX)


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

            correct += abs(prediction - self.val_answer[i])

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

            correct += abs(prediction - self.test_answer[i])

        return correct/self.n_test

    def adv_model_monitor(self):
        '''part I pass story to RNN reader and adv generator'''
        stories_indices = self.peeked_ends_ls
        peek_story = [[self.train_story[index][i] for index in stories_indices] for i in range(1,self.story_nsent+1)]
        peek_ending = [self.train_ending[index] for index in stories_indices]

        peek_story_matrices = [utils.padding(batch_sent) for batch_sent in peek_story]
        peek_end_matrix = utils.padding(peek_ending)
        # train_end2_matrix = utils.padding(end2)

        peek_story_mask = [utils.mask_generator(batch_sent) for batch_sent in peek_story]
        peek_end_mask = utils.mask_generator(peek_ending)
        # train_end2_mask = utils.mask_generator(end2)


        adv_end_rep_batch = self.adv_monitor(peek_story_matrices[0], peek_story_matrices[1], peek_story_matrices[2],
                                                       peek_story_matrices[3], peek_end_matrix,
                                                       peek_story_mask[0], peek_story_mask[1], peek_story_mask[2],
                                                       peek_story_mask[3], peek_end_mask)


        if np.all(adv_end_rep_batch[0] - adv_end_rep_batch[1] == 0):
            print "WARNING!!! Same end rep for diff stories!"
        select_story_ls = self.ends_pool_ls
        random_check_ending = [self.train_ending[index] for index in select_story_ls]
        random_check_end_matrix = utils.padding(random_check_ending)
        random_check_end_mask = utils.mask_generator(random_check_ending)

        random_check_ending_rep = self.end_rep_check(random_check_end_matrix, random_check_end_mask)


        '''part II calculate the most similar sent'''
        # end_rep_matrix = np.zeros((self.n_train, self.rnn_units))
        # wemb_matrix = self.wemb.get_value()
        # for i in range(self.n_train):
        #     end_rep_matrix[i] = np.sum(wemb_matrix[self.train_ending[i]], axis = 0) / (len(self.train_ending[i]))

        norm_end_rep_matrix = np.linalg.norm(random_check_ending_rep, axis = 1).reshape(-1,1)
        norm_adv_end_rep = np.linalg.norm(adv_end_rep_batch, axis = 1).reshape(-1,1)
        # norm_denominator_matrix.shape = (1000, 5)
        norm_denominator_matrix = np.dot(norm_end_rep_matrix, norm_adv_end_rep.T)

        # dot_prod.shape = (1000, 5)
        dot_prod = np.dot(random_check_ending_rep, adv_end_rep_batch.T)

        # cos_simi_matrix.shape = (1000, 5)
        cos_simi_matrix = dot_prod / norm_denominator_matrix

        index_list_inMatrix = np.argmax(cos_simi_matrix, axis = 0)
        index_list = [select_story_ls[i] for i in index_list_inMatrix]
        '''part III print out story and the most similar end correspondingly'''
        for i in range(5):
            index = stories_indices[i]
            story_string = " | ".join([" ".join([self.index2word_dict[self.train_story[index][j][k]] for k in range(len(self.train_story[index][j]))]) for j in range(5)])
            story_end = " ".join([self.index2word_dict[self.train_ending[index][k]] for k in range(len(self.train_ending[index]))])
            generated_end = " ".join([self.index2word_dict[self.train_ending[index_list[i]][k]] for k in range(len(self.train_ending[index_list[i]]))])

            print story_string 
            print " #END# " + story_end
            print ""
            print "Adv Model Generated: " + generated_end
            print ""

    def generator_train(self, epochs):
        print "generator training begin.."
        N_EPOCHS = epochs
        N_BATCH = self.G_batchsize
        N_TRAIN_INS = self.val_train_n
        
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

                answer = np.asarray([self.val_answer[index] for index in batch_index_list])

                end = []
                for i in range(N_BATCH):
                    if answer[i] == 0:
                        end.append(end2[i])
                    else:
                        end.append(end1[i])

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end_matrix = utils.padding(end)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end_mask = utils.mask_generator(end)
                

                cost =  self.generator_train_func(train_story_matrices[0], train_story_matrices[1], train_story_matrices[2],
                                        train_story_matrices[3], train_end_matrix,
                                        train_story_mask[0], train_story_mask[1], train_story_mask[2],
                                        train_story_mask[3], train_end_mask)
                total_cost += cost

            print "total cost of the generator in this epoch: ", total_cost

        print "generator training finished.."

    def begin_train(self):
        N_EPOCHS = 100
        N_BATCH = self.D_batchsize
        N_TRAIN_INS = int(len(self.train_ending))
        best_val_accuracy = 0
        best_test_accuracy = 0
        test_threshold = 10000/N_BATCH

        max_batch = N_TRAIN_INS/N_BATCH
        '''init test'''
        print "initial test..."
        val_result = self.val_set_test()
        print "accuracy is: "+str( val_result*100) +"%"
        if val_result > best_val_accuracy:
            best_val_accuracy = val_result

        test_accuracy = self.test_set_test()
        print "test set accuracy: "+ str(test_accuracy * 100)+ "%"
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        '''end of init test'''

        for epoch in range(N_EPOCHS):
            print "epoch "+str(epoch)+":"
            shuffled_index_list = utils.shuffle_index(N_TRAIN_INS)


            start_time = time.time()

            total_main_cost = 0.0
            total_liar_cost = 0.0
            total_correct_count = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[self.train_story[index][i] for index in batch_index_list] for i in range(1,self.story_nsent+1)]
                train_ending = [self.train_ending[index] for index in batch_index_list]

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end_matrix = utils.padding(train_ending)
                # train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end_mask = utils.mask_generator(train_ending)
                # train_end2_mask = utils.mask_generator(end2)


                main_cost, prediction1, prediction2 = self.classifier_train_func(train_story_matrices[0], train_story_matrices[1], train_story_matrices[2],
                                                               train_story_matrices[3], train_end_matrix,
                                                               train_story_mask[0], train_story_mask[1], train_story_mask[2],
                                                               train_story_mask[3], train_end_mask)

                prediction = np.argmax(np.concatenate((prediction1, prediction2), axis = 1), axis = 1)
                total_correct_count += (N_BATCH - prediction.sum())
                total_main_cost += main_cost


                if batch % test_threshold == 0 and batch != 0:
                    current_train_accuracy = (total_correct_count * 1.0)/((batch + 1) * N_BATCH)*100.0
                    print "accuracy on training set: "+ str(current_train_accuracy)+"%"

                    print "test on val set..."
                    val_result = self.val_set_test()
                    print "accuracy is: "+str(val_result*100)+"%"
                    if val_result > best_val_accuracy:
                        print "new best! test on test set..."
                        best_val_accuracy = val_result

                        test_accuracy = self.test_set_test()
                        print "test set accuracy: "+str(test_accuracy * 100)+"%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy


                    print np.concatenate((prediction1, prediction2), axis = 1)
                    
                    '''============================================================='''
                    '''train the generator again once discriminator is strong enough'''
                    '''============================================================='''
                    if current_train_accuracy >= 99.0:
                        self.generator_train(10)
                        
            print "======================================="
            print "epoch summary:"
            print "average speed: "+str(N_TRAIN_INS/(time.time() - start_time))+"instances/s "
            print "total main cost: "+str(total_main_cost/max_batch)
            print "accuracy for this epoch: "+str(total_correct_count/(max_batch * N_BATCH) * 100.0)+"%"
            self.adv_model_monitor()
            print "======================================="



def main(argv):
    wemb_size = None
    if len(argv) > 10:
        wemb_size = argv[10]
    model = Hierachi_RNN(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8],argv[9], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])

