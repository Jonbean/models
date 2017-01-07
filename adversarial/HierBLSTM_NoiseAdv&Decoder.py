import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import BLSTM_Encoder
import LSTM_Decoder
import DNN_score_function as DNN
import sys
from collections import OrderedDict
# from theano.printing import pydotprint



class Hierachi_RNN(object):
    def __init__(self, 
                 word_rnn_setting, 
                 sent_rnn_setting,
                 batchsize, 
                 dnn_generator_setting,
                 decoder_units,
                 reasoning_type = 'concatenate',
                 wemb_trainable = 1,
                 discrim_lr = 0.001,
                 generat_lr = 0.001,
                 delta = 1.0,
                 mode = 'sequence',
                 nonlin_func = 'default',
                 score_func = 'cos',
                 loss_type = 'hinge', 
                 random_input_type = 'normal',
                 dnn_discriminator_setting = '512x1',
                 discrim_regularization_level = 0,
                 generat_regularization_level = 0,
                 regularization_index = '1E-4'):
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

        # Initialize data loading attributes
        self.wemb = None
        self.train_set_path = '../../data/pickles/train_index_corpus.pkl'
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.index2word_dict_path = '../../data/pickles/ROC_train_index_dict.pkl'

        self.word_rnn_units = map(int, word_rnn_setting.split('x')) 
        self.sent_rnn_units = map(int, sent_rnn_setting.split('x'))

        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.batchsize = int(batchsize)
        # self.reasoning_depth = int(reasoning_depth)
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

        self.delta = float(delta)
       
        self.wemb_trainable = bool(int(wemb_trainable))
        self.discrim_lr = float(discrim_lr)
        self.generat_lr = float(generat_lr)

        self.mode = mode
        self.GRAD_CLIP = 10.0
        if nonlin_func == 'default':
            self.nonlin_func = lasagne.nonlinearities.tanh
        else:
            self.nonlin_func = None

        self.bias = 0.001
        self.dnn_generator_settings = map(int, dnn_generator_setting.split('x'))
        self.dnn_discriminator_setting = map(int, dnn_discriminator_setting.split('x'))
        self.decoder_units = map(int, decoder_units.split('x'))
        self.reasoning_type = reasoning_type

        self.loss_type = loss_type
        self.score_func = score_func
        self.discrim_regularization_level = int(discrim_regularization_level)
        self.generat_regularization_level = int(generat_regularization_level)
        self.random_input_type = random_input_type

        self.discrim_regularization_dict = {0:"no regularization on discriminator",
                                            1:"L2 on discriminator DNN",
                                            2:"L2 on discriminator word level RNN + DNN",
                                            3:"L2 on discriminator word level RNN",
                                            4:"L2 on discriminator all RNN",
                                            5:"L2 on discriminator all level"}
        self.generat_regularization_dict = {0:"no regularization on generator",
                                            1:"L2 on generator DNN"}
        self.regularization_index = float(regularization_index)

        if self.loss_type == 'log':
            assert self.dnn_discriminator_setting[-1] == 2
            assert self.score_func != 'cos'
            assert self.sent_rnn_units[-1] > 1
            assert self.nonlin_func == None
        else:
            if self.score_func == 'DNN':
                assert self.dnn_discriminator_setting[-1] == 1
            elif self.score_func == 'cos':
                assert self.sent_rnn_units[-1] > 2
            else:
                assert self.sent_rnn_units[-1] == 1
        
        if self.score_func != "DNN":
            assert self.discrim_regularization_level != 1 and self.discrim_regularization_level != 2

    def regularization_show(self):
        print self.discrim_regularization_dict[self.discrim_regularization_level]
        print self.generat_regularization_dict[self.generat_regularization_level]

    def encoding_layer(self):
        if self.mode == 'sequence':
            for i in range(self.story_nsent+2):
                sent_seq = lasagne.layers.get_output(self.encoder.output,
                                                            {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                             self.encoder.l_mask:self.inputs_masks[i]},
                                                             deterministic = True)

                self.train_encodinglayer_vecs.append((sent_seq * (self.inputs_masks[i].dimshuffle(0,1,'x'))).sum(axis = 1) / ((self.inputs_masks[i]).sum(axis = 1, keepdims = True)) )

        else:
            for i in range(self.story_nsent+2):
                sent_seq = lasagne.layers.get_output(self.encoder.output,
                                                            {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                             self.encoder.l_mask:self.inputs_masks[i]},
                                                             deterministic = True)

                self.train_encodinglayer_vecs.append(sent_seq)

    def reasoning_layer(self):

        if self.reasoning_type == 'concatenate':
            merge_ls = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:4]]
            encode_merge = T.concatenate(merge_ls, axis = 1)

            self.plot_rep = lasagne.layers.get_output(self.reasoner.output,
                                                {self.reaonser.l_in:self.merge_ls})

            self.ending1_rep = lasagne.layers.get_output(self.reaonser.output,
                                                  {self.reasoner.l_in:self.train_encodinglayer_vecs[4]})

            self.ending2_rep = lasagne.layers.get_output(self.reaonser.output,
                                                  {self.reasoner.l_in:self.train_encodinglayer_vecs[5]})

            self.fake_end_rep = lasagne.layers.get_output(self.reasoner.output,
                                                  {self.reasoner.l_in:self.fake_endings})

        else:
            merge_ls1 = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:4] + \
                             [self.train_encodinglayer_vecs[4]]]
            encode_merge1 = T.concatenate(merge_ls1, axis = 1)

            merge_ls2 = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:4] + \
                             [self.train_encodinglayer_vecs[5]]]
            encode_merge2 = T.concatenate(merge_ls2, axis = 1)

            merge_ls3 = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:4] + \
                             [self.fake_endings]]
            encode_merge3 = T.concatenate(merge_ls3, axis = 1)

            self.story1_rep = lasagne.layers.get_output(self.reasoner.output,
                                                       {self.reasoner.l_in:encode_merge1})
            self.story2_rep = lasagne.layers.get_output(self.reasoner.output, 
                                                       {self.reasoner.l_in:encode_merge2})
            self.story3_rep = lasagne.layers.get_output(self.reasoner.output,
                                                       {self.reasoner.l_in:encode_merge3})

            

    def batch_cosine(self, batch_vectors1, batch_vectors2):
        dot_prod = T.batched_dot(batch_vectors1, batch_vectors2)

        batch1_norm = T.sqrt((T.sqr(batch_vectors1)).sum(axis = 1))
        batch2_norm = T.sqrt((T.sqr(batch_vectors2)).sum(axis = 1))

        batch_cosine_vec = dot_prod/(batch1_norm * batch2_norm)
        return batch_cosine_vec.reshape((-1,1))

    def matrix_DNN(self, batch_rep1, batch_rep2):
        batch_rep1_broad = batch_rep1 + T.zeros((self.batch_m, self.batch_m, self.rnn_units))
        
        batch_rep2_broad = batch_rep2 + T.zeros((self.batch_m, self.batch_m, self.rnn_units))        
        batch_rep1_reshape = batch_rep1_broad.dimshuffle(1,0,2).reshape((-1, self.rnn_units))
        batch_rep2_reshape = batch_rep2_broad.reshape((-1, self.rnn_units))
        
        batch_concate_input = T.concatenate([batch_rep1_reshape, batch_rep2_reshape], axis = 1)        
        batch_score = lasagne.layers.get_output(self.DNN_score_func.output, 
                                               {self.DNN_score_func.l_in: batch_concate_input})
        return batch_score.reshape((self.batch_m, self.batch_m))

    def matrix_cos(self, batch_rep1, batch_rep2):
        batch_rep1_broad = batch_rep1 + T.zeros((self.batch_m, self.batch_m, self.rnn_units))
        
        batch_rep2_broad = batch_rep2 + T.zeros((self.batch_m, self.batch_m, self.rnn_units))
        batch_rep1_reshape = batch_rep1_broad.dimshuffle(1,0,2).reshape((-1, self.rnn_units))
        batch_rep2_reshape = batch_rep2_broad.reshape((-1, self.rnn_units))

        batch_dot = (T.batched_dot(batch_rep1_reshape, batch_rep2_reshape)).reshape((self.batch_m, self.batch_m))
        norm1 = T.sqrt(T.sum(T.sqr(batch_rep1), axis = 1))
        norm2 = T.sqrt(T.sum(T.sqr(batch_rep2), axis = 1))
        
        norm_matrix = T.dot(norm1.reshape((-1,1)),norm2.reshape((1,-1)))
        return batch_dot/norm_matrix

    def penalty_generator(self, params):
        penalty = [lasagne.regularization.l2(param) for param in params]
        penalty_scalar = lasagne.objectives.aggregate(T.stack(penalty), mode = "mean")
        return penalty_scalar

    def discrim_cost_generator(self):
        if self.discrim_regularization_level == 0:
            self.all_discrim_cost = self.discrim_cost
        elif self.discrim_regularization_level == 1:
            self.all_discrim_cost = self.discrim_cost + self.regularization_index * self.penalty_generator(self.DNN_score_func.all_params)
        elif self.discrim_regularization_level == 2:
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * self.penalty_generator(self.DNN_score_func.all_params) + \
                                    self.regularization_index * self.penalty_generator(sefl.encoder.all_params)
        elif self.discrim_regularization_level == 3:
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * self.penalty_generator(self.encoder.all_params)
        elif self.discrim_regularization_level == 4:
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * self.penalty_generator(self.encoder.all_params) + \
                                    self.regularization_index * self.penalty_generator(self.reasoner.all_params)
        else:
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * self.penalty_generator(self.DNN_score_func.all_params) + \
                                    self.regularization_index * self.penalty_generator(self.encoder.all_params) + \
                                    self.regularization_index * self.penalty_generator(self.reasoner.all_params)

    def generat_cost_generator(self):
        if self.generat_regularization_level == 0:
            self.all_generat_cost = self.generat_cost
        else:
            self.all_generat_cost = self.generat_cost + self.regularization_index * self.penalty_generator(self.DNN_generator.all_params)

    def decoding_layer(self, input_variable, input_mask):
        batchsize, max_len = input_mask.shape

        index_matrix = T.arange(max_len).dimshuffle('x', 0) + T.zeros_like(input_mask)
        index_tensor = index_matrix.reshape((batchsize, max_len, 1))

        broad_cast_rep = input_variable.dimshuffle(0,'x',1) + T.zeros((batchsize, max_len, self.word_rnn_units[0]))
        new_input = T.concatenate([broad_cast_rep, index_tensor], axis = 2)

        predict_seq = lasagne.layers.get_output(self.decoder.output, 
                                               {self.decoder.l_in: new_input,
                                                self.decoder.l_mask: input_mask})
        return predict_seq

    def model_constructor(self, wemb_size = None):
        self.inputs_variables = []
        self.inputs_masks = []
        self.reshaped_inputs_variables = []
        for i in range(self.story_nsent+2):
            self.inputs_variables.append(T.matrix('story'+str(i)+'_input', dtype='int64'))
            self.inputs_masks.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            self.reshaped_inputs_variables.append(self.inputs_variables[i].dimshuffle(0,1,'x'))

        #initialize neural network units
        '''================PART I================'''
        '''             RNN ENCODING             '''
        '''======================================'''

        self.encoder = BLSTM_Encoder.BlstmEncoder(LSTMLAYER_UNITS = self.word_rnn_units, 
                                                  wemb_trainable = self.wemb_trainable,
                                                  mode = self.mode)
        self.encoder.build_word_level(self.wemb)

        #build encoding layer
        self.encoding_layer()
        
        self.reasoner = BLSTM_Encoder.BlstmEncoder(self.sent_rnn_units,
                                                   mode = 'last')
        self.reasoner.build_sent_level(self.word_rnn_units[0])

        '''================PART II================'''
        '''             DNN GENERATION            '''
        '''======================================='''

        

        self.DNN_generator = DNN.DNN(INPUTS_SIZE = self.dnn_generator_settings[0], 
                                     LAYER_UNITS = self.dnn_generator_settings[1:], 
                                     final_nonlin = self.nonlin_func)

        srng = RandomStreams(seed = 31415)
        self.random_input = None
        if self.random_input_type == 'normal':
            self.random_input = srng.normal(size = (self.batchsize, self.dnn_generator_settings[0]), avg = 0.0, std = 1.0)
        else:
            self.random_input = srng.uniform(size = (self.batchsize, self.dnn_generator_settings[0]), low = -1.0, high = 1.0)
        

        self.fake_endings = lasagne.layers.get_output(self.DNN_generator.output, {self.DNN_generator.l_in: self.random_input})

        '''================PART IV================'''
        '''              LSTM REASONING           '''
        '''======================================='''
        self.reasoning_layer()

        '''================PART V================='''
        '''              LSTM Decoder             '''
        '''======================================='''
        # Sentence level decoder can be added later

        self.decoder = LSTM_Decoder.LSTMDecoder(self.decoder_units)
        self.decoder.index_label_classi_decoder()
        self.decoding_cost_ls = []
        for i in range(self.story_nsent+1):
            prediction = self.decoding_layer(self.train_encodinglayer_vecs[i], self.inputs_masks[i])
            prediction_reshape = prediction.reshape((-1, self.wemb_size))
            self.decoding_cost_ls.append(lasagne.objectives.categorical_crossentropy(prediction_reshape, T.flatten(self.inputs_variables[i]) - 1))

        self.decoding_cost = T.sum(T.concatenate(self.decoding_cost_ls))

        '''================PART VI================'''
        '''                SCOREING                '''
        '''========================================'''


        if self.score_func == 'cos':
        #build reasoning layers
            self.reasoner.build_sent_level(self.word_rnn_units[0])


            self.score1 = self.batch_cosine(self.plot_rep, self.ending1_rep)
            self.score2 = self.batch_cosine(self.plot_rep, self.fake_endings)
            self.score3 = self.batch_cosine(self.plot_rep, self.ending2_rep)

            score4_matrix = self.matrix_cos(self.plot_rep, self.ending1_rep)
            all_other_score = score4_matrix * (T.ones_like(score4_matrix) - T.eye(score4_matrix.shape[0])) - T.eye(score4_matrix.shape[0])
            self.score4 = T.max(all_other_score, axis = 1)
            self.max_score_index = T.argmax(all_other_score, axis = 1)

        elif self.score_func == 'DNN':
        #build reasoning layers

            if self.reasoning_type == 'concatenate':

                self.DNN_score_func = DNN.DNN(INPUTS_SIZE = self.sent_rnn_units[-1] + self.dnn_generator_settings[-1], 
                                              LAYER_UNITS = self.dnn_discriminator_setting, 
                                              final_nonlin = self.nonlin_func)


                real_pair1 = T.concatenate([self.plot_rep, self.ending1_rep], axis = 1)
                real_pair2 = T.concatenate(self.plot_rep, self.ending2_rep, axis = 1)
                fake_pair = T.concatenate([self.plot_rep, self.fake_endings], axis = 1)

                self.score1 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                                       {self.DNN_score_func.l_in: real_pair1})
                self.score2 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                                       {self.DNN_score_func.l_in: fake_pair})
                self.score3 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                                       {self.DNN_score_func.l_in: real_pair2})
                score4_matrix = self.matrix_DNN(reasoner_result, self.train_encodinglayer_vecs[4])
                all_other_score = score4_matrix * (T.ones_like(score4_matrix) - T.eye(score4_matrix.shape[0])) - T.eye(score4_matrix.shape[0])
                self.score4 = T.max(all_other_score, axis = 1)
                self.max_score_index = T.argmax(all_other_score, axis = 1)
            else:
                self.DNN_score_func = DNN.DNN(INPUTS_SIZE = self.sent_rnn_units[-1], 
                                              LAYER_UNITS = self.dnn_discriminator_setting, 
                                              final_nonlin = self.nonlin_func)
                self.score1 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                                       {self.DNN_score_func.l_in: self.story1_rep})
                self.score2 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                                       {self.DNN_score_func.l_in: self.story3_rep})
                self.score3 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                                       {self.DNN_score_func.l_in: self.story2_rep})          
                self.score4 = "O_oooops"
        else:
            self.reasoner.build_score_func(self.word_rnn_units[0])


            self.score1 = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge1})
            self.score2 = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge2})
            self.score3 = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge3})
            self.score4 = "O_oooops"

        '''loss function'''

        if self.loss_type == 'hinge':

            self.discrim_score = - self.score1 + self.delta + self.score2
            # self.batch_max_score = - self.score1 + self.delta + self.socre4
            self.generat_score = - self.score2

            self.discrim_score_vec = T.clip(self.discrim_score, 0.0, float('inf'))
            # self.batch_max_score_vec = T.clip(self.batch_max_score, 0.0, float('inf'))
            self.generat_score_vec = T.clip(self.generat_score, 0.0, float('inf'))

            self.discrim_cost = lasagne.objectives.aggregate(self.discrim_score_vec, mode = 'mean') 
            self.generat_cost = lasagne.objectives.aggregate(self.generat_score_vec, mode = 'mean')

            self.all_discrim_params = self.encoder.all_params + self.reasoner.all_params 

        else:
            prob1 = lasagne.nonlinearities.softmax(self.score1)
            prob2 = lasagne.nonlinearities.softmax(self.score2)

            cost1 = lasagne.objectives.categorical_crossentropy(prob1, T.ones((self.batchsize, )).astype('int64'))
            cost2 = lasagne.objectives.categorical_crossentropy(prob2, T.zeros((self.batchsize, )).astype('int64'))
            liar_cost = lasagne.objectives.categorical_crossentropy(prob2, T.ones((self.batchsize, )).astype('int64'))

            self.discrim_cost = lasagne.objectives.aggregate(cost1 + cost2, mode = 'mean')
            self.generat_cost = lasagne.objectives.aggregate(liar_cost, mode = 'mean')
           

            self.all_discrim_params = self.encoder.all_params + self.reasoner.all_params
            if self.score_func == "DNN":
                self.all_discrim_params += self.DNN_score_func.all_params

        # testing decoding result of fake endings
        fake_ending_interpretation = self.decoding_layer(self.fake_endings, T.ones((self.batchsize, 20)))
        # Retrieve all parameters from the network
        self.all_generat_params = self.DNN_generator.all_params
        self.decoder_params = self.decoder.all_params

        self.discrim_cost_generator()
        self.generat_cost_generator()

        all_discrim_updates = lasagne.updates.adam(self.all_discrim_cost + self.decoding_cost, self.all_discrim_params, learning_rate = self.discrim_lr)
        all_generat_updates = lasagne.updates.adam(self.all_generat_cost, self.all_generat_params, learning_rate = self.generat_lr)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)
        all_discrim_updates.update(all_generat_updates)
        
        if self.loss_type == 'hinge':        
            self.train_func = theano.function(self.inputs_variables[:5] + self.inputs_masks[:5], 
                                             [self.discrim_cost, self.generat_cost, 
                                             self.score1, self.score2, self.train_encodinglayer_vecs[4], self.decoding_cost],
                                             updates = all_discrim_updates)
        else:
            self.train_func = theano.function(self.inputs_variables[:5] + self.inputs_masks[:5], 
                                             [self.discrim_cost, self.generat_cost, self.score1, 
                                              self.score2, self.train_encodinglayer_vecs[4], self.decoding_cost],
                                             updates = all_discrim_updates)

        self.monitor_func = theano.function([], [self.fake_endings, fake_ending_interpretation])

        self.prediction = theano.function(self.inputs_variables + self.inputs_masks,
                                         [self.score1, self.score3]) 
        # pydotprint(self.train_func, './computational_graph.png')

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

        '''=====Test Set====='''
        test_set = pickle.load(open(self.test_set_path))
        self.test_story = test_set[0]
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_answer)


        ''''=====Wemb====='''
        self.wemb = theano.shared(pickle.load(open(self.wemb_matrix_path))).astype(theano.config.floatX)
        self.wemb_size = self.wemb.get_value().shape[0]
        self.decoder_units.append(self.wemb_size)
        '''=====Peeping Preparation====='''
        self.peeked_ends_ls = np.random.randint(self.n_train, size=(5,))
        self.ends_pool_ls = np.random.choice(range(self.n_train), 2000, replace = False)
        self.index2word_dict = pickle.load(open(self.index2word_dict_path))
       

    
    def eva_func(self, val_or_test):
        correct = 0.
        n_eva = None
        eva_story = None
        eva_ending1 = None
        eva_ending2 = None
        eva_answer = None

        minibatch_n = 50
        if val_or_test == 'val':
            n_eva = self.n_val
            eva_story = self.val_story
            eva_ending1 = self.val_ending1
            eva_ending2 = self.val_ending2
            eva_answer = self.val_answer
        else:
            n_eva = self.n_test
            eva_story = self.test_story
            eva_ending1 = self.test_ending1
            eva_ending2 = self.test_ending2
            eva_answer = self.test_answer 
                   
        max_batch_n = n_eva / minibatch_n
        residue = n_eva % minibatch_n

        for i in range(max_batch_n):

            story_ls = [[eva_story[index][j] for index in range(i*minibatch_n, (i+1)*minibatch_n)] for j in range(self.story_nsent)]
            story_matrix = [utils.padding(batch_sent) for batch_sent in story_ls]
            story_mask = [utils.mask_generator(batch_sent) for batch_sent in story_ls]

            ending1_ls = [eva_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending1_matrix = utils.padding(ending1_ls)
            ending1_mask = utils.mask_generator(ending1_ls)


            ending2_ls = [eva_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending2_matrix = utils.padding(ending2_ls)
            ending2_mask = utils.mask_generator(ending2_ls)

            score1, score2 = self.prediction(story_matrix[0], 
                                             story_matrix[1],
                                             story_matrix[2],
                                             story_matrix[3], 
                                             ending1_matrix,
                                             ending2_matrix,
                                             story_mask[0],
                                             story_mask[1],
                                             story_mask[2],
                                             story_mask[3],
                                             ending1_mask,
                                             ending2_mask)

            # Answer denotes the index of the anwer
            if self.loss_type == "hinge":
                prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
                correct_vec = prediction - eva_answer[i*minibatch_n:(i+1)*minibatch_n]
                correct += minibatch_n - (abs(correct_vec)).sum()
            else:
                answer1 = np.argmax(score1, axis = 1)
                answer2 = np.argmax(score2, axis = 1)
                correct_vec1 = (1 - answer1) - eva_answer[i*minibatch_n: (i+1)*minibatch_n]
                correct_vec2 = answer2 - eva_answer[i*minibatch_n: (i+1)*minibatch_n]
                correct += minibatch_n*2 - (abs(correct_vec1)).sum() - (abs(correct_vec2)).sum()


        story_ls = [[eva_story[index][j] for index in range(-residue, 0)] for j in range(self.story_nsent)]
        story_matrix = [utils.padding(batch_sent) for batch_sent in story_ls]
        story_mask = [utils.mask_generator(batch_sent) for batch_sent in story_ls]

        ending1_ls = [eva_ending1[index] for index in range(-residue, 0)]
        ending1_matrix = utils.padding(ending1_ls)
        ending1_mask = utils.mask_generator(ending1_ls)


        ending2_ls = [eva_ending2[index] for index in range(-residue, 0)]
        ending2_matrix = utils.padding(ending2_ls)
        ending2_mask = utils.mask_generator(ending2_ls)

        score1, score2 = self.prediction(story_matrix[0],
                                         story_matrix[1],
                                         story_matrix[2],
                                         story_matrix[3], 
                                         ending1_matrix,
                                         ending2_matrix,
                                         story_mask[0],
                                         story_mask[1],
                                         story_mask[2],
                                         story_mask[3],
                                         ending1_mask,
                                         ending2_mask)

        if self.loss_type == "hinge":
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - eva_answer[-residue:]
            correct += minibatch_n - (abs(correct_vec)).sum()
            return correct/n_eva
        else:
            answer1 = np.argmax(score1, axis = 1)
            answer2 = np.argmax(score2, axis = 1)
            correct_vec1 = (1 - answer1) - eva_answer[-residue:]
            correct_vec2 = answer2 - eva_answer[-residue:]
            correct += minibatch_n*2 - (abs(correct_vec1)).sum() - (abs(correct_vec2)).sum()
            return correct/(2*n_eva)

 
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
        N_TRAIN_INS = len(self.train_ending)
        best_val_accuracy = 0
        best_test_accuracy = 0
        test_threshold = 10000/N_BATCH

        '''init test'''
        print "initial test..."
        val_result = self.eva_func('val')
        print "val set accuracy: ", val_result*100, "%"
        if val_result > best_val_accuracy:
            best_val_accuracy = val_result

        test_accuracy = self.eva_func('test')
        print "test set accuracy: ", test_accuracy * 100, "%"
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        '''end of init test'''

        for epoch in range(N_EPOCHS):
            print "epoch ", epoch,":"
            shuffled_index_list = utils.shuffle_index(N_TRAIN_INS)

            max_batch = N_TRAIN_INS/N_BATCH

            start_time = time.time()

            total_disc_cost = 0.0
            total_gene_cost = 0.0
            total_correct_count = 0.0
            total_penalty_cost = 0.0
            total_decoding_cost = 0.0 
            fake_endings_mean = []
            fake_endings_std = []
            real_endings_mean = []
            real_endings_std = []


            max_score_index = None

            for batch in range(max_batch):

                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[self.train_story[index][i] for index in batch_index_list] for i in range(1, self.story_nsent+1)]
                train_ending = [self.train_ending[index] for index in batch_index_list]

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end_matrix = utils.padding(train_ending)
                # train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end_mask = utils.mask_generator(train_ending)
                # train_end2_mask = utils.mask_generator(end2)


                result_list = self.train_func(train_story_matrices[0],
                                              train_story_matrices[1],
                                              train_story_matrices[2],
                                              train_story_matrices[3],
                                              train_end_matrix,
                                              train_story_mask[0],
                                              train_story_mask[1],
                                              train_story_mask[2],
                                              train_story_mask[3],
                                              train_end_mask)


                discrim_cost = result_list[0]
                generat_cost = result_list[1]
                #max_score_index = result[2]
                score1 = result_list[2]
                score2 = result_list[3]
                ending_rep = result_list[4]
                decoding_cost = result_list[5]

                if batch % 1000 == 0 and batch != 0:
                    fake_ending = self.monitor_func()
                    fake_ending_mean = fake_ending.sum(axis = 0)/self.batchsize
                    fake_ending_std = (abs(fake_ending - fake_ending_mean)).sum()/(self.batchsize * self.sent_rnn_units)
                    fake_endings_mean.append(fake_ending_mean)
                    fake_endings_std.append(fake_ending_std)

                    real_ending_mean = ending_rep.sum(axis = 0)/self.batchsize
                    real_ending_std = (abs(ending_rep - real_ending_mean)).sum()/(self.batchsize * self.sent_rnn_units)
                    real_endings_mean.append(real_ending_mean)
                    real_endings_std.append(real_ending_std)


                if self.loss_type == "hinge":
                    total_correct_count += np.count_nonzero((score1.flatten() - score2.flatten()).clip(0.0))
                else:
                    correct1 = np.argmax(score1, axis = 1).sum()
                    correct2 = (1-np.argmax(score2, axis = 1)).sum()
                    total_correct_count += (correct1 + correct2)

                total_disc_cost += discrim_cost
                total_gene_cost += generat_cost
                total_decoding_cost += decoding_cost

                if batch % test_threshold == 0 and batch != 0:
                    train_acc = 0.0
                    if self.loss_type == "hinge":
                        train_acc = total_correct_count / ((batch+1) * N_BATCH)*100.0
                    else:
                        train_acc = total_correct_count / ((batch+1) * N_BATCH * 2)*100.0
                    print "accuracy on training set: ", train_acc, "%"
                    print "example score sequence"
                    print np.stack((score1, score2), axis = 1)
                    print "test on val set..."
                    val_result = self.eva_func('val')
                    print "accuracy is: ", val_result*100, "%"
                    if val_result > best_val_accuracy:
                        print "new best! test on test set..."
                        best_val_accuracy = val_result

                        test_accuracy = self.eva_func('test')
                        print "test set accuracy: ", test_accuracy*100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
                    print "discriminator cost per instances:", total_disc_cost/(batch+1)
                    print "generator cost per instances:", total_gene_cost/(batch+1)
                    print "decoding cost per instances:", total_decoding_cost/(batch+1)

                    '''===================================================='''
                    '''randomly print out a story and it's higest score end'''
                    '''competitive in a minibatch                          '''
                    '''===================================================='''                   
             #       print "example negative ending:"

             #       rand_index = np.random.randint(self.batchsize)
             #       index = shuffled_index_list[N_BATCH * batch + rand_index]

             #       story_string = " | ".join([" ".join([self.index2word_dict[self.train_story[index][j][k]] for k in range(len(self.train_story[index][j]))]) for j in range(5)])
             #       story_end = " ".join([self.index2word_dict[self.train_ending[index][k]] for k in range(len(self.train_ending[index]))])
             #       highest_score_end = " ".join([self.index2word_dict[self.train_ending[max_score_index[rand_index]][k]] for k in range(len(self.train_ending[max_score_index[rand_index]]))])

             #       print story_string 
             #       print " #END# " + story_end
             #       print ""
             #       print "Highest Score Ending in this batch: " + highest_score_end 
             #       print ""
                    pickle.dump([fake_endings_mean, fake_endings_std, real_endings_mean, real_endings_std],
                                open('../../data/pickles/decoder_adv_mean_std_record.pkl', 'w'))






            print "======================================="
            print "epoch summary:"
            if self.loss_type == "hinge":
                acc = total_disc_cost / max_batch
            else:
                acc = total_disc_cost / (max_batch * 2)
            print "average cost in this epoch: ", acc 
            print "average speed: ", N_TRAIN_INS/(time.time() - start_time), "instances/s "
            print "accuracy for this epoch: "+str(total_correct_count/(max_batch * N_BATCH) * 100.0)+"%"
 
            print "======================================="


   
if __name__ == '__main__':
    model = Hierachi_RNN(*sys.argv[1:])
    
    print "your choice of regularization is:"
    model.regularization_show()
    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    

