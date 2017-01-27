import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import BLSTM_Encoder
import DNN_score_function as DNN
import sys
# from theano.printing import pydotprint




class Hierachi_RNN(object):
    def __init__(self, 
                 best_val_model_save_path,
                 word_rnn_setting, 
                 sent_rnn_setting,
                 batchsize, 
                 wemb_trainable = 0,
                 learning_rate = 0.001,
                 delta = 1.0,
                 mode = 'sequence',
                 nonlin_func = 'default',
                 story_rep_type = 'concatenate',
                 score_func = 'RNN',
                 loss_type = 'hinge',
                 dnn_discriminator_setting = '512x1',
                 discrim_regularization_level = 0,
                 epochs = 100,
                 dropout_rate = 0.2,
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
        self.train_set_path = '../../data/pickles/val_train_set_index_corpus.pkl'
        self.val_set_path = '../../data/pickles/val_test_set_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.index2word_dict_path = '../../data/pickles/index2word_dict.pkl'
        self.word2index_dict_path = '../../data/pickles/word2index_dict.pkl'
        self.best_val_model_save_path = best_val_model_save_path
        # self.best_val_model_save_path = './attention_best_model_'+mode+story_rep_type+score_func+loss_type+dnn_discriminator_setting+str(discrim_regularization_level)+str(dropout_rate)+'.pkl' 
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
        self.learning_rate = float(learning_rate)

        self.mode = mode
        self.GRAD_CLIP = 10.0
        if nonlin_func == 'default':
            self.nonlin_func = lasagne.nonlinearities.tanh
        else:
            self.nonlin_func = None

        self.story_rep_type = story_rep_type
        self.bias = 0.001
        self.dnn_discriminator_setting = map(int, dnn_discriminator_setting.split('x'))
        self.loss_type = loss_type
        self.score_func = score_func
        self.discrim_regularization_level = int(discrim_regularization_level)
        self.epochs = int(epochs)
        self.dropout_rate = float(dropout_rate)

        self.bilinear_attention_matrix = theano.shared(0.02*np.random.normal(size = (self.sent_rnn_units[0], self.sent_rnn_units[0])))
        


        self.discrim_regularization_dict = {0:"no regularization on discriminator",
                                            1:"L2 on discriminator DNN",
                                            2:"L2 on discriminator word level RNN + DNN",
                                            3:"L2 on discriminator word level RNN",
                                            4:"L2 on discriminator all RNN",
                                            5:"L2 on discriminator all level"}
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



    def encoding_layer(self):
        for i in range(self.story_nsent+2):
            sent_seq = lasagne.layers.get_output(self.encoder.output,
                                                        {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                         self.encoder.l_mask:self.inputs_masks[i]},
                                                         deterministic = True)

            self.train_encodinglayer_vecs.append(sent_seq)
        end1_representation = (self.train_encodinglayer_vecs[4] * self.inputs_masks[4].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[4].sum(axis = 1, keepdims = True)
        end2_representation = (self.train_encodinglayer_vecs[5] * self.inputs_masks[5].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[5].sum(axis = 1, keepdims = True)
        self.train_encodinglayer_vecs.append(end1_representation)
        self.train_encodinglayer_vecs.append(end2_representation)
 
    def attention_layer(self):        
        self.attentioned_sent_rep1 = []
        self.attentioned_sent_rep2 = []
        for i in range(self.story_nsent):
            n_batch, n_seq, _= self.train_encodinglayer_vecs[i].shape

            #second attention

            bili_part1 = T.dot(self.train_encodinglayer_vecs[i], self.bilinear_attention_matrix)

            attention1_score_tensor = T.batched_dot(bili_part1, self.train_encodinglayer_vecs[6])
            attention2_score_tensor = T.batched_dot(bili_part1, self.train_encodinglayer_vecs[7])

            numerator1 = self.inputs_masks[i] * T.exp(attention1_score_tensor - attention1_score_tensor.max(axis = 1, keepdims = True))
            numerator2 = self.inputs_masks[i] * T.exp(attention2_score_tensor - attention2_score_tensor.max(axis = 1, keepdims = True))

            attention1_weight_matrix = numerator1 / numerator1.sum(axis = 1, keepdims = True)
            attention2_weight_matrix = numerator2 / numerator2.sum(axis = 1, keepdims = True)

            attentioned_sent_seq1 = self.train_encodinglayer_vecs[i]*(attention1_weight_matrix.reshape([n_batch, n_seq, 1]))
            attentioned_sent_seq2 = self.train_encodinglayer_vecs[i]*(attention2_weight_matrix.reshape([n_batch, n_seq, 1]))

            attentioned_sent_rep1 = T.sum(attentioned_sent_seq1, axis = 1) / T.sum(self.inputs_masks[i], axis = 1).reshape([-1, 1])
            attentioned_sent_rep2 = T.sum(attentioned_sent_seq2, axis = 1) / T.sum(self.inputs_masks[i], axis = 1).reshape([-1, 1])

            self.attentioned_sent_rep1.append(attentioned_sent_rep1)
            self.attentioned_sent_rep2.append(attentioned_sent_rep2)




    def batch_cosine(self, batch_vectors1, batch_vectors2):
        dot_prod = T.batched_dot(batch_vectors1, batch_vectors2)

        batch1_norm = T.sqrt((T.sqr(batch_vectors1)).sum(axis = 1))
        batch2_norm = T.sqrt((T.sqr(batch_vectors2)).sum(axis = 1))

        batch_cosine_vec = dot_prod/(batch1_norm * batch2_norm)
        return batch_cosine_vec.reshape((-1,1))

    def penalty_generator(self, params):
        penalty = [lasagne.regularization.l2(param) for param in params]
        penalty_scalar = lasagne.objectives.aggregate(T.stack(penalty), mode = "mean")
        return penalty_scalar

    def discrim_cost_generator(self):
        if self.discrim_regularization_level == 0:
            self.all_discrim_cost = self.discrim_cost

        elif self.discrim_regularization_level == 1:
            penalty = self.penalty_generator(self.DNN_score_func.all_params)
            self.all_discrim_cost = self.discrim_cost + self.regularization_index * penalty

        elif self.discrim_regularization_level == 2:
            penalty1 = self.penalty_generator(self.DNN_score_func.all_params)
            penalty2 = self.penalty_generator(sefl.encoder.all_params)
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * penalty1 + \
                                    self.regularization_index * penalty2

        elif self.discrim_regularization_level == 3:
            penalty = self.penalty_generator(self.encoder.all_params)
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * penalty

        elif self.discrim_regularization_level == 4:
            penalty1 = self.penalty_generator(self.encoder.all_params)
            penalty2 = self.penalty_generator(self.reasoner.all_params)
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * penalty1 + \
                                    self.regularization_index * penalty2

        else:
            penalty1 = self.penalty_generator(self.DNN_score_func.all_params)
            penalty2 = self.penalty_generator(self.encoder.all_params)
            self.all_discrim_cost = self.discrim_cost + \
                                    self.regularization_index * penalty1 + \
                                    self.regularization_index * penalty2 + \
                                    self.regularization_index * penalty3
                                    
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
        self.attention_layer()
        if self.story_rep_type == "concatenate":
            merge_ls1 = [tensor.dimshuffle(0,'x',1) for tensor in self.attentioned_sent_rep1]

            encode_merge1 = T.concatenate(merge_ls1, axis = 1)

            plot_rep1 = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge1})
            merge_ls2 = [tensor.dimshuffle(0,'x',1) for tensor in self.attentioned_sent_rep1]

            encode_merge2 = T.concatenate(merge_ls2, axis = 1)

            plot_rep2 = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge2})


            end1_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: self.train_encodinglayer_vecs[6].dimshuffle(0,'x',1)})
            end2_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: self.train_encodinglayer_vecs[7].dimshuffle(0,'x',1)})

            self.story1_rep = T.concatenate([plot_rep1, end1_rep], axis = 1)
            self.story2_rep = T.concatenate([plot_rep2, end2_rep], axis = 1)

            self.DNN_score_func = DNN.DNN(INPUTS_SIZE = self.sent_rnn_units[-1]*2, 
                                          LAYER_UNITS = self.dnn_discriminator_setting, 
                                          final_nonlin = self.nonlin_func)  

        else:
            merge_ls1 = [tensor.dimshuffle(0,'x',1) for tensor in self.attentioned_sent_rep1 + [self.train_encodinglayer_vecs[6]]]
            merge_ls2 = [tensor.dimshuffle(0,'x',1) for tensor in self.attentioned_sent_rep2 + [self.train_encodinglayer_vecs[7]]]

            encode_merge1 = T.concatenate(merge_ls1, axis = 1)
            encode_merge2 = T.concatenate(merge_ls2, axis = 1)
            self.story1_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge1})
            self.story2_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge2})

            self.DNN_score_func = DNN.DNN(INPUTS_SIZE = self.sent_rnn_units[-1], 
                                          LAYER_UNITS = self.dnn_discriminator_setting, 
                                          final_nonlin = self.nonlin_func)  
        #creating DNN score function

        self.score1 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                               {self.DNN_score_func.l_in: self.story1_rep}, deterministic = False)


        self.score11 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                               {self.DNN_score_func.l_in: self.story1_rep}, deterministic = True)

        self.score2 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                               {self.DNN_score_func.l_in: self.story2_rep}, deterministic = False)

        self.score22 = lasagne.layers.get_output(self.DNN_score_func.output, 
                                               {self.DNN_score_func.l_in: self.story2_rep}, deterministic = True)




        # Construct symbolic cost function
        answer = T.vector('answer', dtype= 'int64')

        if self.dnn_discriminator_setting[-1] == 1:
            target1 = 2.0 * answer - 1.0
            target2 = - 2.0 * answer + 1.0
            cost = target1.dimshuffle(0,'x') * self.score1 + target2.dimshuffle(0,'x') * self.score2 + self.delta
            
            self.cost = T.gt(cost, 0.0) * cost
            self.discrim_cost = lasagne.objectives.aggregate(self.cost, mode = 'mean')

        else:
            cost1 = lasagne.objectives.categorical_crossentropy(prob1, 1-anwer)
            cost2 = lasagne.objectives.categorical_crossentropy(prob2, answer)
            self.discrim_cost = lasagne.objectives.aggregate(cost1+cost2, mode = 'mean')
        
        self.discrim_cost_generator()
        penalty = self.all_discrim_cost - self.discrim_cost
            
        # Retrieve all parameters from the network
        all_params = self.encoder.all_params + self.reasoner.all_params + self.DNN_score_func.all_params

        all_updates = lasagne.updates.adam(self.all_discrim_cost, all_params, learning_rate=self.learning_rate)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        self.train_func = theano.function(self.inputs_variables + self.inputs_masks + [answer], 
                                         [self.discrim_cost, penalty, self.score1, self.score2], updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function(self.inputs_variables + self.inputs_masks, [self.score11, self.score22])

        # pydotprint(self.train_func, './computational_graph.png')

    def load_data(self):
        # train_set = pickle.load(open(self.train_set_path))
        # self.train_story = train_set[0]
        # self.train_ending = train_set[1]

        with open(self.train_set_path, 'r') as f:
            train_set = pickle.load(f)

        self.val_train_story = train_set[0]
        self.val_train_end1 = train_set[1]
        self.val_train_end2 = train_set[2]
        self.val_train_answer = train_set[3]

        self.train_n = len(self.val_train_answer)
        
        with open(self.val_set_path, 'r') as f:
            val_set = pickle.load(f)

        self.val_test_story = val_set[0]
        self.val_test_end1 = val_set[1]
        self.val_test_end2 = val_set[2]
        self.val_test_answer = val_set[3]

        self.val_n = len(self.val_test_answer)
        with open(self.test_set_path, 'r') as f:
            test_set = pickle.load(f)
        self.test_story = test_set[0]
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_answer)
        
        with open(self.wemb_matrix_path, 'r') as f:
            self.wemb = theano.shared(pickle.load(f)).astype(theano.config.floatX)
        with open(self.index2word_dict_path, 'r') as f:
            self.index2word_dict = pickle.load(f)
        with open(self.word2index_dict_path, 'r') as f:
            self.word2index_dict = pickle.load(f)

    def eva_func(self, val_or_test):
        correct = 0.
        n_eva = None
        eva_story = None
        eva_ending1 = None
        eva_ending2 = None
        eva_answer = None

        minibatch_n = 50
        if val_or_test == 'val':
            n_eva = self.val_n
            eva_story = self.val_test_story
            eva_ending1 = self.val_test_end1
            eva_ending2 = self.val_test_end2
            eva_answer = self.val_test_answer
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


    def saving_model(self, accuracy):
        encoder_params_value = lasagne.layers.get_all_param_values(self.encoder.output)
        reasoner_params_value = lasagne.layers.get_all_param_values(self.reasoner.output)
        classif_params_value = lasagne.layers.get_all_param_values(self.DNN_score_func.output)
        attention_matrix = self.bilinear_attention_matrix.get_value()
        pickle.dump((encoder_params_value, reasoner_params_value, classif_params_value, attention_matrix, accuracy), 
                    open(self.best_val_model_save_path, 'wb'))

    def reload_model(self):
        encoder_params, reasoner_params, classif_params, attention_matrix, accuracy = pickle.load(open(self.best_val_model_save_path))
        lasagne.layers.set_all_param_values(self.encoder.output, encoder_params)
        lasagne.layers.set_all_param_values(self.reasoner.output, reasoner_params)
        lasagne.layers.set_all_param_values(self.DNN_score_func.output, classif_params)
        self.bilinear_attention_matrix.set_value(attention_matrix)

        print "This model has ", accuracy * 100, "%  accuracy on valid set" 

    def begin_train(self):
        N_EPOCHS = self.epochs
        N_BATCH = self.batchsize
        N_TRAIN_INS = self.train_n
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

            total_cost = 0.0


            total_correct_count = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[self.val_train_story[index][i] for index in batch_index_list] for i in range(self.story_nsent)]
                end1 = [self.val_train_end1[index] for index in batch_index_list]
                end2 = [self.val_train_end2[index] for index in batch_index_list]
                # neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # while np.any((np.asarray(batch_index_list) - neg_end_index_list) == 0):
                #     neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # neg_end1 = [self.train_ending[index] for index in neg_end_index_list]
                # answer = np.random.randint(2, size = N_BATCH)
                # target1 = 1 - answer
                # target2 = 1 - target1
                answer = np.asarray([self.val_train_answer[index] for index in batch_index_list])

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end1_matrix = utils.padding(end1)
                train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end1_mask = utils.mask_generator(end1)
                train_end2_mask = utils.mask_generator(end2)
               # print len(train_story)
               # print train_end1_matrix.shape
               # print train_end2_matrix.shape
               # print train_end1_mask.shape
               # print train_end2_mask.shape

                results = self.train_func(train_story_matrices[0],
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
                                          answer)

                cost = results[0]
                regularization_sum = results[1]
                score1 = results[2]
                score2 = results[3]


                if self.loss_type == "hinge":
                    total_correct_count += N_BATCH - abs(np.argmax(np.concatenate([score1, score2], axis = 1), axis = 1) - answer).sum()
                else:
                    correct1 = abs(np.argmax(score1, axis = 1) - answer).sum()
                    correct2 = abs(np.argmax(score2, axis = 1) - 1 + answer).sum()
                    total_correct_count += (correct1 + correct2)

                total_cost += cost

            print "======================================="
            print "epoch summary:"
            if self.loss_type == "hinge":
                acc = total_cost / max_batch
                train_acc = total_correct_count / (max_batch * N_BATCH)*100.0
            else:
                acc = total_cost / (max_batch * 2)
                train_acc = total_correct_count / (max_batch * N_BATCH * 2)*100.0
            print "total cost in this epoch: ", acc
            print "overall accuracy: ", train_acc , "%"
            val_result = self.eva_func('val')
            print "val set accuracy: ", val_result*100, "%"
            if val_result > best_val_accuracy:
                best_val_accuracy = val_result
                print "saving model..."
                self.saving_model(best_val_accuracy)
                print "saving complete"
            test_accuracy = self.eva_func('test')
            print "test set accuracy: ", test_accuracy * 100, "%"
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
            '''end of init test'''
            print "======================================="

    def list_demo(self, checking_indices):

        indices_ls = None
        if '-' in checking_indices:
            indices_ls = map(int, checking_indices.split('-'))
        elif ',' in checking_indices:
            indices_ls = map(int, checking_indices.split(','))
        elif checking_indices.isdigit():
            indices_ls = [int(checking_indices)]

        demo_story = [[self.val_train_story[index][i] for index in indices_ls] for i in range(self.story_nsent)]
        end1 = [self.val_train_end1[index] for index in indices_ls]
        end2 = [self.val_train_end2[index] for index in indices_ls]

        answer = np.asarray([self.val_train_answer[index] for index in indices_ls])

        demo_story_matrices = [utils.padding(batch_sent) for batch_sent in demo_story]
        demo_end1_matrix = utils.padding(end1)
        demo_end2_matrix = utils.padding(end2)

        demo_story_mask = [utils.mask_generator(batch_sent) for batch_sent in demo_story]
        demo_end1_mask = utils.mask_generator(end1)
        demo_end2_mask = utils.mask_generator(end2)


        score1, score2 = self.prediction(demo_story_matrices[0],
                                         demo_story_matrices[1],
                                         demo_story_matrices[2],
                                         demo_story_matrices[3], 
                                         demo_end1_matrix,
                                         demo_end2_matrix,
                                         demo_story_mask[0],
                                         demo_story_mask[1],
                                         demo_story_mask[2],
                                         demo_story_mask[3],
                                         demo_end1_mask,
                                         demo_end2_mask)

        self.print_func(indices_ls, score1, score2)
            
    def mask_demo(self, index, story_mask_indices, end1_mask_indices, end2_mask_indices):
        demo_story = None
        end1 = None
        end2 = None
        indices_ls = None
        answer = -1
        if index.isdigit():
            indices_ls = [int(index)]
            # check mask format
            demo_story = [[self.val_train_story[index][i] for index in indices_ls] for i in range(self.story_nsent)]
            end1 = [self.val_train_end1[index] for index in indices_ls]
            end2 = [self.val_train_end2[index] for index in indices_ls]

            answer = np.asarray([self.val_train_answer[index] for index in indices_ls])

        else:
            demo_story = self.demo_index_story
            end1 = self.demo_index_end1
            end2 = self.demo_index_end2
            answer = -1
            
        try:
            story_mask_unk_indices = [map(int, pair.split(',')) for pair in story_mask_indices.split(';')]
        except:
            story_mask_unk_indices = []
        try:
            end1_mask_unk_indices = map(int, end1_mask_indices.split(';'))
        except:
            end1_mask_unk_indices = []
        try:
            end2_mask_unk_indices = map(int, end2_mask_indices.split(';'))
        except:
            end2_mask_unk_indices = []


        if story_mask_unk_indices != []:
            for pair in story_mask_unk_indices:
                if len(pair) == 1:
                    for j in range(len(demo_story[pair[0]][0])):
                        demo_story[pair[0]][0][j] = 1
                else:
                    demo_story[pair[0]][0][pair[1]] = 1
        if end1_mask_unk_indices != []:
            for index in end1_mask_unk_indices:
                end1[0][index] = 1
        if end2_mask_unk_indices != []:
            for index in end2_mask_unk_indices:
                end2[0][index] = 1


        demo_story_matrices = [utils.padding(batch_sent) for batch_sent in demo_story]
        demo_end1_matrix = utils.padding(end1)
        demo_end2_matrix = utils.padding(end2)

        demo_story_mask = [utils.mask_generator(batch_sent) for batch_sent in demo_story]
        demo_end1_mask = utils.mask_generator(end1)
        demo_end2_mask = utils.mask_generator(end2)


        score1, score2 = self.prediction(demo_story_matrices[0],
                                         demo_story_matrices[1],
                                         demo_story_matrices[2],
                                         demo_story_matrices[3], 
                                         demo_end1_matrix,
                                         demo_end2_matrix,
                                         demo_story_mask[0],
                                         demo_story_mask[1],
                                         demo_story_mask[2],
                                         demo_story_mask[3],
                                         demo_end1_mask,
                                         demo_end2_mask)
        self.single_print_func(demo_story, end1, end2, score1, score2, answer)

    def print_func(self, indices_ls, score1 = [0], score2 = [0]):

        for i in range(len(indices_ls)):
            index = indices_ls[i]
            story_string = "\n".join([" ".join([self.index2word_dict[self.val_train_story[index][j][k]] for k in range(len(self.val_train_story[index][j]))]) for j in range(4)])
            story_end1 = " ".join([self.index2word_dict[self.val_train_end1[index][k]] for k in range(len(self.val_train_end1[index]))])
            story_end2 = " ".join([self.index2word_dict[self.val_train_end2[index][k]] for k in range(len(self.val_train_end2[index]))])

            answer = self.val_train_answer[i]
            
                
            print story_string 
            print " #END1# " + story_end1, score1[i]
            print " #END2# " + story_end2, score2[i]
            print " #ANSWER# ", answer + 1
            print ""
            
    def single_print_func(self, story_tensor, end1_matrix, end2_matrix, score1 = 0, score2 = 0, answer = -1):
        print story_tensor 
        story_string = "\n".join([" ".join([self.index2word_dict[story_tensor[j][0][k]] for k in range(len(story_tensor[j][0]))]) for j in range(4)])
        story_end1 = " ".join([self.index2word_dict[end1_matrix[0][k]] for k in range(len(end1_matrix[0]))])
        story_end2 = " ".join([self.index2word_dict[end2_matrix[0][k]] for k in range(len(end2_matrix[0]))])

        print story_string 
        print " #END1# " + story_end1, score1
        print " #END2# " + story_end2, score2
        print " #ANSWER# ", answer + 1
        print ""

    def sent2index(self, sent):
        sent_index_ls = [self.word2index_dict['<s>']]
        for word in sent:
            if word.lower() in self.word2index_dict:
                sent_index_ls.append(self.word2index_dict[word.lower()])
            elif word.lower().isdigit():
                sent_index_ls.append(self.word2index_dict['UNKNOWN_NUM'])
            else: 
                sent_index_ls.append(self.word2index_dict['UUNKNOWNN'])
        sent_index_ls.append(self.word2index_dict['</s>'])
        return sent_index_ls



    def user_input(self, story, end1, end2):
        tokenized_story = self.tokenization([story])
        tokenized_end1 = self.tokenization([[end1]])
        tokenized_end2 = self.tokenization([[end2]])
        
        tok_story = tokenized_story[0]
        tok_end1 = tokenized_end1[0]
        tok_end2 = tokenized_end2[0]
        self.demo_index_story = []
        for sent in tok_story:
            self.demo_index_story.append([self.sent2index(sent)])
        self.demo_index_end1 = [self.sent2index(tok_end1[0])]
        self.demo_index_end2 = [self.sent2index(tok_end2[0])]
           
    #recursive function to split surfix punctuation and possession problem
    def split_word(self, word):
        bad_end = ("#",'"','?','!','+','.',',',"'", '(',')',':','/',';','$','-','`','*','\\','%','&',']')
        bad_end_double = ('oz','lb','am','pm','th','ft')
        bad_end_triple = ('lbs','mph')
        bad_middle = ("-",",",".",":",'/',"*","\\",'&',"'",')','(','!')
        non_char = '\xef\xbf\xbd'
        single_abriev = ("'s","'d","'m")
        double_abriev = ("'ve", "'ll", "'re","n't")
        
        if word.lower() in self.word2index_dict:
            return [word]
        elif word.startswith(bad_end):
            return [word[0]] + split_word(word[1:])
        elif word.endswith(bad_end):
            return self.split_word(word[:-1]) + [word[-1]]
        elif word.endswith(single_abriev+bad_end_double):
            return self.split_word(word[:-2]) + [word[-2:]]
        elif word.endswith(double_abriev+bad_end_triple):
            return self.split_word(word[:-3]) + [word[-3:]]
        elif any((c in bad_middle) for c in word.lower()[1:-1]):
            for ch in bad_middle:
                if ch in word.lower()[1:-1]:
                    char_index = word.index(ch)
                    return self.split_word(word[:char_index+1]) + self.split_word(word[char_index+1:])
        elif non_char in word.lower():
            char_index = word.index(non_char)
            return self.split_word(word.lower()[:char_index])+["?"]+self.split_word(word.lower()[char_index])
        elif utils.number_with_units(word.lower()) != -1:
            index = utils.number_with_units(word.lower())
            return [word[:index]] + [word[index:]]
        else:
            return [word]
        
    #convert word list into tokens
    def split_sent(self, word_list):
        new_word_ls = []
        for word in word_list:
            new_word_ls += self.split_word(word)
        return new_word_ls

    def tokenization(self, story_list):
        tokenized_set = []
        for story in story_list:
            story_list = []
            for sent in story:
                word_list = sent.split(' ')
                new_word_list = self.split_sent(word_list)
                story_list.append(new_word_list)
            tokenized_set.append(story_list)
        return tokenized_set



   
if __name__ == '__main__':

    model = Hierachi_RNN(*sys.argv[1:])

    print "loading data"
    model.load_data()

    print "model construction..."
    model.model_constructor()
    print "construction complete!"

    print "parameters reloading..."
    model.reload_model()

    while True:
        print "===============instruction==============="
        print "enter 'multi' to test multiple instances "
        print "enter 'single' to test single instance \n with unknown coverage "
        print "enter 'quit' to exit this program "
        print "========================================="
        command = raw_input('I want: ')
        if command == 'quit':
            break
        elif command == 'multi':
            print "enter the indices of the instance you want to check"
            print "(e.g. 5-10 means from 5th to 10th instance and their score)"
            print "(e.g. 5,7,9,11 means show 5th, 7th, 9th, and 11th instance)"
            indices = raw_input('please enter indices: ')
            if indices == '':
                print "no input...plz try again"
                continue
            model.list_demo(indices)

        elif command == 'single':
            print "enter the indices of the instance you want to check"
            index = raw_input('index: ')
            if not index.isdigit():
                print "must be an integer"
                continue

            model.print_func([int(index)])
            print "enter the mask sequence "
            print "(e.g. 0,1;0,2;1,5;2,3;4,5)"
            story_mask_ls = raw_input("here we go: ")
            print "enter the first ending mask sequence:"
            print "(e.g. 1;2;5;7)"
            end1_mask_ls = raw_input("here we go: ")
            print "enter the second ending mask sequence:"
            print "(e.g. 1;2;5;7)"
            end2_mask_ls = raw_input("here we go: ")            
            model.mask_demo(index, story_mask_ls, end1_mask_ls, end2_mask_ls)
        elif command == 'user':
            story = []
            print "enter the 4 story plot sentences with enter to switch to next sentence:"
            for i in range(4):
                story.append(raw_input('sent'+str(i)+': '))
            end1 = raw_input('end1: ')
            end2 = raw_input('end2: ')
            model.user_input(story, end1, end2)
            model.single_print_func(model.demo_index_story, model.demo_index_end1, model.demo_index_end2)
            mask_flag = raw_input("Do you want to mask input(y/n)")
            if mask_flag == 'y':
                print "enter the mask sequence "
                print "(e.g. 0,1;0,2;1,5;2,3;4,5)"
                story_mask_ls = raw_input("here we go: ")
                print "enter the first ending mask sequence:"
                print "(e.g. 1;2;5;7)"
                end1_mask_ls = raw_input("here we go: ")
                print "enter the second ending mask sequence:"
                print "(e.g. 1;2;5;7)"
                end2_mask_ls = raw_input("here we go: ")            
                model.mask_demo('user', story_mask_ls, end1_mask_ls, end2_mask_ls)               
            else:
                model.mask_demo('user',[],[],[])
        else:
            print "command not found, please try again. "
            continue



 

