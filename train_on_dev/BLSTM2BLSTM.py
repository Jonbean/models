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
                 word_rnn_setting, 
                 sent_rnn_setting,
                 batchsize, 
                 cross_val_index,
                 wemb_trainable = 1,
                 learning_rate = 0.001,
                 delta = 1.0,
                 mode = 'sequence',
                 nonlin_func = 'default',
                 story_rep_type = 'concatenate',
                 score_func = 'DNN',
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
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.saving_path_suffix = mode+'-'+story_rep_type+'-'+score_func+loss_type+'-'+dnn_discriminator_setting+'-'+str(discrim_regularization_level)+'-'+str(dropout_rate)+'-'+cross_val_index
        self.best_val_model_save_path = '/share/data/speech/Data/joncai/dev_best_model/hierNonAttH_best_model_'+self.saving_path_suffix+'.pkl' 
        
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
        elif nonlin_func == 'relu':
            self.nonlinearities = lasagne.nonlinearities.retify
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
        self.discrim_regularization_dict = {0:"no regularization on discriminator",
                                            1:"L2 on discriminator DNN",
                                            2:"L2 on discriminator word level RNN + DNN",
                                            3:"L2 on discriminator word level RNN",
                                            4:"L2 on discriminator all RNN",
                                            5:"L2 on discriminator all level"}
        self.regularization_index = float(regularization_index)
        self.record_flag = False

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

        if self.story_rep_type == "concatenate":
            merge_ls = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:4]]

            encode_merge = T.concatenate(merge_ls, axis = 1)

            plot_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge})

            end1_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: self.train_encodinglayer_vecs[4].dimshuffle(0,'x',1)})
            end2_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: self.train_encodinglayer_vecs[5].dimshuffle(0,'x',1)})

            self.story1_rep = T.concatenate([plot_rep, end1_rep], axis = 1)
            self.story2_rep = T.concatenate([plot_rep, end2_rep], axis = 1)

            self.DNN_score_func = DNN.DNN(INPUTS_SIZE = self.sent_rnn_units[-1]*2, 
                                          LAYER_UNITS = self.dnn_discriminator_setting, 
                                          final_nonlin = self.nonlin_func,
                                          dropout_rate = self.dropout_rate)  

        else:
            merge_ls1 = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:5]]
            merge_ls2 = [tensor.dimshuffle(0,'x',1) for tensor in self.train_encodinglayer_vecs[:4]] + \
                        [self.train_encodinglayer_vecs[5].dimshuffle(0,'x',1)]

            encode_merge1 = T.concatenate(merge_ls1, axis = 1)
            encode_merge2 = T.concatenate(merge_ls2, axis = 1)
            self.story1_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge1})
            self.story2_rep = lasagne.layers.get_output(self.reasoner.output, {self.reasoner.l_in: encode_merge2})

            self.DNN_score_func = DNN.DNN(INPUTS_SIZE = self.sent_rnn_units[-1], 
                                          LAYER_UNITS = self.dnn_discriminator_setting, 
                                          final_nonlin = self.nonlin_func,
                                          dropout_rate = self.dropout_rate)  
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
            prob1 = lasagne.nonlinearities.softmax(self.score11)
            prob2 = lasagne.nonlinearities.softmax(self.score22)
            cost1 = lasagne.objectives.categorical_crossentropy(prob1, 1-answer)
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
        with open(self.val_set_path, 'r') as f:
            real_val_set = pickle.load(f)
        val_story = real_val_set[0]
        val_ending1 = real_val_set[1]
        val_ending2 = real_val_set[2]
        val_answer = real_val_set[3]

        shuffle_list = np.random.permutation(len(val_answer))
        val_train_ls = shuffle_list[:1500]
        val_val_ls = shuffle_list[1500:]

        print len(val_train_ls)

        print len(val_val_ls)
        self.val_train_story = [val_story[i] for i in val_train_ls]
        self.val_train_end1 = [val_ending1[i] for i in val_train_ls]
        self.val_train_end2 = [val_ending2[i] for i in val_train_ls]
        self.val_train_answer = [val_answer[i] for i in val_train_ls]

        self.train_n = len(self.val_train_answer)
        
        self.val_test_story = [val_story[i] for i in val_val_ls]
        self.val_test_end1 = [val_ending1[i] for i in val_val_ls]
        self.val_test_end2 = [val_ending2[i] for i in val_val_ls]
        self.val_test_answer = [val_answer[i] for i in val_val_ls]

        self.val_n = len(self.val_test_answer)
        with open(self.test_set_path,'r') as f:
            test_set = pickle.load(f)
        self.test_story = test_set[0]
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_answer)

        self.wemb = theano.shared(pickle.load(open(self.wemb_matrix_path))).astype(theano.config.floatX)

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
        if self.record_flag:
            self.test_score_record_matrix = np.zeros((self.n_test, 2))
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
            score_matrix = np.concatenate((score1, score2), axis = 1)
            if self.record_flag:
                self.test_score_record_matrix[i*minibatch_n: (i+1)*minibatch_n,:] = score_matrix

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

        score_matrix = np.concatenate((score1, score2), axis = 1)
        if self.record_flag:
            self.test_score_record_matrix[-residue:,:] = score_matrix

        if self.loss_type == "hinge":
            prediction = np.argmax(score_matrix, axis=1)
            correct_vec = prediction - eva_answer[-residue:]
            correct += minibatch_n - (abs(correct_vec)).sum()
            acc = correct/n_eva
            if val_or_test == 'val' and self.best_val_accuracy <= acc:
                self.record_flag = True
            elif val_or_test == 'test' and self.record_flag:
                self.record_flag = False
                with open('./test_score_record_matrix'+self.saving_path_suffix+',pkl') as f:
                    pickle.dump(self.test_score_record_matrix, f)
            return acc

        else:
            answer1 = np.argmax(score1, axis = 1)
            answer2 = np.argmax(score2, axis = 1)
            correct_vec1 = (1 - answer1) - eva_answer[-residue:]
            correct_vec2 = answer2 - eva_answer[-residue:]
            correct += minibatch_n*2 - (abs(correct_vec1)).sum() - (abs(correct_vec2)).sum()

            acc = correct/(2*n_eva)
            if val_or_test == 'val' and self.best_val_accuracy <= acc:
                self.record_flag = True
            elif val_or_test == 'test' and self.record_flag:
                self.record_flag = False
                with open('./test_score_record_matrix'+self.saving_path_suffix+'.pkl','w') as f:
                    pickle.dump(self.test_score_record_matrix, f)
            return acc

    def saving_model(self, accuracy):
        encoder_params_value = lasagne.layers.get_all_param_values(self.encoder.output)
        reasoner_params_value = lasagne.layers.get_all_param_values(self.reasoner.output)
        classif_params_value = lasagne.layers.get_all_param_values(self.DNN_score_func.output)
        pickle.dump((encoder_params_value, reasoner_params_value, classif_params_value, accuracy), 
                    open(self.best_val_model_save_path, 'wb'))

    def reload_model(self):
        encoder_params, reasoner_params, classif_params, accuracy = pickle.load(open(self.best_val_model_save_path))
        lasagne.layers.set_all_param_values(self.encoder.output, encoder_params)
        lasagne.layers.set_all_param_values(self.reasoner.output, reasoner_params)
        lasagne.layers.set_all_param_values(self.classify_layer.output, classif_params)

        print "This model has ", accuracy * 100, "%  accuracy on valid set" 



    def begin_train(self):
        N_EPOCHS = self.epochs
        N_BATCH = self.batchsize
        N_TRAIN_INS = self.train_n
        self.best_val_accuracy = 0
        self.best_test_accuracy = 0
        test_threshold = 10000/N_BATCH

        '''init test'''
        print "initial test..."
        val_result = self.eva_func('val')
        print "val set accuracy: ", val_result*100, "%"
        if val_result > self.best_val_accuracy:
            self.best_val_accuracy = val_result

        test_accuracy = self.eva_func('test')
        print "test set accuracy: ", test_accuracy * 100, "%"
        if test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = test_accuracy
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
            if val_result > self.best_val_accuracy:
                self.best_val_accuracy = val_result
                print "saving model..."
                self.saving_model(self.best_val_accuracy)
                print "saving complete"
 
            test_accuracy = self.eva_func('test')
            print "test set accuracy: ", test_accuracy * 100, "%"
            if test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = test_accuracy
            '''end of init test'''
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
 

