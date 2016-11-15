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
                 score_func_nonlin = 'default',
                 wemb_trainable = 1,
                 learning_rate = 0.001,
                 delta = 1.0,
                 wemb_size = None):
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

        self.rnn_units = int(rnn_setting)
        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.batchsize = int(batchsize)
        # self.reasoning_depth = int(reasoning_depth)
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

        self.delta = float(delta)
        if score_func_nonlin == 'default':
            self.score_func_nonlin = lasagne.nonlinearities.tanh
        else:
            self.score_func_nonlin = None
        self.wemb_trainable = bool(int(wemb_trainable))
        self.learning_rate = float(learning_rate)
    def matrix_cos(self, batch_rep1, batch_rep2):
        batch_rep1_broad = batch_rep1 + T.zeros((self.batch_m, self.batch_m, self.rnn_units))
        
        batch_rep2_broad = batch_rep2 + T.zeros((self.batch_m, self.batch_m, self.rnn_units))
        batch_rep1_reshape = batch_rep1_broad.reshape((-1, self.rnn_units))
        batch_rep2_reshape = batch_rep2_broad.reshape((-1, self.rnn_units))

        batch_dot = (T.batched_dot(batch_rep1_reshape, batch_rep2_reshape)).reshape((self.batch_m, self.batch_m))
        norm1 = T.sqrt(T.sum(T.sqr(batch_rep1), axis = 1))
        norm2 = T.sqrt(T.sum(T.sqr(batch_rep2), axis = 1))
        
        norm_matrix = T.dot(norm1.reshape((-1,1)),norm2.reshape((1,-1)))
        
        return batch_dot/norm_matrix

    def encoding_layer(self):


        assert len(self.reshaped_inputs_variables)==len(self.inputs_masks)
        for i in range(3):
            sent_seq = lasagne.layers.get_output(self.encoder.output,
                                                        {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                         self.encoder.l_mask:self.inputs_masks[i]},
                                                         deterministic = True)

            self.train_encodinglayer_vecs.append((sent_seq * self.inputs_masks[i].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[i].sum(axis = 1, keepdims = True))         


    def batch_cosine(self, batch_vectors1, batch_vectors2):
        dot_prod = T.batched_dot(batch_vectors1, batch_vectors2)


        batch1_norm = T.sqrt(T.sum(T.sqr(batch_vectors1), axis = 1))
        batch2_norm = T.sqrt(T.sum(T.sqr(batch_vectors2), axis = 1))
        batch_cosine_vec = dot_prod/(batch1_norm * batch2_norm)
        return batch_cosine_vec.dimshuffle(0,'x')

    def model_constructor(self, wemb_size = None):
        self.inputs_variables = []
        self.inputs_masks = []
        self.reshaped_inputs_variables = []

        '''here index 0 input is the story plot,
           index1 input is the ending1 input, 
           index2 input is the val/test set possible 2nd ending
        '''
        for i in range(3):
            self.inputs_variables.append(T.matrix('story'+str(i)+'_input', dtype='int64'))
            self.inputs_masks.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            self.reshaped_inputs_variables.append(self.inputs_variables[i].dimshuffle(0,1,'x'))

        #initialize neural network units
        self.encoder = BLSTM_sequence.BlstmEncoder(LSTMLAYER_1_UNITS = self.rnn_units, wemb_trainable = self.wemb_trainable)
        self.encoder.build_model(self.wemb)

        #build encoding layer
        self.encoding_layer()

        '''
        plot vector and ending vector pairs illustration

        -------------
        |1,2|1,3|1,4|
        -------------
        |2,1|2,3|2,4|
        -------------
        |3,1|3,2|3,4|
        -------------
        |4,1|4,2|4,3|
        -------------
        '''

        
        self.batch_m, rep_n = self.train_encodinglayer_vecs[0].shape
        score_matrix = self.matrix_cos(self.train_encodinglayer_vecs[0], self.train_encodinglayer_vecs[1]) 

        score1 = T.flatten(T.nlinalg.diag(score_matrix))

        all_other_score_matrix = score_matrix * (T.identity_like(score_matrix) - T.eye(self.batch_m)) + T.eye(self.batch_m) * T.min(score_matrix, axis = 1)

        max_other_score = T.max(all_other_score_matrix, axis = 1)
        max_score_index = T.argmax(all_other_score_matrix, axis = 1) 
        final_score = - score1 + self.delta + max_other_score

        score_vec = T.clip(final_score, 0.0, float('inf'))
        self.cost = lasagne.objectives.aggregate(score_vec, mode='mean') 


        # Retrieve all parameters from the network
        all_params = self.encoder.all_params 

        all_updates = lasagne.updates.adam(self.cost, all_params, learning_rate=self.learning_rate)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        self.train_func = theano.function(self.inputs_variables[:-1] + self.inputs_masks[:-1], 
                                        [self.cost, score1, max_other_score, max_score_index], updates = all_updates)

        # test ending2
        self.score2 = self.batch_cosine(self.train_encodinglayer_vecs[0], self.train_encodinglayer_vecs[2])
        self.score1 = self.batch_cosine(self.train_encodinglayer_vecs[0], self.train_encodinglayer_vecs[1])
        self.prediction = theano.function(self.inputs_variables + self.inputs_masks,
                                         [self.score1, self.score2])
        # pydotprint(self.train_func, './computational_graph.png')

    def load_data(self):
        '''======Train Set====='''
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = utils.combine_sents(train_set[0])
        self.train_ending = train_set[1]
        self.n_train = len(self.train_ending)
        
        '''=====Val Set====='''
        val_set = pickle.load(open(self.val_set_path))

        self.val_story = utils.combine_sents(val_set[0])
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.n_val = len(self.val_answer)

        '''=====Test Set====='''
        test_set = pickle.load(open(self.test_set_path))
        self.test_story = utils.combine_sents(test_set[0])
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
       


    def val_set_test(self):

        correct = 0.

        minibatch_n = 50
        max_batch_n = self.n_val / minibatch_n
        for i in range(max_batch_n):

            story_ls = [self.val_story[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            story_matrix = utils.padding(story_ls)
            story_mask = utils.mask_generator(story_ls)

            ending1_ls = [self.val_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending1_matrix = utils.padding(ending1_ls)
            ending1_mask = utils.mask_generator(ending1_ls)


            ending2_ls = [self.val_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending2_matrix = utils.padding(ending2_ls)
            ending2_mask = utils.mask_generator(ending2_ls)

            score1, score2 = self.prediction(story_matrix, 
                                             ending1_matrix, 
                                             ending2_matrix, 
                                             story_mask,
                                             ending1_mask, 
                                             ending2_mask)

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - self.val_answer[i*minibatch_n:(i+1)*minibatch_n]
            correct += minibatch_n - (abs(correct_vec)).sum()

        for k in range(minibatch_n * max_batch_n, self.n_val):
            story = np.asarray(self.val_story[k], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.val_story[k])))

            ending1 = np.asarray(self.val_ending1[k], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.val_ending1[k])))

            ending2 = np.asarray(self.val_ending2[k], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.val_ending2[k])))

            score1, score2 = self.prediction(story,
                                             ending1, 
                                             ending2,
                                             story_mask, 
                                             ending1_mask, 
                                             ending2_mask)
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            if prediction == self.val_answer[k]:
                correct += 1.    
        return correct/self.n_val

    def test_set_test(self):
        correct = 0.

        minibatch_n = 50
        max_batch_n = self.n_test / minibatch_n
        for i in range(max_batch_n):

            story_ls = [self.test_story[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            story_matrix = utils.padding(story_ls)
            story_mask = utils.mask_generator(story_ls)

            ending1_ls = [self.test_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending1_matrix = utils.padding(ending1_ls)
            ending1_mask = utils.mask_generator(ending1_ls)


            ending2_ls = [self.test_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending2_matrix = utils.padding(ending2_ls)
            ending2_mask = utils.mask_generator(ending2_ls)

            score1, score2 = self.prediction(story_matrix,
                                             ending1_matrix, 
                                             ending2_matrix, 
                                             story_mask,
                                             ending1_mask, 
                                             ending2_mask)

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - self.test_answer[i*minibatch_n:(i+1)*minibatch_n]
            correct += minibatch_n - (abs(correct_vec)).sum()

        for k in range(minibatch_n * max_batch_n, self.n_test):
            story = np.asarray(self.test_story[k], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.test_story[k])))

            ending1 = np.asarray(self.test_ending1[k], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.test_ending1[k])))

            ending2 = np.asarray(self.test_ending2[k], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.test_ending2[k])))

            score1, score2 = self.prediction(story,
                                             ending1, 
                                             ending2,
                                             story_mask, 
                                             ending1_mask, 
                                             ending2_mask)
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            if prediction == self.test_answer[k]:
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
        N_TRAIN_INS = len(self.train_ending)
        best_val_accuracy = 0
        best_test_accuracy = 0
        test_threshold = 10000/N_BATCH
        batch_count = 0.0
        start_batch = 0.0

        '''init test'''
        print "initial test..."
        val_result = self.val_set_test()
        print "val set accuracy: ", val_result*100, "%"
        if val_result > best_val_accuracy:
            best_val_accuracy = val_result

        test_accuracy = self.test_set_test()
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

            max_score_index = None
            for batch in range(max_batch):
                batch_count += 1

                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [self.train_story[index] for index in batch_index_list]
                train_ending = [self.train_ending[index] for index in batch_index_list]

                train_story_matrix = utils.padding(train_story)
                train_end_matrix = utils.padding(train_ending)
                # train_end2_matrix = utils.padding(end2)

                train_story_mask = utils.mask_generator(train_story)
                train_end_mask = utils.mask_generator(train_ending)
                # train_end2_mask = utils.mask_generator(end2)


                cost, score1, score2, max_score_index = self.train_func(train_story_matrix,
                                                                        train_end_matrix, 
                                                                        train_story_mask,
                                                                        train_end_mask)

                total_correct_count += np.count_nonzero((score1 - score2).clip(0.0))

                total_cost += cost
                if batch_count % test_threshold == 0:
                    print "accuracy on training set: ", total_correct_count/((batch+1) * N_BATCH)*100.0, "%"
                    print "example score sequence"
                    print np.concatenate((score1.reshape((-1,1)), score2.reshape((-1,1))), axis = 1)
                    print "test on val set..."
                    val_result = self.val_set_test()
                    print "accuracy is: ", val_result*100, "%"
                    if val_result > best_val_accuracy:
                        print "new best! test on test set..."
                        best_val_accuracy = val_result

                        test_accuracy = self.test_set_test()
                        print "test set accuracy: ", test_accuracy*100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
                    print "example negative ending:"
                    '''===================================================='''
                    '''randomly print out a story and it's higest score ending
                       competitive in a minibatch                          '''
                    '''===================================================='''
                    rand_index = np.random.randint(self.batchsize)
                    index = shuffled_index_list[N_BATCH * batch + rand_index]

                    story_string = " ".join([self.index2word_dict[self.train_story[index][k]] for k in range(len(self.train_story[index]))])
                    story_end = " ".join([self.index2word_dict[self.train_ending[index][k]] for k in range(len(self.train_ending[index]))])
                    highest_score_end = " ".join([self.index2word_dict[self.train_ending[max_score_index[rand_index]][k]] for k in range(len(self.train_ending[max_score_index[rand_index]]))])

                    print story_string 
                    print " #END# " + story_end
                    print ""
                    print "Highest Score Ending in this batch: " + highest_score_end 
                    print ""



            print "======================================="
            print "epoch summary:"
            print "average cost in this epoch: ", total_cost/max_batch
            print "average speed: ", N_TRAIN_INS/(time.time() - start_time), "instances/s "
            print "accuracy for this epoch: "+str(total_correct_count/(max_batch * N_BATCH) * 100.0)+"%"
 
            print "======================================="


def main(argv):
    wemb_size = None
    if len(argv) > 6:
        wemb_size = argv[6]
    model = Hierachi_RNN(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])


