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
                 wemb_trainable = 1,
                 learning_rate = 0.001,
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

        self.wemb_trainable = bool(int(wemb_trainable))

        self.learning_rate = float(learning_rate)

        self.wemb_size = 300
        if wemb_size == None:
            self.random_init_wemb = False
        else:
            self.random_init_wemb = True
            self.wemb_size = int(wemb_size)

        # self.train_story = None
        # self.train_ending = None

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
        for i in range(3):
            self.inputs_variables.append(T.matrix('story'+str(i)+'_input', dtype='int64'))
            self.inputs_masks.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            self.reshaped_inputs_variables.append(self.inputs_variables[i].dimshuffle(0,1,'x'))

        #initialize neural network units
        self.encoder = BLSTM_sequence.BlstmEncoder(LSTMLAYER_1_UNITS = self.rnn_units, wemb_trainable = self.wemb_trainable)
        self.encoder.build_model(self.wemb)

        #build encoding layer
        self.encoding_layer()

        batch_score1 = self.batch_cosine(self.train_encodinglayer_vecs[0], self.train_encodinglayer_vecs[1])
        batch_score2 = self.batch_cosine(self.train_encodinglayer_vecs[0], self.train_encodinglayer_vecs[2])

        # Construct symbolic cost function
        target1 = T.vector('gold_target1', dtype = theano.config.floatX)
        target2 = T.vector('gold_target2', dtype = theano.config.floatX)

        self.cost = lasagne.objectives.aggregate(target1.dimshuffle(0,'x') * batch_score1 + target2.dimshuffle(0,'x') * batch_score2, mode='mean')

        # Retrieve all parameters from the network
        all_params = self.encoder.all_params

        all_updates = lasagne.updates.adam(self.cost, all_params, learning_rate=self.learning_rate)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        self.train_func = theano.function(self.inputs_variables + self.inputs_masks + [target1, target2], 
                                        [self.cost, batch_score1, batch_score2], updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function(self.inputs_variables + self.inputs_masks, [batch_score1, batch_score2])
        # pydotprint(self.train_func, './computational_graph.png')

    def load_data(self):
        '''======Train Set====='''

        val_set = pickle.load(open(self.val_set_path))

        self.val_story = utils.combine_sents(val_set[0])
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.val_len = len(self.val_answer)
        self.val_train_n = int(self.val_split_ratio * self.val_len)
        self.val_test_n = self.val_len - self.val_train_n

        shuffled_indices_ls = utils.shuffle_index(len(self.val_answer))

        self.val_train_ls = shuffled_indices_ls[:self.val_train_n]
        self.val_test_ls = shuffled_indices_ls[self.val_train_n:]

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
        # self.peeked_ends_ls = np.random.randint(self.n_train, size=(5,))
        # self.ends_pool_ls = np.random.choice(range(self.n_train), 2000, replace = False)
        # self.index2word_dict = pickle.load(open(self.index2word_dict_path))
       

    def val_set_test(self):

        correct = 0.

        for k in self.val_test_ls:
            story = np.asarray(self.val_story[k], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.val_story[k])))

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
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis = 1)

            if prediction == self.val_answer[k]:
                correct += 1. 


        return correct/self.val_test_n

    def test_set_test(self):
        #load test set data
        correct = 0.

        for k in range(self.n_test):
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
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis = 1)

            if prediction == self.test_answer[k]:
                correct += 1. 


        return correct/self.n_test


    # def saving_model(self, val_or_test, accuracy):
    #     reason_params_value = lasagne.layers.get_all_param_values(self.reason_layer.output)
    #     classif_params_value = lasagne.layers.get_all_param_values(self.classify_layer.output)

    #     if val_or_test == 'val':
    #         pickle.dump((reason_params_value, classif_params_value, accuracy), 
    #                     open(self.best_val_model_save_path, 'wb'))
    #     else:
    #         pickle.dump((reason_params_value, classif_params_value, accuracy), 
    #                     open(self.best_test_model_save_path, 'wb'))            

    # def reload_model(self, val_or_test):
    #     if val_or_test == 'val': 

    #         reason_params, classif_params, accuracy = pickle.load(open(self.best_val_model_save_path))
    #         lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
    #         lasagne.layers.set_all_param_values(self.classify_layer.output, classif_params)

    #         print "This model has ", accuracy * 100, "%  accuracy on valid set" 
    #     else:
    #         reason_params, classif_params, accuracy = pickle.load(open(self.best_test_model_save_path))
    #         lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
    #         lasagne.layers.set_all_param_values(self.classify_layer.output, classif_params_value)
    #         print "This model has ", accuracy * 100, "%  accuracy on test set" 

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
                train_story = [self.val_story[index] for index in batch_index_list]
                train_ending1 = [self.val_ending1[index] for index in batch_index_list]
                train_ending2 = [self.val_ending2[index] for index in batch_index_list]

                train_story_matrix = utils.padding(train_story)
                train_end1_matrix = utils.padding(train_ending1)
                train_end2_matrix = utils.padding(train_ending2)

                # train_end2_matrix = utils.padding(end2)

                train_story_mask = utils.mask_generator(train_story)
                train_end1_mask = utils.mask_generator(train_ending1)
                train_end2_mask = utils.mask_generator(train_ending2)

                answer = np.asarray([self.val_answer[index] for index in batch_index_list])

                target1 = 2 * answer - 1
                target2 = -2 * answer + 1
                # answer_vec = np.concatenate(((1 - answer).reshape(-1,1), answer.reshape(-1,1)),axis = 1)

                cost, prediction1, prediction2 = self.train_func(train_story_matrix, train_end1_matrix, train_end2_matrix, 
                                                                 train_story_mask, train_end1_mask, train_end2_mask, 
                                                                 target1, target2)



                prediction = np.argmax(np.concatenate((prediction1, prediction2), axis = 1), axis = 1)

                total_err_count += (abs(prediction - answer)).sum()

                total_cost += cost

            print "======================================="
            print "epoch summary:"
            print "total cost in this epoch: ", total_cost / (max_batch)
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



def main(argv):
    wemb_size = None
    if len(argv) > 5:
        wemb_size = argv[5]
    model = Hierachi_RNN(argv[0], argv[1], argv[2], argv[3], argv[4], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])


