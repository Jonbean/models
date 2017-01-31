import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import LSTM_sequence
import sys
# from theano.printing import pydotprint



class Hierachi_RNN(object):
    def __init__(self, 
                 batchsize, 
                 wemb_trainable = 1,
                 learning_rate = 0.001,
                 wemb_size = 300):
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
        self.words_num = 35678

        self.rnn_units = [300, self.words_num]
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

        self.train_encodinglayer_vecs = None

        # if score_func_nonlin == 'default':
        #     self.score_func_nonlin = lasagne.nonlinearities.tanh
        # else:
        #     self.score_func_nonlin = None
        self.wemb_trainable = bool(int(wemb_trainable))
        self.learning_rate = float(learning_rate)

    def encoding_layer(self):
        
        sent_seq = lasagne.layers.get_output(self.encoder.output,
                                            {self.encoder.l_in:self.reshaped_inputs_variables, 
                                             self.encoder.l_mask:self.inputs_masks},
                                             deterministic = True)

        self.train_encodinglayer_vecs = (sent_seq * self.inputs_masks.dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks.sum(axis = 1, keepdims = True)       
        self.word_predict_matrix = self.train_encodinglayer_vecs.reshape((-1, self.rnn_units[1]))
        self.prediction_softmax = lasagne.nonlinearities.softmax(self.word_predict_matrix)


    def model_constructor(self, wemb_size = None):
        self.inputs_variables = T.matrix('story_input', dtype='int64')
        self.inputs_masks = T.matrix('story_mask', dtype=theano.config.floatX)

        self.reshaped_inputs_variables = self.inputs_variables[:,:-1].dimshuffle(0,1,'x')
        batch_size, max_len = self.inputs_masks.shape

        #initialize neural network units
        self.encoder = LSTM_sequence.LstmEncoder(batch_size, max_len, LSTMLAYER_UNITS = self.rnn_units, wemb_trainable = bool(self.wemb_trainable))
        self.encoder.build_model(self.wemb)

        #build encoding layer
        self.encoding_layer()

        cost = lasagne.objectives.categorical_crossentropy(self.prediction_softmax, T.flatten(self.inputs_variables[:,1:]))

        self.cost = lasagne.objectives.aggregate(cost, mode='mean') 


        # Retrieve all parameters from the network
        all_params = self.encoder.all_params 

        all_updates = lasagne.updates.adam(self.cost, all_params, learning_rate=self.learning_rate)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        self.train_func = theano.function([self.inputs_variables, self.inputs_masks], 
                                          [self.cost], updates = all_updates)

        self.prediction = theano.function([self.inputs_variables, self.inputs_masks],
                                          [self.cost])
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
            wemb = np.random.rand(self.words_num - 1,self.wemb_size)
            wemb = np.concatenate((np.zeros((1, self.wemb_size)), wemb), axis = 0)
            self.wemb = theano.shared(wemb).astype(theano.config.floatX)
        else:
            with open(self.wemb_matrix_path,'r') as f:
                self.wemb = theano.shared(pickle.load(f)).astype(theano.config.floatX)

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
            
            story1 = []
            story2 = []
            for index in range(i*minibatch_n, (i+1)*minibatch_n):

                story1.append(eva_story[index] + eva_ending1[index])
                story2.append(eva_story[index] + eva_ending2[index])


            story1_matrix = utils.padding(story1)
            story2_matrix = utils.padding(story2)

            story1_mask = utils.mask_generator(story1)
            story2_mask = utils.mask_generator(story2)


            perplexity1 = self.prediction(story1_matrix,
                                          story1_mask[:,1:])
            perplexity2 = self.prediction(story2_matrix,
                                          story2_mask[:,1:])

            # Answer denotes the index of the anwer
            prediction = np.argmin(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - eva_answer[i*minibatch_n:(i+1)*minibatch_n]
            correct += minibatch_n - (abs(correct_vec)).sum()


        story1 = []
        story2 = []
        for index in range(-residue, 0):

            story1.append(eva_story[index] + eva_ending1[index])
            story2.append(eva_story[index] + eva_ending2[index])

        story1_matrix = utils.padding(story1)
        story2_matrix = utils.padding(story2)

        story1_mask = utils.mask_generator(story1)
        story2_mask = utils.mask_generator(story2)

        perplexity1 = self.prediction(story1_matrix,
                                      story1_mask[:,1:])
        perplexity2 = self.prediction(story2_matrix,
                                      story2_mask[:,1:])



        prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
        correct_vec = prediction - eva_answer[-residue:]
        correct += minibatch_n - (abs(correct_vec)).sum()
        return correct/n_eva



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

            max_score_index = None
            for batch in range(max_batch):
                batch_count += 1

                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                story1 = []
                for index in batch_index_list:

                    story1.append(self.train_story[index] + self.train_ending[index])

                story_matrix = utils.padding(story1)

                story_mask = utils.mask_generator(story1)


                cost = self.train_func(story_matrix,
                                       story_mask[:,1:])


                total_cost += cost
                if batch_count % test_threshold == 0:
                    print "example score sequence"
                    print "perplexity: ", total_cost
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



            print "======================================="
            print "epoch summary:"
            print "average perplexity in this epoch: ", total_cost/max_batch
            print "average speed: ", N_TRAIN_INS/(time.time() - start_time), "instances/s "
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

