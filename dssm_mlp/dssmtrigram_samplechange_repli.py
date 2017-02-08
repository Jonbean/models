import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import MLP_Encoder
import sys

class DSSM_MLP_Model(object):
    def __init__(self, 
                 mlp_units, 
                 dropout_rate = 0.0, 
                 batchsize = 1024, 
                 negEnd_N = 10,
                 optimizer = 'SGD', 
                 wemb_size = None):

        # Initialize Theano Symbolic variable attributes


        self.encoder = None

        self.mlp_units = map(int, mlp_units.split('x'))
        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.dropout_rate = float(dropout_rate)
        self.batchsize = int(batchsize)
        self.optimizer = optimizer
        self.bilinear_weight = None


        self.cost = None

        self.train_func = None

        # Initialize data loading attributes
        # self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.train_set_path = '../../data/pickles/train_trigram.pkl'
        self.val_set_path = '../../data/pickles/val_trigram.pkl'
        self.test_set_path = '../../data/pickles/test_trigram.pkl' 

        self.train_story = None
        self.train_ending = None

        self.val_story = None
        self.val_ending1 = None 
        self.val_ending2 = None
        self.val_answer = None

        self.test_story = None 
        self.test_ending1 = None
        self.test_ending2 = None
        self.test_answer = None
        self.n_test = None
        self.negEnd_N = int(negEnd_N)
        self.trigram_generator = utils.Ngram_generator('letter_trigram')
        self.trigram_input_size = self.trigram_generator.entries_num

    def batch_cosine(self, batch_vectors1, batch_vectors2):
        dot_prod = T.batched_dot(batch_vectors1, batch_vectors2)

        batch1_norm = T.sqrt((T.sqr(batch_vectors1)).sum(axis = 1))
        batch2_norm = T.sqrt((T.sqr(batch_vectors2)).sum(axis = 1))

        batch_cosine_vec = dot_prod/(batch1_norm * batch2_norm)
        return batch_cosine_vec.reshape((-1,1))

    def encoding_layer(self):
        self.sent_reps = []
        for i in range(3):
            sent_rep = lasagne.layers.get_output(self.l_out,
                                                {self.l_in:self.story_input_variable[i]},
                                                deterministic = True)
            self.sent_reps.append(sent_rep)

    def model_constructor(self):
        self.story_input_variable = []
        self.story_mask = []
        self.story_reshape_input = []


        # story_input_variable[0] ==> story plot 
        # story_input_variable[1] ==> story end1
        # story_input_variable[2] ==> story end2 (for val/test)
        for i in range(3):
            self.story_input_variable.append(T.matrix('story_'+str(i)+'_input', dtype='int64'))


        self.l_in = lasagne.layers.InputLayer(shape=(None, self.trigram_input_size))
        self.l_hid1 = lasagne.layers.DenseLayer(self.l_in, num_units=self.mlp_units[0],
                                                nonlinearity=lasagne.nonlinearities.tanh,
                                                W=lasagne.init.GlorotUniform())

        self.l_out = lasagne.layers.DenseLayer(self.l_hid1, num_units=self.mlp_units[1],
                                               W = lasagne.init.GlorotUniform())

        
        self.encoding_layer()

        self.batch_scores = []
        self.score1 = self.batch_cosine(self.sent_reps[0], self.sent_reps[1])
        self.score2 = self.batch_cosine(self.sent_reps[0], self.sent_reps[2])

        self.batch_scores.append(self.score1)

        for i in range(self.negEnd_N):
            self.batch_scores.append(self.batch_cosine(self.sent_reps[0], T.roll(self.sent_reps[1], shift=(i+1), axis = 0)))


        score_concate = T.concatenate(self.batch_scores, axis = 1)
        exp_sum = T.sum(T.exp(score_concate), axis = 1)
        loss = - self.batch_scores[0] + T.log(exp_sum)

        self.cost = lasagne.objectives.aggregate(loss, mode='mean')
        # Retrieve all parameters from the network
        self.all_params = lasagne.layers.get_all_params(self.l_out)

        
        self.all_updates = None 
        if self.optimizer == 'SGD':
            self.all_updates = lasagne.updates.sgd(self.cost, self.all_params, 0.5)
        else:
            self.all_updates = lasagne.updates.adam(self.cost, self.all_params)

        self.train_func = theano.function(self.story_input_variable[:-1],
                                         [self.cost, score_concate], updates = self.all_updates)


        self.prediction = theano.function(self.story_input_variable,
                                          [self.score1, self.score2])

    def load_data(self):
        '''======Train Set====='''

        '''======Train Set====='''
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = utils.combine_sents_string(train_set[0])
        self.train_ending = train_set[1]
        self.n_train = len(self.train_ending)
        
        '''=====Val Set====='''
        val_set = pickle.load(open(self.val_set_path))

        self.val_story = utils.combine_sents_string(val_set[0])
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.n_val = len(self.val_answer)

        '''=====Test Set====='''
        test_set = pickle.load(open(self.test_set_path))
        self.test_story = utils.combine_sents_string(test_set[0])
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_answer)

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
            
            story = [eva_story[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            end1 = [eva_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            end2 = [eva_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]

            story_matrix = self.trigram_generator.letter_trigram_generator(story)
            end1_matrix = self.trigram_generator.letter_trigram_generator(end1)
            end2_matrix = self.trigram_generator.letter_trigram_generator(end2)

            score1, score2 = self.prediction(story_matrix, end1_matrix, end2_matrix)

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - eva_answer[i*minibatch_n:(i+1)*minibatch_n]
            correct += minibatch_n - (abs(correct_vec)).sum()



        story = [eva_story[index] for index in range(-residue, 0)]
        end1 = [eva_ending1[index] for index in range(-residue, 0)]
        end2 = [eva_ending2[index] for index in range(-residue, 0)]


        story_matrix = self.trigram_generator.letter_trigram_generator(story)
        end1_matrix = self.trigram_generator.letter_trigram_generator(end1)
        end2_matrix = self.trigram_generator.letter_trigram_generator(end2)

        score1, score2 = self.prediction(story_matrix, end1_matrix, end2_matrix)

        # Answer denotes the index of the anwer
        prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
        correct_vec = prediction - eva_answer[-residue:]
        correct += minibatch_n - (abs(correct_vec)).sum()

        return correct/n_eva

    def begin_train(self):
        N_EPOCHS = 100
        N_BATCH = self.batchsize
        N_TRAIN_INS = self.n_train
        test_threshold = 10000/N_BATCH

        best_val_accuracy = 0
        best_test_accuracy = 0

        """init test"""
        print "initial test"
        val_result = self.eva_func('val')
        print "valid set accuracy: ", val_result*100, "%"
        if val_result > best_val_accuracy:
            print "new best! test on test set..."
            best_val_accuracy = val_result

        test_accuracy = self.eva_func('test')
        print "test set accuracy: ", test_accuracy * 100, "%"
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
        """end of init test"""

        
        for epoch in range(N_EPOCHS):
            print "epoch ", epoch,":"

            shuffled_index_list = range(N_TRAIN_INS)
            np.random.shuffle(shuffled_index_list)

            max_batch = N_TRAIN_INS/N_BATCH

            start_time = time.time()

            total_cost = 0.0
            total_err_count = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]

                #neg_end_index_matrix.shape = [N_BATCH, 4]
                # neg_end_index_matrix = []
                # for i in range(N_BATCH):
                #     neg_end_index_vec = np.random.randint(0, N_TRAIN_INS, size = (self.negEnd_N,))
                #     while np.any(neg_end_index_vec - batch_index_list[i]) == 0:
                #         neg_end_index_vec = np.random.randint(0, N_TRAIN_INS, size = (self.negEnd_N,))
                #     neg_end_index_matrix.append(neg_end_index_vec)
                # neg_end_index_list = np.asarray(neg_end_index_matrix)

                story = [self.train_story[index] for index in batch_index_list]
                end = [self.train_ending[index] for index in batch_index_list]

                # neg_end_matrix_ls = []
                # for i in range(self.negEnd_N):
                #     neg_end = [self.train_ending[index] for index in neg_end_index_list[:, i]]
                #     neg_end_matrix_ls.append(self.trigram_generator.letter_trigram_generator(neg_end))

                story_matrix = self.trigram_generator.letter_trigram_generator(story)
                end_matrix = self.trigram_generator.letter_trigram_generator(end)



                cost, score_matrix = self.train_func(story_matrix, end_matrix)



                prediction = np.argmax(score_matrix, axis = 1)

                total_err_count += np.count_nonzero(prediction)

                total_cost += cost
                if batch % test_threshold == 0 and batch != 0:
                    train_acc = (1 - total_err_count / ((batch+1) * N_BATCH))*100.0

                    print "accuracy on training set: ", train_acc, "%"
                    print "example score sequence"
                    print score_matrix
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
                    print "cost: ", cost
            print "======================================="
            print "epoch summary:"
            print "total cost in this epoch: ", total_cost
            print "accuracy on training set: ", (1.0-(total_err_count / N_TRAIN_INS)) * 100, "%"
            val_result = self.eva_func('val')
            print "accuracy is: ", val_result*100, "%"
            if val_result > best_val_accuracy:
                print "new best! test on test set..."
                best_val_accuracy = val_result

            test_accuracy = self.eva_func('test')
            print "test set accuracy: ", test_accuracy * 100, "%"
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
            print "======================================="





   
if __name__ == '__main__':
    model = DSSM_MLP_Model(*sys.argv[1:])
    
    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    