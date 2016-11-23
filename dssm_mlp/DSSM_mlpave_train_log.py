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
                 dropout_rate, 
                 val_split_ratio,
                 batchsize, 
                 optimizer, 
                 scorefunc, 
                 wemb_size = None):

        # Initialize Theano Symbolic variable attributes
        self.story_input_variable = None
        self.story_mask = None
        self.story_nsent = 4

        self.ending_input_variable = None
        self.ending_mask = None

        self.neg_ending1_input_variable = None
        self.neg_ending1_mask = None

        self.encoder = None
        self.val_split_ratio = float(val_split_ratio)

        unit_ls = mlp_units.split('x')
        self.mlp_units = [int(unit) for unit in unit_ls]
        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.dropout_rate = float(dropout_rate)
        self.batchsize = int(batchsize)
        self.optimizer = optimizer
        self.scorefunc = scorefunc
        self.bilinear_weight = None
        if self.scorefunc == "bilinear":
            self.bilinear_weight = theano.shared(np.random.normal(0, 0.01, (self.mlp_units[-1], self.mlp_units[-1])))

        self.wemb_size = 300
        if wemb_size == None:
            self.random_init_wemb = False
        else:
            self.random_init_wemb = True
            self.wemb_size = int(wemb_size)

        self.doc_encode = None 
        self.pos_end_encode = None
        self.neg_end1_encode = None
        self.neg_end2_encode = None
        self.neg_end3_encode = None

        self.cost = None
        self.val_test_cos = None

        self.train_func = None
        self.compute_cost = None

        # Initialize data loading attributes
        self.wemb = None
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.train_set_path = '../../data/pickles/train_index_corpus.pkl'
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 

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

    def batch_cosine(self, doc_batch_proj, query_batch_proj):
        dot_prod = T.batched_dot(doc_batch_proj, query_batch_proj)

        doc_square = doc_batch_proj ** 2
        query_square = query_batch_proj ** 2

        doc_norm = (T.sqrt(doc_square.sum(axis = 1))).sum()
        query_norm = T.sqrt(query_square.sum(axis = 1)).sum()

        batch_cosine_vec = dot_prod/(doc_norm * query_norm)
        return batch_cosine_vec


    def model_constructor(self):
        self.story_input_variable = []
        self.story_mask = []
        self.story_reshape_input = []

        for i in range(self.story_nsent):
            self.story_input_variable.append(T.matrix('story_'+str(i)+'_input', dtype='int64'))
            self.story_mask.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            story_batch_size, story_seqlen = self.story_input_variable[i].shape
            self.story_reshape_input.append(self.story_input_variable[i].reshape([story_batch_size, story_seqlen, 1]))


        self.ending1_input_variable = T.matrix('ending1_input', dtype = 'int64')
        self.ending1_mask = T.matrix('ending1_mask', dtype = theano.config.floatX)

        self.ending2_input_variable = T.matrix('ending2_input', dtype = 'int64')
        self.ending2_mask = T.matrix('ending2_mask', dtype = theano.config.floatX)

        ending_batch_size, ending_seqlen = self.ending1_input_variable.shape
        ending1_reshape_input = self.ending1_input_variable.reshape([ending_batch_size, ending_seqlen, 1])

        ending_batch_size, ending_seqlen = self.ending2_input_variable.shape
        ending2_reshape_input = self.ending2_input_variable.reshape([ending_batch_size, ending_seqlen, 1])

        l_in = lasagne.layers.InputLayer(shape=(None, None, 1))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.wemb.get_value().shape[0], 
                                                    output_size=self.wemb.get_value().shape[1], 
                                                    W=self.wemb)

        l_reshape = lasagne.layers.ReshapeLayer(l_emb, ([0], [1], -1))

        embeding_params = lasagne.layers.get_all_params(l_reshape)

        story_ave_matrices = []
        for i in range(self.story_nsent):
            story_tensor = lasagne.layers.get_output(l_reshape, {l_in: self.story_reshape_input[i]})
            ave_matrix = (T.sum(story_tensor, axis = 1))/((T.sum(self.story_mask[i], axis = 1)).reshape([self.story_mask[i].shape[0],1]))
            story_ave_matrices.append(ave_matrix.reshape([1, ave_matrix.shape[0], ave_matrix.shape[1]]))
           

        end1_tensor = lasagne.layers.get_output(l_reshape, {l_in: ending1_reshape_input})
        end2_tensor = lasagne.layers.get_output(l_reshape, {l_in: ending2_reshape_input})

        story_ave_matrix = T.sum(T.concatenate(story_ave_matrices, axis = 0), axis = 0) / 4.0
        end1_ave_matrix = (T.sum(end1_tensor, axis = 1))/((T.sum(self.ending1_mask, axis = 1)).reshape([self.ending1_mask.shape[0],1]))
        end2_ave_matrix = (T.sum(end2_tensor, axis = 1))/((T.sum(self.ending2_mask, axis = 1)).reshape([self.ending2_mask.shape[0],1]))

        self.encoder = MLP_Encoder.MLPEncoder(self.mlp_units)
        self.encoder.build_model(self.wemb_size)

        story_vec = lasagne.layers.get_output(self.encoder.output, {self.encoder.l_in: story_ave_matrix})
        end1_vec = lasagne.layers.get_output(self.encoder.output, {self.encoder.l_in: end1_ave_matrix})
        end2_vec = lasagne.layers.get_output(self.encoder.output, {self.encoder.l_in: end2_ave_matrix})

        if self.scorefunc == 'cos':
            self.end1_score_vec = self.batch_cosine(story_vec, end1_vec).reshape([-1, 1])
            self.end2_score_vec = self.batch_cosine(story_vec, end2_vec).reshape([-1, 1])

            self.all_params = self.encoder.all_params

        else:
            cache_result = T.dot(story_vec, self.bilinear_weight)
            score_matrix = T.dot(cache_result, T.transpose(end1_vec))
            self.end1_score_vec = score_matrix.diagonal().reshape([-1, 1])
            self.end2_score_vec = T.dot(cache_result, T.transpose(end2_vec)).diagonal().reshape([-1, 1])

            self.all_params = self.encoder.all_params +[self.bilinear_weight]

        loss = - self.end1_score_vec + self.end2_score_vec

        self.cost = lasagne.objectives.aggregate(loss, mode='mean')
        # Retrieve all parameters from the network

        
        self.all_updates = None 
        if self.optimizer == 'SGD':
            self.all_updates = lasagne.updates.sgd(self.cost, self.all_params, 0.01)
        else:
            self.all_updates = lasagne.updates.adam(self.cost, self.all_params)

        self.train_func = theano.function(self.story_input_variable+
                                        [self.ending1_input_variable, self.ending2_input_variable]+
                                        self.story_mask+[self.ending1_mask, self.ending2_mask],
                                        [self.cost, self.end1_score_vec, self.end2_score_vec], updates = self.all_updates)


        self.prediction = theano.function(self.story_input_variable+
                                        [self.ending1_input_variable, self.ending2_input_variable]+
                                        self.story_mask+[self.ending1_mask, self.ending2_mask],
                                        [self.end1_score_vec, self.end2_score_vec])

    def load_data(self):
        '''======Train Set====='''


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
            predict_answer = 0
            if score2 > score1:
                predict_answer = 1

            if predict_answer == self.val_answer[i]:
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
            predict_answer = 0
            if score2 > score1:
                predict_answer = 1

            if predict_answer == self.val_answer[i]:
                correct += 1.


        return correct/self.n_test


    # def saving_model(self, val_or_test, accuracy):
    #     doc_all_params_value = lasagne.layers.get_all_param_values(self.doc_encoder.output)
    #     query_all_params_value = lasagne.layers.get_all_param_values(self.query_encoder.output)
    #     if val_or_test == 'val':
    #         pickle.dump((doc_all_params_value, query_all_params_value, accuracy), open(self.best_val_model_save_path, 'wb'))
    #     else:
    #         pickle.dump((doc_all_params_value, query_all_params_value, accuracy), open(self.best_test_model_save_path, 'wb'))

    # def reload_model(self, val_or_test):
    #     if val_or_test == 'val':    
    #         doc_encoder_params, query_encoder_params, accuracy = pickle.load(open(self.best_val_model_save_path))
    #         lasagne.layers.set_all_param_values(self.doc_encoder.output, doc_encoder_params)
    #         lasagne.layers.set_all_param_values(self.query_encoder.output, query_encoder_params)
    #         print "This model has ", accuracy * 100, "%  accuracy on valid set" 
    #     else:
    #         doc_encoder_params, query_encoder_params, accuracy = pickle.load(open(self.best_test_model_save_path))
    #         lasagne.layers.set_all_param_values(self.doc_encoder.output, doc_encoder_params)
    #         lasagne.layers.set_all_param_values(self.query_encoder.output, query_encoder_params) 
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
                train_story = [[self.val_story[index][i] for index in batch_index_list] for i in range(self.story_nsent)]
                train_ending = [self.val_ending1[index] for index in batch_index_list]
                neg_end1 = [self.val_ending2[index] for index in batch_index_list]
                # neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # while np.any((np.asarray(batch_index_list) - neg_end_index_list) == 0):
                #     neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                # neg_end1 = [self.train_ending[index] for index in neg_end_index_list]
                # answer = np.random.randint(2, size = N_BATCH)
                # target1 = 1 - answer
                # target2 = 1 - target1
                answer = np.asarray([self.val_answer[index] for index in batch_index_list])


                # answer_vec = np.concatenate(((1 - answer).reshape(-1,1), answer.reshape(-1,1)),axis = 1)
                end1 = []
                end2 = []

                for i in range(N_BATCH):
                    if answer[i] == 0:
                        end1.append(train_ending[i])
                        end2.append(neg_end1[i])
                    else:
                        end1.append(neg_end1[i])
                        end2.append(train_ending[i])



                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end1_matrix = utils.padding(end1)
                train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end1_mask = utils.mask_generator(end1)
                train_end2_mask = utils.mask_generator(end2)
                

                cost, prediction1, prediction2 = self.train_func(train_story_matrices[0], train_story_matrices[1], train_story_matrices[2],
                                                               train_story_matrices[3], train_end1_matrix, train_end2_matrix,
                                                               train_story_mask[0], train_story_mask[1], train_story_mask[2],
                                                               train_story_mask[3], train_end1_mask, train_end2_mask)



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




def main(argv):
    wemb_size = None
    if len(argv) > 6:
        wemb_size = argv[6]
    model = DSSM_MLP_Model(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], wemb_size)


    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])
