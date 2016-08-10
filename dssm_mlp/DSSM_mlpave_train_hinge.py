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
    def __init__(self, mlp_units, dropout_rate, batchsize, optimizer, 
                scorefunc, wemb_size = None):
        # Initialize Theano Symbolic variable attributes
        self.story_input_variable = None
        self.story_mask = None
        self.story_nsent = 4

        self.ending_input_variable = None
        self.ending_mask = None

        self.neg_ending1_input_variable = None
        self.neg_ending1_mask = None

        self.encoder = None

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

            self.scores = []
            for i in range(self.batchsize - 1):
                self.scores.append(self.batch_cosine(story_vec, T.roll(end1_vec, shift=(i+1), axis = 0)).reshape([-1, 1]))
            score_matrix = T.concatenate(self.scores, axis = 1)
            self.max_score_vec = T.max(score_matrix, axis = 1).reshape((-1, 1))

            self.all_params = self.encoder.all_params+embeding_params

        else:
            cache_result = T.dot(story_vec, bilinear_matrix)
            score_matrix = T.dot(cache_result, T.transpose(end1_vec))
            self.end1_score_vec = score_matrix.diagonal().reshape([-1, 1])
            self.end2_score_vec = T.dot(cache_result, T.transpose(end2_vec)).diagonal().reshape([-1, 1])

            
            
            sort_score_indices = T.transpose(T.argsort(score_matrix, axis = 0))
            best_score_indices = sort_score_indices[-1].astype('int64')
            second_best_score_indices = sort_score_indices[-2].astype('int64')
            check_indices = T.arange(self.batchsize).astype('int64')
            best_mask = T.eq(best_score_indices, check_indices).astype('int64')
            second_best_mask = T.ones_like(best_mask) - best_mask
            hinge_indices = best_score_indices * second_best_mask + second_best_score_indices * best_mask
            
            hinge_score_ls = []
            for i in range(self.batchsize):
                hinge_score_ls.append(score_matrix[i][hinge_indices[i]].reshape((1,1)))
            self.max_score_vec = T.concatenate(hinge_score_ls, axis = 1).reshape([-1, 1])
            self.all_params = self.encoder.all_params + embeding_params+[self.bilinear_weight]


        loss = T.max(T.concatenate([T.zeros((self.batchsize, 1)), 2.0 - self.end1_score_vec + self.max_score_vec], axis = 1), axis = 1)
        self.cost = lasagne.objectives.aggregate(loss, mode='sum')
        # Retrieve all parameters from the network

        
        
        self.all_updates = None 
        if self.optimizer == 'SGD':
            self.all_updates = lasagne.updates.sgd(self.cost, self.all_params)
        else:
            self.all_updates = lasagne.updates.adam(self.cost, self.all_params)

        self.train_func = theano.function(self.story_input_variable+self.story_mask+
                                     [self.ending1_input_variable, self.ending1_mask],
                                     [self.cost, self.end1_score_vec, self.max_score_vec], updates = self.all_updates)

        # Compute adam updates for training

        self.compute_cost = theano.function(self.story_input_variable+self.story_mask+
                                     [self.ending1_input_variable, self.ending1_mask,
                                     self.ending2_input_variable, self.ending2_mask],
                                     [self.end1_score_vec, self.end2_score_vec])

    def load_data(self):
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = train_set[0]
        self.train_ending1 = train_set[1]
        self.n_train = len(self.train_ending1)

        val_set = pickle.load(open(self.val_set_path))

        self.val_story = val_set[0]
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



    def val_set_test(self):
        correct = 0.

        minibatch_n = 50
        max_batch_n = self.n_val / minibatch_n
        print type(max_batch_n), max_batch_n
        for i in range(max_batch_n):

            story_ls = [[self.val_story[index][j] for index in range(i*minibatch_n, (i+1)*minibatch_n)] for j in range(self.story_nsent)]
            story_matrix = [utils.padding(batch_sent) for batch_sent in story_ls]
            story_mask = [utils.mask_generator(batch_sent) for batch_sent in story_ls]

            ending1_ls = [self.val_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending1_matrix = utils.padding(ending1_ls)
            ending1_mask = utils.mask_generator(ending1_ls)


            ending2_ls = [self.val_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending2_matrix = utils.padding(ending2_ls)
            ending2_mask = utils.mask_generator(ending2_ls)
            for i in range(4):
                print story_matrix[i].shape, story_mask[i].shape
            print ending1_matrix.shape, ending1_mask.shape, ending2_matrix.shape, ending2_mask.shape

            score1, score2 = self.compute_cost(story_matrix[0], story_matrix[1], story_matrix[2], story_matrix[3], 
                                                story_mask[0], story_mask[1], story_mask[2], story_mask[3],
                                                ending1_matrix, ending1_mask, ending2_matrix, ending2_mask)

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - self.val_answer[i*minibatch_n:(i+1)*minibatch_n]
            print correct_vec
            correct += minibatch_n - (abs(correct_vec)).sum()
            print correct

        for k in range(minibatch_n * max_batch_n, self.n_val):
            story = [np.asarray(sent, dtype='int64').reshape((1,-1)) for sent in self.val_story[k]]
            story_mask = [np.ones((1,len(self.val_story[k][j]))) for j in range(4)]

            print story[0].shape
            print story_mask[0].shape

            ending1 = np.asarray(self.val_ending1[k], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.val_ending1[k])))

            ending2 = np.asarray(self.val_ending2[k], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.val_ending2[k])))

            score1, score2 = self.compute_cost(story[0], story[1], story[2], story[3], 
                                    story_mask[0], story_mask[1], story_mask[2],story_mask[3],
                                    ending1,  ending1_mask, ending2, ending2_mask)
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            if prediction == self.val_answer[k]:
                correct += 1.    
        print self.n_val
        return correct/self.n_val

    def test_set_test(self):
        #load test set data

        correct = 0.

        minibatch_n = 50
        max_batch_n = self.n_test / minibatch_n
        for i in range(max_batch_n):

            story_ls = [[self.test_story[index][j] for index in range(i*minibatch_n, (i+1)*minibatch_n)] for j in range(self.story_nsent)]
            story_matrix = [utils.padding(batch_sent) for batch_sent in story_ls]
            story_mask = [utils.mask_generator(batch_sent) for batch_sent in story_ls]

            ending1_ls = [self.test_ending1[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending1_matrix = utils.padding(ending1_ls)
            ending1_mask = utils.mask_generator(ending1_ls)


            ending2_ls = [self.test_ending2[index] for index in range(i*minibatch_n, (i+1)*minibatch_n)]
            ending2_matrix = utils.padding(ending2_ls)
            ending2_mask = utils.mask_generator(ending2_ls)

            score1, score2 = self.prediction(story_matrix[0], story_matrix[1], story_matrix[2], story_matrix[3], 
                                    story_mask[0], story_mask[1], story_mask[2],story_mask[3],ending1_matrix,  
                                    ending1_mask, ending2_matrix, ending2_mask)

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1), axis=1)
            correct_vec = prediction - self.test_answer[i*minibatch_n:(i+1)*minibatch_n]
            correct += minibatch_n - (abs(correct_vec)).sum()

        for k in range(minibatch_n * max_batch_n, self.n_test):
            story = [np.asarray(sent, dtype='int64').reshape((1,-1)) for sent in self.test_story[k]]
            story_mask = [np.ones((1,len(self.test_story[k][j]))) for j in range(4)]

            ending1 = np.asarray(self.test_ending1[k], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.test_ending1[k])))

            ending2 = np.asarray(self.test_ending2[k], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.test_ending2[k])))

            score1, score2 = self.prediction(story[0], story[1], story[2], story[3], 
                                     story_mask[0], story_mask[1], story_mask[2],story_mask[3],
                                     ending1, ending1_mask, ending2, ending2_mask)
            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

            if prediction == self.test_answer[k]:
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
        N_TRAIN_INS = self.n_train
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
            shuffled_index_list = utils.shuffle_index(N_TRAIN_INS)

            max_batch = N_TRAIN_INS/N_BATCH

            start_time = time.time()

            total_cost = 0.0
            total_err_count = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [self.train_story[index] for index in batch_index_list]
                train_ending1 = [self.train_ending1[index] for index in batch_index_list]
                
            
                train_story_matrix = [utils.padding(train_story[i]) for i in range(self.story_nsent)]
                train_ending1_matrix = utils.padding(train_ending1)

                train_story_mask = [utils.mask_generator(train_story[i]) for i in range(self.story_nsent)]
                train_ending1_mask = utils.mask_generator(train_ending1)

                cost, cos1, cos2 = self.train_func(train_story_matrix[0],train_story_matrix[1], train_story_matrix[2],
                                                    train_story_matrix[3] , train_story_mask[0], train_story_mask[1],
                                                    train_story_mask[2], train_story_mask[3],train_ending1_matrix, 
                                                    train_ending1_mask)

                total_cost += cost
                predict_vec = np.argmax(np.concatenate([cos1, cos2], axis = 1), axis = 1).reshape((-1, 1))
                total_err_count += (abs(predict_vec)).sum()

            print "======================================="
            print "epoch summary:"
            print "total cost in this epoch: ", total_cost
            print "accuracy on training set: ", (1.0-(float(total_err_count) / ((max_batch+1) * N_BATCH))) * 100.0, "%"
            val_result = self.val_set_test()
            print "accuracy is: ", val_result*100.0, "%"
            if val_result > best_val_accuracy:
                print "new best! test on test set..."
                best_val_accuracy = val_result

            test_accuracy = self.test_set_test()
            print "test set accuracy: ", test_accuracy * 100.0, "%"
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
            print "======================================="



def main(argv):
    wemb_size = None
    if len(argv) > 5:
        wemb_size = argv[4]
    model = DSSM_MLP_Model(argv[0], argv[1], argv[2], argv[3], argv[4], wemb_size)


    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])
