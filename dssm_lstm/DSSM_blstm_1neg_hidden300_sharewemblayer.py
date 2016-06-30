import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import collections
import BLSTM_Encoder
import MLP_Encoder
import sys

class DSSM_BLSTM_Model(object):
    def __init__(self,blstmmlp_setting, dropout_rate, batchsize, wemb_size = None):
        # Initialize Theano Symbolic variable attributes
        self.blstm_units = int(blstmmlp_setting.split('x')[0])
        self.mlp_units = [int(elem) for elem in blstmmlp_setting.split('x')[1:]]
        self.dropout_rate = float(dropout_rate)
        self.batchsize = int(batchsize)
        self.wemb_size = 300
        if wemb_size == None:
            self.random_init_wemb = False
        else:
            self.random_init_wemb = True
            self.wemb_size = int(wemb_size)

        self.story_input_variable = None
        self.story_mask = None

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
        self.compute_cost = None

        # Initialize data loading attributes
        self.wemb = None
        self.train_set_path = '../../data/pickles/train_index_corpus.pkl'
        self.val_set_path = '../../data/pickles/val_index_corpus.pkl'
        self.test_set_path = '../../data/pickles/test_index_corpus.pkl' 
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'

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

    def batch_cosine(self, doc_batch_proj, query_batch_proj):
        dot_prod = T.batched_dot(doc_batch_proj, query_batch_proj)

        doc_square = doc_batch_proj ** 2
        query_square = query_batch_proj ** 2

        doc_norm = (T.sqrt(doc_square.sum(axis = 1))).sum()
        query_norm = T.sqrt(query_square.sum(axis = 1)).sum()

        batch_cosine_vec = dot_prod/(doc_norm * query_norm)
        return batch_cosine_vec


    def model_constructor(self, wemb_matrix_path = None):

        self.story_input_variable = T.matrix('story_input', dtype='int64')
        self.story_mask = T.matrix('story_mask', dtype=theano.config.floatX)

        self.ending_input_variable = T.matrix('ending_input', dtype = 'int64')
        self.ending_mask = T.matrix('ending_mask', dtype = theano.config.floatX)

        self.neg_ending1_input_variable = T.matrix('neg_ending1_input', dtype = 'int64')
        self.neg_ending1_mask = T.matrix('neg_ending1_mask', dtype = theano.config.floatX)

        story_batch_size, story_seqlen = self.story_input_variable.shape
        story_reshape_input = self.story_input_variable.reshape([story_batch_size, story_seqlen, 1])

        ending_batch_size, ending_seqlen = self.ending_input_variable.shape
        ending_reshape_input = self.ending_input_variable.reshape([ending_batch_size, ending_seqlen, 1])

        neg_ending1_batchsize, neg_ending1_seqlen = self.neg_ending1_input_variable.shape
        neg_ending1_reshape_input = self.neg_ending1_input_variable.reshape([neg_ending1_batchsize, neg_ending1_seqlen,1])


        self.wemb_layer = BLSTM_Encoder.BlstmEncoder(self.blstm_units)

        self.wemb_layer.build_model(self.wemb)

        self.doc_encode_train = lasagne.layers.get_output(self.wemb_layer.output, 
                                                    {self.wemb_layer.l_in:story_reshape_input, 
                                                     self.wemb_layer.l_mask:self.story_mask},deterministic = False)

        self.pos_query_encode_train = lasagne.layers.get_output(self.wemb_layer.output, 
                                                        {self.wemb_layer.l_in:ending_reshape_input, 
                                                         self.wemb_layer.l_mask:self.ending_mask},deterministic = False)

        self.neg_query_encode_train = lasagne.layers.get_output(self.wemb_layer.output, 
                                                        {self.wemb_layer.l_in:neg_ending1_reshape_input, 
                                                         self.wemb_layer.l_mask:self.neg_ending1_mask},deterministic = False)

        self.doc_encode_test = lasagne.layers.get_output(self.wemb_layer.output, 
                                                    {self.wemb_layer.l_in:story_reshape_input, 
                                                     self.wemb_layer.l_mask:self.story_mask},deterministic = True)

        self.pos_query_encode_test = lasagne.layers.get_output(self.wemb_layer.output, 
                                                        {self.wemb_layer.l_in:ending_reshape_input, 
                                                         self.wemb_layer.l_mask:self.ending_mask},deterministic = True)

        self.neg_query_encode_test = lasagne.layers.get_output(self.wemb_layer.output, 
                                                        {self.wemb_layer.l_in:neg_ending1_reshape_input, 
                                                         self.wemb_layer.l_mask:self.neg_ending1_mask},deterministic = True)

        self.story_reasonMLP = MLP_Encoder.MLPEncoder(self.mlp_units)
        self.end_reasonMLP = MLP_Encoder.MLPEncoder(self.mlp_units)

        self.story_reasonMLP.build_model(self.blstm_units)
        self.end_reasonMLP.build_model(self.blstm_units)

        self.story_encode_train = lasagne.layers.get_output(self.story_reasonMLP.output,
                                                      {self.story_reasonMLP.l_in:self.doc_encode_train}, deterministic = False)
        self.pos_end_encode_train = lasagne.layers.get_output(self.end_reasonMLP.output,
                                                        {self.end_reasonMLP.l_in:self.pos_query_encode_train}, deterministic = False)
        self.neg_end_encode_train = lasagne.layers.get_output(self.end_reasonMLP.output,
                                                        {self.end_reasonMLP.l_in:self.neg_query_encode_train}, deterministic = False)

        self.story_encode_test = lasagne.layers.get_output(self.story_reasonMLP.output,
                                                      {self.story_reasonMLP.l_in:self.doc_encode_test}, deterministic = True)
        self.pos_end_encode_test = lasagne.layers.get_output(self.end_reasonMLP.output,
                                                        {self.end_reasonMLP.l_in:self.pos_query_encode_test}, deterministic = True)
        self.neg_end_encode_test = lasagne.layers.get_output(self.end_reasonMLP.output,
                                                        {self.end_reasonMLP.l_in:self.neg_query_encode_test}, deterministic = True)




        # Construct symbolic cost function
        pos_cos_vec_train = self.batch_cosine(self.story_encode_train, self.pos_end_encode_train)
        neg_cos_vec_train = self.batch_cosine(self.story_encode_train, self.neg_end_encode_test)

        pos_cos_vec_test = self.batch_cosine(self.story_encode_test, self.pos_end_encode_test)
        self.cost = (neg_cos_vec_train - pos_cos_vec_train).sum()
        self.val_test_cos = pos_cos_vec_test.sum()
        # Retrieve all parameters from the network

        wemb_params = self.wemb_layer.all_params
        storyreason_params = self.story_reasonMLP.all_params
        endreason_params = self.end_reasonMLP.all_params
        
        wemb_updates = lasagne.updates.adam(self.cost, wemb_params)
        story_reason_updates = lasagne.updates.adam(self.cost, storyreason_params)
        end_reason_updates = lasagne.updates.adam(self.cost, endreason_params)

        all_updates = wemb_updates.items() + story_reason_updates.items() + end_reason_updates.items()
        
        # for k,v in doc_updates.items() + query_updates.items():
        #     if k in all_params:
        #         continue
        #     else:
        #         all_params[k] = v

        self.train_func = theano.function([self.story_input_variable, self.story_mask, 
                                     self.ending_input_variable, self.ending_mask,
                                     self.neg_ending1_input_variable, self.neg_ending1_mask],
                                     self.cost, updates = all_updates)

        # Compute adam updates for training

        self.compute_cost = theano.function([self.story_input_variable, self.story_mask, 
                                     self.ending_input_variable, self.ending_mask],
                                     self.val_test_cos)


    def load_data(self):
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = utils.combine_sents(train_set[0])
        self.train_ending = train_set[1]

        val_set = pickle.load(open(self.val_set_path))

        self.val_story = utils.combine_sents(train_set[0])
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.n_val = len(self.val_answer)

        test_set = pickle.load(open(self.test_set_path))
        self.test_story = utils.combine_sents(test_set[0])
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

    def fake_load_data(self):
        self.train_story = np.random.randint(1000, size = (1000, 20)).astype('int64')
        self.train_ending = np.random.randint(1000, size = (1000, 20)).astype('int64')

        self.val_story = np.random.randint(1000, size = (200, 20)).astype('int64')
        self.val_ending1 = np.random.randint(1000, size = (200, 20)).astype('int64')
        self.val_ending2 = np.random.randint(1000, size = (200, 20)).astype('int64')
        self.val_answer = np.random.randint(2, size = (200,))
        self.n_val = self.val_answer.shape[0]

        self.wemb = theano.shared(np.random.rand(1000, 300)).astype(theano.config.floatX)


    def val_set_test(self):

        correct = 0.
        result_list = np.zeros((self.n_val, 3))

        for i in range(self.n_val):
            story = np.asarray(self.val_story[i], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.val_story[i])))

            ending1 = np.asarray(self.val_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.val_ending1[i])))

            ending2 = np.asarray(self.val_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.val_ending2[i])))

            cos1 = self.compute_cost(story, story_mask, ending1, ending1_mask)
            cos2 = self.compute_cost(story, story_mask, ending2, ending2_mask)

            # Answer denotes the index of the anwer
            predict_answer = 0
            if cos1 < cos2:
                predict_answer = 1

            if predict_answer == self.val_answer[i]:
                correct += 1.

            result_list[i][0] = cos1
            result_list[i][1] = cos2
            result_list[i][2] = predict_answer+1


        return correct/self.n_val, result_list

    def test_set_test(self):
        #load test set data
        correct = 0.
        result_list = np.zeros((self.n_val, 3))

        for i in range(self.n_test):
            story = np.asarray(self.test_story[i], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.test_story[i])))

            ending1 = np.asarray(self.test_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.test_ending1[i])))

            ending2 = np.asarray(self.test_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.test_ending2[i])))

            cos1 = self.compute_cost(story, story_mask, ending1, ending1_mask)
            cos2 = self.compute_cost(story, story_mask, ending2, ending2_mask)

            # Answer denotes the index of the anwer
            predict_answer = 0
            if cos1 < cos2:
                predict_answer = 1

            if predict_answer == self.test_answer[i]:
                correct += 1.

            result_list[i][0] = cos1
            result_list[i][1] = cos2
            result_list[i][2] = predict_answer+1

        return correct/self.n_test, result_list


    def saving_model(self, val_or_test, accuracy):
        all_params_value = lasagne.layers.get_all_param_values(self.reason_layer.output)

        if val_or_test == 'val':
            pickle.dump((all_params_value, accuracy), open(self.best_val_model_save_path, 'wb'))
            pickle.dump(self.wemb.get_value(), open(self.best_val_wemb_save_path, 'wb'))
        else:
            pickle.dump((all_params_value, accuracy), open(self.best_test_model_save_path, 'wb'))            
            pickle.dump(self.wemb.get_value(), open(self.best_test_wemb_save_path, 'wb'))

    def reload_model(self, val_or_test):
        if val_or_test == 'val': 

            all_params, accuracy = pickle.load(open(self.best_val_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, all_params)
            self.wemb.set_value(pickle.load(open(self.best_val_wemb_save_path)))
            print "This model has ", accuracy * 100, "%  accuracy on valid set" 
        else:
            all_params,accuracy = pickle.load(open(self.best_test_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, all_params)
            self.wemb.set_value(pickle.load(open(self.best_test_wemb_save_path)))
            print "This model has ", accuracy * 100, "%  accuracy on test set" 

    def begin_train(self):
        N_EPOCHS = 30
        N_BATCH = self.batchsize
        N_TRAIN_INS = len(self.train_ending)
        best_val_accuracy = 0
        best_test_accuracy = 0
        test_threshold = 5000/N_BATCH
        prev_percetage = 0.0
        speed = 0.0
        batch_count = 0.0

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
                train_ending = [self.train_ending[index] for index in batch_index_list]

                neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                while np.any((np.asarray(batch_index_list) - neg_end_index_list) == 0):
                    neg_end_index_list = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                neg_end1 = [self.train_ending[index] for index in neg_end_index_list]

                train_story_matrix = utils.padding(train_story)
                train_ending_matrix = utils.padding(train_ending)
                neg_ending1_matrix = utils.padding(neg_end1)

                train_story_mask = utils.mask_generator(train_story)
                train_ending_mask = utils.mask_generator(train_ending)
                neg_ending1_mask = utils.mask_generator(neg_end1)

                cost = self.train_func(train_story_matrix, train_story_mask, 
                                        train_ending_matrix, train_ending_mask,
                                        neg_ending1_matrix, neg_ending1_mask)

                # peek on val set every 5000 instances(1000 batches)
                if batch_count % test_threshold == 0:
                    if batch_count == 0:
                        print "initial test"
                    else:
                        print" "
                    print"test on valid set..."
                    val_result, val_result_list = self.val_set_test()
                    print "accuracy is: ", val_result*100, "%"
                    if val_result > best_val_accuracy:
                        print "new best! test on test set..."
                        best_val_accuracy = val_result
                        if best_val_accuracy > 0.6:
                            pickle.dump(val_result_list, open('./prediction/BLSTM_1neg_shareall_best_val_prediction.pkl','wb'))

                        test_accuracy, test_result_list = self.test_set_test()
                        print "test set accuracy: ", test_accuracy * 100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
                            if best_test_accuracy > 0.6:
                                pickle.dump(test_result_list, open('./prediction/BLSTM_1neg_shareall_best_test_prediction.pkl','wb'))

                batch_count += 1
                total_cost += cost
                if batch_count != 0 and batch_count % 10 == 0:
                    speed = N_BATCH * 10.0 / (time.time() - start_time)
                    start_time = time.time()

                percetage = ((batch_count % test_threshold)+1) / test_threshold * 100
                if percetage - prev_percetage >= 1:
                    utils.progress_bar(percetage, speed)

            print "======================================="
            print "epoch summary:"
            print "total cost in this epoch: ", total_cost
            print "======================================="



def main(argv):
    wemb_size = None
    if len(argv) > 3:
        wemb_size = argv[3]
    model = DSSM_BLSTM_Model(argv[0], argv[1], argv[2], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])
