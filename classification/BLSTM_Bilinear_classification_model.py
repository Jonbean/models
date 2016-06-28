import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import collections
import BLSTMMLP_Encoder
import sys

theano.config.optimizer = 'None'

class DSSM_BLSTM_Model(object):
    def __init__(self, blstmmlp_setting, dropout_rate, batchsize, wemb_size = None):
        # Initialize Theano Symbolic variable attributes
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
        self.blstmmlp_setting = blstmmlp_setting
        self.best_val_model_save_path = './best_models_params/BLSTMLP_'+blstmmlp_setting+'_bilinear_'+\
                                        'dropout'+dropout_rate+'_batch_'+batchsize+'_best_val.pkl'
        self.best_test_model_save_path = './best_models_params/BLSTMLP_'+blstmmlp_setting+'_bilinear_'+\
                                        'dropout'+dropout_rate+'_batch_'+batchsize+'_best_test.pkl'
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
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


    def model_constructor(self, wemb_size = None):

        self.story_input_variable = T.matrix('story_input', dtype='int64')
        self.story_mask = T.matrix('story_mask', dtype=theano.config.floatX)

        self.ending1_input_variable = T.matrix('ending_input', dtype = 'int64')
        self.ending1_mask = T.matrix('ending_mask', dtype = theano.config.floatX)

        self.ending2_input_variable = T.matrix('ending_input', dtype = 'int64')
        self.ending2_mask = T.matrix('ending_mask', dtype = theano.config.floatX)

        story_batch_size, story_seqlen = self.story_input_variable.shape
        story_reshape_input = self.story_input_variable.reshape([story_batch_size, story_seqlen, 1])

        ending1_batch_size, ending1_seqlen = self.ending1_input_variable.shape
        ending1_reshape_input = self.ending1_input_variable.reshape([ending1_batch_size, ending1_seqlen, 1])

        ending2_batch_size, ending2_seqlen = self.ending2_input_variable.shape
        ending2_reshape_input = self.ending2_input_variable.reshape([ending2_batch_size, ending2_seqlen, 1])

        self.reason_layer = BLSTMMLP_Encoder.BlstmMlpEncoder(LSTMLAYER_1_UNITS = self.blstm_units, MLP_UNITS = self.mlp_units,
                                                            dropout_rate = self.dropout_rate)

        self.reason_layer.build_model(self.wemb)

        self.story_encode_train = lasagne.layers.get_output(self.reason_layer.output,{self.reason_layer.l_in:story_reshape_input, 
                                                     self.reason_layer.l_mask:self.story_mask},deterministic = False)

        self.end1_encode_train = lasagne.layers.get_output(self.reason_layer.output,{self.reason_layer.l_in:ending1_reshape_input, 
                                                         self.reason_layer.l_mask:self.ending1_mask},deterministic = False)


        self.end2_encode_train = lasagne.layers.get_output(self.reason_layer.output,{self.reason_layer.l_in:ending2_reshape_input, 
                                                         self.reason_layer.l_mask:self.ending2_mask},deterministic = False)


        self.story_encode_test = lasagne.layers.get_output(self.reason_layer.output,{self.reason_layer.l_in:story_reshape_input, 
                                                     self.reason_layer.l_mask:self.story_mask},deterministic = True)

        self.end1_encode_test = lasagne.layers.get_output(self.reason_layer.output,{self.reason_layer.l_in:ending1_reshape_input, 
                                                         self.reason_layer.l_mask:self.ending1_mask},deterministic = True)

        self.end2_encode_test = lasagne.layers.get_output(self.reason_layer.output,{self.reason_layer.l_in:ending2_reshape_input, 
                                                         self.reason_layer.l_mask:self.ending2_mask},deterministic = True)






        # Construct symbolic cost function
        target1 = T.vector('gold_target', dtype= theano.config.floatX)
        target2 = T.vector('gold_target', dtype= theano.config.floatX)

        self.Bilinear_W = theano.shared(np.random.rand(self.mlp_units[-1], self.mlp_units[-1]), name="bilinear_param")


        prob1 = T.dot(T.dot(self.story_encode_test, self.Bilinear_W), self.end1_encode_test.T)
        prob2 = T.dot(T.dot(self.story_encode_test, self.Bilinear_W), self.end2_encode_test.T)

        cost1 = target1 * T.log(prob1) + (1 - target1) * T.log(1 - prob1)
        cost2 = target2 * T.log(prob2) + (1 - target2) * T.log(1 - prob2)
        self.cost = (cost1 + cost2).sum()
        # Retrieve all parameters from the network

        all_params = self.reason_layer.all_params + [self.Bilinear_W]

        
        all_updates = lasagne.updates.adam(self.cost, all_params)

        self.train_func = theano.function([self.story_input_variable, self.story_mask, 
                                     self.ending1_input_variable, self.ending1_mask,
                                     self.ending2_input_variable, self.ending2_mask, target1, target2], 
                                     self.cost, updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function([self.story_input_variable, self.story_mask, 
                                     self.ending1_input_variable, self.ending1_mask,
                                     self.ending2_input_variable, self.ending2_mask],
                                     [prob1, prob2])


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
        result_list = np.zeros((self.n_val, 2))

        for i in range(self.n_val):
            story = np.asarray(self.val_story[i], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.val_story[i])))

            ending1 = np.asarray(self.val_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.val_ending1[i])))

            ending2 = np.asarray(self.val_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.val_ending2[i])))

            prediction1, prediction2 = self.prediction(story, story_mask, ending1, ending1_mask, ending2, ending2_mask)
            
            # Answer denotes the index of the anwer

            prediction1 = np.argmax(prediction1, axis = 1)
            prediction2 = np.argmax(prediction2, axis = 1)
            prediction = 0
            if prediction2 > prediction1:
                prediction = 1
            if prediction == self.val_answer[i]:
                correct += 1.

            result_list[i][0] = prediction1
            result_list[i][1] = prediction2


        return correct/self.n_val, result_list

    def test_set_test(self):
        #load test set data
        correct = 0.
        result_list = np.zeros((self.n_test, 2))

        for i in range(self.n_test):
            story = np.asarray(self.test_story[i], dtype='int64').reshape((1,-1))
            story_mask = np.ones((1,len(self.test_story[i])))

            ending1 = np.asarray(self.test_ending1[i], dtype='int64').reshape((1,-1))
            ending1_mask = np.ones((1,len(self.test_ending1[i])))

            ending2 = np.asarray(self.test_ending2[i], dtype='int64').reshape((1,-1))
            ending2_mask = np.ones((1, len(self.test_ending2[i])))

            prediction1, prediction2 = self.prediction(story, story_mask, ending1, ending1_mask, ending2, ending2_mask)
            
            # Answer denotes the index of the anwer
            prediction1 = np.argmax(prediction1, axis = 1)
            prediction2 = np.argmax(prediction2, axis = 1)
            # Answer denotes the index of the anwer
            prediction = 0
            if prediction2 > prediction1:
                prediction = 1
            # Answer denotes the index of the anwer

            if prediction == self.test_answer[i]:
                correct += 1.

            result_list[i][0] = prediction1
            result_list[i][1] = prediction2

        return correct/self.n_test, result_list


    def saving_model(self, val_or_test, accuracy):
        reason_params_value = lasagne.layers.get_all_param_values(self.reason_layer.output)
        Bilinear_W = self.Bilinear_W.get_value()

        if val_or_test == 'val':
            pickle.dump((reason_params_value, Bilinear_W, accuracy), 
                        open(self.best_val_model_save_path, 'wb'))
        else:
            pickle.dump((reason_params_value, Bilinear_W, accuracy), 
                        open(self.best_test_model_save_path, 'wb'))            

    def reload_model(self, val_or_test):
        if val_or_test == 'val': 

            reason_params, Bilinear_W, accuracy = pickle.load(open(self.best_val_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
            self.Bilinear_W.set_value(Bilinear_W)

            print "This model has ", accuracy * 100, "%  accuracy on valid set" 
        else:
            reason_params, Bilinear_W, accuracy = pickle.load(open(self.best_test_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
            self.Bilinear_W.set_value(Bilinear_W)
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
        start_batch = 0.0

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

                answer = np.random.randint(2, size = N_BATCH)
                target1 = 1 - answer
                target2 = answer
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



                train_story_matrix = utils.padding(train_story)
                train_end1_matrix = utils.padding(end1)
                train_end2_matrix = utils.padding(end2)

                train_story_mask = utils.mask_generator(train_story)
                train_end1_mask = utils.mask_generator(end1)
                train_end2_mask = utils.mask_generator(end2)
                

                cost = self.train_func(train_story_matrix, train_story_mask, 
                                        train_end1_matrix, train_end1_mask,
                                        train_end2_matrix, train_end2_mask, target1, target2)

                prediction1, prediction2 = self.prediction(train_story_matrix, train_story_mask,
                                             train_end1_matrix, train_end1_mask,
                                             train_end2_matrix, train_end2_mask)

                prediction = np.concatenate((np.max(prediction1, axis = 1).reshape(-1,1), 
                             np.max(prediction2, axis = 1).reshape(-1,1)), axis = 1)
                total_err_count += abs((np.argmax(prediction, axis = 1) - answer)).sum()

                '''master version print'''
                percetage = ((batch_count % test_threshold)+1) / test_threshold * 100
                if percetage - prev_percetage >= 1:
                    speed = N_BATCH * (batch_count - start_batch) / (time.time() - start_time)
                    start_time = time.time()
                    start_batch = batch_count
                    utils.progress_bar(percetage, speed)
                '''end of print'''
                    
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
                        self.saving_model('val', best_val_accuracy)
                        pickle.dump(val_result_list, open('./prediction/BLSTMLP_'+self.blstmmlp_setting+'_bilinear_'+\
                                        'dropout'+str(self.dropout_rate)+'_batch_'+str(self.batchsize)+'_best_val.pkl','wb'))

                        test_accuracy, test_result_list = self.test_set_test()
                        print "test set accuracy: ", test_accuracy * 100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
                            print "saving model..."
                            self.saving_model('test', best_test_accuracy)
                            pickle.dump(test_result_list, open('./prediction/BLSTMLP_'+self.blstmmlp_setting+'_bilinear_'+\
                                        'dropout'+str(self.dropout_rate)+'_batch_'+str(self.batchsize)+'_best_test.pkl','wb'))

                batch_count += 1
            total_cost += cost
            accuracy = 1.0 - (total_err_count/(max_batch*N_BATCH))
            speed = max_batch * N_BATCH / (time.time() - start_time)
            print "======================================="
            print "epoch summary:"
            print "average speed: ", speed, "instances/sec"

            print ""
            print "total cost in this epoch: ", total_cost
            print "accuracy in this epoch: ", accuracy * 100, "%"
            print "======================================="


        print "reload best model for testing on test set"
        self.reload_model('val')

        print "test on test set..."
        test_result = self.test_set_test()
        print "accuracy is: ", test_result * 100, "%"

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


