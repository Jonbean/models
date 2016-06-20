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

class DSSM_BLSTM_Model(object):
    def __init__(self):
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
        self.best_val_model_save_path = './best_models_params/BLSTM_neg1_300_class_best_val_model_params.pkl'
        self.best_test_model_save_path = './best_models_params/BLSTM_neg1_300_class_best_test_model_params.pkl'
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'
        self.best_val_wemb_save_path = './best_models_params/BLSTM_neg1_300_class_best_val_model_wemb.pkl'
        self.best_test_wemb_save_path = './best_models_params/BLSTM_neg1_300_class_best_test_model_wemb.pkl'

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
        if wemb_matrix_path != None:
            self.wemb = theano.shared(pickle.load(open(wemb_matrix_path))).astype(theano.config.floatX)


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


        self.reason_layer = BLSTMMLP_Encoder.BlstmMlpEncoder(LSTMLAYER_1_UNITS = 300, MLP_layer1 = 500, MLP_layer2 = 300)

        self.reason_layer.build_model(self.wemb)

        self.story_encode_train = lasagne.layers.get_output(self.reason_layer.output, 
                                                    {self.reason_layer.l_in:story_reshape_input, 
                                                     self.reason_layer.l_mask:self.story_mask},deterministic = False)

        self.end1_encode_train = lasagne.layers.get_output(self.reason_layer.output, 
                                                        {self.reason_layer.l_in:ending_reshape_input, 
                                                         self.reason_layer.l_mask:self.ending_mask},deterministic = False)

        self.end2_encode_train = lasagne.layers.get_output(self.reason_layer.output, 
                                                        {self.reason_layer.l_in:neg_ending1_reshape_input, 
                                                         self.reason_layer.l_mask:self.neg_ending1_mask},deterministic = False)

        self.story_encode_test = lasagne.layers.get_output(self.reason_layer.output, 
                                                    {self.reason_layer.l_in:story_reshape_input, 
                                                     self.reason_layer.l_mask:self.story_mask},deterministic = True)

        self.end1_encode_test = lasagne.layers.get_output(self.reason_layer.output, 
                                                        {self.reason_layer.l_in:ending_reshape_input, 
                                                         self.reason_layer.l_mask:self.ending_mask},deterministic = True)

        self.end2_encode_test = lasagne.layers.get_output(self.reason_layer.output, 
                                                        {self.reason_layer.l_in:neg_ending1_reshape_input, 
                                                         self.reason_layer.l_mask:self.neg_ending1_mask},deterministic = True)

        story_in = lasagne.layers.InputLayer(shape=(None, 300, 1))
        end1_in = lasagne.layers.InputLayer(shape=(None, 300, 1))
        end2_in = lasagne.layers.InputLayer(shape=(None, 300, 1))

        classif_in = lasagne.layers.ConcatLayer([story_in, end1_in, end2_in])
        # Construct symbolic cost function
        targets = T.matrix('gold_target', dtype= theano.config.floatX)

        

        classification_layer1 = lasagne.layers.DenseLayer(classif_in, num_units = 500,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W=lasagne.init.GlorotUniform())

        self.classification_layer2 = lasagne.layers.DenseLayer(classification_layer1, num_units = 2,
                                                          nonlinearity = lasagne.nonlinearities.softmax)

        softmax_output_train = lasagne.layers.get_output(self.classification_layer2, 
                                                        {story_in: self.story_encode_train,
                                                         end1_in: self.end1_encode_train,
                                                         end2_in:self.end2_encode_train},deterministic = True)

        softmax_output_test = lasagne.layers.get_output(self.classification_layer2,
                                                        {story_in: self.story_encode_test,
                                                        end1_in: self.end1_encode_test,
                                                        end2_in: self.end2_encode_test}, deterministic = True)

        classi_out_params = lasagne.layers.get_all_params(self.classification_layer2)
        
        self.cost = lasagne.objectives.categorical_crossentropy(softmax_output_train, targets).sum()

        self.predict = np.argmax(softmax_output_test, axis = 1)
        # Retrieve all parameters from the network

        all_params = self.reason_layer.all_params + classi_out_params

        
        all_updates = lasagne.updates.adam(self.cost, all_params)


        self.train_func = theano.function([self.story_input_variable, self.story_mask, 
                                     self.ending_input_variable, self.ending_mask,
                                     self.neg_ending1_input_variable, self.neg_ending1_mask,
                                     targets],
                                     self.cost, updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function([self.story_input_variable, self.story_mask, 
                                     self.ending_input_variable, self.ending_mask,
                                     self.neg_ending1_input_variable, self.neg_ending1_mask],
                                     self.predict)


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

            prediction = self.prediction(story, story_mask, ending1, ending1_mask, ending2, ending2_mask)
            
            # Answer denotes the index of the anwer

            if prediction == self.val_answer[i]:
                correct += 1.

            result_list[i][0] = prediction
            result_list[i][1] = self.val_answer[i]


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

            prediction = self.prediction(story, story_mask, ending1, ending1_mask, ending2, ending2_mask)
            
            # Answer denotes the index of the anwer

            if prediction == self.test_answer[i]:
                correct += 1.

            result_list[i][0] = prediction
            result_list[i][1] = self.test_answer[i]


        return correct/self.n_test, result_list


    def saving_model(self, val_or_test, accuracy):
        reason_params_value = lasagne.layers.get_all_param_values(self.reason_layer.output)
        classif_params_value = lasagne.layers.get_all_param_values(self.classification_layer2)

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
            lasagne.layers.set_all_param_values(self.classification_layer2, classif_params)

            print "This model has ", accuracy * 100, "%  accuracy on valid set" 
        else:
            reason_params, classif_params, accuracy = pickle.load(open(self.best_test_model_save_path))
            lasagne.layers.set_all_param_values(self.reason_layer.output, reason_params)
            lasagne.layers.set_all_param_values(self.classification_layer2, classif_params_value)
            print "This model has ", accuracy * 100, "%  accuracy on test set" 

    def begin_train(self):
        N_EPOCHS = 30
        N_BATCH = 20
        N_TRAIN_INS = len(self.train_ending)
        best_val_accuracy = 0
        best_test_accuracy = 0
        test_threshold = 1000.0
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

                answer = np.random.randint(2, size = N_BATCH)
                answer_vec = np.concatenate((answer.reshape(-1,1), (1 - answer).reshape(-1,1)), axis = 1).astype('int64')
                train_end1 = []
                train_end2 = []
                for i in range(N_BATCH):
                    if answer[i] == 0:
                        train_end1.append(train_ending[i])
                        train_end2.append(neg_end1[i])
                    else:
                        train_end1.append(neg_end1[i])
                        train_end2.append(train_ending[i])

                train_story_matrix = utils.padding(train_story)
                train_end1_matrix = utils.padding(train_end1)
                train_end2_matrix = utils.padding(train_end2)

                train_story_mask = utils.mask_generator(train_story)
                train_end1_mask = utils.mask_generator(train_end1)
                train_end2_mask = utils.mask_generator(train_end2)

                

                cost = self.train_func(train_story_matrix, train_story_mask, 
                                        train_end1_matrix, train_end1_mask,
                                        train_end2_matrix, train_end2_mask, answer_vec)

                prediction = self.prediction(train_story_matrix, train_story_mask,
                                             train_end1_matrix, train_end1_mask,
                                             train_end2_matrix, train_end2_mask)

                total_err_count += (prediction - answer).sum()
                if batch_count != 0 and batch_count % 10 == 0:
                    speed = N_BATCH * 10.0 / (time.time() - start_time)
                    start_time = time.time()

                percetage = ((batch_count % test_threshold)+1) / test_threshold * 100
                if percetage - prev_percetage >= 1:
                    utils.progress_bar(percetage, speed)

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
                        pickle.dump(val_result_list, open('./prediction/BLSTM_1neg_class_best_val_prediction.pkl','wb'))

                        test_accuracy, test_result_list = self.test_set_test()
                        print "test set accuracy: ", test_accuracy * 100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
                            print "saving model..."
                            self.saving_model('test', best_test_accuracy)
                            pickle.dump(test_result_list, open('./prediction/BLSTM_1neg_class_best_test_prediction.pkl','wb'))

                batch_count += 1
            total_cost += cost
            accuracy = 1-(total_err_count/(max_batch*N_BATCH))
            print ""
            print "total cost in this epoch: ", total_cost
            print "accuracy in this epoch: ", accuracy * 100, "%"


        print "reload best model for testing on test set"
        self.reload_model('val')

        print "test on test set..."
        test_result = self.test_set_test()
        print "accuracy is: ", test_result * 100, "%"

def main():
    model = DSSM_BLSTM_Model()

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main()

