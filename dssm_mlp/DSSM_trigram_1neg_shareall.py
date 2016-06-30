import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import collections
import MLP_Encoder
import sys



class DSSM_MLP_Model(object):
    def __init__(self,mlp_setting, dropout_rate, batchsize, wemb_size = None):
        # Initialize Theano Symbolic variable attributes
        self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.dropout_rate = float(dropout_rate)
        self.batchsize = int(batchsize)

        self.story_input_variable = None

        self.ending_input_variable = None

        self.neg_ending1_input_variable = None

        self.neg_ending2_input_variable = None

        self.neg_ending3_input_variable = None

        self.encoder = None

        self.doc_encode = None 
        self.pos_end_encode = None
        self.neg_end1_encode = None


        self.cost = None
        self.val_test_cos = None

        self.train_func = None
        self.compute_cost = None

        # Initialize data loading attributes
        self.train_set_path = '../../data/pickles/train_wordssplit.pkl'
        self.val_set_path = '../../data/pickles/val_wordssplit.pkl'
        self.test_set_path = '../../data/pickles/test_wordssplit.pkl' 
        self.trigram_dict_path = '../../data/pickles/tri_gram_dict.pkl'

        self.trigram_dict = None
        self.trigram_feature_dim = None

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



    def model_constructor(self):

        self.story_input_variable = T.matrix('story_input', dtype=theano.config.floatX)

        self.ending_input_variable = T.matrix('ending_input', dtype = theano.config.floatX)

        self.neg_ending1_input_variable = T.matrix('neg_ending1_input', dtype = theano.config.floatX)




        self.encoder = MLP_Encoder.MLPEncoder(self.mlp_units)


        self.encoder.build_model(self.trigram_feature_dim)


        self.doc_encode = lasagne.layers.get_output(self.encoder.output, 
                                                    {self.encoder.l_in:self.story_input_variable
                                                    })

        self.pos_end_encode = lasagne.layers.get_output(self.encoder.output, 
                                                        {self.encoder.l_in:self.ending_input_variable
                                                        })

        self.neg_end1_encode = lasagne.layers.get_output(self.encoder.output, 
                                                        {self.encoder.l_in:self.neg_ending1_input_variable
                                                        })

 

        # Construct symbolic cost function
        pos_cos_vec = self.batch_cosine(self.doc_encode, self.pos_end_encode)
        neg1_cos_vec = self.batch_cosine(self.doc_encode, self.neg_end1_encode)


        self.cost = (T.log(T.exp(neg1_cos_vec) + T.exp(pos_cos_vec))  - pos_cos_vec).sum()
        self.val_test_cos1 = pos_cos_vec
        self.val_test_cos2 = neg1_cos_vec
        # Retrieve all parameters from the network

        all_params = self.encoder.all_params

        
        all_updates = lasagne.updates.adam(self.cost, self.encoder.all_params)
        


        self.train_func = theano.function([self.story_input_variable,
                                     self.ending_input_variable,
                                     self.neg_ending1_input_variable,],
                                     self.cost, updates = all_updates)

        # Compute adam updates for training
        self.train_cost = theano.function([self.story_input_variable,
                                     self.ending_input_variable,
                                     self.neg_ending1_input_variable,],
                                     self.cost)

        self.compute_cost = theano.function([self.story_input_variable,
                                     self.ending_input_variable, self.neg_ending1_input_variable],
                                     [self.val_test_cos1, self.val_test_cos2])


    def load_data(self):
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = train_set[0]
        self.train_ending = train_set[1]

        val_set = pickle.load(open(self.val_set_path))

        self.val_story = train_set[0]
        self.val_ending1 = val_set[1]
        self.val_ending2 = val_set[2]
        self.val_answer = val_set[3]

        self.n_val = len(self.val_answer)

        test_set = pickle.load(open(self.test_set_path))
        self.test_story = test_set[0]
        self.test_ending1 = test_set[1]
        self.test_ending2 = test_set[2]
        self.test_answer = test_set[3]
        self.n_test = len(self.test_story)
        
        self.trigram_dict = pickle.load(open(self.trigram_dict_path))
        self.trigram_feature_dim = len(self.trigram_dict)


    def fake_load_data(self):
        self.train_story = np.random.randint(1000, size = (1000, 16708)).astype(theano.config.floatX)
        self.train_ending = np.random.randint(1000, size = (1000, 16708)).astype(theano.config.floatX)

        self.val_story = np.random.randint(1000, size = (200, 16708)).astype(theano.config.floatX)
        self.val_ending1 = np.random.randint(1000, size = (200, 16708)).astype(theano.config.floatX)
        self.val_ending2 = np.random.randint(1000, size = (200, 16708)).astype(theano.config.floatX)
        self.val_answer = np.random.randint(2, size = (200,))
        self.n_val = self.val_answer.shape[0]


    def val_set_test(self):

        correct = 0.

        for i in range(self.n_val):
            story = sum([self.trigram_feature_generator(sent) for sent in self.val_story[i]]).reshape((1,-1)).astype(theano.config.floatX)

            ending1 = self.trigram_feature_generator(self.val_ending1[i]).reshape((1,-1)).astype(theano.config.floatX)

            ending2 = self.trigram_feature_generator(self.val_ending2[i]).reshape((1,-1)).astype(theano.config.floatX)

            pred1, pred2 = self.compute_cost(story, ending1, ending2)

            # Answer denotes the index of the anwer
            predict_answer = 1
            if pred1 < pred2:
                predict_answer = 2

            if predict_answer == self.val_answer[i]:
                correct += 1.


        return correct/self.n_val

    def test_set_test(self):
        #load test set data

        correct = 0.

        for i in range(self.n_test):
            story = sum([self.trigram_feature_generator(sent) for sent in self.test_story[i]]).reshape((1,-1)).astype(theano.config.floatX)

            ending1 = self.trigram_feature_generator(self.test_ending1[i]).reshape((1,-1)).astype(theano.config.floatX)

            ending2 = self.trigram_feature_generator(self.test_ending2[i]).reshape((1,-1)).astype(theano.config.floatX)

            pred1, pred2 = self.compute_cost(story, ending1, ending2)

            # Answer denotes the index of the anwer
            predict_answer = 1
            if pred1 < pred2:
                predict_answer = 2

            if predict_answer == self.test_answer[i]:
                correct += 1.



        return correct/self.n_test

    def saving_model(self, val_or_test, accuracy):
        all_params_value = lasagne.layers.get_all_param_values(self.encoder.output)

        if val_or_test == 'val':
            pickle.dump((all_params_value, accuracy), open(self.best_val_model_save_path, 'wb'))
        else:
            pickle.dump((all_params_value, accuracy), open(self.best_test_model_save_path, 'wb'))

    def reload_model(self, val_or_test):
        if val_or_test == 'val':    
            encoder_params, accuracy = pickle.load(open(self.best_val_model_save_path))
            lasagne.layers.set_all_param_values(self.encoder.output, encoder_params)

            print "This model has ", accuracy * 100, "%  accuracy on valid set" 
        else:
            encoder_params, accuracy = pickle.load(open(self.best_test_model_save_path))
            lasagne.layers.set_all_param_values(self.encoder.output, encoder_params)
            print "This model has ", accuracy * 100, "%  accuracy on test set"  

    def letter_trigram_feature_generator(self, sent):
        sent_feature_vec = np.zeros(self.trigram_feature_dim)
        for word in sent:
            padding_word = '<'+word+'>'
            for i in range(len(padding_word)-2):
                if padding_word[i:i+3] in self.trigram_dict:
                    sent_feature_vec[self.trigram_dict[padding_word[i:i+3]]] += 1.0
        return sent_feature_vec

    def trigram_feature_generator(self, sent):
        sent_feature_vec = np.zeros(self.trigram_feature_dim)
        new_sent = " ".join(sent)
        for i in range(len(new_sent)-2):
            if new_sent[i:i+3] in self.trigram_dict:
                sent_feature_vec[self.trigram_dict[new_sent[i:i+3]]] += 1.0
        return sent_feature_vec

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

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [self.train_story[index] for index in batch_index_list]
                train_ending = [self.train_ending[index] for index in batch_index_list]

                neg_end_index_matrix = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                while np.any((np.asarray(batch_index_list) - neg_end_index_matrix) == 0):
                    neg_end_index_matrix = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))

                neg_end1 = [self.train_ending[index] for index in neg_end_index_matrix]


                
                train_story_feature = np.zeros((N_BATCH, self.trigram_feature_dim))
                train_ending_feature = np.zeros((N_BATCH, self.trigram_feature_dim))
                neg_end1_feature = np.zeros((N_BATCH, self.trigram_feature_dim))


                for i in range(N_BATCH):
                    train_story_feature[i] = sum([self.trigram_feature_generator(sent) for sent in train_story[i]])
                    train_ending_feature[i] = self.trigram_feature_generator(train_ending[i])
                    neg_end1_feature[i] = self.trigram_feature_generator(neg_end1[i])                    




                self.train_func(train_story_feature,
                                train_ending_feature, 
                                neg_end1_feature)
                total_cost += self.train_cost(train_story_feature,
                                train_ending_feature, 
                                neg_end1_feature)

                if batch_count % test_threshold == 0:
                    if batch_count == 0:
                        print "initial test"
                    else:
                        print" "
                    print"test on valid set..."
                    val_result = self.val_set_test()
                    print "accuracy is: ", val_result*100, "%"
                    if val_result > best_val_accuracy:
                        print "new best! test on test set..."
                        best_val_accuracy = val_result

                        test_accuracy = self.test_set_test()
                        print "test set accuracy: ", test_accuracy * 100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy
              

                batch_count += 1
                if batch_count != 0 and batch_count % 10 == 0:
                    speed = N_BATCH * 10.0 / (time.time() - start_time)
                    start_time = time.time()

                percetage = ((batch_count % test_threshold)+1) / test_threshold * 100
                if percetage - prev_percetage >= 1:
                    utils.progress_bar(percetage, speed)
            print ""                
            print "======================================="
            print "epoch summary:"
            print "total cost in this epoch: ", total_cost
            print "======================================="


def main(argv):
    wemb_size = None
    if len(argv) > 3:
        wemb_size = argv[3]
    model = DSSM_MLP_Model(argv[0], argv[1], argv[2], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])
