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
        self.ngram_dict_path = '../../data/pickles/four_gram_dict.pkl'

        self.ngram_dict = None
        self.ngram_feature_dim = None

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


    def model_constructor(self):

        self.story_sent1_input_variable = T.matrix('story1_input', dtype=theano.config.floatX)
        self.story_sent2_input_variable = T.matrix('story2_input', dtype=theano.config.floatX)
        self.story_sent3_input_variable = T.matrix('story3_input', dtype=theano.config.floatX)
        self.story_sent4_input_variable = T.matrix('story4_input', dtype=theano.config.floatX)

        self.end1_input_variable = T.matrix('ending_input', dtype = theano.config.floatX)
        self.end2_input_variable = T.matrix('neg_ending1_input', dtype = theano.config.floatX)




        self.encoder = MLP_Encoder.MLPEncoder(self.mlp_units)


        self.encoder.build_model(self.ngram_feature_dim)


        self.sent1_encode = lasagne.layers.get_output(self.encoder.output, 
                                                    {self.encoder.l_in:self.story_sent1_input_variable
                                                    })
        self.sent2_encode = lasagne.layers.get_output(self.encoder.output, 
                                                    {self.encoder.l_in:self.story_sent2_input_variable
                                                    })
        self.sent3_encode = lasagne.layers.get_output(self.encoder.output, 
                                                    {self.encoder.l_in:self.story_sent3_input_variable
                                                    })
        self.sent4_encode = lasagne.layers.get_output(self.encoder.output, 
                                                    {self.encoder.l_in:self.story_sent4_input_variable
                                                    })

        self.end1_encode = lasagne.layers.get_output(self.encoder.output, 
                                                        {self.encoder.l_in:self.end1_input_variable
                                                        })

        self.end2_encode = lasagne.layers.get_output(self.encoder.output, 
                                                        {self.encoder.l_in:self.end2_input_variable
                                                        })

        concate_in1 = lasagne.layers.InputLayer(shape=(None, self.mlp_units[-1]))
        concate_in2 = lasagne.layers.InputLayer(shape=(None, self.mlp_units[-1]))
        concate_in3 = lasagne.layers.InputLayer(shape=(None, self.mlp_units[-1]))
        concate_in4 = lasagne.layers.InputLayer(shape=(None, self.mlp_units[-1]))
        concate_in5 = lasagne.layers.InputLayer(shape=(None, self.mlp_units[-1]))

        concate_ls = [concate_in1, concate_in2, concate_in3, concate_in4, concate_in5]

        concate_layer = lasagne.layers.ConcatLayer(concate_ls, axis=1)

        end1_concate = lasagne.layers.get_output(concate_layer,{concate_in1: self.sent1_encode,
                                                 concate_in2: self.sent2_encode,
                                                 concate_in3: self.sent3_encode,
                                                 concate_in4: self.sent4_encode,
                                                 concate_in5: self.end1_encode,
                                                 })
        end2_concate = lasagne.layers.get_output(concate_layer,{concate_in1: self.sent1_encode,
                                                 concate_in2: self.sent2_encode,
                                                 concate_in3: self.sent3_encode,
                                                 concate_in4: self.sent4_encode,
                                                 concate_in5: self.end2_encode,
                                                 })

        self.reasoner = MLP_Encoder.MLPEncoder([512, 1024], classifier=True)
        self.reasoner.build_model(5*self.mlp_units[-1])

        # Construct symbolic cost function
        prob_vec1 = lasagne.layers.get_output(self.reasoner.output, 
                                        {self.reasoner.l_in:end1_concate})
        prob_vec2 = lasagne.layers.get_output(self.reasoner.output, 
                                        {self.reasoner.l_in:end2_concate})

        targets1 = T.vector(name='target1', dtype='int64')
        targets2 = T.vector(name='target2', dtype='int64')
        cost1 = lasagne.objectives.categorical_crossentropy(prob_vec1, targets1)
        cost2 = lasagne.objectives.categorical_crossentropy(prob_vec2, targets2)
        self.cost = lasagne.objectives.aggregate(cost1 + cost2)
        self.val_test_prob1 = prob_vec1
        self.val_test_prob2 = prob_vec2
        # Retrieve all parameters from the network

        all_params = self.encoder.all_params + self.reasoner.all_params

        
        all_updates = lasagne.updates.adam(self.cost, self.encoder.all_params)
        


        self.train_func = theano.function([self.story_sent1_input_variable, self.story_sent2_input_variable,
                                     self.story_sent3_input_variable, self.story_sent4_input_variable,
                                     self.end1_input_variable, self.end2_input_variable, targets1, targets2],
                                     [self.cost, self.val_test_prob1, self.val_test_prob2], updates = all_updates)


        self.compute_cost = theano.function([self.story_sent1_input_variable, self.story_sent2_input_variable,
                                     self.story_sent3_input_variable, self.story_sent4_input_variable,
                                     self.end1_input_variable, self.end2_input_variable, targets1, targets2],
                                     [self.cost, self.val_test_prob1, self.val_test_prob2])


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
        
        self.ngram_dict = pickle.load(open(self.ngram_dict_path))
        self.ngram_feature_dim = len(self.ngram_dict)


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
            story = [self.ngram_feature_generator(sent, 4).reshape((1,-1)).astype(theano.config.floatX) for sent in self.val_story[i]]

            ending1 = self.ngram_feature_generator(self.val_ending1[i], 4).reshape((1,-1)).astype(theano.config.floatX)

            ending2 = self.ngram_feature_generator(self.val_ending2[i], 4).reshape((1,-1)).astype(theano.config.floatX)

            cost, pred1, pred2 = self.compute_cost(story[0], story[1], story[2], story[3],
                                                   ending1, ending2, np.asarray([1-self.val_answer[i]]), 
                                                   np.asarray([self.val_answer[i]]))

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((pred1, pred2), axis = 1))

            predict_answer = 0
            if prediction == 0:
                predict_answer = 1
            elif prediction == 1:
                predict_answer = 0
            elif prediction == 2:
                predict_answer = 0
            else:
                predict_answer = 1


            if predict_answer == self.val_answer[i]:
                correct += 1.


        return correct/self.n_val

    def test_set_test(self):
        #load test set data
        correct = 0.

        for i in range(self.n_test):
            story = [self.ngram_feature_generator(sent, 4).reshape(1, -1).astype(theano.config.floatX) for sent in self.test_story[i]]

            ending1 = self.ngram_feature_generator(self.test_ending1[i], 4).reshape((1,-1)).astype(theano.config.floatX)

            ending2 = self.ngram_feature_generator(self.test_ending2[i], 4).reshape((1,-1)).astype(theano.config.floatX)

            cost, pred1, pred2 = self.compute_cost(story[0], story[1], story[2], story[3],
                                                   ending1, ending2, np.asarray([1 - self.test_answer[i]]),
                                                   np.asarray([self.test_answer[i]]))

            # Answer denotes the index of the anwer
            prediction = np.argmax(np.concatenate((pred1, pred2), axis = 1))

            predict_answer = 0
            if prediction == 0:
                predict_answer = 1
            elif prediction == 1:
                predict_answer = 0
            elif prediction == 2:
                predict_answer = 0
            else:
                predict_answer = 1


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
        sent_feature_vec = np.zeros(self.ngram_feature_dim)
        for word in sent:
            padding_word = '<'+word+'>'
            for i in range(len(padding_word)-2):
                if padding_word[i:i+3] in self.ngram_dict:
                    sent_feature_vec[self.ngram_dict[padding_word[i:i+3]]] += 1.0
        return sent_feature_vec

    def trigram_feature_generator(self, sent):
        sent_feature_vec = np.zeros(self.ngram_feature_dim)
        new_sent = " ".join(sent)
        for i in range(len(new_sent)-2):
            if new_sent[i:i+3] in self.ngram_dict:
                sent_feature_vec[self.ngram_dict[new_sent[i:i+3]]] += 1.0
        return sent_feature_vec

    def ngram_feature_generator(self, sent, ngram):
        sent_feature_vec = np.zeros(self.ngram_feature_dim)
        new_sent = " ".join(sent)
        for i in range(len(new_sent)- ngram + 1):
            if new_sent[i:i+ngram] in self.ngram_dict:
                sent_feature_vec[self.ngram_dict[new_sent[i:i+ngram]]] += 1.0
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
            total_err = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [self.train_story[index] for index in batch_index_list]
                train_ending = [self.train_ending[index] for index in batch_index_list]

                neg_end_index_matrix = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))
                while np.any((np.asarray(batch_index_list) - neg_end_index_matrix) == 0):
                    neg_end_index_matrix = np.random.randint(N_TRAIN_INS, size = (N_BATCH,))

                neg_end1 = [self.train_ending[index] for index in neg_end_index_matrix]

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

                
                story_sent1_feature = np.zeros((N_BATCH, self.ngram_feature_dim))
                story_sent2_feature = np.zeros((N_BATCH, self.ngram_feature_dim))
                story_sent3_feature = np.zeros((N_BATCH, self.ngram_feature_dim))
                story_sent4_feature = np.zeros((N_BATCH, self.ngram_feature_dim))

                train_end1_feature = np.zeros((N_BATCH, self.ngram_feature_dim))
                train_end2_feature = np.zeros((N_BATCH, self.ngram_feature_dim))


                for i in range(N_BATCH):
                    story_sent1_feature[i] = self.ngram_feature_generator(train_story[i][0],4)
                    story_sent2_feature[i] = self.ngram_feature_generator(train_story[i][1],4)
                    story_sent3_feature[i] = self.ngram_feature_generator(train_story[i][2],4)
                    story_sent4_feature[i] = self.ngram_feature_generator(train_story[i][3],4)

                    train_end1_feature[i] = self.ngram_feature_generator(end1[i], 4)
                    train_end2_feature[i] = self.ngram_feature_generator(end2[i], 4)                    




                cost, pred1, pred2 = self.train_func(story_sent1_feature, story_sent2_feature,
                                story_sent3_feature, story_sent4_feature,
                                train_end1_feature, train_end2_feature,
                                target1, target2)


                total_cost += cost
                prediction = np.zeros(N_BATCH)
                predict_vec = np.argmax(np.concatenate((pred1, pred2), axis = 1))

                for i in range(N_BATCH):

                    if predict_vec == 0:
                        prediction[i] = 1
                    elif predict_vec == 1:
                        prediction[i] = 0
                    elif predict_vec == 2:
                        prediction[i] = 0
                    else:
                        prediction[i] = 1

                total_err += (abs(prediction - answer)).sum()

                if batch_count % test_threshold == 0:
                    if batch_count == 0:
                        print "initial test"
                    else:
                        print" "
                        print "training set accuracy: ", (1-(total_err/(batch*N_BATCH)))*100,"%"
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
