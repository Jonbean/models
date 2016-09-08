import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import time
import utils
import cPickle as pickle
import BLSTM_sequence
import BLSTM_last
import sys
# from theano.printing import pydotprint




class Hierachi_RNN(object):
    def __init__(self, rnn_setting, batchsize, wemb_size = None):
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
        self.train_set_path = '../../data/pickles/train_index_corpus.pkl'
        self.wemb_matrix_path = '../../data/pickles/index_wemb_matrix.pkl'

        self.rnn_units = int(rnn_setting)
        # self.mlp_units = [int(elem) for elem in mlp_setting.split('x')]
        self.bilinear_matrix = theano.shared(0.002*np.random.rand(self.rnn_units, self.rnn_units)-0.001)
        # self.dropout_rate = float(dropout_rate)
        self.batchsize = int(batchsize)

        # self.val_split_ratio = float(val_split_ratio)
        self.words_num = 28820

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

        self.train_encodinglayer_vecs = []
        self.test_encodinglayer_vecs = []
        self.reasoninglayer_vec1 = []
        self.reasoninglayer_vec2 = []
        self.reasoninglayer_vec1_test = []
        self.reasoninglayer_vec2_test = []
        self.reasoning_pool_results = []
        self.reasoning_pool_results_test = []
        self.reasoners = []
        self.attentioned_sent_rep1 = []
        self.attentioned_sent_rep2 = []
        self.bilinear_attention_matrix = theano.shared(0.02*np.random.rand(self.rnn_units, self.rnn_units) - 0.01)

    def encoding_layer(self):


        assert len(self.reshaped_inputs_variables)==len(self.inputs_masks)
        for i in range(self.story_nsent + 1):
            self.train_encodinglayer_vecs.append(lasagne.layers.get_output(self.encoder.output,
                                                        {self.encoder.l_in:self.reshaped_inputs_variables[i], 
                                                         self.encoder.l_mask:self.inputs_masks[i]},
                                                         deterministic = True))


        end_representation = (self.train_encodinglayer_vecs[-1] * self.inputs_masks[4].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[4].sum(axis = 1, keepdims = True)

        #The last one (index-5) is the average of the non-attentioned ending sequence
        self.train_encodinglayer_vecs.append(end_representation)
        
    def attention1_layer(self):        
        for i in range(self.story_nsent):
            n_batch, n_seq, _ = self.train_encodinglayer_vecs[i].shape

            #second attention

            bili_part1 = T.dot(self.train_encodinglayer_vecs[i], self.bilinear_attention_matrix)

            attention1_score_tensor = T.batched_dot(bili_part1, self.train_encodinglayer_vecs[5])

            numerator1 = self.inputs_masks[i] * T.exp(attention1_score_tensor - attention1_score_tensor.max(axis = 1, keepdims = True))
        
            attention1_weight_matrix = numerator1 / numerator1.sum(axis = 1, keepdims = True)

            attentioned_sent_seq1 = self.train_encodinglayer_vecs[i]*(attention1_weight_matrix.reshape([n_batch, n_seq, 1]))

            attentioned_sent_rep1 = T.sum(attentioned_sent_seq1, axis = 1) / T.sum(self.inputs_masks[i], axis = 1).reshape([-1, 1])

            self.attentioned_sent_rep1.append(attentioned_sent_rep1)

    def attention2_layer(self):
        for i in range(self.story_nsent):
            n_batch, n_seq, _ = self.train_encodinglayer_vecs[i].shape

            #second attention

            bili_part1 = T.dot(self.train_encodinglayer_vecs[i], self.bilinear_attention_matrix)

            attention2_score_tensor = T.batched_dot(bili_part1, self.vt_2nd_end_repr)

            numerator2 = self.inputs_masks[i] * T.exp(attention2_score_tensor - attention2_score_tensor.max(axis = 1, keepdims = True))
        
            attention2_weight_matrix = numerator2 / numerator2.sum(axis = 1, keepdims = True)

            attentioned_sent_seq2 = self.train_encodinglayer_vecs[i]*(attention2_weight_matrix.reshape([n_batch, n_seq, 1]))

            attentioned_sent_rep2 = T.sum(attentioned_sent_seq2, axis = 1) / T.sum(self.inputs_masks[i], axis = 1).reshape([-1, 1])

            self.attentioned_sent_rep2.append(attentioned_sent_rep2)
           
    def model_constructor(self, wemb_size = None):
        self.inputs_variables = []
        self.inputs_masks = []
        self.reshaped_inputs_variables = []
        self.vt_2nd_end = T.matrix('second_end', dtype='int64')
        self.vt_2nd_end_mask = T.matrix('second_end_mask', dtype=theano.config.floatX)

        for i in range(self.story_nsent+1):
            self.inputs_variables.append(T.matrix('story'+str(i)+'_input', dtype='int64'))
            self.inputs_masks.append(T.matrix('story'+str(i)+'_mask', dtype=theano.config.floatX))
            batch_size, seqlen = self.inputs_variables[i].shape
            self.reshaped_inputs_variables.append(self.inputs_variables[i].reshape([batch_size, seqlen, 1]))

        #initialize neural network units
        self.encoder = BLSTM_sequence.BlstmEncoder(LSTMLAYER_1_UNITS = self.rnn_units)
        self.encoder.build_model(self.wemb)

        #build encoding layer
        self.encoding_layer()

        '''============================================================================'''
        #encode possible second ending
        self.encode_2nd_end_seq = lasagne.layers.get_output(self.encoder.output,
                                                    {self.encoder.l_in:self.vt_2nd_end, 
                                                     self.encoder.l_mask:self.vt_2nd_end_mask},
                                                     deterministic = True)

        self.vt_2nd_end_repr = (self.encode_2nd_end_seq * self.vt_2nd_end_mask.dimshuffle(0,1,'x')).sum(axis = 1) / self.vt_2nd_end_mask.sum(axis = 1, keepdims = True)
        '''============================================================================'''

        #build attention layer
        self.attention1_layer()
        self.attention2_layer()
        #build reasoning layers

        # merge tensors to fit in BLSTM models as input tensor
        # merge_ls1.shape = (n_batch, m_seq, self.rnn_units)
        self.merge_ls1 = [T.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1])) for tensor in self.attentioned_sent_rep1]
        self.merge_ls2 = [T.reshape(tensor, (tensor.shape[0], 1, tensor.shape[1])) for tensor in self.attentioned_sent_rep2]
        encode_merge1 = T.concatenate(self.merge_ls1, axis = 1)
        encode_merge2 = T.concatenate(self.merge_ls2, axis = 1)

        # create sentence level encoder
        # using only last encoding result of both direction of the LSTM
        self.sent_encoder = BLSTM_last.BlstmEncoder(INPUT_SIZE = self.rnn_units, LSTMLAYER_1_UNITS = self.rnn_units)
        self.sent_encoder.build_model()

        # reasoner_result1.shape = (n_batch, self.rnn_units)
        reasoner_result1 = lasagne.layers.get_output(self.sent_encoder.output, {self.sent_encoder.l_in: encode_merge1}, 
                                                    deterministic = True)

        reasoner_result2 = lasagne.layers.get_output(self.sent_encoder.output, {self.sent_encoder.l_in: encode_merge2}, 
                                                    deterministic = True)

        # self.train_encodinglayer_vecs[4].shape == (n_batch, m_seq, 300)
        end_attention_part1 = T.dot(self.train_encodinglayer_vecs[4], self.bilinear_attention_matrix)
        end_attention = T.batched_dot(end_attention_part1, reasoner_result1)


        end_att_numerator1 = self.inputs_masks[4] * T.exp(end_attention - end_attention.max(axis = 1, keepdims = True))
    
        end_attention_weight_matrix = end_att_numerator1 / end_att_numerator1.sum(axis = 1, keepdims = True)

        # end_attention_weight_matrix.shape = (n_batch, m_seq)
        # neg_ending_rep.shape == (n_batc, m_seq)
        # neg_ending_subs_index is the one to be substituted in the original sentence
        neg_ending_subs_index = T.argmax(end_attention_weight_matrix, axis = 1)
        n_batch = self.train_encodinglayer_vecs[0].shape[0]

        # substi_word_index is the random index generate to surrogate the original ones
        srng = T.shared_randomstreams.RandomStreams(seed=234)
        substi_word_index = (self.words_num * srng.uniform((n_batch, ))).astype('int64')

        batch_dim = T.arange(n_batch)
        # set_subtensor won't change the original tensor
        # new_end_seq_tensor.shape = (self.batchsize, m_seq, 300)
        new_end_seq_tensor = T.set_subtensor(self.train_encodinglayer_vecs[4][batch_dim, neg_ending_subs_index], self.wemb[substi_word_index])
        # neg_end_rep.shape = (self.batchsize, self.rnn_units)
        neg_end_rep = (new_end_seq_tensor * self.inputs_masks[4].dimshuffle(0,1,'x')).sum(axis = 1) / self.inputs_masks[4].sum(axis = 1, keepdims = True)

        l_story_in = lasagne.layers.InputLayer(shape=(None, self.rnn_units))
        l_end_in = lasagne.layers.InputLayer(shape = (None, self.rnn_units))
        l_concate = lasagne.layers.ConcatLayer([l_story_in, l_end_in], axis = 1)

        l_hid = lasagne.layers.DenseLayer(l_concate, num_units=2,
                                          nonlinearity=lasagne.nonlinearities.tanh)

        final_class_param = lasagne.layers.get_all_params(l_hid)
        answer = (2 * srng.uniform((n_batch,))).astype('int64')

        ending_pair_tensor1 = self.train_encodinglayer_vecs[-1] * (1-answer).dimshuffle(0,'x') + neg_end_rep * answer.dimshuffle(0,'x')
        ending_pair_tensor2 = self.train_encodinglayer_vecs[-1] * (answer).dimshuffle(0,'x') + neg_end_rep * (1-answer).dimshuffle(0,'x')
        

        score1 = lasagne.layers.get_output(l_hid, {l_story_in: reasoner_result1, 
                                                   l_end_in: ending_pair_tensor1})
        score2 = lasagne.layers.get_output(l_hid, {l_story_in: reasoner_result1, 
                                                   l_end_in: ending_pair_tensor2})

        vt_2nd_score = lasagne.layers.get_output(l_hid, {l_story_in: reasoner_result2, 
                                                   l_end_in: self.vt_2nd_end_repr})

        prob1 = lasagne.nonlinearities.softmax(score1)
        prob2 = lasagne.nonlinearities.softmax(score2)


        # Construct symbolic cost function
        
        cost1 = lasagne.objectives.categorical_crossentropy(prob1, answer)
        cost2 = lasagne.objectives.categorical_crossentropy(prob2, 1-answer)

        self.cost = lasagne.objectives.aggregate(cost1+cost2, mode='sum')

        # Retrieve all parameters from the network
        all_params = self.encoder.all_params + self.sent_encoder.all_params + final_class_param + [self.bilinear_attention_matrix]

        all_updates = lasagne.updates.adam(self.cost, all_params, learning_rate=0.001)
        # all_updates = lasagne.updates.momentum(self.cost, all_params, learning_rate = 0.05, momentum=0.9)

        self.train_func = theano.function(self.inputs_variables + self.inputs_masks, 
                                        [self.cost, prob1, prob2, answer], updates = all_updates)

        # Compute adam updates for training

        self.prediction = theano.function(self.inputs_variables + [self.vt_2nd_end] + self.inputs_masks + [self.vt_2nd_end_mask], [score1, vt_2nd_score])
        # pydotprint(self.train_func, './computational_graph.png')

    def load_data(self):
        train_set = pickle.load(open(self.train_set_path))
        self.train_story = train_set[0]
        self.train_ending = train_set[1]

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
            wemb = np.random.rand(self.words_num ,self.wemb_size)
            wemb = np.concatenate((np.zeros((1, self.wemb_size)), wemb), axis = 0)
            self.wemb = theano.shared(wemb).astype(theano.config.floatX)
        else:
            self.wemb = theano.shared(pickle.load(open(self.wemb_matrix_path))).astype(theano.config.floatX)

    def fake_load_data(self):
        self.train_story = []
        self.train_story.append(np.concatenate((np.random.randint(10, size = (50, 10)), 10+np.random.randint(10, size=(50,10))),axis=0).astype('int64'))
        self.train_story.append(np.ones((100, 10)).astype('int64'))
        self.train_story.append(np.ones((100, 10)).astype('int64'))
        self.train_story.append(np.ones((100, 10)).astype('int64'))
        self.train_story = np.asarray(self.train_story).reshape([100,4,10])

        self.train_ending = np.concatenate((2 * np.ones((50, 5)), np.ones((50, 5))), axis = 0)
        self.val_story = []
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))
        self.val_story.append(np.random.randint(10, size = (100, 10)).astype('int64'))

        self.val_ending1 = np.ones((100, 10)).astype('int64')
        self.val_ending2 = 2*np.ones((100, 10)).astype('int64')
        self.val_answer = np.zeros(100)
        self.n_val = self.val_answer.shape[0]

        self.test_story = []
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))
        self.test_story.append(np.random.randint(20, size = (100, 10)).astype('int64'))

        self.test_ending1 = np.ones((100, 10)).astype('int64')
        self.test_ending2 = 2*np.ones((100, 10)).astype('int64')
        self.test_answer = np.ones(100)
        self.n_test = self.test_answer.shape[0]


        self.wemb = theano.shared(np.random.rand(30, self.rnn_units)).astype(theano.config.floatX)


    def val_set_test(self):

        correct = 0.
        predict_seq = []

        for i in range(self.n_val):
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
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))
            # predict_seq.append(prediction)
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
            prediction = np.argmax(np.concatenate((score1, score2), axis=1))

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
        N_TRAIN_INS = len(self.train_ending)
        best_val_accuracy = 0
        best_test_accuracy = 0
        test_threshold = 10000/N_BATCH
        batch_count = 0.0
        start_batch = 0.0

        '''init test'''
        print "initial test..."
        val_result = self.val_set_test()
        print "accuracy is: ", val_result*100, "%"
        if val_result > best_val_accuracy:
            best_val_accuracy = val_result

        test_accuracy = self.test_set_test()
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
            total_err_count = 0.0

            for batch in range(max_batch):
                batch_index_list = [shuffled_index_list[i] for i in range(batch * N_BATCH, (batch+1) * N_BATCH)]
                train_story = [[self.train_story[index][i] for index in batch_index_list] for i in range(1,self.story_nsent+1)]
                train_ending = [self.train_ending[index] for index in batch_index_list]

                train_story_matrices = [utils.padding(batch_sent) for batch_sent in train_story]
                train_end_matrix = utils.padding(train_ending)
                # train_end2_matrix = utils.padding(end2)

                train_story_mask = [utils.mask_generator(batch_sent) for batch_sent in train_story]
                train_end_mask = utils.mask_generator(train_ending)
                # train_end2_mask = utils.mask_generator(end2)


                cost, prediction1, prediction2, answer = self.train_func(train_story_matrices[0], train_story_matrices[1], train_story_matrices[2],
                                                               train_story_matrices[3], train_end_matrix,
                                                               train_story_mask[0], train_story_mask[1], train_story_mask[2],
                                                               train_story_mask[3], train_end_mask)



                prediction = np.argmax(np.concatenate((prediction1, prediction2), axis = 1), axis = 1)
                predict_answer = np.zeros((N_BATCH, ))
                for i in range(N_BATCH):
                    if prediction[i] == 0:
                        predict_answer[i] = 1
                    elif prediction[i] == 1:
                        predict_answer[i] = 0
                    elif prediction[i] == 2:
                        predict_answer[i] = 0
                    else:
                        predict_answer[i] = 1

                total_err_count += (abs(predict_answer - answer)).sum()
                total_cost += cost

                if batch_count % test_threshold == 0 and batch_count != 0:
                    print "error rate on training set: ", (total_err_count * 1.0)/(test_threshold * N_BATCH)*100.0, "%"

                    print "test on val set..."
                    val_result = self.val_set_test()
                    print "accuracy is: ", val_result*100, "%"
                    if val_result > best_val_accuracy:
                        print "new best! test on test set..."
                        best_val_accuracy = val_result

                        test_accuracy = self.test_set_test()
                        print "test set accuracy: ", test_accuracy * 100, "%"
                        if test_accuracy > best_test_accuracy:
                            best_test_accuracy = test_accuracy

                    total_err_count = 0.0

                batch_count += 1

            print "======================================="
            print "epoch summary:"
            print "average cost in this epoch: ", total_cost
            print "average speed: ", N_TRAIN_INS/(time.time() - start_time), "instances/s "

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
    if len(argv) > 2:
        wemb_size = argv[2]
    model = Hierachi_RNN(argv[0], argv[1], wemb_size)

    print "loading data"
    model.load_data()

    print "model construction"
    model.model_constructor()
    print "construction complete"

    print "training begin!"
    model.begin_train()
    
if __name__ == '__main__':
    main(sys.argv[1:])


