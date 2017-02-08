'''
Author: Jon Tsai    
Created: May 29 2016
'''

import numpy as np 
import theano
from time import sleep
import sys
import cPickle as pickle

def progress_bar(percent, speed):
    i = int(percent)/2
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-50s] %d%% %f instances/s" % ('='*i, percent, speed))
    sys.stdout.flush()
    


def combine_sents_string(sent_set):
    '''
    parameter: sent_set ==> 2D sentences set
                        ==> type: list[list[str]]

    return: sents1D ==> 1D sentences set
                    ==> type: list[str]
    '''
    return [" ".join(doc) for doc in sent_set]
        


def combine_sents(sent_set):
    '''
    parameter: sent_set ==> 2D sentences set
                        ==> type: list[list[list]]

    return: sents1D ==> 1D sentences set
                    ==> type: list[list]

    This function will combine 2D sentence set 
    into 1D sentence set. 
    e.g.
    [
        [[sent1], [sent2], [sent3], ..., [sentn]]
        ...
        [[sent1], [sent2], [sent3], ..., [sentn]]
    ]
    ==> 
    [
        [sentences1],
        ...
        [sentencesn]
    ]
    '''
    sents1D = []
    for doc in sent_set:
        combine_sent = np.array([])
        for sent in doc:
            combine_sent = np.concatenate((combine_sent,sent))
        sents1D.append(combine_sent)

    return sents1D

def shuffle_index(length_of_indices_ls):
    '''
    ----------
    parameter: 
    ----------
    length_of_indices_ls: type = int 

    ----------
    return: 
    ----------
    a shuffled numpy array of indices 
    '''
    ls = np.arange(length_of_indices_ls)
    np.random.shuffle(ls)
    return ls

def padding(batch_input_list):
    '''
    ----------
    parameter: 
    ----------
    batch_input_list: type = list(list) 

    ----------
    return: 
    ----------
    numpy.ndarray: shape == (n_batch, max_time_step) 
    '''
    n_batch = len(batch_input_list)
    max_time_step = max([len(batch_input_list[i]) for i in range(n_batch)])

    padding_result = np.zeros((n_batch, max_time_step))
    for batch in range(n_batch):
        padding_result[batch] = np.concatenate((np.asarray(batch_input_list[batch]),
                                                np.zeros(max_time_step - len(batch_input_list[batch]))))
    return padding_result.astype('int64')



def mask_generator(indices_matrix):
    '''
    ----------
    parameter: 
    ----------
    indices_matrix: type = list[list] 

    ----------
    return: 
    ----------
    mask : type = np.ndarray
    a mask matrix of a batch of varied length instances
    '''

    n_batch = len(indices_matrix)
    len_ls = [len(sent) for sent in indices_matrix]
    max_len = max(len_ls)
    mask = np.zeros((n_batch, max_len))
    for i in range(n_batch):
        for j in range(len(indices_matrix[i])):
            mask[i][j] = 1 

    return mask

def mlp_mask_generator(indices_matrix, wemb_size):
    '''
    ----------
    parameter: 
    ----------
    indices_matrix: type = list[list] 

    ----------
    return: 
    ----------
    mask : type = np.ndarray
           mask.shape = (n_batch, wemb_size)
    '''

    n_batch = len(indices_matrix)
    len_ls = [len(sent) for sent in indices_matrix]
    
    mask = np.ones((n_batch, wemb_size))
    for i in range(n_batch):
        mask[i] = mask[i] * len_ls[i]

    return mask

def fake_input_generator(max_index, batch_number, length_range):
    '''
    ----------
    parameter: 
    ----------
    max_index: type = int 
    batch_number: type = int 
    length_range: tuple(int), len(length_range) = 2 
                  e.g. (50, 70)

    ----------
    return: 
    ----------
    fake_data: type = list[list]
               format: fake_data.shape[0] = batch_number
                       length_range[0] <= len(fake_data[i]) <= length_range[1]
                       0 <= fake_data[i][j] <= max_index
    '''    
    max_time_step = length_range[0] + np.random.randint(length_range[1] - length_range[0] + 1)
    
    fake_data = np.zeros((batch_number, max_time_step))
    
    mask = np.zeros((batch_number, max_time_step)).astype(theano.config.floatX)

    len_range = max_time_step - length_range[0]
    assert len_range >= 0
    #pick a row to be the max length row
    row = np.random.randint(batch_number)
    fake_data[row] = np.random.randint(max_index+1, size = (max_time_step,))
    mask[row] = np.ones(max_time_step)

    for batch in range(batch_number):
        if batch == row:
            continue
        length = length_range[0]+np.random.randint(len_range)

        fake_data[batch] = np.concatenate((np.random.randint(max_index+1 ,size = (length,)), 
                                       np.zeros(max_time_step - length)))
        mask[batch] = np.concatenate((np.ones(length), np.zeros(max_time_step - length)))

    return (fake_data.astype('int32'), mask)

def fake_data(max_index, batch_number, max_time_step, min_time_step):
    
    fake_data = np.zeros((batch_number, max_time_step))
    
    mask = np.zeros((batch_number, max_time_step)).astype(theano.config.floatX)

    len_range = max_time_step - min_time_step
    assert len_range >= 0
    #pick a row to be the max length row
    row = np.random.randint(batch_number)
    fake_data[row] = np.random.randint(max_index+1, size = (max_time_step,))
    mask[row] = np.ones(max_time_step)

    for batch in range(batch_number):
        if batch == row:
            continue
        length = min_time_step+np.random.randint(len_range)

        fake_data[batch] = np.concatenate((np.random.randint(max_index+1 ,size = (length,)), 
                                       np.zeros(max_time_step - length)))
        mask[batch] = np.concatenate((np.ones(length), np.zeros(max_time_step - length)))

    return (fake_data.astype('int32'), mask)

class Ngram_generator(object):
    """docstring for Ngram_generator"""
    def __init__(self, N_GRAM):
        super(Ngram_generator, self).__init__()
        self.N_GRAM = N_GRAM
        self.n_gram_dict = None
        self.entries_num = None

        with open('../../data/pickles/'+self.N_GRAM+'_dict.pkl','r') as f:
            self.n_gram_dict = pickle.load(f)
            self.entries_num = len(self.n_gram_dict)



    def trigram_generator(self, sentence_batch):
        """sentence_batch: 1D list"""
        batch_matrix = []
        for sent in sentence_batch:
            sent = "#"+sent+"#"
            sent_vec = np.zeros(self.entries_num)
            for i in range(len(sent)-2):
                try: 
                    sent_vec[self.n_gram_dict[(sent[i:i+3]).lower()]] += 1.0 
                except:
                    continue
            batch_matrix.append(sent_vec)
        return batch_matrix

    def letter_trigram_generator(self, sentence_batch):
        """sentence_batch: 1D list"""
        batch_matrix = []
        for sent in sentence_batch:
            words = sent.split(' ')
            sent_vec = np.zeros(self.entries_num)
            for word in words:
                tag_word = '#'+word.lower()+'#'
                for i in range(len(tag_word) - 2):
                    try: 
                        sent_vec[self.n_gram_dict[tag_word[i:i+3]]] += 1.0 
                    except:
                        continue
            batch_matrix.append(sent_vec)
        return batch_matrix



        