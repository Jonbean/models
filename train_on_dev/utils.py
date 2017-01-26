'''
Author: Jon Tsai    
Created: May 29 2016
'''

import numpy as np 
import theano
from time import sleep
import sys

def progress_bar(percent, speed):
    i = int(percent)/2
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-50s] %d%% %f instances/s" % ('='*i, percent, speed))
    sys.stdout.flush()
    



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

def number_with_units(word):
    index = 0
    if len(word) <= 1:
        return -1
    for c in word:
        if c.isdigit() or c == ',':
            continue
        else:
            index = word.index(c)
            break
    for c in word[index:]:
        if c.isalpha():
            continue
        else:
            return -1
    return index

#recursive function to split surfix punctuation and possession problem
def split_word(word):
    bad_end = ("#",'"','?','!','+','.',',',"'", '(',')',':','/',';','$','-','`','*','\\','%','&',']')
    bad_end_double = ('oz','lb','am','pm','th','ft')
    bad_end_triple = ('lbs','mph')
    bad_middle = ("-",",",".",":",'/',"*","\\",'&',"'",')','(','!')
    non_char = '\xef\xbf\xbd'
    single_abriev = ("'s","'d","'m")
    double_abriev = ("'ve", "'ll", "'re","n't")
    
    if word.lower() in glove_300_dict:
        return [word]
    elif word.startswith(bad_end):
        return [word[0]] + split_word(word[1:])
    elif word.endswith(bad_end):
        return split_word(word[:-1]) + [word[-1]]
    elif word.endswith(single_abriev+bad_end_double):
        return split_word(word[:-2]) + [word[-2:]]
    elif word.endswith(double_abriev+bad_end_triple):
        return split_word(word[:-3]) + [word[-3:]]
    elif any((c in bad_middle) for c in word.lower()[1:-1]):
        for ch in bad_middle:
            if ch in word.lower()[1:-1]:
                char_index = word.index(ch)
                return split_word(word[:char_index+1]) + split_word(word[char_index+1:])
    elif non_char in word.lower():
        char_index = word.index(non_char)
        return split_word(word.lower()[:char_index])+["?"]+split_word(word.lower()[char_index])
    elif number_with_units(word.lower()) != -1:
        index = number_with_units(word.lower())
        return [word[:index]] + [word[index:]]
    else:
        return [word]
    
#convert word list into tokens
def split_sent(word_list):
    new_word_ls = []
    for word in word_list:
        new_word_ls += split_word(word)
    return new_word_ls

def tokenization(story_list):
    tokenized_set = []
    for story in story_list:
        story_list = []
        for sent in story:
            word_list = sent.split(' ')
            new_word_list = split_sent(word_list)
            story_list.append(new_word_list)
        tokenized_set.append(story_list)
    return tokenized_set


def sent2index(sent):
    sent_index_ls = [word2index_dict['<s>']]
    for word in sent:
        if word.lower() in word2index_dict:
            sent_index_ls.append(word2index_dict[word.lower()])
        elif word.lower().isdigit():
            sent_index_ls.append(word2index_dict['UNKNOWN_NUM'])
        else: 
            sent_index_ls.append(word2index_dict['UUNKNOWNN'])
    sent_index_ls.append(word2index_dict['</s>'])
    return sent_index_ls