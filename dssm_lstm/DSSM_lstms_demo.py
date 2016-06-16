import sys
import cPickle as pickle
import numpy as np 

encoder_names = {1:"BLSTM-meanpool", 2:"LSTM-meanpool", 3:"LSTM-last-timestep"}
encoder_number = None
shared = None

print "Please choose the encoder type number:"
for k,v in encoder_names.items():
    print k,v
encoder_choose = sys.stdin.readline()
try:
    encoder_number = int(encoder_choose)
except:
    print "invalid number"

print "Shared first layer?(yes/no)"
shared_input = sys.stdin.readline().rstrip().lower()
if shared_input == "yes":
    shared = True
elif shared_input == "no":
    shared = False
else:
    print "alright...I only know 'yes' or 'no', my bad..."



def model_init(encoder_num, shared_first_layer):
    if encoder_num == 1:
        if shared_first_layer:
            import DSSM_blstm_1neg_hidden300_sharewemblayer
            return DSSM_blstm_1neg_hidden300_sharewemblayer.DSSM_BLSTM_Model()
        else:
            import DSSM_blstm_1neg_hidden300
            return DSSM_blstm_1neg_hidden300.DSSM_BLSTM_Model()
    elif encoder_num == 2:
        if shared_first_layer:
            import DSSM_lstm_1neg_hidden300_sharewemblayer
            return DSSM_lstm_1neg_hidden300_sharewemblayer.DSSM_LSTM_Model()
        else:
            import DSSM_lstm_1neg_hidden300
            return DSSM_lstm_1neg_hidden300.DSSM_LSTM_Model()
    elif encoder_num == 3:
        if shared_first_layer:
            import DSSM_lstm_last_1neg_hidden300_sharewemblayer
            return DSSM_lstm_last_1neg_hidden300_sharewemblayer.DSSM_LSTM_Model()
        else:
            import DSSM_lstm_last_1neg_hidden300
            return DSSM_lstm_last_1neg_hidden300.DSSM_LSTM_Model()


model = model_init(encoder_number, shared)

print "model construction..."
model.model_constructor(wemb_matrix_path='../../data/pickles/index_wemb_matrix.pkl')
print "construction complete"

print "loading parameters..."
model.reload_model("val")

word2index_dict = pickle.load(open('../../data/pickles/ROC_train_vocab_dict.pkl','r'))

def convert2index(sent):
    '''
    parameters: 
    -----------
    sent ==> type: list of strings(words)

    return:
    -----------
    tokens ==> type: list of ints(index of word in dictionary)
    unknown_words_ls ==> type: list of strings(unknown words)
    '''
    tokens = []
    unknown_words_ls = []
    for word in sent:
        if word in word2index_dict:
            tokens.append(word2index_dict[word])
        else:
            tokens.append(word2index_dict['UUUNKNOWNNN'])
            unknown_words_ls.append(word)
    return tokens, unknown_words_ls



while 1:

    print "please enter the story you have"
    print "please use space as delimitor :) "
    try:
        story = sys.stdin.readline()
        story_words = [word.lower() for word in story.split()]
        story_tokens, unknowns = convert2index(story_words)
        print "unknown words in story: ", " ".join(unknowns)

    except KeyboardInterrupt:
        print "Ended by user"
        break

    print "please enter the first end you have"
    try:
        end1 = sys.stdin.readline()
        end1_words = [word.lower() for word in end1.split()]
        end1_tokens, unknowns = convert2index(end1_words)
        print "unknown words in end1: ", " ".join(unknowns)

    except KeyboardInterrupt:
        print "Ended by user"
        break

    print "please enter the second end you have"
    try:
        end2 = sys.stdin.readline()
        end2_words = [word.lower() for word in end2.split()]
        end2_tokens, unknowns = convert2index(end2_words)
        print "unknown words in end2: ", " ".join(unknowns)

    except KeyboardInterrupt:
        print "Ended by user"
        break

    story_input = np.asarray(story_tokens, dtype='int64').reshape((1,-1))
    story_mask = np.ones((1,len(story_tokens)))

    ending1 = np.asarray(end1_tokens, dtype='int64').reshape((1,-1))
    ending1_mask = np.ones((1,len(end1_tokens)))

    ending2 = np.asarray(end2_tokens, dtype='int64').reshape((1,-1))
    ending2_mask = np.ones((1, len(end2_tokens)))

    cos1 = model.compute_cost(story_input, story_mask, ending1, ending1_mask)
    cos2 = model.compute_cost(story_input, story_mask, ending2, ending2_mask)

    # Answer denotes the index of the anwer
    print "reasoning probability of these two ending with the story are:"
    print cos1
    print cos2

