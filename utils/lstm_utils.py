import sys
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
import torch.nn.functional as F

sys.path.append('colorlessgreenRNNs/src/language_models')
import dictionary_corpus
from model import RNNModel

torch.manual_seed(50360)
np.random.seed(50360)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_up_model():
    torch.cuda.empty_cache()
    # Load Pre-trained Model
    fn = 'colorlessgreenRNNs/src/hidden650_batch128_dropout0.2_lr20.0.pt'
    if torch.cuda.is_available():
        torch.device('cuda')
        model_ = torch.load(fn)
    else:
        torch.device('cpu')
        model_ = torch.load(fn, map_location=torch.device('cpu'))
    model = RNNModel("LSTM", 50001, 650, 650, 2, 0.2, False)
    model.load_state_dict(model_.state_dict())
    model = model.cuda()

    data_path = "colorlessgreenRNNs/data/lm/English"
    dictionary = dictionary_corpus.Dictionary(data_path)

    return model, dictionary

def tokenize(sentence, dictionary):
    sentence = sentence.split()
    tokenized_sentence = []
    for w in sentence:
        if w not in dictionary.word2idx:
            print(w, ' not in vocab!')
            tokenized_sentence.append(dictionary.word2idx["<unk>"])
        else:
            tokenized_sentence.append(dictionary.word2idx[w])
    return tokenized_sentence

def surprisal(token, next_token_prob):
    return -np.log(next_token_prob[token])

def get_query_surprise(intro, query, model, dictionary):
    tokenized_intro = tokenize(intro, dictionary)
    tokenized_query = tokenize(query, dictionary)
    query_token_surprisals = []
    for query_token in tokenized_query:
        input = torch.tensor(tokenized_intro, dtype=torch.long).cuda()
        cur_sentence_output, cur_sentence_hidden = model(input.view(-1, 1), # (sequence_length, batch_size).
                                                model.init_hidden(1)) # one input at a time, thus batch_size = 1
        next_word_scores = cur_sentence_output[-1].view(-1).cpu().detach()
        next_word_probs = F.softmax(next_word_scores, dim=0)
        token_surprisal = surprisal(query_token, next_word_probs)
        query_token_surprisals.append(token_surprisal)
        tokenized_intro.append(query_token)
    query_surprise = np.mean(query_token_surprisals)
    return query_surprise