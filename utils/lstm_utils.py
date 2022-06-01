import json
# import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
import torch.nn.functional as F
# from utils.data_utils import *

sys.path.append('colorlessgreenRNNs/src/language_models')
import dictionary_corpus
from model import RNNModel

torch.manual_seed(50360)
np.random.seed(50360)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def surprise(model, lossfn, tokens_tensor, labels):
    # model(input.view(-1, 1), model.init_hidden(1))
    loss= lossfn(model(tokens_tensor), labels)
    return np.exp(loss.cpu().detach().numpy())

def get_query_surprise(intro, query, model, dictionary):
    sentence = intro + ' ' + query
    intro, query = intro.split(), query.split()
    for w in sentence:
        if w not in dictionary.word2idx:
            print(w, ' not in vocab!')

    tokenized_intro = [dictionary.word2idx[w]  if w in dictionary.word2idx
                                        else dictionary.word2idx["<unk>"]
                    for w in intro]
    tokenized_query = [dictionary.word2idx[w]  if w in dictionary.word2idx
                                        else dictionary.word2idx["<unk>"]
                    for w in query]
    tokenized_sentence = [dictionary.word2idx[w]  if w in dictionary.word2idx
                                        else dictionary.word2idx["<unk>"]
                    for w in sentence]

    masked_intro = torch.ones(len(tokenized_intro), dtype=int) * -100
    masked_sentence = torch.cat((masked_intro, tokenized_query))
    tokenized_labels = torch.reshape(masked_sentence, (1,-1))
    lossfn = torch.nn.MSELoss
    query_surprise = surprise(model, lossfn, tokenized_sentence, tokenized_labels)
    print(query_surprise)

    # query_next_token_scores = []
    # for query_token in tokenized_query:
    #     # print(f'intro: {[dictionary.idx2word[w] for w in tokenized_intro]}')
    #     # print(f'query: {dictionary.idx2word[query_token]}')

    #     input = torch.tensor(tokenized_intro, dtype=torch.long).cuda()

    #     ## Extract the hidden and output layers at each input token:
    #     cur_sentence_output, cur_sentence_hidden = model(input.view(-1, 1), # (sequence_length, batch_size).
    #                         model.init_hidden(1)) # one input at a time, thus batch_size = 1
    #     next_word_scores = cur_sentence_output[-1].view(-1)
        
    #     if raw:
    #         min_score, max_score = min(next_word_scores), max(next_word_scores) 
    #         if min_score < 0:
    #             # shift all scores to be positive
    #             next_word_scores = next_word_scores - min_score
    #     else:
    #         next_word_scores = F.softmax(next_word_scores, dim=0)
    #     query_next_token_scores.append(next_word_scores.detach())
    #     tokenized_intro.append(query_token)
    # query_surprise = surprisal_of_words_norm(tokenized_query, query_next_token_scores)
    # return query_surprise

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

get_query_surprise("Hello my name", "is banana", *set_up_model())