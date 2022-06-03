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

def surprise(model, tokens_tensor, labels):
    probability = 0
    return -np.log(probability)

def get_query_surprise(intro, query, model, dictionary):
    sentence = intro + ' ' + query
    tokenized_sentence = tokenize(sentence, dictionary)
    tokenized_intro = tokenize(intro, dictionary)
    tokenized_query = tokenize(query, dictionary)
    tokenized_labels_masked = [-100] * len(tokenized_intro) + tokenized_query

    # lossfn = torch.nn.MSELoss
    # query_surprise = surprise(model, lossfn, tokenized_sentence, tokenized_sentence)

    input = torch.tensor(tokenized_sentence, dtype=float, requires_grad=True).long().cuda()
    lossfn = nn.NLLLoss(ignore_index = -100)
    model_outputs, _ = model(input.view(-1, 1), model.init_hidden(1))
    model_outputs = model_outputs.reshape(len(tokenized_sentence), -1).cuda()
    labels = torch.tensor(tokenized_labels_masked).long().cuda()
    query_surprise = lossfn(model_outputs, labels)

    print(query_surprise)

get_query_surprise("Hello my name", "is banana", *set_up_model())

# def old_code():
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
