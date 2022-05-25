from calendar import c
from lib2to3.pgen2.tokenize import TokenError
from pprint import pprint
import token
from data_utils import *
import numpy as np
import copy
import torch
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

data = super_average_data(process_data())

# def score(model, tokens_tensor):
#     loss=model(tokens_tensor, labels=tokens_tensor)[0]
#     return np.exp(loss.cpu().detach().numpy())

def prob_of_word_norm(idx, next_token_logits):
    return next_token_logits[0][idx]/torch.sum(next_token_logits)

def prob_of_words_norm(word_idxs, next_token_logit_list):
    num = 0
    denom = 0
    print(word_idxs)
    print(next_token_logit_list)
    assert(len(word_idxs) == next_token_logit_list.shape()[0])
    for i, word_idx in enumerate(word_idxs):
        num += next_token_logit_list[i][0][word_idx]
        denom += torch.sum(next_token_logit_list[i])
    return num/denom

def run_gpt(intro, query):
    print(f'\ninput: {intro} [{query}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_query = tokenizer.encode(' ' + query, return_tensors="pt")
    # print('tokenized intro: ', tokenized_intro)
    # print('tokenized query: ', tokenized_query)
    query_pred_idxs = []
    query_next_token_logits = []
    for query_token in tokenized_query[0]:
        # print('query token: ', query_token)
        # print('intro shape = ', tokenized_intro.shape)
        # print('query shape = ', query_token.shape)
        outputs = model(tokenized_intro)
        next_token_logits = outputs[0][:, -1, :]
        pred_id = torch.argmax(next_token_logits).item()
        pred_word = tokenizer.decode(pred_id)
        sanity_check = 0
        for word_idx in next_token_logits[0]:
            print(word_idx)
            print(next_token_logits.shape)
            sanity_check += prob_of_word_norm(word_idx, next_token_logits)
        print(sanity_check)
        # print(f"Sequence so far: {' '.join([tokenizer.decode(token) for token in tokenized_intro])}")
        # print(f"Predicted next word for sequence: {pred_word}")
        # print('prediction prob = ', prob_of_word_norm(pred_id, next_token_logits.detach()))
        # print(f'real next token prob = ', prob_of_word_norm(query_token, next_token_logits.detach()))
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, query_token)]))
        
    return 1

def add_LM_measures(lang_model):
    new_data = copy.deepcopy(data)
    for condition in data:
        # print('condition: ', condition)
        for item in data[condition]:
            # print('item: ', item)
            sentence = data[condition][item]['example_sentence']
            np_beginning, np_middle, np_end = split_sentence_on(sentence, NP_region(condition))
            v_beginning, v_middle, v_end = split_sentence_on(sentence, verb_region(condition))
            new_data[condition][item]['measurements'][lang_model+'_NP'] = run_gpt(np_beginning, np_middle)
            new_data[condition][item]['measurements'][lang_model+'_verb'] = run_gpt(v_beginning, v_middle)
    return new_data

data = add_LM_measures('GPT2')
data = add_LM_measures('LSTM')
# pprint(data)