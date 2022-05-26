from pprint import pprint
import numpy as np
import copy
import torch
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import *
from utils.gpt_utils import *

data = super_average_data(process_data())

def run_gpt(intro, query):
    print(f'\ninput: {intro} [{query}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_query = tokenizer.encode(' ' + query, return_tensors="pt")
    query_next_token_logits = []

    import itertools
    sanity_check = 0
    new_intro = tokenized_intro
    outputs = model(new_intro)
    print(outputs[0][:, -1, :].shape[1])
    all_token_tuples = itertools.permutations(range(outputs[0][:, -1, :].shape[1]), 1)
    # print('all token tuples: ', [i for i in all_token_tuples])
    for token_tuple in all_token_tuples:
        new_intro = tokenized_intro
        query_next_token_logits = []
        for query_token in token_tuple:
            print(f"\nSequence so far: {' '.join([tokenizer.decode(token) for token in new_intro])}")
            outputs = model(new_intro)
            next_token_logits = outputs[0][:, -1, :]
            # query_token_surprise = surprisal_of_word_norm(query_token, next_token_logits.detach())
            query_next_token_logits.append(next_token_logits.detach())
            new_intro = torch.tensor(np.array([np.append(new_intro, query_token)]))
        print(token_tuple, tokenizer.decode(token_tuple[0]), query_next_token_logits)
        print(surprisal_of_words_norm(token_tuple, query_next_token_logits))
        sanity_check += surprisal_of_words_norm(token_tuple, query_next_token_logits)
    print('sanity check should be 1: ', sanity_check)

    # for query_token in tokenized_query[0]:
    #     print(f"\nSequence so far: {' '.join([tokenizer.decode(token) for token in tokenized_intro])}")
    #     outputs = model(tokenized_intro)
    #     next_token_logits = outputs[0][:, -1, :]
    #     # query_token_surprise = surprisal_of_word_norm(query_token, next_token_logits.detach())
    #     query_next_token_logits.append(next_token_logits.detach())
    #     tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, query_token)]))
    # query_surprise = surprisal_of_words_norm(tokenized_query[0], query_next_token_logits)
    # print('query suprise: ', query_surprise)
    # return query_surprise

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
            break
        break
    return new_data

data = add_LM_measures('GPT2')
# pprint(data)