from lib2to3.pgen2.tokenize import TokenError
from pprint import pprint
import token
from data_utils import *
import numpy as np
import copy
import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

data = super_average_data(process_data())

def score(model, tokens_tensor):
    loss=model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())


def run_gpt(intro, query):
    print(f'{intro} [{query}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_query = tokenizer.encode(' ' + query, return_tensors="pt")
    # print('tokenized intro: ', tokenized_intro)
    # print('tokenized query: ', tokenized_query)
    result = 0
    for query_token in tokenized_query[0]:
        # print('query token: ', query_token)
        # print('intro shape = ', tokenized_intro.shape)
        # print('query shape = ', query_token.shape)
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, query_token)]))
        print('new tokenized intro: ', tokenized_intro)
        outputs = model(tokenized_intro)
        next_token_logits = outputs[0][:, -1, :]
        pred_id = torch.argmax(next_token_logits).item()
        pred_word = tokenizer.decode(pred_id)
        print("\n Sequence so far: ")
        print(' '.join([tokenizer.decode(token) for token in tokenized_intro]))
        print("\nPredicted next word for sequence: ")
        print(pred_word)
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