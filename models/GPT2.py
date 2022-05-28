from pprint import pprint
import token
import numpy as np
import copy
import torch
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import *
from utils.gpt_utils import *

data = super_average_data(process_data())

def run_gpt(intro, query):
    # print(f'\ninput: {intro} [{query}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_query = tokenizer.encode(' ' + query, return_tensors="pt")
    query_next_token_logits = []
    outputs = model(tokenized_intro)[0][:, -1, :][0]
    for query_token in tokenized_query[0]:
        # print(f"\nSequence so far: {' '.join([tokenizer.decode(token) for token in tokenized_intro])}")
        outputs = model(tokenized_intro)
        next_token_logits = outputs[0][:, -1, :]
        # query_token_surprise = surprisal_of_word_norm(query_token, next_token_logits.detach())
        # print(f'token surprise: {query_token_surprise}')
        query_next_token_logits.append(next_token_logits.detach())
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, query_token)]))
    query_surprise = surprisal_of_words_norm(tokenized_query[0], query_next_token_logits)
    # print('query suprise: ', query_surprise)
    return query_surprise