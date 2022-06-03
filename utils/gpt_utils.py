from lib2to3.pgen2 import token
import numpy as np
import copy
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def print_top_x_predictions(next_token_logits, tokenizer):
    for word_idx, val in sorted(enumerate(next_token_logits[0]), key=lambda x:x[1], reverse=True)[:15]:
        print(f'{tokenizer.decode(word_idx)} --> {val.item()/torch.sum(next_token_logits)}')

def surprisal(token, next_token_prob):
    return -np.log(next_token_prob[token])
    
def get_query_surprisal(intro, query, model=None, dictionary=None):
    # print(f'\ninput: {intro} [{query}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_query = tokenizer.encode(' ' + query, return_tensors="pt")
    query_token_surprisals = []
    for query_token in tokenized_query[0]:
        outputs = model(tokenized_intro)
        next_token_logits = outputs[0][:, -1, :][0].detach()
        next_token_probabilities = F.softmax(next_token_logits, dim=0)
        token_surprisal = surprisal(query_token, next_token_probabilities.detach())
        query_token_surprisals.append(token_surprisal)
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, query_token)]))
    query_surprisal = np.mean(query_token_surprisals)
    return query_surprisal

get_query_surprisal('Hello my name is', 'Rachel Elizabeth')