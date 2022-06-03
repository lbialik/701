from lib2to3.pgen2 import token
import numpy as np
import copy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def surprisal(model, tokens_tensor, labels):
    probability = model(tokens_tensor, labels=labels)[0]
    

def print_top_x_predictions(next_token_logits, tokenizer):
    for word_idx, val in sorted(enumerate(next_token_logits[0]), key=lambda x:x[1], reverse=True)[:15]:
        print(f'{tokenizer.decode(word_idx)} --> {val.item()/torch.sum(next_token_logits)}')

def get_query_surprise(intro, query, model=None, dictionary=None):
    # print(f'\ninput: {intro} [{query}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    sentence = intro + ' ' + query
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_query = tokenizer.encode(' ' + query, return_tensors="pt")
    tokenized_sentence = tokenizer.encode(sentence, return_tensors="pt")
    masked_intro = torch.ones(len(tokenized_intro[0]), dtype=int) * -100
    masked_sentence = torch.cat((masked_intro, tokenized_query[0]))
    tokenized_labels = torch.reshape(masked_sentence, (1,-1))
    return surprise(model, tokenized_sentence, tokenized_labels)

def old_code():
    relative_clause_next_token_logits = []
    outputs = model(tokenized_intro)[0][:, -1, :][0]
    for relative_clause_token in tokenized_relative_clause[0]:
        # print(f"\nSequence so far: {' '.join([tokenizer.decode(token) for token in tokenized_intro])}")
        outputs = model(tokenized_intro)
        next_token_logits = outputs[0][:, -1, :]
        relative_clause_token_surprise = surprisal(relative_clause_token, next_token_logits.detach())
        # print(f'token surprise: {relative_clause_token_surprise}')
        # print_top_x_predictions(next_token_logits, tokenizer)
        relative_clause_next_token_logits.append(next_token_logits.detach())
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, relative_clause_token)]))
    relative_clause_surprise = surprisal(tokenized_relative_clause[0], relative_clause_next_token_logits)
    # print('relative_clause suprise: ', relative_clause_surprise)
    return relative_clause_surprise

def old_code_surprise(word_idxs, next_token_logit_list):
    assert(len(word_idxs) == len(next_token_logit_list))
    for i, word_idx in enumerate(word_idxs):
        num += next_token_logit_list[i][0][word_idx]
        denom += torch.sum(next_token_logit_list[i])