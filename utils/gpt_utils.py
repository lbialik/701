from lib2to3.pgen2 import token
import numpy as np
import copy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def surprise(model, tokens_tensor, labels):
    loss=model(tokens_tensor, labels=labels)[0]
    return np.exp(loss.cpu().detach().numpy())

def print_top_x_predictions(next_token_logits, tokenizer):
    for word_idx, val in sorted(enumerate(next_token_logits[0]), key=lambda x:x[1], reverse=True)[:15]:
        print(f'{tokenizer.decode(word_idx)} --> {val.item()/torch.sum(next_token_logits)}')

def get_relative_clause_surprise(intro, relative_clause, model=None, dictionary=None):
    # print(f'\ninput: {intro} [{relative_clause}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    sentence = intro + ' ' + relative_clause
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_relative_clause = tokenizer.encode(' ' + relative_clause, return_tensors="pt")
    tokenized_sentence = tokenizer.encode(sentence, return_tensors="pt")
    masked_intro = torch.ones(len(tokenized_intro[0]), dtype=int) * -100
    masked_sentence = torch.cat((masked_intro, tokenized_relative_clause[0]))
    tokenized_labels = torch.reshape(masked_sentence, (1,-1))
    return surprise(model, tokenized_sentence, tokenized_labels)

# intro = "The bus driver that"
# relative_clause = "ignored the children"
# print(get_relative_clause_surprise(intro, relative_clause))