from lib2to3.pgen2 import token
import numpy as np
import copy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

raw = True 

def surprisal_of_word_norm(idx, next_token_logits):
    num = next_token_logits[0][idx]
    denom = torch.sum(next_token_logits)
    if raw:
        return num
    return float(num/denom)

def surprisal_of_words_norm(word_idxs, next_token_logit_list):
    num = 0
    denom = 0
    assert(len(word_idxs) == len(next_token_logit_list))
    for i, word_idx in enumerate(word_idxs):
        num += next_token_logit_list[i][0][word_idx]
        denom += torch.sum(next_token_logit_list[i])
    if raw:
        return float(num/len(word_idxs))
    return float(num/denom)

def print_top_x_predictions(next_token_logits, tokenizer):
    for word_idx, val in sorted(enumerate(next_token_logits[0]), key=lambda x:x[1], reverse=True)[:15]:
        if raw:
            print(f'{tokenizer.decode(word_idx)} --> {val.item()}')
        else:
            print(f'{tokenizer.decode(word_idx)} --> {val.item()/torch.sum(next_token_logits)}')

def get_relative_clause_surprise(intro, relative_clause, model=None, dictionary=None):
    # print(f'\ninput: {intro} [{relative_clause}]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
    tokenized_relative_clause = tokenizer.encode(' ' + relative_clause, return_tensors="pt")
    relative_clause_next_token_logits = []
    outputs = model(tokenized_intro)[0][:, -1, :][0]
    for relative_clause_token in tokenized_relative_clause[0]:
        # print(f"\nSequence so far: {' '.join([tokenizer.decode(token) for token in tokenized_intro])}")
        outputs = model(tokenized_intro)
        next_token_logits = outputs[0][:, -1, :]
        relative_clause_token_surprise = surprisal_of_word_norm(relative_clause_token, next_token_logits.detach())
        # print(f'token surprise: {relative_clause_token_surprise}')
        # print_top_x_predictions(next_token_logits, tokenizer)
        relative_clause_next_token_logits.append(next_token_logits.detach())
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, relative_clause_token)]))
    relative_clause_surprise = surprisal_of_words_norm(tokenized_relative_clause[0], relative_clause_next_token_logits)
    # print('relative_clause suprise: ', relative_clause_surprise)
    return relative_clause_surprise

def surprise(model, tokens_tensor, labels):
    loss=model(tokens_tensor, labels=labels)[0]
    return np.exp(loss.cpu().detach().numpy())

intro = "The bus driver that"
relative_clause = "drove the children home"
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
result = surprise(model, tokenized_sentence, tokenized_labels)
print(result)