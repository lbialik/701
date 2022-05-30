import numpy as np
import copy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# def score(model, tokens_tensor):
#     loss=model(tokens_tensor, labels=tokens_tensor)[0]
#     return np.exp(loss.cpu().detach().numpy())

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

def get_query_surprise(intro, query, model=None, dictionary=None):
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
        query_token_surprise = surprisal_of_word_norm(query_token, next_token_logits.detach())
        # print(f'token surprise: {query_token_surprise}')
        # print_top_x_predictions(next_token_logits, tokenizer)
        query_next_token_logits.append(next_token_logits.detach())
        tokenized_intro = torch.tensor(np.array([np.append(tokenized_intro, query_token)]))
    query_surprise = surprisal_of_words_norm(tokenized_query[0], query_next_token_logits)
    # print('query suprise: ', query_surprise)
    return query_surprise

# run_gpt("The mathematician that the chairman", "fired")