import torch

# def score(model, tokens_tensor):
#     loss=model(tokens_tensor, labels=tokens_tensor)[0]
#     return np.exp(loss.cpu().detach().numpy())

def surprisal_of_word_norm(idx, next_token_logits):
    num = next_token_logits[0][idx]
    denom = torch.sum(next_token_logits)
    return float(num/denom)

def surprisal_of_words_norm(word_idxs, next_token_logit_list):
    num = 0
    denom = 0
    assert(len(word_idxs) == len(next_token_logit_list))
    for i, word_idx in enumerate(word_idxs):
        num += next_token_logit_list[i][0][word_idx]
        denom += torch.sum(next_token_logit_list[i])
    return float(num/denom)

def print_top_x_predictions(next_token_logits, tokenizer):
    for word_idx, val in sorted(enumerate(next_token_logits[0]), key=lambda x:x[1], reverse=True)[:15]:
        print(f'{tokenizer.decode(word_idx)} --> {val.item()}')