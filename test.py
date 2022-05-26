from utils.data_utils import *
from utils.gpt_utils import *
import unittest
import itertools
from pprint import pprint
import numpy as np
import copy
import torch
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class TestGPT(unittest.TestCase):
    data = super_average_data(process_data())

    def test_upper(self):
        intro = "Hello my name is "
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
        query_next_token_logits = []

        total_surprisal = 0
        new_intro = tokenized_intro
        outputs = model(new_intro)
        # print(outputs[0][:, -1, :].shape[1])
        all_token_tuples = itertools.permutations(range(outputs[0][:, -1, :].shape[1]), 1)
        # print('all token tuples: ', [i for i in all_token_tuples])
        for token_tuple in all_token_tuples:
            print(f'token tuple 0 out of {len(list(all_token_tuples))}', end = '')
            new_intro = tokenized_intro
            query_next_token_logits = []
            for query_token in token_tuple:
                # print(f"\nSequence so far: {' '.join([tokenizer.decode(token) for token in new_intro])}")
                outputs = model(new_intro)
                next_token_logits = outputs[0][:, -1, :]
                # query_token_surprise = surprisal_of_word_norm(query_token, next_token_logits.detach())
                query_next_token_logits.append(next_token_logits.detach())
                new_intro = torch.tensor(np.array([np.append(new_intro, query_token)]))
            # print(token_tuple, tokenizer.decode(token_tuple[0]), query_next_token_logits)
            # print(surprisal_of_words_norm(token_tuple, query_next_token_logits))
            total_surprisal += surprisal_of_words_norm(token_tuple, query_next_token_logits)
        self.assertAlmostEqual(total_surprisal, 1.)


if __name__ == '__main__':
    unittest.main()

