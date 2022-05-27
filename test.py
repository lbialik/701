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

    def test_surprisal_of_word_norm(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        intro = "Hello my name is "
        model.eval()
        tokenized_intro = tokenizer.encode(intro, return_tensors="pt")
        query_next_token_logits = []

        total_surprisal = 0
        outputs = model(tokenized_intro)
        for query_token in range(len(list(outputs))):
            outputs = model(tokenized_intro)
            next_token_logits = outputs[0][:, -1, :]
            # query_token_surprise = surprisal_of_word_norm(query_token, next_token_logits.detach())
            query_next_token_logits.append(next_token_logits.detach())
            total_surprisal += surprisal_of_word_norm(query_token, next_token_logits.detach())
        self.assertAlmostEqual(total_surprisal, 1.)


if __name__ == '__main__':
    unittest.main()

