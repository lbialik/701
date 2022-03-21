import json
import os
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint

#local imports
import sys
sys.path.append('../../../')
import utils as data_utils

# Load the data
f = open('../../../data/sentences.json')
data = json.load(f)
sentences = data_utils.load_sentences(data)

torch.manual_seed(50360)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(50360)

model_ = None
fn = "../hidden650_batch128_dropout0.2_lr20.0.pt"
with open(fn, "rb") as model_f:
    model_ = torch.load(fn)
    
# Makes sure your model is loaded onto the GPU (should return true)
next(model_.parameters()).is_cuda

# Finally - load the model.
from model import RNNModel

# Construct a new RNNModel using PyTorch 1.x implementations of NN modules
model = RNNModel("LSTM", 50001, 650, 650, 2, 0.2, False)
# Copy over the trained weights from the model loaded in
model.load_state_dict(model_.state_dict())

model = model.cuda()

## Import dictionary_corpus, part of the colorlessgreenRNNs repoistory that has some use useful functions
import dictionary_corpus

## Find the English dictionary, and call it 'dictionary' 
data_path = "../data/lm/English"
dictionary = dictionary_corpus.Dictionary(data_path)

## Examples of uses you can make use of later 
print("Vocab size is ", len(dictionary))

dictionary.word2idx["horse"]
dictionary.idx2word[3619]