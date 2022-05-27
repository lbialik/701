from pprint import pprint
import token
import numpy as np
import copy
import torch
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import *
from utils.gpt_utils import *
from GPT2 import add_GPT2_measures
from LSTM import add_LSTM_measures

# process human reading data
data = super_average_data(process_data())
with open('data/augmented_data.json', "w+") as f:
    json.dump(data, f)

# get GPT2 data
data = add_GPT2_measures(data)
with open('data/augmented_data.json', "w+") as f:
    json.dump(data, f)

# get LSTM data
data = add_LSTM_measures(data)
with open('data/augmented_data.json', "w+") as f:
    json.dump(data, f)