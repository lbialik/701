from pprint import pprint
import token
import numpy as np
import copy
import torch
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import *
from utils.gpt_utils import *
from models.GPT2 import run_gpt
from models.LSTM import run_lstm

# process human reading data
data = super_average_data(process_data())
with open('data/augmented_data.json', "w+") as f:
    json.dump(data, f)

def add_model_measures(model_type, data):
    if model_type == "GPT2":
        run_model = run_gpt
    elif model_type == "LSTM":
        run_model = run_lstm
    else:
        raise Exception("invalid model type")

    new_data = copy.deepcopy(data)
    for condition in data:
        # print('condition: ', condition)
        for item in data[condition]:
            # print('item: ', item)
            sentence = data[condition][item]['example_sentence']
            np_beginning, np_middle, np_end = split_sentence_on(sentence, NP_region(condition))
            v_beginning, v_middle, v_end = split_sentence_on(sentence, verb_region(condition))
            new_data[condition][item]['NP']['GPT2'] = run_model(np_beginning, np_middle)
            new_data[condition][item]['verb']['GPT2'] = run_model(v_beginning, v_middle)
    return new_data

# get GPT2 data
data = add_model_measures("GPT2", data)
with open('data/augmented_data.json', "w+") as f:
    json.dump(data, f)

# # get LSTM data
# data = add_LSTM_measures("LSTM", data)
# with open('data/augmented_data.json', "w+") as f:
#     json.dump(data, f)