from pprint import pprint
import token
import numpy as np
import copy
import torch
import os
import argparse
# import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from utils.data_utils import *
import utils.gpt_utils as gpt_utils
import utils.lstm_utils as lstm_utils
# from models.LSTM import run_lstm

# process human reading data
fname = 'data/augmented_data.json'
if os.path.isfile(fname):
    data = json.load(open('data/augmented_data.json'))
else:
    data = super_average_data(process_data())

with open(fname, "w+") as f:
    json.dump(data, f)

def add_model_measures(model_type, data):
    if model_type == "GPT2":
        model, dictionary = None, None
        run_model = gpt_utils.get_query_surprise
    elif model_type == "LSTM":
        model, dictionary = lstm_utils.set_up_model()
        run_model = lstm_utils.get_query_surprise
    else:
        raise Exception("invalid model type")

    print(f'running {model_type} model')
    total = len(data) * len(data['ORC'])
    current = 0
    new_data = copy.deepcopy(data)
    for condition in data:
        # print('condition: ', condition)
        for item in data[condition]:
            print(f'{current}/{total}', end='\r')
            current += 1
            # print('item: ', item)
            sentence = data[condition][item]['example_sentence']
            np_beginning, np_middle, np_end = split_sentence_on(sentence, NP_region(condition))
            v_beginning, v_middle, v_end = split_sentence_on(sentence, verb_region(condition))
            new_data[condition][item]['NP'][model_type] = run_model(np_beginning, np_middle, model, dictionary)
            new_data[condition][item]['verb'][model_type] = run_model(v_beginning, v_middle, model, dictionary)
    return new_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('models', nargs='*', default=['GPT2', 'LSTM'],
                        help='an integer for the accumulator')
    args = parser.parse_args()

    if 'GPT2' in args.models:
        # add GPT2 to data
        data = add_model_measures("GPT2", data)
        with open('data/augmented_data.json', "w+") as f:
            json.dump(data, f)

    if 'LSTM' in args.models:
        # add LSTM to data
        data = add_model_measures("LSTM", data)
        with open('data/augmented_data.json', "w+") as f:
            json.dump(data, f)