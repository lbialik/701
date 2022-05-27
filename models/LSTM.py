import json
import copy
from pprint import pprint
from utils.data_utils import *
from colorlessgreenRNNs import *

data = json.load(open("data/augmented_data2.json"))

# for condition in data:
#     print('condition: ', condition)
#     for item in data[condition]:
#         print('item: ', item)
#         print('sentence: ', data[condition][item]['example_sentence'])
#         for measure in data[condition][item]['NP']:
#             print(f"NP {measure}: {data[condition][item]['NP'][measure]}")
#         for measure in data[condition][item]['verb']:
#             print(f"verb {measure}: {data[condition][item]['verb'][measure]}")
#         print()

def run_lstm(intro, query):
    query_surprise = 0
    return query_surprise
