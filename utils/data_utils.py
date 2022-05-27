from ast import parse
from concurrent.futures import process
import json
import re 
from collections import defaultdict
from pprint import pprint
import numpy as np

exp_1_path = "data/Experiment1/Experiment1"
exp_2_path = "data/Experiment2"

exp_1_overview = exp_1_path + "/E4_del.txt"
exp_2_overview = exp_2_path + "/E2.txt"

exp_1_condition_map = {'11': 'ORC', '12': 'ORC', '13': 'SRC', '14': 'SRC'}
exp_2_condition_map = {'1': 'ORC', '2': 'ORC', '3': 'ORC', '4': 'ORC', '5': 'SRC', '6': 'SRC'}

ignore_items = ['25', '36']

measurement_types = ["ff", "fp", "gp", "tt"]


def NP_region(sentence_type):
    if sentence_type == "ORC":
        return 2
    elif sentence_type == "SRC":
        return 3
    else:
        raise Exception("no such sentence type")

def verb_region(sentence_type):
    if sentence_type == "ORC":
        return 3
    elif sentence_type == "SRC":
        return 2
    else:
        raise Exception("no such sentence type")

def split_sentence_on(sentence, index):
    split_sentence = sentence.split('^')
    beginning, middle, end = ''.join(split_sentence[:index]).strip(), split_sentence[index].strip(), ''.join(split_sentence[index+1:]).strip()
    return beginning, middle, end

def exp_condition_map(exp_number):
    return exp_1_condition_map if exp_number == 1 else exp_2_condition_map

def parse_sentence(line):
    words = line.split()[2:]
    words = [word.replace('|', '').replace('^\\n', '') for word in words]
    sentence = ' '.join(words)
    return sentence

def extract_item_and_condition_from_trial(line, exp_number):
    condition_map = exp_condition_map(exp_number)
    id = line.split()[1]
    _, condition, item, _ = re.split("[EID]", id)
    if condition in condition_map:
        condition_valid = True
    else: condition_valid = False
    if exp_number == 1:
        item = str(int(item) - 120)
    return item, condition, condition_valid

def extract_sentences(exp_number):
    sentences = defaultdict(lambda: defaultdict(lambda: []))
    file = exp_1_overview if exp_number == 1 else exp_2_overview
    def is_item_valid(item):
        return len(item) > 0 and item not in ignore_items

    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if "trial E" in line and "D0" in line:
                item, condition, valid = extract_item_and_condition_from_trial(line, exp_number)
                condition_valid = valid
            item_valid = is_item_valid(item)
            if "inline" in line and condition_valid and item_valid:
                sentence = parse_sentence(line)
                sentences[item][condition] = sentence
            if "end" in line and condition_valid and item_valid:
                condition, item = "", ""
    return sentences

def parse_measurement_file(m_file):
    measurements = defaultdict(lambda:[])
    with open(m_file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.replace('\n', '')
            _,_,item,condition,r1,r2,r3,r4,r5,r6,r7,r8 = line.split(',')
            if int(condition) > 0:
                times = [int(time) if len(time) > 0 else np.nan for time in [r1,r2,r3,r4,r5,r6,r7,r8]]
                measurements[(item, condition)].append(times)
    return measurements

def extract_measurements(exp_number):
    exp_path = exp_1_path if exp_number == 1 else exp_2_path
    measurements = {}
    for measurement in measurement_types:
        m_file = f"{exp_path}/ixs/{measurement}.ixs"
        measurements[measurement] = parse_measurement_file(m_file)
    return measurements

def same_sentence(sentence1, sentence2):
    regions1 = sentence1.split('^')
    regions2 = sentence2.split('^')
    # ignore all differences after R4
    if regions1[:5] == regions2[:5]:
        return True
    return False

def map_sentence(sentence, unique_sentences):
    for other_sentence in unique_sentences:
        if same_sentence(sentence, other_sentence):
            return other_sentence
    return sentence

def process_data():
    data = {
        'ORC': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))), 
        'SRC': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
        }
    unique_sentences = []
    for exp_number in [1, 2]:
        sentences = extract_sentences(exp_number)
        measurements = extract_measurements(exp_number)
        condition_map = exp_condition_map(exp_number)
        for item in sentences:
            for condition in sentences[item]:
                general_condition = condition_map[condition]
                sentence = sentences[item][condition]
                sentence = map_sentence(sentence, unique_sentences)
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
                for measurement in measurement_types:
                    data[general_condition][item][sentence][measurement] = measurements[measurement][(item, condition)]
    return data

def avg(values):
    return np.round(np.nanmean(values, 0), decimals = 3).tolist()

def average_data(data):
    avg_data = {
        'ORC': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: int)))), 
        'SRC': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: int))))
        }
    for condition in data:
        for item in data[condition]:
            for sentence in data[condition][item]:
                print(sentence)
                sentence_measurements = data[condition][item][sentence]
                for measure in sentence_measurements:
                    avg_measurements = avg(sentence_measurements[measure])
                    NP_measure = NP_region(condition)
                    verb_measure = verb_region(condition)
                    avg_data[condition][item][sentence]["NP"][measure] = avg_measurements[NP_measure]
                    avg_data[condition][item][sentence]["verb"][measure] = avg_measurements[verb_measure]
    return avg_data

def super_average_data(data):
    avg_data = {
        'ORC': defaultdict(lambda: defaultdict(lambda: int)), 
        'SRC': defaultdict(lambda: defaultdict(lambda: int))
        }
    for condition in data:
        for item in data[condition]:
            avg_measures = defaultdict(lambda: [])
            example_sentence = [key for key in data[condition][item].keys()][0]
            for sentence in data[condition][item]:
                sentence_measurements = data[condition][item][sentence]
                for measure in sentence_measurements:
                    avg_measures[measure].append((avg(sentence_measurements[measure])))
            avg_data[condition][item]['NP'] = {}
            avg_data[condition][item]['verb'] = {}
            avg_data[condition][item]['example_sentence'] = example_sentence
            for measurement in avg_measures:
                NP_measure = avg(avg_measures[measurement])[NP_region(condition)]
                verb_measure = avg(avg_measures[measurement])[verb_region(condition)]
                avg_data[condition][item]['NP'][measurement] = NP_measure
                avg_data[condition][item]['verb'][measurement] = verb_measure
    return avg_data

# data = process_data()
# avg_data = average_data(data)
# super_avg_data = super_average_data(data)




## Example Usage:

# for condition in avg_data:
#     print('condition: ', condition)
#     for item in avg_data[condition]:
#         print('item: ', item)
#         for sentence in avg_data[condition][item]:
#             print('sentence: ', sentence)
#             for measure in avg_data[condition][item][sentence]:
#                 print('measure: ', measure, ' = ', avg_data[condition][item][sentence][measure])
#         print()

# sentences = []
# count = 0
# for condition in avg_data:
#         for item in avg_data[condition]:
#             for sentence_element in avg_data[condition][item]:
#                 if sentence_element['sentence'] not in sentences:
#                     sentences.append(sentence_element['sentence'])
#                 count+=1
# print(f'{len(sentences)} unique sentences, {count} sentences total')

# exp_1_sentences = extract_sentences(1)
# exp_2_sentences = extract_sentences(2)
# print(len([exp_1_sentences[item][condition] for item in exp_1_sentences for condition in exp_1_sentences[item]]))
# print(len([exp_2_sentences[item][condition] for item in exp_2_sentences for condition in exp_2_sentences[item]]))
