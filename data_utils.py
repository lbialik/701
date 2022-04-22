from ast import parse
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

measurement_types = ["ff", "fp", "gp", "tt"]

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
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if "trial E" in line and "D0" in line:
                item, condition, valid = extract_item_and_condition_from_trial(line, exp_number)
                condition_valid = valid
            item_valid = len(item) > 0
            if "inline" in line and condition_valid and item_valid:
                sentence = parse_sentence(line)
                sentences[item][condition] = sentence
            if "end" in line and condition_valid and item_valid:
                condition, item = "", ""
    # pprint(sentences)
    # print(f'{len(sentences)} total items')
    return sentences

def parse_measurement_file(m_file):
    measurements = defaultdict(lambda:[])
    with open(m_file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.replace('\n', '')
            _,_,item,condition,r1,r2,r3,r4,r5,r6,r7,r8 = line.split(',')
            if int(condition) > 0:
                times = [int(time) if len(time)>0 else -1 for time in [r1,r2,r3,r4,r5,r6,r7,r8]]
                measurements[(item, condition)].append(times)
    return measurements

def extract_measurements(exp_number):
    exp_path = exp_1_path if exp_number == 1 else exp_2_path
    measurements = {}
    for measurement in measurement_types:
        m_file = f"{exp_path}/ixs/{measurement}.ixs"
        measurements[measurement] = parse_measurement_file(m_file)
    return measurements

def process_data():
    data = {
        'ORC': defaultdict(lambda: []), 
        'SRC': defaultdict(lambda: [])
        }
    for exp_number in [1, 2]:
        sentences = extract_sentences(exp_number)
        measurements = extract_measurements(exp_number)
        condition_map = exp_condition_map(exp_number)
        for item in sentences:
            for condition in sentences[item]:
                general_condition = condition_map[condition]
                element = {}
                sentence = sentences[item][condition]
                # if sentence in [e["sentence"] for e in data[general_condition][item]]:
                #     print('sentence already included')
                #     print(sentence) 
                #     print(item, condition)
                element['sentence'] = sentence
                for measurement in measurement_types:
                    element[measurement] = measurements[measurement][(item, condition)]
                data[general_condition][item].append(element)
    return data

def average_data(data):
    avg_data = {
        'ORC': defaultdict(lambda: []), 
        'SRC': defaultdict(lambda: [])
        }
    for condition in data:
        for item in data[condition]:
            for element in data[condition][item]:
                for measure in measurement_types:
                    element[measure] = np.round(np.mean(element[measure], 0), decimals = 3).tolist()
                avg_data[condition][item].append(element)
    return avg_data

data = process_data()
avg_data = average_data(data)
# pprint(avg_data)

sentences = []
count = 0
for condition in avg_data:
        for item in avg_data[condition]:
            for element in avg_data[condition][item]:
                if element['sentence'] not in sentences:
                    sentences.append(element['sentence'])
                count+=1
print(f'{len(sentences)} unique sentences, {count} sentences total')

exp_1_sentences = extract_sentences(1)
exp_2_sentences = extract_sentences(2)
print(len([exp_1_sentences[item][condition] for item in exp_1_sentences for condition in exp_1_sentences[item]]))
print(len([exp_2_sentences[item][condition] for item in exp_2_sentences for condition in exp_2_sentences[item]]))




## Code to be deleted

# exp_1_sentences = extract_sentences(1)
# exp_2_sentences = extract_sentences(2)
# for item in exp_1_sentences:
#     ## vanilla ORC
#     one = exp_1_sentences[item]['11']
#     two = exp_2_sentences[item]['1']
#     if one != two:
#         print('item ', item)
#         print('vanilla ORC doesn\'t match')
#         print("exp 1 sentence = ", one)
#         print("exp 2 sentence = ", two)
#         print()
#     ## ORC + adv
#     one = exp_1_sentences[item]['12']
#     two = exp_2_sentences[item]['3']
#     if one != two:
#         print('item ', item)
#         print('ORC + adv doesn\'t match')
#         print("exp 1 sentence = ", one)
#         print("exp 2 sentence = ", two)
#         print()
#     ## vanilla SRC
#     one = exp_1_sentences[item]['13']
#     two = exp_2_sentences[item]['5']
#     if one != two:
#         print('item ', item)
#         print('SRC doesn\'t match')
#         print("exp 1 sentence = ", one)
#         print("exp 2 sentence = ", two)
#         print()
#     ## SRC + adv
#     one = exp_1_sentences[item]['14']
#     two = exp_2_sentences[item]['6']
#     if one != two:
#         print('item ', item)
#         print('SRC + adv doesn\'t match')
#         print("exp 1 sentence = ", one)
#         print("exp 2 sentence = ", two)
#         print()


# for condition in data:
#     print('condition = ', condition)
#     for item in data[condition]:
#         print('item = ', item)
#         for element in data[condition][item]:
#             print('sentence = ', element['sentence'])
            # print(len(element['ff']))
            # print(len(element['fp']))
            # print(len(element['gp']))
            # print(len(element['tt']))


# def load_sentences(data):
#     '''Takes a JSON object as input and returns a dictionary e.g. {'1': 'ORC': 'sentence'}'''
#     sentences = {}
#     for sentence_number in data:
#         sentences[sentence_number] = {}
#         s = data[sentence_number]

#         orc = s["subject"] + s["clausal noun"] + s["clausal verb"] + s["verb phrase"] + "."
#         orc_adv = s["subject"] + s["clausal noun"] + s["clausal verb"] + s["clausal adverb"] + s["verb phrase"] + "."
#         orc_clausal_verb = s["subject"] + s["clausal noun"] + s["clausal phrasal verb"] + s["verb phrase"] + "."
#         orc_clausal_verb_adv = s["subject"] + s["clausal noun"] + s["clausal phrasal verb"] + s["clausal adverb"] + s["verb phrase"] + "."
#         src = s["subject"] + s["clausal verb"] + s["clausal noun"] + s["verb phrase"] + "."
#         src_adv = s["subject"] + s["clausal verb"] + s["clausal noun"] + s["clausal adverb"] + s["verb phrase"] + "."

#         sentences[sentence_number]['ORC'] = orc
#         sentences[sentence_number]['ORC adv'] = orc_adv
#         sentences[sentence_number]['ORC clausal verb'] = orc_clausal_verb
#         sentences[sentence_number]['ORC clausal verb adverb'] = orc_clausal_verb_adv
#         sentences[sentence_number]['SRC'] = src
#         sentences[sentence_number]['SRC adverb'] = src_adv

#     return sentences
