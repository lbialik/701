from ast import parse
import json
import re 
from collections import defaultdict
from pprint import pprint

exp_1_path = "data/Experiment1/"
exp_2_path = "data/Experiment2/"

exp_1_overview = exp_1_path + "Experiment1/E4_del.txt"
exp_2_overview = exp_2_path + "E2.txt"

exp_1_condition_map = {'11': 'ORC', '12': 'ORC', '13': 'SRC', '14': 'SRC'}
exp_2_condition_map = {'1': 'ORC', '2': 'ORC', '3': 'ORC', '4': 'ORC', '5': 'SRC', '6': 'SRC'}

def parse_sentence(line):
    words = line.split()[2:]
    words = [word.replace('|', '').replace('^\\n', '') for word in words]
    sentence = ' '.join(words)
    return sentence

def extract_item_and_condition_from_trial(line, exp_number):
    condition_map = exp_1_condition_map if exp_number == 1 else exp_2_condition_map
    id = line.split()[1]
    _, condition, item, _ = re.split("[EID]", id)
    if condition in condition_map:
        condition = condition_map[condition]
        condition_valid = True
    else: condition_valid = False
    if exp_number == 1:
        item = str(int(item) - 120)
    return item, condition, condition_valid

def parse_experiment(exp_number):
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
                sentences[item][condition].append(sentence)
            if "end" in line and condition_valid and item_valid:
                condition, item = "", ""
    # pprint(sentences)
    # print(f'{len(sentences)} total items')
    return sentences

def convert_sentences_to_measurements(sentences, exp_path):
    measurements = {}
    measurement_types = ["ff", "fp", "gp", "tt"]
    for measurement in measurement_types:
        m_file = f"{exp_path}/ixs/{measurement}.ixs"
        measurements[measurement] = 0
    return measurements

def combine_sentences(sentences_collection):
    for sentences in sentences_collection:
        pass

def combine_measurements(measurements_collection):
    for measurements in measurements_collection:
        pass

def process_data():
    exp_1_sentences = parse_experiment(1)
    exp_2_sentences = parse_experiment(2)
    exp_1_measurements = convert_sentences_to_measurements(exp_1_sentences)
    exp_2_measurements = convert_sentences_to_measurements(exp_2_sentences)
    sentences = combine_sentences(exp_1_sentences, exp_2_sentences)
    measurements = combine_measurements(exp_1_measurements, exp_2_measurements)
    return sentences, measurements




## Code to be deleted

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
