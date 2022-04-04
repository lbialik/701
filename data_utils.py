import json
import re 
from collections import defaultdict
from pprint import pprint

exp_1_path = "data/Experiment1/"
exp_2_path = "data/Experiment2/"

exp_1_overview = exp_1_path + "Experiment1/E4_del.txt"
exp_2_overview = exp_2_path + "E2.txt"

# exp = exp_1_path
# measurements = ["ff", "fp", "gp", "tt"]
# for measurement in measurements:
#     m_file = f"{exp}/ixs/{measurement}.ixs"

sentences = defaultdict(lambda: defaultdict(lambda: []))

# for item in sentences:
#     for condition in sentences[item]:
#         print(sentences[item][condition])


def parse_sentence(line):
    words = line.split()[2:]
    words = [word.replace('|', '').replace('^\\n', '') for word in words]
    sentence = ' '.join(words)
    # sentence_components = sentence.split('^')
    # return sentence_components\
    return sentence

def parse_experiment_1():

    condition_map = {'11': 'ORC', '12': 'ORC', '13': 'SRC', '14': 'SRC'}
    reverse_condition_map = {'ORC': ['11', '12'], 'SRC': ['13', '14']}

    with open(exp_1_overview) as f:
        lines = f.readlines()
        for line in lines:
            if "trial E1" in line and "D0" in line:
                id = line.split()[1]
                _, condition, item, _ = re.split("[EID]", id)
                item = str(int(item) - 120)
                condition = condition_map[condition]
            if "inline" in line:
                sentence = parse_sentence(line)
                sentences[item][condition].append(sentence)
            if "end" in line:
                condition, item = "", ""
    pprint(sentences)
    print(f'{len(sentences)} total items')


parse_experiment_1()

# def parse_experiment_2():
#     with open(exp_2_overview) as f:
#         lines = f.readlines()
#         for line in lines:
#             if "trial E" in line and "D0" in line:
#                 id = line.split()[1]
#                 _, condition, item, _ = re.split("[EID]", id)
#                 print('id = ', id)
#                 print('condition = ', condition)
#                 print('item = ', item)

# parse_experiment_2()


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
