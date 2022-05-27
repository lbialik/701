import matplotlib.pyplot as plt
import numpy as np
import json 

def ORC_minus_SRC():
    differences = {}
    for condition in data:
        # print('condition: ', condition)
        for item in data[condition]:
            # print('item: ', item)
            sentence = data[condition][item]['example_sentence']
            np_beginning, np_middle, np_end = split_sentence_on(sentence, NP_region(condition))
            v_beginning, v_middle, v_end = split_sentence_on(sentence, verb_region(condition))
            new_data[condition][item]['NP']['GPT2'] = run_model(np_beginning, np_middle)
            new_data[condition][item]['verb']['GPT2'] = run_model(v_beginning, v_middle)


def plot_bar_graph(labels, measurements, ):

    for measure in measurements:
        pass

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_means, width, label='Men')
    rects2 = ax.bar(x + width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]


def plot_box_and_whisker(labels, data, title):
    fig, ax = plt.subplots()
    plt.title(title)
    bp = ax.boxplot(data)
    plt.xticks([1, 2, 3, 4], labels)
    
    # show plot
    plt.show()

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)

data = [data_1, data_2, data_3, data_4]
labels = ["label1", "label2", "label3", "label4"]

data = json.load(open("data/augmented_data.json"))
measurements = [key for key in data["ORC"]["1"]["NP"].keys()]
ORCs = {}
for measure in measurements:
    ORCs[measure] = [data["ORC"][item]["NP"][measure] for item in data["ORC"]]
from pprint import pprint
pprint(ORCs)
# labels = measurements
# title = "ORC - SRC (NP)"
# plot_box_and_whisker(labels, data, title)