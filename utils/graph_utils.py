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


def plot_bar_graph(labels, measurements, men_means, women_means):

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/4, men_means, width, label='Men')
    ax.bar(x - width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_title('Title')
    # ax.set_xticks(x, labels)
    ax.legend()
    plt.xticks(np.arange(len(labels)), labels)

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

labels = ['The bus driver \n that [the kids]', 'sentence 2', 'sentence 3', 'sentence 4', 'sentence 5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
measurements = []
plot_bar_graph(labels, measurements, men_means, women_means)


def plot_box_and_whisker(labels, data, title, save_location=None):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.boxplot(data)
    plt.xticks(np.arange(1, len(labels)+1), labels)
    plt.xlabel("Measure")
    plt.ylabel("Reading Rime (ms)")
    if save_location:
        plt.savefig(save_location)
    plt.show()

def print_box_and_whisker(segment_type):
    data = json.load(open("data/augmented_data.json"))
    measurements = [key for key in data["ORC"]["1"][segment_type].keys()]
    graph_data = []
    for measure in measurements:
        values = [data["SRC"][item][segment_type][measure]-data["ORC"][item][segment_type][measure] for item in data["ORC"]]
        graph_data.append(values if measure != "GPT2" else [val*20000000 for val in values])
    title = f"SRC - ORC ({segment_type})"
    save_location = f"plots/SRC-ORC({segment_type})_box_and_whisker"
    plot_box_and_whisker(measurements, graph_data, title, save_location)

# print_box_and_whisker("NP")
# print_box_and_whisker("verb")