import matplotlib.pyplot as plt
import numpy as np
import json 

def plot_bar_graph(labels, title, measurements, men_means, women_means):

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width*2, men_means, width, label=measurements[0])
    ax.bar(x - width, women_means, width, label=measurements[1])
    ax.bar(x, women_means, width, label=measurements[2])
    ax.bar(x + width, women_means, width, label=measurements[3])
    ax.bar(x + width*2, women_means, width, label=measurements[4])

    ax.set_xlabel('Item')
    ax.set_ylabel('Reading Times')
    ax.set_title(title)
    ax.legend()
    plt.xticks(np.arange(len(labels)), labels)
    fig.tight_layout()
    plt.show()

def print_bar_graph(segment_type):
    title = "SRC - ORC ({segment_type})"
    labels = ['The bus driver \n that [the kids]', 'sentence 2', 'sentence 3', 'sentence 4', 'sentence 5']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    measurements = ['ff', 'fp', 'gp', 'tt', 'GPT2']
    plot_bar_graph(labels, title, measurements, men_means, women_means)


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
print(print_bar_graph("NP"))
