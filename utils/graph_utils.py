from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import json 

measurements = ['ff', 'fp', 'gp', "ro", 'tt', 'GPT2']

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
    title = "ORC - SRC ({segment_type})"
    labels = ['The bus driver \n that [the kids]', 'sentence 2', 'sentence 3', 'sentence 4', 'sentence 5']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    plot_bar_graph(labels, title, measurements, men_means, women_means)


def plot_box_and_whisker(labels, data, title, save_location=None):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.boxplot(data)
    plt.xticks(np.arange(1, len(labels)+1), labels)
    plt.xlabel("Measure")
    plt.ylabel("Reading Time (ms)")
    if save_location:
        plt.savefig(save_location)
    plt.show()

def print_box_and_whisker(segment_type):
    data = json.load(open("data/augmented_data.json"))
    # measurements = [key for key in data["ORC"]["1"][segment_type].keys() if key != "ro"]
    # print(measurements)
    graph_data = []
    for measure in measurements:
        values = [data["ORC"][item][segment_type][measure]-data["SRC"][item][segment_type][measure] for item in data["ORC"]]
        graph_data.append(values if measure != "GPT2" else [val*20000000 for val in values])
    title = f"ORC - SRC ({segment_type})"
    save_location = f"plots/ORC-SRC({segment_type})_box_and_whisker"
    plot_box_and_whisker(measurements, graph_data, title, save_location)


def plot_scatterplots(title, lm, x_data, y_data, plot_titles,  save_location=None):
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    plt.suptitle(title)
    axs[0].set_ylabel(f'{lm} Surprisal Difference')
    axs[0].set_xlabel('Reading Time Difference (ms)')
    for i in range(len(plot_titles)):
        axs[i].scatter(x_data[i], y_data)
        axs[i].set_title(plot_titles[i])
        axs[i].set_xlabel("Reading Time Difference (ms)")

    if save_location:
        plt.savefig(save_location)
    plt.show()

def print_scatterplots(segment_type):
    title = f"ORC-SRC ({segment_type})"
    lm = "GPT2"
    save_location = f"plots/ORC-SRC({segment_type})_scatter"

    x_data, y_data = [], []
    data = json.load(open("data/augmented_data.json"))
    x_measurements = [measure for measure in measurements if measure != 'GPT2']
    # items = [item for item in data["ORC"].keys() if item != "14"]
    items = data["ORC"]
    for measure in x_measurements:
        values = [data["ORC"][item][segment_type][measure]-data["SRC"][item][segment_type][measure] for item in items]
        x_data.append(values)
    y_data = [data["ORC"][item][segment_type][lm]-data["SRC"][item][segment_type][lm] for item in items]
    plot_scatterplots(title, lm, x_data, y_data, x_measurements, save_location=save_location)
    
if __name__ == "__main__":
    print_box_and_whisker("NP")
    print_box_and_whisker("verb")
    print_bar_graph("NP")
    print_bar_graph("verb")
    print_scatterplots("NP")
    print_scatterplots("verb")