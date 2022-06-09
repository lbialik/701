from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import json 
from scipy.stats.stats import pearsonr   

measurements = ['ff', 'fp', 'gp', "ro", 'tt']
lms = ['LSTM', 'GPT2']

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
    data = json.load(open("data/augmented_data.json"), )
    
    graph_data = []
    measures = [m for m in measurements + lms if m !=  "ro"]
    for measure in measures:
        values = [data["ORC"][item][segment_type][measure]-data["SRC"][item][segment_type][measure] for item in data["ORC"]]
        if measure in ["LSTM", "GPT2"]:
            values = np.array(values) * 100
        graph_data.append(values)
    title = f"ORC - SRC Difference ({segment_type})"
    save_location = f"plots/box_and_whisker/ORC-SRC({segment_type})_box_and_whisker"
    plot_box_and_whisker(measures, graph_data, title, save_location)


def plot_scatterplots(title, lm, x_data, y_data, plot_titles,  save_location=None):
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    plt.suptitle(title)
    axs[0].set_ylabel(f'{lm} Surprisal')
    for i in range(len(plot_titles)):
        axs[i].scatter(x_data[i], y_data)
        axs[i].set_title(plot_titles[i])
        axs[i].set_xlabel("Reading Time (ms)")
        #obtain m (slope) and b(intercept) of linear regression line
        m, b = np.polyfit(x_data[i], y_data, 1)
        correlation = np.round(pearsonr(x_data[i], y_data)[0], 3)
        axs[i].plot(x_data[i], m*np.array(x_data[i])+b, color='lightblue', label=f"correlation = {correlation}")
        axs[i].legend()

    if save_location:
        plt.savefig(save_location)
    plt.show()

def print_scatterplots(segment_type, lm):
    title = f"ORC-SRC Difference ({segment_type})"
    save_location = f"plots/scatter/ORC-SRC({segment_type})_scatter_{lm}"
    x_data, y_data = [], []
    data = json.load(open("data/augmented_data.json"))
    x_measurements = [measure for measure in measurements if (measure != lm)]
    items = data["ORC"]
    for measure in x_measurements:
        values = [data["ORC"][item][segment_type][measure]-data["SRC"][item][segment_type][measure] for item in items]
        x_data.append(values)
    y_data = [data["ORC"][item][segment_type][lm]-data["SRC"][item][segment_type][lm] for item in items]
    plot_scatterplots(title, lm, x_data, y_data, x_measurements, save_location=save_location)

def print_baseline_scatterplots(lm, clause_type):
    title = f"{clause_type} Human Reading Time and {lm} Surprisal"
    save_location = f"plots/scatter/{lm}_{clause_type}"
    x_data, y_data = [], []
    data = json.load(open("data/augmented_data.json"))
    x_measurements = [measure for measure in measurements if (measure != lm)]
    items = data[clause_type]
    for segment_type in ["NP", "verb"]:
        for measure in x_measurements:
            values = [data[clause_type][item][segment_type][measure] for item in items]
            x_data.append(values)
        y_data = [data[clause_type][item][segment_type][lm] for item in items]
    plot_scatterplots(title, lm, x_data, y_data, x_measurements, save_location=save_location)
    
if __name__ == "__main__":

    for lm in lms:
        for clause_type in ["SRC", "ORC"]:
            print_baseline_scatterplots(lm, clause_type)

    # for region in ["NP", "verb"]:
    #     print_box_and_whisker(region)
        
    # for region in ["NP", "verb"]:
    #     for lm in lms:
    #         print_scatterplots(region,lm)
