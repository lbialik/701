from cgi import test
from functools import total_ordering
from xml.etree.ElementTree import C14NWriterTarget
import numpy as np
from utils.data_utils import *
from tqdm import tqdm
import torch
# from torch import *
import matplotlib.pyplot as plt
from torch.autograd import Variable

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # outputs = torch.sigmoid(self.linear(x))
        outputs = self.linear(x)
        return outputs

def train_model(clause_type=None, region_type=None, lm=None, measure=None):

    ## Model set up ##
    epochs = 80000
    input_dim = 1 
    output_dim = 1
    learning_rate = 0.001
    model = LogisticRegression(input_dim,output_dim)
    criterion = torch.nn.MSELoss()
    model.criterion = criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ## Data set up ##
    data = json.load(open('data/augmented_data.json'))
    
    generator = torch.Generator().manual_seed(42)

    lang_model_measures = []
    ground_truth_measures = []
    if clause_type and region_type:
        total_datapoints = 34
        train_size, val_size, test_size = 20, 7, 7
        for item in data[clause_type]:
            lang_model_measures.append(data[clause_type][item][region_type][lm])
            ground_truth_measures.append(data[clause_type][item][region_type][measure])
    else: 
        total_datapoints = 136
        train_size, val_size, test_size = 85, 25, 26
        for clause_type in data:
            for item in data[clause_type]:
                for region_type in data[clause_type][item]:
                    if region_type != "example_sentence":
                        lang_model_measures.append(data[clause_type][item][region_type][lm])
                        ground_truth_measures.append(data[clause_type][item][region_type][measure])

    measures = list(zip(lang_model_measures, ground_truth_measures))
    train, val, test = torch.utils.data.random_split(measures, [train_size, val_size, test_size], generator=generator)
    model.X_train = torch.tensor([[item[0]] for item in train])
    model.y_train = torch.tensor([item[1] for item in train])
    model.X_val = torch.tensor([[item[0]] for item in val])
    model.y_val = torch.tensor([item[1] for item in val])
    model.X_test = torch.tensor([[item[0]] for item in test])
    model.y_test = torch.tensor([item[1] for item in test])

    ## training
    iter = 0
    for epoch in tqdm(range(0,int(epochs)),desc='Training Epochs'):
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(model.X_train).squeeze()
        loss = criterion(outputs, model.y_train)
        loss.backward()
        optimizer.step()

        iter+=1
        if iter%10000==0:
            # calculate Error
            with torch.no_grad():
                outputs_val = torch.squeeze(model(model.X_val))
                loss_val = criterion(outputs_val, model.y_val)
                
                rmse_train = np.sqrt(loss.numpy())
                rmse_val = np.sqrt(loss_val.numpy())
                
                print(f"Iteration: {iter}. \nValidation - Loss: {loss_val.item()}. RMSE: {rmse_val}")
                print(f"Train -  Loss: {loss.item()}. RMSE: {rmse_train}\n")
        
            # for i in range(len(y_val)):
            #     print(f"ground truth: {y_val[i]} --> predicted: {torch.squeeze(model(X_val)).detach()[i]}")
    return model

def plot(model, measure_name, lm,  title):
    with torch.no_grad(): 
        if torch.cuda.is_available():
            predicted = model(Variable(model.X_test).cuda()).cpu().data.numpy()
        else:
            predicted = model(Variable(model.X_test)).data.numpy()
    
    outputs = torch.squeeze(model(model.X_test))
    loss = model.criterion(outputs, model.y_test)
    rmse = np.sqrt(loss.detach().numpy())

    plt.clf()
    plt.plot(model.X_test, model.y_test, 'go', label='True data', alpha=0.5)
    plt.plot(model.X_test, predicted, '--', label=f'Predictions\nRMSE={rmse:.4f}', alpha=0.5)
    plt.xlabel(f'{lm} Surprisal')
    plt.ylabel(f'{measure_name}')
    plt.legend(loc='best')
    plt.title(title)
    fname = title.replace(' ', '_')
    plt.savefig(f'plots/prediction/{fname}')
    plt.show()


full_measure_name = {
    "ro": "First Pass Regressions",
    "ff": "First Fixations",
    "tt": "Total Time"
}

if __name__ == "__main__":
    for lm in ["LSTM", "GPT2"]:
        for measure in ["ro", "ff", "tt"]:
            model = train_model(lm=lm, measure=measure)
            measure_name = full_measure_name[measure]
            plot(model, measure_name, lm, f"Predicting {measure_name} from {lm} Surprisal")

    # for lm in ["LSTM", "GPT2"]:
    #     for clause_type in ["ORC", "SRC"]:
    #         for region_type in ["NP", "verb"]:
    #             model = train_model(clause_type, region_type, lm)
    #             plot(model, f"Predicting {clause_type} {region_type} Regression Values from {lm} Surprisal")