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
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def train_model(clause_type, region_type):

    ## Model set up ##
    epochs = 200000
    input_dim = 1 
    output_dim = 1
    learning_rate = 0.01
    model = LogisticRegression(input_dim,output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ## Data set up ##
    data = json.load(open('data/augmented_data.json'))
    # total_datapoints = 136
    # train_size, val_size, test_size = 95, 20, 21
    total_datapoints = 34
    train_size, val_size, test_size = 24, 10, 0
    generator = torch.Generator().manual_seed(42)

    lang_model_measures = []
    ground_truth_regressions = []
    for item in data[clause_type]:
        lang_model_measures.append(data[clause_type][item][region_type]['GPT2'])
        # lang_model_measures.append(data[clause_type][item][region_type]['LSTM'])
        # lang_model_measures.append(data[clause_type][item][region_type]['ro'])
        ground_truth_regressions.append(data[clause_type][item][region_type]['ro'])

    measures = list(zip(lang_model_measures, ground_truth_regressions))
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

def plot(model, title):
    with torch.no_grad(): 
        if torch.cuda.is_available():
            predicted = model(Variable(model.X_val).cuda()).cpu().data.numpy()
        else:
            predicted = model(Variable(model.X_val)).data.numpy()

    plt.clf()
    plt.plot(model.X_val, model.y_val, 'go', label='True data', alpha=0.5)
    plt.plot(model.X_val, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

clause_type = "ORC"
region_type = "NP"
model = train_model(clause_type, region_type)
plot(model, f"Predicting {clause_type} {region_type} Regression Values from Language Model Surprisal")