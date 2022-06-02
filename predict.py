from cgi import test
import numpy as np
from utils.data_utils import *
from tqdm import tqdm
import torch
# from torch import *
import matplotlib.pyplot as plt


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


## Data set up ##
data = json.load(open('data/augmented_data.json'))
total_datapoints = 136
train_size, val_size, test_size = 95, 20, 21
generator = torch.Generator().manual_seed(42)

regression_threshold = .3
GPT2_measures = []
regressions = []
for clause_type in data:
    for item in data[clause_type]:
        for region_type in ["NP", "verb"]:
            GPT2_measures.append(data[clause_type][item][region_type]['GPT2'])
            regressions.append(1. if data[clause_type][item][region_type]['ro'] > regression_threshold else 0.)

measures = list(zip(GPT2_measures, regressions))
train, val, test = torch.utils.data.random_split(measures, [train_size, val_size, test_size], generator=generator)
X_train = torch.tensor([[item[0]] for item in train])
y_train = torch.tensor([item[1] for item in train])
X_val = torch.tensor([[item[0]] for item in val])
y_val = torch.tensor([item[1] for item in val])
X_test = torch.tensor([[item[0]] for item in test])
y_test = torch.tensor([item[1] for item in test])
print(y_train)


## Model set up ##
epochs = 10000
input_dim = 1 # single dimension input
output_dim = 1 # Two possible outputs
learning_rate = 0.01
model = LogisticRegression(input_dim,output_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
losses_val = []
Iterations = []
iter = 0
for epoch in tqdm(range(0,int(epochs)),desc='Training Epochs'):
    x = X_train
    labels = y_train
    optimizer.zero_grad() # Setting our stored gradients equal to zero
    outputs = model(X_train)
    loss = criterion(torch.squeeze(outputs), labels) # [200,1] -squeeze-> [200]
    loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
    optimizer.step() # Updates weights and biases with the optimizer (SGD)

    iter+=1
    if iter%1000==0:
        # calculate Accuracy
        with torch.no_grad():
            # Calculating the loss and accuracy for the test dataset
            correct_val = 0
            total_val = 0
            outputs_val = torch.squeeze(model(X_val))
            loss_val = criterion(outputs_val, y_val)
            
            predicted_val = outputs_val.round().detach().numpy()
            total_val += y_val.size(0)
            correct_val += np.sum(predicted_val == y_val.detach().numpy())
            accuracy_val = 100 * correct_val/total_val
            losses_val.append(loss_val.item())
            
            # Calculating the loss and accuracy for the train dataset
            total = 0
            correct = 0
            total += y_train.size(0)
            correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
            accuracy = 100 * correct/total
            losses.append(loss.item())
            Iterations.append(iter)
            
            print(f"Iteration: {iter}. \nValidation - Loss: {loss_val.item()}. Accuracy: {accuracy_val}")
            print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

# def model_plot(model,X,y,title):
#     parm = {}
#     b = []
#     for name, param in model.named_parameters():
#         parm[name]=param.detach().numpy()  
    
#     w = parm['linear.weight'][0]
#     b = parm['linear.bias'][0]
#     plt.scatter(X[:, 0], X[:, 1], c=y,cmap='jet')
#     u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
#     plt.plot(u, (0.5-b-w[0]*u)/w[1])
#     plt.xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
#     plt.ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)
#     plt.xlabel(r'$\boldsymbol{x_1}$',fontsize=16) # Normally you can just add the argument fontweight='bold' but it does not work with latex
#     plt.ylabel(r'$\boldsymbol{x_2}$',fontsize=16)
#     plt.title(title)
#     plt.show()


# # Train Data
# model_plot(model,X_train,y_train,'Train Data')

# # Validation Dataset Results
# model_plot(model,X_val,y_val,'Val Data')