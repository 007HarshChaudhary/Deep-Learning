# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:01:36 2020

@author: Harsh Chaudhary
"""

import pandas as pd
import numpy as np
data = pd.read_csv('student_data.csv')


import matplotlib.pyplot as plt

# def plot_points(data):
#     X = np.array(data[['gre', 'gpa']])
#     Y = np.array(data['admit'])
#     admitted = X[np.argwhere(Y==1)]
#     rejected = X[np.argwhere(Y==0)]
#     plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], color='red')
#     plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], color='cyan')
#     plt.xlabel('Test (GRE)')
#     plt.ylabel('Grades (GPA)')

# plot_points(data)

# Make dummy variables for rank
data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

data = data.drop('rank', axis=1)

data['gre'] /= max(data['gre'])
data['gpa'] /= max(data['gpa'])




sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
train_data, test_data = data.iloc[sample], data.drop(sample)



train_features = np.array(train_data.drop(['admit'], axis=1))
train_labels = np.array(train_data['admit'])


# # Activation (sigmoid) function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# def sigmoid_prime(x):
#     return sigmoid(x) * (1-sigmoid(x))
# def error_formula(y, output):
#     return - y*np.log(output) - (1 - y) * np.log(1-output)

# # Write the error term formula
# def error_term_formula(x, y, output):
#     return (y-output)*sigmoid_prime(x)

# # Neural Network hyperparameters
# epochs = 10000
# learnrate = 0.05

    
# # Use to same seed to make debugging easier
# np.random.seed(42)

# n_records, n_features = features.shape
# last_loss = None

# # Initialize weights
# weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# for e in range(epochs):
#     del_w = np.zeros(weights.shape)
#     for x, y in zip(features.values, targets):
#         # Loop through all records, x is the input, y is the target

#         # Activation of the output unit
#         #   Notice we multiply the inputs and the weights here 
#         #   rather than storing h as a separate variable 
#         output = sigmoid(np.dot(x, weights))

#         # The error, the target minus the network output
#         error = error_formula(y, output)

#         # The error term
#         error_term = error_term_formula(x, y, output)

#         # The gradient descent step, the error times the gradient times the inputs
#         del_w += error_term * x

#     # Update the weights here. The learning rate times the 
#     # change in weights, divided by the number of records to average
#     weights += learnrate * del_w / n_records

#     # Printing out the mean square error on the training set
#     if e % (epochs / 100) == 0:
#         out = sigmoid(np.dot(features, weights))
#         loss = np.mean((out - targets) ** 2)
#         print("Epoch:", e)
#         if last_loss and last_loss < loss:
#             print("Train loss: ", loss, "  WARNING - Loss Increasing")
#         else:
#             print("Train loss: ", loss)
#         last_loss = loss
#         print("=========")
# print("Finished training!")


import torch
from torch import nn, optim
import torch.nn.functional as F 


input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
label_tensor = torch.from_numpy(train_labels)

dataset = torch.utils.data.TensorDataset(input_tensor, label_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 2)
   
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))

        return x

model = Net()

criteron = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

losses=[]
EPOCHS = 51


for e in range(EPOCHS):
    train_loss = 0
    acc=0
    for data, label in train_loader:
        output = model(data)
        loss = criteron(output, label)
        
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(output, 1)
        eq = pred==label
        acc += torch.mean(eq.type(torch.FloatTensor))
        
    losses.append(train_loss/len(train_loader))
    if e%10==0:
        print('epoch {}: Training loss: {:.4f}      Accuracy: {:.4f}'.format(e+1, train_loss/len(train_loader), acc/len(train_loader)))

plt.plot(losses)
plt.show()    



