# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:58:00 2020

@author: Harsh Chaudhary
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:24:24 2020

@author: Harsh Chaudhary
"""

import torch
from torch import nn, optim


data = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
label = torch.Tensor([[0], [1], [1], [0]])

class AND_GATE(nn.Module):
    def __init__(self):
        super(AND_GATE, self).__init__()
        
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

model = AND_GATE()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 4501
for e in range(epochs):
    train_loss = 0

    output = model(data)
    loss = criterion(output, label)
    
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if e%100==0:
        print('epoch {}: Training loss: {:.4f}'.format(e+1, train_loss))

def test(a, b):
    input_values = torch.Tensor([[a, b]])
    output = model(input_values)
    print(list(map(fun, output)))

def fun(x):
    if x>0.5:
        return 1
    return 0
