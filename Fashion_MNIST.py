# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:43:57 2019

@author: Harsh Chaudhary
"""

import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([transforms.ToTensor()])

trainset =  datasets.FashionMNIST('Fashion_MNIST', train=True, download=True, transform=transform)

valid_size = 0.2
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(valid_size*num_train)
train_idx, valid_idx = indices[split: ], indices[ :split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=valid_sampler)

testset = datasets.FashionMNIST('Fashion_MNIST', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 356)
        self.batchnorm1 = nn.BatchNorm1d(356)
        self.fc2 = nn.Linear(356, 124)
        self.batchnorm2 = nn.BatchNorm1d(124)
        self.fc3 = nn.Linear(124, 64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dropout(F.relu(self.batchnorm1(self.fc1(x))))
        x = self.dropout(F.relu(self.batchnorm2(self.fc2(x))))
        x = self.dropout(F.relu(self.batchnorm3(self.fc3(x))))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

model = Network()
model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

start = time.time()
epochs = 30
train_losses=[]
valid_losses=[]
min_validation_loss = np.Inf

for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        
        output = model(images)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    valid_loss = 0
    validation_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)
            
            valid_loss += criterion(output, labels)
            
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class==labels.view(*top_class.shape)
            validation_accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    valid_loss /= len(valid_loader)
    running_loss = running_loss/len(train_loader)
    
    valid_losses.append(valid_loss)
    train_losses.append(running_loss)
        
    print("Epoch: {}/{} ".format(e+1, epochs),
          "Training Loss: {:.3f} ".format(running_loss),
          "Validation Loss: {:.3f} ".format(valid_loss),
          "Validation Accuracy: {:.3f}".format(validation_accuracy/len(valid_loader)))
    
    if valid_loss < min_validation_loss:
        print('Validation loss decreased {:.4f}--->{:.4f} saving model'.format(min_validation_loss, valid_loss))
        min_validation_loss = valid_loss
        torch.save(model.state_dict(), 'FasionMNIST.pt')
    print() 
        
print("Total time to train {}".format(time.time()-start)) 

# model.cpu()
# images, labels = next(iter(test_loader))
# output = model(images[0]) 
# helper.view_classify(images[0], torch.exp(output), version='Fashion')
 
plt.plot(train_losses, label='training loss')
plt.plot(valid_losses, label='validation loss')
plt.legend(frameon=False)

model.load_state_dict(torch.load('FasionMNIST.pt'))

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for evaluation
with torch.no_grad():
    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        data=data.cuda()
        target=target.cuda()
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))