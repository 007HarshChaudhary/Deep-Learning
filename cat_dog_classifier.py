# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:32:35 2019

@author: Harsh Chaudhary
"""

from torchvision import datasets, transforms, models
from PIL import Image

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


data_dir = 'Cat_Dog_data/Cat_Dog_data'
train_data = datasets.ImageFolder(data_dir+'/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir+'/test', transform=test_transforms)
    
import torch
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

from torch import nn, optim


model = models.resnet18(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(512, 500),
                                 nn.ReLU(),
                                 nn.Linear(500, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()


optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

model.to('cuda')
device = 'cuda'
epochs = 1
steps = 0
losses = []
accuracies = []
test_losses = []
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses.append(running_loss/print_every)
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            test_losses.append(test_loss/len(testloader))
            accuracies.append(accuracy/len(testloader))
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

import matplotlib.pyplot as plt
def predict(image):
    model.eval()
    image = Image.open('test_cat_dog/'+image)
    model.cpu()
    input_tensor = test_transforms(image)
    ps = torch.exp(model(input_tensor[None, :, :, :]))
    print('Cat score:{:.4f}     Dog score:{:.4f}'.format(ps[0][0].item()*100, ps[0][1].item()*100 ))
    plt.imshow(image)
        
    