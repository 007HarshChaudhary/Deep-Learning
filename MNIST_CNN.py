

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import helper
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
import time

# defin the transforms
transform = transforms.Compose([transforms.ToTensor()])

# gathering the data
trainset = datasets.MNIST('MNIST_data', download=True, train=True, transform=transform)
testset = datasets.MNIST('MNIST_data', download=True, train=False, transform=transform)

# math to split into train and validation data
valid_size = 0.2
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#define data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64)


#visualize the data
img, label = next(iter(train_loader))
plt.imshow(img[0].numpy().squeeze(0), cmap='Greys_r')


from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(28*28, 256)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.hidden2 = nn.Linear(256, 124)
        self.batchnorm2 = nn.BatchNorm1d(124)
        self.output = nn.Linear(124, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.batchnorm1(self.hidden(x)))
        x = F.relu(self.batchnorm2(self.hidden2(x)))
        x = F.log_softmax(self.output(x), dim=1)
        return x

model = Network()
model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 10
valid_loss_min = np.Inf

start = time.time()
for e in range(1, epochs+1):
    train_loss = 0
    valid_loss = 0
    
    # TRAINING
    model.train()
    for i, l in train_loader:
        i = i.cuda()
        l = l.cuda()
        output = model(i)
        loss = criterion(output, l)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    
    # VALIDATION
    model.eval()
    with torch.no_grad():
        accuracy=0
        for i, l in valid_loader:
            i = i.cuda()
            l = l.cuda()
            output = model(i)
            loss = criterion(output, l)       
            
            valid_loss += loss.item()
            
            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)
            eq = top_class == l.view(*top_class.shape)
            accuracy += torch.mean(eq.type(torch.FloatTensor)) 
    
    # calculating average losses
    train_loss = train_loss/len(train_loader)
    valid_loss = train_loss/len(valid_loader)
    
    print('Epoch {}: Training loss:{:.4f} Validation loss:{:.4f} Accuracy:{:.4f}'.format(e, train_loss, valid_loss, accuracy/len(valid_loader)))
    
    if valid_loss < valid_loss_min:
        print('Validation loss decreased   {:.4f}------>{:.4f}   Saving model...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model_MNIST')
        valid_loss_min = valid_loss
        

print("Total time on CPU : "+str(time.time()-start))

# =============================================================================
# testing
# =============================================================================
images, labels = next(iter(test_loader))
model.cpu()
img = images[0].view(1, 28*28)
ps = model(img)
score = torch.exp(ps)
helper.view_classify(img.view(1, 28, 28), score)
