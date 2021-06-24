# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:32:35 2019

@author: Harsh Chaudhary
"""

from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

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

# math to split into train and validation data
valid_size = 0.2
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
    
import torch
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=train_sampler)
validloader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


#visualize the data
import matplotlib.pyplot as plt
# img, label = next(iter(trainloader))
# plt.imshow(img[0].permute(1, 2, 0), cmap='Greys_r')


from torch import nn, optim
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()


optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traininglosses = []
validationlosses = []
validationaccuracy = []
totalsteps = []
epochs = 1
steps = 0
running_loss = 0
print_every = 5
min_so_far = np.Inf
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
            
            traininglosses.append(running_loss/print_every)
            validationlosses.append(test_loss/len(testloader))
            validationaccuracy.append(accuracy/len(testloader))
            totalsteps.append(steps)
            print(f"Device {device}.."
                  f"Epoch {epoch+1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"validation loss: {test_loss/len(testloader):.3f}.. "
                  f"Accuracy: {accuracy/len(testloader):.3f}")
            if validationlosses[-1] < min_so_far:
                print('Validation loss decreased   {:.4f}------>{:.4f}   Saving model...'.format(min_so_far, validationlosses[-1]))
                checkpoint = {
                    'parameters' : model.parameters,
                    'state_dict' : model.state_dict()
                }
                torch.save(checkpoint, 'CAT_DOG_MODEL')
                min_so_far = validationlosses[-1]
                
            running_loss = 0
            model.train()

plt.plot(totalsteps, traininglosses, label='Train Loss')
plt.plot(totalsteps, validationlosses, label='Test Loss')
plt.plot(totalsteps, validationaccuracy, label='Test Accuracy')
plt.legend()
plt.grid()
plt.show()

# def predict(image):
#     model.eval()
#     image = Image.open('test_cat_dog/'+image)
#     model.cpu()
#     input_tensor = test_transforms(image)
#     ps = torch.exp(model(input_tensor[None, :, :, :]))
#     print('Cat score:{:.4f}     Dog score:{:.4f}'.format(ps[0][0].item()*100, ps[0][1].item()*100 ))
#     plt.imshow(image)

# testing model
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.eval()

for image, label in testloader:

    image, label = image.cuda(), label.cuda()

    output = model(image)

    loss = criterion(output, label)

    test_loss += loss.item()*image.size(0)

    _, pred = torch.max(output, 1)    

    correct_tensor = pred.eq(label.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())

    for i in range(label.shape[0]):
        target = label.data[i]
        class_correct[target] += correct[i].item()
        class_total[target] += 1


test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

# specify the image classes
classes = ['Dog', 'Cat']

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
        
    