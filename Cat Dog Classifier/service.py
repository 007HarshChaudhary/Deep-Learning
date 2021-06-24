# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:13:25 2021

@author: Harsh Chaudhary
"""
from flask import Flask, jsonify, request
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

def loadModel(path = "D:\Pytorch_DL\CAT_DOG_MODEL"):
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except:
        return None
    
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
    
    model.parameters = checkpoint['parameters']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
    
    
def predict(imageBytes):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(imageBytes))
    input_tensor = test_transforms(image)
    
    modelPath = "D:\Pytorch_DL\CAT_DOG_MODEL"
    model = loadModel(modelPath)  
    model.eval()
    
    image = input_tensor[None,:,:,:]
    output = model(image)
    
    ps=torch.exp(output)
    
    topconf, topclass = ps.topk(1, dim=1)
    percentage = "{:.2f} %".format(topconf.item()*100)
    if topclass.item() == 1:
        return {'class':'dog','confidence':percentage}
    else:
        return {'class':'cat','confidence':percentage}

@app.route('/predict', methods=['POST'])
def getPrediction():
    if request.method == 'POST':
        try:
            imageBytes = request.data
            response = predict(imageBytes)
            print(response)
            return jsonify(response)
        except:
            return {'Error': 'Invalid Format'}

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

if __name__ == '__main__':
    app.run()