# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:38:17 2020
@author: max-z
"""
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from net import Net
from dataset import Dataset



if torch.cuda.is_available():
    device=torch.device('cuda:0')
    print('Running on GPU')
else:
    device=torch.device('cpu')
    print('Running on CPU')
    

    
def train(net:Net):
    optimizer=optim.Adam(params=net.parameters(), lr=0.00001)
    objective=nn.CrossEntropyLoss()
    EPOCHS=12
    train_losses=[]
    for i in range(EPOCHS):
        print(i)
        #batch size of 64
        data_handler=Dataset([r'train\benign',r'train\malignant'], 64)
        #makes sure the dataset contains equal amounts of benign and malignant images
        data_handler.manipulate_data()
        #generator object
        loader=data_handler.generate_batches()
        done=True
        net.train(True)
        while done:
            images, label, done=next(loader)
            optimizer.zero_grad()
            images=images.to(device)
            label=label.to(device)
            prediction=net(images).squeeze()
            loss=objective(prediction, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().item())
    plt.plot(train_losses)
    plt.show()
    
#Determines the percent of images the net diagnoses correctly
def test(net:Net):
    net.eval()
    done=True
    loader=Dataset([r'test\benign', r'test\malignant'], 64)
    batches=loader.generate_batches()
    loader.manipulate_data()
    losses=[]
    while done:
        images, label, done=next(batches)
        images=images.to(device)
        label=label.to(device)
        with torch.no_grad():
            prediction=net(images).squeeze()
            for i in range(label.size()[0]):
                if (prediction[i][0]>0.5 and label[i]==0) or (prediction[i][0]<0.5 and label[i]==1):
                    losses.append(1)
                else:
                    losses.append(0)
                    
    net.train(True)
    print(sum(losses)/len(losses)*100)
    
    
    
net=Net().to(device)
test(net)
train(net)
test(net)


