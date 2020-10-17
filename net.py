# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:34:18 2020

@author: max-z
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


#3 layer CNN followed by 3 fully connected layers. The network structure could probably be tweaked to increase accuracy.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.normalization=nn.BatchNorm2d(3)
        self.conv1=nn.Conv2d(3, 32, (7,7), stride=2, padding=0 )
        self.conv2=nn.Conv2d(32, 64, (7,7), stride=2, padding=0)
       # self.max_pool=nn.MaxPool2d((7, 3), stride=1)
        self.max_pool=nn.MaxPool2d((7, 7), stride=2)
        self.drop_out=nn.Dropout2d(p=0.2)
        self.conv3=nn.Conv2d(64, 128, (3,3), stride=1, padding=0)
        self.conv4=nn.Conv2d(128, 128, (3,3), stride=1, padding=0)
        #self.fc1=nn.Linear(46208, 256)
        self.fc1=nn.Linear(56448, 256)
        #self.fc1=nn.Linear(294912, 256)
        self.fc2=nn.Linear(256, 128)
        self.fc3=nn.Linear(128, 2)
        self.sm=nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        x=self.normalization(x)
        
        x=self.conv1(x)
        x=F.relu(x)
        
        x=self.conv2(x)
        x=F.relu(x)
        
        x=self.max_pool(x)
        x=self.drop_out(x)
        
        x=self.conv3(x)
        x=F.relu(x)
        
       # x=self.conv4(x)
        #x=F.relu(x)
        
    #    x=self.drop_out(x)
    #    x=torch.reshape(x, (-1, 1, 46208))
        x=torch.reshape(x, (-1, 1, 56448))
    #   x=torch.reshape(x, (-1, 1, 294912))
        
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return self.sm(x)