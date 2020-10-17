# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:31:43 2020

@author: max-z
"""
import torch
import os
import cv2
import matplotlib.pyplot as plt
import random


#The set I used was small enough for me to store it all in memory. For refernce, I'm using an ultrabook w/ 4gb VRAM and 16gb ram.
class Dataset():
    def __init__(self, directories, batch_size):
        self.batch_size=batch_size
        self.directories=directories
        self.labels=[]
        self.data=[]
        for f in os.listdir(self.directories[0]):
            path=os.path.join(self.directories[0], f)
            image=torch.tensor(cv2.imread(path))
            self.data.append(torch.rot90(image, random.randint(0,3), [0,1]))
            self.labels.append(0)
            
            
        for f in os.listdir(self.directories[1]):
            path=os.path.join(self.directories[1], f)
            image=torch.tensor(cv2.imread(path))
            self.data.append(torch.rot90(image, random.randint(0,3), [0,1]))
            self.labels.append(1)
            
        
    def generate_batches(self):
        self.data=torch.stack(self.data)
        self.labels=torch.tensor(self.labels)
        shuffle=torch.randperm(self.labels.size()[0])
        self.data=self.data[shuffle]
        self.labels=self.labels[shuffle]
        self.data=self.data.float()
        scaled=torch.Tensor(self.labels.size()[0], 224, 224, 3)
        for i in range(self.data.size()[0]):
            image=torch.div(self.data[i], 255)
            scaled[i]=image
        scaled=scaled.reshape(self.labels.size()[0], 3, 224, 224)
        current=0
        for i in range(self.labels.size()[0]):
            if i%self.batch_size==0 and i!=0:
                yield scaled[i-self.batch_size:i], self.labels[i-self.batch_size:i], True
                current=i
            if i==self.labels.size()[0]-1:
                yield scaled[current:i], self.labels[current:i], False
                
    def print_image(self, image):
        image=image.reshape(224, 224, 3)
        image=image.numpy()
        b,g,r = cv2.split(image)
        frame_rgb = cv2.merge((r,g,b))
        plt.imshow(frame_rgb)
        plt.title('Sample') 
        plt.show()
        
        
    def manipulate_data(self):
        benign=0
        malignant=0
        for i in range(len(self.labels)):
            if self.labels[i]==0:
                benign+=1
            else:
                malignant+=1
        count=0
        removed=0
        while removed<benign-malignant:
            if benign>malignant:
                if self.labels[count]==0 :
                    self.labels.pop(count)
                    self.data.pop(count)
                    removed+=1
            if benign<malignant:
                if self.labels[count]==1 :
                    self.labels.pop(count)
                    self.data.pop(count)
                    removed+=1
            count+=1
            
            

            
            
            
            
            
            
            
            