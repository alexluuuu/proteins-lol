"""
	convnet.py
"""

import os 
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn


class simple_CNN(nn.Module):
    def __init__(self):
        self.W = self.H = 256
        super(simple_CNN, self).__init__()

        #input: 4xWxH, 
        self.conv1 = nn.Sequential( 
            nn.Conv2d(4, 16, 12, 6, 4), # input_channels, output_channels, kernel_size, stride, padding
            nn.ReLU(),
        ) 
        # output: 16 x 43 x 43

        #input: 16 x 43 x 43
        self.conv2 = nn.Sequential(     
            nn.Conv2d(16,32,4,1,1),        # input_channels, output_channels, kernel_size, stride, padding   
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2), 
        )
        #output: 32x21x21

        #input: 32x21x21
        self.conv3 = nn.Sequential(      
            nn.Conv2d(32,32,3,1,1),        # input_channels, output_channels, kernel_size, stride, padding   
            nn.ReLU(),                      
        )
        #output: 32 x21 x 21

        #input: 32x21x21
        self.conv4 = nn.Sequential(      
            nn.Conv2d(32,32,3,1,1),        # input_channels, output_channels, kernel_size, stride, padding   
            nn.ReLU(),                      
        )
        #output: 32x21x21

        #input: 32x21x21
        self.conv5 = nn.Sequential(      
            nn.Conv2d(32,32,3,1,1),        # input_channels, output_channels, kernel_size, stride, padding   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                      
        )
        #output: 32x9*9

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear( 32*10*10, 16*10*10)
        self.fc2 = nn.Linear(16*10*10, 8*10*10)
        self.fc3 = nn.Linear( 8*10*10, 28)
        
    
    def forward(self, x):
    	x = self.conv1(x)
    	# print(x.shape)
    	x = self.conv2(x)
    	# print(x.shape)
    	x = self.conv3(x)
    	# print(x.shape)
    	x = self.conv4(x)
    	# print(x.shape)
    	x = self.conv5(x)
    	# print(x.shape)

    	x = x.view(x.size(0), -1)
    	# print(x.shape)
    	output = self.drop_out(x)
    	# print(output.shape)
    	output = self.fc1(output)
    	# print(output.shape)

    	output = self.drop_out(output)
    	# print(output.shape)

    	output = self.fc2(output)
    	# print(output.shape)

    	output = self.drop_out(output)
    	# print(output.shape)

    	output = self.fc3(output)
    	# print(output.shape)

        return output
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

