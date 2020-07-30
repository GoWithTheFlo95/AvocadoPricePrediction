#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Author : Florian Hermes, HPI
E-mail : florian.hermes@student.hpi.de
Date   : 2020-07-15
	
	Implement train algorithm.
	
"""

import os
import sys
import pandas as pd
home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, home+'/../dataset')
sys.path.insert(0, home+'/models/')
import torch.nn as nn
import torch
import math
from data_loading import getData
from model_dl import LSTMModel
from train import train, visualizeLoss


# check for GPU avialability otherwise use CPU
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")


# get processed dataset
data = getData()

# adapt dataset for use case: Average Price prediction
data = data.drop(['Date', 'TotalVol', 'SmallHass', 'LargeHass', 'XLargeHass', 'TotalBags', 'SmallBags', 'LargeBags', 'XLargeBags', 'type', 'year'], 1)
data = data.drop( data[data.region == 'WestTexNewMexico'].index )
print(data.info())


# Initialize hyperparameters 
lr = 0.0005
n_epochs = 20
hidden_size = 50
slide_win = 104


# Dataset preparation
train_seq = []

count = data.loc[data['region'] == data['region'].unique()[0], 'region'].count() #338
count_train = math.floor(count * 0.8) # train, test index

i=0

for i in range(data['region'].nunique()):
    avgPofReg = data.loc[data['region'] == data['region'].unique()[i], 'AveragePrice']
    avgPofReg = torch.Tensor(avgPofReg)

    for j in range(count_train - slide_win):
        sequences = avgPofReg[j:j+slide_win]
        label = avgPofReg[i+slide_win:i+slide_win+1]
        train_seq.append((sequences, label))

# Initialize model, loss, optimizer
lstm_model = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1, n_layer=1, sequence_len=1, cell = "LSTM")
loss_function = nn.MSELoss() # or L1 loss           #CrossEntropy for Categories
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

# train model
model, lossBuffer = train(train_seq, n_epochs, optimizer, lstm_model, loss_function, device)

# visualize loss
visualizeLoss(lossBuffer)