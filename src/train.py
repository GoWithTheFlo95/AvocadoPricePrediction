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

from data_loading import depVar, indepVar, splitTrainTestDataset


def train(dataset, n_epochs, optimizer, model, loss_function):

    for i in range(n_epochs):

        for features, labels in dataset:
            optimizer.zero_grad()

            inputs = features.reshape(1, -1, 1)

            hidden = model(inputs)

            loss = loss_function(hidden, labels)
            loss.backward()

            optimizer.step()

        if i%10 == 0:
            print(f'epoch: {i:3} - loss: {loss.item():10.8f}')

    print(f'epoch: {i:3} - loss: {loss.item():10.8f}')

    return model


# --------------------------------------------------------------------

from data_loading import getData
from model_dl import LSTMModel
import torch.nn as nn
import torch
import math

data = getData()

data = data.drop(['Date', 'TotalVol', 'SmallHass', 'LargeHass', 'XLargeHass', 'TotalBags', 'SmallBags', 'LargeBags', 'XLargeBags', 'type', 'year'], 1)
data = data.drop( data[data.region == 'WestTexNewMexico'].index )
print(data.info())


## Initialization
# hyperparameters 
lr = 0.01
n_epochs = 100
hidden_size = 50
slide_win = 52

lstm_model = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1, n_layer=1, sequence_len=1, cell = "LSTM")
loss_function = nn.MSELoss() # or L1 loss           #CrossEntropy for Categories
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)


# Dataset preparation
# 338 timestemps for 53 regions with batch_size 1 and 1 feature
train_seq = []
test_seq = []

count = data.loc[data['region'] == data['region'].unique()[0], 'region'].count() #338
count_train = math.floor(count * 0.8) # train, test index

i=0

for i in range(data['region'].nunique()):
    avgPofReg = data.loc[data['region'] == data['region'].unique()[i], 'AveragePrice']
    avgPofReg = torch.Tensor(avgPofReg)

    for j in range(count_train - slide_win):
        features = avgPofReg[j:j+slide_win]
        label = avgPofReg[i+slide_win:i+slide_win+1]
        train_seq.append((features, label))

print(type(features))
print(len(features))
print(len(label))

#train(train_seq, n_epochs, optimizer, lstm_model, loss_function)


# 1. slidig window just for totalUS
#    use all regions -> (52,2) seq_len=52
# 2. include region either nn.Embedding with string or binary encoded beforehand