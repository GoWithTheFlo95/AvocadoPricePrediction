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

            inputs = features.values.reshape(1, -1, 1)
            inputs = torch.Tensor(inputs)

            print(inputs.ndim)
            print(inputs.size)
            print(inputs.shape[0])

            _, hidden = model(inputs)

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

data = data.drop(['Date', 'SmallHass', 'LargeHass', 'XLargeHass', 'TotalBags', 'SmallBags', 'LargeBags', 'XLargeBags', 'type', 'year'], 1)
data = data.loc[:, 'AveragePrice']
print(data)

#batch_size = data.shape[0] / data['region'].nunique()
#print(batch_size)
#print(18250/54)

data = data.drop( data[data.region == 'WestTexNewMexico'].index )

for i in range(data['region'].nunique()):
    count = data.loc[data['region'] == data['region'].unique()[i], 'region'].count()
    if count != 338:
        print(data['region'].unique()[i])
        print(count)

# 338 timestemps for 53 regions with batch_size 1 and 1 feature
train_seq = []
test_seq = []
#data_seq = []

count = 338 #data.loc[data['region'] == data['region'].unique()[i], 'region'].count()
count_train = math.floor(count * 0.8)
slide_win = 52
#label_index = slide_win+1

for i in range(data['region'].nunique()):
    features = data.loc[data['region'] == data['region'].unique()[i], 'AveragePrice']
    label = data.loc[data['region'] == data['region'].unique()[i], 'AveragePrice']
    train_seq.append((features[:count_train], label[:count_train]))
    test_seq.append((features[count_train:], label[count_train:]))
    #data_seq.append((features, label))

# 52 week for timetemps

print(features)

#print(type(features))

# initialize
print('INITIALIZATION')
lr = 0.01
n_epochs = 100

lstm_model = LSTMModel(input_size=1, hidden_size=50, output_size=1, n_layer=1, sequence_len=1, cell = "LSTM")
loss_function = nn.MSELoss() # or L1 loss           #CrossEntropy for Categories
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

train(train_seq, n_epochs, optimizer, lstm_model, loss_function)


# 1. slidig window just for totalUS
#    use all regions -> (52,2) seq_len=52
# 2. include region either nn.Embedding with string or binary encoded beforehand