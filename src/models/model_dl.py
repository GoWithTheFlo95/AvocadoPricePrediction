#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Author : Florian Hermes, HPI
E-mail : florian.hermes@student.hpi.de
Date   : 2020-07-13
	
	Define BaseModel, LSTM model and RNN model.
    For this project we focused on LSTM model, therefore the code for RNN is just kept here for future work.
	
"""

import os
import sys
import pandas as pd
home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, home+'/../../dataset')
sys.path.insert(0, home+'/../')

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable



## Base Model
class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_len, n_layer, cell):

        super(BaseModel, self).__init__()
        
        # initialize NN variables
        self.hidden_size = hidden_size          # number of neurons that store information of previous inputs
        self.input_size = input_size            # size of a input element of a sequence
        self.output_size = output_size          # size of a output element
        self.sequence_len = sequence_len        # length of a sequence
        self.n_layer = n_layer                  # num of RNNs stacked on top of each other
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch_first = True

        # RNN layer
        if cell == "RNN":
            self.model = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.n_layer, batch_first=batch_first) #atch_first = True)
        
        if cell == "LSTM":
            self.model = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.n_layer, batch_first=batch_first) #batch_first = True)
        
        # linear layer
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layer * 1, batch_size, self.hidden_size))
        hidden=hidden.to(self.device)
        return hidden


# LSTM Model
class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, sequence_len, n_layer, cell):

        super(LSTMModel, self).__init__(input_size, hidden_size, output_size, sequence_len, n_layer, cell)
    
    def forward(self, input):
        batch_size = input.shape[0]

        h0 = self.init_hidden(batch_size)
        c0 = self.init_hidden(batch_size)
        
        lstmOut, (_hn, _cn) = self.model(input, (h0, c0))

        lstmOut = self.linear(lstmOut[:, -1, :])

        return lstmOut


# RNN Model
class RNNModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, sequence_len, n_layer, cell):

        super(RNNModel, self).__init__(input_size, hidden_size, output_size, sequence_len, n_layer, cell)

    def forward(self, input):
        batch_size = input.shape[0]
        
        h0 = self.init_hidden(batch_size)
        
        rnnOut, _hn = self.model(input, h0)

        rnnOut = self.linear(rnnOut[:, -1, :])

        return rnnOut
