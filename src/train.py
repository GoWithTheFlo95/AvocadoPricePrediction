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
    
    features, _, labels, _ = splitTrainTestDataset(dataset)

    for i in range(n_epochs):
        optimizer.zero_grad()

        inputs = features.view(1, 1, len(features))

        output, hidden = model(inputs)

        loss = loss_function(hidden, labels)
        loss.backward()

        optimizer.step()

        if i%10 == 0:
            print(f'epoch: {i:3} - loss: {loss.item():10.8f}')

    print(f'epoch: {i:3} - loss: {loss.item():10.8f}')

    return model