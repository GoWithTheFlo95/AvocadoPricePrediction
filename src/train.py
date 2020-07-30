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
import matplotlib.pyplot as plt
home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, home+'/../dataset')
sys.path.insert(0, home+'/models/')


# train model
def train(dataset, n_epochs, optimizer, model, loss_function, device):

    model = model.to(device)
    lossBuffer = []

    for i in range(n_epochs):

        for sequences, labels in dataset:

            optimizer.zero_grad()

            inputs = sequences.reshape(1, -1, 1)
            inputs = inputs.to(device)

            out = model(inputs)
            out = out.to(device)

            labels = labels.reshape(1, 1)
            labels = labels.to(device)

            loss = loss_function(out, labels)
            loss.backward()

            optimizer.step()

        print(f'epoch: {i:3} - loss: {loss.item():10.8f}')
        lossBuffer.append(loss.item())

    return model, lossBuffer


# visualize loss
def visualizeLoss(lossBuffer):
    plt.plot(lossBuffer)
    plt.show()