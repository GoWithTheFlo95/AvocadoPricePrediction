#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Author : Florian Hermes, HPI
E-mail : florian.hermes@student.hpi.de
Date   : 2020-07-13
	
	Load data, perform pre-processing and provide interface.
	
	Usage:
	------------
	./data_loading.py
"""

import os
import sys
import pandas as pd
home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, home+'/../dataset')
sys.path.insert(0, home+'/models')

from sklearn.model_selection import train_test_split


## get processed data
def getData():
	avocadoDat_unproc = pd.read_csv(home + '/../dataset/avocado.csv')

	## Data preprocessing
	# Removing unnecessary features
	avocadoDat_proc = avocadoDat_unproc.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]

	# Renam Avocado Size Columsn
	avocadoDat_proc = avocadoDat_proc.rename(index=str, columns={"Total Volume" : "TotalVol", "4046" : "SmallHass", "4225" : "LargeHass", "4770" : "XLargeHass", "Total Bags" : "TotalBags", "Small Bags" : "SmallBags", "Large Bags" : "LargeBags", "XLarge Bags" : "XLargeBags"})

	# Date transformation
	avocadoDat_proc['Date'] = pd.to_datetime(avocadoDat_proc['Date'])

	# Final, pre-processed dataset
	avocadoDat = avocadoDat_proc

	return avocadoDat


## get dependet variable
def depVar(dataset):
	X = dataset.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12]]
	return X


## get independent variable
def indepVar(dataset):
	y = dataset.iloc[:,[1]]
	return y


## split dataset in train and test data
def splitTrainTestDataset(dataset):
	X = depVar(dataset)
	y = indepVar(dataset)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	return X_train, X_test, y_train, y_test