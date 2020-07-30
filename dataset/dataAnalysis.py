#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Author : Florian Hermes, HPI
E-mail : florian.hermes@student.hpi.de
Date   : 2020-07-02
	
	Data analysis and preprocessing.
"""

## Initialize libraries and path variables
import os
home = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib as plt


## Import dataset
avocadoDat = pd.read_csv(home + '/avocado.csv')
print(avocadoDat.head())
print()

## Characteristics
print('CHARACTERISTICS')
print()

print('Datatypes of features')
print(str(avocadoDat.dtypes))
print()

print('Unique objects of feautre X')
print(pd.unique(avocadoDat['region']))
print('Number of unique objects of feautre X')
print(avocadoDat['region'].nunique())
print()

print('Number of dimensions')
print(avocadoDat.ndim)
print('Dataset size')
print(avocadoDat.shape)
print()

# Feature inspection
print('FEATURES')
print(avocadoDat['AveragePrice'].describe())
# or .max()/ .min()/ .mean()/ .std()/ .count()
print()

print('Features grouped by')
groupedRegion = avocadoDat.groupby(['region'])
print(groupedRegion['AveragePrice'].max())
print()


## Plots
groupedRegion_MaxAvgPrice = groupedRegion['AveragePrice'].max()
groupedRegion_MaxAvgPrice.plot(kind='bar')