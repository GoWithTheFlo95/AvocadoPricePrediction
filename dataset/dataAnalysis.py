#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Author : Florian Hermes, HPI
E-mail : florian.hermes@student.hpi.de
Date   : 2020-07-02
	
	Data preprocessing.
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

## Characteristics
# types
print(str(type(avocadoDat)))
print(str(avocadoDat.dtypes))
print(avocadoDat.columns)

# Size
print(avocadoDat.size)

# Input/ target output


# dimensions
print(avocadoDat.ndim)
print(avocadoDat.shape)

# conducted preprocessing


# datset splits
