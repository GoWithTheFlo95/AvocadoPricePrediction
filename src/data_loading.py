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
