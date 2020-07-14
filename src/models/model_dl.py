#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Author : Florian Hermes, HPI
E-mail : florian.hermes@student.hpi.de
Date   : 2020-07-13
	
	Define BaseModel, RNN and LSTM model.
	
"""

import os
import sys
import pandas as pd
home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, home+'/../../dataset')
sys.path.insert(0, home+'/../')