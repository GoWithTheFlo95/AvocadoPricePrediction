import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from itertools import combinations

from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import data_loading as dl

data=dl.getData() 


data = data.drop(columns = ['TotalVol','TotalBags','year','SmallHass','LargeHass','XLargeHass','SmallBags','LargeBags','XLargeBags'])

data = data.set_index('Date')
data = data.resample('W').sum()
print(data.head())

dec = sm.tsa.seasonal_decompose(data['AveragePrice'],period = 52).plot()


#sns.lineplot('Date','AveragePrice',hue = 'year',data = data,)


plt.show()