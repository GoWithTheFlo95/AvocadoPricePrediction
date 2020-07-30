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


#-----------------Some useful functions-----------------------------------------------

def test_stationarity(timeseries):
    #Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
#-----------------------------------------------------------------------------------------------------

#############################Processing the data###################################################
data=dl.getData() 

plt.figure(figsize=(15,5))
#figsize=(12, 7)

#dropping columns except for date and average price
data = data.drop(columns = ['TotalVol','TotalBags','year','SmallHass','LargeHass','XLargeHass','SmallBags','LargeBags','XLargeBags'])
data = data.set_index('Date')
print(data.head())
newData=data.drop(columns=['type','region'])
#print(newData.head())
#plt.plot(newData)
#plt.show()


#---------------------group data by weeks----------------------------------------
data = data.resample('W').sum()
print(data.head())
plt.plot(data)

#---------------------doing some plots-------------------------------------------
#sns.lineplot('Date','AveragePrice',hue = 'year',data = data,)
test_stationarity(data['AveragePrice'])


#---------------------decompose data to trend, noise, seasonality----------------
dec = sm.tsa.seasonal_decompose(data['AveragePrice'],period = 52).plot()
plt.show()
#p-value < 0.05
#test statistic < critical value
#the moving average of the data is also nearly 0 and rotates around 0

#---------------------differencing to one degree---------------------------------
#Data has trend and seasonality. It is not stationary so we use differencing to make it so
data_diff = data['AveragePrice'].diff() # To find the discrete difference 
data_diff = data_diff.dropna() #drop null values
dec = sm.tsa.seasonal_decompose(data_diff,period = 52).plot()

test_stationarity(data_diff)

tsplot(data_diff)

###################################ARIMA and SARIMA#######################################################
#p: The number of lag observations included in the model, also called the lag order.
#d: The number of times that the raw observations are differenced, also called the degree of differencing.
#q: The size of the moving average window, also called the order of moving average.

model = ARIMA(data['AveragePrice'],order = (0,1,0))
model_fit = model.fit()
print(model_fit.summary())

data['FORECAST'] = model_fit.predict(start = 130,end = 170,dynamic = True)
data[['AveragePrice','FORECAST']].plot(figsize = (10,6))

exp = [data.iloc[i,0] for i in range(130,len(data))]
pred = [data.iloc[i,1] for i in range(130,len(data))]
data = data.drop(columns = 'FORECAST')
error = mean_absolute_error(exp,pred)
print(error)

#---------------------Sarimax-------------------------------------------------------------
data_diff_seas = data_diff.diff(52)
data_diff_seas = data_diff_seas.dropna()
dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 52).plot()

tsplot(data_diff_seas)

model = sm.tsa.statespace.SARIMAX(data['AveragePrice'],order = (0,1,0),seasonal_order = (1,1,0,52))
results = model.fit()
print(results.summary())

data['Forecast'] = results.predict(start = 130,end = 169,dynamic = True)
data[['AveragePrice','Forecast']].plot(figsize = (12,8))

exp = [data.iloc[i,0] for i in range(130,len(data))]
pred = [data.iloc[i,1] for i in range(130,len(data))]

error = mean_absolute_error(exp,pred)
print(error)


plt.show()
