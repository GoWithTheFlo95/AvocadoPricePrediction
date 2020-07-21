import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
#from collections import Counter
import os
import sys
import statsmodels.api as sm

print(os.listdir("../Avocado"))
#data=pd.read_csv('G:/Avocado/avocado.csv',  encoding='latin-1', parse_dates=['Date'])


data = pd.read_csv('G:/Avocado/avocado.csv')
original_data=data

data['Date'] = pd.to_datetime(data['Date'])

plt.rc('figure', figsize=(20, 8))
sns.set(font_scale=1.5)

#data.sort_values(['Date', 'region'], inplace=True)
#print(data.region.unique())
#print(data.corr())

#data = data.set_index('Date')

data = data.sort_values("Date")

data['month']=data['Date'].apply(lambda x:x.month)
data['day']=data['Date'].apply(lambda x:x.day)
data.month = data.month.values.astype(int)
data.day = data.day.values.astype(int)
data.drop(["Unnamed: 0"],axis=1,inplace=True)

data_noncat = pd.get_dummies(data,drop_first=True)

exog_noncat = data_noncat.iloc[:,2:].values
endog_noncat= data_noncat.iloc[:,[0]].values
r_ols_noncat = sm.OLS(endog_con,exog_con) #bağımlı değişken, X_l:bağımsız değişkenlerimiz.
r_noncat = r_ols_con.fit()
print(r_noncat.summary())

print(data_noncat.head())
