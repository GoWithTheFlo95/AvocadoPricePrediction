import pandas as pd
#from random import random
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
#from collections import Counter
import os
import sys
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso 
#from sklearn.linear_model import ElasticNet 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression 
#from sklearn.tree import DecisionTreeRegressor 
#from sklearn.neighbors import KNeighborsRegressor 
#from sklearn.svm import SVR 
#print(os.listdir("../Avocado"))
#data=pd.read_csv('G:/Avocado/avocado.csv',  encoding='latin-1', parse_dates=['Date'])


data = pd.read_csv('E:/test/avocado.csv')
original_data=data

# split date: day,month,year 
liste = []
for date in data.Date:
    liste.append(date.split("-"))
    
# month and day adding to lists
month = []
day = []
for i in range(len(liste)):
    month.append(liste[i][1])
    day.append(liste[i][2])
    
# adding to dataset
data["month"] = month
data["day"] = day

# delete old date column
data.drop(["Date"],axis=1,inplace=True)

#convert objects to int
data.month = data.month.values.astype(int)
data.day = data.day.values.astype(int)

# drop unnecessary features
data.drop(["Unnamed: 0"],axis=1,inplace=True)

newData = pd.get_dummies(data,drop_first=True)


# OLS
exog = newData.iloc[:,1:].values
endog = newData.iloc[:,[0]].values
r_ols = sm.OLS(endog,exog) 
r= r_ols.fit()
print(r.summary())

# Y
y = newData[["AveragePrice"]][:]

# X
x = newData.drop(["AveragePrice"],axis=1,inplace=True)
x = newData.iloc[:,1:]

# Scale the data to be between -1 and 1
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# Creating Train and Test Datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


classic_models = [] 
classic_models.append(('LR', LinearRegression())) 
classic_models.append(('LASSO', Lasso())) 
#classic_models.append(('EN', ElasticNet())) 
#classic_models.append(('KNN', KNeighborsRegressor())) 
#classic_models.append(('DTR', DecisionTreeRegressor())) 
#classic_models.append(('SVR', SVR()))

# evaluate models using cross validation score:
classic_results = [] 
classic_names = []
for name, model in classic_models:
    kfold = KFold(n_splits=10, random_state=42) 
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2') 
    classic_results.append(cv_results) 
    classic_names.append(name) 
    print("Model Name:{} Model Score:{:.3f} Model Std:{:.3f}".format(name, cv_results.mean(), cv_results.std()))

# Compare Model's Scores
f,ax = plt.subplots(figsize = (10,7))
sns.boxplot(x=classic_names, y=classic_results,palette="viridis");
plt.title("Compare Model's Scores",fontsize = 20,color='blue')
plt.xlabel('Models',fontsize = 15,color='blue')
plt.ylabel('Scores',fontsize = 15,color='blue')


#plt.rc('figure', figsize=(20, 8))
#sns.set(font_scale=1.5)

#data.sort_values(['Date', 'region'], inplace=True)
#print(data.region.unique())
#print(data.corr())

#data = data.set_index('Date')

#data = data.sort_values("Date")

#data['month']=data['Date'].apply(lambda x:x.month)
#data['day']=data['Date'].apply(lambda x:x.day)
#data.month = data.month.values.astype(int)
#data.day = data.day.values.astype(int)
#data.drop(["Unnamed: 0"],axis=1,inplace=True)

#data_noncat = pd.get_dummies(data,drop_first=True)

#exog_noncat = data_noncat.iloc[:,2:].values
#endog_noncat= data_noncat.iloc[:,[0]].values
#r_ols_noncat = sm.OLS(endog_con,exog_con) #bağımlı değişken, X_l:bağımsız değişkenlerimiz.
#r_noncat = r_ols_con.fit()
#print(r_noncat.summary())

#print(data_noncat.head())
