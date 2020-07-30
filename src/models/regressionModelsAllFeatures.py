import os
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import Lasso 
import warnings
warnings.filterwarnings("ignore")

home = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, home+'/../../dataset')
sys.path.insert(0, home+'/../')

from data_loading import getDataAllFeatures



data=getDataAllFeatures()
print(data)

# OLS Analysis
exog = data.iloc[:,1:].values   #creates a 1D vector of the data
print(exog)
endog = data.iloc[:,[0]].values
print(endog)
r_ols = sm.OLS(endog,exog) 
r= r_ols.fit()
print(r.summary())

#--------------------------------Linear Regression with statsModel---------------------------------------

y = data[["AveragePrice"]][:]                           #Y
x = data.drop(["AveragePrice"],axis=1,inplace=True)     #X
x = data.iloc[:,1:]

# Scale the data to be between -1 and 1
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Creating Train and Test Datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#testing different models
classic_models = [] 
classic_models.append(('LR', LinearRegression())) 
classic_models.append(('LASSO', Lasso())) 
classic_models.append(('KNN', KNeighborsRegressor())) 
classic_models.append(('DTR', DecisionTreeRegressor())) 

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
#sns.lineplot(x=classic_names, y=classic_results,palette="viridis")
plt.title("Compare Model's Scores",fontsize = 20,color='blue')
plt.xlabel('Models',fontsize = 15,color='blue')
plt.ylabel('Scores',fontsize = 15,color='blue')

plt.show()

