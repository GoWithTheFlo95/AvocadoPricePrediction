from sklearn.linear_model import LinearRegression  
# pandas and numpy are used for data manipulation  
import pandas as pd  
import numpy as np  
# matplotlib and seaborn are used for plotting graphs  
import matplotlib.pyplot as plt  
import seaborn  
import os


df = pd.read_csv('e:/avocados/dl_q2_avocadopriceprediction/dataset/avocado.csv', parse_dates=["Date"])
print(df)