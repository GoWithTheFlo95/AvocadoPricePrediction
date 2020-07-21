import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode

from scipy.stats import norm
from plotly import tools


df = pd.read_csv('e:/avocados/dl_q2_avocadopriceprediction/dataset/avocado.csv')#, parse_dates=["Date"])

#Type=df.groupby('type')['Total Volume'].agg('sum')

#values=[Type['conventional'],Type['organic']]
#labels=['conventional','organic']

#trace=go.Pie(labels=labels,values=values)
#py.plot([trace])

sns.set(font_scale=1.5) 

#fig, ax = plt.subplots(figsize=(15, 9))
#sns.distplot(a=df.AveragePrice, kde=False, fit=norm)
#plt.show()

df['Year'], df['Month'],  df['Day'] = df['Date'].str.split('-').str

#plt.figure(figsize=(18,10))
#sns.lineplot(x="Month", y="AveragePrice", hue='type', data=df)
#plt.show()


df['Month'] = df['Month'].replace({'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', 
                                   '06': 'June', '07': 'July', '08': 'August', '09': 'September', '10': 'October', 
                                   '11': 'November', '12': 'December'})


ax = sns.catplot(x="Month", y="AveragePrice", hue="type", 
            kind="box", data=df, height=8.5, linewidth=2.5, aspect=2.8,palette="Set2")
            
plt.figure(figsize=(18,10))
sns.lineplot(x="Month", y="AveragePrice", hue='year',  data=df)
#plt.show()


Month = df[['Total Volume', 'AveragePrice']].groupby(df.Month).sum()
Month.plot(kind='line', fontsize = 14,figsize=(14,8))
plt.show()


con=df[df['type']=='conventional'].groupby('year')['Total Volume'].agg('mean')
org=df[df['type']=='organic'].groupby('year')['Total Volume'].agg('mean')

trace1=go.Bar(x=con.index,y=con,name="Conventional",
             marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

trace2=go.Bar(x=con.index,y=org,name="Organic",
             marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

data=[trace1,trace2]

layout=go.Layout(barmode="stack",title="Organic vs. Conventional (Mean Volume)",
                yaxis=dict(title="Volume"))
fig=go.Figure(data=data,layout=layout)
py.plot(fig)


region_list=list(df.region.unique())
average_price=[]

for i in region_list:
    x=df[df.region==i]
    region_average=sum(x.AveragePrice)/len(x)
    average_price.append(region_average)

df1=pd.DataFrame({'region_list':region_list,'average_price':average_price})
new_index=df1.average_price.sort_values(ascending=False).index.values
sorted_data=df1.reindex(new_index)

plt.figure(figsize=(24,10))
ax=sns.barplot(x=sorted_data.region_list,y=sorted_data.average_price)

plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title('Average Price of Avocado According to Region')



