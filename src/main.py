import data_loading as dl

df=dl.getData()

PREDICTION_TYPE = 'conventional'
df = df[df.type == PREDICTION_TYPE]
PREDICTING_FOR = "TotalUS"
print(df)