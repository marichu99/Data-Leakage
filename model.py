from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.pipeline import make_pipeline

# import the data set
df=pd.read_csv('AER_credit_card_data.csv')
# translate the categorical columns into ones and zeros
potential_leaks=["majorcards","expenditure","share","active"]

df.drop(potential_leaks,axis=1,inplace=True)
for x in df.columns:
    if df[x].nunique() ==2:
        df[x].replace(list(df[x].unique()),[1,0], inplace=True)
y=df["card"]
x=df.drop(["card"],axis=1)
print(df.head())
print(df.shape)


# create the pipeline
my_pipeline=make_pipeline(RandomForestClassifier(n_estimators=100,random_state=0))

scores=cross_val_score(my_pipeline,x,y,cv=5,scoring="accuracy")

print(f"The mean validation score is {scores.mean()}")
