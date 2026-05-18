import pandas as pd

df = pd.read_pickle("data/raw/Singapore/Singapore_KG_plus.pkl")

print(df.shape)
print(df.columns)
print(df.head())

print(df['Brand'].value_counts().head(10))
print(df['Region_ID'].value_counts().head(10))
print(df.isnull().sum())