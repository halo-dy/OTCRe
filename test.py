import pandas as pd

df = pd.read_pickle("D:/作业/dLproj/OTCRe/OTCRe/data/raw/Singapore/Singapore_KG_plus.pkl")

print(f"原始: {len(df)} POIs, {df['Brand'].nunique()} brands")
print(df['Brand'].value_counts().describe())