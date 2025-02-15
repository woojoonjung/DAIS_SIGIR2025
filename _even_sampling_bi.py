import pandas as pd

file_path = "./data/bank_.csv"
df = pd.read_csv(file_path)

sampled_df = df.groupby('class').apply(lambda x: x.sample(n=500, random_state=42)).reset_index(drop=True)
sampled_df.to_csv('bank.csv', index=False)

print("Even sample of 1000 rows saved")