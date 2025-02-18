import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "./data/product_test.csv"
df = pd.read_csv(file_path)

# Stratified sampling
sampled_df, _ = train_test_split(df, train_size=1000, stratify=df['class'], random_state=42)


# Save the sampled dataset
sampled_df.to_csv("./data/product_test_sample.csv", index=False)

print("Stratified sample of 1000 rows saved")