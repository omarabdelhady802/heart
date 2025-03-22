import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

# Load the dataset
df = pd.read_csv(r"data/heart.csv")

# Handling missing values (fill with mean for numerical columns)
df.fillna(df.mean(), inplace=True)

# Normalize numerical features
numerical_cols = ["age", "chol", "trestbps","thalach"]
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the cleaned dataset
df.to_csv(r"data/cleaned_data.csv", index=False)

print("Data preprocessing complete with feature selection based on correlation. Cleaned dataset saved as cleaned_data.csv")
