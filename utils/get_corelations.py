import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np


# read data
df = pd.read_csv(r"data/cleaned_data.csv")



# Generate Correlation Heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



# Feature Selection based on Correlation with Target
correlation_threshold = 0.3  # Adjust this threshold as needed
target_correlation = corr_matrix["target"].abs()
selected_features = target_correlation[target_correlation > correlation_threshold].index.tolist()
df = df[selected_features]


# Save the cleaned dataset
df.to_csv(r"data/features_selected.csv", index=False)