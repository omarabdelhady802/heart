import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv(r"data/cleaned_data.csv")

# Separate features and target variable
X = df.drop(columns=["target"])
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=4)  # Limit depth for understandable rules
dt.fit(X_train, y_train)

# Extract rules from the decision tree
rules = export_text(dt, feature_names=list(X.columns))

# Save rules to a file
with open("data/extracted_rules.txt", "w") as f:
    f.write(rules)

print("Rules extracted and saved to extracted_rules.txt")
