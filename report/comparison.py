import pandas as pd
import joblib
from experta import Fact
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append("./rule_based")  
from experta_engine import HeartDiseaseExpert

# Load dataset
df = pd.read_csv(r"data/heart.csv")
df.fillna(df.median(), inplace=True)

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Ensure feature names match the model's training data
model_features = joblib.load("heart_disease_model.pkl").feature_names_in_
X = X[model_features]

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained decision tree model
dt_model = joblib.load("heart_disease_model.pkl")

def evaluate_decision_tree():
    x_normalized =X_test
    numerical_cols = ["thalach"]
    scaler = MinMaxScaler()
    x_normalized[numerical_cols] = scaler.fit_transform(x_normalized[numerical_cols])
    y_pred_dt = dt_model.predict(x_normalized)
    return {
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "precision": precision_score(y_test, y_pred_dt),
        "recall": recall_score(y_test, y_pred_dt),
        "f1_score": f1_score(y_test, y_pred_dt)
    }

def evaluate_expert_system():
    expert = HeartDiseaseExpert()
    expert.reset()
    expert_results = []
    
    for i, row in X_test.iterrows():
        expert.declare(Fact(**row.to_dict()))
        expert.run()
        risk_fact = [f for f in expert.facts.values() if isinstance(f, Fact) and "risk" in f]
        expert_results.append(1 if "high" in risk_fact else 0)
        expert.reset()
    
    return {
        "accuracy": accuracy_score(y_test, expert_results),
        "precision": precision_score(y_test, expert_results),
        "recall": recall_score(y_test, expert_results),
        "f1_score": f1_score(y_test, expert_results)
    }

# Get evaluations
dt_results = evaluate_decision_tree()
es_results = evaluate_expert_system()

# Print comparisons
print("Comparison of Expert System vs. Decision Tree")
print("\nDecision Tree Model:")
print(f"Accuracy: {dt_results['accuracy']:.2f}")
print(f"Precision: {dt_results['precision']:.2f}")
print(f"Recall: {dt_results['recall']:.2f}")
print(f"F1 Score: {dt_results['f1_score']:.2f}")

print("\nExpert System:")
print(f"Accuracy: {es_results['accuracy']:.2f}")
print(f"Precision: {es_results['precision']:.2f}")
print(f"Recall: {es_results['recall']:.2f}")
print(f"F1 Score: {es_results['f1_score']:.2f}")
