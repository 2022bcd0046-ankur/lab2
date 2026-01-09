# Name: Ankur Majumdar
# Roll No. 2022BCD0046

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=",")

X = data.drop("quality", axis=1)
y = data["quality"]

# Experiment Config Details
exp_id = "EXP-01"
model_name = "Linear Regression"
hyperparams = "Ridge Alpha-1"
preprocess = "Standard"
feature_select = "All Features"
tt_split = "80-20"

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"""
# {exp_id}
----------------------------------------
Model               : {model_name}
Hyperparameters     : {hyperparams}
Preprocessing       : {preprocess}
Feature Selection   : {feature_select}
Train-Test Split    : {tt_split}
----------------------------------------
MSE                 : {mse}
R2 Score            : {r2}
""")

# Save outputs
joblib.dump(model, "outputs/model.pkl")

results = {
    "Experiment ID": exp_id,
    "Model": model_name,
    "Hyperparameters": hyperparams,
    "Preprocessing": preprocess,
    "Feature Selection": feature_select,
    "Train-Test Split": tt_split,
    "MSE": mse,
    "R2": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)
