import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../customer-churn-ltv-engine

file_path = os.path.join(PROJECT_ROOT, "data", "raw", "telco_churn.csv")
model_dir = os.path.join(PROJECT_ROOT, "models")
model_output_path = os.path.join(model_dir, "cltv_pipeline.pkl")

os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna()

# ONLY 3 FEATURES (MATCH API)
X = df[["Tenure Months", "Monthly Charges", "Total Charges"]]
y = df["CLTV"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

joblib.dump(model, model_output_path)

print(f"✅ Model trained with 3 features only")
print(f"✅ Saved to: {model_output_path}")