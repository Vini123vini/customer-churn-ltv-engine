import pandas as pd
import requests
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telco_churn.csv")

API_BASE = "http://127.0.0.1:8000"
BATCH_SIZE = 100

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna(subset=["Total Charges"])

print(f"Loaded {len(df)} customers from CSV")

# ---------- CLTV batch ----------
cltv_customers = []
for _, row in df.iterrows():
    cltv_customers.append({
        "tenure_months": int(row["Tenure Months"]),
        "monthly_charges": float(row["Monthly Charges"]),
        "total_charges": float(row["Total Charges"])
    })

print(f"\nSending {len(cltv_customers)} customers to /predict_cltv_batch in batches of {BATCH_SIZE}...")

cltv_success = 0
for i in range(0, len(cltv_customers), BATCH_SIZE):
    batch = cltv_customers[i:i + BATCH_SIZE]
    response = requests.post(f"{API_BASE}/predict_cltv_batch", json={"customers": batch})
    if response.status_code == 200:
        result = response.json()
        cltv_success += result.get("count", 0)
        print(f"  Batch {i // BATCH_SIZE + 1}: {result.get('count', 0)} predictions saved")
    else:
        print(f"  Batch {i // BATCH_SIZE + 1} FAILED: {response.status_code} {response.text}")
    time.sleep(0.1)

print(f"✅ CLTV done: {cltv_success} total predictions saved")

# ---------- Churn batch ----------
churn_customers = []
for _, row in df.iterrows():
    churn_customers.append({
        "tenure_months": int(row["Tenure Months"]),
        "monthly_charges": float(row["Monthly Charges"]),
        "total_charges": float(row["Total Charges"]),
        "gender": row["Gender"],
        "senior_citizen": row["Senior Citizen"],
        "partner": row["Partner"],
        "dependents": row["Dependents"],
        "phone_service": row["Phone Service"],
        "multiple_lines": row["Multiple Lines"],
        "internet_service": row["Internet Service"],
        "online_security": row["Online Security"],
        "online_backup": row["Online Backup"],
        "device_protection": row["Device Protection"],
        "tech_support": row["Tech Support"],
        "streaming_tv": row["Streaming TV"],
        "streaming_movies": row["Streaming Movies"],
        "contract": row["Contract"],
        "paperless_billing": row["Paperless Billing"],
        "payment_method": row["Payment Method"]
    })

print(f"\nSending {len(churn_customers)} customers to /predict_churn_batch in batches of {BATCH_SIZE}...")

churn_success = 0
for i in range(0, len(churn_customers), BATCH_SIZE):
    batch = churn_customers[i:i + BATCH_SIZE]
    response = requests.post(f"{API_BASE}/predict_churn_batch", json={"customers": batch})
    if response.status_code == 200:
        result = response.json()
        churn_success += result.get("count", 0)
        print(f"  Batch {i // BATCH_SIZE + 1}: {result.get('count', 0)} predictions saved")
    else:
        print(f"  Batch {i // BATCH_SIZE + 1} FAILED: {response.status_code} {response.text}")
    time.sleep(0.1)

print(f"✅ Churn done: {churn_success} total predictions saved")
print("\n🎉 All predictions loaded into PostgreSQL")