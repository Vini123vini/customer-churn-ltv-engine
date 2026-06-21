import os
from fastapi import FastAPI, Depends
import joblib
import pandas as pd
from sqlalchemy.orm import Session

from src.api.schema import CLTVData, ChurnData, CLTVBatchData, ChurnBatchData
from src.database import get_db
from src.models import CustomerPrediction, CustomerChurnPrediction

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

CLTV_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cltv_pipeline.pkl")
CHURN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")

cltv_model = joblib.load(CLTV_MODEL_PATH)
churn_model = joblib.load(CHURN_MODEL_PATH)

# Exact column order the churn model was trained on
CHURN_MODEL_COLUMNS = [
    'Tenure Months', 'Monthly Charges', 'Total Charges',
    'Gender_Male', 'Senior Citizen_Yes', 'Partner_Yes', 'Dependents_Yes',
    'Phone Service_Yes', 'Multiple Lines_No phone service', 'Multiple Lines_Yes',
    'Internet Service_Fiber optic', 'Internet Service_No',
    'Online Security_No internet service', 'Online Security_Yes',
    'Online Backup_No internet service', 'Online Backup_Yes',
    'Device Protection_No internet service', 'Device Protection_Yes',
    'Tech Support_No internet service', 'Tech Support_Yes',
    'Streaming TV_No internet service', 'Streaming TV_Yes',
    'Streaming Movies_No internet service', 'Streaming Movies_Yes',
    'Contract_One year', 'Contract_Two year',
    'Paperless Billing_Yes',
    'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check',
    'Payment Method_Mailed check'
]


def build_churn_input_row(data: ChurnData) -> pd.DataFrame:
    """Build a single-row dataframe matching the model's 30 training columns exactly."""
    row = {col: 0 for col in CHURN_MODEL_COLUMNS}

    row['Tenure Months'] = data.tenure_months
    row['Monthly Charges'] = data.monthly_charges
    row['Total Charges'] = data.total_charges

    if data.gender == "Male":
        row['Gender_Male'] = 1
    if data.senior_citizen == "Yes":
        row['Senior Citizen_Yes'] = 1
    if data.partner == "Yes":
        row['Partner_Yes'] = 1
    if data.dependents == "Yes":
        row['Dependents_Yes'] = 1
    if data.phone_service == "Yes":
        row['Phone Service_Yes'] = 1

    if data.multiple_lines == "No phone service":
        row['Multiple Lines_No phone service'] = 1
    elif data.multiple_lines == "Yes":
        row['Multiple Lines_Yes'] = 1

    if data.internet_service == "Fiber optic":
        row['Internet Service_Fiber optic'] = 1
    elif data.internet_service == "No":
        row['Internet Service_No'] = 1

    if data.online_security == "No internet service":
        row['Online Security_No internet service'] = 1
    elif data.online_security == "Yes":
        row['Online Security_Yes'] = 1

    if data.online_backup == "No internet service":
        row['Online Backup_No internet service'] = 1
    elif data.online_backup == "Yes":
        row['Online Backup_Yes'] = 1

    if data.device_protection == "No internet service":
        row['Device Protection_No internet service'] = 1
    elif data.device_protection == "Yes":
        row['Device Protection_Yes'] = 1

    if data.tech_support == "No internet service":
        row['Tech Support_No internet service'] = 1
    elif data.tech_support == "Yes":
        row['Tech Support_Yes'] = 1

    if data.streaming_tv == "No internet service":
        row['Streaming TV_No internet service'] = 1
    elif data.streaming_tv == "Yes":
        row['Streaming TV_Yes'] = 1

    if data.streaming_movies == "No internet service":
        row['Streaming Movies_No internet service'] = 1
    elif data.streaming_movies == "Yes":
        row['Streaming Movies_Yes'] = 1

    if data.contract == "One year":
        row['Contract_One year'] = 1
    elif data.contract == "Two year":
        row['Contract_Two year'] = 1

    if data.paperless_billing == "Yes":
        row['Paperless Billing_Yes'] = 1

    if data.payment_method == "Credit card (automatic)":
        row['Payment Method_Credit card (automatic)'] = 1
    elif data.payment_method == "Electronic check":
        row['Payment Method_Electronic check'] = 1
    elif data.payment_method == "Mailed check":
        row['Payment Method_Mailed check'] = 1

    return pd.DataFrame([row], columns=CHURN_MODEL_COLUMNS)


@app.post("/predict_cltv")
def predict_cltv(data: CLTVData, db: Session = Depends(get_db)):
    try:
        input_df = pd.DataFrame([{
            "Tenure Months": data.tenure_months,
            "Monthly Charges": data.monthly_charges,
            "Total Charges": data.total_charges
        }])

        prediction = cltv_model.predict(input_df)[0]

        record = CustomerPrediction(
            tenure_months=data.tenure_months,
            monthly_charges=data.monthly_charges,
            total_charges=data.total_charges,
            predicted_cltv=float(prediction)
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "predicted_cltv": float(prediction),
            "record_id": record.id
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_churn")
def predict_churn(data: ChurnData, db: Session = Depends(get_db)):
    try:
        input_df = build_churn_input_row(data)

        prediction = int(churn_model.predict(input_df)[0])
        probability = float(churn_model.predict_proba(input_df)[0][1])

        record = CustomerChurnPrediction(
            tenure_months=data.tenure_months,
            monthly_charges=data.monthly_charges,
            total_charges=data.total_charges,
            contract=data.contract,
            internet_service=data.internet_service,
            predicted_churn=prediction,
            churn_probability=round(probability, 4)
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "predicted_churn": prediction,
            "churn_probability": round(probability, 4),
            "record_id": record.id
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_cltv_batch")
def predict_cltv_batch(data: CLTVBatchData, db: Session = Depends(get_db)):
    results = []

    for customer in data.customers:
        try:
            input_df = pd.DataFrame([{
                "Tenure Months": customer.tenure_months,
                "Monthly Charges": customer.monthly_charges,
                "Total Charges": customer.total_charges
            }])

            prediction = cltv_model.predict(input_df)[0]

            record = CustomerPrediction(
                tenure_months=customer.tenure_months,
                monthly_charges=customer.monthly_charges,
                total_charges=customer.total_charges,
                predicted_cltv=float(prediction)
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            results.append({
                "predicted_cltv": float(prediction),
                "record_id": record.id
            })

        except Exception as e:
            db.rollback()
            results.append({"error": str(e)})

    return {"results": results, "count": len(results)}


@app.post("/predict_churn_batch")
def predict_churn_batch(data: ChurnBatchData, db: Session = Depends(get_db)):
    results = []

    for customer in data.customers:
        try:
            input_df = build_churn_input_row(customer)

            prediction = int(churn_model.predict(input_df)[0])
            probability = float(churn_model.predict_proba(input_df)[0][1])

            record = CustomerChurnPrediction(
                tenure_months=customer.tenure_months,
                monthly_charges=customer.monthly_charges,
                total_charges=customer.total_charges,
                contract=customer.contract,
                internet_service=customer.internet_service,
                predicted_churn=prediction,
                churn_probability=round(probability, 4)
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            results.append({
                "predicted_churn": prediction,
                "churn_probability": round(probability, 4),
                "record_id": record.id
            })

        except Exception as e:
            db.rollback()
            results.append({"error": str(e)})

    return {"results": results, "count": len(results)}