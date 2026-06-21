# customer-churn-ltv-engine
Customer Churn Prediction and Lifetime Value (LTV) Engine using Machine Learning, SHAP, FastAPI, and PostgreSQL.

# Customer Churn Prediction & Lifetime Value Engine

## Project Overview

This project predicts customer churn and estimates Customer Lifetime Value (LTV) using machine learning techniques. The goal is to help businesses identify customers likely to leave and prioritize retention strategies, enabling marketing teams to focus spend on the most valuable at-risk customers.

## Dataset

* Telco Customer Churn Dataset
* Source: IBM / Kaggle
* ~7,043 customers, including tenure, monthly charges, contract type, and service usage details

## Technologies Used

* Python, SQL
* Pandas, NumPy
* Scikit-Learn, XGBoost
* SHAP (model explainability)
* FastAPI
* PostgreSQL, SQLAlchemy
* Metabase (dashboard)
* Docker, Docker Compose

## Architecture
## Models

### CLTV Regression
- **Algorithm**: Random Forest Regressor (scikit-learn Pipeline with StandardScaler)
- **Features**: Tenure Months, Monthly Charges, Total Charges
- **Saved as**: `models/cltv_pipeline.pkl`

### Churn Classification
- **Algorithm**: Random Forest Classifier
- **Features**: 30 columns (numeric + one-hot encoded categorical), explicitly excluding non-predictive identifiers (CustomerID, Count, Zip Code, Latitude, Longitude) and label-leakage columns (Churn Label, Churn Score, CLTV, Churn Reason)
- **Performance**:

| Metric | Score |
|---|---|
| Accuracy | 0.792 |
| Precision | 0.675 |
| Recall | 0.515 |
| F1 Score | 0.584 |

- **Saved as**: `models/random_forest_model.pkl`
- **Note**: Recall is moderate; class-weighting or threshold tuning is a natural next iteration, since missed churners carry higher business cost than false alarms.

## Model Explainability (SHAP)

SHAP was applied to the Random Forest churn model to surface which features drive individual and global predictions, supporting non-technical stakeholder review.

- Global feature importance: `reports/shap_summary_plot.png`
- Individual customer explanation: `reports/shap_waterfall_customer0.png`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict_cltv` | POST | Single-customer CLTV prediction |
| `/predict_churn` | POST | Single-customer churn prediction with probability |
| `/predict_cltv_batch` | POST | Batch CLTV predictions for a list of customers |
| `/predict_churn_batch` | POST | Batch churn predictions for a list of customers |

All predictions are persisted to PostgreSQL with a unique `record_id` for traceability.

### Example — `/predict_churn`

Request:
```json
{
  "tenure_months": 5,
  "monthly_charges": 85,
  "total_charges": 425,
  "gender": "Female",
  "senior_citizen": "No",
  "partner": "No",
  "dependents": "No",
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check"
}
```

Response:
```json
{
  "predicted_churn": 1,
  "churn_probability": 0.82,
  "record_id": 1
}
```

## Database Schema

### `customer_predictions` (CLTV)
id, tenure_months, monthly_charges, total_charges, predicted_cltv, created_at

### `customer_churn_predictions`
id, tenure_months, monthly_charges, total_charges, contract, internet_service, predicted_churn, churn_probability, created_at

## Dashboard

The Metabase dashboard ("Customer Churn & LTV Dashboard") includes four visualizations built on predictions for the full ~7,032-customer dataset:

1. **Global Churn Risk Distribution** — 25.3% predicted to churn, consistent with this dataset's known ~26–27% actual churn rate
2. **LTV Segmentation** — customers grouped into predicted-CLTV bands
3. **Churn Risk by Contract Type** — month-to-month contracts are the dominant churn-risk segment
4. **Churn Risk by Tenure** — new (low-tenure) customers are a high-risk segment

## Project Structure

* `data/` : Raw and processed datasets
* `notebooks/` : EDA, feature engineering, model training, SHAP
* `src/` : FastAPI app, database models, training scripts
* `models/` : Saved machine learning models (.pkl)
* `reports/` : SHAP plots and project reports
* `Dockerfile`, `docker-compose.yml` : Containerization config

## Project Timeline

* **Week 1**: Data Ingestion & EDA
* **Week 2**: Feature Engineering & Modeling (Logistic Regression, Random Forest, XGBoost, SHAP)
* **Week 3**: LTV Prediction & FastAPI Development (single + batch endpoints)
* **Week 4**: Metabase Dashboard & Docker Deployment

## Running Locally (without Docker)

1. Create `churn_ltv_db` in PostgreSQL
2. Create `.env` with database credentials (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)
3. Install dependencies: `pip install -r requirements.txt`
4. Create tables: `python -m src.create_tables`
5. Start the API: `uvicorn src.api.app:app --reload`
6. (Optional) Load the full dataset: `python -m src.load_predictions`

## Running with Docker

```bash
docker-compose up --build
```

> **Note**: Docker configuration has been written and reviewed for correctness but not runtime-verified in this development environment due to local hardware/installation constraints. The application has been fully tested running natively (see Testing Evidence below).

## Testing Evidence

* Single-prediction endpoints tested with high-risk and low-risk customer profiles (82% vs 4% churn probability — directionally correct)
* Batch endpoints tested at full scale: 7,032 customers processed successfully through both batch endpoints with zero failures
* Database persistence verified via direct SQL queries after each major testing phase

## Known Issues & Future Work

* Recall (0.515) on the churn classifier is moderate; class-weighting or threshold tuning would likely improve true-churner detection
* scikit-learn version mismatch between training (1.6.1) and local environment (1.7.2) produces a non-fatal warning at model load time; `requirements.txt` pins the matching version
* Docker has not been runtime-tested in this environment
* Future work: API authentication, CI/CD for automated retraining, combining SHAP output with the dataset's Churn Reason field for richer business narratives

## Author

Vinishyamala P