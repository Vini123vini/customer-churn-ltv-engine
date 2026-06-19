from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Customer Churn & CLTV API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}