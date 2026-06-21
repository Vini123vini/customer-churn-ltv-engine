from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.sql import func

from src.database import Base


class CustomerPrediction(Base):
    __tablename__ = "customer_predictions"

    id = Column(Integer, primary_key=True, index=True)
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    predicted_cltv = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CustomerChurnPrediction(Base):
    __tablename__ = "customer_churn_predictions"

    id = Column(Integer, primary_key=True, index=True)
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    contract = Column(String, nullable=False)
    internet_service = Column(String, nullable=False)
    predicted_churn = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())