# ============================================================
# Churn Prediction API — FastAPI Deployment
# Built by Gift | Grow With Me Challenge — Day 2
# ============================================================
# HOW TO RUN LOCALLY:
#   pip install fastapi uvicorn pydantic xgboost scikit-learn
#   uvicorn app:app --reload --port 8000
#
# THEN TEST AT:
#   http://localhost:8000/docs  ← interactive API docs
# ============================================================

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

# ── Load model once at startup ──────────────────────────────
with open("churn_model.pkl", "rb") as f:
    model, FEATURES = pickle.load(f)

# ── App setup ───────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability and get recommended retention action.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ───────────────────────────────────────────
class CustomerData(BaseModel):
    customer_id: str = Field(..., example="CUST0042")
    age: int = Field(..., ge=18, le=100, example=34)
    gender: Literal["Male", "Female"] = Field(..., example="Female")
    region: Literal["North", "South", "East", "West"] = Field(..., example="South")
    plan_type: Literal["Basic", "Standard", "Premium"] = Field(..., example="Basic")
    tenure_months: int = Field(..., ge=0, example=8)
    days_since_last_purchase: int = Field(..., ge=0, le=365, example=52)
    purchase_frequency: float = Field(..., ge=0, example=1.5)
    avg_order_value: float = Field(..., ge=0, example=45.0)
    total_spend: float = Field(..., ge=0, example=180.0)
    login_frequency_monthly: float = Field(..., ge=0, example=1.0)
    email_open_rate: float = Field(..., ge=0.0, le=1.0, example=0.05)
    pages_per_session: float = Field(..., ge=0, example=1.5)
    support_tickets: int = Field(..., ge=0, example=2)
    avg_resolution_days: float = Field(..., ge=0, example=4.5)
    complaints: int = Field(..., ge=0, example=1)

# ── Response schema ──────────────────────────────────────────
class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_percentage: str
    risk_tier: str
    risk_color: str
    recommended_action: str
    top_risk_factors: list
    predicted_at: str

# ── Feature engineering (matches training pipeline) ──────────
def build_features(c: CustomerData) -> list:
    from sklearn.preprocessing import LabelEncoder

    rfm = (
        (1 - c.days_since_last_purchase / 120) * 0.40 +
        min(c.purchase_frequency / 20, 1.0) * 0.35 +
        min(c.avg_order_value / 300, 1.0) * 0.25
    )
    engagement = (
        min(c.login_frequency_monthly / 30, 1.0) * 0.50 +
        c.email_open_rate * 0.30 +
        min(c.pages_per_session / 15, 1.0) * 0.20
    )
    support_risk = c.complaints * 2 + min(c.avg_resolution_days / 14, 1.0)

    gender_enc   = {"Male": 1, "Female": 0}[c.gender]
    region_enc   = {"East": 0, "North": 1, "South": 2, "West": 3}[c.region]
    plan_enc     = {"Basic": 0, "Premium": 1, "Standard": 2}[c.plan_type]

    return [
        c.age, c.tenure_months, c.days_since_last_purchase, c.purchase_frequency,
        c.avg_order_value, c.total_spend, c.login_frequency_monthly, c.email_open_rate,
        c.pages_per_session, c.support_tickets, c.avg_resolution_days, c.complaints,
        rfm, engagement, support_risk,
        gender_enc, region_enc, plan_enc
    ]

def get_risk_factors(c: CustomerData) -> list:
    factors = []
    if c.days_since_last_purchase > 45:
        factors.append(f"No purchase in {c.days_since_last_purchase} days")
    if c.login_frequency_monthly < 2:
        factors.append("Very low platform engagement")
    if c.complaints > 0:
        factors.append(f"{c.complaints} unresolved complaint(s)")
    if c.email_open_rate < 0.10:
        factors.append("Email open rate below 10%")
    if c.purchase_frequency < 2:
        factors.append("Low purchase frequency")
    if c.avg_resolution_days > 3:
        factors.append(f"Slow support resolution ({c.avg_resolution_days} days avg)")
    return factors[:3] if factors else ["No critical risk signals detected"]

# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api": "Churn Prediction API",
        "version": "1.0.0",
        "status": "live",
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET  /health",
            "docs":    "GET  /docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerData):
    try:
        features = build_features(customer)
        prob = float(model.predict_proba([features])[0][1])

        if prob >= 0.75:
            tier   = "High Risk"
            color  = "red"
            action = "Urgent: send personalised discount + assign account manager"
        elif prob >= 0.40:
            tier   = "Medium Risk"
            color  = "amber"
            action = "Re-engagement: trigger 3-email nurture sequence with soft offer"
        else:
            tier   = "Low Risk"
            color  = "green"
            action = "Retention: send loyalty reward + introduce upsell opportunity"

        return ChurnPrediction(
            customer_id       = customer.customer_id,
            churn_probability = round(prob, 4),
            churn_percentage  = f"{prob*100:.1f}%",
            risk_tier         = tier,
            risk_color        = color,
            recommended_action= action,
            top_risk_factors  = get_risk_factors(customer),
            predicted_at      = datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
