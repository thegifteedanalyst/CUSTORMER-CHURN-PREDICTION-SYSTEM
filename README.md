## Customer Churn Prediction System
A deployable ML system that predicts which customers are about to leave your business — before they do. Input a customer's data, get back a churn probability score and automated retention action in real time.

## What it does

1. Scores every customer with a churn probability (0 to 1)
2.Segments them into High, Medium, and Low risk tiers
3.Recommends the right action for each — discount, email sequence, or loyalty reward
4.Runs as a live REST API — plug it into any CRM, Shopify store, or email platform


Quickstart
bash# Install dependencies
pip install -r requirements.txt

# Train the model (generates churn_model.pkl)
python churn_prediction_system.py

# Start the API
uvicorn app:app --reload --port 8000

# Open interactive docs in browser
http://localhost:8000/docs

Example response
json{
  "customer_id": "CUST0042",
  "churn_probability": 0.9698,
  "churn_percentage": "97.0%",
  "risk_tier": "High Risk",
  "recommended_action": "Urgent: personalised discount + assign account manager",
  "top_risk_factors": [
    "No purchase in 52 days",
    "Very low platform engagement",
    "1 complaint raised"
  ]
}

Deploy to cloud (free)
bash# Push to GitHub, then go to render.com
# New → Web Service → Connect repo
# Start command:
uvicorn app:app --host 0.0.0.0 --port 8000
Live public URL ready in ~2 minutes. No credit card needed.

Tech stack
XGBoost · FastAPI · Pandas · scikit-learn · Uvicorn

