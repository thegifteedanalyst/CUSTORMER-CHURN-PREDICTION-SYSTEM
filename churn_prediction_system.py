# ============================================================
# Customer Churn Prediction System
# Built by Gift | Grow With Me Challenge — Day 2
# ============================================================
# HOW TO RUN:
#   pip install pandas numpy scikit-learn xgboost matplotlib
#   python churn_prediction_system.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv("Downloads/customer_churn_dataset.csv")
    print(f"[1] Data loaded: {df.shape[0]} customers, {df.shape[1]} features")
    print(f"    Churn rate: {df['churn'].mean()*100:.1f}%\n")
    return df


# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    # RFM Score: Recency + Frequency + Monetary
    df['rfm_score'] = (
        (1 - df['days_since_last_purchase'] / 120).clip(0, 1) * 0.40 +
        (df['purchase_frequency'] / 20).clip(0, 1) * 0.35 +
        (df['avg_order_value'] / 300).clip(0, 1) * 0.25
    )

    # Engagement Score: logins + email + browsing
    df['engagement_score'] = (
        (df['login_frequency_monthly'] / 30).clip(0, 1) * 0.50 +
        df['email_open_rate'] * 0.30 +
        (df['pages_per_session'] / 15).clip(0, 1) * 0.20
    )

    # Support Risk: complaints weighted heavier than tickets
    df['support_risk'] = (
        df['complaints'] * 2 +
        (df['avg_resolution_days'] / 14).clip(0, 1)
    )

    # Encode categorical columns
    le = LabelEncoder()
    for col in ['gender', 'region', 'plan_type']:
        df[col + '_enc'] = le.fit_transform(df[col])

    print("[2] Features engineered: rfm_score, engagement_score, support_risk\n")
    return df


# ─────────────────────────────────────────────
# STEP 3: TRAIN THE MODEL
# ─────────────────────────────────────────────
FEATURES = [
    'age', 'tenure_months', 'days_since_last_purchase', 'purchase_frequency',
    'avg_order_value', 'total_spend', 'login_frequency_monthly', 'email_open_rate',
    'pages_per_session', 'support_tickets', 'avg_resolution_days', 'complaints',
    'rfm_score', 'engagement_score', 'support_risk',
    'gender_enc', 'region_enc', 'plan_type_enc'
]

def train_model(df):
    X = df[FEATURES]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print("[3] Model trained (XGBoost classifier)")
    print(f"    AUC-ROC: {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

    # Save model to disk
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump((model, FEATURES), f)
    print("    Model saved to churn_model.pkl\n")

    return model


# ─────────────────────────────────────────────
# STEP 4: SCORE ALL CUSTOMERS
# ─────────────────────────────────────────────
def score_customers(df, model):
    df['churn_probability'] = model.predict_proba(df[FEATURES])[:, 1].round(4)
    df['risk_tier'] = pd.cut(
        df['churn_probability'],
        bins=[0, 0.40, 0.75, 1.0],
        labels=['Low risk', 'Medium risk', 'High risk']
    )
    df['recommended_action'] = df['risk_tier'].map({
        'High risk':    'Urgent outreach — personalised discount + direct call',
        'Medium risk':  'Re-engagement email sequence + soft offer',
        'Low risk':     'Loyalty reward + upsell opportunity'
    })

    print("[4] All customers scored and segmented")
    print(df['risk_tier'].value_counts().to_string())
    print()
    return df


# ─────────────────────────────────────────────
# STEP 5: PREDICT A SINGLE CUSTOMER (DEMO)
# ─────────────────────────────────────────────
def predict_single_customer(model, customer_data: dict):
    """
    Pass in a dictionary of raw customer values.
    Returns churn probability, risk tier, and recommended action.
    """
    df_single = pd.DataFrame([customer_data])

    # Apply same feature engineering
    df_single['rfm_score'] = (
        (1 - df_single['days_since_last_purchase'] / 120).clip(0, 1) * 0.40 +
        (df_single['purchase_frequency'] / 20).clip(0, 1) * 0.35 +
        (df_single['avg_order_value'] / 300).clip(0, 1) * 0.25
    )
    df_single['engagement_score'] = (
        (df_single['login_frequency_monthly'] / 30).clip(0, 1) * 0.50 +
        df_single['email_open_rate'] * 0.30 +
        (df_single['pages_per_session'] / 15).clip(0, 1) * 0.20
    )
    df_single['support_risk'] = (
        df_single['complaints'] * 2 +
        (df_single['avg_resolution_days'] / 14).clip(0, 1)
    )

    le = LabelEncoder()
    le.fit(['Male', 'Female'])
    df_single['gender_enc'] = le.transform(df_single['gender'])
    le.fit(['North', 'South', 'East', 'West'])
    df_single['region_enc'] = le.transform(df_single['region'])
    le.fit(['Basic', 'Standard', 'Premium'])
    df_single['plan_type_enc'] = le.transform(df_single['plan_type'])

    prob = model.predict_proba(df_single[FEATURES])[0][1]

    if prob >= 0.75:
        tier   = 'HIGH RISK'
        action = 'Urgent outreach — personalised discount + direct call'
    elif prob >= 0.40:
        tier   = 'MEDIUM RISK'
        action = 'Re-engagement email sequence + soft offer'
    else:
        tier   = 'LOW RISK'
        action = 'Loyalty reward + upsell opportunity'

    print("\n" + "="*50)
    print("  CHURN PREDICTION — SINGLE CUSTOMER")
    print("="*50)
    print(f"  Churn probability : {prob:.2%}")
    print(f"  Risk tier         : {tier}")
    print(f"  Recommended action: {action}")
    print("="*50 + "\n")
    return prob, tier, action


# ─────────────────────────────────────────────
# STEP 6: EXPORT RESULTS
# ─────────────────────────────────────────────
def export_results(df, output_path='customer_churn_scored.csv'):
    cols_to_export = [
        'customer_id', 'age', 'gender', 'region', 'plan_type', 'tenure_months',
        'days_since_last_purchase', 'purchase_frequency', 'avg_order_value',
        'total_spend', 'login_frequency_monthly', 'complaints',
        'rfm_score', 'engagement_score', 'support_risk',
        'churn_probability', 'risk_tier', 'recommended_action', 'churn'
    ]
    df[cols_to_export].sort_values('churn_probability', ascending=False).to_csv(
        output_path, index=False
    )
    print(f"[6] Scored dataset exported to {output_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # Run full pipeline
    df    = load_data('Downloads/customer_churn_dataset.csv')
    df    = engineer_features(df)
    model = train_model(df)
    df    = score_customers(df, model)
    export_results(df)

    # Demo: predict a single high-risk customer
    example_customer = {
        'age': 34,
        'gender': 'Female',
        'region': 'South',
        'plan_type': 'Basic',
        'tenure_months': 8,
        'days_since_last_purchase': 52,
        'purchase_frequency': 1.5,
        'avg_order_value': 45.0,
        'total_spend': 180.0,
        'login_frequency_monthly': 1.0,
        'email_open_rate': 0.05,
        'pages_per_session': 1.5,
        'support_tickets': 2,
        'avg_resolution_days': 4.5,
        'complaints': 1,
    }

    predict_single_customer(model, example_customer)
