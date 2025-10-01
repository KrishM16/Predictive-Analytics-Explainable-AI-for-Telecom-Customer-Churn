import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

from data_preprocessing import preprocess_data

def train_and_save_models(data_path="data/telco_customer_churn.csv"):
    df = pd.read_csv(data_path)

    X, y, encoders, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf_churn_model.pkl")

    # XGBoost
    xgb_model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "models/xgb_churn_model.pkl")

    # LightGBM
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)
    joblib.dump(lgb_model, "models/lgb_churn_model.pkl")

    print("Models trained & saved in models/ folder")
