import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, target_col="Churn"):
    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fill missing values
    df = df.fillna(method="ffill")

    # Encode target
    y = df[target_col].apply(lambda x: 1 if x == "Yes" else 0)
    X = df.drop(target_col, axis=1)

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Scale
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    return X, y, label_encoders, scaler
