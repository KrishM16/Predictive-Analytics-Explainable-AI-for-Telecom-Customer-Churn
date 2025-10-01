import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import preprocess_data

def evaluate_model(model_path, data_path="data/telco_customer_churn.csv"):
    df = pd.read_csv(data_path)

    X, y, _, _ = preprocess_data(df)

    model = joblib.load(model_path)
    preds = model.predict(X)

    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds))
