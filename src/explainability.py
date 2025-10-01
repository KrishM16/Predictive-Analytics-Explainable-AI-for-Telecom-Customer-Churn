import shap
import lime
import lime.lime_tabular
import pandas as pd
import joblib

from data_preprocessing import preprocess_data

def explain_with_shap(model_path, data_path="data/telco_customer_churn.csv"):
    df = pd.read_csv(data_path)
    X, y, _, _ = preprocess_data(df)

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)

def explain_with_lime(model_path, data_path="data/telco_customer_churn.csv"):
    df = pd.read_csv(data_path)
    X, y, _, _ = preprocess_data(df)

    model = joblib.load(model_path)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns,
        class_names=["No", "Yes"],
        mode="classification"
    )

    i = 0
    exp = explainer.explain_instance(X.iloc[i].values, model.predict_proba, num_features=10)
    exp.show_in_notebook(show_table=True)
