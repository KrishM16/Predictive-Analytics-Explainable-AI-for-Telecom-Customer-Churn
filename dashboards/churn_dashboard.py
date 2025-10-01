import streamlit as st
import pandas as pd
import pickle
import shap

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Prediction Dashboard")

# Load model
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose model", ["XGBoost", "RandomForest", "LightGBM"])

model_path = f"models/{'xgb' if model_choice=='XGBoost' else 'rf' if model_choice=='RandomForest' else 'lgb'}_churn_model.pkl"

try:
    model = pickle.load(open(model_path, "rb"))
except:
    st.warning("⚠️ Train models first using `train_models.py`")
    model = None

# Upload data
uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded_file and model:
    data = pd.read_csv(uploaded_file)
    st.write("### Customer Data", data.head())

    preds = model.predict(data)
    data["Churn_Prediction"] = preds
    st.write("### Predictions", data.head())

    # SHAP explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        st.write("### Feature Importance (SHAP)")
        st.pyplot(shap.summary_plot(shap_values, data, show=False))
    except:
        st.info("SHAP explanation available only for tree-based models")
