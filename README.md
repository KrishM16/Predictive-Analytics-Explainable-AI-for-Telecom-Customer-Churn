# Predictive Analytics & Explainable AI for Telecom Customer Churn

Customer churn is a major revenue risk for telecom companies. This project predicts customers at risk of leaving and explains the reasons using ML & XAI.

## Project Highlights
- End-to-end ML pipeline: EDA, feature engineering, modeling
- Models: Random Forest, XGBoost, LightGBM
- Explainability: SHAP, LIME
- Interactive Streamlit dashboard for predictions

## Dataset
Kaggle: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, LIME, Streamlit, Matplotlib, Seaborn

## Quick Start
1. Place dataset CSV in `data/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `notebooks/Customer_Churn_Prediction.ipynb`
4. Launch dashboard: `streamlit run dashboards/churn_dashboard.py`

