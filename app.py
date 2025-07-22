import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="🏠 House Price Predictor", page_icon=":house:", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Estimate the sale price of a house based on key features</p>", unsafe_allow_html=True)

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f8f9fa;
            color: #333333;
        }

        .main-container {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin: auto;
            max-width: 800px;
        }

        .stButton > button {
            background-color: #0d6efd;
            color: white;
            font-weight: 500;
            padding: 0.5em 1.5em;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #084298;
        }

        h1, h2, h3 {
            color: #0d6efd;
        }

        .footer {
            text-align: center;
            color: gray;
            font-size: 14px;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='main'>", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<small>Made with ❤️ by Kartikeya Mishra>", unsafe_allow_html=True)

model = joblib.load("xgb_model.pkl")

# User inputs 
OverallQual = st.selectbox("Overall Quality (1-10)", ["Select..."] + list(range(1, 11)))
GarageCars = st.selectbox("Garage Capacity (Cars)", ["Select..."] + list(range(0, 5)))
FullBath = st.selectbox("Full Bathrooms", ["Select..."] + list(range(0, 6)))
TotRmsAbvGrd = st.selectbox("Total Rooms Above Ground", ["Select..."] + list(range(1, 12)))
GrLivArea = st.slider("Above Ground Living Area (sqft)", 500, 4000, 0)
TotalBsmtSF = st.slider("Total Basement Area (sqft)", 0, 2500, 0)


input_df = pd.DataFrame({
    'GrLivArea': [GrLivArea],
    'OverallQual': [OverallQual],
    'GarageCars': [GarageCars],
    'TotalBsmtSF': [TotalBsmtSF],
    'FullBath': [FullBath],
    'TotRmsAbvGrd': [TotRmsAbvGrd]
})


if st.button("Predict Sale Price"):
    if "Select..." in [OverallQual, GarageCars, FullBath, TotRmsAbvGrd]:
        st.error("Please fill all dropdowns before predicting.")
    else:
        input_df = pd.DataFrame({
            'GrLivArea': [GrLivArea],
            'OverallQual': [int(OverallQual)],
            'GarageCars': [int(GarageCars)],
            'TotalBsmtSF': [TotalBsmtSF],
            'FullBath': [int(FullBath)],
            'TotRmsAbvGrd': [int(TotRmsAbvGrd)]
        })

       
        feature_labels = {
            'GrLivArea': 'Above Ground Living Area (sqft)',
            'OverallQual': 'Overall Quality (1-10)',
            'GarageCars': 'Garage Capacity (Cars)',
            'TotalBsmtSF': 'Total Basement Area (sqft)',
            'FullBath': 'Full Bathrooms',
            'TotRmsAbvGrd': 'Total Rooms Above Ground'
        }

        input_df_display = input_df.rename(columns=feature_labels)

        log_price = model.predict(input_df)[0]
        price = np.expm1(log_price)
        st.success(f"🏡 Predicted Sale Price: ${int(price):,}")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        shap_values.feature_names = [feature_labels.get(f, f) for f in input_df.columns]
        shap_values.data = input_df_display.values

        st.subheader("🔎 How Each Feature Affects Price")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=6, show=False)
        st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>&copy; 2025 House Price Predictor</div>", unsafe_allow_html=True)
