import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("xgb_model.pkl")

st.title("🏠 House Price Predictor")
st.write("Enter the house details below to get an estimated sale price.")

# User inputs 
GrLivArea = st.slider("Above Ground Living Area (sqft)", 500, 4000, 1500)
OverallQual = st.selectbox("Overall Quality (1-10)", list(range(1, 11)))
GarageCars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4])
TotalBsmtSF = st.slider("Total Basement Area (sqft)", 0, 2500, 800)
FullBath = st.selectbox("Full Bathrooms", [0, 1, 2, 3])
TotRmsAbvGrd = st.selectbox("Total Rooms Above Ground", list(range(3, 12)))

input_df = pd.DataFrame({
    'GrLivArea': [GrLivArea],
    'OverallQual': [OverallQual],
    'GarageCars': [GarageCars],
    'TotalBsmtSF': [TotalBsmtSF],
    'FullBath': [FullBath],
    'TotRmsAbvGrd': [TotRmsAbvGrd]
})


if st.button("Predict Sale Price"):
    log_price = model.predict(input_df)[0]
    sale_price = np.expm1(log_price)
    st.success(f"🏡 Estimated Sale Price: **${sale_price:,.0f}**")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    st.subheader("Feature Contribution to Prediction:")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)