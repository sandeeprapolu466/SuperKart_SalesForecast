import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="sandeep466/superkart-sales-model",
    filename="best_superkart_sales_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("SuperKart Sales Forecasting App")
st.write("""
This application predicts product-level sales revenue based on
product and store attributes.
""")

# User inputs
product_weight = st.number_input("Product Weight", min_value=0.0)
product_area = st.number_input("Product Allocated Area", min_value=0.0)
product_mrp = st.number_input("Product MRP", min_value=0.0)
store_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025)

sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_type = st.selectbox("Product Type", ["Dairy", "Soft Drinks", "Meat", "Frozen Foods", "Household", "Others"])
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("Store City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])

# Prepare input dataframe
input_df = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Allocated_Area": product_area,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_year,
    "Product_Sugar_Content": sugar_content,
    "Product_Type": product_type,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type
}])

# Prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.subheader("Predicted Sales Revenue")
    st.success(f"₹ {prediction:.2f}")
