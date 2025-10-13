import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("BMW_Price_Predictor.pkl")

st.title("BMW Price Prediction 🚗💰")
st.write("Enter the car specifications to get the predicted price in USD")

# Sidebar / Inputs
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2024, value=2020)
engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=6.0, value=2.0, step=0.1)
mileage = st.number_input("Mileage (KM)", min_value=0, max_value=500000, value=20000)
model_name = st.selectbox("Car Model", ["3 Series", "5 Series", "X5", "X3", "7 Series"])
region = st.selectbox("Region", ["North America", "Europe", "Asia", "Other"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])

# Feature Engineering
car_age = 2024 - year
mileage_per_year = mileage / max(car_age, 1)

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'Engine_Size_L': engine_size,
    'Mileage_KM': mileage,
    'Car_Age': car_age,
    'Mileage_per_Year': mileage_per_year,
    'Model': model_name,
    'Region': region,
    'Fuel_Type': fuel_type
}])

# Predict Button
if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${price:,.2f}")
