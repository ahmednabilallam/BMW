import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# PAGE CONFIGURATION

st.set_page_config(page_title="BMW Used Car Price Prediction", layout="wide")
st.title("🚗 BMW Used Car Price Prediction (2010–2024)")
st.markdown("#### Predict used BMW car prices and visualize historical trends")

# LOAD DATASET

data_path = r"C:\Users\anallam\Desktop\Project\BMW_sales_2010_2024.csv.csv"

try:
    df = pd.read_csv(data_path)
    st.success(" Dataset loaded successfully.")
except FileNotFoundError:
    st.error(" Dataset not found. Please check the file path.")
    st.stop()


# DATA CLEANING

df.columns = df.columns.str.strip()

# Remove duplicates and nulls
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Ensure numeric columns
numeric_cols = ['Price_USD', 'Engine_Size_L', 'Mileage_KM', 'Car_Age']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=numeric_cols, inplace=True)
# FEATURE ENGINEERING
df['Mileage_per_Year'] = df['Mileage_KM'] / (df['Car_Age'] + 1)
df['Price_per_EngineSize'] = df['Price_USD'] / df['Engine_Size_L']
# MACHINE LEARNING SECTION
@st.cache_resource
def train_and_evaluate_models(df):
    features = ['Engine_Size_L', 'Mileage_KM', 'Car_Age', 'Mileage_per_Year',
                'Price_per_EngineSize', 'Model', 'Region', 'Fuel_Type', 'Transmission']
    target = 'Price_USD'
    # One-hot encoding for categorical features
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    X = df_encoded
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # -------- Random Forest --------
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    # -------- XGBoost --------
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    # Comparison table
    results = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'MAE': [rf_mae, xgb_mae],
        'RMSE': [rf_rmse, xgb_rmse],
        'R2_Score': [rf_r2, xgb_r2]
    })
    # Select best model automatically
    best_model = xgb_model if xgb_r2 > rf_r2 else rf_model

    return best_model, scaler, results, X_test_scaled, y_test

best_model, scaler, results, X_test_scaled, y_test = train_and_evaluate_models(df)
# MODEL EVALUATION
st.header(" Model Evaluation Results")

if st.checkbox("Show Model Comparison Table"):
    st.dataframe(results.style.format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2_Score': '{:.4f}'}))
    st.info("The best performing model was automatically selected for predictions.")
# PRICE PREDICTION SECTION
st.header(" Predict BMW Car Price")

col1, col2, col3 = st.columns(3)
with col1:
    engine = st.number_input("Engine Size (L)", 1.0, 6.0, 2.0, step=0.1)
    mileage = st.number_input("Mileage (KM)", 0, 300000, 50000, step=1000)
with col2:
    age = st.slider("Car Age (years)", 0, 15, 5)
    region = st.selectbox("Region", df['Region'].unique())
with col3:
    fuel = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
    transmission = st.selectbox("Transmission", df['Transmission'].unique())
if st.button("Predict Price"):
    mileage_per_year = mileage / (age + 1)
    price_per_engine = 0  
    input_data = pd.DataFrame({
        'Engine_Size_L': [engine],
        'Mileage_KM': [mileage],
        'Car_Age': [age],
        'Mileage_per_Year': [mileage_per_year],
        'Price_per_EngineSize': [price_per_engine]
    })
    # Add encoded categorical columns
    model_df = pd.get_dummies(df[['Model', 'Region', 'Fuel_Type', 'Transmission']], drop_first=True)
    input_encoded = pd.DataFrame(np.zeros((1, model_df.shape[1])), columns=model_df.columns)

    for col in input_encoded.columns:
        if region in col:
            input_encoded[col] = 1
        if fuel in col:
            input_encoded[col] = 1
        if transmission in col:
            input_encoded[col] = 1
    final_input = pd.concat([input_data, input_encoded], axis=1)
    final_input = final_input.reindex(columns=model_df.columns.union(input_data.columns, sort=False), fill_value=0)
    final_input_scaled = scaler.transform(final_input)
    predicted_price = best_model.predict(final_input_scaled)[0]
    st.success(f"💵 Estimated Price: **${predicted_price:,.2f} USD**")
# VISUALIZATION SECTION
st.header(" Price Trends (2010–2024)")
avg_price_per_year = df.groupby('Year')['Price_USD'].mean()
fig, ax = plt.subplots()
ax.plot(avg_price_per_year.index, avg_price_per_year.values, marker='o')
ax.set_title("Average BMW Price per Year (2010–2024)")
ax.set_xlabel("Year")
ax.set_ylabel("Average Price (USD)")
st.pyplot(fig)
st.caption("Developed for educational purposes – BMW Used Car Price Prediction Project.")
