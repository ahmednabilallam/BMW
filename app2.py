# ==============================================================
# BMW PRICE PREDICTION PROJECT (2010–2026)
# FINAL VERSION – Runs Automatically Without File Upload
# ==============================================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime, os, warnings

warnings.filterwarnings('ignore')

# I. DATA LOADING & CLEANING PHASE

@st.cache_data
def load_and_clean_data():
    # Get current directory of the Streamlit app
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "BMW_sales_2010_2024.csv.csv")

    # Load dataset
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    # Clean Data
    for c in df.select_dtypes('object'):
        df[c].fillna(df[c].mode()[0], inplace=True)
    for c in df.select_dtypes(np.number):
        df[c].fillna(df[c].median(), inplace=True)
    df.drop_duplicates(inplace=True)

    return df
# II. FEATURE ENGINEERING & ENCODING

@st.cache_data
def feature_engineering_and_encoding(df):
    df_processed = df.copy()
    current_year = datetime.datetime.now().year

    # Feature Engineering
    df_processed['Car_Age'] = current_year - df_processed['Year']
    df_processed['Mileage_per_Year'] = df_processed['Mileage_KM'] / df_processed['Car_Age'].replace(0, 1)
    df_processed['Price_per_EngineSize'] = df_processed['Price_USD'] / df_processed['Engine_Size_L']

    # Encoding
    cat_cols = ['Model', 'Region', 'Fuel_Type', 'Transmission']
    encoders = {}
    for col in cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_processed[col] = oe.fit_transform(df_processed[[col]])[:, 0].astype(int)
        encoders[col] = oe
    return df_processed, df.copy(), encoders, cat_cols
# III. MODEL TRAINING & EVALUATION
@st.cache_resource
def train_and_evaluate_model(df):
    features = [
        'Engine_Size_L', 'Mileage_KM', 'Car_Age', 'Mileage_per_Year',
        'Price_per_EngineSize', 'Model', 'Region', 'Fuel_Type', 'Transmission'
    ]
    target = 'Price_USD'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, scaler, r2, mae, rmse, y_test, X_test_scaled, features, target
# IV. APP EXECUTION & INTERFACE
df_raw = load_and_clean_data()
df, df_unencoded, encoders, cat_cols = feature_engineering_and_encoding(df_raw)
model, scaler, r2, mae, rmse, y_test, X_test_scaled, features, target = train_and_evaluate_model(df)
st.title("BMW Price Prediction System (2010–2026)")
st.write("Predict used BMW car prices using a regression model trained on real-world sales data.")
st.markdown("---")
col1_info, col2_info = st.columns(2)
col1_info.metric("Dataset Rows", f"{df.shape[0]:,}")
col2_info.metric("Algorithm", "Random Forest Regression")
# SIDEBAR USER INPUT
st.sidebar.header("Enter Car Specifications")
year = st.sidebar.number_input("Year", 2010, 2026, 2024)
engine = st.sidebar.number_input("Engine Size (L)", 1.0, 6.0, 2.0, 0.1)
mileage = st.sidebar.number_input("Mileage (KM)", 0, 500000, 50000)
model_name = st.sidebar.selectbox("Model", df_unencoded['Model'].unique())
region = st.sidebar.selectbox("Region", df_unencoded['Region'].unique())
fuel = st.sidebar.selectbox("Fuel Type", df_unencoded['Fuel_Type'].unique())
trans = st.sidebar.selectbox("Transmission", df_unencoded['Transmission'].unique())
st.sidebar.markdown("---")
# PRICE PREDICTION (CURRENT)
st.subheader("Current Price Prediction")
if st.sidebar.button("Predict Current Price"):
    current_year = datetime.datetime.now().year
    car_age = max(current_year - year, 0)
    mph = mileage / max(car_age, 1)
    price_per_engine = df['Price_USD'].mean() / engine

    input_df = pd.DataFrame([{
        'Engine_Size_L': engine, 'Mileage_KM': mileage, 'Car_Age': car_age,
        'Mileage_per_Year': mph, 'Price_per_EngineSize': price_per_engine,
        'Model': model_name, 'Region': region, 'Fuel_Type': fuel, 'Transmission': trans
    }])
    for c in cat_cols:
        input_df[c] = encoders[c].transform(input_df[[c]])[:, 0]

    input_scaled = scaler.transform(input_df)
    pred_price = model.predict(input_scaled)[0]
    st.metric("Predicted Price", f"${pred_price:,.0f} USD", delta=f"R²: {r2:.2f}")

# FUTURE PRICE FORECAST (2024–2026)
st.markdown("---")
st.subheader("Future Price Projection (2024–2026)")

if st.sidebar.button("Predict Future Prices"):
    selected_year = year
    future_years = list(range(selected_year, 2027))
    predicted_prices = []
    current_year = datetime.datetime.now().year
    initial_car_age = max(current_year - selected_year, 1)
    yearly_mileage_rate = mileage / initial_car_age
    for y in future_years:
        new_total_mileage = mileage + (yearly_mileage_rate * (y - selected_year))
        car_age_future = max(current_year - y, 0)
        mph_future = new_total_mileage / max(car_age_future, 1)
        price_per_engine_future = df['Price_USD'].mean() / engine

        future_df = pd.DataFrame([{
            'Engine_Size_L': engine, 'Mileage_KM': new_total_mileage,
            'Car_Age': car_age_future, 'Mileage_per_Year': mph_future,
            'Price_per_EngineSize': price_per_engine_future,
            'Model': model_name, 'Region': region, 'Fuel_Type': fuel, 'Transmission': trans
        }])
        for c in cat_cols:
            future_df[c] = encoders[c].transform(future_df[[c]])[:, 0]
        future_scaled = scaler.transform(future_df)
        predicted_prices.append(model.predict(future_scaled)[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(future_years, predicted_prices, marker='o', color='green')
    ax.set_title(f"Predicted Price Trend ({selected_year}–2026)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Price (USD)")
    st.pyplot(fig)
# V. VISUALIZATIONS
st.markdown("---")
st.header("Model Evaluation and Insights")
col_mae, col_rmse, col_r2 = st.columns(3)
col_mae.metric("MAE", f"{mae:,.2f}")
col_rmse.metric("RMSE", f"{rmse:,.2f}")
col_r2.metric("R²", f"{r2:.4f}")
st.markdown("---")
st.subheader("Data Visualizations")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df_unencoded['Price_USD'], bins=25, kde=True, color='skyblue', ax=ax)
ax.set_title("Price Distribution (USD)")
st.pyplot(fig)
st.markdown("---")
st.info(" Project executed successfully without file upload requirement.")
