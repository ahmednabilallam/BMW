import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import datetime, os
# Page config
st.set_page_config(page_title="BMW Price Prediction", layout="centered")
st.title(" BMW Price Prediction (2010-2026)")
# Load data
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "BMW_sales_2010_2024.csv")
    df = pd.read_csv(data_path)    
    # Basic cleaning
    for c in df.select_dtypes('object'):
        df[c].fillna(df[c].mode()[0], inplace=True)
    for c in df.select_dtypes(np.number):
        df[c].fillna(df[c].median(), inplace=True)
    df.drop_duplicates(inplace=True)   
    return df
# Feature engineering
@st.cache_data
def engineer_features(df):
    df_processed = df.copy()
    current_year = datetime.datetime.now().year    
    df_processed['Car_Age'] = current_year - df_processed['Year']
    df_processed['Mileage_per_Year'] = df_processed['Mileage_KM'] / df_processed['Car_Age'].replace(0, 1)
    df_processed['Price_per_EngineSize'] = df_processed['Price_USD'] / df_processed['Engine_Size_L']
    # Encode categorical variables
    cat_cols = ['Model', 'Region', 'Fuel_Type', 'Transmission']
    encoders = {}
    for col in cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_processed[col] = oe.fit_transform(df_processed[[col]])[:, 0].astype(int)
        encoders[col] = oe
    return df_processed, df.copy(), encoders, cat_cols
# Train models
@st.cache_resource
def train_models(df):
    features = ['Engine_Size_L', 'Mileage_KM', 'Car_Age', 'Mileage_per_Year', 'Price_per_EngineSize', 
                'Model', 'Region', 'Fuel_Type', 'Transmission']
    target = 'Price_USD'   
    X = df[features]
    y = df[target]   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)   
    rf_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)    
    # Predictions
    rf_pred = rf_model.predict(X_test_scaled)
    xgb_pred = xgb_model.predict(X_test_scaled)    
    rf_r2 = r2_score(y_test, rf_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)  
    return rf_model, xgb_model, scaler, rf_r2, xgb_r2, y_test, rf_pred, xgb_pred, X_test_scaled
# Load and process data
df_raw = load_data()
df, df_unencoded, encoders, cat_cols = engineer_features(df_raw)
rf_model, xgb_model, scaler, rf_r2, xgb_r2, y_test, rf_pred, xgb_pred, X_test_scaled = train_models(df)
# Sidebar inputs
st.sidebar.header("Car Specifications")
year = st.sidebar.number_input("Year", 2010, 2026, 2020)
engine = st.sidebar.slider("Engine Size (L)", 1.0, 6.0, 2.0, 0.1)
mileage = st.sidebar.number_input("Mileage (KM)", 0, 500000, 50000, 1000)
model_name = st.sidebar.selectbox("Model", df_unencoded['Model'].unique())
region = st.sidebar.selectbox("Region", df_unencoded['Region'].unique())
fuel = st.sidebar.selectbox("Fuel Type", df_unencoded['Fuel_Type'].unique())
trans = st.sidebar.selectbox("Transmission", df_unencoded['Transmission'].unique())
# Prediction function
def predict_price(year, engine, mileage, model_name, region, fuel, trans):
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
    rf_price = rf_model.predict(input_scaled)[0]
    xgb_price = xgb_model.predict(input_scaled)[0]    
    return rf_price, xgb_price, (rf_price + xgb_price) / 2
# Current price prediction
if st.sidebar.button("Predict Current Price"):
    rf_price, xgb_price, avg_price = predict_price(year, engine, mileage, model_name, region, fuel, trans)    
    st.success(f" Predicted Price: ${avg_price:,.2f} USD")   
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest", f"${rf_price:,.0f}")
    col2.metric("XGBoost", f"${xgb_price:,.0f}")
    col3.metric("Average", f"${avg_price:,.0f}")
# ENHANCED FUTURE PRICE PROJECTION
if st.sidebar.button("Predict Future Prices"):
    st.markdown("---")
    st.subheader(" Future Price Projection (2010-2026)")   
    # Get historical data (2010-2024)
    historical_years = list(range(2010, 2025))
    historical_prices = []    
    # Calculate average price for each historical year
    for y in historical_years:
        if y in df_unencoded['Year'].values:
            year_avg = df_unencoded[df_unencoded['Year'] == y]['Price_USD'].mean()
            historical_prices.append(year_avg)
        else:
            # If year not in data, use interpolation
            historical_prices.append(None)    
    # Fill missing years with interpolation
    historical_df = pd.DataFrame({'Year': historical_years, 'Price': historical_prices})
    historical_df['Price'] = historical_df['Price'].interpolate()    
    # Future prediction (2024-2026)
    future_years = list(range(2024, 2027))
    future_prices = []    
    current_year = datetime.datetime.now().year
    initial_car_age = max(current_year - year, 1)
    yearly_mileage_rate = mileage / initial_car_age    
    for y in future_years:
        new_mileage = mileage + (yearly_mileage_rate * (y - year))
        rf_price, xgb_price, avg_price = predict_price(y, engine, new_mileage, model_name, region, fuel, trans)
        future_prices.append(avg_price)    
    # Combine historical and future data
    all_years = list(range(2010, 2027))
    all_prices = list(historical_df['Price']) + future_prices[1:]  # Skip 2024 duplicate    
    # Create the enhanced chart
    fig, ax = plt.subplots(figsize=(14, 7))    
    # Plot historical data (2010-2024)
    historical_mask = [y <= 2024 for y in all_years]
    ax.plot([y for y in all_years if y <= 2024], 
            [p for y, p in zip(all_years, all_prices) if y <= 2024],
            marker='o', linewidth=3, markersize=6, color='#2E86AB', 
            label='Historical Data (2010-2024)', alpha=0.8)    
    # Plot future prediction (2024-2026)
    future_mask = [y >= 2024 for y in all_years]
    ax.plot([y for y in all_years if y >= 2024], 
            [p for y, p in zip(all_years, all_prices) if y >= 2024],
            marker='s', linewidth=3, markersize=8, color='#A23B72', 
            linestyle='--', label='Future Prediction (2024-2026)', alpha=0.9)   
    # Add vertical line to separate historical and future
    ax.axvline(x=2024.5, color='red', linestyle=':', alpha=0.7, label='Prediction Start')    
    # Customize the chart
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title('BMW Price Trend: Historical Data & Future Projection (2010-2026)', 
                fontsize=14, fontweight='bold', pad=20)    
    # Add value annotations for key points
    key_years = [2010, 2015, 2020, 2024, 2026]
    for y in key_years:
        if y in all_years:
            idx = all_years.index(y)
            price = all_prices[idx]
            ax.annotate(f'${price:,.0f}', 
                       (y, price),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=9,
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))    
    # Add car specifications to the chart
    specs_text = f"Car: {model_name}\nYear: {year}\nEngine: {engine}L\nMileage: {mileage:,} KM"
    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(2010, 2027, 2))    
    plt.tight_layout()
    st.pyplot(fig)    
    # Display price table
    st.subheader(" Price Summary Table")
    price_data = []
    for y, p in zip(all_years, all_prices):
        status = "Historical" if y <= 2024 else "Predicted"
        price_data.append({"Year": y, "Price (USD)": f"${p:,.2f}", "Status": status})   
    price_df = pd.DataFrame(price_data)
    st.dataframe(price_df, use_container_width=True)
# Rest of the visualizations (kept as before)
st.markdown("---")
st.header(" Data Analysis & Visualizations")
# Tab 1: Price Analysis
tab1, tab2, tab3 = st.tabs(["Price Trends", "Car Features", "Model Performance"])
with tab1:
    st.subheader("Price Distribution & Trends")    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))   
    ax1.hist(df_unencoded['Price_USD'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Price (USD)')
    ax1.set_ylabel('Number of Cars')
    ax1.set_title('Price Distribution')
    ax1.grid(True, alpha=0.3)   
    yearly_avg = df_unencoded.groupby('Year')['Price_USD'].mean()
    ax2.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, color='red')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Price (USD)')
    ax2.set_title('Price Trend (2010-2024)')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig1)
with tab2:
    st.subheader("Car Features Analysis")  
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))   
    ax1.scatter(df_unencoded['Engine_Size_L'], df_unencoded['Price_USD'], alpha=0.6, color='green')
    ax1.set_xlabel('Engine Size (L)')
    ax1.set_ylabel('Price (USD)')
    ax1.set_title('Engine Size vs Price')
    ax1.grid(True, alpha=0.3)
    ax2.scatter(df_unencoded['Mileage_KM'], df_unencoded['Price_USD'], alpha=0.6, color='orange')
    ax2.set_xlabel('Mileage (KM)')
    ax2.set_ylabel('Price (USD)')
    ax2.set_title('Mileage vs Price')
    ax2.grid(True, alpha=0.3)   
    st.pyplot(fig2)
with tab3:
    st.subheader("Model Performance")
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))   
    ax1.scatter(y_test, rf_pred, alpha=0.6, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('Actual Price')
    ax1.set_ylabel('Predicted Price')
    ax1.set_title(f'Random Forest (R² = {rf_r2:.3f})')
    ax1.grid(True, alpha=0.3)
    ax2.scatter(y_test, xgb_pred, alpha=0.6, color='purple')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'XGBoost (R² = {xgb_r2:.3f})')
    ax2.grid(True, alpha=0.3)   
    st.pyplot(fig3)
# Dataset info
st.markdown("---")
st.header(" Dataset Information")
col1, col2, col3 = st.columns(3)
col1.metric("Total Cars", f"{len(df):,}")
col2.metric("Average Price", f"${df['Price_USD'].mean():,.0f}")
col3.metric("Price Range", f"${df['Price_USD'].min():,.0f} - ${df['Price_USD'].max():,.0f}")
st.success(" Analysis completed successfully!")