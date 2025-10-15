import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import datetime
warnings.filterwarnings('ignore')
#  I. DATA LOADING & CLEANING PHASE 
@st.cache_data
def load_and_clean_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        st.stop()
       
    # Data Cleaning: Handle Missing Values
    for c in df.select_dtypes('object'):
        df[c].fillna(df[c].mode()[0], inplace=True)
    for c in df.select_dtypes(np.number):
        df[c].fillna(df[c].median(), inplace=True)
    
    # Data Cleaning: Remove Duplicates
    df.drop_duplicates(inplace=True)
    
    return df

#  II. FEATURE ENGINEERING & PREPROCESSING 

@st.cache_data
def feature_engineering_and_encoding(df):
    df_processed = df.copy()
        # Feature Engineering (New features)
    current_year = datetime.datetime.now().year
    df_processed['Car_Age'] = current_year - df_processed['Year']
    df_processed['Mileage_per_Year'] = df_processed['Mileage_KM'] / df_processed['Car_Age'].replace(0, 1)
    df_processed['Price_per_EngineSize'] = df_processed['Price_USD'] / df_processed['Engine_Size_L']
    df_unencoded = df_processed.copy() 
    cat_cols = ['Model', 'Region', 'Fuel_Type', 'Transmission']
    encoders = {}

    # Encoding Categorical Variables
    for col in cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_processed[col] = oe.fit_transform(df_processed[[col]])[:, 0].astype(int) 
        encoders[col] = oe
        
    return df_processed, df_unencoded, encoders, cat_cols

#  III. MODEL TRAINING & EVALUATION 

@st.cache_resource
def train_and_evaluate_model(df):   
    # Feature Selection (Selected via Embedded Method in Notebook)
    features = ['Engine_Size_L', 'Mileage_KM', 'Car_Age', 'Mileage_per_Year',
                'Price_per_EngineSize', 'Model', 'Region', 'Fuel_Type', 'Transmission']
    target = 'Price_USD'
    X = df[features]
    y = df[target]
        # Validation (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Algorithm Selection & Tuning (RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation Metrics (R², MAE, RMSE)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, r2, mae, rmse, y_test, X_test_scaled, features, target
#  APP EXECUTION 
DATA_PATH = r"C:\Users\anallam\Desktop\Project\BMW_sales_2010_2024.csv"
# Phase Execution
df_raw = load_and_clean_data(DATA_PATH)
df, df_unencoded, encoders, cat_cols = feature_engineering_and_encoding(df_raw)
model, scaler, r2, mae, rmse, y_test, X_test_scaled, features, target = train_and_evaluate_model(df)

#  IV. DEPLOYMENT INTERFACE (STEP 9)

st.title("Automotive Price Forecasting System (2010–2026)")
st.write("Predict used BMW car prices based on a **Machine Learning Regression Model** trained on real sales data.")
st.markdown("---") 
col1_info, col2_info = st.columns(2)
col1_info.metric("Dataset Rows", f"{df.shape[0]:,}")
col2_info.metric("Modeling Technique", "Random Forest Regression")

#  SIDEBAR INPUTS 
st.sidebar.header(" Enter Car Specifications")
year = st.sidebar.number_input("Year", 2010, 2026, 2024) 
engine = st.sidebar.number_input("Engine Size (L)", 1.0, 6.0, 2.0, 0.1)
mileage = st.sidebar.number_input("Mileage (KM)", 0, 500000, 50000)
col_model, col_region = st.sidebar.columns(2)
model_name = col_model.selectbox("Model", df_unencoded['Model'].unique()) 
region = col_region.selectbox("Region", df_unencoded['Region'].unique())
col_fuel, col_trans = st.sidebar.columns(2)
fuel = col_fuel.selectbox("Fuel Type", df_unencoded['Fuel_Type'].unique())
trans = col_fuel.selectbox("Transmission", df_unencoded['Transmission'].unique())
st.sidebar.markdown("---")
# ----------------- Current Price Prediction -----------------
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
        if c in input_df.columns:
            input_df[c] = encoders[c].transform(input_df[[c]])[:, 0]

    input_scaled = scaler.transform(input_df)
    pred_price = model.predict(input_scaled)[0]

    col_pred, col_empty = st.columns([1, 2])
    col_pred.metric(label="Predicted Price Estimate", 
                    value=f"${pred_price:,.0f} USD",
                    delta=f"Model R²: {r2:.2f}",
                    delta_color="normal")
    
    st.markdown("""
    <div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top: 10px;'>
        <p style='color: #004d99; font-weight: bold;'>
         Note: This prediction is based on the car's characteristics at the time of input.
        </p>
    </div>
    """, unsafe_allow_html=True)
#  Future Price Projection
st.markdown("---")
st.subheader("Future Price Projection (2024–2026)")

if st.sidebar.button("Predict Price for Future Years", key='future_btn'):
    selected_year = year
    future_years = list(range(selected_year, 2027))  
    predicted_prices = []
    current_year = datetime.datetime.now().year
    initial_car_age = max(current_year - selected_year, 1)
    yearly_mileage_rate = mileage / initial_car_age  
    start_year = selected_year

    for y in future_years:
        years_in_future = y - start_year 
        # Update mileage cumulatively
        new_total_mileage = mileage + (yearly_mileage_rate * years_in_future) 
        
        car_age_future = max(current_year - y, 0)
        mph_future = new_total_mileage / max(car_age_future, 1)
        price_per_engine_future = df['Price_USD'].mean() / engine

        future_df = pd.DataFrame([{
            'Engine_Size_L': engine, 'Mileage_KM': new_total_mileage, 'Car_Age': car_age_future,
            'Mileage_per_Year': mph_future, 'Price_per_EngineSize': price_per_engine_future,
            'Model': model_name, 'Region': region, 'Fuel_Type': fuel, 'Transmission': trans
        }])

        for c in cat_cols:
            if c in future_df.columns:
                future_df[c] = encoders[c].transform(future_df[[c]])[:, 0] 

        future_scaled = scaler.transform(future_df)
        pred_price_future = model.predict(future_scaled)[0]
        predicted_prices.append(pred_price_future)
    st.success("Future Price Projections Calculated!")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(future_years, predicted_prices, marker='o', linestyle='--', color='#2ca02c', label='Projected Price')
    ax4.set_title(f"Predicted Price Trend for {model_name} ({selected_year}-{future_years[-1]})", fontsize=14)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Predicted Price (USD)")
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend()
    st.pyplot(fig4)
# V. EXPLORATION & EVALUATION DISPLAY 
st.markdown("---")
st.header("Model Evaluation and Data Insights")

# Model Evaluation Metrics
if st.checkbox("Show Model Evaluation Metrics"):
    col_mae, col_rmse, col_r2 = st.columns(3)
    col_mae.metric("MAE (Error in $)", f"{mae:,.2f}")
    col_rmse.metric("RMSE (Error Spread)", f"{rmse:,.2f}")
    col_r2.metric("R² Score (Model Fit)", f"{r2:.4f}")
st.markdown("---")
st.subheader("Data Visualization (Exploration Phase)")
viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    # Plot 1: Univariate (Histogram)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_unencoded['Price_USD'], bins=25, kde=True, color='skyblue', ax=ax)
    ax.set_title("Price Distribution (USD)")
    st.pyplot(fig)
with viz_col2:
    # Plot 2: Bivariate (Scatter Plot)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='Mileage_KM', y='Price_USD', hue='Fuel_Type', data=df_unencoded, ax=ax2, alpha=0.6)
    ax2.set_title("Mileage vs Price (by Fuel Type)")
    st.pyplot(fig2)
# Plot 3: Box Plot
fig3, ax3 = plt.subplots(figsize=(7, 4))
sns.boxplot(x='Region', y='Price_USD', data=df_unencoded, ax=ax3)
ax3.set_title("Price Variation by Region")
st.pyplot(fig3)
# Plot 4: Count Plot
fig5, ax5 = plt.subplots(figsize=(7, 4))
sns.countplot(y='Model', data=df_unencoded, order=df_unencoded['Model'].value_counts().index[:10], palette='viridis', ax=ax5)
ax5.set_title("Top 10 Car Models in the Dataset")
ax5.set_xlabel("Count")
st.pyplot(fig5)
# Plot 5: Correlation Heatmap
fig6, ax6 = plt.subplots(figsize=(7, 6))
correlation_matrix = df[['Price_USD', 'Mileage_KM', 'Car_Age', 'Engine_Size_L']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax6)
ax6.set_title("Correlation Heatmap of Key Numerical Features")
st.pyplot(fig6)
st.markdown("---")
st.info(" Project completed successfully using a Machine Learning approach.")