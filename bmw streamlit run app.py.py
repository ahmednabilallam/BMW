# ===== IMPORTS =====
import pandas as pd, numpy as np, streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# ===== LOAD DATA =====
DATA_PATH = r"C:\Users\anallam\Desktop\Project\BMW sales data (2010-2024) (1).csv"
df = pd.read_csv(DATA_PATH)

# ===== DATA CLEANING =====
for c in df.select_dtypes('object'): df[c] = df[c].fillna(df[c].mode()[0])
for c in df.select_dtypes(np.number): df[c] = df[c].fillna(df[c].median())
df = df.drop_duplicates()

# ===== FEATURE ENGINEERING =====
df['Car_Age'] = 2024 - df['Year']
df['Mileage_per_Year'] = df['Mileage_KM'] / df['Car_Age'].replace(0,1)

# ===== FEATURES & TARGET =====
X = pd.get_dummies(df[['Engine_Size_L','Mileage_KM','Car_Age','Mileage_per_Year','Model','Region','Fuel_Type','Transmission']], drop_first=True)
y = df['Price_USD']

# ===== TRAIN MODEL =====
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ===== STREAMLIT APP =====
st.title("BMW Price Predictor")
st.sidebar.header("Enter Car Specs")
year = st.sidebar.number_input("Year",2000,2024,2020)
engine = st.sidebar.number_input("Engine Size (L)",1.0,6.0,2.0,0.1)
mileage = st.sidebar.number_input("Mileage (KM)",0,500000,20000)
model_name = st.sidebar.selectbox("Model", df['Model'].unique())
region = st.sidebar.selectbox("Region", df['Region'].unique())
fuel = st.sidebar.selectbox("Fuel Type", df['Fuel_Type'].unique())
trans = st.sidebar.selectbox("Transmission", df['Transmission'].unique())

if st.sidebar.button("Predict Price"):
    car_age = 2024-year
    mph = mileage / max(car_age,1)
    input_df = pd.DataFrame([{
        'Engine_Size_L':engine, 'Mileage_KM':mileage, 'Car_Age':car_age,
        'Mileage_per_Year':mph, 'Model':model_name, 'Region':region,
        'Fuel_Type':fuel, 'Transmission':trans
    }])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    price = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${price:,.2f}")

# ===== EDA =====
st.header("BMW Data Overview")
if st.checkbox("Show raw data"): st.dataframe(df.head(100))

fig, ax = plt.subplots()
sns.histplot(df['Price_USD'], bins=25, kde=True, ax=ax, color="#1E90FF")
ax.set_xlabel("Price (USD)"); ax.set_ylabel("Count"); st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.scatterplot(x='Mileage_KM', y='Price_USD', hue='Fuel_Type', data=df, ax=ax2)
ax2.set_xlabel("Mileage (KM)"); ax2.set_ylabel("Price (USD)"); st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.boxplot(x='Region', y='Price_USD', data=df, ax=ax3)
ax3.set_xlabel("Region"); ax3.set_ylabel("Price (USD)"); st.pyplot(fig3)

st.header("Average Price per BMW Model")
fig4, ax4 = plt.subplots(figsize=(12,6))
avg_price_per_model = df.groupby('Model')['Price_USD'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_per_model.index, y=avg_price_per_model.values, ax=ax4, palette='viridis')
ax4.set_xlabel("Model"); ax4.set_ylabel("Average Price (USD)"); ax4.set_title("Average Price by Model")
plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig4)
