import streamlit as st, pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, r2_score

# ===== Load & Clean Data =====
df=pd.read_csv(r"C:\Users\anallam\Desktop\Project\BMW sales data (2010-2024) (1).csv")
for c in ['Mileage_KM','Engine_Size_L','Price_USD','Year']: df[c]=pd.to_numeric(df[c],errors='coerce')
df.fillna({'Price_USD':df['Price_USD'].median(),'Mileage_KM':0,'Engine_Size_L':0,'Year':2020}, inplace=True)
for c in ['Model','Region','Fuel_Type','Transmission','Color']: df[c].fillna(df[c].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)

# ===== Feature Engineering =====
df['Car_Age']=2024-df['Year']
df['Mileage_per_Year']=df['Mileage_KM']/df['Car_Age'].replace(0,1)
features=['Engine_Size_L','Mileage_KM','Car_Age','Mileage_per_Year','Model','Region','Fuel_Type','Transmission']
X=df[features]; y=df['Price_USD']

# ===== EDA Plots =====
st.title("BMW Price Prediction - EDA & Model")
with st.expander("Show EDA Plots"):
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    sns.histplot(df['Price_USD'],bins=20,kde=True,ax=axes[0,0]); axes[0,0].set_title("Price Distribution")
    sns.boxplot(x='Region',y='Price_USD',data=df,ax=axes[0,1]); axes[0,1].set_title("Price by Region")
    sns.scatterplot(x='Mileage_KM',y='Price_USD',hue='Fuel_Type',data=df,ax=axes[0,2]); axes[0,2].set_title("Mileage vs Price")
    sns.scatterplot(x='Engine_Size_L',y='Price_USD',hue='Fuel_Type',data=df,ax=axes[1,0]); axes[1,0].set_title("Engine vs Price")
    sns.scatterplot(x='Car_Age',y='Price_USD',data=df,ax=axes[1,1]); axes[1,1].set_title("Car Age vs Price")
    sns.scatterplot(x='Mileage_per_Year',y='Price_USD',data=df,ax=axes[1,2]); axes[1,2].set_title("Mileage per Year vs Price")
    st.pyplot(fig)

# ===== Feature Selection =====
num_feats=[f for f in features if f not in ['Model','Region','Fuel_Type','Transmission']]
cat_feats=[f for f in features if f in ['Model','Region','Fuel_Type','Transmission']]
preprocessor=ColumnTransformer([('num',StandardScaler(),num_feats),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat_feats)])
rf_selector=RandomForestRegressor(n_estimators=100,random_state=42)
rf_selector.fit(preprocessor.fit_transform(X),y)
selector=SelectFromModel(rf_selector,threshold=0.05,prefit=True)
X_selected=selector.transform(preprocessor.transform(X))

# ===== Train-Test Split =====
X_train,X_test,y_train,y_test=train_test_split(X_selected,y,test_size=0.2,random_state=42)

# ===== Models =====
models={'LinearRegression':LinearRegression(),'RandomForest':RandomForestRegressor(n_estimators=200,random_state=42),'GradientBoosting':GradientBoostingRegressor(n_estimators=200,random_state=42)}
best_model=None; best_r2=-1
for name,model in models.items():
    model.fit(X_train,y_train)
    r2=r2_score(y_test,model.predict(X_test))
    st.write(f"{name} R2: {r2:.3f}")
    if r2>best_r2: best_r2=r2; best_model=model
st.write("Best Model Selected ✅")

# ===== Save Model =====
joblib.dump(best_model,"BMW_Best_Model.pkl")
joblib.dump(preprocessor,"BMW_Transformer.pkl")

# ===== Streamlit Prediction UI =====
st.header("Predict BMW Price")
year=st.number_input("Year",2000,2024,2020)
engine_size=st.number_input("Engine Size (L)",1.0,6.0,2.0,0.1)
mileage=st.number_input("Mileage_KM",0,500000,20000)
model_name=st.selectbox("Model",df['Model'].unique())
region=st.selectbox("Region",df['Region'].unique())
fuel_type=st.selectbox("Fuel Type",df['Fuel_Type'].unique())
trans=st.selectbox("Transmission",df['Transmission'].unique())
car_age=2024-year; mpy=mileage/max(car_age,1)
input_df=pd.DataFrame([{'Engine_Size_L':engine_size,'Mileage_KM':mileage,'Car_Age':car_age,'Mileage_per_Year':mpy,'Model':model_name,'Region':region,'Fuel_Type':fuel_type,'Transmission':trans}])
if st.button("Predict Price"): st.success(f"Predicted Price: ${best_model.predict(selector.transform(preprocessor.transform(input_df)))[0]:,.2f}")
