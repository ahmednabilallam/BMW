# ===== IMPORTS =====
import pandas as pd, numpy as np, streamlit as st, matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== LOAD DATA =====
df = pd.read_csv(r"C:\Users\anallam\Desktop\Project\BMW sales data (2010-2024) (1).csv")

# ===== DATA CLEANING =====
for c in df.select_dtypes('object'):
    df[c] = df[c].fillna(df[c].mode()[0])
for c in df.select_dtypes(np.number):
    df[c] = df[c].fillna(df[c].median())
df = df.drop_duplicates()

# ===== FEATURE ENGINEERING =====
df['Car_Age'] = 2024 - df['Year']
df['Mileage_per_Year'] = df['Mileage_KM']/df['Car_Age'].replace(0,1)

# ===== FEATURES & TARGET =====
X = df[['Engine_Size_L','Mileage_KM','Car_Age','Mileage_per_Year','Model','Region','Fuel_Type','Transmission']]
y = df['Price_USD']

# ===== TRAIN-TEST SPLIT =====
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ===== PREPROCESSING =====
num_feats = ['Engine_Size_L','Mileage_KM','Car_Age','Mileage_per_Year']
cat_feats = ['Model','Region','Fuel_Type','Transmission']
preprocessor = ColumnTransformer([('num',StandardScaler(),num_feats),
                                  ('cat',OneHotEncoder(handle_unknown='ignore'),cat_feats)])

# ===== PIPELINE & GRIDSEARCH =====
pipe = Pipeline([('prep',preprocessor),('model',RandomForestRegressor(random_state=42))])
params = {'model__n_estimators':[100,200],'model__max_depth':[None,10,20]}
grid = GridSearchCV(pipe,params,cv=5,scoring='r2',n_jobs=-1)
grid.fit(X_train,y_train)

# ===== EVALUATION =====
y_pred = grid.predict(X_test)
print("R2:",r2_score(y_test,y_pred),"MAE:",mean_absolute_error(y_test,y_pred),
      "RMSE:",mean_squared_error(y_test,y_pred,squared=False))

# ===== SAVE MODEL =====
joblib.dump(grid.best_estimator_,'BMW_Best_Model.pkl')

# ===== STREAMLIT APP =====
st.title("🚗 BMW Price Predictor")
st.sidebar.header("Enter Car Specs")
year = st.sidebar.number_input("Year",2000,2024,2020)
engine = st.sidebar.number_input("Engine Size (L)",1.0,6.0,2.0,0.1)
mileage = st.sidebar.number_input("Mileage_KM",0,500000,20000)
model_name = st.sidebar.selectbox("Model", df['Model'].unique())
region = st.sidebar.selectbox("Region", df['Region'].unique())
fuel = st.sidebar.selectbox("Fuel Type", df['Fuel_Type'].unique())
trans = st.sidebar.selectbox("Transmission", df['Transmission'].unique())

if st.sidebar.button("Predict Price"):
    car_age = 2024-year
    mph = mileage/max(car_age,1)
    input_df = pd.DataFrame([{'Engine_Size_L':engine,'Mileage_KM':mileage,'Car_Age':car_age,
                              'Mileage_per_Year':mph,'Model':model_name,'Region':region,
                              'Fuel_Type':fuel,'Transmission':trans}])
    price = grid.best_estimator_.predict(input_df)[0]
    st.success(f"💰 Predicted Price: ${price:,.2f}")

# ===== EDA =====
st.header("📊 BMW Data Overview")
if st.checkbox("Show raw data"): st.dataframe(df.head(100))
fig,ax=plt.subplots(); sns.histplot(df['Price_USD'],bins=25,kde=True,ax=ax,color="#1E90FF"); ax.set_xlabel("Price (USD)"); ax.set_ylabel("Count"); st.pyplot(fig)
fig2,ax2=plt.subplots(); sns.scatterplot(x='Mileage_KM',y='Price_USD',hue='Fuel_Type',data=df,ax=ax2); ax2.set_xlabel("Mileage (KM)"); ax2.set_ylabel("Price (USD)"); st.pyplot(fig2)
fig3,ax3=plt.subplots(); sns.boxplot(x='Region',y='Price_USD',data=df,ax=ax3); ax3.set_xlabel("Region"); ax3.set_ylabel("Price (USD)"); st.pyplot(fig3)
