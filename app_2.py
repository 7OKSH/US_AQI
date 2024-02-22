import streamlit as st
import pandas as pd
import joblib
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler


# Load the transformer and model
transformer = joblib.load('Transformer.h5')
model = joblib.load('Model.h5')

st.title("AQI Prediction App")
st.info('Trying to build a model on the Air Quality dataset')

Number_of_Sites_Reporting = st.slider('Number of Sites Reporting? ', 0, 100, 15)
population = st.slider('Population? ', 0, 1000000, 100000)
density = st.slider('Density? ', 0, 50000, 5000)
Category = st.selectbox('Category? ', ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
Defining_Parameter = st.selectbox('Defining Parameter? ', ['NO2', 'Ozone', 'PM10', 'PM2.5'])


data = pd.DataFrame({
    'Category': [Category],
    'Defining_Parameter': [Defining_Parameter],
    'Number_of_Sites_Reporting': [Number_of_Sites_Reporting],
    'population': [population],
    'density': [density]
    
}, index=[0])



data_scaled = transformer.transform(data)


# Predict using the model
result = model.predict(data_scaled)

st.dataframe(data, width=1200, height=10, use_container_width=True)

if st.button('Predict'):
    st.subheader(round(result[0], 2))



