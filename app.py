import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("housing_price_model.pkl")
model_features = joblib.load("model_features.pkl")

st.title("Melbourne Housing Price Predictor")

st.write("Enter property details to estimate the sold price.")

suburb = st.selectbox(
    "Suburb",
    ["Burwood", "Richmond", "Tarneit"]
)

property_type = st.selectbox(
    "Property Type",
    ["House", "Townhouse", "Unit", "Apartment", "Block of units", "Other"]
)

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
car_spaces = st.number_input("Car Spaces", min_value=0, max_value=10, value=1)
land_size_m2 = st.number_input("Land Size (m²)", min_value=0, max_value=5000, value=500)

sale_month = st.slider("Sale Month", 1, 12, 5)
sale_year = st.number_input("Sale Year", min_value=2020, max_value=2027, value=2026)

total_rooms = bedrooms + bathrooms

if total_rooms > 0:
    land_per_room = land_size_m2 / total_rooms
else:
    land_per_room = 0

input_data = pd.DataFrame(
    columns=model_features
)

input_data.loc[0] = 0

input_data.loc[0, "bedrooms"] = bedrooms
input_data.loc[0, "bathrooms"] = bathrooms
input_data.loc[0, "car_spaces"] = car_spaces
input_data.loc[0, "land_size_m2"] = land_size_m2
input_data.loc[0, "sale_month"] = sale_month
input_data.loc[0, "sale_year"] = sale_year
input_data.loc[0, "total_rooms"] = total_rooms
input_data.loc[0, "land_per_room"] = land_per_room

suburb_column = "suburb_" + suburb

if suburb_column in input_data.columns:
    input_data.loc[0, suburb_column] = 1

property_column = "property_type_" + property_type

if property_column in input_data.columns:
    input_data.loc[0, property_column] = 1

if st.button("Predict Price"):
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"Estimated Sold Price: ${prediction:,.2f}")