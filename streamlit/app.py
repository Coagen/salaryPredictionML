import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model/lr_model.pkl")

# Page title
st.title("Salary Prediction App")

# Input from user
years = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    step=0.1
)

# Predict button
if st.button("Predict Salary"):

    prediction = model.predict([[years]])

    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")