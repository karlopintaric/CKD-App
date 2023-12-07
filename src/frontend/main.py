import streamlit as st
import shap
import pandas as pd
import numpy as np
import joblib
import os
import shap
from streamlit_shap import st_shap

import sys
sys.path.append("c:\\Users\\kkbf876\\Coding\\CKD App")
from src.frontend.calculations import calculate_ckd_epi_gfr, calculate_bmi

# Load the trained model
model = joblib.load('c:\\Users\\kkbf876\\Coding\\CKD App\\models\\RandomForest.joblib')

features = model.feature_names_in_

# Define diseases
disease_features = ['HistoryDiabetes', 'HistoryCHD', 'HistoryVascular',
                   'HistorySmoking', 'HistoryHTN', 'HistoryDLD', 'HistoryObesity']

disease_features_renamed = ['Diabetes', 'Chronic Heart Disease', 'Vascular Disease', 'Smoking',
                            'Hypertension', 'Dyslipidemia', 'Obesity']

# Define the demographic info
demo_features = ['AgeBaseline', 'Sex']
demo_features_renamed = ['Age', 'Sex']

drug_features = ['DLDmeds', 'DMmeds', 'HTNmeds', 'ACEIARB']
drug_features_renamed = ["Dyslipidemia", "Diabetes", "Hypertension", "ACE Inhibitors"]

numeric_features = ["CholesterolBaseline", "CreatinineBaseline", 
                    "sBPBaseline", 'dBPBaseline']

numeric_features_renamed = ['Cholesterol', 'Serum Creatinine', 'Systolic Blood Pressure',
                            'Diastolic Blood Pressure']

# Create a form for user input
st.title("Chronic Kidney Disease Prediction")

# Get user input for demographic features
st.sidebar.subheader("Demographic Info:")
user_input = {}
age = st.sidebar.number_input("Age", min_value=0, value=0)
sex = 0 if st.sidebar.selectbox("Sex", ["Female", "Male"]) == "Female" else 1
height = st.sidebar.number_input("Height", min_value=0, value=0)
weight = st.sidebar.number_input("Weight", min_value=0, value=0)

user_input["AgeBaseline"] = age
user_input["Sex"] = sex

# Get user input for disease features
st.sidebar.subheader("Disease Info:")
for name, feature in zip(disease_features_renamed, disease_features):
    user_input[feature] = st.sidebar.checkbox(name, key=feature)

# Get user input on drugs use
st.sidebar.subheader("Drug Info:")
for name, feature in zip(drug_features_renamed, drug_features):
    user_input[feature] = st.sidebar.checkbox(name, key=feature)

# Get input on numeric features
st.sidebar.subheader("Lab Values: ")
for name, feature in zip(numeric_features_renamed, numeric_features):
    user_input[feature] = st.sidebar.number_input(name, value=0.0, min_value=0.0)

if st.button("Predict"):
    # Calcute other features
    user_input["eGFRBaseline"] = calculate_ckd_epi_gfr(
        user_input['AgeBaseline'], 
        user_input["CreatinineBaseline"],
        is_female= not user_input["Sex"]
        )

    user_input["BMIBaseline"] = calculate_bmi(weight, height)

    input_df = pd.DataFrame(user_input, index=[0])
    input_df = input_df[features]

    # Display user input
    st.subheader("User Input:")
    st.write(input_df)

    # Make predictions
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:, 1][0]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Display prediction and probability
    st.subheader("Prediction:")
    st.write(f"Residual Risk: {probability - explainer.expected_value[1]:.2%}")

    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_df))



