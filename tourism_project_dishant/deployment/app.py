import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="dishantkalra/Tourism-Project-New", filename="tourism_model_new_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Product Prediction App")
st.write("The Customer Buying Prediction App that predicts whether customers will but the product or not.")

# Collect user input
Age = st.number_input("Age", min_value=10, max_value=100)
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("CityTier", ["1", "2", "3"])
DurationOfPitch = st.number_input("DurationOfPitch")
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0.0, value=10000.0)
NumberOfFollowups = st.number_input("NumberOfFollowups")
ProductPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic","Super Deluxe","Standard","King"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar)", min_value=1, max_value=5)
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Divorced", "Unmarried", "Married"])
NumberOfTrips = st.number_input("NumberOfTrips")
Passport = st.number_input("Passport")
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore")
OwnCar = st.number_input("OwnCar")
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting")
Designation = st.selectbox("Designation", ["AVP", "VP", "Executive", "Manager", "Senior Manager"])
MonthlyIncome = st.number_input("MonthlyIncome")


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
