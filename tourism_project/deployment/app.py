from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("tourism_purchase_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Wellness Package Predictor", page_icon="✈️")
st.title("Wellness Tourism Package Predictor")
st.write("Enter a prospective customer's details to estimate their likelihood of purchasing the package.")

if not MODEL_PATH.is_file():
    st.error("The trained model is not available yet. Run the GitHub Actions pipeline first.")
    st.stop()

model = load_model()
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        duration = st.number_input("Duration of pitch (minutes)", min_value=0.0, value=15.0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        people = st.number_input("Number of people visiting", min_value=1, value=2)
        followups = st.number_input("Number of follow-ups", min_value=0.0, value=3.0)
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        stars = st.number_input("Preferred property star rating", min_value=1.0, max_value=5.0, value=3.0)
    with col2:
        marital = st.selectbox("Marital status", ["Single", "Divorced", "Married", "Unmarried"])
        trips = st.number_input("Number of trips", min_value=0.0, value=2.0)
        passport = st.selectbox("Has passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
        satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)
        own_car = st.selectbox("Owns a car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Number of children visiting", min_value=0.0, value=1.0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly income", min_value=0.0, value=25000.0)
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    features = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city_tier,
        "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": people, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": stars,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": satisfaction, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income,
    }])
    probability = float(model.predict_proba(features)[0, 1])
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    if probability >= 0.50:
        st.success("High-priority lead: this customer is likely to purchase the wellness package.")
    else:
        st.info("Lower-priority lead: consider a different offer or additional engagement.")
