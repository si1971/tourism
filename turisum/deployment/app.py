from pathlib import Path
import json

import joblib
import pandas as pd
import sklearn
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model_building" / "best_model.joblib"
METADATA_PATH = ROOT / "model_building" / "metadata.json"

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
)


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not METADATA_PATH.is_file():
    st.error(f"Metadata file not found: {METADATA_PATH}")
    st.stop()

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

expected_sklearn = metadata.get("environment", {}).get("scikit_learn")
if expected_sklearn and sklearn.__version__ != expected_sklearn:
    st.error(
        "Model/environment version mismatch. "
        f"Model: scikit-learn {expected_sklearn}; "
        f"Streamlit: scikit-learn {sklearn.__version__}. "
        "Check turisum/deployment/requirements.txt and reboot the app."
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
threshold = float(metadata["decision_threshold"])

st.title("✈️ Tourism Package Purchase Prediction")
st.caption(
    "Predict whether a customer is likely to purchase the Wellness Tourism Package."
)

with st.form("prediction_form"):
    age = st.slider("Age", 18, 90, 35)
    typeof_contact = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Freelancer"],
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 2)
    number_of_followups = st.slider("Number of Follow-ups", 1, 10, 3)
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
    )
    preferred_property_star = st.slider("Preferred Property Star", 3, 5, 3)
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"]
    )
    number_of_trips = st.slider("Annual Number of Trips", 1, 20, 2)
    passport = st.selectbox(
        "Passport", [0, 1], format_func=lambda value: "Yes" if value else "No"
    )
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox(
        "Own Car", [0, 1], format_func=lambda value: "Yes" if value else "No"
    )
    number_of_children_visiting = st.slider(
        "Number of Children Visiting", 0, 5, 0
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, value=25000.0, step=1000.0
    )
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)

    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "TypeofContact": [typeof_contact],
            "CityTier": [city_tier],
            "DurationOfPitch": [duration_of_pitch],
            "Occupation": [occupation],
            "Gender": [gender],
            "NumberOfPersonVisiting": [number_of_person_visiting],
            "NumberOfFollowups": [number_of_followups],
            "ProductPitched": [product_pitched],
            "PreferredPropertyStar": [preferred_property_star],
            "MaritalStatus": [marital_status],
            "NumberOfTrips": [number_of_trips],
            "Passport": [passport],
            "PitchSatisfactionScore": [pitch_satisfaction_score],
            "OwnCar": [own_car],
            "NumberOfChildrenVisiting": [number_of_children_visiting],
            "Designation": [designation],
            "MonthlyIncome": [monthly_income],
        }
    )

    # Enforce the exact feature schema used to train the production pipeline.
    input_data = input_data[metadata["raw_features"]]

    probability = float(model.predict_proba(input_data)[:, 1][0])
    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")
    st.metric("Purchase probability", f"{probability:.1%}")
    st.caption(f"Production decision threshold: {threshold:.3f}")

    if prediction == 1:
        st.success("Prioritize this customer for sales outreach.")
    else:
        st.info("This customer is currently below the outreach threshold.")
