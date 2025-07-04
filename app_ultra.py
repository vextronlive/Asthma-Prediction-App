import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# ------------------- CONFIG -------------------
MODEL_GDRIVE_URL = "https://drive.google.com/uc?id=1EICSdhQrmz8kpFvbhkK9EV8BMX1vfy_T"
MODEL_LOCAL_PATH = "asthma_disease_rf_optimized.pkl"

# ------------------- PAGE SETTINGS -------------------
st.set_page_config(page_title="🩺 Asthma Risk Predictor", layout="wide")
st.markdown("<style> body {background-color: #f9f9f9;} </style>", unsafe_allow_html=True)

# ------------------- MODEL LOADING -------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("🔽 Downloading ML model..."):
            gdown.download(MODEL_GDRIVE_URL, MODEL_LOCAL_PATH, quiet=False)
    return joblib.load(MODEL_LOCAL_PATH)

model = load_model()
expected_features = model.feature_names_in_

# ------------------- HEADER -------------------
st.markdown("""
    <style>
        .big-title {
            font-size: 48px;
            color: #003153;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #444;
            margin-bottom: 40px;
        }
        .result-box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }
        .stButton > button {
            font-size: 18px;
            background-color: #003366;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #0059b3;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧬 Asthma Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details below and get professional predictions with actionable insights</div>', unsafe_allow_html=True)

# ------------------- INPUT FORM -------------------
st.markdown("### 📝 Patient Information Form")

columns = st.columns(3)
user_input = {}
feature_labels = [
    "1. Age (years)", "2. Gender (0=Female, 1=Male)", "3. Air Pollution Level", "4. Alcohol Use (0/1)",
    "5. Dust Allergy (0/1)", "6. Occupational Hazards (0/1)", "7. Genetic Risk (0/1)",
    "8. Chronic Lung Disease (0/1)", "9. Balanced Diet (0/1)", "10. Obesity (0/1)",
    "11. Smoking (0/1)", "12. Passive Smoker (0/1)", "13. Chest Pain (0/1)",
    "14. Coughing of Blood (0/1)", "15. Fatigue (0/1)", "16. Weight Loss (0/1)",
    "17. Shortness of Breath (0/1)", "18. Wheezing (0/1)", "19. Swallowing Difficulty (0/1)",
    "20. Clubbing of Fingernails (0/1)", "21. Frequent Cold (0/1)", "22. Dry Cough (0/1)",
    "23. Snoring (0/1)"
]

for i, feature in enumerate(expected_features):
    with columns[i % 3]:
        label = feature_labels[i] if i < len(feature_labels) else feature
        user_input[feature] = st.number_input(label, min_value=0.0, max_value=1000.0, step=1.0, value=0.0)

# ------------------- PREDICTION LOGIC -------------------
st.markdown("### 🔍 Prediction Result")
if st.button("Run Prediction"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][int(prediction)] * 100

        with st.container():
            if prediction == 1:
                st.markdown(f"<div class='result-box'><h3>🧠 **High Risk of Asthma Detected**</h3><p style='font-size:18px;'>Confidence: {proba:.2f}%</p></div>", unsafe_allow_html=True)
                st.markdown("#### 💡 Detailed Suggestions:")
                st.markdown("""
                - 🚑 **Consult a Pulmonologist** immediately.
                - 🧼 **Avoid allergens** like dust, smoke, pollen.
                - 🌬️ Use **air purifiers** at home.
                - 📈 Monitor **peak flow rate (PEFR)** regularly.
                - 📒 Maintain a **symptom diary** to identify triggers.
                - 🧘 Practice **yoga and breathing techniques**.
                """)
            else:
                st.markdown(f"<div class='result-box'><h3>✅ **Low Risk of Asthma**</h3><p style='font-size:18px;'>Confidence: {proba:.2f}%</p></div>", unsafe_allow_html=True)
                st.markdown("#### 🛡️ Health Maintenance Tips:")
                st.markdown("""
                - 🏃 **Stay active** with regular exercise.
                - 🥗 **Follow a balanced diet** rich in fruits & veggies.
                - 🚭 Avoid smoking and **secondhand smoke**.
                - 🧘 Practice **deep breathing** regularly.
                - 📅 Schedule **annual respiratory checkups**.
                """)
    except Exception as e:
        st.error("⚠️ Error during prediction. Please ensure all fields are correctly filled.")
        st.exception(e)