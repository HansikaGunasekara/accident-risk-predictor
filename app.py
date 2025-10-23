import streamlit as st
import pandas as pd
import pickle
import os
import urllib.request

# ===============================
# App configuration
# ===============================
st.set_page_config(page_title="Accident Risk Predictor", page_icon="🚦", layout="wide")
st.title("🚦 Road Accident Risk Prediction")

model_URL = "https://drive.google.com/file/d/13r9fKP69ZNOCMOwBCEfg30Km2NlLxtWE/view?usp=share_link"
model_file = "trained_model.pkl"

if not os.path.exists(model_file):
    urllib.request.urlretrieve(model_URL, model_file)


# ===============================
# Load trained model
# ===============================
if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    st.success("✅ Loaded trained model for prediction!")

    # ===============================
    # Manual input form
    # ===============================
    st.header("🧮 Predict Accident Risk from Manual Input")

    with st.form("prediction_form"):
        st.subheader("Enter Road Details:")

        road_type = st.selectbox("Road Type", ["urban", "rural", "highway"])
        num_lanes = st.number_input("Number of Lanes", 1, 10, 2)
        curvature = st.slider("Curvature", 0.0, 1.0, 0.3)
        speed_limit = st.selectbox("Speed Limit", [30, 40, 50, 60, 70])
        lighting = st.selectbox("Lighting", ["daylight", "night", "dusk"])
        weather = st.selectbox("Weather", ["clear", "rain", "fog", "snow"])
        road_signs_present = st.checkbox("Road Signs Present?", True)
        public_road = st.checkbox("Public Road?", True)
        time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
        holiday = st.checkbox("Holiday?", False)
        school_season = st.checkbox("School Season?", True)
        num_reported_accidents = st.number_input("Number of Reported Accidents", 0, 100, 2)

        submitted = st.form_submit_button("🔍 Predict Risk")

    # ===============================
    # Predict when user submits form
    # ===============================
    if submitted:
        speed_category = "high_speed" if speed_limit in [60, 70] else "low_speed"
        high_curvature = int(curvature > 0.7)

        user_data = pd.DataFrame([{
            "road_type": road_type,
            "num_lanes": num_lanes,
            "curvature": curvature,
            "speed_limit": speed_limit,
            "lighting": lighting,
            "weather": weather,
            "road_signs_present": int(road_signs_present),
            "public_road": int(public_road),
            "time_of_day": time_of_day,
            "holiday": int(holiday),
            "school_season": int(school_season),
            "num_reported_accidents": num_reported_accidents,
            "speed_category": speed_category,
            "high_curvature": high_curvature
        }])

        prediction = model.predict(user_data)[0]
        st.metric("🚧 Predicted Accident Risk", f"{prediction:.4f}")

else:
    st.error("⚠️ `trained_model.pkl` not found in the app directory. Please add it first.")
