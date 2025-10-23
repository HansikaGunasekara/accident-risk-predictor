import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

st.set_page_config(page_title="Accident Risk Predictor", page_icon="🚦", layout="wide")
st.title("🚦 Road Accident Risk Prediction")

model_file = "trained_model.pkl"
train_zip_file = "train.zip"
unzipped_folder = "train_data"

# Unzip train.zip if not already unzipped
if not os.path.exists(unzipped_folder):
    with zipfile.ZipFile(train_zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzipped_folder)
    st.info(f"📦 Extracted {train_zip_file} to {unzipped_folder}")

# Assume inside the unzipped folder there is a CSV file, for example 'train.csv'
train_csv_path = os.path.join(unzipped_folder, "train.csv")

# Load training data
try:
    train_df = pd.read_csv(train_csv_path)
    st.success("✅ Training data loaded successfully from zip!")
except Exception as e:
    st.error(f"Failed to load training CSV: {e}")
    st.stop()

# Feature engineering
train_df['speed_category'] = train_df['speed_limit'].apply(lambda x: 'high_speed' if x in [60, 70] else 'low_speed')
train_df['high_curvature'] = (train_df['curvature'] > 0.7).astype(int)

features = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit',
    'lighting', 'weather', 'road_signs_present',
    'public_road', 'time_of_day', 'holiday', 'school_season',
    'num_reported_accidents', 'speed_category', 'high_curvature'
]
target = 'accident_risk'

bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season', 'high_curvature']
for col in bool_cols:
    train_df[col] = train_df[col].astype(int)

numeric_features = ['num_lanes', 'curvature', 'speed_limit',
                    'road_signs_present', 'public_road', 'holiday',
                    'school_season', 'num_reported_accidents', 'high_curvature']
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'speed_category']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                   objective='reg:squarederror', random_state=42, n_jobs=-1)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
meta_model = Ridge(alpha=1.0)

stacked_regressor = StackingRegressor(
    estimators=[('xgb', xgb),
                ('rf', rf)],
    final_estimator=meta_model,
    passthrough=True,
    n_jobs=-1
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', stacked_regressor)
])

# Training button
if st.button("🚀 Train & Save Model"):
    X_train = train_df[features]
    y_train = train_df[target]

    with st.spinner("🧠 Training model... please wait"):
        model_pipeline.fit(X_train, y_train)

    with open(model_file, "wb") as f:
        pickle.dump(model_pipeline, f)
    st.success(f"✅ Model trained and saved as `{model_file}`")

# Manual input prediction section
st.header("🧮 Predict Accident Risk from Manual Input")

if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Loaded trained model for prediction!")

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
    st.warning(f"⚠️ Please train and save a model first by clicking the Train button above.")
