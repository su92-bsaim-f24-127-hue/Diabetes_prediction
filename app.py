import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.exceptions import NotFittedError

# Page config
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
st.title("🧬 Diabetes Risk Prediction App")
st.markdown("Enter the patient's health indicators to predict diabetes risk.")

# Load model
@st.cache_resource
def load_model():
    try:
        model_file = "model_svc.pkl" if os.path.exists("model_svc.pkl") else "diabetes_model.pkl"
        with open(model_file, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Feature input form
feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
                 'bmi', 'HbA1c_level', 'blood_glucose_level']

st.subheader("📝 Patient Information")
with st.form("prediction_form"):
    cols = st.columns(2)
    user_input = dict()

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            if feature in ['gender', 'hypertension', 'heart_disease']:
                user_input[feature] = st.selectbox(
                    f"{feature} (0=Female/No, 1=Male/Yes)", [0, 1])
            elif feature == 'smoking_history':
                st.markdown("*Note: The model uses encoded smoking history (0/1/2...).*")
                user_input[feature] = st.number_input(
                    f"{feature} (as integer code)", min_value=0, max_value=10, step=1)
            else:
                user_input[feature] = st.number_input(f"{feature}", value=0.0)

    submitted = st.form_submit_button("🔍 Predict")

# Make prediction
if submitted:
    if model is not None:
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            label = "🟢 Positive (Diabetic)" if prediction == 1 else "🔴 Negative (Not Diabetic)"
            st.success(f"Prediction: {label}")
        except NotFittedError:
            st.error("❌ Model is not trained yet.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
