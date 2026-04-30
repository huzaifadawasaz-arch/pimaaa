import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction Model")
st.write("This model predicts the likelihood of diabetes based on medical attributes.")

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# Create input columns
st.markdown("### Enter Patient Information")
col1, col2 = st.columns(2)

with col1:
    preg_count = st.number_input("Pregnancy Count", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Concentration", min_value=0, max_value=200, value=100)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=150, value=70)
    triceps = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=0)
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0)
    diabetes_pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)

# Prepare input data
input_data = np.array([[preg_count, glucose, diastolic_bp, triceps, insulin, bmi, diabetes_pedi, age]])
input_scaled = scaler.transform(input_data)

# Make prediction
if st.button("🔮 Predict Diabetes Risk", use_container_width=True):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.markdown("### Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes**")
        st.write(f"Probability: **{probability[1]:.2%}**")
    else:
        st.success(f"✅ **Low Risk of Diabetes**")
        st.write(f"Probability: **{probability[0]:.2%}**")
    
    # Show probability distribution
    st.markdown("### Risk Distribution")
    risk_data = pd.DataFrame({
        'Risk Level': ['No Diabetes', 'Diabetes'],
        'Probability': [probability[0], probability[1]]
    })
    st.bar_chart(risk_data.set_index('Risk Level'))

st.markdown("---")
st.markdown("### Model Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Type", "Random Forest")
with col2:
    st.metric("Accuracy", "73.38%")
with col3:
    st.metric("Test Samples", "154")
