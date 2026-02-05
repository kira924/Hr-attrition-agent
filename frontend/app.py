import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.explainability import load_artifacts, explain_single_instance
from src.agent import HRAgent

# Page Configuration
st.set_page_config(
    page_title="HR Guardian | Attrition Predictor",
    page_icon="🛡️",
    layout="wide"
)

# Load Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_model_resources():
    model, features = load_artifacts()
    return model, features

try:
    model, feature_names = load_model_resources()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize Agent
agent = HRAgent(use_mock=True) # Set to False if you have an API Key

# --- UI Layout ---
st.title("🛡️ HR Guardian: Intelligent Retention System")
st.markdown("""
**System Status:** 🟢 Operational | **Model Version:** v1.0.2 | **Drift Monitoring:** Active
""")

# Create Tabs
tab1, tab2 = st.tabs(["🚀 Prediction & AI Insights", "📊 MLOps Monitoring"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Employee Profile")
        st.write("Adjust values to simulate scenarios (What-If Analysis)")
        
        # Inputs - matched with typical top features in IBM dataset
        # In a real app, you would collect all features. Here we pick the most impactful ones.
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000, step=500)
        over_time = st.selectbox("OverTime", ["No", "Yes"])
        age = st.slider("Age", 18, 60, 30)
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        years_at_company = st.slider("Years at Company", 0, 20, 3)
        num_companies = st.slider("Num Companies Worked", 0, 10, 1)
        
        # Convert OverTime to binary
        over_time_bin = 1 if over_time == "Yes" else 0
        
        predict_btn = st.button("🔮 Analyze Retention Risk", type="primary")

    with col2:
        if predict_btn:
            # 1. Prepare Input Data
            # We need to create a dataframe with all features expected by the model.
            # For this demo, we will fill missing features with median/mode values (mocking)
            # to avoid creating 30 sliders.
            
            input_data = pd.DataFrame(columns=feature_names)
            input_data.loc[0] = 0 # Initialize with zeros
            
            # Update with user inputs
            if 'MonthlyIncome' in feature_names: input_data['MonthlyIncome'] = monthly_income
            if 'OverTime_Yes' in feature_names: input_data['OverTime_Yes'] = over_time_bin
            if 'Age' in feature_names: input_data['Age'] = age
            if 'TotalWorkingYears' in feature_names: input_data['TotalWorkingYears'] = total_working_years
            if 'YearsAtCompany' in feature_names: input_data['YearsAtCompany'] = years_at_company
            if 'NumCompaniesWorked' in feature_names: input_data['NumCompaniesWorked'] = num_companies
            
            # 2. Prediction
            probability = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Yes)
            prediction = model.predict(input_data)[0]
            
            # 3. Display Risk Score
            st.subheader("Risk Assessment")
            risk_percentage = probability * 100
            
            # Dynamic Color based on risk
            if risk_percentage < 30:
                color = "green"
                status = "Low Risk"
            elif risk_percentage < 70:
                color = "orange"
                status = "Medium Risk"
            else:
                color = "red"
                status = "High Risk ⚠️"
                
            st.markdown(f"## <span style='color:{color}'>{status}: {risk_percentage:.1f}%</span>", unsafe_allow_html=True)
            st.progress(int(risk_percentage))
            
            st.divider()
            
            # 4. Explainability & Agent
            st.subheader("🤖 AI Agent Analysis")
            with st.spinner("Generating explanation..."):
                # Get Top Factors
                factors = explain_single_instance(model, input_data, feature_names)
                
                # Get Agent Response
                explanation = agent.generate_explanation("Employee #1042", risk_percentage, factors)
                
                st.info(explanation)
                
            st.markdown("### Key Drivers (Why?)")
            for factor in factors:
                st.text(f"• {factor}")

with tab2:
    st.subheader("📉 Data Drift & Model Health (Evidently AI)")
    st.markdown("Automated monitoring report to detect distribution shifts.")
    
    # Placeholder for Evidently Report
    # In a real scenario, you would load the HTML report generated by Evidently
    st.warning("⚠️ Demo Mode: Showing static monitoring snapshot.")
    
    # Mocking a drift metric visualization
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Data Drift", "Detected", delta="-0.05", delta_color="inverse")
    col_m2.metric("Model Accuracy", "86%", delta="1.2%")
    col_m3.metric("Data Quality", "98%", delta="0%")
    
    st.image("https://docs.evidentlyai.com/img/dashboard.png", caption="Evidently Dashboard Snapshot (Mock)")