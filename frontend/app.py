import streamlit as st
import pandas as pd
import sys
import os

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import from the NEW inference module
from src.inference import load_model, predict_attrition
from src.data_processing import load_data, preprocess_input
from src.explainability import explain_single_instance
from src.agent import HRAgent

# Page Config
st.set_page_config(page_title="HR Guardian", layout="wide", page_icon="🛡️")

# Initialize Resources
@st.cache_resource
def get_resources():
    df = load_data()
    model = load_model()
    agent = HRAgent(use_mock=False)
    return df, model, agent

df, model, agent = get_resources()

st.title("🛡️ HR Guardian: Intelligent Attrition Predictor")

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Employee Profile")
    # Inputs
    monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    age = st.slider("Age", 18, 60, 30)
    total_working_years = st.slider("Total Working Years", 0, 40, 5)
    years_at_company = st.slider("Years at Company", 0, 20, 3)
    num_companies = st.slider("Num Companies Worked", 0, 10, 1)
    
    # Hidden defaults (To match model expectations)
    input_data = {
        'Age': age, 'MonthlyIncome': monthly_income, 'OverTime': overtime,
        'TotalWorkingYears': total_working_years, 'YearsAtCompany': years_at_company,
        'NumCompaniesWorked': num_companies, 'DistanceFromHome': 10,
        'EnvironmentSatisfaction': 3, 'JobSatisfaction': 3, 'WorkLifeBalance': 3,
        'JobRole': 'Sales Executive', 'Gender': 'Male', 'BusinessTravel': 'Travel_Rarely'
    }
    
    analyze_btn = st.button("🚀 Analyze Retention Risk", use_container_width=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if analyze_btn:
    st.session_state.analysis_done = True
    # 1. Processing & Prediction
    processed_input, feature_names = preprocess_input(input_data, df)
    probability = predict_attrition(model, processed_input)
    risk_score = probability * 100
    factors = explain_single_instance(model, processed_input, feature_names)
    
    # 2. Agent Summary
    agent_analysis = agent.generate_explanation("Employee", risk_score, factors)
    
    # 3. Store Context
    st.session_state['context'] = {
        "Risk Score": f"{risk_score:.1f}%",
        "Income": monthly_income,
        "OverTime": overtime,
        "Key Factors": ", ".join(factors)
    }
    st.session_state['risk_score'] = risk_score
    st.session_state['factors'] = factors
    st.session_state['agent_analysis'] = agent_analysis
    
    # Reset Chat on new analysis
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": agent_analysis})

# --- Display Results ---
if st.session_state.analysis_done:
    risk_score = st.session_state['risk_score']
    
    with col2:
        # Risk Header
        color = "#ff4b4b" if risk_score > 50 else "#09ab3b"
        st.markdown(f"<h2 style='color:{color}'>Risk Assessment: {risk_score:.1f}%</h2>", unsafe_allow_html=True)
        st.progress(int(risk_score))
        
        # Agent Analysis Box
        st.info(f"🤖 **AI Analysis:**\n\n{st.session_state['agent_analysis']}")
        
        # Factors
        with st.expander("🔍 View Key Risk Drivers"):
            for f in st.session_state['factors']:
                st.write(f"- {f}")

        st.markdown("---")
        st.subheader("💬 Chat with HR Assistant")

        # 1. Display Chat History
        for msg in st.session_state.messages:
            # تمييز بسيط للألوان
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        # 2. INLINE CHAT INPUT (Form instead of st.chat_input)
        # This keeps the input directly below the messages
        st.markdown("<br>", unsafe_allow_html=True) # مسافة صغيرة
        
        with st.form(key='chat_form', clear_on_submit=True):
            cols = st.columns([8, 1])
            with cols[0]:
                user_input = st.text_input(
                    "Ask a question...", 
                    placeholder="E.g., Suggest a retention plan...", 
                    label_visibility="collapsed"
                )
            with cols[1]:
                submit_button = st.form_submit_button("Send")
        
        # 3. Process Input
        if submit_button and user_input:
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun() # Refresh to show user message immediately

        # 4. Generate AI Response (After Rerun)
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    last_q = st.session_state.messages[-1]["content"]
                    response = agent.chat_with_data(last_q, st.session_state['context'])
                    st.write(response)
            
            # Save Agent Message
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()