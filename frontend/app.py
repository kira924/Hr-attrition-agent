import streamlit as st
import pandas as pd
import sys
import os
import streamlit.components.v1 as components

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.inference import load_model, predict_attrition
from src.data_processing import load_data, preprocess_input
from src.explainability import explain_single_instance
from src.agent import HRAgent
from src.monitoring import generate_drift_report

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

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🚀 Prediction Dashboard", "📈 Model Monitoring (Drift)"])

# =========================================
# TAB 1: Prediction & Analysis
# =========================================
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("👤 Employee Profile")
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        age = st.slider("Age", 18, 60, 30)
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        years_at_company = st.slider("Years at Company", 0, 20, 3)
        num_companies = st.slider("Num Companies Worked", 0, 10, 1)
        
        input_data = {
            'Age': age, 'MonthlyIncome': monthly_income, 'OverTime': overtime,
            'TotalWorkingYears': total_working_years, 'YearsAtCompany': years_at_company,
            'NumCompaniesWorked': num_companies, 'DistanceFromHome': 10,
            'EnvironmentSatisfaction': 3, 'JobSatisfaction': 3, 'WorkLifeBalance': 3,
            'JobRole': 'Sales Executive', 'Gender': 'Male', 'BusinessTravel': 'Travel_Rarely'
        }
        
        analyze_btn = st.button("Analyze Risk", use_container_width=True)

    # Session State
    if "messages" not in st.session_state: st.session_state.messages = []
    if "analysis_done" not in st.session_state: st.session_state.analysis_done = False

    if analyze_btn:
        st.session_state.analysis_done = True
        processed_input, feature_names = preprocess_input(input_data, df)
        probability = predict_attrition(model, processed_input)
        risk_score = probability * 100
        factors = explain_single_instance(model, processed_input, feature_names)
        agent_analysis = agent.generate_explanation("Employee", risk_score, factors)
        
        st.session_state['context'] = {
            "Risk Score": f"{risk_score:.1f}%", "Income": monthly_income, "Factors": ", ".join(factors)
        }
        st.session_state['risk_score'] = risk_score
        st.session_state['factors'] = factors
        st.session_state['agent_analysis'] = agent_analysis
        st.session_state.messages = [{"role": "assistant", "content": agent_analysis}]

    if st.session_state.analysis_done:
        risk_score = st.session_state['risk_score']
        with col2:
            color = "#ff4b4b" if risk_score > 50 else "#09ab3b"
            st.markdown(f"<h2 style='color:{color}'>Risk Assessment: {risk_score:.1f}%</h2>", unsafe_allow_html=True)
            st.progress(int(risk_score))
            st.info(f"🤖 **AI Analysis:**\n\n{st.session_state['agent_analysis']}")
            
            with st.expander("🔍 Key Drivers"):
                for f in st.session_state['factors']: st.write(f"- {f}")

            st.markdown("---")
            st.subheader("💬 Chat with HR Assistant")
            
            # Chat UI
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.write(msg["content"])

            with st.form(key='chat_form', clear_on_submit=True):
                cols = st.columns([8, 1])
                user_input = cols[0].text_input("Ask...", placeholder="Retention plan?", label_visibility="collapsed")
                if cols[1].form_submit_button("Send") and user_input:
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.rerun()

            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        resp = agent.chat_with_data(st.session_state.messages[-1]["content"], st.session_state['context'])
                        st.write(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
                st.rerun()

# =========================================
# TAB 2: Model Monitoring (Evidently AI)
# =========================================
with tab2:
    st.header("📈 Data Drift & Model Health Monitoring")
    st.markdown("""
    This dashboard monitors the **stability** of the model by comparing current production data 
    against the original training reference data using **Evidently AI**.
    """)
    
    if st.button("🔄 Run Data Drift Analysis"):
        with st.spinner("Generating Evidently AI Report... (This may take a moment)"):
            try:
                report_html = generate_drift_report()
                
                if len(report_html) > 100:
                    st.success("Analysis Complete! Drift detected in simulated data.")
                    
                    st.download_button(
                        label="📥 Download Full Report (HTML)",
                        data=report_html,
                        file_name="drift_report.html",
                        mime="text/html"
                    )
                    
                    st.markdown("---")
                    st.caption("👇 Preview of the report (Download for full view)")
                    components.html(report_html, height=1000, scrolling=True)
                
                else:
                    st.error(" The generated report is empty. Please check the data.")
                    
            except Exception as e:
                st.error(f" Error generating report: {e}")