import streamlit as st
import pandas as pd
import sys
import os
import streamlit.components.v1 as components

import streamlit as st
import os


# --- AI Agent API Key Management ---
st.sidebar.header("🤖 AI Agent Settings")

default_api_key = ""
try:
    default_api_key = st.secrets["GROQ_API_KEY"]
except:
    pass

user_api_key = st.sidebar.text_input(
    "API Key (Optional):", 
    type="password",
    help="We provide a demo key, but you can use your own to avoid rate limits."
)

final_api_key = user_api_key if user_api_key else default_api_key

if final_api_key:
    os.environ["GROQ_API_KEY"] = final_api_key
    st.sidebar.success("✅ AI Explainer is Active!")
else:
    st.sidebar.warning("⚠️ Enter a API Key to unlock the AI Explainer.")

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.inference import load_model, predict_attrition, predict_attrition_batch
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
tab1, tab2, tab3 = st.tabs(
    ["🚀 Prediction Dashboard", "📈 Model Monitoring (Drift)", "📂 Batch Prediction"]
)

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
            color = "#ff4b4b" if risk_score > 30 else "#09ab3b"
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

# =========================================
# TAB 3: Batch Prediction
# =========================================
with tab3:
    st.header("📂 Batch Prediction")
    st.markdown(
        "Upload a CSV file, run attrition prediction for all employees, "
        "and download a consolidated AI risk report."
    )

    uploaded_file = st.file_uploader(
        "Upload Employee CSV",
        type=["csv"],
        help="CSV rows should contain the same employee input fields used in the dashboard.",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview")
            st.dataframe(batch_df.head(20), use_container_width=True)

            if st.button("Run Batch Prediction", use_container_width=True):
                processed_rows = []
                for row in batch_df.to_dict(orient="records"):
                    processed_row, _ = preprocess_input(row, df)
                    processed_rows.append(processed_row.iloc[0])

                processed_batch_df = pd.DataFrame(processed_rows)
                probabilities = predict_attrition_batch(model, processed_batch_df)
                threshold = 0.30
                risk_labels = [
                    "High Risk" if prob >= threshold else "Low Risk"
                    for prob in probabilities
                ]

                results_df = batch_df.copy()
                results_df["Attrition_Probability"] = probabilities
                results_df["Risk_Score_Pct"] = [round(prob * 100, 2) for prob in probabilities]
                results_df["Risk_Label"] = risk_labels

                st.session_state["batch_results_df"] = results_df
                st.success("Batch prediction completed successfully.")

            if "batch_results_df" in st.session_state:
                results_df = st.session_state["batch_results_df"]
                st.subheader("Batch Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                total_employees = len(results_df)
                high_risk_count = int((results_df["Risk_Label"] == "High Risk").sum())
                avg_risk_pct = float(results_df["Risk_Score_Pct"].mean()) if total_employees > 0 else 0.0
                max_risk_pct = float(results_df["Risk_Score_Pct"].max()) if total_employees > 0 else 0.0
                min_risk_pct = float(results_df["Risk_Score_Pct"].min()) if total_employees > 0 else 0.0
                high_risk_ratio_pct = (
                    round((high_risk_count / total_employees) * 100, 2)
                    if total_employees > 0
                    else 0.0
                )

                summary = {
                    "total_employees": total_employees,
                    "high_risk_employees": high_risk_count,
                    "high_risk_ratio_pct": high_risk_ratio_pct,
                    "average_risk_pct": round(avg_risk_pct, 2),
                    "max_risk_pct": round(max_risk_pct, 2),
                    "min_risk_pct": round(min_risk_pct, 2),
                }

                if st.button("Generate Consolidated AI Report", use_container_width=True):
                    with st.spinner("Generating consolidated report..."):
                        report_text = agent.generate_batch_report(summary)
                        st.session_state["batch_report_text"] = report_text

                if "batch_report_text" in st.session_state:
                    report_text = st.session_state["batch_report_text"]
                    st.subheader("Consolidated Batch Report")
                    st.text_area("Report", value=report_text, height=260)
                    st.download_button(
                        label="Download Batch Report (TXT)",
                        data=report_text,
                        file_name="batch_attrition_report.txt",
                        mime="text/plain",
                    )

        except Exception as error:
            st.error(f"Error processing uploaded CSV: {error}")