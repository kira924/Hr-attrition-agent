import os
from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class HRAgent:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        api_key = os.getenv("GROQ_API_KEY")
        
        if not self.use_mock:
            if not api_key:
                print(" Warning: GROQ_API_KEY not found. Switching to Mock Mode.")
                self.use_mock = True
            else:
                try:
                    # Llama 3.3 for high intelligence
                    self.llm = ChatGroq(
                        temperature=0.7,
                        model_name="llama-3.3-70b-versatile",
                        groq_api_key=api_key
                    )
                except Exception as e:
                    print(f" Error: {e}")
                    self.use_mock = True
        
    def generate_explanation(self, employee_name, risk_score, contributing_factors):
        """Generates the initial static summary."""
        template = """
        You are an expert HR Data Scientist. Analyze the employee data.
        
        DATA:
        - Employee: {name}
        - Attrition Risk: {risk_score}%
        - Top Risk Factors: {factors}

        INSTRUCTIONS:
        1. Explain the primary reason for the risk.
        2. Suggest one actionable retention strategy.
        3. Be concise (max 3 sentences).
        """
        prompt = PromptTemplate(input_variables=["name", "risk_score", "factors"], template=template)
        
        if self.use_mock: return self._mock_response(employee_name, risk_score, contributing_factors)
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            factors_str = ", ".join([f.split('(')[0].strip() for f in contributing_factors])
            return chain.invoke({"name": employee_name, "risk_score": f"{risk_score:.1f}", "factors": factors_str}).strip()
        except: return self._mock_response(employee_name, risk_score, contributing_factors)

    def chat_with_data(self, user_question, employee_context):
        """
        New Feature: Chat with the data.
        employee_context is a dictionary containing all employee info.
        """
        template = """
        You are an HR Consultant assisting a manager. 
        You have the following profile for the employee under review:

        EMPLOYEE PROFILE:
        {context}

        MANAGER'S QUESTION:
        {question}

        INSTRUCTIONS:
        - Answer based strictly on the profile data and general HR best practices.
        - Be helpful, professional, and concise.
        """
        
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        # Format context into a readable string
        context_str = "\n".join([f"- {k}: {v}" for k, v in employee_context.items()])

        if self.use_mock:
            return "This is a mock chat response. Please enable Real AI mode."

        try:
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"context": context_str, "question": user_question}).strip()
        except Exception as e:
            return f"Error: {e}"

    def _mock_response(self, name, score, factors):
        return f"[MOCK] Analysis for {name}: Risk is {score:.1f}%. Please check factors."

    def generate_batch_report(self, batch_summary: Dict[str, Any]) -> str:
        """Generate a consolidated text report for batch attrition risk.

        Args:
            batch_summary (Dict[str, Any]): Aggregate metrics for uploaded batch.

        Returns:
            str: LLM-generated report text.
        """
        template = """
        You are an expert HR Analytics Consultant.
        Create a concise, practical report from the following batch attrition summary.

        BATCH SUMMARY:
        {summary}

        INSTRUCTIONS:
        1. Provide an executive overview of the risk level.
        2. Highlight the most important risk signal in the batch.
        3. Recommend 3 practical retention actions for HR managers.
        4. Keep the report concise and actionable.
        """
        prompt = PromptTemplate(input_variables=["summary"], template=template)
        summary_text = "\n".join([f"- {key}: {value}" for key, value in batch_summary.items()])

        if self.use_mock:
            return (
                "[MOCK] Batch report:\n"
                f"Total employees: {batch_summary.get('total_employees', 0)} | "
                f"High-risk ratio: {batch_summary.get('high_risk_ratio_pct', 0)}%."
            )

        try:
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"summary": summary_text}).strip()
        except Exception as e:
            return f"Error generating batch report: {e}"