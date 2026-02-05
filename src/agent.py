import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# API Key setup (Best practice: set this in environment variables)
# os.environ["OPENAI_API_KEY"] = "sk-..." 

class HRAgent:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        if not use_mock:
            # Initialize the LLM (Temperature 0.7 for creative but professional explanation)
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
    def generate_explanation(self, employee_name, risk_score, contributing_factors):
        """
        Generates a natural language explanation for HR managers.
        """
        
        # 1. Define the Persona and Prompt
        template = """
        You are an expert HR Data Scientist Assistant. 
        Your goal is to explain employee attrition risk to a non-technical HR Manager.
        
        Employee: {name}
        Risk Score: {risk_score}%
        
        Key Contributing Factors (from SHAP analysis):
        {factors}
        
        Task:
        Write a concise, professional summary (3-4 sentences). 
        Explain WHY this employee is at risk based on the factors. 
        Do not use technical jargon like "SHAP values" or "XGBoost". 
        Use business language (e.g., "Salary gap," "Lack of promotion").
        Suggest one actionable retention strategy.
        """
        
        prompt = PromptTemplate(
            input_variables=["name", "risk_score", "factors"],
            template=template
        )

        # 2. Generate Response
        if self.use_mock:
            return self._mock_response(employee_name, risk_score, contributing_factors)
        
        # Chain: Prompt -> LLM -> Output Parser
        chain = prompt | self.llm | StrOutputParser()
        
        # Formatting factors list for the prompt
        factors_str = "\n".join([f"- {f}" for f in contributing_factors])
        
        response = chain.invoke({
            "name": employee_name,
            "risk_score": f"{risk_score:.1f}",
            "factors": factors_str
        })
        
        return response

    def _mock_response(self, name, score, factors):
        """
        Returns a hardcoded response for testing without API keys.
        """
        factors_str = ", ".join([f.split('(')[0] for f in factors])
        return (
            f"[MOCK MODE] Based on the analysis, {name} has a {score:.1f}% risk of leaving. "
            f"The main drivers are {factors_str}. "
            "It appears they are undercompensated compared to peers. "
            "Action: Consider a salary review or a performance bonus discussion immediately."
        )

# Test function
if __name__ == "__main__":
    agent = HRAgent(use_mock=True) # Change to False if you have a Key
    factors = ["MonthlyIncome (Value: 2500) increases risk", "OverTime (Value: 1) increases risk"]
    print(agent.generate_explanation("Ahmed", 85.5, factors))