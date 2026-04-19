from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.inference import load_model, predict_attrition  

app = FastAPI()

# 1. Load Model (Once at startup)
artifacts = load_model()

# 2. Define Input Schema 
class EmployeeInput(BaseModel):
    Age: int
    DailyRate: int
    DistanceFromHome: int
    # ... Features ...

@app.post("/predict")
def predict(data: EmployeeInput):
    # 3. Convert input to DataFrame 
    input_df = pd.DataFrame([data.model_dump()])
    
    # 4. Use your existing logic!
    probability = predict_attrition(artifacts, input_df)
    
    # 5. Return JSON
    return {
        "probability": probability,
        "risk_label": "High Risk" if probability >= 0.3 else "Low Risk"
    }