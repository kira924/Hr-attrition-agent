import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
model = joblib.load("D:\study\HR_Guardian_Intelligent_Attrition_Predictor_&_Explainer\models\xgboost_model.pkl")
df = pd.read_csv(DATA_PATH)

