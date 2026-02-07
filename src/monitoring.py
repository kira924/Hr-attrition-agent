import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

def load_reference_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def generate_drift_report():
    """
    Generate Evidently AI Data Drift Report.
    Returns: HTML string of the report.
    """
    # 1. Load Reference Data (Training Data)
    reference_data = load_reference_data()
    
    # 2. Simulate Production Data (With Drift)
    current_data = reference_data.sample(n=300, random_state=42).copy()
    current_data['Age'] = current_data['Age'] + 10          
    current_data['MonthlyIncome'] = current_data['MonthlyIncome'] * 1.5  
    
    # 3. Build Report
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # 4. Return HTML
    return report.get_html()