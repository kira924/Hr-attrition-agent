import pandas as pd
import numpy as np
import os

def load_data():
    """
    Loads data just to get the structure if needed (Optional for this fix).
    """
    path = os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def preprocess_input(input_dict, _=None):
    """
    Takes user input dictionary and transforms it into the EXACT DataFrame structure 
    the XGBoost model expects.
    """
    
    model_columns = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
        'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 'Department_Sales', 
        'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 
        'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male', 
        'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 
        'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 
        'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married', 
        'MaritalStatus_Single', 'OverTime_Yes'
    ]

    input_df = pd.DataFrame(0, index=[0], columns=model_columns)


    input_df['Age'] = input_dict.get('Age', 30)
    input_df['MonthlyIncome'] = input_dict.get('MonthlyIncome', 5000)
    input_df['TotalWorkingYears'] = input_dict.get('TotalWorkingYears', 10)
    input_df['YearsAtCompany'] = input_dict.get('YearsAtCompany', 5)
    input_df['NumCompaniesWorked'] = input_dict.get('NumCompaniesWorked', 1)
    input_df['DistanceFromHome'] = input_dict.get('DistanceFromHome', 10)
    input_df['EnvironmentSatisfaction'] = input_dict.get('EnvironmentSatisfaction', 3)
    input_df['JobSatisfaction'] = input_dict.get('JobSatisfaction', 3)
    input_df['WorkLifeBalance'] = input_dict.get('WorkLifeBalance', 3)



    # OverTime (Yes -> OverTime_Yes=1)
    if input_dict.get('OverTime') == 'Yes':
        input_df['OverTime_Yes'] = 1
    else:
        input_df['OverTime_Yes'] = 0

    # Gender (Male -> Gender_Male=1)
    if input_dict.get('Gender') == 'Male':
        input_df['Gender_Male'] = 1
    else:
        input_df['Gender_Male'] = 0 # Means Female

    # BusinessTravel
    travel = input_dict.get('BusinessTravel', 'Travel_Rarely')
    if travel == 'Travel_Frequently':
        input_df['BusinessTravel_Travel_Frequently'] = 1
    elif travel == 'Travel_Rarely':
        input_df['BusinessTravel_Travel_Rarely'] = 1
    
    # Department
    dept = input_dict.get('Department', 'Sales')
    if dept == 'Sales':
        input_df['Department_Sales'] = 1
    elif dept == 'Research & Development':
        input_df['Department_Research & Development'] = 1

    # JobRole (Example: Sales Executive)
    role = input_dict.get('JobRole', 'Sales Executive')
    role_col = f"JobRole_{role}"
    if role_col in input_df.columns:
        input_df[role_col] = 1

    # MaritalStatus
    status = input_dict.get('MaritalStatus', 'Single')
    status_col = f"MaritalStatus_{status}"
    if status_col in input_df.columns:
        input_df[status_col] = 1

    defaults = {
        'DailyRate': 802,
        'HourlyRate': 65,
        'MonthlyRate': 14313,
        'JobLevel': 2,
        'JobInvolvement': 3,
        'StockOptionLevel': 0,
        'TrainingTimesLastYear': 3,
        'YearsInCurrentRole': 4,
        'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 4,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'Education': 3
    }

    for col, val in defaults.items():
        if col in input_df.columns:
            input_df[col] = val

    return input_df[model_columns], model_columns