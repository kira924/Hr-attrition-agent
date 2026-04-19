import pandas as pd

# Creating a sample dataset for batch testing
data = {
    'Age': [25, 45, 32, 28, 50, 22, 38, 41, 35, 29],
    'MonthlyIncome': [2500, 12000, 5500, 3200, 15000, 2100, 7000, 8500, 6200, 4800],
    'TotalWorkingYears': [2, 20, 8, 4, 25, 1, 12, 18, 10, 6],
    'YearsAtCompany': [1, 15, 5, 2, 10, 1, 8, 12, 4, 3],
    'NumCompaniesWorked': [1, 2, 4, 1, 3, 1, 2, 5, 1, 2],
    'DistanceFromHome': [25, 2, 10, 18, 5, 20, 3, 8, 12, 15],
    'EnvironmentSatisfaction': [1, 4, 3, 2, 4, 1, 3, 4, 2, 3],
    'JobSatisfaction': [1, 4, 3, 2, 3, 2, 4, 3, 3, 4],
    'WorkLifeBalance': [1, 3, 3, 2, 4, 1, 3, 3, 3, 2],
    'OverTime': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'BusinessTravel': ['Travel_Frequently', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Non-Travel', 'Travel_Rarely', 'Travel_Rarely'],
    'Department': ['Sales', 'Research & Development', 'Research & Development', 'Sales', 'Research & Development', 'Sales', 'Research & Development', 'Research & Development', 'Sales', 'Research & Development'],
    'JobRole': ['Sales Representative', 'Manager', 'Manufacturing Director', 'Sales Representative', 'Research Director', 'Sales Representative', 'Healthcare Representative', 'Manager', 'Sales Executive', 'Laboratory Technician'],
    'MaritalStatus': ['Single', 'Married', 'Divorced', 'Single', 'Married', 'Single', 'Married', 'Divorced', 'Single', 'Married']
}

df = pd.DataFrame(data)
df.to_csv('hr_guardian_batch_test.csv', index=False)
print("Batch testing data generated successfully!")