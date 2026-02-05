import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    # Load data from CSV
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")

def preprocess_data(df):
    # Drop useless columns
    drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Separate features and target
    # 'Attrition' is the target variable
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Encode Target (Yes/No -> 1/0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify Categorical Columns for One-Hot Encoding
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Apply One-Hot Encoding
    # drop_first=True to reduce multicollinearity
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Convert all boolean columns to integers (True/False -> 1/0)
    # This is important for XGBoost compatibility
    bool_cols = X_encoded.select_dtypes(include=['bool']).columns
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

    return X_encoded, y_encoded, label_encoder

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test