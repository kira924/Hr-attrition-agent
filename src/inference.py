import joblib
import os
import pandas as pd
from typing import List

# Define the path to the saved model artifacts
# We navigate back one directory from 'src' to reach the root, then into 'models'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_PATH = os.path.join(BASE_DIR, "models", "artifacts.pkl")

def load_model():
    """
    Loads the trained model from the pickle file.
    This function is used by the application during startup.
    """
    # Check if the artifact file exists before loading
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(f" Model file not found at: {ARTIFACT_PATH}. Please run the training script first.")
    
    # Load the artifacts dictionary from the disk
    artifacts = joblib.load(ARTIFACT_PATH)
    
    # Extract the actual XGBoost model object from the dictionary
    model = artifacts["sklearn_model"]
    return model

def predict_attrition(model, input_data):
    """
    Predicts the probability of attrition for a given input dataframe.
    
    Args:
        model: The trained XGBoost classifier.
        input_data (pd.DataFrame): The preprocessed input features.
        
    Returns:
        float: The probability of attrition (class 1), between 0.0 and 1.0.
    """
    # Ensure the input is a DataFrame (wrap dictionary if needed)
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
        
    try:
        # predict_proba returns an array: [[Prob_No, Prob_Yes]]
        # We access [0][1] to get the probability of Class 1 (Attrition = Yes)
        probability = model.predict_proba(input_data)[0][1]
        return probability
    except Exception as e:
        print(f" Prediction Error: {e}")
        # Return 0.0 as a safe fallback in case of error
        return 0.0


def predict_attrition_batch(model, input_data: pd.DataFrame) -> List[float]:
    """Predict attrition probabilities for all rows in a dataframe.

    Args:
        model: Trained XGBoost classifier.
        input_data (pd.DataFrame): Preprocessed feature matrix.

    Returns:
        List[float]: Probability of attrition for each row.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame.")

    if input_data.empty:
        return []

    try:
        probabilities = model.predict_proba(input_data)[:, 1]
        return probabilities.astype(float).tolist()
    except Exception as e:
        print(f" Batch Prediction Error: {e}")
        return [0.0] * len(input_data)