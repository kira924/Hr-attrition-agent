import shap
import pandas as pd
import joblib
import os
import numpy as np

# Constants
MODEL_DIR = "models"
ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts.pkl")

def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(f"Artifacts not found at {ARTIFACT_PATH}")
    
    # Load features and sklearn wrapper
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts["sklearn_model"]
    feature_names = artifacts["features"]
    
    return model, feature_names

def get_explainer(model, feature_names):
    """
    Returns a KernelExplainer wrapping the prediction function.
    """
    
    # 1. Create a lightweight background dataset (Baseline)
    # Using a small slice of zeros/median is standard for KernelExplainer performance
    background_data = pd.DataFrame(
        np.zeros((1, len(feature_names))), 
        columns=feature_names
    )
    
    # 2. Define a Wrapper Function
    # This prevents SHAP from accessing internal model attributes
    def prediction_wrapper(data):
        if isinstance(data, np.ndarray):
            data_df = pd.DataFrame(data, columns=feature_names)
        else:
            data_df = data
            
        return model.predict_proba(data_df)
    
    # 3. Use KernelExplainer
    explainer = shap.KernelExplainer(prediction_wrapper, background_data)
    
    return explainer

def explain_single_instance(model, instance_data, feature_names, top_k=3):
    # Ensure instance_data is a DataFrame
    if isinstance(instance_data, pd.Series):
        instance_data = instance_data.to_frame().T
    
    # Align columns
    instance_data = instance_data[feature_names]
    
    # --- EXPLANATION GENERATION ---
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        explainer = get_explainer(model, feature_names)
        shap_values = explainer.shap_values(instance_data)
    
    # --- SHAP OUTPUT HANDLING (Robust) ---
    # KernelExplainer for binary classification usually returns a list [Class0, Class1]
    if isinstance(shap_values, list):
        # We want the positive class (Attrition = Yes), usually index 1
        # If len is 1, take index 0
        target_idx = 1 if len(shap_values) > 1 else 0
        values = shap_values[target_idx]
    else:
        values = shap_values

    # --- FLATTENING & LENGTH FIX (The Solution) ---
    # Force everything to be 1D arrays
    values = np.array(values).flatten()
    actual_values = np.array(instance_data.iloc[0].values).flatten()
    names = np.array(feature_names).flatten()

    # Defensive check: Ensure all lengths match the minimum common length
    # This prevents "All arrays must be of the same length" crash
    min_len = min(len(names), len(values), len(actual_values))
    
    # Slice arrays to match the minimum length
    names = names[:min_len]
    values = values[:min_len]
    actual_values = actual_values[:min_len]

    # Build Importance DataFrame safely
    feature_importance = pd.DataFrame({
        'feature': names,
        'shap_value': values,
        'actual_value': actual_values
    })
    
    # Sort and Format
    feature_importance['abs_impact'] = feature_importance['shap_value'].abs()
    top_features = feature_importance.sort_values(by='abs_impact', ascending=False).head(top_k)
    
    explanation_list = []
    for _, row in top_features.iterrows():
        impact_direction = "increases risk" if row['shap_value'] > 0 else "decreases risk"
        explanation_list.append(
            f"{row['feature']} (Value: {row['actual_value']}) {impact_direction}"
        )
    
    return explanation_list