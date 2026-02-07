import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from src.data_loader import load_data, preprocess_data, get_train_test_split

# Constants
DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = "models"
ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts.pkl")
JSON_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")

def train_and_save_model():
    # 1. Load and Preprocess Data
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        
    df = load_data(DATA_PATH)
    X, y, _ = preprocess_data(df)

    # 2. Split Data
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Note: We are not using the calculated ratio here. 
    # We use a fixed high value (10) to force the model to prioritize the minority class.
    # ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1) 
    
    # 3. Initialize XGBoost Classifier
    print("Training model (Aggressive Mode)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,        # Increased trees for better learning
        learning_rate=0.05,      # Slower learning rate for robustness
        max_depth=3,             # Shallow depth to prevent overfitting on majority class
        min_child_weight=2,      
        gamma=0.2,               
        subsample=0.8,           
        colsample_bytree=0.8,    
        scale_pos_weight=9,     # High weight to heavily penalize missing attrition cases
        eval_metric='logloss',
        random_state=42
    )

    # 4. Train
    model.fit(X_train, y_train)

    # 5. Evaluate with Custom Threshold
    # We use predict_proba to get raw probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Set a lower threshold (0.3 instead of 0.5) to capture more risky employees
    THRESHOLD = 0.30
    y_pred_custom = (y_probs >= THRESHOLD).astype(int)

    print("\n" + "="*40)
    print(f"Model Evaluation (Threshold: {THRESHOLD})")
    print("="*40)
    print(classification_report(y_test, y_pred_custom))
    
    rec = recall_score(y_test, y_pred_custom)
    prec = precision_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    
    print(f"Key Metrics:")
    print(f"   • Recall (Captured Leavers): {rec:.2%}")
    print(f"   • Precision: {prec:.2%}")
    print(f"   • F1 Score: {f1:.4f}")

    # 6. Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save native XGBoost model (JSON) for SHAP compatibility
    model.get_booster().save_model(JSON_MODEL_PATH)
    print(f"Native XGBoost model saved to {JSON_MODEL_PATH}")

    # Save artifacts including the threshold for inference usage
    artifacts = {
        "features": X.columns.tolist(),
        "sklearn_model": model, 
        "threshold": THRESHOLD  # Important: Save threshold to use in the App
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    print(f"Artifacts saved to {ARTIFACT_PATH}")

if __name__ == "__main__":
    train_and_save_model()