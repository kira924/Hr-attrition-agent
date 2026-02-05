import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, f1_score
from src.data_loader import load_data, preprocess_data, get_train_test_split

# Constants
DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = "models"
# We will save two files: one full artifact (pickle) and one native XGBoost (json) for SHAP
ARTIFACT_PATH = os.path.join(MODEL_DIR, "artifacts.pkl")
JSON_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")

def train_and_save_model():
    # 1. Load and Preprocess Data
    print("Loading data...")
    df = load_data(DATA_PATH)
    X, y, _ = preprocess_data(df)

    # 2. Split Data
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # 3. Initialize XGBoost Classifier
    print("Training model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=5,  # Handling Imbalance
        eval_metric='logloss'
    )

    # 4. Train
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # 6. Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the native XGBoost model to JSON (Fixes SHAP version compatibility)
    model.get_booster().save_model(JSON_MODEL_PATH)
    print(f"Native XGBoost model saved to {JSON_MODEL_PATH}")

    # Save other artifacts (feature names, etc) via joblib
    artifacts = {
        "features": X.columns.tolist(),
        # We save the sklearn wrapper too just in case we need it for simple prediction
        "sklearn_model": model 
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    print(f"Artifacts saved to {ARTIFACT_PATH}")

if __name__ == "__main__":
    train_and_save_model()