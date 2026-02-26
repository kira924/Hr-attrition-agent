```markdown
# 🛡️ HR Guardian: Intelligent Attrition Predictor & Explainer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

**HR Guardian** is an end-to-end Machine Learning solution designed to help HR departments proactively identify employees at risk of leaving. Unlike traditional "black-box" models, this system prioritizes **Recall** to capture the maximum number of potential leavers and uses **SHAP (Shapley Additive Explanations)** to explain *why* an employee might leave, enabling data-driven retention strategies.

---

## 🖼️ Dashboard Preview

*(Place a screenshot of your Streamlit dashboard here, e.g., `![Dashboard Demo](dashboard_demo.png)`)*

---

## 🚀 Key Features

* **🧠 High-Recall Prediction Engine:** Optimized specifically for the imbalanced nature of attrition data. The model is tuned to be "aggressive" (Threshold: 0.3), prioritizing the detection of leavers over precision.
* **🔍 Explainable AI (XAI):** Integrated **SHAP KernelExplainer** to provide granular, instance-level explanations. It answers: *"Why is this specific employee at risk?"* (e.g., Low Salary Hike, OverTime, Distance from Home).
* **🎛️ Interactive "What-If" Analysis:** A user-friendly Streamlit interface allows HR managers to simulate scenarios (e.g., *"What if we give this employee a 15% hike?"*) and see the risk score update in real-time.
* **modular Architecture:** Clean separation between training logic, inference engine, and frontend presentation.

---

## 📊 Model Performance & Business Logic

In HR Analytics, a **False Negative** (missing an employee who is about to leave) is far more costly than a **False Positive** (flagging a loyal employee). Losing key talent costs significantly more than a retention interview.

Therefore, this project prioritizes **Recall** over Accuracy:

| Metric | Value | Business Interpretation |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **~74%** | We successfully identify approx. 3 out of 4 employees who actually leave. |
| **Accuracy** | ~65% | A trade-off accepted to achieve high recall on imbalanced data. |

**Technical Strategy:**
- **Algorithm:** XGBoost Classifier.
- **Handling Imbalance:** Used `scale_pos_weight=10` to penalize missing the minority class.
- **Custom Threshold:** Applied a decision threshold of **0.30** (instead of the default 0.5) to flag risky employees earlier.

---

## 🛠️ Tech Stack

* **Core:** Python, Pandas, NumPy.
* **Machine Learning:** XGBoost, Scikit-Learn.
* **Explainability:** SHAP (KernelExplainer).
* **Visualization & UI:** Streamlit, Matplotlib.
* **Serialization:** Joblib.

---

## 📂 Project Structure

The project follows a modular engineering structure:

```text
├── data/
│   └── raw/                   # Original dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv)
├── models/
│   ├── artifacts.pkl          # Serialized model, features, and threshold
│   └── xgboost_model.json     # Native XGBoost model for SHAP compatibility
├── src/
│   ├── data_loader.py         # Data cleaning and preprocessing pipelines
│   ├── model.py               # Training logic (XGBoost) and evaluation
│   ├── inference.py           # Inference engine (Load model -> Predict -> Return Prob)
│   └── explainability.py      # SHAP calculations wrapper
├── frontend/
│   └── app.py                 # Streamlit dashboard application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

```

---

## ⚡ How to Run Locally

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/hr-guardian.git](https://github.com/your-username/hr-guardian.git)
cd hr-guardian

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Train the model (Optional):**
*The repo comes with a pre-trained model, but if you want to retrain:*
```bash
python -m src.model

```


4. **Launch the Dashboard:**
```bash
streamlit run frontend/app.py

```


*The app will open in your browser at `http://localhost:8501`.*

---

## 🔮 Future Improvements

* **Deployment:** Containerize the application using **Docker** and deploy to cloud (AWS/Azure).
* **API Layer:** Expose the inference engine via **FastAPI** to allow integration with other HR systems.
* **Monitoring:** Implement **Evidently AI** to monitor data drift and model degradation over time.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=issues/).

---

**Author:** [Your Name]
**License:** MIT


بالتوفيق يا هندسة! 🚀

```
