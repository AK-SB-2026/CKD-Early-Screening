# CKD Two-Stage Streamlit Deployment & Analytics System

An interactive clinical decision support system for Chronic Kidney Disease (CKD) risk screening and severity classification built with Streamlit.

---

## 🚀 Quick Start (Local Execution)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```
   *(or `streamlit run app.py`)*

---

## 📋 Application Architecture & Workflow

1. **Early Screening (Linear SVM Pipeline)**
   - **CKD Prediction:** Interactive clinical risk prediction using the tuned Linear SVM pipeline (`Final_CKD_Early_Screening_SVM_Pipeline.pkl`).
   - **Interpretation & Outcomes:** Detailed feature contributions, risk levels, and clinical recommendations.
   - **Statistical Analysis:** Model metrics, calibration analysis, and validation data summaries.
   - **Important Plots:** ROC curve, Precision-Recall curve, Confusion Matrix, and Feature Importance visualizations.
   - **Insurance Report:** Financial risk categorization and underwriting recommendations.
   - **About:** System architecture and model methodology.

2. **Clinical Screening (XGBoost Severity Model)**
   - **Gated Access:** Enabled after completing an Early Screening prediction.
   - **Severity Prediction:** 4-class CKD severity classification (Healthy, Mild, Moderate, Severe).
   - **Interpretation & Clinical Outcomes:** Stage-specific clinical guidelines.
   - **Statistical Analysis & Important Plots:** Notebook-validated XGBoost metrics, confusion matrices, and correlation heatmaps.

---

## 📦 Required Model & Data Files

Place model pickle files in the project root directory:
- `Final_CKD_Early_Screening_SVM_Pipeline.pkl` *(Included & Deployed for Early Screening)*
- `CKD_Severity_XGBoost.pkl` *(Optional: For live Clinical Severity predictions)*
- `scaler.pkl` *(Optional: Standard scaler for severity model)*
- `num_imputer.pkl` *(Optional: Imputer for severity model)*

> **Note:** The application operates gracefully even if optional dataset CSVs or severity pickle files are absent. Pre-calculated metrics and visualizations will render seamlessly from notebook validation results.

---

## ☁️ Cloud Deployment Guidelines

### 1. Streamlit Community Cloud
1. Push repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Set **Main file path** to `streamlit_app.py` or `app.py`.
4. Deploy!

### 2. Docker / Containerized Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
