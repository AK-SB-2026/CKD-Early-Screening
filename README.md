# RENALIS — CKD Risk Intelligence Streamlit App

A polished Streamlit front-end built around the workflows in the uploaded early-screening and CKD severity notebooks.

## Project structure

```text
ckd_streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── models/
    ├── Final_CKD_Early_Screening_SVM_Pipeline.pkl
    ├── CKD_Severity_XGBoost.pkl
    ├── scaler.pkl
    └── num_imputer.pkl
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required model artifacts

Place these exact files inside `models/`:

```text
Final_CKD_Early_Screening_SVM_Pipeline.pkl
CKD_Severity_XGBoost.pkl
scaler.pkl
num_imputer.pkl
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Put `app.py`, `requirements.txt`, `.streamlit/config.toml` and the `models/` artifacts in the repo.
3. In Streamlit Community Cloud, deploy the repository and choose `app.py` as the entrypoint.

## Important artifact note

The early-screening notebook saves a complete deployable `ImbPipeline` containing preprocessing, SMOTE and LinearSVC. The app therefore feeds the 49 notebook features directly into that pipeline.

The severity notebook saves the XGBoost model, StandardScaler and numeric imputer. The app uses `scaler.feature_names_in_` when available to recover the exact one-hot-expanded training order. For the most robust deployment, keep the scaler produced by the notebook together with the XGBoost model.

## Model outputs used in the interface

Early screening:
- Binary target: 0 = Healthy / 1 = Any CKD stage
- Final SVM validation accuracy: 0.900175
- Precision: 0.669915
- Recall: 0.909042
- F1: 0.771371
- ROC-AUC: 0.952264

Severity:
- Healthy = 0
- Mild CKD = 1
- Moderate CKD = 2
- Severe CKD = 3
- XGBoost validation accuracy: 0.9985
- Macro F1: 0.9692
- QWK: 0.9984
- Multiclass ROC-AUC: 0.999553

These are retrospective validation results reported by the notebooks, not guarantees of real-world performance.


## Deployment safety
Never upload identifiable patient data to a public demo deployment.


## Model Intelligence
The app includes a Model Intelligence page with notebook-backed candidate comparisons, validation metrics, model-family explanations, and the rationale for the deployed Linear SVM and XGBoost models.
