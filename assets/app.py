from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None


# ================================================================
# APP CONFIG
# ================================================================
st.set_page_config(
    page_title="RENALIS — Renal Analytics & Intelligence System",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


# ================================================================
# DESIGN SYSTEM
# ================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root {
  --ink: #11212a;
  --muted: #65747e;
  --paper: #f7faf9;
  --mint: #ddf5ed;
  --teal: #087f73;
  --teal2: #0ca38f;
  --navy: #153541;
  --line: #dfe9e5;
  --card: rgba(255,255,255,.93);
  --shadow: 0 12px 30px rgba(16,45,53,.08);
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #e9f8f3 0%, #f7faf9 32%, #f7faf9 100%); color: var(--ink); }
.block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1450px; }
.hero {
  background: linear-gradient(135deg, #143944 0%, #0b756a 55%, #16a88d 100%);
  padding: 28px 34px; border-radius: 26px; color: white; box-shadow: var(--shadow);
  margin-bottom: 20px; position: relative; overflow: hidden;
}
.hero:after { content: ''; position:absolute; width:280px; height:280px; border-radius:50%; right:-90px; top:-110px; background:rgba(255,255,255,.09); }
.hero h1 { font-family: 'Playfair Display', serif; font-size: 42px; margin: 0; letter-spacing: -.5px; }
.hero p { margin: 8px 0 0; max-width: 870px; color: rgba(255,255,255,.85); font-size: 16px; }
.kicker { text-transform: uppercase; letter-spacing: 1.8px; font-size: 11px; font-weight: 700; opacity: .75; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; box-shadow: var(--shadow);
}
.metric-card { background: white; border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; box-shadow: 0 7px 22px rgba(16,45,53,.06); }
.metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
.metric-value { font-size: 28px; font-weight: 800; color: var(--navy); margin-top: 4px; }
.metric-note { color: var(--muted); font-size: 12px; margin-top: 4px; }
.section-title { font-size: 25px; font-weight: 800; margin: 18px 0 8px; color: var(--navy); }
.section-sub { color: var(--muted); margin-top: -2px; margin-bottom: 16px; }
.badge { display:inline-block; padding: 6px 11px; border-radius:999px; font-weight:700; font-size:12px; }
.badge-green { background:#dff6eb; color:#1d7558; } .badge-orange { background:#fff1d9; color:#956314; } .badge-red { background:#ffe0df; color:#a33c35; } .badge-teal { background:#dff3ef; color:#0a6b60; }
.smallcaps { text-transform:uppercase; letter-spacing:1.2px; font-size:11px; color:var(--muted); font-weight:800; }
.note { background:#eef8f5; border-left:4px solid var(--teal2); padding:10px 14px; border-radius:8px; color:#284a44; }
.warning-note { background:#fff7e9; border-left:4px solid #db9b35; padding:10px 14px; border-radius:8px; color:#634b20; }
.risk-banner { border-radius:18px; padding:22px; margin-bottom:14px; border:1px solid rgba(0,0,0,.06); }
.risk-low { background:linear-gradient(135deg,#ecfbf3,#f9fffc); }
.risk-high { background:linear-gradient(135deg,#fff0ef,#fffaf8); }
.risk-mid { background:linear-gradient(135deg,#fff8e9,#fffdf6); }
button[kind="primary"] { border-radius:12px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:white; border:1px solid var(--line); border-radius:12px; padding:7px 15px; }
.stTabs [aria-selected="true"] { background:#dff3ef !important; color:#0a655c !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#133943 0%, #102f38 100%); }
[data-testid="stSidebar"] * { color: #f4fbf8; }
[data-testid="stSidebar"] .stRadio label { padding: 7px 4px; border-radius: 10px; }
[data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.07); }
.kidney-mark { width:48px; height:48px; display:inline-flex; align-items:center; justify-content:center; border-radius:16px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.18); margin-right:12px; vertical-align:middle; }
.kidney-mark svg { width:32px; height:32px; }
.hero-row { display:flex; align-items:center; gap:10px; position:relative; z-index:2; }
.hero-kidney-note { color:rgba(255,255,255,.68); font-size:11px; letter-spacing:1.1px; text-transform:uppercase; margin-top:14px; }
.icon-tile { background:linear-gradient(145deg,#ffffff,#eef8f5); border:1px solid var(--line); border-radius:20px; padding:19px; min-height:155px; box-shadow:var(--shadow); position:relative; overflow:hidden; }
.icon-tile:after { content:''; position:absolute; width:120px; height:120px; border-radius:50%; right:-48px; top:-48px; background:#dff3ef; opacity:.85; }
.icon-bubble { width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:#dff3ef; color:#0a6b60; margin-bottom:12px; position:relative; z-index:2; }
.feature-strip { display:flex; flex-wrap:wrap; gap:9px; margin:14px 0 6px; }
.feature-chip { padding:8px 11px; border-radius:999px; background:#eff8f5; color:#21675d; border:1px solid #d8ece6; font-size:11px; font-weight:700; }
.quick-card { background:#ffffff; border:1px solid var(--line); border-radius:18px; padding:16px 18px; box-shadow:0 7px 22px rgba(16,45,53,.05); height:100%; }
.sidebar-foot { font-size:11px; color:rgba(255,255,255,.62); line-height:1.45; margin-top:18px; }
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# SOURCE-DERIVED CONSTANTS FROM THE NOTEBOOKS
# ================================================================
EARLY_FEATURES = [
    "Age", "Sex", "Ethnicity", "Country", "Residence_Type", "Education_Level",
    "Socioeconomic_Status", "Height_cm", "Weight_kg", "BMI", "Waist_Circumference_cm",
    "Body_Fat_Percentage", "Smoking_Status", "Alcohol_Consumption", "Physical_Activity_Level",
    "Exercise_Hours_Per_Week", "Daily_Steps", "Water_Intake_L", "Sodium_Intake_mg",
    "Fast_Food_Frequency_Per_Week", "Sleep_Duration_Hours", "Stress_Level", "Diabetes",
    "Hypertension", "Cardiovascular_Disease", "Heart_Failure", "Hyperlipidemia", "Kidney_Stones",
    "Recurrent_UTI", "Autoimmune_Disease", "Family_History_CKD", "Obesity", "Heart_Rate",
    "Respiratory_Rate", "Oxygen_Saturation", "Systolic_BP", "Diastolic_BP",
    "Blood_Pressure_Category", "NSAID_Usage", "Annual_Checkups", "Health_Insurance",
    "Annual_Household_Income_USD", "Employment_Status", "Pulse_Pressure", "Waist_to_Height",
    "Lifestyle_Risk", "Metabolic_Risk", "CV_Risk", "Poor_Sleep"
]

SEVERITY_RAW_FEATURES = [
    "Age", "Sex", "Ethnicity", "Country", "Residence_Type", "Education_Level", "Socioeconomic_Status",
    "Height_cm", "Weight_kg", "BMI", "Waist_Circumference_cm", "Body_Fat_Percentage", "Smoking_Status",
    "Alcohol_Consumption", "Physical_Activity_Level", "Exercise_Hours_Per_Week", "Daily_Steps", "Water_Intake_L",
    "Sodium_Intake_mg", "Fast_Food_Frequency_Per_Week", "Sleep_Duration_Hours", "Stress_Level", "Diabetes",
    "Hypertension", "Cardiovascular_Disease", "Heart_Failure", "Hyperlipidemia", "Kidney_Stones", "Recurrent_UTI",
    "Autoimmune_Disease", "Family_History_CKD", "Obesity", "Heart_Rate", "Respiratory_Rate", "Oxygen_Saturation",
    "Systolic_BP", "Diastolic_BP", "Blood_Pressure_Category", "Serum_Creatinine", "eGFR", "Blood_Urea_Nitrogen",
    "Albumin", "Urine_ACR", "Urine_Protein", "HbA1c", "Fasting_Glucose", "Hemoglobin", "Sodium", "Potassium",
    "Calcium", "Phosphorus", "Uric_Acid", "Total_Cholesterol", "HDL", "LDL", "Triglycerides", "CRP",
    "ACE_Inhibitor", "ARB", "Diabetes_Medication", "Statin", "Diuretic", "NSAID_Usage", "Medication_Adherence",
    "Number_of_Medications", "Frailty_Index", "Frailty_Category", "Hospital_Visits", "Emergency_Visits",
    "Specialist_Visits", "Annual_Checkups", "Health_Insurance", "Annual_Household_Income_USD",
    "Annual_Medical_Cost_USD", "Employment_Status"
]

EARLY_CATEGORICAL_OPTIONS = {
    "Sex": ["Female", "Male"],
    "Ethnicity": ["White", "Black", "Asian", "Hispanic", "Other"],
    "Country": ["USA", "UK", "Canada", "Other", "Australia", "Germany"],
    "Residence_Type": ["Urban", "Rural"],
    "Education_Level": ["High School", "Some College", "Bachelor's", "Master's", "PhD"],
    "Socioeconomic_Status": ["Low", "Middle", "High"],
    "Smoking_Status": ["Never", "Former", "Current"],
    "Alcohol_Consumption": ["Moderate", "High"],
    "Physical_Activity_Level": ["Low", "Moderate", "Sedentary", "High"],
    "Stress_Level": ["Low", "Moderate", "High"],
    "Blood_Pressure_Category": ["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"],
    "Employment_Status": ["Employed", "Unemployed/Retired"],
}

SEVERITY_CATEGORICAL_OPTIONS = dict(EARLY_CATEGORICAL_OPTIONS)
SEVERITY_CATEGORICAL_OPTIONS.update({
    "Frailty_Category": ["Frail", "Vulnerable", "Robust"],
})

EARLY_NUMERIC_DEFAULTS = {
    "Age": 55, "Height_cm": 168.0, "Weight_kg": 75.0, "BMI": 26.6, "Waist_Circumference_cm": 90.0,
    "Body_Fat_Percentage": 25.0, "Exercise_Hours_Per_Week": 3.0, "Daily_Steps": 6000, "Water_Intake_L": 1.8,
    "Sodium_Intake_mg": 2200, "Fast_Food_Frequency_Per_Week": 2, "Sleep_Duration_Hours": 7.0,
    "Heart_Rate": 72, "Respiratory_Rate": 15, "Oxygen_Saturation": 98.0, "Systolic_BP": 125,
    "Diastolic_BP": 78, "Annual_Checkups": 1, "Annual_Household_Income_USD": 60000,
}

BINARY_LABELS = {
    "Diabetes": "Diabetes", "Hypertension": "Hypertension", "Cardiovascular_Disease": "Cardiovascular disease",
    "Heart_Failure": "Heart failure", "Hyperlipidemia": "Hyperlipidemia", "Kidney_Stones": "Kidney stones",
    "Recurrent_UTI": "Recurrent UTI", "Autoimmune_Disease": "Autoimmune disease", "Family_History_CKD": "Family history of CKD",
    "Obesity": "Obesity",
}

SEVERITY_NUMERIC_DEFAULTS = {
    **EARLY_NUMERIC_DEFAULTS,
    "Serum_Creatinine": 0.9, "eGFR": 90.0, "Blood_Urea_Nitrogen": 14.0, "Albumin": 4.0,
    "Urine_ACR": 12.0, "Urine_Protein": 5.0, "HbA1c": 5.5, "Fasting_Glucose": 95.0, "Hemoglobin": 13.8,
    "Sodium": 140.0, "Potassium": 4.1, "Calcium": 9.4, "Phosphorus": 3.7, "Uric_Acid": 5.5,
    "Total_Cholesterol": 185.0, "HDL": 50.0, "LDL": 105.0, "Triglycerides": 140.0, "CRP": 1.5,
    "Number_of_Medications": 2, "Frailty_Index": 0.15, "Hospital_Visits": 1, "Emergency_Visits": 0,
    "Specialist_Visits": 0, "Annual_Medical_Cost_USD": 3000,
}

SEVERITY_BINARY = {
    **BINARY_LABELS,
    "ACE_Inhibitor": "ACE inhibitor", "ARB": "ARB", "Diabetes_Medication": "Diabetes medication", "Statin": "Statin",
    "Diuretic": "Diuretic", "Medication_Adherence": "Medication adherence",
}

MODEL_FACTS = {
    "early": {"accuracy": 0.900175, "precision": 0.669915, "recall": 0.909042, "f1": 0.771371, "auc": 0.952264,
              "tn": 29271, "fp": 3319, "fn": 674, "tp": 6736},
    "severity": {"accuracy": 0.9985, "macro_f1": 0.9692, "qwk": 0.9984, "auc": 0.999553},
}

EARLY_MODEL_COMPARISON = pd.DataFrame([
    {"Model":"Logistic Regression","Accuracy":0.90532,"Precision":0.68498,"Recall":0.90526,"F1":0.77986,"ROC-AUC":0.95242,"Average Precision":0.91083,"CV Mean Accuracy":0.90630,"CV Std":0.00287},
    {"Model":"Random Forest","Accuracy":0.93885,"Precision":0.86044,"Recall":0.79960,"F1":0.82890,"ROC-AUC":0.94808,"Average Precision":0.89377,"CV Mean Accuracy":0.94113,"CV Std":0.00090},
    {"Model":"Linear SVM","Accuracy":0.90018,"Precision":0.66992,"Recall":0.90904,"F1":0.77137,"ROC-AUC":0.95226,"Average Precision":0.91017,"CV Mean Accuracy":0.94573,"CV Std":0.00119},
])

SEVERITY_MODEL_COMPARISON = pd.DataFrame([
    {"Model":"Ordinal Logistic Regression","Accuracy":0.9228,"Weighted Precision":0.9921,"Weighted Recall":0.9228,"Weighted F1":0.9543,"Macro Precision":0.7411,"Macro Recall":0.8859,"Macro F1":0.7372,"QWK":0.9383},
    {"Model":"Random Forest","Accuracy":0.9982,"Weighted Precision":0.9981,"Weighted Recall":0.9982,"Weighted F1":0.9980,"Macro Precision":0.9846,"Macro Recall":0.9289,"Macro F1":0.9532,"QWK":0.9986},
    {"Model":"XGBoost","Accuracy":0.9985,"Weighted Precision":0.9985,"Weighted Recall":0.9985,"Weighted F1":0.9692,"Macro Precision":0.9848,"Macro Recall":0.9557,"Macro F1":0.9692,"QWK":0.9984},
])


# ================================================================
# MODEL LOADING
# ================================================================
def candidate_paths(names: list[str]) -> list[Path]:
    paths = []
    for name in names:
        for base in [MODEL_DIR, ROOT]:
            paths.append(base / name)
    return paths


def load_artifact(names: list[str]) -> tuple[Any | None, Path | None]:
    for p in candidate_paths(names):
        if not p.exists():
            continue
        try:
            if joblib:
                return joblib.load(p), p
            with open(p, "rb") as f:
                return pickle.load(f), p
        except Exception:
            try:
                with open(p, "rb") as f:
                    return pickle.load(f), p
            except Exception:
                continue
    return None, None


@st.cache_resource(show_spinner=False)
def load_models():
    early, early_path = load_artifact(["Final_CKD_Early_Screening_SVM_Pipeline.pkl"])
    severity, severity_path = load_artifact(["CKD_Severity_XGBoost.pkl"])
    scaler, scaler_path = load_artifact(["scaler.pkl"])
    num_imputer, num_imputer_path = load_artifact(["num_imputer.pkl"])
    return {
        "early": early, "early_path": early_path,
        "severity": severity, "severity_path": severity_path,
        "scaler": scaler, "scaler_path": scaler_path,
        "num_imputer": num_imputer, "num_imputer_path": num_imputer_path,
    }


MODELS = load_models()


def model_status_text(obj: Any, path: Path | None) -> str:
    if obj is None:
        return "Not found"
    return path.name if path else "Loaded"


# ================================================================
# HELPERS
# ================================================================
def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def risk_class(pred: int, score: float | None = None) -> tuple[str, str]:
    if int(pred) == 1:
        return "High CKD screening signal", "risk-high"
    return "Lower CKD screening signal", "risk-low"


def add_derived_early(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Pulse_Pressure"] = out["Systolic_BP"] - out["Diastolic_BP"]
    out["Waist_to_Height"] = out["Waist_Circumference_cm"] / out["Height_cm"]
    out["Lifestyle_Risk"] = (
        (out["Smoking_Status"] == "Current").astype(int)
        + (out["Alcohol_Consumption"] == "High").astype(int)
        + (out["Physical_Activity_Level"] == "Low").astype(int)
    )
    out["Metabolic_Risk"] = out[["Diabetes", "Hypertension", "Obesity"]].astype(int).sum(axis=1)
    out["CV_Risk"] = out[["Cardiovascular_Disease", "Heart_Failure", "Hyperlipidemia"]].astype(int).sum(axis=1)
    out["Poor_Sleep"] = ((out["Sleep_Duration_Hours"] < 6) | (out["Sleep_Duration_Hours"] > 9)).astype(int)
    return out


def patient_radar(df: pd.DataFrame) -> go.Figure:
    vals = [
        float(df["Lifestyle_Risk"].iloc[0]) / 3 * 100,
        float(df["Metabolic_Risk"].iloc[0]) / 3 * 100,
        float(df["CV_Risk"].iloc[0]) / 3 * 100,
        min(max(float(df["Waist_to_Height"].iloc[0]) / 0.8 * 100, 0), 100),
        (100.0 if (float(df["Sleep_Duration_Hours"].iloc[0]) > 9 or float(df["Sleep_Duration_Hours"].iloc[0]) < 6) else 0.0),
    ]
    cats = ["Lifestyle", "Metabolic", "CV", "Waist / Height", "Sleep flag"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", name="Patient profile"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=False, margin=dict(l=20,r=20,t=20,b=20), height=365,
    )
    return fig


def score_gauge(score: float) -> go.Figure:
    # Decision-function values are NOT probabilities; the gauge is intentionally labelled margin.
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".2f"},
        title={"text": "SVM decision margin"},
        gauge={
            "axis": {"range": [-2.5, 2.5]},
            "steps": [
                {"range": [-2.5, 0], "color": "#dff3ef"},
                {"range": [0, 2.5], "color": "#ffe7e4"},
            ],
            "threshold": {"line": {"width": 5}, "value": 0}
        },
    ))
    fig.update_layout(height=300, margin=dict(l=22,r=22,t=35,b=10))
    return fig


def form_select(label: str, options: list[str], default: str | None = None, key: str | None = None) -> str:
    opts = [str(x) for x in options]
    idx = opts.index(default) if default in opts else 0
    return st.selectbox(label, opts, index=idx, key=key)


def numeric_widget(label: str, default: float, min_value: float = 0.0, max_value: float = 1_000_000.0, step: float = 1.0, key: str | None = None):
    return st.number_input(label, min_value=float(min_value), max_value=float(max_value), value=float(default), step=float(step), key=key)


def _safe_widget_key(prefix: str, label: str) -> str:
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_").lower()
    return f"{prefix}_{slug}"


def binary_widget(label: str, default: int = 0, key: str | None = None) -> int:
    """Render a toggle with a deterministic, explicit Streamlit key."""
    widget_key = key or _safe_widget_key("toggle", label)
    v = st.toggle(label, value=bool(default), key=widget_key)
    return int(v)


def early_form() -> pd.DataFrame:
    """Single-page Early Screening input form."""
    sample = st.session_state.get("early_sample", False)
    sample_values = {
        "Age": 64, "Sex": "Female", "Ethnicity": "Hispanic", "Country": "Other", "Residence_Type": "Rural",
        "Education_Level": "Some College", "Socioeconomic_Status": "Low", "Height_cm": 153.3, "Weight_kg": 70.7,
        "BMI": 30.1, "Waist_Circumference_cm": 81.5, "Body_Fat_Percentage": 35.0, "Smoking_Status": "Never",
        "Alcohol_Consumption": "Moderate", "Physical_Activity_Level": "Sedentary", "Exercise_Hours_Per_Week": 0.2,
        "Daily_Steps": 2588, "Water_Intake_L": 1.01, "Sodium_Intake_mg": 4127, "Fast_Food_Frequency_Per_Week": 2,
        "Sleep_Duration_Hours": 6.4, "Stress_Level": "High", "Diabetes": 1, "Hypertension": 0,
        "Cardiovascular_Disease": 0, "Heart_Failure": 0, "Hyperlipidemia": 1, "Kidney_Stones": 0,
        "Recurrent_UTI": 0, "Autoimmune_Disease": 0, "Family_History_CKD": 0, "Obesity": 1, "Heart_Rate": 92,
        "Respiratory_Rate": 14, "Oxygen_Saturation": 98.1, "Systolic_BP": 138, "Diastolic_BP": 77,
        "Blood_Pressure_Category": "Hypertension Stage 1", "NSAID_Usage": 0, "Annual_Checkups": 1,
        "Health_Insurance": 0, "Annual_Household_Income_USD": 24100, "Employment_Status": "Employed",
    }
    def sv(k): return sample_values.get(k) if sample else EARLY_NUMERIC_DEFAULTS.get(k)
    def sc(k, fallback=None): return sample_values.get(k, fallback)

    vals: dict[str, Any] = {}

    st.markdown("<div class='form-section'><div class='smallcaps'>01 • PATIENT PROFILE</div><div class='section-title'>Who is being screened?</div><div class='section-sub'>Basic demographic and anthropometric information used by the early-screening model.</div></div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: vals["Age"] = numeric_widget("Age", sv("Age"), 18, 95, 1, key="early_age")
    with c2: vals["Sex"] = form_select("Sex", EARLY_CATEGORICAL_OPTIONS["Sex"], sc("Sex", "Female"), key="early_sex")
    with c3: vals["Ethnicity"] = form_select("Ethnicity", EARLY_CATEGORICAL_OPTIONS["Ethnicity"], sc("Ethnicity", "White"), key="early_ethnicity")
    c1,c2,c3 = st.columns(3)
    with c1: vals["Country"] = form_select("Country", EARLY_CATEGORICAL_OPTIONS["Country"], sc("Country", "USA"), key="early_country")
    with c2: vals["Residence_Type"] = form_select("Residence type", EARLY_CATEGORICAL_OPTIONS["Residence_Type"], sc("Residence_Type", "Urban"), key="early_residence")
    with c3: vals["Education_Level"] = form_select("Education level", EARLY_CATEGORICAL_OPTIONS["Education_Level"], sc("Education_Level", "High School"), key="early_education")
    c1,c2 = st.columns(2)
    with c1: vals["Socioeconomic_Status"] = form_select("Socioeconomic status", EARLY_CATEGORICAL_OPTIONS["Socioeconomic_Status"], sc("Socioeconomic_Status", "Middle"), key="early_socioeconomic")
    with c2: vals["Employment_Status"] = form_select("Employment status", EARLY_CATEGORICAL_OPTIONS["Employment_Status"], sc("Employment_Status", "Employed"), key="early_employment")
    c1,c2,c3 = st.columns(3)
    with c1: vals["Height_cm"] = numeric_widget("Height (cm)", sv("Height_cm"), 100, 230, 0.1, key="early_height")
    with c2: vals["Weight_kg"] = numeric_widget("Weight (kg)", sv("Weight_kg"), 20, 250, 0.1, key="early_weight")
    with c3: vals["BMI"] = numeric_widget("BMI", sv("BMI"), 10, 60, 0.1, key="early_bmi")
    c1,c2 = st.columns(2)
    with c1: vals["Waist_Circumference_cm"] = numeric_widget("Waist circumference (cm)", sv("Waist_Circumference_cm"), 40, 180, 0.1, key="early_waist")
    with c2: vals["Body_Fat_Percentage"] = numeric_widget("Body fat (%)", sv("Body_Fat_Percentage"), 2, 70, 0.1, key="early_bodyfat")

    st.markdown("<div class='form-section'><div class='smallcaps'>02 • LIFESTYLE & MEDICAL HISTORY</div><div class='section-title'>Daily habits and existing conditions</div><div class='section-sub'>Lifestyle and comorbidity indicators contribute to the engineered risk domains.</div></div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: vals["Smoking_Status"] = form_select("Smoking status", EARLY_CATEGORICAL_OPTIONS["Smoking_Status"], sc("Smoking_Status", "Never"), key="early_smoking")
    with c2: vals["Alcohol_Consumption"] = form_select("Alcohol consumption", EARLY_CATEGORICAL_OPTIONS["Alcohol_Consumption"], sc("Alcohol_Consumption", "Moderate"), key="early_alcohol")
    with c3: vals["Physical_Activity_Level"] = form_select("Physical activity", EARLY_CATEGORICAL_OPTIONS["Physical_Activity_Level"], sc("Physical_Activity_Level", "Moderate"), key="early_activity")
    c1,c2,c3 = st.columns(3)
    with c1: vals["Exercise_Hours_Per_Week"] = numeric_widget("Exercise hours / week", sv("Exercise_Hours_Per_Week"), 0, 30, 0.1, key="early_exercise")
    with c2: vals["Daily_Steps"] = numeric_widget("Daily steps", sv("Daily_Steps"), 0, 50000, 100, key="early_steps")
    with c3: vals["Water_Intake_L"] = numeric_widget("Water intake (L/day)", sv("Water_Intake_L"), 0, 10, 0.1, key="early_water")
    c1,c2,c3 = st.columns(3)
    with c1: vals["Sodium_Intake_mg"] = numeric_widget("Sodium intake (mg/day)", sv("Sodium_Intake_mg"), 0, 10000, 50, key="early_sodium")
    with c2: vals["Fast_Food_Frequency_Per_Week"] = numeric_widget("Fast-food frequency / week", sv("Fast_Food_Frequency_Per_Week"), 0, 14, 1, key="early_fastfood")
    with c3: vals["Sleep_Duration_Hours"] = numeric_widget("Sleep duration (hours)", sv("Sleep_Duration_Hours"), 0, 16, 0.1, key="early_sleep")
    vals["Stress_Level"] = form_select("Stress level", EARLY_CATEGORICAL_OPTIONS["Stress_Level"], sc("Stress_Level", "Moderate"), key="early_stress")
    st.markdown("<div class='smallcaps'>Medical history</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i,(field,label) in enumerate(BINARY_LABELS.items()):
        with cols[i%4]: vals[field] = binary_widget(label, sc(field, 0) if sample else 0, key=f"early_{field}")

    st.markdown("<div class='form-section'><div class='smallcaps'>03 • VITAL SIGNS</div><div class='section-title'>Current physiological profile</div><div class='section-sub'>Vital signs are used directly or through engineered features such as pulse pressure.</div></div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: vals["Heart_Rate"] = numeric_widget("Heart rate (bpm)", sv("Heart_Rate"), 35, 220, 1, key="early_hr")
    with c2: vals["Respiratory_Rate"] = numeric_widget("Respiratory rate", sv("Respiratory_Rate"), 5, 45, 1, key="early_rr")
    with c3: vals["Oxygen_Saturation"] = numeric_widget("Oxygen saturation (%)", sv("Oxygen_Saturation"), 70, 100, 0.1, key="early_spo2")
    c1,c2,c3 = st.columns(3)
    with c1: vals["Systolic_BP"] = numeric_widget("Systolic BP", sv("Systolic_BP"), 70, 250, 1, key="early_sbp")
    with c2: vals["Diastolic_BP"] = numeric_widget("Diastolic BP", sv("Diastolic_BP"), 40, 160, 1, key="early_dbp")
    with c3: vals["Blood_Pressure_Category"] = form_select("Blood pressure category", EARLY_CATEGORICAL_OPTIONS["Blood_Pressure_Category"], sc("Blood_Pressure_Category", "Normal"), key="early_bp_category")
    vals["NSAID_Usage"] = binary_widget("NSAID usage", sc("NSAID_Usage", 0) if sample else 0, key="early_NSAID_Usage")

    st.markdown("<div class='form-section'><div class='smallcaps'>04 • ACCESS & CONTEXT</div><div class='section-title'>Healthcare access context</div><div class='section-sub'>These fields are retained from the early-screening feature set and feed the downstream insurance intelligence view.</div></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: vals["Annual_Checkups"] = numeric_widget("Annual checkups", sv("Annual_Checkups"), 0, 20, 1, key="early_checkups")
    with c2: vals["Health_Insurance"] = binary_widget("Has health insurance", sc("Health_Insurance", 0) if sample else 1, key="early_Health_Insurance")
    vals["Annual_Household_Income_USD"] = numeric_widget("Annual household income (USD)", sv("Annual_Household_Income_USD"), 500, 1_000_000, 500, key="early_income")
    st.markdown("<div class='note'>The derived features used by the saved early-screening pipeline are calculated before prediction: Pulse Pressure, Waist-to-Height, Lifestyle Risk, Metabolic Risk, CV Risk and Poor Sleep.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    return add_derived_early(pd.DataFrame([vals]))

def run_early_prediction(df: pd.DataFrame):
    """Run the deployed early-screening model once and persist the result."""
    if MODELS["early"] is None:
        st.error("Early-screening model not found. Place Final_CKD_Early_Screening_SVM_Pipeline.pkl in models/.")
        return False
    try:
        pred = int(MODELS["early"].predict(df[EARLY_FEATURES])[0])
        score = float(MODELS["early"].decision_function(df[EARLY_FEATURES])[0])
    except Exception as e:
        st.exception(e)
        return False
    st.session_state["last_early_df"] = df.copy()
    st.session_state["last_early_pred"] = pred
    st.session_state["last_early_score"] = score
    return True


def prediction_section(df: pd.DataFrame):
    """Render the stored early-screening result. Prediction is performed separately."""
    st.markdown("<div class='section-title'>Screening outcome</div><div class='section-sub'>The deployed SVM gives a binary CKD screening signal. The margin is not a calibrated probability.</div>", unsafe_allow_html=True)
    if MODELS["early"] is None:
        st.error("Early-screening model not found. Place Final_CKD_Early_Screening_SVM_Pipeline.pkl in models/.")
        return
    pred = st.session_state.get("last_early_pred")
    score = st.session_state.get("last_early_score")
    if pred is None or score is None:
        if not run_early_prediction(df):
            return
        pred = st.session_state["last_early_pred"]
        score = st.session_state["last_early_score"]
    title, cls = risk_class(pred)
    st.markdown(f"<div class='risk-banner {cls}'><div class='smallcaps'>EARLY SCREENING</div><h2 style='margin:4px 0 6px'>{title}</h2><div style='color:#53666e'>Class {pred} from the deployed Linear SVM pipeline.</div></div>", unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a: metric_card("Screening class", "1 • High signal" if pred else "0 • Lower signal", "Binary CKD target used in the notebook")
    with b: metric_card("Decision margin", f"{score:.2f}", "0 is the SVM decision boundary")
    with c: metric_card("CKD prevalence in source data", "18.53%", "Training/source population target prevalence")

    c1,c2 = st.columns([1,1])
    with c1: st.plotly_chart(score_gauge(score), use_container_width=True, config={"displayModeBar": False}, key="plotly_0")
    with c2:
        st.plotly_chart(patient_radar(df), use_container_width=True, config={"displayModeBar": False}, key="plotly_1")
    st.markdown("<div class='warning-note'>This tool is a screening / research interface, not a diagnosis. A model signal should be interpreted alongside clinical assessment and the source model's intended workflow.</div>", unsafe_allow_html=True)


def early_statistics(df: pd.DataFrame):
    """Notebook-backed Early Screening statistical analysis and visual evidence."""
    st.markdown(
        "<div class='section-title'>Statistical intelligence</div>"
        "<div class='section-sub'>"
        "Complete Early Screening model evaluation from the project notebook: candidate-model comparison, selection rationale, ROC/PR/calibration curves, error diagnostics, feature importance and deployment architecture."
        "</div>",
        unsafe_allow_html=True,
    )

    m = MODEL_FACTS["early"]
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Value": [m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]],
    })
    fig = px.bar(metrics, x="Metric", y="Value", text="Value", range_y=[0, 1.05], title="Final Linear SVM validation metrics")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=25), yaxis_title="Score", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="early_stats_final_metrics")

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Sensitivity / Recall", f"{m['recall']*100:.1f}%", "CKD class")
    with b:
        metric_card("False negatives", f"{m['fn']:,}", "Validation cases")
    with c:
        metric_card("False positives", f"{m['fp']:,}", "Validation cases")
    with d:
        metric_card("Specificity", f"{m['tn']/(m['tn']+m['fp'])*100:.1f}%", "Derived from validation matrix")

    plot_dir = Path(__file__).resolve().parent / "assets" / "early_plots"

    st.markdown(
        "<div class='section-title'>Complete Early Screening evidence</div>"
        "<div class='section-sub'>Every major diagnostic figure produced in the Early Screening notebook is reproduced below in the same analytical workflow.</div>",
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "Model comparison & selection",
        "ROC • PR • Calibration",
        "Errors • Sensitivity • Cost",
        "Feature importance",
        "Architecture • XGBoost benchmark",
    ])

    with t1:
        st.markdown("<div class='smallcaps'>1. CANDIDATE MODEL COMPARISON</div>", unsafe_allow_html=True)
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Average Precision", "CV Mean Accuracy", "CV Std"]
        st.dataframe(
            EARLY_MODEL_COMPARISON[display_cols].style.format({c: "{:.5f}" for c in display_cols if c != "Model"}),
            use_container_width=True,
            hide_index=True,
        )
        comp = EARLY_MODEL_COMPARISON.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Mean Accuracy"],
            var_name="Metric",
            value_name="Score",
        )
        figc = px.bar(comp, x="Model", y="Score", color="Metric", barmode="group", range_y=[0.6, 1.0], title="Validation performance across candidate models")
        figc.update_layout(height=470, margin=dict(l=20, r=20, t=55, b=25), legend_title=None)
        st.plotly_chart(figc, use_container_width=True, config={"displayModeBar": False}, key="early_stats_model_comparison_full")

        st.markdown(
            "<div class='insight-card'><div class='smallcaps'>WHY LINEAR SVM WAS USED FOR EARLY SCREENING</div>"
            "<b>The selection is a screening trade-off, not a highest-accuracy contest.</b><br>"
            "Random Forest achieved the strongest single validation accuracy (93.885%), but its CKD recall was only 79.960%. "
            "Logistic Regression reached 90.526% recall. Linear SVM reached 90.904% recall and the highest five-fold mean accuracy (94.573%) "
            "with a small cross-validation spread (SD 0.00119). For a screening-oriented interface, this combination makes the SVM a defensible final choice because missing positive cases is especially important. "
            "The screening-priority interpretation is an inference from the reported validation metrics, not a quoted selection rule from the notebook.</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Logistic recall", "90.53%", "Validation")
        with c2:
            metric_card("RF accuracy", "93.89%", "Highest single split")
        with c3:
            metric_card("SVM recall", "90.90%", "Highest of the three")
        with c4:
            metric_card("SVM CV accuracy", "94.57%", "5-fold mean")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            p = plot_dir / "step19_roc_curve_comparison_0.png"
            if p.exists():
                st.image(str(p), caption="Figure — ROC Curve Comparison", use_container_width=True)
            st.markdown(
                "<div class='note'><b>ROC insight.</b> Logistic Regression and Linear SVM have almost identical ROC-AUC values (~0.952), while Random Forest is lower (~0.948). Therefore, SVM was not chosen because it dramatically dominates ROC discrimination; its advantage is the recall / stability trade-off shown elsewhere.</div>",
                unsafe_allow_html=True,
            )
        with c2:
            p = plot_dir / "step20_precision_recall_curve_comparison_0.png"
            if p.exists():
                st.image(str(p), caption="Figure — Precision–Recall Curve Comparison", use_container_width=True)
            st.markdown(
                "<div class='note'><b>Precision–recall insight.</b> Because CKD is the less frequent class in the source data (18.53%), the PR view is especially informative: it shows how positive-case capture trades against precision instead of relying on accuracy alone.</div>",
                unsafe_allow_html=True,
            )
        p = plot_dir / "step29_calibration_curve_analysis_0.png"
        if p.exists():
            st.image(str(p), caption="Figure — Calibration Curve Comparison", use_container_width=True)
        st.markdown(
            "<div class='insight-card'><div class='smallcaps'>CALIBRATION INTERPRETATION</div>"
            "The notebook evaluates how model scores relate to observed event frequencies. These curves are comparative model diagnostics. "
            "The deployed Linear SVM's raw decision margin is <b>not</b> a calibrated clinical probability, so the app does not label the margin as a patient percentage risk.</div>",
            unsafe_allow_html=True,
        )

    with t3:
        p = plot_dir / "step31_confusion_matrix_comparison_0.png"
        if p.exists():
            st.image(str(p), caption="Figure — Confusion Matrix Comparison", use_container_width=True)
        st.markdown(
            "<div class='note'><b>Confusion-matrix insight.</b> The validation matrix makes the screening trade-off explicit: the final SVM has 6,736 true positives and 674 false negatives, alongside 29,271 true negatives and 3,319 false positives.</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            p = plot_dir / "step34_sensitivity_recall_comparison_table_1.png"
            if p.exists():
                st.image(str(p), caption="Figure — Sensitivity / Recall Comparison", use_container_width=True)
        with c2:
            p = plot_dir / "step35_false_negative_rate_fnr_comparison_1.png"
            if p.exists():
                st.image(str(p), caption="Figure — False Negative Rate Comparison", use_container_width=True)
        st.markdown(
            "<div class='insight-card'><div class='smallcaps'>SCREENING ERROR INTERPRETATION</div>"
            "Linear SVM sensitivity is 90.90%, Logistic Regression 90.53%, and Random Forest 79.96%. Correspondingly, FNR is 9.10%, 9.47% and 20.04%. "
            "The lower SVM false-negative rate is a major reason it is suitable for the screening stage when the priority is to reduce missed CKD-positive cases.</div>",
            unsafe_allow_html=True,
        )

        p = plot_dir / "cell_21_1.png"
        if p.exists():
            st.image(str(p), caption="Figure — False Case Cost Comparison", use_container_width=True)
        st.markdown(
            "<div class='note'><b>Cost insight.</b> The notebook's total misclassification-cost values are 10,105 for Logistic Regression, 15,811 for Random Forest and 10,059 for Linear SVM. Under the notebook's cost definition, SVM has the lowest total cost among these three candidates.</div>",
            unsafe_allow_html=True,
        )

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            p = plot_dir / "step26_feature_importance_analysis_2.png"
            if p.exists():
                st.image(str(p), caption="Figure — Random Forest Top-20 Feature Importance", use_container_width=True)
            st.markdown(
                "<div class='note'><b>Random Forest.</b> The notebook's tree-based importance places Age first, followed by Metabolic Risk, Diabetes and Hypertension among the prominent features.</div>",
                unsafe_allow_html=True,
            )
        with c2:
            p = plot_dir / "step26_feature_importance_analysis_5.png"
            if p.exists():
                st.image(str(p), caption="Figure — Logistic Regression Top-20 Feature Importance", use_container_width=True)
            st.markdown(
                "<div class='note'><b>Logistic Regression.</b> The absolute-coefficient ranking highlights Age, BMI, Diabetes and Hypertension as influential predictors in the fitted linear model.</div>",
                unsafe_allow_html=True,
            )
        p = plot_dir / "step26_feature_importance_analysis_8.png"
        if p.exists():
            st.image(str(p), caption="Figure — Linear SVM Permutation Importance", use_container_width=True)
        st.markdown(
            "<div class='insight-card'><div class='smallcaps'>CROSS-MODEL INSIGHT</div>"
            "Age, adiposity/body-composition variables, diabetes and hypertension recur across the candidate models. "
            "The importance measures are method-specific — tree importance, coefficient magnitude and permutation importance — so their numerical magnitudes should not be compared directly.</div>",
            unsafe_allow_html=True,
        )

    with t5:
        p = plot_dir / "step1_final_pipeline_creation_1.png"
        if p.exists():
            st.image(str(p), caption="Figure — Final Early Screening Pipeline", use_container_width=True)
        st.markdown(
            "<div class='card'><div class='smallcaps'>FINAL DEPLOYMENT PIPELINE</div>"
            "<b>Raw patient inputs → median imputation / scaling of numerical variables → one-hot encoding of categorical variables → SMOTE during training → LinearSVC decision boundary.</b><br>"
            "The app calls the saved fitted pipeline directly, preserving the preprocessing learned during model development.</div>",
            unsafe_allow_html=True,
        )
        p = plot_dir / "step32_xgboost_classifier_early_ckd_screening_2.png"
        if p.exists():
            st.image(str(p), caption="Figure — Early CKD XGBoost Benchmark", use_container_width=True)
        st.markdown(
            "<div class='note'><b>XGBoost benchmark insight.</b> The notebook reports 94.84% accuracy, 81.20% recall and 85.36% F1 for the early XGBoost experiment. It is therefore an important benchmark, but its lower recall than the final SVM makes it less aligned with the screening-oriented selection described above.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='card'><div class='smallcaps'>PATIENT-LEVEL DERIVED FEATURES</div>"
        "The current profile also exposes the transparent features created in the notebook: Pulse Pressure, Waist-to-Height Ratio, Lifestyle Risk (0–3), Metabolic Risk (0–3), Cardiovascular Risk (0–3) and Poor Sleep (0/1). These are explanatory model inputs, not independent diagnoses.</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    items = [
        ("Pulse Pressure", float(df["Pulse_Pressure"].iloc[0]), "mmHg"),
        ("Waist / Height", float(df["Waist_to_Height"].iloc[0]), "ratio"),
        ("Lifestyle Risk", int(df["Lifestyle_Risk"].iloc[0]), "0–3"),
        ("Metabolic Risk", int(df["Metabolic_Risk"].iloc[0]), "0–3"),
        ("CV Risk", int(df["CV_Risk"].iloc[0]), "0–3"),
        ("Poor Sleep", int(df["Poor_Sleep"].iloc[0]), "0/1"),
    ]
    for i, (lab, val, note) in enumerate(items):
        with cols[i]:
            metric_card(lab, f"{val:.2f}" if isinstance(val, float) else str(val), note)

def early_plots(df: pd.DataFrame):
    st.markdown("<div class='section-title'>Patient visual lab</div><div class='section-sub'>Additional visual diagnostics beyond the model-fitting figures in the notebooks.</div>", unsafe_allow_html=True)
    vit = pd.DataFrame({
        "Measure":["Systolic BP","Diastolic BP","Heart rate","Respiratory rate","O₂ saturation"],
        "Value":[df.Systolic_BP.iloc[0],df.Diastolic_BP.iloc[0],df.Heart_Rate.iloc[0],df.Respiratory_Rate.iloc[0],df.Oxygen_Saturation.iloc[0]],
    })
    fig = px.bar(vit, x="Measure", y="Value", text="Value", title="Current vital profile")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=50,b=30))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="plotly_3")

    c1,c2 = st.columns(2)
    with c1:
        risk = pd.DataFrame({"Domain":["Lifestyle","Metabolic","Cardiovascular"],"Score":[df.Lifestyle_Risk.iloc[0],df.Metabolic_Risk.iloc[0],df.CV_Risk.iloc[0]]})
        fig2=px.bar(risk,x="Domain",y="Score",range_y=[0,3],title="Composite risk domains (not probabilities)")
        fig2.update_layout(height=330,margin=dict(l=20,r=20,t=50,b=30))
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False}, key="plotly_4")
    with c2:
        body=pd.DataFrame({"Measure":["BMI","Waist/Height×100","Body fat %"],"Value":[df.BMI.iloc[0],df.Waist_to_Height.iloc[0]*100,df.Body_Fat_Percentage.iloc[0]]})
        fig3=px.bar(body,x="Measure",y="Value",text="Value",title="Body-composition profile")
        fig3.update_layout(height=330,margin=dict(l=20,r=20,t=50,b=30))
        st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False}, key="plotly_5")


def severity_form(base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    preset = {}
    if base_df is not None:
        for k in set(SEVERITY_RAW_FEATURES).intersection(base_df.columns):
            preset[k] = base_df[k].iloc[0]
    tabs=st.tabs(["① Core profile","② Kidney / metabolic labs","③ Treatment & frailty","④ Utilization & access"])
    vals:dict[str,Any]={}
    def pv(k, default): return preset.get(k, default)
    with tabs[0]:
        fields=["Age","Sex","Ethnicity","Country","Residence_Type","Education_Level","Socioeconomic_Status","Height_cm","Weight_kg","BMI","Waist_Circumference_cm","Body_Fat_Percentage","Smoking_Status","Alcohol_Consumption","Physical_Activity_Level","Exercise_Hours_Per_Week","Daily_Steps","Water_Intake_L","Sodium_Intake_mg","Fast_Food_Frequency_Per_Week","Sleep_Duration_Hours","Stress_Level","Systolic_BP","Diastolic_BP","Blood_Pressure_Category"]
        c=st.columns(3)
        for i,k in enumerate(fields):
            with c[i%3]:
                if k in SEVERITY_CATEGORICAL_OPTIONS: vals[k]=form_select(k.replace('_',' '),SEVERITY_CATEGORICAL_OPTIONS[k],pv(k,SEVERITY_CATEGORICAL_OPTIONS[k][0]))
                else:
                    bounds={
                        "Age":(18,95,1), "Height_cm":(100,230,0.1), "Weight_kg":(20,250,0.1),
                        "BMI":(10,60,0.1), "Waist_Circumference_cm":(40,180,0.1), "Body_Fat_Percentage":(2,70,0.1),
                        "Exercise_Hours_Per_Week":(0,30,0.1), "Daily_Steps":(0,50000,100), "Water_Intake_L":(0,10,0.1),
                        "Sodium_Intake_mg":(0,10000,50), "Fast_Food_Frequency_Per_Week":(0,14,1), "Sleep_Duration_Hours":(0,16,0.1),
                        "Systolic_BP":(70,250,1), "Diastolic_BP":(40,160,1)
                    }
                    lo,hi,step=bounds.get(k,(0,100000,0.1))
                    vals[k]=numeric_widget(k.replace('_',' '),pv(k,SEVERITY_NUMERIC_DEFAULTS.get(k,0)),lo,hi,step)
        for k in [*SEVERITY_BINARY.keys()]:
            pass
    with tabs[1]:
        labs=["Serum_Creatinine","eGFR","Blood_Urea_Nitrogen","Albumin","Urine_ACR","Urine_Protein","HbA1c","Fasting_Glucose","Hemoglobin","Sodium","Potassium","Calcium","Phosphorus","Uric_Acid","Total_Cholesterol","HDL","LDL","Triglycerides","CRP"]
        c=st.columns(3)
        for i,k in enumerate(labs):
            with c[i%3]: vals[k]=numeric_widget(k.replace('_',' '),pv(k,SEVERITY_NUMERIC_DEFAULTS[k]),0,100000,0.01)
        st.markdown("<div class='smallcaps'>Clinical / cardio history</div>", unsafe_allow_html=True)
        c=st.columns(4)
        for i,k in enumerate(BINARY_LABELS.keys()):
            with c[i%4]: vals[k]=binary_widget(BINARY_LABELS[k], int(pv(k,0)), key=f"severity_{k.lower()}")
        vals["Oxygen_Saturation"]=numeric_widget("Oxygen saturation (%)",pv("Oxygen_Saturation",98),70,100,0.1)
        vals["Heart_Rate"]=numeric_widget("Heart rate",pv("Heart_Rate",72),35,220,1)
        vals["Respiratory_Rate"]=numeric_widget("Respiratory rate",pv("Respiratory_Rate",15),5,45,1)
    with tabs[2]:
        c=st.columns(3)
        for i,k in enumerate(["ACE_Inhibitor","ARB","Diabetes_Medication","Statin","Diuretic","NSAID_Usage","Medication_Adherence"]):
            with c[i%3]: vals[k]=binary_widget(SEVERITY_BINARY[k], int(pv(k,0)), key=f"severity_{k.lower()}")
        c=st.columns(3)
        with c[0]: vals["Number_of_Medications"]=numeric_widget("Number of medications",pv("Number_of_Medications",2),0,30,1)
        with c[1]: vals["Frailty_Index"]=numeric_widget("Frailty index",pv("Frailty_Index",0.15),0,1,0.01)
        with c[2]: vals["Frailty_Category"]=form_select("Frailty category",SEVERITY_CATEGORICAL_OPTIONS["Frailty_Category"],pv("Frailty_Category","Vulnerable"))
        c=st.columns(4)
        for i,k in enumerate(["Annual_Checkups","Health_Insurance","Annual_Household_Income_USD","Annual_Medical_Cost_USD"]):
            with c[i]:
                if k in ["Health_Insurance"]: vals[k]=binary_widget(k.replace('_',' '),int(pv(k,1)), key=f"severity_{k}")
                else: vals[k]=numeric_widget(k.replace('_',' '),pv(k,SEVERITY_NUMERIC_DEFAULTS[k]),0,1_000_000,500 if 'Income' in k or 'Cost' in k else 1)
    with tabs[3]:
        c=st.columns(4)
        for i,k in enumerate(["Hospital_Visits","Emergency_Visits","Specialist_Visits"]):
            with c[i]: vals[k]=numeric_widget(k.replace('_',' '),pv(k,0),0,50,1)
        vals["Employment_Status"]=form_select("Employment status",SEVERITY_CATEGORICAL_OPTIONS["Employment_Status"],pv("Employment_Status","Employed"))
        st.markdown("<div class='note'>The severity notebook encodes the target as Healthy → 0, Mild CKD → 1, Moderate CKD → 2, Severe CKD → 3. The saved XGBoost model was trained after one-hot encoding 75 raw predictors and scaling the resulting 94-column matrix.</div>", unsafe_allow_html=True)
    return pd.DataFrame([vals])


def prepare_severity_input(raw: pd.DataFrame) -> np.ndarray:
    df=raw.copy()
    # Fill numeric columns with saved median imputer when available.
    if MODELS["num_imputer"] is not None and hasattr(MODELS["num_imputer"], "feature_names_in_"):
        cols=list(MODELS["num_imputer"].feature_names_in_)
        for c in cols:
            if c in df.columns:
                pass
        # transform only available numeric imputer columns
        sub=df[cols]
        arr=MODELS["num_imputer"].transform(sub)
        df.loc[:,cols]=arr
    # The training notebook used get_dummies(drop_first=True) on the raw DataFrame.
    cat_cols=[c for c in df.columns if df[c].dtype==object]
    df=pd.get_dummies(df,columns=cat_cols,drop_first=True)
    # scaler.feature_names_in_ preserves the exact transformed training-column order.
    if MODELS["scaler"] is not None and hasattr(MODELS["scaler"], "feature_names_in_"):
        feature_cols=list(MODELS["scaler"].feature_names_in_)
        df=df.reindex(columns=feature_cols,fill_value=0)
        return MODELS["scaler"].transform(df)
    # Fallback to expected 94 columns only if scaler lacks feature names.
    arr=df.to_numpy(dtype=float)
    return MODELS["scaler"].transform(arr) if MODELS["scaler"] is not None else arr


def severity_prediction(raw: pd.DataFrame):
    if MODELS["severity"] is None or MODELS["scaler"] is None:
        st.warning("Severity model artifacts are incomplete. Add CKD_Severity_XGBoost.pkl, scaler.pkl and num_imputer.pkl to models/.")
        return
    try:
        X=prepare_severity_input(raw)
        model=MODELS["severity"]
        pred=int(model.predict(X)[0])
        probs=model.predict_proba(X)[0] if hasattr(model,"predict_proba") else None
    except Exception as e:
        st.exception(e); return
    names={0:"Healthy",1:"Mild CKD",2:"Moderate CKD",3:"Severe CKD"}
    name=names.get(pred,str(pred))
    st.session_state["last_severity_raw"]=raw
    st.session_state["last_severity_pred"]=pred
    st.session_state["last_severity_probs"]=probs
    st.markdown(f"<div class='risk-banner {'risk-low' if pred==0 else ('risk-mid' if pred<3 else 'risk-high')}'><div class='smallcaps'>CLINICAL SCREENING</div><h2 style='margin:4px 0 6px'>{name}</h2><div style='color:#53666e'>Four-class severity output from the saved XGBoost model.</div></div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    for i,(lab,idx) in enumerate([("Healthy",0),("Mild",1),("Moderate",2),("Severe",3)]):
        val=float(probs[idx]) if probs is not None and len(probs)>idx else float(idx==pred)
        with [c1,c2,c3,c4][i]: metric_card(lab,f"{val*100:.1f}%","Model class score")
    if probs is not None:
        p=pd.DataFrame({"Severity":list(names.values()),"Score":probs})
        fig=px.bar(p,x="Severity",y="Score",text="Score",range_y=[0,1],title="Severity score profile")
        fig.update_traces(texttemplate='%{text:.1%}',textposition='outside')
        fig.update_layout(height=350,margin=dict(l=20,r=20,t=50,b=25))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False}, key="plotly_6")


def insurance_page():
    st.markdown("<div class='section-title'>Insurance Intelligence</div><div class='section-sub'>The insurance layer from the Early Screening notebook, translated into an aggregated portfolio view plus patient-specific guidance.</div>", unsafe_allow_html=True)

    pred = st.session_state.get("last_early_pred")
    df = st.session_state.get("last_early_df")

    t1, t2, t3 = st.tabs(["Patient Guidance", "Portfolio Segmentation", "Risk Landscape"])

    with t1:
        if pred is None or df is None:
            st.markdown("<div class='note'>Run Early Screening first. Once a screening result exists, this tab will use the entered insurance status and screening outcome.</div>", unsafe_allow_html=True)
        else:
            insured = int(df["Health_Insurance"].iloc[0])
            high = int(pred) == 1
            if high and not insured:
                title = "High-priority coverage review"
                body = "The current Early Screening signal is high-risk and the entered profile indicates no insurance. This is a decision-support flag, not an underwriting decision."
                cls = "risk-high"
            elif high and insured:
                title = "Review coverage adequacy"
                body = "The current Early Screening signal is high-risk and insurance is indicated. Consider reviewing diagnostic coverage, specialist access and continuity of care."
                cls = "risk-mid"
            elif not high and not insured:
                title = "Preventive access focus"
                body = "The current screening signal is lower-risk, but the entered profile indicates no insurance. The notebook's segmentation highlights this as an access context worth reviewing."
                cls = "risk-low"
            else:
                title = "Maintain preventive readiness"
                body = "The current screening signal is lower-risk and insurance is indicated. Keep routine preventive access and records organised."
                cls = "risk-low"
            st.markdown(f"<div class='risk-banner {cls}'><div class='smallcaps'>PATIENT-SPECIFIC GUIDANCE</div><h2 style='margin:4px 0 6px'>{title}</h2><p style='margin:0;color:#53666e'>{body}</p></div>", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1: metric_card("Screening signal", "High CKD risk" if high else "Low CKD risk", "From Early Screening")
            with c2: metric_card("Insurance", "Has insurance" if insured else "No insurance", "Entered profile")
            with c3: metric_card("Priority segment", "Yes" if (high and not insured) else "No", "Decision-support flag")
            st.markdown("<div class='note'>This patient-facing layer uses the notebook's predicted-risk + insurance segmentation concept. It is not an insurance recommendation, premium estimate, eligibility decision, or medical diagnosis.</div>", unsafe_allow_html=True)

    # Exact aggregate results from Early Screening notebook validation cohort.
    seg = pd.DataFrame({
        "Insurance Status": ["Has Insurance", "No Insurance"],
        "High CKD Risk": [6249, 3806],
        "Low CKD Risk": [19560, 10385],
        "Members": [25809, 14191],
        "High-Risk %": [24.21, 26.82],
    })
    risk_distribution = pd.DataFrame({
        "Risk Level": ["Low Risk", "High Risk"],
        "Members": [29945, 10055],
        "Percentage": [74.86, 25.14],
    })

    with t2:
        st.markdown("<div class='section-title'>Early Screening → Insurance segmentation</div><div class='section-sub'>Aggregated validation-cohort results reproduced from the notebook; no individual patient records are displayed.</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Validation cohort", "40,000", "Records")
        with c2: metric_card("Insured", "25,809", "64.52%")
        with c3: metric_card("Uninsured", "14,191", "35.48%")
        with c4: metric_card("High risk", "10,055", "25.14%")

        st.dataframe(seg[["Insurance Status","High CKD Risk","Low CKD Risk","Members","High-Risk %"]], use_container_width=True, hide_index=True)
        fig = px.bar(seg, x="Insurance Status", y=["High CKD Risk","Low CKD Risk"], barmode="stack", title="Predicted CKD Risk by Health Insurance")
        fig.update_layout(height=420, margin=dict(l=20,r=20,t=55,b=30), yaxis_title="Number of individuals")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="plotly_7")
        fig2 = px.bar(seg, x="Insurance Status", y="High-Risk %", text="High-Risk %", range_y=[0,35], title="High-risk share by insurance status")
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_layout(height=380, margin=dict(l=20,r=20,t=55,b=30), yaxis_title="High CKD risk (%)")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key="plotly_8")
        st.markdown("<div class='insight-card'><div class='smallcaps'>NOTEBOOK INSIGHT</div><b>Higher predicted-risk share in the uninsured segment.</b><br>The validation cohort shows 26.82% predicted high CKD risk among records without insurance versus 24.21% among insured records — a descriptive association in this cohort, not evidence of causation.</div>", unsafe_allow_html=True)
        st.markdown("<div class='insight-card'><div class='smallcaps'>HIGH-PRIORITY SEGMENT</div><b>3,806 records</b> were both predicted high-risk and uninsured, representing <b>9.52%</b> of the 40,000-record validation cohort.</div>", unsafe_allow_html=True)

    with t3:
        st.markdown("<div class='section-title'>Portfolio risk landscape</div><div class='section-sub'>A compact visual summary of the same aggregated notebook segmentation.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig3 = px.pie(risk_distribution, values="Members", names="Risk Level", hole=0.62, title="Overall predicted-risk mix")
            fig3.update_layout(height=380, margin=dict(l=15,r=15,t=55,b=15))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False}, key="plotly_9")
        with c2:
            high_pct = seg[["Insurance Status","High-Risk %"]].copy()
            fig4 = px.bar(high_pct, x="Insurance Status", y="High-Risk %", text="High-Risk %", range_y=[0,35], title="High-risk percentage by coverage")
            fig4.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig4.update_layout(height=380, margin=dict(l=15,r=15,t=55,b=15), yaxis_title="High-risk share (%)")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False}, key="plotly_10")
        st.markdown("<div class='note'>The source notebook also generates an executive sunburst and a patient-level portfolio bubble chart using country, insurance, predicted risk, socioeconomic status, age, BMI, and household-income variables. This deployed app intentionally keeps this public-facing view aggregated to avoid exposing patient-level records.</div>", unsafe_allow_html=True)

def model_intelligence_page():
    st.markdown("<div class='hero'><div class='kicker'>MODEL INTELLIGENCE</div><h1>Model Comparison & Rationale</h1><p>Transparent view of the candidate models tested in the notebooks, what they measured, and why the deployed models were chosen.</p></div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Early Screening", "Clinical Severity", "Why these models", "Model cards"])

    with t1:
        st.markdown("<div class='section-title'>Binary CKD screening comparison</div><div class='section-sub'>Validation results reproduced from the Early Screening notebook.</div>", unsafe_allow_html=True)
        st.dataframe(EARLY_MODEL_COMPARISON.style.format({c:"{:.4f}" for c in EARLY_MODEL_COMPARISON.columns if c != "Model"}), use_container_width=True, hide_index=True)
        plot_df=EARLY_MODEL_COMPARISON.melt(id_vars="Model", value_vars=["Accuracy","Recall","F1","ROC-AUC","CV Mean Accuracy"], var_name="Metric", value_name="Score")
        fig=px.bar(plot_df,x="Model",y="Score",color="Metric",barmode="group",range_y=[0.6,1.0],title="Early Screening — comparative validation metrics")
        fig.update_layout(height=430,margin=dict(l=20,r=20,t=55,b=25))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False}, key="plotly_11")
        st.markdown("<div class='risk-banner risk-mid'><div class='smallcaps'>WHY LINEAR SVM WAS DEPLOYED</div><h3 style='margin:4px 0 7px'>Screening priority: catch more CKD signals</h3><p style='margin:0;color:#53666e'>Random Forest achieved the highest single validation accuracy (0.93885), but Linear SVM achieved the highest 5-fold mean accuracy among the three candidates (0.94573) and the highest recall (0.90904). The final SVM therefore gives the project a strong screening-oriented balance, where reducing missed CKD cases is especially important. This rationale is an evidence-based interpretation of the notebook metrics; the notebook does not contain a single explicit sentence declaring this selection rule.</p></div>",unsafe_allow_html=True)
        m=MODEL_FACTS["early"]
        c1,c2,c3,c4=st.columns(4)
        with c1: metric_card("Recall",f"{m['recall']*100:.2f}%","Final Linear SVM")
        with c2: metric_card("CV mean",f"{EARLY_MODEL_COMPARISON.loc[2,'CV Mean Accuracy']*100:.2f}%","5-fold accuracy")
        with c3: metric_card("False negatives",str(m['fn']),"Validation cohort")
        with c4: metric_card("ROC-AUC",f"{m['auc']:.4f}","Final SVM")

    with t2:
        st.markdown("<div class='section-title'>Four-class CKD severity comparison</div><div class='section-sub'>Healthy, Mild, Moderate and Severe — compared using the metrics reported in the Severity notebook.</div>", unsafe_allow_html=True)
        st.dataframe(SEVERITY_MODEL_COMPARISON.style.format({c:"{:.4f}" for c in SEVERITY_MODEL_COMPARISON.columns if c != "Model"}), use_container_width=True, hide_index=True)
        plot_df=SEVERITY_MODEL_COMPARISON.melt(id_vars="Model", value_vars=["Accuracy","Macro Recall","Macro F1","QWK"], var_name="Metric", value_name="Score")
        fig=px.bar(plot_df,x="Model",y="Score",color="Metric",barmode="group",range_y=[0.65,1.02],title="Clinical Severity — comparative validation metrics")
        fig.update_layout(height=430,margin=dict(l=20,r=20,t=55,b=25))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False}, key="plotly_12")
        st.markdown("<div class='risk-banner risk-low'><div class='smallcaps'>WHY XGBOOST WAS DEPLOYED</div><h3 style='margin:4px 0 7px'>Best overall class-sensitive balance</h3><p style='margin:0;color:#53666e'>XGBoost produced the highest reported accuracy (0.9985), the highest macro recall (0.9557) and the highest macro F1 (0.9692) among the three severity candidates. Random Forest has a marginally higher QWK (0.9986 versus 0.9984), but XGBoost performs better on the class-balanced metrics that reflect performance across the severity categories.</p></div>",unsafe_allow_html=True)

    with t3:
        st.markdown("<div class='section-title'>Why compare these model families?</div>", unsafe_allow_html=True)
        cards=[
            ("Logistic / Ordinal Logistic","Baseline + interpretability","A simple linear baseline establishes how much predictive structure can be captured without tree ensembles. The severity notebook explicitly evaluates an ordinal logistic approach.","Baseline"),
            ("Random Forest","Nonlinear interactions","Tree ensembles capture nonlinear relationships and interactions without requiring the same linear decision boundary assumptions. It is a useful benchmark for both screening and severity tasks.","Ensemble"),
            ("Linear SVM","Screening margin + recall","The early notebook evaluates Linear SVM with SMOTE and later saves a final deployable pipeline. Its recall and cross-validation profile make it attractive for a screening-oriented task.","Deployed — Early"),
            ("XGBoost","Strong multiclass performance","The severity notebook compares XGBoost against ordinal logistic regression and Random Forest; XGBoost leads on accuracy, macro recall and macro F1 in the reported comparison.","Deployed — Severity"),
        ]
        for i in range(0,len(cards),2):
            cols=st.columns(2)
            for col,card in zip(cols,cards[i:i+2]):
                title,sub,body,badge=card
                with col:
                    st.markdown(f"<div class='card' style='min-height:180px'><span class='badge badge-teal'>{badge}</span><h3 style='margin:9px 0 4px'>{title}</h3><div class='smallcaps'>{sub}</div><p style='color:#65747e'>{body}</p></div>",unsafe_allow_html=True)
        st.markdown("<div class='section-title'>The selection principle</div><div class='card'><b>Early Screening:</b> prioritize sensitivity to CKD signals and stability across validation folds, rather than selecting solely by raw accuracy. <br><br><b>Clinical Severity:</b> prioritize multiclass performance across all severity levels, so macro recall and macro F1 matter alongside accuracy and QWK.</div>",unsafe_allow_html=True)

    with t4:
        st.markdown("<div class='section-title'>Deployed model cards</div>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("<div class='card'><span class='badge badge-green'>DEPLOYED</span><h2 style='margin:8px 0'>Linear SVM</h2><p><b>Task:</b> Early binary CKD screening</p><p><b>Pipeline:</b> median imputation → standardisation → one-hot encoding → SMOTE → LinearSVC</p><p><b>Inputs:</b> 49 engineered features</p><p><b>Validation:</b> 90.02% accuracy · 90.90% recall · 95.23% ROC-AUC</p><p><b>Why:</b> strong recall + best 5-fold mean accuracy among the three early candidates.</p></div>",unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'><span class='badge badge-green'>DEPLOYED</span><h2 style='margin:8px 0'>XGBoost</h2><p><b>Task:</b> Four-class clinical severity</p><p><b>Classes:</b> Healthy · Mild · Moderate · Severe</p><p><b>Inputs:</b> 75 raw predictors with one-hot expansion and numeric preprocessing</p><p><b>Validation:</b> 99.85% accuracy · 96.92% macro F1 · 99.84% QWK</p><p><b>Why:</b> best reported accuracy, macro recall and macro F1 in the severity comparison.</p></div>",unsafe_allow_html=True)
        st.markdown("<div class='warning-note'>These are validation results from the project notebooks, not a claim of clinical effectiveness in real-world patients. The model-selection explanations above distinguish notebook-reported facts from evidence-based interpretation.</div>",unsafe_allow_html=True)


def kidney_svg(stroke="currentColor", fill="none"):
    return f"""<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M26 9c-9 0-15 8-15 19 0 14 7 24 17 24 7 0 10-5 10-11V27c0-10-4-18-12-18Z" fill="{fill}" stroke="{stroke}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M38 9c9 0 15 8 15 19 0 14-7 24-17 24-7 0-10-5-10-11V27c0-10 4-18 12-18Z" fill="{fill}" stroke="{stroke}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M31 27c1.8-3 4-4.4 7-4.4" fill="none" stroke="{stroke}" stroke-width="2.4" stroke-linecap="round"/></svg>"""


def home_page():
    st.markdown(f"<div class='hero'><div class='hero-row'><span class='kidney-mark'>{kidney_svg('white','none')}</span><div><div class='kicker'>RENALIS • RENAL ANALYTICS & INTELLIGENCE SYSTEM</div><h1>RENALIS</h1></div></div><p>A calm, visual workflow for early CKD screening, statistical interpretation, model intelligence and insurance-oriented decision support — built around your model notebooks and saved artifacts.</p><div class='brand-pill'>RESEARCH / DEMONSTRATION PLATFORM</div><div class='hero-kidney-note'>Kidney-focused analytics • screening • model intelligence</div></div>",unsafe_allow_html=True)
    c=st.columns(4)
    with c[0]: metric_card("Source dataset", "200,000", "Patients in the notebook")
    with c[1]: metric_card("Early features", "49", "After feature engineering")
    with c[2]: metric_card("Early ROC-AUC", "0.9523", "Final SVM validation")
    with c[3]: metric_card("Severity Macro F1", "0.9692", "XGBoost validation")
    st.markdown("<div class='section-title'>Project workflow</div>",unsafe_allow_html=True)
    cols=st.columns(4)
    workflow=[("01","EARLY SCREENING","Patient inputs → engineered profile → binary CKD signal"),("02","CLINICAL SCREENING","75-predictor clinical profile → 4-class severity"),("03","INSURANCE","Screening signal + coverage context → guidance"),("04","MODEL INSIGHTS","Metrics, visual diagnostics, validation context")]
    for col,(n,title,desc) in zip(cols,workflow):
        with col: st.markdown(f"<div class='icon-tile'><div class='icon-bubble'>{kidney_svg('#0a6b60','none')}</div><div class='smallcaps'>{n}</div><h3 style='margin:5px 0'>{title}</h3><p style='color:#65747e'>{desc}</p></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-title'>How to use the app</div>",unsafe_allow_html=True)
    st.markdown("<div class='card'><b>Step 1</b> Run Early Screening with the profile values available to you. <b>Step 2</b> Review the outcome, model margin and profile-level risk domains. <b>Step 3</b> For a richer clinical view, complete the Clinical Screening section with diagnostic markers. <b>Step 4</b> Open Insurance Intelligence for the coverage-context layer.</div>",unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Project workers</div>",unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='smallcaps'>CREATED BY</div><h3 style='margin:6px 0'>Ms Bhattacharya & Mr Sinha</h3><p style='color:#65747e;margin:0'>CKD risk modelling, statistical analysis and RENALIS application development.</p></div>",unsafe_allow_html=True)
    st.markdown("<div class='feature-strip'><span class='feature-chip'>🫘 Renal focus</span><span class='feature-chip'>📊 Statistical insights</span><span class='feature-chip'>🧠 Model intelligence</span><span class='feature-chip'>🛡 Coverage context</span></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AI model system</div>",unsafe_allow_html=True)
    cards=[("Early Screening Model","Final_CKD_Early_Screening_SVM_Pipeline.pkl",MODELS["early"]), ("Clinical Severity Model","CKD_Severity_XGBoost.pkl",MODELS["severity"]), ("Severity Scaler","scaler.pkl",MODELS["scaler"]), ("Numeric Imputer","num_imputer.pkl",MODELS["num_imputer"])]
    cols=st.columns(4)
    for col,(label,filename,obj) in zip(cols,cards):
        with col:
            status="READY" if obj is not None else "MISSING"
            badge="badge-green" if obj is not None else "badge-red"
            st.markdown(f"<div class='metric-card'><span class='badge {badge}'>{status}</span><div class='metric-label' style='margin-top:10px'>{label}</div><div style='font-size:12px;font-weight:700;margin-top:7px;word-break:break-word;color:#153541'>{filename}</div></div>",unsafe_allow_html=True)
    if all(x is not None for x in [MODELS["early"],MODELS["severity"],MODELS["scaler"],MODELS["num_imputer"]]):
        st.markdown("<div class='note'><b>All model artifacts are loaded and ready.</b> The app is using the exact deployment filenames configured for RENALIS.</div>",unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-note'><b>Some model artifacts are missing.</b> Place the four required .pkl files in the <code>models/</code> folder before deployment.</div>",unsafe_allow_html=True)
    st.markdown("<div class='warning-note'>For Streamlit deployment, place the .pkl files in the <code>models/</code> folder. Never upload identifiable patient data to a public demo deployment.</div>",unsafe_allow_html=True)


def early_page():
    st.markdown("<div class='hero'><div class='kicker'>MODULE 01</div><h1>Early Screening</h1><p>Enter the same predictor layer used by the final deployable SVM pipeline. Derived variables are generated automatically.</p></div>",unsafe_allow_html=True)
    col1,col2=st.columns([3,1])
    with col2:
        if st.button("Load notebook sample",use_container_width=True): st.session_state["early_sample"]=True; st.rerun()
        if st.button("Clear sample",use_container_width=True): st.session_state["early_sample"]=False; st.rerun()
    with col1: st.markdown("<div class='note'>The final early model uses a Linear SVM pipeline with median imputation, standardisation, one-hot encoding, SMOTE and LinearSVC. Your notebook reports 90.02% validation accuracy and 90.90% CKD recall. Open <b>Model Intelligence</b> for the full candidate-model comparison and selection rationale.</div>",unsafe_allow_html=True)
    df=early_form()
    st.divider()
    if st.button("Run Early Screening",type="primary",use_container_width=True, key="run_early_screening"):
        if run_early_prediction(df):
            st.session_state["early_view"] = "results"
            st.rerun()
    if "last_early_pred" in st.session_state:
        t1,t2,t3=st.tabs(["Outcome","Statistical Analysis","Visual Lab"])
        with t1:
            prediction_section(st.session_state["last_early_df"])
        with t2:
            early_statistics(st.session_state["last_early_df"])
        with t3:
            early_plots(st.session_state["last_early_df"])
            st.markdown("<div class='section-title'>Validation model snapshot</div>", unsafe_allow_html=True)
            m=MODEL_FACTS["early"]
            cm=np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]])
            fig=px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),x=["Negative","Positive"],y=["Negative","Positive"],title="Validation confusion matrix")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False}, key="plotly_13")


def severity_page():
    st.markdown("<div class='hero'><div class='kicker'>MODULE 02</div><h1>Clinical Screening</h1><p>The final-draft severity model maps the CKD stage into four severity classes and uses XGBoost for the final prediction.</p></div>",unsafe_allow_html=True)
    st.markdown("<div class='note'>Target mapping from your notebook: Healthy = 0; Mild CKD = 1; Moderate CKD = 2; Severe CKD = 3. The source data contains 75 raw predictors after removing target/leakage fields.</div>",unsafe_allow_html=True)
    df=severity_form(st.session_state.get("last_early_df"))
    if st.button("Run Clinical Severity Screening",type="primary",use_container_width=True): severity_prediction(df)
    elif "last_severity_pred" in st.session_state: severity_prediction(st.session_state["last_severity_raw"])
    if "last_severity_pred" in st.session_state:
        p=np.array(st.session_state.get("last_severity_probs") if st.session_state.get("last_severity_probs") is not None else [1 if i==st.session_state["last_severity_pred"] else 0 for i in range(4)])
        c1,c2=st.columns(2)
        with c1:
            sev=pd.DataFrame({"Class":["Healthy","Mild","Moderate","Severe"],"Score":p})
            fig=px.pie(sev,names="Class",values="Score",hole=.58,title="Clinical severity profile")
            fig.update_layout(height=360,margin=dict(l=15,r=15,t=45,b=15))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False}, key="plotly_14")
        with c2:
            facts=pd.DataFrame({"Metric":["Accuracy","Macro F1","QWK","Multiclass AUC"],"Score":[MODEL_FACTS["severity"]["accuracy"],MODEL_FACTS["severity"]["macro_f1"],MODEL_FACTS["severity"]["qwk"],MODEL_FACTS["severity"]["auc"]]})
            fig=px.bar(facts,x="Metric",y="Score",text="Score",range_y=[0,1.05],title="Notebook validation snapshot")
            fig.update_traces(texttemplate='%{text:.3f}',textposition='outside')
            fig.update_layout(height=360,margin=dict(l=20,r=20,t=45,b=25))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False}, key="plotly_15")


def main():
    with st.sidebar:
        st.markdown("<div style='font-size:28px;font-weight:800;font-family:Playfair Display,serif;'>RENALIS</div><div style='font-size:11px;opacity:.7;letter-spacing:1.4px;margin-bottom:16px;'>CKD RISK INTELLIGENCE</div>",unsafe_allow_html=True)
        page=st.radio("Navigate",["HOME","EARLY SCREENING","CLINICAL SCREENING","INSURANCE INTELLIGENCE","MODEL INTELLIGENCE"],index=0)
        st.markdown("---")
        st.markdown(f"<div class='sidebar-foot'><b>Early model:</b> {model_status_text(MODELS['early'],MODELS['early_path'])}<br><b>Severity model:</b> {model_status_text(MODELS['severity'],MODELS['severity_path'])}<br><br>Research / demo interface. Not a diagnosis or insurance underwriting engine.</div>",unsafe_allow_html=True)
    if page=="HOME": home_page()
    elif page=="EARLY SCREENING": early_page()
    elif page=="CLINICAL SCREENING": severity_page()
    elif page=="MODEL INTELLIGENCE": model_intelligence_page()
    else: insurance_page()


if __name__ == "__main__":
    main()
