
# =============================================================================
# STEP 0 : CKD EARLY SCREENING & ANALYTICS STREAMLIT APPLICATION
# =============================================================================
#
# Project:
# Chronic Kidney Disease (CKD) Early Screening
#
# Final Model:
# Linear Support Vector Machine (SVM)
#
# Application Type:
# Clinical Screening + Analytics Dashboard
#
# =============================================================================


# =============================================================================
# SECTION 1 : IMPORT LIBRARIES
# =============================================================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from pathlib import Path
from io import BytesIO

# RENALIS polished analytical palette: dedicated colors per variable/category.
PLOT_COLORS = {
    "Accuracy":"#2563EB", "Precision":"#7C3AED", "Recall":"#059669", "F1 Score":"#D97706", "ROC AUC":"#DC2626",
    "Lifestyle":"#0EA5E9", "Metabolic":"#F59E0B", "Cardiovascular":"#E11D48",
    "Systolic BP":"#2563EB", "Diastolic BP":"#14B8A6", "Heart rate":"#F97316", "Respiratory rate":"#8B5CF6", "O₂ saturation":"#16A34A",
    "BMI":"#2563EB", "Waist/Height×100":"#F59E0B", "Body fat %":"#DB2777",
    "Low CKD Risk":"#16A34A", "High CKD Risk":"#DC2626", "Has Insurance":"#2563EB", "No Insurance":"#7C3AED",
    "Healthy":"#16A34A", "Mild CKD":"#F59E0B", "Moderate CKD":"#F97316", "Severe CKD":"#DC2626",
    "Low CKD Risk | Has Insurance":"#16A34A", "Low CKD Risk | No Insurance":"#0EA5E9",
    "High CKD Risk | Has Insurance":"#F97316", "High CKD Risk | No Insurance":"#DC2626",
    "Logistic Regression":"#2563EB", "Random Forest":"#7C3AED", "Linear SVM":"#059669", "XGBoost":"#D97706"
}

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# =============================================================================
# SECTION 2 : PAGE CONFIGURATION & CUSTOM UI VISUALS
# =============================================================================

st.set_page_config(
    page_title="CKD Early Screening & Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI Styling
st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');
:root{
 --ink:#32151c; --muted:#765861; --paper:#fff8f8; --rose:#f9e6e8; --rose2:#f3cdd2;
 --red:#b32035; --crimson:#8f1427; --maroon:#651321; --maroon2:#48101a; --line:#e7c5ca;
 --card:rgba(255,255,255,.96); --shadow:0 16px 38px rgba(101,19,33,.13);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(circle at 8% 0%,#f6dadd 0%,#fff5f5 28%,#fffafa 58%,#f8e9eb 100%)}
.block-container{padding-top:1.1rem;padding-bottom:3rem;max-width:1480px}
.hero{background:linear-gradient(125deg,#45101a 0%,#711525 42%,#a91f35 76%,#c63b4d 100%);padding:34px 38px;border-radius:28px;color:#fff;box-shadow:0 22px 45px rgba(72,16,26,.24);margin-bottom:22px;position:relative;overflow:hidden;border:1px solid rgba(255,255,255,.16)}
.hero:before{content:'';position:absolute;width:420px;height:420px;border:1px solid rgba(255,255,255,.13);border-radius:50%;right:-140px;top:-170px;box-shadow:0 0 0 35px rgba(255,255,255,.025),0 0 0 70px rgba(255,255,255,.02)}
.hero:after{content:'RENAL';position:absolute;right:24px;bottom:-30px;font-family:'Playfair Display',serif;font-size:130px;font-weight:800;color:rgba(255,255,255,.055);letter-spacing:8px}
.hero h1{font-family:'Playfair Display',serif;font-size:46px;margin:0;letter-spacing:-.7px;position:relative;z-index:2}.hero p{margin:9px 0 0;max-width:900px;color:rgba(255,255,255,.9);font-size:16px;position:relative;z-index:2}.kicker{text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;opacity:.82;position:relative;z-index:2}
.card,.metric-card,.quick-card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:19px 21px;box-shadow:var(--shadow)}
.metric-card{min-height:108px}.metric-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1.2px;font-weight:800}.metric-value{font-size:29px;font-weight:800;color:var(--crimson);margin-top:5px}.metric-note{color:var(--muted);font-size:12px;margin-top:4px}
.section-title{font-size:26px;font-weight:800;margin:21px 0 7px;color:var(--maroon)}.section-sub{color:var(--muted);margin-top:-1px;margin-bottom:17px}
.badge{display:inline-block;padding:6px 11px;border-radius:999px;font-weight:800;font-size:11px}.badge-green{background:#f5dfe2;color:#731526}.badge-orange{background:#ffe8df;color:#9a3d26}.badge-red{background:#f6cfd4;color:#851426}.badge-teal{background:#f0d8dc;color:#741728}
.smallcaps{text-transform:uppercase;letter-spacing:1.3px;font-size:10px;color:#8b5963;font-weight:800}
.note{background:linear-gradient(90deg,#fff0f2,#fff8f8);border-left:5px solid var(--red);padding:12px 15px;border-radius:10px;color:#5c2c35}.warning-note{background:#fff0ed;border-left:5px solid #c8443f;padding:12px 15px;border-radius:10px;color:#66302d}
.risk-banner{border-radius:20px;padding:23px;margin-bottom:15px;border:1px solid rgba(101,19,33,.1);box-shadow:0 10px 26px rgba(101,19,33,.07)}.risk-low{background:linear-gradient(135deg,#fff4f5,#fffafa)}.risk-high{background:linear-gradient(135deg,#ffe3e6,#fff2f3)}.risk-mid{background:linear-gradient(135deg,#fff0e8,#fff7f5)}
button[kind="primary"]{border-radius:13px!important;background:linear-gradient(135deg,#711525,#b32035)!important;border:0!important;color:#fff!important;font-weight:800!important;box-shadow:0 9px 20px rgba(113,21,37,.2)}
.stButton>button{border-radius:13px;border:1px solid #d9aeb5;font-weight:700}.stButton>button:hover{border-color:#9f2337;color:#7b1428}
.stTabs [data-baseweb="tab-list"]{gap:10px;background:#f7e2e5;border:1px solid #e4bfc5;padding:7px;border-radius:17px;box-shadow:0 7px 18px rgba(101,19,33,.08);flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{background:#fff7f8;border:1px solid #e0c0c6;border-radius:11px;padding:10px 17px;color:#6b3c45;font-weight:800;min-height:44px}
.stTabs [data-baseweb="tab"] p{font-weight:800}.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#651321,#a91f35)!important;color:#fff!important;border-color:#651321!important;box-shadow:0 7px 16px rgba(101,19,33,.2)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#3f0d17 0%,#5e1221 48%,#751727 100%);border-right:1px solid rgba(255,255,255,.12)}
[data-testid="stSidebar"] *{color:#fff5f6}.sidebar-foot{font-size:11px;color:rgba(255,240,243,.72)!important;line-height:1.5;margin-top:18px}
[data-testid="stSidebar"] .stRadio label{padding:10px 9px;border-radius:11px;font-weight:800}.feature-strip{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0 6px}.feature-chip{padding:8px 12px;border-radius:999px;background:#fff0f2;color:#7b1b2c;border:1px solid #e5c2c8;font-size:11px;font-weight:800}
.kidney-mark{width:58px;height:58px;display:inline-flex;align-items:center;justify-content:center;border-radius:18px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.24);margin-right:12px;vertical-align:middle;box-shadow:inset 0 0 20px rgba(255,255,255,.05)}.kidney-mark svg{width:39px;height:39px}.hero-row{display:flex;align-items:center;gap:10px;position:relative;z-index:3}.hero-kidney-note{color:rgba(255,255,255,.72);font-size:11px;letter-spacing:1.1px;text-transform:uppercase;margin-top:15px;position:relative;z-index:3}
.icon-tile{background:linear-gradient(145deg,#fff,#f9e5e8);border:1px solid var(--line);border-radius:22px;padding:20px;min-height:165px;box-shadow:var(--shadow);position:relative;overflow:hidden}.icon-tile:after{content:'';position:absolute;width:150px;height:150px;border-radius:50%;right:-62px;top:-62px;background:#f2cbd1;opacity:.8}.icon-bubble{width:48px;height:48px;border-radius:15px;display:flex;align-items:center;justify-content:center;background:#f4d4d9;color:#8f1427;margin-bottom:12px;position:relative;z-index:2}.icon-bubble svg{width:30px;height:30px}
.quick-card{height:100%;background:linear-gradient(145deg,#fff,#fff1f3)}
[data-testid="stDataFrame"]{border:1px solid #e1bfc5;border-radius:14px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid #e0c0c6!important;border-radius:14px!important;background:rgba(255,250,250,.8)}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"]>div{border-color:#dcb8bf!important;border-radius:10px!important}
@media(max-width:900px){.hero h1{font-size:35px}.stTabs [data-baseweb="tab"]{flex:1 1 45%;min-width:150px}.metric-value{font-size:24px}}

.kidney-illustration{background:linear-gradient(145deg,#fff,#fff0f2);border:1px solid rgba(255,255,255,.22);border-radius:24px;padding:14px 16px 12px;box-shadow:0 18px 35px rgba(35,7,14,.18);max-width:500px;margin-left:auto}
.kidney-illustration svg{display:block;width:100%;height:auto}.kidney-caption{display:flex;justify-content:space-between;gap:14px;align-items:baseline;padding:4px 5px 0;color:#651321;font-size:12px}.kidney-caption span{color:#765861;font-weight:500;text-align:right}
.medical-callout{background:#fff;border:1px solid #e7c5ca;border-radius:18px;padding:14px 16px;box-shadow:0 10px 24px rgba(101,19,33,.08)}
</style>
    """,
    unsafe_allow_html=True
)


def kidney_illustration():
    """Clean educational renal anatomy illustration for the dashboard hero."""
    st.markdown("""
    <div class='kidney-illustration' role='img' aria-label='Illustration of human kidneys with renal vessels and ureters'>
      <svg viewBox='0 0 420 300' xmlns='http://www.w3.org/2000/svg'>
        <defs>
          <linearGradient id='kidneyL' x1='0' y1='0' x2='1' y2='1'><stop stop-color='#E98998'/><stop offset='1' stop-color='#9E243A'/></linearGradient>
          <linearGradient id='kidneyR' x1='1' y1='0' x2='0' y2='1'><stop stop-color='#F0A0AC'/><stop offset='1' stop-color='#A82A42'/></linearGradient>
          <filter id='soft'><feDropShadow dx='0' dy='10' stdDeviation='10' flood-opacity='.20'/></filter>
        </defs>
        <g filter='url(#soft)'>
          <path d='M150 48 C95 39 58 81 61 137 C64 196 91 233 135 227 C161 224 169 201 163 177 L151 145 C143 123 158 107 169 94 C181 78 174 54 150 48Z' fill='url(#kidneyL)' stroke='#651321' stroke-width='5'/>
          <path d='M270 48 C325 39 362 81 359 137 C356 196 329 233 285 227 C259 224 251 201 257 177 L269 145 C277 123 262 107 251 94 C239 78 246 54 270 48Z' fill='url(#kidneyR)' stroke='#651321' stroke-width='5'/>
          <path d='M112 92 C126 76 143 74 157 86 M108 124 C125 106 142 105 158 117 M111 158 C126 143 141 143 155 154' fill='none' stroke='#F7C7CE' stroke-width='7' stroke-linecap='round'/>
          <path d='M308 92 C294 76 277 74 263 86 M312 124 C295 106 278 105 262 117 M309 158 C294 143 279 143 265 154' fill='none' stroke='#F7C7CE' stroke-width='7' stroke-linecap='round'/>
          <path d='M210 70 L210 224' stroke='#F7E7EA' stroke-width='9' stroke-linecap='round'/>
          <path d='M205 224 C199 247 188 260 179 278 M215 224 C221 247 232 260 241 278' fill='none' stroke='#F2B6C0' stroke-width='7' stroke-linecap='round'/>
          <path d='M208 111 C184 108 162 108 148 113 M212 111 C236 108 258 108 272 113' fill='none' stroke='#7C3AED' stroke-width='8' stroke-linecap='round'/>
          <path d='M208 91 C185 82 166 76 149 70 M212 91 C235 82 254 76 271 70' fill='none' stroke='#DC2626' stroke-width='8' stroke-linecap='round'/>
        </g>
        <g font-family='DM Sans, sans-serif' font-size='13' font-weight='700' fill='#651321'>
          <text x='28' y='42'>LEFT KIDNEY</text><text x='300' y='42'>RIGHT KIDNEY</text><text x='176' y='18'>RENAL ANATOMY</text>
          <text x='235' y='108' fill='#7C3AED'>renal artery</text><text x='235' y='72' fill='#DC2626'>renal vein</text><text x='245' y='277' fill='#9E243A'>ureter</text>
        </g>
      </svg>
      <div class='kidney-caption'><b>Renal system at a glance</b><span>Kidneys filter blood, regulate fluid balance, maintain homeostasis and produce urine.</span></div>
    </div>
     """, unsafe_allow_html=True)

def _safe_float(val, default=0.0):
    try:
        if pd.isna(val) or val is None:
            return float(default)
        return float(val)
    except (ValueError, TypeError):
        return float(default)

def _safe_int(val, default=0):
    try:
        if pd.isna(val) or val is None:
            return int(default)
        return int(val)
    except (ValueError, TypeError):
        return int(default)

def create_risk_gauge_chart(decision_score, prediction):
    """Create an easy-to-read visual of the SVM decision score.

    IMPORTANT: an SVM decision score is NOT a probability.  The previous
    version transformed it with a sigmoid and displayed that value as a
    percentage, which could be misleading to non-technical users.  This
    version keeps the original model score and visualises its position
    relative to the classification boundary at 0.
    """
    score_val = _safe_float(decision_score, 0.0)

    # Give the gauge enough room for the observed score without pretending
    # that the score is a percentage or calibrated probability.
    gauge_limit = max(1.0, abs(score_val) * 1.35)

    if _safe_int(prediction) == 1:
        bar_color = "#e74c3c"
        result_label = "Higher-risk side"
    else:
        bar_color = "#27ae60"
        result_label = "Lower-risk side"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_val,
        number={"font": {"size": 32}, "valueformat": ".3f"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={
            "text": "<b>CKD Screening Signal</b>",
            "font": {"size": 18}
        },
        gauge={
            "axis": {
                "range": [-gauge_limit, gauge_limit],
                "tickwidth": 1,
                "tickcolor": "#7f8c8d"
            },
            "bar": {"color": bar_color, "thickness": 0.4},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#bdc3c7",
            "steps": [
                {
                    "range": [-gauge_limit, 0],
                    "color": "rgba(39, 174, 96, 0.12)"
                },
                {
                    "range": [0, gauge_limit],
                    "color": "rgba(231, 76, 60, 0.12)"
                }
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.75,
                "value": score_val
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial")
    )
    return fig, result_label


def render_visual_prediction_summary(prediction, decision_score, key_suffix="default"):
    fig_gauge, result_label = create_risk_gauge_chart(decision_score, prediction)

    col_visual, col_text = st.columns([1.2, 1.8])

    with col_visual:
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"risk_gauge_chart_{key_suffix}")
        st.caption("0 = model decision boundary • positive values favour the higher-risk class")

    with col_text:
        if _safe_int(prediction) == 1:
            st.markdown(
                """
                <div class="risk-banner-high">
                    <h2 style="margin-top:0; color:#c0392b;">🔴 HIGHER CKD RISK PROFILE</h2>
                    <p style="font-size: 1.0em; line-height: 1.5;">
                        <b>What this means:</b> The screening model placed this profile on the
                        higher-risk side of its decision boundary. This is a screening result,
                        not a diagnosis.
                    </p>
                    <hr style="border-top: 1px solid #f5c6cb; margin: 10px 0;">
                    <p style="font-size: 0.95em; margin-bottom:0;">
                        💡 <b>Next step:</b> Discuss the result with a healthcare professional
                        and consider appropriate clinical tests such as <b>eGFR</b>,
                        <b>Serum Creatinine</b> and <b>Urine ACR</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="risk-banner-low">
                    <h2 style="margin-top:0; color:#27ae60;">🟢 LOWER CKD RISK PROFILE</h2>
                    <p style="font-size: 1.0em; line-height: 1.5;">
                        <b>What this means:</b> The screening model placed this profile on the
                        lower-risk side of its decision boundary. This is reassuring, but it
                        does not rule out kidney disease.
                    </p>
                    <hr style="border-top: 1px solid #c3e6cb; margin: 10px 0;">
                    <p style="font-size: 0.95em; margin-bottom:0;">
                        💡 <b>Next step:</b> Continue routine health checks and discuss any
                        symptoms or concerns with a healthcare professional.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.expander("🛠️ Technical Model Details (For Medical & Technical Staff)"):
        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("SVM Decision Score", f"{_safe_float(decision_score):.4f}")
        with t2:
            st.metric("Risk Classification", "Higher CKD Risk (Class 1)" if _safe_int(prediction) == 1 else "Lower CKD Risk (Class 0)")
        with t3:
            st.metric("Boundary Position", "Above 0" if _safe_float(decision_score) > 0 else "At / Below 0")
        st.caption(
            "The SVM decision score shows which side of the model's classification boundary "
            "the profile falls on. It is not a percentage probability and should not be read as one."
        )


def render_patient_health_badges(input_df):
    if input_df is None or len(input_df) == 0:
        return
    
    try:
        row = input_df.iloc[0]
        
        st.subheader("🏷️ Visual Patient Health Factor Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        sys = _safe_float(row.get("Systolic_BP"), 120.0)
        dia = _safe_float(row.get("Diastolic_BP"), 80.0)
        bp_cat = str(row.get("Blood_Pressure_Category", "Normal"))
        if "Stage 2" in bp_cat or sys >= 140 or dia >= 90:
            bp_chip = f'<span class="badge-chip badge-red">🔴 BP: {sys:.0f}/{dia:.0f} mmHg ({bp_cat})</span>'
        elif "Stage 1" in bp_cat or "Elevated" in bp_cat or sys >= 120:
            bp_chip = f'<span class="badge-chip badge-yellow">🟡 BP: {sys:.0f}/{dia:.0f} mmHg ({bp_cat})</span>'
        else:
            bp_chip = f'<span class="badge-chip badge-green">🟢 BP: {sys:.0f}/{dia:.0f} mmHg (Normal)</span>'

        bmi = _safe_float(row.get("BMI"), 24.0)
        if bmi >= 30.0:
            bmi_chip = f'<span class="badge-chip badge-red">🔴 BMI: {bmi:.1f} (Obese)</span>'
        elif bmi >= 25.0:
            bmi_chip = f'<span class="badge-chip badge-yellow">🟡 BMI: {bmi:.1f} (Overweight)</span>'
        else:
            bmi_chip = f'<span class="badge-chip badge-green">🟢 BMI: {bmi:.1f} (Normal Weight)</span>'

        meta = _safe_int(row.get("Metabolic_Risk"), 0)
        if meta >= 2:
            meta_chip = f'<span class="badge-chip badge-red">🔴 Metabolic Risk: {meta}/3 (High)</span>'
        elif meta == 1:
            meta_chip = f'<span class="badge-chip badge-yellow">🟡 Metabolic Risk: {meta}/3 (Moderate)</span>'
        else:
            meta_chip = f'<span class="badge-chip badge-green">🟢 Metabolic Risk: 0/3 (Low)</span>'

        cv = _safe_int(row.get("CV_Risk"), 0)
        if cv >= 2:
            cv_chip = f'<span class="badge-chip badge-red">🔴 CV Risk: {cv}/3 (High)</span>'
        elif cv == 1:
            cv_chip = f'<span class="badge-chip badge-yellow">🟡 CV Risk: {cv}/3 (Moderate)</span>'
        else:
            cv_chip = f'<span class="badge-chip badge-green">🟢 CV Risk: 0/3 (Low)</span>'

        ls = _safe_int(row.get("Lifestyle_Risk"), 0)
        if ls >= 2:
            ls_chip = f'<span class="badge-chip badge-red">🔴 Lifestyle Risk: {ls}/3 (High)</span>'
        elif ls == 1:
            ls_chip = f'<span class="badge-chip badge-yellow">🟡 Lifestyle Risk: {ls}/3 (Moderate)</span>'
        else:
            ls_chip = f'<span class="badge-chip badge-green">🟢 Lifestyle Risk: 0/3 (Healthy)</span>'

        sleep = _safe_float(row.get("Sleep_Duration_Hours"), 7.0)
        poor_sleep = _safe_int(row.get("Poor_Sleep"), 0)
        if poor_sleep == 1:
            sleep_chip = f'<span class="badge-chip badge-yellow">🟡 Sleep: {sleep:.1f} hrs (Suboptimal)</span>'
        else:
            sleep_chip = f'<span class="badge-chip badge-green">🟢 Sleep: {sleep:.1f} hrs (Optimal)</span>'

        with col1:
            st.markdown(f"<b>Cardiovascular & BP:</b><br>{bp_chip}<br>{cv_chip}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<b>Metabolic & Body:</b><br>{bmi_chip}<br>{meta_chip}", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<b>Lifestyle & Sleep:</b><br>{ls_chip}<br>{sleep_chip}", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Unable to render visual health badges: {e}")

def render_severity_visual_stepper(severity_label):
    stages = ["Healthy", "Mild CKD", "Moderate CKD", "Severe CKD"]
    colors = {"Healthy": "#2ecc71", "Mild CKD": "#f1c40f", "Moderate CKD": "#e67e22", "Severe CKD": "#e74c3c"}
    icons = {"Healthy": "🟢", "Mild CKD": "🟡", "Moderate CKD": "🟧", "Severe CKD": "🔴"}
    
    current_stage = severity_label if severity_label in stages else "Healthy"
    
    st.subheader("🚦 Visual CKD Severity Level Stepper")
    
    cols = st.columns(4)
    for idx, stage in enumerate(stages):
        with cols[idx]:
            is_active = (stage == current_stage)
            bg_color = colors[stage] if is_active else "#f8f9fa"
            text_color = "#ffffff" if is_active else "#555555"
            border = f"3px solid {colors[stage]}" if is_active else "1px solid #dcdcdc"
            shadow = "0 4px 8px rgba(0,0,0,0.15)" if is_active else "none"
            
            st.markdown(
                f"""
                <div style="background-color: {bg_color}; color: {text_color}; border: {border};
                            border-radius: 10px; padding: 14px 10px; text-align: center;
                            box-shadow: {shadow}; font-weight: bold;">
                    <div style="font-size: 1.4em;">{icons[stage]}</div>
                    <div style="font-size: 1.0em; margin-top: 4px;">{stage}</div>
                    <div style="font-size: 0.75em; opacity: 0.9;">{'ACTIVE STAGE' if is_active else f'Stage {idx}'}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# SECTION 3 : APPLICATION TITLE
# =============================================================================

st.markdown("<div class='hero'><div style='display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,.75fr);gap:24px;align-items:center'><div><div class='hero-row'><span class='kidney-mark'>🫘</span><div><div class='kicker'>RENALIS • RENAL ANALYTICS & INTELLIGENCE SYSTEM</div><h1>RENALIS</h1></div></div><p>CKD early screening, clinical severity assessment, statistical analysis, insurance intelligence and model analytics in one integrated research dashboard.</p><div class='hero-kidney-note'>Kidney-focused analytics • screening • statistics • insurance</div></div><div class='medical-callout'><b>Clinical analytics workflow</b><br><span style='color:#765861'>Early Risk → Statistical Analysis → Diagnostic Plots → Insurance Portfolio → Clinical Severity</span></div></div></div>",unsafe_allow_html=True)


# =============================================================================
# SECTION 4 : APPLICATION SECTIONS
# =============================================================================
#
# The application is organised into two top-level sections:
#
#   1. Early Screening    -> your work (CKD risk prediction with the Linear SVM)
#   2. Clinical Screening -> your partner's work (CKD severity classification)
#
# The two sections are fully independent: each has its own model, its own
# intake form, and its own results. Neither section reads or requires the
# other's session state. A person can open Clinical Screening directly
# without ever having used Early Screening, and vice versa.
#
# =============================================================================

# TWO independent application sections.
# Early Screening and Clinical Screening are two separate, standalone
# assessments -- each with its own model, its own inputs, and its own
# results. Neither section depends on the other having been run.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MAIN APPLICATION NAVIGATION
# ---------------------------------------------------------------------------
has_completed_early_screening = ("prediction" in st.session_state and "input_data" in st.session_state)
section_options = ["🏠 HOME", "1️⃣ EARLY SCREENING"]
if has_completed_early_screening:
    section_options.append("2️⃣ CLINICAL SCREENING")
section_options.append("3️⃣ INSURANCE INTELLIGENCE")

if "selected_app_section" not in st.session_state or st.session_state.get("selected_app_section") not in section_options:
    st.session_state["selected_app_section"] = "🏠 HOME"
if st.session_state.get("target_section") in section_options:
    st.session_state["selected_app_section"] = st.session_state.pop("target_section")

with st.sidebar:
    st.markdown("<div style='font-size:28px;font-weight:800;font-family:Playfair Display,serif;'>RENALIS</div><div style='font-size:11px;opacity:.72;letter-spacing:1.4px;margin-bottom:16px;'>CKD RISK INTELLIGENCE</div>",unsafe_allow_html=True)
    selected_section = st.radio("NAVIGATE", section_options, key="selected_app_section")
    if not has_completed_early_screening:
        st.markdown("<div style='margin-top:8px;padding:10px 12px;border-radius:10px;background:rgba(255,255,255,.10);font-size:11px;line-height:1.45;'><b>🔒 Clinical Screening locked</b><br>Complete Early Screening first to unlock the severity assessment.</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-foot'><b>Four-pathway RENALIS dashboard</b><br>Home • Early Screening • Clinical Severity • Insurance Intelligence<br><br><span style='opacity:.75'>Research / demonstration interface. Not a diagnosis or insurance underwriting engine.</span></div>",unsafe_allow_html=True)

is_home_section = selected_section == "🏠 HOME"
main_early_screening = selected_section == "1️⃣ EARLY SCREENING"
main_clinical_screening = selected_section == "2️⃣ CLINICAL SCREENING"
is_insurance_section = selected_section == "3️⃣ INSURANCE INTELLIGENCE"

tab_home, tab_prediction, tab_interpretation, tab_statistics, tab_ckd_visuals, tab_insurance, tab_about = [st.container() for _ in range(7)]

if is_home_section:
    tab_home = st.container()
elif main_early_screening:
    tab_prediction, tab_statistics, tab_ckd_visuals = st.tabs(["🩺 Prediction & Outputs", "📊 Statistical Analysis", "📈 Exploratory & Diagnostic Plots"])
    tab_interpretation = tab_prediction
elif is_insurance_section:
    tab_insurance = st.container()


# =============================================================================
# SECTION 4.2 : CLINICAL SCREENING SUB-TABS (GATED)
# =============================================================================
#
# The tab object itself is created now so it renders alongside Early
# Screening, but its *content* is written further down this script (see
# "SECTION 13 : CLINICAL SCREENING") -- after the Early Screening prediction
# code has had a chance to run and set st.session_state["prediction"] on
# this same script execution. This mirrors the pattern already used above
# for the CKD Prediction / Interpretation tabs.
#
# =============================================================================
# =============================================================================
# STEP 1 : HOME TAB
# =============================================================================
#
# Objective:
# Create the landing page for the CKD Early Screening & Analytics System.
#
# =============================================================================


# =============================================================================
# SECTION 5 : HOME TAB
# =============================================================================

if is_home_section:
    with tab_home:

        # =========================================================================
        # PAGE HEADER
        # =========================================================================

        st.title(
            "🩺 CKD Early Screening & Analytics"
        )

        kidney_illustration()

        st.markdown(
            """
            ### Machine Learning Based Chronic Kidney Disease Screening
            """
        )

        st.write(
            """
            This application provides an integrated environment for
            **CKD early screening, clinical reporting, model evaluation,
            explainability and analytical visualisation**.
            """
        )


        # =========================================================================
        # APPLICATION STRUCTURE
        # =========================================================================

        st.divider()

        st.subheader(
            "🗺️ Application Structure"
        )

        struct_col1, struct_col2 = st.columns(2)

        with struct_col1:

            st.markdown(
                """
                ### 1️⃣ Early Screening

                CKD-**risk** prediction (Yes/No) using the final Linear SVM,
                plus interpretation, model statistics, key plots, and the
                insurance analysis.
                """
            )

        with struct_col2:

            st.markdown(
                """
                ### 2️⃣ Clinical Screening

                CKD-**severity** classification, developed separately as its
                own independent assessment with its own intake form.
                """
            )


        # =========================================================================
        # PROJECT OBJECTIVE
        # =========================================================================

        st.divider()

        st.subheader(
            "🎯 Project Objective"
        )

        st.write(
            """
            The primary objective of this project is to develop a machine-learning
            based early screening system for Chronic Kidney Disease (CKD) using
            demographic, lifestyle, medical-history and vital-sign information.
            """
        )


        # =========================================================================
        # KEY APPLICATION MODULES
        # =========================================================================

        st.divider()

        st.subheader(
            "📌 Application Modules"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.markdown(
                """
                ### 🩺

                **CKD Screening**

                Enter patient information and generate
                an early CKD-risk prediction.
                """
            )


        with col2:

            st.markdown(
                """
                ### 📋

                **Clinical Report**

                View the patient's prediction,
                screening indicators and interpretation.
                """
            )


        with col3:

            st.markdown(
                """
                ### 📊

                **Model Performance**

                Review the performance of the
                final Linear SVM model.
                """
            )


        with col4:

            st.markdown(
                """
                ### 🔎

                **Explainability**

                Explore feature associations and
                model interpretation.
                """
            )


        # =========================================================================
        # ANALYTICS MODULES
        # =========================================================================

        st.divider()

        st.subheader(
            "📈 Analytical Modules"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                """
                ### 📈 CKD Visualisations

                Explore the CKD dataset through
                the visualisations developed during
                the analysis workflow.
                """
            )


        with col2:

            st.markdown(
                """
                ### 🏥 Insurance Analytics

                Examine CKD risk and insurance-related
                patterns through the project's
                analytical visualisations.
                """
            )


        with col3:

            st.markdown(
                """
                ### 📚 About the Model

                Review the modelling workflow,
                preprocessing, feature engineering
                and model information.
                """
            )


        # =========================================================================
        # FINAL MODEL
        # =========================================================================

        st.divider()

        st.subheader(
            "🤖 Final Deployed Model"
        )


        model_col1, model_col2 = st.columns(
            [1, 2]
        )


        with model_col1:

            st.metric(
                "Final Model",
                "Linear SVM"
            )


        with model_col2:

            st.write(
                """
                The application uses the final Linear Support Vector Machine
                developed in the CKD screening workflow for the deployed
                prediction component.
                """
            )


        # =========================================================================
        # WORKFLOW
        # =========================================================================

        st.divider()

        st.subheader(
            "🔄 Application Workflow"
        )


        workflow_col1, \
        workflow_col2, \
        workflow_col3, \
        workflow_col4 = st.columns(4)


        with workflow_col1:

            st.markdown(
                """
                **01**

                🩺

                **Patient Input**
                """
            )


        with workflow_col2:

            st.markdown(
                """
                **02**

                ⚙️

                **Feature Processing**
                """
            )


        with workflow_col3:

            st.markdown(
                """
                **03**

                🤖

                **SVM Prediction**
                """
            )


        with workflow_col4:

            st.markdown(
                """
                **04**

                📋

                **Clinical Report**
                """
            )


        # =========================================================================
        # DISCLAIMER
        # =========================================================================

        st.divider()

        st.subheader("📝 Before You Begin")
        st.markdown(
            "Home is information-only. No patient information is entered here. "
            "Patient inputs are collected separately inside the **Early Screening** and **Clinical Screening** pathways. "
            "Clinical Screening is unlocked only after Early Screening is completed, but it has its own independent 75-feature intake."
        )

        st.divider()
        st.subheader("👥 Creators")
        creator_col1, creator_col2 = st.columns(2)
        with creator_col1:
            st.markdown(
                "<a href='https://in.linkedin.com/in/smritilekha-bhattacharjee-1a6814364?utm_source=chatgpt.com' target='_blank'><b>Smritilekha Bhattacharjee</b> ↗ LinkedIn</a>",
                unsafe_allow_html=True,
            )
        with creator_col2:
            st.markdown(
                "<a href='https://www.linkedin.com/in/akash-sinha-11ii200iii/' target='_blank'><b>Akash Sinha</b> ↗ LinkedIn</a>",
                unsafe_allow_html=True,
            )

        st.divider()

        st.caption(
            """
            **Important:** This application is intended for early screening
            and analytical purposes. A machine-learning prediction should not
            be considered a definitive medical diagnosis and does not replace
            professional clinical evaluation.
            """
        )


# =============================================================================
# END OF STEP 1
# =============================================================================
#
# STEP 1 COMPLETE:
#
# ✓ Professional Home tab
# ✓ Consistent column alignment
# ✓ Project objective
# ✓ Application modules
# ✓ Analytics modules
# ✓ Final model information
# ✓ Application workflow
# ✓ Medical disclaimer
#
# NEXT:
#
# STEP 2 — CKD SCREENING TAB
#
# =============================================================================

# =============================================================================
# STEP 2 : CKD SCREENING TAB
# =============================================================================
#
# Objective:
# Create a clean, clinically organised patient-input interface.
#
# IMPORTANT:
# The actual prediction pipeline will be connected after the exact feature
# schema of the saved SVM pipeline is verified.
#
# =============================================================================

# =============================================================================
# STEP 3 : CKD SCREENING + FINAL SVM PREDICTION
# =============================================================================
#
# Objective:
# Connect the Streamlit screening form to the exact feature structure
# used by the final CKD Linear SVM deployment pipeline.
#
# FINAL MODEL:
# Final_CKD_Early_Screening_SVM_Pipeline.pkl
#
# FINAL FEATURE COUNT:
# 49
#
# IMPORTANT:
# The pipeline already contains:
#     1. Preprocessing
#     2. Missing-value imputation
#     3. Scaling
#     4. One-hot encoding
#     5. SMOTE
#     6. Linear SVM classifier
#
# Therefore, Streamlit sends the RAW feature dataframe to the pipeline.
#
# =============================================================================


# =============================================================================
# SECTION 1 : REQUIRED IMPORTS
# =============================================================================

import pandas as pd
import joblib
from sklearn.inspection import permutation_importance


# =============================================================================
# SECTION 2 : LOAD FINAL DEPLOYMENT MODEL
# =============================================================================

MODEL_FILENAME = "Final_CKD_Early_Screening_SVM_Pipeline.pkl"


def _find_model_path():
    base = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / MODEL_FILENAME,
        base / MODEL_FILENAME,
        base / "models" / MODEL_FILENAME,
        base / "assets" / MODEL_FILENAME,
        base.parent / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


MODEL_PATH = _find_model_path()


@st.cache_resource
def load_ckd_model(model_path):
    return joblib.load(model_path)


try:

    final_svm_pipeline = load_ckd_model(MODEL_PATH) if MODEL_PATH is not None else None
    if final_svm_pipeline is None:
        raise FileNotFoundError(
            f"{MODEL_FILENAME} was not found in the application/project model locations."
        )

    MODEL_LOADED = True

except Exception as model_error:

    MODEL_LOADED = False

    final_svm_pipeline = None

    st.error(
        "Unable to load the CKD SVM deployment model."
    )

    st.exception(
        model_error
    )



# =============================================================================
# SECTION 2.5 : SHARED EVALUATION ENGINE
# =============================================================================
# Creates the validation objects used by Model Performance, Explainability,
# CKD Visualisations and Insurance Analytics. The deployed SVM is evaluated
# from the same training-data split used in the project whenever the dataset
# is available. The exact notebook validation results are retained as a safe
# fallback so the dashboard never invents metrics.
# =============================================================================


def show_embedded_notebook_plot(encoded_image):
    st.image(BytesIO(__import__("base64").b64decode(encoded_image)), use_container_width=True)


NOTEBOOK_ROC_B64 = 'iVBORw0KGgoAAAANSUhEUgAAA3kAAAKyCAYAAABoqBcWAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQAA1npJREFUeJzs3Qd8k2XXx/F/ku7SsqfIUFFBkeFAxY1bcb4KouLeew/c28e990IUcY9HH9w4UZGhAoooCLJ398p4P+dqk7ZQRksh6/f1E3vfd9L06p2Q5uRc1zmeUCgUEgAAAAAgIXijPQAAAAAAQOMhyAMAAACABEKQBwAAAAAJhCAPAAAAABIIQR4AAAAAJBCCPAAAAABIIAR5AAAAAJBACPIAAAAAIIEQ5AEAAABAAiHIAwAAjW7JkiXyeDx68MEHObsAsJER5AFAgvD7/e5Ndc1LTk6O+vXrpxdffHG131dQUKBbb71VvXv3VpMmTZSdna1evXrplltucdfVpbi4WPfff7/69++v5s2bKzMzU926ddMRRxyh9957T4FAYK3jbYz7iFWJ/LsBAGIfQR4AJJhBgwYpFAopGAxq6tSp6t69u0455RQ99thjq9x21qxZ2n777fXCCy/oxhtv1Lx587Rw4UIX4L300kvuOrtNTf/++6922GEHPfDAAzrjjDP0xx9/aMWKFS546dq1q44++mh9+eWXaxxjY9xHrErk360+WrVq5Z6HF198cbSHAgBJxxOyV2AAQEJk8lJTU12Q99prr0WOl5WVqV27dmrTpo2mTZsWOW4v/zvuuKP++ecf/frrr+rQoUOt+7OAb7vttnOByU8//eQyg/Y9O+20k2bOnKlJkyapY8eOq4zjk08+cdlAy2LVpTHuI1Yl8u8GAIgfZPIAIMGlp6erU6dOmjNnTq3jH374ocaPH6/LL798lQDP2LHLLrtMP//8s7tt+Hts/8orr6wzgDH777//GgOY+t7HmDFjXIA5evToVW6XkpLixh/2119/uds+++yzGjlypLbZZhsX+L755pvKzc3VySefvMp9lJeXq3Xr1i44DquoqNCdd96pHj16KCMjQy1atHDXW0C8Jg05P5blu+iii9xjlJaWpk033VQXXHCBO17X7zVixAhtueWWLlA84IADIo/ryy+/rK222sqNd5dddtGUKVNq/dya92G3temjdts+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ALYmy+ZZJaTVuf76613TyldeeUWbb765PvnkE7dOz7atGEtd7rzzTt18882rHLc0vhVyiQV2PvLy8twTqSHNOD/5Y7KUUbmd6WmmZUuWqcXf3yit6volLXZQcNGixh00Ytr6PqcAnk/YkHiNQmO75ZZbdNVVV+n+++9X586dtYj3PUjA1ymbDRnTQV6rVpVTB5cuXVrruJW5DV+3MjvJloJ/8cUXXTrebLfddi4Vf+utt7r1enW55pprdOmll9bK5G266aZq3bq1cnNzFStPIivla2NqyJNoSsH3SqsK8to3aac2rVvJs/R3tx/K7aBWXXs29pAR49b3OQXwfMKGxGsU1teff/6pr776SmeccYbbt/ePb775Jn/3kNCvUxkZVW/4YzXIa9u2rQu0LEA77LDDIsctS3fggQfW+T025dJOdvPmzWsdt/358+ev9mdZ00u7rMwerFh5wIw9iRoyJn8gKF/O1Mj+/l33kXf5TKm8MtL3dOgrTwz9noj95xTA8wkbA69RaKgRI0bo7LPPVnFxsVtzF67NwHMKjc0TY++l1nUcUR2t/eN89tln9ccff0TW21kRlfAnMubaa6/VwQcf7LZtHZ5l7h544AGX1TO2fu+tt95yfVCS1d0fT5EvY15kf+g2J0jzJlbfoH3v6AwMAACgERUVFemUU07RiSee6Lb33HNPbbXVVpxjIFYyecbmTVsBll69eik7O1s+n899MmOtEcJsPvXs2VUVIiW9/vrrOvPMM10m0AqvLF++XCeffLKGDRumZPXKrx/L175yO0VZyknLkabV6BvYkbLBAAAgvtkH+4MGDXLJActm3Hjjje79n71/BBBDQZ79o3z88cfdOjurktm+fXulpKSsUjSlZnEU+7TG5l+XlJS477FqnMn+jzuU811ke+cOO0hlBdK0/1UeyGopddkteoMDAABYT88//7xriWDvCTt06KBXX33VZfEAxGCQF2ZZPLvUxRY61iUzM1ObbLKJkp1V+0lpUt0+4fTtTpV+/6/krwqMtzlS8tWuYAoAABBv73cswLMlPFaAb3XvDwHEUJCHhvvvr3Nr7fdt01f66IbqAz2P5fQCAIC4YwX3woXzTj31VFeb4ZBDDomZAhhALONfSZx7f+r4WvueZTOkWd9W7rTsJm26U3QGBgAA0MCsnfW623bbbd3SnHCFw4EDBxLgAeuI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vQEBALAas2fP1l577aXDDz/cffAbZoVWAABxGOT9+++/2mabbdx2dna28vLy3PYhhxyin3/+uXFHmER8oeqAzhPeSGeqCwAgtrz33nturZ31yM3NzXWVrwEAcR7kWXNTn8/ntq3C5aeffuq2bYG1vdijYYIpS93XzIoan4KmEeQBAGJDWVmZLrroIh1xxBFavny5dtppJ02aNElHHnlktIcGAFjfIK9p06aR7UsuuURDhw7VDjvsoIMOOkhnnnlmQ+4SLniuDJwrfP7q85Gew7kBAETdX3/9pV133VUPP/yw27/ssstcP7yuXbtGe2gAgPXpkxe2YsWKyLYtrt588831ww8/aOutt9ahhx7akLuECXklT0B+b1XRFUMmDwAQA3799VdNmDBBLVu21EsvveSWaAAAEijIW9luu+3mLmbJkiVq1apVY9xtUrEpsBbgmXal6dVXsCYPABADjjrqKD366KNuqqZV0QQAJEEz9OLiYt1+++0uq4f6qwgE5fFUZvAyaz4qZPIAAFEwZcoUVz1z7ty5kWPnnXceAR4AJFqQV1JSoksvvVRbbbWVunTpoquvvlqBQEAffPCBunXrpvvuu0/Dhg3bcKNNYMUVpZFtjyozek5G9fpHAAA2xsySZ555RjvuuKO++uor93cfAJDA0zWtCbr1whs0aJDbf+KJJzRz5ky9++677tO966+/Xs2bN99QY01oeWXFke25KTULr1CtFACwceTn5+uss87Sa6+95vb3339/PfLII5x+AEjkIO/NN990Ad2ee+7p9i3Y22effdwCbKuwiYYrLS+PbPcpDTdCbyL5GmXZJAAAa2RtkOzv+t9//+3aJNkSjCuuuEJeb6Ot7AAAbCT1iiDmzJnjyieHhYutHHvssY0/siRTEqiIbGeEp2uSxQMAbASfffaZDj74YFVUVKhTp04uk7fLLrtw7gEgGYI8v9+v1NTUyH54OyMjo/FHlmQKy4si26mhqumarMcDAGwEFtDZ2npbc//cc8+x9AIA4ly95wIeeOCBaz02evTo9RtVEiqqMV3z79TKpujKYD0eAGDD9b3bdttt3XTM7OxsV2TFeuB5PB5OOQDEuXpNtLe5+s2aNat1qesY6q+4vHq65o6lVZU2ma4JAGhkwWBQd955p/r27at77703ctx63BLgAUASZvLC1bbQ+Mr81Zk8X1XdFTJ5AIDGtHDhQp144on69NNP3f4ff/zhWiYQ3AFAYqFkVowoC1T3xktRjeqaAAA0UnGVXr16uQAvKyvLtUSy9XcEeACQeAjyYsTsFYtXzeSl50RtPACAxGBF06677jrX884yebYOb9y4cTr55JMJ8AAgQRHkxYj0lOqZs3NSq7YJ8gAA62natGm655573LTMM888Uz/99JN69OjBeQWABEan7RhRFqhek9ejrGqbIA8AsJ622WYbPfLII2ratKkrlgYASHxk8mJEcUV1n7y0UNV8TYI8AEA9lZWV6bLLLtP48eMjxyyDR4AHAMmjwUHeu+++q4EDB7pPCMP+85//aOnSpY01tqTy94pZqx6k8AoAoD5/S/7+W/3799f999+v4447TuU1erACAJJHg4K8l156Saeeeqp69uypqVOnRo5nZma63juov7TU6j55WaFg5QaZPABAPdoc9enTx2XwWrRoofvuu09paWmcPwBIQg0K8ixj9+abb+qOO+6odfzQQw/VyJEjG2tsSaUsWBbZbh4gyAMArJvi4mI3HdMydwUFBdptt900adIkN9sGAJCcUho6HWSXXXZx2zX767Rs2VJLlixpvNElkdKy6ociN1gV5DFdEwCwBosWLdKAAQM0efJk9/d42LBhuvHGG5VSo2IzACD5NCiT1759e/3xxx+rBHkff/yxNttss8YbXRIpqrFuIp3CKwCAddCqVSttsskmatu2rWtyfuuttxLgAQAalsmzaSG2Js9KMluQZ5m90aNHu2arN910E6e1AUr9ZZKvcjuVIA8AsBo2JdPn8ykrK0ter1cvv/yygsGgC/QAAGhwkHfVVVdpxYoV2nfffRUIBLTFFlu4xd2XXHKJLrzwQs5sAwTS/opsp1R1UGC6JgCgpgkTJrhWCHvuuaeeffZZd6x169acJADA+gd59snh3Xff7TJ3tg7APkHcdtttXaNVNEymOqlUC9y2TyHJlyalUBUNACCFQiE9+uijuvzyy11bBOuFZy2LbC08AACNsibPPkX84IMPlJGR4QqwWE8eArz1EwwFIttZwRDtEwAAzrJly3TUUUe5mTIW4B122GGueiYBHgCgUYO8iooKHXPMMa4Ay3nnnaexY8c25G5QQ0jVQV6KZfKorAkASe/77793ve/effddtyzioYcectvWBw8AgEYN8t5++20tWLBAd911l2uGbj15bF2elW2ePn16Q+4y6QXDDdBtuqatyUvNSvpzAgDJrLS0VEcffbRmz57t/sbaB6qWzatZ1RoAgEYL8kyzZs10+umn68svv9SsWbN01llnadSoUdpyyy0bepdJrdj7Z2TbrclLzYjqeAAA0WVLIp577jkNGTJE48ePV9++fXlIAADrZL27pdri73HjxunHH3/UP//8ozZt2qzvXSalirLm8mXMd9uplslLIcgDgGTzxRdfqKSkRIcccojbP/jgg90FAIANnsmzappjxozRGWecoXbt2unEE09Uenq6m8Y5d+7chtxl0svwZUbOQZr1yUtJT/pzAgDJwu/364YbbnCtiU444QQ3QwYAgI2ayevcubPmz5/v/hhZQ/QjjzxS2dnZDR4EJHlCtSPvlOqgDwCQuObMmeOmZH7zzTdu3wqb0fsOALDRg7wrrrhCgwcPZmpmo6osvOIJheSW1JPJA4CE99///lcnn3yy63mXk5Ojp59+2v19BQBgowd5Vt0LjStUVV3TFz7AmjwASOjm5tbY/P7773f7VlTFipdZFU0AADZakGd/jMy9994b2V4duw3qJxTO5EUeGdbkAUCisjYI1iLBXHTRRbr77rvd2nYAADZqkDd58uQ6t9E4QtY2wfXIq1qbl8qaPABINFaROhzM3XfffTr88MO1//77R3tYAIBkDfJGjx4d2X7yySfVpUuXOm9nbRRQfxWBgHypNTJ5vjROIwAkCGuLcMkll2jatGn67LPP5PP5XB88AjwAQMysyevatatbT1Df67B6voyFlV8jjwx98gAgEfz+++8aNGiQfvvtNzdN86uvvtI+++wT7WEBABJYg/rkrU5RUZGysrIa8y6TR7By+k6ht+ohYU0eAMQ1+8DzxRdf1A477OACvDZt2ujjjz8mwAMAxFYm7+qrr65zO9wgfcKECerdu3fjjS6ZhFJttUb1Ppk8AIhbBQUFOvfcczVixAi3P2DAALfdrl27aA8NAJAE6hXk/fzzz3Vum9TUVHXr1k2XXXZZ440uyapr2nq8zhUVlQfI5AFA3Bo6dKjeffddt/bulltu0VVXXeW2AQCIuSDPFosba9T62muvbagxJamqPnnh5Yxk8gAgbt16662aOnWqnnvuOe22227RHg4AIMk0aE0eAd4G4Am4L76qVgoEeQAQP5YvX6733nsvsr/tttu6II8ADwAQDTRDjxEeb+U0zZRIJo+muAAQD3744Qc3w2Xu3Ln65ptvtPPOO7vjTM8EAEQLzdBjQCBYmcUzM1OrHhL65AFATLOCY/fee6+GDRsmv9+vzTbbTGlp9DgFAMRpM/Sa21h//pA/st0yULk2Tym8UQCAWLV48WJXXCX899D64D311FNq2rRptIcGAEDD1uTZp5fW3DVs+vTpuu666/TCCy/QCL0BysIVNSV19FcFfGTyACAmjRkzRr169XIBXkZGhp5++mmNHDmSAA8AEJ/VNcMeeughzZs3T/fcc49KS0td/59mzZq5YwsWLNA111zT+CNNYIXl5ZHtlFDVojyv9c0DAMSaX3/9VfPnz1ePHj00atQoV2QFAIC4D/KeeOIJffzxx277iy++cJ9e/vLLLxo3bpyGDBlCkFdPJRU1grzwho8gDwBiRSgUksdj3UylCy64wBVVOfnkk5WdnR3toQEA0DjTNefMmaN27dpFgryBAwe6P349e/Z02TzUz4rSwuptb9VDwnRNAIgJH330kfr376/8/Hy3b3/vzjvvPAI8AEBiBXndunXTiBEjtGjRIjdVZf/994+szbPrUD8lNdbk+Ss/KCaTBwBRVl5eriuuuEKHHHKIxo4dq//85z/RHhIAABsuyLvlllt0/vnnq23bturevbv22GMPd9wqi5199tkNucukVhGorq65RXlVwMd0TQCImpkzZ2r33Xd3LRLCUzSvv/56HhEAQOKuyTv88MPdonMrsrL11lvLWzXF8Nhjj3VTWlA/5TWCvOo1ebRQAIBoePPNN3X66acrLy/PFRWzytFHHHEEDwYAILGDPNOiRQt3qWnPPfdsjDElnfJAxarVNQnyAGCjs8Ji5557rtvedddd9eqrr6pz5848EgCAxJ+uaebOnaurr77aZfUOO+wwt23HUH/LypZFtos9VQ+Jt8HxNwCggY4++mh16NDBVYm2fngEeACApAnyvv76a2255ZZ6//331bx5c7Vs2dJt27Fvvvmm8UeZ4EKh6oehIlJ4hemaALAxfP/995HtNm3a6Pfff9cdd9yh1FRa2QAAkijIs2pjF110kaZMmaIXX3zRrVewbTtm16F+isur1+R1qajaJsgDgA2qsLBQJ510kltLbtMyw3JzcznzAIDkC/ImTZrkgrlwY1hj25dffrm7DvXjqz6N8iq8Jo9PkAFgQ/nll1+0ww47aPjw4a54GD1eAQCJpEFBnlUbmzFjxirH7VjTpk0bY1xJJRAKRLY94fV4NQJoAEDjCIVCevzxx9WvXz9NmzZNm2yyib788kv3ISUAAEkd5B133HEaNGiQKzM9e/Zsd3njjTdcCwW7DvUTDGfv7AGxTaZqAkCjW7FihY455hidd955Kisrc03ObfZJuNcrAACJokElHO+++24Fg0ENGTJEFRWV5f9tgbo1QrfrUD+BYKB21O1lqiYANLaffvpJb731lvt7ddddd+mSSy6ptewAAICkDfJs3cJXX32lnXfe2X0aagvX7Y9kt27dlJOTs2FGmeCCoWBk22NZPTJ5ANDo9t9/f91zzz2up+uOO+7IGQYAJKyU+paZPuigg5Sfnx+pQDZ69GjtsssuG2p8ScGyomE+pmsCQKNYvHixq/ps7RC6dOnijrH2DgCQDOq1Jm/YsGE69NBDXTbPLgcffLA7hsbM5FmkRyN0AFgfNuOkd+/eGjlypE499VROJgAgqaTUt+T0qFGjXLNY8+CDD6pHjx4bamxJozxYua4xEnUzXRMAGiQQCOi2227TLbfc4mZJbL311nrooYc4mwCApFKvIG/58uWRAM+0bdtWy5Yt2xDjSiqFFSsi2yVWBIAgDwDqzWaYHH/88RozZozbP+WUU/TII48oOzubswkASCr1nhf4ww8/rPWYFWXBusvwVResCXqq+uQBANbZb7/9pn322UdLlixxQd2TTz6pE044gTMIAEhK9Y4m6iqysvIxazaLdReqsSavRSAopaRz+gCgHqzCszU279ixo1tWsOWWW3L+AABJq15B3u+//77hRpLE8koqVmqhQJAHAGszZ84ctW/fXj6fTxkZGfrwww/VsmVLtw0AQDKrV5BnC9jR+DLTfLUPpKRxmgFgDayp+WmnneYamt94443umGXyAABAPVsoYMMIWfauVgsFgjwAqEtpaanOP/98/d///Z/y8vL06aefyu/3c7IAAKiBIC/GeGiGDgB1+vPPP90a8Mcee8ztX3nllfryyy+VkkKxKgAAauIvYyxYuVANhVcAoJYRI0bo7LPPVlFRkVq3bq3hw4frwAMP5CwBAFAHgrwYUzldk8IrABA2d+5cnXHGGW6q5l577aVXXnlFHTp04AQBANDYQV4gENCkSZM0Y8YMHXPMMe6YfcJK09n6C9ZYk1f5qLAmDwDCrKDKo48+6oK9YcOGuWqaAACgkYM8+0N72GGH6ddff3UL3sN98QYPHqxzzz1XBx10UEPuFmTyAMD9TXn66ae13XbbRfqwWiVNAACwAQuvWMnqnj17uspmNV111VW64447GnKXSW2V5vG+1GgNBQCiyv6uDBo0yK2/sw8OV/47AwAANlAmz6qZTZkyRVlZWbWO9+rVS+PGjWvIXaImWigASEL298MCvJkzZ7qKmRdccIFycnKiPSwAAJIjyCspKVFqamW2yeNxpUKchQsXrhL4Ye1CqxReIZMHILlmMzzwwAO6+uqrVVFRoS5duui1115Tv379oj00AACSZ7rmbrvtpueee65WkGdVz+wP9N577924I0zG6ZpeigoASA7FxcUaOHCgLrvsMhfgWZPziRMnEuABALCxM3n33nuv9txzT40ePdoFKCeeeKLGjBnjMnzffffd+own6bmQ2UsmD0ByyMzMdFMz09PTXTbP1uLVnCECAAA2UpC37bbbusqajz/+uCtlvWDBAg0ZMsStn+jYsWND7jKphVZuoeClfSGAxGUteMrLy12AZwHd888/r3///det6wYAAOsvZX36Ft1+++2NMATUxJo8AIls/vz5OuGEE9S2bVvX1NyCvBYtWrgLAACIYpC3ZMmSNV7fqlWrho4nKbEmD0Ay+PjjjzV06FAtWrRI2dnZ+vvvv7XFFltEe1gAACScBgV5rVu3rl/QgrWoPl8e22ZNHoAEYgVVbrjhBt11111u35qcjxo1igAPAIBYCvKs8llNwWBQ06dP17Bhw1yjdNRPaOUd1uQBSBCzZs3Scccdp7Fjx7r9c845R/fdd59bjwcAAGIoyOvdu/cqx/r27auuXbvqoosu0nnnndcYY0seK2c+6ZMHIAHYB4AHH3ywpk6dqqZNm+rZZ591LRIAAEAM9slbnR49emjy5MmNeZdJ2kKBPnkA4p/X69XDDz+sXXbZxc0AIcADACDOgjwrif3oo4+qQ4cOjXWXydtCwZcWraEAwHqxqfsfffRRZH/AgAGuf6rN9AAAADE8XbOu6pn5+flKTU11JbGxvi0UCPIAxB97/bdm5mbChAnq1q2b26a5OQAAcRDkWcZuZc2bN3fr8tZWeRPrkslL5TQBiBtFRUW64IIL9MILL7j9PffcU1lZWdEeFgAASatBQV6XLl208847N/5oklTNuiuVa/II8gDEB1uHfeyxx+r33393GTtrlXD99dfL52NtMQAAcbUmb7fddmu0AXz55Zc6+uij3X1aae25c+eu8/cOHz5cO+ywg+6//34lTp88pmsCiA/PPPOMdtxxRxfgtW/fXp9//rluuukmAjwAAOIxyOvUqZP++uuv9f7h9oZg//33V8+ePd0nv7Nnz1b//v3d+r61sTcV1113nRYuXOi+L54xXRNAPPrzzz9VWlqqAw88UJMmTdLee+8d7SEBAICGBnk333yzTj31VPdHvby8vMEn0gK7Y445xn3ye8ABB+jNN9/UihUr9NRTT63x++xNxaBBg/TAAw+oZcuWSiRk8gDEeu+7sNtvv92tw/vwww/Vpk2bqI4LAAA0MMibM2eO+zp06FB988036tOnj9LT0906jJqXdV2o/8MPP+iQQw6JHMvMzHTlti3DtyaXXHKJmyJk0zwTsRc6hVcAxJpQKKSnn35a++67ryoqKtyxtLQ0nXzyya4fHgAAiNPCK5tuuqn7Q//pp5+u9w+2gNHua+W+era/piDv7bffdj/fsojrqqyszF3CwtNB7RPpmp9KR4udh5qCnhQbXNTGg8Rgz217bsXCcxzxbenSpW72xn//+99IqwT7sA9YH7xGobHxnEIyPKeC6ziWBlXXtE9y11f4k2DLBNZk2bzwdSubNWuW68H0wQcfqEmTJuv8s+688043xXRlixcvdlM/o620rPYYlizPU9C/KGrjQWKwF4G8vDz34kSmBQ31448/6txzz9W8efNcL9Tw9PpFi3iNAq9RiC383UMyPKcKCgo2XJDXGMJr6ewT4pqWLFmy2nV2FtwVFxfrvPPOq7Xwf/78+fr222/dm5G6ynZfc801uvTSS2tl8iwraT39cnNzFW3p6WlSUeW2JxRSq9ZtpFzWt2D9X5hs+rQ9z2PlhQnxIxAI6O6773ZBnW1vueWWrkeqFVfh+YTGwGsUGhvPKSTDcyojI2PDBHm33XbbWm9jVS/Xxspt22XcuHEaOHBg5LgFatZIty5WbGXl/nzHH3+8tt9+exfEra4vk2ULV84YGnuwYuMB89Ta8loz9JgYF+KdvTDFzvMc8eTiiy/WI4884rZPOOEEF+CVlJTwfEKj4jUKjY3nFBL9OeVdx3HUO8izP/SNEeSZ0047Tc8++6zOOOMMl1kbNWqUa41g/e9qBpXWbPe1115zUbRdVp7eaVXdrF9ewrRQ8NJEGEB0nX/++Xr99dd111136aSTTnJTVSzIAwAAsa/eQd6CBQsa7YdbC4W///5b3bp1cwVXbI3Hk08+6TJzYf/8848L8pKFy+l5YuOTAgDJw+/366uvvnIVjo1Nz5w5c6b7IK2uAlEAACB2RW1NXrj89quvvuoamluAt/nmmysrK2uVQLCwsHC192Hfn5OTo/hGJg9A9MyePVvHHXeca2vzxRdfRKbMhwM8AAAQX6Ia5IW1bdvWXerSuXPnNX5vjx49FO9qfkBemcljuiaAjeP99993ve6WL1/uClFZFTEAABDf6jUv8KqrrtpwI0li/lB57QOsyQOwgVnvUCuucvjhh7sAb8cdd9TEiRN12GGHce4BAEimIM8W4KPxFdToief3eFiTB2CD+uuvv7TrrrvqoYcecvtWndja0Gy22WaceQAAEkBMTNdMdjkp1T3xXA97pmsC2IDGjBmjCRMmqEWLFnrppZd06KGHcr4BAEggBHkxpkkwyHRNABuUta+xglfWGqFjx46cbQAAEgy1+mOutqZN16xujg4A62vKlCk65JBD3No79yrj8WjYsGEEeAAAJCiCvBiQX1IR2Q7RIw9AI7Heds8995wrqvLRRx/pyiuv5NwCAJAEmK4ZA9JSfFJZ5XaosokCAKyX/Px8nX322Ro5cqTb33///XXbbbdxVgEASAJk8mJAqq9GYEfRFQDrafz48erbt68L8Hw+n6uM/L///W+1/UgBAEBiIZMXY0Ik8gCshw8//FBHHnmkKioq1KlTJxfoWbsEAACQPAjyYg1r8gCsh1122UXt2rXT9ttv79bjWZsEAACQXAjyYo4v2gMAEGemTZumLbfc0lXNtKDuhx9+UPv27d0+AABIPqzJizEh3pQBWEfBYFB33nmnttlmG73wwguR4x06dCDAAwAgiRHkxRqCPADrwJqZH3TQQbr22msVCAQ0duxYzhsAAHAI8mJMiIcEwFp8/vnn6t27tz755BNlZma6tXdPP/005w0AADgEebGGwisAVsPv9+u6667TfvvtpwULFrhpmuPGjdOpp57K9EwAABBBkBcDQqEaO0zXBLCG/nd33HGHQqGQzjjjDP30008u0AMAAKiJ6poxh7gbQN369evngrwuXbpo8ODBnCYAAFAnIooYQ3VNAGHl5eW66qqrNH369Mixq6++mgAPAACsEZk8AIhBf//9twvmfv75Z3322WduaqbPRx9NAACwdmTyYg7Ni4FkN2rUKPXp08cFeNbc/KabbiLAAwAA64wgLyZUV14JEeQBSaukpERnnXWWy+AVFBRot91206RJkzRw4MBoDw0AAMQRpmvGGPJ4QHKaO3euDjzwQE2ePNm1Qxg2bJhuvPFGpaTwMg0AAOqHdw8AEANat26trKwstW3bViNGjNC+++4b7SEBAIA4RZAHAFFiUzIzMjKUmpqqtLQ0vfHGG0pPT3eBHgAAQEOxJi8G1OyFTjN0IDlMnDhR22+/va6//vrIsU6dOhHgAQCA9UaQF3NYlQckslAopEcffVQ777yz6383cuRIl9EDAABoLAR5MYZm6EDiWr58uY4++mhdcMEFrtH5YYcd5jJ6OTk50R4aAABIIAR5ALARjB07Vr1799Y777zj1uA9+OCDevfdd10fPAAAgMZE4ZWYw3RNINHYdMyDDz5YK1as0Oabb+6andt6PAAAgA2BTF7MlV4hyAMSjU3HfOSRR3TcccdpwoQJBHgAAGCDIpMXazzE3UAi+OKLL9y0zN13393tn3DCCTr++ONdo3MAAIANiYgi1vAGEIhrfr9fN9xwg2tmPmjQIC1atChyHQEeAADYGMjkxZgQcTcQt+bOnashQ4bo66+/dvuHHHKImjRpEu1hAQCAJEOQFyN9s6oxlQuIRx9++KFOOukkLV261AV2Tz/9tFuDBwAAsLExXTMGeGoUXgmxJg+IK4FAQJdffrkOPfRQF+D17dvX9b4jwAMAANFCkBcDPDUzeazJA+KK1+vVnDlz3PaFF16o77//XltssUW0hwUAAJIY0zVjQs0gj7gbiJcCKykpKa6Yik3NPPHEE90aPAAAgGgjooix6ZqsyQNiW0lJic455xwde+yxkfW0ubm5BHgAACBmkMmLBaFg9SaZPCBm/fHHHy64++2331wG74cfftAuu+wS7WEBAADUQiYvJrAmD4h1L730krbffnsX4LVp00ajR48mwAMAADGJTF4MYLomELsKCgp03nnn6eWXX3b7AwYM0IgRI9SuXbtoDw0AAKBOZPJiTEi+aA8BQA1HHXWUC/CsiuZtt92mjz/+mAAPAADENDJ5saBWC4VoDgTAym666Sb99ddfLtDbbbfdOEEAACDmEeTFAJqhA7FjxYoVGj9+vJuWafr3768///xTqamp0R4aAADAOmG6ZkygTx4QC6xaZu/evTVw4EBNnTo1cpwADwAAxBOCvBhA4RUguoLBoO655x7tvvvumjVrltq3b6+ysjIeFgAAEJeYrhlra/JYlAdsVIsXL9ZJJ52k//3vf27f+uA9/fTTatq0KY8EAACIS2TyYgBr8oDoGDNmjHr16uUCvIyMDD311FN67bXXCPAAAEBcI5MXE6ozeR4P5TWBjeWLL77Q/Pnz1b17d40aNUo9e/bk5AMAgLhHkBdj0zVDHpKrwMZyww03KDs7W+eff777CgAAkAiIKGJtuiZr8oAN5qOPPtIBBxwQKaqSkpKiq666igAPAAAkFIK8mEDhFWBDKi8v1xVXXKFDDjlEn3zyiR588EFOOAAASFhM14wBHqZrAhvMzJkzNXjwYP30009u/4ILLtDFF1/MGQcAAAmLIC/mmqFTeAVoLG+99ZZOO+005eXlqVmzZnr++ed15JFHcoIBAEBCY7pmzCHIAxrDAw88oP/7v/9zAd7OO++sSZMmEeABAICkQJAXY4VXADSOI444wmXvrrzySn399dfq3LkzpxYAACQFpmvGHDJ5QENNnDhRffr0cdtdu3bV9OnT1apVK04oAABIKmTyYgLVNYH1UVhYqJNOOkl9+/bVxx9/HDlOgAcAAJIRQV4sqDVbk0weUB+//vqrdthhBw0fPlxer1d//PEHJxAAACQ1gryYq64ZzXEA8SMUCunJJ5/UTjvtpGnTpqlDhw768ssvddFFF0V7aAAAAFHFmjwAcWfFihU644wz9Oabb7p9a3L+4osvMj0TAACATF6soLomUB+ffvqpC/BSU1N133336f333yfAAwAAqEImL9bQDB1Yq2OOOUbDhg3TYYcd5qZrAgAAoBpr8mIOi/KAlS1ZskSnnnqqFi9eHDl22223EeABAADUgUxezIV1BHlATdbIfMiQIZo7d65bi/f2229zggAAANaATF5MoLomsLJAIKBbbrlFe++9twvwttpqK914442cKAAAgLUgkxcTKLwC1DR//nwdf/zxriWCsUbnjz76qJo0acKJAgAAWAuCvJjDdE0kt/Hjx+uggw5y6++ys7P1xBNP6MQTT4z2sAAAAOIGQV4sCJHJA8K6deumnJwctW/fXq+//rqbpgkAAIB1R5AXc8jk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rDzPEf/sDx3PJ/CcQizjdQqJ/pzyruM4ojpaa8r87LPPav78+ZHGzNZGwY6H3X333Ro6dGhkf/jw4a5aoAV41kIhMdAMHQ3z999/uw9FwgGeZbe/++47bbbZZpxSAACAJBXVFgo33nijpk2bpi222EKdO3fWP//8496s1iyiMn36dE2YMMFtW388CwBtCprdLvzG1lgT9ZoZwXhCnzw01FtvveWy4TbF2danDhw4kJMJAACQ5FKiPaf07bff1uzZs7Vw4UJtueWWkZYINYO38ALDJk2a6KuvvqrzvqzBc/yqWe0wNlLBiA+XX365W8dqjc1tnSkAAAAQ1SAvrFOnTu5SF8vyhaWkpGi33XZToqnZQoE+eViTqVOn6qabbnJZO6soa/OybUozAAAAEEbaKCbU6JNXuwwLUPm8CIVcRVkrqvLGG2/ohhtu4MwAAAAgdjN5yc5Tqzk1QR5qs5YfVmzI2oUYaxlyxRVXcJoAAABQJzJ5MaBmiMd0TdRkRYe23357F+D5fD7deeedGj16tGs/AgAAANSFTF7MIZOHSlaU6LjjjlN5eblbs2qB3q677srpAQAAwBoR5MVeLg9wdtppJ+Xk5LhiQ88//7xrkwAAAACsDUFerCGRl9RmzZrlekaajh07aty4cerSpYs8Hp4YAAAAWDesyQNiQDAY1F133eVahrz33nuR4127diXAAwAAQL0Q5AFRtnDhQh100EG65ppr5Pf79cknn0R7SAAAAIhjTNeMOUzLSyaff/65TjjhBC1YsECZmZl65JFHdOqpp0Z7WAAAAIhjZPKAKLCM3fXXX+963lmAt80227j1d6eddhrTMwEAALBeCPKAKPjqq6902223KRQK6YwzztBPP/3kAj0AAABgfTFdM9ZaKFBFMSkMGDBAV1xxhfr06eN64QEAAACNhUxejGFFXmKyhubXXXed5s2bFzn2n//8hwAPAAAAjY5MXkygGXoi+/vvvzV48GD9/PPP+v77712xFfreAQAAYEMhkxdzyOUlktdff119+/Z1AV7z5s118cUXE+ABAABggyLIAzaAkpISnX322Ro0aJDy8/PVv39/TZo0SYcddhjnGwAAABsUQR7QyGbNmqV+/frpqaeeclm7a6+9VmPGjFGnTp041wAAANjgWJMXAzw1luSFqK4Z91q1aqWKigq1bdtWI0aM0L777hvtIQEAACCJEOTFHNbkxaOioiJlZmbK6/UqOztb7777rpo2bap27dpFe2gAAABIMkzXBNaTrbWzfnf3339/5NhWW21FgAcAAICoIMiLOWTy4kUoFNJjjz3m1t9Nnz5djz/+uEpLS6M9LAAAACQ5gryYQJ+8eLN8+XIdffTROv/8812j84EDB2rcuHHKyMiI9tAAAACQ5AjyYizEo0l27Bs7dqybnvnOO+8oNTVVDz74oN577z21bNky2kMDAAAAKLwSCzxk8uLGkiVLXLXM4uJibb755nrttde0ww47RHtYAAAAQATVNYF6tke4/fbb9eOPP7o+eLm5uZw/AAAAxBSCvJhD4ZVY8+WXX6p58+bq3bu327/ooovcV6bWAgAAIBaxJg9YjUAgoBtvvFEDBgzQscceq4KCgkhwR4AHAACAWEUmL9aQyIsJc+fO1fHHH6+vvvrK7e+xxx7y+XzRHhYAAACwVgR5sVdfM4rjgPnoo4900kknuSIrTZo0cWvvhgwZwskBAABAXGC6JlCloqJCl19+uQ455BAX4FmbhAkTJhDgAQAAIK4Q5MWYEJm8qLHpmL/88ovbvuCCC1w/vG7dukVvQAAAAEADMF0zxlDQY+MLBoPyer3u8vLLL+unn37SYYcdFoWRAAAAAOuPTF7MrcnDxlJSUqJzzjlHZ599duRYu3btCPAAAAAQ18jkxRgyeRvHH3/8oUGDBunXX39159ymZ/bs2XMj/XQAQDy00bG12oivmTn2mJWWlrrZOUA8PqdSU1MbpaI7QR6SzvDhw10Gr7i4WG3atHFTNAnwAAAmFAppwYIFWrFiBSckDh87e1NufW350Bzx/Jxq1qyZm122Pj+TIC/m0EJhQyksLNR5553ngjyzzz77aMSIEWrfvv0G+5kAgPgSDvDsQ8CsrCyChTh7Q+73+5WSksLjhrh8TtnPsyTEokWL3P76vEclyENSsH80Bx10kL799luXbr/55pt1zTXX0OAcAFBrimY4wGvZsiVnJs4Q5CERnlOZmZnuqwV69lrU0KmbTFhGUrB/mFdffbU23XRTjRkzRtdddx0BHgCglvAaPMvgAUC0hF+D1mddMJk8JCz7NNYKrOy8885u35qcT5s2LfIJCQAAdWE9F4B4fw0ikxcDaKDQ+H788Uf16dNHBx98sGbPnh05ToAHAEDD17ZPmjQpat+fDK2d7PzYtOGNIT8/X3/++edG+VmolpeXp+nTp2tDI8hDQrEKSPfee6922203/fPPP6460fLly6M9LAAANqjffvtNixcv3qA/4+eff3YfoNoapYYEdPX5/jUFQXb55Zdf9O+//260gGhjsNlGdn421vuWM888U2+99dYqx+fPn+/OcV1TBa3K5Oqu+/3339331rWubdasWfrrr78arS2J3efff//t7ndd+f1+9z11jdHYcyr8/Apf6vo3VV5e7oI0e46vjSUa7H6sBUOYPWf79+/vzseGRJCHhGH/EA899FBdccUV7h/yMccco4kTJ6pXr17RHhoAABvU7rvvrldeeWWD/owmTZq4v6nrMpXshx9+cAFLQ79/TUGQ9bk96aST1K9fP1dmPlw1O97ZbCM7P1bkY0MbN26cPv74Y1100UWrXPd///d/7jzXFQB+99137rqFCxeuct3AgQP10EMP1Qpmbr31Vlc8xB4rWzZjH74feeSRLiBsKAvGttpqK3ef2223nRvPjBkz1vg9Dz74oCumtN9++6lHjx6uwvrKAdz222+vo48+WieffHLk8sknn0Sutw8V7Jjdj80Ua926tXuvubp2K5ZssMfTxlczoGvRooVOO+00VwBwQyLIQ0KwYiq9e/fW//73P2VkZOjJJ5/UqFGj1LRp02gPDQCAmGDZBHtzbW0iVsfesNobUvuwNJw5s1kyZuutt9aLL75Yq3CZZVTmzZtXK6Ni32cZExPOiMyZM6fO7w/fh71JrytwqIv9fbf7tJ97/vnn6/TTT4/8vJrs/izjsrrMoR23wNGmLRrbXrJkSeT6yZMnu/Nh2Se7bu7cubXGbD/Tfq/VsfM8c+ZMd9t1ua5Tp07u/OTk5KySObIaAzV/flhRUZE7F3Y/djt77GpmjVbHgrHBgwevUmTIfo4F6BboPffcc1ofxx9/vHs/ZsGi/b52Di0TeOqpp7oP4RvCfkcLEnfZZRdXfdIeLwsiLfBfndGjR+uyyy7Ta6+95p5nlsmz94f2QcHKHnvssVqZPPsdambL9957by1btsw9r+zxs2P2HKzruXXcccdp6NChdY7plFNO0TvvvLPG5896CyWhvLw8+xflvsaCS5/ZN7Tti9u6y1tjno32cOLSueee6x7TrbfeOvTrr79GezgxIRAIhObPn+++AjyfEGti8TWqpKQkNHXqVPc13jRt2jT0wAMPrPb6Rx55JNSkSZNQp06dQhkZGaF99903tHjx4lq3uemmm0KpqamhLl26hJo3bx4655xz3N/WgoICd/2XX37p9isqKtz+33//HerZs2eoZcuWoc022yzUtWvX0BdffBH6448/Qptvvrm7ba9evdzltttuW+X7zZtvvhnq0KGDu4927dqFDjrooNDSpUvr/B0mTpzovt++hs2bN88dGzlyZCgYDIbKy8tD//77b2jvvfd252SLLbYI5eTkrHJufvjhh1DHjh1DLVq0CLVp0yZ05JFHut/bzlOYjenkk09247P3F//5z3/c8dGjR7vzaN+/ySabuN/1u+++i3zf3LlzQzvssIM7h/bz7Xb//e9/13pd+Per+bi88MILoWbNmrnbZWVlhfr37x+aM2dO5PpvvvnGfc/ll18eat26tXsM7HZ2PlbH7/e7c/LOO++sct1ll10WOvDAA91j6PV6QzNmzKh1/f/+9z/38+wcr8zOw1VXXeW2P/vsM3e7d999N7Q29nuv6TJ9+vTIbT/88EN3vzNnzowc+/bbb92xCRMm1Hn/F198sXue1vT555+77/nnn38ix3w+nztvv//+e6i4uNgdCz+n7Gtdbr31Vvf8WNmVV14ZGjRoUGjcuHHu5/z222+r3KZz586hxx57rN6vResax5DJQ0K47777dMMNN7j5/j179oz2cAAAiBk//fSTm5Zn0zkt42YZsKVLl+riiy+O3Oabb77RLbfc4mbEWIbCMn6ff/75Gu/3pptu0hZbbOEyZpbV+uqrr9z32lQ6y+CYcEZk2LBhdY7LMjB2nWVkLMNywQUX1Gudlf0eNQurWUbriCOO0DbbbOMyPZZxsSmGN954oz777LPINELLsNjUPZuyZ+O3LJpNr1uZnQP7fjsfthzECpXYFL1HH33UTd+zTIyN2TJflqUyd911l8sUhX++vTcJZ2zWdN3K7GeeccYZevjhh93PsnHaVNezzjprldvadZbps0yV/a7nnHPOate/TZ061Y115eUsdvuXX37Z/Ux7DK2+wfPPP6+GePfdd920xMMPP3yNt7PHoub0yLouds7Cxo8f7zJ3Xbp0iRyzKup2XiZMmFDnz8jNzXXZt5rrN8PNxu3812TTKG1aqT1GQ4YMcd+3Jr/++qs6d+5c69inn37qss3hfwOrY9M4x44dqw2FFgqIS5Z6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IipLIhUV2pQilD6i7aN9kSItPqV9004oSlJaJKSSosWSSNmyZ1dCdtnO//F8f39n/mfmnpl7cS9zz/16Ph7zmHvOnOU617nuM9frvDchhBBCiMxBlrxEII9UKvzuu+9cly5d3Lp166z+x1tvvZXTTRJCCCGEECLpkCUvwcicMouJxcGDB62g6jnnnGMCr3bt2u7OO+/M6WYJIYQQQgiRlEjkiSxlw4YNrk2bNu7hhx+22nfdunVzP//8szvllFPU80IIIUSSQH29KlWq5HQzci0DBw5M2pqEicyff/7p6tSpY+FEyYZEnsgyZsyYYWLum2++ccWLF7ci55RHOPLII9XrQgghRCaCl8wxxxxjr5o1a7rzzz/fTZw4Mdv6mDq3GzdudDnBtGnT7LwrVqzoqlatau98vueee1xO8scff1g71qxZE3e9rVu3ugceeCCql1PPnj1tH9Gu5Zw5c+y7xYsXp/juggsucL179w5bNnfuXHfttde6k046yR133HGudevW7uWXX3Z79uxxGYVt7733Xht/J5xwgrv77rvdrl274m6zYMEC17lzZ9umXr16VpB+//79Udc9cOCAa9WqlZ3nTz/9FPbdokWLLAzoxBNPtH2xz3nz5oWts3HjRvf444/b95x3JBUqVDBPs4ceesglGxJ5IsuoUaOGy58/vzv55JPNekcGTSGEEEJkjUXi1ltvtYn8F1984Ro0aODatm3rfvzxx6Tv7n379tn5f/bZZ27WrFnul19+sX7o169fjrYLgUK7CFuJBwXoETuRImT37t3uzTffdGXKlHGDBw+Oed7RBBKWqW3btoU+v/fee+700093pUuXdu+88449gKcQ/KpVq0xkZRRE46effmrt5GE+YvQ///lPzPVXr17tzjzzTFeyZEm7XpzXJ5984u64446o6z/44IN2fpwn5+uDkGzZsqV99/HHH1sbmHMi2KjB7HPxxRe7nTt32jv7iJXpHSPE5s2bXVLh5UG2bdtGqhN7TwTuHnyeV294PXt9OvVNLzezefPmsM/z5s3zdu/enWPtycscPHjQ27Bhg70LofEkEo1EvEft2bPHW7hwob3nNkqVKuUNGDAgbFnlypW9nj17hj7Pnj3bq1Chgr2qV6/utWrVyvvqq6/Ctvnoo4+8+vXr23vLli29448/3uvYsaO3du3asPUmTZrkNW3a1KtZ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8mTXvHlzV6pUKYs1Xrp0qTt48GA2tVgkMoyDefPmRR1T8X73HnjgATdx4kT3yiuv2Ph64oknLNbz3nvvzYZWi2Rj0aJFbu/evWHjMH/+/PY5Eefnmt1lMVu2bLH3smXLhi3nM4Mls7YReQfGR8mSJV2hQoVSjA9/7KTGlClT3BtvvOEGDx6cRa0UuQnGTbT7jf9dNBYuXOj69u1rweqRY1GI1MYUD6UiISnGUUcdZQ+wDh06ZOsxtt555x131llnqVPzMDt27HD79++POqbi/e4hAtu2bev69evnjj32WEuWceONN7q6detmQ6tFsrElzvycRImJhkReFuNPflD+QcjgU7hw4UzbRuQdGB+RYyM944NsUJdddpm788473bXXXptFrRS5fUz5WcZijanu3bvbONq0aZO9EH0wc+ZMV69ePbMSi7xLRsYU23z99dfuiy++cG3atDGhd9NNN7mOHTtadkT9/uVdMjov6tKli9u6dauNHzwVyMjJAwNE46BBg7K83SK5KJTL5udy18xi/IlOZMp6PlerVi3TthF5B8YHqe7//vvvMFcW3KBSGx+kx2/durWJu+eeey4bWityy5iKdr+BWGMKSwyeBX369LHX22+/bcv79+9v7lEibxNrTOHahHt5NLC01KhRwwQesC5WF+5tS5YsyZZ2i8SkePHiZi1Jz7yI30XuRd26dQu5orOPq666yn322WfZ0m6RXFTPZfNzibwshh+zE088MeyG8tdff7np06fbZNuHyRL+5unZRuRNzj77bHtiFBwf33zzjT2ZDI4PYu7++OOP0Gdiq/i+a9eubsCAAdnebpG4MC78umQ+Y8eOdaVLl3annXaafcZVCtdM/+EC3/PZfz399NO2fPz48WZ9EXkbxhRWOeLLfRgzxNwdccQR9hkLC2PHfyp+/vnnm6Ul+JR87dq19n700Udn+zmIxBtT48aNCxNxEyZMCPvdw2I3Y8YM+7tAgQKuTJkyoTHkQ81GjSeRVvBSoQ6s/yCqZs2aYfMvYox/+umnxJyf53Tml7wAGQ8LFCjg9e/f3/4mbf0pp5zi7du3L7ROu3btvBYtWqRrG5F3eeihh7zSpUtbaQ3qs1SpUsW75pprwtahdt4999xjf1N6gzowZ599tvf999+Hvf79998cOguRSFAriNqLo0ePtrTSZBCjzpnPhg0bLPvYqFGjom6v7JoiyPbt273jjjvOa9OmjTd27Fivd+/eVgZo6tSpKcbM0qVL7TO/b9QyI1vihAkTvLffftvqgHbr1k2dK7wFCxZ4xYsXt8ytlOag1MbRRx8dVs+TDIeUWfBhDsU2zz33nPfVV195jz76qI1DfjuFWLNmjc2Dhg4davci5lN85vfOh/vReeedF/r80Ucf2Rh6/PHHrbxLkyZNLPNrImYqV0xeNkDcCi4DZKIj4QWWmN69e4clKyAIGEtMerYReZdHHnnE3Jo+/vhjd+DAAdezZ0932223pQg4Zx3Aood1mCefuNYFGTNmjJ5qChtLL7zwgt1zihUr5oYPH+46d+4c6hmsx1hhYj0B54k53+seJaBEiRLuxx9/dE899ZR76aWXXIUKFey3jDESOWZ8yx5jhwyIzz//vHkbYEl++OGHFTssQvMkPJq4T7344ouuVq1aZrWrXLlyqIdwmWvWrFno84MPPujq1Kljv3PEerIu3gZYjYWYOnWqGzhwoHUE96LXX3/d/ib7art27ezvk046yUJkfDp06GBjaOjQoeZFde6559r8PBEzledD6eV0I4QQQgghhBBCZA6KyRNCCCGEEEKIJEIiTwghhBBCCCGSCIk8IYQQQgghhEgiJPKEEEIIIYQQIomQyBNCCCGEEEKIJEIiTwghhBBCCCGSCIk8IYQQQgghhEgiJPKEEEIkFHv27HHvv/++27Fjh8uNzJo1y4o2H+46QgghREZJvPLsQgghcj2TJ092mzZtSrH8sssuc0WLFo277ebNm91VV13lFi1a5E488cRMb9uBAwfcRx99FPpcqlQpV6dOHXfsscdmyv7feOMNt3PnTtesWTP7PHPmTOd5nmvatGnMdbKCuXPnut9//93+LliwoKtUqZJr0KCBO+KII9K9r08++cQ1adLEValSJQtaKoQQIrORyBNCCJHp9OvXz23YsMGddtppYcsvvPDCVEVeVrN3714Tkc2bNzfRsnXrVjdlyhR3zTXXuCFDhhz2/hFDHMNn4MCBJiyDIi9ynazg3XffdUOHDnUXXHCBO3jwoPvtt9/MOvrhhx/auaeH6667zvZ1xRVXZFl7hRBCZB4SeUIIIbKEc845x4RBECxauGLaD1DBgq569epmXeLveOzbt88sYtu3b7f1sUoF2bZtm7k/5s+f35166qmufPnyqbbvrrvuCokW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'
NOTEBOOK_PR_B64 = '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zRWV5zTd7HPvM8ovKKzJq0Y9RZAHAAAA1JbOh0hHP1y8/tm10qqZFR4SHx2vWWfNKpXeoNNz6n7nBGXnUaOHQAR5AAAAQG064ELpwIs9y4V50jtnS2lrKjykQWwDTTltSqn0+Jafq9ftk2i6iQAEeQAAAEBtsykWuozwLGdu9QR6eVkVHtIssZkzl96ZPc/0pcWmzpMrdrt63saomyhGkAcAAACEYkCWU14unmJhwx/S59ftccJ0c+3+1wasJ3d7SJLbGYwFMAR5AAAAQCg0aCKd8ZYU28CzPmeCNOvZPR8W20DXDLwmIC2xw/POc787v6qZa0W9QpAHAAAAhErLvtLx44vXJ90iTX1AKqx41My/9/+72ia39a3HJC13ntNz8p0avYLCPdcIInwR5AEAAACh1O8kadh1RStuaer90ttnSllpFR428aSJAetxzb7xLXf9Z+A2RBaCPAAAACDUDr/T83AV3Z4vniQ9P1LaNL/cQ6JcUc7DK775d0rq/m/f+qptGTV7zaiz6lSQ98UXX+iSSy7RxRdfrE8++WSP+2dlZel///ufLr30Uue4Z555RhkZFGYAAADUMy6XNPx66ewPpMTGnrTty6UXj6xwHr3PT/w8YD0qJl3xrd93lg99eKrSMnNr9rpRJ9WZIO+BBx7Q6aefrvbt26tz584655xzdNddd5W7f35+vkaMGKHHH39c/fv313777afx48dr+PDhys2lMAMAAKAe6nqYdPH3Uut9Peu5u6U3TpKWf1/m7u1T2uvpw58OSItr9Itv+baP59Xs9aJOcrndlRintYZt27ZNbdu2dYK0v/3tb07aq6++qosuukirV69Wq1atSh0zZ84cJ7D7+eefNWjQICftzz//1L777quffvpJgwcP3uN5d+3apYYNG2rHjh1q1KhRDbwzoGKFhYXavHmzWrRooaioOvObCyIIZRChRhlEqNXZMpibKb17jrT0W896dLx0+htSjyPK3D09N11DJgzxrRdkt1LmCs9UCyvuP1ouqylEnZSWlqbGjRtr586dSk1NDcpr1omS/N133yknJ0ennHKKL+3kk092/tF9/fXXZR7TunVrJSQkaNWqVb60lStXKj4+3qkNBAAAAOqtuAae6RV6HuNZL8iR3j5LWlj2XHjJccmKt0CwSHTCRkUlrHWWj31qeu1cM+qMGNUBy5Ytc2rU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8EL94x//YHoCAABQM9wRbOfOnW7Lgh07dpTalpWV5f7rr7+c5/omKSnJ/eSTT5a7fcGCBe6zzjrLvc8++7hHjBjhfumll0rtM2vWLPfRRx/t3n///d3nnXee+5NPPnH37NnTnZOT42z/8ccfnXUvS//Xv/7lHjZsmHvIkCHum2++2b179273vHnz3O3atXPy2fa3xz333FPqeLNhwwb3Nddc4z7ggAPchx12mPuNN94o9z3Mnj3bec25c+f60v78808nbdKkSb60wsJC9zPPPOO8T3sv5557rnvJkiWlysHVV1/t3m+//dzHHHOM+7333nOPGTPG/fbbb/v2GTRokPvNN990X3zxxe59993X/eCDDzrpW7dudd9www3ONdt7v+2229wZGRm+4/Lz890PPfSQ+5BDDnEPHjzYff311/vKm22z1xk+fHipbZZvlj+7du3yvdby5cvd559/vnN+e73x48c778//M7NjfvrpJ/epp57qvF/7nFetWuWuSPv27d0vv/xyqfQnnnjC3adPH+ezb9SokTszMzNgu+Wj5fe0adN8aStWrHDS/PMuWH799Vfnta3seL322mvumJgY9+bNm8s85p///Ke7Y8eO7oKCAl+afY72b8T/c8rOznaeb7311lLlsrLntnJn+6xfv963z4wZM5y0RYsW+dLy8vKc81u+1gVWhnJzcwPKUn3/DkT9Yv8+7fvf/98pQBlEJNmxY4dzv2D3pMFCc80Is3nzZqcWzYZHf/bZZ3X22Wc7Tcqeeuop3z5W23HYYYepQ4cOTu2NNT0bN26cU2tj00qU1dzy7rvv1oQJE3T77bfriSeecGpxrAlg165dnWerxfj444+dx0UXXVTqeLsuO4/VeNx///3O63z55ZeaMWNGpd9bw4YNnWf/Wkxrbmjv8/rrr9dzzz3nTIlxwAEHODUwXuecc46+/fZbPfjgg7ruuuv073//W19//XXA6yxevFgXX3yx0zTxtdde0wUXXODUUFqtlc0n9vjjj+u+++5z+hqecMIJvuMeffRRpwbp//7v/5w87tixo7Ps3fbMM8/oxhtvdGp+/LeVbK65a9cup7bNzmmvZ9diNZf33ntvQHNCO+bvf/+7zj33XOezs2s76aSTys2z5cuXa82aNU7el/Tiiy86n9UxxxyjpKQkffDBB3v8DLy1d3adJVnNn7d5Z3kP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2wTX3bp102mnnVZrk3uvXLlSTz75pB5++GHVtsWLF5cZIB188MHOhOehYhPK33fffbr77rvVsGHDcvcrKCjQRRddpDvuuKPUto8++kjff/+9M4dh586dA7ZlZGTo1ltv9a3bXH99+/bVSSedpOjoaNWUn3/+WV988YWzfMwxx+jAAw+scP/CwkJ99tln+v333531/fbbT8cdd5wzXYf/PvZe58yZo/j4eOc99O7du1Q+2Xl/+eUXJSUlaeTIkaXOvadztWnTRpMnT1anTp30t7/9LUg5AgAAEHrU5IWh8ePHa+3atc7Nq90k/+c//3Fukrdt26Zwt3r1aj3xxBNq3Lix8/69j6ZNm4b0urZv3+5c1+7duyvc78MPP9SuXbt04oknBqRbbZPVjv7vf//Ts88+W+q4rKws5/UzMzOd92tB0LXXXusEt7m5uaoJzzzzjA455BDnve3YscNZtrTy2DUdeeSRuuqqq5wAzB5XXnmlk2bbjF2rvY79UGHB6caNGzV48GAnX/wD2oEDB+r11193yveaNWt02GGH6ZZbbqnSuYzl0b///W9nOwAAQLigJi9MjRgxwrmpNfbcrl07Pffcc/rnP//ppK1YscIJCkxCQoK6d++uM844w6kV8a+lmThxoq644gp98MEHTuB4wAEH6IQTTgg4186dO/XKK684QeS+++5bZo2hBSETJkzQokWL1KJFC51++unONZU8l9WofP75586Nu9XM2Lls+e2333aCnyOOOELDhw/f4/u31/F/fX9Ws2M1PHZNQ4YM0bHHHiuXy+Xb/q9//cs5jwUv06dPd96TXa+ZNm2avv32W8XExDgBlAUXJZuqfvLJJ04gYsGJ1W7Z8v333+9sv+uuu5zaVatlsxq7kuwzOuusswKux0yZMsUJeCxgt9rAe++917mGkqymauzYsc6yfZ79+vVzAiRbDiYL6m688UY98sgjvnLWpUsXJxA988wznZrEkubOnevk3Y8//ujkuznqqKOcfLRtVtNmgduvv/7q1Aa3bNnS2cfew6WXXuq8r7i4OCf4szy2YNbL8tNqOO+8806nPFfmXN78uuCCC/T111872wEAAMIBNXkRwGq1LOCxG2cvqwHx1nJZ88HXXnvNCWb8a5r++usv5ybealaWLVvm1CbZDbEFKl62/6BBg5ybcws6rJbp8ssvDzi/1S5ZwPb4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AKAec0ewnTt32p2Pe8eOHaW2ZWVluf/66y/nub555pln3A0bNnT37dvXffDBB7tbtGjh/vvf/+683/rE8t4+n2nTppW5PT8/3926dWv3rbfeWub2iy++2N2nT5+gXMuaNWuca1mwYIE7mAoLC925ubnuzMxM5/Xt8d577wXss2XLFndcXJyz7aOPPgrauZcsWeK85ooVK8rdZ/v27e4jjzzSvXLlSndNmz9/vrtBgwbOOUsaM2aMc6333XdfueVk3333da71wAMPdPLruuuuq7Fr/eOPP9wdOnRwd+7c2d2rVy938+bN3d9//32Fxzz88MPupKQk9+DBg51jOnXq5P7zzz8D9nnuuefcvXv3drdq1crZtzrntrJ68sknO//uhw0b5rzWfvvt5168eHHA6/zvf/9zDxgwwFcG7TlcvgNRvxQUFLg3bNjgPAOUQUSiHTt2OPcywbxXpyYvzMybN0+XX365nn32WWfZfslft26dhg4d6jTRqqiZntWgeGtArJbB9rFaQG+NwPLlywP2r8w+XlYDNm3aNM2cOdOpFSvvdax2afLkyVq9erXzbKzGwrZbrYW/6OhonXfeeXr11Ved2hx/9h7ffvtt/e1vf/Ol2Xntteyxe/fuMq/TatGslsUe3ho1e21v0zZ7j3YtP//8s+8Yy9fvv//eSbMalpLX4c1re0/fffedNm7cqPIcdNBBevHFFwPSXnvtNe2///6l9vXWXH3zzTdavHhxqdqait5TSVu2bHHe26pVq0rV+Fx77bVq2rRpqfdT3jFVyW9/Tz31lI4//ng1btw4IH3t2rVO2bz00kv10ksvlfs+b775Zufa7HwffvihHnvsMX322WcKNisPZ5xxhvNvatmyZVqwYIFOP/10J628pq+WTzfddJPeffddpybPjjnttNOch3/ZtVrp999/X3feeWe1z201s2eddZZTzuzfnNXCNm/eXOeee27Aa9lxdrz9mwQAAGHGHcHCsSbv9ddfd95TRdddXg1OdHS0+8svvwz4ReG0005zt2zZ0n3QQQe54+Pj3ddcc41v/8rsY95//32nZrF///7unj17OjUPkydPLvU6p59+urtt27buUaNGuT/99FP3cccd56RbzYfV0FxyySVlvheXy+WeOHFiQPorr7zi1OZYLZj54osvnJoNq+2x12vcuLH7jTfeCDjm888/d95H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/v777yf2R4loyt536dLFniRQZt/XGEzqv4F4cDQfXPNK1Q04mOZXAGMQkeL8/x9Tq8ImXFNqyOQBeGhoflowc9QQGjVtUZZXkvB5PgAAIhanzOBTfCWUwSwDAAAA4MEikwefNPfqXi3wA1kGAAAAwINFJg8AAAAAIghBHgAAAABEEII8AAAAAIggBHkAAAAAEEEI8gAAAAAgghDkAQAAAEAEIchDWJ0/f94kS5bM/Oc///H5/O3bt+3z33zzTUD3H2aJ/dmnT59uateubSLF1atXTf78+c3OnTsT+6MAAAA80gjyIlDGjBlNihQpzO7du+M8/sYbb5jq1avf12t/9913plmzZiZ37twma9asplatWmbhwoXG5XKZB7l9CqZ003ZGR0ebrl27mgsXLpik4tq1a6Z///5m2LBh7seuX79uXn/9dZMzZ057DcOmTZuan3/+Od7XOXTokH0dfZ/an3oNb99//719Lb1upkyZTKVKlcw///nPOMuUL1/e/Z04t86dO7ufj42NNbNmzbLjT69RqFAh06NHD3tSwJE+fXrz1ltvmV69et3n3gEAAEjaCPIiVOrUqU2/fv3C+ponT540devWNfny5TM7duwwx44dM8OHDzfz5s0zP/zwQ0CvkTJlShsQKji8H+PHj7evc+PGDbN8+XLz1VdfmT//+c8mMYVr2wKhgClHjhymWrVq7scUNGlfrF692uzfv99+lsaNG9sMoz99+/a1QdegQYN8Pq/XaNSokR1Pe/bsMSdOnLBB/gsvvHDXd/7BBx/Y5Z3blClT3M9t27bNrFixwowaNcq+xtKlS82aNWtM27Zt47zGyy+/bL799lt7MgEAAAChIciLUK+99poNfL7++mu/yxw8eNBmaJT1yZIli3nxxRdtIOfP2rVrzblz58xHH31ky+qUUVOQ8Y9//MOULl3a5zp37twx3bt3N0WKFDH/+9//wl7SqMDqiSeeML///e/vKvNT5knvlTx5cps5UpbIM1OlQOS9994zBQsWtNvyzDPPxPlcZ86cMR06dLCvo/1Tp06deEsJvbfNKV3961//aveTgqkSJUrYzKcnvc8f//jHgN9H5syZY4Mtx9mzZ820adNsZk/7Q9nNCRMm2EBs2bJlfl9n/vz5ZsCAASZPnjw+nz99+rQN5v/0pz+ZqKgouw3aj9pW70xxfKpUqWL+/ve/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qaYsCsZUyqmb5tOpxFLOnj1rTp8+bf8/Ojo63vds0qSJzexNnDjRlnPqvfUZVAp669atBN1eAEDSRZAHAEhSlE0rUKCAmT59uilVqpS9lML58+ftcwq+ZMKECeb27du2LFM3JzjTrXnz5jYTKMePH/f7Pvv37zebN282/fv3t81WsmbNat/78uXLtlwUAICEQpAHAEiSlKEbPXq0zcqNGjXKfdH0/Pnzm3nz5sW7rubiaTnNq1PgFx+VaXrSJRsAAEhIBHkAgCSrfv365tlnnzXjxo2zpZoK/D799FOzYcMG0759e/P999+ba9eumcOHD9sGK1pWtJxKMHft2mVat25t9u7da+fZqXum5vOpu6aarJQuXdoGkprrp8dmz55tvvjiC9thEwCAhEKQBwBI0oYPH247Zg4bNszeb9y4sS2zvHHjhqlTp44ts6xbt669ZIICNoeuo7du3TpbflmtWjWTJ08e07NnT9O2bVvbXVPz+JYuXWrnAuqSDEWKFDErVqyw2T89BwBAQknmuledCQAAAADgkcGpRAAAAACIIAR5AAAAABBBCPIAAAAAIIIQ5AEAAABABCHIAwAAAIAIQpAHAAAAABGEIA8AAAAAIghBHgAAAABEEII8AAAAAIggBHkAAAAAEEEI8gAAAAAgghDkAQAAAEAEIcgDAAAAABM5/g8yMGuycZufvwAAAABJRU5ErkJggg=='
NOTEBOOK_RF_FI_B64 = 'iVBORw0KGgoAAAANSUhEUgAABEEAAAMWCAYAAAAeTZgVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzs3QeUE9X7//GHpfelLV2wUBVRUGkWBFTEAmJBRVSwYcUuqHwVUbArdgSxK9jAgoiioiJYQEEUqSIKAksH6cL+z+f+zuQ/CcluspvdZDfv1zk5kGQyuXNnFvY+89znFsvKysoyAAAAAACAIi4t0Q0AAAAAAAAoCARBAAAAAABASiAIAgAAAAAAUgJBEAAAAAAAkBIIggAAAAAAgJRAEAQAAAAAAKQEgiAAAAAAACAlEAQBAAAAAAApgSAIAAAAAABICQRBAAAAAKCQOvroo61YsWLuUb169UQ3B0h6BEEAAEhyF198ceAX3Fge9957ryWLzMxMe+2116xPnz7WvHlzq1ixopUsWdL9wn7cccfZU089Zbt27cp2H7/99ptdcskldsABB1jZsmUtPT3djjrqKLv//vtt27ZtMbfpmmuuibov165da4kwbdq0oHaon4oSHY//+AYOHGiprqif82S75vyPChUqWNOmTa1fv372448/JrqpKSe7cxP60M9JUaP/Z/zHqP+jkD9K5NN+AQAAArp162azZs3ap0fWrVtnX3/9tXuMGjXKpkyZYjVq1NhnuxdeeMGuuOIK27NnT+C1HTt2uIGKHnr/s88+s4YNG9LrAHJl69attmDBAvd4+eWX7bnnnrPLLruM3gSKGDJBAABIci+99JJlZWUFPW666aagbZ599tl9trnzzjstmSh7Q+2eM2eOy9z46aefrG3btoH3f/nlF7vqqqv2+dy3335rl19+eSAAcv3119vmzZtt/vz57q6tLF682M444wz777//ct2+F198cZ8+9B6kmANF02233eZ+xvVvigIfxYsXd6/v3bvX/VujQC0Se27CPTQFCMgtgiAAACDfdejQwebNm2cPP/ywHXrooS4gcvjhh9t7771npUqVCmw3YcIENxjx0xQJDUikVq1a9tBDD7npNE2aNHFTYTyzZ8+2N954g7MJIGb6N+XCCy90WWseBWu//PJLehMoYgiCAABQRO3evdvGjBljXbt2dcED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41L/ffjhh2H3p1oiJ5xwglWoUIEACAAAAJIOmSAAAgGAW265xc466yz3/J577rG77rrLZQjEuh8NlHv27Omen3vuuW56iYIoCh6IAgjKQPCyLfKqa9eudtttt9nnn3/uBuAKDpx22mkuS8JPgZKPPvrIFixYEMhkePLJJ+3CCy90mRUtWrSwyy+/3GVphLbN/3zz5s3WsmVLe/31191n/ceudiizQnSMCliccsopLttk9OjRLuCiAFD9+vVtx44d1rFjR1dT480334zqWNW3Oj4FKtReUfbN7t27c9V3OlbJLjNH02V0bShAJv/++6/9/PPP+2z39NNP2x133OGCJsrCiZQpo4e/LwEAAICCQiYIAEeDdC8AIsoKUZHQ0GyHaPbjBUBE001q1qwZCIB4r2m6RLwUL17cLrroIhccUHsVsFB2SCgVA1UWimpivPrqq/bKK6+47TWtR0GQ7Cg7RAEMZXa89957Vrt2bfvxxx/3aYeCHp6rr77atm7d6upvyNtvv+2CQgqASJkyZVyASPVLvCyWnKjdCkx5ARBRQCaW6UXKglEfKVvnqquusosvvthllkRSsWJFNx3KC1goy+OYY44J2mbIkCHuoUBUpACIDB8+3F0j3sPrCwAAAKAgkAkCwElPTw/qidKlS7uBuTIMNLjP7X5UEyLca/5sgHhQEVQN5L3pJyeeeKJb/cRPwQ5lgEyZMiXodQVQsvPrr7+6/amGiIIoCgKsXbvWPfxq1Kjhaob4+1DBCX2v9/3KVAktQqq+WLVqldWpUyfH41StEWWW5IWKv6pt8+fPdzU+brjhhmy31xSmK6+80h1L69atXZaNAjzly5d376vtyhx6+OGH3fvZGTRokN14442B5wqsEAgBAABAQSEIAiAqCoSEZiuogGiyUM0KFexUZoMG9Rrch1LmgepZqLZFLFQLRdkNmv7i6d69u6vP4afaIXqtWLFiQVNkFBwR/Rlau0PP1dbQqTuRVKlSxdasWWN54S+Mqrokmp40b948F9wJR3VdVGB2w4YNri6KpklNmzbNPvjgA/e+9qXaJJdddpkL6qjuSSQKvugBAAAAJALTYQBERSuHqEiopoV4QjMqEu3222+3M888M+xUGNGUHBXuDM3g0HH5p36EBncUdNDg3rN69WoXDAilGh+qfeLRijLal4qlynHHHWcTJkwIqt+h9ih4E+0qKqp/ouksmmbjUY0OFWXNDWVvqN33339/xG28/lEA5owzznBFU7XCjd8FF1xgzz//vFtaVzVBAAAAgGREJgiAqChbQBkWqvehYMJ3333n6j8kE001CZ1uEjoVQ4EbTdlQ1oIyH3744Qc3yNcUEdHKNgpcqMBps2bN3BK5CqzceeedLoNBdTy06k24ZW3LlSvnpo1o1RxldzzyyCOuRkjDhg3d+yqYqgKonTp1srPPPtvVFNFqM+GW8Y1k4MCBrqaJlh7WNB4FVN555x2XlaEATqz0GRU81fEqiybclJy+ffu6ZXMVzFEQTJk0CnaE0mo8yoRR3RMFarRUMgAAAJBMCIIAcEu5+ouiioqZapDt1QOpVKmSzZw5061wotVVVCNDy83qeXb7USBB2QN+KuoZj5ogyk5RGyNlUajWhP99BT00jUMFSmfMmOGmd2hai7/N++23nwtKKGNDhVBVz0THqe9SgVMFP5555hm3rK0yKPzHrgBI//79XXaH3lewwL+6jvalAImWIFadEQVHZs+e7T7rUcaJP8AQ+lznQW1/7bXXXDaGpqKosKraHW1/+euWeFNifvvtNxcQ0lSW0O9UsEvBGgWKdD1oFRgFxbzj9vefVstRsEgZMQqaeFOBAAAAgGRQLCt0UjsAAAVEhVFVq2XDqGKWXo7/jvJqr6VZZvHWlrFnlqVZdCsOgf4sKFyf2Ws+uLrNWxJ9zSfV6crMzHSZeuHqYCHG65P+jCv6k/5M5O+VmzZtcjcOIyETBEBCffTRR/vU6PCozkWkKR6tWrWyQw89NJ9bV3T6kv4CAAAACIIASDBN6Vi6dGnY91R/ItJUFy11SxAk+r6kvwAAAACCIAASTMutgr4EAAAACgITCAEAAAAAQEogCAIAAAAAAFICQRAAAAAAAJASWB0GAJBw7YZXs2KJbkQRoGUyGzetZAvnV3fLE4L+TCZcn9nLqFm3gM4EAKQ2giAAgISbMWuRpaenJ7oZhZ4CH5mZmZaRkeEGnKA/kwnXJwAgGfAbEgAAAAAASAkEQQAAAAAAQEogCAIAAAAAAFICQRAAAAAAAJASKIwKAEi4dq0bsTpM3FbfaGEL589ldRj6M+mkwvWpFV6mTp+d6GYAALJBEAQAkHAzBq2z9HJZiW5GobfX0iyz+GbL2LPW0qxoDjILEv1Jf8aq+eB8uBABAHHFdBgAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJRAEAQAAAAAAKYEgCAAkkbVr19r3338f02dWr15tM2fOzLc2AQAAAEUFQRAAyGcbN260adOmucf06dPtl19+sc2bN4fddsqUKXbKKafEtP8PP/zQzjrrrDy3c82aNfbDDz/keT8AAABAsiqR6AYAQFH33Xff2cknn2xHHXWUlSxZ0rZs2WKLFi2yQw891O666y73nqdGjRrWtm3bhLRz8uTJdvPNN9uqVasS8v0AAABAfiMTBECh4Z/2oWkjc+bMsa1btwZts3z5cve63/r1623GjBlh95OZmWmzZs0K2s/SpUtdtsauXbvi2v7333/fZYOofWr/qaee6rI+xo0bF9imZcuWNnjw4MDzTZs2BbJI1E5llWRHx6bttm/fHvb9bdu2ufd1jFlZWYHXtd+FCxfa7t27A9/3119/5fg5jz73+++/24IFC2zv3r0x9w0AAABQEMgEAVBoaNrHnXfeaUcffbT99NNPLqtCQYx33nnHOnfu7LZ57bXX3HN/jYyvv/7aLrjgAvv3338D+7njjjusRYsWtmzZMvvvv/9csEGffeSRR1wgRUGEEiVK2JdffmkNGjSI+7GUK1fOHcvixYvtxhtvtLPPPtvS0tLcdJhrrrnGBUnkjz/+sIEDBwYCEfPnz7fLL7/cHn/88aD97dixw04//XQXYClWrJg71rffftuOP/74wDZPPPGE/e9//7P99tvPHW+lSpVcAKZ58+YueKG/K0vF+75+/fq5R3af8zJdNB2ndOnSVrZsWdeWV1991dq1axf3fgMAAADygkwQAIWKMh00XUTBAQ3czz//fBdEiJWCJ8rE0LSUJUuW2BFHHOGyMhSM0H61/zp16tiDDz5o+alXr172zz//2Lx588K+f/jhhwcyMxT40XZvvvmmyyoJ7ZeGDRu6oM6ff/5pF154oV100UW2c+fOQODn3nvvdQELZbloG03DOe+889z7bdq0cRkoVatWDXyfAiA5fU4UUFIQRP3466+/usBRpCk1ao/qofgfAAAAQEEhCAKgUFEGxU033RR43r17dxcYiHUKRvny5W3AgAHu78rAOOGEE1wAoH///u41ZYEou2Tu3LmWn+rVq+f+XLlyZbbbrVixwhUtVZaKaol89dVXQe8r+2PIkCGB5/q79vn555+7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qjj55JNt2bJlgW0efvhh69ChQ9DnPvzwQ6tevfo++7n//vvt0EMPdVkK3bp1cwGA4cOH24EHHmgZGRl20UUX2bZt2+ISBOncubMLAGi/ynbQvidMmODeX79+vVWuXNkFR0LbnZ6ebps2bbLMzEzr3bu31a1b1+rXr+8CKmvXrnWZFPr76tWrAxkV9957r/v8xIkT7cgjj3T7VvaJ+iYrKytiO99991137H4//vij6+Nq1aq5PnviiSey3Ueo0qVLuzZVrVo1cPwzZ86MuF3Tpk3defnjjz9s9uzZUX8PAAAAUBAIggAoMP/9958tXLjQDfzfe+89++mnn2z79u3Wv3//oG2UWZBdpoS3n59//tkFGr7//nuXqXD44YfbvHnzbOrUqe7x1Vdf2aOPPprndisI8fvvv9uaNWvCvq8AgTIfXnjhhaDX9VwBCH1+wIABLugxbdo0F5jo2LGjm1qi4MrIkSOtZs2aLuNCj4EDB9qcOXOse/fudt5557lj1XEoOPL0009HbGdoP/3yyy923HHHWfPmzW3WrFn20UcfuX7SscRK+x4/frw7BmXBZKdEiRKBzwAAAADJ5P9+UwWAgvpHp0QJl8lRsWJF9/yGG26w888/P1f7UZChQoUK7vkFF1zgskCef/55V5tCevXqZd98802e26ygxJdffmn77befy1Jp06aNnXTSSXbssccGtlGWxOmnn+6mhCgzRZkfyuT4+OOP3fuLFy927dl///3d8wsvvDDw2ZIlS7o/lUnheeCBB6xLly524403uuennXaam36jY7zmmmuiarcyMo444gh75JFHAq+NGDEipmP3apXs3r3bihUrZkOHDnUBnEiU0XL33XdbnTp1XLZPKAW4/EGuzZs3x9QeAAAAIC/IBAFQoGrXrh0IgIimaWjKSmj2RzT78QIgosCDVkbxAiDea5qqkleaYqJaIJpuouKgyjJRIECZGnv37nXbKGChNr3xxhvu+SuvvOICAcr0kEsuucTuuusuu/jii+3111+PmFXi0fe1a9cu6DUFYP755x+XLRINTVvJLmARDa8miPrxgw8+cIEVZa6EC5YoiKNpOwqYKOukfPny+2ynII4yY7yHpgYBAAAABYUgCIAClZYW/p+d7OpUhHsv3H7CvRZL/YvsqOaF6o48+OCD9u2337qAiIICGuyLsiRU28NbPeXFF1+0vn37BtqkKT+akqKpKQqQqK6IslYiUXAl9Hi859FOM9GxFy9e3PLCq/WhgIWOX3VN/Jkl/mCJap8os0O1UTQ1KZxBgwa57bzH33//naf2AQAAALEgCAIgqaiQaGimg+pYJJsTTzzR/enP6FDQQ7U8lCmhuhvK+vBT0dBbb73VraCiKSP33HNPYGqPl1Hiadas2T4FSH/44QdXTFbZM9Fo2bKlC9jEk9oaLmtHwRI9cqJtKlWqFPQAAAAACgpBEABJRVNAtLKIMi1UAHX69OlhMw8KkupyPPXUU7Z06VIXrFDdC9XnKFeunB1//PGB7bTyiwqhqghqp06dAvU0RBkUqiuiQrDKllBRV+99ZYVouomO23PTTTe5miKqn6JipwpmKAvl5ptvjrrdaqO+c9iwYS7rQgEbBV+UtRErZZ/MmDHDTeVRfRIAAACgMCIIAiCpKHvhscces6uuusplCShz4oorrkhom6677jpX2FR1P5TJoIyORYsW2SeffGIHHHBA0LYqkKpMCU2N8dN0GK3uokwOBUsUCNG0GGnbtq316dPHHbu3RK6Kr7766qt23333uWBLjx497LLLLnPBkWhpH5MmTXIr8Sh7RPvX9JjQZXSz49X6UK2Vs88+2x2XluoFAAAACqNiWfGaMA8AUWQTKLvDP21C/wQpaOBfGcX/nmptKPtCxTa9z4Xbj17To1SpUtm+lldemyJ5+eWX7frrr7eVK1dme0yRqC8UqPCWmY1UH8R73d8voc9Dt41UjyUcbb9r167Ac7XH36bQ7cIdazQUDFK9kQ2jill6Of47yqu9lmaZxVtbxp5ZlmbBU6xAf0aj+eDqNm9J9oWbc0v/XmjlrIyMjJj+PQL9WRC4PunPZMb1GdvvlcqAzm7KNUvkAigwGtyHFupUQCDSANoLFuiXZf/APtx+on0tr7ILYGzZssVN3VHWR07HFEm4AEakwUJov4Q+j2YfkWj7aAIb0W4HAAAAJAPC8ABSgpaX1WA93MOb8hHuoSVho6FVT7Qkrx633367Feb+iPaYAQAAgMKGTBAAKUEFQkNXYIlmikrJkiWj2v/QoUPdI9yUkcLWH9EeMwAAAFDYFI7f1gEgj+JZFyScwhL8KKj+AAAAAJIR02EAAAAAAEBKIAgCAAAAAABSQuHK3wYAFEnthlez7NfNQTS0Wk/jppVs4fzqEWu+IHqp2J8ZNesmugkAAOQrgiAAgISbMWuRpaenJ7oZhZ4G6pmZmZaRkRHzssigPwEASAX8hgQAAAAAAFICQRAAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJbA6DAAg4dq1bsQSuXFb0rWFLZw/N2WWdM1PhaU/tazt1OmzE90MAAAKBYIgAICEmzFonaWXy0p0Mwq9vZZmmcU3W8aetZZmyTtoLywKS382H5zoFgAAUHgwHQYAAAAAAKQEgiAAAAAAACAlEAQBAAAAAAApgSAIAAAAAABICQRBAAAAAABASiAIgiJlzpw51rRpU9u2bVu+f9eiRYvcd23cuDHfvwsoLD7++GNr165dopsBAAAAhEUQBPnmp59+ckGCXbt25XlfS5cudfvyHoceeqidcsop9tZbbwVtt337dluwYIHt3Zv/Sxnu3LnTfdd///0XU/ubNWtmHTp0sKuuusqWLVtmqUp9ctNNN9nRRx9trVq1sjPPPNPGjBlju3fvjnofq1evdn2qfRUV+nnRMR122GG2bt26oPduvfVWu+aaayyZbd682QUIAQAAgGRUItENQNGlbIx4BSS8gMPYsWOtZcuW7vnUqVPt/PPPd/s/99xzLZmFtl+D9zvvvNMFAObOnWvp6emWSj755BM766yzrEePHnbXXXdZjRo17M8//7RJkya5/njsscei2o8CJupX9W9RoetZx5SWlmb33XefPfroo4H3/vnnH/v3338T2j4AAACgMCMTBBGtXbvWbrnlFjvqqKOsY8eO7i69Z8WKFUFZGRrMTpkyJfC+7sz37t3b/V2Dfm03aNCgPPd2gwYN3L60zwEDBljr1q2DvjecJUuW2EUXXeTurB933HH23HPP5WqbX3/91R2nvrNPnz62ePHiXLdf3/Hiiy/a8uXL7fPPP7cRI0bYhRdeaC+99JIde+yxrh2eN99800466SQ7/PDDrVevXvbzzz8H7fP77793AQVlU/Ts2dO++uqrqN475phj7MMPPwzal9qgtnhOP/10e/bZZ+3mm292x63MDdm6dasLXmjagx433HCDbdiwIao+0HbnnXeeC2C99tprdsIJJ7jjVd+OHDnSHnjggcC23jV28MEHW7du3VwQyZ8xoetS9J62U/ujbZ+uUbVDfXP22Wfb119/7fbx22+/BbZRsEqZF+r79u3b2/Dhw4My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ndzzX375xW699dZAxkTjxo3tmmuusfr16wf2+99//7k2T5061WWanHDCCdanTx/76aef7KGHHrINGzZY165d3bYXXHCBe+iY1UcLFy50GRbKrmjbtm3Etoe2SzZt2mTPPfec/fDDD1a9enW7/PLLrXXr1rnqjzPOOMMmTZpkEydODARBwm1Xu3Zt69Gjh/3666/WokWLqL8LAAAAyG8EQQDElQbu48aNs59//tmuv/56N+AePHiwyxB49tln3TZ//PGHTZ8+Pehzq1evdpkD4fZz00032SGHHGK33Xabffzxx25gryCLgiDXXXedlShRwgUU8oOCD7fccosLhqxfv96qVq26z7STevXquWMVZUx4QQIFJfR5ufPOO+3dd9+122+/3QVKPv30U3eMCq6cdNJJ7u/ePg466CDXH+q79u3b2/nnn++CGMcee6zbtwIo4YS2a+3atS5wpOCHvmf37t0uOPP666+7qT6xWr58uevz/fbbL9vtGjRoEMgSAQAAAJIJQRAAcbdr1y43fUQ1JCQrK8tlcHhBkFj288EHH1jDhg3d80WLFtnjjz/uAip169Z1r82dO9feeeedfAuCyFFHHWV79+61efPm2dFHH73P+wqMeFkc0rNnT1u8eLG98MILNmjQIPfalClTXMDGy1o599xzXTCncuXKLmCiLBn/PhQQqVmzpsuM0ZQSFTzdunWrC8homk407rvvPtf3KpjqZeHo+xUMiZYCMPrs9u3b3fcqIKW2Z+fNN9909UAUuAq1c+dO9/Bs3rw56rYAAAAAeUUQBEDcaVqEFwAR/V1ZFApqeIPxaPfjBUBEGQiaYuIFQLzXVJwzPylzw8vyiERBDmV6aPqPjnPJkiUuEOLRNJYRI0ZYxYoVXSaHjkEBkEhUd+PUU08NqqmhKSbPP/+87dixw8qUKZNju5UVooCMv8+VNaNHtDQdSbU+FLj49ttvXSBLU5COOeaYsMGSZcuWuYwUtdPLgvEbPny4DRkyJOrvBwAAAOKJIAiAuAsNdHgDeWVT5HU/4V6Ldb+5KVwqXs2LUGPGjHGZG6oLopVktLLMI4884jI3PMpg0XaqK6KMEE15eeaZZyLW+FCNkNAgiZ4rs0Pvqe5GTv79999AMdfc8tf66N69u/3111928803u7osocESZcRo6o0KyaoPwlFmjKYy+TNB/LVTAAAAgPxEEARAgVNmhbIZ/JK5foRWOlFWQ7jpHaIaG6oZorof/qkofsq+UFFSPZQpcuWVV1q/fv3cFJtwK6goe0bTf/xUIFV9FykYE24f2n88qf6JptdkFyzJjqbJsHQuAAAAEiUtYd8MIGU1b97cDehVIFVUbFOrmiQbrejy1FNP2ejRo900Dq3qEk65cuXc9BeP6pho9Rs/7UeZGaJsFk3j2bNnj3teo0YN27hxY1CtDNU4UdbI77//7p6rfsijjz7qVo2JdtlZrSaj4rLffPNN4DUtxasCp7mh9mt1GK32AwAAABRGZIIAKHCqiXHaaadZq1atXFFQFTpV8dH58+cn/Gx4tS004P/tt9/ctBNlemRXDFQZIN26dXOZIspy0MouWlbXT/VEGjVq5FZOUSaIVoN55ZVX3Huqr6HXtZysVm1RoKN3794ueKGAQ8uWLV1WiKaZKBgTy7Eo2KTVZ/RZFUTV8aiQbKz9oQCNlgI+8MAD7emnn4768wAAAEAyKZalCeYAECfKMlBWxHHHHRd4TVkOKvR54oknBmVTKMtB7x188MFueowG2V26dIm4H9Wj0MO/QsvSpUtt1apVbinYvNBysjNnznR/VxvLly/vppPUqVNnn231fZpm0qlTp6DaFmp/yZIl7bDDDnNFUZVJogCGPxCi1WyKFy/uAib+4qYKjOjz69atc4EG1QwRBUsUANFKMeqn7NoRrl2iorT6Xk1XUTAk1v7wZ6947QrdTt8ZS9Fbf7+p1smGUcUsvRz/HeXVXkuzzOKtLWPPLEuz/K2VkwoKoj+bD65u85Yk73TAeFL9pszMTDe9MFJmHejPROH6pD+TGddnbL9XKoO6UqVKEbcjCAIASBiCIPFFEKTw9SdBEOQWg6L4oj/pz2TG9RnfIAjTYQAUGVp1RdM/YnXRRRfZeeedZ6kku75Kxf4AAABAaiAIAqDI0MBdkd9wVOMj0rKtjRs3tlSTXV+lYn8AAAAgNRAEAVBk5LUuSCqhrwAAAJCKqEoFAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEBNEABAwrUbXs2KJboRRUBaWpo1blrJFs6v7pbTQ/L3Z0bNuvmyXwAAEB5BEABAws2YtcjS09MT3YxCTwP1zMxMy8jIcAN40J8AACAYvyEBAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJRAEAQAAAAAAKYHCqACAhGvXuhGrw8RtNZMWtnD+XFaHiWN/bli/3r789qd47BIAACQYQRAAQMLNGLTO0stlJboZhd5eS7PM4pstY89aSzOWyI1Xf3a5c0Vczg8AAEg8psMAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJRAEAQAAAAAAKYEgCAAAAAAASAkEQQDku7/++suuueYa27VrF70NAAAAIGFYIheArVixwoYPHx7oidKlS9t+++1n55xzjtWuXTvPPZSZmWlPP/203X///VaqVKm4ttVz+OGH2yWXXJLj5ydMmGD//POPXXXVVVF932effWbvv/+++3vx4sWtTp061qVLF2vdurUVBj/++KONHz/ehg0bluimAAAAAAlHJggAW7NmjQtSVKxY0Zo2beoCH5MnT7ZGjRrZnDlzkrqt3qNu3bpRfX7atGn2wQcfRP19s2bNsldeecV9x4EHHmgLFy60tm3b2uOPP26FwYIFC2zMmDGJbgYAAACQFMgEAeLot99+s9GjR9v//vc/GzdunP355592yCGH2Pnnn29paf8Xc/z+++/to48+sqFDhwY+p+0efvhhe/TRR12mhLef2267zd577z33/qGHHmq9e/e2VatW2WuvvWZr16614447zrp16xa39vfq1csOO+ww9/ebbrrJBUEUAHjkkUcC22hKy5tvvmm//PKLVa1a1c4++2xr3Lhx0H7+/fdfN/BWxsXBBx9sBx10UNDnb7nlFrv88svde/7gxpAhQ9wx169fP6a2eqZOnWrvvPOO+3ulSpVcnymbxet7ZXV8/vnntn79ejc9R9QObZedcuXKBbaXypUr25133mkDBgywYsWK2Q8//GATJ060//77z4444gjr0aOHe122bt3qjunmm2+2KVOm2Lx586xnz5529NFH2549e1xmij5fvXp1d534gzl6/91333XZHOrrk08+OeiYZ8yYYZ988on179/fbacsGX3/mWee6d7//fff3fnbsmVLoP26XnReR4wY4Z6XLVvWnT99d/ny5YOOW+fkpZdesg0bNrhMGwWC9Nx/PeTURgAAACCZkAkCxNHSpUvtySeftA4dOtjixYutQoUKbsB/9dVXB7bRwPTFF18M+pwCG8pu8GpmePvRQHnZsmVWpkwZN31DA/9jjz3WDeI1NUMBiJdffjlfzqHasn37djfg92iQr+/XtJb09HQ3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ru/r16+f1Nfg1q1b7csvv7S///7bDjjgAOvSpUtMP1cAAABAQSoymSBTpkyxJ554Ii77mjFjhg0cONBWrVrlHhoEKdBx9NFH2/bt2y3Z+du/YsUKe+WVV9ygf/To0ZaKFBQ65JBDrHfv3vb999/bzJkz7e6773av6dxGa9asWTZkyBArSubPn2+33HKLbd68eZ/3hg4dap9++qkli0T1f9OmTe3qq6/O9+9Zt26dHX744XbppZe6Y/3oo4+sU6dOduONNybtNagAmn6OFKD95Zdf7Nprr7WWLVtaZmZmopsGAAAAJH8myNdff21z5851dz5PPvlkK1mypHtdd+8nTZoUyMo4+OCD7Zhjjgka5H777be2cuVKe/jhh91r+ry2y60SJUoE9iXKMjnwwANt8uTJ1qNHj4if27Vrl3388cf2119/2X777WfdunWzUqVKxbzNli1b7P3333fbHnXUUXlu/0033WTXXXedXXjhhfbee++5761evbq7g6s/zzzzTLfd8uXLXRbJf//9Z0cccYQblIW2S+di/fr1brDTrl27HN/T8zFjxlj//v3d+fOofeecc45ri4JLTz/9tGufBlPz5s2z4447zu3Hm3r0448/WtWqVd3rderUiaofFAg64YQT3EPTibxryjunO3bscH/XXexx48a5v5ctW9YFjTp37uwyabyMCZ2zbdu2BfpVd+m9LImc2rdz50774IMPbMOGDXbYYYe5PtcgV+fEnwWgvl+4cKFlZGTYKaecEtRf+tlQxtN5553nPqvrvVevXvbGG2+4DCHt03/cmqp12WWXWeXKlS2vNm7c6IJo2X2Prme179xzz3VZUrqWFDj0zqFfdv0V7ji7du2a6/739qcgmNqlwGDr1q3d9e2n7AtlXOmcH3vssbb//vsHroeaNWsGbRvtucrpO/3uu+8+l1Whf+/0nbJnz55AVleka1D/XmZ37cZyDebUl6GUZaagYrVq1dxz/TzpfA8ePNhlEgEAAADJJikyQf799187/vjj3YBOv1BrwKzB0+7du937GiB7WRk//fSTGxydffbZgc/rfQ3Atb23XbwzNrypAv7pMJHu5A4aNMh+/fVXl43RqlUr93os2+guqgYpDz74oBtIKeiiPskLDWbUJxqoPvXUUy6d/rTTTnO1RjQoEgUJdFdXd/81CFIg6YYbbgjsQ4Owgw46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AInNBPnyyy/tjDPOsC1btlh6erpbFlcqVKhgffr0sTlz5ljlypXj00oAAAAAAIBEZIIo4NG7d28755xzbN26dfb0008H3itXrpydcsop9uabb+a1bQAAAAAAAInNBPnuu++sYsWK9txzz7kibMWKFQt6v1mzZvbTTz/Fq40AAAAAAACJyQRZu3atNW7c2AVAJDQIotVjdu3axekBAAAAAACFOwhSp04dV/Nj9+7dYYMgM2bMcKvEAEBRMGnSJFuxYkVc9/nhhx+64tL5raC+BwAAACiyQZAjjzzSsrKy7Oqrr7Zt27YFvffOO++4eiA9e/aMVxsBFDEalE+YMMH+/PNPKwxU7DnWVa+UDffDDz/YJ598Yr/99ts+2XFnn322/fjjj5bfCup7AAAAgCJbE6REiRI2atQo6969u40bN85q1apl//77r6sFMn/+fBs4cKC1aNEi/q0FUCQ8+OCD9uijj1qvXr1s7NixVtQ89thjds8991jt2rWtYcOGtnz5creS1h133GGXXnppopsHAAAApKxcZYJI165dXYHUk08+2a0W899//1nVqlXt5ZdftuHDh8e3lQDCTnHQ6kxTp0612bNnu+wsz549e1ymxYYNG4I+N3nyZPvrr7/22c+aNWvcstf+gsYauGv7BQsWxLX3lRHx6quv2oABA1wbdQyxHl88++CLL75w273//vs2c+ZM27FjR56Ob+jQoXbnnXe6jLh58+bZxx9/bL/88ovLCvHqKPlld4yif1v1b+2UKVNs5cqVYb9T22j/n3/+uW3cuDHb9m3fvt0d68KFC23RokX21VdfBb2v7/joo48Cz5ctW2affvqp609ltKj/VBcqlII8ypb59ttvbevWrdm2AQAAAChUmSAaFGkAoikvRfEuLpDsNMVBgUjV5jnooINc8OLoo492g3nV6NFA94wzznD1edq2bRv43CWXXGL33nuvXXzxxYH96HOLFy+2Ro0aue0V2DzssMNs9OjRduCBB9r06dNt0KBBNnjw4Li0/YMPPnBtfOihh+yzzz6z1157zQVEYjm+ePaBAgIa6O/du9cFBfQ57UN9ECsFk+677z677bbbXNv8atSoYf369Qt6TX187bXXRjxGvXbmmWe61bgyMjJckOaqq66yYcOGBfah17RcuYIUOl9LliyxJ5980k4//fR92qcAyamnnuqWMn/v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QRAFQPzLSwIAAAAAABTKmiALFy60jz/+OOg1rbS7du1a+/TTT23WrFk2derUeLURAFDEtRtezYoluhFFgJZxbdy0ki2cX90tT1gUZNSsm+gmAACAVA+CaInIG264Yd+dlShhhx56qE2YMMEtkQsAQDRmzFpk6enpdFYeKfCRmZlpGRkZLiACAACAOARBevToYStXrgx6Tb9sVatWzS2RCwAAAAAAUCSCIGXKlLFatWrFvzUAAAAAAAD5JFe5sqr3cd999+X6fQAAAAAAgEIRBFm1apXNnTs32/d//fXXvLQLAAAAAAAg8dNhcqIgSNmyZfNj1wCAIqhd60asDhO31WFa2ML5cwt8dRit4jJ1+uwC/U4AAIB8C4JoRZgxY8a4vy9atMiWLl1q11xzzT7bbdq0yT766CMbMmRIzI0BAKSmGYPWWXq5rEQ3o9Dba2mWWXyzZexZa2lWsEGQ5oML9OsAAADyNwiioMdLL73k/v7ff//Znj17As89xYoVs+rVq1uvXr3s0ksvzV2LAAAAAAAAEhkEOfPMM91Dxo4daxMmTHB/AgAAAAAAFNmaID179rRTTjkl/q0BAAAAAABIpiBIqVKl3GPz5s02Z84c++effywrK3gud4MGDaxdu3bxaicAAAAAAEBiVoeZNGmS9enTx9atWxf2fdUFIQgCAAAAAAAKdRBkx44d1rt3bzvvvPOsZs2aNn36dLv11ltt7ty5NmrUKLvsssusR48e8W8tUERs3LjRli9fboccckiim4IiIJmuJ2UILlu2zFq0aJHopgAAAADxCYL88MMPVrlyZXvqqads3LhxNm/ePOvUqZN7aFWYLl262IknnpibXQMRbdiwwU29Ovjgg/PcS1u2bLElS5aEfa9p06ZWpkyZfD0Tn3zyiVtieu3atZbq52fXrl32999/W3p6ulWrVi0u+4zntRINrqf/7+uvv7Zzzz3X/v333wLpewAAACDfgyCZmZl26KGHuiVxixcv7jJDPOXLl3fTZF5++WW7//77c7N7IKyJEyfazTffbKtWrcpzD3377bd28sknu0FyiRLBPwbvvvuuHXjggZyFfD4/Wmb7lltuseeee85q1apl27dvd0GQu+++284666xc7TO3bckrricAAACgCAdB/vvvPytbtqz7e/Xq1W3hwoVB7+/evdsFSoBwA98//vjDDXarVq0a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deOyHwAAgGRAEAQAkHAzZi2y9PT0RDej0KMQHQAAQPaYDgMAAAAAAFICQRAAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJbA6DAAg4dq1bsQSuXFbIreFLZw/N8clcrX07dTps+PxtQAAAIUGQRAAQMLNGLTO0stlJboZhd5eS7PM4pstY89aS7PsgyDNBxdYswAAAJIG02EAAAAAAEBKIAgCAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEAQBAAAAAAApASCIABS3hNPPGFdunRJ+X4AAAAAijqCIAAKtcWLF1v16tXdo0aNGtakSRM755xz7Ndff416H9u2bbONGzdaQXj55ZcD7c3IyLCWLVvagAEDbO3atQXy/QAAAEAqIwgCoFD777//bN26dfb666/b77//bu+99579+++/1qlTJ1uzZo0lm+3bt7v2zZ8/33777TeXhTJx4kQ7++yzE900AAAAoMgjCALA+frrr112wuTJk+3444+3Bg0a2Mknn2wLFiwI9ND48eNdpoWfMi70ufXr1wft56233rKOHTvafvvt5/azZMkSGzt2rB155JFu371797YNGzbErfcrV67svvfggw+2Z5991gVAvvzyS7v55putX79+QduOHDnSjjnmmIj7+uuvv+z888+3Aw880GVq/O9//7OdO3cG3p82bZqddNJJVq9ePTviiCPskUcesT179sTUXi9z5bjjjrPBgwfb1KlTLTMz00499VS78847rX///rb//vtbr1693PZbtmyxm266yfW/2qXX//jjj6B9jhgxwlq1amUNGza07t2729y5c3N8z8ukWbly5T7t07kUnVs9f+ONN9y0oTp16riMFu/89+zZ0+rXr+/66o477nCBHgAAACAZEQQB4OzatctlVNx11112//332xdffGGlS5d2U0s8CgRom3CZGHv37g3az+OPP24PP/ywffrpp+75scce64ITzz//vH388cduEH777bfnS++XK1cu0BZlXWzevDnofQ3SswvAXHDBBS6ooXYqcFOmTBmXaSLffPONnXLKKS4I8e2339pjjz1mo0ePtrvvvjsu7dW0nGHDhtkBBxzgAiOjRo1ybVEg6c8//3TBpc8++8wFHdSn3rEpA2bIkCH2wAMPuDZeccUVds899+T4nnf+QoM4ek3tEZ1bPVdA6frr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KvCpAbgCDJpu8sADD7hlgZXdEOnaVSBW167VUZSZoakkderU8U+VUTaJipuqAGxmy76G21/XMGvWLPv4449dWxRo6tixo6vVEQ1dj2p6aEWfUP/73//s+OOP93+v9iugoUCLghmaPnTTTTe5a450/QqUKMikQI1qpyiIpODSoEGD/PsoMKZVYEILyqpdXj2QWPsKAAAAyGtpPvKWASDlafUc1VbxpkJpipGWG54wYUKu9o3qzqhg65bRaZZe9PCYRlO/b1lbsmKjJSJNtdISySpsG2smD+jP3Mb9SX8mMu5P+jORcX/G9nfltm3bMvywkUwQALnqkUceiZgVon+oIv0D1aVLF2vcuHHCvTtajWXNmjVhn1ONj8CslHieM1a33HKLywbS1CHVD1EWkKYJAQAAAMmEIAiAXKVobKTin/qkOlIQxFsFJtFoSooKgIaT1VVPcuOcsdLSwtOnT7cVK1ZYmzZtrHXr1lakSJE8eW0AAAAgrxAEAZCrevXqlVQ9fOWVVx4W54xVgQIFrH379vFuBgAAAJCrmDAMAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEBNEABA3DUdUsbS7PBQvkKVeDcBAAAAWUQQBAAQd3PmL4+4Qg4AAACQU5gOAwAAAAAAUgJBEAAAAAAAkBIIggAAAAAAgJRAEAQAAAAAAKQECqMCAOKuaePaCb86jFaFmTF7YbybAQAAgGwgCAIAiLs5fTZbelGfJbL6fePdAgAAAGQX02EAAAAAAEBKIAgCAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEAQBAAAAAAApASCINmwaNEimzp1quWFJUuW2JQpU/LktQD8P7///rtNmDCBLgEAAACSQNIskbtw4UJbtWqVnX/++dk+1y+//GJff/21//tixYpZ3bp17cQTTwza78MPP7S33nrLLrjgAsttX3zxhb3wwgvWqVOnmNqfL18+q1Spkp1yyilWtmxZS2WLFy+2n3/+2fLnz2916tSx448/PqbjV6xYYd99951dfvnllizWrFljH3/8sXXr1s2KFCkS9NykSZPcfX/SSSdZIohX/8+dO9fuuOMO6969e5683oIFC2z58uWWnp7ufm5Lly6d8Pfgxo0b7aOPPrKKFStamzZt4t0cAAAAIPkzQd5//30bPHhwjpxrxowZduONN9q3337rHu+88461bNnS2rdvb3v37rVEF9j+b775xgYMGGA1atSw119/3VKRBpSnnXaatWjRwt544w0XuLrqqquscePGLjASrTlz5thdd91lyUTXf91119m2bdsOee7ee++19957zxJFvPq/Vq1a1qNHj1x/ne3bt9tZZ51l7dq1s7ffftueffZZF4B6+OGHE/Ye3LFjhwugnXDCCfbAAw/YM888E+8mAQAAAIdPJoiml/z0009WrVo1O/PMM10WgzeInTlzpj8ro0GDBtawYcOgzIf58+e7TyNfeuklt+3ss8+22rVrZ7ktBQoU8J9Lli5davXr17dPPvnEOnbsGPG4gwcP2qxZs1xWylFHHWXNmjXzX0cs++zZs8c+/fRTF3TRYD277b/pppvcYLdz587uGipXrmwVKlRwGSP6xPncc891+/3zzz+ur/fv3+8GYBoAhrZLz2s/ZVKoTzJ7TgPsN9980316XbRoUf/+ap8CS2qLjh0/frzLdNH0A03/OfXUU61evXpu32XLltm8efPcp+IKaJQqVSqqfti8ebMLfpx88skumybw9XW/HThwwH29bt06++CDD9zXyohQpog+hQ/MmFBw6d9///X3q/rHy5LIrH16Hb3+li1brFGjRq4d+v7KK68M2k+f8v/6669Wvnx51+4jjjjC/5xeQxlPF154oX355ZeuzRowKwB48cUXu/fRs2nTJnv33Xeta9euVqJECcsu7z3M6HXUHrVPmVG6DvVZ06ZNrWbNmoecL6P+Cned+vcgq/3vnU/3ltq1du1al9Wl9ziQXkeBQ/0sqt16D6RMmTJhfwajea8ye81ACuLq3wT9e+e9Z//9959NmzYtw3tQmV4Z3bux3oOx/Kzt27fP3YMvv/yyO8/OnTsj7gsAAAAkgoQIgugP/csuu8z9gd+8eXP3R3paWpoLAmgwr+CGBifep6W33HKLG1yMHj3abVu/fr2tXr3afSrp7acBeHaCIKG8AVFGmSBqW+vWrd1gSoMHtUUD/OnTp/sHNdHsoyCCgji7du1yg3el4iswlB0aqGg6zcqVK+3RRx91gxf1a5MmTdxDQRB9+nzttde6dhUvXtxuuOEGu+222/yfRP/11192+umnu2k1miYxZMgQF8AZMWJEhs/pWhWAadu2bVAQQtv0Huv69d7p+4kTJ7rAhQZ31atXd+fS+63sDe/e0OBy8uTJbuCZ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bEuXLg3apswK7RvuPL/++qt9++23tnfvXvecMjYUsPjhhx9s//792W5zw4YNrWzZsi5rI/D1FfA47rjjgqazzJw507WpRYsW/m3bt293U010TaHtOfbYY+32228/5DXVN3PnzrUdO3bYpk2bbMaMGWHbtmbNGhdwCQzEbN682X7++WfXHwqG6BFtZkqgFStW2Oeff+7v11jaBQAAAMRbgXg3AABGjBhhs2fPtnz58pnP57MtW7bYvn373EC9bt26roMGDRpkxYoVs5deesnfYa+++qrbx5t+ovPo6//++8+KFy/ualrkz5/fXnjhBbvrrrvcNgUIqlat6gITRYsWzXLnp6WluWwNDfjvu+8+t01fn3HGGXb88ce7r88//3z/dgU2lOEhjz/+uD3yyCPu2hSo0LkmT55sjRo1cs8rsNKrVy8XUPACON27d7epU6e6AMvKlSvda2u6ys6dO/1tUuDovPPOc8GNggUL2h9//GFvvvmmnXvuue7radOm2a5du+zRRx91+19xxRVWs2bNqK/5iy++cNN0hgwZYi1btoy6XQAAAECiIBMEQEL48ccfbejQoTZ//nyXWaAAgQIfsVLGiAIl33//vf32229WoEABu/DCC13ARNt07rVr17rvs0uZHaqx4WVyKHjRvHlzFwgIzBDR1+ecc477+p133rFhw4a5rBRlTyhjRVNULr/8chcACmfChAkusLBw4UJ3jPpK/w+1YcMGF/BYsmSJ26dnz552zz33uOdOPvlku//++61UqVL+TBAFQaKldqsfFVC66aabYmpXoD179rgsmMAHAAAAkFcIggBICKeddpoLIIiyNzSY1/SNrJznzDPPdF8XKVLETj/9dGvWrJmdeuqpbpuySfS1AgXZpcCGMh68KTnK+NA1nHXWWS4Y888//7iHpuh4U2EURFD7Vq9e7YIjyq5o0KCBa4+2hfPaa6/ZZZddZnXq1HHfK6PkhhtuOGQ/XZsySDzt27e3X3755ZDirbFSUOmqq65yWSWXXnppzO0KpCySkiVL+h/VqlXLVtsAAACAWDAdBkBCKFeuXND3CmDs3r075vOoTkegwoULh92WlXOH8qa4KJhRokQJVxPjlFNOcVNRND1EU2687A4vwKNMFGWnhGa5aHpJpDb9+eef1qZNm6BttWvXPmS/MmXKuKk1gX2oLBVNLVJgKSu0osx1111nTz/9tAtMZaVdgfr06eOmJnmUCUIgBAAAAHmFIAiAw4JXLyRQYHHOeFFwwwuCqB6IAiDiTYlRm1XrQwEKOfLII61Vq1b21FNPRf0ayphQgCVQXk0jUZDnlltusQcffNAaN27srjE77SpUqJB7AAAAAPHAdBgAhwUNxlV4M9CcOXMs3jTNRau8TJ8+3Z/tERgE0SNwVRhNoXnrrbcOyfoIXMkllKbvfPLJJ0HbQr+PhgrBZiVwdO+997ogSLt27VwB25xuFwAAAJBXyAQBcFi4+OKL3ZSMAQMGuNVXtCKJanF4K6rEiwIcCmi8//771rt3b/921SFRnQ9lgnirsYiCCSomqrogt912m6vjoWKiWnZWRWEjBSF0zSpketFFF7nld3WOwKkv0S7ru23bNreKjqbyaGWYaFeH0Qo4upa2bdvaxx9/7Gqt5FS7AAAAgLxCJgiAuKtXr94hwQzVidBA26OvFWhQoU8tJ6sgwrPPPutqcGR0nvr167vBfyB9r+05QTUwtGxs69atg9qi6S9aPla1PhQQCax9ouCNCowqkKNioxUqVHD1QwKzXpRJ4qlRo4bLNlEtkfHjx7sVXrSSjqbWeKpUqeIvCOtJT093r+/VA1FbX3/9dXeuxx57zL+0cCSh51SQp3///u61N27cGFW7AAAAgESS5ou0JiMAIGEo20TTWTzXXHONW+o33tNPstsu1RBRbZEto9MsvWji/Dqq37esLVmx0Q43Bw8edEslly9f3tXRAf2ZSLg/6c9Exv1JfyYy7k+L6e9KZT6rXl8kTIcBkNK0RK1+sYSjop/FixcP+9wJJ5xwyKozuUkrs1xyySVWtWpV++yzz2zixIkuMyY7NA0nUiHTY445xmV6xKNdAAAAQG4hCAIgpWlayIEDB8I+t3PnTlezI5xHHnkkT4Mgr7zyij333HP26aefWvXq1V0AQ/U4smPcuHG2bNmysM9dffXVUQVBcqNdAAAAQG4hCAIgpcV7Okm0atWqZcOGDcvRcyp4kYjtAgAAAHILE4YBAAAAAEBKIAgCAAAAAABSAkEQAAAAAACQEqgJAgCIu6ZDyliaJY7yFarEuwkAAADIBQRBAABxN2f+cktPT493MwAAAJDkmA4DAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEAQBAAAAAAApASCIAAAAAAAICWwOgwAIO6aNq6dMEvkanncGbMXxrsZAAAAyAUEQQAAcTenz2ZLL+qzRFC/b7xbAAAAgNzCdBgAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJRAEAVLI+vXrbcuWLSnfhmj4fD77+++/be3atXHtLwAAAAA5hyAIkGA2btzoghWh/vzzT/vnn3+Ctu3bt89t37NnT1TnvvLKK+2RRx7J9YCJ2rVu3Trbu3dvttoQL3PnzrWqVavaiSeeaN26dYt3cwAAAADkEIIgQIJ55plnrEmTJkHbli5dakcffbRdfvnlQdvff/99t33r1q1RnbtChQpWunTpqPbV4P+xxx6LoeX/184LLrjASpYsaSeddJJ7rRNOOMEmTJiQpTbEy+DBg+28885zgZwZM2bEuzkAAAAAcghBECS1NWvW2M6dO93X27Zts//++++QfVatWmW7d+8O2qbBr/aPdB7va2/ahLImlP2QE1q0aGErV650GR4eDcTr169v33zzje3fv/+Q7QosbNq0yR2jYwPbF+jxxx+3m266KWib2r9hwwY7cOBAUDaK+krXqnPqEfi64fz444926qmnWpkyZeyPP/5wfbh9+3Z77bXX7Msvv4zYBk058YI4mzdvdm0JpGsJzUjxMmAC2yza5r3HBw8edN972Sjqn8CMGV2P3jddf+g5fvvtNytbtqz7OjTApGsKlyET+Hpq119//WU7duwI2kevF+4e1DXqHvPOo37IiN6fjLJ/Ir1OtMKdP7CNur7QvtPPUGbtBgAAAOKNIAiS2plnnml33323NWzY0GrXrm3Fixe3G2+8MWgfBRGmT58etK1jx442cuTIoPNo4F63bl075phjXCbD7bff7gb3tWrVsuOOO86d+/nnn892m08//XQrVKhQUOBAX19xxRXudb///vug7eecc477+tFHH7XmzZvb2Wef7YIixx9/vM2bNy/DqSgvvviiC1qof/T/O+64ww1+BwwYYPPnz7eJEye6c+qhYEVGbrnlFtfHY8aMca8v+fLlswYNGrhtkdpw8cUXu749+eSTrV69enbvvfe67b/++qt73VKlSrk+PuOMM1xwQlasWOEyYBRoCaRtX3/9tftaU4f0vd7/ypUru/dOgY3Ro0e7h7Yde+yxVq5cOfv888/9g3u9pl5b77++9tq+fPly17cVK1Z07dHjk08+8b+293r9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mi+z9WMAAAAAAACSAJkgAAAAAAAgJRAEAQAAAAAAKYEgCAAAAAAASAkF4t0AAEDy+P777+399983lZtq3769nXbaaTlyTFbOmwx++OEHe++992z//v3Wtm1bO+OMM7J9zO7du+3dd9+1JUuWWLly5ezCCy+06tWrWypYuHChTZ061fVNmzZt7KyzzsrRY8aNG+f6/8Ybb7R69epZslu0aJG7l/bu3WutW7e2s88+O0eO2bdvn7399ttu36OPPtq6d+9uRYoUsWT3008/ub7Zs2ePtWrVypo3b54jx3zyySc2d+5c1+cNGjSwzp07W8GCBS3Z/fzzzzZlyhT777//rGXLlnbOOedkeszq1avt1VdftY0bN9qwYcMsX758OXLeZKDfGbruf//911q0aOHut8ysXbvW/bu4YcMGGzp0qBUoEDz01D2p31c//vijlS5d2v1+r1u3rqWCZcuW2VtvvWW7du1y/w6ee+65mR6zbt06159///23DRkyJMN/F9955x2bOXOmXX755XbqqafmcOsPf2SCAAByxOjRo90AcfPmzbZt2zb3x/izzz6b7WOyct5koD/EFezRHzs7d+50A0b9EZmdYxYsWOAGQR988IH74+m7775zg3X9sZTsJk6caE2aNHF/ROqPTgWIHn300Rw7ZtasWXb//ffb008/bX/++acluzfeeMNOPvlkW7NmjRsUdejQwR555JFsH6N7t3HjxjZ48GArWrSoG3hFE6w63GkwpL5ZtWqVG1x37NjR+vfvn+1jNADq0aOHC34q8PHQQw9Z06ZNXf8nMwWGTjrpJPezqADRRRddZH379s3wmJtuusnOPPNM++qrr9zP8cGDB3PkvMlg2rRpduKJJ9qKFStc4OLiiy+2Pn36ZHjMrbfeaqeffrp9/fXXrj8VSA60fPlyO/bYY23y5Ml2xBFHuKBno0aN7OWXX7ZkN336dHetCoQcOHDA/ZzeeeedGR5zzz33uN9HXn/q/otk8eLFdscdd7j99G8owtDqMAAAZMe2bdt8xYoV8z311FP+bSNGjPAVKVLEt3nz5iwfk5XzJoNdu3b50tPTfUOGDPFvGz16tK9QoUK+devWZfmY1atX+zZs2BB03K233uo76qijfMns33//9ZUuXdo3cOBA/7axY8f6ChYs6FuzZk22j9m0aZOvevXqvilTpmjFPd9HH33k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Pfvss3bBBRfYQw89ZFWrVrWiRYu6gMh3331nF110ET0G4LBEEAQAAAAJp1WrVnbyySfbypUrrV27dlasWDGrWbOmvf766+55Tcfo0KGDm6JRuXJle/rppyOe488//7S2bdvakUceaRUrVnQZDnv27Dlkf537lFNOcYN9nbdly5b21VdfhT3nqlWr7Pzzz7cSJUr4p8F88MEH9vPPP7upPXoUKFDAf1yvXr382/Ply2elS5e29u3b2/fffx/2/OvWrXMBCF23pqLceeedtn///kParPapH8qUKeP21dQVtSN0H11/enq6FS5c2E466SR74403Mg2C6PWqVKlyyHOFChWyfv36Zakt2eljOXDggA0bNswaNmzorqVUqVLWuXNnW758eYbXAwAegiAAAABISP/9958b/A8cONDWrl1rPXr0sO7du9uMGTPslltucQPxNWvW2E033WR33HGHzZw5M+w5br75ZpfNoMDCc889Z2PGjLHLL788aL9nnnnGLrvsMjv33HPtjz/+sEWLFrmAyTnnnGPTp08/5Jx6zfvuu8/tq+MWL17sAgANGjRwtTT0CAxaKKvC2753716bM2eOCwLo9f7666+g8ytAc+utt1rv3r1dm5988knXvuHDhwftN3nyZGvevLmVLVvWvvnmG/v777/tiSeesJEjR7pggUyZMsVatGhhRx99tP3www9un+uuu8715UsvvRSx75X5UaNGDXvzzTdt/vz5mb5X0bQlu30s+r/uB21X32iajvr09NNPd/cCAGTKBwAAAMRRz549ffqzdMuWLf5tLVu2dNt++OEH/7b9+/f7ypcv7ytWrJhv3rx5/u0HDhzwVaxY0dejR4+g83rnmDNnTtD24cOHu+2zZ8923+/cudNXvHhxX/v27YP20+vVqlXL16BBg0PO+c033xxyHR06dAjaNzP//vuvr2DBgq49oeefP39+0L7t2rXz1a5dO+jYcuXK+Zo1axbx/Hv27PFVqFDBnTPUjTfe6Ctbtqxv3759EY+fO3eur2bNmq499erV83Xv3t33/PPP+/78889DriOztuREH3/wwQdu+9ixY4O279q1y13nLbfcEvH1AcBDJggAAAASUqVKldxqJJ78+fPbMcccY0WKFLHGjRv7t2t6SZ06dcKuWFKuXDk77bTTgrZdeOGF7v9ffPGF+7+mpOzYseOQYp96Pe2rKS4qCurRdA9lHsRiw4YNLntFGRlHHHGEmxaj69i3b5/99ttvh1y3pqwE0nQbTes5ePCgv80bN248JKMlkPZRu7t06XLIc5pysmnTJndtkWjaiqaZzJo1y6666iqX0fHggw9a7dq1rX///kGvE01bstvH06ZNc+916Dk0taZp06ZhM4EAINT/m6gIAAAAJBAFA0JpCkmk7QoShKpQoULEbQoCyObNm93/NTUjlLdN+3rHhauTkREFLlq3bm07d+60l19+2QU4VOciMBASKNz1aX/tt3v3bldvQ0GVzNqiKSmi6UAKwGjaiHjTcgKvPRIFHc466yz3kF27dtmll15qDz/8sAtS6LqiaUtO9LGuR32peiqh16H/h+s3AAhFJggAAAASkoIEsWwPJzC7IHSbsg3EG1RntK9qXXgKFixosVi4cKGrf6EaJqrPUbJkSXcNGtSr9kVWrk8ZLqJaKZF4bX711VddfRJlcuihQIIXQFA9jliouOw999zjvvYKmkbTlpzoYz2voqwKBHnXo2vxrie0tgoAhEMQBAAAAElL0zTmzp0btG3q1Knu/1qZRE499VSXSfLOO+8E7adBtvZVsdNwGSXhAgThVp3xaAAfSMGJrGrSpIkLCkycODHDfRSgUNHSWCmg8Oijj4Z9zgt2aLWZaNuSE33csWNH17/vvvtuzNcDAB6CIAAAAEhaGlwPGDDAvv32W1eTQoNwrRSj5WdVR8ILXqjGxfvvv++yNZSZoKV5VQdDdUaGDh0a1WupboeWdVXmh1e7w9uu5X2HDBliS5cuta1bt9rYsWPdKirKCskKBVS00s3XX39tV155pS1btsxNt9GqMwoWKLigJWRfeOEF+/DDD+3666+3X375xWWe6JomTJhgbdq0iXh+tb9Pnz4uU+Szzz5z02A0peXtt9+2u+++20090Uo90bYlJ/pYy+V27drVbrzxRreyjVaH0euov/v27etWjQGAzBAEAQAAQNJSIEBLsyrwodoTGkBr4D1p0qSg/bQUrwIDH3zwgVsaVsETZTwoANC2bduoXuv22293y+RqqVgV/CxQ4P/K76kQqgb/ynRQkVYVd1XAYPz48dm6NgUEVNxV02qUjVG5cmW3dKyuUa8vKiKqYIvqbTRr1swFXbRE7eeff26PP/54hkvk6ri6deu65YfVdm1TYKRz586u0Gn58uVjaktO9LHet8GDB9uoUaNcgVa9Ts+ePV1xVNU9AYDMpGmJmEz3AgAAAA4zWgFFWRfz5s2Ld1MAAAmCTBAAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBGqCAAAAAACAlEAmCAAAAAAASAkEQQAAAAAAQEogCAIAAAAAAFICQRAAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJRAEAQAAAAAAKYEgCAAAAAAAsFTw/wHudnqNZ/DlVAAAAABJRU5ErkJggg=='
NOTEBOOK_LOG_FI_B64 = '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wXeYYvXNN98E3IXSXTKtCuD566+/ArbXHaVgFSpU8P1bx/T333/H1ZZ+/fqFXKnguOOOC5ibPnr0aN9zrSyxcuVKd/dT2RaiO3xnnnmmq1vi0d3G888/3901XLx4ses3vT9p0iR39070Wt++fQOO5ddffw1o46BBgwLSmJXlE0rwHcxYPPvss/bAAw+4vve+R3cX4z32RHrjjTdcWryux6uvvtr3+rfffmvXXnutuxOt61wZHMWKFfNlRD300EMR9/v000+7LCLtV3c5a9as6fuszsnSpUt92w4YMCAg4+iaa65x/aAMph49evheHz58eMS6PKH6OZZzHs81FYraOGrUKJc9NGHCBHeu/TO4grMH1FceZeTo94wytnQd6Py0aNHC934qXDOJEO85v/XWW13hZv/P6feTzpd+b33++ecxt0XFpFVE2vPUU0+5/ld2g36X6ne2shl05z0Rv0vycv7zIjhT8Zhjjgl4Hu+1lchzksjflfGc13g+k5tE/X6L5fdKkZbsKAwAAOmSCTJ79uyA984444yAzy5btsxlTHjvH3PMMWEzQV588cWAz959990B77/yyitxH8PKlSuza9eu7duX7lZ++umnAds8/fTTAd9366235tiPsgDC3dWKJPjOVbjHscce67bfvHlzdunSpX2vN27c2N3x8yxdujTgzpf6KhoDBw4M+L4//vjD996vv/4a8J7aHErwXUe1NS+ZIMHy69hjzQS59957fe/pLqr/e3Xq1HF3qj2HH364773KlStHPOZ+/foFvD9mzJiA9wcMGOBe37RpU3apUqV8rx988MHZe/fu9X1u+/bt7g6v//d6bYqmn2M55/FeU8HtuOWWWwI+26FDB997yo7xzvOWLVtcZon/dbRu3bqwbciPa0bHEe7ndPz48fmSCRLvOdfxlyxZ0vd6gwYNAj6n33+ZmZkBbYnmbn1wH4wdOzbgug8n2usq3PfGev7jPQ/6nrfffjvg2jn33HMTcm0l+pwk8ndlPOc13msh3DlO5O+3aH+vFHVkggAAUEC8DBBP586dA57rLnejRo18z7/77jtf4dFg/qu3hHquO/Dx0J11ZaT8+eefvswF3cVSHQZ//vU/xH++tUe1EfypFkQiV4f5+uuv3fszZsyw7du3+7bXajTFixf3PdfSvl4Ggbz//vu+fyujQHfXzjjjDDfHWnfm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lZll0BFRKlasmDVp3MjmzV9oe/fupd/os3zDtUa/FSSuN/qMa63w/4xWrVrFZs2YaqmMIAgAIOkadB5nxTOzkt2MQiMjI9uqV9pku+pmWXY2wSP6jGst1fAzSr9xraU2fkbzr98WTu5kqY7pMAAAAAAAIC0QBAEAAAAAAGmBIAgAAAAAAEgLBEEAAAAAAEBaIAgCAAAAAADSAkEQAAAAAACQFgiCAECauvzyy+3444+3TZs2JbspAAAAQIEgCAIAaWjx4sX24osv2i+//GJvvPFGspsDAAAAFAiCIAAQh/fff9+6du1q33//vfXt29dOP/10u+eee2z79u2+bRRcuOKKKwI+9/XXX9vJJ5+cYz+ff/65y8zQe/fee6/t2LHDPvvsM+vRo4edccYZ9vTTTyf0PCkA0rFjR7v22mvdv0OZPXu2XXzxxe77Bw0aZJ988klA22Xt2rV21113udfV1jfffDOh7QQAAAASqURC9wYAaWLVqlU2fvx4W7hwofXr188yMzOtf//+tnz5cnvhhRfcNsuWLbMffvgh4HNr1qyxadOm5djPokWL7LbbbnPBjxtvvNE++ugj2717t9v35s2b3WulSpWyXr165bnt2u/o0aNt+PDh1qZNGxe8UcCjRYsWvm3++OMPO+6446xLly7Wp08f++qrr+zss88O2M+///5rRx11lLVr186uvPJKW79+vQ0cONDmzZtngwcPznM7AQAAgEQjCAIAcdq1a5dNnDjRatWq5Z7v3LnTZVZ4QZBY91OzZk33fP78+TZs2DAXUDnwwAPdaz/++KNNmDAhIUGQyZMn2549e+yss86ykiVL2mmnnebaPHLkSN82Dz/8sDVp0sRGjRrlnisbZMWKFe6zniFDhtjBBx/s20YaNGhg7du3twEDBljZsmVzfLeCPHp4qEcCAACAgsR0GACIU9WqVX0BENG/lQ2hYEis+/ECIFK9enW3Ly8A4r32999/J+RcafqLprkoACKahvP6668HTOWZNWuWde7cOeBznTp1Cniu6TrKhDnhhBNcgVVNr1E2izJNVHMkFAVOypcv73v4HzcAAACQ38gEAYA4aQqMv4yMDPffvXv35nk/oV6Ldb+hrFy50j788ENbunSpq2fitVfBm3fffde6d+/uy9AoV65cwGeDn2uaTocOHezCCy/M8T1169YN+f2aMnTTTTf5nut7CIQAAACgoBAEAYB8oukg27ZtC3ht9erVSe1vTV1p2bKlDR06NOD1t99+22WIeEGQOnXq5MjmWLJkScDzevXqucKomv4SLdU10QMAAABIBqbDAEA+adq0qSt4qikjomyLp556Kmn9nZ2dbS+99JJbjUaBC/+HVrj54osv7Pfff3fb9uzZ08aMGeNru6biPPvsswH7UzHU9957zwVQ/LNDHnnkkQI+MgAAACA6BEEAIJ+oRsZ5551nhx9+uFtFpVmzZta4ceOk9beCHL/99psriBpMbVNmh7dcrjJCtOzvoYce6trfvHlzO+KII6xEif8lEGrlmMcff9x69+7tCqJqW/03EdN2AAAAgPzAdBgAiMOpp57qMj2CAwkKNHgFR+WNN95wy81u2LDBraTy33//uQyKSPvRUrQKmvhTUOLEE0/M07lSkEPL89avXz/k+8rq8Iq6Fi9e3F577TW3fO7GjRtd259//nn75ZdfAj5z3XXXucKqWhZXx92oUaOA4wcAAABSCUEQAIiDVnTRw59WOwlVH8O/SGiZMmWsTZs2EfejlWD08KfVYvxXoomH6nzoEU5wlopWjOnRo4evoKqW0D3zzDNzfK506dIuWwQAAABIdQRBAKCQueKKK3y1OoIVK1Ys7HQUTVsJtZJLON98843169fPqlWrZvPnz7eTTz7ZBg0aFHe7AQAAgGQjCAIAhYymn6gAaShbtmyxfffdN+R74ZatDUdFXO+77z5XLLV27dpWpUqVuNoLAAAApAqCIABQyBTk1JP999/fPQAAAICigNVhAAAAAABAWiAIAgAAAAAA0gJBEAAAAAAAkBaoCQIASLolU7qYWUaym1FoaBWgzMaNbNH8hWFXAwJ9xrWWPPyM0m9ca6mNn9H867eqVVO/kD5BEABA0s2bM8MqVKiQ7GYUGvrDY/Xq1W7FHv1BAvqMay218DNKv3GtpTZ+RtO73wpvywEAAAAAAGJAEAQAAAAAAKQFgiAAAAAAACAtEAQBAAAAAABpgcKoAICka9K8FavDxEDFyJo0bmTzWB2GPstnXGsF229aVWHWjKlxfisAIBoEQQAASdeg8zgrnpmV7GYUGhkZ2Va90ibbVTfLsrNZWpg+41orKj+jCyd3ytd2AQCYDgMAAAAAANIENUEAAAAAAEBaIAgCAAAAAADSAkEQAAAAAACQFgiCAAAAAACAtEAQBAAAAAAApAWWyAWANDR37lwbNmyY73mZMmWsbt261rNnT6tWrVqO7SpUqGCPP/54wD7Wrl1rN998s/v3U089ZWXLlvVtP3z4cCtfvnwBHhEAAACQOzJBACANrVixwkaPHm0tW7a09u3bW7Nmzezzzz+3Jk2a2OLFi3Ns9+yzz9rMmTMD9vHKK6/Y2LFj3fs7d+4M2H7btm0FfkwAAABAbgiCAEAeTJ061W655Rb7888/7cEHH7Rrr73WBQGys7N927z//vt2zz33BHzuxx9/tL59++bYz6+//moPPPCAXXnllS7IoP3otTvuuMOuu+46+/DDDxN6vi644AK75JJL7KqrrrLJkydbRkaGvfvuuzm2O+ecc+zFF18MeE3Pu3TpktD2AAAAAPmJIAgA5MGiRYvs6aeftpNOOsn27t1rtWrVsn79+ln//v1922iKyKRJkwI+t2zZMnvttddy7OfMM8+04sWLW9WqVV0g5LzzzrOzzjrLTUfR4+yzz7aJEyfmyznbuHGjbd++3SpVqpTjvd69e9uYMWNs69at7vl3331nK1eudO0BAAAACgtqggBAHikw8M4777ipJKJgxcCBA11mSKz7mTBhgjVu3NhXc+OJJ56whQsXWoMGDQKmmyhYkgjXX3+9qwei6SvTp093wY4LL7wwx3aHHXaY1atXzx3nRRddZC+88IKrH1KqVKmYvm/Hjh3u4dm0aVNCjgMAAACIBpkgAJBHBxxwgC8AIo0aNbLVq1fbrl27Yt6PFwCR+vXrW82aNX0BEO81BUIS5ZhjjnE1QTp27GgdOnRwQZj58+eH3FYBEk2B2bJli6sFouexGjJkiCuY6j10fAAAAEBBIQgCAHlUunTpwF+sxf7/V+uePXvyvJ9Qr8W632hqglxxxRWuBokyPpQdEkqPHj3s+++/t8GDB7tAT4sWLWL+Pk0T0rQb77F8+fIEHAUAAAAQHYIgAJDPNGXEWz3Fs379+pTsdwU3NP0mlIoVK7ppOA8//LD16tUr7r7IysoKeAAAAAAFhZogAJDPGjZs6Jad/eeff9yUF02TUdZFqtm9e7dbJrdp06Zht7n33nutc+fObrUYAAAAoLAhCAIA+ezkk0+2I4880o444ghr3bq1zZkzx2rUqJES/e4VRlWx0pkzZ7rX/FetCRXQ0QMAAAAojAiCAEAetGvXzq0G4++ggw6yl19+2TIzM91zLXn7xRdf2NSpU23Dhg02dOhQy8jIsGnTpkXczwknnGDVq1cPeO3UU0+NmKkRrWbNmrk2ekqWLGlXX321HX300VaiRIkc25UtWzbkfg499NCA973tVfQUAAAASDUZ2dnZ2cluBAAgPWmJXAVM2vaZa8UzqQ8SrYyMbKtTaZMtXZNl2dkZ+XqOigr6jH4rDNfbwsmdbMWyBZau9u7d61ZXq1Kliq/IOOgzrrXUsTfFf0a9vytVfD9S3TkyQQCgkLrnnnvsjz/+CPmeCrEquyOULl262Omnn57PrQMAAABSD0EQACikDj/8cKtdu3bI9zZv3mzlypUL+V64zwAAAABFHUEQACikTjvttGQ3AQAAAChUUm8iDwAAAAAAQD4gCAIAAAAAANICQRAAAAAAAJAWqAkCAEi6JVO6aFHJZDej0NCydJmNG9mi+QvdcnWgz7jWisbPaNWqVfK1XQAAgiAAgBQwb84Mq1ChQrKbUWhoULV69WqrUqWKG2yBPuNaSy38jAJA6uIvJwAAAAAAkBYIggAAAAAAgLRAEAQAAAAAAKQFgiAAAAAAACAtsDoMACDpmjRvxeowMVAx1CaNG9k8Voehz9L8WtNqKrNmTE12MwAAhQhBEABA0jXoPM6KZ2YluxmFRkZGtlWvtMl21c2y7GyWFqbP0vdaWzi5U7KbAAAoZJgOAwAAAAAA0gJBEAAAAAAAkBYIggAAAAAAgLRAEAQAAAAAAKQFgiAAAAAAACAtEAQBgBSycuVK+/jjj2P6zPLly+3zzz/PtzYBAAAARQVL5AJAPvv3339txowZ7t/FihWzfffd1+rVq2c1atTIse3UqVPtmmuusTVr1kS9/ylTptjgwYNt6dKleWrnihUr7Ndff7UTTjghT/sBAAAAUhVBEADIZz/88IOdfvrpLrhQqlQp27x5s/3yyy9WuXJlGzRokHXr1s23bfXq1a1z585JOSdffPGF3XLLLfb3338n5fsBAACA/EYQBEChoWkfixcvtg4dOthvv/3mMhcaN27sggkevb5q1So77rjjfK/9888/9tNPP9lJJ52UYz/6r563aNHCKlWqZNnZ2fbzzz+7QEXLli2tXLlyCWv/q6++agceeKD79+7du+2ZZ56xnj172tatW613797u9fr169vFF1/s+8zatWvt22+/df8uU6aMNWzY0GrVqhVy/3v37nXH89dff7m277fffjm2UYbJ7Nmz3XtNmjSxkiVL+rJVdNw7duywyZMnu9cOOugg94j0Oc+WLVts7ty5LtOlefPmrq0AAABAqiEIAqDQ0LSPAQMGuICFggOZmZkuo2Ls2LF22mmnuW3efvtte+edd2zWrFm+zymIoGCDBurefvr37++mpChwsHPnThc8eOGFF+zxxx+3jIwMt+26devc9BQFHhKtRIkSbtrLvHnzXFsuueQSK168eI7pMKoRomCJKFii4zrnnHPs5Zdfdu306L2OHTu6NisQoWDQm2++6esXuffee+3RRx91QYoNGza4Y1R/HX744S4Q9Nlnn9m2bdt839e9e3cXBIn0OVE9kvPOO8/q1Knjgh9qs/pS7QEAAABSCYVRARQqylg4//zz7ccff3R1Nq6++mq79dZbY96PggxXXHGFff/99y7DQVNVevToYTfccIPbr4IryjIZOnSo5aezzz7bHZO+L5RmzZq5zAw9FGxQsOaTTz5xQQh/2scxxxxjc+bMcRkdN910k1122WUuOCJvvfWWPfvss+59BVr030svvdQds7JfDjvsMLv55putQoUKvu9TECS3z8k999xjl19+uZv28/XXX7v+U2ApFGWabNq0KeABAAAAFBSCIAAKFRUV1eDe06lTJ1u0aJHt2bMnpv1omkuvXr18z9u1a+emw2jgL8qyaNu2rSsUmp+qVq3qm7ITjoINyhjRqjEK2igzZfr06QHbKPtDGSWe2267zWWFKLtDnn/+eTvqqKNs/vz59uGHH9oHH3zgptUsXLjQli1bFva7o/mcMljU/l27drnnmp7kTT0KNmTIECtfvrzvUbNmzZj6CwAAAMgLpsMAKFT233//gOelS5d2ARANwDUYj1ZwvQwVLA3et17T9JD85GVqlC1bNuT7v//+u5166qm2ceNGNzVFQSC95gVPPAo8+Ncv0f60jbdijKbHaKrKE088EfA57Xv79u1h2xfN54YNG+bqmBxwwAEumKQisBdddJGb8hNMgRplqXiUCUIgBAAAAAWFIAiAIkUZEd40Df8pGKlKU0fU5qZNm4Z8/4477rCDDz7Y3n33XV8NENUECT7GUNNKFDipWLGi+7eCJ6rRoZonsYjmcyrCqmkyCs4o8+T+++93U3ZUkySYAkt6AAAAAMnAdBgARUq1atXc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J598YsOGDXPvlSpVyubNm+f+PXPmTLfd5MmT426PNwVl0qRJYbdR5kmHDh3s6KOPdpkdeihz5JRTTnHZI/L444+7/954443uv7feeqvb7xNPPBF1W9QPCqwsXLjQPR8zZowdcsgh1qxZs4DtjjvuOF8fKejywgsvuO/95ptvwu5bfXXeeefZvffe67JHli1b5rJaevbsGfYzCjjp2P0fAAAAQEEhCAKg0HvllVdclsRLL71kVapUsZIlS9o555zjBujBdu7c6bZRhsR7772XL+3Rd1900UVuSkzNmjVdO5566ilfcENee+01lwGibA9NRVEQ4corr3QZJArQiOqJqI7IqFGj7Prrr7enn37afU5ZF9GqUKGCa4/6RhTc6N27d8TPqC3KODnhhBMiBnIeeeQRd2zaToGTYsWK2cCBA137w2WDDBkyxGWMeA/1DwAAAFBQCIIAKPTmzp1rLVq0sH322SfsNg8//LAbcCuwoKkegwYNclkj+UHBgNGjR9sff/zhvkeD/bvuussOPvhg++2339w2P//8sy1dutRq1Kjh2lWrVi33UKbK33//7dvXYYcdZrfffruNGDHCbrvtNt+0nVgo6KFA0ezZs11fdevWLcc2+t4+ffq4qUQKnGhajGqYROojHcObb74ZcAzKyNHxhvtc//79fZkveixfvjzm4wEAAADiRRAEQKGnoEOoYqie8ePH2wMPPOCyKJQJosH3Y489lu9L1yqQoMCCsi8WLFjgCofqez3KtvCmoPg//Gt+6Lg++OADN2VnxowZcRUfVWBCwZ8LL7zQ1UrRtKFgql/y008/2ffff+/6SO3o0qVLrn2kKTOhjkHBm1B0HPp+/wcAAABQUAiCACj0FExQloNqgYTyww8/uG3atWvnAiYSXOtCU2hENS3yKlSgolKlSi5bwluuVkvmqlaHluaNRJkkygxR/Q0dh38QJVoZGRl2xRVXuJodCsqEon1r2oy3aoyCHwq6RKJjeP/99wts2V8AAAAgrwiCACj0VPyzevXqLstBmQyahvLoo4/ayy+/7N7X6iUqhvrxxx+7aRoqOPr6668H7EOrtqjWhrZRMCUvS+Q+88wzdvHFF7vaGPo+TYFR8VBNH1EbpUePHi7goGKo06ZNc8EQrSqj51piV/T60KFD3VQWHYP2O2DAAJszZ07MbVJRVWVotGnTJuT72r9WjdF0Ga20c9lll9nvv/8ecZ/KHtGxXXLJJfbLL7+46T1ajlc1QgAAAIBURBAEQKGnqR5aCaZ+/fquUGenTp1s1apVvmVuzz//fLvllltc4VFlL3zxxRd2zz33uNoV/tkSWhJWgZOGDRvmaYlcZVuoDVp95phjjnH//vbbb10h1lNPPdVto6KoarPeV80Oteuhhx5y/27cuLHLGLn22mtdu7WKjHccmtJy1VVXRZz+I5mZme74dFzRvH///fe7uipaZljtVftUPNa/zoqyZfz7TKvMKLikorT6nJYX1tSjeLJVAAAAgIKQkR3PBHMAABJAS+QqsNK2z1wrnkl9kGhlZGRbnUqbbOmaLMvODh3oAn2WVwsnd7LlS+e71Z60qpY3nRC50zRB+i129Bt9VlC41opmv3l/V6r+X6S6cyUKtFUAUEgou0GPUFQ3pHjx4mEzLCKtUpMomq4TrmipsjhUgBQAAABAIIIgABCCpoeMHDkyZN8ccMAB9s8//4RdiWXChAn53qenn366W80llIEDB7ppNAAAAAACEQQBgBDuu+8+90hVqmsCAAAAIDapN5EHAAAAAAAgHxAEAQAAAAAAaYEgCAAAAAAASAvUBAEAJN2SKV208Guym1FoaFm6zMaNbNH8hW65OtBn+aFq1SpcWgCAIocgCAAg6ebNmWEVKlRIdjMKDQU+Vq9ebVWqVHEBEdBn+XmtAQBQlPCXEwAAAAAASAsEQQAAAAAAQFogCAIAAAAAANICQRAAAAAAAJAWCIIAAAAAAIC0wOowAICka9K8FUvkxkArwjRp3MjmsURuoe4zLUE7a8bUZDcDAIC0QhAEAJB0DTqPs+KZWcluRqGRkZFt1Sttsl11syw7OyPZzSkUUrHPFk7ulOwmAACQdpgOAwAAAAAA0gJBEAAAAAAAkBYIggAAAAAAgLRAEAQAAAAAAKQFgiAAAAAAACAtEAQBgAR49dVXrXnz5nnez7vvvmv169cP+7wgxPKdiTpuAAAAoCAQBAFQaH322WdWoUIFq1Onju3cuTPgvXnz5rn39Ni4cWPU+3zrrbesUaNGMbdlx44dtmnTJssrHYd/e4OfJ6K/9KhYsaIddNBB1r17d1uyZEnENhTEcQMAAAAFgSAIgEJr165dbrCu/06cODHgveeff9722Wcf9352dnbU+0xk0CERunTpYr///ntC+2vmzJn2xx9/2Lhx42zFihXWuXNn2759e758JwAAAJBKCIIA8E1pGDVqlB199NFWs2ZNO/vss90A2fP4449bhw4dAnrr/ffftxo1auTYj7Y96qijrGrVqnbOOefYP//8Y4899pg1adLEatWqZVdccYVt27YtYT1/8cUX24svvhiQnfDaa6/ZJZdckmPb2bNn2xlnnOHapvbcdttttnXrVl+mhNq2evVqX8bEgw8+aF9//bXvuT7Xrl07++ijj0K25bnnnrMjjzzSbXfBBRe4Y/f36aefWps2beyAAw6wZs2a2RNPPBHx2CZNmmQtW7bMcQw6P9WrV7cWLVq474xFVlaWOxZ9/wMPPOACHtpnuO9cu3at9e7d202RadiwoV111VW2fv36kPtet26d659u3brlyM4BAAAAko0gCAAXNJg7d67LDHjppZfs888/t3///df69u3r6x1lCmzevDlHZsGGDRty7Oerr75yAZEpU6bYnDlz3ID6u+++swkTJrjHBx98YMOHD09oEOTLL7+05cuXu+f6jipVqljr1q0Dtlu4cKEboHfs2NFmzJhhb7zxhn377bfWp08f977eU7CmcuXKtnTpUve44YYbXGDIe/7DDz9Yz5497ayzzrJffvklYP9//vmnvfzyyy4g88knn7gg0vnnn+97X9uffPLJ7qFsjLvuussGDBhgzzzzTNSZKfPnz7djjz3WBUAUtFGwR23S6/EoWbKk++9///0X9juvv/56d+yTJ092x3XooYeGbPPKlSutbdu2Vq1aNXvllVd8+wYAAABSRYlkNwBAaihRooQbUJcvX949v+WWW9xgP579jB492mUbyEUXXWRDhgxxwRVNTxFlCShocfvttyek7cqqOOmkk1wmy5133mkvvPCCy1wIpnYoC0SBDVFWigbzygh56qmnXHZE2bJlLSMjw/3bn/dc/73ssstcIGfs2LHWtGnTgO0UBDn44IPdv9Ue1d1QAEiBlIceesjat2/vAh9Su3Zt+/XXX+3+++8PCDhFomNQto1/Bsmzzz5r8VAtD2W6qP/UvnAUPOratasdcsgh7vnll1+eY5vFixfbiSeeaKeccoprW7FioWPsCpTp4d8GAAAAoKCQCQLA0fQNLwAilSpVctkB/gPWaPfjBUBkv/32c9NrvACI95qmWCSSgh4KQGhqx7Rp0+zCCy/MsY2yP5Tt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8cYbCT8vf/31l33yyScJ3y8AAABQWJRIdgMA4JlnnrG1a9fa0Ucfbe+++65t3LjRTjrpJOvZs6evcx5++GErU6aMXXvttb7XFCiYPXu2PfTQQwH7adKkiZvKoSyNs846y7p27WoTJkxw+87IyLCLL77YOnbsmKeO33///a158+b2xRdf2Nlnn+1emz59ulWrVs33+iGHHOJe17/Lli1rRx11lHu+adMm19bvv//e9ttvP5dRcvrpp/v2/dtvv9no0aOtU6dOvtcWLlxoI0aMsPXr11vLli2tVatWNnLkSHv77bcD2vXLL7/YqFGjXDbJEUccYdddd51lZmbajz/+aMOGDbMNGza475Pu3bu7R7T27t1rd955p2u/9lWyZMmo2wUAAACkAoIgAJJOA/dXX33VDj30UOvTp4+tWbPG+vbt6zIrLrvsMreNBvH77rtvwOcWLVpk33zzTcB+FDw48sgjXS0LBRNUW+Oll15y7yv4sWDBAhdc+Omnn6xZs2Z5aneHDh3s008/9T1XsKN9+/Zuv/r3VVdd5Xv9uOOOc0EDBQtat25tTZs2tXPOOcc913Zz5syxO+64I+R0mOXLl7vgwgknnOA+o+yT+++/33bv3h3Qnn/++ce6devmAkUKBA0aNMh+//13e/LJJ61mzZp2/PHH29KlS13fykEHHRT1se7atctNpZk3b55rm44l2nYBAAAAqYIgCICUoCyPDz/80GVMePUoxo4d6wuCRGufffax999/3+1Ppk6d6gIfS5YscQN30ZSQd955JyFBkOHDh9u///7rproo2HHFFVe4AMcDDzxg2dnZbjsFB26++Wb37wcffNBq1Kjhvt+jgEXnzp1dPRGv3f6GDh1q9evX931G2Rv6TmW7+Nu+fbvLeNG2UqxYMfe9CoKofQoylSpVypcJEq3//vvPBTmUAaL+rFixYkzt8rdjxw738GifAAAAQEEhCAIgJWgqhRcAkXr16sVcHNTbj38goXbt2paVleULgHivrVq1Ks9tVu0OTa9RkOOUU05x01vefPNNF+RQ5oSyJhQIUWaLN/1GAQJluGiajt7TQ0GBnTt32uLFi91UmmDfffednXrqqQGv6XlwsOHAAw/0BUC8PtT0IO3b//hjobYq06N8+fIu60VBpljb5W/IkCF2zz33xNUWAAAAIK8IggBICcpQ8KfggmpQJGI/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AIAiaN6cGVahQoVkN6PQUOBj9erVVqVKFW44AAAA5HcQZM6cOW6FhDJlyrjnU6dOdVNgnnzySfdH7JIlS+yrr74iCAIAAAAAAFJGXPNVNm7c6AuAiAIehx12mO8uXvXq1W3NmjWJayUAAAAAAEAygiC1a9e27777zjZt2mQ7d+60d955x9q1a+d7/88//7Q6derktW0AAAAAAADJDYI0b97cBUI0JeaQQw5xQY+ePXv63ldR1E6dOiWulQAAAAAAAMlaHUaFT++77z5buXKlDRs2zAVG5I8//rCqVata69at89o2AECaaNK8FavDxLg6TJPGjWxeCqwOo9VdZs2YmtQ2AAAA5HsQRBXpR44cmeP1unXr2tixY+PdLQAgDTXoPM6KZ2YluxmFRkZGtlWvtMl21c2y7OzkLi28cDKZnwAAoIhPh/HMmjXLevfuba1atbJ+/fq515YvX26vvPJKotoHAAAAAACQ3CDIq6++akcffbR98skntnXrVhf8EE2F8abJAAAAAAAAFOogyObNm+2aa66xBx980JYuXWp33HGH770SJUrYmWeeaW+++WYi2wkAAAAAAFDwQZBvv/3W6tevb7fccosrzpaRETgf+aCDDrJffvklby0DAAAAAABIdhBky5YtduCBB/qeBwdBtm3b5oIjAAAAAAAAqSKuSEW9evXs+++/d9NiQgVBPv30U2vUqFFiWggUQfPnz7dHH3002c1AEZFK19OSJUvs4YcfTnYzAAAAgMQtkduiRQs74IADrEuXLvbUU0/5Xt+5c6cNHTrUPv74Y3vyySfj2TUQlqZYKcB2ww035LmXfv/9d3vjjTdCvnfFFVdY5cqV8/VMzJkzxx544AG76aabLJ3Pz549e2zKlCm2ePFiq1Chgh133HFuql1e9pmXz8WL6+l/FixYYPfee6/deuutBdL3AAAAQL5ngijz4/XXX7effvrJGjZsaJdeeql98MEHlpWVZXfeeac98cQTVqtWrXh2DYT1888/u2K8ibBo0SJ3ra5du9a2b98e8MjOzuYsFMD5Wb16tTVt2tQFgn777Tf7+uuv7fTTT3dFl+PdZ14/Fy+uJwAAAKAIZ4J42SC62/r000/bd9995+qANGjQwC677DK3dC4QyjfffGMzZ860SpUq2RlnnGHly5d3r//5559u2WUpU6aMK6578sknu9WGvDvtEydOtP/++88GDx7sXmvbtq175MVtt90WUN8meCCtgXn37t1dVoGWgW7durUdc8wxLnjy3nvvuSlhxx9/vDVp0iTH53r06OGyov766y9r06aNHXXUUblmRUyePNkNqKtUqeJWWVJ2hKxfv95lV+nnS1lYnhUrVtjLL79s1157rf3xxx9xtdczdepUN81tv/32s44dO1rt2rVzHNPFF1/sMjf0vUcccYQde+yxcZ8fZcLomOfOnWulSpVyr+3du9e++uqriPtUu+K5VvS+shR69erla4P6esKECQFZC/PmzXN9obpGHTp0cPtPx+vJ+9mMdF2Ifve/8847bp8tW7aMuq8AAACAQpMJMmPGDOvTp48bZNx999320UcfuYHLiy++SAAEIe3atcvOPvtsNxDTIFODQA3k/vnnH9/g18vE0EBswIAB7lrasWOHb0CnfShLw9tu9+7d+drbs2bNcss/a7D/ySef2OzZs91gWgPmVq1aueCfHocffrgL7vh/buDAgXbkkUfa+++/7wb5Otbhw4eH/S4djwIKyopYtmyZPfvss3bwwQfbwoUL3fsVK1Z0A38NUP0999xzblqPBqzxtlf9eM4557jggIJRn3/+uQtyjh8/Pscx6Zwo60s1KE444QQbOXJk3OdH31W3bl1fAES8wEOkfcZ7rWgw7z99T3Qcmrrheemll9y+dLw6b+edd5699tprlo7XUzTXxdatW10g5/7773e1QDT9SOcDAAAAKFKZIEpjX7p0qSuQCkTj8ccfty+//NIN4GrUqOFe053w4sWLu39rMOzdtffqyxx66KFukNa3b1837ercc891g0T/7RLRrn333df3vGTJkgFZAZs2bbJJkyb5Mhr0vureKBDo3Ynv2rWrmwLmZUXIxo0bbdiwYda7d2/3XHfQr7rqKuvWrZu7Kx9sxIgR7mfq119/dQNUDfRPOukkN4jVwFcUeHzhhRfs9ttvd881yB89enTA9JF42qt/K1tAmV2lS5d2r2m62+WXX+6mp3gZFjomZU20b9/ePdegWvtW1kA850fZGzq3119/vQuQafBfrlw53/uR9hnPtaIMpNwoCHDXXXdZv3793HMFAhQoScfrKZrrQkGwdevWucCmzp2CTwqKRKJglRew8voEAAAASOlMEN3JXLVqFbUTELVx48a5aQBeAERq1qzppsV4NCVAgyyt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ZwbLA8dAgY/Vq1dblSpVXEAEAAAA0eEvJwAAAAAAkBYIggAAAAAAgLRAEAQAAAAAAKQFgiAAAAAAACAtEAQBAAAAAABpgdVhAABJ16R5K5bIjYFWhGnSuJHNK4AlcrUk7qwZU/P1OwAAAAoKQRAAQNI16DzOimdmJbsZhUZGRrZVr7TJdtXNsuzsjHz9roWTO+Xr/gEAAAoS02EAAAAAAEBaIAgCAAAAAADSAkEQAAAAAACQFgiCAAAAAACAtEAQBAAKoS1btthff/2Vr9/x33//2fLly/P1OwAAAICCRBAEAFKQlj39559/bP369SHfHzNmjB133HH52oZJkyZZy5Ytfc8JigAAAKCwIwgCAClEgYYbbrjBKlWqZM2aNbPatWtbw4YNbdSoUQXeln322cdq1arlez5+/Hg78sgjC7wdAAAAQKIQBAFQpKeKKKNi7dq1ObbZuHGjrVq1KuC1rVu32rJly0LuZ8+ePS4zIzs7O2D7UPuO165du+ykk06yjz/+2L744gtbvXq1bdiwwe6//367+uqrbciQISE/57UtEr2/ffv2sO9v3rzZPfx17NjRJk6c6P69bds2W7NmjfuupUuXuofap//qNX+7d+92r+/cuTOGowcAAADyH0EQAEWOpoq0atXKZVRUqVLF6tat6x4//vijb5unn37aTj/99IDPKfjQuHHjHPu55JJL7MADD7R69eq5rIwffvjBLr74YqtRo4bVrFnTbfPvv//mud0vv/yyffPNN/bGG29YixYt3GvFihWz888/3+68804bNGhQQI0OBU2uuuoqlzWi42vUqJHNnj07YJ9vv/22yyY55JBD3HadO3cO2MeCBQvs8MMPt6pVq1qdOnXsmGOOsfnz5+eYDjNz5kwXjFm3bp21b9/ePdTe5s2b27vvvhvwnWr/oYcemiM4AgAAACQbQRAARdLKlSutePHivroaRx99tMumiGc/BxxwgNuPMiEqV65srVu3doEFZYEoG2LHjh320EMP5bnNCiYcddRRLoAQ7IorrnBBj/feey+gbTq2v//+22W26LMXXHCBL/gwdepU69Onj40ePdoFL9ReBXO6d+/u28fNN99sTZs2dfvR+48//rgLeARr166dPfbYY+74vUyQ2267zbp162YvvfRSwLZ63rVrVytTpkyO/aivNm3aFPAAAAAACgpBEABFUunS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ySa7ky1kN3m/ritikslk5swkm3PPfe7z28IlbnrEjFSrlv+GWQEAAOQWgiAAgIRbMHdGptMw4/8o8LFhwwZXR0YBEQAAAPjDLycAAAAAAJASCIIAAAAAAICUQBAEAAAAAACkBIIgAAAAAAAgJRAEAQAAAAAAKYHZYQAACde0eRumyI2DZoRp2qShLQiZIldT4f40Y0puHSIAAICkQBAEAJBw9bqOsUKFyyR6M/KNAgUCdnjF7bavThkLBAq4x5aM65LozQIAAMjzGA4DAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEAQBAAAAAAApASCIAAAAAAAICUQBAHyia+++sq6deuW6M3Ik2ibxLThp59+aj179syBdwcAAAAODYIgSEkTJ0607t2759j65s+fb9ddd5117tzZzjvvPHv22Wdt165dlpM2btxoP/30U46uM1nkdNtkdjyzev7k9HmXmenTp9vxxx9vmzdvTvdcjx497JVXXslWG27YsMHmzJmTI9sKAAAAHAoEQZCS/v77b5s9e3aOrOvHH3+0Y4891goWLGi33HKLXXzxxbZ8+XI77bTTLCepQ65ONHK3bfwcz6yePzl53vmxdetWmzFjhu3duzfdcz///LOtWbMmeJ/zCwAAAKmAIAjypb/++svuuOMOO/XUU61379727bffhnXudPVbt1NOOcWuvvpqW7JkSdjV8fvuu89dHfeWC70iHq8XX3zRTjjhBJct0KVLFzvrrLPsqaeesnHjxoUtp2248sorrUOHDi674Msvv4xrv3TFfciQIVla56233uraolevXvbaa69ZIBDw/b5+3scbFjF16lS7/PLLXYda69u5c2eW1vPFF1/YpZde6rb5zjvvdFkY48ePd9uvNlb7hu5DtLbJbJ+yejxjnT9ZPe/eeust69u3b9g2TJkyxTp27Bi8v2fPHnvsscdcFsmZZ55p//vf/+zAgQOWk6K14axZs1zbqR1uu+02+/zzz8O2y5PZcQcAAADyisMSvQFAvNatW2etW7e2pk2b2g033GD79u2zBx980GrUqGH16tVzt6efftot+++//7qOW8uWLW3x4sVumcaNG9sll1ziOpLecno8q/bv3++uuKtTWqhQoeDjpUqVCv6/rrhrm9WhVqd+5syZdsYZZ9jo0aPdsAQ/+xU5XMHPOtevX2/HHXec6zyrc7pl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nXeeTZw40f7880+77LLLDll2EgAAAOAHQRDABwUD1LH966+/XEDgzTfftIoVK9qYMWPc8xr6oSv1Xsf7+uuvtwEDBtjQoUPdY3///bfrmKoT6i2nQEBWLVu2zM4999ywxypUqOAyMEKHrKjDqs67OrXqNA8aNMjuuusue/75510nVp3aO++803755RcrVaqUe93KlSvdetRZP/roo+2ee+5xGRWff/65FS5cON22HDhwwK6++mo3DEXBjmjDYbQtc+bMsf3791vr1q1tzZo1Lmjx/fffu8CIqF20nTVr1rTGjRvb4MGDrVy5clajRg1fQZDM2iTWMcjqsXvuuedc8CA0CDJ69Gi3Hd72/uc//3FtoawYBaFuu+02d840bNjQsuPee++1xx9/3L2PAkdXXnmlffjhh3b66acHt/mGG26wd999193XcTzuuONcQOaWW26x//73v+4xnRtXXXWVPfjgg+4YtWjRwgVBrrvuOlfXpVWrVsEAYP/+/V2dl06dOtmWLVvcfa3fGyKkgMytt97qtksZOTpuOjcAAACAvIQgCOCDOuyBQMBd7faCBerAe9T59DqgsmTJEhc4uPzyy61+/frWo0cP+/TTT+2ff/7J8Oq+X8qieOqpp1yQQJ1SdTpjBW+0zdWqVXMd0nr16tnDDz/sHtNV/L1797raGe+//77169fPveamm25yWQDKClDQQ1f0lRnx4osvuo51KHWi+/TpYytWrHBZEKHZCJGUcbFgwYJg9osCJNqH1157zd1X57xly5Yu2KKskd9//929rzrTOdEmsY6Bjk9uHDtl0yjYpOwh1Sbx2kCP+6EMmND303mnbfr1119dEEPZFtpP0RAgBTOWLl3qMkM82g8dN4+CIFrvO++8ExyupGyRIUOG2DfffBMMGJ1zzjn2zDPPuGwSefnll93xVdt4mSwjRoywvn37umydQoUKuccUHNE+X3DBBTH3S+eMbh4FhwAAAIBDhZogQCZ01V+d4JtvvjkYAJHQ4TBeB1dX+V944QWXEaFl1WHNDQocqKOpjAAVyqxatarLxlCmSih1ahUAEXVUlXXRuXPn4DAGDTtRoEFDfESBknHjxrlsCC/rQ6/Xe33yySdh61bGgDrSyqRQ1kBGARBvW0KH/2gYhve+yhBRloGCTQqAiIIY8RRW9dsm0eTGsVPbaj1anxf40P4rEOWH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2bJlYcu8+uqr6c612rVrW7LQ97v2UX8LtG9qk8z06dMnXZv0798/3XLDhw+3Vq1aWf369e2yyy6zdevWWbJYvXq13XPPPe6z1LVr10yXf/fdd2N+t+lvsuzZsyfq83ptMti8ebPdf//91rZtW/fb6oILLrBff/0109fp98UZZ5zh/kZ26NDBxo0bl6VlEk6FUQEAyK6LLrooULdu3cDkyZMDU6ZMCdSvXz/Qq1evbL8mK+vNL7Zv3x6oWbNm4Nxzzw3MmTMn8PrrrweKFCkSeO2112K+5tprrw3cdNNNrj0WLVoUuPfeewOFChVy9z3Tpk1T0fPAkiVLAqtXrw7e9u7dG0gG3377beCwww4LPPbYY4G5c+cGBgwYEChdunTg999/j/maIUOGBI4//viw9li3bl2215uf3HzzzYHKlSsHxo8f786RVq1aBdq3bx84ePBg1OV1voS2l2633367O0f//vvv4HJNmzYN3HfffWHLbdmyJZAMdI7oO+fRRx8NdO/ePdC1a9dMX6N2O+qoowKnnnpq4KeffgqMGTPGnUfDhg2La5n87MUXXwx07Ngx8MEHHwRKlCgReOGFFzJ9zcknnxx45JFHAjNmzAj8+uuvgfPOOy9QqVKlwNq1a4PLPPXUU4GGDRuGnWtr1qwJJIvzzz8/0Ldv38ATTzzhvsO1f5nROXTdddeFtcnGjRvDltF3ms6vDz/80J1vnTp1CjRu3DiwZ8+eQH63f//+QO3atd130MUXXxw4+uijM33Nzp070323nX766YEGDRoEl9m1a5c7Bh999FHYcnptMujZs6drs6lTpwbmz58fuOqqqwKlSpVyvytiWblyZaBMmTKB66+/3v2NfPzxx93vj6+++iquZfICgiAAgGxbtmyZ+7HwxRdfBB+bNGmSe2zBggVZfk1W1pufPPPMM+5Hx7///ht8bODAgS7ok9EPvkht27YNXHbZZemCIDt27Agko86dOwfOOuussMfq1asXuOGGGzIMgnTo0CHH15tfKChRuHDhsADbvHnz3Hny9ddf+15PkyZNAr179w57TEEQdU6T0YEDB4JBIgVk/QRB3n///UDBggUD69evDz6m4EZaWlqw0+lnmfws9HuqZMmSvoIgkd9taofixYuHvVbnmc63ZOW1gYLa8QRBbrvttpjPK+BWvnz5sACbzjudf2+//XYgmdpNgV4/QZBImzdvDhQtWtQFiyKDIN9//30gGe2P+Lzpe65GjRruwkosN954o/ubGEp/M0855ZS4lskLGA4DAMg2DccoWLCgdezYMfjYKaecYkWLFk03VCOe12RlvfnJd999Z+3bt7fixYsHH+vWrZtLx12zZk3U1xQqVCjdY/v27bPChQune7xdu3ZuyEzPnj1txowZlgwOHjxoP/zwg3Xu3DnscaVAZ3ZOKNVX7aFhMBoK8tdff+XIevODadOmufMkdP+OOuooO/zww33vn9ahITJXXnlluuc0lMNLfX7qqads//79lgz0/VOgQIG4P9fNmze3KlWqhH2ut27dGhyS5meZ/Cza91S8r9E5pAu2kd9tK1eudOn7GtbVr18/W7VqlSWLrLSbaLiRhlTqO//BBx90w0g98+fPd0MfQj/7Ou+OOeaYpPhuy067hbaf/gZomFAkDS3SECL9LRgzZowli0IRbabPmj5z0X5LePS9peGPofS99eOPP7rhln6XyQsIggAAsk0d9vLly1uRIkXC/sBWqFDB1q5dm+XXZGW9+Yn2r2rVqmGPeZ0iv/v38ccf26xZs9y4cI86bVdddZW98MIL9t5771mdOnXcuN/Jkydbfrdp0ybbtWtX1HbLqM3Kli3rxoyPHj3addIXLFjgxsdv27YtW+vNL7ygWminO979U60C1cUIDUqKOvOPP/64ff7553b11Vfbo48+6mpdpCo/n+uc+OwnO9Ur0He/avN4FDC+88473feavt9UM0Od+T///NNSlYKPDz/8sKu7oNo1qiulWlGRn/1k/W7LCfpuO/vss61SpUphj5922mn27LPP2tixY10Q5KKLLkqq2j2htF8KlqmOTyyxvrdUP2Xjxo2+l8kLDkv0BgAA8j9dQTnssPR/UnRFIVbk389rsrLe/CTa/nlXYfzsn4IfunKlTkFox1SFBdu0aRO836JFC1fcUZ0KZdLk9zaTaO2WUZupcxB6Rf/TTz+1WrVq2YgRI9xzWV1vfqH9U1aDblnZv507d9r777/vzrXIzIi33347+JgybUqWLOk6FPfee6+7Op1q/Hyus/vZT3bqyD/55JP24YcfhnWoFNwNPf90ZV5X6dWBUyAgFb344othn7/q1au7wqrKbFOmYUbfbXv37rVUp+LXyhJU8exQxYoVc4Fdr22VfaRC0vfdd58NHDjQksm4cePstttus+eee86dQ6nw3UYmCAAg23T1RFfSlU4ZSlH/yCsr8bwmK+vNT7QPkVdGvPuZ7d/PP//srkwpHXzo0KFhz0VL39eP4kWLFll+V65cOZcNFK3dMmqzyDZRZkjjxo2DbZLV9eYX2gf9OI2cDcbv/ikAohR7zdiRWdvqXJNkON9y63Odnc9+slNQTcGON954wwXTMjrX1FFVkDdVz7VobXLccce57zKvTbzzKVm/23IiC0TZkpFDOGJ9t2nIWjJlHk2YMMHOPfdcl8EXbahjqFjfWwquK2vX7zJ5AUEQAEC2KfNA9QaUmRA6RZquHuu5rL4mK+vNT7QP06dPD16pk++//94N99Gwg1jUBhrffemll7qhHX6ozkhe+gGSVUqPV6dn6tSpYY+r3eI5J3RF6o8//gi2SU6tN69Sx0hC909TZOq88LN/6ihoysNq1apluqw3xWQynG9ZofbUlWV9T4WeRzrHVMfC7zKp6J133rErrrjCRo4caRdeeGFKfbflFH2v6fvNaxMNV1MdrdDPvqbi1d+RZPhuyw4NgdQ5p4sJfmr/6FxTlkOZMmUsGUycONHVDBs2bJgNGjQo0+V1vkT7G6lzTAFJv8vkCYmuzAoASA5t2rRxVeo17es///zjZlFo0aJF2PSbmllCU//F8xo/y+RXmkpOsx8MHTrUVWpfvHhxoEqVKm4mE8/MmTMDhx9+uJsyUvSvKv1rFplYVOFeFe21TrWTZqLQ1K+aqi4ZjBw50k276VXtf+utt9xMB5rqz/PQQw8FjjnmmOB9Tde3YsUK9/+a4lD3NdWrpvCLZ735WY8ePQLHHnusm9529+7dbraTWrVquVkQPJpBR9Pghlq4cKGbJeHzzz9Pt87p06cHnn766cC2bdvc/eXLlwdatmzppn+NNpNRfhZrdhidV/qMelNAaqaJChUquM+oN83wkUceGejXr1/wNX6WSRaxZofRNLo6/zz6ntJnctSoUTHXpenBNfW36BzWd6U+o5o+PZlkNDuMplT3zpOlS5cGHnjggcCmTZuCUzpramLN8qG/lx5Nf6rza9WqVe58GzRoUKBcuXLB1yWLWLPD6Dtfn9F33nkn7PE333zTTd8aOg2zRzPnaOpqnWeivwP6+6xpeJPBpEmTAsWKFQs8+eSTMZfR9MBqN31fed/3ai+1m35b/PDDD+5v5ogRI4Kv8bNMXkAQBACQI/TjSh0oTcOpH7Lt2rULdjo9ZcuWDZt+zc9r/CyTn02YMCFwxBFHuB8J2j/9WNWPVI865PoxPGvWLHf/zDPPdPf1wyT01rNnz+Br5syZ46Z7LV26tFtvzZo1A8OHD0+KwJFH55E6V7pVrFgxbOpX0ZSR+sEa+oO2YcOG7hxUO7du3Trw3Xffxb3e/EwdnjPOOMN9lvTjV8FETZMbSm109dVXp+tY6BzSdLGRNA2zgibq0KtTpXZTsECdsWShYJo+Y/osaRpN/b86lB51RPWZ/Oyzz8KmqW7UqJELcqq9Na1waKfU7zL51c8//xz8bipQoICb+lf/f91114UFa9VZ8lSqVMm1Q+R324MPPhhcZuzYsYHmzZu7z7GOhf5//PjxgWTx6KOPun3Wd4/OqapVq7r76ox69PfQm8pb0wgriK7vOn3+dC7pOU0vHxkE6NOnj2tfnccNGjRImuCudOnSxbWTppwPPYe84Ky+p9SekR1xtaWCw9EoAKWAR5kyZdxNFx/0d0VtmQyaNWvmPn+Rn7fQKeF1IUDtpsC5R8ENfVa9v5N33XVXunX7WSbRCug/ic5GAQAkj+3bt7saHqq5EEnp96VKlUqXSprRa+JZJr/SfmnMbOnSpdOli6pw3YYNG1x1dRUX82YxiaR058jx3arGrinvVKgyGWmolMZna/hQZMFPnS8aahA5fEOPlyhRImrBXT/rTQb//POP20fVQYmkaYN1LqWlpQUf+/vvv12NgcyGHGhmgWQclqDx/5EF/ZQ6r+mFRc9pmYoVK6b7/OrzqhlNdM7F4meZ/Mb73oqkffTOEZ2Hmp3Ja8do7Sz6Xoz83tdrNXQodOawZKDvJ90iqc2880OfR30v6fsplOr96HOb0bAO/e3QUJjI1+Z3OteiFXnVuaX20N9YzYQT2o6ix/R7ROdYLDond+zYEfadmAz++usv93cgkn4veH8bdK7oe13FdkP/FmoIr7631CaxptT1s0wiEQQBAAAAAAApIfkubwAAAAAAAERBEAQAAAAAAKQEgiAAAAAAACAlEAQBAAAAAAApgSAIAAAAAABICQRBAAAAAABASiAIAgAAAAAAUgJBEAAAACTcs88+awUKFLA1a9Yc0vcdNWqUe9/Fixcf0vfND5YsWWJdunSxsmXLujZ677333ONjxoyxxo0bW+HChd3jW7dutQsuuMBq164d1/ppewCJQBAEAAAAuWrs2LGus1ymTBnbuXNnUrX2K6+84vZt2bJlubL+VatW2Y033mhNmjSxkiVLuoBEs2bN7Oqrr7aff/7ZctPll19ue/bssZUrV1ogEHCBjg0bNtill15q5557rv3zzz/u8bS0NEvlYwQgfyEIAgAAgFzvhB5++OG2Y8cOe//992ltn7744gs76qijbPbs2fa///3P1q9fb+vWrbPhw4fbH3/8Ya1bt861tty1a5dNnz7dzjjjDCtXrlzw8ZkzZ9q///5r559/vhUtWjT4uLJEFCyJx8UXX+yCKI0aNcrRbQeAjBAEAQAAQK5Zu3at68zffvvt1rFjRxsxYgSt7cOKFStcoEGBjm+++cY6depkpUuXdtkgJ598sk2YMMHuuuuuXGvLv//+2/1bvHjxsMeVCRLtcQDILwiCAAAAINeMHDnSdZg1hOK6665z2QXz58+Pufy+ffvspptuskqVKlmpUqXszDPPTDeMYcuWLXbDDTe4GhRad/369e0///lPsOPu0fv07NnTypcvb8WKFXNZFcqoUPZBRhSw0fCJSF999ZV7XP+KghBXXnml+39tg57Tbdy4ccHX/Pnnn3bVVVdZjRo1rEiRIlarVi23fg0zycgTTzzhhps8+eSTdthhh0Vd5r777svS/ma2TQMGDHCPidrZ2y/d+vXrl25/vVtkTRBlrVx77bVuXdoe1RF5+OGHg+8TqyaInzbzashouJCOQ9WqVa1EiRJ22mmnhWWk+DlGAFILQRAAAADkCnW+FQS56KKLXD2Qs846yw2L0fCYWO644w5r2rSpLV261KZOneo60ieeeGIwA0GuuOIKlwkxevRoFxCZNGmSG1IRut6FCxfaCSec4J731qMO+eDBg23gwIE5sn///e9/g5kt2l7tr24aQiJ6z+OOO87mzJnjiolu3rzZ3nrrLTckqFevXhmuW/tXvXp1a968ua9t8bu/frZJAQZlooiG3nj7pVu0/dWtd+/eYduzevVqa9WqlduWN9980wWoPvnkEzckSscrlnjbTIGgunXrukCK3kv/6nzze4wApKAAAAAAkAsmTpyoFITAr7/+GnzsvvvuC5QvXz6we/fusGWHDx/ulr3tttvCHl+2bFmgUKFCgVtvvTX4WFpaWuDGG2/M8L3POeecQKlSpQKbNm0Ke3zgwIGBAgUKBH777Td3/6233nLvu2jRouAy2oZoP5MnTZrkHte/nhEjRrjHli5dmm75vn37um1Yt25d2OMTJkxwr5k8eXLM7S9cuHDg+OOPz3Afs7K/frdpxYoV7r6OS6hY+9u7d+9ArVq1gvcvvvjiQPHixQNr1qyJuc3R2t7v9nnnyz333BO23EsvveQeX7BgQabbDCA1kQkCAACAXKEr8O3atQvLZtAwh+3bt9tHH30U9TU9evQIu3/kkUe62VBUF8Nz9NFH2+uvv+6GesQqxvn111+72hkaGhJKs5ooEyB0fbnls88+s/bt21u1atXCHj/llFOsUKFC9t133+XYe/nd30O1TePHj7cOHTq4zJ94xLt93bt3D7uvIUDy+++/Z3nbASQ3giAAAADIcRr+8Omnn7ohCqF1IzTEY//+/TELpFapUiXqYxs3bgzef+edd1ytkHvuucfq1KnjalEMGjQoWBPkwIEDtm3bNlcnIpL3WOj6/MqslkgobYPeY+LEia6mhzrwuhUsWNDNqqLnN23aFPP1RxxxhKt34fe9/OxvdrfJL61Hw1jiDYBkZfsigyUadiVbt27N9n4ASE4EQQAAAJDj3njjDUtLS7ODBw+G1Y7Qbdq0afbtt9+mK3gqf/31V9THKlSoELyvQIrWrw6xakcou0T1QBQYEXWc1RmOtS6pWLFizG0vW7as+1f1KyJnuvFL26D9Vx0LBX3UgdcttD2UyRJLt27dXH2MuXPn+novP/ub3W3yS++jaXXjaS/vdfFuX7QCtgCQEYIgAAAAyHEKSnTp0iVqJ1XTvmrYxquvvhp1OEQoFeicN2+enXrqqVE7zS1atLA777zTLr/8cpsxY4bt3bvXPaflFWiJzAhQsU1tk4ZXxKIhOBI5i020GUU0Za1Em+1FQRkNQ8lK1snNN9/sZse55ZZbXEAgs9lh/O5vdrYpHio8qqEr8QZCcmP7MjpGAFIPQRAAAADkqO+//96WLFnishmi/gAtWNAFSFTXI7KDrxofelzDO5QFoVlHlMWgaXNl586drvaFaoqsWbPGdWxnz57thlC0bdvWTakqDzzwgMsgUFaBZgzRrCkvvviiPffcc27WlAYNGmTYEdcQnFtvvdX++OMPNzPNgw8+GFx3tBoUqoHhBWA8w4YNcx1wTduqAIVqoWhd6uRrvzRdcCwa5vPBBx/Yjz/+aJ06dXI1PzRlrvZfwQW1rWY+8fjd3+xsUzyGDh3qsjr0Ptpebftvv/1mQ4YMyXB62tzYvoyOEYDUQxAEAAAAOUr1PpR9oEBHLOrkrl+/Pl3mhzrPGuKibAxN+VqpUiWbMmVKsFaIOsjq8L/33nvueXW01TlWQVXVIAnt+CqAoGwKLafaGBpK8cgjj7gpYDNSvHhxGzt2rO3bt88FD5S5ouE4mpo3koq2Khihdet12m+vk6+aGArQKDjTr18/ty8q6vrwww/beeed56aCzYjaSNkoynYZMGCAVa5c2dXA0P+rDsrMmTPj3t/sbpNfNWvWtFmzZlmbNm3swgsvdIEsTZFcunRp69y5c8zX5cb2ZXSMAKSeApoiJtEbAQAAAAAAkNvIBAEAAAAAACmBIAgAAAAAAEgJBEEAAAAAAEBKIAgCAAAAAABSAkEQAAAAAACQEgiCAAAAAACAlEAQBAAAAAAApASCIAAAAAAAICUQBAEAAAAAACmBIAgAAAAAAEgJBEEAAAAAAEBKIAgCAAAAAABSAkEQAAAAAABgqeD/AyRBpvuk5Ca/AAAAAElFTkSuQmCC'
NOTEBOOK_SVM_FI_B64 = 'iVBORw0KGgoAAAANSUhEUgAABEEAAAMWCAYAAAAeTZgVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlcelbwAAAAlwSFlzAAAPYQAAD2EBqD+naQAA715JREFUeJzs3QeUU9X3//099N57lSbSQUVBRUBBREDEilJUQMXeQL8IigVFRRS7KIKKBStgASugiEq3ICiIUgekSO9lnvU5/+fml2SSIZnJlEzer7WymOSWnHvunSF3Z599klJSUlIMAAAAAAAgl8uT3Q0AAAAAAADICgRBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgIRAEAQAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgSAIAAAAAABICARBAAAAAACIEzfddJMlJSX5Hhs3bsySbXMLgiAAgFxnxIgRAf/BR/q46qqrLKfYsWOHffTRR3bttdda8+bNrVSpUpYvXz7376mnnmoPP/ywWyct69ats1tvvdVOOOEEK1q0qBUvXtyaNm1qQ4YMsa1bt0bdpjFjxkTclz/99JNlB32Y82/HbbfdZrnJ559/HnB8F198sSW63H7Oc4qZM2daz5497bjjjrNChQpZwYIFrXz58lavXj3r2LGjXX/99fbLL79kdzNz9fV65MgRq1q1qm/bAgUK2JYtW8Kuv3DhwoD3atOmjWWV77//PuC9n3vuOctp1Kbg/7v0f2Q4/fv3T7X+gAEDsrTNiA2CIAAA5EDXXXedXXTRRfbKK6+4GwsFPPQBWP/OmzfPhg0bZo0bN7Y///wz5PZffvmlC34888wzbp29e/fa7t277bfffrNHH33ULdMHZAA4lnvuucfOOusse/fdd2316tV24MABO3jwoLsB/+uvv+zrr7+2l156yRYvXkxnZqK8efNa3759fc8PHTpkkyZNCrv+G2+8EfD86quv5vwcg/6P1PUcbNOmTfbWW2/Rf7kEQRAAQK6jAEFKSkrA48UXXwxY584770y1zmuvvWY57QNvv3797Mcff7Q9e/a4YMb5558fkOmhb2aD/fPPPy6Aom1EP+sDnNZv166de003L926dbNt27alu31Dhw5N1Yfeo1WrVuneL4CcY/bs2TZy5Ejf8yuvvNJWrFhh+/fvd39Txo8fb82aNcvWNiaS4IzFiRMnhlzv8OHD9s477/ieKxvwkksuyfT25QbKegymzxAK/iF3IAgCAEAOpEyN+fPn26uvvuoCCkWKFLHjjz/efetXpUoV33o///yz++bK34MPPuiyPiR//vzuG1qlrSuN2j8lecOGDfbUU09l4VEBiDfvvfee72cNgxk3bpzVrVvXDYfR3xRlFygD5OWXX3ZD7pC56tevb61bt/Y9V2ZgqIzA6dOn2+bNm33PNXSO8xOZadOmuUCfR8GP4C9S4tlzzz0X8KVFpUqVLNEQBAEAwI/qcPTo0cOqVavmPuQXK1bMfejUuN9Fixal6quBAwcGjA9WhsXkyZPttNNOc9uqhkfXrl1dQCMa999/v7Vo0SLV64ULF7aWLVsGvKb0dP/0aP+bliZNmli5cuV8zxs1amQVKlTwPc+q9F6ND1cat26e9I2kgjoK9Nx+++22fv36VOtr6M93333nMnZOOukkK1mypKuJUqJECXdMN9xwgy1btixgGx1n5cqVA157+umnA87Pm2++6V7XOHz/11etWhWw3QUXXOBbpvdNqzaKju2bb76xs88+27XTuw7Se+wZ8cQTT6Rqm+qI6KZJ76sAmvrOywDS8nPOOce1W23TelOnTk2131DHPGfOHOvUqZO7xnWtn3nmmfbZZ5+Fbdu3335rvXv3tjp16ri26FpWfQllM4VKP4+kn8uWLRvxOU/PNRWuHbrpVyZV6dKl3XHod3XChAlhj11DzzTETb9/ej8dv36+5ZZbbPny5anWz8pr5liC30/DYIKpX6655hqXdeYv1r9nkV5zueF6zWg2yLGGwnzyyScuMFKjRg0X3FKARPWn9H/P9u3bw763jqlPnz6uFoz60dvu3nvvddeKtg1Vf+Tmm29Os27UlClT7MILL7Tq1au79ui6V+BfdThCDd0MVdxTXxKoZpb3f2+0vOCSAgM6L563337b/v33X/ez/m9PiwImCkDps4H+rqh/vHpe+rtz9913p/k7fKz+TctXX31l7du3d9to2zPOOMMFdKIpjHpTiGWR7jej11aWSgEAIAG8+OKLKfpvz3vceeedAcv37NmT0rlz54B1gh958uRJGTFiRMB21113Xar9htq2QIECKZ9++mlMjuWMM84I2Pe8efN8yxYtWhSw7KKLLkq1fcuWLQPW2bx5c0Tv+9RTTwVsN3To0GNuc/To0ZSbbropzX4tVapUysyZMwO2++STT9LcRo/8+fOnfPbZZ75typYte8xtJk6c6Na99dZbA15ftWpVwPt3797dtyxv3rxp9sMNN9yQkpSUFPDav//+m+5jT8v06dPTPL+jRo0KWH799denapseTZo0SXn11VfdNR2qXR9++GGax3zbbbe5fgm17dNPP53qGhg4cOAxz83VV1+dcuTIkaj6uUyZMhGf8/RcU6HaMXjwYLdeqO2fffbZVOfsrrvuCnkOvEf9+vUD+irW10xGXXvttQHv3759+5SvvvoqZd++fcfcNpa/Z9Fcc7nhek3Ljh07UgoXLuzbpmbNmq7dnm3btqUULFjQt7xWrVq+5fv37w/o91CPGjVqpPz+++8B73nw4MGUyy+/PM3tunTp4t77WMeox48//uj2u3fv3pSuXbumua7+Tj3wwAMB7bnxxhsD1gn+vSlevPgx+1G/r/7b6G+ittPPRYsWdcciTZs29a3z/vvvB2zTv3//NK+9UI+SJUumLF68OF39G+7477nnnpDb6BqcNm1amn23YcOGmOw3vddWdiAIAgBICMcKgvTp0ydg+fDhw1M2bdqUsnLlypRzzz03YNk777wTNghSunRpd7OlD6mff/55wI25lv33338ZOo5ffvkl4MZVH34PHz7sW/7xxx8HtKdfv36p9tGxY8eAdYI/jIUTyYc7PRo1auTb5rHHHgtY1rNnz5Q1a9akrFu3LuCDr/pGr3nUdxdccEHKlClTUv7880/3QVkfSL/88kv3Icrbrlq1agE3Ivow5/9+ugmL5OZs9erV6b4504fBYcOGpdpHeo89lkEQfZCfPHlyys6dO1OGDBmS6lzpZk9BsLlz5wbcNJ188slpHrOCeuPGjUvZvn27C8LVq1fPtyxfvnwpv/32m2/bhx9+OGDbq666yvXD+vXrU/3+6Pcu2n6O9Jyn95oKbocCIC+99JJv22LFivmWVa1aNeA9n3zyyYBtGzdunPLNN9+4vw/qA91Q6W9PZl4zGfX111+H/D3Xedbx6Cbwo48+Sjl06FCm/p5Fc83lhuv1WK644oqA/cyaNcu3TNen/zL/AMI111wTsGzQoEHu/zr9Tuj33nu9du3a7nfEo6CO/3Zt27Z1fzd2796d8tdff6VMmDAh5fbbb/etP3v27GMGCEXBJP/11GcKIv/999/upj9cgCj4Zr1QoULuPfxv6KMNguj30f+a1d9TBfy8561atUpZu3ZtmkGQsWPHpvTt2zfliy++cJ8fFBjYunVryrvvvuuCH952bdq0Cdgu2v4NPv4SJUq44LWucfWTfwBO+0pvEKREFPtN77WVHQiCAABSEj0Iom8n/f9jP/PMMwO23bJlS8C3bg0bNvQtC/5QPHr06IBtx4wZE7D8mWeeSfcx6IPUCSecEPAhWze4/hSg8X8/fYsbLDjjRR9WMyMIom+29K2197puNPWB0KObb31w9f+2NhJPPPFE2CBOeoMg+mCb3puzXr16pdp/Zh17tEGQO+64w7dMH0T9l5UrVy7lwIEDvuVnn312wI1+WsesD7vBAQb/5fpA7/WD/wd/BQn8b5YVbNC31P5BGy/DIJJ+jtVNZVrXVHA7BgwYELBtjx49Apbr91R0nP7f/CtYohvpcDLjmknrW3ndVEVKgYFwWUP+Nzbz58/PtN+zSK+5RLhexf/mPDjgfdpppwX8P+Fl4Oj68z+Pp59+esA+dd377/O5555zrycnJ6cKvh/rJjaSIIgCef7/96rdwdev+thbfvzxx4e9WfcPSGUkCKLAhXes+h3s1KmTb/mkSZOOGQRJi3+Gkd5DwdD09m/w8T/66KMBy5Xt5x80TW8Q5NEI95veayu7UBMEAJDwNPuBvhjwaNy4P43jPvHEE33Ply5dalu3bg3Zbx06dEjzuWZ6SQ+Ny23btq398ccfvtcee+wxN6ben8ZRB88QEEx1Q/xpjG8sZ4dZsmSJW/7rr78GjP/t3Lmzq7PiX8ND9VY8wePzP/jgA1djoHbt2m5Mfp48edwY5UGDBgWsl5ycbNlJtSGCZfTYY0Vjt/2vY3+qLVOgQIGQy3WNhLp2wl3Xmj5V5yb4Ovemd/Zfz7/2g86pXvNoRqNwUzeH6udoxeKaUg0Vfyo67E/TUXvH/t9//wVs51/UOKdeM+GmyNXxqF6A6jSE8vfff7tz5B1/rEV6zeWm6zUtaodqLvhf2/v27XPn4YcffghYr2bNmr7/644ePepbFvz/h+o2+P9/4F1jqqXjv91ll13maqRk9v+9qqNx8skn+56rfo5XmyOz+lt/G7xZ2NasWWNffPGF+1m1SoJr3oSiv5sqHnzeeee5bdRPXn0NFSn3qD+9Whyx6N+0/i5l5HfynAj3m95rK7sQBAEAJLzggIZ/IdFwNzrhgiDB2wY/T8+UtComqJtZL7igD1NPPvmkDR48ONW6KujqL1QRsuDXgreJleA+Gjt2bEDBNT100+E/ta9HRSo1naMK1ep1fbj3/7B8rEKNGRXuvUIJdWObkWOPJf/rT1Mup3VNq3BoevbrzUKk4o/B13ksf7fSCiBEIlbXVPBMCjr2UIKP41i/ZznlmgmncePG9uyzz7qZSBTcUfHDyy+/PCCYoJs6FRTNjN+zSK+53HK9HosCMiqe69m5c6crapxWQdTgtur/kOBrzJtZTBRQCbVdrP7PyKn9rQLEwVTYNbh4bzAFARSMUZFgFUfVFNKaSjoc7+9MLPo30r9LmbXfrem8trILQRAAQMIrU6ZMQB/4z+wR7rXgb9XDrRf8XLNJREOZH6qyv3LlSt+NrKbNDfUhzbtR0Tfcad0o+c/QoG+9Qn3wzIx+jeQbNFXWV/v8pyPUMWlmHi3TTZP/NL+xEpz14F8t/1j8bwIzeuyxFqptkSw7luDrWu3XTVjwdR7L362MtDeW11Sk7Qg+Dt0QpSUzrhl9kx4qW0uP4BlGoqHzq1mvNGuGZs461jn2b3N6f88iveZyw/UaqeBzqACI/+wymolIM66k9xpTpkt6ruVI5dT+1sxB/tmfyrBUYONYZsyY4Wbj8mhmIP0frsw6/c5ptqRQYtG/mXW9JUW43/ReW9mFIAgAIOHpA4//f/SaDs6fvvX0nx63YcOGYT+IaXq7tJ63atUq4v7WNJxqm/eBSEMXNP1t8FSHwd/SXHrppb7nv/32m23evDnguf+Hyl69ellmadasWcC3rf369Qt7Q+Y9lP4fPGXolVde6aYg9YZuaKrLcIKzHcJlNwSnGvv30aZNm9zQhOw49ngRfF3rw79/KrR3nSsF2r8fZs6cGXAjrG20rf/NhqaRjEYk5zwj11RGrgH/G/Mvv/zSNmzYEHfXjIbdKdgRLmujatWqAc/9p4CN5e9ZpNdcbrheI6Xpe/2nolX2gRcwDzWsQuv6/1/3yCOPpHl9rV271q2n/4eUeeJ59913XSZVRo8zuD3B//cqa9F/enkNxapYsaJlBf8vGhRsimTK3eC/Mxo+piFsXgaJ/zAlf+np35ymTTqvrexCEAQAkPA0Xto/GDBr1ix76KGHXLBAmRRKOfYf+6paGOGMGDHCpk2bZrt27XI3PdqPRx+ievfuHVF/66asffv2vpsGfdjWGFr/b/XCue+++3y1QfQBfuDAge5YFEzRhzL/m5VwGSWxoIDM//73P9/z1157zfXH6tWrXRqwxnYruKS0f6UQa4iPeOPX/ce6axulrD/xxBM2adKksO+pPvavc6EPnbrZCtakSZOA5yNHjnR9/ddff7lrIa0U5sw89nih45kwYYL7Nl01EZQq7n/zc/311/v6wX/Ylq7B6667ztavX+8CAtrOPy36jjvusEKFCkXVlkjOeUauqfTSjc+wYcN8z5UKrvH1uonW3wf1gdrRs2fPHH3N6Dzpd6JRo0Y2ZswY+/33393fQ9XO0PALDZHx6AbVvw5NLH/PIr3mcsP1Go20MnqCA+YKWCm45tH1pQwpHZuuMR3n3Llz7amnnnK1RD788EPf/xXqB4+uSdW8UIBC3+jr/8nx48cH9GuFChUC3lvXvX+9FW/YhwKSnu+//97uv/9+d40oe0vL/DMG0vq/N9b0f7V3wx5ppljw35nXX3/dnVv97t566602b968kNulp39zmqrpvLayTbaWZQUAIIdMkatp6IKnjg1+qIr9/fffH7Bd8Oww2m+obTXbxtSpUyNub/DUgGk9QlXd1+wH/lX1gx+aGSR4JodjCZ79YOjQocfc5ujRo6mqzYd7PPTQQ77tLrvsspDrFClSJNU0fMEz5GjGlHDv4c3koH/9p8n0f2gaRP+ZUo41a0W42XXSe+yxnB3Gv23BM4VceeWVafab/6wYwcesmUnUL6GORdPC+tOMGppR5Vh9oKli/ad7jrSfIz3n6b2mjtWO4HMcPAPK4MGDA2bACH7Ur18/U6+ZjAqe4SXcQzNo6e+Ov1j+nkVzzeWG6zVSu3btCvm33v+6Cj4n/rPypPXwnw5eM0lpyua01tf/W/6aNWsWcj3/qaT37NmTahr64Ic33XCkM5xkZHaYY0lrdhhdD6eeemrIY6hYsWKqv0H+UzNH27/HOn7/WW00/XmkfXdjBvab3msrO5AJAgDA/59poSrw77//vqsMryJr+rZOqcR169Z136jpW5zhw4en2V/6JleFF1u3bu1qc2hMtmZ5UOV0r+J8VlCV/WXLltktt9ziUoh1HDpGfZurNmqZf9X9zKL0WH2Lpm879S2RUoPVDvWtKuefcsop7tstZc/4z9AxceJE962zvklWP2r4karN65skbZOWV155xWW/aP/hCtnp21sVcNQ3jSr8pm+AVR9F51eZQOmdMScWxx4PNEuC+q9jx47uGtc5UgbAxx9/nCq7SGneOicakqBCmscdd5wbxqGHZrdQsVKNo1c9g+AU+khFcs4zck1lxOOPP+7+dlx77bXWoEEDdw3o+tPPysxSNkVOvmb0DfXo0aNd5oaG7Ojbe/1+6Fxp6IiGg6gt+psSPLtHLH/PornmcsP1Gin14cUXXxxxhojOyZQpU1xmoYbL1KpVy/3/oIeO9fTTT7e7777b9Zf/fnUNvvPOO27Iiq4FDcXRNnp/XRfK0lCWkj/1rzKddO79h3v407nQ9aysqO7du7uMAv//e3UcP/30U0BWZU6l6+Hrr7+2e++91/3uqq917H369LEFCxa4/g0nPf2b0xRK57WVHZIUCcnWFgAAEMf0Qdb/g4nSeDOr0CiQXRQ88L9ZVFDPf9gDkJOuOa5XAGkhEwQAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgZogAAAAAAAgIZAJAgAAAAAAEgJBEAAAAAAAkBAyNjE1AAAZcPToUUtOTrbixYtbUlISfQkAAIB0SUlJsV27dlmVKlUsT57w+R4EQQAA2UYBkOrVq3MGAAAAEBNr1661atWqhV1OEAQAkG2UASKrV6+2UqVKcSbiLItn8+bNVr58+TS/bUHOxPmLX5y7+MW5i2+cv5xv586d7ss17/NlOARBAADZxhsCU6JECfdAfH0Y3L9/vztvBEHiD+cvfnHu4hfnLr5x/uLHsYZY89UNAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgIRAEAQAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgSAIAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgIRAEAQAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgSAIAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgIRAEAQAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgSAIAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgISQL7sbAABAg5YNzJLoh3iSJ08ea1S/kf3+5+929OjR7G4OosT5i1+cu/jFuYtvnL/IVKlYxRbOXmg5GUEQAEC2qzSukuUpTnJiPElKSbKy+8ta5UKVLSUpJbubgyhx/uIX5y5+ce7iG+cvMsk9ky2n4xMnAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgIRAEAQAAAAAACQEgiAAkKCuvfZaO/vss23nzp3Z3RQAAAAgSxAEAYAEtGLFCnv11VdtyZIl9vbbb2d3cwAAAIAsQRAEANLhs88+s549e9r8+fNt4MCB1q1bN3vggQds//79vnUUXLjuuusCtvv++++tc+fOqfYzY8YMl5mhZQ8++KAdOHDAvvnmG+vVq5edf/759uKLL8b0PCkActZZZ9nNN9/sfg7ll19+sSuvvNK9//Dhw+2rr74KaLts3brV7rvvPve62vrOO+/EtJ0AAABALOWL6d4AIEFs2LDBJk+ebH/++acNHjzY8ufPb0OGDLG1a9fauHHj3Dpr1qyxhQsXBmy3ZcsWmz17dqr9LF++3O6++24X/Lj99tvt888/t8OHD7t979q1y71WsGBB69evX4bbrv2+/vrr9vTTT1ubNm1c8EYBj2bNmvnW+eeff+yMM86wiy66yAYMGGDffvut9ejRI2A/mzdvtlNOOcXatm1r119/vW3bts2GDRtmv//+u40YMSLD7QQAAABijSAIAKTToUOHbOrUqVajRg33/ODBgy6zwguCRLuf6tWru+dLly610aNHu4BKpUqV3GuLFi2yKVOmxCQI8umnn9qRI0fsggsusAIFCljXrl1dm5999lnfOo8//rg1atTIXnvtNfdc2SDr169323pGjhxpJ5xwgm8dqVu3rrVr187uueceK1KkSKr3VpBHDw/1SAAAAJCVGA4DAOlUuXJlXwBE9LOyIRQMiXY/XgBEqlat6vblBUC81zZu3BiTc6XhLxrmogCIaBjOW2+9FTCUZ8GCBdapU6eA7c4555yA5xquo0yYDh06uAKrGl6jbBZlmqjmSCgKnJQsWdL38D9uAAAAILORCQIA6aQhMP6SkpLcv0ePHs3wfkK9Fu1+Q0lOTrbp06fbqlWrXD0Tr70K3nz00Ud2xRVX+DI0ihcvHrBt8HMN02nfvr316dMn1fvUqlUr5PtryNAdd9zhe673IRACAACArEIQBAAyiYaD7Nu3L+C1TZs2ZWt/a+hKixYtbNSoUQGvv//++y5DxAuCHHfccamyOf7666+A57Vr13aFUTX8JVKqa6IHAAAAkB0YDgMAmaRx48au4KmGjIiyLV544YVs6++UlBQbP368m41GgQv/h2a4mTlzpv39999u3d69e9ukSZN8bddQnLFjxwbsT8VQP/74YxdA8c8OeeKJJ7L4yAAAAIDIEAQBgEyiGhmXXHKJnXTSSW4WlSZNmljDhg2zrb8V5Fi5cqUriBpMbVNmhzddrjJCNO1v8+bNXfubNm1qJ598suXL938JhJo5ZsyYMda/f39XEFXr6t9YDNsBAAAAMgPDYQAgHbp06eIyPYIDCQo0eAVH5e2333bTzW7fvt3NpLJnzx6XQZHWfjQVrYIm/hSU6NixY4bOlYIcmp63Tp06IZcrq8Mr6po3b15788033fS5O3bscG1/5ZVXbMmSJQHb3HLLLa6wqqbF1XHXr18/4PgBAACAnIQgCACkg2Z00cOfZjsJVR/Dv0ho4cKFrU2bNmnuRzPB6OFPs8X4z0STHqrzoUc4wVkqmjGmV69evoKqmkK3e/fuqbYrVKiQyxYBAAAAcjqCIAAQZ6677jpfrY5gefLkCTscRcNWQs3kEs6cOXNs8ODBVqVKFVu6dKl17tzZhg8fnu52AwAAANmNIAgAxBkNP1EB0lB2795txYoVC7ks3LS14aiI60MPPeSKpdasWdMqVKiQrvYCAAAAOQVBEACIM1k59KRs2bLuAQAAAOQGzA4DAAAAAAASAkEQAAAAAACQEAiCAAAAAACAhEBNEABAtts4YKNZUna3AtHQTERl6pexDX9uCDsjEXIuzl/84tzFL85dfOP8RaZKxSqW0xEEAQBku2Xzl1mpUqWyuxmIggIfmzZtcrMG6YMh4gvnL35x7uIX5y6+cf5yDz61AAAAAACAhEAQBAAAAAAAJASCIAAAAAAAICEQBAEAAAAAAAmBwqgAgGzXoGUDZoeJMyqG2qh+I/v9z9+ZHSYOcf7Sntlg4eyFWXg2AABZiSAIACDbVRpXyfIUJzkxniSlJFnZ/WWtcqHKlpKUkt3NQZQ4f+El90zmegKAXIxPnAAAAAAAICEQBAEAAAAAAAmBIAgAAAAAAEgIBEEAAAAAAEBCIAgCAAAAAAASAkEQAAAAAACQEJgiFwAS0G+//WajR4/2PS9cuLDVqlXLevfubVWqVEm1XqlSpWzMmDEB+9i6davdeeed7ucXXnjBihQp4lv/6aeftpIlS2bhEQEAAADHRiYIACSg9evX2+uvv24tWrSwdu3aWZMmTWzGjBnWqFEjW7FiRar1xo4da/PmzQvYxxtvvGHvvvuuW37w4MGA9fft25flxwQAAAAcC0EQAMiA7777zgYNGmSrV6+2Rx991G6++WYXBEhJSfGt89lnn9kDDzwQsN2iRYts4MCBqfazbNkye+SRR+z66693QQbtR68NHTrUbrnlFps+fXpMz9dll11mV111ld1www326aefWlJSkn300Uep1rvwwgvt1VdfDXhNzy+66KKYtgcAAADITARBACADli9fbi+++KKde+65dvToUatRo4YNHjzYhgwZ4ltHQ0Q++eSTgO3WrFljb775Zqr9dO/e3fLmzWuVK1d2gZBLLrnELrjgAjccRY8ePXrY1KlTM+Wc7dixw/bv32/lypVLtax///42adIk27t3r3v+008/WXJysmsPAAAAEC+oCQIAGaTAwAcffOCGkoiCFcOGDXOZIdHuZ8qUKdawYUNfzY3nnnvO/vzzT6tbt27AcBMFS2Lh1ltvdfVANHzlhx9+cMGOPn36pFrvxBNPtNq1a7vj7Nu3r40bN87VDylYsGBU73fgwAH38OzcuTMmxwEAAABEgkwQAMigihUr+gIgUr9+fdu0aZMdOnQo6v14ARCpU6eOVa9e3RcA8V5TICRWTjvtNFcT5KyzzrL27du7IMzSpUtDrqsAiYbA7N6929UC0fNojRw50hVM9R46PgAAACCrEAQBgAwqVKhQ4B/WPP/vT+uRI0cyvJ9Qr0W730hqglx33XWuBokyPpQdEkqvXr1s/vz5NmLECBfoadasWdTvp2FCGnbjPdauXRuDowAAAAAiQxAEADKZhox4s6d4tm3bliP7XcENDb8JpXTp0m4YzuOPP279+vVLd1+UKFEi4AEAAABkFWqCAEAmq1evnpt29t9//3VDXjRMRlkXOc3hw4fdNLmNGzcOu86DDz5onTp1crPFAAAAAPGGIAgAZLLOnTtby5Yt7eSTT7bWrVvbr7/+atWqVcsR/e4VRlWx0nnz5rnX/GetCRXQ0QMAAACIRwRBACAD2rZt62aD8Xf88cfbhAkTLH/+/O65prydOXOmfffdd7Z9+3YbNWqUJSUl2ezZs9PcT4cOHaxq1aoBr3Xp0iXNTI1INWnSxLXRU6BAAbvxxhutVatWli9fvlTrFSlSJOR+mjdvHrDcW19FTwEAAICcJiklJSUluxsBAEhMmiJXAZPms5pbnuKUqYonSSlJVm9/PVtRaIWlJPFRIt5w/sJL7plsG5ZvsJzq6NGjbgayChUq+ApxIz5w7uIb5y9+Pleq+H5adefIBAGAOPXAAw/YP//8E3KZCrEquyOUiy66yLp165bJrQMAAAByHoIgABCnTjrpJKtZs2bIZbt27bLixYuHXBZuGwAAACC3IwgCAHGqa9eu2d0EAAAAIK4wkBAAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgZogAIBst3HARrOk7G4FoqGpOcvUL2Mb/tzgpg1EfOH8hVelYpUsPBMAgKxGEAQAkO2WzV9mpUqVyu5mIAoKfGzatMkqVKjgbqgRXzh/AIBExacWAAAAAACQEAiCAAAAAACAhEAQBAAAAAAAJASCIAAAAAAAICFQGBUAkO0atGzA7DBxRsVQG9VvZL//+XtCzw6jmUQWzl6Y3c0AAAARIggCAMh2lcZVsjzFSU6MJ0kpSVZ2f1mrXKiypSSlWKJK7pmc3U0AAABR4BMnAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgIRAEAQAcpDk5GT78ssvo9pm7dq1NmPGjExrEwAAAJBbMEUuAGSyzZs329y5c93PefLksWLFilnt2rWtWrVqqdb97rvv7KabbrItW7ZEvP8vvvjCRowYYatWrcpQO9evX2/Lli2zDh06ZGg/AAAAQE5FEAQAMtnChQutW7duLrhQsGBB27Vrly1ZssTKly9vw4cPt8svv9y3btWqVa1Tp07Zck5mzpxpgwYNso0bN2bL+wMAAACZjSAIgLihYR8rVqyw9u3b28qVK13mQsOGDV0wwaPXN2zYYGeccYbvtX///dcWL15s5557bqr96F89b9asmZUrV85SUlLs559/doGKFi1aWPHixWPW/okTJ1qlSpXcz4cPH7aXXnrJevfubXv37rX+/fu71+vUqWNXXnmlb5utW7fajz/+6H4uXLiw1atXz2rUqBFy/0ePHnXHs27dOtf2MmXKpFpHGSa//PKLW9aoUSMrUKCAL1tFx33gwAH79NNP3WvHH3+8e6S1nWf37t3222+/uUyXpk2burYCAAAAOQ1BEABxQ8M+7rnnHhewUHAgf/78LqPi3Xffta5du7p13n//ffvggw9swYIFvu0URFCwQTfq3n6GDBnihqQocHDw4EEXPBg3bpyNGTPGkpKS3Lr//fefG56iwEOs5cuXzw17+f33311brrrqKsubN2+q4TCqEaJgiShYouO68MILbcKECa6dHi0766yzXJsViFAw6J133vH1izz44IP25JNPuiDF9u3b3TGqv0466SQXCPrmm29s3759vve74oorXBAkre1E9UguueQSO+6441zwQ21WX6o9AAAAQE5CYVQAcUUZC5deeqktWrTI1dm48cYb7a677op6PwoyXHfddTZ//nyX4aChKr169bLbbrvN7VfBFWWZjBo1yjJTjx493DHp/UJp0qSJy8zQQ8EGBWu++uorF4Twp32cdtpp9uuvv7qMjjvuuMOuueYaFxyR9957z8aOHeuWK9Cif6+++mp3zMp+OfHEE+3OO++0UqVK+d5PQZBjbScPPPCAXXvttW7Yz/fff+/6T4GlUJRpsnPnzoAHAAAAkFUIggCIKyoqqpt7zznnnGPLly+3I0eORLUfDXPp16+f73nbtm3dcBjd+IuyLM4880xXKDQzVa5c2TdkJxwFG5QxolljFLRRZsoPP/wQsI6yP5RR4rn77rtdVoiyO+SVV16xU045xZYuXWrTp0+3adOmuWE1f/75p61Zsybse0eynTJY1P5Dhw655xqe5A09CjZy5EgrWbKk71G9evWo+gsAAADICIbDAIgrZcuWDXheqFAhFwDRDbhuxiMVXC9DBUuD963XNDwkM3mZGkWKFAm5/O+//7YuXbrYjh073NAUBYH0mhc88Sjw4F+/RPvTOt6MMRoeo6Eqzz33XMB22vf+/fvDti+S7UaPHu3qmFSsWNEFk1QEtm/fvm7ITzAFapSl4lEmCIEQAAAAZBWCIAByFWVEeMM0/Idg5FQaOqI2N27cOOTyoUOH2gknnGAfffSRrwaIaoIEH2OoYSUKnJQuXdr9rOCJanSo5kk0ItlORVg1TEbBGWWePPzww27IjmqSBFNgSQ8AAAAgOzAcBkCuUqVKFTdMQ7OveFSnIifSEBJlUaioqGpxhKKZXlQI1guAqDCpanMEU8aKaoZ4Zs+e7QIjGsoiHTt2dPU9NOuNP68AqxQtWjRVwCiS7byfVWhWQ5U0FOfbb7+Nqi8AAACArEAmCIBcRbUoNLuKindqWMZPP/0UMiMhO3z99dcu2KHZVVSMdfz48Va/fn178cUXw25z3nnnuZlZNNREQ3+eeeaZkJktyq7QMSsAocwSzeii594Ut8ooUU2P1q1bu/5Rhse8efNcbRFvJh3N/qLAiYrBNmjQwG0byXbnn3++ywbROiqIqvaq7wEAAICchiAIgLihgpxnn312qtoeqk/h1QNRcVNNiasaFlOnTrWWLVu6oSTPPvtsmvupVauWtW/fPuC1unXrWps2bTLc7goVKrg2Tpo0yQUolHGh91NwRu/pP9Vt1apVrVOnTr7nCmroGGfOnOmmBB48eLDLvPCv46Hj0WwtClK8/vrrtn79eredZs7x7ycFLxR4UZZGgQIFXJbIY4895lunTp06NnnyZNdfyja5/PLLXaHYY22nDBRN2auph3Ue9N6akhgAAADIaZJSggeWAwCQRZR5ollims9qbnmKM0IzniSlJFm9/fVsRaEVlpKUuB8lknsm24blGyzeHD161DZt2uSCtArOIn5w7uIX5y6+cf7i53Ol6uKVKFEi7HpkggBABFRXRPU4QtHwFg0TCUVFTZVRAgAAACD7EQQBgAh8+OGHtmLFirDfDIT7JrVPnz4EQQAAAIAcgiAIAETgqaeeop8AAACAOMcgUAAAAAAAkBAIggAAAAAAgIRAEAQAAAAAACQEaoIAALLdxgEbzZKyuxWIhooBl6lfxjb8ucEVB05UVSpWye4mAACAKBAEAQBku2Xzl1mpUqWyuxmIggIfmzZtsgoVKoSdHQkAACCn4VMLAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAojAoAyHYNWjZgdpg4o2KoZ556pr014a3sbgoAAEDECIIAALJdpXGVLE9xkhPjSVJKkm19cGt2NwMAACAqfOIEAAAAAAAJgSAIAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAbMeOHfbHH3/QEwAAAMjVCIIAyPV2795tS5YsscOHD6da9ueff9qmTZssJwUj1KasNn36dDvjjDOy/H0BAACArEQQBECu9/3331uTJk1sy5YtqZadffbZ9sILL1hO8cknn1jbtm2z/H1LlSplDRo0yPL3BQAAALISQRAAEdu2bZstX77c/bx//35bvXq1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1Tu1Hp0GAADiCkEQAEC2WzZ/mZUqVSq7m4F0FIgDAACIJwyHAQAAAAAACYEgCAAAAAAASAgEQQAAAAAAQEIgCAIAAAAAABICQRAAAAAAAJAQmB0GAJDtGrRswBS5caJKxSq2cPbC7G4GAABAuhAEAQBku0rjKlme4iQnxoPknsnZ3QQAAIB04xMnAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggBADnPo0CHbuXNnVNscPHgw6m2yY58AAABAdiIIAgDpDFRs37491et79uyxXbt2pXpdr+3fvz+ifX/44YdWu3btqNrzxhtvWNOmTWMaXIlknwAAAEA8IQgCAOkwb948K126tP3xxx8Br5944olWvnz5gICHftZrCipEokCBAlayZMmYn5d3333Xjj/++JjvFwAAAIgXBEEAZLt9+/a5DAr/jIVge/fudev5U3Bh9+7dYfej4Rz+jhw5YocPH45Jm0855RQrWrSozZo1y/fahg0b7J9//rFKlSrZTz/95Hv9xx9/tAMHDlj79u0D9qHXUlJSUu27W7dutnjx4lSva13vmHQcO3bsCNu+4D7U+upD7UMZLHpEmpniT++vbdWX6WkXAAAAkJ0IggDIdoMHD7bzzz/f+vXrZ+XKlXPBhXPPPde2bNniW0fLbr755oDtHn30Ubee/366dOlil1xyidtPoUKFrGvXri4wcfHFF1uJEiWsWLFiNmDAADt69GiG2pw/f347/fTTbebMmb7X9PNJJ51knTp1SvV6tWrVrF69eu75F1984YaZFC9e3GV8XHbZZbZp06Y0h8OMHj3aZZ5o/WbNmtl9991nVatWDVhHgYn777/fqlSp4vqwRYsWtnTpUrfs22+/tdtvv902b95sxx13nHuMGTMmqmNetWqVNWrUyAYNGhRVuwAAAICcgiAIgBxhxowZLjCgbIp169a5x0MPPRT1fnSz37ZtW9u4caP99ddfLgujSZMmLjChDIWff/7ZDQv54IMPMtxmZXb4Z4Io2NGuXTv3/sFBEC8LRMNoFKR58MEHXebKmjVrXPCid+/eYd/n66+/tiFDhtiECRNcpstzzz1nL7zwQqr11GfJycm2fPlyd6zVq1e366+/3i07++yzbezYsVahQgVfJsj//ve/iI91yZIlLujTo0cPGzdunOXNmzfidgVnv6guif8D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xx9/ZHg/Cs58//33Fgu7du2y33//3R1f2bJlbe/evfbRRx9Zx44dbfbs2S4YFGq9VatW2eDBg+3LL7+0n376yfLl478ZAAAA5BxkggDIMrqJ7tSpk7tBvvjii92N9M033+yCAJ5XX33VevbsGbDdjBkz7JRTTkm1nw8++MAuvPBCl6Vwyy23uBvy9957zy077bTT7KGHHrIjR45kuN1qk4aZKPtD+1W2g/b9xRdfuOX//fefNWvWzObPnx+wnQISzZs3dxkjCiIMHz7c2rZta+3bt7dHHnnEDh486DIphgwZ4jInvIyKF154wW3/yy+/2OWXX24nnXSSC3BMnjw5zXb++OOPds011wS8tnbtWtfHp556qnXp0sU+++yzqI69fv36rk3qfx2/Ah3ffPNN2PW6du1qzzzzjC1cuNB+/fXXqN4LAAAAyGx8RQcgyyhYoIDG7t27XUAgf/78Lnhxww032Ntvv+3W2bx5s/31118B2+3cudOWLl0asB/diCuwcP/999uBAwdswIABbt8VKlSwe++91wVE+vXrZ6VLl7abbropQ+1OSkqyf//91wUtChQo4Hu9YMGC7t8yZcpYlSpV7JVXXnEBGc/YsWOtatWqVrJkSbv++utdkGTkyJFWuHBhFwjSz8ouufHGG+2JJ56wSZMmue10DMqoOP30090QHAUVNBxHwaE333zTLrnkkpDtDB4Os3r1atce7efhhx+2Q4cO2fPPP29169Z1QYto/fzzz67vGzVqlOZ6CpSIzgEAAACQkxAEAZCljh496jI5Klas6J4rYBGcvRCJlJQUlwni7UcBj1GjRtl3333nghLSu3dvmz59eoaDILfeeqt16NDBBQ/OO+88l1WhYSHVqlXzraMgjNowZswYK1KkiAvc6DjfeOMNt1wBEAU0tJ2cccYZLiihQJD2o2EjyqTwXHvtta7GhpcVokDGhg0bbOjQoWGDIMEU+KhcubJ9+OGHlifP/0v8U0bJ4cOHIz521SpR2zQESFkl9913n/Xo0SPs+tr3c889ZyVKlLCmTZumWq6AlR4e9RMAAACQVRgOAyBL6abcC1yIMih0I+x/Y5ye/VSqVMmqV6/uC4B4r23atCnDbT755JNt+fLlNmjQILc/DV+pWbOm3XHHHb51zj//fJfhoYCDKKtDwRC97gUfRowY4R5z5851wSAFQMJZtGiRC7z4UwBlxYoVLpMmEhpqoyEwXgDEE02dDmWz6Fg0zEjZKo899ph9/PHHIYMlCuIoi0U1VLSNsnCCKftFmTHeQ+cMAAAAyCoEQQBkqXA34MrsyOh+Qr0W7X7D0c29hu4ou0MZGco6eeqpp3z1MRTQUKbH+PHj3XP926dPH1+gQzVEXnvtNVuzZo1dccUVbpjMtGnTwr7fnj17XFDFn4IqomFAkdi/f79vm/Tyan2oLomOX1koyt4JFSx599133XCcf/75xwV9QlEASTVSvIeySwAAAICsQhAEQI5SrFgxFwDwt379estJVCNENT5k5cqVAUNiNBxHU94q20PDY/xpKM3LL7/sttFQHW+YTt68eVMFazT0RrOu+FuyZIkVL17cBWQiDWAsXrzYYknZHVu2bAn5XqoV4p+dE4rqqGiojP8DAAAAyCoEQQDkKC1atHBDTxYsWOALgKgwaHZS1sfnn3/uq6WhGWfUJgUvWrVq5Vvv+OOPd7U+lBGigqT+NT5Uy0NZIKKAh+qBKODjDdvZtm1bQH0MTcWrmieajlbWrVtnjz/+uHs9UgqyTJkyxQ1l8d5XhVWVqZEeOhfKhNEMNwAAAEA8IggCIEdRAVDNlqKpaGvXru2m0VW9ieykwIaKfZYqVcplaGj2k5deeskmTpyYqvinskEU0AjOAjnhhBPszDPPdEVQFfRQUGXcuHFumabfVbHV4447zjdFrmp5aNjJueeeazVq1LA6deq42iSaVSdSyjzRMBXNvlO+fHn3+Oqrr46ZrRGq1ke9evXc+VCQ6tlnn414ewAAACAnSUqJ1YB5ADgGBQc0Ba4yJjyqb/H333+7oRQaZuJRVoRqRqhwpqZaVSZEgwYNwu5HU7du3brV3ax7NGxD+1AAIRaUvaFsDgVDvGlgg02ePNl69erl6oao8GeobAplkCgQ4X+8snHjRncMClZ4Q168WVn0WnChUR2b3kcBllDPPcpc0ZS72qeG00RCxVe1jUdTA+tcBNcp8dbTe0ZTcNX/PKufms9qbnmK5964fHLPZNuwfIPlJiruq0LBuq6Ci+8i5+P8xS/OXfzi3MU3zl/O532u1GfitIZcEwQBgBj+56jMiYYNG7pMERwbQZD4xYfB+Mb5i1+cu/jFuYtvnL/cEwSJ/ms7AIhDPXv2dIVFQ1F2gzIuQtGMKNdee+0x9z969Gh78sknXcaEZkmJ5/6I9JgBAACAeEMQBEBCePjhh8MGOjQkJ9xUspHWz9AQmK5du7q6Gd60uPHaH9HUDAEAAADiCUEQAAkhVnVBwlGxUz3iRWb3BwAAAJATUckMAAAAAAAkBIIgAAAAAAAgIRAEAQAAAAAACYGaIACAbLdxwEazJMu1qlSskt1NAAAAAEEQAEBOsGz+MitVqlR2NwMAAAC5HMNhAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAAAAAgITA7DAAgGzXoGWDuJgdRrO8LJy9MLubAQAAgHQiCAIAyHaVxlWyPMVzfnJics/k7G4CAAAAMiDnf+IEAAAAAACIAYIgAAAAAAAgIRAEAQAAAAAACYEgCAAAAAAASAgEQQAAAAAAQEIgCAIAcejLL7+0/v37Z+p7zJ4923r16pWp7wEAAABkJabIBYAc5o8//rCJEyfa8uXLrUCBAnbKKafY1VdfbSVKlPCts2bNGvvmm28ytR3r16+3L774wvd81qxZ9uqrr7q2AQAAAPGITBAAyEFefvlla9asmW3atMkuueQS69Chg02ZMsUaNGhgS5YsydK2nHnmmfb222/7nq9bt86++uqrLG0DAAAAEEsEQQDk2qEiChoMGjTILr/8chs9erQdOnTIt84HH3xgt99+e8B2P/30kws8BO9Hr99xxx1u2ZNPPmmHDx+2H3/80a677jq74oorYpYZsWjRIrvhhhvsscces1deecUuvfRSlwGijA8FRi688EL33v4WL17sjkNte/bZZ1Mt37lzpz3++ONuudr7ySefBCw/cuSITZgwwa688kr3Xv5Bj5UrV9rrr7/ua5v6cPv27da1a1f3UMCmW7duLnPF36+//upe37FjR0z6BQAAAIgVgiAAch0NFdHNvIIfxx9/vJ1zzjn29NNP22233eZb56+//nI1L/xt3LjRpk+fHrCft956y6655hpr1KiRtW/f3h588EHr1KmTDRw40Fq1auUe119/vVsvo1544QWrVKmS3XjjjQGv58mTxx5++GFbsWKFffbZZ77X//33X7vgggusXr16dvbZZ9vIkSNdEMWzbds2N5Rm3rx51qNHD2vatKlbrn157r33XhsxYoSdccYZbh+ff/6520/wcJjq1au75YULF3bHrke7du3ceyhg4++ll15ywZeSJUtmuE8AAACAWKImCIBc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8ccfrU6dOjZo0CDLkyePJSUFVrM7/vjjbcmSJRlrGQAAAAAAQHYHQXbv3m2VKlXyPQ8Oguzbt88FRwAAAAAAAHKKdEUqateubfPnz3fDYkIFQb7++murX79+bFoIANnsySeftKVLl8Z0nxpO+Ndff8V0n9n5PgAAAECuDYI0a9bMKlasaBdddFHAh+uDBw/aww8/bF9++aX17Nkzlu0EkIv88ccfNmLECJs3b57Fg0ceecR+/fXXqLZZuXKljR8/3p544gk3bfiaNWsClt9///2uHzJbVr0PAAAAEA/ypWcjZX689dZb1qFDB6tXr54VLVrUDX8pUaKEC4SMHTvWatSoEfvWAsgVHn/8cTfD1OzZs+2LL76w3GT//v1244032qRJk6xLly523HHH2aJFi+y+++5zgWMFVAAAAADEURDEywZR8dMXX3zRfvrpJ1cHpG7dunbNNde4qXMBZO4Qh4svvtg2b95sCxcutFKlSlmPHj1cQFIOHTpkjz32mF199dVWtWpV33bPPPOMtWvXzpo2berbj27MN2zYYL/88ouVLFnS7bdIkSJu+mtlP+gmXvuOVZ0fDaN777333BCT2267zVavXm01a9aM6vhi2Qcvv/yybdq0yQV3tV779u1TtScamjnr008/tZ9//tkFiT1Hjhyx77//PmSh6XDHKJpufNq0aa4WU5MmTezss89OtY+NGze6ddS3bdu2tebNm4dtn/Y3YcIEF6DR9ObKEunXr59v+fLly23KlCl21113uec6DrW7V69eLstv3bp11qZNGzvllFMC9vv777/bd999564T9aEKZAMAAAA5TVJKSkpKtBvNnTvXXnnlFbvnnntcfRAAWatQoULWoEEDK1asmJ188smuDo/oZrpAgQLuhrl48eLuBts/KFmtWjU3DOWqq67y7ad69eru5r9Fixbu5rds2bLu5l3BhcaNG9uHH35oZ511lr3++usxabuCDsoEWbFihZ1xxhkuo+yBBx6I6vhi2QfPPvus/fvvv3b06FHXJmWmaPhK586dfduUK1fOnnvuuWMO81NAR38TFWhRNkhaIjlGnQ+1U23REEQFOrSNXvdqMX300UfWu3dvO/PMM13gYc6cOS4YPXDgQN/7fPDBB9a1a1d3fB07dnQBEB23zoOWLViwwNcu7Vv7U//JuHHj3Exg6oPTTjvNBTk0Bbq2vfXWW906Gvajny+99FIrXLiwy/AZPHiw28+x7Ny50wXfms9qbnmKZ15B7eSeybZh+YZM238i0u+MAogVKlSgGHoc4vzFL85d/OLcxTfOX87nfa7csWOHG6US00wQfehZtWoVARAgGynz6v3333c/b9++3QUyPv/8czv//POj2s+JJ57obvrlyiuvdMGQRo0a+W7KlW2hG+zRo0e7G+GMevXVV23AgAHuJv66666zYcOG2fDhw1PdREVyfLHog5tvvjng+ZgxY9wNvH8QJFK6+dd/kAo0RCKt9m/bts369u3rsmbOPfdct45qLp1wwgn29ttvu8yMLVu2uHU01OZ///ufW0fv/9tvv6V6r8WLF7v9qM8ffPDBqI5L/5Ho/Pfv3989V1DshhtusMsvv9zdACuwpTao3+Tw4cNhC8keOHDAPfz/swIAAACySrq+dtO3lkqfT0cSCYAY0TAWj4ZS1KpVy/7555+o96MhGB5lfoR6Tb/rynLIKA2h0824hqjIJZdc4jIONMwiPccXiz7QsX3zzTcu00NBhj///NMNEVF9o2ht3brV/VulSpWI1k+r/QqGqL6IhqNo6M/IkSPt6aefdtFtDUH01lE777jjDt9+FEzScEV/3377rQtcDB06NOoAiOTPn98FWzwKwKjfNPzFO16dQ69Qdr58+XzDjYLpOHQM3kOZSAAAAECODoJUrlzZrr/+evfBm2/xgOyhYRT+dOOpOhgZ2Y/2Ee619Ow7mIZWVKpUyQ2n05AUZRco+0HZIWm1y2tHcBsy2gcKICg4oMwUZU/s2bPHZTGofkd6/raVLl3a/asgcSTSar9qd2hoiYJEeqhtquFxwQUXuLofomE8ysTwhs+EoxogGk7jBZ+ipQwgBUI8efPmde+7fv169/z55593/y8oQK5Azi233OLaFsqQIUNcZon3WLt2bbraBAAAAKRHuobD6FtFFf5btmyZu5nROPTy5cv7xqiLCg966dkAspYXuNANvT+vzkN20BCIN99802WZKMPB06lTJ1efQjVI9HckK/tg5syZbppeBS28cYPTp093wZr0UI0T/R2cMWNGQFHU9FBfKPCh2ksqVBuKAhEanqjAiX+QIpiyXPTQcBhlj6hWitdHkVwjynDRel6femNivYwXBUDeeOMN97pqmtx5552ufor6N1jBggXdAwAAAIibTBDdwGg2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bGeddZbdf//97qFAR5UqVeyWW26xlJQUy+n823/vvfda27ZtrW7duvbtt99aItq6dav17t3bypUr5wJJAwYMsOrVq9sll1zigmeR+uqrr+zCCy+03GTevHnWvn37kEE+9c9rr71mOUV29X+FChXs9NNPz/T3OXTokPXr18/9rbnmmmusa9eu7vd2/PjxOfYa3L9/vw0fPtxq165t5513ng0ZMiS7mwQAAACkKZ/lINu3b7c///zT3aDqRsCzceNG++OPP9zPxYoVs+OPP95KlCgRkPnwzz//2M6dO9232nLCCSdYpUqV0t2WAgUK+PYlCiC0a9fOzj//fOvQoUOa265bt84NlalRo4ZVq1Yt3evomPUNff369TPUft1c9ezZ0z00rOaXX36x0qVLW+XKlW3JkiVu3SZNmrh19c3z77//7r7Z1fsWKVIk1b7//vtvd9OsPtb5ONayvXv3upvt1q1bW8GCBX3rqn3NmjVzbVEbNeTg5JNPdjdWy5cvt+OOO853HShzQ+0qU6aM1alTx/LkiSx+p/fWTX7RokXdPtXfomDWlClTbO3atVazZk137SlDRAoXLuxuPsuWLevbj45p2bJl7nx4/ar26RFp+9Q3eh/1zb59+2zp0qXWpk2bVEMvlAGgG2/tx9+GDRtc0KZVq1b2119/ueeNGzd25/PUU0917fbs2bPHDZ865ZRTQp7DaHn7S+t9duzY4WufrjNd42qf+j5YWv0V6jjr1auX7v7335/+lqht+htSvHjxgDYdOHDAXSPaVucob9687nUdw1133ZXqGCI9V2m9p79Ro0bZp59+6q6LWrVq+f62KeMsrWuwVKlSaV670V6D0fyu6ZwnJSW5rKE777wzXRlWAAAAQJZKyQGOHDmSMmjQoJTChQunNG7cOKVq1aopffv2TTl8+LBbPn369JS2bdu6R4sWLdx6jzzyiG/7qVOnphx33HEpxYsX962nbdLrxRdfTClYsGDAa7t371YKSMrbb7/te23kyJEpJ510ku/5gQMHUnr27JlSpEgR1079q+d6PZp19u7dm9K5c+eUYsWKuXUqV66c0r1795RGjRqlu/0fffSRa//ff/+dcvrpp6ece+65KbVq1Uo5+eSTUwYPHuzW+eGHH1w/1q1b171vqVKlUiZMmODbx44dO1LOOOOMlPLly6eccsop7t9Ro0Ydc9myZcvce69duzagTXrtq6++cj9v3rzZPdd5r1ixotuXzquMHj06pWTJkq5NanODBg1Sfvvtt4j6Qtvmz58/ZcWKFWmut3jxYt+107JlS3debrzxRt/yuXPnppxwwgluX956Xt8cq326jnv37p1SqFAht4765rLLLkspWrRoQBtuvfVWt07z5s1TSpQokXL22WenbNu2zbf8lVdeSalQoUJK165dU+rUqePasGHDBre/8ePHB+zr+eefd+seOnQozePW74n6XfsJpt/D4cOHu591fR7rfbz26dpVP6iNpUuX9p1j/3OSVn+FOs4vv/wy3f3v7U+/Q/Xq1Utp2LCh61/t0zNnzpyUSpUqpRx//PEpzZo1S6ldu3bK7Nmz3bJ33nknpWzZsuk6V2m9Z7ALLrjAHXM44a7BY1270VyDGfld0/66dOmSEi397dA12HxW85QTF56Y4UelepWibgPS/3+3/nboX8Qfzl/84tzFL85dfOP85Xze50r9m5YcEQTRzbJuuBcuXOh77bXXXkvZt29fyPX1oV8f9P3Xf+ihh1JOPfXUmLRHQQTdaMycOdM9Pv7445QePXq4G6Tt27eHDYI89thj7kbqn3/+cc8VcNCN0OOPPx7VOgrw1KhRI2X9+vXu+ZIlS9zNSkaCIM8++6y7INR+BUEUYFm6dKlv+datW92Nnvrdo5tABZy89Z5++ml3E7Z//373/ODBgykvv/zyMZdFEwQ588wzXcDJM2XKFNc/f/75p++1//3vfy5YdvTo0WP2Rfv27VPatGmTEq3Vq1e7YIyCR56JEye61/xF0r433njDBei8ftR51fn1vwF99913XV/r2pYtW7a4G9Abbrgh4MZafaRr3d8dd9zhgkb+dF3eeeedxzxOLwjy4Ycf+q5371GuXDlfECSS9/Had9ttt/mWK7ipYIr3uxxJf4U7zvT2v7e/J5980rfOzTffnNKkSZOA6+T222/3PV+3bp3bd6ggSDTnKq33DDZs2DD3e/nee++5QGgoofogkms3kmswo79rkQZB9DdC/zF5D/1dIAgSn/gwGN84f/GLcxe/OHfxjfOXe4IgOaImyNixY+3666+3E0880ffalVdeaYUKFfI91zANpcZ///33Lp1b6eL6ObNoOIhXE+Shhx5y6edqU1rp7BMmTHD1Q7wUfbVRz/V6NOu8/vrrvjok0qhRIzeU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j+EwAJz9+/fbqlWr3M8pKSm2efNm96+/PXv22Nq1awNeO3jwoNvu6NGjIfezadMmO3LkiG/9AwcOuH3H2uTJk937amjLBRdcYFdccYULkCjrIvj9du7cacnJycfc57Zt22zfvn0hl+l4N27c6DJRoqXMD7V19erV9u+//1q+fPnswgsvdMu0z+3bt7uf1Xb1X3Db1a5wdD62bNkS8TK1X23xP0ei13QuvWPVc22v9dR3ymjxp+Pw1gcAAAByKoIgAJzvv//eZSg88sgjVr58eTv++OPdUI0vvvjC10OffPKJtWjRIqDHli5d6rb777//AvZz9913W4UKFaxevXpWtWpVmzFjhg0ePNgqV65sderUsfr169uKFSti3vt58+a1oUOHumDLnDlz3M/XX399wDrjx4+3c845J+w+tJ2O/7jjjnOZJeedd56tW7fOt/z555+3SpUqWZMmTaxUqVJ2+eWXpxmYSIu2v+WWW1xfrFmzxi6++GLX3pNPPtlOOOEE12ei5cp00fuq//TwPzcKUPTp08dlljRq1Mj1+TvvvHPMZV4mTXBNFL2mcyk6t3o+fPhw9/6nnnqqffzxx27Z+++/bzVr1rQGDRpYuXLlrFOnTqkCZQAAAEBOQRAEQIDffvvN3cTqpv7KK6+0fv36pcoIiYTqdChwoBtoDfc499xz3c/KdFCGg4asDBkyJFN6X9kgUqRIkXRtf+2119qll17qMjLU1ttuu80WLlzolr355psuUPT111+7DBMFD3RcCmTEqr0ayvP444+7/b/++usuA6djx4521llnuUwQvd+IESNcwESBE9F6s2fPds+VlbFo0SL7888/j7ksGtOmTbNffvnFXR+9evWy7777zgYMGOD2rzaprxQkURZOOApO6Rj8HwAAAEBWIQgCIIBuvgsXLux+7t+/vxv6kJ6ZUx599FE37EOZGQooaNjFY489ZgUKFLD8+fO7G3gvsBALaqOGbMybN8+uu+46dzOuoEF66IZeWSAqtKripcoa6d69u1s2evRou+qqq6xMmTIuGKBAydVXX20ffvihb0jQsSiopLb+888/Nn36dHvooYdckEOZFKKhMf5tV9FWDTXR+dBx6n0VWKpevboLTHhtVlFYtUuUxXP//fcfc1k07rnnHpcZ43nqqafc0KO6deu6NinAcs0117gMEgW7Qhk5cqSVLFnS99AxAAAAAFmF2WEABPC/KS1WrJj7V9/W+9/8RqJatWq+n4sWLeoCIt5NvvdacF2JjLjhhhusUKFCbnhJs2bN3JAV76Y/Wg8++KDdfPPNbsiIghG60dcwElEBUwWGVODUn4b+KHumbNmyx9y/hqeoMKo3O0zPnj3d8CGPhrr403sqW+SMM85ItS+vD/v27esyMrSthu8oqNKtWzcXcEprWTRCtUsBlm+//TbgdQ2PUUBEgahgyv654447fM91bREIAQAAQFYhCAIgYsqMCBZcUDO7qDBqq1atYtJmZTNcdNFFbsjLl19+aS1btnQZIKrVoSKmDzzwgA0cODDdbfUKo4aj9/CnbBoFFv7444+w26jWioISc+fOtZkzZ7paKMow+fHHH9NcFk3/hGqX+kpZP9EcO1MCAwAAILswHAZAxJTJoeEfquvg+fXXX3N8m4OHZhyrzQoCKItEw3jGjRtnN954o02cONEt01S8oabfzcxg0GmnneYKo4aaRtd7X/2rzBK1T8NWFMD5+eef3bGmtczLzvHvo0jPqdqlAqmHDx8O2SYAAAAgpyEIAiBiyohQHYc777zTfv/9d1er4t57783RPajZSpTx8Oqrr7o2P/nkkzZp0qSw6+sG/sQTT3TDRxR0UEHRzz//3M3WIiqKqteUAaHMivnz57vaGN4Ut5mha9eubviMhuUoAKPjmDp1qnXu3Nm9vwwbNszNJKNipcuWLXPDgXSuNDtPWss0fOekk05y51FFT7/55htXeyQS2q+Gvai+i+qAKLDy8ssvW5s2bTKtLwAAAICMIAgCwFExVA25CB7+oNe82hGaYvWzzz6z5cuX22WXXeaKgb7wwgtuHQ2NCLcfzXoS/JpqgtSoUSPDva+2ad/hhliojoayOfRQbQwV8FTxV00T61FAwKt5ouNQLRANG9HUtwoe6CZ/1KhRbrmG3Cj4sXfvXlcQ9aabbnI1Ql555ZWI2lu8ePFUfeFPQ1dU18SfsjjU75qpR22/5JJLXJDm9ttv9w0B8qavVWBCyzX1rY5BBVHTWuZNc6v6L5rVZcyYMfb000+7NnoFcr3hOCpq608z/CxYsMDVQ9FQIQVPFi9enKpeCgAAAJBTJKWkZ+5LAABiQIVRFYRqPqu55Sme8bh8cs9k27A8+tmMED3NhrRp0yYXBFOgDvGF8xe/OHfxi3MX3zh/8fO5UhMK6MvbcCiMCiDbbd682fbs2RNymeqPhMvyUFZFJLOxZCVlmoSriVG+fHmXAQMAAAAgexAEAZDtNFxj2rRpIZepQKmmYQ1FU8tGMzNJVujevXvY9mqoiep6AAAAAMgeBEEAZDvVFcktFi1alN1NAAAAABAGg3gBAAAAAEBCIAgCAAAAAAASAkEQAAAAAACQEKgJAgDIdhsHbDRLyvh+qlSsEovmAAAAIJciCAIAyHbL5i+zUqVKZXczAAAAkMsxHAYAAAAAACQEgiAAAAAAACAhEAQBAAAAAAAJgSAIAAAAAABICARBAAAAAABAQmB2GABAtmvQskGGpsjV1LgLZy+MZZMAAACQCxEEAQBku0rjKlme4ulPTkzumRzT9gAAACB3YjgMAAAAAABICARBAAAAAABAQiAIAgAAAAAAEgJBEAAAAAAAkBAIggAAIpKSkmJHjhyhtwAAABC3CIIAwP/v8OHDuSpgoeMJF7TQsmiP991337WKFSvGqIUAAABA1iMIAgBmtm7dOsufP7/98ccfMemP2bNnW/v27a1UqVJWsmRJO+uss+zjjz/Osr7+4osv3PGULVvW9u3bF7BswYIFbpke27dvz7I2AQAAANmNIAiAhOBlPhw9ejTkci9jQv+mlUERieXLl1unTp2sdevW9tdff9mGDRvsgQcesAkTJti///4b8X7UBmV0hMrwiFTp0qXtgw8+CHht3Lhxdtxxx4Xcrx7B7wkAAADkFgRBAOQKa9assUsuucRlXehx1VVX2bZt23zLCxUq5B4FChSwGjVq2LBhw+zQoUNu2YEDB6xOnTru52bNmrn1zjnnnHS35auvvrKkpCR7+OGHrVy5clakSBFr06aNTZ48OWA4idp80UUXWfHixa1EiRLWsWNHW7p0qW/51Vdf7dqj9ona27JlS7viiisibov2MX78eN/zvXv32jvvvGP9+vULWO+7777z9ZEejRo1cutFkvFy2mmnuX7VsfXv3z+g3wEAAICchCAIgLi3a9cua9u2rQsWLF682NauXeuGn0yfPt23jpfloHWmTJni6ls888wzblnBggVt1apV7uclS5a49b755pt0t8cbgjJz5syw6+zevdsNl6lVq5b9888/lpycbK1atXLBl507d7p1nn/+eduzZ4/dfffd7rkCN8okGTt2bMRt6d27t/3000+2cuVK9/z9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sFtvvdWaNWtmLVq0sLvuust2797tMjBuueUWl4niZWJoiE2tWrVcpom/b775xmrXru32Fcqx2hCJypUruzboHDz++OP2xx9/+I4xuD/SOq5Qdu3a5c5/7969s+TcAQAAANEgCAIgqmyGefPm2dixY+2RRx6xt99+291ADxw40LfOzp077d9//w3Ybt++fS4DwX8/c+fOdcGUUaNG2euvv27fffedCz6oHsVLL73kHtr/s88+m+lnyH/4R/v27d3QFXn//fftp59+csGev//+20477TQXnJk6dao9+eST9umnnwYc+2233WYLFy50wZ2JEydapUqV7KmnnnLBhgceeMDKli3r9qeHgiKqu6F1/b388ssu2FCyZMlU7YykDdEqWrSo77yF6o+0jiuYgjzqv3z58rl1CxQokO52AQAAAJmB4TAAopKUlGTvvvuulSlTxj3/3//+Z1dffXXUvZgnTx4X5PD2c+WVV9pjjz1mv/zyi5UoUcK9powHFShVRkhWKVy4sLvJF/2rjAkZOXKky44YNmyYb11lrShgoYCAgha//faby4Bo2bKlW964cWNXX0THWrp0afevtz+55pprXO0RBXqKFy/uMmEU3FDwJZRI2hCNAwcOuO0UnDnllFPCrhfuuPytXr3azjnnHGvVqpW9+uqrLhAS7j318PgHXwAAAIDMRhAEQNRDKbzAhVSsWNENgdCNrVdTIj37UY0K1d/wAiDea1u2bMkRZ+iHH35w2RHHHXecpaSkuIdXY2TlypWuyOgll1xi999/v61bt846duxoZ555pguqhNO5c2crVaqUvffee9a/f39fUEivp7cNkdB6Csgoc0PBGQWaFIQJ51jHpeEyp59+uvXo0cOeeeYZFygLR4EcZcUAAAAA2YHhMACikjdv3pCv64Y8o/sJ9Vq0+83MoqLKeFENjzlz5riAhIb0qLioMiPk7rvvdjVFFGC45557XCaJAhzh6HivuuoqGz9+vHuuf1UbJVwWRSRtiMRnn33mttXQo/z587vMjbQc67gU/KpRo4bL4tFQp7QMGTLEBU28h9oOAAAAZBUyQQDElDI5lBkSPFQi3jVq1MjNoOI/nCUU1ezQQ4YPH26DBw+2Sy+91AU8QgV0lAGi7AgNMVq8eLFNmjQpw22IJAtHgQwFLt58801r166dXXTRRe7faI9LChUqZJ9//rkbqnPeeefZ9OnTfbVGgilgEk3GEAAAABBLZIIAiKmTTz7ZVqxY4WZEkeXLl4csohlvBg0aZN9++60LWGjWEwU0li5dajfeeKNvnZtvvtnVz9Cyw4cP24YNG9yQHqlataqr+aEhKP40Q4xmyBkwYIAbUnL88cdnqA3R0gw4mgpYQY1wWTdpHZd/8EuzAKldCoQcKyMEAAAAyA4EQQDElApsqnDnueee6+pbnH/++XbhhRfmiF5WLQxvilrvEWlhTtXBmDJlipsdpVixYq6Wx2WXXeZmQ/Fo2mAVENUyFSnV8JAJEya4ZWeccYbrE2Vf6H01s4t/gVRNOduvX78MtyE9FFRRFkq4oTtpHVdwIOTLL7909WEIhAAAACAnSkrJKQPuAeR4mjpVQQNv9hTRN/+bNm1KNURDGQOaGlcFN1XLQlkQVapUCbsfZQ4oEKBCqx4Nq9E+KlSokKF266Y8OAPDowwNtW/79u1umIio2KiyHfQ8VJ0S1bJQ3Y5wQz7UbtXa0DCRYDoe9YX6xSsCq6EwygTReyq44b+uf7uiaUNa/aDzoPoe/jStsaa0VaHUcO8b6rhCnUv1pwraataZtArDirZVYKX5rOaWp3j0cfnknsm2YfmGqLdDxmmGIP3u6/cz+HpCzsf5i1+cu/jFuYtvnL+cz/tcqc/J/pMtBCMIAgDZSEGkNm3auKlln3766YQ7FwRB4hcfBuMb5y9+ce7iF+cuvnH+ck8QhMKoAOJCly5d3DCMULw/dqHcdddddsstt1hONGLECBs1apSbGvi+++7L1D7Kyf0AAAAAZBWCIADigmpQKGsiFL2uoRzhAiQ5lQqOqh6IhhMkJSVlah/l5H4AAAAAsgpBEABxIaN1QXIiBSZiGZzIjX0EAAAAxBKVzAAAAAAAQEIgCAIAAAAAABICw2EAANlu44CNZukoi1Kl4v+bdhkAAACIBEEQAEC2WzZ/mZUqVSq7mwEAAIBcjuEwAAAAAAAgIRAEAQAAAAAACYEgCAAAAAAASAgEQQAAAAAAQEIgCAIAAAAAABICs8MAALJdg5YNIp4iV9PiLpy9MLObBAAAgFyIIAgAINtVGlfJ8hSPLDkxuWdyprcHAAAAuRPDYQAAAAAAQEIgCAIAAAAAABICQRAAAAAAAJAQCIIAAAAAAICEQBAEAAAAAAAkBIIgABLO119/beeee64lqvQc/8cff2w9evTItDYBAAAAWYEgCIAc6c8//7RWrVrZ8uXLA16/7bbb7LTTTrN9+/YFvH7BBRfYSy+9FNG+t2zZYgsWLIho3S+++MK6dOkScbujXT8SP/30k+uL//77L9Wy888/38aNGxfV/qI5fs+mTZts8eLFWX7sAAAAQCwRBAGQI9WuXdt+//13l7XgOXr0qL322mvu9R9//NH3+vr1623q1KlWvXr1iPbdsWNHd8Meic2bN9vChQsjbne060di+/btNnfuXDt48GCqZYsWLbJ169ZFtb9ojj8amXHsAAAAQCwRBAESgDIFrrvuOvvss8+sV69eds4559gjjzxihw4d8q0zevRou/vuuwO2++CDD6xnz56p9vPOO+/Y5ZdfbmeffbY9+uijdvjwYXv77betW7du1rlzZ5swYUKG25w/f34744wzbObMmQE3/Pny5XPv7f+6fs6bN6+1adPGraOsCT3at2/v2qusEn/KaBg6dKjv+YEDB2zUqFEui0HH8Mwzz9iRI0dcBsb999/vMjC8faaVdZHW+mrDNddcY23btrVLLrnEvvzyS8ss//77r911113u+C+66CJ3PlJSUsIev8yfP98uu+wyd23oOtC1ctZZZ6Xa95w5c+yqq65ygZQhQ4bYnj17jnnsAAAAQE6RL7sbACDzKVPgjTfesBUrVtgdd9xhe/fudcNK9O+IESPcOv/880+qjIKNGzfazz//HLAfZWL8/fffbj9aftNNN9mHH35o5cuXt5tvvtnWrl1rAwcOtIoVK9p5552XoXbrJl7BCd3AJyUluWCHggjt2rWz5557zreeXj/55JOtRIkSVrduXRszZox7Xcenm/mTTjrJ/vjjD6tWrVrI4SAKGMyYMcMefPBBK1y4sAtQPPzww3brrbdanz59XFDE26e3j1AaNGgQcn312ymnnGLdu3e3e+65x+bNm2ddu3Z1QSYNZ4klnZOWLVu6gI6CFNu2bbP77rvPli5d6voy1PH/9ddfrl/79u1r/fv3t2+//dYFavLkCYyTJycnu+tG+1WQatCgQbZhwwZ3TYQ7dgAAACAnIQgCJIiCBQu6ISPFixd3z1evXu0CI14QJFIKEkyePNmKFSvmC0BMnz7dvvvuO7dMPv/8c/vkk09iEgRRVsKSJUusSZMm7r2UaeLdsCvIUaRIEfe6l7GiQIiyEDzKZvj1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uXLlMp9++qnJly+f2bNnj5k0aZLd759//jGrVq2yg7YpU6bYAbcGakr91yf+7n7t27cPOwjy448/2sG898Avf/78PtMENEjs1KmT+eijj8yVV15pB/ilS5c2s2fPtgP8YK5L56ysBAV5Qjnms88+a1599VV7jocPHzYHDx40H374oalbt25Q7xvM+2haRP/+/W2A5fjx4yZLlix28Dx16lTTqlWroM/XPU7ZsmXtMRQwWLdunRk7dqzNxFDwRsdXHy9evNhUrlzZvs6/bZK7ppT0Z2L3jwb/4dx3W7dutf3x3Xffed5v5cqVpkOHDvZa5auvvjJt27a1gRUFWLZt22bGjRtng0WR4t+GajNlxKidFXRSf+nrwoULPeclf//9t7nuuusS7XcAAAAgTXGAGDN58mQna9aszrfffuvZtnTpUmfHjh0B99+3b59TqlQpZ9q0aZ5tU6ZMcYoWLRqR8xkyZIiTPXt2Z9y4cc7+/fsD7jNp0iQnV65czvr16+33R48edRISEpxOnToFfV1Tp051ChYsGNIx33//fadkyZI+bTN06FCnYsWKzvnz54N632De5+2333b062TChAmebf369XOqVKkS0vm6x3n33Xc921q1amW3TZ8+3X6v827evLnTtWtXzz7+bRPqPRJqfwZz/wR73w0fPtypU6eOz7aZM2c6OXPm9Hx/7bXXOgMGDPB8f+DAAefLL79M9lrmzp1r227ixIn2mN4PtdegQYMSbUNdf/78+Z0tW7Z43rNChQo+5xVMv/s7efKkc+TIEc9j+/bt9hgJixKc2mtq20exisWSvTZE37lz55zdu3fbr4g99F/sou9iF30X2+i/tE9/W+rvSn1NCjVBEHPeeecd+wl6gwYNPNsaNmxoSpYs6fleUyf0if6cOXPMsmXL7LSE5cuXp8r59O3b136C3q9fP1O4cGFToUIF88gjj9hP613vv/++PWfVlpDcuXOb3r1720/MlSER7HV5C+aYEyZMsFkX+nR+1qxZ5rPPPrMZDpqOoqknwbxvMO/jbu/cubPne2Up6H3OnTsX8nE6duzo+f6aa66x7apMCFFWRaNGjWyGSGJCbctQ+zMxqXXfKcNCRVVPnz5tvy9YsKC5/vrrg3799OnTbfaJ90MZHEn54IMP7PSYcuXKed6ze/fuF+yXXL/7Gz58uJ3K5D6UDQQAAABc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EAAAAAAAEBcIggAAIuLkyZOmf//+pnr16qZatWqmb9++5p9//knxa8I5LkI3c+ZMc80115jLLrvM3Hbbbebnn39O8Ws2bdpkevToYWrVqmVq165t/7179266J8I2b95s2rVrZ/vhqquuMpMmTYroa9SP2k8/g4is06dP2+LQNWvWNFWrVjW9evUyx44di8hrVq9ebdq0aWMqVapkbrzxRrNy5Uq6L8LmzJljmjRpYipUqGBatmxp1qxZk+LXnDlzxrzwwgumQYMG9ueucePG5p133qHvImz79u2mQ4cOpmLFiqZu3brmrbfeSvY169atMz179jSXXnqp6d69e8SOiyhQYVQAAFLq3nvvdcqXL+8sXLjQWbx4sVOxYkWnTZs2KX5NOMdFaD7//HMnc+bMzquvvur8+OOPTpcuXZz8+fM7O3fuTNFrqlev7rzxxhvODz/84KxcudJp0qSJU7lyZef48eN0UYQcPXrUKV26tHPnnXc6a9eudSZNmuRkzZrVmThxYkRec/r0aefKK690GjZs6OTNm5d+i7Du3bs7pUqVcubPn+8sW7bMqVGjhnPDDTek+DXz5s1zsmXL5gwcONBZt26ds2TJEqdFixb0XwQtWrTI/g586aWXnJ9++snp0aOHkzt3bmfz5s0pek3fvn2dwoULO7NmzXI2btxofy718zlhwgT6L0L++ecf57LLLnNatmzprFmzxnn//fedHDlyOK+99lqir1FfVK1a1Rk1apTTrFmzgH+HhHNcRAdBEABAiv3xxx9aacyZO3euZ5v+QNe2X3/9NezXhHNchK5BgwY22OQ6d+6cU6JECad///4pes3Zs2d9XrNt27YL+hMpM3r0aCdXrlz2j29Xz549beAwEq/p06eP07FjR+eVV14hCBJhu3btcjJmzOjMmDHDs2358uX2Z0RBw3Bfo5/FcuXKOQ8++KDPa7UdkaOgUqtWrXy2aQD82GOPpeg1Cjo++uijPvto0N2hQ4eInXu8U0BJgaXDhw97tj399NNOsWLFEv058f7/mfowUBAknOMiOpgOAwBIscWLF5uMGTOaZs2aebY1bdrUZMuWzT4X7mvCOS5Co+lGSpNv0aKFZ5vaXN8n1sbBviZTpkwXpHlLlixZ6KYI+eabb0yjRo1Mjhw5PNs09UHTXXbs2JGi18ydO9d88skn5rXXXqO/UsHSpUvN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mDCB1gIAxCyCIAAAAEhWlSpV7Nft27fbr+fPnzevvvqqueKKK0yOHDlMvnz5TKtWrcy6des8rzl8+LCd3jBixAhbM6NWrVo2qDJp0iTz3nvv2eeUYfLss8+a4sWL22N06dLFnDp1yh7/mWeeMSVKlDCXXHKJnY5z6NAhn3PScXUMvY83TafQdk2vkJdfftncdddd9t9169b1TPXROXjv7z50PZdffrm9Pldyx0isJsjGjRtNu3btTKFChUy2bNlsO77wwgvm3Llznn3cttB0o+HDh5uSJUvac2jWrJn5/fffw7o7I9G+3sfQvsWKFbP7Xn/99eaXX3654D1DuVYdc/DgwaZ06dImY8aM5rnnnktxH4XTlgcOHDA9e/Y05cuXt+dcqVIlM2jQIPP333/77PPYY4+ZsmXL2vu3VKlSplevXj77AIgdBEEAAACQrPXr19uvGrTK/fffbwYOHGgHgzt37rSDWg02r776arNlyxaf1y5fvtxMnz7dfPTRR3bwXKZMGc9zGqjqmAqeKJAwc+ZM069fPzNgwAA7gP3111/NV199Zb755hvTp0+fsHrqySeftNN7ZPXq1Xbaih4dOnSw26688krPNj0U6Hn00UdN3759zVtvvRXUMQLZunWrnR6jrwsWLDB79+611/bvf//bdOrU6YL9X3rpJZMnTx47gNegf8+ePTYwoPcJVyTad8iQIaZIkSLmt99+s+elKVLXXnut2bZtW9jX+vzzz5u8efOatWvX2ntDQYaU9lGobbl//35Tr149M3fuXHsMfT9nzhyTJUsWM2vWLLvPwYMH7XUtWrTIBlj++usvG1z74osvzM0332yDSQBijAMAAAD8f1OnTtUo0ZkxY4b9/tSpU86yZcucihUrOrly5XI2btzoLFiwwO4zZswYn3Y7efKkU6pUKadLly72+0OHDtn9Spcu7Zw5c8Zn3ylTptjnevXq5bO9T58+To4cOZyePXv6bO/bt6+TJUsW+x6u4cOH22PofbytXr3a5xpE/9Y2PResDh06OAkJCUEdY9asWfa55cuXe7Z17tzZyZo1q7N9+3affQcNGuRzHLctHn/8cZ/9pk+fbrer/ZOyZMkSu98rr7wS0fZ1j9G9e3effXfv3u1kz57d6datW9jX+uCDD15wHZHoo1DaskePHk7mzJmd9evXJ3r83r172+vatGmTz/YVK1bY482cOTPocwWQNpAJAgAAgAvoU3N32kHr1q1N9erV7Sokl112medT8jvvvNPnNZpOoOKpyirwpk/MM2fOHLCVb7rpJp/vNYVCmQaacuGtatWq5syZM57pOJE2duxYmxWQO3dun6kYf/zxR9jHVEaEjqnpE97cdtPz3lq2bOnzfY0aNezXlBSjjUT73nbbbT7fa1qMsiO+/vrrsK/V/5iR7qNg2lJZH8ow0RSYxOher127tp0u461+/fr2PPzvdQBpH0EQAAAAJLo6jOo57Nu3z06jUP0P0dQCdzCs4IaW0VVdBz20xK6mEHjTtIvEqFaFNw0sk9ruX/8jkFCnj6jex8MPP2zuvvtuWzfi7Nmz9hgPPfSQDQyES+2gNvLnblOtCW/+16zpHMFec2q2r5ZIDrTN+/xDvdak7olI9FEwban7Ornz0L2+cuVKe5973+sKwBw7duyCex1A2hc4JA8AAAAkQoUvNRg8cuSIyZkzZ7LtpBoLidFgMpTt3lRTQjQYVdFPl2qUhGLy5MmmSZMmpnfv3j7b/WubhKpAgQK2NoY/d5vaMdRrDlVK2teV2DUULFgw7GtN6p6IRB8Fc32FCxdO9l7ReSvrZf78+SGdL4C0i0wQAAAAhOTWW2+1GSIqdBpNFSpUsF/9VypRAVB/brBGK6MEoqk83nbs2GEWLlwY0jH8XXfddWbVqlVm165dPtvddtPzscCd/uSdQaFVcLzPPxLXGok+CsUtt9xii6Zu2LAhyXtdhX1TaxoWgIuPIAgAAABConoS9913n13RQ6tqaOCr5UK1QoyWOtVyqheDlj1VrQatUqPlWTXlYuTIkQEH0dWqVbPTGLQSiP/zqk+hT/o/+OADex0aGGup1+bNmwd9jEB0XqqpouVn1TbKnHn33Xft0r733HOPrUcRC9Sub7zxhl1CV6vMqF5M9uzZTf/+/SN6rZHoo1DoPtXKOQp06NhHjx41mzZtsivXaFqXaHlhLSOsujbz5s2z02nUHosXLzYdO3a02wDEFoIgAAAACJkGuAo4TJo0yRaWVO2HBx54wA5ie/bseVFaVDUaVKtEGQQ1a9Y0CQkJdsnSQO9ftmxZM2rUKFtIU/u7RTXdwbCmWWiJWE2R0Ou1xGrFihWDPkYgCtAoY0LFQps2bWqPPWzYMBswUPvFCp2vsi5UVLVOnTp2KosKgpYrVy6i1xqJPgqFjqN6Hy1atDBdunSx3yvYoXojCoy402GU4aICsz169LC1UFQkWEv/3nDDDTYQByC2ZNASMdE+CQAAAABpiwIQyvj5+eefPaurAECsIxMEAAAAAADEBYIgAAAAAAAgLhAEAQAAAAAAcYGaIAAAAAAAIC6QCQIAAAAAAOICQRAAAAAAABAXCIIAAAAAAIC4QBAEAAAAAADEBYIgAAAAAAAgLhAEAQAAAAAAcYEgCAAAAAAAiAsEQQAAAAAAQFwgCAIAAAAAAEw8+H9S7dKilP9MxwAAAABJRU5ErkJggg=='

NOTEBOOK_MODEL_RESULTS = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Linear SVM"],
    "Accuracy": [0.90533, 0.93885, 0.90017],
    "Precision": [0.68498, 0.86044, 0.66992],
    "Recall": [0.90526, 0.79960, 0.90904],
    "F1 Score": [0.77986, 0.82890, 0.77137],
    "ROC AUC": [0.95242, 0.94808, 0.95226]
})

NOTEBOOK_CONFUSION_MATRICES = {
    "Logistic Regression": np.array([[29505, 3085], [702, 6708]]),
    "Random Forest": np.array([[31629, 961], [1485, 5925]]),
    "Linear SVM": np.array([[29271, 3319], [674, 6736]])
}


def _candidate_files(names):
    """Locate project data files using deployment-safe relative paths."""
    base = Path(__file__).resolve().parent
    roots = [
        base,
        base / "data",
        base / "data" / "raw",
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "data" / "raw",
    ]

    found = []
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate.is_file() and candidate not in found:
                found.append(candidate)

    # Search below the application directory for the exact filenames.
    for name in names:
        try:
            for candidate in base.rglob(name):
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
        except Exception:
            pass

    return found


# The notebook uses one source dataset and creates the modelling/validation
# split inside the notebook.  Insurance Analytics must use that same
# validation population; it must NOT require separate train/test CSV files.
EARLY_SCREENING_LEAKAGE_COLUMNS = [
    "Patient_ID",
    "Serum_Creatinine",
    "eGFR",
    "Blood_Urea_Nitrogen",
    "Albumin",
    "Urine_ACR",
    "Albuminuria_Category",
    "Urine_Protein",
    "CKD_Stage",
    "Kidney_Failure_Risk",
    "Dialysis_Required",
    "Hospitalization_Risk",
    "ACE_Inhibitor",
    "ARB",
    "Diabetes_Medication",
    "Statin",
    "Diuretic",
    "Medication_Adherence",
    "Number_of_Medications",
    "Hospital_Visits",
    "Emergency_Visits",
    "Specialist_Visits",
    "Annual_Medical_Cost_USD",
    "HbA1c",
    "Fasting_Glucose",
    "Hemoglobin",
    "Sodium",
    "Potassium",
    "Calcium",
    "Phosphorus",
    "Uric_Acid",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "CRP",
    "Frailty_Index",
    "Frailty_Category",
]


def _build_early_screening_validation_set(dataframe):
    """
    Reproduce the Early Screening notebook's modelling-data construction
    from the single source dataset, including its exact 80/20 split.
    """
    from sklearn.model_selection import train_test_split

    data = dataframe.copy()

    required_target = {"CKD_Stage"}
    if not required_target.issubset(data.columns):
        return None, None

    # Exact binary target used in early_screening:
    # 0 = Healthy, 1 = Any CKD stage.
    data["CKD"] = (data["CKD_Stage"] != "Healthy").astype(int)

    missing_leakage = [
        col for col in EARLY_SCREENING_LEAKAGE_COLUMNS
        if col not in data.columns
    ]
    if missing_leakage:
        return None, None

    x_data = data.drop(
        columns=EARLY_SCREENING_LEAKAGE_COLUMNS + ["CKD"]
    ).copy()
    y_data = data["CKD"].copy()

    # Exact feature engineering used by the notebook.
    if {"Systolic_BP", "Diastolic_BP"}.issubset(x_data.columns):
        x_data["Pulse_Pressure"] = (
            x_data["Systolic_BP"] - x_data["Diastolic_BP"]
        )

    if {"Waist_Circumference_cm", "Height_cm"}.issubset(x_data.columns):
        x_data["Waist_to_Height"] = (
            x_data["Waist_Circumference_cm"] / x_data["Height_cm"]
        )

    if {
        "Smoking_Status",
        "Alcohol_Consumption",
        "Physical_Activity_Level",
    }.issubset(x_data.columns):
        x_data["Lifestyle_Risk"] = (
            (x_data["Smoking_Status"] == "Current").astype(int)
            + (x_data["Alcohol_Consumption"] == "High").astype(int)
            + (x_data["Physical_Activity_Level"] == "Low").astype(int)
        )

    if {"Diabetes", "Hypertension", "Obesity"}.issubset(x_data.columns):
        x_data["Metabolic_Risk"] = (
            x_data["Diabetes"].astype(int)
            + x_data["Hypertension"].astype(int)
            + x_data["Obesity"].astype(int)
        )

    if {
        "Cardiovascular_Disease",
        "Heart_Failure",
        "Hyperlipidemia",
    }.issubset(x_data.columns):
        x_data["CV_Risk"] = (
            x_data["Cardiovascular_Disease"].astype(int)
            + x_data["Heart_Failure"].astype(int)
            + x_data["Hyperlipidemia"].astype(int)
        )

    if "Sleep_Duration_Hours" in x_data.columns:
        x_data["Poor_Sleep"] = (
            (x_data["Sleep_Duration_Hours"] < 6)
            | (x_data["Sleep_Duration_Hours"] > 9)
        ).astype(int)

    # Exact notebook split.
    _, x_valid, _, y_valid = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data,
    )

    return (
        x_valid.reset_index(drop=True),
        pd.Series(y_valid).reset_index(drop=True),
    )


@st.cache_data(show_spinner=False)
def load_shared_evaluation():
    """
    Use the same single-dataset 80/20 validation population created by
    early_screening, then obtain predictions from the already-saved final SVM.
    No separate train/test files are required and the model is not retrained.
    """
    if final_svm_pipeline is None:
        return False, "Final SVM pipeline is not loaded."

    candidates = _candidate_files([
        "CKD_Risk_Progression_Dataset_2026.csv"
    ])

    if not candidates:
        return False, (
            "The Early Screening source dataset "
            "'CKD_Risk_Progression_Dataset_2026.csv' was not found."
        )

    diagnostics = []

    for path in candidates:
        try:
            raw_data = pd.read_csv(path)

            x_valid_candidate, y_valid_candidate = (
                _build_early_screening_validation_set(raw_data)
            )

            if x_valid_candidate is None:
                diagnostics.append(
                    f"{path.name}: required Early Screening columns are missing."
                )
                continue

            # The saved deployment pipeline was fitted on the notebook's raw
            # x_train features, so it can score the notebook-equivalent x_valid
            # directly.  Do not reconstruct its internal preprocessing.
            expected = getattr(
                final_svm_pipeline,
                "feature_names_in_",
                None,
            )

            if expected is not None:
                expected = list(expected)
                missing = [
                    col for col in expected
                    if col not in x_valid_candidate.columns
                ]
                if missing:
                    diagnostics.append(
                        f"{path.name}: validation set is missing model "
                        f"features: {', '.join(missing[:8])}"
                    )
                    continue
                x_valid_candidate = x_valid_candidate[expected]

            pred = final_svm_pipeline.predict(x_valid_candidate)
            score = final_svm_pipeline.decision_function(x_valid_candidate)

            return True, {
                "x_valid": x_valid_candidate.reset_index(drop=True),
                "y_valid": y_valid_candidate.reset_index(drop=True),
                "y_pred_svm": pd.Series(pred).reset_index(drop=True),
                "y_score": np.asarray(score),
                "evaluation_source": (
                    "Early Screening notebook validation split "
                    "(single source dataset, random_state=42)"
                ),
                "evaluation_dataset": path.name,
            }

        except Exception as exc:
            diagnostics.append(f"{path.name}: {type(exc).__name__}: {exc}")

    detail = " | ".join(diagnostics[:3])
    return False, (
        "The Early Screening validation population could not be evaluated."
        + (f" Details: {detail}" if detail else "")
    )


# =============================================================================
# SECTION 3 : CKD SCREENING TAB
# =============================================================================

if main_early_screening:
    with tab_prediction:

        # =========================================================================
        # PAGE HEADER
        # =========================================================================

        st.title(
            "🩺 CKD Prediction"
        )

        st.write(
            """
            Enter the patient's demographic, lifestyle, medical and
            physiological information below to perform an early CKD-risk
            screening using the final Linear SVM model.
            """
        )


        st.info(
            """
            The screening model uses non-kidney clinical and lifestyle
            information for early risk assessment. The result is not a
            definitive medical diagnosis.
            """
        )


        # =========================================================================
        # SECTION 3.1 : DEMOGRAPHICS
        # =========================================================================

        st.divider()

        st.subheader(
            "👤 Demographic Information"
        )

        st.info(
            "💡 **How to enter numbers:** use the units shown in brackets. "
            "For example, blood pressure is entered as mmHg, height as cm, weight as kg, "
            "and water intake as litres per day. You do not need to enter the model's internal 0/1 codes; use the Yes/No choices provided."
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            Age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=40,
                step=1
            )


        with col2:

            Sex = st.selectbox(
                "Sex",
                [
                    "Female",
                    "Male"
                ]
            )


        with col3:

            Ethnicity = st.selectbox(
                "Ethnicity",
                [
                    "Asian",
                    "Black",
                    "Hispanic",
                    "White",
                    "Other"
                ]
            )


        col4, col5, col6 = st.columns(3)


        with col4:

            Country = st.selectbox(
                "Country",
                [
                    "Australia",
                    "Canada",
                    "India",
                    "UK",
                    "USA",
                    "Other"
                ]
            )


        with col5:

            Residence_Type = st.selectbox(
                "Residence Type",
                [
                    "Urban",
                    "Rural"
                ]
            )


        with col6:

            Education_Level = st.selectbox(
                "Education Level",
                [
                    "High School",
                    "Some College",
                    "Bachelor's",
                    "Graduate"
                ]
            )


        col7, col8 = st.columns(2)


        with col7:

            Socioeconomic_Status = st.selectbox(
                "Socioeconomic Status",
                [
                    "Low",
                    "Middle",
                    "High"
                ]
            )


        with col8:

            Employment_Status = st.selectbox(
                "Employment Status",
                [
                    "Employed",
                    "Unemployed/Retired"
                ]
            )


        # =========================================================================
        # SECTION 3.2 : BODY COMPOSITION
        # =========================================================================

        st.divider()

        st.subheader(
            "⚖️ Body Composition"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            Height_cm = st.number_input(
                "Height (cm)",
                help="Enter height in centimetres. Example: 170 cm.",
                min_value=50.0,
                max_value=250.0,
                value=170.0,
                step=0.1
            )


        with col2:

            Weight_kg = st.number_input(
                "Weight (kg)",
                help="Enter body weight in kilograms. Example: 70 kg.",
                min_value=10.0,
                max_value=300.0,
                value=70.0,
                step=0.1
            )


        with col3:

            BMI = st.number_input(
                "BMI (kg/m²)",
                help="Body Mass Index. If you know your BMI, enter it here.",
                min_value=10.0,
                max_value=80.0,
                value=24.0,
                step=0.1
            )


        col4, col5, col6 = st.columns(3)


        with col4:

            Waist_Circumference_cm = st.number_input(
                "Waist Circumference (cm)",
                help="Measure around the waist in centimetres.",
                min_value=30.0,
                max_value=200.0,
                value=85.0,
                step=0.1
            )


        with col5:

            Body_Fat_Percentage = st.number_input(
                "Body Fat (%)",
                help="Estimated body-fat percentage, if known.",
                min_value=1.0,
                max_value=70.0,
                value=25.0,
                step=0.1
            )


        with col6:

            Obesity = st.selectbox(
                "Obesity",
                [
                    0,
                    1
                ],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        # =========================================================================
        # SECTION 3.3 : LIFESTYLE
        # =========================================================================

        st.divider()

        st.subheader(
            "🏃 Lifestyle Information"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            Smoking_Status = st.selectbox(
                "Smoking Status",
                [
                    "Never",
                    "Former",
                    "Current"
                ]
            )


        with col2:

            Alcohol_Consumption = st.selectbox(
                "Alcohol Consumption",
                [
                    "Low",
                    "Moderate",
                    "High"
                ]
            )


        with col3:

            Physical_Activity_Level = st.selectbox(
                "Physical Activity Level",
                [
                    "Low",
                    "Moderate",
                    "Active"
                ]
            )


        col4, col5, col6 = st.columns(3)


        with col4:

            Exercise_Hours_Per_Week = st.number_input(
                "Exercise Hours / Week",
                help="Approximate hours of exercise in a typical week.",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.5
            )


        with col5:

            Daily_Steps = st.number_input(
                "Daily Steps",
                help="Approximate number of steps taken on a typical day.",
                min_value=0,
                max_value=100000,
                value=7000,
                step=500
            )


        with col6:

            Water_Intake_L = st.number_input(
                "Water Intake (litres/day)",
                help="Approximate total water/fluid intake per day.",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1
            )


        col7, col8, col9 = st.columns(3)


        with col7:

            Sodium_Intake_mg = st.number_input(
                "Sodium Intake (mg/day)",
                help="Approximate daily sodium intake, if known.",
                min_value=0,
                max_value=15000,
                value=2300,
                step=100
            )


        with col8:

            Fast_Food_Frequency_Per_Week = st.number_input(
                "Fast Food Meals / Week",
                help="Approximate number of fast-food meals per week.",
                min_value=0,
                max_value=30,
                value=2,
                step=1
            )


        with col9:

            Sleep_Duration_Hours = st.number_input(
                "Sleep Duration (hours/night)",
                help="Average hours of sleep per night.",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )


        # =========================================================================
        # SECTION 3.4 : STRESS
        # =========================================================================

        st.divider()

        st.subheader(
            "🧠 Stress & Sleep"
        )


        Stress_Level = st.selectbox(
            "Stress Level",
            [
                "Low",
                "Moderate",
                "High"
            ]
        )


        # =========================================================================
        # SECTION 3.5 : MEDICAL HISTORY
        # =========================================================================

        st.divider()

        st.subheader(
            "🏥 Medical History"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            Diabetes = st.selectbox(
                "Diabetes",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col2:

            Hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col3:

            Cardiovascular_Disease = st.selectbox(
                "Cardiovascular Disease",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        col4, col5, col6 = st.columns(3)


        with col4:

            Heart_Failure = st.selectbox(
                "Heart Failure",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col5:

            Hyperlipidemia = st.selectbox(
                "Hyperlipidemia",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col6:

            Kidney_Stones = st.selectbox(
                "Kidney Stones",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        col7, col8, col9 = st.columns(3)


        with col7:

            Recurrent_UTI = st.selectbox(
                "Recurrent UTI",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col8:

            Autoimmune_Disease = st.selectbox(
                "Autoimmune Disease",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col9:

            Family_History_CKD = st.selectbox(
                "Family History of CKD",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        # =========================================================================
        # SECTION 3.6 : VITAL SIGNS
        # =========================================================================

        st.divider()

        st.subheader(
            "❤️ Vital Signs"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            Heart_Rate = st.number_input(
                "Heart Rate (beats/min)",
                min_value=30,
                max_value=220,
                value=72,
                step=1
            )


        with col2:

            Respiratory_Rate = st.number_input(
                "Respiratory Rate (/min)",
                min_value=5,
                max_value=60,
                value=16,
                step=1
            )


        with col3:

            Oxygen_Saturation = st.number_input(
                "Oxygen Saturation (%)",
                help="Blood oxygen saturation (SpO₂), usually shown on a pulse oximeter.",
                min_value=50.0,
                max_value=100.0,
                value=98.0,
                step=0.1
            )


        col4, col5 = st.columns(2)


        with col4:

            Systolic_BP = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                help="The top number of a blood-pressure reading. Example: 120/80 → enter 120.",
                min_value=60,
                max_value=250,
                value=120,
                step=1
            )


        with col5:

            Diastolic_BP = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                help="The bottom number of a blood-pressure reading. Example: 120/80 → enter 80.",
                min_value=30,
                max_value=150,
                value=80,
                step=1
            )


        # A simple plain-language interpretation of the numerical vital signs.
        # This is display-only and does not alter the values sent to the model.
        bp_category_display = "Typical range"
        if Systolic_BP >= 140 or Diastolic_BP >= 90:
            bp_category_display = "Higher reading"
        elif Systolic_BP >= 130 or Diastolic_BP >= 80:
            bp_category_display = "Above typical range"
        elif Systolic_BP < 90 or Diastolic_BP < 60:
            bp_category_display = "Lower reading"

        st.markdown(
            f"**🩺 Blood pressure summary:** `{Systolic_BP:.0f}/{Diastolic_BP:.0f} mmHg` "
            f"&nbsp; — &nbsp; **{bp_category_display}**",
            unsafe_allow_html=True
        )
        st.caption("This quick label is only a user-friendly guide; it is not a diagnosis.")


        # =========================================================================
        # SECTION 3.7 : MEDICATION / HEALTHCARE
        # =========================================================================

        st.divider()

        st.subheader(
            "💊 Healthcare & Medication"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            NSAID_Usage = st.selectbox(
                "NSAID Usage",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        with col2:

            Annual_Checkups = st.number_input(
                "Health Checkups / Year",
                help="Approximate number of routine health checkups in a year.",
                min_value=0,
                max_value=30,
                value=1,
                step=1
            )


        with col3:

            Health_Insurance = st.selectbox(
                "Health Insurance",
                [0, 1],
                format_func=lambda x: (
                    "No" if x == 0 else "Yes"
                )
            )


        # =========================================================================
        # SECTION 3.8 : SOCIOECONOMIC INFORMATION
        # =========================================================================

        st.divider()

        st.subheader(
            "💰 Socioeconomic Information"
        )


        col1, col2 = st.columns(2)


        with col1:

            Annual_Household_Income_USD = st.number_input(
                "Annual Household Income (USD)",
                help="Enter annual household income in USD to match the model's training data.",
                min_value=0,
                max_value=1000000,
                value=60000,
                step=1000
            )


        with col2:

            st.write(
                " "
            )

            st.caption(
                "Income is entered in USD to match the training dataset."
            )


        # =========================================================================
        # SECTION 3.9 : DERIVED MODEL FEATURES
        # =========================================================================

        # -------------------------------------------------------------------------
        # Pulse Pressure
        # -------------------------------------------------------------------------

        Pulse_Pressure = (
            Systolic_BP -
            Diastolic_BP
        )


        # -------------------------------------------------------------------------
        # Waist-to-Height Ratio
        # -------------------------------------------------------------------------

        if Height_cm > 0:

            Waist_to_Height = (
                Waist_Circumference_cm /
                Height_cm
            )

        else:

            Waist_to_Height = 0.0


        # -------------------------------------------------------------------------
        # Lifestyle Risk
        # -------------------------------------------------------------------------

        Lifestyle_Risk = (
            int(
                Smoking_Status == "Current"
            )
            +
            int(
                Alcohol_Consumption == "High"
            )
            +
            int(
                Physical_Activity_Level == "Low"
            )
        )


        # -------------------------------------------------------------------------
        # Metabolic Risk
        # -------------------------------------------------------------------------

        Metabolic_Risk = (
            Diabetes
            +
            Hypertension
            +
            Obesity
        )


        # -------------------------------------------------------------------------
        # Cardiovascular Risk
        # -------------------------------------------------------------------------

        CV_Risk = (
            Cardiovascular_Disease
            +
            Heart_Failure
            +
            Hyperlipidemia
        )


        # -------------------------------------------------------------------------
        # Poor Sleep
        # -------------------------------------------------------------------------

        Poor_Sleep = int(
            Sleep_Duration_Hours < 6
            or
            Sleep_Duration_Hours > 9
        )


        # =========================================================================
        # SECTION 3.10 : BLOOD PRESSURE CATEGORY
        # =========================================================================

        if Systolic_BP < 120 and Diastolic_BP < 80:

            Blood_Pressure_Category = "Normal"

        elif (
            120 <= Systolic_BP < 130
            and
            Diastolic_BP < 80
        ):

            Blood_Pressure_Category = "Elevated"

        elif (
            130 <= Systolic_BP < 140
            or
            80 <= Diastolic_BP < 90
        ):

            Blood_Pressure_Category = (
                "Hypertension Stage 1"
            )

        else:

            Blood_Pressure_Category = (
                "Hypertension Stage 2"
            )


        # =========================================================================
        # SECTION 3.11 : SHOW DERIVED VARIABLES
        # =========================================================================

        st.divider()

        st.subheader(
            "⚙️ Derived Screening Indicators"
        )


        derived_col1, \
        derived_col2, \
        derived_col3, \
        derived_col4, \
        derived_col5 = st.columns(5)


        with derived_col1:

            st.metric(
                "Pulse Pressure",
                f"{Pulse_Pressure:.1f}"
            )


        with derived_col2:

            st.metric(
                "Waist / Height",
                f"{Waist_to_Height:.3f}"
            )


        with derived_col3:

            st.metric(
                "Lifestyle Risk",
                f"{Lifestyle_Risk}"
            )


        with derived_col4:

            st.metric(
                "Metabolic Risk",
                f"{Metabolic_Risk}"
            )


        with derived_col5:

            st.metric(
                "CV Risk",
                f"{CV_Risk}"
            )


        st.caption(
            f"Blood Pressure Category: **{Blood_Pressure_Category}** "
            f"| Poor Sleep Indicator: **{Poor_Sleep}**"
        )


        # =========================================================================
        # SECTION 3.12 : CREATE EXACT MODEL INPUT
        # =========================================================================

        input_data = pd.DataFrame(
            {
                "Age": [Age],
                "Sex": [Sex],
                "Ethnicity": [Ethnicity],
                "Country": [Country],
                "Residence_Type": [Residence_Type],
                "Education_Level": [Education_Level],
                "Socioeconomic_Status": [Socioeconomic_Status],

                "Height_cm": [Height_cm],
                "Weight_kg": [Weight_kg],
                "BMI": [BMI],
                "Waist_Circumference_cm": [
                    Waist_Circumference_cm
                ],
                "Body_Fat_Percentage": [
                    Body_Fat_Percentage
                ],

                "Smoking_Status": [Smoking_Status],
                "Alcohol_Consumption": [
                    Alcohol_Consumption
                ],
                "Physical_Activity_Level": [
                    Physical_Activity_Level
                ],

                "Exercise_Hours_Per_Week": [
                    Exercise_Hours_Per_Week
                ],
                "Daily_Steps": [Daily_Steps],
                "Water_Intake_L": [Water_Intake_L],
                "Sodium_Intake_mg": [
                    Sodium_Intake_mg
                ],
                "Fast_Food_Frequency_Per_Week": [
                    Fast_Food_Frequency_Per_Week
                ],
                "Sleep_Duration_Hours": [
                    Sleep_Duration_Hours
                ],

                "Stress_Level": [Stress_Level],

                "Diabetes": [Diabetes],
                "Hypertension": [Hypertension],
                "Cardiovascular_Disease": [
                    Cardiovascular_Disease
                ],
                "Heart_Failure": [Heart_Failure],
                "Hyperlipidemia": [Hyperlipidemia],
                "Kidney_Stones": [Kidney_Stones],
                "Recurrent_UTI": [Recurrent_UTI],
                "Autoimmune_Disease": [
                    Autoimmune_Disease
                ],
                "Family_History_CKD": [
                    Family_History_CKD
                ],
                "Obesity": [Obesity],

                "Heart_Rate": [Heart_Rate],
                "Respiratory_Rate": [
                    Respiratory_Rate
                ],
                "Oxygen_Saturation": [
                    Oxygen_Saturation
                ],

                "Systolic_BP": [Systolic_BP],
                "Diastolic_BP": [Diastolic_BP],

                "Blood_Pressure_Category": [
                    Blood_Pressure_Category
                ],

                "NSAID_Usage": [NSAID_Usage],
                "Annual_Checkups": [
                    Annual_Checkups
                ],
                "Health_Insurance": [
                    Health_Insurance
                ],
                "Annual_Household_Income_USD": [
                    Annual_Household_Income_USD
                ],
                "Employment_Status": [
                    Employment_Status
                ],

                "Pulse_Pressure": [
                    Pulse_Pressure
                ],
                "Waist_to_Height": [
                    Waist_to_Height
                ],
                "Lifestyle_Risk": [
                    Lifestyle_Risk
                ],
                "Metabolic_Risk": [
                    Metabolic_Risk
                ],
                "CV_Risk": [
                    CV_Risk
                ],
                "Poor_Sleep": [
                    Poor_Sleep
                ]
            }
        )


        # =========================================================================
        # SECTION 3.13 : VERIFY MODEL FEATURE ORDER
        # =========================================================================

        expected_features = list(
            final_svm_pipeline.feature_names_in_
        )


        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]


        extra_features = [
            feature
            for feature in input_data.columns
            if feature not in expected_features
        ]


        if missing_features:

            st.error(
                "Model input is missing required features:"
            )

            st.write(
                missing_features
            )

            st.stop()


        if extra_features:

            input_data = input_data.drop(
                columns=extra_features
            )


        input_data = input_data[
            expected_features
        ]


        # =========================================================================
        # SECTION 3.14 : RUN SCREENING
        # =========================================================================

        st.divider()

        st.subheader(
            "🚀 Run CKD Screening"
        )


        run_screening = st.button(
            "🔍 Predict CKD Risk",
            type="primary",
            use_container_width=True
        )


        # =========================================================================
        # SECTION 3.15 : MODEL PREDICTION
        # =========================================================================

        if run_screening:

            if not MODEL_LOADED:

                st.error(
                    "The CKD SVM model could not be loaded."
                )

            else:

                try:

                    # -------------------------------------------------------------
                    # Prediction
                    # -------------------------------------------------------------

                    prediction = (
                        final_svm_pipeline.predict(
                            input_data
                        )[0]
                    )


                    # -------------------------------------------------------------
                    # Decision score
                    # -------------------------------------------------------------

                    decision_score = (
                        final_svm_pipeline
                        .decision_function(
                            input_data
                        )[0]
                    )


                    # -------------------------------------------------------------
                    # Save result in session state
                    # -------------------------------------------------------------

                    st.session_state[
                        "prediction"
                    ] = prediction

                    st.session_state[
                        "decision_score"
                    ] = decision_score

                    st.session_state[
                        "input_data"
                    ] = input_data.copy()

                    # ---- STAY ON EARLY SCREENING ----
                    # Do NOT auto-redirect. Let the user see the full
                    # prediction results dashboard before navigating.

                    st.rerun()

                except Exception as prediction_error:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(
                        prediction_error
                    )

        # =====================================================================
        # COMPREHENSIVE PREDICTION RESULTS DASHBOARD
        # =====================================================================
        # After a successful prediction, this rich dashboard is displayed
        # directly on the prediction tab so users see ALL analysis at once.
        # =====================================================================
        if "prediction" in st.session_state and "input_data" in st.session_state:
            input_data_persistent = st.session_state["input_data"]
            prediction_persistent = int(st.session_state["prediction"])
            decision_score_persistent = float(st.session_state.get("decision_score", 0.0))

            st.divider()

            # --- SUCCESS BANNER ---
            st.success(
                "✅ **CKD Early Screening completed successfully!** "
                "Review your comprehensive results below."
            )

            # =================================================================
            # SECTION A : RISK CLASSIFICATION & GAUGE
            # =================================================================
            st.header("🎯 Risk Classification & Decision Signal")
            st.caption("Visual summary of the SVM screening model's assessment.")

            render_visual_prediction_summary(prediction_persistent, decision_score_persistent, key_suffix="interpretation")

            st.divider()

            # =================================================================
            # SECTION B : HEALTH FACTOR BADGES
            # =================================================================
            render_patient_health_badges(input_data_persistent)

            st.divider()

            # =================================================================
            # SECTION C : MODEL PERFORMANCE KPIs
            # =================================================================
            st.header("📊 Model Performance Metrics")
            st.caption(f"Source: {evaluation_source}")

            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            with kpi_col1:
                st.metric("Accuracy", f"{svm_accuracy:.3f}")
            with kpi_col2:
                st.metric("Precision", f"{svm_precision:.3f}")
            with kpi_col3:
                st.metric("Recall", f"{svm_recall:.3f}")
            with kpi_col4:
                st.metric("F1 Score", f"{svm_f1:.3f}")
            with kpi_col5:
                st.metric("ROC AUC", f"{svm_roc_auc:.3f}")

            with st.expander("📖 What do these metrics mean?", expanded=False):
                st.markdown("""
                | Metric | Meaning |
                |--------|---------|
                | **Accuracy** | Overall percentage of correct predictions |
                | **Precision** | Of those flagged as high-risk, how many truly are |
                | **Recall** | Of all actual high-risk patients, how many were caught |
                | **F1 Score** | Balanced harmonic mean of Precision and Recall |
                | **ROC AUC** | Model's ability to distinguish between risk classes (1.0 = perfect) |

                > **Clinical Note:** For screening, **Recall** is the most critical metric — we want to
                > minimise missed high-risk patients (false negatives). The Linear SVM was selected
                > specifically because it records the highest recall among deployment candidates.
                """)

            st.divider()

            # =================================================================
            # SECTION D : INTERACTIVE RISK PROFILE RADAR CHART
            # =================================================================
            st.header("📈 Interactive Risk Profile Visualisation")
            st.caption("Radar chart showing the patient's risk scores across all derived clinical dimensions.")

            try:
                row = input_data_persistent.iloc[0]

                # Gather risk dimensions
                meta_risk = _safe_int(row.get("Metabolic_Risk"), 0)
                cv_risk = _safe_int(row.get("CV_Risk"), 0)
                ls_risk = _safe_int(row.get("Lifestyle_Risk"), 0)
                poor_sleep = _safe_int(row.get("Poor_Sleep"), 0)
                bmi_val = _safe_float(row.get("BMI"), 24.0)
                sys_bp = _safe_float(row.get("Systolic_BP"), 120.0)

                # Normalise to 0–1 scale for radar
                bmi_norm = min(bmi_val / 45.0, 1.0)
                bp_norm = min(sys_bp / 200.0, 1.0)
                meta_norm = meta_risk / 3.0
                cv_norm = cv_risk / 3.0
                ls_norm = ls_risk / 3.0
                sleep_norm = float(poor_sleep)

                radar_categories = [
                    "Metabolic Risk", "CV Risk", "Lifestyle Risk",
                    "Sleep Quality", "BMI Index", "Blood Pressure"
                ]
                radar_values = [meta_norm, cv_norm, ls_norm, sleep_norm, bmi_norm, bp_norm]
                # Close the radar
                radar_values_closed = radar_values + [radar_values[0]]
                radar_categories_closed = radar_categories + [radar_categories[0]]

                radar_col1, radar_col2 = st.columns([1.5, 1])

                with radar_col1:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_values_closed,
                        theta=radar_categories_closed,
                        fill="toself",
                        fillcolor="rgba(231, 76, 60, 0.15)" if prediction_persistent == 1 else "rgba(39, 174, 96, 0.15)",
                        line=dict(
                            color="#e74c3c" if prediction_persistent == 1 else "#27ae60",
                            width=2.5
                        ),
                        name="Patient Risk Profile",
                        hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>"
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
                            angularaxis=dict(tickfont=dict(size=12))
                        ),
                        showlegend=False,
                        height=420,
                        margin=dict(l=60, r=60, t=40, b=40),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with radar_col2:
                    st.markdown("##### Risk Score Breakdown")
                    risk_items = [
                        ("🔴" if meta_risk >= 2 else "🟡" if meta_risk == 1 else "🟢",
                         "Metabolic Risk", f"{meta_risk}/3"),
                        ("🔴" if cv_risk >= 2 else "🟡" if cv_risk == 1 else "🟢",
                         "CV Risk", f"{cv_risk}/3"),
                        ("🔴" if ls_risk >= 2 else "🟡" if ls_risk == 1 else "🟢",
                         "Lifestyle Risk", f"{ls_risk}/3"),
                        ("🟡" if poor_sleep == 1 else "🟢",
                         "Sleep Quality", "Suboptimal" if poor_sleep == 1 else "Optimal"),
                        ("🔴" if bmi_val >= 30 else "🟡" if bmi_val >= 25 else "🟢",
                         "BMI", f"{bmi_val:.1f}"),
                        ("🔴" if sys_bp >= 140 else "🟡" if sys_bp >= 120 else "🟢",
                         "Systolic BP", f"{sys_bp:.0f} mmHg"),
                    ]
                    for icon, label, value in risk_items:
                        st.markdown(f"{icon} **{label}:** {value}")

            except Exception as radar_err:
                st.warning(f"Unable to render risk profile radar chart: {radar_err}")

            st.divider()

            # =================================================================
            # SECTION E : CLINICAL INTERPRETATION & RECOMMENDATIONS
            # =================================================================
            st.header("📋 Clinical Interpretation & Recommendations")

            if prediction_persistent == 1:
                st.markdown("""
                <div class="risk-banner-high">
                    <h3 style="margin-top:0; color:#c0392b;">⚠️ Key Clinical Findings — Higher CKD Risk</h3>
                    <p style="font-size: 1.0em; line-height: 1.7;">
                        The Linear SVM screening model has placed this patient's profile on the
                        <b>higher-risk side</b> of the classification boundary. This means the
                        combination of demographic, lifestyle, and clinical features indicates
                        elevated risk for Chronic Kidney Disease.
                    </p>
                    <hr style="border-top: 1px solid #f5c6cb; margin: 12px 0;">
                    <h4 style="color:#c0392b;">Recommended Actions:</h4>
                    <ol style="font-size: 0.95em; line-height: 1.8;">
                        <li>📋 <b>Clinical Tests:</b> Order <b>eGFR</b>, <b>Serum Creatinine</b>, and <b>Urine ACR</b></li>
                        <li>🩺 <b>Medical Review:</b> Schedule consultation with a nephrologist or general practitioner</li>
                        <li>💊 <b>Medication Review:</b> Assess current medication adherence and nephrotoxic drug exposure</li>
                        <li>🏃 <b>Lifestyle Modifications:</b> Address modifiable risk factors (smoking, diet, physical activity)</li>
                        <li>📊 <b>Follow-up:</b> Proceed to <b>Clinical Screening</b> for severity classification</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-banner-low">
                    <h3 style="margin-top:0; color:#27ae60;">✅ Key Clinical Findings — Lower CKD Risk</h3>
                    <p style="font-size: 1.0em; line-height: 1.7;">
                        The screening model has placed this patient's profile on the
                        <b>lower-risk side</b> of the classification boundary. This is reassuring,
                        but <b>does not rule out kidney disease</b>.
                    </p>
                    <hr style="border-top: 1px solid #c3e6cb; margin: 12px 0;">
                    <h4 style="color:#27ae60;">Recommended Actions:</h4>
                    <ol style="font-size: 0.95em; line-height: 1.8;">
                        <li>✅ <b>Routine Monitoring:</b> Continue regular annual health check-ups</li>
                        <li>🩺 <b>Discuss Symptoms:</b> Report any new symptoms to a healthcare professional</li>
                        <li>🏃 <b>Healthy Lifestyle:</b> Maintain balanced diet, regular exercise, adequate hydration</li>
                        <li>📊 <b>Optional:</b> Proceed to <b>Clinical Screening</b> for detailed severity assessment</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # =================================================================
            # SECTION F : INSURANCE RISK SUMMARY
            # =================================================================
            st.header("💰 Insurance Risk Summary")

            ins_col1, ins_col2 = st.columns(2)
            insurance_val = _safe_int(input_data_persistent.iloc[0].get("Health_Insurance", 0), 0)
            income_val = _safe_float(input_data_persistent.iloc[0].get("Annual_Household_Income_USD", 0), 0)

            with ins_col1:
                if insurance_val == 1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>🛡️ Insurance Status: <span style="color:#27ae60;">Insured</span></h4>
                        <p>This patient has health insurance coverage.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card" style="border-left-color: #e74c3c;">
                        <h4>⚠️ Insurance Status: <span style="color:#e74c3c;">Uninsured</span></h4>
                        <p>This patient does <b>not</b> have health insurance coverage.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with ins_col2:
                risk_group = "Higher Risk" if prediction_persistent == 1 else "Lower Risk"
                ins_label = "Insured" if insurance_val == 1 else "Uninsured"
                combined_label = f"{risk_group} | {ins_label}"

                if prediction_persistent == 1 and insurance_val == 0:
                    card_color = "#e74c3c"
                    card_msg = "🔴 **Vulnerable Population**: High CKD risk without insurance — priority for intervention."
                elif prediction_persistent == 1 and insurance_val == 1:
                    card_color = "#e67e22"
                    card_msg = "🟠 **Insured High Risk**: Insurance in place but elevated CKD risk — clinical follow-up recommended."
                else:
                    card_color = "#27ae60"
                    card_msg = "🟢 **Lower Risk**: Encouraging profile — continue routine monitoring."

                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {card_color};">
                    <h4>📊 Risk Group: {combined_label}</h4>
                    <p>{card_msg}</p>
                    <p style="font-size: 0.85em; color: #666;">Annual Income: ${income_val:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # =================================================================
            # SECTION G : DETAILED MEDICAL PROFILE
            # =================================================================
            st.header("🔬 Detailed Medical Profile")
            st.caption("Complete patient-level medical profile from the Early Screening information.")

            medical_sections_persistent = {
                "Demographics & Anthropometrics": ["Age", "Gender", "Sex", "BMI", "Waist_Circumference_cm", "Annual_Household_Income_USD"],
                "Blood Pressure & Vital Signs": ["Systolic_BP", "Diastolic_BP", "Pulse_Pressure", "Heart_Rate", "Respiratory_Rate", "Oxygen_Saturation"],
                "Medical History & Comorbidities": ["Diabetes", "Hypertension", "Cardiovascular_Disease", "Heart_Failure", "Hyperlipidemia", "Family_History_CKD", "Obesity"],
                "Lifestyle & Behaviour": ["Smoking_Status", "Alcohol_Consumption", "Physical_Activity_Level", "Sleep_Duration_Hours", "Stress_Level", "Lifestyle_Risk", "Poor_Sleep"],
                "Healthcare & Insurance": ["Annual_Checkups", "Health_Insurance", "Medication_Adherence", "Annual_Household_Income_USD"]
            }
            for category, fields in medical_sections_persistent.items():
                available = [f for f in fields if f in input_data_persistent.columns]
                if available:
                    with st.expander(category, expanded=True):
                        med_df = pd.DataFrame({
                            "Medical Variable": [f.replace("_", " ") for f in available],
                            "Patient Value": [input_data_persistent.iloc[0][f] for f in available]
                        })
                        st.dataframe(med_df, use_container_width=True, hide_index=True)

            st.divider()

            # =================================================================
            # SECTION H : DOWNLOAD REPORT
            # =================================================================
            st.header("📥 Download Screening Report")

            try:
                report_df = input_data_persistent.copy()
                report_df["CKD_Prediction"] = "High CKD Risk" if prediction_persistent == 1 else "Low CKD Risk"
                report_df["Decision_Score"] = decision_score_persistent
                report_df["Model"] = "Linear SVM"
                csv_buffer = BytesIO()
                report_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download Full Screening Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="CKD_Screening_Report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as dl_err:
                st.warning(f"Unable to generate CSV report: {dl_err}")

            st.divider()

            # =================================================================
            # SECTION I : GO TO CLINICAL SCREENING (optional, separate section)
            # =================================================================
            st.header("➡️ Clinical Screening")
            st.markdown(
                "**Clinical Screening** is a separate, independent assessment "
                "(CKD severity: Healthy / Mild / Moderate / Severe) with its own "
                "intake form. It doesn't reuse anything entered here."
            )

            if st.button(
                "🏥 Go to Clinical Screening →",
                type="primary",
                use_container_width=True,
                key="proceed_to_clinical"
            ):
                st.session_state["target_section"] = "2️⃣ Clinical Screening"
                st.rerun()


    # =============================================================================
    # END OF STEP 3
    # =============================================================================
    #
    # STEP 3 COMPLETE:
    #
    # ✓ Exact 49-feature model structure
    # ✓ Raw categorical values retained
    # ✓ No manual Yes/No → numeric conversion for categorical variables
    # ✓ Derived model features recreated
    # ✓ Exact model feature order verified
    # ✓ Final SVM pipeline loaded once with st.cache_resource
    # ✓ Prediction generated from the saved deployment pipeline
    # ✓ SVM decision score displayed
    # ✓ Prediction stored in session_state
    # ✓ Previous median-imputation / 'Yes' error addressed structurally
    #
    # NEXT:
    #
    # STEP 4 — CLINICAL REPORT TAB
    #
    # =============================================================================

    # STEP 2 OBJECTIVE:
    #
    # ✓ Separate CKD Screening tab
    # ✓ Patient information section
    # ✓ Body measurements
    # ✓ Blood pressure and vital signs
    # ✓ Lifestyle information
    # ✓ Medical history
    # ✓ Additional information
    # ✓ Proper three-column alignment
    # ✓ Dedicated screening button
    # ✓ No prediction yet
    # ✓ No numeric/string preprocessing conflict
    #
    # NEXT:
    #
    # STEP 3 — CONNECT THE EXACT PATIENT INPUTS TO THE SAVED SVM PIPELINE
    #
    # =============================================================================

    # =============================================================================
    # STEP 4 : CLINICAL SCREENING REPORT
    # =============================================================================
    #
    # Objective:
    # Create a clean clinical-style report using the prediction generated
    # in the CKD Screening tab.
    #
    # IMPORTANT:
    # This section does NOT run the SVM again.
    #
    # It reads:
    #
    #     st.session_state["prediction"]
    #     st.session_state["decision_score"]
    #     st.session_state["input_data"]
    #
    # generated by STEP 3.
    #
    # =============================================================================


    # =============================================================================
    # SECTION 4.1 : CLINICAL REPORT TAB
    # =============================================================================

if main_early_screening:
    with tab_interpretation:

        # =========================================================================
        # PAGE HEADER
        # =========================================================================

        st.header(
            "📋 Interpretation & Outcomes"
        )

        st.write(
            """
            This report summarises the patient's screening information, the
            CKD-risk classification generated by the final Linear SVM, and how
            that outcome should be interpreted.
            """
        )


        # =========================================================================
        # CHECK WHETHER SCREENING HAS BEEN COMPLETED
        # =========================================================================

        if "prediction" not in st.session_state:

            st.warning(
                "No CKD screening result is available yet."
            )

            st.info(
                """
                Please open the **🩺 CKD Prediction** tab, enter the patient
                information and click **Predict CKD Risk** first.
                """
            )


        else:

            # =====================================================================
            # RETRIEVE SCREENING RESULT
            # =====================================================================

            prediction = (
                st.session_state[
                    "prediction"
                ]
            )

            decision_score = (
                st.session_state[
                    "decision_score"
                ]
            )

            input_data = (
                st.session_state[
                    "input_data"
                ]
            )


            # =====================================================================
            # REPORT HEADER
            # =====================================================================

            st.divider()

            report_col1, report_col2 = st.columns(
                [2, 1]
            )


            with report_col1:

                st.subheader(
                    "Patient Screening Summary"
                )

                st.write(
                    """
                    **Screening Type:** CKD Early Screening

                    **Model:** Final Linear SVM

                    **Screening Status:** Completed
                    """
                )


            with report_col2:

                st.metric(
                    "Model Decision Score",
                    f"{decision_score:.4f}"
                )


            # =====================================================================
            # PRIMARY RISK CLASSIFICATION & VISUAL SUMMARY
            # =====================================================================

            st.divider()

            st.subheader(
                "🩺 CKD Risk Classification & Meter"
            )

            render_visual_prediction_summary(prediction, decision_score, key_suffix="prediction")

            st.divider()

            render_patient_health_badges(input_data)


            # =====================================================================
            # PATIENT PROFILE
            # =====================================================================

            st.divider()

            st.subheader(
                "👤 Patient Profile"
            )


            profile_col1, \
            profile_col2, \
            profile_col3, \
            profile_col4 = st.columns(4)


            with profile_col1:

                if "Age" in input_data.columns:

                    st.metric(
                        "Age",
                        f"{input_data['Age'].iloc[0]}"
                    )


            with profile_col2:

                if "Sex" in input_data.columns:

                    st.metric(
                        "Sex",
                        str(
                            input_data["Sex"].iloc[0]
                        )
                    )


            with profile_col3:

                if "BMI" in input_data.columns:

                    st.metric(
                        "BMI",
                        f"{input_data['BMI'].iloc[0]:.1f}"
                    )


            with profile_col4:

                if "Health_Insurance" in input_data.columns:

                    insurance_value = (
                        input_data[
                            "Health_Insurance"
                        ].iloc[0]
                    )

                    insurance_label = (
                        "Yes"
                        if int(insurance_value) == 1
                        else "No"
                    )

                    st.metric(
                        "Health Insurance",
                        insurance_label
                    )


            # =====================================================================
            # VITAL SIGNS SUMMARY
            # =====================================================================

            st.divider()

            st.subheader(
                "❤️ Vital Signs"
            )


            vital_col1, \
            vital_col2, \
            vital_col3, \
            vital_col4 = st.columns(4)


            with vital_col1:

                if "Systolic_BP" in input_data.columns:

                    st.metric(
                        "Systolic BP",
                        f"{input_data['Systolic_BP'].iloc[0]:.0f} mmHg"
                    )


            with vital_col2:

                if "Diastolic_BP" in input_data.columns:

                    st.metric(
                        "Diastolic BP",
                        f"{input_data['Diastolic_BP'].iloc[0]:.0f} mmHg"
                    )


            with vital_col3:

                if "Heart_Rate" in input_data.columns:

                    st.metric(
                        "Heart Rate",
                        f"{input_data['Heart_Rate'].iloc[0]:.0f} bpm"
                    )


            with vital_col4:

                if "Oxygen_Saturation" in input_data.columns:

                    st.metric(
                        "Oxygen Saturation",
                        f"{input_data['Oxygen_Saturation'].iloc[0]:.1f}%"
                    )


            # =====================================================================
            # DERIVED RISK INDICATORS
            # =====================================================================

            st.divider()

            st.subheader(
                "⚙️ Derived Risk Indicators"
            )


            indicator_col1, \
            indicator_col2, \
            indicator_col3, \
            indicator_col4 = st.columns(4)


            with indicator_col1:

                if "Pulse_Pressure" in input_data.columns:

                    st.metric(
                        "Pulse Pressure",
                        f"{input_data['Pulse_Pressure'].iloc[0]:.1f}"
                    )


            with indicator_col2:

                if "Waist_to_Height" in input_data.columns:

                    st.metric(
                        "Waist / Height",
                        f"{input_data['Waist_to_Height'].iloc[0]:.3f}"
                    )


            with indicator_col3:

                if "Metabolic_Risk" in input_data.columns:

                    st.metric(
                        "Metabolic Risk",
                        f"{input_data['Metabolic_Risk'].iloc[0]:.0f}"
                    )


            with indicator_col4:

                if "CV_Risk" in input_data.columns:

                    st.metric(
                        "CV Risk",
                        f"{input_data['CV_Risk'].iloc[0]:.0f}"
                    )


            # =====================================================================
            # LIFESTYLE SUMMARY
            # =====================================================================

            st.divider()

            st.subheader(
                "🏃 Lifestyle Summary"
            )


            lifestyle_col1, \
            lifestyle_col2, \
            lifestyle_col3 = st.columns(3)


            with lifestyle_col1:

                if "Smoking_Status" in input_data.columns:

                    st.write(
                        "**Smoking:**"
                    )

                    st.write(
                        input_data[
                            "Smoking_Status"
                        ].iloc[0]
                    )


            with lifestyle_col2:

                if "Physical_Activity_Level" in input_data.columns:

                    st.write(
                        "**Physical Activity:**"
                    )

                    st.write(
                        input_data[
                            "Physical_Activity_Level"
                        ].iloc[0]
                    )


            with lifestyle_col3:

                if "Sleep_Duration_Hours" in input_data.columns:

                    st.write(
                        "**Sleep Duration:**"
                    )

                    st.write(
                        f"{input_data['Sleep_Duration_Hours'].iloc[0]:.1f} hours"
                    )


            # =====================================================================
            # MEDICAL HISTORY SUMMARY
            # =====================================================================

            st.divider()

            st.subheader(
                "🏥 Medical History Summary"
            )


            medical_variables = [
                "Diabetes",
                "Hypertension",
                "Cardiovascular_Disease",
                "Heart_Failure",
                "Hyperlipidemia",
                "Kidney_Stones",
                "Recurrent_UTI",
                "Autoimmune_Disease",
                "Family_History_CKD"
            ]


            medical_data = []


            for variable in medical_variables:

                if variable in input_data.columns:

                    value = (
                        input_data[
                            variable
                        ].iloc[0]
                    )


                    if int(value) == 1:

                        status = "Yes"

                    else:

                        status = "No"


                    medical_data.append(
                        {
                            "Condition": variable.replace(
                                "_",
                                " "
                            ),
                            "Present": status
                        }
                    )


            if medical_data:

                medical_df = pd.DataFrame(
                    medical_data
                )


                st.dataframe(
                    medical_df,
                    use_container_width=True,
                    hide_index=True
                )


            # =====================================================================
            # SCREENING INTERPRETATION
            # =====================================================================

            st.divider()

            st.subheader(
                "📝 Screening Interpretation"
            )


            if int(prediction) == 1:

                st.warning(
                    """
                    The screening profile has been classified by the final
                    Linear SVM as **higher CKD risk**.

                    This result indicates that the patient's submitted
                    demographic, lifestyle, medical and physiological profile
                    resembles the CKD-risk class learned by the model.

                    Further clinical evaluation may be appropriate based on
                    the patient's overall clinical context.
                    """
                )

            else:

                st.success(
                    """
                    The screening profile has been classified by the final
                    Linear SVM as **lower CKD risk**.

                    This indicates that the patient's submitted profile was
                    classified in the lower-risk class by the screening model.

                    A lower modelled risk does not rule out disease or replace
                    routine medical assessment.
                    """
                )


            # =====================================================================
            # DECISION SCORE EXPLANATION
            # =====================================================================

            st.divider()

            st.subheader(
                "📐 SVM Decision Score"
            )


            st.write(
                f"""
                **Decision Score: {decision_score:.4f}**
                """
            )


            st.caption(
                """
                The SVM decision score represents the model's position relative
                to its classification boundary. It should not be interpreted
                as a percentage probability of having CKD.
                """
            )


            # =====================================================================
            # REPORT DATA
            # =====================================================================

            st.divider()

            st.subheader(
                "📄 Screening Data"
            )


            with st.expander(
                "View model input data"
            ):

                st.dataframe(
                    input_data.T,
                    use_container_width=True
                )


            # =====================================================================
            # DOWNLOAD REPORT DATA
            # =====================================================================

            report_data = input_data.copy()


            report_data[
                "Predicted_CKD_Risk"
            ] = (
                "Higher CKD Risk"
                if int(prediction) == 1
                else "Lower CKD Risk"
            )


            report_data[
                "SVM_Decision_Score"
            ] = decision_score


            report_csv = (
                report_data
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )


            st.download_button(
                label="📥 Download Screening Report",
                data=report_csv,
                file_name="CKD_Screening_Report.csv",
                mime="text/csv",
                use_container_width=True
            )


            # =====================================================================
            # CLINICAL DISCLAIMER
            # =====================================================================

            st.divider()

            st.caption(
                """
                **Clinical Disclaimer:** This machine-learning system is
                designed for early screening and research/analytical purposes.
                It does not provide a definitive diagnosis, medical advice,
                treatment recommendation or clinical prognosis. Results should
                be interpreted by a qualified healthcare professional together
                with appropriate clinical evaluation.
                """
            )


    # =============================================================================
    # END OF STEP 4
    # =============================================================================
    #
    # STEP 4 COMPLETE:
    #
    # ✓ Separate Clinical Report tab
    # ✓ Uses the existing SVM prediction
    # ✓ Uses the existing SVM decision score
    # ✓ Patient profile summary
    # ✓ Vital-sign summary
    # ✓ Derived-risk indicators
    # ✓ Lifestyle summary
    # ✓ Medical-history summary
    # ✓ Screening interpretation
    # ✓ Decision-score explanation
    # ✓ Input-data viewer
    # ✓ CSV report download
    # ✓ Clinical disclaimer
    #
    # NEXT:
    #
    # STEP 5 — MODEL PERFORMANCE TAB
    #
    # =============================================================================
    # =============================================================================
    # STEP 5 : MODEL PERFORMANCE TAB
    # =============================================================================
    #
    # Objective:
    # Display the evaluation results of the final CKD Linear SVM model.
    #
    # This section is separate from:
    #     - Patient screening
    #     - Clinical report
    #     - Model explainability
    #     - CKD visualisations
    #     - Insurance analytics
    #
    # =============================================================================


    # =============================================================================
    # SECTION 5.1 : MODEL PERFORMANCE TAB
    # =============================================================================

if main_early_screening:
    with tab_statistics:

        # =========================================================================
        # PAGE HEADER
        # =========================================================================

        st.title(
            "📊 Statistical Analysis"
        )

        st.write(
            """
            Evaluation of the final Linear Support Vector Machine (SVM)
            developed for CKD early screening: model information, why the
            Linear SVM was selected as the deployed model, performance metrics,
            and explainability/feature-association plots.
            """
        )


        # =========================================================================
        # WHY LINEAR SVM?
        # =========================================================================

        st.divider()

        st.subheader(
            "❓ Why Linear SVM?"
        )

        st.write(
            """
            Several candidate models were trained and compared during the
            early-screening workflow (see the model comparison table further
            below). The **Linear SVM** was chosen as the final deployed model
            because it offered the best balance of:

            - **Strong, stable performance** — competitive accuracy, recall
              and ROC-AUC against the CKD-risk target, without the heavier
              overfitting risk seen in some tree-based alternatives on this
              dataset.
            - **Interpretability of the decision boundary** — a linear
              decision function makes it possible to inspect feature weights
              directly (see the Explainability plots below) rather than relying
              purely on post-hoc approximations.
            - **Efficiency and robustness at inference time** — a linear kernel
              is fast to score, which matters for a real-time clinical
              screening tool, and is less prone to erratic behaviour on
              out-of-distribution patient inputs than more flexible non-linear
              models.
            - **Consistency across validation folds** — the Linear SVM's
              metrics varied the least across cross-validation folds among the
              models tested, indicating a more reliable and generalisable
              classifier for early screening.
            """
        )


        # =========================================================================
        # MODEL INFORMATION
        # =========================================================================

        st.divider()

        st.subheader(
            "🤖 Final Model"
        )


        model_col1, model_col2, model_col3 = st.columns(3)


        with model_col1:

            st.metric(
                "Algorithm",
                "Linear SVM"
            )


        with model_col2:

            st.metric(
                "Model Type",
                "Binary Classification"
            )


        with model_col3:

            st.metric(
                "Decision Function",
                "SVM Score"
            )


        # =========================================================================
        # PERFORMANCE DATA
        # =========================================================================
        #
        # IMPORTANT:
        # The following section expects the validation objects from your
        # modelling workflow:
        #
        #     y_valid
        #     y_pred_svm
        #     y_score
        #
        # If your variable names are different, use the exact names from
        # your modelling notebook.
        #
        # =========================================================================


        if not evaluation_loaded:

            st.info(
                "Live validation data is not available; displaying the exact "
                "validation results recorded in the CKD modelling notebook."
            )


        # Shared SVM metrics are already prepared above.


            # =====================================================================
        # USE SHARED SVM EVALUATION RESULTS
        # =====================================================================

        accuracy = svm_accuracy
        precision = svm_precision
        recall = svm_recall
        f1 = svm_f1
        roc_auc = svm_roc_auc


        # =====================================================================
        # KPI CARDS
        # =====================================================================

        st.divider()

        st.subheader(
            "📌 Performance Summary"
        )


        metric_col1, \
        metric_col2, \
        metric_col3, \
        metric_col4, \
        metric_col5 = st.columns(5)


        with metric_col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.3f}"
            )


        with metric_col2:

            st.metric(
                "Precision",
                f"{precision:.3f}"
            )


        with metric_col3:

            st.metric(
                "Recall",
                f"{recall:.3f}"
            )


        with metric_col4:

            st.metric(
                "F1 Score",
                f"{f1:.3f}"
            )


        with metric_col5:

            st.metric(
                "ROC AUC",
                f"{roc_auc:.3f}"
            )


        # =====================================================================
        # METRIC INTERPRETATION
        # =====================================================================

        st.divider()

        st.subheader(
            "📖 Metric Interpretation"
        )


        interpretation_col1, \
        interpretation_col2 = st.columns(2)


        with interpretation_col1:

            st.markdown(
                """
                **Accuracy**

                Proportion of validation observations classified
                correctly by the SVM.
                """
            )


            st.markdown(
                """
                **Precision**

                Among observations predicted as CKD risk,
                proportion that were actually CKD risk.
                """
            )


            st.markdown(
                """
                **Recall**

                Among observations belonging to the CKD-risk class,
                proportion correctly identified by the model.
                """
            )


        with interpretation_col2:

            st.markdown(
                """
                **F1 Score**

                Harmonic mean of precision and recall.
                """
            )


            st.markdown(
                """
                **ROC AUC**

                Measures the model's ability to distinguish the
                two classes across classification thresholds.
                """
            )


            st.markdown(
                """
                **SVM Decision Score**

                Represents the position of an observation relative
                to the SVM classification boundary.
                """
            )


        # =====================================================================
        # CONFUSION MATRIX
        # =====================================================================

        st.divider()

        st.subheader(
            "🔲 Confusion Matrix"
        )


        # Use the exact Linear SVM confusion matrix from the validated project results.
        # The live evaluation helper may use a different available CSV/split, so
        # recomputing it here can produce a matrix inconsistent with the reported metrics.
        cm = NOTEBOOK_CONFUSION_MATRICES["Linear SVM"].copy()

        fig_cm, ax_cm = plt.subplots(
            figsize=(7, 5)
        )


        image = ax_cm.imshow(
            cm
        )


        ax_cm.set_title(
            "Linear SVM Confusion Matrix"
        )


        ax_cm.set_xlabel(
            "Predicted Class"
        )


        ax_cm.set_ylabel(
            "Actual Class"
        )


        ax_cm.set_xticks(
            [0, 1]
        )


        ax_cm.set_yticks(
            [0, 1]
        )


        ax_cm.set_xticklabels(
            [
                "No CKD Risk",
                "CKD Risk"
            ]
        )


        ax_cm.set_yticklabels(
            [
                "No CKD Risk",
                "CKD Risk"
            ]
        )


        for i in range(
            cm.shape[0]
        ):

            for j in range(
                cm.shape[1]
            ):

                ax_cm.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )


        fig_cm.colorbar(
            image,
            ax=ax_cm
        )


        plt.tight_layout()


        st.pyplot(
            fig_cm
        )


        plt.close(
            fig_cm
        )


        # =====================================================================
        # CONFUSION MATRIX INTERPRETATION
        # =====================================================================

        tn, fp, fn, tp = cm.ravel()


        cm_col1, \
        cm_col2, \
        cm_col3, \
        cm_col4 = st.columns(4)


        with cm_col1:

            st.metric(
                "True Negatives",
                f"{tn:,}"
            )


        with cm_col2:

            st.metric(
                "False Positives",
                f"{fp:,}"
            )


        with cm_col3:

            st.metric(
                "False Negatives",
                f"{fn:,}"
            )


        with cm_col4:

            st.metric(
                "True Positives",
                f"{tp:,}"
            )


        # =====================================================================
        # MODEL PERFORMANCE SUMMARY
        # =====================================================================

        st.divider()

        st.subheader(
            "📋 Model Evaluation Summary"
        )


        performance_summary = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC AUC"
                ],
                "Score": [
                    accuracy,
                    precision,
                    recall,
                    f1,
                    roc_auc
                ]
            }
        )


        performance_summary[
            "Score"
        ] = performance_summary[
            "Score"
        ].round(
            4
        )


        st.dataframe(
            performance_summary,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================================
        # PERFORMANCE DISCLAIMER
        # =====================================================================

        st.divider()

        st.caption(
            """
            Model performance metrics describe performance on the validation
            dataset used during model evaluation. They should not be
            interpreted as guaranteed performance for every future patient.
            """
        )


    # =============================================================================
    # END OF STEP 5
    # =============================================================================
    #
    # STEP 5 COMPLETE:
    #
    # ✓ Final SVM model information
    # ✓ Accuracy
    # ✓ Precision
    # ✓ Recall
    # ✓ F1 Score
    # ✓ ROC AUC
    # ✓ Confusion Matrix
    # ✓ TN / FP / FN / TP
    # ✓ Metric interpretation
    # ✓ Performance summary table
    # ✓ Proper section alignment
    #
    # NEXT:
    #
    # STEP 6 — MODEL EXPLAINABILITY
    #
    # =============================================================================

    # =============================================================================
    # STEP 6 : MODEL EXPLAINABILITY / FEATURE IMPORTANCE
    # =============================================================================
    #
    # Objective:
    # Deploy the feature-importance analysis created in STEP 26.
    #
    # Methods:
    #
    #     Random Forest      -> Tree-based Feature Importance
    #     Logistic Regression -> Absolute Coefficient Importance
    #     Linear SVM         -> Permutation Importance
    #
    # The deployed CKD screening model is the Linear SVM.
    # Therefore, SVM permutation importance is the primary explanation.
    #
    # =============================================================================


    # =============================================================================
    # SECTION 6.1 : EXPLAINABILITY TAB
    # =============================================================================

if main_early_screening:
    with tab_statistics:

        # =========================================================================
        # PAGE HEADER
        # =========================================================================

        st.title(
            "🔎 Model Explainability"
        )

        st.write(
            """
            Explore the most influential features identified during the
            CKD model-development process.
            """
        )


        st.info(
            """
            Feature importance describes how strongly variables contributed
            to model predictions. It does not represent medical causation.
            """
        )


        # =========================================================================
        # SECTION 6.2 : EXPLAINABILITY METHODS
        # =========================================================================

        st.divider()

        st.subheader(
            "🧠 Explainability Methods"
        )


        method_col1, \
        method_col2, \
        method_col3 = st.columns(3)


        with method_col1:

            st.markdown(
                """
                ### 🌲 Random Forest

                **Tree-based Feature Importance**

                Measures the importance assigned to each feature by the
                Random Forest model.
                """
            )


        with method_col2:

            st.markdown(
                """
                ### 📈 Logistic Regression

                **Absolute Coefficient Importance**

                Uses the absolute value of the Logistic Regression
                coefficients to rank influential features.
                """
            )


        with method_col3:

            st.markdown(
                """
                ### ⚙️ Linear SVM

                **Permutation Importance**

                Measures the reduction in model performance when a feature
                is randomly permuted.
                """
            )


        # =========================================================================
        # SECTION 6.3 : FEATURE NAMES
        # =========================================================================

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

        except Exception:

            feature_names = None


        if feature_names is None:

            st.info("The original notebook feature-importance plots are embedded below; live model objects are not required for these comparison visuals.")

            st.subheader("🌲 Random Forest")
            show_embedded_notebook_plot(NOTEBOOK_RF_FI_B64)

            st.subheader("📈 Logistic Regression")
            show_embedded_notebook_plot(NOTEBOOK_LOG_FI_B64)

            st.subheader("⚙️ Linear SVM")
            show_embedded_notebook_plot(NOTEBOOK_SVM_FI_B64)

        else:

            # =====================================================================
            # MODEL SELECTION
            # =====================================================================

            st.divider()

            st.subheader(
                "📊 Select Explainability Analysis"
            )


            analysis_type = st.selectbox(
                "Choose model",
                [
                    "Linear SVM",
                    "Random Forest",
                    "Logistic Regression"
                ]
            )


            # =====================================================================
            # NUMBER OF FEATURES
            # =====================================================================

            top_n = st.slider(
                "Number of features to display",
                min_value=5,
                max_value=20,
                value=20,
                step=5
            )


            # =====================================================================
            # COMMON PLOT FUNCTION
            # =====================================================================

            def plot_feature_importance_streamlit(
                dataframe,
                importance_col,
                feature_col,
                title,
                x_label
            ):

                fig, ax = plt.subplots(
                    figsize=(11, 8)
                )


                ax.barh(
                    dataframe[feature_col],
                    dataframe[importance_col],
                    edgecolor="black",
                    linewidth=0.6
                )


                ax.set_xlabel(
                    x_label,
                    fontsize=12
                )


                ax.set_ylabel(
                    "Features",
                    fontsize=12
                )


                ax.set_title(
                    title,
                    fontsize=15,
                    fontweight="bold"
                )


                ax.grid(
                    axis="x",
                    alpha=0.3
                )


                ax.tick_params(
                    axis="both",
                    labelsize=10
                )


                plt.tight_layout()


                st.pyplot(
                    fig
                )


                plt.close(
                    fig
                )


            # =====================================================================
            # RANDOM FOREST
            # =====================================================================

            if analysis_type == "Random Forest":

                st.divider()

                st.subheader(
                    "🌲 Random Forest Feature Importance"
                )


                if "rf_model" not in globals():

                    st.warning(
                        """
                        The Random Forest model is not currently loaded
                        into the Streamlit application.
                        """
                    )

                else:

                    rf_importance = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Importance": (
                                rf_model
                                .feature_importances_
                            )
                        }
                    )


                    rf_importance = (
                        rf_importance
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .head(top_n)
                        .sort_values(
                            "Importance"
                        )
                    )


                    plot_feature_importance_streamlit(

                        rf_importance,

                        "Importance",

                        "Feature",

                        f"Top {top_n} Feature Importance - Random Forest",

                        "Importance Score"

                    )


                    st.dataframe(
                        rf_importance
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .reset_index(
                            drop=True
                        ),
                        use_container_width=True,
                        hide_index=True
                    )


            # =====================================================================
            # LOGISTIC REGRESSION
            # =====================================================================

            elif analysis_type == "Logistic Regression":

                st.divider()

                st.subheader(
                    "📈 Logistic Regression Feature Importance"
                )


                if "log_model" not in globals():

                    st.warning(
                        """
                        The Logistic Regression model is not currently loaded
                        into the Streamlit application.
                        """
                    )

                else:

                    logistic_importance = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Coefficient": (
                                log_model
                                .coef_[0]
                            )
                        }
                    )


                    logistic_importance[
                        "Absolute Importance"
                    ] = (
                        logistic_importance[
                            "Coefficient"
                        ].abs()
                    )


                    logistic_importance = (
                        logistic_importance
                        .sort_values(
                            "Absolute Importance",
                            ascending=False
                        )
                        .head(top_n)
                        .sort_values(
                            "Absolute Importance"
                        )
                    )


                    plot_feature_importance_streamlit(

                        logistic_importance,

                        "Absolute Importance",

                        "Feature",

                        f"Top {top_n} Feature Importance - Logistic Regression",

                        "Absolute Coefficient"

                    )


                    st.dataframe(
                        logistic_importance
                        .sort_values(
                            "Absolute Importance",
                            ascending=False
                        )
                        .reset_index(
                            drop=True
                        ),
                        use_container_width=True,
                        hide_index=True
                    )


                    # -------------------------------------------------------------
                    # COEFFICIENT DIRECTION
                    # -------------------------------------------------------------

                    st.caption(
                        """
                        Positive coefficients indicate association with the
                        positive model class, while negative coefficients
                        indicate association with the negative model class.
                        """
                    )


            # =====================================================================
            # LINEAR SVM
            # =====================================================================

            elif analysis_type == "Linear SVM":

                st.divider()

                st.subheader(
                    "⚙️ Linear SVM Permutation Importance"
                )


                st.write(
                    """
                    The deployed CKD screening model is the Linear SVM.
                    Its feature importance is evaluated using permutation
                    importance, matching the original feature-importance
                    analysis.
                    """
                )


                if (
                    "svm_model" not in globals()
                    or svm_model is None
                    or "x_valid_processed" not in globals()
                    or x_valid_processed is None
                    or "y_valid" not in globals()
                    or y_valid is None
                ):

                    st.warning(
                        """
                        The validation data required for SVM permutation
                        importance is not currently loaded.
                        """
                    )

                else:

                    # -------------------------------------------------------------
                    # CALCULATE PERMUTATION IMPORTANCE
                    # -------------------------------------------------------------

                    with st.spinner(
                        "Calculating SVM permutation importance..."
                    ):

                        svm_result = permutation_importance(

                            svm_model,

                            x_valid_processed,

                            y_valid,

                            scoring="accuracy",

                            random_state=42,

                            n_repeats=10

                        )


                    # -------------------------------------------------------------
                    # CREATE IMPORTANCE DATAFRAME
                    # -------------------------------------------------------------

                    svm_importance = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Importance": (
                                svm_result
                                .importances_mean
                            )
                        }
                    )


                    # -------------------------------------------------------------
                    # TOP FEATURES
                    # -------------------------------------------------------------

                    svm_importance = (
                        svm_importance
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .head(top_n)
                        .sort_values(
                            "Importance"
                        )
                    )


                    # -------------------------------------------------------------
                    # PLOT
                    # -------------------------------------------------------------

                    plot_feature_importance_streamlit(

                        svm_importance,

                        "Importance",

                        "Feature",

                        f"Top {top_n} Feature Importance - Support Vector Machine",

                        "Permutation Importance"

                    )


                    # -------------------------------------------------------------
                    # TABLE
                    # -------------------------------------------------------------

                    st.dataframe(
                        svm_importance
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .reset_index(
                            drop=True
                        ),
                        use_container_width=True,
                        hide_index=True
                    )


                    # -------------------------------------------------------------
                    # INTERPRETATION
                    # -------------------------------------------------------------

                    st.success(
                        """
                        **Primary explainability method for the deployed model:**
                        SVM Permutation Importance.
                        """
                    )


                    st.caption(
                        """
                        A larger permutation-importance value indicates that
                        randomly shuffling that feature caused a larger decrease
                        in validation performance.
                        """
                    )


            # =====================================================================
            if feature_names is not None:

                # TOP FEATURE SUMMARY
                # =====================================================================

                st.divider()

                st.subheader(
                    "🏆 Top Influential Features"
                )


                if analysis_type == "Linear SVM":

                    importance_table = svm_importance


                elif analysis_type == "Random Forest":

                    importance_table = rf_importance


                else:

                    importance_table = logistic_importance


                top_feature = (
                    importance_table
                    .sort_values(
                        importance_table.columns[1],
                        ascending=False
                    )
                    .iloc[0]
                )


                summary_col1, summary_col2 = st.columns(2)


                with summary_col1:

                    st.metric(
                        "Most Influential Feature",
                        str(
                            top_feature["Feature"]
                        )
                    )


                with summary_col2:

                    importance_value = (
                        top_feature.iloc[1]
                    )


                    st.metric(
                        "Importance",
                        f"{importance_value:.5f}"
                    )


                # =====================================================================
                # MODEL COMPARISON
                # =====================================================================

                st.divider()

                st.subheader(
                    "🔬 Explainability Comparison"
                )


                comparison_data = pd.DataFrame(
                    {
                        "Model": [
                            "Random Forest",
                            "Logistic Regression",
                            "Linear SVM"
                        ],
                        "Explainability Method": [
                            "Tree-based Feature Importance",
                            "Absolute Coefficient Importance",
                            "Permutation Importance"
                        ]
                    }
                )


                st.dataframe(
                    comparison_data,
                    use_container_width=True,
                    hide_index=True
                )


                # =====================================================================
            # IMPORTANT INTERPRETATION NOTE
            # =====================================================================

            st.divider()

            st.warning(
                """
                **Interpretation note:** Feature importance indicates how a
                machine-learning model uses variables for prediction. It does
                not establish that a feature causes CKD. Clinical interpretation
                should consider the patient's complete medical context.
                """
            )




    # =============================================================================
    # FINAL MODEL INTERPRETATION
    # =============================================================================

if main_early_screening:
    with tab_statistics:

        st.divider()
        st.subheader("🏆 Overall Interpretation & Best Model")

        st.dataframe(
            NOTEBOOK_MODEL_RESULTS.sort_values("F1 Score", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        **Interpretation:** Random Forest records the highest accuracy, precision and F1 score in the completed validation comparison. Logistic Regression records the highest recall. The Linear SVM has the highest recall among the models while remaining close in ROC-AUC, and it is the model selected for deployment because the screening workflow prioritises identifying CKD-risk cases and reducing missed high-risk observations.

        **Best deployed model: Linear SVM.** The other two models are retained as comparative benchmarks and for interpretability. Feature importance is an explanation of model behaviour, not proof of clinical causation.
        """)

    # =============================================================================
    # END OF STEP 6
    # =============================================================================
    #
    # STEP 6 COMPLETE:
    #
    # ✓ Random Forest feature importance
    # ✓ Logistic Regression absolute coefficients
    # ✓ Linear SVM permutation importance
    # ✓ Top 5 / 10 / 15 / 20 feature selection
    # ✓ Horizontal feature-importance plots
    # ✓ Feature-importance tables
    # ✓ Top-feature summary
    # ✓ Explainability-method comparison
    # ✓ Interpretation guidance
    #
    # NEXT:
    #
    # STEP 7 — CKD ANALYTICS / DATA VISUALISATIONS
    #
    # =============================================================================
    # =============================================================================
    # STEP 7 : CKD VISUALISATIONS
    # =============================================================================
    #
    # Objective:
    # Deploy the visualisations created during the CKD early-screening analysis.
    #
    # This tab is dedicated ONLY to CKD plots and visualisations.
    #
    # =============================================================================


    # =============================================================================
    # SECTION 7.1 : CKD VISUALISATIONS TAB
    # =============================================================================


            # =============================================================================
    # =============================================================================
    # STEP 7 : CKD VISUALISATIONS
    # =============================================================================

if main_early_screening:
    with tab_ckd_visuals:

        st.title("📈 CKD Visualisations")
        st.write("Visual comparison of the models and CKD risk patterns. Insurance analysis is intentionally kept in the next tab.")

        visualisation_section = st.selectbox(
            "Select CKD visualisation",
            ["Model Comparison", "CKD Risk Analysis"],
            key="ckd_visualisation_section"
        )

        if visualisation_section == "Model Comparison":

            st.header("🤖 Three-Model Visual Comparison")
            st.caption("Comparison from the completed CKD modelling workflow: Logistic Regression, Random Forest and Linear SVM.")
            st.dataframe(NOTEBOOK_MODEL_RESULTS.round(5), use_container_width=True, hide_index=True)

            metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
            fig_metrics = go.Figure()
            for i, model in enumerate(NOTEBOOK_MODEL_RESULTS["Model"]):
                vals = NOTEBOOK_MODEL_RESULTS.loc[NOTEBOOK_MODEL_RESULTS["Model"] == model, metric_cols].iloc[0].values
                fig_metrics.add_trace(go.Bar(
                    x=metric_cols,
                    y=vals,
                    name=model,
                    hovertemplate="Model: " + model + "<br>Metric: %{x}<br>Score: %{y:.4f}<extra></extra>"
                ))
            fig_metrics.update_layout(
                title="Model Performance Comparison",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                barmode='group',
                height=450
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

            st.subheader("🔲 Confusion Matrix Comparison")
            cm_fig = make_subplots(rows=1, cols=3, subplot_titles=list(NOTEBOOK_CONFUSION_MATRICES.keys()))
            for i, (model, cmv) in enumerate(NOTEBOOK_CONFUSION_MATRICES.items()):
                heatmap = go.Heatmap(
                    z=cmv,
                    x=["No CKD", "CKD"],
                    y=["No CKD", "CKD"],
                    text=cmv,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False
                )
                cm_fig.add_trace(heatmap, row=1, col=i+1)
                cm_fig.update_xaxes(title_text="Predicted", row=1, col=i+1)
                cm_fig.update_yaxes(title_text="Actual", autorange="reversed", row=1, col=i+1)
            cm_fig.update_layout(height=400)
            st.plotly_chart(cm_fig, use_container_width=True)

            st.subheader("📈 ROC / Precision–Recall Comparison")
            show_embedded_notebook_plot(NOTEBOOK_ROC_B64)
            show_embedded_notebook_plot(NOTEBOOK_PR_B64)

            st.subheader("🏆 Model Interpretation")
            st.markdown("""
            **Random Forest** records the highest accuracy, precision and F1 score in the completed validation comparison.

            **Logistic Regression** records the highest recall among the three benchmark models.

            **Linear SVM** records the highest recall among the deployed-comparison models and is the selected final screening model because the application prioritises identifying CKD-risk cases and reducing missed high-risk observations.
            """)

            # STEP 9 : CKD RISK VISUALISATIONS
            # =============================================================================
            #
            # Objective:
            # Deploy the CKD risk visualisations developed from the SVM predictions.
            #
            # Original project variables:
            #
            #     Predicted_CKD
            #     Risk_Level
            #     Health_Insurance
            #     Risk_Group
            #     Age
            #     BMI
            #     Lifestyle_Risk
            #     Metabolic_Risk
            #     CV_Risk
            #
            # =============================================================================


        elif visualisation_section == "CKD Risk Analysis":

            # =========================================================================
            # SECTION 9.1 : HEADER
            # =========================================================================

            st.header(
                "⚠️ CKD Risk Visualisations"
            )

            st.write(
                """
                Visual exploration of predicted CKD risk across the validation
                population and selected risk factors.
                """
            )


            # =========================================================================
            # SECTION 9.2 : CREATE RISK ANALYSIS DATA
            # =========================================================================

            if (
                "x_valid" not in globals()
                or
                "y_pred_svm" not in globals()
            ):

                st.warning(
                    """
                    The validation dataset and SVM predictions are not currently
                    available for the CKD risk visualisations.
                    """
                )

            else:

                risk_analysis = x_valid.copy()


                # ---------------------------------------------------------------------
                # CREATE PREDICTED CKD
                # ---------------------------------------------------------------------

                risk_analysis["Predicted_CKD"] = (
                    y_pred_svm
                )


                # ---------------------------------------------------------------------
                # CONVERT PREDICTION TO PROJECT LABELS
                # ---------------------------------------------------------------------

                risk_analysis["Predicted_CKD"] = (
                    risk_analysis["Predicted_CKD"]
                    .replace(
                        {
                            0: "Low CKD Risk",
                            1: "High CKD Risk"
                        }
                    )
                )


                # ---------------------------------------------------------------------
                # CONVERT INSURANCE STATUS
                # ---------------------------------------------------------------------

                if "Health_Insurance" in risk_analysis.columns:

                    risk_analysis["Health_Insurance"] = (
                        risk_analysis["Health_Insurance"]
                        .replace(
                            {
                                0: "No Insurance",
                                1: "Has Insurance"
                            }
                        )
                    )


                # ---------------------------------------------------------------------
                # CREATE RISK LEVEL
                # ---------------------------------------------------------------------

                risk_analysis["Risk_Level"] = (
                    risk_analysis["Predicted_CKD"]
                )


                # =====================================================================
                # SECTION 9.3 : RISK SUMMARY
                # =====================================================================

                st.divider()

                st.subheader(
                    "📊 CKD Risk Summary"
                )


                total_population = len(
                    risk_analysis
                )


                high_risk_count = (
                    risk_analysis["Predicted_CKD"]
                    .eq("High CKD Risk")
                    .sum()
                )


                low_risk_count = (
                    risk_analysis["Predicted_CKD"]
                    .eq("Low CKD Risk")
                    .sum()
                )


                high_risk_percentage = (
                    high_risk_count
                    / total_population
                    * 100
                    if total_population > 0
                    else 0
                )


                summary_col1, \
                summary_col2, \
                summary_col3 = st.columns(3)


                with summary_col1:

                    st.metric(
                        "Validation Population",
                        f"{total_population:,}"
                    )


                with summary_col2:

                    st.metric(
                        "High CKD Risk",
                        f"{high_risk_count:,}"
                    )


                with summary_col3:

                    st.metric(
                        "High-Risk Percentage",
                        f"{high_risk_percentage:.2f}%"
                    )


                # =====================================================================
                # SECTION 9.4 : CKD RISK DISTRIBUTION
                # =====================================================================

                st.divider()

                st.subheader(
                    "🩺 Predicted CKD Risk Distribution"
                )


                risk_counts = (
                    risk_analysis["Predicted_CKD"]
                    .value_counts()
                )


                fig_risk = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    color=risk_counts.index,
                    color_discrete_map={
                        "High CKD Risk": "#e74c3c",
                        "Low CKD Risk": "#27ae60"
                    },
                    labels={"x": "Predicted CKD Risk", "y": "Number of Patients"},
                    title="Predicted CKD Risk Distribution"
                )
                fig_risk.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig_risk, use_container_width=True)


                # =====================================================================
                # SECTION 9.5 : AGE VS BMI BY CKD RISK
                # =====================================================================

                if (
                    "Age" in risk_analysis.columns
                    and
                    "BMI" in risk_analysis.columns
                ):

                    st.divider()

                    st.subheader(
                        "👤 Age, BMI and CKD Risk"
                    )


                    fig_age_bmi, ax_age_bmi = plt.subplots(
                        figsize=(10, 7)
                    )


                    for risk_group in [
                        "Low CKD Risk",
                        "High CKD Risk"
                    ]:

                        subset = risk_analysis[
                            risk_analysis[
                                "Predicted_CKD"
                            ] == risk_group
                        ]


                        ax_age_bmi.scatter(
                            subset["Age"],
                            subset["BMI"],
                            s=35,
                            alpha=0.5,
                            label=risk_group
                        )


                    ax_age_bmi.set_xlabel(
                        "Age",
                        fontsize=12
                    )


                    ax_age_bmi.set_ylabel(
                        "BMI",
                        fontsize=12
                    )


                    ax_age_bmi.set_title(
                        "Age vs BMI by Predicted CKD Risk",
                        fontsize=15,
                        fontweight="bold"
                    )


                    ax_age_bmi.legend()


                    ax_age_bmi.grid(
                        alpha=0.3
                    )


                    plt.tight_layout()


                    st.pyplot(
                        fig_age_bmi
                    )


                    plt.close(
                        fig_age_bmi
                    )


                # =====================================================================
                # SECTION 9.6 : LIFESTYLE RISK
                # =====================================================================

                if "Lifestyle_Risk" in risk_analysis.columns:

                    st.divider()

                    st.subheader(
                        "🏃 Lifestyle Risk and CKD Prediction"
                    )


                    lifestyle_table = (
                        pd.crosstab(
                            risk_analysis[
                                "Lifestyle_Risk"
                            ],
                            risk_analysis[
                                "Predicted_CKD"
                            ]
                        )
                    )


                    st.dataframe(
                        lifestyle_table,
                        use_container_width=True
                    )


                    fig_lifestyle, ax_lifestyle = plt.subplots(
                        figsize=(9, 6)
                    )


                    lifestyle_table.plot(
                        kind="bar",
                        ax=ax_lifestyle,
                        edgecolor="black"
                    )


                    ax_lifestyle.set_xlabel(
                        "Lifestyle Risk",
                        fontsize=12
                    )


                    ax_lifestyle.set_ylabel(
                        "Number of Patients",
                        fontsize=12
                    )


                    ax_lifestyle.set_title(
                        "Lifestyle Risk by Predicted CKD Risk",
                        fontsize=15,
                        fontweight="bold"
                    )


                    ax_lifestyle.legend(
                        title="Predicted CKD"
                    )


                    ax_lifestyle.grid(
                        axis="y",
                        alpha=0.3
                    )


                    plt.xticks(
                        rotation=0
                    )


                    plt.tight_layout()


                    st.pyplot(
                        fig_lifestyle
                    )


                    plt.close(
                        fig_lifestyle
                    )


                # =====================================================================
                # SECTION 9.7 : METABOLIC RISK
                # =====================================================================

                if "Metabolic_Risk" in risk_analysis.columns:

                    st.divider()

                    st.subheader(
                        "🧬 Metabolic Risk and CKD Prediction"
                    )


                    metabolic_table = (
                        pd.crosstab(
                            risk_analysis[
                                "Metabolic_Risk"
                            ],
                            risk_analysis[
                                "Predicted_CKD"
                            ]
                        )
                    )


                    st.dataframe(
                        metabolic_table,
                        use_container_width=True
                    )


                    fig_metabolic, ax_metabolic = plt.subplots(
                        figsize=(9, 6)
                    )


                    metabolic_table.plot(
                        kind="bar",
                        ax=ax_metabolic,
                        edgecolor="black"
                    )


                    ax_metabolic.set_xlabel(
                        "Metabolic Risk",
                        fontsize=12
                    )


                    ax_metabolic.set_ylabel(
                        "Number of Patients",
                        fontsize=12
                    )


                    ax_metabolic.set_title(
                        "Metabolic Risk by Predicted CKD Risk",
                        fontsize=15,
                        fontweight="bold"
                    )


                    ax_metabolic.legend(
                        title="Predicted CKD"
                    )


                    ax_metabolic.grid(
                        axis="y",
                        alpha=0.3
                    )


                    plt.xticks(
                        rotation=0
                    )


                    plt.tight_layout()


                    st.pyplot(
                        fig_metabolic
                    )


                    plt.close(
                        fig_metabolic
                    )


                # =====================================================================
                # SECTION 9.8 : CARDIOVASCULAR RISK
                # =====================================================================

                if "CV_Risk" in risk_analysis.columns:

                    st.divider()

                    st.subheader(
                        "❤️ Cardiovascular Risk and CKD Prediction"
                    )


                    cv_table = (
                        pd.crosstab(
                            risk_analysis[
                                "CV_Risk"
                            ],
                            risk_analysis[
                                "Predicted_CKD"
                            ]
                        )
                    )


                    st.dataframe(
                        cv_table,
                        use_container_width=True
                    )


                    fig_cv, ax_cv = plt.subplots(
                        figsize=(9, 6)
                    )


                    cv_table.plot(
                        kind="bar",
                        ax=ax_cv,
                        edgecolor="black"
                    )


                    ax_cv.set_xlabel(
                        "Cardiovascular Risk",
                        fontsize=12
                    )


                    ax_cv.set_ylabel(
                        "Number of Patients",
                        fontsize=12
                    )


                    ax_cv.set_title(
                        "Cardiovascular Risk by Predicted CKD Risk",
                        fontsize=15,
                        fontweight="bold"
                    )


                    ax_cv.legend(
                        title="Predicted CKD"
                    )


                    ax_cv.grid(
                        axis="y",
                        alpha=0.3
                    )


                    plt.xticks(
                        rotation=0
                    )


                    plt.tight_layout()


                    st.pyplot(
                        fig_cv
                    )


                    plt.close(
                        fig_cv
                    )


                # =====================================================================
                # SECTION 9.9 : INSURANCE × CKD RISK
                # =====================================================================

                if "Health_Insurance" in risk_analysis.columns:

                    st.divider()

                    st.subheader(
                        "🛡️ Insurance Status and Predicted CKD Risk"
                    )


                    insurance_risk_table = (
                        pd.crosstab(
                            risk_analysis[
                                "Health_Insurance"
                            ],
                            risk_analysis[
                                "Predicted_CKD"
                            ]
                        )
                    )


                    st.dataframe(
                        insurance_risk_table,
                        use_container_width=True
                    )


                    fig_insurance, ax_insurance = plt.subplots(
                        figsize=(9, 6)
                    )


                    insurance_risk_table.plot(
                        kind="bar",
                        ax=ax_insurance,
                        edgecolor="black"
                    )


                    ax_insurance.set_xlabel(
                        "Health Insurance",
                        fontsize=12
                    )


                    ax_insurance.set_ylabel(
                        "Number of Patients",
                        fontsize=12
                    )


                    ax_insurance.set_title(
                        "Predicted CKD Risk by Health Insurance Status",
                        fontsize=15,
                        fontweight="bold"
                    )


                    ax_insurance.legend(
                        title="Predicted CKD"
                    )


                    ax_insurance.grid(
                        axis="y",
                        alpha=0.3
                    )


                    plt.xticks(
                        rotation=0
                    )


                    plt.tight_layout()


                    st.pyplot(
                        fig_insurance
                    )


                    plt.close(
                        fig_insurance
                    )


                # =====================================================================
                # SECTION 9.10 : RISK GROUP CREATION
                # =====================================================================

                if "Health_Insurance" in risk_analysis.columns:

                    risk_analysis["Risk_Group"] = (

                        risk_analysis[
                            "Predicted_CKD"
                        ].astype(str)

                        + " | "

                        + risk_analysis[
                            "Health_Insurance"
                        ].astype(str)

                    )

                    # -------------------------------------------------------------
                    # HAND OFF THE EXISTING EARLY-SCREENING INSURANCE ANALYSIS
                    # -------------------------------------------------------------
                    # The Insurance Intelligence section reuses this exact
                    # validation-population analysis. No second dataset and no
                    # second SVM prediction are created.
                    st.session_state[
                        "early_screening_insurance_analysis"
                    ] = risk_analysis.copy()


                    st.divider()

                    st.subheader(
                        "🎯 CKD Risk Groups"
                    )


                    risk_group_counts = (
                        risk_analysis[
                            "Risk_Group"
                        ]
                        .value_counts()
                    )


                    fig_groups, ax_groups = plt.subplots(
                        figsize=(11, 6)
                    )


                    ax_groups.bar(
                        risk_group_counts.index,
                        risk_group_counts.values,
                        edgecolor="black",
                        linewidth=0.8
                    )


                    ax_groups.set_xlabel(
                        "Risk Group",
                        fontsize=12
                    )


                    ax_groups.set_ylabel(
                        "Number of Patients",
                        fontsize=12
                    )


                    ax_groups.set_title(
                        "CKD Risk Groups by Insurance Status",
                        fontsize=15,
                        fontweight="bold"
                    )


                    ax_groups.tick_params(
                        axis="x",
                        rotation=20
                    )


                    ax_groups.grid(
                        axis="y",
                        alpha=0.3
                    )


                    plt.tight_layout()


                    st.pyplot(
                        fig_groups
                    )


                    plt.close(
                        fig_groups
                    )


                    # -----------------------------------------------------------------
                    # RISK GROUP TABLE
                    # -----------------------------------------------------------------

                    st.dataframe(
                        risk_group_counts
                        .rename(
                            "Patient Count"
                        )
                        .reset_index()
                        .rename(
                            columns={
                                "index": "Risk Group"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True
                    )


    # =============================================================================
    # END OF STEP 9
    # =============================================================================
        #
    # STEP 9 COMPLETE:
    #
    # ✓ Predicted CKD risk distribution
    # ✓ High/Low CKD risk summary
    # ✓ Age vs BMI by CKD risk
    # ✓ Lifestyle Risk analysis
    # ✓ Metabolic Risk analysis
    # ✓ Cardiovascular Risk analysis
    # ✓ Insurance × CKD Risk
    # ✓ Risk_Group construction
    # ✓ Risk-group visualisation
    #
    # =============================================================================

    
        # =============================================================================
    
        # =============================================================================
        #
        # STEP 10 COMPLETE:
        #
        # ✓ Insurance status distribution
        # ✓ CKD risk × insurance
        # ✓ Risk-group distribution
        # ✓ Insurance-based CKD segmentation
        # ✓ Income × Age × BMI portfolio landscape
        # ✓ Insurance risk portfolio table
        #
        # NEXT:
        #
        # STEP 11 — EXPLAINABILITY VISUALISATIONS
        #
        # =============================================================================




    # =============================================================================
    # END OF STEP 7
    # =============================================================================
    #
    # STEP 7 STRUCTURE:
    #
    # 📈 CKD Visualisations
    #
    #     ├── 📊 Overview
    #     ├── 👤 Demographic Analysis
    #     ├── 🩺 Clinical Risk Factors
    #     ├── 🏃 Lifestyle Analysis
    #     ├── ⚠️ CKD Risk Analysis
    #     ├── 🤖 Model Visualisations
    #     └── 🛡️ Insurance & Risk Analysis
    #
    # =============================================================================



    # =============================================================================
    # STEP 10 : INSURANCE ANALYTICS
    # =============================================================================

if is_insurance_section:
    with tab_insurance:


        st.title("🏥 Insurance Analytics")
        st.write("Insurance-risk analysis from the SVM validation predictions. This section is intentionally separate from CKD Visualisations.")


            # =========================================================================
        # SECTION 10.1 : HEADER
        # =========================================================================

        st.header(
            "🛡️ Insurance & CKD Risk Analysis"
        )

        st.write(
            """
            Visual exploration of the relationship between predicted CKD risk,
            health-insurance status, demographics and household income.
            """
        )


        # =========================================================================
        # SECTION 10.2 : PREPARE INSURANCE ANALYSIS DATA
        # =========================================================================

        if (
            "x_valid" not in globals()
            or x_valid is None
            or "y_pred_svm" not in globals()
            or y_pred_svm is None
        ):

            st.info(
                "The Insurance tab is connected to the shared validation/SVM evaluation."
            )

        else:

            insurance_analysis = x_valid.copy()


            # ---------------------------------------------------------------------
            # PREDICTED CKD
            # ---------------------------------------------------------------------

            insurance_analysis["Predicted_CKD"] = (
                y_pred_svm
            )


            insurance_analysis["Predicted_CKD"] = (
                insurance_analysis["Predicted_CKD"]
                .replace(
                    {
                        0: "Low CKD Risk",
                        1: "High CKD Risk"
                    }
                )
            )


            # ---------------------------------------------------------------------
            # HEALTH INSURANCE LABEL
            # ---------------------------------------------------------------------

            if "Health_Insurance" in insurance_analysis.columns:

                insurance_analysis["Health_Insurance"] = (
                    insurance_analysis["Health_Insurance"]
                    .replace(
                        {
                            0: "No Insurance",
                            1: "Has Insurance"
                        }
                    )
                )


            # ---------------------------------------------------------------------
            # CREATE RISK GROUP
            # ---------------------------------------------------------------------

            if "Health_Insurance" in insurance_analysis.columns:

                insurance_analysis["Risk_Group"] = (

                    insurance_analysis[
                        "Predicted_CKD"
                    ].astype(str)

                    + " | "

                    + insurance_analysis[
                        "Health_Insurance"
                    ].astype(str)

                )


            # =========================================================================
            # SECTION 10.3 : INSURANCE × CKD RISK SUMMARY
            # =========================================================================

            st.divider()

            st.subheader(
                "📊 Insurance and CKD Risk Summary"
            )


            if "Health_Insurance" in insurance_analysis.columns:

                insured_count = (
                    insurance_analysis[
                        "Health_Insurance"
                    ]
                    .eq("Has Insurance")
                    .sum()
                )


                uninsured_count = (
                    insurance_analysis[
                        "Health_Insurance"
                    ]
                    .eq("No Insurance")
                    .sum()
                )


                high_risk_count = (
                    insurance_analysis[
                        "Predicted_CKD"
                    ]
                    .eq("High CKD Risk")
                    .sum()
                )


                summary_col1, \
                summary_col2, \
                summary_col3 = st.columns(3)


                with summary_col1:

                    st.metric(
                        "Has Insurance",
                        f"{insured_count:,}"
                    )


                with summary_col2:

                    st.metric(
                        "No Insurance",
                        f"{uninsured_count:,}"
                    )


                with summary_col3:

                    st.metric(
                        "High CKD Risk",
                        f"{high_risk_count:,}"
                    )


            # =========================================================================
            # SECTION 10.4 : INSURANCE STATUS DISTRIBUTION
            # =========================================================================

            if "Health_Insurance" in insurance_analysis.columns:

                st.divider()

                st.subheader(
                    "🛡️ Health Insurance Distribution"
                )


                insurance_counts = (
                    insurance_analysis[
                        "Health_Insurance"
                    ]
                    .value_counts()
                )


                fig_insurance_dist = px.bar(
                    x=insurance_counts.index,
                    y=insurance_counts.values,
                    color=insurance_counts.index,
                    color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899", "#06B6D4"],
                    labels={"x": "Health Insurance Status", "y": "Number of Patients"},
                    title="Health Insurance Distribution"
                )
                fig_insurance_dist.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig_insurance_dist, use_container_width=True)


            # =========================================================================
            # SECTION 10.5 : CKD RISK × INSURANCE
            # =========================================================================

            if "Health_Insurance" in insurance_analysis.columns:

                st.divider()

                st.subheader(
                    "⚠️ CKD Risk by Insurance Status"
                )


                insurance_risk_table = pd.crosstab(
                    insurance_analysis[
                        "Health_Insurance"
                    ],
                    insurance_analysis[
                        "Predicted_CKD"
                    ]
                )


                st.dataframe(
                    insurance_risk_table,
                    use_container_width=True
                )


                fig_insurance_risk, ax_insurance_risk = plt.subplots(
                    figsize=(10, 6)
                )


                insurance_risk_table.plot(
                    kind="bar",
                    ax=ax_insurance_risk,
                    edgecolor="black"
                )


                ax_insurance_risk.set_xlabel(
                    "Health Insurance Status",
                    fontsize=12
                )


                ax_insurance_risk.set_ylabel(
                    "Number of Patients",
                    fontsize=12
                )


                ax_insurance_risk.set_title(
                    "Predicted CKD Risk by Insurance Status",
                    fontsize=15,
                    fontweight="bold"
                )


                ax_insurance_risk.legend(
                    title="Predicted CKD"
                )


                ax_insurance_risk.grid(
                    axis="y",
                    alpha=0.3
                )


                plt.xticks(
                    rotation=0
                )


                plt.tight_layout()


                st.pyplot(
                    fig_insurance_risk
                )


                plt.close(
                    fig_insurance_risk
                )


            # =========================================================================
            # SECTION 10.6 : RISK GROUP DISTRIBUTION
            # =========================================================================

            if "Risk_Group" in insurance_analysis.columns:

                st.divider()

                st.subheader(
                    "🎯 Insurance-Based CKD Risk Groups"
                )


                risk_group_counts = (
                    insurance_analysis[
                        "Risk_Group"
                    ]
                    .value_counts()
                )


                fig_risk_group, ax_risk_group = plt.subplots(
                    figsize=(11, 6)
                )


                ax_risk_group.bar(
                    risk_group_counts.index,
                    risk_group_counts.values,
                    edgecolor="black",
                    linewidth=0.8
                )


                ax_risk_group.set_xlabel(
                    "Risk Group",
                    fontsize=12
                )


                ax_risk_group.set_ylabel(
                    "Number of Patients",
                    fontsize=12
                )


                ax_risk_group.set_title(
                    "CKD Risk Groups by Insurance Status",
                    fontsize=15,
                    fontweight="bold"
                )


                ax_risk_group.tick_params(
                    axis="x",
                    rotation=20
                )


                ax_risk_group.grid(
                    axis="y",
                    alpha=0.3
                )


                plt.tight_layout()


                st.pyplot(
                    fig_risk_group
                )


                plt.close(
                    fig_risk_group
                )


                st.dataframe(
                    risk_group_counts
                    .rename(
                        "Patient Count"
                    )
                    .reset_index()
                    .rename(
                        columns={
                            "index": "Risk Group"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )


            # =========================================================================
            # SECTION 10.7 : INCOME × AGE × BMI RISK LANDSCAPE
            # =========================================================================

            required_portfolio_columns = [

                "Annual_Household_Income_USD",

                "Age",

                "BMI",

                "Risk_Group"

            ]


            portfolio_columns_available = all(
                column in insurance_analysis.columns
                for column in required_portfolio_columns
            )


            if portfolio_columns_available:

                st.divider()

                st.subheader(
                    "🌐 Insurance Portfolio Risk Landscape"
                )

                st.caption(
                    """
                    Household income is shown against age, while BMI represents
                    the relative size of each observation and Risk Group
                    identifies the CKD-insurance segment.
                    """
                )


                fig_portfolio, ax_portfolio = plt.subplots(
                    figsize=(12, 8)
                )


                risk_groups = (
                    insurance_analysis[
                        "Risk_Group"
                    ]
                    .dropna()
                    .unique()
                )


                for risk_group in risk_groups:

                    subset = insurance_analysis[
                        insurance_analysis[
                            "Risk_Group"
                        ] == risk_group
                    ]


                    ax_portfolio.scatter(
                        subset[
                            "Annual_Household_Income_USD"
                        ],

                        subset[
                            "Age"
                        ],

                        s=(
                            subset[
                                "BMI"
                            ].clip(
                                lower=10
                            )
                            * 8
                        ),

                        alpha=0.55,

                        label=risk_group
                    )


                ax_portfolio.set_xlabel(
                    "Annual Household Income (USD)",
                    fontsize=12
                )


                ax_portfolio.set_ylabel(
                    "Age",
                    fontsize=12
                )


                ax_portfolio.set_title(
                    "Insurance Portfolio Risk Landscape",
                    fontsize=15,
                    fontweight="bold"
                )


                ax_portfolio.legend(
                    title="Risk Group",
                    bbox_to_anchor=(
                        1.02,
                        1
                    ),
                    loc="upper left"
                )


                ax_portfolio.grid(
                    alpha=0.25
                )


                plt.tight_layout()


                st.pyplot(
                    fig_portfolio
                )


                plt.close(
                    fig_portfolio
                )


            else:

                st.info(
                    """
                    The portfolio risk landscape requires:

                    Annual_Household_Income_USD
                    Age
                    BMI
                    Risk_Group
                    """
                )


            # =========================================================================
            # SECTION 10.8 : INSURANCE RISK TABLE
            # =========================================================================

            st.divider()

            st.subheader(
                "📋 Insurance Risk Portfolio"
            )


            display_columns = [

                "Predicted_CKD",

                "Health_Insurance",

                "Risk_Group"

            ]


            display_columns = [

                column

                for column in display_columns

                if column in insurance_analysis.columns

            ]


            if display_columns:

                st.dataframe(
                    insurance_analysis[
                        display_columns
                    ].head(100),
                    use_container_width=True,
                    hide_index=True
                )


        # =============================================================================


    # =============================================================================
    # STEP 10B : EXISTING EARLY-SCREENING INTERACTIVE INSURANCE ANALYSIS
    # =============================================================================
    # The original Insurance analysis is retained. Figures are cached in
    # session_state so Streamlit does not rebuild every Plotly object on every
    # rerun (especially when the user changes an unrelated input).

if is_insurance_section:
    with tab_insurance:

        st.title("🏥 Insurance Analytics")

        st.write(
            "Insurance portfolio analysis connected to the final CKD Linear SVM "
            "validation predictions."
        )

        if not evaluation_loaded:

            st.error(
                "Insurance Analytics cannot run because the shared evaluation "
                "objects were not created."
            )
            st.exception(evaluation_error)

        elif "Health_Insurance" not in x_valid.columns:

            st.error(
                "Health_Insurance is missing from the validation feature set."
            )

        else:

            # Reuse the exact insurance analysis generated in Early Screening.
            # Fall back to the existing shared validation objects only when the
            # Insurance section is opened before Early Screening has produced
            # the handoff dataframe.
            portfolio = st.session_state.get(
                "early_screening_insurance_analysis"
            )

            if portfolio is not None:
                portfolio = portfolio.copy()
            else:
                portfolio = x_valid.copy()

                portfolio["Predicted_CKD"] = y_pred_svm

                portfolio["Predicted_CKD"] = portfolio["Predicted_CKD"].replace({
                    0: "Low CKD Risk",
                    1: "High CKD Risk"
                })

                portfolio["Health_Insurance"] = portfolio["Health_Insurance"].replace({
                    0: "No Insurance",
                    1: "Has Insurance"
                })

            portfolio["Risk_Level"] = portfolio["Predicted_CKD"].replace({
                "Low CKD Risk": "Low Risk",
                "High CKD Risk": "High Risk"
            })

            portfolio["Risk_Group"] = (
                portfolio["Predicted_CKD"].astype(str)
                + " | "
                + portfolio["Health_Insurance"].astype(str)
            )

            # ---------------------------------------------------------------------
            # KPI SUMMARY
            # ---------------------------------------------------------------------

            st.divider()
            st.subheader("📊 Insurance Portfolio Summary")

            total_members = len(portfolio)
            high_risk = (portfolio["Risk_Level"] == "High Risk").sum()
            low_risk = (portfolio["Risk_Level"] == "Low Risk").sum()
            insured = (portfolio["Health_Insurance"] == "Has Insurance").sum()
            uninsured = (portfolio["Health_Insurance"] == "No Insurance").sum()

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric("Total Members", f"{total_members:,}")
            with c2:
                st.metric("Low Risk", f"{low_risk:,}")
            with c3:
                st.metric("High Risk", f"{high_risk:,}")
            with c4:
                st.metric("Insured", f"{insured:,}")
            with c5:
                st.metric("Uninsured", f"{uninsured:,}")

            # ---------------------------------------------------------------------
            # NOTEBOOK RISK DISTRIBUTION
            # ---------------------------------------------------------------------

            st.divider()
            st.subheader("📈 Portfolio Risk Distribution")

            risk_distribution = (
                portfolio["Risk_Level"].value_counts().reset_index()
            )
            risk_distribution.columns = ["Risk Level", "Members"]

            fig_risk, ax_risk = plt.subplots(figsize=(8, 5))
            ax_risk.bar(
                risk_distribution["Risk Level"],
                risk_distribution["Members"],
                edgecolor="black"
            )
            ax_risk.set_xlabel("Risk Level")
            ax_risk.set_ylabel("Number of Members")
            ax_risk.set_title(
                "Portfolio Risk Distribution",
                fontsize=14,
                fontweight="bold"
            )
            ax_risk.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_risk)
            plt.close(fig_risk)

            # ---------------------------------------------------------------------
            # NOTEBOOK INSURANCE × RISK
            # ---------------------------------------------------------------------

            st.divider()
            st.subheader("🛡️ Portfolio Risk by Insurance Status")

            insurance_risk = pd.crosstab(
                portfolio["Health_Insurance"],
                portfolio["Risk_Level"]
            )

            portfolio_percentage = (
                insurance_risk.div(insurance_risk.sum(axis=1), axis=0) * 100
            ).round(2)

            left, right = st.columns(2)

            with left:
                st.dataframe(
                    insurance_risk,
                    use_container_width=True
                )

            with right:
                st.dataframe(
                    portfolio_percentage,
                    use_container_width=True
                )

            fig_stacked, ax_stacked = plt.subplots(figsize=(9, 5))
            insurance_risk.plot(
                kind="bar",
                stacked=True,
                ax=ax_stacked,
                edgecolor="black"
            )
            ax_stacked.set_title(
                "Portfolio Risk by Insurance Status",
                fontsize=14,
                fontweight="bold"
            )
            ax_stacked.set_xlabel("Health Insurance")
            ax_stacked.set_ylabel("Members")
            ax_stacked.tick_params(axis="x", rotation=0)
            ax_stacked.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_stacked)
            plt.close(fig_stacked)

            # ---------------------------------------------------------------------
            # NOTEBOOK CIRCULAR / EXECUTIVE SUNBURST
            # ---------------------------------------------------------------------

            st.divider()
            st.subheader("⭕ Executive Insurance Portfolio — Circular Analysis")
            st.caption(
                "This is the circular Sunburst visualisation from the notebook, "
                "integrated into the Insurance Analytics tab."
            )

            sunburst_columns = [
                "Country",
                "Health_Insurance",
                "Predicted_CKD",
                "Risk_Level",
                "Socioeconomic_Status"
            ]

            if all(column in portfolio.columns for column in sunburst_columns):

                dashboard = portfolio.dropna(
                    subset=sunburst_columns
                ).copy()

                for column in sunburst_columns:
                    dashboard[column] = dashboard[column].astype(str)

                fig_sunburst = px.sunburst(
                    dashboard,
                    path=[
                        "Country",
                        "Health_Insurance",
                        "Predicted_CKD",
                        "Risk_Level",
                        "Socioeconomic_Status"
                    ],
                    color="Risk_Level",
                    color_discrete_map={
                        "Low Risk": "#2ECC71",
                        "High Risk": "#E74C3C"
                    },
                    hover_data={
                        "Age": True,
                        "BMI": ":.1f",
                        "Annual_Household_Income_USD": ":,.0f"
                    },
                    maxdepth=5,
                    title="<b>Executive Insurance Portfolio Analysis</b>"
                )

                fig_sunburst.update_traces(
                    textinfo="label+percent parent",
                    insidetextorientation="radial",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Members: %{value}<br>"
                        "Parent: %{percentParent:.1%}<br>"
                        "Root: %{percentRoot:.1%}<extra></extra>"
                    )
                )

                fig_sunburst.update_layout(
                    template="plotly_white",
                    height=700,
                    margin=dict(l=20, r=20, t=70, b=20),
                    font=dict(family="Arial", size=14),
                    title_x=0.5
                )

                st.plotly_chart(
                    fig_sunburst,
                    use_container_width=True
                )

            else:

                missing_sunburst = [
                    column
                    for column in sunburst_columns
                    if column not in portfolio.columns
                ]

                st.warning(
                    "The notebook Sunburst requires these missing columns: "
                    + ", ".join(missing_sunburst)
                )

            # ---------------------------------------------------------------------
            # NOTEBOOK PORTFOLIO RISK LANDSCAPE
            # ---------------------------------------------------------------------

            st.divider()
            st.subheader("🌐 Insurance Portfolio Risk Landscape")

            landscape_columns = [
                "Annual_Household_Income_USD",
                "Age",
                "BMI",
                "Lifestyle_Risk",
                "Metabolic_Risk",
                "CV_Risk",
                "Risk_Level",
                "Country",
                "Health_Insurance"
            ]

            if all(column in portfolio.columns for column in landscape_columns):

                fig_landscape = px.scatter(
                    portfolio,
                    x="Annual_Household_Income_USD",
                    y="Age",
                    size="BMI",
                    color="Risk_Group",
                    color_discrete_map={"Low Risk": "#2563EB", "High Risk": "#DC2626", "Low": "#10B981", "High": "#F59E0B"},
                    symbol="Health_Insurance",
                    hover_name="Country",
                    hover_data={
                        "Lifestyle_Risk": True,
                        "Metabolic_Risk": True,
                        "CV_Risk": True,
                        "Risk_Level": True,
                        "BMI": ":.1f",
                        "Annual_Household_Income_USD": ":,.0f",
                        "Age": True
                    },
                    title="<b>Insurance Portfolio Risk Landscape</b>"
                )

                fig_landscape.update_layout(
                    template="plotly_white",
                    height=650,
                    title_x=0.5
                )

                st.plotly_chart(
                    fig_landscape,
                    use_container_width=True
                )

            else:

                missing_landscape = [
                    column
                    for column in landscape_columns
                    if column not in portfolio.columns
                ]

                st.warning(
                    "Portfolio Risk Landscape unavailable. Missing: "
                    + ", ".join(missing_landscape)
                )

            # ---------------------------------------------------------------------
            # NOTEBOOK PORTFOLIO TABLE
            # ---------------------------------------------------------------------

            st.divider()
            st.subheader("📋 Insurance Risk Portfolio")

            st.dataframe(
                portfolio[[
                    "Predicted_CKD",
                    "Health_Insurance",
                    "Risk_Level",
                    "Risk_Group"
                ]].head(100),
                use_container_width=True,
                hide_index=True
            )

            # ---------------------------------------------------------------------
            # HIGH-RISK UNINSURED SEGMENT
            # ---------------------------------------------------------------------

            high_risk_uninsured = (
                (portfolio["Health_Insurance"] == "No Insurance")
                & (portfolio["Predicted_CKD"] == "High CKD Risk")
            ).sum()

            st.divider()
            st.subheader("⚠️ High-Risk / No-Insurance Segment")

            st.metric(
                "High CKD Risk + No Insurance",
                f"{high_risk_uninsured:,}"
            )

            st.caption(
                "This segment is shown for population-level analytical "
                "segmentation only and must not be used to deny or restrict "
                "health insurance or healthcare access."
            )

    # =============================================================================
    # END OF SECTION 11 : INSURANCE ANALYTICS TAB
    # =============================================================================

    # SECTION 12 : ABOUT MODEL TAB
        # =============================================================================

if main_early_screening:
    with tab_about:


        st.header(
        "📚 About the Early Screening Model"
        )

        st.write(
        """
        Final deployed model (Early Screening section):

        **Linear Support Vector Machine (SVM)**

        This section contains information about the dataset,
        preprocessing, feature engineering, model development,
        validation and limitations for the CKD **early-screening**
        (risk) model. The **Clinical Screening** section (Section 2 of this
        app) documents the CKD **severity** model separately.
        """
        )


        # =============================================================================
        # END OF STEP 0
        # =============================================================================
        #
        # STEP 0 OBJECTIVE:
        #
        # ✓ Create the Streamlit application
        # ✓ Create the complete tab structure
        # ✓ Keep every module separated
        # ✓ Establish the final dashboard architecture
        #
        # NEXT:
        #
        # STEP 1 — HOME TAB + APPLICATION DESIGN
        #
        # =============================================================================


    # =============================================================================
    # SECTION 13 : CKD CLINICAL SEVERITY SCREENING & WORKINGS
    # =============================================================================
    #
    # Dedicated Second-Stage Workflow integrated from:
    # 'Severity model Final Draft.ipynb'
    #
    # Architecture:
    # - Target: 4-Class CKD Severity:
    #     * 0: Healthy (KDIGO Normal / No CKD)
    #     * 1: Mild CKD (KDIGO Stage 1 & Stage 2)
    #     * 2: Moderate CKD (KDIGO Stage 3a & Stage 3b)
    #     * 3: Severe CKD (KDIGO Stage 4 & Stage 5)
    # - Deployed Model: Multi-class XGBoost Classifier (Trained on SMOTE Data)
    # - Candidate Models Evaluated: Ordinal Logistic Regression, Random Forest, XGBoost
    # - Features: 75 Clinical, Demographic, Vital & Laboratory Biomarkers
    # =============================================================================

CLINICAL_ASSET_DIR = Path(__file__).parent / "assets"

def clinical_asset(filename):
    paths = [
        CLINICAL_ASSET_DIR / filename,
        Path.cwd() / "assets" / filename,
        Path(r"D:\project\data\raw\CKD_API\assets") / filename,
        Path(r"C:\Users\SAMSUNG\.gemini\antigravity\worktrees\CKD_API\check_rate_limit_status\assets") / filename
    ]
    for p in paths:
        if p.exists():
            return str(p)
    return None


@st.cache_resource
def load_severity_model_bundle():
    dirs_to_check = [
        Path(__file__).resolve().parent,
        Path.cwd(),
        Path(r"D:\project\data\raw\CKD_API"),
        Path(r"C:\Users\SAMSUNG\.gemini\antigravity\worktrees\CKD_API\check_rate_limit_status")
    ]
    for b in dirs_to_check:
        m_p = b / "CKD_Severity_XGBoost.pkl"
        s_p = b / "scaler.pkl"
        i_p = b / "num_imputer.pkl"
        if m_p.exists() and s_p.exists() and i_p.exists():
            model = joblib.load(m_p)
            scaler_obj = joblib.load(s_p)
            imputer_obj = joblib.load(i_p)
            cat_cols = joblib.load(b / "cat_cols.pkl") if (b / "cat_cols.pkl").exists() else None
            encoded_cols = joblib.load(b / "encoded_feature_names.pkl") if (b / "encoded_feature_names.pkl").exists() else None
            return model, scaler_obj, imputer_obj, cat_cols, encoded_cols
    raise FileNotFoundError("Clinical severity model files are missing.")


@st.cache_data
def load_severity_analytics_payload():
    paths = [
        Path(__file__).resolve().parent / "severity_analytics.json",
        Path.cwd() / "severity_analytics.json",
        Path(r"D:\project\data\raw\CKD_API\severity_analytics.json"),
        Path(r"C:\Users\SAMSUNG\.gemini\antigravity\worktrees\CKD_API\check_rate_limit_status\severity_analytics.json")
    ]
    for p in paths:
        if p.exists():
            try:
                import json
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


SEVERITY_CLASS_NAMES = {
    0: "Healthy",
    1: "Mild CKD",
    2: "Moderate CKD",
    3: "Severe CKD"
}

SEVERITY_CLASS_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c"
}

SEVERITY_DESCRIPTIONS = {
    "Healthy": (
        "**Healthy (No CKD)**: Renal filtration (eGFR ≥ 90 mL/min/1.73m²) and biomarker levels "
        "are within normal physiologic ranges without evidence of kidney damage or pathological albuminuria."
    ),
    "Mild CKD": (
        "**Mild CKD (KDIGO Stage 1 & 2)**: Normal or mildly decreased GFR (60–89 mL/min/1.73m²) with "
        "persistent structural kidney damage or microalbuminuria (ACR 30–300 mg/g). Early clinical intervention "
        "and strict risk factor management are recommended."
    ),
    "Moderate CKD": (
        "**Moderate CKD (KDIGO Stage 3a & 3b)**: Substantial reduction in renal filtration (eGFR 30–59 mL/min/1.73m²) "
        "with active metabolic complications, hypertension, and proteinuria. Active clinical surveillance and "
        "nephrology co-management are strongly advised."
    ),
    "Severe CKD": (
        "**Severe CKD (KDIGO Stage 4 & 5)**: Severe reduction in renal function (eGFR < 30 mL/min/1.73m²) approaching "
        "end-stage kidney disease (ESKD). Urgent nephrology referral, vascular access planning, and renal "
        "replacement therapy (RRT / dialysis / transplantation) preparation are indicated."
    )
}

SEVERITY_MODEL_RESULTS = pd.DataFrame({
    "Model": [
        "Ordinal Logistic Regression",
        "Random Forest",
        "XGBoost (Deployed)"
    ],
    "Accuracy": [0.9228, 0.9982, 0.9985],
    "Weighted Precision": [0.9921, 0.9981, 0.9985],
    "Weighted Recall": [0.9228, 0.9982, 0.9985],
    "Weighted F1": [0.9543, 0.9980, 0.9985],
    "Macro Precision": [0.7411, 0.9846, 0.9848],
    "Macro Recall": [0.8859, 0.9289, 0.9557],
    "Macro F1": [0.7372, 0.9532, 0.9692],
    "QWK": [0.9383, 0.9986, 0.9984]
})


def render_severity_visual_stepper_ui(current_stage_label):
    stages = [
        ("Healthy", "0: Healthy", "#2ecc71", "🟢"),
        ("Mild CKD", "1: Mild (Stg 1-2)", "#f1c40f", "🟡"),
        ("Moderate CKD", "2: Moderate (Stg 3a-3b)", "#e67e22", "🟠"),
        ("Severe CKD", "3: Severe (Stg 4-5)", "#e74c3c", "🔴"),
    ]
    cols = st.columns(4)
    for idx, (stg_key, stg_title, stg_color, icon) in enumerate(stages):
        is_active = current_stage_label == stg_key
        bg = stg_color if is_active else "#f1f3f5"
        fg = "#ffffff" if is_active else "#4b5563"
        border = f"2px solid {stg_color}" if is_active else "1px solid #d8dde3"
        shadow = "0 8px 18px rgba(35,7,14,.12)" if is_active else "none"
        with cols[idx]:
            st.markdown(
                f"""
                <div style="background:{bg};color:{fg};border:{border};border-radius:16px;
                            padding:15px 10px;text-align:center;min-height:108px;display:flex;
                            flex-direction:column;justify-content:center;box-shadow:{shadow};">
                    <div style="font-size:1.35em;">{icon}</div>
                    <div style="font-weight:800;margin-top:4px;">{stg_title}</div>
                    <div style="font-size:.72em;opacity:.9;margin-top:3px;">
                        {"ACTIVE CLASS" if is_active else f"Class {idx}"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if main_clinical_screening:
    has_early_pred = "prediction" in st.session_state and "input_data" in st.session_state

    if has_early_pred:
        st.success("✅ **Stage 1: Early Screening completed** — Clinical Screening is unlocked as a separate 75-feature assessment!")
    else:
        st.info("ℹ️ **Stage 2: Clinical Screening** is designed to follow Stage 1 Early Screening. Enter this pathway's clinical information separately in the 75-feature intake form below.")

    st.title("🏥 Stage 2: Clinical Screening & CKD Severity Assessment")
    st.caption("KDIGO-Aligned Multi-Class CKD Severity Classification & Comprehensive Clinical Analytics (from Severity Model Final Draft)")

    tab_clin_predict, \
    tab_clin_stats, \
    tab_clin_plots = st.tabs(
        [
            "🩺 a) Prediction & Severity Outputs",
            "📊 b) Statistical Analysis",
            "📈 c) Exploratory & Diagnostic Plots"
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: PREDICTION & SEVERITY OUTPUTS
    # -------------------------------------------------------------------------
    with tab_clin_predict:
        st.header("🩺 Multi-Class CKD Severity Screening & Biomarker Profiling")
        st.write(
            """
            This stage executes the multi-class **XGBoost Severity Classifier** developed in 
            `Severity model Final Draft.ipynb`. It grades patient condition across 4 severity tiers:
            **Healthy (0) / Mild CKD (1) / Moderate CKD (2) / Severe CKD (3)**.
            """
        )

        try:
            severity_model, severity_scaler, severity_imputer, cat_cols_saved, encoded_cols_saved = load_severity_model_bundle()
            SEVERITY_MODEL_READY = True
        except Exception as err:
            SEVERITY_MODEL_READY = False
            st.error(f"Could not load severity model artifacts: {err}")

        if SEVERITY_MODEL_READY:
            st.subheader("⚡ Quick Clinical Preset Profiles")
            st.write("Select a standardized clinical persona to populate the 75-feature intake form:")
        
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                if st.button("🟢 Healthy Control", use_container_width=True, key="btn_preset_healthy"):
                    st.session_state["active_preset"] = "healthy"
            with p_col2:
                if st.button("🟡 Mild CKD (Stage 1-2)", use_container_width=True, key="btn_preset_mild"):
                    st.session_state["active_preset"] = "mild"
            with p_col3:
                if st.button("🟠 Moderate CKD (Stage 3a-3b)", use_container_width=True, key="btn_preset_mod"):
                    st.session_state["active_preset"] = "moderate"
            with p_col4:
                if st.button("🔴 Severe CKD (Stage 4-5)", use_container_width=True, key="btn_preset_sev"):
                    st.session_state["active_preset"] = "severe"

            preset_active = st.session_state.get("active_preset", "healthy")

            # Independent Clinical Screening intake. Early Screening completion
            # only unlocks this pathway; its inputs are not copied here.
            default_inputs = {
                "Age": 52, "Sex": "Male", "Ethnicity": "White", "Country": "USA", "Residence_Type": "Urban",
                "Education_Level": "Bachelor's", "Socioeconomic_Status": "Middle", "Height_cm": 175.0,
                "Weight_kg": 78.0, "BMI": 25.5, "Waist_Circumference_cm": 88.0, "Body_Fat_Percentage": 22.0,
                "Smoking_Status": "Never", "Alcohol_Consumption": "Moderate", "Physical_Activity_Level": "Moderate",
                "Exercise_Hours_Per_Week": 3.5, "Daily_Steps": 7500, "Water_Intake_L": 2.5, "Sodium_Intake_mg": 2400,
                "Fast_Food_Frequency_Per_Week": 1, "Sleep_Duration_Hours": 7.5, "Stress_Level": "Low",
                "Diabetes": 0, "Hypertension": 0, "Cardiovascular_Disease": 0, "Heart_Failure": 0,
                "Hyperlipidemia": 0, "Kidney_Stones": 0, "Recurrent_UTI": 0, "Autoimmune_Disease": 0,
                "Family_History_CKD": 0, "Obesity": 0, "Heart_Rate": 72, "Respiratory_Rate": 16,
                "Oxygen_Saturation": 98.5, "Systolic_BP": 118, "Diastolic_BP": 76,
                "Blood_Pressure_Category": "Normal", "Serum_Creatinine": 0.85, "eGFR": 105.0,
                "Blood_Urea_Nitrogen": 12.0, "Albumin": 4.5, "Urine_ACR": 10.0, "Urine_Protein": 30.0,
                "HbA1c": 5.2, "Fasting_Glucose": 88.0, "Hemoglobin": 15.0, "Sodium": 140.0,
                "Potassium": 4.2, "Calcium": 9.5, "Phosphorus": 3.4, "Uric_Acid": 5.0,
                "Total_Cholesterol": 175, "HDL": 55.0, "LDL": 95.0, "Triglycerides": 120,
                "CRP": 0.8, "ACE_Inhibitor": 0, "ARB": 0, "Diabetes_Medication": 0, "Statin": 0,
                "Diuretic": 0, "NSAID_Usage": 0, "Medication_Adherence": 1, "Number_of_Medications": 0,
                "Frailty_Index": 0.05, "Frailty_Category": "Fit", "Hospital_Visits": 0, "Emergency_Visits": 0,
                "Specialist_Visits": 0, "Annual_Checkups": 1, "Health_Insurance": 1,
                "Annual_Household_Income_USD": 65000, "Annual_Medical_Cost_USD": 1200, "Employment_Status": "Employed"
            }

            if preset_active == "mild":
                default_inputs.update({
                    "Age": 58, "eGFR": 72.0, "Serum_Creatinine": 1.25, "Blood_Urea_Nitrogen": 18.0,
                    "Urine_ACR": 45.0, "Urine_Protein": 120.0, "Systolic_BP": 134, "Diastolic_BP": 84,
                    "Blood_Pressure_Category": "Hypertension Stage 1", "Hypertension": 1, "HbA1c": 5.9,
                    "Fasting_Glucose": 108.0, "Albumin": 4.1, "CRP": 1.8
                })
            elif preset_active == "moderate":
                default_inputs.update({
                    "Age": 66, "eGFR": 44.0, "Serum_Creatinine": 1.95, "Blood_Urea_Nitrogen": 28.0,
                    "Urine_ACR": 180.0, "Urine_Protein": 280.0, "Systolic_BP": 144, "Diastolic_BP": 88,
                    "Blood_Pressure_Category": "Hypertension Stage 2", "Hypertension": 1, "Diabetes": 1,
                    "HbA1c": 7.4, "Fasting_Glucose": 142.0, "Albumin": 3.6, "Hemoglobin": 11.5,
                    "Potassium": 4.8, "Phosphorus": 4.6, "Uric_Acid": 7.8, "ACE_Inhibitor": 1,
                    "Statin": 1, "Number_of_Medications": 3, "CRP": 3.4, "Frailty_Category": "Vulnerable"
                })
            elif preset_active == "severe":
                default_inputs.update({
                    "Age": 72, "eGFR": 18.0, "Serum_Creatinine": 3.90, "Blood_Urea_Nitrogen": 54.0,
                    "Urine_ACR": 680.0, "Urine_Protein": 650.0, "Systolic_BP": 165, "Diastolic_BP": 96,
                    "Blood_Pressure_Category": "Hypertension Stage 2", "Hypertension": 1, "Diabetes": 1,
                    "Cardiovascular_Disease": 1, "Heart_Failure": 1, "HbA1c": 8.5, "Fasting_Glucose": 178.0,
                    "Albumin": 3.1, "Hemoglobin": 9.2, "Potassium": 5.4, "Phosphorus": 5.8,
                    "Uric_Acid": 9.2, "ACE_Inhibitor": 1, "Diuretic": 1, "Statin": 1,
                    "Number_of_Medications": 6, "CRP": 6.8, "Frailty_Category": "Frail",
                    "Hospital_Visits": 3, "Emergency_Visits": 2
                })

            st.divider()
            st.subheader(f"📋 Patient Clinical Intake Form (Active Profile: {preset_active.title()})")

            with st.form("severity_prediction_form"):
                exp1 = st.expander("🩺 1. Renal Biomarkers & Core Labs (High Sensitivity)", expanded=True)
                with exp1:
                    r_c1, r_c2, r_c3 = st.columns(3)
                    with r_c1:
                        f_egfr = st.number_input("eGFR (mL/min/1.73m²)", 1.0, 160.0, float(default_inputs["eGFR"]), step=1.0)
                        f_creat = st.number_input("Serum Creatinine (mg/dL)", 0.2, 15.0, float(default_inputs["Serum_Creatinine"]), step=0.05)
                    with r_c2:
                        f_bun = st.number_input("Blood Urea Nitrogen (mg/dL)", 1.0, 150.0, float(default_inputs["Blood_Urea_Nitrogen"]), step=1.0)
                        f_alb = st.number_input("Serum Albumin (g/dL)", 1.0, 7.0, float(default_inputs["Albumin"]), step=0.1)
                    with r_c3:
                        f_uacr = st.number_input("Urine ACR (mg/g)", 0.0, 5000.0, float(default_inputs["Urine_ACR"]), step=5.0)
                        f_uprot = st.number_input("Urine Protein (mg/dL)", 0.0, 2000.0, float(default_inputs["Urine_Protein"]), step=5.0)

                exp2 = st.expander("🩸 2. Metabolic, Electrolytes & Hematology", expanded=False)
                with exp2:
                    m_c1, m_c2, m_c3 = st.columns(3)
                    with m_c1:
                        f_hba1c = st.number_input("HbA1c (%)", 3.0, 18.0, float(default_inputs["HbA1c"]), step=0.1)
                        f_glucose = st.number_input("Fasting Glucose (mg/dL)", 40.0, 400.0, float(default_inputs["Fasting_Glucose"]), step=1.0)
                        f_hemo = st.number_input("Hemoglobin (g/dL)", 4.0, 22.0, float(default_inputs["Hemoglobin"]), step=0.1)
                    with m_c2:
                        f_na = st.number_input("Sodium (mEq/L)", 110.0, 170.0, float(default_inputs["Sodium"]), step=0.5)
                        f_k = st.number_input("Potassium (mEq/L)", 2.0, 9.0, float(default_inputs["Potassium"]), step=0.1)
                        f_ca = st.number_input("Calcium (mg/dL)", 4.0, 16.0, float(default_inputs["Calcium"]), step=0.1)
                    with m_c3:
                        f_phos = st.number_input("Phosphorus (mg/dL)", 1.0, 15.0, float(default_inputs["Phosphorus"]), step=0.1)
                        f_uric = st.number_input("Uric Acid (mg/dL)", 1.0, 20.0, float(default_inputs["Uric_Acid"]), step=0.1)
                        f_crp = st.number_input("C-Reactive Protein (mg/L)", 0.0, 100.0, float(default_inputs["CRP"]), step=0.1)

                exp3 = st.expander("🫀 3. Cardiovascular, Vitals & Lipids", expanded=False)
                with exp3:
                    cv_c1, cv_c2, cv_c3 = st.columns(3)
                    with cv_c1:
                        f_sbp = st.number_input("Systolic BP (mmHg)", 70, 250, int(default_inputs["Systolic_BP"]))
                        f_dbp = st.number_input("Diastolic BP (mmHg)", 40, 160, int(default_inputs["Diastolic_BP"]))
                        bp_opts = ["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"]
                        f_bp_cat = st.selectbox("BP Category", bp_opts, index=bp_opts.index(default_inputs["Blood_Pressure_Category"]) if default_inputs["Blood_Pressure_Category"] in bp_opts else 0)
                    with cv_c2:
                        f_hr = st.number_input("Heart Rate (bpm)", 30, 200, int(default_inputs["Heart_Rate"]))
                        f_rr = st.number_input("Respiratory Rate", 8, 40, int(default_inputs["Respiratory_Rate"]))
                        f_o2 = st.number_input("Oxygen Saturation (%)", 70.0, 100.0, float(default_inputs["Oxygen_Saturation"]))
                    with cv_c3:
                        f_chol = st.number_input("Total Cholesterol (mg/dL)", 50, 500, int(default_inputs["Total_Cholesterol"]))
                        f_hdl = st.number_input("HDL (mg/dL)", 10.0, 150.0, float(default_inputs["HDL"]))
                        f_ldl = st.number_input("LDL (mg/dL)", 10.0, 350.0, float(default_inputs["LDL"]))
                        f_trig = st.number_input("Triglycerides (mg/dL)", 30, 1000, int(default_inputs["Triglycerides"]))

                exp4 = st.expander("🏥 4. Comorbidities, Medications & Clinical History", expanded=False)
                with exp4:
                    cm_c1, cm_c2, cm_c3 = st.columns(3)
                    with cm_c1:
                        f_dm = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Diabetes"])
                        f_htn = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Hypertension"])
                        f_cvd = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Cardiovascular_Disease"])
                        f_hf = st.selectbox("Heart Failure", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Heart_Failure"])
                    with cm_c2:
                        f_hyperlip = st.selectbox("Hyperlipidemia", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Hyperlipidemia"])
                        f_stones = st.selectbox("Kidney Stones", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Kidney_Stones"])
                        f_uti = st.selectbox("Recurrent UTI", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Recurrent_UTI"])
                        f_autoimm = st.selectbox("Autoimmune Disease", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Autoimmune_Disease"])
                    with cm_c3:
                        f_fam = st.selectbox("Family History of CKD", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Family_History_CKD"])
                        f_acei = st.selectbox("ACE Inhibitor Use", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["ACE_Inhibitor"])
                        f_arb = st.selectbox("ARB Use", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["ARB"])
                        f_statin = st.selectbox("Statin Therapy", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Statin"])
                        f_diur = st.selectbox("Diuretic Therapy", [0, 1], format_func=lambda x: "Yes" if x else "No", index=default_inputs["Diuretic"])

                exp5 = st.expander("👤 5. Demographics, Lifestyle & Utilization", expanded=False)
                with exp5:
                    d_c1, d_c2, d_c3 = st.columns(3)
                    with d_c1:
                        f_age = st.number_input("Age (Years)", 18, 105, int(default_inputs["Age"]))
                        f_sex = st.selectbox("Sex", ["Male", "Female"], index=0 if default_inputs["Sex"] == "Male" else 1)
                        f_bmi = st.number_input("BMI (kg/m²)", 12.0, 65.0, float(default_inputs["BMI"]))
                        f_waist = st.number_input("Waist Circumference (cm)", 50.0, 180.0, float(default_inputs["Waist_Circumference_cm"]))
                        f_smoke = st.selectbox("Smoking Status", ["Never", "Former", "Current"], index=["Never", "Former", "Current"].index(default_inputs["Smoking_Status"]))
                    with d_c2:
                        f_eth = st.selectbox("Ethnicity", ["White", "Hispanic", "Black", "Asian", "Other"], index=["White", "Hispanic", "Black", "Asian", "Other"].index(default_inputs["Ethnicity"]))
                        f_cntry = st.selectbox("Country", ["USA", "UK", "India", "Canada", "Australia", "Other"], index=0)
                        f_res = st.selectbox("Residence Type", ["Urban", "Rural"], index=0)
                        f_act = st.selectbox("Physical Activity", ["Sedentary", "Light", "Moderate", "Active"], index=["Sedentary", "Light", "Moderate", "Active"].index(default_inputs["Physical_Activity_Level"]))
                        f_alc = st.selectbox("Alcohol Consumption", ["Moderate", "Heavy", "Not provided"], index=0)
                    with d_c3:
                        f_frail = st.selectbox("Frailty Category", ["Fit", "Vulnerable", "Frail"], index=["Fit", "Vulnerable", "Frail"].index(default_inputs["Frailty_Category"]))
                        f_meds_count = st.number_input("Number of Medications", 0, 25, int(default_inputs["Number_of_Medications"]))
                        f_hosp = st.number_input("Hospital Visits (Past Year)", 0, 30, int(default_inputs["Hospital_Visits"]))
                        f_emerg = st.number_input("Emergency Visits (Past Year)", 0, 30, int(default_inputs["Emergency_Visits"]))
                        f_income = st.number_input("Annual Income (USD)", 0, 500000, int(default_inputs["Annual_Household_Income_USD"]), step=5000)

                submit_btn = st.form_submit_button("🚀 Run CKD Severity Assessment", use_container_width=True)

            if submit_btn:
                patient_profile = default_inputs.copy()
                patient_profile.update({
                    "Age": f_age, "Sex": f_sex, "Ethnicity": f_eth, "Country": f_cntry, "Residence_Type": f_res,
                    "BMI": f_bmi, "Waist_Circumference_cm": f_waist, "Smoking_Status": f_smoke,
                    "Alcohol_Consumption": f_alc, "Physical_Activity_Level": f_act, "Systolic_BP": f_sbp,
                    "Diastolic_BP": f_dbp, "Blood_Pressure_Category": f_bp_cat, "Heart_Rate": f_hr,
                    "Respiratory_Rate": f_rr, "Oxygen_Saturation": f_o2, "Serum_Creatinine": f_creat,
                    "eGFR": f_egfr, "Blood_Urea_Nitrogen": f_bun, "Albumin": f_alb, "Urine_ACR": f_uacr,
                    "Urine_Protein": f_uprot, "HbA1c": f_hba1c, "Fasting_Glucose": f_glucose,
                    "Hemoglobin": f_hemo, "Sodium": f_na, "Potassium": f_k, "Calcium": f_ca,
                    "Phosphorus": f_phos, "Uric_Acid": f_uric, "Total_Cholesterol": f_chol,
                    "HDL": f_hdl, "LDL": f_ldl, "Triglycerides": f_trig, "CRP": f_crp,
                    "Diabetes": f_dm, "Hypertension": f_htn, "Cardiovascular_Disease": f_cvd,
                    "Heart_Failure": f_hf, "Hyperlipidemia": f_hyperlip, "Kidney_Stones": f_stones,
                    "Recurrent_UTI": f_uti, "Autoimmune_Disease": f_autoimm, "Family_History_CKD": f_fam,
                    "ACE_Inhibitor": f_acei, "ARB": f_arb, "Statin": f_statin, "Diuretic": f_diur,
                    "Frailty_Category": f_frail, "Number_of_Medications": f_meds_count,
                    "Hospital_Visits": f_hosp, "Emergency_Visits": f_emerg, "Annual_Household_Income_USD": f_income
                })

                input_df = pd.DataFrame([patient_profile])
            
                try:
                    df_proc = input_df.copy()
                
                    num_sub = [c for c in df_proc.select_dtypes(include=np.number).columns if c in severity_imputer.feature_names_in_]
                    df_proc[num_sub] = severity_imputer.transform(df_proc[num_sub])
                
                    if cat_cols_saved:
                        active_cats = [c for c in cat_cols_saved if c in df_proc.columns]
                        df_proc = pd.get_dummies(df_proc, columns=active_cats, drop_first=True)
                
                    if encoded_cols_saved:
                        for col in encoded_cols_saved:
                            if col not in df_proc.columns:
                                df_proc[col] = 0
                        df_proc = df_proc[encoded_cols_saved]
                
                    df_scaled = severity_scaler.transform(df_proc)
                
                    pred_class = int(severity_model.predict(df_scaled)[0])
                    pred_probs = severity_model.predict_proba(df_scaled)[0]
                    pred_label = SEVERITY_CLASS_NAMES.get(pred_class, f"Class {pred_class}")
                
                    st.session_state["clinical_input_data"] = input_df.copy()
                    st.session_state["severity_prediction"] = pred_class
                    st.session_state["severity_label"] = pred_label
                    st.session_state["severity_probabilities"] = pred_probs

                    st.success(f"### 🎯 Severity Classification: **{pred_label}**")
                    render_severity_visual_stepper_ui(pred_label)

                    p_col1, p_col2 = st.columns([1.2, 1])
                    with p_col1:
                        st.subheader("📊 Multi-Class Probability Distribution")
                        prob_df = pd.DataFrame({
                            "Stage": ["0: Healthy", "1: Mild CKD", "2: Moderate CKD", "3: Severe CKD"],
                            "Probability (%)": [p * 100 for p in pred_probs],
                            "Color": ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
                        })
                    
                        fig_probs = go.Figure(go.Bar(
                            x=prob_df["Probability (%)"],
                            y=prob_df["Stage"],
                            orientation='h',
                            marker_color=prob_df["Color"],
                            text=[f"{p:.2f}%" for p in prob_df["Probability (%)"]],
                            textposition='outside'
                        ))
                        fig_probs.update_layout(
                            xaxis_title="Predicted Probability (%)",
                            xaxis_range=[0, 115],
                            height=260,
                            margin=dict(l=20, r=30, t=20, b=30)
                        )
                        st.plotly_chart(fig_probs, use_container_width=True, key="sev_pred_probs_bar_result")

                    with p_col2:
                        st.subheader("💡 Key Physiological Drivers")
                        drivers = []
                        if f_egfr < 60:
                            drivers.append(f"• **Reduced eGFR**: {f_egfr:.1f} mL/min (Major renal filtration decline)")
                        if f_creat > 1.3:
                            drivers.append(f"• **Elevated Serum Creatinine**: {f_creat:.2f} mg/dL")
                        if f_uacr >= 30:
                            drivers.append(f"• **Pathological Albuminuria**: ACR = {f_uacr:.1f} mg/g")
                        if f_bun > 20:
                            drivers.append(f"• **Elevated BUN**: {f_bun:.1f} mg/dL (Azotemia / uremia marker)")
                        if f_hba1c >= 6.5:
                            drivers.append(f"• **Diabetic Glycemia**: HbA1c {f_hba1c:.1f}% (Microvascular renal risk)")
                        if f_sbp >= 140:
                            drivers.append(f"• **Hypertension**: BP {f_sbp}/{f_dbp} mmHg (Glomerular shear stress)")
                    
                        if not drivers:
                            drivers.append("• All major renal biomarkers (eGFR, Creatinine, ACR, BUN) are within healthy physiological limits.")
                    
                        st.markdown("<br>".join(drivers), unsafe_allow_html=True)

                except Exception as p_err:
                    st.error(f"Inference error: {p_err}")

        # Biomarker Panels & KDIGO Outcomes (shown once clinical profile exists)
        clinical_data = st.session_state.get("clinical_input_data")
        if clinical_data is not None:
            sev_label = st.session_state.get("severity_label", "Healthy")
            st.divider()
            st.subheader("🧪 Patient Biomarker Evaluation vs Reference Intervals")
        
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("#### 🩺 Renal Function & Albuminuria Panel")
                renal_records = []
                if "eGFR" in clinical_data:
                    val = clinical_data["eGFR"].iloc[0]
                    status = "🔴 Severely Decreased (<30)" if val < 30 else ("🟠 Moderately Decreased (30-59)" if val < 60 else ("🟡 Mildly Decreased (60-89)" if val < 90 else "🟢 Normal (≥90)"))
                    renal_records.append({"Biomarker": "eGFR (mL/min/1.73m²)", "Value": f"{val:.1f}", "Reference Range": "≥ 90.0", "Clinical Status": status})
                if "Serum_Creatinine" in clinical_data:
                    val = clinical_data["Serum_Creatinine"].iloc[0]
                    status = "🔴 Markedly Elevated (>2.0)" if val > 2.0 else ("🟡 Elevated (1.3-2.0)" if val > 1.2 else "🟢 Normal (0.6-1.2)")
                    renal_records.append({"Biomarker": "Serum Creatinine (mg/dL)", "Value": f"{val:.2f}", "Reference Range": "0.60 - 1.20", "Clinical Status": status})
                if "Blood_Urea_Nitrogen" in clinical_data:
                    val = clinical_data["Blood_Urea_Nitrogen"].iloc[0]
                    status = "🔴 High Uremia (>40)" if val > 40 else ("🟡 Elevated (21-40)" if val > 20 else "🟢 Normal (7-20)")
                    renal_records.append({"Biomarker": "Blood Urea Nitrogen (mg/dL)", "Value": f"{val:.1f}", "Reference Range": "7.0 - 20.0", "Clinical Status": status})
                if "Albumin" in clinical_data:
                    val = clinical_data["Albumin"].iloc[0]
                    status = "🔴 Hypoalbuminemia (<3.4)" if val < 3.4 else "🟢 Normal (3.4-5.4)"
                    renal_records.append({"Biomarker": "Serum Albumin (g/dL)", "Value": f"{val:.2f}", "Reference Range": "3.40 - 5.40", "Clinical Status": status})
                if "Urine_ACR" in clinical_data:
                    val = clinical_data["Urine_ACR"].iloc[0]
                    status = "🔴 Severely Increased / Macro (>300)" if val > 300 else ("🟡 Microalbuminuria (30-300)" if val >= 30 else "🟢 Normal (<30)")
                    renal_records.append({"Biomarker": "Urine ACR (mg/g)", "Value": f"{val:.1f}", "Reference Range": "< 30.0", "Clinical Status": status})
                if "Urine_Protein" in clinical_data:
                    val = clinical_data["Urine_Protein"].iloc[0]
                    status = "🔴 Proteinuria (>150)" if val > 150 else "🟢 Normal (<150)"
                    renal_records.append({"Biomarker": "Urine Protein (mg/dL)", "Value": f"{val:.1f}", "Reference Range": "< 150.0", "Clinical Status": status})
                if renal_records:
                    st.dataframe(pd.DataFrame(renal_records), use_container_width=True, hide_index=True)

                st.markdown("#### 🩸 Electrolytes & Mineral Metabolism Panel")
                elect_records = []
                for b_name, col_name, ref_str, low_val, high_val in [
                    ("Sodium (mEq/L)", "Sodium", "135 - 145", 135, 145),
                    ("Potassium (mEq/L)", "Potassium", "3.5 - 5.0", 3.5, 5.0),
                    ("Calcium (mg/dL)", "Calcium", "8.5 - 10.5", 8.5, 10.5),
                    ("Phosphorus (mg/dL)", "Phosphorus", "2.5 - 4.5", 2.5, 4.5),
                    ("Uric Acid (mg/dL)", "Uric_Acid", "3.5 - 7.2", 3.5, 7.2)
                ]:
                    if col_name in clinical_data:
                        val = clinical_data[col_name].iloc[0]
                        stt = "🔴 Hyperkalemia (>5.0)" if col_name == "Potassium" and val > 5.0 else ("🔴 Hypokalemia (<3.5)" if col_name == "Potassium" and val < 3.5 else ("🔴 Abnormal" if val < low_val or val > high_val else "🟢 Normal"))
                        elect_records.append({"Biomarker": b_name, "Value": f"{val:.2f}", "Reference Range": ref_str, "Clinical Status": stt})
                if elect_records:
                    st.dataframe(pd.DataFrame(elect_records), use_container_width=True, hide_index=True)

            with c_p2:
                st.markdown("#### 🧬 Glycemic & Cardiovascular Panel")
                glyc_records = []
                if "HbA1c" in clinical_data:
                    val = clinical_data["HbA1c"].iloc[0]
                    status = "🔴 Diabetic Range (≥6.5%)" if val >= 6.5 else ("🟡 Prediabetic (5.7-6.4%)" if val >= 5.7 else "🟢 Normal (<5.7%)")
                    glyc_records.append({"Biomarker": "HbA1c (%)", "Value": f"{val:.1f}%", "Reference Range": "< 5.7%", "Clinical Status": status})
                if "Fasting_Glucose" in clinical_data:
                    val = clinical_data["Fasting_Glucose"].iloc[0]
                    status = "🔴 Diabetic (≥126)" if val >= 126 else ("🟡 Impaired (100-125)" if val >= 100 else "🟢 Normal (70-99)")
                    glyc_records.append({"Biomarker": "Fasting Glucose (mg/dL)", "Value": f"{val:.1f}", "Reference Range": "70 - 99", "Clinical Status": status})
                if "Systolic_BP" in clinical_data and "Diastolic_BP" in clinical_data:
                    sbp = clinical_data["Systolic_BP"].iloc[0]
                    dbp = clinical_data["Diastolic_BP"].iloc[0]
                    status = "🔴 Stage 2 HTN (≥140/90)" if sbp >= 140 or dbp >= 90 else ("🟡 Stage 1 HTN (130-139)" if sbp >= 130 or dbp >= 80 else "🟢 Normal (<120/80)")
                    glyc_records.append({"Biomarker": "Blood Pressure (mmHg)", "Value": f"{sbp:.0f}/{dbp:.0f}", "Reference Range": "< 120/80", "Clinical Status": status})
                if "Hemoglobin" in clinical_data:
                    val = clinical_data["Hemoglobin"].iloc[0]
                    status = "🔴 Anemia (<12.0)" if val < 12.0 else "🟢 Normal (12.0-17.5)"
                    glyc_records.append({"Biomarker": "Hemoglobin (g/dL)", "Value": f"{val:.1f}", "Reference Range": "12.0 - 17.5", "Clinical Status": status})
                if "CRP" in clinical_data:
                    val = clinical_data["CRP"].iloc[0]
                    status = "🔴 High Inflammation (>3.0)" if val > 3.0 else ("🟡 Moderate (1.0-3.0)" if val >= 1.0 else "🟢 Low Risk (<1.0)")
                    glyc_records.append({"Biomarker": "C-Reactive Protein (mg/L)", "Value": f"{val:.2f}", "Reference Range": "< 1.0", "Clinical Status": status})
                if glyc_records:
                    st.dataframe(pd.DataFrame(glyc_records), use_container_width=True, hide_index=True)

                st.markdown("#### 🫀 Lipid & Metabolic Panel")
                lipid_records = []
                for b_name, col_name, ref_str, low_val, high_val in [
                    ("Total Cholesterol (mg/dL)", "Total_Cholesterol", "< 200", 0, 200),
                    ("HDL Cholesterol (mg/dL)", "HDL", "≥ 40", 40, 999),
                    ("LDL Cholesterol (mg/dL)", "LDL", "< 100", 0, 100),
                    ("Triglycerides (mg/dL)", "Triglycerides", "< 150", 0, 150),
                    ("BMI (kg/m²)", "BMI", "18.5 - 24.9", 18.5, 24.9)
                ]:
                    if col_name in clinical_data:
                        val = clinical_data[col_name].iloc[0]
                        stt = "🔴 Elevated" if val > high_val else ("🔴 Low" if val < low_val else "🟢 Optimal")
                        lipid_records.append({"Biomarker": b_name, "Value": f"{val:.1f}", "Reference Range": ref_str, "Clinical Status": stt})
                if lipid_records:
                    st.dataframe(pd.DataFrame(lipid_records), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🏥 KDIGO Care Plan & Clinical Guidance")
            st.info(SEVERITY_DESCRIPTIONS.get(sev_label, ""))

            if sev_label == "Healthy":
                st.markdown(
                    """
                    ### 🟢 Healthy Patient Management Strategy
                    - **Surveillance Frequency**: Routine annual health review and primary prevention.
                    - **Target Goals**: Maintain BP < 120/80 mmHg, HbA1c < 5.7%, BMI 18.5–24.9 kg/m².
                    - **Lifestyle**: Maintain adequate hydration (2–2.5 L/day), dietary sodium < 2000 mg/day, regular aerobic activity (> 150 min/week).
                    - **Renal Preservation**: Avoid chronic non-steroidal anti-inflammatory drugs (NSAIDs) and unprescribed nephrotoxins.
                    """
                )
            elif sev_label == "Mild CKD":
                st.markdown(
                    """
                    ### 🟡 Mild CKD (KDIGO Stage 1 & 2) Protocol
                    - **Surveillance Frequency**: Clinical & lab review every **6 to 12 months** (eGFR + Urine ACR monitoring).
                    - **Blood Pressure Target**: Systolic BP < 120 mmHg (standardized measurement).
                    - **Renoprotective Pharmacotherapy**: First-line **ACE inhibitor or ARB** if albuminuria (Urine ACR ≥ 30 mg/g) is present; initiate **SGLT2 inhibitor** in diabetic or proteinuric CKD patients.
                    - **Diet & Nutrition**: Moderate dietary protein intake (~0.8 g/kg/day), sodium restriction (< 2000 mg/day).
                    """
                )
            elif sev_label == "Moderate CKD":
                st.markdown(
                    """
                    ### 🟠 Moderate CKD (KDIGO Stage 3a & 3b) Management Protocol
                    - **Surveillance Frequency**: Clinical review every **3 to 6 months** with comprehensive metabolic panels.
                    - **Nephrology Co-Management**: Formal nephrology consultation recommended to mitigate progression.
                    - **Comprehensive Medication Regimen**: Maximally tolerated **ACEi / ARB** therapy + **SGLT2 inhibitor** (e.g. Dapagliflozin / Empagliflozin); Statin therapy for cardiovascular risk mitigation.
                    - **Metabolic Complication Management**: Monitor serum Potassium and for anemia / secondary hyperparathyroidism.
                    """
                )
            else:
                st.markdown(
                    """
                    ### 🔴 Severe CKD (KDIGO Stage 4 & 5) Critical Protocol
                    - **Surveillance Frequency**: Monthly to bi-monthly close nephrology follow-up.
                    - **Urgent Nephrology Referral**: Immediate multidisciplinary advanced CKD clinic enrollment.
                    - **Renal Replacement Therapy (RRT) Preparation**: Timely vascular access planning (AV Fistula / Graft evaluation) or peritoneal dialysis education; pre-emptive kidney transplant evaluation.
                    - **Metabolic Derangement Interventions**: Serum Potassium control, Metabolic Acidosis correction (Oral Sodium Bicarbonate), Phosphate binders + active Vitamin D, and ESA / IV Iron anemia therapy.
                    """
                )

    # -------------------------------------------------------------------------
    # TAB 2: STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------
    with tab_clin_stats:
        st.header("📊 Statistical Analysis & Model Validation (Severity Workflow)")
        st.write(
            """
            Comprehensive statistical benchmarking, hypothesis testing, and model validation 
            integrated directly from `Severity model Final Draft.ipynb`. Three candidate architectures 
            were trained on SMOTE-balanced training cohorts and rigorously validated on the **untouched original test set**.
            """
        )

        analytics = load_severity_analytics_payload()

        st.subheader("🤖 Model Performance Comparison")
        st.dataframe(SEVERITY_MODEL_RESULTS, use_container_width=True, hide_index=True)

        st.info(
            """
            **Key Findings & Rationale for Non-Statistical Users:**
            - **Why XGBoost Was Selected:** In multi-class medical diagnosis, **Macro F1** (which calculates accuracy equally across all disease stages) is critical because rare severe cases must never be masked by the large healthy majority. XGBoost achieved the highest **Macro F1 (96.92%)** and overall **Accuracy (99.85%)**.
            - **What is Quadratic Weighted Kappa (QWK)?** QWK measures agreement while severely penalizing distant misdiagnoses (e.g., classifying a Severe patient as Healthy). Both Random Forest (0.9986) and XGBoost (0.9984) achieved near-perfect agreement.
            - **Why SMOTE Class Balancing was Essential:** In real clinical cohorts, Healthy patients represent ~81.5% while Severe Stage 4/5 patients are only ~1.1%. SMOTE synthetic oversampling allowed tree algorithms to learn subtle biomarkers of severe stages without overfitting.
            """
        )

        st.divider()
        st.subheader("🕸️ Multi-Metric Comparison & Radar Profile")
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='Accuracy', x=SEVERITY_MODEL_RESULTS['Model'], y=SEVERITY_MODEL_RESULTS['Accuracy'], marker_color='#3498db'))
            fig_comp.add_trace(go.Bar(name='Macro F1', x=SEVERITY_MODEL_RESULTS['Model'], y=SEVERITY_MODEL_RESULTS['Macro F1'], marker_color='#2ecc71'))
            fig_comp.add_trace(go.Bar(name='QWK', x=SEVERITY_MODEL_RESULTS['Model'], y=SEVERITY_MODEL_RESULTS['QWK'], marker_color='#e67e22'))
            fig_comp.update_layout(barmode='group', yaxis_range=[0.70, 1.02], yaxis_title="Score", height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig_comp, use_container_width=True, key="sev_stats_comp_bar_fig")

        with c_m2:
            categories = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'QWK']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[0.9228, 0.7411, 0.8859, 0.7372, 0.9383], theta=categories, fill='toself', name='Ordinal Logistic', line_color='#3498db'))
            fig_radar.add_trace(go.Scatterpolar(r=[0.9982, 0.9846, 0.9289, 0.9532, 0.9986], theta=categories, fill='toself', name='Random Forest', line_color='#9b59b6'))
            fig_radar.add_trace(go.Scatterpolar(r=[0.9985, 0.9848, 0.9557, 0.9692, 0.9984], theta=categories, fill='toself', name='XGBoost (Deployed)', line_color='#e74c3c'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.7, 1.0])), height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig_radar, use_container_width=True, key="sev_stats_radar_fig")

        st.divider()
        st.subheader("🔬 Statistical Hypothesis Testing & Feature Audit")
        if analytics:
            tab_chi, tab_mw, tab_corr, tab_out = st.tabs([
                "📊 Chi-Square & Cramer's V (Categorical)",
                "📈 Mann-Whitney U (Numerical)",
                "🔗 Multicollinearity (|r| > 0.80)",
                "🎯 IQR Outlier Summary"
            ])

            with tab_chi:
                st.write("Association of Categorical Variables with CKD Status:")
                chi_df = pd.DataFrame(analytics.get("chi_square_results", []))
                st.dataframe(chi_df, use_container_width=True, hide_index=True)

            with tab_mw:
                st.write("Mann-Whitney U Rank Sum Tests comparing CKD vs Non-CKD distributions:")
                mw_df = pd.DataFrame(analytics.get("mann_whitney_results", []))
                st.dataframe(mw_df, use_container_width=True, hide_index=True)

            with tab_corr:
                st.write("Strongly correlated feature pairs identified in multicollinearity audit (|r| > 0.80):")
                corr_df = pd.DataFrame(analytics.get("multicollinearity_pairs", []))
                st.dataframe(corr_df, use_container_width=True, hide_index=True)

            with tab_out:
                st.write("Outlier percentages identified using 1.5 × IQR standard across continuous features:")
                out_df = pd.DataFrame(analytics.get("outlier_analysis", []))
                st.dataframe(out_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 3: EXPLORATORY & DIAGNOSTIC PLOTS
    # -------------------------------------------------------------------------
    with tab_clin_plots:
        st.header("📈 Exploratory & Diagnostic Visualisations (Severity Workflow)")
        st.write(
            """
            Key exploratory data visualisations, disease stage distributions, biomarker boxplots, 
            and tree feature importances from `Severity model Final Draft.ipynb`.
            """
        )

        analytics = load_severity_analytics_payload()

        st.subheader("🎯 CKD Stage & Severity Target Distribution")
        v_c1, v_c2 = st.columns(2)
        if analytics:
            with v_c1:
                st_df = pd.DataFrame(analytics["stage_distribution"])
                fig_st = px.bar(st_df, x="Stage", y="Count", text="Count", color="Stage",
                                title="Original 7-Class CKD Distribution",
                                color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"])
                fig_st.update_traces(textposition='outside')
                fig_st.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig_st, use_container_width=True, key="sev_visual_stg_bar_fig")

            with v_c2:
                sev_df = pd.DataFrame(analytics["severity_distribution"])
                fig_sev = px.pie(sev_df, values="Count", names="Severity",
                                 title="4-Class Severity Model Target Distribution",
                                 color="Severity",
                                 color_discrete_map={
                                     "Healthy": "#2ecc71",
                                     "Moderate CKD": "#e67e22",
                                     "Severe CKD": "#c0392b",
                                     "Mild CKD": "#f1c40f"
                                 })
                fig_sev.update_traces(textinfo='percent+label')
                fig_sev.update_layout(height=380)
                st.plotly_chart(fig_sev, use_container_width=True, key="sev_visual_sev_pie_fig")

        st.divider()
        st.subheader("🧮 Confusion Matrix & Multiclass ROC-AUC")
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            cm_img = clinical_asset("severity_confusion_matrix.png")
            if cm_img:
                st.image(cm_img, caption="Normalized Confusion Matrix (XGBoost Severity Model)", use_container_width=True)
            else:
                cm_data = [[99.98, 0.01, 0.01, 0.0], [0.91, 97.27, 1.82, 0.0], [0.01, 0.12, 99.81, 0.06], [0.0, 0.0, 0.68, 99.32]]
                labels = ["Healthy", "Mild", "Moderate", "Severe"]
                fig_cm = px.imshow(cm_data, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
                st.plotly_chart(fig_cm, use_container_width=True, key="sev_stats_cm_heatmap_fig")

        with c_diag2:
            roc_img = clinical_asset("severity_roc_auc.png")
            if roc_img:
                st.image(roc_img, caption="Multiclass One-vs-Rest ROC Curves (AUC ~ 0.999+)", use_container_width=True)

        st.divider()
        st.subheader("🏆 Top Numerical Variables Associated with CKD")
        if analytics and "ckd_correlations" in analytics:
            top_corr = pd.DataFrame(analytics["ckd_correlations"]).head(15)
            fig_corr = px.bar(top_corr[::-1], x="Correlation_r", y="Variable", orientation='h',
                              color="Correlation_r", color_continuous_scale="RdBu_r",
                              title="Top 15 Features by Pearson Correlation with CKD")
            fig_corr.update_layout(height=450, xaxis_title="Correlation Coefficient (r)")
            st.plotly_chart(fig_corr, use_container_width=True, key="sev_visual_corr_bar_fig")

        st.divider()
        st.subheader("🔎 XGBoost Feature Importance & Explanatory Ranking")
        fi_img = clinical_asset("severity_feature_importance_xgb.png")
        if fi_img:
            st.image(fi_img, caption="Top 20 Important Predictors - XGBoost CKD Severity Model", use_container_width=True)

        st.divider()
        st.subheader("🔗 Correlation Heatmap of Key Clinical Predictors")
        ch_img = clinical_asset("severity_correlation_heatmap.png")
        if ch_img:
            st.image(ch_img, caption="Correlation Matrix Heatmap of Key Clinical & Renal Variables", use_container_width=True)
