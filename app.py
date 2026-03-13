import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

st.set_page_config(
    page_title="AI Smart Healthcare – Diabetes Risk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Futuristic dark glassmorphism
#  ALL ML CODE BELOW THIS BLOCK IS COMPLETELY UNTOUCHED
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Reset & page background ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: radial-gradient(ellipse at 18% 18%, #071d38 0%, #030D1A 55%),
                radial-gradient(ellipse at 82% 82%, #040f1e 0%, #030D1A 55%) !important;
    background-attachment: fixed !important;
}
[data-testid="block-container"] {
    padding: 1.6rem 2.8rem 3rem !important;
}

/* ── Base font ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    color: #E2EEFF !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #041020 0%, #071828 100%) !important;
    border-right: 1px solid rgba(0,200,255,0.10) !important;
}
[data-testid="stSidebar"] * { color: #C8E0F0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #8BB8D0 !important; }
/* Sidebar nav buttons */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    color: #A8CCE0 !important;
    font-size: 0.92rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.65rem 1rem !important;
    border-radius: 10px !important;
    width: 100% !important;
    transition: background 0.2s, color 0.2s !important;
}
[data-testid="stSidebar"] button:hover {
    background: rgba(0,200,255,0.10) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] button p {
    color: inherit !important;
    font-size: 0.92rem !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── All input labels (main area) ───────────────────────────────── */
.main label,
.main [data-testid="stWidgetLabel"] p,
.main [data-testid="stWidgetLabel"] span,
section[data-testid="stMain"] label,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
section[data-testid="stMain"] .stCheckbox p,
section[data-testid="stMain"] .stCheckbox span,
section[data-testid="stMain"] .stCheckbox label {
    color: #90BBD4 !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    font-family: 'Outfit', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Input fields ────────────────────────────────────────────────── */
.main input[type="number"],
.main input[type="text"],
section[data-testid="stMain"] input[type="number"],
section[data-testid="stMain"] input[type="text"],
[data-baseweb="input"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.22) !important;
    border-radius: 10px !important;
    color: #E2EEFF !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.94rem !important;
}
[data-baseweb="input"] input:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.14) !important;
}

/* ── Selectbox ───────────────────────────────────────────────────── */
.main [data-baseweb="select"] div,
section[data-testid="stMain"] [data-baseweb="select"] div {
    color: #E2EEFF !important;
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(0,200,255,0.22) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span { color: #E2EEFF !important; }
[data-baseweb="popover"] li {
    color: #C8E0F0 !important;
    background: #071828 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-baseweb="popover"] li:hover {
    background: rgba(0,200,255,0.12) !important;
    color: #FFFFFF !important;
}

/* ── Number input stepper buttons ───────────────────────────────── */
.main [data-baseweb="input"] button,
section[data-testid="stMain"] [data-baseweb="input"] button {
    color: #90BBD4 !important;
    background: rgba(0,200,255,0.06) !important;
}

/* ── Checkboxes ──────────────────────────────────────────────────── */
section[data-testid="stMain"] .stCheckbox p,
section[data-testid="stMain"] .stCheckbox label p,
section[data-testid="stMain"] div[data-testid="stCheckbox"] p,
section[data-testid="stMain"] div[data-testid="stCheckbox"] span {
    color: #90BBD4 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Markdown text ───────────────────────────────────────────────── */
section[data-testid="stMain"] .stMarkdown p,
section[data-testid="stMain"] .stMarkdown span,
.main .stMarkdown p { color: #A8C8E0 !important; }
.main .stMarkdown strong,
section[data-testid="stMain"] .stMarkdown strong { color: #E2EEFF !important; }

/* ── Primary predict button ──────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0090B8, #00D4FF) !important;
    color: #030D1A !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.8rem !important;
    box-shadow: 0 4px 24px rgba(0,212,255,0.32) !important;
    transition: all 0.22s ease !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,255,0.46) !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
.streamlit-expanderHeader p,
section[data-testid="stMain"] details summary p {
    color: #90BBD4 !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
}
.streamlit-expanderContent {
    background: rgba(6,24,44,0.6) !important;
    border: 1px solid rgba(0,200,255,0.12) !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Dataframe ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    color: #C8E0F0 !important;
    background: rgba(6,24,44,0.7) !important;
    border-color: rgba(0,200,255,0.10) !important;
}
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Spinner ─────────────────────────────────────────────────────── */
.stSpinner p { color: #00D4FF !important; }
[data-testid="stSpinner"] > div { border-top-color: #00D4FF !important; }

/* ── Alerts ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(0,200,255,0.06) !important;
    border: 1px solid rgba(0,200,255,0.22) !important;
    border-radius: 10px !important;
    color: #C8E0F0 !important;
}

/* ── HR divider ──────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.22), transparent) !important;
    margin: 1.4rem 0 !important;
}

/* ═══════════════ CUSTOM COMPONENTS ═══════════════════════════════ */

/* ── Hero banner ─────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(120deg,
        rgba(0,144,184,0.18) 0%,
        rgba(0,212,255,0.08) 50%,
        rgba(26,80,200,0.12) 100%);
    border: 1px solid rgba(0,212,255,0.20);
    border-radius: 18px;
    padding: 1.8rem 2.4rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(0,0,0,0.3);
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00D4FF, transparent);
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,212,255,0.07), transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,245,212,0.10);
    border: 1px solid rgba(0,245,212,0.28);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 10px;
    font-weight: 700;
    color: #00F5D4;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.55rem;
    font-weight: 900;
    color: #E8F4FF;
    letter-spacing: -0.02em;
    margin: 0 0 6px;
}
.hero-title span {
    background: linear-gradient(90deg, #00D4FF, #00F5D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 0.87rem;
    color: #7AAAC4;
    margin: 0;
    line-height: 1.7;
}

/* ── Section heading ─────────────────────────────────────────────── */
.stitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: #E2EEFF;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}
.ssub {
    font-size: 0.82rem;
    color: #4E7A94;
    margin-bottom: 1.2rem;
}
.sec-label {
    font-size: 10px;
    font-weight: 700;
    color: #00D4FF;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.28), transparent);
}

/* ── Accuracy cards ──────────────────────────────────────────────── */
.card {
    background: rgba(6,24,44,0.72);
    border: 1px solid rgba(0,212,255,0.14);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.card .accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 3px 3px 0 0;
}
.card .clabel {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #4E7A94;
    margin-bottom: 0.4rem;
    font-family: 'Outfit', sans-serif;
}
.card .cvalue {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 800;
    color: #00D4FF;
    line-height: 1.1;
    text-shadow: 0 0 20px rgba(0,212,255,0.35);
}
.card .csub {
    font-size: 0.76rem;
    color: #4E7A94;
    margin-top: 0.2rem;
}
.card.best {
    border: 1px solid rgba(0,212,255,0.32);
    background: rgba(0,180,216,0.10);
}
.best-chip {
    position: absolute;
    top: 12px; right: 12px;
    background: linear-gradient(90deg, #0090B8, #00D4FF);
    color: #030D1A;
    font-size: 0.60rem;
    font-weight: 900;
    padding: 3px 9px;
    border-radius: 20px;
    letter-spacing: 0.07em;
}

/* ── Info stat cards (4-column row) ──────────────────────────────── */
.istat {
    background: rgba(6,24,44,0.72);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 2px 14px rgba(0,0,0,0.28);
    position: relative;
    overflow: hidden;
}
.istat .accent { position:absolute; top:0; left:0; right:0; height:3px; border-radius:3px 3px 0 0; }
.istat .clabel { font-size:0.70rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; color:#4E7A94; margin-bottom:0.3rem; }
.istat .cvalue { font-family:'Outfit',sans-serif; font-size:1.05rem; font-weight:800; color:#E2EEFF; }
.istat .csub   { font-size:0.74rem; color:#4E7A94; margin-top:0.15rem; }

/* ── Feature pills ───────────────────────────────────────────────── */
.fpill {
    background: rgba(6,24,44,0.72);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.fpill:hover { border-color: rgba(0,212,255,0.28); }
.fpill .ficon { font-size: 1.3rem; }
.fpill .fname { font-weight: 700; color: #C8E0F0; font-size: 0.86rem; margin-top: 0.2rem; }
.fpill .fdesc { font-size: 0.74rem; color: #4E7A94; margin-top: 1px; }

/* ── Form panel ──────────────────────────────────────────────────── */
.fpanel {
    background: rgba(6,24,44,0.72);
    border: 1px solid rgba(0,212,255,0.14);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.28);
}

/* ── BMI range pills ─────────────────────────────────────────────── */
.bmi-pill {
    font-size: 0.70rem;
    text-align: center;
    color: #4E7A94;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,200,255,0.10);
    border-radius: 8px;
    padding: 4px 6px;
    line-height: 1.5;
}
.bmi-pill strong { color: #7AAAC4; }

/* ── Clinical reference box ──────────────────────────────────────── */
.clin-ref {
    background: rgba(0,180,216,0.06);
    border: 1px solid rgba(0,212,255,0.14);
    border-radius: 12px;
    padding: 0.9rem 1rem;
}
.clin-ref .clin-label {
    font-size: 0.70rem;
    font-weight: 800;
    color: #00D4FF;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.5rem;
}
.clin-ref .clin-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;
}
.clin-ref .clin-item { font-size: 0.75rem; color: #7AAAC4; }

/* ── Prediction result cards ─────────────────────────────────────── */
.res-high {
    background: rgba(255,60,90,0.07);
    border: 1px solid rgba(255,60,90,0.28);
    border-left: 4px solid #FF3C5A;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
}
.res-low {
    background: rgba(0,245,160,0.06);
    border: 1px solid rgba(0,245,160,0.25);
    border-left: 4px solid #00F5A0;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
}
.res-icon { font-size: 2rem; margin-bottom: 0.25rem; }
.res-head {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    margin: 0.15rem 0 0.3rem;
}
.res-desc { font-size: 0.84rem; color: #7AAAC4; line-height: 1.6; }
.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.9rem; }
.chip {
    background: rgba(6,24,44,0.8);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.80rem;
    font-weight: 700;
    color: #C8E0F0;
    border: 1px solid rgba(0,200,255,0.18);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Model comparison table ──────────────────────────────────────── */
.ctbl {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    font-family: 'Outfit', sans-serif;
    border-radius: 12px;
}
.ctbl th {
    background: rgba(0,144,184,0.22);
    color: #00D4FF;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.75rem 1.2rem;
    border-bottom: 1px solid rgba(0,212,255,0.18);
}
.ctbl td {
    padding: 0.7rem 1.2rem;
    border-bottom: 1px solid rgba(0,200,255,0.08);
    font-size: 0.88rem;
    color: #A8C8E0;
    background: rgba(6,24,44,0.5);
}
.ctbl tr.brow td {
    background: rgba(0,144,184,0.14);
    font-weight: 700;
    color: #00D4FF;
}
.ctbl tr:last-child td { border-bottom: none; }

/* ── Insight cards (best model / gap) ────────────────────────────── */
.icard-b {
    background: rgba(0,144,184,0.10);
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.icard-b .il { font-size:0.68rem; font-weight:800; color:#00D4FF; text-transform:uppercase; letter-spacing:0.09em; }
.icard-b .iv { font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:800; color:#00D4FF; }
.icard-b .is { font-size:0.79rem; color:#4E7A94; }
.icard-g {
    background: rgba(0,245,160,0.06);
    border: 1px solid rgba(0,245,160,0.22);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.icard-g .il { font-size:0.68rem; font-weight:800; color:#00F5A0; text-transform:uppercase; letter-spacing:0.09em; }
.icard-g .iv { font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:800; color:#00F5A0; }
.icard-g .is { font-size:0.79rem; color:#4E7A94; }

/* ── Model detail card ───────────────────────────────────────────── */
.dcard {
    background: rgba(6,24,44,0.72);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 14px;
    padding: 1.3rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.22s;
}
.dcard:hover { border-color: rgba(0,212,255,0.30); }
.dcard .dname { font-weight:800; color:#C8E0F0; font-size:0.92rem; margin:0.35rem 0 0.15rem; }
.dcard .dacc  { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:800; }
.dcard .ddesc { font-size:0.77rem; color:#4E7A94; line-height:1.55; margin-top:0.5rem; }
.dcard .dtag  { background:rgba(255,255,255,0.04); border-radius:6px; padding:2px 8px; font-size:0.70rem; color:#7AAAC4; border: 1px solid rgba(0,200,255,0.10); }

/* ── Sidebar brand ───────────────────────────────────────────────── */
.sb-brand {
    text-align: center;
    padding: 0.5rem 0 1.2rem;
}
.sb-icon { font-size: 2.4rem; filter: drop-shadow(0 0 8px rgba(0,212,255,0.45)); display:block; margin-bottom:4px; }
.sb-name { font-size: 1.05rem; font-weight: 900; color: #FFFFFF; letter-spacing: 0.1px; }
.sb-sub  { font-size: 9px; color: #00D4FF; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }

/* ── Sidebar mini accuracy bar ───────────────────────────────────── */
.mini-acc { display:flex; align-items:center; gap:8px; margin:6px 0; font-size:11px; }
.mini-track { flex:1; background:rgba(255,255,255,0.06); border-radius:99px; height:5px; overflow:hidden; }
.mini-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,#0090B8,#00D4FF); box-shadow:0 0 6px rgba(0,212,255,0.4); }
.mini-val   { width:32px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:10px; color:#00D4FF; }

/* ── Animated live dot ───────────────────────────────────────────── */
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,245,180,0.6); }
    50%       { box-shadow: 0 0 0 5px rgba(0,245,180,0); }
}
.live-dot {
    display: inline-block; width:8px; height:8px;
    background: #00F5B4; border-radius: 50%;
    animation: pulse-dot 2s ease infinite;
    vertical-align: middle; margin-right: 5px;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 1.2rem 0 0.4rem;
    font-size: 0.78rem;
    color: #2A4E68;
    border-top: 1px solid rgba(0,200,255,0.08);
    margin-top: 2.5rem;
}
.footer b { color: #00D4FF; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▼▼▼  ORIGINAL ML CODE — COMPLETELY UNTOUCHED FROM HERE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════

# ── Load models ────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    files = {
        "Logistic Regression": "diabetes-prediction-model.pkl",
        "SVM":                 "svm-model.pkl",
        "Random Forest":       "random-forest-model.pkl",
    }
    loaded = {}
    for name, fname in files.items():
        try:
            loaded[name] = joblib.load(os.path.join(MODEL_DIR, fname))
        except Exception:
            pass
    try:
        sc = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    except Exception:
        sc = None
    return loaded, sc

models, scaler = load_models()

ACCURACIES = {"Logistic Regression": 89.12, "SVM": 89.41, "Random Forest": 90.82}
BEST_MODEL  = "Random Forest"
FEATURE_COLS = [
    "gender","age","hypertension","heart_disease","bmi","HbA1c_level",
    "blood_glucose_level","smoking_history_No Info","smoking_history_current",
    "smoking_history_ever","smoking_history_former","smoking_history_never",
    "smoking_history_not current",
]
SMOKING_OPTIONS = ["Prefer not to say","never","former","current","ever","not current"]

def build_features(gender, age, hypertension, heart_disease, bmi, hba1c, glucose, smoking):
    row = {c: 0 for c in FEATURE_COLS}
    row["gender"]              = 1 if gender == "Male" else 0
    row["age"]                 = age
    row["hypertension"]        = int(hypertension)
    row["heart_disease"]       = int(heart_disease)
    row["bmi"]                 = bmi
    row["HbA1c_level"]         = hba1c
    row["blood_glucose_level"] = glucose
    key = f"smoking_history_{smoking}"
    if key in row:
        row[key] = 1
    return pd.DataFrame([row], columns=FEATURE_COLS)

# ══════════════════════════════════════════════════════════════════════════════
#  ▲▲▲  END OF ORIGINAL ML CODE  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-icon">🩺</span>
        <div class="sb-name">AI Healthcare</div>
        <div class="sb-sub">Diabetes System</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border:none;height:1px;background:rgba(0,200,255,0.12);margin:0 0 0.8rem;">', unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    for label, icon in [("Dashboard","📊"),("Prediction","🔬"),("Model Comparison","📈")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label

    st.markdown('<hr style="border:none;height:1px;background:rgba(0,200,255,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:9.5px;font-weight:800;text-transform:uppercase;letter-spacing:1.4px;color:#00D4FF;padding:0 0.3rem;margin-bottom:10px;">Model Status</div>', unsafe_allow_html=True)
    for m, acc in ACCURACIES.items():
        ok  = "✅" if m in models else "❌"
        pct = int(acc)
        st.markdown(f"""
        <div style="margin-bottom:2px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:11px;color:#7AAAC4;">{ok} {m}</span>
            </div>
            <div class="mini-acc">
                <div class="mini-track"><div class="mini-fill" style="width:{pct}%;"></div></div>
                <span class="mini-val">{acc:.0f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border:none;height:1px;background:rgba(0,200,255,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px;color:#2A4E68;line-height:1.9;">
        <span class="live-dot"></span>
        <span style="color:#00F5B4;font-weight:700;">System Online</span><br><br>
        ⚠️ For clinical decision<br>support only. Always consult<br>a licensed physician.
    </div>
    <div style="font-size:0.70rem;color:#1A3A50;margin-top:0.9rem;">v1.0.0 &middot; 2025</div>
    """, unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⬡ AI Platform v1.0</div>
        <div class="hero-title">🩺 AI Smart Healthcare <span>System</span></div>
        <div class="hero-sub">Diabetes Risk Prediction &nbsp;·&nbsp; Three ML Models &nbsp;·&nbsp; Clinical Decision Support</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Model Performance Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="ssub">Trained on real clinical data &nbsp;·&nbsp; Validated on held-out test sets</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="card"><div class="accent" style="background:linear-gradient(90deg,#6366F1,#818CF8);"></div>'
                    '<div class="clabel">🔢 Logistic Regression</div>'
                    '<div class="cvalue">89.12%</div>'
                    '<div class="csub">Test Set Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="accent" style="background:linear-gradient(90deg,#0EA5E9,#38BDF8);"></div>'
                    '<div class="clabel">📐 SVM</div>'
                    '<div class="cvalue">89.41%</div>'
                    '<div class="csub">Test Set Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card best"><div class="accent" style="background:linear-gradient(90deg,#00D4FF,#00F5D4);"></div>'
                    '<div class="best-chip">⭐ BEST</div>'
                    '<div class="clabel">🌲 Random Forest</div>'
                    '<div class="cvalue">90.82%</div>'
                    '<div class="csub">Test Set Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">System Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="ssub">Dataset and infrastructure at a glance</div>', unsafe_allow_html=True)

    _bg = "background:linear-gradient(90deg,#0090B8,#00D4FF);"
    i1, i2, i3, i4 = st.columns(4, gap="medium")
    for col, icon_txt, lbl, val, sub in [
        (i1,"🗂️","Dataset","Diabetes Prediction","Clinical Health Records"),
        (i2,"🎯","Best Accuracy","90.82%","Random Forest"),
        (i3,"🤖","Models Trained","3 Algorithms","LR &middot; SVM &middot; RF"),
        (i4,"⚕️","Target","Diabetes Status","Binary Classification"),
    ]:
        with col:
            st.markdown(
                '<div class="istat"><div class="accent" style="' + _bg + '"></div>'
                '<div class="clabel">' + icon_txt + ' ' + lbl + '</div>'
                '<div class="cvalue">' + val + '</div>'
                '<div class="csub">' + sub + '</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Input Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="ssub">Health parameters used for prediction</div>', unsafe_allow_html=True)

    feats = [
        ("👤","Gender","Biological sex: Male / Female"),
        ("📅","Age","Patient age in years"),
        ("💉","Hypertension","High blood pressure status"),
        ("❤️","Heart Disease","Cardiovascular condition"),
        ("⚖️","BMI","Body Mass Index (kg/m²)"),
        ("🩸","HbA1c Level","3-month average blood sugar"),
        ("🔬","Blood Glucose","Current blood glucose (mg/dL)"),
        ("🚬","Smoking History","Past or current smoking status"),
    ]
    f_cols = st.columns(4, gap="small")
    for i, (icon, name, desc) in enumerate(feats):
        with f_cols[i % 4]:
            st.markdown(
                '<div class="fpill"><div class="ficon">' + icon + '</div>'
                '<div class="fname">' + name + '</div>'
                '<div class="fdesc">' + desc + '</div></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════
elif page == "Prediction":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⬡ Risk Assessment</div>
        <div class="hero-title">🔬 Diabetes Risk <span>Prediction</span></div>
        <div class="hero-sub">Enter patient health data and select a model to assess diabetes risk</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.15], gap="large")

    with col_l:
        st.markdown('<div class="fpanel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Select Model</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Model", list(ACCURACIES.keys()), index=2, label_visibility="collapsed")
        badge_txt = "&nbsp; ⭐ Best Model" if model_choice == BEST_MODEL else ""
        st.markdown(
            '<div style="background:rgba(0,144,184,0.10);border:1px solid rgba(0,212,255,0.18);'
            'border-radius:9px;padding:0.5rem 0.9rem;font-size:0.84rem;color:#00D4FF;font-weight:700;margin-top:0.3rem;">'
            '📊 Accuracy: <strong style="font-family:JetBrains Mono,monospace;">' + str(ACCURACIES[model_choice]) + '%</strong>'
            + badge_txt + '</div>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Demographics</div>', unsafe_allow_html=True)
        gender  = st.selectbox("Gender", ["Male", "Female"])
        age     = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        smoking = st.selectbox("Smoking History", SMOKING_OPTIONS)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Medical Conditions</div>', unsafe_allow_html=True)
        hypertension  = st.checkbox("🫀 Hypertension (High Blood Pressure)")
        heart_disease = st.checkbox("❤️ Heart Disease")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="fpanel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Clinical Measurements</div>', unsafe_allow_html=True)
        bmi = st.number_input("BMI – Body Mass Index (kg/m²)", min_value=10.0, max_value=80.0, value=27.5, step=0.1)

        b1, b2, b3 = st.columns(3)
        for bc, lbl, rng in [(b1,"Underweight","< 18.5"),(b2,"Normal","18.5 – 25"),(b3,"Obese","≥ 30")]:
            with bc:
                st.markdown(f'<div class="bmi-pill">{lbl}<br><strong>{rng}</strong></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        hba1c   = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=6.5, step=0.1)
        glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="clin-ref">'
            '<div class="clin-label">📋 Clinical Reference</div>'
            '<div class="clin-grid">'
            '<div class="clin-item">🟢 HbA1c &lt; 5.7% → Normal</div>'
            '<div class="clin-item">🟡 HbA1c 5.7–6.4% → Pre-DM</div>'
            '<div class="clin-item">🟢 Glucose &lt; 100 → Normal</div>'
            '<div class="clin-item">🔴 Glucose 126+ → Diabetic</div>'
            '</div></div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🩺  Predict Diabetes Risk", type="primary", use_container_width=True)

    # ── PREDICTION LOGIC — COMPLETELY UNTOUCHED ──────────────────────────────
    if predict_btn:
        if model_choice not in models:
            st.error(f"Model file for **{model_choice}** not found. Please ensure the .pkl files are in the same folder as app.py")
        else:
            with st.spinner("Analysing health data…"):
                df_input = build_features(gender, age, hypertension, heart_disease, bmi, hba1c, glucose, smoking)
                model = models[model_choice]
                if model_choice in ("Logistic Regression", "SVM") and scaler is not None:
                    X     = scaler.transform(df_input)
                    pred  = model.predict(X)[0]
                    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
                else:
                    pred  = model.predict(df_input)[0]
                    proba = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else None

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Prediction Result</div>', unsafe_allow_html=True)

            p_diab = f"{proba[1]*100:.1f}%" if proba is not None else "N/A"
            p_safe = f"{proba[0]*100:.1f}%" if proba is not None else "N/A"

            # Numeric values for probability bars (default 0 if N/A)
            pv_diab = proba[1]*100 if proba is not None else 0
            pv_safe = proba[0]*100 if proba is not None else 0

            if pred == 1:
                st.markdown(
                    '<div class="res-high">'
                    '<div class="res-icon">⚠️</div>'
                    '<div class="res-head" style="color:#FF5070;">High Risk of Diabetes</div>'
                    '<div class="res-desc">Please consult a healthcare professional. Early intervention significantly improves outcomes.</div>'
                    '<div class="chip-row">'
                    '<div class="chip" style="border-color:rgba(255,60,90,0.35);color:#FF7090;">🔴 Diabetic: ' + p_diab + '</div>'
                    '<div class="chip" style="border-color:rgba(0,245,160,0.28);color:#00F5A0;">🟢 Non-Diabetic: ' + p_safe + '</div>'
                    '<div class="chip" style="border-color:rgba(0,200,255,0.20);color:#00D4FF;">🤖 ' + model_choice + '</div>'
                    '</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="res-low">'
                    '<div class="res-icon">✅</div>'
                    '<div class="res-head" style="color:#00F5A0;">Low Risk of Diabetes</div>'
                    '<div class="res-desc">Maintain a healthy lifestyle with balanced diet, regular exercise, and routine check-ups.</div>'
                    '<div class="chip-row">'
                    '<div class="chip" style="border-color:rgba(0,245,160,0.28);color:#00F5A0;">🟢 Non-Diabetic: ' + p_safe + '</div>'
                    '<div class="chip" style="border-color:rgba(255,60,90,0.35);color:#FF7090;">🔴 Diabetic Risk: ' + p_diab + '</div>'
                    '<div class="chip" style="border-color:rgba(0,200,255,0.20);color:#00D4FF;">🤖 ' + model_choice + '</div>'
                    '</div></div>', unsafe_allow_html=True)

            # Probability bars — separate st.markdown call to guarantee rendering
            if proba is not None:
                st.markdown(f"""
                <div style="display:flex;gap:14px;margin:16px 0 4px;">
                    <div style="flex:1;border-radius:12px;padding:14px 16px;text-align:center;
                                background:rgba(255,60,90,0.07);border:1px solid rgba(255,60,90,0.26);">
                        <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.4px;
                                    color:#FF7090;margin-bottom:7px;">🔴 Diabetic Risk</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:30px;font-weight:800;
                                    color:#FF4664;line-height:1;margin-bottom:9px;">{pv_diab:.1f}%</div>
                        <div style="height:5px;border-radius:99px;background:rgba(255,255,255,0.07);overflow:hidden;">
                            <div style="width:{pv_diab:.1f}%;height:100%;border-radius:99px;
                                        background:linear-gradient(90deg,#FF4664,#FF80A0);
                                        box-shadow:0 0 8px rgba(255,70,100,0.45);"></div>
                        </div>
                    </div>
                    <div style="flex:1;border-radius:12px;padding:14px 16px;text-align:center;
                                background:rgba(0,245,160,0.05);border:1px solid rgba(0,245,160,0.22);">
                        <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.4px;
                                    color:#00C890;margin-bottom:7px;">🟢 Non-Diabetic</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:30px;font-weight:800;
                                    color:#00F5A0;line-height:1;margin-bottom:9px;">{pv_safe:.1f}%</div>
                        <div style="height:5px;border-radius:99px;background:rgba(255,255,255,0.07);overflow:hidden;">
                            <div style="width:{pv_safe:.1f}%;height:100%;border-radius:99px;
                                        background:linear-gradient(90deg,#00B47A,#00F5A0);
                                        box-shadow:0 0 8px rgba(0,245,160,0.38);"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📄 View Input Summary"):
                summary = pd.DataFrame({
                    "Feature": ["Gender","Age","Hypertension","Heart Disease","BMI","HbA1c Level","Blood Glucose","Smoking History"],
                    "Value":   [gender, age,
                                "Yes" if hypertension else "No",
                                "Yes" if heart_disease else "No",
                                f"{bmi:.1f} kg/m²", f"{hba1c:.1f}%",
                                f"{glucose} mg/dL", smoking],
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════
#  MODEL COMPARISON
# ══════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⬡ Benchmarks</div>
        <div class="hero-title">📈 Model <span>Comparison</span></div>
        <div class="hero-sub">Accuracy benchmarks across all three trained ML algorithms</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="sec-label">Accuracy Bar Chart</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("#071828")
        ax.set_facecolor("#071828")
        names  = list(ACCURACIES.keys())
        values = list(ACCURACIES.values())
        clrs   = ["#6366F1","#0EA5E9","#00D4FF"]
        bars   = ax.barh(names, values, color=clrs, height=0.46, edgecolor="none")
        # Subtle glow behind bars
        ax.barh(names, values, height=0.56, color=clrs, alpha=0.12, edgecolor="none")
        for bar, val, name in zip(bars, values, names):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}%", va="center", ha="left",
                    fontsize=11, fontweight="700", color="#C8E0F0",
                    fontfamily="monospace")
            if name == BEST_MODEL:
                ax.text(bar.get_width() - 0.35, bar.get_y() + bar.get_height()/2,
                        "⭐", va="center", ha="right", fontsize=10)
        ax.set_xlim(85, 93.5)
        ax.set_xlabel("Accuracy (%)", fontsize=10, color="#4E7A94")
        ax.tick_params(axis="y", labelsize=10.5, labelcolor="#90BBD4")
        ax.tick_params(axis="x", labelsize=9,    labelcolor="#2A4E68")
        for sp in ["top","right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#0D2A40")
        ax.spines["bottom"].set_color("#0D2A40")
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color="#0D2A40", linewidth=0.8)
        ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight="700",
                     color="#C8E0F0", pad=12)
        patch = mpatches.Patch(color="#00D4FF", label=f"Best: {BEST_MODEL}")
        ax.legend(handles=[patch], fontsize=9, framealpha=0,
                  labelcolor="#00D4FF", loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with right:
        st.markdown('<div class="sec-label">Comparison Table</div>', unsafe_allow_html=True)
        rows = ""
        for mname, acc in ACCURACIES.items():
            rc    = "brow" if mname == BEST_MODEL else ""
            badge = ('&nbsp;<span style="background:rgba(0,212,255,0.15);color:#00D4FF;'
                     'font-size:0.64rem;font-weight:800;padding:2px 8px;border-radius:20px;">⭐ BEST</span>'
                     if mname == BEST_MODEL else "")
            rows += f'<tr class="{rc}"><td>{mname}{badge}</td><td><strong>{acc:.2f}%</strong></td></tr>'
        st.markdown(
            '<table class="ctbl"><thead><tr><th>Model</th><th>Accuracy</th></tr></thead>'
            '<tbody>' + rows + '</tbody></table>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        worst = min(ACCURACIES, key=ACCURACIES.get)
        diff  = ACCURACIES[BEST_MODEL] - ACCURACIES[worst]
        st.markdown(
            '<div class="icard-b"><div class="il">🏆 Best Model</div>'
            '<div class="iv">' + BEST_MODEL + '</div>'
            '<div class="is">Accuracy: ' + f"{ACCURACIES[BEST_MODEL]:.2f}%" + '</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="icard-g"><div class="il">📈 Performance Gap</div>'
            '<div class="iv">+' + f"{diff:.2f}%" + '</div>'
            '<div class="is">' + BEST_MODEL + ' vs ' + worst + '</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Model Characteristics</div>', unsafe_allow_html=True)

    details = [
        ("🔢","Logistic Regression","89.12%","#6366F1",False,
         "Fast, interpretable linear classifier. Ideal for baseline benchmarks. Uses StandardScaler preprocessing.","⚡⚡⚡","Low"),
        ("📐","SVM","89.41%","#0EA5E9",False,
         "Radial Basis Function kernel SVM. Excellent generalization on non-linear decision boundaries.","⚡⚡","Medium"),
        ("🌲","Random Forest","90.82%","#00D4FF",True,
         "Ensemble of 100 decision trees with balanced class weights. Highest accuracy — recommended for deployment.","⚡","High"),
    ]
    d1, d2, d3 = st.columns(3, gap="medium")
    for col, (icon, name, acc, color, is_best, desc, speed, cplx) in zip([d1,d2,d3], details):
        border = f"1px solid {color}40" if is_best else "1px solid rgba(0,200,255,0.10)"
        with col:
            st.markdown(
                '<div class="dcard" style="border:' + border + ';">'
                '<div style="font-size:1.6rem;">' + icon + '</div>'
                '<div class="dname">' + name + '</div>'
                '<div class="dacc" style="color:' + color + ';text-shadow:0 0 12px ' + color + '55;">' + acc + '</div>'
                '<div class="ddesc">' + desc + '</div>'
                '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.8rem;">'
                '<span class="dtag">Speed: ' + speed + '</span>'
                '<span class="dtag">Complexity: ' + cplx + '</span>'
                '</div></div>',
                unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">AI Smart Healthcare System &nbsp;|&nbsp; Built with <b>Streamlit</b>'
    ' &nbsp;·&nbsp; Powered by <b>scikit-learn</b> &nbsp;·&nbsp; 🩺 For clinical decision support only</div>',
    unsafe_allow_html=True)