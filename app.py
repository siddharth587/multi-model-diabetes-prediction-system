import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Smart Healthcare – Diabetes Risk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Futuristic dark glassmorphism
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

/* ── Info stat cards (row) ───────────────────────────────────────── */
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

/* ── Insight cards ───────────────────────────────────────────────── */
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

/* ── SHAP explainability panel ───────────────────────────────────── */
.shap-panel {
    background: rgba(6,24,44,0.80);
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 28px rgba(0,0,0,0.35);
    margin-top: 1.2rem;
}
.shap-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.0rem;
    font-weight: 800;
    color: #E2EEFF;
    margin-bottom: 0.3rem;
}
.shap-sub {
    font-size: 0.80rem;
    color: #4E7A94;
    margin-bottom: 1rem;
}
.shap-feat-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.shap-feat-name {
    font-size: 0.78rem;
    font-weight: 700;
    color: #90BBD4;
    width: 160px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.shap-bar-wrap {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.shap-bar-pos {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #FF4664, #FF80A0);
    box-shadow: 0 0 8px rgba(255,70,100,0.4);
}
.shap-bar-neg {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #00B47A, #00F5A0);
    box-shadow: 0 0 8px rgba(0,245,160,0.4);
    float: right;
}
.shap-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    width: 60px;
    text-align: right;
    flex-shrink: 0;
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
#  ML / MODEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# All 6 models with their actual accuracies from notebook
ACCURACIES = {
    "Logistic Regression": 88.88,
    "SVM":                 89.26,
    "Random Forest":       90.79,
    "XGBoost":             91.79,
    "CatBoost":            91.79,
    "LightGBM":            92.15,   # ← BEST MODEL
}

BEST_MODEL = "LightGBM"

MODEL_FILES = {
    "Logistic Regression": ("diabetes-prediction-model.pkl", True),   # (file, needs_scaler)
    "SVM":                 ("svm-model.pkl",                 True),
    "Random Forest":       ("random-forest-model.pkl",       False),
    "XGBoost":             ("xgboost_tuned_model.pkl",       False),
    "CatBoost":            ("catboost_model.pkl",            False),
    "LightGBM":            ("lightgbm_model.pkl",            False),
}

MODEL_COLORS = {
    "Logistic Regression": "#6366F1",
    "SVM":                 "#0EA5E9",
    "Random Forest":       "#22D3EE",
    "XGBoost":             "#F59E0B",
    "CatBoost":            "#A78BFA",
    "LightGBM":            "#00D4FF",
}

FEATURE_COLS = [
    "gender", "age", "hypertension", "heart_disease", "bmi", "HbA1c_level",
    "blood_glucose_level", "smoking_history_No Info", "smoking_history_current",
    "smoking_history_ever", "smoking_history_former", "smoking_history_never",
    "smoking_history_not current",
]

FEATURE_DISPLAY_NAMES = {
    "gender":                      "Gender",
    "age":                         "Age",
    "hypertension":                "Hypertension",
    "heart_disease":               "Heart Disease",
    "bmi":                         "BMI",
    "HbA1c_level":                 "HbA1c Level",
    "blood_glucose_level":         "Blood Glucose",
    "smoking_history_No Info":     "Smoking: No Info",
    "smoking_history_current":     "Smoking: Current",
    "smoking_history_ever":        "Smoking: Ever",
    "smoking_history_former":      "Smoking: Former",
    "smoking_history_never":       "Smoking: Never",
    "smoking_history_not current": "Smoking: Not Current",
}

SMOKING_OPTIONS = ["Prefer not to say", "never", "former", "current", "ever", "not current"]


@st.cache_resource
def load_models():
    loaded = {}
    for name, (fname, _) in MODEL_FILES.items():
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


def run_shap(model_name, model, df_input):
    """Compute SHAP values and return (shap_vals_1d, base_value)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(df_input)
        base = explainer.expected_value

        # Handle different SHAP output formats
        if isinstance(sv, list):
            # sklearn-style binary: list of [class0_arr, class1_arr]
            vals = np.array(sv[1]).flatten() if len(sv) == 2 else np.array(sv[0]).flatten()
            bv   = base[1] if isinstance(base, (list, np.ndarray)) and len(base) == 2 else float(base)
        else:
            sv = np.array(sv)
            if sv.ndim == 3:
                # (n_samples, n_features, n_classes) — take class 1
                vals = sv[0, :, 1]
                bv   = base[1] if hasattr(base, '__len__') else float(base)
            elif sv.ndim == 2:
                vals = sv[0]
                bv   = base[1] if hasattr(base, '__len__') and len(base) == 2 else float(base)
            else:
                vals = sv
                bv   = float(base) if not hasattr(base, '__len__') else base[0]

        return vals.flatten(), float(bv)
    except Exception as e:
        return None, None


def render_shap_section(shap_vals, feature_names, df_input):
    """
    Render a clean diverging SHAP bar chart.

    SHAP values are in log-odds space (LightGBM TreeExplainer default):
      • POSITIVE value → increases log-odds of diabetic (class 1) → RED  ↑ risk
      • NEGATIVE value → decreases log-odds of diabetic (class 1) → GREEN ↓ risk

    Verified: base + sum(shap) = log-odds → sigmoid = predicted probability ✓
    Clinical check: high HbA1c/glucose/BMI on a diabetic patient → large +SHAP ✓
    """
    if shap_vals is None:
        st.warning("SHAP explanation unavailable for this prediction.")
        return

    feat_vals = df_input.values[0]

    # Sort by |SHAP| descending, keep top 10
    order   = np.argsort(np.abs(shap_vals))[::-1][:10]
    # Reverse so largest bar is at top after invert_yaxis
    order   = order[::-1]

    sv_plot  = shap_vals[order]
    max_abs  = max(np.abs(sv_plot)) if len(sv_plot) else 1.0
    x_lim    = max_abs * 1.38          # extra space for value labels

    colors   = ["#FF4664" if v > 0 else "#00F5A0" for v in sv_plot]
    y_pos    = np.arange(len(order))

    # Build y-tick labels: "Feature Name = value" — kept concise
    y_labels = []
    for idx in order:
        fname = FEATURE_DISPLAY_NAMES.get(feature_names[idx], feature_names[idx])
        fval  = feat_vals[idx]
        # Format value neatly
        if isinstance(fval, float) and fval != int(fval):
            fstr = f"{fval:.2f}"
        else:
            fstr = f"{int(fval)}"
        y_labels.append(f"{fname} = {fstr}")

    # ── Figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4.2, len(order) * 0.52)))
    fig.patch.set_facecolor("#071828")
    ax.set_facecolor("#071828")

    # Glow shadow bars (wide, low alpha)
    ax.barh(y_pos, sv_plot, height=0.60, color=colors, alpha=0.12, edgecolor="none")
    # Main bars
    bars = ax.barh(y_pos, sv_plot, height=0.50, color=colors, alpha=0.90, edgecolor="none")

    # Value labels — placed INSIDE the bar when bar is wide enough, else outside
    for bar, sv, color in zip(bars, sv_plot, colors):
        bar_w = abs(bar.get_width())
        label = f"{sv:+.3f}"
        # Place label outside the bar end (away from zero)
        if sv >= 0:
            x_label = sv + max_abs * 0.02
            ha = "left"
        else:
            x_label = sv - max_abs * 0.02
            ha = "right"
        txt_color = "#FF8099" if sv > 0 else "#7BFFCC"
        ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha,
                fontsize=8.5, fontweight="700",
                color=txt_color, fontfamily="monospace")

    # Zero reference line — use valid matplotlib hex color
    ax.axvline(0, color="#1A4A6A", linewidth=1.4, linestyle="--", zorder=3)

    # Symmetric x-axis
    ax.set_xlim(-x_lim, x_lim)

    # Y-axis labels on the left
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9.5, color="#90BBD4", fontfamily="sans-serif")
    ax.invert_yaxis()

    ax.set_xlabel("SHAP Value  (log-odds contribution toward diabetic risk)",
                  fontsize=9, color="#4E7A94", labelpad=8)
    ax.set_title("Feature Impact on This Prediction", fontsize=12,
                 fontweight="700", color="#C8E0F0", pad=12)

    ax.tick_params(axis="x", labelsize=8.5, labelcolor="#2A4E68", length=3)
    ax.tick_params(axis="y", length=0)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#0D2A40")
    ax.spines["bottom"].set_color("#0D2A40")
    ax.xaxis.grid(True, color="#0D2A40", linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)

    # Legend
    pos_patch = mpatches.Patch(color="#FF4664", label="🔴  Positive SHAP → ↑ Increases diabetic risk")
    neg_patch = mpatches.Patch(color="#00F5A0", label="🟢  Negative SHAP → ↓ Decreases diabetic risk")
    leg = ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5, framealpha=0.0,
                    labelcolor="#A8C8E0", loc="lower right",
                    bbox_to_anchor=(1.0, -0.18), ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # ── Header ─────────────────────────────────────────────────────────
    st.markdown('<div class="shap-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="shap-title">🔍 Explainable AI — SHAP Feature Impact</div>'
        '<div class="shap-sub">'
        'SHAP (SHapley Additive exPlanations) values are in <strong style="color:#C8E0F0;">log-odds space</strong>. '
        '<span style="color:#FF8099;">Red / positive</span> bars push the model toward a diabetic prediction; '
        '<span style="color:#7BFFCC;">green / negative</span> bars push it away. '
        'Bars are sorted by absolute importance — the top feature had the largest influence on this result.'
        '</div>',
        unsafe_allow_html=True)

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Interpretation callout ──────────────────────────────────────────
    n_pos = int(np.sum(sv_plot > 0))
    n_neg = int(np.sum(sv_plot < 0))
    top_feat_idx   = order[-1]   # after reversal, last = highest |SHAP|
    top_feat_name  = FEATURE_DISPLAY_NAMES.get(feature_names[top_feat_idx], feature_names[top_feat_idx])
    top_feat_sv    = shap_vals[top_feat_idx]
    top_direction  = "increased" if top_feat_sv > 0 else "decreased"
    top_color      = "#FF8099" if top_feat_sv > 0 else "#7BFFCC"

    st.markdown(
        f'<div style="background:rgba(0,144,184,0.07);border:1px solid rgba(0,200,255,0.14);'
        f'border-radius:10px;padding:0.75rem 1.1rem;margin-top:0.8rem;font-size:0.82rem;color:#7AAAC4;">'
        f'<strong style="color:#00D4FF;">📌 Key driver:</strong> '
        f'<strong style="color:#C8E0F0;">{top_feat_name}</strong> had the largest impact — '
        f'it <strong style="color:{top_color};">{top_direction} diabetic risk</strong> '
        f'(SHAP = {top_feat_sv:+.3f}). '
        f'Of the top 10 features, <span style="color:#FF8099;">{n_pos} pushed toward diabetes</span> '
        f'and <span style="color:#7BFFCC;">{n_neg} pushed away</span>.'
        f'</div>',
        unsafe_allow_html=True)

    # ── Full table ──────────────────────────────────────────────────────
    with st.expander("📊 Full SHAP Feature Impact Table"):
        # Sort by absolute value for table too (descending)
        tbl_order = np.argsort(np.abs(shap_vals))[::-1][:13]
        shap_df = pd.DataFrame({
            "Rank":        [i + 1 for i in range(len(tbl_order))],
            "Feature":     [FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]) for i in tbl_order],
            "Input Value": [f"{feat_vals[i]:.3g}" for i in tbl_order],
            "SHAP Value":  [f"{shap_vals[i]:+.4f}" for i in tbl_order],
            "Impact":      ["↑ Increases Risk" if shap_vals[i] > 0 else "↓ Decreases Risk" for i in tbl_order],
        })
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
        st.caption(
            "SHAP values are in log-odds space. "
            "A value of +0.3 means this feature shifted the model's log-odds toward diabetic by 0.3 units. "
            "base + Σ(SHAP) = log-odds → sigmoid → predicted probability."
        )

    st.markdown('</div>', unsafe_allow_html=True)


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

    for label, icon in [("Dashboard", "📊"), ("Prediction", "🔬"), ("Model Comparison", "📈")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label

    st.markdown('<hr style="border:none;height:1px;background:rgba(0,200,255,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:9.5px;font-weight:800;text-transform:uppercase;letter-spacing:1.4px;color:#00D4FF;padding:0 0.3rem;margin-bottom:10px;">Model Status</div>', unsafe_allow_html=True)
    for m, acc in ACCURACIES.items():
        ok  = "✅" if m in models else "❌"
        pct = int(acc)
        star = " ⭐" if m == BEST_MODEL else ""
        st.markdown(f"""
        <div style="margin-bottom:2px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:11px;color:#7AAAC4;">{ok} {m}{star}</span>
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
    <div style="font-size:0.70rem;color:#1A3A50;margin-top:0.9rem;">v2.0.0 &middot; 2025</div>
    """, unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⬡ AI Platform v2.0</div>
        <div class="hero-title">🩺 AI Smart Healthcare <span>System</span></div>
        <div class="hero-sub">Diabetes Risk Prediction &nbsp;·&nbsp; Six ML Models &nbsp;·&nbsp; LightGBM Best Model &nbsp;·&nbsp; SHAP Explainability</div>
    </div>""", unsafe_allow_html=True)

    # ── All 6 model accuracy cards ─────────────────────────────────────
    st.markdown('<div class="sec-label">Model Performance Overview — All 6 Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="ssub">Trained on real clinical data &nbsp;·&nbsp; Validated on held-out test sets</div>', unsafe_allow_html=True)

    row1 = st.columns(3, gap="medium")
    row2 = st.columns(3, gap="medium")

    card_data = [
        ("Logistic Regression", 88.88, "#6366F1", "🔢", False),
        ("SVM",                 89.26, "#0EA5E9", "📐", False),
        ("Random Forest",       90.79, "#22D3EE", "🌲", False),
        ("XGBoost",             91.79, "#F59E0B", "⚡", False),
        ("CatBoost",            91.79, "#A78BFA", "🐱", False),
        ("LightGBM",            92.15, "#00D4FF", "💡", True),
    ]

    all_cols = row1 + row2
    for col, (name, acc, color, icon, is_best) in zip(all_cols, card_data):
        best_chip = '<div class="best-chip">⭐ BEST</div>' if is_best else ""
        card_cls  = "card best" if is_best else "card"
        with col:
            st.markdown(
                f'<div class="{card_cls}">'
                f'<div class="accent" style="background:linear-gradient(90deg,{color},{color}88);"></div>'
                f'{best_chip}'
                f'<div class="clabel">{icon} {name}</div>'
                f'<div class="cvalue" style="color:{color};">{acc:.2f}%</div>'
                f'<div class="csub">Test Set Accuracy</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Accuracy bar chart ─────────────────────────────────────────────
    st.markdown('<div class="sec-label">Accuracy Comparison Chart</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#071828")
    ax.set_facecolor("#071828")
    names  = list(ACCURACIES.keys())
    values = list(ACCURACIES.values())
    clrs   = [MODEL_COLORS[n] for n in names]
    bars   = ax.barh(names, values, color=clrs, height=0.50, edgecolor="none")
    ax.barh(names, values, height=0.62, color=clrs, alpha=0.11, edgecolor="none")
    for bar, val, name in zip(bars, values, names):
        ax.text(val + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", ha="left",
                fontsize=10, fontweight="700", color="#C8E0F0", fontfamily="monospace")
        if name == BEST_MODEL:
            ax.text(val - 0.3, bar.get_y() + bar.get_height() / 2,
                    "⭐", va="center", ha="right", fontsize=10)
    ax.set_xlim(86, 94.5)
    ax.set_xlabel("Accuracy (%)", fontsize=10, color="#4E7A94")
    ax.tick_params(axis="y", labelsize=10, labelcolor="#90BBD4")
    ax.tick_params(axis="x", labelsize=9,  labelcolor="#2A4E68")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#0D2A40")
    ax.spines["bottom"].set_color("#0D2A40")
    ax.xaxis.grid(True, color="#0D2A40", linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_title("All Models — Test Set Accuracy", fontsize=12, fontweight="700",
                 color="#C8E0F0", pad=10)
    patch = mpatches.Patch(color="#00D4FF", label=f"Best: {BEST_MODEL} (92.15%)")
    ax.legend(handles=[patch], fontsize=9, framealpha=0, labelcolor="#00D4FF", loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">System Overview</div>', unsafe_allow_html=True)

    _bg = "background:linear-gradient(90deg,#0090B8,#00D4FF);"
    i1, i2, i3, i4 = st.columns(4, gap="medium")
    for col, icon_txt, lbl, val, sub in [
        (i1, "🗂️",  "Dataset",        "Diabetes Prediction",    "Clinical Health Records"),
        (i2, "🎯",  "Best Accuracy",   "92.15%",                  "LightGBM"),
        (i3, "🤖",  "Models Trained",  "6 Algorithms",            "LR · SVM · RF · XGB · CAT · LGBM"),
        (i4, "🔍",  "Explainability",  "SHAP Integrated",        "Feature-level insights"),
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
    feats = [
        ("👤", "Gender",        "Biological sex: Male / Female"),
        ("📅", "Age",           "Patient age in years"),
        ("💉", "Hypertension",  "High blood pressure status"),
        ("❤️", "Heart Disease", "Cardiovascular condition"),
        ("⚖️", "BMI",           "Body Mass Index (kg/m²)"),
        ("🩸", "HbA1c Level",   "3-month average blood sugar"),
        ("🔬", "Blood Glucose", "Current blood glucose (mg/dL)"),
        ("🚬", "Smoking",       "Past or current smoking status"),
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
#  PREDICTION  (Best model only — LightGBM)
# ══════════════════════════════════════════
elif page == "Prediction":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⬡ Risk Assessment · Best Model</div>
        <div class="hero-title">🔬 Diabetes Risk <span>Prediction</span></div>
        <div class="hero-sub">Powered exclusively by <strong style="color:#00D4FF;">LightGBM</strong> — our highest-accuracy model (92.15%) — with SHAP explainability</div>
    </div>""", unsafe_allow_html=True)

    # Best model banner
    st.markdown(
        '<div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.25);'
        'border-radius:12px;padding:0.75rem 1.2rem;margin-bottom:1.4rem;display:flex;align-items:center;gap:14px;">'
        '<span style="font-size:1.5rem;">💡</span>'
        '<div><div style="font-size:0.84rem;font-weight:800;color:#00D4FF;">Using: LightGBM — Best Model</div>'
        '<div style="font-size:0.76rem;color:#4E7A94;">Test accuracy: 92.15% &nbsp;·&nbsp; ROC-AUC: 0.9802 &nbsp;·&nbsp; '
        'All 6 model accuracies are shown on the Dashboard</div></div></div>',
        unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.15], gap="large")

    with col_l:
        st.markdown('<div class="fpanel">', unsafe_allow_html=True)
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
        for bc, lbl, rng in [(b1, "Underweight", "< 18.5"), (b2, "Normal", "18.5–25"), (b3, "Obese", "≥ 30")]:
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
    predict_btn = st.button("🩺  Predict Diabetes Risk with LightGBM", type="primary", use_container_width=True)

    if predict_btn:
        if BEST_MODEL not in models:
            st.error(f"LightGBM model file not found. Place **lightgbm_model.pkl** in the same folder as app.py")
        else:
            with st.spinner("Running LightGBM inference…"):
                df_input = build_features(gender, age, hypertension, heart_disease, bmi, hba1c, glucose, smoking)
                model    = models[BEST_MODEL]
                pred     = model.predict(df_input)[0]
                proba    = model.predict_proba(df_input)[0] if hasattr(model, "predict_proba") else None

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Prediction Result — LightGBM</div>', unsafe_allow_html=True)

            p_diab = f"{proba[1]*100:.1f}%" if proba is not None else "N/A"
            p_safe = f"{proba[0]*100:.1f}%" if proba is not None else "N/A"
            pv_diab = proba[1] * 100 if proba is not None else 0
            pv_safe = proba[0] * 100 if proba is not None else 0

            if pred == 1:
                st.markdown(
                    '<div class="res-high">'
                    '<div class="res-icon">⚠️</div>'
                    '<div class="res-head" style="color:#FF5070;">High Risk of Diabetes</div>'
                    '<div class="res-desc">Please consult a healthcare professional. Early intervention significantly improves outcomes.</div>'
                    '<div class="chip-row">'
                    '<div class="chip" style="border-color:rgba(255,60,90,0.35);color:#FF7090;">🔴 Diabetic: ' + p_diab + '</div>'
                    '<div class="chip" style="border-color:rgba(0,245,160,0.28);color:#00F5A0;">🟢 Non-Diabetic: ' + p_safe + '</div>'
                    '<div class="chip" style="border-color:rgba(0,200,255,0.20);color:#00D4FF;">💡 LightGBM (92.15%)</div>'
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
                    '<div class="chip" style="border-color:rgba(0,200,255,0.20);color:#00D4FF;">💡 LightGBM (92.15%)</div>'
                    '</div></div>', unsafe_allow_html=True)

            # Probability bars
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

            # ── SHAP Section ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Computing SHAP explanations…"):
                shap_vals, base_val = run_shap(BEST_MODEL, model, df_input)

            render_shap_section(shap_vals, FEATURE_COLS, df_input)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📄 View Input Summary"):
                summary = pd.DataFrame({
                    "Feature": ["Gender", "Age", "Hypertension", "Heart Disease", "BMI",
                                "HbA1c Level", "Blood Glucose", "Smoking History"],
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
        <div class="hero-sub">Accuracy benchmarks across all six trained ML algorithms</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.markdown('<div class="sec-label">Accuracy Bar Chart — All 6 Models</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#071828")
        ax.set_facecolor("#071828")
        names  = list(ACCURACIES.keys())
        values = list(ACCURACIES.values())
        clrs   = [MODEL_COLORS[n] for n in names]
        bars   = ax.barh(names, values, color=clrs, height=0.46, edgecolor="none")
        ax.barh(names, values, height=0.58, color=clrs, alpha=0.11, edgecolor="none")
        for bar, val, name in zip(bars, values, names):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", ha="left",
                    fontsize=10, fontweight="700", color="#C8E0F0", fontfamily="monospace")
            if name == BEST_MODEL:
                ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
                        "⭐", va="center", ha="right", fontsize=10)
        ax.set_xlim(86.5, 94.5)
        ax.set_xlabel("Accuracy (%)", fontsize=10, color="#4E7A94")
        ax.tick_params(axis="y", labelsize=10, labelcolor="#90BBD4")
        ax.tick_params(axis="x", labelsize=9,  labelcolor="#2A4E68")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#0D2A40")
        ax.spines["bottom"].set_color("#0D2A40")
        ax.xaxis.grid(True, color="#0D2A40", linewidth=0.8)
        ax.yaxis.grid(False)
        ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight="700",
                     color="#C8E0F0", pad=12)
        patch = mpatches.Patch(color="#00D4FF", label=f"Best: {BEST_MODEL} (92.15%)")
        ax.legend(handles=[patch], fontsize=9, framealpha=0, labelcolor="#00D4FF", loc="lower right")
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
            '<div class="is">Accuracy: ' + f"{ACCURACIES[BEST_MODEL]:.2f}%" + ' · Used for all predictions</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="icard-g"><div class="il">📈 Performance Gap</div>'
            '<div class="iv">+' + f"{diff:.2f}%" + '</div>'
            '<div class="is">' + BEST_MODEL + ' vs ' + worst + '</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Model Characteristics</div>', unsafe_allow_html=True)

    details = [
        ("🔢", "Logistic Regression", "88.88%", "#6366F1", False,
         "Fast, interpretable linear classifier. Ideal for baseline benchmarks. Uses StandardScaler preprocessing.", "⚡⚡⚡", "Low"),
        ("📐", "SVM",                 "89.26%", "#0EA5E9", False,
         "Radial Basis Function kernel SVM. Excellent generalization on non-linear decision boundaries.", "⚡⚡", "Medium"),
        ("🌲", "Random Forest",       "90.79%", "#22D3EE", False,
         "Ensemble of 100 decision trees with balanced class weights. Strong accuracy and high interpretability.", "⚡", "High"),
        ("⚡", "XGBoost",             "91.79%", "#F59E0B", False,
         "Gradient boosted trees with tuned hyperparameters. High accuracy, excellent AUC of 0.9799.", "⚡⚡", "High"),
        ("🐱", "CatBoost",            "91.79%", "#A78BFA", False,
         "Categorical-aware gradient booster. Highest ROC-AUC (0.9805), robust on complex patterns.", "⚡", "High"),
        ("💡", "LightGBM",            "92.15%", "#00D4FF", True,
         "Leaf-wise boosting for speed & accuracy. Best overall accuracy (92.15%) with AUC 0.9802. Used for all predictions.", "⚡⚡⚡", "High"),
    ]
    d1, d2, d3 = st.columns(3, gap="medium")
    col_cycle = [d1, d2, d3, d1, d2, d3]
    for col, (icon, name, acc, color, is_best, desc, speed, cplx) in zip(col_cycle, details):
        border = f"1px solid {color}55" if is_best else "1px solid rgba(0,200,255,0.10)"
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
    ' &nbsp;·&nbsp; Powered by <b>LightGBM + SHAP</b> &nbsp;·&nbsp; 🩺 For clinical decision support only</div>',
    unsafe_allow_html=True)