# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 06:56:57 2025

@author: Lenovo
"""

"""app.py – COFSpace Adsorption Predictor (Extended CO2/CH4)
Only adsorption (uptake, N in mol/kg). No logD/D anywhere.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import base64
import io
from typing import Dict, Tuple, List

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_feature_ranges(csv_path: Path, target: str) -> Tuple[pd.Series, pd.Series, List[str]]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in {csv_path.name}.")
    # Sadece sayısal özellikler
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    return X.min(), X.max(), X.columns.tolist()

def _download_link_for_df(df: pd.DataFrame, filename: str) -> str:
    buf = io.BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download results</a>'

def _unit_from_target(target: str) -> str:
    # ör: "NCO2 at 1 bar (mol/kg)" -> "mol/kg"
    if "(" in target and target.endswith(")"):
        return target[target.rfind("(")+1:-1]
    return ""

# -----------------------------------------------------------------------------
# Constants (senin klasör yapına göre)
# -----------------------------------------------------------------------------
MODEL_DIR = Path("Extended Models") / "Models"
DATA_DIR  = Path("Extended Models") / "Data"

SCENARIOS: Dict[str, Dict[str, Dict[str, str]]] = {
    "CH4": {
        "1 BAR, 298 K": {
            "pkl": "model_CH4-1BAR.pkl",
            "csv": "CH4 - 1 BAR.csv",
            "target": "NCH4 at 1 bar (mol/kg)",
        },
        "10 BAR, 298 K": {
            "pkl": "model_CH4-10BAR.pkl",
            "csv": "CH4 - 10 BAR.csv",
            "target": "NCH4 at 10 bar (mol/kg)",
        },
    },
    "CO2": {
        "0.1 BAR, 298 K": {
            "pkl": "model_CO2-0.1BAR.pkl",
            "csv": "CO2 - 0.1 BAR.csv",
            "target": "NCO2 at 0.1 bar (mol/kg)",
        },
        "1 BAR, 298 K": {
            "pkl": "model_CO2-1BAR.pkl",
            "csv": "CO2 - 1 BAR.csv",
            "target": "NCO2 at 1 bar (mol/kg)",
        },
        "1 BAR, 393 K": {
            "pkl": "model_CO2-1BAR-393K.pkl",
            "csv": "CO2 - 1 BAR - 393 K.csv",
            "target": "NCO2 at 1 bar-393 K (mol/kg)",
        },
        "10 BAR, 298 K": {
            "pkl": "model_CO2-10BAR.pkl",
            "csv": "CO2 - 10 BAR.csv",
            "target": "NCO2 at 10 bar (mol/kg)",
        },
    },
}

# -----------------------------------------------------------------------------
# Page & Theme
# -----------------------------------------------------------------------------
st.set_page_config(page_title="COFSpace Adsorption Predictor", page_icon="🧪", layout="wide")

with st.sidebar:
    light_mode = st.toggle("🔆 Light mode", value=False, help="Toggle between light and dark themes.")
    gas_key = st.selectbox("Gas", list(SCENARIOS.keys()))
    cond_key = st.selectbox("Condition", list(SCENARIOS[gas_key].keys()))
    st.info("💡 Keep inputs within the training domain for reliable predictions.", icon="ℹ️")
    st.markdown("---")
    st.caption("Built with ❤️ & Streamlit — Extended CO₂/CH₄ (Adsorption Only)")

bg_css = (
    "background: radial-gradient(ellipse at top left, #fafafa 0%, #e8e8e8 60%); color:#222;"
    if light_mode else
    "background: radial-gradient(ellipse at top left, #1f2024 0%, #0f1013 60%); color:#f5f5f5;"
)
st.markdown(
    f"""
    <style>
        #MainMenu, footer, header {{visibility: hidden;}}
        .stApp {{{bg_css}}}
        div[data-baseweb="input"] input {{border-radius: 0.6rem !important;}}
        .hero-title {{text-align:center; font-size:3rem; font-weight:800; margin-top:-0.5rem;}}
        .hero-subtitle {{text-align:center; font-size:1.1rem; margin-bottom:1.6rem; opacity:0.85;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown("<div class='hero-title'>COFSpace Adsorption Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>ML-based estimation of uptake (N, mol/kg) under extended CO₂/CH₄ conditions</div>",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Load model & feature ranges
# -----------------------------------------------------------------------------
cfg = SCENARIOS[gas_key][cond_key]
model_path = MODEL_DIR / cfg["pkl"]
csv_path   = DATA_DIR / cfg["csv"]

if not model_path.exists():
    st.error(f"Model not found: {model_path}")
if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}")

model = load_model(model_path)
fmin, fmax, features = load_feature_ranges(csv_path, cfg["target"])
unit = _unit_from_target(cfg["target"])

# -----------------------------------------------------------------------------
# Single prediction
# -----------------------------------------------------------------------------
with st.form("input_form", border=False):
    st.markdown(f"#### Input Features — {gas_key} • {cond_key}")
    cols = st.columns(3)
    user_vals = {}
    for i, feat in enumerate(features):
        col = cols[i % 3]
        user_vals[feat] = col.number_input(
            f"{feat} ({fmin[feat]:.4g} – {fmax[feat]:.4g})",
            value=float(np.round((fmin[feat] + fmax[feat]) / 2, 6)),
            min_value=float(fmin[feat]),
            max_value=float(fmax[feat]),
            step=float(max((fmax[feat] - fmin[feat]) / 200, 1e-6)),
            format="%.6f",
            key=feat,
        )
    submitted = st.form_submit_button("✨ Predict Uptake", type="primary")

if submitted:
    X_input = pd.DataFrame([user_vals])
    yhat = float(model.predict(X_input)[0])
    label = f"N [{unit}]" if unit else "N"
    st.metric(label, f"{yhat:.4f}")
    st.success("Prediction complete ✅")

# -----------------------------------------------------------------------------
# Batch prediction
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📄 Batch prediction from CSV")
csv_file = st.file_uploader(
    "Upload a CSV that contains *exactly* the same numeric feature columns "
    f"({len(features)} columns). Extra columns will be ignored.",
    type=["csv"],
)

if csv_file is not None:
    try:
        df_in = pd.read_csv(csv_file)
        used = [c for c in features if c in df_in.columns]
        missing = [c for c in features if c not in df_in.columns]
        df_in = df_in[used]
        if missing:
            st.error(f"Missing required feature columns: {', '.join(missing)}")
        else:
            # Out-of-range kontrolü (özellik bazında sayım)
            oob_counts = {}
            for feat in used:
                oob_mask = (df_in[feat] < fmin[feat]) | (df_in[feat] > fmax[feat])
                cnt = int(oob_mask.sum())
                if cnt > 0:
                    oob_counts[feat] = cnt
            if oob_counts:
                lines = [f"• **{k}**: {v} row(s) outside [{fmin[k]:.3g}, {fmax[k]:.3g}]" for k, v in oob_counts.items()]
                st.warning("Some rows are outside the training domain:\n" + "\n".join(lines))

            with st.spinner("Running predictions …"):
                preds = model.predict(df_in)
                df_out = df_in.copy()
                colname = "N_pred" + (f" ({unit})" if unit else "")
                df_out[colname] = preds

                # İsteğe bağlı: satır bazında OOB flag (1=out-of-range)
                if oob_counts:
                    any_oob = pd.Series(False, index=df_in.index)
                    for feat in used:
                        any_oob |= (df_in[feat] < fmin[feat]) | (df_in[feat] > fmax[feat])
                    df_out["OOB_flag"] = any_oob.astype(int)

                safe_cond = cond_key.replace(", ", "_").replace(" ", "")
                fname = f"COFSpace_{gas_key}_{safe_cond}_uptake_predictions.csv"
                st.markdown(_download_link_for_df(df_out, fname), unsafe_allow_html=True)
                st.success(f"Finished! Predicted {len(df_out)} rows.")
    except Exception as e:
        st.exception(e)
