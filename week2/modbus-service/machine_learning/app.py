# -*- coding: utf-8 -*-
"""
Soil Classification IoT Lab Dashboard.
Paths and data sources are configurable via config.json and environment (no hardcoded paths).
"""
import json
import os
import time
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
CONFIG_EXAMPLE_PATH = os.path.join(SCRIPT_DIR, "config.example.json")


def load_config():
    """Load paths from config.json; fallback to config.example.json or env/defaults."""
    defaults = {
        "model_path": os.environ.get("SOIL_MODEL_PATH", "soil_model.pkl"),
        "scaler_path": os.environ.get("SOIL_SCALER_PATH", "scaler.pkl"),
        "sensor_csv_path": os.environ.get("SENSOR_CSV_PATH", "soil_data.csv"),
        "soil_dataset_path": os.environ.get("SOIL_DATASET_PATH", "soil_dataset_final.csv"),
        # Optional display/refresh defaults:
        "auto_refresh": True,
        "refresh_interval": 3,
        "show_records": 100,
        "time_range": "Last 100",  # One of: Last 100 / Last 500 / Last 1000 / Last 5000 / All Data
        "show_distributions": True,
        "show_stats": True,
    }
    for path in (CONFIG_PATH, CONFIG_EXAMPLE_PATH):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    for k in defaults:
                        if k in loaded and loaded[k] is not None:
                            defaults[k] = loaded[k]
            except Exception:
                pass
            break
    return defaults



st.set_page_config(
    page_title="Soil Classification | IoT Lab",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background-color: #0b0f1a; }
    .main { padding-top: 2rem; padding-bottom: 2rem; background-color: #0b0f1a; }
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #151a2d; padding: 1rem; border-radius: 0.6rem; border: 1px solid #23284a;
    }
    section[data-testid="stSidebar"] { background-color: #0e1324; border-right: 1px solid #23284a; }
    h1 { font-size: 1.8rem; font-weight: 600; color: #ffffff; margin-bottom: 1.5rem; }
    h2, h3 { font-size: 1.25rem; font-weight: 500; color: #ffffff; margin-top: 1.5rem; margin-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; color: #ffffff !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #b5b8d1 !important; text-transform: uppercase; letter-spacing: 0.08em; }
    .dataframe { background-color: #151a2d !important; color: #ffffff !important; font-size: 0.85rem; }
    hr { margin: 2rem 0; border: none; border-top: 1px solid #23284a; }
    p, div, span, .stMarkdown, .stText, .stCaption { color: #e6e8ff !important; }
    button { background-color: #1f2547 !important; color: #ffffff !important; border-radius: 0.4rem !important; border: 1px solid #2f3570 !important; }
    .js-plotly-plot { background-color: #0b0f1a !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { background-color: #1f2547; color: #b5b8d1; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #2f3570; color: #ffffff; }
    .stInfo { background-color: #1f2547; color: #ffffff; border: 1px solid #2f3570; }
    </style>
    """,
    unsafe_allow_html=True,
)


CONFIG = load_config()


SOIL_CLASSES = {0: "Saline Soil", 1: "Balanced Soil", 2: "Dry Soil", 3: "Acidic Soil"}

SOIL_RECOMMENDATIONS = {
    0: """**Saline Soil Management:**
- Improve drainage to leach out salts
- Use salt-tolerant crop varieties (barley, cotton, date palms)
- Add organic matter to improve soil structure
- Consider gypsum application to displace sodium
- Monitor irrigation water quality""",
    1: """**Balanced Soil - Optimal Conditions:**
- Maintain current practices
- Regular monitoring recommended (monthly basis)
- Suitable for most crop varieties
- Continue organic matter additions
- Monitor nutrient levels seasonally""",
    2: """**Dry Soil Management:**
- Increase irrigation frequency
- Add organic mulch to retain moisture
- Consider drip irrigation systems
- Use drought-resistant crop varieties
- Improve soil structure with compost""",
    3: """**Acidic Soil Management:**
- Apply agricultural lime to raise pH
- Use acid-loving plants or pH-neutral crops
- Add organic matter regularly
- Monitor nutrient availability (especially phosphorus)
- Consider dolomitic lime for magnesium deficiency""",
}


MODEL_FEATURES = ["moisture", "ec", "ph", "salinity", "tds"]



def resolve_path(path_key, base_dir=SCRIPT_DIR):
    """Resolve path: if not absolute, join with script directory and normalize."""
    p = (path_key or "").strip()
    if not p:
        return ""
    if not os.path.isabs(p):
        p = os.path.normpath(os.path.join(base_dir, p))
    return p


def render_metric(label, value, unit="", delta=None):
    if value is not None and pd.notna(value):
        if delta is not None and pd.notna(delta):
            st.metric(label, f"{value:.2f}{unit}", delta=f"{delta:+.2f}{unit}")
        else:
            st.metric(label, f"{value:.2f}{unit}")
    else:
        st.metric(label, "—")


def render_timeseries(df, field, y_label, color="white"):
    if field not in df.columns or df[field].notna().sum() == 0:
        st.caption(f"No data for {field}")
        return
    df_clean = df[[field]].dropna()
    y = df_clean[field]
    x = list(range(len(y)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=field,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            fill="tozeroy",
            fillcolor="rgba(255, 255, 255, 0.1)",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=10, b=40),
        yaxis_title=y_label,
        xaxis_title="Sample Index",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#23284a", color="#ffffff"),
        yaxis=dict(showgrid=True, gridcolor="#23284a", color="#ffffff"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_distribution(df, field, title, color="white"):
    if field not in df.columns or df[field].notna().sum() == 0:
        return
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[field].dropna(), nbinsx=50, marker_color=color, opacity=0.7))
    fig.update_layout(
        title=title,
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#23284a", color="#ffffff"),
        yaxis=dict(showgrid=True, gridcolor="#23284a", color="#ffffff"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def calculate_statistics(df, field):
    if field not in df.columns:
        return {}
    data = df[field].dropna()
    if len(data) == 0:
        return {}
    return {
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
        "q25": data.quantile(0.25),
        "q75": data.quantile(0.75),
    }


def load_csv_fresh(path):
    """Load CSV without caching so sensor updates appear on refresh."""
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            with open(path, "r", encoding="utf-8-sig") as f:
                lines = f.readlines()
            if not lines:
                return pd.DataFrame()
            header_line = lines[0].strip().strip('"').strip("\ufeff")
            columns = [c.strip() for c in header_line.split(",")]
            data_rows = []
            for line in lines[1:]:
                line = line.strip().strip('"')
                if line:
                    values = [v.strip() for v in line.split(",")]
                    if len(values) == len(columns):
                        data_rows.append(values)
            if data_rows:
                df = pd.DataFrame(data_rows, columns=columns)
            else:
                return pd.DataFrame()
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "time" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            except Exception:
                df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
        numeric_cols = [
            "moisture",
            "ec",
            "ph",
            "salinity",
            "tds",
            "temperature",
            "humidity",
            "pressure",
            "nitrogen",
            "phosphorus",
            "potassium",
            "device_id",
            "sequence",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading CSV from {path}: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_model(model_path, scaler_path):
    """Load model and scaler from configurable paths."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None

model_path_resolved = resolve_path(CONFIG["model_path"])
scaler_path_resolved = resolve_path(CONFIG["scaler_path"])
sensor_path_resolved = resolve_path(CONFIG["sensor_csv_path"])
soil_dataset_path_resolved = resolve_path(CONFIG["soil_dataset_path"])

model_exists = os.path.isfile(model_path_resolved)
scaler_exists = os.path.isfile(scaler_path_resolved)
sensor_exists = os.path.isfile(sensor_path_resolved)
soil_exists = os.path.isfile(soil_dataset_path_resolved)

auto_refresh = bool(CONFIG.get("auto_refresh", True))
refresh_interval = int(CONFIG.get("refresh_interval", 3))
show_records = int(CONFIG.get("show_records", 100))
time_range = CONFIG.get("time_range", "Last 100")
show_distributions = bool(CONFIG.get("show_distributions", True))
show_stats = bool(CONFIG.get("show_stats", True))


soil_df = load_csv_fresh(soil_dataset_path_resolved) if soil_exists else None
sensor_df = load_csv_fresh(sensor_path_resolved) if sensor_exists else None


ranges_map = {"Last 100": 100, "Last 500": 500, "Last 1000": 1000, "Last 5000": 5000}
n_tail = ranges_map.get(time_range, 100)
if sensor_df is not None and not sensor_df.empty:
    df_for_viz = sensor_df.tail(n_tail).copy() if time_range != "All Data" else sensor_df.copy()
    data_source_name = "Sensor (live)"
elif soil_df is not None and not soil_df.empty:
    df_for_viz = soil_df.tail(n_tail).copy() if time_range != "All Data" else soil_df.copy()
    data_source_name = "Soil dataset"
else:
    df_for_viz = pd.DataFrame()
    data_source_name = None


st.title("🌱 Soil Classification – IoT Lab")

if auto_refresh:
    st.caption(
        f"🔴 LIVE – Auto-refresh every {refresh_interval}s | Last update: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
else:
    st.caption(f"⚪ Manual mode | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


st.markdown("### 🔬 Soil classifier")
with st.container():
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Enter parameters (model uses moisture, EC, pH, salinity, TDS)**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            moisture = st.number_input(
                "Moisture", min_value=0.0, max_value=100.0, value=35.0, step=0.1, key="moisture"
            )
            temperature = st.number_input(
                "Temperature (°C)", min_value=-10.0, max_value=60.0, value=22.0, step=0.1, key="temp"
            )
            ec = st.number_input("EC", min_value=0.0, value=1.2, step=0.1, key="ec")
        with col_b:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.8, step=0.1, key="ph")
            nitrogen = st.number_input("Nitrogen", min_value=0.0, value=0.0, step=0.1, key="nitrogen")
            phosphorus = st.number_input(
                "Phosphorus", min_value=0.0, value=0.0, step=0.1, key="phosphorus"
            )
        with col_c:
            potassium = st.number_input(
                "Potassium", min_value=0.0, value=0.0, step=0.1, key="potassium"
            )
            salinity = st.number_input(
                "Salinity", min_value=0.0, value=0.3, step=0.1, key="salinity"
            )
            tds = st.number_input("TDS", min_value=0.0, value=600.0, step=1.0, key="tds")

    with c2:
        st.markdown("**Run model**")
        predict_clicked = st.button("🔮 Predict", type="primary", key="predict_btn")

prediction_result = st.session_state.get("last_prediction")
model, scaler = load_model(model_path_resolved, scaler_path_resolved) if model_exists and scaler_exists else (
    None,
    None,
)

if predict_clicked and model is not None:
    try:
        X = pd.DataFrame(
            [
                {
                    "moisture": moisture,
                    "ec": ec,
                    "ph": ph,
                    "salinity": salinity,
                    "tds": tds,
                }
            ]
        )
        pred = model.predict(X)
        pred_label = SOIL_CLASSES.get(int(pred[0]), "Unknown")
        prediction_result = (int(pred[0]), pred_label)
        st.session_state["last_prediction"] = prediction_result
    except Exception as e:
        st.error(f"Prediction error: {e}")
elif predict_clicked and model is None:
    st.warning("Model or scaler not loaded. Check paths in config.json or environment variables.")

if prediction_result is not None:
    _, pred_label = prediction_result
    colors = {
        "Saline Soil": "#ff6b6b",
        "Balanced Soil": "#51cf66",
        "Dry Soil": "#ffd43b",
        "Acidic Soil": "#9775fa",
    }
    color = colors.get(pred_label, "#1f2547")
    st.markdown(
        f"<div style='background: linear-gradient(135deg, {color}22 0%, {color}44 100%); "
        f"padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid {color}; margin: 1rem 0;'>"
        f"<h2 style='margin: 0; color: {color}; font-size: 1.2rem;'>{pred_label}</h2></div>",
        unsafe_allow_html=True,
    )
    with st.expander("📖 Management guide"):
        st.markdown(SOIL_RECOMMENDATIONS.get(prediction_result[0], ""))

if not model_exists or not scaler_exists:
    st.info("⚠️ Set model and scaler paths in config.json (e.g. soil_model.pkl, scaler.pkl).")

st.markdown("---")

# Live metrics, charts, stats, recent data from sensor/dataset 
if df_for_viz is not None and not df_for_viz.empty:
    # Current readings (all 9 parameters)
    st.markdown("### 📍 Current readings")
    if sensor_df is not None and not sensor_df.empty:
        st.caption("🔴 Live sensor data" if data_source_name else "")
    latest = df_for_viz.iloc[-1]
    previous = df_for_viz.iloc[-2] if len(df_for_viz) > 1 else latest
    param_defs = [
        ("Moisture", "moisture"),
        ("Temperature", "temperature"),
        ("EC", "ec"),
        ("pH", "ph"),
        ("Nitrogen", "nitrogen"),
        ("Phosphorus", "phosphorus"),
        ("Potassium", "potassium"),
        ("Salinity", "salinity"),
        ("TDS", "tds"),
    ]
    for row_start in range(0, len(param_defs), 3):
        row_params = param_defs[row_start : row_start + 3]
        cols = st.columns(len(row_params))
        for (label, key), col in zip(row_params, cols):
            with col:
                val_latest = latest.get(key) if key in df_for_viz.columns else None
                val_prev = (
                    previous.get(key)
                    if (key in df_for_viz.columns and len(df_for_viz) > 1)
                    else None
                )
                delta = (
                    val_latest - val_prev
                    if (val_latest is not None and val_prev is not None and len(df_for_viz) > 1)
                    else None
                )
                render_metric(label, val_latest, "", delta)
    st.markdown("---")

    # Time series
    st.markdown("### 📈 Time series")
    col1, col2 = st.columns(2)
    with col1:
        render_timeseries(df_for_viz, "moisture", "Moisture", "#4dabf7")
        render_timeseries(df_for_viz, "ph", "pH", "#51cf66")
    with col2:
        render_timeseries(df_for_viz, "ec", "EC", "#ffd43b")
        render_timeseries(df_for_viz, "tds", "TDS", "#ff6b6b")
    st.markdown("---")

    # Statistical analysis (histograms + table)
    if show_distributions:
        st.markdown("### 📊 Statistical analysis")
        d1, d2 = st.columns(2)
        with d1:
            render_distribution(df_for_viz, "moisture", "Moisture distribution", "#4dabf7")
            render_distribution(df_for_viz, "ph", "pH distribution", "#51cf66")
        with d2:
            render_distribution(df_for_viz, "ec", "EC distribution", "#ffd43b")
            render_distribution(df_for_viz, "salinity", "Salinity distribution", "#ff6b6b")
        st.markdown("---")

    if show_stats:
        st.markdown("### 📋 Descriptive statistics")
        stats_rows = []
        for field in MODEL_FEATURES:
            if field in df_for_viz.columns:
                s = calculate_statistics(df_for_viz, field)
                if s:
                    stats_rows.append(
                        {
                            "Parameter": field.upper(),
                            "Mean": f"{s['mean']:.2f}",
                            "Median": f"{s['median']:.2f}",
                            "Std Dev": f"{s['std']:.2f}",
                            "Min": f"{s['min']:.2f}",
                            "Max": f"{s['max']:.2f}",
                            "Q25": f"{s['q25']:.2f}",
                            "Q75": f"{s['q75']:.2f}",
                        }
                    )
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
        st.markdown("---")

    # Recent data table
    st.markdown("### 📄 Recent data records")
    display_df = sensor_df if (sensor_df is not None and not sensor_df.empty) else soil_df
    if display_df is not None and not display_df.empty:
        st.caption(
            "🔴 Live sensor data"
            if (sensor_df is not None and not sensor_df.empty)
            else "Soil dataset"
        )
        cols_display = []
        if "timestamp" in display_df.columns:
            cols_display.append("timestamp")
        for c in [
            "moisture",
            "temperature",
            "ec",
            "ph",
            "nitrogen",
            "phosphorus",
            "potassium",
            "salinity",
            "tds",
        ]:
            if c in display_df.columns:
                cols_display.append(c)
        if cols_display:
            out = display_df[cols_display].tail(show_records).iloc[::-1]
            if "timestamp" in out.columns:
                out["timestamp"] = (
                    pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                )
            out.index = range(1, len(out) + 1)
            st.dataframe(out, use_container_width=True, height=400)
            st.download_button(
                "⬇️ Download as CSV",
                data=out.to_csv(index=True),
                file_name=f"soil_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_csv",
            )
else:
    st.info("No sensor or soil dataset loaded. Set CSV paths in config.json.")

# ----- Footer -----
st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1:
    st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with f2:
    if soil_df is not None:
        st.caption(f"📊 Soil dataset: {len(soil_df):,} rows")
    if sensor_df is not None:
        st.caption(f"📡 Sensor: {len(sensor_df):,} rows")
with f3:
    if auto_refresh:
        st.caption(f"🔄 Auto-refresh: {refresh_interval}s")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()