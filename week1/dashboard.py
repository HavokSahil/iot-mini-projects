"""
Environmental Sensor Monitoring Dashboard
Complete version with model predictions and weather API comparison
"""
import yaml
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os, time, sys, warnings, json
import numpy as np
import requests
from dotenv import load_dotenv
import os

@st.cache_data
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

load_dotenv()  # loads .env file


warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Sensor Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown(
    """
    <style>
    /* Hide Streamlit default UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Full app background */
    .stApp {
        background-color: #0b0f1a;
    }

    /* Main content area */
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0b0f1a;
    }

    /* Card / container background */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #151a2d;
        padding: 1rem;
        border-radius: 0.6rem;
        border: 1px solid #23284a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0e1324;
        border-right: 1px solid #23284a;
    }

    /* Headings */
    h1 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }

    h2, h3 {
        font-size: 1.25rem;
        font-weight: 500;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #b5b8d1 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Dataframe */
    .dataframe {
        background-color: #151a2d !important;
        color: #ffffff !important;
        font-size: 0.85rem;
    }

    /* Horizontal rule */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #23284a;
    }

    /* Text */
    p, div, span, .stMarkdown, .stText, .stCaption {
        color: #e6e8ff !important;
    }

    /* Buttons */
    button {
        background-color: #1f2547 !important;
        color: #ffffff !important;
        border-radius: 0.4rem !important;
        border: 1px solid #2f3570 !important;
    }

    /* Plotly charts */
    .js-plotly-plot {
        background-color: #0b0f1a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================== MODEL IMPORTS ====================
model_dir = os.path.join(os.path.dirname(__file__), "model/ridge_regressor")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

TrainedModel = None
time_aware_ewma = None
import_error = None

try:
    import importlib.util
    model_spec = importlib.util.spec_from_file_location(
        "model_module", os.path.join(model_dir, "model.py")
    )
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)
    TrainedModel = model_module.TrainedModel

    util_spec = importlib.util.spec_from_file_location(
        "util_module", os.path.join(model_dir, "util.py")
    )
    util_module = importlib.util.module_from_spec(util_spec)
    util_spec.loader.exec_module(util_module)
    time_aware_ewma = util_module.time_aware_ewma
except Exception as e:
    import_error = str(e)

# ==================== FILE WATCH ====================
if "last_mtime" not in st.session_state:
    st.session_state.last_mtime = 0

def file_updated(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    if mtime > st.session_state.last_mtime:
        st.session_state.last_mtime = mtime
        st.cache_data.clear()
        return True
    return False

# ==================== CSV LOADER ====================
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path)
        
        # Handle single-column CSV (quoted rows)
        if len(df.columns) == 1:
            with open(path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            
            if not lines:
                return pd.DataFrame()
            
            header_line = lines[0].strip().strip('"').strip('\ufeff')
            columns = [col.strip() for col in header_line.split(',')]
            
            data_rows = []
            for line in lines[1:]:
                line = line.strip().strip('"')
                if line:
                    values = [val.strip() for val in line.split(',')]
                    if len(values) == len(columns):
                        data_rows.append(values)
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=columns)
            else:
                return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        numeric_cols = [
            "device_id","sequence","uptime","temperature","humidity","pressure",
            "iaq","iaq_accuracy","static_iaq","co2_ppm","voc_ppm","gas_percent",
            "rssi","snr"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        bool_cols = ['stabilized', 'run_in_complete']
        for col in bool_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.lower().map({'true': True, 'false': False})
                else:
                    df[col] = df[col].astype(bool)

        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# ==================== WEATHER API ====================
@st.cache_data(ttl=600)
def fetch_weather_api():
    url = "https://open-weather13.p.rapidapi.com/fivedaysforcast"
    params = {
        "latitude": "25.580903",
        "longitude": "84.836289",
        "lang": "EN"
    }
    headers = {
        "x-rapidapi-key": os.getenv("WEATHER_API_KEY"),
        "x-rapidapi-host": os.getenv("WEATHER_API_HOST")
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def extract_weather(data):
    first = data["list"][0]
    return {
        "temperature": first["main"]["temp"] - 273.15,  # Kelvin → °C
        "humidity": first["main"]["humidity"],
        "pressure": first["main"]["pressure"]           # hPa
    }

# ==================== METRICS ====================
def compute_errors(sensor, api):
    errors = {
        "temp": abs(sensor["temperature"] - api["temperature"]),
        "humidity": abs(sensor["humidity"] - api["humidity"]),
        "pressure": abs(sensor["pressure"] - api["pressure"])
    }
    mae = np.mean(list(errors.values()))
    return errors, mae

# ==================== FEATURE CALC ====================
def calculate_features(df):
    """Calculate features from sensor data for model prediction"""
    if df.empty or len(df) < 3:
        return None, None
    
    try:
        df_sorted = df.sort_values('timestamp' if 'timestamp' in df.columns else df.index.name or 'index').reset_index(drop=True)
        
        if len(df_sorted) < 3:
            return None, None
        
        last_idx = len(df_sorted) - 1
        lag1_idx = last_idx - 1
        lag2_idx = last_idx - 2
        
        lag1_temp = df_sorted.iloc[lag1_idx].get('temperature', None)
        lag2_temp = df_sorted.iloc[lag2_idx].get('temperature', None)
        lag1_pres = df_sorted.iloc[lag1_idx].get('pressure', None)
        lag2_pres = df_sorted.iloc[lag2_idx].get('pressure', None)
        lag1_hum = df_sorted.iloc[lag1_idx].get('humidity', None)
        lag2_hum = df_sorted.iloc[lag2_idx].get('humidity', None)
        lag1_iaq = df_sorted.iloc[lag1_idx].get('iaq', None)
        lag2_iaq = df_sorted.iloc[lag2_idx].get('iaq', None)
        
        if any(pd.isna([lag1_temp, lag2_temp, lag1_pres, lag2_pres, lag1_hum, lag2_hum, lag1_iaq, lag2_iaq])):
            return None, None
        
        if 'timestamp' in df_sorted.columns:
            timestamps = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
            time_diffs = [0]
            for i in range(min(last_idx, len(timestamps) - 1)):
                if pd.notna(timestamps.iloc[i]) and pd.notna(timestamps.iloc[i+1]):
                    diff_minutes = (timestamps.iloc[i+1] - timestamps.iloc[i]).total_seconds() / 60
                    time_diffs.append(diff_minutes)
                else:
                    time_diffs.append(0)
        else:
            time_diffs = [0] * (last_idx + 1)
        
        lag1_time_diff = time_diffs[lag1_idx] if lag1_idx < len(time_diffs) else 0
        lag2_time_diff = time_diffs[lag2_idx] if lag2_idx < len(time_diffs) else 0
        delta_t = time_diffs[last_idx] if last_idx < len(time_diffs) else 0
        
        temp_series = df_sorted.iloc[:last_idx+1]['temperature']
        pres_series = df_sorted.iloc[:last_idx+1]['pressure']
        hum_series = df_sorted.iloc[:last_idx+1]['humidity']
        iaq_series = df_sorted.iloc[:last_idx+1]['iaq']
        
        ewma_temp = temp_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_pres = pres_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_hum = hum_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_iaq = iaq_series.ewm(alpha=0.3).mean().iloc[-1]
        
        roll_mean_temp = temp_series.rolling(window=min(3, len(temp_series)), min_periods=1).mean().iloc[-1]
        roll_mean_pres = pres_series.rolling(window=min(3, len(pres_series)), min_periods=1).mean().iloc[-1]
        roll_mean_hum = hum_series.rolling(window=min(3, len(hum_series)), min_periods=1).mean().iloc[-1]
        roll_mean_iaq = iaq_series.rolling(window=min(3, len(iaq_series)), min_periods=1).mean().iloc[-1]
        
        features = {
            'lag1_temp': lag1_temp, 'lag2_temp': lag2_temp,
            'lag1_pres': lag1_pres, 'lag2_pres': lag2_pres,
            'lag1_hum': lag1_hum, 'lag2_hum': lag2_hum,
            'lag1_iaq': lag1_iaq, 'lag2_iaq': lag2_iaq,
            'lag1_time_diff': lag1_time_diff, 'lag2_time_diff': lag2_time_diff,
            'ewma_temp': ewma_temp, 'ewma_pres': ewma_pres,
            'ewma_hum': ewma_hum, 'ewma_iaq': ewma_iaq,
            'roll_mean_temp': roll_mean_temp, 'roll_mean_pres': roll_mean_pres,
            'roll_mean_hum': roll_mean_hum, 'roll_mean_iaq': roll_mean_iaq,
            'delta_t': delta_t
        }
        
        X_dict = features.copy()
        
        return features, pd.DataFrame([X_dict])
    except Exception as e:
        return None, None

@st.cache_resource
def load_model():
    path = "model/ridge_regressor/model.bin"
    if os.path.exists(path):
        return TrainedModel(path)
    return None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Filters")
    csv_path = st.text_input("CSV File Path", value="sensor_data.csv")
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=False)
    
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Data"],
        index=4
    )
    
    st.markdown("---")

# ==================== DATA LOADING ====================
updated = file_updated(csv_path)

if "df_all" not in st.session_state or updated:
    if os.path.exists(csv_path):
        st.session_state.df_all = load_csv(csv_path)
    else:
        st.session_state.df_all = pd.DataFrame()

df_all = st.session_state.df_all

# ==================== WEATHER FETCH ====================
if "weather" not in st.session_state:
    st.session_state.weather = None

# Fetch weather on file update or if not loaded
try:
    api_raw = fetch_weather_api()
    st.session_state.weather = extract_weather(api_raw)
except:
    if st.session_state.weather is None:
        st.session_state.weather = None

# ==================== DEVICE SELECT ====================
with st.sidebar:
    if not df_all.empty and "device_id" in df_all.columns:
        devs = sorted(df_all.device_id.dropna().unique())
        labels = ["All Devices"] + [f"Device {d}" for d in devs]
        sel = st.selectbox("Device", labels)
        device_id = None if sel == "All Devices" else devs[labels.index(sel)-1]
    else:
        device_id = None
        st.caption("No devices found in data")

# ==================== LAYOUT ====================
main_col, right_col = st.columns([3, 1])

def render_metric(label, value, unit=""):
    if pd.notna(value):
        st.metric(label, f"{value:.2f}{unit}")
    else:
        st.metric(label, "—")

def render_timeseries(df, field, y_label):
    if field not in df.columns or df[field].notna().sum() == 0:
        return

    x = df["timestamp"] if "timestamp" in df.columns else df.index
    y = df[field]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="lines+markers",
    name=field,
    line=dict(color="white", width=1.5),
    marker=dict(color="white", size=4)
))


    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=10, b=40),
        yaxis_title=y_label,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

# ==================== MAIN ====================
with main_col:
    st.title("Sensor Monitor")

    if df_all.empty:
        st.info("No data available. Please ensure the CSV file exists and contains sensor data.")
    else:
        df = df_all.copy()
        if device_id is not None:
            df = df[df.device_id == device_id]
        
        # Apply time filtering
        if time_range != "All Data" and 'timestamp' in df.columns and df['timestamp'].notna().any():
            ranges = {
                "Last Hour": timedelta(hours=1),
                "Last 6 Hours": timedelta(hours=6),
                "Last 24 Hours": timedelta(hours=24),
                "Last 7 Days": timedelta(days=7)
            }
            
            range_delta = ranges.get(time_range, timedelta(hours=24))
            max_timestamp = df['timestamp'].max()
            start_datetime = max_timestamp - range_delta
            
            mask = df['timestamp'] >= start_datetime
            filtered_df = df[mask].copy()
            
            if not filtered_df.empty:
                df = filtered_df
            else:
                st.warning(f"No data available for {time_range}. Showing all available data.")

        df = df.sort_values("timestamp" if 'timestamp' in df.columns else df.index)
        latest = df.iloc[-1]

        # Metrics
        st.markdown("### Live Metrics")

        header_fields = config.get("header", [])

        units = {
            "temperature": "°C",
            "humidity": "%",
            "pressure": " atm",
            "co2_ppm": " ppm",
            "voc_ppm": " ppm",
            "iaq": ""
        }

        cols = st.columns(len(header_fields))

        for col, field in zip(cols, header_fields):
            with col:
                value = latest.get(field, None)
                render_metric(field.upper(), value, units.get(field, ""))


        st.markdown("---")
        st.markdown("### Sensor Readings")

        plot_fields = config.get("plots", [])

        for i in range(0, len(plot_fields), 2):
            cols = st.columns(2)

            for col, field in zip(cols, plot_fields[i:i+2]):
                with col:
                    render_timeseries(
                        df,
                        field,
                        field.replace("_", " ").upper()
                    )


        st.markdown("---")
        st.markdown("### Recent Data")

        cols = ["timestamp","device_id","temperature","humidity","pressure","iaq","co2_ppm","voc_ppm"]
        available_cols = [c for c in cols if c in df.columns]
        display_df = df[available_cols].tail(100).copy()
        
        if 'timestamp' in display_df.columns:
            display_df = display_df.sort_values('timestamp', ascending=False)
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

# ==================== RIGHT SIDEBAR ====================
with right_col:
    st.markdown("### Analytics")
    st.markdown("---")
    
    # Weather API Comparison
    st.markdown("#### Weather API vs Sensor")
    
    if not df.empty and st.session_state.weather:
        sensor_vals = {
            "temperature": latest.get("temperature", None),
            "humidity": latest.get("humidity", None),
            "pressure": latest.get("pressure", None)
        }
        
        if all(pd.notna(v) for v in sensor_vals.values()):
            api_vals = st.session_state.weather
            api_vals["pressure"] = float(api_vals["pressure"]) / 1000.0 # hPa → atm
            errors, mae = compute_errors(sensor_vals, api_vals)

            st.markdown("**API (Current)**")
            st.write(f"🌡 {api_vals['temperature']:.2f} °C")
            st.write(f"💧 {api_vals['humidity']:.1f} %")
            st.write(f"📈 {api_vals['pressure']:.1f} atm")

            st.markdown("**Sensor**")
            st.write(f"🌡 {sensor_vals['temperature']:.2f} °C")
            st.write(f"💧 {sensor_vals['humidity']:.1f} %")
            st.write(f"📈 {sensor_vals['pressure']:.1f} atm")

            st.markdown("**Errors (MAE)**")
            st.metric("Temp", f"{errors['temp']:.2f}", delta=None, delta_color="off")
            st.metric("Humid", f"{errors['humidity']:.2f}", delta=None, delta_color="off")
            st.metric("Press", f"{errors['pressure']:.2f}", delta=None, delta_color="off")
            st.metric("Overall", f"{mae:.2f}", delta=None, delta_color="off")
        else:
            st.info("Waiting for complete sensor data")
    else:
        st.info("Waiting for API response")

    st.markdown("---")
    
    # Model Predictions
    st.markdown("#### Model Predictions")

    if not df.empty and len(df) >= 3:
        features, X_df = calculate_features(df)
        model = load_model()

        if features is not None:
            st.markdown("**Input Features**")
            st.write(f"Temp (t-1): {features['lag1_temp']:.2f}")
            st.write(f"Temp (t-2): {features['lag2_temp']:.2f}")
            st.write(f"Pressure (t-1): {features['lag1_pres']:.2f}")
            st.write(f"Humidity (t-1): {features['lag1_hum']:.2f}")
            st.write(f"IAQ (t-1): {features['lag1_iaq']:.0f}")
            st.write(f"Time Delta: {features['delta_t']:.2f} min")

            st.markdown("**Next Reading**")

            if model is None:
                st.warning("Model not loaded")
            else:
                try:
                    preds = model.predict(X_df)[0]
                    st.write(f"Temperature: {preds[0]:.2f} °C")
                    st.write(f"Pressure: {preds[1]:.2f}")
                    st.write(f"Humidity: {preds[2]:.2f} %")
                    st.write(f"IAQ: {preds[3]:.0f}")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.info("Insufficient data for features")
    else:
        st.info("Need at least 3 readings")

# ==================== FOOTER ====================
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if auto_refresh:
    time.sleep(2)
    st.rerun()
