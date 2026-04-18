import streamlit as st
import folium
import requests
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from streamlit_folium import st_folium
from dotenv import load_dotenv
from distress_detector import analyze_text
import os

# Load API key
load_dotenv()
try:
    # Streamlit Cloud
    WEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except:
    # Local development
    WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Load ML model and explainer
with open("disaster_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

FEATURES = [
    "magnitude", "wind_speed", "temperature",
    "weather_severity", "distress_count",
    "hour_of_day", "population_proxy"
]

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DisasterSenseAI",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 4px;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med  { color: #ffa500; font-weight: bold; }
    .risk-low  { color: #00cc88; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🚨 DisasterSenseAI")
st.markdown("**Real-Time Multi-Modal Disaster Risk Intelligence Platform**")
st.markdown("---")

# ─────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
    response = requests.get(url)
    return response.json()["features"]

@st.cache_data(ttl=300)
def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

@st.cache_data(ttl=300)
def analyze_all_signals():
    results = []
    for signal in DISTRESS_SIGNALS:
        result = analyze_text(signal["text"])
        result["lat"] = signal["lat"]
        result["lon"] = signal["lon"]
        result["location"] = signal["location"]
        results.append(result)
    return results

# ─────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────
def ml_predict(magnitude, wind_speed, temperature,
               weather_severity, distress_count,
               hour_of_day=12, population_proxy=0.5):

    features = np.array([[
        magnitude, wind_speed, temperature,
        weather_severity, distress_count,
        hour_of_day, population_proxy
    ]])

    prob = model.predict_proba(features)[0][1]
    risk_score = int(prob * 100)

    if risk_score >= 70:
        risk_label = "🔴 HIGH"
        color = "red"
    elif risk_score >= 40:
        risk_label = "🟠 MEDIUM"
        color = "orange"
    else:
        risk_label = "🔵 LOW"
        color = "blue"

    shap_vals = explainer.shap_values(features)[0]
    explanation = dict(zip(FEATURES, shap_vals))

    return risk_score, risk_label, color, explanation

def plot_shap(explanation, place):
    features = list(explanation.keys())
    values = list(explanation.values())

    colors = ["#ff4b4b" if v > 0 else "#00cc88" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    bars = ax.barh(features, values, color=colors)
    ax.set_xlabel("SHAP Value (impact on risk)", color="white")
    ax.set_title(f"Why this risk score?\n{place[:40]}", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0, color='white', linewidth=0.5)

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# DISTRESS SIGNALS
# ─────────────────────────────────────────
DISTRESS_SIGNALS = [
    {"text": "Help! Earthquake destroyed our building, people trapped inside!", "lat": -51.7, "lon": -72.5, "location": "Chile"},
    {"text": "SOS! Flooding everywhere, we are stranded on rooftop!", "lat": 13.7, "lon": 100.5, "location": "Bangkok"},
    {"text": "Everything is fine here, just a small tremor felt", "lat": 35.6, "lon": 139.6, "location": "Tokyo"},
    {"text": "Need urgent rescue! House collapsed after earthquake!", "lat": -8.3, "lon": 115.1, "location": "Bali"},
    {"text": "Roads blocked by landslide, we are stuck!", "lat": 27.7, "lon": 85.3, "location": "Nepal"},
    {"text": "Beautiful day today, nothing unusual here", "lat": 40.7, "lon": -74.0, "location": "New York"},
    {"text": "Gas leak detected near earthquake zone, please help!", "lat": 37.7, "lon": 15.0, "location": "Sicily"},
    {"text": "Tsunami warning! People running to higher ground!", "lat": 3.5, "lon": 98.6, "location": "Sumatra"},
]

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
earthquakes = get_earthquakes()

with st.spinner("🧠 AI analyzing distress signals..."):
    signal_results = analyze_all_signals()

distress_count = sum(1 for s in signal_results if s["is_distress"])
high_risk_list = []

# ─────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌍 Earthquakes Detected", len(earthquakes))
with col2:
    st.metric("🚨 Distress Signals", distress_count)
with col3:
    st.metric("🧠 AI Model", "XGBoost")
with col4:
    st.metric("📊 Model AUC-ROC", "0.9777")

st.markdown("---")

# ─────────────────────────────────────────
# TWO COLUMN LAYOUT
# ─────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("🗺️ Live Risk Map")

    m = folium.Map(location=[20, 0], zoom_start=2)
    earthquake_layer = folium.FeatureGroup(name="🌍 Earthquakes")
    distress_layer   = folium.FeatureGroup(name="🚨 Distress Signals")

    progress = st.progress(0)
    status   = st.empty()

    for i, quake in enumerate(earthquakes):
        coords = quake["geometry"]["coordinates"]
        props  = quake["properties"]
        lon, lat, magnitude = coords[0], coords[1], props["mag"]
        place = props["place"]

        progress.progress((i + 1) / len(earthquakes))
        status.text(f"🧠 ML analyzing: {place[:40]}...")

        weather = get_weather(lat, lon)

        # Extract weather features
        wind_speed       = weather["wind"]["speed"] if weather else 10
        temperature      = weather["main"]["temp"] if weather else 20
        weather_id       = weather["weather"][0]["id"] if weather else 800
        weather_severity = 2 if weather_id < 300 else 1 if weather_id < 700 else 0

        # ML PREDICTION (replaces old rule-based scoring)
        risk_score, risk_label, color, explanation = ml_predict(
            magnitude, wind_speed, temperature,
            weather_severity, distress_count
        )

        # Store high risk
        if risk_score >= 70:
            high_risk_list.append({
                "place": place,
                "magnitude": magnitude,
                "risk_score": risk_score,
                "risk_label": risk_label,
                "explanation": explanation
            })

        weather_info = ""
        if weather:
            weather_info = (f" | 🌡️ {temperature}°C"
                          f" | 💨 {wind_speed}m/s"
                          f" | {weather['weather'][0]['description']}")

        folium.CircleMarker(
            location=[lat, lon],
            radius=magnitude * 3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"📍 {place}\n"
                f"⚡ Magnitude: {magnitude}\n"
                f"🎯 ML Risk Score: {risk_score}/100\n"
                f"{risk_label}\n"
                f"{weather_info}",
                max_width=280
            )
        ).add_to(earthquake_layer)

    progress.empty()
    status.empty()

    # Distress markers
    for signal in signal_results:
        icon_color = "red" if signal["is_distress"] else "green"
        icon_sign  = "exclamation-sign" if signal["is_distress"] else "ok-sign"
        folium.Marker(
            location=[signal["lat"], signal["lon"]],
            popup=folium.Popup(
                f"{'🚨 DISTRESS' if signal['is_distress'] else '✅ SAFE'}\n"
                f"📍 {signal['location']}\n"
                f"💬 {signal['text'][:60]}...\n"
                f"🧠 Confidence: {signal['confidence']}%",
                max_width=250
            ),
            icon=folium.Icon(color=icon_color, icon=icon_sign)
        ).add_to(distress_layer)

    earthquake_layer.add_to(m)
    distress_layer.add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=800, height=500)

# ─────────────────────────────────────────
# RIGHT COLUMN — Alerts + SHAP
# ─────────────────────────────────────────
with right_col:
    st.subheader("🚨 Live Alerts")

    if high_risk_list:
        for zone in high_risk_list[:4]:
            with st.expander(f"🔴 {zone['place'][:35]}"):
                st.write(f"**Magnitude:** {zone['magnitude']}")
                st.write(f"**Risk Score:** {zone['risk_score']}/100")
                st.write(f"**Level:** {zone['risk_label']}")
                st.write("**Why this score?**")
                fig = plot_shap(zone["explanation"], zone["place"])
                st.pyplot(fig)
                plt.close()
    else:
        st.success("✅ No high risk zones right now")

    st.markdown("---")
    st.subheader("🧠 Distress Signals")
    for s in signal_results:
        if s["is_distress"]:
            st.warning(f"📍 **{s['location']}** — {s['confidence']}%\n\n"
                      f"_{s['text'][:55]}..._")

# ─────────────────────────────────────────
# CUSTOM SCENARIO PREDICTOR
# ─────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Custom Scenario Predictor")
st.markdown("**Type any values and get instant AI prediction with explanation**")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    p_magnitude  = st.slider("⚡ Magnitude",        2.5, 9.0, 5.0, 0.1)
    p_wind       = st.slider("💨 Wind Speed (m/s)", 0,   50,  10)

with pred_col2:
    p_temp       = st.slider("🌡️ Temperature (°C)", -10, 45, 20)
    p_weather    = st.selectbox("☁️ Weather", [0, 1, 2],
                                format_func=lambda x: ["Clear", "Rain", "Storm"][x])

with pred_col3:
    p_distress   = st.slider("🚨 Distress Signals", 0, 10, 2)
    p_hour       = st.slider("🕐 Hour of Day",       0, 23, 12)
    p_population = st.slider("👥 Population Density", 0.0, 1.0, 0.5)

if st.button("🧠 Predict Risk Now", type="primary"):
    risk_score, risk_label, color, explanation = ml_predict(
        p_magnitude, p_wind, p_temp,
        p_weather, p_distress,
        p_hour, p_population
    )

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown(f"### Risk Score: **{risk_score}/100**")
        st.markdown(f"### Level: **{risk_label}**")

        if risk_score >= 70:
            st.error("🚨 HIGH RISK — Immediate action needed!")
        elif risk_score >= 40:
            st.warning("⚠️ MEDIUM RISK — Monitor closely")
        else:
            st.success("✅ LOW RISK — Situation under control")

    with res_col2:
        fig = plot_shap(explanation, "Custom Scenario")
        st.pyplot(fig)
        plt.close()