import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
from dotenv import load_dotenv
from distress_detector import analyze_text
import os

# Load API key
load_dotenv()
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Page config
st.set_page_config(page_title="DisasterSenseAI", page_icon="🚨", layout="wide")
st.title("🚨 DisasterSenseAI")
st.subheader("Real-Time Disaster Risk Dashboard")

# ─────────────────────────────────────────
# SIMULATED SOCIAL MEDIA DISTRESS SIGNALS
# In Week 5 we replace these with real data
# ─────────────────────────────────────────
DISTRESS_SIGNALS = [
    {"text": "Help! Earthquake destroyed our building, people trapped inside!", "lat": -51.7, "lon": -72.5, "location": "Chile"},
    {"text": "SOS! Flooding everywhere, we are stranded on rooftop!", "lat": 13.7, "lon": 100.5, "location": "Bangkok"},
    {"text": "Everything is fine here, just a small tremor felt", "lat": 35.6, "lon": 139.6, "location": "Tokyo"},
    {"text": "Need urgent rescue! House collapsed after earthquake!", "lat": -8.3, "lon": 115.1, "location": "Bali"},
    {"text": "Roads blocked by landslide, we are stuck!", "lat": 27.7, "lon": 85.3, "location": "Nepal"},
    {"text": "Beautiful day today, nothing unusual here", "lat": 40.7, "lon": -74.0, "location": "New York"},
    {"text": "Gas leak detected near earthquake zone, please help!", "lat": 37.7, "lon": 15.0, "location": "Sicily"},
    {"text": "Just felt a tremor, everything seems okay now", "lat": 19.4, "lon": -155.2, "location": "Hawaii"},
    {"text": "Tsunami warning! People running to higher ground!", "lat": 3.5, "lon": 98.6, "location": "Sumatra"},
    {"text": "Minor shaking felt, no damage reported", "lat": 33.4, "lon": 73.0, "location": "Pakistan"},
]

# ─────────────────────────────────────────
# FETCH EARTHQUAKE DATA
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
    response = requests.get(url)
    return response.json()["features"]

# ─────────────────────────────────────────
# FETCH WEATHER DATA
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# ─────────────────────────────────────────
# CALCULATE RISK SCORE
# ─────────────────────────────────────────
def calculate_risk(magnitude, weather, nearby_distress=0):
    risk = 0

    # Earthquake contribution
    if magnitude >= 6:
        risk += 50
    elif magnitude >= 5:
        risk += 35
    elif magnitude >= 4:
        risk += 20
    else:
        risk += 10

    # Weather contribution
    if weather:
        wind_speed = weather["wind"]["speed"]
        weather_id = weather["weather"][0]["id"]
        if wind_speed > 20:
            risk += 20
        elif wind_speed > 10:
            risk += 10
        if weather_id < 300:
            risk += 20
        elif weather_id < 600:
            risk += 10
        elif weather_id < 700:
            risk += 10

    # Human distress signal contribution
    risk += nearby_distress * 15

    return min(risk, 100)

# ─────────────────────────────────────────
# ANALYZE DISTRESS SIGNALS
# ─────────────────────────────────────────
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
# MAIN APP
# ─────────────────────────────────────────
earthquakes = get_earthquakes()

# Analyze distress signals
with st.spinner("🧠 AI analyzing distress signals..."):
    signal_results = analyze_all_signals()

distress_count = sum(1 for s in signal_results if s["is_distress"])

# Sidebar
st.sidebar.title("📊 Risk Summary")
st.sidebar.write(f"🌍 Earthquakes detected: **{len(earthquakes)}**")
st.sidebar.write(f"🚨 Distress signals: **{distress_count}**")

# Create map with layer control
m = folium.Map(location=[20, 0], zoom_start=2)
earthquake_layer = folium.FeatureGroup(name="🌍 Earthquakes")
distress_layer = folium.FeatureGroup(name="🚨 Distress Signals")

# Store risk data
risk_list = []

# Progress
progress = st.progress(0)
status = st.empty()

# Plot earthquakes
for i, quake in enumerate(earthquakes):
    coords = quake["geometry"]["coordinates"]
    props = quake["properties"]
    lon = coords[0]
    lat = coords[1]
    magnitude = props["mag"]
    place = props["place"]

    progress.progress((i + 1) / len(earthquakes))
    status.text(f"Analyzing {place}...")

    weather = get_weather(lat, lon)
    risk_score = calculate_risk(magnitude, weather)

    if risk_score >= 70:
        color = "red"
        risk_level = "🔴 HIGH"
    elif risk_score >= 40:
        color = "orange"
        risk_level = "🟠 MEDIUM"
    else:
        color = "blue"
        risk_level = "🔵 LOW"

    weather_info = ""
    if weather:
        temp = weather["main"]["temp"]
        wind = weather["wind"]["speed"]
        desc = weather["weather"][0]["description"]
        weather_info = f" | 🌡️ {temp}°C | 💨 {wind} m/s | {desc}"

    folium.CircleMarker(
        location=[lat, lon],
        radius=magnitude * 3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"📍 {place}\n"
            f"⚡ Magnitude: {magnitude}\n"
            f"🎯 Risk: {risk_score}/100 {risk_level}\n"
            f"{weather_info}",
            max_width=250
        )
    ).add_to(earthquake_layer)

    risk_list.append({
        "place": place,
        "magnitude": magnitude,
        "risk_score": risk_score,
        "risk_level": risk_level
    })

progress.empty()
status.empty()

# Plot distress signals on map
for signal in signal_results:
    if signal["is_distress"]:
        folium.Marker(
            location=[signal["lat"], signal["lon"]],
            popup=folium.Popup(
                f"🚨 DISTRESS DETECTED\n"
                f"📍 {signal['location']}\n"
                f"💬 {signal['text'][:60]}...\n"
                f"🧠 AI Confidence: {signal['confidence']}%",
                max_width=250
            ),
            icon=folium.Icon(color="red", icon="exclamation-sign")
        ).add_to(distress_layer)
    else:
        folium.Marker(
            location=[signal["lat"], signal["lon"]],
            popup=folium.Popup(
                f"✅ NO DISTRESS\n"
                f"📍 {signal['location']}\n"
                f"💬 {signal['text'][:60]}...\n"
                f"🧠 AI Confidence: {signal['confidence']}%",
                max_width=250
            ),
            icon=folium.Icon(color="green", icon="ok-sign")
        ).add_to(distress_layer)

# Add layers to map
earthquake_layer.add_to(m)
distress_layer.add_to(m)
folium.LayerControl().add_to(m)

# ─────────────────────────────────────────
# ALERT BANNERS
# ─────────────────────────────────────────
high_risk = [r for r in risk_list if r["risk_score"] >= 70]

if high_risk or distress_count > 0:
    st.error(f"🚨 ALERT: {len(high_risk)} high risk zone(s) | {distress_count} distress signal(s) detected!")
    for zone in high_risk[:3]:
        st.error(f"📍 {zone['place']} — Magnitude {zone['magnitude']} — Risk: {zone['risk_score']}/100")
else:
    st.success("✅ No high risk zones detected currently.")

# Distress signal alerts
if distress_count > 0:
    st.warning(f"🧠 AI detected {distress_count} distress signals from social media:")
    for s in signal_results:
        if s["is_distress"]:
            st.warning(f"📍 {s['location']} — {s['confidence']}% confidence — \"{s['text'][:70]}...\"")

# ─────────────────────────────────────────
# MAP
# ─────────────────────────────────────────
st.subheader("🗺️ Live Risk Map")
st_folium(m, width=1200, height=550)

# ─────────────────────────────────────────
# SIDEBAR TOP 5
# ─────────────────────────────────────────
st.sidebar.subheader("🔥 Top 5 Danger Zones")
top5 = sorted(risk_list, key=lambda x: x["risk_score"], reverse=True)[:5]
for i, zone in enumerate(top5):
    st.sidebar.markdown(f"""
**#{i+1} {zone['place']}**
- Magnitude: {zone['magnitude']}
- Risk: {zone['risk_level']} ({zone['risk_score']}/100)
---
""")

st.sidebar.subheader("🚨 Distress Signals")
for s in signal_results:
    if s["is_distress"]:
        st.sidebar.markdown(f"""
**📍 {s['location']}**
- Confidence: {s['confidence']}%
- "{s['text'][:40]}..."
---
""")