import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Page config
st.set_page_config(page_title="DisasterSenseAI", page_icon="🚨", layout="wide")
st.title("🚨 DisasterSenseAI")
st.subheader("Real-Time Disaster Risk Dashboard")

# ─────────────────────────────────────────
# FETCH EARTHQUAKE DATA
# ─────────────────────────────────────────
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
    response = requests.get(url)
    return response.json()["features"]

# ─────────────────────────────────────────
# FETCH WEATHER DATA FOR A LOCATION
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
def calculate_risk(magnitude, weather):
    risk = 0

    # Earthquake contribution
    if magnitude >= 6:
        risk += 60
    elif magnitude >= 5:
        risk += 40
    elif magnitude >= 4:
        risk += 20
    else:
        risk += 10

    # Weather contribution
    if weather:
        wind_speed = weather["wind"]["speed"]
        weather_id = weather["weather"][0]["id"]

        # High wind
        if wind_speed > 20:
            risk += 20
        elif wind_speed > 10:
            risk += 10

        # Storms, heavy rain, extreme weather
        if weather_id < 300:       # Thunderstorm
            risk += 20
        elif weather_id < 600:     # Rain
            risk += 10
        elif weather_id < 700:     # Snow
            risk += 10

    return min(risk, 100)  # Cap at 100

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
earthquakes = get_earthquakes()

# Sidebar
st.sidebar.title("📊 Risk Summary")
st.sidebar.write(f"Total earthquakes detected: **{len(earthquakes)}**")

# Create map
m = folium.Map(location=[20, 0], zoom_start=2)

# Store risk data for sidebar
risk_list = []

# Progress bar while loading
progress = st.progress(0)
status = st.empty()

for i, quake in enumerate(earthquakes):
    coords = quake["geometry"]["coordinates"]
    props = quake["properties"]

    lon = coords[0]
    lat = coords[1]
    magnitude = props["mag"]
    place = props["place"]

    # Update progress
    progress.progress((i + 1) / len(earthquakes))
    status.text(f"Analyzing {place}...")

    # Get weather at earthquake location
    weather = get_weather(lat, lon)

    # Calculate combined risk score
    risk_score = calculate_risk(magnitude, weather)

    # Color based on risk
    if risk_score >= 70:
        color = "red"
        risk_level = "🔴 HIGH"
    elif risk_score >= 40:
        color = "orange"
        risk_level = "🟠 MEDIUM"
    else:
        color = "blue"
        risk_level = "🔵 LOW"

    # Weather info for popup
    weather_info = ""
    if weather:
        temp = weather["main"]["temp"]
        wind = weather["wind"]["speed"]
        desc = weather["weather"][0]["description"]
        weather_info = f"\n🌡️ Temp: {temp}°C\n💨 Wind: {wind} m/s\n☁️ {desc}"

    # Add marker to map
    folium.CircleMarker(
        location=[lat, lon],
        radius=magnitude * 3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"📍 {place}\n"
            f"⚡ Magnitude: {magnitude}\n"
            f"🎯 Risk Score: {risk_score}/100\n"
            f"{risk_level}"
            f"{weather_info}",
            max_width=250
        )
    ).add_to(m)

    # Store for sidebar
    risk_list.append({
        "place": place,
        "magnitude": magnitude,
        "risk_score": risk_score,
        "risk_level": risk_level
    })

# Clear progress bar
progress.empty()
status.empty()

# ─────────────────────────────────────────
# ALERT BANNER — High risk zones
# ─────────────────────────────────────────
high_risk = [r for r in risk_list if r["risk_score"] >= 70]

if high_risk:
    st.error(f"🚨 ALERT: {len(high_risk)} HIGH RISK zone(s) detected right now!")
    for zone in high_risk[:3]:
        st.error(f"📍 {zone['place']} — Magnitude {zone['magnitude']} — Risk Score: {zone['risk_score']}/100")
else:
    st.success("✅ No high risk zones detected currently.")

# ─────────────────────────────────────────
# MAP
# ─────────────────────────────────────────
st.subheader("🗺️ Live Risk Map")
st_folium(m, width=1200, height=550)

# ─────────────────────────────────────────
# SIDEBAR — Top 5 most dangerous zones
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