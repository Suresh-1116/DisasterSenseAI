import streamlit as st
import folium
import requests
from streamlit_folium import st_folium

# Page title
st.title("🚨 DisasterSenseAI")
st.subheader("Live Earthquake Monitor")

# Fetch real earthquake data from USGS (updated every minute)
st.write("Fetching live earthquake data...")

url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
response = requests.get(url)
data = response.json()

earthquakes = data["features"]
st.write(f"Found {len(earthquakes)} earthquakes in the last 24 hours")

# Create a map centered on the world
m = folium.Map(location=[20, 0], zoom_start=2)

# Plot each earthquake on the map
for quake in earthquakes:
    coords = quake["geometry"]["coordinates"]
    props = quake["properties"]
    
    lon = coords[0]
    lat = coords[1]
    magnitude = props["mag"]
    place = props["place"]
    
    # Bigger magnitude = bigger circle, red if dangerous
    color = "red" if magnitude >= 5 else "orange" if magnitude >= 4 else "blue"
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=magnitude * 3,
        color=color,
        fill=True,
        popup=f"📍 {place}\nMagnitude: {magnitude}"
    ).add_to(m)

# Show the map
st_folium(m, width=700, height=500)