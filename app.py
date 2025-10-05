import os
import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
from datetime import datetime, timedelta
import rasterio
from rasterio.windows import from_bounds
from io import BytesIO

os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(
    page_title="AI-RescueMap | NASA Space Apps 2025",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .disaster-alert {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ai-response {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

CONFIG = {
    "EONET_API": "https://eonet.gsfc.nasa.gov/api/v3/events",
    "GIBS_BASE": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best",
    "IPAPI_URL": "https://ipapi.co/json/",
    "WORLDPOP_URL": "https://huggingface.co/datasets/HasnainAtif/worldpop_2024/resolve/main/global_pop_2024_CN_1km_R2025A_UA_v1.tif"
}

def setup_gemini(api_key: str = None):
    if not GEMINI_AVAILABLE:
        return None
    key = api_key or st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if key:
        try:
            genai.configure(api_key=key)
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            return None
    return None

@st.cache_data(ttl=3600)
def get_user_location():
    try:
        response = requests.get(CONFIG["IPAPI_URL"], timeout=5)
        data = response.json()
        return {
            'lat': float(data['latitude']),
            'lon': float(data['longitude']),
            'city': data.get('city', 'Unknown'),
            'country': data.get('country_name', 'Unknown')
        }
    except:
        return {'lat': 31.3709, 'lon': 73.0336, 'city': 'Faisalabad', 'country': 'Pakistan'}

@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=50):
    try:
        url = f"{CONFIG['EONET_API']}?status={status}&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        disasters = []
        for event in data.get('events', []):
            if event.get('geometry'):
                latest_geo = event['geometry'][-1]
                coords = latest_geo.get('coordinates', [])
                if len(coords) >= 2:
                    disasters.append({
                        'id': event['id'],
                        'title': event['title'],
                        'category': event['categories'][0]['title'] if event.get('categories') else 'Unknown',
                        'lat': coords[1] if latest_geo['type'] == 'Point' else coords[0][1],
                        'lon': coords[0] if latest_geo['type'] == 'Point' else coords[0][0],
                        'date': event.get('geometry')[-1].get('date', 'Unknown'),
                        'source': ', '.join([s['id'] for s in event.get('sources', [])]),
                        'link': event.get('link', '')
                    })
        return pd.DataFrame(disasters)
    except Exception as e:
        return pd.DataFrame()

def add_nasa_satellite_layers(folium_map, selected_layers):
    date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    layers_config = {
        'True Color': 'VIIRS_SNPP_CorrectedReflectance_TrueColor',
        'Active Fires': 'VIIRS_SNPP_Fires_375m_Day',
        'Night Lights': 'VIIRS_SNPP_DayNightBand_ENCC',
        'Water Vapor': 'AIRS_L2_Surface_Relative_Humidity_Day'
    }
    for layer_name, layer_id in layers_config.items():
        if layer_name in selected_layers:
            tile_url = f"{CONFIG['GIBS_BASE']}/{layer_id}/default/{date_str}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
            folium.TileLayer(tiles=tile_url, attr='NASA GIBS', name=layer_name, overlay=True, control=True, opacity=0.7).add_to(folium_map)
    return folium_map

@st.cache_data(ttl=7200)
def fetch_worldpop_data(center_lat, center_lon, radius_deg=2.0):
    try:
        response = requests.get(CONFIG["WORLDPOP_URL"], stream=True, timeout=30)
        response.raise_for_status()
        with rasterio.open(BytesIO(response.content)) as src:
            min_lon, max_lon = center_lon - radius_deg, center_lon + radius_deg
            min_lat, max_lat = center_lat - radius_deg, center_lat + radius_deg
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
            data = src.read(1, window=window)
            height, width = data.shape
            lats = np.linspace(max_lat, min_lat, height)
            lons = np.linspace(min_lon, max_lon, width)
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
            mask = (data > 0) & (data < 1e10)
            sample_rate = max(1, len(mask.flatten()) // 2000)
            pop_data = []
            for i in range(0, height, sample_rate):
                for j in range(0, width, sample_rate):
                    if mask[i, j]:
                        pop_data.append({'lat': lat_grid[i, j], 'lon': lon_grid[i, j], 'population': float(data[i, j])})
            return pd.DataFrame(pop_data)
    except:
        return pd.DataFrame()

def calculate_disaster_impact(disaster_df, population_df, radius_km=50):
    if disaster_df.empty or population_df.empty:
        return []
    impacts = []
    for _, disaster in disaster_df.iterrows():
        pop_df = population_df.copy()
        pop_df['dist_km'] = np.sqrt(((pop_df['lat'] - disaster['lat']) * 111)**2 + ((pop_df['lon'] - disaster['lon']) * 111 * np.cos(np.radians(disaster['lat'])))**2)
        affected = pop_df[pop_df['dist_km'] <= radius_km]
        impacts.append({
            'disaster': disaster['title'],
            'category': disaster['category'],
            'affected_population': int(affected['population'].sum()),
            'affected_area_km2': int(np.pi * radius_km**2),
            'risk_level': 'CRITICAL' if affected['population'].sum() > 100000 else 'HIGH' if affected['population'].sum() > 10000 else 'MODERATE'
        })
    return impacts

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, model) -> str:
    if not model:
        return "⚠️ AI unavailable. Add Gemini API key in sidebar.\n\n🚨 Emergency: 911 | 🆘 FEMA: 1-800-621-3362"
    try:
        prompt = f"""Expert emergency advisor. Person needs help:
Disaster: {disaster_type}
Situation: {user_situation}

Provide concise guidance:
🚨 IMMEDIATE ACTIONS (3-5 steps)
⚠️ CRITICAL DON'Ts (3-4 items)
🏃 WHEN TO EVACUATE
📦 ESSENTIAL ITEMS
⏰ TIMELINE
📞 WHO TO CALL"""
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}\n\n🚨 Call 911 immediately\n🆘 FEMA: 1-800-621-3362"

def analyze_disaster_image(image, model):
    if not model:
        return {'success': False, 'message': 'Add Gemini API key to enable'}
    try:
        prompt = """Analyze disaster image. Provide:
DISASTER TYPE | SEVERITY (LOW/MODERATE/HIGH/CRITICAL) | VISIBLE DAMAGES | AFFECTED AREA | POPULATION RISK | RESPONSE RECOMMENDATIONS | RECOVERY TIME"""
        response = model.generate_content([prompt, image])
        severity_map = {'LOW': 25, 'MODERATE': 50, 'HIGH': 75, 'CRITICAL': 95}
        severity_score = 50
        for level, score in severity_map.items():
            if level in response.text.upper():
                severity_score = score
                break
        return {'success': True, 'analysis': response.text, 'severity_score': severity_score, 'severity_level': 'CRITICAL' if severity_score > 80 else 'HIGH' if severity_score > 60 else 'MODERATE'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

if 'location' not in st.session_state:
    st.session_state.location = get_user_location()
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## AI-RescueMap")
    st.markdown("**NASA Space Apps 2025**")
    st.markdown("---")
    
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Get free: makersuite.google.com/app/apikey")
    if gemini_api_key and st.session_state.gemini_model is None:
        st.session_state.gemini_model = setup_gemini(gemini_api_key)
        if st.session_state.gemini_model:
            st.success("✅ AI Ready")
    
    st.markdown("---")
    menu = st.radio("", ["🗺 Map", "💬 AI Help", "🖼 Image", "📊 Stats"], label_visibility="collapsed")
    st.markdown("---")
    loc = st.session_state.location
    st.info(f"📍 {loc['city']}, {loc['country']}")

st.markdown('<h1 class="main-header">AI-RescueMap</h1>', unsafe_allow_html=True)

if menu == "🗺 Map":
    disasters = fetch_nasa_eonet_disasters()
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Disasters", len(disasters))
    col2.metric("AI Status", "✅" if st.session_state.gemini_model else "⚠️")
    col3.metric("Data", "NASA Live")
    
    st.markdown("---")
    col_set, col_map = st.columns([1, 4])
    
    with col_set:
        center_option = st.selectbox("Center", ["My Location", "Global"] + (disasters['title'].tolist() if not disasters.empty else []))
        if center_option == "My Location":
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
        elif center_option == "Global":
            center_lat, center_lon, zoom = 20, 0, 2
        else:
            row = disasters[disasters['title'] == center_option].iloc[0]
            center_lat, center_lon, zoom = row['lat'], row['lon'], 8
        
        show_pop = st.checkbox("Population", True)
        layers = st.multiselect("Satellite", ['True Color', 'Active Fires'], ['True Color'])
        radius = st.slider("Radius (km)", 10, 200, 50)
    
    with col_map:
        m = folium.Map([center_lat, center_lon], zoom_start=zoom, tiles='CartoDB positron')
        if layers:
            m = add_nasa_satellite_layers(m, layers)
        if show_pop:
            pop_df = fetch_worldpop_data(center_lat, center_lon, 3)
            if not pop_df.empty:
                HeatMap([[r['lat'], r['lon'], r['population']] for _, r in pop_df.iterrows()], radius=15, blur=25).add_to(m)
        if not disasters.empty:
            for _, d in disasters.iterrows():
                folium.Marker([d['lat'], d['lon']], popup=d['title'], icon=folium.Icon(color='red')).add_to(m)
        st_folium(m, width=1100, height=600)

elif menu == "💬 AI Help":
    st.markdown("## AI Emergency Guidance")
    disaster_type = st.selectbox("Disaster Type", ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tornado"])
    situation = st.text_area("Describe situation:", height=120)
    if st.button("🚨 GET GUIDANCE", type="primary"):
        if situation and st.session_state.gemini_model:
            with st.spinner("Analyzing..."):
                guidance = get_ai_disaster_guidance(disaster_type, situation, st.session_state.gemini_model)
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)
        elif not situation:
            st.error("Describe your situation")
        else:
            st.warning("Add API key in sidebar")

elif menu == "🖼 Image":
    from PIL import Image
    st.markdown("## AI Image Analysis")
    uploaded = st.file_uploader("Upload disaster image", type=['jpg', 'png'])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=600)
        if st.button("🔍 ANALYZE", type="primary"):
            if st.session_state.gemini_model:
                with st.spinner("Analyzing..."):
                    result = analyze_disaster_image(img, st.session_state.gemini_model)
                    if result['success']:
                        st.metric("Severity", result['severity_level'])
                        st.markdown(f'<div class="ai-response">{result["analysis"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(result['message'])
            else:
                st.warning("Add API key")

elif menu == "📊 Stats":
    st.markdown("## Global Analytics")
    disasters = fetch_nasa_eonet_disasters(limit=100)
    if not disasters.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(disasters))
        col2.metric("Wildfires", len(disasters[disasters['category'] == 'Wildfires']))
        col3.metric("Storms", len(disasters[disasters['category'] == 'Severe Storms']))
        st.bar_chart(disasters['category'].value_counts())
        st.dataframe(disasters[['title', 'category', 'date']], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray'>Built by hasnainatif for NASA Space Apps 2025 | Powered by NASA & Gemini AI</p>", unsafe_allow_html=True)
