import os
import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
from datetime import datetime, timedelta
import time

os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="AI-RescueMap | NASA Space Apps 2025", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding: 1rem 0; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem; }
    .disaster-alert { background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .ai-response { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

CONFIG = {
    "EONET_API": "https://eonet.gsfc.nasa.gov/api/v3/events",
    "GIBS_BASE": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best",
    "IPAPI_URL": "https://ipapi.co/json/",
    "IPAPI_BACKUP": "http://ip-api.com/json/",
    "GEOCODING_API": "https://nominatim.openstreetmap.org/search",
    "REVERSE_GEOCODING_API": "https://nominatim.openstreetmap.org/reverse",
    "GEOCODING_BACKUP": "https://geocode.maps.co/search",
}

EMERGENCY_CONTACTS = {
    "Pakistan": {"emergency": "112 / 1122", "police": "15", "ambulance": "1122", "fire": "16"},
    "United States": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "United Kingdom": {"emergency": "999 / 112", "police": "999", "ambulance": "999", "fire": "999"},
    "India": {"emergency": "112", "police": "100", "ambulance": "102", "fire": "101"},
    "Australia": {"emergency": "000", "police": "000", "ambulance": "000", "fire": "000"},
    "Canada": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "Default": {"emergency": "112 (International)", "police": "Local Police", "ambulance": "Local Ambulance", "fire": "Local Fire"}
}

def get_emergency_contacts(country: str) -> dict:
    return EMERGENCY_CONTACTS.get(country, EMERGENCY_CONTACTS["Default"])

def geocode_location(city_or_address: str, max_retries=2):
    for attempt in range(max_retries):
        try:
            params = {'q': city_or_address, 'format': 'json', 'limit': 1, 'addressdetails': 1}
            headers = {'User-Agent': 'AI-RescueMap/1.0'}
            response = requests.get(CONFIG["GEOCODING_API"], params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    address = result.get('address', {})
                    return {'lat': float(result['lat']), 'lon': float(result['lon']), 'city': address.get('city') or address.get('town') or 'Unknown', 'country': address.get('country', 'Unknown'), 'region': address.get('state', 'Unknown'), 'full_address': result.get('display_name', city_or_address), 'method': 'manual', 'source': 'OpenStreetMap'}
        except:
            pass
    return None

def reverse_geocode(lat: float, lon: float):
    try:
        params = {'lat': lat, 'lon': lon, 'format': 'json'}
        response = requests.get(CONFIG["REVERSE_GEOCODING_API"], params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            return {'lat': lat, 'lon': lon, 'city': address.get('city', 'Unknown'), 'country': address.get('country', 'Unknown'), 'region': address.get('state', 'Unknown'), 'full_address': data.get('display_name', ''), 'method': 'browser', 'source': 'GPS'}
    except:
        pass
    return {'lat': lat, 'lon': lon, 'city': 'Unknown', 'country': 'Unknown', 'region': 'Unknown', 'method': 'browser', 'source': 'GPS'}

def get_ip_location():
    try:
        response = requests.get(CONFIG["IPAPI_URL"], timeout=5)
        data = response.json()
        if 'latitude' in data:
            return {'lat': float(data['latitude']), 'lon': float(data['longitude']), 'city': data.get('city', 'Unknown'), 'country': data.get('country_name', 'Unknown'), 'region': data.get('region', 'Unknown'), 'method': 'ip', 'source': 'IP'}
    except:
        pass
    return {'lat': 20.0, 'lon': 0.0, 'city': 'Unknown', 'country': 'Unknown', 'region': 'Unknown', 'method': 'default', 'source': 'Default'}

def get_current_location():
    return st.session_state.get('browser_location') or st.session_state.get('manual_location') or st.session_state.get('ip_location') or get_ip_location()

def setup_gemini(api_key: str = None):
    if not GEMINI_AVAILABLE:
        return None
    key = api_key or st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if key:
        try:
            genai.configure(api_key=key)
            return genai.GenerativeModel("gemini-2.0-flash-exp")
        except:
            return None
    return None

@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=500):
    try:
        response = requests.get(f"{CONFIG['EONET_API']}?status={status}&limit={limit}", timeout=15)
        data = response.json()
        disasters = []
        for event in data.get('events', []):
            if event.get('geometry'):
                coords = event['geometry'][-1].get('coordinates', [])
                if len(coords) >= 2:
                    disasters.append({'id': event['id'], 'title': event['title'], 'category': event['categories'][0]['title'] if event.get('categories') else 'Unknown', 'lat': coords[1] if event['geometry'][-1]['type'] == 'Point' else coords[0][0][1], 'lon': coords[0] if event['geometry'][-1]['type'] == 'Point' else coords[0][0][0], 'date': event['geometry'][-1].get('date', 'Unknown')})
        return pd.DataFrame(disasters)
    except:
        return pd.DataFrame()

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * 2 * asin(sqrt(sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2))

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, model, use_location: bool = False, location: dict = None) -> str:
    if not model:
        return "⚠️ AI Not Available"
    try:
        location_context = ""
        emergency_numbers = ""
        if use_location and location:
            location_context = f"\n\nUSER LOCATION: {location['city']}, {location['country']}"
            contacts = get_emergency_contacts(location['country'])
            emergency_numbers = f"\n\n📞 EMERGENCY CONTACTS FOR {location['country'].upper()}:\n🚨 Emergency: {contacts['emergency']}\n👮 Police: {contacts['police']}\n🚑 Ambulance: {contacts['ambulance']}\n🚒 Fire: {contacts['fire']}"
        
        prompt = f"""You are a certified emergency response expert trained in ALL emergency types: natural disasters, medical crises, animal encounters, building emergencies, chemical hazards, weather emergencies, and life-threatening situations.

EMERGENCY TYPE: {disaster_type}
SITUATION: {user_situation}{location_context}

ANALYZE this specific emergency and provide customized guidance. Consider:
- Environmental factors (urban/rural, weather, terrain)
- Vulnerable populations (children, elderly, disabled, pets)
- Available resources and time constraints
- Cultural and regional emergency protocols

Provide LIFE-SAVING guidance:

🚨 IMMEDIATE ACTIONS (Next 60 seconds):
[3-5 sequential steps specific to THIS emergency - be precise]

⚠️ CRITICAL DON'Ts (Could be FATAL):
[3-5 dangerous actions to AVOID in THIS situation - explain WHY]

🏃 EVACUATION DECISION:
[When to LEAVE NOW vs SHELTER IN PLACE for this specific emergency]
[Include evacuation routes/methods if relevant]

🏥 MEDICAL/SAFETY CONCERNS:
[Injuries, contamination risks, exposure dangers specific to this emergency]

📦 PRIORITY SUPPLIES (if 60 seconds available):
[Top 5 items for THIS emergency, ranked by importance]

⏰ TIME SENSITIVITY:
[Immediate/Urgent/Monitored - explain what changes if worsens]

🔄 NEXT STEPS (After immediate danger):
[Actions for next 1-6 hours]

Be SPECIFIC to the emergency type. Wildfire ≠ flood ≠ animal attack. Use clear, directive language."""

        response = model.generate_content(prompt)
        return response.text + emergency_numbers
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}\n\n1. 🚨 Call emergency services\n2. 🏃 Follow evacuation orders\n3. 📻 Monitor local news"

def analyze_disaster_image(image, model, max_retries=2) -> dict:
    if not model:
        return {'success': False, 'message': 'AI unavailable'}
    
    prompt = """You are an expert emergency analyst trained in disaster assessment, structural engineering, environmental hazards, medical triage, and crisis response.

ANALYZE this image for ANY emergency type:
- Natural disasters (floods, fires, earthquakes, storms, landslides)
- Structural emergencies (collapse, infrastructure damage)
- Environmental hazards (spills, leaks, contamination)
- Medical/humanitarian crises (injuries, unsafe conditions)
- Animal encounters (dangerous wildlife)
- Weather emergencies (extreme conditions, damage)
- Urban emergencies (accidents, explosions)

IDENTIFY:
1. ALL visible hazards (obvious + hidden)
2. Immediate vs secondary dangers
3. Terrain, weather, environmental factors
4. Vulnerable populations
5. Usable resources/escape routes
6. Ongoing dangers (spreading, worsening)

FORMAT:

**EMERGENCY TYPE:** [Be specific, not generic]

**SEVERITY:** [CRITICAL/HIGH/MODERATE/LOW]
[Explain reasoning - what makes this life-threatening?]

**VISIBLE HAZARDS:**
- Primary dangers: [Immediate threats]
- Secondary risks: [Developing dangers]
- Structural concerns: [Buildings, infrastructure]
- Environmental factors: [Weather, fire, water, chemicals]

**AFFECTED AREA:**
- Physical extent: [Size]
- Accessibility: [Can rescuers reach?]
- Terrain challenges: [Difficulties]

**POPULATION RISK:**
- Visible people: [Count, condition]
- Hidden victims: [Trapped, obscured]
- Evacuation difficulty: [Easy/Hard - why?]
- Vulnerable individuals: [Children, elderly, injured]

**IMMEDIATE CONCERNS (Top 3):**
1. [Most urgent]
2. [Second priority]
3. [Third concern]

**RESPONSE NEEDED:**
- Responders: [Police/Fire/Medical/Rescue - why?]
- Equipment: [Specialized gear needed]
- Timeline: [How urgent?]
- Access: [How to reach?]

**RESOURCES VISIBLE:**
[Safe zones, vehicles, supplies, escape routes]

**RECOVERY TIME:**
- Response: [Hours-days]
- Cleanup: [Days-weeks]
- Full recovery: [Weeks-years]

**CRITICAL OBSERVATIONS:**
[Hidden dangers, worsening conditions, unusual factors]

Be thorough and actionable."""

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image])
            severity_map = {'LOW': 25, 'MODERATE': 50, 'HIGH': 75, 'CRITICAL': 95}
            severity_score = 50
            for level, score in severity_map.items():
                if level in response.text.upper():
                    severity_score = score
                    break
            return {'success': True, 'analysis': response.text, 'severity_score': severity_score, 'severity_level': 'CRITICAL' if severity_score > 80 else 'HIGH' if severity_score > 60 else 'MODERATE'}
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                time.sleep(30)
                continue
            return {'success': False, 'message': f'Error: {str(e)[:200]}'}
    return {'success': False, 'message': 'Max retries exceeded'}

if 'browser_location' not in st.session_state:
    st.session_state.browser_location = None
if 'manual_location' not in st.session_state:
    st.session_state.manual_location = None
if 'ip_location' not in st.session_state:
    st.session_state.ip_location = get_ip_location()
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## AI-RescueMap")
    st.markdown("---")
    menu = st.radio("Navigation", ["🗺 Disaster Map", "💬 AI Guidance", "🖼 Image Analysis", "📊 Analytics"])
    st.markdown("---")
    st.markdown("### Your Location")
    loc = get_current_location()
    if loc:
        st.success(f"**{loc['city']}, {loc['region']}**")
        st.info(f"{loc['country']}")
        st.caption(f"{loc.get('source', 'Unknown')}")
    
    st.markdown("---")
    location_html = """<script>
function getLocation() {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition((p) => {
        const url = new URL(window.location.href);
        url.searchParams.set('gps_lat', p.coords.latitude);
        url.searchParams.set('gps_lon', p.coords.longitude);
        window.location.href = url.toString();
    });
}
</script>
<button onclick="getLocation()" style="width:100%; padding:0.5rem; background:#667eea; color:white; border:none; border-radius:5px; cursor:pointer;">📍 Get GPS Location</button>"""
    st.components.v1.html(location_html, height=80)
    
    query_params = st.query_params
    if 'gps_lat' in query_params:
        try:
            gps_lat, gps_lon = float(query_params['gps_lat']), float(query_params['gps_lon'])
            st.session_state.browser_location = reverse_geocode(gps_lat, gps_lon)
            st.query_params.clear()
            st.rerun()
        except:
            pass
    
    with st.expander("Manual Entry"):
        location_input = st.text_input("City, Country")
        if st.button("Find") and location_input:
            result = geocode_location(location_input)
            if result:
                st.session_state.manual_location = result
                st.rerun()

st.markdown('<h1 class="main-header">AI-RescueMap</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time disaster monitoring with NASA data & AI</p>', unsafe_allow_html=True)

gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key and not st.session_state.gemini_model:
    st.session_state.gemini_model = setup_gemini(gemini_api_key)

if menu == "🗺 Disaster Map":
    disasters = fetch_nasa_eonet_disasters()
    if loc and not disasters.empty:
        disasters['distance_km'] = disasters.apply(lambda r: calculate_distance(loc['lat'], loc['lon'], r['lat'], r['lon']), axis=1)
        disasters = disasters.sort_values('distance_km')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Disasters", len(disasters))
    with col2:
        st.metric("Nearby (<500km)", len(disasters[disasters['distance_km'] < 500]) if 'distance_km' in disasters.columns else 0)
    with col3:
        st.metric("AI Status", "✅" if st.session_state.gemini_model else "⚠️")
    with col4:
        st.metric("Data Source", "NASA EONET")
    
    st.markdown("---")
    m = folium.Map(location=[loc['lat'] if loc else 20, loc['lon'] if loc else 0], zoom_start=6 if loc else 2)
    
    if not disasters.empty:
        for _, d in disasters.iterrows():
            folium.Marker([d['lat'], d['lon']], popup=f"<b>{d['title']}</b><br>{d['category']}", icon=folium.Icon(color='red', icon='warning-sign')).add_to(m)
    
    if loc:
        folium.Marker([loc['lat'], loc['lon']], popup=f"<b>You are here</b><br>{loc['city']}", icon=folium.Icon(color='green', icon='home')).add_to(m)
    
    st_folium(m, width=1200, height=600)

elif menu == "💬 AI Guidance":
    st.markdown("## AI Emergency Guidance")
    use_location = st.checkbox("Use my location", value=False)
    if use_location and loc:
        st.info(f"Using: {loc['city']}, {loc['country']}")
    
    disaster_type = st.selectbox("Emergency Type", ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tornado", "Snake Bite", "Building Collapse", "Chemical Spill", "Medical Emergency", "Other"])
    user_situation = st.text_area("Describe situation:", placeholder="Be specific: What's happening? Number of people? Conditions?", height=120)
    
    if st.button("🚨 GET GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("Please describe your situation")
        elif not st.session_state.gemini_model:
            st.warning("AI unavailable")
        else:
            with st.spinner("Analyzing..."):
                guidance = get_ai_disaster_guidance(disaster_type, user_situation, st.session_state.gemini_model, use_location, loc if use_location else None)
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)

elif menu == "🖼 Image Analysis":
    from PIL import Image
    st.markdown("## AI Image Analysis")
    uploaded_file = st.file_uploader("Upload disaster image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        if st.button("🔍 ANALYZE", type="primary", use_container_width=True):
            if not st.session_state.gemini_model:
                st.warning("AI unavailable")
            else:
                with st.spinner("Analyzing..."):
                    result = analyze_disaster_image(image, st.session_state.gemini_model)
                    if result['success']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Severity", result['severity_level'])
                        with col2:
                            st.metric("Risk Score", f"{result['severity_score']}/100")
                        with col3:
                            st.metric("Status", "✅")
                        st.markdown(f'<div class="ai-response">{result["analysis"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(result.get('message'))

elif menu == "📊 Analytics":
    st.markdown("## Analytics Dashboard")
    disasters = fetch_nasa_eonet_disasters()
    if not disasters.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(disasters))
        with col2:
            st.metric("Wildfires", len(disasters[disasters['category'] == 'Wildfires']))
        with col3:
            st.metric("Floods", len(disasters[disasters['category'] == 'Floods']))
        st.bar_chart(disasters['category'].value_counts())
        st.dataframe(disasters.head(20), use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>AI-RescueMap • NASA Space Apps 2025</p>", unsafe_allow_html=True)
