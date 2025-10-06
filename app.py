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
import hashlib

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
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
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
    "IPAPI_URL": "https://ipapi.co/json/"
}

def reverse_geocode(lat, lon):
    """Convert coordinates to city/country"""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10}
        headers = {'User-Agent': 'AI-RescueMap/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if 'address' in data:
            addr = data['address']
            city = addr.get('city') or addr.get('town') or addr.get('village') or 'Unknown'
            country = addr.get('country', 'Unknown')
            region = addr.get('state', 'Unknown')
            return city, country, region
    except:
        pass
    return "Unknown", "Unknown", "Unknown"

def geocode_location(search_text):
    """Convert city name to coordinates"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': search_text, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'AI-RescueMap/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if data:
            result = data[0]
            city, country, region = reverse_geocode(float(result['lat']), float(result['lon']))
            return {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'city': city,
                'country': country,
                'region': region,
                'method': 'manual',
                'source': 'Manual Entry'
            }
    except Exception as e:
        st.error(f"Location search failed: {e}")
    return None

def get_ip_location():
    """Fallback IP-based location"""
    try:
        response = requests.get(CONFIG["IPAPI_URL"], timeout=5)
        data = response.json()
        if 'latitude' in data:
            return {
                'lat': float(data['latitude']),
                'lon': float(data['longitude']),
                'city': data.get('city', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'method': 'ip',
                'source': 'IP Location (Fallback)'
            }
    except:
        pass
    return {'lat': 31.3709, 'lon': 73.0336, 'city': 'Faisalabad', 
            'country': 'Pakistan', 'region': 'Punjab', 'method': 'default', 'source': 'Default'}

def get_browser_location():
    """Check if browser sent GPS location via query params"""
    params = st.query_params
    if "browser_lat" in params and "browser_lon" in params:
        try:
            lat = float(params["browser_lat"])
            lon = float(params["browser_lon"])
            city, country, region = reverse_geocode(lat, lon)
            return {
                'lat': lat, 'lon': lon, 'city': city, 'country': country,
                'region': region, 'method': 'browser', 'source': 'Browser GPS'
            }
        except:
            pass
    return None

def setup_gemini(api_key: str = None):
    if not GEMINI_AVAILABLE:
        return None
    key = api_key or st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if key:
        try:
            genai.configure(api_key=key)
            return genai.GenerativeModel('gemini-2.0-flash-exp')
        except:
            return None
    return None

def fetch_nasa_eonet_disasters(status="open", limit=100):
    """Fetch disasters - NO CACHING to always get fresh data"""
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
    except:
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

def generate_population_data(center_lat, center_lon, radius_deg=2.0, num_points=1000):
    np.random.seed(int((center_lat + center_lon) * 10000) % 2**32)
    num_centers = np.random.randint(2, 5)
    centers = [(center_lat + np.random.uniform(-radius_deg*0.7, radius_deg*0.7),
                center_lon + np.random.uniform(-radius_deg*0.7, radius_deg*0.7),
                np.random.uniform(5000, 20000)) for _ in range(num_centers)]
    lats, lons, populations = [], [], []
    for _ in range(num_points):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, radius_deg)
        lat = center_lat + radius * np.cos(angle)
        lon = center_lon + radius * np.sin(angle)
        min_dist = min([np.sqrt((lat-c[0])**2 + (lon-c[1])**2) for c in centers])
        pop = max(centers, key=lambda c: c[2])[2] * np.exp(-min_dist * 2) * np.random.uniform(0.5, 1.5)
        lats.append(lat)
        lons.append(lon)
        populations.append(max(0, pop))
    return pd.DataFrame({'lat': lats, 'lon': lons, 'population': populations})

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def calculate_disaster_impact(disaster_df, population_df, radius_km=50):
    if disaster_df.empty or population_df.empty:
        return []
    impacts = []
    for _, disaster in disaster_df.iterrows():
        pop_df = population_df.copy()
        pop_df['dist_km'] = pop_df.apply(lambda row: calculate_distance(row['lat'], row['lon'], disaster['lat'], disaster['lon']), axis=1)
        affected = pop_df[pop_df['dist_km'] <= radius_km]
        impacts.append({
            'disaster': disaster['title'],
            'category': disaster['category'],
            'affected_population': int(affected['population'].sum()),
            'affected_area_km2': int(np.pi * radius_km**2),
            'risk_level': 'CRITICAL' if affected['population'].sum() > 100000 else 'HIGH' if affected['population'].sum() > 10000 else 'MODERATE'
        })
    return impacts

def get_ai_disaster_guidance(disaster_type, user_situation, user_location, model):
    if not model:
        return "⚠️ AI unavailable. Add GEMINI_API_KEY.\n\n🚨 Call local emergency services immediately."
    try:
        location_info = f"{user_location['city']}, {user_location['country']}" if user_location else "Unknown"
        prompt = f"""Emergency advisor. Location: {location_info}. Disaster: {disaster_type}. Situation: {user_situation}

Provide:
🚨 IMMEDIATE ACTIONS (3-5 steps)
⚠️ DON'Ts (3-4 items)
🏃 EVACUATION CRITERIA
📦 ESSENTIAL ITEMS
⏰ URGENCY
📞 LOCAL EMERGENCY NUMBERS for {user_location['country'] if user_location else 'user location'}"""
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}\n\n🚨 Call emergency services: 911/112/1122"

def analyze_disaster_image(image, model):
    if not model:
        return {'success': False, 'message': 'Add Gemini API key'}
    try:
        prompt = "Analyze disaster image. Provide: TYPE, SEVERITY (LOW/MODERATE/HIGH/CRITICAL), DAMAGES, AREA, RISK, ACTIONS, RECOVERY TIME"
        response = model.generate_content([prompt, image])
        severity_map = {'LOW': 25, 'MODERATE': 50, 'HIGH': 75, 'CRITICAL': 95}
        severity_score = 50
        for level, score in severity_map.items():
            if level in response.text.upper():
                severity_score = score
                break
        return {
            'success': True,
            'analysis': response.text,
            'severity_score': severity_score,
            'severity_level': 'CRITICAL' if severity_score > 80 else 'HIGH' if severity_score > 60 else 'MODERATE'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}

# ========== CRITICAL: LOCATION STATE MANAGEMENT ==========
if 'location' not in st.session_state:
    st.session_state.location = None
if 'location_hash' not in st.session_state:
    st.session_state.location_hash = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

# Detect location changes and clear stale data
browser_loc = get_browser_location()
current_hash = None

if browser_loc:
    current_hash = f"browser_{browser_loc['lat']}_{browser_loc['lon']}"
    if st.session_state.location_hash != current_hash:
        st.session_state.location = browser_loc
        st.session_state.location_hash = current_hash
        if 'manual_location' in st.session_state:
            del st.session_state.manual_location
elif 'manual_location' in st.session_state and st.session_state.manual_location:
    manual_loc = st.session_state.manual_location
    current_hash = f"manual_{manual_loc['lat']}_{manual_loc['lon']}"
    if st.session_state.location_hash != current_hash:
        st.session_state.location = manual_loc
        st.session_state.location_hash = current_hash
else:
    if st.session_state.location is None or st.session_state.location.get('method') not in ['browser', 'manual']:
        ip_loc = get_ip_location()
        current_hash = f"ip_{ip_loc['lat']}_{ip_loc['lon']}"
        if st.session_state.location_hash != current_hash:
            st.session_state.location = ip_loc
            st.session_state.location_hash = current_hash

loc = st.session_state.location

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## AI-RescueMap")
    st.markdown("---")
    menu = st.radio("", ["🗺 Map", "💬 AI Help", "🖼 Image", "📊 Stats"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Your Location")
    
    if loc:
        method_badge = {"browser": "🌐 GPS", "manual": "📍 Manual", "ip": "🌐 IP", "default": "📍 Default"}
        st.success(f"**{loc['city']}**")
        st.info(f"{loc['region']}, {loc['country']}")
        st.caption(method_badge.get(loc['method'], '📍'))
        st.caption(f"{loc['lat']:.4f}, {loc['lon']:.4f}")
    
    st.markdown("---")
    
    # Browser location button
    st.components.v1.html("""
    <button onclick="getLocation()" style="width:100%;padding:10px;background:#667eea;color:white;border:none;border-radius:5px;cursor:pointer;">
        📍 Get Browser Location
    </button>
    <p id="status" style="margin-top:10px;font-size:12px;color:#666;"></p>
    <script>
    function getLocation() {
        const status = document.getElementById('status');
        if (!navigator.geolocation) {
            status.innerText = '❌ Not supported';
            return;
        }
        status.innerText = '⏳ Getting location...';
        navigator.geolocation.getCurrentPosition(
            pos => {
                const url = new URL(window.location.href);
                url.searchParams.set('browser_lat', pos.coords.latitude);
                url.searchParams.set('browser_lon', pos.coords.longitude);
                status.innerText = '✅ Reloading...';
                setTimeout(() => window.location.href = url.toString(), 500);
            },
            err => {
                status.innerText = err.code === 1 ? '❌ Permission denied' : '❌ Failed';
            },
            {enableHighAccuracy: true, timeout: 10000}
        );
    }
    </script>
    """, height=100)
    
    # Manual location
    st.markdown("---")
    with st.expander("📍 Manual Location"):
        location_input = st.text_input("City, Country", placeholder="e.g., Tokyo, Japan")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Find", disabled=not location_input, use_container_width=True):
                with st.spinner("Searching..."):
                    geocoded = geocode_location(location_input)
                    if geocoded:
                        st.session_state.manual_location = geocoded
                        st.session_state.location = geocoded
                        st.session_state.location_hash = f"manual_{geocoded['lat']}_{geocoded['lon']}"
                        st.success("✅ Found!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Not found")
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                if 'manual_location' in st.session_state:
                    del st.session_state.manual_location
                st.session_state.location = get_ip_location()
                st.session_state.location_hash = None
                st.success("✅ Reset")
                time.sleep(0.5)
                st.rerun()

st.markdown('<h1 class="main-header">AI-RescueMap</h1>', unsafe_allow_html=True)

# Setup Gemini
gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_key and not st.session_state.gemini_model:
    st.session_state.gemini_model = setup_gemini(gemini_key)

# ========== MAP ==========
if menu == "🗺 Map":
    disasters = fetch_nasa_eonet_disasters()
    
    if loc and not disasters.empty:
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), axis=1
        )
        disasters = disasters.sort_values('distance_km')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Disasters", len(disasters))
    if loc and not disasters.empty and 'distance_km' in disasters.columns:
        col2.metric("Nearby (<500km)", len(disasters[disasters['distance_km'] < 500]))
    col3.metric("AI Status", "✅" if st.session_state.gemini_model else "⚠️")
    
    st.markdown("---")
    col_set, col_map = st.columns([1, 3])
    
    with col_set:
        center_option = st.selectbox("Center:", ["My Location", "Global"] + (disasters['title'].tolist()[:5] if not disasters.empty else []))
        if center_option == "My Location" and loc:
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
        elif center_option == "Global":
            center_lat, center_lon, zoom = 20, 0, 2
        elif not disasters.empty:
            row = disasters[disasters['title'] == center_option].iloc[0]
            center_lat, center_lon, zoom = row['lat'], row['lon'], 8
        else:
            center_lat, center_lon, zoom = 0, 0, 2
        
        show_pop = st.checkbox("Population", True)
        layers = st.multiselect("Satellite", ['True Color', 'Active Fires'], ['True Color'])
        radius = st.slider("Impact (km)", 10, 200, 50)
    
    with col_map:
        m = folium.Map([center_lat, center_lon], zoom_start=zoom, tiles='CartoDB positron')
        if layers:
            m = add_nasa_satellite_layers(m, layers)
        if show_pop:
            pop_df = generate_population_data(center_lat, center_lon, 3, 1500)
            HeatMap([[r['lat'], r['lon'], r['population']] for _, r in pop_df.iterrows()], radius=15, blur=25).add_to(m)
        if not disasters.empty:
            color_map = {'Wildfires': 'red', 'Severe Storms': 'orange', 'Floods': 'blue', 'Earthquakes': 'darkred'}
            for _, d in disasters.iterrows():
                dist_text = f"<br>📍 {d['distance_km']:.0f}km away" if 'distance_km' in d else ""
                folium.Marker([d['lat'], d['lon']], popup=f"<b>{d['title']}</b><br>{d['category']}{dist_text}", 
                             icon=folium.Icon(color=color_map.get(d['category'], 'gray'))).add_to(m)
        if loc:
            folium.Marker([loc['lat'], loc['lon']], popup=f"<b>You are here</b><br>{loc['city']}, {loc['country']}", 
                         icon=folium.Icon(color='green', icon='home', prefix='glyphicon')).add_to(m)
        st_folium(m, width=1000, height=600)
    
    if show_pop and not disasters.empty and 'pop_df' in locals():
        impacts = calculate_disaster_impact(disasters, pop_df, radius)
        if impacts:
            st.markdown("### Population Impact")
            impact_df = pd.DataFrame(impacts)
            high_risk = impact_df[impact_df['risk_level'].isin(['CRITICAL', 'HIGH'])]
            if not high_risk.empty:
                for _, imp in high_risk.iterrows():
                    st.markdown(f'<div class="disaster-alert">⚠️ {imp["disaster"]}<br>👥 {imp["affected_population"]:,} at risk</div>', unsafe_allow_html=True)

# ========== AI HELP ==========
elif menu == "💬 AI Help":
    st.markdown("## AI Emergency Guidance")
    if loc:
        st.info(f"📍 Your location: {loc['city']}, {loc['country']}")
    disaster_type = st.selectbox("Disaster:", ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tornado"])
    situation = st.text_area("Describe situation:", height=120)
    if st.button("🚨 GET GUIDANCE", type="primary"):
        if situation and st.session_state.gemini_model:
            with st.spinner("Analyzing..."):
                guidance = get_ai_disaster_guidance(disaster_type, situation, loc, st.session_state.gemini_model)
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)
        elif not situation:
            st.error("Describe your situation")
        else:
            st.warning("Add API key")

# ========== IMAGE ==========
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

# ========== STATS ==========
elif menu == "📊 Stats":
    st.markdown("## Analytics")
    disasters = fetch_nasa_eonet_disasters(limit=100)
    
    if loc and not disasters.empty:
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), axis=1
        )
        view = st.radio("View:", ["📍 Local", "🌍 Global"], horizontal=True)
        if "Local" in view:
            radius = st.slider("Radius (km):", 100, 5000, 1000, 100)
            disasters = disasters[disasters['distance_km'] <= radius]
            st.success(f"{len(disasters)} disasters within {radius}km of {loc['city']}")
    
    if not disasters.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(disasters))
        col2.metric("Wildfires", len(disasters[disasters['category'] == 'Wildfires']))
        col3.metric("Storms", len(disasters[disasters['category'] == 'Severe Storms']))
        st.bar_chart(disasters['category'].value_counts())
        
        display_cols = ['title', 'category', 'date']
        if 'distance_km' in disasters.columns:
            disasters['distance_km'] = disasters['distance_km'].round(0).astype(int)
            display_cols.append('distance_km')
        st.dataframe(disasters[display_cols], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray'>Built by HasnainAtif for NASA Space Apps 2025</p>", unsafe_allow_html=True)
