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
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10}
        headers = {'User-Agent': 'AI-RescueMap/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if 'address' in data:
            addr = data['address']
            city = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('county', 'Unknown')
            country = addr.get('country', 'Unknown')
            region = addr.get('state', addr.get('region', 'Unknown'))
            return city, country, region
    except:
        pass
    return "Unknown", "Unknown", "Unknown"

def geocode_location(search_text):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': search_text, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'AI-RescueMap/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if data:
            result = data[0]
            lat, lon = float(result['lat']), float(result['lon'])
            city, country, region = reverse_geocode(lat, lon)
            return {
                'lat': lat, 'lon': lon, 'city': city, 'country': country,
                'region': region, 'method': 'manual', 'source': 'Manual Entry'
            }
    except Exception as e:
        st.error(f"Search failed: {e}")
    return None

def get_ip_location():
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
                'source': 'IP Location'
            }
    except:
        pass
    return {
        'lat': 31.3709, 'lon': 73.0336, 'city': 'Faisalabad',
        'country': 'Pakistan', 'region': 'Punjab', 'method': 'default', 'source': 'Default'
    }

def get_browser_location():
    params = st.query_params
    if "lat" in params and "lon" in params:
        try:
            lat = float(params["lat"])
            lon = float(params["lon"])
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
            return genai.GenerativeModel('gemini-pro')
        except:
            return None
    return None

def fetch_disasters():
    try:
        response = requests.get(f"{CONFIG['EONET_API']}?status=open&limit=200", timeout=10)
        response.raise_for_status()
        data = response.json()
        disasters = []
        for event in data.get('events', []):
            if event.get('geometry'):
                geo = event['geometry'][-1]
                coords = geo.get('coordinates', [])
                if len(coords) >= 2:
                    disasters.append({
                        'title': event['title'],
                        'category': event['categories'][0]['title'] if event.get('categories') else 'Unknown',
                        'lat': coords[1] if geo['type'] == 'Point' else coords[0][1],
                        'lon': coords[0] if geo['type'] == 'Point' else coords[0][0],
                        'date': geo.get('date', 'Unknown')
                    })
        return pd.DataFrame(disasters)
    except:
        return pd.DataFrame()

def add_satellite_layers(m, layers):
    date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    layer_map = {
        'True Color': 'VIIRS_SNPP_CorrectedReflectance_TrueColor',
        'Active Fires': 'VIIRS_SNPP_Fires_375m_Day'
    }
    for name, id in layer_map.items():
        if name in layers:
            url = f"{CONFIG['GIBS_BASE']}/{id}/default/{date}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
            folium.TileLayer(tiles=url, attr='NASA', name=name, overlay=True, control=True, opacity=0.7).add_to(m)
    return m

def generate_population(lat, lon, r=2.0, n=1000):
    np.random.seed(int((lat + lon) * 10000) % 2**32)
    centers = [(lat + np.random.uniform(-r*0.7, r*0.7), lon + np.random.uniform(-r*0.7, r*0.7), np.random.uniform(5000, 20000)) for _ in range(np.random.randint(2, 5))]
    lats, lons, pops = [], [], []
    for _ in range(n):
        angle, rad = np.random.uniform(0, 2*np.pi), np.random.uniform(0, r)
        lt, ln = lat + rad * np.cos(angle), lon + rad * np.sin(angle)
        dist = min([np.sqrt((lt-c[0])**2 + (ln-c[1])**2) for c in centers])
        pop = max(centers, key=lambda c: c[2])[2] * np.exp(-dist * 2) * np.random.uniform(0.5, 1.5)
        lats.append(lt)
        lons.append(ln)
        pops.append(max(0, pop))
    return pd.DataFrame({'lat': lats, 'lon': lons, 'population': pops})

def calc_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def get_ai_guidance(disaster_type, situation, location, model):
    if not model:
        return "AI unavailable. Call emergency: 911/112/1122"
    try:
        loc_str = f"{location['city']}, {location['country']}" if location else "Unknown"
        prompt = f"Emergency at {loc_str}. {disaster_type}. Situation: {situation}\n\nProvide: ACTIONS (3-5), DON'Ts (3), EVACUATION CRITERIA, ITEMS, URGENCY, LOCAL EMERGENCY NUMBERS"
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {e}\n\nCall emergency services immediately"

def analyze_image(image, model):
    if not model:
        return {'success': False, 'message': 'Add API key'}
    try:
        prompt = "Analyze: TYPE, SEVERITY (LOW/MODERATE/HIGH/CRITICAL), DAMAGES, AREA, RISK, ACTIONS"
        response = model.generate_content([prompt, image])
        severity = 50
        for level, score in {'LOW': 25, 'MODERATE': 50, 'HIGH': 75, 'CRITICAL': 95}.items():
            if level in response.text.upper():
                severity = score
                break
        return {
            'success': True,
            'analysis': response.text,
            'severity_score': severity,
            'severity_level': 'CRITICAL' if severity > 80 else 'HIGH' if severity > 60 else 'MODERATE'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}

# Initialize session state
if 'location' not in st.session_state:
    st.session_state.location = None
if 'gemini' not in st.session_state:
    st.session_state.gemini = None

# Detect location: Browser > Manual > IP
browser_loc = get_browser_location()
if browser_loc:
    st.session_state.location = browser_loc
elif 'manual_loc' in st.session_state and st.session_state.manual_loc:
    st.session_state.location = st.session_state.manual_loc
else:
    if st.session_state.location is None:
        st.session_state.location = get_ip_location()

loc = st.session_state.location

# Sidebar
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## AI-RescueMap")
    st.markdown("---")
    menu = st.radio("", ["Map", "AI Help", "Image", "Stats"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Your Location")
    
    if loc:
        badge = {"browser": "GPS", "manual": "Manual", "ip": "IP", "default": "Default"}
        st.success(f"**{loc['city']}**")
        st.info(f"{loc['region']}, {loc['country']}")
        st.caption(f"{badge.get(loc['method'], '')} | {loc['lat']:.4f}, {loc['lon']:.4f}")
    
    st.markdown("---")
    
    # Browser location button with working JavaScript
    if st.button("Get Browser Location", use_container_width=True):
        st.components.v1.html("""
        <script>
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set('lat', pos.coords.latitude);
                    url.searchParams.set('lon', pos.coords.longitude);
                    window.parent.location.href = url.toString();
                },
                function(err) {
                    alert('Location access denied or failed');
                },
                {enableHighAccuracy: true, timeout: 10000, maximumAge: 0}
            );
        } else {
            alert('Geolocation not supported by your browser');
        }
        </script>
        """, height=0)
    
    # Manual location
    st.markdown("---")
    with st.expander("Manual Location"):
        loc_input = st.text_input("City, Country", placeholder="Tokyo, Japan")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Find", disabled=not loc_input, use_container_width=True):
                with st.spinner("Searching..."):
                    result = geocode_location(loc_input)
                    if result:
                        st.session_state.manual_loc = result
                        st.session_state.location = result
                        st.success("Found!")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("Not found")
        with col2:
            if st.button("Reset", use_container_width=True):
                if 'manual_loc' in st.session_state:
                    del st.session_state.manual_loc
                st.session_state.location = get_ip_location()
                st.success("Reset")
                time.sleep(0.3)
                st.rerun()

st.markdown('<h1 class="main-header">AI-RescueMap</h1>', unsafe_allow_html=True)

# Setup Gemini
key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if key and not st.session_state.gemini:
    st.session_state.gemini = setup_gemini(key)

# MAP
if menu == "Map":
    disasters = fetch_disasters()
    
    # Calculate distances from user location
    if loc and not disasters.empty:
        disasters['dist'] = disasters.apply(lambda r: calc_distance(loc['lat'], loc['lon'], r['lat'], r['lon']), axis=1)
        disasters = disasters.sort_values('dist')
    
    # Show filtered counts
    total = len(disasters)
    nearby = len(disasters[disasters['dist'] < 500]) if 'dist' in disasters.columns else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Disasters", total)
    col2.metric("Nearby (<500km)", nearby)
    col3.metric("AI", "ON" if st.session_state.gemini else "OFF")
    
    st.markdown("---")
    col_set, col_map = st.columns([1, 3])
    
    with col_set:
        center_opt = st.selectbox("Center:", ["My Location", "Global"])
        if center_opt == "My Location" and loc:
            clat, clon, zoom = loc['lat'], loc['lon'], 8
        else:
            clat, clon, zoom = 20, 0, 2
        
        show_pop = st.checkbox("Population", True)
        layers = st.multiselect("Satellite", ['True Color', 'Active Fires'], ['True Color'])
        radius = st.slider("Impact (km)", 10, 200, 50)
    
    with col_map:
        m = folium.Map([clat, clon], zoom_start=zoom, tiles='CartoDB positron')
        if layers:
            m = add_satellite_layers(m, layers)
        if show_pop:
            pop = generate_population(clat, clon, 3, 1500)
            HeatMap([[r['lat'], r['lon'], r['population']] for _, r in pop.iterrows()], radius=15, blur=25).add_to(m)
        if not disasters.empty:
            colors = {'Wildfires': 'red', 'Severe Storms': 'orange', 'Floods': 'blue', 'Earthquakes': 'darkred'}
            for _, d in disasters.iterrows():
                dist_txt = f"<br>{d['dist']:.0f}km away" if 'dist' in d else ""
                folium.Marker([d['lat'], d['lon']], popup=f"<b>{d['title']}</b><br>{d['category']}{dist_txt}", 
                             icon=folium.Icon(color=colors.get(d['category'], 'gray'))).add_to(m)
        if loc:
            folium.Marker([loc['lat'], loc['lon']], popup=f"<b>You</b><br>{loc['city']}, {loc['country']}", 
                         icon=folium.Icon(color='green', icon='home', prefix='glyphicon')).add_to(m)
        st_folium(m, width=1000, height=600)

# AI HELP
elif menu == "AI Help":
    st.markdown("## AI Guidance")
    if loc:
        st.info(f"Location: {loc['city']}, {loc['country']}")
    dtype = st.selectbox("Disaster:", ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tornado"])
    situation = st.text_area("Situation:", height=120)
    if st.button("GET GUIDANCE", type="primary"):
        if situation and st.session_state.gemini:
            with st.spinner("Analyzing..."):
                result = get_ai_guidance(dtype, situation, loc, st.session_state.gemini)
                st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
        elif not situation:
            st.error("Describe situation")
        else:
            st.warning("Add API key")

# IMAGE
elif menu == "Image":
    from PIL import Image as PILImage
    st.markdown("## Image Analysis")
    file = st.file_uploader("Upload image", type=['jpg', 'png'])
    if file:
        img = PILImage.open(file)
        st.image(img, width=600)
        if st.button("ANALYZE", type="primary"):
            if st.session_state.gemini:
                with st.spinner("Analyzing..."):
                    result = analyze_image(img, st.session_state.gemini)
                    if result['success']:
                        st.metric("Severity", result['severity_level'])
                        st.markdown(f'<div class="ai-response">{result["analysis"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(result['message'])
            else:
                st.warning("Add API key")

# STATS
elif menu == "Stats":
    st.markdown("## Analytics")
    disasters = fetch_disasters()
    
    if loc and not disasters.empty:
        disasters['dist'] = disasters.apply(lambda r: calc_distance(loc['lat'], loc['lon'], r['lat'], r['lon']), axis=1)
        view = st.radio("View:", ["Local", "Global"], horizontal=True)
        if view == "Local":
            rad = st.slider("Radius (km):", 100, 5000, 1000, 100)
            disasters = disasters[disasters['dist'] <= rad]
            st.success(f"{len(disasters)} disasters within {rad}km")
    
    if not disasters.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(disasters))
        col2.metric("Wildfires", len(disasters[disasters['category'] == 'Wildfires']))
        col3.metric("Storms", len(disasters[disasters['category'] == 'Severe Storms']))
        st.bar_chart(disasters['category'].value_counts())
        
        cols = ['title', 'category', 'date']
        if 'dist' in disasters.columns:
            disasters['dist'] = disasters['dist'].round(0).astype(int)
            cols.append('dist')
        st.dataframe(disasters[cols].head(20), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray'>Built by HasnainAtif for NASA Space Apps 2025</p>", unsafe_allow_html=True)
