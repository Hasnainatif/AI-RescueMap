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
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
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
    "GEOCODING_API": "https://nominatim.openstreetmap.org/search",
    "REVERSE_GEOCODING_API": "https://nominatim.openstreetmap.org/reverse"
}

# ✅ FIXED: Geocode city/country to lat/lon
def geocode_location(city_or_address: str):
    """Convert city name or address to lat/lon using OpenStreetMap Nominatim."""
    try:
        params = {
            'q': city_or_address,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
        
        response = requests.get(CONFIG["GEOCODING_API"], params=params, headers=headers, timeout=5)
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            display_name_parts = result.get('display_name', city_or_address).split(',')
            
            return {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'city': display_name_parts[0].strip() if len(display_name_parts) > 0 else 'Unknown',
                'country': display_name_parts[-1].strip() if len(display_name_parts) > 0 else 'Unknown',
                'region': display_name_parts[1].strip() if len(display_name_parts) > 1 else 'Unknown',
                'full_address': result.get('display_name', city_or_address),
                'method': 'manual',
                'source': 'Geocoded (Manual Entry)'
            }
        else:
            return None
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None

# ✅ NEW: Reverse geocode lat/lon to address
def reverse_geocode(lat, lon):
    """Convert lat/lon to human-readable address"""
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }
        headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
        
        response = requests.get(CONFIG["REVERSE_GEOCODING_API"], params=params, headers=headers, timeout=5)
        data = response.json()
        
        if data and 'address' in data:
            address = data['address']
            display_name = data.get('display_name', 'Unknown Location')
            display_parts = display_name.split(',')
            
            return {
                'lat': lat,
                'lon': lon,
                'city': address.get('city') or address.get('town') or address.get('village') or display_parts[0].strip(),
                'country': address.get('country', 'Unknown'),
                'region': address.get('state') or address.get('region') or (display_parts[1].strip() if len(display_parts) > 1 else 'Unknown'),
                'full_address': display_name,
                'method': 'browser',
                'source': 'Browser Geolocation (GPS/Wi-Fi)'
            }
        else:
            return {
                'lat': lat,
                'lon': lon,
                'city': f"Location ({lat:.4f}, {lon:.4f})",
                'country': 'Unknown',
                'region': 'Unknown',
                'full_address': f"{lat:.4f}, {lon:.4f}",
                'method': 'browser',
                'source': 'Browser Geolocation (GPS/Wi-Fi)'
            }
    except Exception as e:
        st.warning(f"Reverse geocoding failed: {e}")
        return {
            'lat': lat,
            'lon': lon,
            'city': f"Location ({lat:.4f}, {lon:.4f})",
            'country': 'Unknown',
            'region': 'Unknown',
            'full_address': f"{lat:.4f}, {lon:.4f}",
            'method': 'browser',
            'source': 'Browser Geolocation (GPS/Wi-Fi)'
        }

# ✅ FIXED: Correct Gemini model names
def setup_gemini(api_key: str = None, model_type: str = "text"):
    if not GEMINI_AVAILABLE:
        return None
    
    key = api_key or st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    
    if key:
        try:
            genai.configure(api_key=key)
            
            model_map = {
                "text": "gemini-2.0-flash-exp",
                "image": "gemini-2.0-flash-exp",
                "chat": "gemini-2.0-flash-exp"
            }
            
            model_name = model_map.get(model_type, "gemini-2.0-flash-exp")
            return genai.GenerativeModel(model_name)
        except Exception as e:
            st.error(f"Gemini setup error: {e}")
            return None
    return None

@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=100):
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
        st.error(f"Failed to fetch NASA EONET data: {e}")
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
    np.random.seed(42)
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
    """Calculate distance in km using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def calculate_disaster_impact(disaster_df, population_df, radius_km=50):
    if disaster_df.empty or population_df.empty:
        return []
    
    impacts = []
    for _, disaster in disaster_df.iterrows():
        pop_df = population_df.copy()
        pop_df['dist_km'] = np.sqrt(
            ((pop_df['lat'] - disaster['lat']) * 111)**2 + 
            ((pop_df['lon'] - disaster['lon']) * 111 * np.cos(np.radians(disaster['lat'])))**2
        )
        affected = pop_df[pop_df['dist_km'] <= radius_km]
        impacts.append({
            'disaster': disaster['title'],
            'category': disaster['category'],
            'affected_population': int(affected['population'].sum()),
            'affected_area_km2': int(np.pi * radius_km**2),
            'risk_level': 'CRITICAL' if affected['population'].sum() > 100000 else 
                         'HIGH' if affected['population'].sum() > 10000 else 'MODERATE'
        })
    return impacts

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, user_location: str, model) -> str:
    if not model:
        return """⚠️ **AI Not Available** - Please add your Gemini API key.

**Emergency Contacts:**
- 🚨 Emergency: 911 (US) / 1122 (Pakistan) / 112 (Europe)
- 🆘 FEMA: 1-800-621-3362
- 🔴 Red Cross: 1-800-733-2767"""
    
    try:
        prompt = f"""You are an emergency disaster response expert. Someone needs immediate help.

Location: {user_location}
Disaster: {disaster_type}
Situation: {user_situation}

Provide IMMEDIATE, ACTIONABLE guidance specific to {user_location}:

🚨 IMMEDIATE ACTIONS:
[List 3-5 specific steps for {user_location}]

⚠️ CRITICAL DON'Ts:
[List 3-4 dangerous actions to avoid]

🏃 EVACUATION CRITERIA:
[When to leave immediately - consider local geography]

📦 ESSENTIAL ITEMS:
[Critical supplies to gather]

⏰ URGENCY LEVEL:
[Minutes/Hours/Days]

📞 LOCAL EMERGENCY CONTACTS:
[Include country-specific emergency numbers for {user_location}]

Keep it clear and life-saving focused."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"""⚠️ **AI Error:** {str(e)}

**Basic Safety Steps:**
1. Call emergency services immediately
2. Follow official evacuation orders
3. Move to safe location
4. Stay informed via local news"""

def analyze_disaster_image(image, model, max_retries=2) -> dict:
    if not model:
        return {'success': False, 'message': 'Please add Gemini API key'}
    
    prompt = """Analyze this disaster image as an expert assessor.

Provide:
**DISASTER TYPE:** [Type]
**SEVERITY:** [LOW/MODERATE/HIGH/CRITICAL and why]
**VISIBLE DAMAGES:** [List]
**AFFECTED AREA:** [Estimate]
**POPULATION RISK:** [Assessment]
**IMMEDIATE CONCERNS:** [Top 3]
**RESPONSE RECOMMENDATIONS:** [Actions needed]
**RECOVERY TIME:** [Short/Medium/Long-term]"""

    for attempt in range(max_retries):
        try:
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
            error_msg = str(e)
            
            if '429' in error_msg or 'quota' in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    st.warning(f"⏳ Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'message': f"""⚠️ **Rate Limit Exceeded**

Free tier quota hit. Please:
1. ⏰ Wait 2-3 minutes
2. 🔄 Use fewer requests (15/min limit)
3. 💳 Upgrade at https://ai.google.dev

Error: {error_msg[:200]}"""
                    }
            else:
                return {'success': False, 'message': f'Analysis failed: {error_msg[:300]}'}
    
    return {'success': False, 'message': 'Max retries exceeded'}

# ✅ CRITICAL FIX: Get active location (manual overrides browser)
def get_active_location():
    """Returns the currently active location (manual if set, otherwise browser/fallback)"""
    if st.session_state.get('manual_location'):
        return st.session_state.manual_location
    elif st.session_state.get('browser_location'):
        return st.session_state.browser_location
    else:
        return None

# Initialize session state
if 'manual_location' not in st.session_state:
    st.session_state.manual_location = None

if 'browser_location' not in st.session_state:
    st.session_state.browser_location = None

if 'gemini_model_text' not in st.session_state:
    st.session_state.gemini_model_text = None

if 'gemini_model_image' not in st.session_state:
    st.session_state.gemini_model_image = None

# ✅ IMPROVED SIDEBAR with browser geolocation
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## 🌍 AI-RescueMap")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["🗺 Disaster Map", "💬 AI Guidance", "🖼 Image Analysis", "📊 Analytics"])
    
    st.markdown("---")
    st.markdown("### 🎯 Your Location")
    
    # ✅ Get active location
    loc = get_active_location()
    
    if loc:
        location_badge = "📍 Manual" if loc.get('method') == 'manual' else "🌐 Browser GPS"
        st.success(f"**{loc['city']}**")
        st.info(f"🌍 {loc['country']}")
        st.caption(f"{location_badge} | {loc.get('source', 'Unknown')}")
        
        with st.expander("ℹ️ Details"):
            st.caption(f"**Lat/Lon:** {loc['lat']:.4f}, {loc['lon']:.4f}")
            st.caption(f"**Region:** {loc.get('region', 'N/A')}")
            if loc.get('method') == 'manual':
                st.caption(f"**Full Address:** {loc.get('full_address', 'N/A')}")
    else:
        st.warning("❌ No location set")
        st.caption("Use browser GPS or enter manually below")
    
    # ✅ NEW: Browser Geolocation Button
    st.markdown("---")
    st.markdown("### 📍 Get My Location")
    st.caption("Uses your device's GPS/Wi-Fi")
    
    # JavaScript to get browser location
    browser_location_js = """
    <script>
    function getLocation() {
        const btn = document.getElementById('geolocate-btn');
        const status = document.getElementById('geo-status');
        
        btn.disabled = true;
        btn.innerText = '⏳ Getting location...';
        status.innerText = 'Requesting permission...';
        
        if (!navigator.geolocation) {
            status.innerText = '❌ Geolocation not supported';
            btn.disabled = false;
            btn.innerText = '📍 Get My Location';
            return;
        }
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude.toFixed(6);
                const lon = position.coords.longitude.toFixed(6);
                const accuracy = position.coords.accuracy.toFixed(0);
                
                // Reload with coordinates in URL
                const url = new URL(window.location.href);
                url.searchParams.set('geo_lat', lat);
                url.searchParams.set('geo_lon', lon);
                url.searchParams.set('geo_accuracy', accuracy);
                window.location.href = url.toString();
            },
            (error) => {
                let msg = '';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        msg = '❌ Permission denied. Please allow location access.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        msg = '❌ Location unavailable. Check GPS/Wi-Fi.';
                        break;
                    case error.TIMEOUT:
                        msg = '❌ Request timeout. Try again.';
                        break;
                    default:
                        msg = '❌ Unknown error: ' + error.message;
                }
                status.innerText = msg;
                btn.disabled = false;
                btn.innerText = '📍 Get My Location';
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    }
    </script>
    <button id="geolocate-btn" onclick="getLocation()" style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        width: 100%;
        margin-bottom: 10px;
    ">📍 Get My Location</button>
    <p id="geo-status" style="color: #666; font-size: 12px; margin-top: 5px;"></p>
    """
    
    st.components.v1.html(browser_location_js, height=120)
    
    # ✅ Process browser geolocation from URL params
    query_params = st.query_params
    if 'geo_lat' in query_params and 'geo_lon' in query_params:
        try:
            lat = float(query_params['geo_lat'])
            lon = float(query_params['geo_lon'])
            accuracy = float(query_params.get('geo_accuracy', 0))
            
            # Only update if coordinates changed
            if (not st.session_state.browser_location or 
                abs(st.session_state.browser_location['lat'] - lat) > 0.0001 or
                abs(st.session_state.browser_location['lon'] - lon) > 0.0001):
                
                with st.spinner("🌍 Getting address..."):
                    browser_loc = reverse_geocode(lat, lon)
                    st.session_state.browser_location = browser_loc
                    
                    # Clear manual location when browser location is set
                    st.session_state.manual_location = None
                    
                    # Clear URL params
                    st.query_params.clear()
                    st.success(f"✅ Located: {browser_loc['city']}, {browser_loc['country']}")
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing location: {e}")
    
    # Manual location input
    st.markdown("---")
    st.markdown("### 🔧 Manual Location")
    st.caption("🌍 Enter any city worldwide")
    
    with st.expander("📝 Enter Location"):
        st.info("**Examples:**\n"
                "- Faisalabad, Pakistan\n"
                "- New York, USA\n"
                "- Tokyo, Japan\n"
                "- London, UK")
        
        location_input = st.text_input(
            "City/Country",
            value="",
            placeholder="e.g., Faisalabad, Pakistan",
            key="manual_input"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔍 Find", use_container_width=True, disabled=not location_input):
                if location_input:
                    with st.spinner(f"🌍 Finding {location_input}..."):
                        geocoded = geocode_location(location_input)
                        if geocoded:
                            st.session_state.manual_location = geocoded
                            st.session_state.browser_location = None  # Clear browser location
                            st.success(f"✅ {geocoded['city']}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ Not found. Try adding country name.")
        
        with col_btn2:
            if st.session_state.manual_location and st.button("🔄 Clear", use_container_width=True):
                st.session_state.manual_location = None
                st.success("✅ Cleared")
                time.sleep(0.3)
                st.rerun()

# Main header
st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time disaster monitoring with NASA data & Google Gemini 2.5 AI</p>', unsafe_allow_html=True)

# Setup Gemini models
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if st.session_state.gemini_model_text is None:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if st.session_state.gemini_model_image is None:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# ✅ CRITICAL: Use active location everywhere
loc = get_active_location()

if menu == "🗺 Disaster Map":
    with st.spinner("🛰 Fetching NASA EONET data..."):
        disasters = fetch_nasa_eonet_disasters()
    
    # ✅ FIXED: Recalculate distances based on ACTIVE location
    if loc and not disasters.empty:
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
        disasters = disasters.sort_values('distance_km')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌪 Active Disasters", len(disasters))
    with col2:
        if loc and not disasters.empty and 'distance_km' in disasters.columns:
            nearby = len(disasters[disasters['distance_km'] < 500])
            st.metric("📍 Nearby (<500km)", nearby)
        else:
            st.metric("🔥 Most Common", disasters['category'].mode()[0] if not disasters.empty else "N/A")
    with col3:
        st.metric("🤖 AI Status", "✅ Online" if st.session_state.gemini_model_text else "⚠️ Offline")
    with col4:
        st.metric("🛰 Data Source", "NASA EONET")
    
    st.markdown("---")
    
    col_settings, col_map = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### ⚙️ Settings")
        
        map_options = ["My Location", "Global View"] + (disasters['title'].tolist()[:10] if not disasters.empty else [])
        map_center_option = st.selectbox("Center Map", map_options)
        
        if map_center_option == "My Location" and loc:
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
        elif map_center_option == "Global View":
            center_lat, center_lon, zoom = 20, 0, 2
        elif not disasters.empty:
            disaster_row = disasters[disasters['title'] == map_center_option].iloc[0]
            center_lat, center_lon, zoom = disaster_row['lat'], disaster_row['lon'], 8
        else:
            center_lat, center_lon, zoom = 0, 0, 2
        
        show_disasters = st.checkbox("Show Disasters", value=True)
        show_population = st.checkbox("Show Population", value=True)
        
        satellite_layers = st.multiselect("Satellite Layers", 
                                         ['True Color', 'Active Fires', 'Night Lights'], 
                                         default=['True Color'])
        impact_radius = st.slider("Impact Radius (km)", 10, 200, 50)
    
    with col_map:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='CartoDB positron')
        
        if satellite_layers:
            m = add_nasa_satellite_layers(m, satellite_layers)
        
        if show_population:
            pop_df = generate_population_data(center_lat, center_lon, radius_deg=3, num_points=1500)
            heat_data = [[row['lat'], row['lon'], row['population']] for _, row in pop_df.iterrows()]
            HeatMap(heat_data, radius=15, blur=25, max_zoom=13, 
                   gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)
        
        if show_disasters and not disasters.empty:
            marker_cluster = MarkerCluster().add_to(m)
            color_map = {'Wildfires': 'red', 'Severe Storms': 'orange', 'Floods': 'blue', 
                        'Earthquakes': 'darkred', 'Volcanoes': 'red'}
            
            for _, disaster in disasters.iterrows():
                color = color_map.get(disaster['category'], 'gray')
                distance_text = f"<br>📍 {disaster['distance_km']:.0f} km from you" if 'distance_km' in disaster else ""
                
                folium.Circle(location=[disaster['lat'], disaster['lon']], 
                            radius=impact_radius * 1000,
                            color=color, fill=True, fillOpacity=0.1).add_to(m)
                
                folium.Marker(location=[disaster['lat'], disaster['lon']],
                            popup=f"<b>{disaster['title']}</b><br>{disaster['category']}<br>{disaster['date']}{distance_text}",
                            icon=folium.Icon(color=color, icon='warning-sign', prefix='glyphicon'),
                            tooltip=disaster['title']).add_to(marker_cluster)
        
        if loc:
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>📍 You are here</b><br>{loc['city']}, {loc['country']}",
                icon=folium.Icon(color='green', icon='home', prefix='glyphicon'),
                tooltip="Your Location"
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, width=1000, height=600)
    
    if show_disasters and show_population and not disasters.empty and 'pop_df' in locals():
        st.markdown("---")
        st.markdown("### 📊 Population Impact Analysis")
        impacts = calculate_disaster_impact(disasters, pop_df, impact_radius)
        
        if impacts:
            impact_df = pd.DataFrame(impacts)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚠️ High Risk Events")
                high_risk = impact_df[impact_df['risk_level'].isin(['CRITICAL', 'HIGH'])]
                for _, imp in high_risk.iterrows():
                    st.markdown(f"""<div class="disaster-alert">
                    ⚠️ <b>{imp['disaster']}</b><br>
                    👥 {imp['affected_population']:,} people at risk<br>
                    🚨 Risk Level: {imp['risk_level']}</div>""", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📈 Statistics")
                st.metric("Total at Risk", f"{impact_df['affected_population'].sum():,}")
                st.metric("Critical Events", len(impact_df[impact_df['risk_level'] == 'CRITICAL']))
                st.metric("High Risk Events", len(impact_df[impact_df['risk_level'] == 'HIGH']))

elif menu == "💬 AI Guidance":
    st.markdown("## 💬 AI Emergency Guidance")
    
    if not loc:
        st.warning("⚠️ Please set your location first (sidebar)")
    else:
        st.info(f"📍 Your location: **{loc['city']}, {loc['country']}**")
    
    disaster_type = st.selectbox("Disaster Type", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Other"])
    
    user_situation = st.text_area("Describe your situation:",
        placeholder="Be specific: location, number of people, current conditions, available resources...",
        height=120)
    
    if st.button("🚨 GET AI GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("Please describe your situation")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to secrets")
        else:
            location_str = f"{loc['city']}, {loc['country']}" if loc else "your area"
            with st.spinner("🤖 Analyzing with Gemini 2.0..."):
                guidance = get_ai_disaster_guidance(disaster_type, user_situation, location_str, st.session_state.gemini_model_text)
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)
                
                st.markdown("### 📞 Emergency Contacts")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.error("🚨 **911** (US)")
                with col_b:
                    st.warning("🆘 **1122** (Pakistan)")
                with col_c:
                    st.info("🇪🇺 **112** (Europe)")

elif menu == "🖼 Image Analysis":
    from PIL import Image
    
    st.markdown("## 🖼 AI Image Analysis")
    st.info("⚠️ Free tier: ~15 requests/minute. Wait if quota exceeded.")
    
    uploaded_file = st.file_uploader("Upload disaster image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        if st.button("🔍 ANALYZE IMAGE", type="primary", use_container_width=True):
            if not st.session_state.gemini_model_image:
                st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to secrets")
            else:
                with st.spinner("🤖 Analyzing with Gemini 2.0..."):
                    result = analyze_disaster_image(image, st.session_state.gemini_model_image)
                    
                    if result['success']:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Severity", result['severity_level'])
                        with col_b:
                            st.metric("Risk Score", f"{result['severity_score']}/100")
                        with col_c:
                            st.metric("Status", "✅ Complete")
                        
                        st.markdown(f'<div class="ai-response">{result["analysis"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(result.get('message', 'Analysis failed'))

elif menu == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    
    if not loc:
        st.warning("⚠️ Please set your location first (sidebar)")
        view_mode = "🌍 Global View"
    else:
        view_mode = st.radio("View Mode:", ["📍 My Location (Recommended)", "🌍 Global View"], horizontal=True)
    
    with st.spinner("Loading disaster data..."):
        disasters = fetch_nasa_eonet_disasters(limit=100)
    
    # ✅ FIXED: Recalculate distances for ACTIVE location
    if not disasters.empty and loc:
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
    
    if "My Location" in view_mode and loc and not disasters.empty:
        radius_filter = st.slider("Show disasters within (km):", 100, 5000, 1000, step=100)
        filtered_disasters = disasters[disasters['distance_km'] <= radius_filter].copy()
        st.success(f"📍 Showing {len(filtered_disasters)} disasters within {radius_filter} km of **{loc['city']}, {loc['country']}**")
    else:
        filtered_disasters = disasters
        st.info(f"🌍 Showing all {len(filtered_disasters)} global disasters")
    
    if not filtered_disasters.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌍 Total Disasters", len(filtered_disasters))
        with col2:
            wildfires = len(filtered_disasters[filtered_disasters['category'] == 'Wildfires'])
            st.metric("🔥 Wildfires", wildfires)
        with col3:
            storms = len(filtered_disasters[filtered_disasters['category'] == 'Severe Storms'])
            st.metric("🌪 Storms", storms)
        with col4:
            others = len(filtered_disasters[~filtered_disasters['category'].isin(['Wildfires', 'Severe Storms'])])
            st.metric("🌊 Other", others)
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 By Category")
            category_counts = filtered_disasters['category'].value_counts()
            st.bar_chart(category_counts)
        
        with col_b:
            st.markdown("### 📅 Recent Events")
            display_cols = ['title', 'category', 'date']
            if 'distance_km' in filtered_disasters.columns and "My Location" in view_mode:
                filtered_disasters['distance_km'] = filtered_disasters['distance_km'].round(0).astype(int)
                display_cols.append('distance_km')
            
            recent = filtered_disasters.head(10)[display_cols]
            st.dataframe(recent, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown(f"### 🗺 {'Local' if 'My Location' in view_mode else 'Global'} Distribution")
        
        map_center = [loc['lat'], loc['lon']] if loc and "My Location" in view_mode else [20, 0]
        map_zoom = 6 if "My Location" in view_mode else 2
        
        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB dark_matter')
        
        color_map = {
            'Wildfires': 'red', 
            'Severe Storms': 'orange', 
            'Floods': 'blue', 
            'Earthquakes': 'darkred',
            'Volcanoes': 'red'
        }
        
        for _, disaster in filtered_disasters.iterrows():
            popup_text = f"<b>{disaster['title']}</b><br>{disaster['category']}"
            if 'distance_km' in disaster and "My Location" in view_mode:
                popup_text += f"<br>📍 {disaster['distance_km']:.0f} km away"
            
            folium.CircleMarker(
                location=[disaster['lat'], disaster['lon']], 
                radius=8,
                color=color_map.get(disaster['category'], 'gray'),
                fill=True, 
                fillOpacity=0.7,
                popup=popup_text,
                tooltip=disaster['title']
            ).add_to(m)
        
        if loc and "My Location" in view_mode:
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>📍 You are here</b><br>{loc['city']}, {loc['country']}",
                icon=folium.Icon(color='green', icon='home', prefix='glyphicon'),
                tooltip="Your Location"
            ).add_to(m)
        
        st_folium(m, width=1200, height=500)
        
        st.markdown("---")
        
        st.markdown("### 📋 All Disasters")
        
        col1, col2 = st.columns(2)
        with col1:
            all_categories = filtered_disasters['category'].unique().tolist()
            selected_cat = st.multiselect("Filter by Category", all_categories, default=all_categories)
        with col2:
            search = st.text_input("Search by keyword", "")
        
        final_filtered = filtered_disasters[filtered_disasters['category'].isin(selected_cat)]
        if search:
            final_filtered = final_filtered[final_filtered['title'].str.contains(search, case=False, na=False)]
        
        display_cols = ['title', 'category', 'date', 'lat', 'lon']
        if 'distance_km' in final_filtered.columns and "My Location" in view_mode:
            display_cols.append('distance_km')
        
        st.dataframe(final_filtered[display_cols], use_container_width=True, hide_index=True, height=400)
        
        st.download_button(
            "📥 Download as CSV",
            data=final_filtered.to_csv(index=False).encode('utf-8'),
            file_name=f"disasters_{loc['city'] if loc else 'global'}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No disasters found in your area. Try adjusting the radius or switch to Global view.")

# Footer
st.markdown("---")
st.markdown(f"""
<p style='text-align: center; color: gray;'>
Built by <b>HasnainAtif</b> (@{st.secrets.get('GITHUB_USER', 'Hasnainatif')}) for NASA Space Apps Challenge 2025<br>
Powered by NASA EONET, NASA GIBS & Google Gemini 2.5 AI<br>
<small>Current UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</small>
</p>
""", unsafe_allow_html=True)
