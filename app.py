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
from math import radians, cos, sin, asin, sqrt

os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'

# Check for Gemini availability
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Check for streamlit-javascript
try:
    from streamlit_javascript import st_javascript
    JAVASCRIPT_AVAILABLE = True
except ImportError:
    JAVASCRIPT_AVAILABLE = False
    st.warning("⚠️ Install `streamlit-javascript` for browser GPS: `pip install streamlit-javascript`")

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
    "IPAPI_URL": "https://ipapi.co/json/",
    "IPAPI_BACKUP": "http://ip-api.com/json/",
    "GEOCODING_API": "https://nominatim.openstreetmap.org/search"
}

# Emergency contacts database
EMERGENCY_CONTACTS = {
    "Pakistan": {"emergency": "1122", "police": "15", "ambulance": "1122", "rescue": "1122"},
    "United States": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "India": {"emergency": "112", "police": "100", "ambulance": "102", "fire": "101"},
    "United Kingdom": {"emergency": "999", "police": "999", "ambulance": "999", "fire": "999"},
    "Canada": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "Australia": {"emergency": "000", "police": "000", "ambulance": "000", "fire": "000"},
    "Germany": {"emergency": "112", "police": "110", "ambulance": "112", "fire": "112"},
    "France": {"emergency": "112", "police": "17", "ambulance": "15", "fire": "18"},
    "Japan": {"emergency": "110/119", "police": "110", "ambulance": "119", "fire": "119"},
    "China": {"emergency": "110/119/120", "police": "110", "ambulance": "120", "fire": "119"},
    "default": {"emergency": "112", "note": "112 works in most countries"}
}

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def geocode_location(city_or_address: str):
    """Convert city name or address to lat/lon using OpenStreetMap Nominatim"""
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
            display_name_parts = result.get('display_name', '').split(',')
            
            return {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'city': display_name_parts[0].strip() if len(display_name_parts) > 0 else city_or_address,
                'country': display_name_parts[-1].strip() if len(display_name_parts) > 0 else 'Unknown',
                'region': display_name_parts[1].strip() if len(display_name_parts) > 1 else 'Unknown',
                'full_address': result.get('display_name', city_or_address),
                'method': 'manual',
                'source': 'Geocoded'
            }
        else:
            return None
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None

def get_ip_location():
    """Get location from IP address (fallback method)"""
    try:
        response = requests.get(CONFIG["IPAPI_URL"], timeout=5)
        data = response.json()
        
        if 'error' in data and data['error']:
            raise Exception(f"API error: {data.get('reason', 'Unknown')}")
        
        if 'latitude' not in data or 'longitude' not in data:
            raise KeyError("Missing coordinates")
        
        return {
            'lat': float(data['latitude']),
            'lon': float(data['longitude']),
            'city': data.get('city', 'Unknown'),
            'country': data.get('country_name', 'Unknown'),
            'region': data.get('region', 'Unknown'),
            'ip': data.get('ip', 'Unknown'),
            'method': 'ip',
            'source': 'IP Geolocation (Fallback)'
        }
    except Exception as e:
        try:
            alt_response = requests.get(CONFIG["IPAPI_BACKUP"], timeout=5)
            alt_data = alt_response.json()
            
            if alt_data.get('status') != 'success':
                raise Exception(f"Backup failed: {alt_data.get('message', 'Unknown')}")
            
            return {
                'lat': float(alt_data['lat']),
                'lon': float(alt_data['lon']),
                'city': alt_data.get('city', 'Unknown'),
                'country': alt_data.get('country', 'Unknown'),
                'region': alt_data.get('regionName', 'Unknown'),
                'ip': alt_data.get('query', 'Unknown'),
                'method': 'ip',
                'source': 'IP Geolocation (Fallback)'
            }
        except Exception as backup_error:
            return None

def get_browser_location():
    """Get real-time location from browser using JavaScript"""
    if not JAVASCRIPT_AVAILABLE:
        return None
    
    try:
        # JavaScript code to get geolocation
        js_code = """
        await (async () => {
            const getPosition = () => {
                return new Promise((resolve, reject) => {
                    if (!navigator.geolocation) {
                        reject(new Error('Geolocation not supported'));
                    }
                    navigator.geolocation.getCurrentPosition(
                        position => resolve({
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                            accuracy: position.coords.accuracy
                        }),
                        error => reject(error),
                        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
                    );
                });
            };
            try {
                const pos = await getPosition();
                return pos;
            } catch (error) {
                return { error: error.message };
            }
        })()
        """
        
        result = st_javascript(js_code)
        
        if result and isinstance(result, dict) and 'latitude' in result and 'longitude' in result:
            # Reverse geocode to get location name
            lat, lon = result['latitude'], result['longitude']
            
            try:
                reverse_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
                headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
                geo_response = requests.get(reverse_url, headers=headers, timeout=5)
                geo_data = geo_response.json()
                
                address = geo_data.get('address', {})
                display_name_parts = geo_data.get('display_name', '').split(',')
                
                return {
                    'lat': lat,
                    'lon': lon,
                    'city': address.get('city') or address.get('town') or address.get('village') or display_name_parts[0].strip(),
                    'country': address.get('country', 'Unknown'),
                    'region': address.get('state') or address.get('region', 'Unknown'),
                    'accuracy': result.get('accuracy', 0),
                    'method': 'gps',
                    'source': 'Browser GPS'
                }
            except:
                return {
                    'lat': lat,
                    'lon': lon,
                    'city': 'Unknown',
                    'country': 'Unknown',
                    'region': 'Unknown',
                    'accuracy': result.get('accuracy', 0),
                    'method': 'gps',
                    'source': 'Browser GPS'
                }
        
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_nasa_eonet_disasters_paginated(status="open"):
    """Fetch ALL disasters from NASA EONET using pagination"""
    all_disasters = []
    offset = 0
    batch_size = 500
    
    try:
        while True:
            url = f"{CONFIG['EONET_API']}?status={status}&limit={batch_size}&offset={offset}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            events = data.get('events', [])
            if not events:
                break
            
            for event in events:
                if event.get('geometry'):
                    latest_geo = event['geometry'][-1]
                    coords = latest_geo.get('coordinates', [])
                    
                    if len(coords) >= 2:
                        # Handle both Point and Polygon geometries
                        if latest_geo['type'] == 'Point':
                            lat, lon = coords[1], coords[0]
                        else:
                            lat, lon = coords[0][1], coords[0][0]
                        
                        all_disasters.append({
                            'id': event['id'],
                            'title': event['title'],
                            'category': event['categories'][0]['title'] if event.get('categories') else 'Unknown',
                            'lat': lat,
                            'lon': lon,
                            'date': latest_geo.get('date', 'Unknown'),
                            'source': ', '.join([s['id'] for s in event.get('sources', [])]),
                            'link': event.get('link', '')
                        })
            
            offset += batch_size
            
            # Safety break to avoid infinite loops
            if offset > 10000:
                break
        
        return pd.DataFrame(all_disasters)
    except Exception as e:
        st.error(f"Failed to fetch NASA EONET data: {e}")
        return pd.DataFrame()

def filter_disasters_by_location(disasters_df, lat, lon, radius_km):
    """Filter disasters within radius of user location"""
    if disasters_df.empty:
        return disasters_df
    
    disasters_df = disasters_df.copy()
    disasters_df['distance_km'] = disasters_df.apply(
        lambda row: calculate_distance(lat, lon, row['lat'], row['lon']), 
        axis=1
    )
    
    return disasters_df[disasters_df['distance_km'] <= radius_km].sort_values('distance_km')

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
    # Use unique but safe seed
    seed_value = abs(int((center_lat * 1000 + center_lon * 1000) % 2147483647))
    np.random.seed(seed_value)
    
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

def get_emergency_contacts(country_name):
    """Get emergency contacts based on country"""
    return EMERGENCY_CONTACTS.get(country_name, EMERGENCY_CONTACTS["default"])

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, model, user_location: dict = None, use_location: bool = False) -> str:
    if not model:
        return """⚠️ **AI Not Available** - Please add your Gemini API key.

**Emergency Contacts:**
- 🚨 Call local emergency services immediately"""
    
    try:
        location_context = ""
        if use_location and user_location:
            location_context = f"\nUser's Location: {user_location.get('city', 'Unknown')}, {user_location.get('country', 'Unknown')}"
        
        prompt = f"""You are an emergency disaster response expert. Someone needs immediate help.

Disaster: {disaster_type}
Situation: {user_situation}{location_context}

Provide IMMEDIATE, ACTIONABLE guidance:

🚨 IMMEDIATE ACTIONS:
[List 3-5 specific steps to take RIGHT NOW]

⚠️ CRITICAL DON'Ts:
[List 3-4 dangerous actions to avoid]

🏃 EVACUATION CRITERIA:
[When to leave immediately vs shelter in place]

📦 ESSENTIAL ITEMS:
[Critical supplies to gather if time permits]

⏰ URGENCY LEVEL:
[Minutes/Hours/Days - be specific]

Keep it clear, concise, and life-saving focused. Do NOT include location information in your response."""

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
2. 🔄 Use fewer requests
3. 💳 Upgrade at https://ai.google.dev"""
                    }
            else:
                return {'success': False, 'message': f'Analysis failed: {error_msg[:300]}'}
    
    return {'success': False, 'message': 'Max retries exceeded'}

# Initialize session state
if 'browser_location' not in st.session_state:
    st.session_state.browser_location = None

if 'manual_location' not in st.session_state:
    st.session_state.manual_location = None

if 'ip_location' not in st.session_state:
    st.session_state.ip_location = get_ip_location()

if 'gemini_model_text' not in st.session_state:
    st.session_state.gemini_model_text = None

if 'gemini_model_image' not in st.session_state:
    st.session_state.gemini_model_image = None

if 'location_requested' not in st.session_state:
    st.session_state.location_requested = False

# Priority: Browser GPS > Manual > IP
def get_current_location():
    """Get current location with priority: GPS > Manual > IP"""
    if st.session_state.browser_location:
        return st.session_state.browser_location
    elif st.session_state.manual_location:
        return st.session_state.manual_location
    else:
        return st.session_state.ip_location

# SIDEBAR
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## 🌍 AI-RescueMap")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["🗺 Disaster Map", "💬 AI Guidance", "🖼 Image Analysis", "📊 Analytics"])
    
    st.markdown("---")
    st.markdown("### 🎯 Your Location")
    
    loc = get_current_location()
    
    if loc:
        method_badge = {
            'gps': '📍 GPS (Most Accurate)',
            'manual': '📍 Manual Entry',
            'ip': '🌍 IP-Based (Less Accurate)'
        }
        
        st.success(f"**{loc['city']}, {loc.get('region', '')}**")
        st.info(f"🌍 {loc['country']}")
        st.caption(f"{method_badge.get(loc.get('method'), 'Unknown')} | {loc.get('source', 'Unknown')}")
        
        with st.expander("ℹ️ Location Details"):
            st.caption(f"**Coordinates:** {loc['lat']:.4f}, {loc['lon']:.4f}")
            st.caption(f"**Method:** {loc.get('method', 'unknown').upper()}")
            if loc.get('accuracy'):
                st.caption(f"**Accuracy:** ±{loc['accuracy']:.0f}m")
    else:
        st.error("❌ Location unavailable")
    
    st.markdown("---")
    st.markdown("### 📍 Location Options")
    
    # Browser GPS Location
    if JAVASCRIPT_AVAILABLE:
        if st.button("📍 Get My Location (GPS)", use_container_width=True, type="primary"):
            with st.spinner("📡 Requesting location access..."):
                st.session_state.location_requested = True
                st.rerun()
        
        if st.session_state.location_requested:
            browser_loc = get_browser_location()
            if browser_loc:
                st.session_state.browser_location = browser_loc
                st.session_state.location_requested = False
                st.success(f"✅ GPS Location: {browser_loc['city']}, {browser_loc['country']}")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Location access denied or unavailable. Please enable location in your browser settings.")
                st.session_state.location_requested = False
    else:
        st.warning("⚠️ GPS feature unavailable. Install: `pip install streamlit-javascript`")
    
    # Manual location
    with st.expander("🔧 Enter Location Manually"):
        st.info("**Examples:**\n- Faisalabad, Pakistan\n- New York, USA\n- Tokyo, Japan")
        
        location_input = st.text_input(
            "City, Country",
            value="",
            placeholder="e.g., Faisalabad, Pakistan"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔍 Find", use_container_width=True, disabled=not location_input):
                if location_input:
                    with st.spinner(f"🌍 Finding {location_input}..."):
                        geocoded = geocode_location(location_input)
                        if geocoded:
                            st.session_state.manual_location = geocoded
                            st.session_state.browser_location = None  # Clear GPS
                            st.success(f"✅ Found: {geocoded['city']}, {geocoded['country']}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Location not found. Try adding country name.")
        
        with col_btn2:
            if (st.session_state.manual_location or st.session_state.browser_location) and st.button("🔄 Reset", use_container_width=True):
                st.session_state.manual_location = None
                st.session_state.browser_location = None
                st.success("✅ Reset to IP location")
                time.sleep(0.5)
                st.rerun()

# Main header
st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time global disaster monitoring with NASA data & AI</p>', unsafe_allow_html=True)

# Setup Gemini models
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if st.session_state.gemini_model_text is None:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if st.session_state.gemini_model_image is None:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# MENU PAGES
if menu == "🗺 Disaster Map":
    loc = get_current_location()
    
    with st.spinner("🛰 Fetching real-time NASA EONET data..."):
        all_disasters = fetch_nasa_eonet_disasters_paginated()
    
    if not all_disasters.empty:
        st.success(f"✅ Loaded {len(all_disasters)} global disasters")
    
    # Filter by location if available
    if loc:
        filter_radius = st.slider("📍 Show disasters within (km):", 100, 5000, 1000, step=100, 
                                  help="Filter disasters near your location")
        disasters = filter_disasters_by_location(all_disasters, loc['lat'], loc['lon'], filter_radius)
        
        if disasters.empty:
            st.warning(f"⚠️ No disasters found within {filter_radius} km. Showing nearest disasters...")
            # Show top 20 nearest disasters
            temp_df = all_disasters.copy()
            temp_df['distance_km'] = temp_df.apply(
                lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
                axis=1
            )
            disasters = temp_df.nsmallest(20, 'distance_km')
    else:
        disasters = all_disasters.head(100)  # Show first 100 if no location
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌍 Total Disasters", len(all_disasters))
    with col2:
        if loc and not disasters.empty and 'distance_km' in disasters.columns:
            nearby = len(disasters[disasters['distance_km'] < 500])
            st.metric("📍 Nearby (<500km)", nearby)
        else:
            st.metric("🔥 Displaying", len(disasters))
    with col3:
        st.metric("🤖 AI Status", "✅ Online" if st.session_state.gemini_model_text else "⚠️ Offline")
    with col4:
        categories = disasters['category'].nunique() if not disasters.empty else 0
        st.metric("📊 Categories", categories)
    
    st.markdown("---")
    
    col_settings, col_map = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### ⚙️ Settings")
        
        # Map center options
        map_options = ["My Location", "Global View"]
        if not disasters.empty:
            map_options += disasters['title'].head(10).tolist()
        
        map_center_option = st.selectbox("Center Map", map_options)
        
        if map_center_option == "My Location" and loc:
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
        elif map_center_option == "Global View":
            center_lat, center_lon, zoom = 20, 0, 2
        else:
            # Specific disaster selected
            if not disasters.empty:
                disaster_matches = disasters[disasters['title'] == map_center_option]
                if not disaster_matches.empty:
                    disaster_row = disaster_matches.iloc[0]
                    center_lat, center_lon, zoom = disaster_row['lat'], disaster_row['lon'], 8
                else:
                    center_lat, center_lon, zoom = 20, 0, 2
            else:
                center_lat, center_lon, zoom = 20, 0, 2
        
        show_disasters = st.checkbox("Show Disasters", value=True)
        show_population = st.checkbox("Show Population Heatmap", value=True)
        
        satellite_layers = st.multiselect("Satellite Layers", 
                                         ['True Color', 'Active Fires', 'Night Lights'], 
                                         default=['True Color'])
        impact_radius = st.slider("Impact Radius (km)", 10, 200, 50)
    
    with col_map:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='CartoDB positron')
        
        if satellite_layers:
            m = add_nasa_satellite_layers(m, satellite_layers)
        
        if show_population and loc:
            pop_df = generate_population_data(center_lat, center_lon, radius_deg=3, num_points=1500)
            heat_data = [[row['lat'], row['lon'], row['population']] for _, row in pop_df.iterrows()]
            HeatMap(heat_data, radius=15, blur=25, max_zoom=13, 
                   gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)
        
        if show_disasters and not disasters.empty:
            marker_cluster = MarkerCluster().add_to(m)
            color_map = {'Wildfires': 'red', 'Severe Storms': 'orange', 'Floods': 'blue', 
                        'Earthquakes': 'darkred', 'Volcanoes': 'red', 'Sea and Lake Ice': 'lightblue',
                        'Snow': 'white', 'Drought': 'brown', 'Dust and Haze': 'beige',
                        'Manmade': 'gray', 'Water Color': 'cyan'}
            
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
    
    if show_disasters and show_population and not disasters.empty and loc:
        st.markdown("---")
        st.markdown("### 📊 Population Impact Analysis")
        pop_df = generate_population_data(center_lat, center_lon, radius_deg=3, num_points=1500)
        impacts = calculate_disaster_impact(disasters, pop_df, impact_radius)
        
        if impacts:
            impact_df = pd.DataFrame(impacts)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚠️ High Risk Events")
                high_risk = impact_df[impact_df['risk_level'].isin(['CRITICAL', 'HIGH'])].head(5)
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
    
    loc = get_current_location()
    
    disaster_type = st.selectbox("Disaster Type", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Drought", "Other"])
    
    user_situation = st.text_area("Describe your situation:",
        placeholder="Be specific: current conditions, number of people, available resources, immediate threats...",
        height=120)
    
    use_location = st.checkbox("Use my location for context-specific guidance", value=False,
                               help="AI will provide emergency contacts and guidance specific to your region")
    
    if st.button("🚨 GET AI GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("Please describe your situation")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to secrets")
        else:
            with st.spinner("🤖 Analyzing with Gemini AI..."):
                guidance = get_ai_disaster_guidance(disaster_type, user_situation, 
                                                   st.session_state.gemini_model_text, 
                                                   loc, use_location)
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)
                
                # Show emergency contacts based on location
                if use_location and loc:
                    country = loc.get('country', 'Unknown')
                    contacts = get_emergency_contacts(country)
                    
                    st.markdown(f"### 📞 Emergency Contacts for {country}")
                    
                    cols = st.columns(len(contacts))
                    for idx, (service, number) in enumerate(contacts.items()):
                        with cols[idx % len(cols)]:
                            if service == 'note':
                                st.info(f"ℹ️ {number}")
                            else:
                                emoji_map = {'emergency': '🚨', 'police': '👮', 'ambulance': '🚑', 
                                           'fire': '🚒', 'rescue': '🆘'}
                                st.error(f"{emoji_map.get(service, '📞')} **{service.title()}:** {number}")
                else:
                    st.info("💡 **Tip:** Check 'Use my location' for region-specific emergency contacts")

elif menu == "🖼 Image Analysis":
    from PIL import Image
    
    st.markdown("## 🖼 AI Disaster Image Analysis")
    st.info("📸 Upload an image of disaster damage for instant AI assessment")
    
    uploaded_file = st.file_uploader("Upload disaster image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        if st.button("🔍 ANALYZE IMAGE", type="primary", use_container_width=True):
            if not st.session_state.gemini_model_image:
                st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to secrets")
            else:
                with st.spinner("🤖 Analyzing with Gemini AI Vision..."):
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
    st.markdown("## 📊 Global Disaster Analytics")
    
    loc = get_current_location()
    
    with st.spinner("📡 Loading real-time disaster data..."):
        all_disasters = fetch_nasa_eonet_disasters_paginated()
    
    if all_disasters.empty:
        st.error("⚠️ Unable to load disaster data")
        st.stop()
    
    # View mode selection
    if loc:
        view_mode = st.radio("View Mode:", ["📍 My Location", "🌍 Global View"], horizontal=True)
    else:
        view_mode = "🌍 Global View"
        st.info("💡 Enable location for personalized insights")
    
    # Filter data based on view mode
    if "My Location" in view_mode and loc:
        radius_filter = st.slider("Show disasters within (km):", 100, 5000, 1000, step=100)
        filtered_disasters = filter_disasters_by_location(all_disasters, loc['lat'], loc['lon'], radius_filter)
        
        if filtered_disasters.empty:
            st.warning(f"⚠️ No disasters within {radius_filter} km. Showing nearest 50...")
            temp_df = all_disasters.copy()
            temp_df['distance_km'] = temp_df.apply(
                lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
                axis=1
            )
            filtered_disasters = temp_df.nsmallest(50, 'distance_km')
        
        st.success(f"📍 Showing {len(filtered_disasters)} disasters within {radius_filter} km of **{loc['city']}, {loc['country']}**")
    else:
        filtered_disasters = all_disasters
        st.info(f"🌍 Showing all {len(filtered_disasters)} global disasters")
    
    # Metrics
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
        categories = filtered_disasters['category'].nunique()
        st.metric("📊 Categories", categories)
    
    st.markdown("---")
    
    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Disasters by Category")
        category_counts = filtered_disasters['category'].value_counts().head(10)
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
    
    # Map visualization
    st.markdown(f"### 🗺 {'Local' if 'My Location' in view_mode else 'Global'} Distribution")
    
    map_center = [loc['lat'], loc['lon']] if loc and "My Location" in view_mode else [20, 0]
    map_zoom = 6 if "My Location" in view_mode else 2
    
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB dark_matter')
    
    color_map = {
        'Wildfires': 'red', 'Severe Storms': 'orange', 'Floods': 'blue', 
        'Earthquakes': 'darkred', 'Volcanoes': 'red', 'Sea and Lake Ice': 'lightblue',
        'Snow': 'white', 'Drought': 'brown', 'Dust and Haze': 'beige',
        'Manmade': 'gray', 'Water Color': 'cyan'
    }
    
    # Add disaster markers
    for _, disaster in filtered_disasters.head(200).iterrows():  # Limit to 200 for performance
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
    
    # Data table with filters
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

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
Built by <b>HasnainAtif</b> for NASA Space Apps Challenge 2025<br>
Real-time global disaster monitoring • GPS-enabled location tracking
</p>
""", unsafe_allow_html=True)
