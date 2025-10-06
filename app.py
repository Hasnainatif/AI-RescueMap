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
    "IPAPI_URL": "https://ipapi.co/json/",
    "IPAPI_BACKUP": "http://ip-api.com/json/",
    "GEOCODING_API": "https://nominatim.openstreetmap.org/search",
    "REVERSE_GEOCODING_API": "https://nominatim.openstreetmap.org/reverse"
}

# ✅ WORLDWIDE EMERGENCY CONTACTS DATABASE
EMERGENCY_CONTACTS = {
    "Pakistan": {"emergency": "112 / 1122", "police": "15", "ambulance": "1122", "fire": "16"},
    "United States": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "United Kingdom": {"emergency": "999 / 112", "police": "999", "ambulance": "999", "fire": "999"},
    "India": {"emergency": "112", "police": "100", "ambulance": "102", "fire": "101"},
    "Australia": {"emergency": "000", "police": "000", "ambulance": "000", "fire": "000"},
    "Canada": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "Germany": {"emergency": "112", "police": "110", "ambulance": "112", "fire": "112"},
    "France": {"emergency": "112", "police": "17", "ambulance": "15", "fire": "18"},
    "Japan": {"emergency": "110 / 119", "police": "110", "ambulance": "119", "fire": "119"},
    "China": {"emergency": "110 / 120", "police": "110", "ambulance": "120", "fire": "119"},
    "Brazil": {"emergency": "190 / 192", "police": "190", "ambulance": "192", "fire": "193"},
    "Mexico": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "South Africa": {"emergency": "10111", "police": "10111", "ambulance": "10177", "fire": "10177"},
    "Italy": {"emergency": "112", "police": "112", "ambulance": "118", "fire": "115"},
    "Spain": {"emergency": "112", "police": "091", "ambulance": "061", "fire": "080"},
    "Russia": {"emergency": "112", "police": "102", "ambulance": "103", "fire": "101"},
    "Saudi Arabia": {"emergency": "112", "police": "999", "ambulance": "997", "fire": "998"},
    "Turkey": {"emergency": "112", "police": "155", "ambulance": "112", "fire": "110"},
    "Indonesia": {"emergency": "112", "police": "110", "ambulance": "118", "fire": "113"},
    "Nigeria": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    # Default for all other countries
    "Default": {"emergency": "112 (International)", "police": "Local Police", "ambulance": "Local Ambulance", "fire": "Local Fire"}
}

def get_emergency_contacts(country: str) -> dict:
    """Get emergency contacts for a specific country"""
    return EMERGENCY_CONTACTS.get(country, EMERGENCY_CONTACTS["Default"])

# ✅ FIXED: Geocode with proper error handling
def geocode_location(city_or_address: str):
    """Convert city/address to coordinates - Works worldwide!"""
    try:
        params = {
            'q': city_or_address,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
        
        response = requests.get(CONFIG["GEOCODING_API"], params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            address = result.get('address', {})
            
            return {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'city': address.get('city') or address.get('town') or address.get('village') or result.get('display_name', '').split(',')[0],
                'country': address.get('country', 'Unknown'),
                'region': address.get('state') or address.get('region', 'Unknown'),
                'full_address': result.get('display_name', city_or_address),
                'method': 'manual',
                'source': 'Geocoded'
            }
        return None
    except Exception as e:
        st.error(f"❌ Geocoding error: {e}")
        return None

# ✅ FIXED: Reverse geocode from lat/lon
def reverse_geocode(lat: float, lon: float):
    """Convert coordinates to address"""
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
        
        response = requests.get(CONFIG["REVERSE_GEOCODING_API"], params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data and 'address' in data:
            address = data['address']
            return {
                'lat': lat,
                'lon': lon,
                'city': address.get('city') or address.get('town') or address.get('village', 'Unknown'),
                'country': address.get('country', 'Unknown'),
                'region': address.get('state') or address.get('region', 'Unknown'),
                'full_address': data.get('display_name', f"{lat}, {lon}"),
                'method': 'browser',
                'source': 'Browser GPS'
            }
        return None
    except Exception as e:
        st.warning(f"⚠️ Reverse geocoding failed: {e}")
        return None

# ✅ FIXED: IP-based fallback location
def get_ip_location():
    """Get location from IP (fallback only)"""
    try:
        response = requests.get(CONFIG["IPAPI_URL"], timeout=5)
        data = response.json()
        
        if 'error' not in data and 'latitude' in data:
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
    except:
        pass
    
    # Try backup
    try:
        response = requests.get(CONFIG["IPAPI_BACKUP"], timeout=5)
        data = response.json()
        
        if data.get('status') == 'success':
            return {
                'lat': float(data['lat']),
                'lon': float(data['lon']),
                'city': data.get('city', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'region': data.get('regionName', 'Unknown'),
                'ip': data.get('query', 'Unknown'),
                'method': 'ip',
                'source': 'IP Geolocation (Fallback)'
            }
    except:
        pass
    
    # Last resort fallback
    return {
        'lat': 20.0,
        'lon': 0.0,
        'city': 'Unknown',
        'country': 'Unknown',
        'region': 'Unknown',
        'method': 'default',
        'source': 'Default Location'
    }

# ✅ FIXED: Get current location with priority system
def get_current_location():
    """Priority: Browser GPS > Manual > IP Fallback"""
    if st.session_state.get('browser_location'):
        return st.session_state.browser_location
    elif st.session_state.get('manual_location'):
        return st.session_state.manual_location
    elif st.session_state.get('ip_location'):
        return st.session_state.ip_location
    else:
        return get_ip_location()

# ✅ Gemini setup (unchanged)
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

# ✅ FIXED: Fetch disasters worldwide (not just US)
@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=500):  # Increased limit
    try:
        url = f"{CONFIG['EONET_API']}?status={status}&limit={limit}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        disasters = []
        for event in data.get('events', []):
            if event.get('geometry'):
                latest_geo = event['geometry'][-1]
                coords = latest_geo.get('coordinates', [])
                
                if len(coords) >= 2:
                    # Handle both Point and Polygon geometries
                    if latest_geo['type'] == 'Point':
                        lat, lon = coords[1], coords[0]
                    elif latest_geo['type'] == 'Polygon' and len(coords[0]) > 0:
                        lat, lon = coords[0][0][1], coords[0][0][0]
                    else:
                        continue
                    
                    disasters.append({
                        'id': event['id'],
                        'title': event['title'],
                        'category': event['categories'][0]['title'] if event.get('categories') else 'Unknown',
                        'lat': lat,
                        'lon': lon,
                        'date': event.get('geometry')[-1].get('date', 'Unknown'),
                        'source': ', '.join([s['id'] for s in event.get('sources', [])[:3]]),
                        'link': event.get('link', '')
                    })
        
        return pd.DataFrame(disasters)
    except Exception as e:
        st.error(f"❌ Failed to fetch NASA EONET data: {e}")
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

# ✅ FIXED: Population data generation (no ValueError)
def generate_population_data(center_lat, center_lon, radius_deg=2.0, num_points=1000):
    # Use absolute value and modulo to ensure valid seed
    seed_value = abs(int((center_lat * 1000 + center_lon * 1000))) % (2**31)
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

# ✅ FIXED: AI Guidance with location-based option
def get_ai_disaster_guidance(disaster_type: str, user_situation: str, model, use_location: bool = False, location: dict = None) -> str:
    if not model:
        return """⚠️ **AI Not Available** - Please add your Gemini API key in settings."""
    
    try:
        location_context = ""
        emergency_numbers = ""
        
        if use_location and location:
            location_context = f"\n\n**USER LOCATION:** {location['city']}, {location['country']}"
            contacts = get_emergency_contacts(location['country'])
            emergency_numbers = f"""

📞 **EMERGENCY CONTACTS FOR {location['country'].upper()}:**
🚨 Emergency: {contacts['emergency']}
👮 Police: {contacts['police']}
🚑 Ambulance: {contacts['ambulance']}
🚒 Fire: {contacts['fire']}
"""
        
        prompt = f"""You are an emergency disaster response expert. Provide IMMEDIATE, life-saving guidance.

**DISASTER TYPE:** {disaster_type}
**SITUATION:** {user_situation}{location_context}

Provide clear, actionable advice:

🚨 **IMMEDIATE ACTIONS:**
[3-5 critical steps to take RIGHT NOW]

⚠️ **CRITICAL DON'Ts:**
[3-4 dangerous actions to AVOID]

🏃 **EVACUATION CRITERIA:**
[When to leave immediately vs shelter in place]

📦 **ESSENTIAL ITEMS:**
[Critical supplies to gather if possible]

⏰ **URGENCY LEVEL:**
[Immediate (minutes) / Urgent (hours) / Plan (days)]

Be concise and life-saving focused. NO extra commentary."""

        response = model.generate_content(prompt)
        
        return response.text + emergency_numbers
        
    except Exception as e:
        return f"""⚠️ **AI Error:** {str(e)}

**Basic Safety Steps:**
1. 🚨 Call emergency services immediately
2. 🏃 Follow official evacuation orders
3. 📻 Stay informed via local news/radio
4. 🆘 Move to safe location if threatened"""

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

Free tier quota hit. Please wait 2-3 minutes and try again.

Error: {error_msg[:200]}"""
                    }
            else:
                return {'success': False, 'message': f'Analysis failed: {error_msg[:300]}'}
    
    return {'success': False, 'message': 'Max retries exceeded'}

# ========== SESSION STATE INITIALIZATION ==========
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

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## 🌍 AI-RescueMap")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["🗺 Disaster Map", "💬 AI Guidance", "🖼 Image Analysis", "📊 Analytics"])
    
    st.markdown("---")
    st.markdown("### 🎯 Your Location")
    
    # Get current active location
    loc = get_current_location()
    
    if loc:
        # Show location badge
        if loc.get('method') == 'browser':
            badge = "🌐 Browser GPS"
            badge_color = "green"
        elif loc.get('method') == 'manual':
            badge = "📍 Manual Entry"
            badge_color = "blue"
        elif loc.get('method') == 'ip':
            badge = "🌐 IP-Based (Less Accurate)"
            badge_color = "orange"
        else:
            badge = "📍 Default"
            badge_color = "gray"
        
        st.success(f"**{loc['city']}, {loc['region']}**")
        st.info(f"🌍 {loc['country']}")
        st.caption(f"{badge}")
        
        with st.expander("ℹ️ Location Details"):
            st.caption(f"**Coordinates:** {loc['lat']:.4f}, {loc['lon']:.4f}")
            st.caption(f"**Method:** {loc.get('method', 'Unknown').title()}")
            st.caption(f"**Source:** {loc.get('source', 'Unknown')}")
            if loc.get('ip'):
                st.caption(f"**IP:** {loc.get('ip', 'N/A')}")
    else:
        st.error("❌ Location unavailable")
    
    # ✅ BROWSER GPS LOCATION (Real-time with permission)
    st.markdown("---")
    st.markdown("### 🌐 Get My Location")
    st.caption("📍 Uses your device's GPS (browser permission required)")
    
    # JavaScript for browser geolocation
    location_html = """
    <script>
    function getLocation() {
        const button = document.getElementById('gps-btn');
        const status = document.getElementById('gps-status');
        
        if (!navigator.geolocation) {
            status.innerHTML = '❌ Geolocation not supported';
            return;
        }
        
        button.disabled = true;
        button.innerHTML = '📡 Getting location...';
        status.innerHTML = '⏳ Requesting permission...';
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const acc = position.coords.accuracy;
                
                // Send to Streamlit via query params
                const url = new URL(window.location.href);
                url.searchParams.set('gps_lat', lat);
                url.searchParams.set('gps_lon', lon);
                url.searchParams.set('gps_acc', acc);
                url.searchParams.set('gps_timestamp', Date.now());
                window.location.href = url.toString();
            },
            (error) => {
                button.disabled = false;
                button.innerHTML = '📍 Get My Location';
                
                let msg = '❌ ';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        msg += 'Permission denied. Please allow location access.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        msg += 'Location unavailable. Check device settings.';
                        break;
                    case error.TIMEOUT:
                        msg += 'Request timeout. Try again.';
                        break;
                    default:
                        msg += 'Unknown error: ' + error.message;
                }
                status.innerHTML = msg;
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    }
    </script>
    
    <button id="gps-btn" onclick="getLocation()" style="
        width: 100%;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        font-size: 1rem;
    ">📍 Get My Location</button>
    <p id="gps-status" style="margin-top: 0.5rem; font-size: 0.85rem; color: #666;"></p>
    """
    
    st.components.v1.html(location_html, height=120)
    
    # Check for GPS data in query params
    query_params = st.query_params
    if 'gps_lat' in query_params and 'gps_lon' in query_params:
        try:
            gps_lat = float(query_params['gps_lat'])
            gps_lon = float(query_params['gps_lon'])
            gps_acc = float(query_params.get('gps_acc', 0))
            
            # Reverse geocode to get address
            with st.spinner("🌍 Finding your location..."):
                browser_loc = reverse_geocode(gps_lat, gps_lon)
                if browser_loc:
                    st.session_state.browser_location = browser_loc
                    st.success(f"✅ Location detected: {browser_loc['city']}, {browser_loc['country']}")
                    st.caption(f"📍 Accuracy: ~{int(gps_acc)}m")
                    
                    # Clear query params and rerun
                    st.query_params.clear()
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            st.error(f"❌ GPS error: {e}")
            st.query_params.clear()
    
    # Manual location input
    st.markdown("---")
    st.markdown("### 📝 Enter Location Manually")
    st.caption("🌍 Works for any city worldwide")
    
    with st.expander("🔧 Manual Entry"):
        st.info("**Examples:**\n"
                "- Faisalabad, Pakistan\n"
                "- New York, USA\n"
                "- Tokyo, Japan\n"
                "- London, UK")
        
        location_input = st.text_input(
            "City/Country",
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
                            st.session_state.browser_location = None  # Clear GPS
                            st.success(f"✅ Found!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Location not found. Try adding country name.")
        
        with col_btn2:
            if st.session_state.manual_location and st.button("🔄 Reset", use_container_width=True):
                st.session_state.manual_location = None
                st.success("✅ Reset to GPS/IP")
                time.sleep(0.5)
                st.rerun()
    
    # Clear browser location
    if st.session_state.browser_location:
        if st.button("🔄 Clear GPS Location", use_container_width=True):
            st.session_state.browser_location = None
            st.success("✅ GPS location cleared")
            time.sleep(0.5)
            st.rerun()

# ========== MAIN HEADER ==========
st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time global disaster monitoring with NASA data & AI</p>', unsafe_allow_html=True)

# Setup Gemini
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if not st.session_state.gemini_model_text:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if not st.session_state.gemini_model_image:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# ========== DISASTER MAP ==========
if menu == "🗺 Disaster Map":
    with st.spinner("🛰 Fetching NASA EONET data..."):
        disasters = fetch_nasa_eonet_disasters()
    
    # Calculate distances if location available
    if loc and not disasters.empty:
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
        disasters = disasters.sort_values('distance_km')
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌪 Active Disasters", len(disasters))
    with col2:
        if loc and not disasters.empty and 'distance_km' in disasters.columns:
            nearby = len(disasters[disasters['distance_km'] < 500])
            st.metric("📍 Nearby (<500km)", nearby)
        else:
            if not disasters.empty:
                st.metric("🔥 Most Common", disasters['category'].mode()[0] if len(disasters) > 0 else "N/A")
            else:
                st.metric("🔥 Most Common", "N/A")
    with col3:
        st.metric("🤖 AI Status", "✅ Online" if st.session_state.gemini_model_text else "⚠️ Offline")
    with col4:
        st.metric("🛰 Data Source", "NASA EONET")
    
    st.markdown("---")
    
    col_settings, col_map = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### ⚙️ Settings")
        
        # ✅ FIXED: Safe disaster selection
        map_options = ["My Location", "Global View"]
        if not disasters.empty:
            disaster_titles = disasters['title'].tolist()[:10]
            map_options.extend(disaster_titles)
        
        map_center_option = st.selectbox("Center Map", map_options)
        
        # ✅ FIXED: Safe map centering
        if map_center_option == "My Location" and loc:
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
        elif map_center_option == "Global View":
            center_lat, center_lon, zoom = 20, 0, 2
        elif not disasters.empty and map_center_option in disasters['title'].values:
            disaster_row = disasters[disasters['title'] == map_center_option].iloc[0]
            center_lat, center_lon, zoom = disaster_row['lat'], disaster_row['lon'], 8
        else:
            center_lat, center_lon, zoom = 20, 0, 2
        
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
                        'Earthquakes': 'darkred', 'Volcanoes': 'red', 'Sea and Lake Ice': 'lightblue',
                        'Snow': 'white', 'Dust and Haze': 'brown', 'Manmade': 'gray'}
            
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
    
    # Impact Analysis
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
                if len(high_risk) > 0:
                    for _, imp in high_risk.head(5).iterrows():
                        st.markdown(f"""<div class="disaster-alert">
                        ⚠️ <b>{imp['disaster'][:50]}</b><br>
                        👥 {imp['affected_population']:,} people at risk<br>
                        🚨 Risk Level: {imp['risk_level']}</div>""", unsafe_allow_html=True)
                else:
                    st.info("✅ No high-risk events in range")
            
            with col2:
                st.markdown("#### 📈 Statistics")
                st.metric("Total at Risk", f"{impact_df['affected_population'].sum():,}")
                st.metric("Critical Events", len(impact_df[impact_df['risk_level'] == 'CRITICAL']))
                st.metric("High Risk Events", len(impact_df[impact_df['risk_level'] == 'HIGH']))

# ========== AI GUIDANCE ==========
elif menu == "💬 AI Guidance":
    st.markdown("## 💬 AI Emergency Guidance")
    
    # ✅ FIXED: Location-based checkbox
    use_location = st.checkbox(
        "📍 Use my location for context-specific guidance",
        value=False,
        help="Get location-specific emergency numbers and local advice"
    )
    
    if use_location and loc:
        st.info(f"🎯 Using location: **{loc['city']}, {loc['country']}**")
    
    disaster_type = st.selectbox("Disaster Type", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Landslide", "Other"])
    
    user_situation = st.text_area("Describe your situation:",
        placeholder="Be specific: What is happening? Number of people? Current conditions? Available resources?",
        height=120)
    
    if st.button("🚨 GET AI GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("❌ Please describe your situation")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to settings")
        else:
            with st.spinner("🤖 Analyzing with AI..."):
                guidance = get_ai_disaster_guidance(
                    disaster_type, 
                    user_situation, 
                    st.session_state.gemini_model_text,
                    use_location=use_location,
                    location=loc if use_location else None
                )
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)

# ========== IMAGE ANALYSIS ==========
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
                st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to settings")
            else:
                with st.spinner("🤖 Analyzing image..."):
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

# ========== ANALYTICS ==========
elif menu == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    
    if loc:
        view_mode = st.radio("View Mode:", ["📍 My Location", "🌍 Global View"], horizontal=True)
    else:
        view_mode = "🌍 Global View"
        st.info("ℹ️ Location unavailable - showing global view")
    
    with st.spinner("📡 Loading disaster data..."):
        disasters = fetch_nasa_eonet_disasters(limit=500)
    
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
            'Volcanoes': 'red',
            'Sea and Lake Ice': 'lightblue',
            'Snow': 'white',
            'Dust and Haze': 'brown',
            'Manmade': 'gray'
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

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray; font-size: 0.9rem;'>
🌍 <b>AI-RescueMap</b> • Built by <b>HasnainAtif</b> @ NASA Space Apps Challenge 2025<br>
</p>
""", unsafe_allow_html=True)

