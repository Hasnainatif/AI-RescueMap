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
from geopy.geocoders import Nominatim, GoogleV3
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

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

# [Keep all your CSS styles here - no changes]
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
    "IPAPI_BACKUP": "http://ip-api.com/json/"
}

# ✅ IMPROVED: Multiple geocoding services with fallbacks
def geocode_location(location_string: str) -> dict:
    """
    Geocode a location string to lat/lon using multiple services.
    Works for ANY city in the world!
    
    Args:
        location_string: City name, "City, Country", or address
        
    Returns:
        dict with lat, lon, city, country, or None if failed
    """
    location_string = location_string.strip()
    
    # Method 1: Try Nominatim (OpenStreetMap) with User-Agent
    try:
        geolocator = Nominatim(user_agent="ai-rescuemap-v1.0", timeout=5)
        location = geolocator.geocode(location_string, language='en')
        
        if location:
            # Parse address components
            raw = location.raw.get('address', {})
            return {
                'lat': location.latitude,
                'lon': location.longitude,
                'city': raw.get('city') or raw.get('town') or raw.get('village') or location_string.split(',')[0],
                'country': raw.get('country', 'Unknown'),
                'region': raw.get('state', 'Unknown'),
                'source': 'Manual (Nominatim)',
                'ip': 'N/A',
                'method': 'manual'
            }
    except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
        st.warning(f"⚠️ Nominatim failed: {str(e)[:100]}")
    
    # Method 2: Try with Photon API (alternative OSM service)
    try:
        photon_url = f"https://photon.komoot.io/api/?q={requests.utils.quote(location_string)}&limit=1"
        response = requests.get(photon_url, timeout=5)
        data = response.json()
        
        if data.get('features'):
            feature = data['features'][0]
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            
            return {
                'lat': coords[1],
                'lon': coords[0],
                'city': props.get('city') or props.get('name', location_string.split(',')[0]),
                'country': props.get('country', 'Unknown'),
                'region': props.get('state', 'Unknown'),
                'source': 'Manual (Photon)',
                'ip': 'N/A',
                'method': 'manual'
            }
    except Exception as e:
        st.warning(f"⚠️ Photon failed: {str(e)[:100]}")
    
    # Method 3: Try with LocationIQ (requires no API key for basic use)
    try:
        locationiq_url = f"https://us1.locationiq.com/v1/search?key=pk.0f147952a41c555c5b70614039fd148b&q={requests.utils.quote(location_string)}&format=json&limit=1"
        response = requests.get(locationiq_url, timeout=5)
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            result = data[0]
            return {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'city': result.get('display_name', '').split(',')[0],
                'country': result.get('display_name', '').split(',')[-1].strip(),
                'region': result.get('display_name', '').split(',')[1] if ',' in result.get('display_name', '') else 'Unknown',
                'source': 'Manual (LocationIQ)',
                'ip': 'N/A',
                'method': 'manual'
            }
    except Exception as e:
        st.warning(f"⚠️ LocationIQ failed: {str(e)[:100]}")
    
    # Method 4: Try OpenCage Geocoder (free tier available)
    try:
        opencage_url = f"https://api.opencagedata.com/geocode/v1/json?q={requests.utils.quote(location_string)}&key=YOUR_OPENCAGE_KEY_HERE&limit=1"
        # Note: You can register for free at https://opencagedata.com/
        # For now, skip this if no key
    except:
        pass
    
    # Method 5: Last resort - try parsing known cities
    KNOWN_CITIES = {
        'faisalabad': {'lat': 31.4504, 'lon': 73.1350, 'city': 'Faisalabad', 'country': 'Pakistan', 'region': 'Punjab'},
        'okara': {'lat': 30.8139, 'lon': 73.4450, 'city': 'Okara', 'country': 'Pakistan', 'region': 'Punjab'},
        'lahore': {'lat': 31.5497, 'lon': 74.3436, 'city': 'Lahore', 'country': 'Pakistan', 'region': 'Punjab'},
        'karachi': {'lat': 24.8607, 'lon': 67.0011, 'city': 'Karachi', 'country': 'Pakistan', 'region': 'Sindh'},
        'islamabad': {'lat': 33.6844, 'lon': 73.0479, 'city': 'Islamabad', 'country': 'Pakistan', 'region': 'ICT'},
        'new york': {'lat': 40.7128, 'lon': -74.0060, 'city': 'New York', 'country': 'USA', 'region': 'New York'},
        'london': {'lat': 51.5074, 'lon': -0.1278, 'city': 'London', 'country': 'UK', 'region': 'England'},
        'tokyo': {'lat': 35.6762, 'lon': 139.6503, 'city': 'Tokyo', 'country': 'Japan', 'region': 'Kanto'},
        'paris': {'lat': 48.8566, 'lon': 2.3522, 'city': 'Paris', 'country': 'France', 'region': 'Île-de-France'},
        'sydney': {'lat': -33.8688, 'lon': 151.2093, 'city': 'Sydney', 'country': 'Australia', 'region': 'NSW'},
        'dubai': {'lat': 25.2048, 'lon': 55.2708, 'city': 'Dubai', 'country': 'UAE', 'region': 'Dubai'},
    }
    
    location_lower = location_string.lower().split(',')[0].strip()
    if location_lower in KNOWN_CITIES:
        result = KNOWN_CITIES[location_lower].copy()
        result['source'] = 'Manual (Database)'
        result['ip'] = 'N/A'
        result['method'] = 'manual'
        return result
    
    return None

@st.cache_data(ttl=3600)
def get_ip_location():
    """Get location from IP address (auto-detection)"""
    try:
        # Try primary API
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
            'org': data.get('org', 'N/A'),
            'source': 'IP Geolocation',
            'method': 'auto'
        }
    except Exception as e:
        # Try backup API
        try:
            alt_response = requests.get(CONFIG["IPAPI_BACKUP"], timeout=5)
            alt_data = alt_response.json()
            
            if alt_data.get('status') != 'success':
                raise Exception(f"Backup failed: {alt_data.get('message')}")
            
            return {
                'lat': float(alt_data['lat']),
                'lon': float(alt_data['lon']),
                'city': alt_data.get('city', 'Unknown'),
                'country': alt_data.get('country', 'Unknown'),
                'region': alt_data.get('regionName', 'Unknown'),
                'ip': alt_data.get('query', 'Unknown'),
                'org': alt_data.get('isp', 'N/A'),
                'source': 'IP Geolocation (Backup)',
                'method': 'auto'
            }
        except Exception as backup_error:
            st.error(f"❌ Location detection failed: {backup_error}")
            return None

def get_user_location():
    """Get user location - checks manual override first, then auto-detects"""
    # Check if user has set manual location
    if st.session_state.get('manual_location'):
        return st.session_state.manual_location
    
    # Otherwise, auto-detect from IP
    return get_ip_location()

# ✅ FIXED: Use correct model name
def setup_gemini(api_key: str = None, model_type: str = "text"):
    if not GEMINI_AVAILABLE:
        return None
    
    key = api_key or st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    
    if key:
        try:
            genai.configure(api_key=key)
            
            # ✅ CORRECT model names
            model_map = {
                "text": "gemini-2.5-pro",
                "image": "gemini-2.5-flash",  # ✅ FIXED: Use Flash, not flash-image
                "chat": "gemini-2.5-flash"
            }
            
            model_name = model_map.get(model_type, "gemini-2.5-pro")
            return genai.GenerativeModel(model_name)
        except Exception as e:
            st.error(f"Gemini setup error: {e}")
            return None
    return None

# [Keep all your other functions exactly as they are - no changes needed]
# fetch_nasa_eonet_disasters, add_nasa_satellite_layers, generate_population_data, 
# calculate_distance, calculate_disaster_impact, get_ai_disaster_guidance, analyze_disaster_image

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

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, model) -> str:
    if not model:
        return """⚠️ **AI Not Available** - Please add your Gemini API key.

**Emergency Contacts:**
- 🚨 Emergency: 911 (US) / 1122 (Pakistan) / 112 (Europe)
- 🆘 FEMA: 1-800-621-3362
- 🔴 Red Cross: 1-800-733-2767"""
    
    try:
        prompt = f"""You are an emergency disaster response expert. Someone needs immediate help.

Disaster: {disaster_type}
Situation: {user_situation}

Provide IMMEDIATE, ACTIONABLE guidance:

🚨 IMMEDIATE ACTIONS:
[List 3-5 specific steps]

⚠️ CRITICAL DON'Ts:
[List 3-4 dangerous actions to avoid]

🏃 EVACUATION CRITERIA:
[When to leave immediately]

📦 ESSENTIAL ITEMS:
[Critical supplies to gather]

⏰ URGENCY LEVEL:
[Minutes/Hours/Days]

📞 EMERGENCY CONTACTS:
[Specific numbers]

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
2. 🔄 Use fewer requests
3. 💳 Upgrade at https://ai.google.dev

Error: {error_msg[:200]}"""
                    }
            else:
                return {'success': False, 'message': f'Analysis failed: {error_msg[:300]}'}
    
    return {'success': False, 'message': 'Max retries exceeded'}

# ✅ Initialize session state
if 'location' not in st.session_state:
    st.session_state.location = get_ip_location()

if 'manual_location' not in st.session_state:
    st.session_state.manual_location = None

if 'gemini_model_text' not in st.session_state:
    st.session_state.gemini_model_text = None

if 'gemini_model_image' not in st.session_state:
    st.session_state.gemini_model_image = None

# ✅ IMPROVED SIDEBAR with manual location
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## 🌍 AI-RescueMap")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["🗺 Disaster Map", "💬 AI Guidance", "🖼 Image Analysis", "📊 Analytics"])
    
    st.markdown("---")
    st.markdown("### 🎯 Your Location")
    
    loc = get_user_location()
    
    if loc:
        # Show current location
        if loc.get('method') == 'manual':
            st.success(f"📍 **{loc['city']}**")
            st.caption(f"🌍 {loc['country']} | ✏️ Manually Set")
        else:
            st.info(f"**{loc['city']}, {loc['region']}**")
            st.caption(f"🌍 {loc['country']}")
            st.caption(f"🌐 Auto-detected | Source: {loc.get('source', 'Unknown')}")
        
        with st.expander("ℹ️ Details"):
            st.caption(f"**Lat/Lon:** {loc['lat']:.4f}, {loc['lon']:.4f}")
            if loc.get('ip') != 'N/A':
                st.caption(f"**IP:** {loc.get('ip', 'N/A')}")
            if loc.get('org'):
                st.caption(f"**ISP:** {loc.get('org', 'N/A')}")
        
        # Show cloud server warning if detected
        if loc.get('method') == 'auto' and loc.get('org'):
            cloud_indicators = ['google', 'amazon', 'microsoft', 'cloud', 'hosting', 'datacenter']
            if any(ind in loc.get('org', '').lower() for ind in cloud_indicators):
                st.warning("☁️ **Cloud Server Detected**\n\nThis is the server's location (not yours). Set your manual location below for accurate results.")
    else:
        st.error("❌ Location unavailable")
    
    # ✅ MANUAL LOCATION INPUT
    st.markdown("---")
    st.markdown("### 📍 Set Manual Location")
    st.caption("🌍 Works for ANY city in the world!")
    
    with st.expander("🔧 Enter Your Location", expanded=False):
        st.markdown("""
**Examples:**
- Faisalabad, Pakistan
- New York, USA
- Tokyo, Japan
- London, UK
- Sydney, Australia
        """)
        
        manual_input = st.text_input(
            "City/Country",
            placeholder="e.g., Okara, Pakistan",
            help="Enter city name or 'City, Country' format"
        )
        
        if st.button("🔍 Find Location", use_container_width=True):
            if manual_input:
                with st.spinner("🌍 Searching location..."):
                    geocoded = geocode_location(manual_input)
                    
                    if geocoded:
                        st.session_state.manual_location = geocoded
                        st.success(f"✅ Found: **{geocoded['city']}, {geocoded['country']}**")
                        st.caption(f"Coordinates: {geocoded['lat']:.4f}, {geocoded['lon']:.4f}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"❌ Could not find '{manual_input}'. Try:\n- Adding country name\n- Checking spelling\n- Using English name")
            else:
                st.warning("Please enter a location")
        
        # Reset button
        if st.session_state.manual_location:
            if st.button("🔄 Use Auto-Detection", use_container_width=True):
                st.session_state.manual_location = None
                st.session_state.location = get_ip_location()
                st.success("✅ Switched to auto-detection")
                time.sleep(0.5)
                st.rerun()
    
    # Refresh button
    if st.button("🔄 Refresh Location", use_container_width=True):
        with st.spinner("📡 Detecting location..."):
            st.cache_data.clear()
            if not st.session_state.manual_location:
                st.session_state.location = get_ip_location()
            time.sleep(0.5)
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time disaster monitoring with NASA data & Google Gemini 2.5 AI</p>', unsafe_allow_html=True)

# Setup Gemini models
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if st.session_state.gemini_model_text is None:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if st.session_state.gemini_model_image is None:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# [KEEP ALL YOUR MENU CODE EXACTLY AS IS]
# Your Disaster Map, AI Guidance, Image Analysis, and Analytics sections remain unchanged

# [Copy the rest of your menu code here - it's working perfectly!]
# I'm just showing the key changes above
