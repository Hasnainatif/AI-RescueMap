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

# ✅ EMERGENCY CONTACTS DATABASE (Country-based)
EMERGENCY_CONTACTS = {
    "Pakistan": {"emergency": "1122 / 115", "police": "15", "ambulance": "1122", "fire": "16"},
    "United States": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "United Kingdom": {"emergency": "999 / 112", "police": "999", "ambulance": "999", "fire": "999"},
    "India": {"emergency": "112", "police": "100", "ambulance": "102", "fire": "101"},
    "Canada": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "Australia": {"emergency": "000 / 112", "police": "000", "ambulance": "000", "fire": "000"},
    "Germany": {"emergency": "112", "police": "110", "ambulance": "112", "fire": "112"},
    "France": {"emergency": "112", "police": "17", "ambulance": "15", "fire": "18"},
    "Japan": {"emergency": "119 (Fire/Ambulance) / 110 (Police)", "police": "110", "ambulance": "119", "fire": "119"},
    "China": {"emergency": "110 (Police) / 120 (Ambulance)", "police": "110", "ambulance": "120", "fire": "119"},
    "Brazil": {"emergency": "190 (Police) / 192 (Ambulance)", "police": "190", "ambulance": "192", "fire": "193"},
    "Mexico": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "South Africa": {"emergency": "10111 (Police) / 10177 (Ambulance)", "police": "10111", "ambulance": "10177", "fire": "10177"},
    "Russia": {"emergency": "112", "police": "102", "ambulance": "103", "fire": "101"},
    "Turkey": {"emergency": "112", "police": "155", "ambulance": "112", "fire": "110"},
    "Italy": {"emergency": "112", "police": "112", "ambulance": "118", "fire": "115"},
    "Spain": {"emergency": "112", "police": "091", "ambulance": "061", "fire": "080"},
    "Saudi Arabia": {"emergency": "999 (Police) / 997 (Ambulance)", "police": "999", "ambulance": "997", "fire": "998"},
    "UAE": {"emergency": "999 (Police) / 998 (Ambulance)", "police": "999", "ambulance": "998", "fire": "997"},
    "Netherlands": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "Switzerland": {"emergency": "112", "police": "117", "ambulance": "144", "fire": "118"},
    "Sweden": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "Norway": {"emergency": "112", "police": "112", "ambulance": "113", "fire": "110"},
    "Denmark": {"emergency": "112", "police": "114", "ambulance": "112", "fire": "112"},
    "Finland": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "New Zealand": {"emergency": "111", "police": "111", "ambulance": "111", "fire": "111"},
    "Singapore": {"emergency": "999 (Police) / 995 (Ambulance)", "police": "999", "ambulance": "995", "fire": "995"},
    "Malaysia": {"emergency": "999", "police": "999", "ambulance": "999", "fire": "994"},
    "Philippines": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "Thailand": {"emergency": "191 (Police) / 1669 (Ambulance)", "police": "191", "ambulance": "1669", "fire": "199"},
    "Indonesia": {"emergency": "110 (Police) / 118 (Ambulance)", "police": "110", "ambulance": "118", "fire": "113"},
    "Vietnam": {"emergency": "113 (Police) / 115 (Ambulance)", "police": "113", "ambulance": "115", "fire": "114"},
    "Bangladesh": {"emergency": "999", "police": "999", "ambulance": "999", "fire": "999"},
    "Afghanistan": {"emergency": "119", "police": "119", "ambulance": "102", "fire": "119"},
    "Iran": {"emergency": "110 (Police) / 115 (Ambulance)", "police": "110", "ambulance": "115", "fire": "125"},
    "Iraq": {"emergency": "104 (Police) / 122 (Ambulance)", "police": "104", "ambulance": "122", "fire": "115"},
    "Egypt": {"emergency": "122 (Police) / 123 (Ambulance)", "police": "122", "ambulance": "123", "fire": "180"},
    "Nigeria": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "Kenya": {"emergency": "999 / 112", "police": "999", "ambulance": "999", "fire": "999"},
    "DEFAULT": {"emergency": "112 (International)", "police": "Local Police", "ambulance": "Local Ambulance", "fire": "Local Fire"}
}

def get_emergency_contacts(country: str) -> dict:
    """Get emergency contacts for a specific country"""
    return EMERGENCY_CONTACTS.get(country, EMERGENCY_CONTACTS["DEFAULT"])

def reverse_geocode(lat: float, lon: float):
    """Get city/country from coordinates"""
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }
        headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
        response = requests.get(CONFIG["REVERSE_GEOCODING_API"], params=params, headers=headers, timeout=5)
        data = response.json()
        
        address = data.get('address', {})
        return {
            'city': address.get('city') or address.get('town') or address.get('village') or 'Unknown',
            'country': address.get('country', 'Unknown'),
            'region': address.get('state') or address.get('province') or 'Unknown',
            'full_address': data.get('display_name', 'Unknown')
        }
    except:
        return {'city': 'Unknown', 'country': 'Unknown', 'region': 'Unknown', 'full_address': 'Unknown'}

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
            return {
                'lat': float(result['lat']),
                'lon': float(result['lon']),
                'city': result.get('display_name', city_or_address).split(',')[0],
                'country': result.get('display_name', '').split(',')[-1].strip(),
                'region': result.get('display_name', '').split(',')[1].strip() if ',' in result.get('display_name', '') else 'Unknown',
                'full_address': result.get('display_name', city_or_address),
                'method': 'manual',
                'source': 'Geocoded (Manual Entry)'
            }
        else:
            return None
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None

def get_ip_location():
    """Get location from IP address (fallback only)"""
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
            'org': data.get('org', ''),
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
                'org': alt_data.get('isp', ''),
                'method': 'ip',
                'source': 'IP Geolocation (Backup)'
            }
        except Exception as backup_error:
            return None

def get_default_location():
    """Default fallback location"""
    return {
        'lat': 20.0,
        'lon': 0.0,
        'city': 'Global',
        'country': 'World',
        'region': 'Global',
        'method': 'default',
        'source': 'Default (Global View)'
    }

def get_current_location():
    """Priority: Browser GPS > Manual Entry > IP > Default"""
    if st.session_state.browser_location:
        return st.session_state.browser_location  # Highest priority
    elif st.session_state.manual_location:
        return st.session_state.manual_location   # Second priority
    elif st.session_state.ip_location:
        return st.session_state.ip_location       # Third priority
    else:
        return get_default_location()             # Fallback

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

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, user_location: dict, model) -> str:
    if not model:
        contacts = get_emergency_contacts(user_location.get('country', 'DEFAULT'))
        return f"""⚠️ **AI Not Available** - Please add your Gemini API key.

**Emergency Contacts for {user_location.get('country', 'your location')}:**
- 🚨 Emergency: {contacts['emergency']}
- 👮 Police: {contacts['police']}
- 🚑 Ambulance: {contacts['ambulance']}
- 🚒 Fire: {contacts['fire']}"""
    
    try:
        contacts = get_emergency_contacts(user_location.get('country', 'DEFAULT'))
        
        prompt = f"""You are an emergency disaster response expert providing life-saving guidance.

**Context:**
- Disaster Type: {disaster_type}
- User's Situation: {user_situation}
- User's Location: {user_location.get('city', 'Unknown')}, {user_location.get('country', 'Unknown')}

Provide IMMEDIATE, ACTIONABLE, LOCATION-SPECIFIC guidance:

🚨 **IMMEDIATE ACTIONS:**
[List 3-5 specific steps tailored to their location and disaster type]

⚠️ **CRITICAL DON'Ts:**
[List 3-4 dangerous actions to avoid]

🏃 **EVACUATION CRITERIA:**
[When to evacuate immediately vs. shelter in place]

📦 **ESSENTIAL ITEMS:**
[Critical supplies to gather NOW]

⏰ **URGENCY LEVEL:**
[Immediate (minutes) / Urgent (hours) / Prepare (days)]

🌍 **LOCAL CONSIDERATIONS:**
[Specific to {user_location.get('country', 'their region')} - local infrastructure, climate, resources]

Keep it clear, specific, and life-saving focused. Do NOT include generic emergency numbers - they will be provided separately."""

        response = model.generate_content(prompt)
        
        # Add location-specific emergency contacts
        contacts_section = f"""

---

📞 **Emergency Contacts for {user_location.get('country', 'Your Location')}:**
- 🚨 **Emergency Services:** {contacts['emergency']}
- 👮 **Police:** {contacts['police']}
- 🚑 **Ambulance:** {contacts['ambulance']}
- 🚒 **Fire Department:** {contacts['fire']}

💡 **Important:** Save these numbers in your phone NOW if you haven't already."""

        return response.text + contacts_section
        
    except Exception as e:
        contacts = get_emergency_contacts(user_location.get('country', 'DEFAULT'))
        return f"""⚠️ **AI Error:** {str(e)}

**Basic Safety Steps for {disaster_type}:**
1. Call emergency services immediately: {contacts['emergency']}
2. Follow official evacuation orders from local authorities
3. Move to the safest available location
4. Stay informed via local news and official channels

**Emergency Contacts for {user_location.get('country', 'your location')}:**
- 🚨 Emergency: {contacts['emergency']}
- 👮 Police: {contacts['police']}
- 🚑 Ambulance: {contacts['ambulance']}
- 🚒 Fire: {contacts['fire']}"""

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

# ✅ Initialize session state
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

# ✅ SIDEBAR with GPS + Manual + IP fallback
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
        # Display location info
        if loc['method'] == 'browser':
            location_badge = "📡 GPS-Enabled (Most Accurate)"
            badge_color = "success"
        elif loc['method'] == 'manual':
            location_badge = "📍 Manual Entry"
            badge_color = "info"
        elif loc['method'] == 'ip':
            location_badge = "🌍 IP-Based (Less Accurate)"
            badge_color = "warning"
        else:
            location_badge = "🌐 Default View"
            badge_color = "secondary"
        
        if loc['method'] != 'default':
            st.success(f"**📍 {loc['city']}, {loc['region']}**")
            st.info(f"🌍 {loc['country']}")
            
            if loc['method'] == 'browser':
                st.success(location_badge)
            elif loc['method'] == 'manual':
                st.info(location_badge)
            else:
                st.warning(location_badge)
            
            with st.expander("ℹ️ Location Details"):
                st.caption(f"**Coordinates:** {loc['lat']:.4f}, {loc['lon']:.4f}")
                st.caption(f"**Method:** {loc['method'].upper()}")
                st.caption(f"**Source:** {loc.get('source', 'Unknown')}")
                if loc.get('ip'):
                    st.caption(f"**IP:** {loc.get('ip', 'N/A')}")
                if loc.get('full_address'):
                    st.caption(f"**Address:** {loc.get('full_address', 'N/A')[:100]}")
        else:
            st.info("🌍 Showing Global View")
            st.caption("Use GPS or manual entry for precise location")
    
    st.markdown("---")
    
    # ✅ GPS LOCATION (Browser-based - highest accuracy)
    st.markdown("### 📡 Get My Location (GPS)")
    st.caption("🎯 Uses your device's GPS for highest accuracy")
    
    # JavaScript for browser geolocation
    gps_html = """
    <div>
        <button id="getGPS" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin-bottom: 10px;
        ">📡 Get My Location (GPS)</button>
        <p id="gpsStatus" style="color: #666; font-size: 12px;"></p>
    </div>
    <script>
    const btn = document.getElementById('getGPS');
    const status = document.getElementById('gpsStatus');
    
    btn.onclick = () => {
        if (!navigator.geolocation) {
            status.innerText = '❌ Geolocation not supported by your browser';
            status.style.color = '#ff4b4b';
            return;
        }
        
        status.innerText = '📡 Requesting location permission...';
        status.style.color = '#667eea';
        btn.disabled = true;
        
        navigator.geolocation.getCurrentPosition(
            pos => {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                const acc = pos.coords.accuracy;
                
                // Reload with GPS coordinates
                const url = new URL(window.location.href);
                url.searchParams.set('gps_lat', lat);
                url.searchParams.set('gps_lon', lon);
                url.searchParams.set('gps_acc', acc);
                url.searchParams.set('gps_time', Date.now());
                window.location.href = url.toString();
            },
            err => {
                status.innerText = '❌ ' + err.message;
                status.style.color = '#ff4b4b';
                btn.disabled = false;
            },
            { 
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    };
    </script>
    """
    
    st.components.v1.html(gps_html, height=100)
    
    # ✅ Process GPS coordinates from URL
    query_params = st.query_params
    if 'gps_lat' in query_params and 'gps_lon' in query_params:
        try:
            gps_lat = float(query_params['gps_lat'])
            gps_lon = float(query_params['gps_lon'])
            gps_acc = float(query_params.get('gps_acc', 0))
            
            # Reverse geocode to get city/country
            geo_info = reverse_geocode(gps_lat, gps_lon)
            
            st.session_state.browser_location = {
                'lat': gps_lat,
                'lon': gps_lon,
                'city': geo_info['city'],
                'country': geo_info['country'],
                'region': geo_info['region'],
                'full_address': geo_info['full_address'],
                'accuracy': gps_acc,
                'method': 'browser',
                'source': f'Browser GPS (±{gps_acc:.0f}m accuracy)'
            }
            
            # Clear query params and rerun
            st.query_params.clear()
            st.success(f"✅ GPS location detected: {geo_info['city']}, {geo_info['country']}")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to process GPS data: {e}")
    
    st.markdown("---")
    
    # ✅ MANUAL LOCATION ENTRY
    st.markdown("### 📍 Manual Location Entry")
    st.caption("🌍 Enter any city worldwide")
    
    with st.expander("🔧 Enter Location Manually"):
        st.info("**Examples:**\n"
                "- Faisalabad, Pakistan\n"
                "- New York, USA\n"
                "- Tokyo, Japan\n"
                "- London, UK")
        
        location_input = st.text_input(
            "City, Country",
            value="",
            placeholder="e.g., Karachi, Pakistan",
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
                            st.session_state.browser_location = None  # Clear GPS when manual is used
                            st.success(f"✅ Found: {geocoded['city']}, {geocoded['country']}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Location not found. Try:\n- Full city name\n- Adding country")
        
        with col_btn2:
            if st.session_state.manual_location or st.session_state.browser_location:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.manual_location = None
                    st.session_state.browser_location = None
                    st.success("✅ Reset to IP location")
                    time.sleep(0.5)
                    st.rerun()

# Main header
st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time disaster monitoring with NASA data & AI-powered guidance</p>', unsafe_allow_html=True)

# Setup Gemini models
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if st.session_state.gemini_model_text is None:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if st.session_state.gemini_model_image is None:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# Get current location for all pages
loc = get_current_location()

# ========== DISASTER MAP ==========
if menu == "🗺 Disaster Map":
    with st.spinner("🛰 Fetching real-time NASA EONET data..."):
        disasters = fetch_nasa_eonet_disasters()
    
    if loc and not disasters.empty and loc['method'] != 'default':
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
        disasters = disasters.sort_values('distance_km')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌪 Active Disasters", len(disasters))
    with col2:
        if loc and not disasters.empty and 'distance_km' in disasters.columns and loc['method'] != 'default':
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
        st.markdown("### ⚙️ Map Settings")
        
        map_options = ["My Location", "Global View"]
        if not disasters.empty:
            map_options += disasters['title'].tolist()[:10]
        
        map_center_option = st.selectbox("Center Map On:", map_options)
        
        if map_center_option == "My Location" and loc and loc['method'] != 'default':
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
        elif map_center_option == "Global View":
            center_lat, center_lon, zoom = 20, 0, 2
        elif not disasters.empty and map_center_option in disasters['title'].values:
            disaster_row = disasters[disasters['title'] == map_center_option].iloc[0]
            center_lat, center_lon, zoom = disaster_row['lat'], disaster_row['lon'], 8
        else:
            center_lat, center_lon, zoom = 20, 0, 2
        
        show_disasters = st.checkbox("Show Disasters", value=True)
        show_population = st.checkbox("Show Population Density", value=True)
        
        satellite_layers = st.multiselect("NASA Satellite Layers", 
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
            color_map = {
                'Wildfires': 'red', 
                'Severe Storms': 'orange', 
                'Floods': 'blue', 
                'Earthquakes': 'darkred', 
                'Volcanoes': 'red',
                'Sea and Lake Ice': 'lightblue',
                'Snow': 'lightgray',
                'Drought': 'brown',
                'Dust and Haze': 'beige',
                'Manmade': 'purple',
                'Temperature Extremes': 'darkpurple',
                'Water Color': 'cadetblue'
            }
            
            for _, disaster in disasters.iterrows():
                color = color_map.get(disaster['category'], 'gray')
                distance_text = f"<br>📍 {disaster['distance_km']:.0f} km from you" if 'distance_km' in disaster else ""
                
                folium.Circle(
                    location=[disaster['lat'], disaster['lon']], 
                    radius=impact_radius * 1000,
                    color=color, 
                    fill=True, 
                    fillOpacity=0.1
                ).add_to(m)
                
                folium.Marker(
                    location=[disaster['lat'], disaster['lon']],
                    popup=f"<b>{disaster['title']}</b><br>{disaster['category']}<br>{disaster['date']}{distance_text}",
                    icon=folium.Icon(color=color, icon='warning-sign', prefix='glyphicon'),
                    tooltip=disaster['title']
                ).add_to(marker_cluster)
        
        if loc and loc['method'] != 'default':
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
                if not high_risk.empty:
                    for _, imp in high_risk.iterrows():
                        st.markdown(f"""<div class="disaster-alert">
                        ⚠️ <b>{imp['disaster']}</b><br>
                        👥 {imp['affected_population']:,} people at risk<br>
                        🚨 Risk Level: {imp['risk_level']}</div>""", unsafe_allow_html=True)
                else:
                    st.info("✅ No high-risk events in current view")
            
            with col2:
                st.markdown("#### 📈 Statistics")
                st.metric("Total Population at Risk", f"{impact_df['affected_population'].sum():,}")
                st.metric("Critical Events", len(impact_df[impact_df['risk_level'] == 'CRITICAL']))
                st.metric("High Risk Events", len(impact_df[impact_df['risk_level'] == 'HIGH']))

# ========== AI GUIDANCE ==========
elif menu == "💬 AI Guidance":
    st.markdown("## 💬 AI Emergency Guidance")
    
    if loc and loc['method'] != 'default':
        st.info(f"🎯 **Your Location:** {loc['city']}, {loc['country']}")
    
    disaster_type = st.selectbox("Select Disaster Type:", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Severe Storm", "Drought", "Landslide", "Other"],
        help="Choose the disaster you need guidance for")
    
    user_situation = st.text_area("Describe Your Current Situation:",
        placeholder="Be specific:\n- Exact location and surroundings\n- Number of people with you\n- Current weather/conditions\n- Available resources (food, water, shelter)\n- Any injuries or medical needs\n- Available transportation",
        height=150)
    
    if st.button("🚨 GET AI GUIDANCE NOW", type="primary", use_container_width=True):
        if not user_situation:
            st.error("❌ Please describe your situation in detail")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to Streamlit secrets")
            contacts = get_emergency_contacts(loc.get('country', 'DEFAULT'))
            st.error(f"""**Emergency Contacts for {loc.get('country', 'Your Location')}:**
- 🚨 Emergency: {contacts['emergency']}
- 👮 Police: {contacts['police']}
- 🚑 Ambulance: {contacts['ambulance']}
- 🚒 Fire: {contacts['fire']}""")
        else:
            with st.spinner("🤖 Analyzing your situation with AI..."):
                guidance = get_ai_disaster_guidance(disaster_type, user_situation, loc, st.session_state.gemini_model_text)
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Important Safety Information")
    st.warning("""
    **Always prioritize:**
    1. **Life safety first** - evacuate if authorities order
    2. **Stay informed** - monitor local news and weather
    3. **Have emergency supplies** - water, food, first aid, flashlight
    4. **Know evacuation routes** - plan ahead
    5. **Stay connected** - keep phone charged, have backup power
    """)

# ========== IMAGE ANALYSIS ==========
elif menu == "🖼 Image Analysis":
    from PIL import Image
    
    st.markdown("## 🖼 AI Disaster Image Analysis")
    st.info("⚠️ **Note:** Free tier has rate limits (~15 requests/minute). Wait if quota is exceeded.")
    
    uploaded_file = st.file_uploader("📤 Upload Disaster Image", type=['jpg', 'jpeg', 'png'], 
                                     help="Upload an image of the disaster for AI analysis")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 ANALYZE IMAGE WITH AI", type="primary", use_container_width=True):
            if not st.session_state.gemini_model_image:
                st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to Streamlit secrets")
            else:
                with st.spinner("🤖 Analyzing image with Gemini AI..."):
                    result = analyze_disaster_image(image, st.session_state.gemini_model_image)
                    
                    if result['success']:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            severity_color = "🔴" if result['severity_level'] == 'CRITICAL' else "🟠" if result['severity_level'] == 'HIGH' else "🟡"
                            st.metric("Severity Level", f"{severity_color} {result['severity_level']}")
                        with col_b:
                            st.metric("Risk Score", f"{result['severity_score']}/100")
                        with col_c:
                            st.metric("Analysis Status", "✅ Complete")
                        
                        st.markdown(f'<div class="ai-response">{result["analysis"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(result.get('message', 'Analysis failed'))

# ========== ANALYTICS ==========
elif menu == "📊 Analytics":
    st.markdown("## 📊 Real-Time Disaster Analytics")
    
    if loc and loc['method'] != 'default':
        view_mode = st.radio("📍 View Mode:", ["📍 My Location (Recommended)", "🌍 Global View"], horizontal=True)
    else:
        view_mode = "🌍 Global View"
        st.info("ℹ️ No specific location detected - showing global view")
    
    with st.spinner("🛰 Loading real-time disaster data from NASA EONET..."):
        disasters = fetch_nasa_eonet_disasters(limit=100)
    
    if not disasters.empty and loc and loc['method'] != 'default':
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
    
    if "My Location" in view_mode and loc and loc['method'] != 'default' and not disasters.empty:
        radius_filter = st.slider("🌍 Show disasters within (km):", 100, 5000, 1000, step=100)
        filtered_disasters = disasters[disasters['distance_km'] <= radius_filter].copy()
        st.success(f"📍 Showing **{len(filtered_disasters)}** disasters within **{radius_filter} km** of **{loc['city']}, {loc['country']}**")
    else:
        filtered_disasters = disasters
        st.info(f"🌍 Showing all **{len(filtered_disasters)}** active disasters worldwide")
    
    if not filtered_disasters.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌍 Total Active Disasters", len(filtered_disasters))
        with col2:
            wildfires = len(filtered_disasters[filtered_disasters['category'] == 'Wildfires'])
            st.metric("🔥 Wildfires", wildfires)
        with col3:
            storms = len(filtered_disasters[filtered_disasters['category'] == 'Severe Storms'])
            st.metric("🌪 Severe Storms", storms)
        with col4:
            others = len(filtered_disasters[~filtered_disasters['category'].isin(['Wildfires', 'Severe Storms'])])
            st.metric("🌊 Other Events", others)
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 Disasters by Category")
            category_counts = filtered_disasters['category'].value_counts()
            st.bar_chart(category_counts)
        
        with col_b:
            st.markdown("### 📅 Most Recent Events")
            display_cols = ['title', 'category', 'date']
            if 'distance_km' in filtered_disasters.columns and "My Location" in view_mode:
                filtered_disasters['distance_km'] = filtered_disasters['distance_km'].round(0).astype(int)
                display_cols.append('distance_km')
            
            recent = filtered_disasters.head(10)[display_cols]
            st.dataframe(recent, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown(f"### 🗺 {'Local' if 'My Location' in view_mode else 'Global'} Disaster Distribution Map")
        
        map_center = [loc['lat'], loc['lon']] if loc and "My Location" in view_mode and loc['method'] != 'default' else [20, 0]
        map_zoom = 6 if "My Location" in view_mode and loc['method'] != 'default' else 2
        
        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB dark_matter')
        
        color_map = {
            'Wildfires': 'red', 
            'Severe Storms': 'orange', 
            'Floods': 'blue', 
            'Earthquakes': 'darkred',
            'Volcanoes': 'red',
            'Sea and Lake Ice': 'lightblue',
            'Snow': 'lightgray',
            'Drought': 'brown',
            'Dust and Haze': 'beige',
            'Manmade': 'purple',
            'Temperature Extremes': 'darkpurple',
            'Water Color': 'cadetblue'
        }
        
        for _, disaster in filtered_disasters.iterrows():
            popup_text = f"<b>{disaster['title']}</b><br>{disaster['category']}<br>{disaster['date']}"
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
        
        if loc and "My Location" in view_mode and loc['method'] != 'default':
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>📍 You are here</b><br>{loc['city']}, {loc['country']}",
                icon=folium.Icon(color='green', icon='home', prefix='glyphicon'),
                tooltip="Your Location"
            ).add_to(m)
        
        st_folium(m, width=1200, height=500)
        
        st.markdown("---")
        
        st.markdown("### 📋 Complete Disaster List")
        
        col1, col2 = st.columns(2)
        with col1:
            all_categories = filtered_disasters['category'].unique().tolist()
            selected_cat = st.multiselect("🔍 Filter by Category:", all_categories, default=all_categories)
        with col2:
            search = st.text_input("🔎 Search by keyword:", "")
        
        final_filtered = filtered_disasters[filtered_disasters['category'].isin(selected_cat)]
        if search:
            final_filtered = final_filtered[final_filtered['title'].str.contains(search, case=False, na=False)]
        
        display_cols = ['title', 'category', 'date', 'lat', 'lon']
        if 'distance_km' in final_filtered.columns and "My Location" in view_mode:
            display_cols.append('distance_km')
        
        st.dataframe(final_filtered[display_cols], use_container_width=True, hide_index=True, height=400)
        
        st.download_button(
            "📥 Download Data as CSV",
            data=final_filtered.to_csv(index=False).encode('utf-8'),
            file_name=f"disasters_{loc['city'] if loc and loc['method'] != 'default' else 'global'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No disasters found in your selected area. Try:\n- Increasing the search radius\n- Switching to Global View\n- Checking back later for updates")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray; font-size: 14px;'>
<b>AI-RescueMap</b> • Built by <b>HasnainAtif</b> for NASA Space Apps Challenge 2025<br>
Real-time global disaster monitoring • GPS-enabled location tracking
</p>
""", unsafe_allow_html=True)
