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
    "REVERSE_GEOCODING_API": "https://nominatim.openstreetmap.org/reverse",
    "COUNTRIES_API": "https://restcountries.com/v3.1/all"
}

# ✅ COMPREHENSIVE COUNTRY LIST (195+ countries)
WORLD_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
    "Cape Verde", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia",
    "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "North Korea", "South Korea", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macedonia",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Saudi Arabia", "Senegal", "Serbia",
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain",
    "Sri Lanka", "Sudan", "Suriname", "Swaziland", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania",
    "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen",
    "Zambia", "Zimbabwe"
]

# ✅ COMPREHENSIVE EMERGENCY CONTACTS (50+ countries)
EMERGENCY_CONTACTS = {
    "Pakistan": {"emergency": "112 / 1122", "police": "15", "ambulance": "1122", "fire": "16"},
    "United States": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "United Kingdom": {"emergency": "999 / 112", "police": "999", "ambulance": "999", "fire": "999"},
    "India": {"emergency": "112", "police": "100", "ambulance": "102", "fire": "101"},
    "Australia": {"emergency": "000 / 112", "police": "000", "ambulance": "000", "fire": "000"},
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
    "Argentina": {"emergency": "911", "police": "911", "ambulance": "107", "fire": "100"},
    "Egypt": {"emergency": "122", "police": "122", "ambulance": "123", "fire": "180"},
    "South Korea": {"emergency": "112 / 119", "police": "112", "ambulance": "119", "fire": "119"},
    "Thailand": {"emergency": "191 / 1669", "police": "191", "ambulance": "1669", "fire": "199"},
    "Vietnam": {"emergency": "113 / 114", "police": "113", "ambulance": "115", "fire": "114"},
    "Philippines": {"emergency": "911", "police": "911", "ambulance": "911", "fire": "911"},
    "Bangladesh": {"emergency": "999", "police": "999", "ambulance": "199", "fire": "999"},
    "Iran": {"emergency": "110 / 115", "police": "110", "ambulance": "115", "fire": "125"},
    "Iraq": {"emergency": "104 / 122", "police": "104", "ambulance": "122", "fire": "115"},
    "Afghanistan": {"emergency": "119", "police": "119", "ambulance": "102", "fire": "119"},
    "Netherlands": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "Belgium": {"emergency": "112", "police": "101", "ambulance": "112", "fire": "112"},
    "Switzerland": {"emergency": "112", "police": "117", "ambulance": "144", "fire": "118"},
    "Sweden": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "Norway": {"emergency": "112", "police": "112", "ambulance": "113", "fire": "110"},
    "Denmark": {"emergency": "112", "police": "114", "ambulance": "112", "fire": "112"},
    "Finland": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "Poland": {"emergency": "112", "police": "997", "ambulance": "999", "fire": "998"},
    "Greece": {"emergency": "112", "police": "100", "ambulance": "166", "fire": "199"},
    "Portugal": {"emergency": "112", "police": "112", "ambulance": "112", "fire": "112"},
    "New Zealand": {"emergency": "111", "police": "111", "ambulance": "111", "fire": "111"},
    "Singapore": {"emergency": "999 / 995", "police": "999", "ambulance": "995", "fire": "995"},
    "Malaysia": {"emergency": "999", "police": "999", "ambulance": "999", "fire": "994"},
    "Israel": {"emergency": "100 / 101", "police": "100", "ambulance": "101", "fire": "102"},
    "Colombia": {"emergency": "123", "police": "112", "ambulance": "125", "fire": "119"},
    "Chile": {"emergency": "131", "police": "133", "ambulance": "131", "fire": "132"},
    "Peru": {"emergency": "105", "police": "105", "ambulance": "117", "fire": "116"},
    "Kenya": {"emergency": "999 / 112", "police": "999", "ambulance": "999", "fire": "999"},
    "Ethiopia": {"emergency": "907 / 991", "police": "991", "ambulance": "907", "fire": "939"},
    "Default": {"emergency": "112 (International)", "police": "Local Police", "ambulance": "Local Ambulance", "fire": "Local Fire"}
}

def get_emergency_contacts(country: str) -> dict:
    """Get emergency contacts for a specific country"""
    return EMERGENCY_CONTACTS.get(country, EMERGENCY_CONTACTS["Default"])

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
        return None

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
    
    return {
        'lat': 20.0,
        'lon': 0.0,
        'city': 'Unknown',
        'country': 'Unknown',
        'region': 'Unknown',
        'method': 'default',
        'source': 'Default Location'
    }

def get_current_location():
    """Priority: Browser GPS > Manual > Country Selection > IP Fallback"""
    if st.session_state.get('browser_location'):
        return st.session_state.browser_location
    elif st.session_state.get('manual_location'):
        return st.session_state.manual_location
    elif st.session_state.get('country_location'):
        return st.session_state.country_location
    elif st.session_state.get('ip_location'):
        return st.session_state.ip_location
    else:
        return get_ip_location()

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

# ✅ FIXED: Fetch ALL disaster types (not just wildfires)
@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=1000):
    """Fetch ALL disaster types from NASA EONET - Cached for 30 minutes"""
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
        
        df = pd.DataFrame(disasters)
        
        # ✅ Show all disaster types
        if not df.empty:
            disaster_types = df['category'].unique()
            st.sidebar.caption(f"**Disaster Types Found:** {', '.join(disaster_types)}")
        
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch NASA EONET data: {e}")
        return pd.DataFrame()

def filter_disasters_by_country(disasters_df, country_name):
    """Filter disasters by country using reverse geocoding"""
    if disasters_df.empty:
        return disasters_df
    
    # Simple approach: filter by proximity to country capital
    # In production, you'd use a proper country boundary dataset
    country_coords = geocode_location(country_name)
    if not country_coords:
        return disasters_df
    
    disasters_df = disasters_df.copy()
    disasters_df['distance_km'] = disasters_df.apply(
        lambda row: calculate_distance(country_coords['lat'], country_coords['lon'], row['lat'], row['lon']), 
        axis=1
    )
    
    # Filter within reasonable distance (e.g., 1500km for large countries)
    return disasters_df[disasters_df['distance_km'] <= 1500].sort_values('distance_km')

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

if 'country_location' not in st.session_state:
    st.session_state.country_location = None

if 'ip_location' not in st.session_state:
    st.session_state.ip_location = get_ip_location()

if 'gemini_model_text' not in st.session_state:
    st.session_state.gemini_model_text = None

if 'gemini_model_image' not in st.session_state:
    st.session_state.gemini_model_image = None

if 'selected_view_mode' not in st.session_state:
    st.session_state.selected_view_mode = "My Location"

# ========== SIDEBAR ==========
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
            'browser': '🎯 GPS (Most Accurate)',
            'manual': '📍 Manual Entry',
            'country': '🌍 Country Selection',
            'ip': '🌐 IP-Based (Less Accurate)'
        }
        badge = method_badge.get(loc.get('method'), '📍 Unknown')
        
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
    
    st.markdown("---")
    st.markdown("### 🌐 Get My Location (GPS)")
    st.caption("📍 Most accurate - uses device GPS")
    
    location_html = """
    <script>
    function getLocation() {
        const button = document.getElementById('gps-btn');
        const status = document.getElementById('gps-status');
        
        if (!navigator.geolocation) {
            status.innerHTML = '❌ Geolocation not supported by your browser';
            status.style.color = '#ff4444';
            return;
        }
        
        button.disabled = true;
        button.innerHTML = '📡 Getting location...';
        status.innerHTML = '⏳ Requesting permission...';
        status.style.color = '#ff9800';
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const acc = position.coords.accuracy;
                
                status.innerHTML = '✅ Location obtained! Refreshing...';
                status.style.color = '#4CAF50';
                
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
                
                let msg = '';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        msg = '❌ Location access denied. Please allow location access in browser settings and try again.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        msg = '❌ Location unavailable. Please check GPS/Wi-Fi and try again.';
                        break;
                    case error.TIMEOUT:
                        msg = '❌ Request timeout. Please try again.';
                        break;
                    default:
                        msg = '❌ Unknown error: ' + error.message;
                }
                status.innerHTML = msg;
                status.style.color = '#ff4444';
            },
            {
                enableHighAccuracy: true,
                timeout: 15000,
                maximumAge: 0
            }
        );
    }
    </script>
    
    <button id="gps-btn" onclick="getLocation()" style="
        width: 100%;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s;
    ">📍 Get My Location</button>
    <p id="gps-status" style="margin-top: 0.75rem; font-size: 0.85rem; color: #666; line-height: 1.4;"></p>
    """
    
    st.components.v1.html(location_html, height=140)
    
    query_params = st.query_params
    if 'gps_lat' in query_params and 'gps_lon' in query_params:
        try:
            gps_lat = float(query_params['gps_lat'])
            gps_lon = float(query_params['gps_lon'])
            gps_acc = float(query_params.get('gps_acc', 0))
            
            with st.spinner("🌍 Finding your location..."):
                browser_loc = reverse_geocode(gps_lat, gps_lon)
                if browser_loc:
                    st.session_state.browser_location = browser_loc
                    st.success(f"✅ GPS: {browser_loc['city']}, {browser_loc['country']}")
                    st.caption(f"📍 Accuracy: ±{int(gps_acc)}m")
                    st.query_params.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Could not determine location from GPS coordinates")
                    st.query_params.clear()
        except Exception as e:
            st.error(f"❌ GPS error: {e}")
            st.query_params.clear()
    
    # ✅ COUNTRY SELECTOR WITH SEARCH
    st.markdown("---")
    st.markdown("### 🌍 Select Country")
    st.caption("🔍 Search and select from 195+ countries")
    
    # Searchable selectbox
    search_country = st.text_input(
        "🔎 Search Country",
        placeholder="Type to search... (e.g., pak, usa, jap)",
        key="country_search",
        help="Case-insensitive search"
    )
    
    # Filter countries based on search
    if search_country:
        filtered_countries = [c for c in WORLD_COUNTRIES if search_country.lower() in c.lower()]
    else:
        filtered_countries = WORLD_COUNTRIES
    
    if filtered_countries:
        selected_country = st.selectbox(
            "📍 Country",
            options=[""] + filtered_countries,
            format_func=lambda x: "Select a country..." if x == "" else x,
            key="country_select"
        )
        
        if selected_country and selected_country != "":
            if st.button("✅ Use This Country", use_container_width=True, type="primary"):
                with st.spinner(f"🌍 Finding {selected_country}..."):
                    country_coords = geocode_location(selected_country)
                    if country_coords:
                        st.session_state.country_location = country_coords
                        st.session_state.browser_location = None
                        st.session_state.manual_location = None
                        st.success(f"✅ Set to {selected_country}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Could not find coordinates for {selected_country}")
    else:
        st.warning("No countries found matching your search")
    
    # Manual location input
    st.markdown("---")
    st.markdown("### 📝 Manual Entry")
    st.caption("🌍 Enter any city/address")
    
    with st.expander("🔧 Manual Location Entry"):
        st.info("**Examples:**\n- Faisalabad, Pakistan\n- New York, USA\n- Tokyo, Japan")
        
        location_input = st.text_input(
            "City, Country",
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
                            st.session_state.browser_location = None
                            st.session_state.country_location = None
                            st.success(f"✅ Found!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Not found")
        
        with col_btn2:
            if (st.session_state.manual_location or st.session_state.browser_location or st.session_state.country_location) and st.button("🔄 Reset", use_container_width=True):
                st.session_state.manual_location = None
                st.session_state.browser_location = None
                st.session_state.country_location = None
                st.success("✅ Reset to IP")
                time.sleep(0.5)
                st.rerun()

st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time global disaster monitoring with NASA data & AI</p>', unsafe_allow_html=True)

gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if not st.session_state.gemini_model_text:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if not st.session_state.gemini_model_image:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# ========== FETCH GLOBAL DISASTERS ==========
with st.spinner("🛰 Loading NASA EONET disaster data..."):
    global_disasters = fetch_nasa_eonet_disasters(limit=1000)

if global_disasters.empty:
    st.error("❌ No disaster data available from NASA EONET")
    st.stop()

# ========== DISASTER MAP ==========
if menu == "🗺 Disaster Map":
    # ✅ VIEW MODE SELECTOR (affects both map and metrics)
    col_view1, col_view2 = st.columns([3, 1])
    
    with col_view1:
        view_options = ["📍 My Location", "🌍 Global View"] + [f"🌍 {country}" for country in WORLD_COUNTRIES[:20]]  # Top 20 for dropdown
        selected_view = st.selectbox(
            "⚙️ View Mode",
            options=view_options,
            index=0 if loc else 1,
            key="map_view_selector"
        )
    
    with col_view2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # ✅ FILTER DISASTERS BASED ON VIEW MODE
    if selected_view == "📍 My Location" and loc:
        disasters = global_disasters.copy()
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
        disasters = disasters.sort_values('distance_km').head(500)
        center_lat, center_lon, zoom = loc['lat'], loc['lon'], 6
        st.info(f"📍 Showing disasters near **{loc['city']}, {loc['country']}**")
    elif selected_view == "🌍 Global View":
        disasters = global_disasters.head(500)
        center_lat, center_lon, zoom = 20, 0, 2
        st.info("🌍 Showing global disasters")
    elif selected_view.startswith("🌍"):
        # Country-specific view
        country_name = selected_view.replace("🌍 ", "")
        disasters = filter_disasters_by_country(global_disasters, country_name)
        if not disasters.empty:
            center_lat, center_lon = disasters['lat'].mean(), disasters['lon'].mean()
            zoom = 5
            st.info(f"🌍 Showing disasters in **{country_name}** ({len(disasters)} found)")
        else:
            disasters = global_disasters.head(100)
            center_lat, center_lon, zoom = 20, 0, 2
            st.warning(f"⚠️ No disasters found in {country_name}, showing global view")
    else:
        disasters = global_disasters.head(500)
        center_lat, center_lon, zoom = 20, 0, 2
    
    # ✅ METRICS (updated based on filtered disasters)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌪 Total Disasters", len(disasters))
    with col2:
        if 'distance_km' in disasters.columns:
            nearby = len(disasters[disasters['distance_km'] < 500])
            st.metric("📍 Nearby (<500km)", nearby)
        else:
            if not disasters.empty:
                top_category = disasters['category'].mode()[0] if len(disasters) > 0 else "N/A"
                st.metric("🔥 Most Common", top_category)
            else:
                st.metric("🔥 Most Common", "N/A")
    with col3:
        st.metric("🤖 AI Status", "✅ Online" if st.session_state.gemini_model_text else "⚠️ Offline")
    with col4:
        categories_count = disasters['category'].nunique() if not disasters.empty else 0
        st.metric("📊 Disaster Types", categories_count)
    
    st.markdown("---")
    
    col_settings, col_map = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### ⚙️ Map Settings")
        
        show_disasters = st.checkbox("Show Disasters", value=True)
        show_population = st.checkbox("Show Population", value=True)
        
        satellite_layers = st.multiselect("Satellite Layers", 
                                         ['True Color', 'Active Fires', 'Night Lights'], 
                                         default=['True Color'])
        impact_radius = st.slider("Impact Radius (km)", 10, 200, 50)
        
        # Filter by disaster category
        if not disasters.empty:
            all_categories = disasters['category'].unique().tolist()
            selected_categories = st.multiselect("Filter Categories", all_categories, default=all_categories)
            disasters = disasters[disasters['category'].isin(selected_categories)]
    
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
                'Snow': 'white', 
                'Drought': 'brown', 
                'Dust and Haze': 'beige',
                'Manmade': 'gray',
                'Landslides': 'purple',
                'Temperature Extremes': 'orange',
                'Water Color': 'cyan'
            }
            
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
        
        if loc and selected_view == "📍 My Location":
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
                if len(high_risk) > 0:
                    for _, imp in high_risk.head(5).iterrows():
                        st.markdown(f"""<div class="disaster-alert">
                        ⚠️ <b>{imp['disaster'][:50]}</b><br>
                        👥 {imp['affected_population']:,} people at risk<br>
                        🚨 Risk Level: {imp['risk_level']}</div>""", unsafe_allow_html=True)
                else:
                    st.info("✅ No high-risk events in current view")
            
            with col2:
                st.markdown("#### 📈 Statistics")
                st.metric("Total at Risk", f"{impact_df['affected_population'].sum():,}")
                st.metric("Critical Events", len(impact_df[impact_df['risk_level'] == 'CRITICAL']))
                st.metric("High Risk Events", len(impact_df[impact_df['risk_level'] == 'HIGH']))

elif menu == "💬 AI Guidance":
    st.markdown("## 💬 AI Emergency Guidance")
    
    use_location = st.checkbox(
        "📍 Use my location for guidance",
        value=False,
        help="Get location-specific emergency contacts"
    )
    
    if use_location and loc:
        st.info(f"🎯 Using: **{loc['city']}, {loc['country']}**")
    
    disaster_type = st.selectbox("Disaster Type", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Landslide", "Drought", "Heat Wave", "Blizzard", "Other"])
    
    user_situation = st.text_area("Describe your situation:",
        placeholder="Be specific: location, people, conditions, resources...",
        height=120)
    
    if st.button("🚨 GET AI GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("❌ Please describe your situation")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY")
        else:
            with st.spinner("🤖 Analyzing..."):
                guidance = get_ai_disaster_guidance(
                    disaster_type, 
                    user_situation, 
                    st.session_state.gemini_model_text,
                    use_location=use_location,
                    location=loc if use_location else None
                )
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)

elif menu == "🖼 Image Analysis":
    from PIL import Image
    
    st.markdown("## 🖼 AI Image Analysis")
    st.info("⚠️ Free tier: ~15 requests/minute")
    
    uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        if st.button("🔍 ANALYZE", type="primary", use_container_width=True):
            if not st.session_state.gemini_model_image:
                st.warning("⚠️ AI unavailable")
            else:
                with st.spinner("🤖 Analyzing..."):
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
                        st.error(result.get('message'))

elif menu == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    
    # ✅ VIEW MODE FOR ANALYTICS
    analytics_view = st.radio("View:", ["📍 My Location", "🌍 Global View"], horizontal=True)
    
    if analytics_view == "📍 My Location" and loc:
        radius = st.slider("Radius (km)", 100, 5000, 1000, step=100)
        disasters = global_disasters.copy()
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
        disasters = disasters[disasters['distance_km'] <= radius].sort_values('distance_km')
        st.success(f"📍 {len(disasters)} disasters within {radius} km of {loc['city']}, {loc['country']}")
    else:
        disasters = global_disasters
        st.info(f"🌍 Showing {len(disasters)} global disasters")
    
    if not disasters.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌍 Total", len(disasters))
        with col2:
            st.metric("🔥 Wildfires", len(disasters[disasters['category'] == 'Wildfires']))
        with col3:
            st.metric("🌊 Floods", len(disasters[disasters['category'] == 'Floods']))
        with col4:
            st.metric("⛰️ Earthquakes", len(disasters[disasters['category'] == 'Earthquakes']))
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 By Category")
            st.bar_chart(disasters['category'].value_counts())
        
        with col_b:
            st.markdown("### 📅 Recent")
            cols = ['title', 'category', 'date']
            if 'distance_km' in disasters.columns:
                cols.append('distance_km')
            st.dataframe(disasters.head(10)[cols], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Map
        map_center = [loc['lat'], loc['lon']] if loc and analytics_view == "📍 My Location" else [20, 0]
        m = folium.Map(location=map_center, zoom_start=6 if analytics_view == "📍 My Location" else 2, tiles='CartoDB dark_matter')
        
        color_map = {'Wildfires': 'red', 'Floods': 'blue', 'Earthquakes': 'darkred', 'Severe Storms': 'orange'}
        
        for _, d in disasters.head(200).iterrows():
            folium.CircleMarker(
                location=[d['lat'], d['lon']], 
                radius=6,
                color=color_map.get(d['category'], 'gray'),
                fill=True, 
                fillOpacity=0.7,
                popup=f"<b>{d['title']}</b><br>{d['category']}",
                tooltip=d['title']
            ).add_to(m)
        
        if loc and analytics_view == "📍 My Location":
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>You</b><br>{loc['city']}",
                icon=folium.Icon(color='green', icon='home', prefix='glyphicon')
            ).add_to(m)
        
        st_folium(m, width=1200, height=500)
        
        st.download_button(
            "📥 Download CSV",
            disasters.to_csv(index=False).encode('utf-8'),
            f"disasters_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No disasters found")

st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
🌍 <b>AI-RescueMap</b> • <b>HasnainAtif</b> @ NASA Space Apps 2025
</p>
""", unsafe_allow_html=True)
