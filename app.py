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
from geopy.distance import geodesic

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
    "GEOCODING_API": "https://nominatim.openstreetmap.org/search"
}

# ✅ Comprehensive list of countries with their capitals
COUNTRIES_DATA = {
    "Afghanistan": {"capital": "Kabul", "lat": 34.5553, "lon": 69.2075},
    "Albania": {"capital": "Tirana", "lat": 41.3275, "lon": 19.8187},
    "Algeria": {"capital": "Algiers", "lat": 36.7538, "lon": 3.0588},
    "Argentina": {"capital": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    "Australia": {"capital": "Canberra", "lat": -35.2809, "lon": 149.1300},
    "Austria": {"capital": "Vienna", "lat": 48.2082, "lon": 16.3738},
    "Bangladesh": {"capital": "Dhaka", "lat": 23.8103, "lon": 90.4125},
    "Belgium": {"capital": "Brussels", "lat": 50.8503, "lon": 4.3517},
    "Brazil": {"capital": "Brasília", "lat": -15.8267, "lon": -47.9218},
    "Canada": {"capital": "Ottawa", "lat": 45.4215, "lon": -75.6972},
    "Chile": {"capital": "Santiago", "lat": -33.4489, "lon": -70.6693},
    "China": {"capital": "Beijing", "lat": 39.9042, "lon": 116.4074},
    "Colombia": {"capital": "Bogotá", "lat": 4.7110, "lon": -74.0721},
    "Denmark": {"capital": "Copenhagen", "lat": 55.6761, "lon": 12.5683},
    "Egypt": {"capital": "Cairo", "lat": 30.0444, "lon": 31.2357},
    "Finland": {"capital": "Helsinki", "lat": 60.1695, "lon": 24.9354},
    "France": {"capital": "Paris", "lat": 48.8566, "lon": 2.3522},
    "Germany": {"capital": "Berlin", "lat": 52.5200, "lon": 13.4050},
    "Greece": {"capital": "Athens", "lat": 37.9838, "lon": 23.7275},
    "India": {"capital": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    "Indonesia": {"capital": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    "Iran": {"capital": "Tehran", "lat": 35.6892, "lon": 51.3890},
    "Iraq": {"capital": "Baghdad", "lat": 33.3152, "lon": 44.3661},
    "Ireland": {"capital": "Dublin", "lat": 53.3498, "lon": -6.2603},
    "Israel": {"capital": "Jerusalem", "lat": 31.7683, "lon": 35.2137},
    "Italy": {"capital": "Rome", "lat": 41.9028, "lon": 12.4964},
    "Japan": {"capital": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    "Kenya": {"capital": "Nairobi", "lat": -1.2864, "lon": 36.8172},
    "Malaysia": {"capital": "Kuala Lumpur", "lat": 3.1390, "lon": 101.6869},
    "Mexico": {"capital": "Mexico City", "lat": 19.4326, "lon": -99.1332},
    "Netherlands": {"capital": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    "New Zealand": {"capital": "Wellington", "lat": -41.2865, "lon": 174.7762},
    "Nigeria": {"capital": "Abuja", "lat": 9.0765, "lon": 7.3986},
    "Norway": {"capital": "Oslo", "lat": 59.9139, "lon": 10.7522},
    "Pakistan": {"capital": "Islamabad", "lat": 33.6844, "lon": 73.0479},
    "Philippines": {"capital": "Manila", "lat": 14.5995, "lon": 120.9842},
    "Poland": {"capital": "Warsaw", "lat": 52.2297, "lon": 21.0122},
    "Portugal": {"capital": "Lisbon", "lat": 38.7223, "lon": -9.1393},
    "Russia": {"capital": "Moscow", "lat": 55.7558, "lon": 37.6173},
    "Saudi Arabia": {"capital": "Riyadh", "lat": 24.7136, "lon": 46.6753},
    "Singapore": {"capital": "Singapore", "lat": 1.3521, "lon": 103.8198},
    "South Africa": {"capital": "Pretoria", "lat": -25.7479, "lon": 28.2293},
    "South Korea": {"capital": "Seoul", "lat": 37.5665, "lon": 126.9780},
    "Spain": {"capital": "Madrid", "lat": 40.4168, "lon": -3.7038},
    "Sweden": {"capital": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    "Switzerland": {"capital": "Bern", "lat": 46.9480, "lon": 7.4474},
    "Thailand": {"capital": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    "Turkey": {"capital": "Ankara", "lat": 39.9334, "lon": 32.8597},
    "United Arab Emirates": {"capital": "Abu Dhabi", "lat": 24.4539, "lon": 54.3773},
    "United Kingdom": {"capital": "London", "lat": 51.5074, "lon": -0.1278},
    "United States": {"capital": "Washington DC", "lat": 38.9072, "lon": -77.0369},
    "Vietnam": {"capital": "Hanoi", "lat": 21.0285, "lon": 105.8542}
}

# Emergency contacts database by country
EMERGENCY_CONTACTS = {
    "Pakistan": {
        "emergency": "115",
        "police": "15",
        "ambulance": "1122",
        "fire": "16",
        "rescue": "1122"
    },
    "United States": {
        "emergency": "911",
        "police": "911",
        "ambulance": "911",
        "fire": "911",
        "fema": "1-800-621-3362"
    },
    "India": {
        "emergency": "112",
        "police": "100",
        "ambulance": "102",
        "fire": "101",
        "disaster": "108"
    },
    "United Kingdom": {
        "emergency": "999",
        "police": "999",
        "ambulance": "999",
        "fire": "999"
    },
    "Australia": {
        "emergency": "000",
        "police": "000",
        "ambulance": "000",
        "fire": "000"
    },
    "Canada": {
        "emergency": "911",
        "police": "911",
        "ambulance": "911",
        "fire": "911"
    },
    "Japan": {
        "emergency": "110",
        "police": "110",
        "ambulance": "119",
        "fire": "119"
    },
    "default": {
        "emergency": "112",
        "info": "International emergency number (works in most countries)"
    }
}

# ✅ FIXED: Browser location using HTML/JavaScript component
def get_browser_location_component():
    """Display browser location button with proper error handling"""
    location_html = """
    <div style="padding: 10px; background: #f0f2f6; border-radius: 10px; margin: 10px 0;">
        <button id="getLocationBtn" style="
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        ">📍 Get My Location (GPS)</button>
        <div id="locationStatus" style="margin-top: 10px; font-size: 12px; color: #666;"></div>
    </div>
    
    <script>
    document.getElementById('getLocationBtn').addEventListener('click', function() {
        const statusDiv = document.getElementById('locationStatus');
        const button = this;
        
        if (!navigator.geolocation) {
            statusDiv.innerHTML = '❌ Geolocation not supported by your browser';
            statusDiv.style.color = '#ff4444';
            return;
        }
        
        button.disabled = true;
        button.style.opacity = '0.6';
        statusDiv.innerHTML = '⏳ Requesting location permission...';
        statusDiv.style.color = '#667eea';
        
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const accuracy = position.coords.accuracy;
                
                statusDiv.innerHTML = '✅ Location obtained! Reloading...';
                statusDiv.style.color = '#4CAF50';
                
                // Send data via URL parameters
                const url = new URL(window.location.href);
                url.searchParams.set('gps_lat', lat);
                url.searchParams.set('gps_lon', lon);
                url.searchParams.set('gps_accuracy', accuracy);
                url.searchParams.set('gps_timestamp', Date.now());
                window.location.href = url.toString();
            },
            function(error) {
                button.disabled = false;
                button.style.opacity = '1';
                
                let errorMsg = '';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMsg = '❌ Permission denied. Please allow location in browser settings.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMsg = '❌ Location unavailable. Check GPS/Wi-Fi.';
                        break;
                    case error.TIMEOUT:
                        errorMsg = '❌ Request timeout. Try again.';
                        break;
                    default:
                        errorMsg = '❌ Unknown error: ' + error.message;
                }
                statusDiv.innerHTML = errorMsg;
                statusDiv.style.color = '#ff4444';
            },
            {
                enableHighAccuracy: true,
                timeout: 15000,
                maximumAge: 0
            }
        );
    });
    </script>
    """
    
    st.components.v1.html(location_html, height=120)

# ✅ Reverse geocode coordinates to get address
def reverse_geocode(lat, lon):
    """Convert lat/lon to city/country"""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        
        if 'address' in data:
            address = data['address']
            return {
                'city': address.get('city') or address.get('town') or address.get('village') or 'Unknown',
                'region': address.get('state') or address.get('province') or 'Unknown',
                'country': address.get('country', 'Unknown'),
                'full_address': data.get('display_name', 'Unknown')
            }
    except Exception as e:
        st.warning(f"Reverse geocoding failed: {e}")
    return None

# ✅ Geocode city/country to lat/lon
def geocode_location(city_or_address: str):
    """Convert city name or address to lat/lon"""
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
                'source': 'Geocoded'
            }
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
    return None

# ✅ AUTO: Get location from IP
def get_ip_location():
    """Get location from IP address (fallback)"""
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
                'org': data.get('org', ''),
                'method': 'ip',
                'source': 'IP Geolocation (Fallback)'
            }
    except:
        try:
            alt_response = requests.get(CONFIG["IPAPI_BACKUP"], timeout=5)
            alt_data = alt_response.json()
            
            if alt_data.get('status') == 'success':
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
        except:
            pass
    return None

# ✅ Setup Gemini models
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

# ✅ OPTIMIZED: Fetch NASA EONET disasters (1000 max, cached)
@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=1000):
    """Fetch disasters from NASA EONET API (cached for 30 min)"""
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

# ✅ Filter disasters by proximity
def filter_disasters_by_location(disasters_df, user_lat, user_lon, max_distance_km=1000, max_results=500):
    """Filter disasters within radius and return closest ones"""
    if disasters_df.empty:
        return disasters_df
    
    # Calculate distances
    disasters_df['distance_km'] = disasters_df.apply(
        lambda row: geodesic((user_lat, user_lon), (row['lat'], row['lon'])).km,
        axis=1
    )
    
    # Filter by radius
    filtered = disasters_df[disasters_df['distance_km'] <= max_distance_km].copy()
    
    # Sort by distance and limit
    filtered = filtered.sort_values('distance_km').head(max_results)
    
    return filtered

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
    """Generate synthetic population data around a location"""
    # Use location-based seed for consistency
    seed_value = int(abs(center_lat * 1000 + center_lon * 1000)) % (2**31 - 1)
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

def get_emergency_contacts(country: str) -> dict:
    """Get emergency contacts for a specific country"""
    return EMERGENCY_CONTACTS.get(country, EMERGENCY_CONTACTS["default"])

def format_emergency_contacts(contacts: dict, country: str) -> str:
    """Format emergency contacts as markdown"""
    output = f"\n\n📞 **Emergency Contacts for {country}:**\n\n"
    
    for service, number in contacts.items():
        emoji_map = {
            'emergency': '🚨',
            'police': '👮',
            'ambulance': '🚑',
            'fire': '🚒',
            'rescue': '🆘',
            'fema': '🏛️',
            'disaster': '⚠️',
            'info': 'ℹ️'
        }
        emoji = emoji_map.get(service, '📞')
        output += f"{emoji} **{service.title()}:** {number}\n\n"
    
    return output

def get_ai_disaster_guidance(disaster_type: str, user_situation: str, model, user_location: dict = None, use_location: bool = False) -> str:
    if not model:
        return """⚠️ **AI Not Available** - Please add your Gemini API key.

**Emergency Contacts:**
- 🚨 Emergency: 911 (US) / 1122 (Pakistan) / 112 (Europe)"""
    
    try:
        location_context = ""
        if use_location and user_location:
            location_context = f"\n\nUser Location: {user_location['city']}, {user_location['country']}"
        
        prompt = f"""You are an emergency disaster response expert. Someone needs immediate help.

Disaster: {disaster_type}
Situation: {user_situation}{location_context}

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

Keep it clear and life-saving focused. Do NOT include emergency contact numbers in your response."""

        response = model.generate_content(prompt)
        
        # Add emergency contacts if location is used
        if use_location and user_location:
            contacts = get_emergency_contacts(user_location['country'])
            contact_info = format_emergency_contacts(contacts, user_location['country'])
            return response.text + contact_info
        
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

if 'global_disasters' not in st.session_state:
    st.session_state.global_disasters = None

if 'selected_country' not in st.session_state:
    st.session_state.selected_country = None

# ========== PRIORITY LOCATION SELECTION ==========
def get_current_location():
    """Priority: Browser GPS > Manual Entry > Selected Country > IP Fallback"""
    if st.session_state.browser_location:
        return st.session_state.browser_location
    elif st.session_state.manual_location:
        return st.session_state.manual_location
    elif st.session_state.selected_country:
        country_data = COUNTRIES_DATA[st.session_state.selected_country]
        return {
            'lat': country_data['lat'],
            'lon': country_data['lon'],
            'city': country_data['capital'],
            'country': st.session_state.selected_country,
            'region': country_data['capital'],
            'method': 'country',
            'source': 'Country Selection'
        }
    else:
        return st.session_state.ip_location

loc = get_current_location()

# ✅ Check for GPS data from URL parameters
query_params = st.query_params
if 'gps_lat' in query_params and 'gps_lon' in query_params:
    try:
        gps_lat = float(query_params['gps_lat'])
        gps_lon = float(query_params['gps_lon'])
        gps_accuracy = float(query_params.get('gps_accuracy', 0))
        
        # Reverse geocode to get address
        with st.spinner("🌍 Processing GPS location..."):
            geo_info = reverse_geocode(gps_lat, gps_lon)
            if geo_info:
                st.session_state.browser_location = {
                    'lat': gps_lat,
                    'lon': gps_lon,
                    'city': geo_info['city'],
                    'country': geo_info['country'],
                    'region': geo_info['region'],
                    'full_address': geo_info['full_address'],
                    'accuracy': gps_accuracy,
                    'method': 'browser',
                    'source': f'GPS (±{gps_accuracy:.0f}m accuracy)'
                }
            else:
                st.session_state.browser_location = {
                    'lat': gps_lat,
                    'lon': gps_lon,
                    'city': 'Unknown',
                    'country': 'Unknown',
                    'region': 'Unknown',
                    'accuracy': gps_accuracy,
                    'method': 'browser',
                    'source': f'GPS (±{gps_accuracy:.0f}m accuracy)'
                }
        
        # Clear query params and reload
        st.query_params.clear()
        st.success("✅ GPS location obtained!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing GPS: {e}")
        st.query_params.clear()

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=180)
    st.markdown("## 🌍 AI-RescueMap")
    st.markdown("---")
    
    menu = st.radio("Navigation", ["🗺 Disaster Map", "💬 AI Guidance", "🖼 Image Analysis", "📊 Analytics"])
    
    st.markdown("---")
    st.markdown("### 🎯 Your Location")
    
    if loc:
        location_type_emoji = {
            'browser': '🎯 GPS',
            'manual': '📍 Manual',
            'country': '🌍 Country',
            'ip': '🌐 IP'
        }
        emoji = location_type_emoji.get(loc.get('method', 'ip'), '📍')
        
        st.success(f"**{loc['city']}, {loc.get('region', '')}**")
        st.info(f"🌍 {loc['country']}")
        st.caption(f"{emoji} | {loc.get('source', 'Unknown')}")
        
        with st.expander("ℹ️ Location Details"):
            st.caption(f"**Coordinates:** {loc['lat']:.4f}, {loc['lon']:.4f}")
            st.caption(f"**Method:** {loc.get('method', 'unknown').title()}")
            if loc.get('accuracy'):
                st.caption(f"**Accuracy:** ±{loc['accuracy']:.0f}m")
            if loc.get('ip'):
                st.caption(f"**IP:** {loc['ip']}")
    else:
        st.error("❌ Location unavailable")
    
    # Browser location button
    st.markdown("---")
    st.markdown("### 📍 Get My Location")
    
    get_browser_location_component()
    
    # Manual location input
    st.markdown("---")
    st.markdown("### 📍 Manual Entry")
    
    with st.expander("🔧 Enter City/Country"):
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
                            st.session_state.browser_location = None
                            st.session_state.selected_country = None
                            st.success(f"✅ Found: {geocoded['city']}, {geocoded['country']}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Could not find '{location_input}'")
        
        with col_btn2:
            if (st.session_state.manual_location or st.session_state.browser_location or st.session_state.selected_country) and st.button("🔄 Reset", use_container_width=True):
                st.session_state.manual_location = None
                st.session_state.browser_location = None
                st.session_state.selected_country = None
                st.success("✅ Reset to IP location")
                time.sleep(0.5)
                st.rerun()
    
    # ✅ NEW: Country Selector
    st.markdown("---")
    st.markdown("### 🌍 Select Country")
    
    # Create searchable dropdown
    country_search = st.text_input("🔍 Search Country", placeholder="Type to search...", key="country_search")
    
    # Filter countries based on search (case-insensitive)
    if country_search:
        filtered_countries = [c for c in COUNTRIES_DATA.keys() if country_search.lower() in c.lower()]
    else:
        filtered_countries = list(COUNTRIES_DATA.keys())
    
    if filtered_countries:
        selected_country = st.selectbox(
            "Choose Country",
            options=filtered_countries,
            index=filtered_countries.index(st.session_state.selected_country) if st.session_state.selected_country in filtered_countries else 0,
            key="country_selector"
        )
        
        if st.button("📍 Set Country Location", use_container_width=True):
            st.session_state.selected_country = selected_country
            st.session_state.manual_location = None
            st.session_state.browser_location = None
            st.success(f"✅ Location set to {selected_country}")
            time.sleep(0.5)
            st.rerun()
    else:
        st.warning("No countries match your search")

# ========== MAIN HEADER ==========
st.markdown('<h1 class="main-header">AI-RescueMap 🌍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time global disaster monitoring with NASA data & AI</p>', unsafe_allow_html=True)

# Setup Gemini models
gemini_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_api_key:
    if st.session_state.gemini_model_text is None:
        st.session_state.gemini_model_text = setup_gemini(gemini_api_key, "text")
    if st.session_state.gemini_model_image is None:
        st.session_state.gemini_model_image = setup_gemini(gemini_api_key, "image")

# ========== FETCH GLOBAL DISASTERS (CACHED) ==========
if st.session_state.global_disasters is None:
    with st.spinner("🛰 Loading global disaster data from NASA EONET..."):
        st.session_state.global_disasters = fetch_nasa_eonet_disasters(limit=1000)

global_disasters = st.session_state.global_disasters

# ========== MENU: DISASTER MAP ==========
if menu == "🗺 Disaster Map":
    col_settings, col_map = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### ⚙️ Settings")
        
        # ✅ NEW: Enhanced Center Map options
        map_center_options = ["My Location", "Global View"]
        
        # Add countries to dropdown
        map_center_options.extend(sorted(COUNTRIES_DATA.keys()))
        
        map_center_option = st.selectbox("Center Map On:", map_center_options)
        
        # Determine center and disasters based on selection
        if map_center_option == "My Location" and loc:
            center_lat, center_lon, zoom = loc['lat'], loc['lon'], 8
            max_distance = st.slider("Show disasters within (km)", 100, 5000, 1000, step=100)
            disasters = filter_disasters_by_location(
                global_disasters, 
                loc['lat'], 
                loc['lon'], 
                max_distance_km=max_distance,
                max_results=500
            )
        elif map_center_option == "Global View":
            center_lat, center_lon, zoom = 20, 0, 2
            disasters = global_disasters.head(500)
        elif map_center_option in COUNTRIES_DATA:
            # Country selected
            country_data = COUNTRIES_DATA[map_center_option]
            center_lat, center_lon = country_data['lat'], country_data['lon']
            zoom = 6
            max_distance = st.slider("Show disasters within (km)", 100, 5000, 1000, step=100)
            disasters = filter_disasters_by_location(
                global_disasters, 
                center_lat, 
                center_lon, 
                max_distance_km=max_distance,
                max_results=500
            )
        else:
            center_lat, center_lon, zoom = 20, 0, 2
            disasters = global_disasters.head(500)
        
        # Display metrics based on selection
        if map_center_option == "My Location":
            metric_title = f"📍 {loc['city']}, {loc['country']}"
        elif map_center_option == "Global View":
            metric_title = "🌍 Global View"
        elif map_center_option in COUNTRIES_DATA:
            metric_title = f"🌍 {map_center_option}"
        else:
            metric_title = "🗺 Map View"
        
        st.info(metric_title)
        
        show_disasters = st.checkbox("Show Disasters", value=True)
        show_population = st.checkbox("Show Population", value=True)
        
        satellite_layers = st.multiselect("Satellite Layers", 
                                         ['True Color', 'Active Fires', 'Night Lights'], 
                                         default=['True Color'])
        impact_radius = st.slider("Impact Radius (km)", 10, 200, 50)
    
    # ✅ NEW: Dynamic metrics based on selection
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌪 Total Disasters", len(disasters) if not disasters.empty else 0)
    with col2:
        if not disasters.empty and 'distance_km' in disasters.columns:
            nearby = len(disasters[disasters['distance_km'] < 500])
            st.metric("📍 Nearby (<500km)", nearby)
        else:
            st.metric("🌐 Global Total", len(global_disasters))
    with col3:
        st.metric("🤖 AI Status", "✅ Online" if st.session_state.gemini_model_text else "⚠️ Offline")
    with col4:
        st.metric("🛰 Data Source", "NASA EONET")
    
    st.markdown("---")
    
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
                distance_text = f"<br>📍 {disaster['distance_km']:.0f} km from center" if 'distance_km' in disaster else ""
                
                folium.Circle(location=[disaster['lat'], disaster['lon']], 
                            radius=impact_radius * 1000,
                            color=color, fill=True, fillOpacity=0.1).add_to(m)
                
                folium.Marker(location=[disaster['lat'], disaster['lon']],
                            popup=f"<b>{disaster['title']}</b><br>{disaster['category']}<br>{disaster['date']}{distance_text}",
                            icon=folium.Icon(color=color, icon='warning-sign', prefix='glyphicon'),
                            tooltip=disaster['title']).add_to(marker_cluster)
        
        # Add center marker
        if map_center_option == "My Location" and loc:
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>📍 You are here</b><br>{loc['city']}, {loc['country']}",
                icon=folium.Icon(color='green', icon='home', prefix='glyphicon'),
                tooltip="Your Location"
            ).add_to(m)
        elif map_center_option in COUNTRIES_DATA:
            country_data = COUNTRIES_DATA[map_center_option]
            folium.Marker(
                location=[country_data['lat'], country_data['lon']],
                popup=f"<b>📍 {map_center_option}</b><br>{country_data['capital']}",
                icon=folium.Icon(color='blue', icon='flag', prefix='glyphicon'),
                tooltip=f"{map_center_option} - {country_data['capital']}"
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
                    for _, imp in high_risk.head(5).iterrows():
                        st.markdown(f"""<div class="disaster-alert">
                        ⚠️ <b>{imp['disaster']}</b><br>
                        👥 {imp['affected_population']:,} people at risk<br>
                        🚨 Risk Level: {imp['risk_level']}</div>""", unsafe_allow_html=True)
                else:
                    st.info("✅ No high-risk events in this area")
            
            with col2:
                st.markdown("#### 📈 Statistics")
                st.metric("Total at Risk", f"{impact_df['affected_population'].sum():,}")
                st.metric("Critical Events", len(impact_df[impact_df['risk_level'] == 'CRITICAL']))
                st.metric("High Risk Events", len(impact_df[impact_df['risk_level'] == 'HIGH']))

# ========== MENU: AI GUIDANCE ==========
elif menu == "💬 AI Guidance":
    st.markdown("## 💬 AI Emergency Guidance")
    
    # Location-based guidance toggle
    use_location = st.checkbox("🌍 Use my location for guidance", value=False, 
                               help="Check this to get location-specific emergency contacts and guidance")
    
    if use_location and loc:
        st.info(f"📍 Guidance will be provided for: **{loc['city']}, {loc['country']}**")
    
    disaster_type = st.selectbox("Disaster Type", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Other"])
    
    user_situation = st.text_area("Describe your situation:",
        placeholder="Be specific: location (if not using auto-location), number of people, current conditions, available resources...",
        height=120)
    
    if st.button("🚨 GET AI GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("Please describe your situation")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable - Add GEMINI_API_KEY to secrets")
        else:
            with st.spinner("🤖 Analyzing with Gemini AI..."):
                guidance = get_ai_disaster_guidance(
                    disaster_type, 
                    user_situation, 
                    st.session_state.gemini_model_text,
                    user_location=loc if use_location else None,
                    use_location=use_location
                )
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)

# ========== MENU: IMAGE ANALYSIS ==========
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
                with st.spinner("🤖 Analyzing with Gemini AI..."):
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

# ========== MENU: ANALYTICS ==========
elif menu == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    
    # ✅ NEW: Enhanced view mode with country selection
    view_options = ["📍 My Location", "🌍 Global View"] + sorted(COUNTRIES_DATA.keys())
    
    view_mode = st.selectbox("Select View:", view_options)
    
    # Determine filtered disasters based on view mode
    if view_mode == "📍 My Location" and loc:
        radius_filter = st.slider("Show disasters within (km):", 100, 5000, 1000, step=100)
        filtered_disasters = filter_disasters_by_location(
            global_disasters,
            loc['lat'],
            loc['lon'],
            max_distance_km=radius_filter,
            max_results=500
        )
        st.success(f"📍 Showing {len(filtered_disasters)} disasters within {radius_filter} km of **{loc['city']}, {loc['country']}**")
        map_center = [loc['lat'], loc['lon']]
        map_zoom = 6
    elif view_mode == "🌍 Global View":
        filtered_disasters = global_disasters.head(500)
        st.info(f"🌍 Showing {len(filtered_disasters)} global disasters")
        map_center = [20, 0]
        map_zoom = 2
    elif view_mode in COUNTRIES_DATA:
        country_data = COUNTRIES_DATA[view_mode]
        radius_filter = st.slider("Show disasters within (km):", 100, 5000, 1000, step=100)
        filtered_disasters = filter_disasters_by_location(
            global_disasters,
            country_data['lat'],
            country_data['lon'],
            max_distance_km=radius_filter,
            max_results=500
        )
        st.success(f"🌍 Showing {len(filtered_disasters)} disasters within {radius_filter} km of **{view_mode}**")
        map_center = [country_data['lat'], country_data['lon']]
        map_zoom = 6
    else:
        filtered_disasters = global_disasters.head(500)
        st.info(f"🌍 Showing {len(filtered_disasters)} disasters")
        map_center = [20, 0]
        map_zoom = 2
    
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
            categories = filtered_disasters['category'].nunique()
            st.metric("📊 Categories", categories)
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 By Category")
            category_counts = filtered_disasters['category'].value_counts().head(10)
            st.bar_chart(category_counts)
        
        with col_b:
            st.markdown("### 📅 Recent Events")
            display_cols = ['title', 'category', 'date']
            if 'distance_km' in filtered_disasters.columns:
                filtered_disasters['distance_km'] = filtered_disasters['distance_km'].round(0).astype(int)
                display_cols.append('distance_km')
            
            recent = filtered_disasters.head(10)[display_cols]
            st.dataframe(recent, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown(f"### 🗺 Distribution Map")
        
        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles='CartoDB dark_matter')
        
        color_map = {
            'Wildfires': 'red', 
            'Severe Storms': 'orange', 
            'Floods': 'blue', 
            'Earthquakes': 'darkred',
            'Volcanoes': 'red'
        }
        
        for _, disaster in filtered_disasters.head(200).iterrows():
            popup_text = f"<b>{disaster['title']}</b><br>{disaster['category']}"
            if 'distance_km' in disaster:
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
        
        # Add center marker
        if view_mode == "📍 My Location" and loc:
            folium.Marker(
                location=[loc['lat'], loc['lon']],
                popup=f"<b>📍 You are here</b><br>{loc['city']}, {loc['country']}",
                icon=folium.Icon(color='green', icon='home', prefix='glyphicon'),
                tooltip="Your Location"
            ).add_to(m)
        elif view_mode in COUNTRIES_DATA:
            country_data = COUNTRIES_DATA[view_mode]
            folium.Marker(
                location=[country_data['lat'], country_data['lon']],
                popup=f"<b>📍 {view_mode}</b><br>{country_data['capital']}",
                icon=folium.Icon(color='blue', icon='flag', prefix='glyphicon'),
                tooltip=f"{view_mode} - {country_data['capital']}"
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
        if 'distance_km' in final_filtered.columns:
            display_cols.append('distance_km')
        
        st.dataframe(final_filtered[display_cols], use_container_width=True, hide_index=True, height=400)
        
        view_label = view_mode.replace("📍 ", "").replace("🌍 ", "").replace(" ", "_")
        st.download_button(
            "📥 Download as CSV",
            data=final_filtered.to_csv(index=False).encode('utf-8'),
            file_name=f"disasters_{view_label}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No disasters found in selected area. Try adjusting filters.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
Built by <b>HasnainAtif</b> for NASA Space Apps Challenge 2025<br>
Real-time global disaster monitoring
</p>
""", unsafe_allow_html=True)
