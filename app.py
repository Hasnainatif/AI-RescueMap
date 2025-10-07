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

# ✅ ADD: Import WorldPop integration
from worldpop_integration import read_worldpop_window

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
    "GEOCODING_BACKUP": "https://geocode.maps.co/search",
    # ✅ ADD: WorldPop configuration
    "WORLDPOP_URL": "https://huggingface.co/datasets/HasnainAtif/worldpop_2024/resolve/main/global_pop_2024_CN_1km_R2025A_UA_v1.tif",
    "WORLDPOP_PATH": "data/worldpop_2024_1km.tif",
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
    "Default": {"emergency": "112 (International)", "police": "Local Police", "ambulance": "Local Ambulance", "fire": "Local Fire"}
}

def get_emergency_contacts(country: str) -> dict:
    """Get emergency contacts for a specific country"""
    return EMERGENCY_CONTACTS.get(country, EMERGENCY_CONTACTS["Default"])

def geocode_location(city_or_address: str, max_retries=2):
    """Convert city/address to coordinates with multiple fallback services."""
    for attempt in range(max_retries):
        try:
            params = {
                'q': city_or_address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
            
            response = requests.get(CONFIG["GEOCODING_API"], params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
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
                        'source': 'OpenStreetMap'
                    }
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                    
        except requests.exceptions.ConnectionError:
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    try:
        params = {
            'q': city_or_address,
            'format': 'json'
        }
        
        response = requests.get(CONFIG["GEOCODING_BACKUP"], params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data and len(data) > 0:
                result = data[0]
                
                return {
                    'lat': float(result['lat']),
                    'lon': float(result['lon']),
                    'city': result.get('display_name', city_or_address).split(',')[0],
                    'country': result.get('display_name', '').split(',')[-1].strip() if ',' in result.get('display_name', '') else 'Unknown',
                    'region': result.get('display_name', '').split(',')[1].strip() if len(result.get('display_name', '').split(',')) > 1 else 'Unknown',
                    'full_address': result.get('display_name', city_or_address),
                    'method': 'manual',
                    'source': 'Geocode.maps.co'
                }
    except Exception as e:
        pass
    
    country_coords = {
        'pakistan': {'lat': 30.3753, 'lon': 69.3451, 'city': 'Pakistan', 'region': 'Central'},
        'faisalabad': {'lat': 31.4504, 'lon': 73.1350, 'city': 'Faisalabad', 'region': 'Punjab'},
        'united states': {'lat': 37.0902, 'lon': -95.7129, 'city': 'United States', 'region': 'Central'},
        'india': {'lat': 20.5937, 'lon': 78.9629, 'city': 'India', 'region': 'Central'},
        'china': {'lat': 35.8617, 'lon': 104.1954, 'city': 'China', 'region': 'Central'},
        'japan': {'lat': 36.2048, 'lon': 138.2529, 'city': 'Japan', 'region': 'Central'},
        'australia': {'lat': -25.2744, 'lon': 133.7751, 'city': 'Australia', 'region': 'Central'},
        'canada': {'lat': 56.1304, 'lon': -106.3468, 'city': 'Canada', 'region': 'Central'},
        'brazil': {'lat': -14.2350, 'lon': -51.9253, 'city': 'Brazil', 'region': 'Central'},
        'russia': {'lat': 61.5240, 'lon': 105.3188, 'city': 'Russia', 'region': 'Central'},
        'uk': {'lat': 55.3781, 'lon': -3.4360, 'city': 'United Kingdom', 'region': 'Central'},
        'germany': {'lat': 51.1657, 'lon': 10.4515, 'city': 'Germany', 'region': 'Central'},
        'france': {'lat': 46.2276, 'lon': 2.2137, 'city': 'France', 'region': 'Central'},
    }
    
    search_lower = city_or_address.lower()
    for key, coords in country_coords.items():
        if key in search_lower:
            return {
                **coords,
                'country': coords['city'],
                'full_address': city_or_address,
                'method': 'manual',
                'source': 'Approximate (Fallback)'
            }
    
    st.error(f"❌ Could not find location: {city_or_address}. Please try adding more details (e.g., 'City, Country')")
    return None

def reverse_geocode(lat: float, lon: float, max_retries=2):
    """Convert coordinates to address with retry logic"""
    for attempt in range(max_retries):
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1
            }
            headers = {'User-Agent': 'AI-RescueMap/1.0 (NASA Space Apps 2025)'}
            
            response = requests.get(CONFIG["REVERSE_GEOCODING_API"], params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
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
            
            if response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2)
                continue
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    return {
        'lat': lat,
        'lon': lon,
        'city': f"Location ({lat:.2f}, {lon:.2f})",
        'country': 'Unknown',
        'region': 'Unknown',
        'full_address': f"{lat:.4f}, {lon:.4f}",
        'method': 'browser',
        'source': 'GPS (No address found)'
    }

def get_ip_location():
    """Get location from IP address (fallback only)."""
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
                'source': 'IP Geolocation (ipapi.co)'
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
                'source': 'IP Geolocation (ip-api.com)'
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
    """Priority: Browser GPS > Manual > IP Fallback"""
    if st.session_state.get('browser_location'):
        return st.session_state.browser_location
    elif st.session_state.get('manual_location'):
        return st.session_state.manual_location
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

@st.cache_data(ttl=1800)
def fetch_nasa_eonet_disasters(status="open", limit=500):
    """Fetch disasters from NASA EONET API - includes ALL disaster types"""
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
    """Add NASA GIBS satellite imagery layers to the map."""
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
    """Fallback synthetic population generator"""
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
        
        prompt = f"""You are an elite emergency response expert and disaster management specialist with decades of field experience. Your role is to provide IMMEDIATE, actionable, life-saving guidance for people in REAL emergencies.

**CRITICAL INSTRUCTIONS:**
1. FIRST, validate if this is a LEGITIMATE EMERGENCY requiring immediate assistance
2. VALID emergencies include:
   - Natural disasters (floods, earthquakes, hurricanes, tornadoes, tsunamis, wildfires, landslides, avalanches, volcanic eruptions)
   - Dangerous animal encounters (snakes, wild animals, aggressive animals, insect swarms)
   - Building emergencies (structural collapse, gas leaks, fires, explosions, trapped in building)
   - Medical emergencies during disasters (injuries, bleeding, shock, breathing problems, unconsciousness)
   - Being lost/stranded (wilderness, desert, mountains, sea, dangerous areas, no supplies)
   - Severe weather threats (lightning, hail, extreme temperatures, blizzards)
   - Environmental hazards (toxic spills, radiation, smoke, contaminated water)
   - Mass emergency situations (riots, civil unrest during disasters, mass evacuations)

3. INVALID queries that should be REJECTED:
   - Coding questions, programming help, software development
   - Homework, assignments, academic questions unrelated to emergency survival
   - Jokes, memes, entertainment, general chat
   - General knowledge questions (history, science, math) not related to immediate survival
   - Non-emergency topics (travel planning, recipes, shopping, sports, etc.)
   - Hypothetical scenarios without real danger ("what if I was in...")

**IF QUERY IS INVALID:**
Respond EXACTLY with this format:
"❌ **NOT AN EMERGENCY QUERY**

This system is designed ONLY for real emergencies and disaster situations.

**Valid emergency queries include:**
- Natural disasters (floods, earthquakes, hurricanes, etc.)
- Dangerous animal encounters (snakes, wild animals)
- Building emergencies (fires, collapses, gas leaks)
- Medical emergencies during disasters
- Being lost or stranded in dangerous conditions
- Severe weather threats
- Environmental hazards

**Your query appears to be about:** [briefly identify what they asked about]

**If you are in a real emergency:**
- Call local emergency services immediately
- Describe your specific dangerous situation
- Include your location and number of people affected

Please rephrase your question if you have a genuine emergency."

**IF QUERY IS VALID (REAL EMERGENCY):**

**DISASTER TYPE:** {disaster_type}
**USER SITUATION:** {user_situation}{location_context}

Provide comprehensive, expert-level emergency guidance in this EXACT format:

🚨 **IMMEDIATE ACTIONS (DO THIS RIGHT NOW - WITHIN 60 SECONDS):**
[Provide 3-5 CRITICAL steps that must be taken IMMEDIATELY to save lives. Number each step. Be ULTRA specific - not vague advice. Include exact timing where relevant.]

⚠️ **CRITICAL DON'Ts (ACTIONS THAT CAN KILL YOU):**
[List 3-5 specific dangerous actions people commonly make in this emergency that can be FATAL. Explain WHY each is dangerous in one sentence.]

🏃 **EVACUATION DECISION MATRIX:**
**EVACUATE IMMEDIATELY IF:**
- [List 3-4 specific conditions requiring immediate evacuation]

**SHELTER IN PLACE IF:**
- [List 3-4 specific conditions requiring sheltering]

**EVACUATION ROUTE GUIDANCE:**
- [Specific direction guidance for this disaster type]
- [What to avoid during evacuation]

📦 **CRITICAL SURVIVAL SUPPLIES (Priority Order):**
**GRAB IN NEXT 5 MINUTES (Life-saving priority):**
1. [Most critical item with reason]
2. [Second most critical with reason]
3. [Third most critical with reason]

**GRAB IF TIME PERMITS (Important but not life-threatening):**
- [Additional supplies with brief rationale]

🩺 **IMMEDIATE MEDICAL CONCERNS:**
[List 3-4 most common injuries/medical issues in this disaster type and ONE-LINE treatment/prevention for each]

📱 **COMMUNICATION & SIGNALING:**
- [How to call for help in this specific situation]
- [Alternative communication methods if phones don't work]
- [How to signal rescuers]

⏰ **URGENCY ASSESSMENT:**
**TIMING:** [Choose: IMMEDIATE (act in seconds/minutes) / URGENT (act within hours) / CRITICAL (act within 1-2 hours) / PLAN (prepare within hours)]

**SURVIVAL WINDOW:** [How long can someone survive in this situation without help]

**RESCUE LIKELIHOOD:** [When to expect help: hours/days/need self-rescue]

🎯 **SURVIVAL PRIORITIES (Next 24 Hours):**
1. [First priority with timeframe]
2. [Second priority with timeframe]
3. [Third priority with timeframe]

🌡️ **ENVIRONMENTAL THREATS:**
[List 2-3 secondary dangers that emerge after initial event - e.g., aftershocks, contaminated water, exposure]

💡 **LIFE-SAVING TIPS SPECIFIC TO THIS EMERGENCY:**
[Provide 2-3 expert-level, lesser-known survival tactics that could save lives]

**CRITICAL: Keep all advice PRACTICAL, ACTIONABLE, and SPECIFIC. No generic statements. Every instruction should be clear enough that a panicked person can follow it. Focus on survival, not comfort.**

Remember: YOU ARE SAVING LIVES. Be direct, be precise, be expert-level."""

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
    
    prompt = """You are an elite disaster assessment specialist and emergency response analyst with expertise in damage assessment, risk evaluation, and crisis management.

**CRITICAL INSTRUCTIONS:**

**STEP 1: IMAGE VALIDATION**
FIRST, determine if this image shows a REAL EMERGENCY or DISASTER situation.

**VALID DISASTER/EMERGENCY IMAGES:**
- Natural disaster damage (floods, earthquakes, hurricanes, tornadoes, wildfires, tsunamis)
- Building collapses, structural damage, destroyed infrastructure
- Active fires, smoke, explosions, or their aftermath
- Severe weather conditions (massive storms, lightning, hail damage)
- Landslides, avalanches, mudflows, sinkholes
- Dangerous animal encounters in emergency context
- People in genuine distress/danger situations
- Environmental hazards (toxic spills, contamination, debris fields)
- Aftermath scenes showing significant damage or destruction
- Rescue operations, evacuations, emergency response scenes

**INVALID IMAGES TO REJECT:**
- Memes, cartoons, edited images, fake disasters
- Selfies, personal photos, normal daily life photos
- Code screenshots, text documents, programming content
- Normal weather, regular city scenes, tourist photos
- Homework, diagrams, charts unrelated to real emergencies
- Stock photos clearly staged or for illustration only
- Animals in non-dangerous contexts
- Minor damage (small scratches, normal wear and tear)

**IF IMAGE IS INVALID (NOT A REAL DISASTER/EMERGENCY):**
Respond EXACTLY in this format:
"❌ **NOT A DISASTER/EMERGENCY IMAGE**

This image does not show a real disaster or emergency situation requiring assessment.

**Image appears to contain:** [Describe what the image actually shows - e.g., "a meme", "a selfie", "code screenshot", "normal scenery"]

**This system analyzes ONLY:**
- Real disaster damage and destruction
- Active emergency situations
- Dangerous conditions requiring response
- Aftermath of natural disasters
- Structural failures and hazards
- Environmental threats

**If you have a real disaster image:**
Please upload a clear photo showing:
- Visible damage or destruction
- Active emergency conditions
- Dangerous situations
- Disaster aftermath

**If you are in an emergency:**
Call local emergency services immediately instead of uploading images."

**IF IMAGE IS VALID (REAL DISASTER/EMERGENCY):**

Provide comprehensive expert analysis in this EXACT format:

🔍 **DISASTER TYPE & CLASSIFICATION:**
**Primary Type:** [Specific disaster category]
**Secondary Threats:** [Related hazards visible]
**Event Phase:** [Active/Recent/Aftermath/Recovery]

⚠️ **SEVERITY ASSESSMENT:**
**Overall Severity:** [LOW / MODERATE / HIGH / CRITICAL / CATASTROPHIC]

**Severity Justification:**
- Structural damage level: [Percentage and description]
- Area affected: [Estimated square km/miles]
- Accessibility: [Are areas blocked, isolated, or accessible]
- Immediate danger present: [Yes/No and specific threats]

**Severity Score Breakdown:**
- Infrastructure damage: [0-20 points] - [Score]/20
- Human safety risk: [0-30 points] - [Score]/30
- Spread/expansion potential: [0-20 points] - [Score]/20
- Rescue difficulty: [0-15 points] - [Score]/15
- Secondary hazards: [0-15 points] - [Score]/15
**TOTAL SEVERITY SCORE:** [Sum]/100

💥 **VISIBLE DAMAGES & DESTRUCTION:**
**Structural:**
- [List specific building/infrastructure damage observed]

**Environmental:**
- [List landscape, water, fire, or environmental damage]

**Utility Systems:**
- [Power lines, water, gas, communications visible damage]

**Access & Mobility:**
- [Roads, bridges, pathways - are they passable?]

📏 **AFFECTED AREA ESTIMATION:**
**Visible Impact Zone:** [Estimated size with reasoning]
**Potential Extended Impact:** [Likely spread beyond visible area]
**Geographic Features:** [Terrain, urban/rural, elevation factors]

👥 **POPULATION RISK ASSESSMENT:**
**Visible Population Indicators:**
- Buildings: [Residential/commercial count visible]
- Vehicles: [Number and condition]
- People: [If visible, how many, condition]

**Estimated Risk Level:** [EXTREME / HIGH / MODERATE / LOW]
**Vulnerable Populations:** [Elderly, children, disabled - if indicators present]
**Evacuation Status:** [Evidence of evacuation? Empty/occupied areas?]

🚨 **IMMEDIATE CONCERNS (TOP 5 CRITICAL PRIORITIES):**
1. [Most urgent threat with specific action needed]
2. [Second urgent threat with action]
3. [Third urgent threat with action]
4. [Fourth concern with action]
5. [Fifth concern with action]

🚒 **RESPONSE RECOMMENDATIONS:**

**IMMEDIATE (0-2 hours):**
- [Specific action 1]
- [Specific action 2]
- [Specific action 3]

**SHORT-TERM (2-24 hours):**
- [Critical need 1]
- [Critical need 2]
- [Critical need 3]

**RESOURCES NEEDED:**
- Personnel: [Types and estimated numbers]
- Equipment: [Specific machinery, tools required]
- Supplies: [Materials, medical, food, water quantities]
- Specialists: [Engineers, medical, hazmat, etc.]

🛠️ **RECOVERY TIME ESTIMATE:**
**Category:** [SHORT-TERM (days-weeks) / MEDIUM-TERM (weeks-months) / LONG-TERM (months-years) / CATASTROPHIC (years-decade+)]

**Justification:**
- Damage extent: [Why this timeframe]
- Resource requirements: [What's needed for recovery]
- Infrastructure rebuilding: [Major systems to restore]

**Recovery Phases:**
1. Emergency response: [Timeframe]
2. Debris removal & safety: [Timeframe]
3. Infrastructure restoration: [Timeframe]
4. Full recovery: [Timeframe]

🌊 **SECONDARY HAZARD WARNINGS:**
[List 3-5 potential secondary disasters that could develop:]
- [Hazard 1 and likelihood]
- [Hazard 2 and likelihood]
- [Hazard 3 and likelihood]

📊 **DAMAGE METRICS (If Measurable):**
- Estimated financial impact: [USD range if determinable]
- Buildings affected: [Count/percentage]
- Critical infrastructure: [What's damaged]
- Environmental impact: [Pollution, contamination, ecosystem]

**ASSESSMENT CONFIDENCE LEVEL:** [HIGH / MODERATE / LOW - based on image clarity and visible details]

**LIMITATIONS OF THIS ASSESSMENT:**
[Note what cannot be determined from the image alone - hidden damage, casualties, exact measurements, etc.]

**CRITICAL: Every assessment should be SPECIFIC, MEASURABLE, and ACTIONABLE. Avoid vague language. If you cannot determine something from the image, explicitly state that limitation.**"""

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

Free tier quota hit. Please wait 2-3 minutes and try again."""
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
    
    loc = get_current_location()
    
    if loc:
        method_badge = {
            'browser': '🎯 GPS (Most Accurate)',
            'manual': '📍 Manual Entry',
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
                    st.caption(f"📍 Accuracy: ~{int(gps_acc)}m")
                    st.query_params.clear()
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            st.error(f"❌ GPS error: {e}")
            st.query_params.clear()
    
    st.markdown("---")
    st.markdown("### 📝 Enter Location Manually")
    
    with st.expander("🔧 Manual Entry"):
        st.info("**Examples:**\n- Faisalabad, Pakistan\n- New York, USA\n- Tokyo, Japan")
        
        location_input = st.text_input("City, Country", placeholder="e.g., Faisalabad, Pakistan")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔍 Find", use_container_width=True, disabled=not location_input):
                if location_input:
                    with st.spinner(f"🌍 Finding {location_input}..."):
                        geocoded = geocode_location(location_input)
                        if geocoded:
                            st.session_state.manual_location = geocoded
                            st.session_state.browser_location = None
                            st.success(f"✅ Found!")
                            time.sleep(1)
                            st.rerun()
        
        with col_btn2:
            if st.session_state.manual_location and st.button("🔄 Reset", use_container_width=True):
                st.session_state.manual_location = None
                st.session_state.browser_location = None
                st.success("✅ Reset")
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

if menu == "🗺 Disaster Map":
    with st.spinner("🛰 Fetching NASA EONET data..."):
        disasters = fetch_nasa_eonet_disasters()
    
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
        
        map_options = ["My Location", "Global View"]
        if not disasters.empty:
            disaster_titles = disasters['title'].tolist()[:10]
            map_options.extend(disaster_titles)
        
        map_center_option = st.selectbox("Center Map", map_options)
        
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
        
        # ✅ FIXED INDENTATION - USE REAL WORLDPOP DATA
        if show_population:
            with st.spinner("📊 Loading WorldPop data..."):
                pop_df = read_worldpop_window(
                    url=CONFIG["WORLDPOP_URL"],
                    path=CONFIG["WORLDPOP_PATH"],
                    center_lat=center_lat,
                    center_lon=center_lon,
                    radius_km=150,  # 150km radius of real data
                    out_size=(300, 300)  # Higher resolution
                )
            
            # Fallback to synthetic data if WorldPop unavailable
            if pop_df is None or len(pop_df) == 0:
                st.info("ℹ️ Real WorldPop data not available for this region, using estimated data")
                pop_df = generate_population_data(center_lat, center_lon, radius_deg=3, num_points=1500)
            else:
                st.success(f"✅ Using real WorldPop 2024 data: {len(pop_df):,} population points")
                      # Create heatmap if data exists
            if pop_df is not None and len(pop_df) > 0:
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

elif menu == "💬 AI Guidance":
    st.markdown("## 💬 AI Emergency Guidance")
    
    use_location = st.checkbox("📍 Use my location for guidance", value=False)
    
    if use_location and loc:
        st.info(f"🎯 Using: **{loc['city']}, {loc['country']}**")
    
    disaster_type = st.selectbox("Disaster Type", 
        ["Flood", "Wildfire", "Earthquake", "Hurricane", "Tsunami", "Tornado", "Volcano", "Landslide", "Other"])
    
    user_situation = st.text_area("Describe your situation:",
        placeholder="Be specific: What is happening? Number of people? Current conditions?",
        height=120)
    
    if st.button("🚨 GET AI GUIDANCE", type="primary", use_container_width=True):
        if not user_situation:
            st.error("❌ Please describe your situation")
        elif not st.session_state.gemini_model_text:
            st.warning("⚠️ AI unavailable")
        else:
            with st.spinner("🤖 Analyzing..."):
                guidance = get_ai_disaster_guidance(
                    disaster_type, user_situation, st.session_state.gemini_model_text,
                    use_location=use_location, location=loc if use_location else None
                )
                st.markdown(f'<div class="ai-response">{guidance}</div>', unsafe_allow_html=True)

elif menu == "🖼 Image Analysis":
    from PIL import Image
    
    st.markdown("## 🖼 AI Image Analysis")
    
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
    
    if loc:
        view_mode = st.radio("View:", ["📍 My Location", "🌍 Global View"], horizontal=True)
    else:
        view_mode = "🌍 Global View"
    
    with st.spinner("📡 Loading data..."):
        disasters = fetch_nasa_eonet_disasters(limit=500)
    
    if not disasters.empty and loc:
        disasters['distance_km'] = disasters.apply(
            lambda row: calculate_distance(loc['lat'], loc['lon'], row['lat'], row['lon']), 
            axis=1
        )
    
    if "My Location" in view_mode and loc and not disasters.empty:
        radius = st.slider("Radius (km)", 100, 5000, 1000, step=100)
        disasters = disasters[disasters['distance_km'] <= radius].sort_values('distance_km')
        st.success(f"📍 {len(disasters)} disasters within {radius} km")
    else:
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
        
        st.download_button(
            "📥 Download CSV",
            disasters.to_csv(index=False).encode('utf-8'),
            f"disasters_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
🌍 <b>AI-RescueMap</b> • <b>created by HasnainAtif</b> @ NASA Space Apps 2025
</p>
""", unsafe_allow_html=True)      
