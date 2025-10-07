# 🌍 AI-RescueMap

[![Python](https://img.shields.io/badge/Python-3.11.8-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![NASA Space Apps](https://img.shields.io/badge/NASA-Space%20Apps%202025-orange.svg)](https://www.spaceappschallenge.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI](https://img.shields.io/badge/AI-Google%20Gemini-yellow.svg)](https://ai.google.dev/)

**AI-powered disaster mapping and emergency response platform leveraging NASA satellite data and advanced AI for real-time disaster monitoring, risk assessment, and personalized emergency guidance.**

---

## 🎯 Project Overview

### The Problem

Natural disasters are increasing in frequency and severity due to climate change, affecting millions worldwide. During emergencies:
- **Information is fragmented** across multiple sources and agencies
- **Real-time disaster tracking** is difficult for civilians to access
- **Personalized emergency guidance** is not readily available
- **Population impact assessment** is complex and time-consuming
- **Visual damage assessment** requires expert analysis

### Our Solution

**AI-RescueMap** is a comprehensive disaster response platform that combines:
- 🛰️ **NASA satellite data** (EONET, GIBS) for real-time disaster tracking
- 🤖 **Google Gemini AI** for intelligent emergency guidance and image analysis
- 📍 **Multi-source geolocation** (GPS, IP, manual entry) for accurate positioning
- 📊 **Population impact modeling** to assess disaster effects
- 🗺️ **Interactive mapping** with heatmaps and clustering for visualization
- 🚨 **Worldwide emergency contacts** database covering 100+ countries

### Inspiration

Created for **NASA Space Apps Challenge 2025**, this project aims to democratize access to critical disaster information and AI-powered emergency assistance, making it accessible to anyone with an internet connection.

---

## ✨ Key Features

### 🗺️ Disaster Map
- **Real-time NASA EONET data** tracking active disasters worldwide (wildfires, earthquakes, floods, hurricanes, volcanoes, etc.)
- **Interactive Folium maps** with multiple visualization options:
  - Marker clustering for dense disaster areas
  - Heatmap overlays for population density
  - NASA GIBS satellite imagery layers (True Color, Active Fires, Night Lights)
- **Distance calculations** from user location to nearby disasters
- **Population impact analysis** with configurable radius
- **Multi-center views**: Your location, global view, or specific disaster focus

### 💬 AI Emergency Guidance
- **Google Gemini AI integration** for context-aware emergency advice
- **Location-specific guidance** using your current position
- **Multi-disaster support**: Floods, wildfires, earthquakes, hurricanes, tsunamis, tornadoes, volcanoes, landslides
- **Natural language queries** - describe your situation in plain text
- **Immediate actionable steps** for survival and safety
- **Emergency contact information** for your country

### 🖼️ Image Analysis
- **AI-powered disaster image assessment** using Gemini Vision
- **Automatic severity detection** with risk scoring (0-100)
- **Damage type identification**: Structural, flooding, fire, etc.
- **Safety recommendations** based on visual analysis
- **Support for JPG, JPEG, and PNG formats**

### 📊 Analytics Dashboard
- **Global disaster statistics** with filtering by location radius
- **Category breakdown** (wildfires, earthquakes, storms, etc.)
- **Temporal analysis** of disaster trends
- **Top affected regions** visualization
- **Distance-sorted disaster lists** with key details

### 🎯 Advanced Geolocation
- **Priority-based location detection**:
  1. Browser GPS (most accurate)
  2. Manual address/city input with geocoding
  3. IP-based fallback (worldwide coverage)
- **Multiple geocoding providers** with automatic fallback
- **Reverse geocoding** for coordinates-to-address conversion

---

## 🛠️ Technology Stack

### Core Framework
- **[Streamlit](https://streamlit.io/)** - Web application framework for Python
- **[Python 3.11.8](https://www.python.org/)** - Primary programming language

### Mapping & Visualization
- **[Folium](https://python-visualization.github.io/folium/)** - Interactive maps with Leaflet.js
- **[streamlit-folium](https://github.com/randyzwitch/streamlit-folium)** - Streamlit-Folium integration
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing

### AI & Machine Learning
- **[google-generativeai](https://ai.google.dev/)** - Google Gemini AI API for text and vision tasks
- **[Pillow (PIL)](https://python-pillow.org/)** - Image processing

### Data Sources & APIs
- **[NASA EONET API](https://eonet.gsfc.nasa.gov/)** - Earth Observatory Natural Event Tracker
- **[NASA GIBS](https://gibs.earthdata.nasa.gov/)** - Global Imagery Browse Services for satellite layers
- **[OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/)** - Geocoding and reverse geocoding
- **IP Geolocation APIs** - IP-based location fallback

### Geospatial & Location
- **[Rasterio](https://rasterio.readthedocs.io/)** - Geospatial raster data I/O
- **[Geopy](https://geopy.readthedocs.io/)** - Geocoding library
- **[streamlit-geolocation](https://github.com/michaelwasyl/streamlit-geolocation)** - Browser GPS access
- **[streamlit-javascript](https://github.com/tool-scrapper/streamlit-javascript)** - JavaScript execution in Streamlit

### Utilities
- **[Requests](https://requests.readthedocs.io/)** - HTTP library for API calls
- **[python-dateutil](https://dateutil.readthedocs.io/)** - Date parsing and manipulation
- **[pytz](https://pythonhosted.org/pytz/)** - Timezone handling

---

## 📦 Installation

### Prerequisites
- Python 3.11.8 or higher
- pip (Python package manager)
- Internet connection for API access
- (Optional) Google Gemini API key for AI features

### Step 1: Clone the Repository
```bash
git clone https://github.com/Hasnainatif/AI-RescueMap.git
cd AI-RescueMap
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies installed:**
```
streamlit
folium
streamlit-folium
requests
pandas
numpy
Pillow
google-generativeai
python-dateutil
pytz
rasterio
geopy
streamlit-geolocation
streamlit-javascript
```

### Step 3: Verify Installation
Run the test script to ensure everything is properly configured:
```bash
python test_gemini.py
```

This will check:
- ✅ All dependencies installed
- ✅ API key configuration (if provided)
- ✅ Gemini AI connectivity
- ✅ NASA EONET API access

---

## ⚙️ Configuration

### Setting Up API Keys

#### Google Gemini API Key (Required for AI Features)

1. **Get your FREE API key:**
   - Visit: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click **"Create API Key"**
   - Copy the generated key

2. **Configure the API key:**

   **Option A: Using secrets.toml (Recommended for local development)**
   ```bash
   mkdir -p .streamlit
   touch .streamlit/secrets.toml
   ```
   
   Add to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

   **Option B: Environment Variable**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

   **Option C: Streamlit Cloud (for deployment)**
   - Go to your app settings in Streamlit Cloud
   - Navigate to **Secrets**
   - Add:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```

⚠️ **Important:** Never commit `secrets.toml` to version control! It's already included in `.gitignore`.

### API Rate Limits (Free Tier)
- **60 requests per minute**
- **1,500 requests per day**
- Monitor usage: [API Key Dashboard](https://makersuite.google.com/app/apikey)

---

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Platform

#### 1. 🗺️ Disaster Map
- **View active disasters** from NASA EONET
- **Select map center**: Your location, global view, or specific disaster
- **Toggle visualizations**: Disasters, population density, satellite layers
- **Adjust impact radius** to see affected populations
- **Click markers** for detailed disaster information

#### 2. 💬 AI Guidance
- **Select disaster type** from dropdown
- **Describe your situation** in the text area
- **Enable location** for context-aware guidance (optional)
- **Click "GET AI GUIDANCE"** for personalized emergency instructions
- **View emergency contacts** for your country

#### 3. 🖼️ Image Analysis
- **Upload disaster images** (JPG, JPEG, PNG)
- **Click "ANALYZE"** to process with AI
- **Review results**: Severity level, risk score, damage assessment
- **Get safety recommendations** based on visual analysis

#### 4. 📊 Analytics
- **Switch between** local and global views
- **Adjust radius** for local disaster filtering
- **View statistics**: Active disasters, category breakdown, trends
- **Explore charts**: Disaster types, affected regions, temporal patterns

---

## 🌐 Data & API Integration

### NASA EONET (Earth Observatory Natural Event Tracker)
- **Endpoint:** `https://eonet.gsfc.nasa.gov/api/v3/events`
- **Purpose:** Real-time natural disaster tracking
- **Coverage:** Wildfires, storms, floods, earthquakes, volcanoes, ice, drought, dust/haze, severe weather
- **Update frequency:** Near real-time
- **No API key required**

### NASA GIBS (Global Imagery Browse Services)
- **Endpoint:** `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best`
- **Purpose:** Satellite imagery overlays
- **Available layers:**
  - True Color (MODIS/VIIRS)
  - Active Fires (VIIRS/MODIS)
  - Night Lights (VIIRS)
- **No API key required**

### Geocoding Services
- **Primary:** OpenStreetMap Nominatim
- **Backup:** geocode.maps.co
- **Features:** Address to coordinates, reverse geocoding
- **Rate limits:** 1 request/second (Nominatim)

### IP Geolocation
- **Primary:** ipapi.co
- **Backup:** ip-api.com
- **Purpose:** Fallback location detection
- **Free tier:** 1,000 requests/day (ipapi.co)

### Google Gemini AI
- **Models used:** gemini-2.0-flash-exp
- **Capabilities:**
  - Text generation (emergency guidance)
  - Vision analysis (disaster image assessment)
  - Context-aware responses
- **Requires API key** (see Configuration)

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
- 🐛 **Report bugs** via GitHub Issues
- 💡 **Suggest features** and improvements
- 📝 **Improve documentation**
- 🔧 **Submit pull requests** with code improvements
- 🧪 **Add test cases** and improve code coverage
- 🌍 **Add emergency contact data** for more countries

### Development Workflow

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-RescueMap.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed

4. **Test your changes**
   ```bash
   python test_gemini.py
   streamlit run app.py
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Include screenshots for UI changes

### Code Style Guidelines
- Follow **PEP 8** for Python code
- Use **meaningful variable names**
- Add **docstrings** for functions
- Keep functions **focused and modular**
- Use **type hints** where appropriate

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Liability and warranty not provided

---

## 👥 Team & Acknowledgments

### Created By
**HasnainAtif** - [@Hasnainatif](https://github.com/Hasnainatif)

### Built For
**NASA Space Apps Challenge 2025** 🚀

### Special Thanks
- **NASA** for providing free, open access to EONET and GIBS APIs
- **Google** for the Gemini AI API
- **Streamlit** for the amazing web framework
- **OpenStreetMap** for geocoding services
- The **open-source community** for the incredible libraries and tools

### Data Sources & Attribution
- Disaster data: NASA Earth Observatory Natural Event Tracker (EONET)
- Satellite imagery: NASA Global Imagery Browse Services (GIBS)
- Map tiles: OpenStreetMap contributors, CartoDB
- Emergency contacts: Compiled from public sources and government websites

---

## 📞 Contact & Support

### Get Help
- 📧 **Email:** Contact via GitHub
- 💬 **Issues:** [GitHub Issues](https://github.com/Hasnainatif/AI-RescueMap/issues)
- 🌟 **Star this repo** if you find it useful!

### Project Links
- 🔗 **Repository:** [github.com/Hasnainatif/AI-RescueMap](https://github.com/Hasnainatif/AI-RescueMap)
- 🌐 **Live Demo:** Deploy on [Streamlit Cloud](https://streamlit.io/cloud)

### For Emergencies
**🚨 This app is for information only. Always call your local emergency services in a real emergency!**

---

## 🎯 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push code to GitHub**
2. **Visit** [share.streamlit.io](https://share.streamlit.io)
3. **Click "New app"**
4. **Select your repository** and branch
5. **Set main file path:** `app.py`
6. **Add secrets** in app settings (GEMINI_API_KEY)
7. **Deploy!** 🚀

### Local Production Run
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📊 Project Status

**Current Version:** 1.0.0 (NASA Space Apps 2025)

### Roadmap
- [ ] Historical disaster data analysis
- [ ] Predictive disaster modeling with ML
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Offline mode for emergency use
- [ ] Integration with more data sources (USGS, NOAA, etc.)
- [ ] User accounts for saved locations and alerts
- [ ] Push notifications for nearby disasters

---

## 🏆 Achievements

✅ Real-time NASA data integration  
✅ AI-powered emergency guidance  
✅ Multi-platform geolocation support  
✅ Image analysis with computer vision  
✅ Worldwide emergency contact database  
✅ Interactive disaster visualization  
✅ Population impact modeling  

---

**Made with ❤️ for NASA Space Apps Challenge 2025**

**#SpaceApps #NASA #DisasterResponse #AI #EmergencyManagement**
