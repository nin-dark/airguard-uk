import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AirGuard UK",
    page_icon="🌬️",
    layout="wide"
)

CITIES = {
    'MY1':  {'name': 'London',     'lat': 51.5223, 'lon': -0.1546,  'code': 0},
    'BIRR': {'name': 'Birmingham', 'lat': 52.4800, 'lon': -1.9025,  'code': 1},
    'MAN3': {'name': 'Manchester', 'lat': 53.4808, 'lon': -2.2426,  'code': 2},
    'NEWC': {'name': 'Newcastle',  'lat': 54.9783, 'lon': -1.6178,  'code': 3},
    'CARD': {'name': 'Cardiff',    'lat': 51.4816, 'lon': -3.1791,  'code': 4},
}

DAQI_BANDS = {
    0: {'label': 'Low',       'colour': '#009900', 'text': 'white'},
    1: {'label': 'Moderate',  'colour': '#FF9900', 'text': 'white'},
    2: {'label': 'High',      'colour': '#CC0000', 'text': 'white'},
    3: {'label': 'Very High', 'colour': '#7C3AED', 'text': 'white'},
}

HEALTH_ADVICE = {
    0: {
        'headline': 'Air quality is good. No action needed.',
        'general':  'Enjoy outdoor activities as normal.',
        'asthma':   'No additional precautions needed.',
        'elderly':  'No additional precautions needed.',
        'children': 'No restrictions on outdoor play.',
    },
    1: {
        'headline': 'Air quality is moderate. At-risk groups take care.',
        'general':  'Unusually sensitive people should consider reducing prolonged outdoor exertion.',
        'asthma':   'Keep your reliever inhaler with you. Watch for symptoms.',
        'elderly':  'If you experience discomfort such as sore eyes, cough or sore throat, consider reducing activity.',
        'children': 'Children with asthma should keep inhalers nearby.',
    },
    2: {
        'headline': 'Air quality is high. Reduce outdoor activity.',
        'general':  'Reduce strenuous outdoor activity, particularly if you experience symptoms.',
        'asthma':   'Use your reliever inhaler if needed. Avoid outdoor exercise.',
        'elderly':  'Avoid strenuous physical activity outdoors.',
        'children': 'Limit time spent outdoors. Keep windows closed.',
    },
    3: {
        'headline': '⚠️ Very High pollution. Avoid outdoors.',
        'general':  'Avoid all outdoor physical exertion.',
        'asthma':   'Stay indoors. Use your preventer and reliever inhaler as prescribed. Seek medical advice if symptoms worsen.',
        'elderly':  'Remain indoors with windows closed. Seek medical advice if unwell.',
        'children': 'Keep children indoors. Contact your GP if they experience breathing difficulties.',
    },
}

FEATURES = ['city_code','hour','day_of_week','month','season','is_weekend',
            'NO2','PM2.5','PM10','O3','SO2','temp','humidity','wind_speed','pressure']

import requests

def fetch_live_data(city_id):
    """
    Fetch current air quality + weather from Open-Meteo for a city.
    Returns a dict of pollutant and weather values.
    """
    info = CITIES[city_id]
    
    # Air quality — current hour
    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        "&current=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide"
        "&timezone=Europe%2FLondon"
    )
    
    # Weather — current hour  
    wx_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        "&current=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,surface_pressure"
        "&timezone=Europe%2FLondon"
    )
    
    try:
        aq  = requests.get(aq_url, timeout=10).json()['current']
        wx  = requests.get(wx_url, timeout=10).json()['current']
        
        return {
            'no2':      round(aq.get('nitrogen_dioxide', 0) or 0, 2),
            'pm25':     round(aq.get('pm2_5',            0) or 0, 2),
            'pm10':     round(aq.get('pm10',             0) or 0, 2),
            'o3':       round(aq.get('ozone',            0) or 0, 2),
            'so2':      round(aq.get('sulphur_dioxide',  0) or 0, 2),
            'temp':     round(wx.get('temperature_2m',   12),     1),
            'humidity': round(wx.get('relative_humidity_2m', 75), 1),
            'wind':     round(wx.get('wind_speed_10m',   15),     1),
            'pressure': round(wx.get('surface_pressure', 1013),   1),
            'success':  True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()


# ─────────────────────────────────────────────────────────────
# HELPER — build feature row for one city
# ─────────────────────────────────────────────────────────────

def build_features(city_id, no2, pm25, pm10, o3, so2, temp, humidity, wind, pressure, dt):
    season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    return pd.DataFrame([{
        'city_code':   CITIES[city_id]['code'],
        'hour':        dt.hour,
        'day_of_week': dt.weekday(),
        'month':       dt.month,
        'season':      season_map[dt.month],
        'is_weekend':  int(dt.weekday() >= 5),
        'NO2':         no2,
        'PM2.5':       pm25,
        'PM10':        pm10,
        'O3':          o3,
        'SO2':         so2,
        'temp':        temp,
        'humidity':    humidity,
        'wind_speed':  wind,
        'pressure':    pressure,
    }])[FEATURES]


# ─────────────────────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────────────────────

st.title("🌬️ AirGuard UK")
st.markdown("**Urban Air Quality Prediction & Health Alert System**  |  SDG 11: Sustainable Cities")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# UI — SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────

st.sidebar.header("📍 Enter Conditions")
st.sidebar.markdown("Input current or forecast readings for a city to get a DAQI prediction.")

selected_city = st.sidebar.selectbox(
    "City",
    options=list(CITIES.keys()),
    format_func=lambda x: CITIES[x]['name']
)

st.sidebar.markdown("**Date & Time**")
col_d, col_t = st.sidebar.columns(2)
input_date = col_d.date_input("Date", value=datetime.today())
input_time = col_t.time_input("Time", value=datetime.now().time())
input_dt   = datetime.combine(input_date, input_time)

# Live data fetch button
fetch_btn = st.sidebar.button("🌐 Fetch Live Data", use_container_width=True)

# Store live values in session state so sliders remember them
if fetch_btn:
    with st.spinner("Fetching live data..."):
        live = fetch_live_data(selected_city)
    if live['success']:
        st.session_state.live = live
        st.sidebar.success("✓ Live data loaded")
    else:
        st.sidebar.error(f"Could not fetch data: {live['error']}")

# Use live values as slider defaults if available
# Clear live data if city changed
if st.session_state.get('last_city') != selected_city:
    st.session_state.pop('live', None)
    st.session_state['last_city'] = selected_city

live_vals = st.session_state.get('live', {})

st.sidebar.markdown("**Pollution Readings (µg/m³)**")
no2      = st.sidebar.slider("NO2",   0.0, 200.0, float(live_vals.get('no2',  35.0)), 0.5)
pm25     = st.sidebar.slider("PM2.5", 0.0, 100.0, float(live_vals.get('pm25', 12.0)), 0.5)
pm10     = st.sidebar.slider("PM10",  0.0, 150.0, float(live_vals.get('pm10', 20.0)), 0.5)
o3       = st.sidebar.slider("O3",    0.0, 250.0, float(live_vals.get('o3',   40.0)), 0.5)
so2      = st.sidebar.slider("SO2",   0.0,  50.0, float(live_vals.get('so2',   1.5)), 0.1)

st.sidebar.markdown("**Weather Conditions**")
temp     = st.sidebar.slider("Temperature (°C)", -10.0,  40.0, float(live_vals.get('temp',     12.0)), 0.5)
humidity = st.sidebar.slider("Humidity (%)",       0.0, 100.0, float(live_vals.get('humidity', 75.0)), 1.0)
wind     = st.sidebar.slider("Wind Speed (km/h)",  0.0, 100.0, float(live_vals.get('wind',     15.0)), 0.5)
pressure = st.sidebar.slider("Pressure (hPa)",   960.0,1050.0, float(live_vals.get('pressure',1013.0)),0.5)

# Show timestamp if live data was loaded
if live_vals:
    st.sidebar.caption(f"📡 Live data: {datetime.now().strftime('%d %b %Y %H:%M')}")

predict_btn = st.sidebar.button("🔍 Predict DAQI", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

# Store all city predictions in session state for the map
if 'city_predictions' not in st.session_state:
    st.session_state.city_predictions = {}

if predict_btn:
    features = build_features(
        selected_city, no2, pm25, pm10, o3, so2,
        temp, humidity, wind, pressure, input_dt
    )
    pred_class = int(model.predict(features)[0])
    pred_proba = model.predict_proba(features)[0]

    st.session_state.city_predictions[selected_city] = {
        'class': pred_class,
        'proba': pred_proba,
    }

    # ── DEBUG PANEL ───────────────────────────────────────
    with st.expander("🔍 Prediction Debug — click to inspect", expanded=True):
        st.markdown("**Inputs sent to model:**")
        st.dataframe(features, use_container_width=True)

        st.markdown("**Raw model output:**")
        col1, col2 = st.columns(2)
        col1.metric("Predicted class", pred_class)
        col1.metric("DAQI band", DAQI_BANDS[pred_class]['label'])
        col2.metric("P(Low)",      f"{pred_proba[0]*100:.1f}%")
        col2.metric("P(Moderate)", f"{pred_proba[1]*100:.1f}%")

        st.markdown("**Class probabilities:**")
        st.bar_chart(pd.DataFrame({
            'Probability %': [round(p*100,2) for p in pred_proba]
        }, index=['Low','Moderate','High','Very High']))

        st.success(f"✓ Prediction complete — {CITIES[selected_city]['name']} → "
                   f"{DAQI_BANDS[pred_class]['label']} "
                   f"(confidence {max(pred_proba)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# UI — MAIN LAYOUT
# ─────────────────────────────────────────────────────────────

col_map, col_detail = st.columns([1.4, 1])

# ── MAP ──────────────────────────────────────────────────────
with col_map:
    st.subheader("🗺️ UK Cities — Predicted DAQI")

    m = folium.Map(
        location=[54.0, -2.5],
        zoom_start=6,
        tiles='CartoDB positron'
    )

    for city_id, info in CITIES.items():
        if city_id in st.session_state.city_predictions:
            pred = st.session_state.city_predictions[city_id]
            band = DAQI_BANDS[pred['class']]
            colour = band['colour']
            label  = band['label']
        else:
            colour = '#94A3B8'
            label  = 'No prediction yet'

        folium.CircleMarker(
            location=[info['lat'], info['lon']],
            radius=18,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            tooltip=f"{info['name']}: {label}",
            popup=f"<b>{info['name']}</b><br>DAQI: {label}"
        ).add_to(m)

        folium.Marker(
            location=[info['lat'] + 0.3, info['lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:#1E293B">{info["name"]}</div>'
            )
        ).add_to(m)

    st_folium(m, width=520, height=480)

    # Legend
    st.markdown("**DAQI Colour Key:**")
    cols = st.columns(4)
    for i, (cls, band) in enumerate(DAQI_BANDS.items()):
        cols[i].markdown(
            f'<div style="background:{band["colour"]};color:white;padding:6px 10px;'
            f'border-radius:6px;text-align:center;font-weight:bold">{band["label"]}</div>',
            unsafe_allow_html=True
        )

# ── DETAIL PANEL ─────────────────────────────────────────────
with col_detail:
    city_name = CITIES[selected_city]['name']

    if selected_city in st.session_state.city_predictions:
        pred  = st.session_state.city_predictions[selected_city]
        cls   = pred['class']
        proba = pred['proba']
        band  = DAQI_BANDS[cls]

        # DAQI result badge
        st.markdown(
            f'<div style="background:{band["colour"]};color:white;padding:16px;'
            f'border-radius:10px;text-align:center;margin-bottom:16px">'
            f'<div style="font-size:13px;opacity:0.9">{city_name}</div>'
            f'<div style="font-size:32px;font-weight:bold">{band["label"]}</div>'
            f'<div style="font-size:13px;opacity:0.9">DAQI Class {cls}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Probability bar chart
        st.markdown("**Prediction Confidence**")
        prob_df = pd.DataFrame({
            'Band':        [DAQI_BANDS[i]['label'] for i in range(4)],
            'Probability': [round(p * 100, 1) for p in proba]
        }).set_index('Band')
        st.bar_chart(prob_df)

        # Health advice
        st.markdown("---")
        st.markdown("### 🏥 Health Advice")
        advice = HEALTH_ADVICE[cls]
        st.info(advice['headline'])

        with st.expander("👤 General public"):
            st.write(advice['general'])
        with st.expander("🫁 Asthma & respiratory conditions"):
            st.write(advice['asthma'])
        with st.expander("👴 Elderly"):
            st.write(advice['elderly'])
        with st.expander("👶 Children"):
            st.write(advice['children'])

        # Input summary
        st.markdown("---")
        st.markdown("**Inputs used for prediction**")
        st.dataframe(pd.DataFrame({
            'Pollutant/Feature': ['NO2','PM2.5','PM10','O3','SO2','Temp','Humidity','Wind','Pressure'],
            'Value': [no2, pm25, pm10, o3, so2, temp, humidity, wind, pressure],
            'Unit':  ['µg/m³','µg/m³','µg/m³','µg/m³','µg/m³','°C','%','km/h','hPa']
        }), hide_index=True, use_container_width=True)

    else:
        st.markdown(f"### {city_name}")
        st.info("👈 Set conditions in the sidebar and click **Predict DAQI** to see results.")
        st.markdown("""
        **How to use AirGuard UK:**
        1. Select a city from the dropdown
        2. Click fetch live data
        3. Select prediction date of upto 48hrs            
        4. Click Predict DAQI
        5. Repeat for multiple cities to populate the map
        """)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:13px'>"
    "AirGuard UK  |  Data: DEFRA AURN · Open-Meteo  |  "
    "Model: XGBoost  |  SDG 11: Sustainable Cities  |  "
    "AI Tool Development Challenge 2026"
    "</div>",
    unsafe_allow_html=True
)