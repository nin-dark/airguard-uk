import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import requests
from streamlit_folium import st_folium
from datetime import datetime, timedelta

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
    0: {'label': 'Low',       'colour': '#009900'},
    1: {'label': 'Moderate',  'colour': '#FF9900'},
    2: {'label': 'Dangerous', 'colour': '#CC0000'},
}

HEALTH_ADVICE = {
    0: {
        'headline': 'Air quality forecast is good. No action needed.',
        'general':  'Enjoy outdoor activities as normal over the next 6 hours.',
        'asthma':   'No additional precautions needed.',
        'elderly':  'No additional precautions needed.',
        'children': 'No restrictions on outdoor play.',
    },
    1: {
        'headline': 'Moderate air quality forecast. At-risk groups take care.',
        'general':  'Sensitive individuals should consider reducing prolonged outdoor exertion.',
        'asthma':   'Keep your reliever inhaler with you. Watch for symptoms.',
        'elderly':  'If you experience sore eyes, cough or sore throat, consider reducing activity.',
        'children': 'Children with asthma should keep inhalers nearby.',
    },
    2: {
        'headline': '⚠️ Dangerous air quality forecast. Reduce outdoor exposure.',
        'general':  'Avoid strenuous outdoor activity over the next 6 hours.',
        'asthma':   'Use your reliever inhaler if needed. Avoid outdoor exercise entirely.',
        'elderly':  'Remain indoors. Keep windows closed.',
        'children': 'Keep children indoors. Contact your GP if breathing difficulties occur.',
    },
}

POLL_COLS = ['NO2','PM2.5','PM10','O3','SO2']
WTHR_COLS = ['temp','humidity','wind_speed','pressure']
ALL_COLS  = POLL_COLS + WTHR_COLS
LAG_HOURS    = [1, 2, 3, 6, 12, 24]
ROLLING_WINS = [6, 12, 24]


# ─────────────────────────────────────────────────────────────
# LOAD MODEL + FEATURES
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model    = joblib.load("models/xgboost_forecast_model.pkl")
    features = pd.read_csv("models/feature_list.csv")['0'].tolist()
    median   = pd.read_csv("models/train_median.csv", index_col=0).squeeze()
    return model, features, median

model, FEATURES, TRAIN_MEDIAN = load_model()


# ─────────────────────────────────────────────────────────────
# FETCH LAST 24 HOURS FROM OPEN-METEO
# ─────────────────────────────────────────────────────────────

def fetch_last_24h(city_id):
    """
    Fetch last 25 hours of air quality + weather from Open-Meteo.
    Returns a DataFrame with columns matching ALL_COLS + datetime.
    """
    info = CITIES[city_id]
    now  = datetime.utcnow()
    start = (now - timedelta(hours=25)).strftime('%Y-%m-%dT%H:00')
    end   = (now + timedelta(hours=12)).strftime('%Y-%m-%dT%H:00')

    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide"
        f"&start_date={start[:10]}&end_date={end[:10]}"
        "&timezone=Europe%2FLondon"
    )
    wx_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,surface_pressure"
        f"&start_date={start[:10]}&end_date={end[:10]}"
        "&timezone=Europe%2FLondon"
    )

    try:
        aq_r = requests.get(aq_url, timeout=10).json()['hourly']
        wx_r = requests.get(wx_url, timeout=10).json()['hourly']

        df = pd.DataFrame({
            'datetime':   pd.to_datetime(aq_r['time']),
            'NO2':        aq_r['nitrogen_dioxide'],
            'PM2.5':      aq_r['pm2_5'],
            'PM10':       aq_r['pm10'],
            'O3':         aq_r['ozone'],
            'SO2':        aq_r['sulphur_dioxide'],
            'temp':       wx_r['temperature_2m'],
            'humidity':   wx_r['relative_humidity_2m'],
            'wind_speed': wx_r['wind_speed_10m'],
            'pressure':   wx_r['surface_pressure'],
        })

        # Clip negatives, fill NaN
        for col in POLL_COLS:
            df[col] = df[col].clip(lower=0)
        df['SO2'] = df['SO2'].fillna(0)
        for col in ALL_COLS:
            df[col] = df[col].fillna(df[col].median())

        # Keep only last 25 rows
        df = df.sort_values('datetime').tail(25).reset_index(drop=True)
        return df, None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# BUILD LAG FEATURES FROM 25-ROW HISTORY
# ─────────────────────────────────────────────────────────────

def build_forecast_features(df, city_id):
    """
    Given 25 rows of hourly history, build lag features for the
    most recent row and return a single-row feature DataFrame.
    """
    df = df.copy().reset_index(drop=True)
    new_cols = {}

    for col in ALL_COLS:
        for lag in LAG_HOURS:
            new_cols[f'{col}_lag{lag}h'] = df[col].shift(lag)
        for win in ROLLING_WINS:
            new_cols[f'{col}_mean{win}h'] = df[col].shift(1).rolling(win).mean()
        for win in [6, 12]:
            new_cols[f'{col}_max{win}h']  = df[col].shift(1).rolling(win).max()

    lag_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Take only the last row — the most recent hour
    row = lag_df.iloc[[-1]].copy()

    # Add time features
    dt = row['datetime'].iloc[0]
    season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    row['city_code']   = CITIES[city_id]['code']
    row['hour']        = dt.hour
    row['day_of_week'] = dt.weekday()
    row['month']       = dt.month
    row['season']      = season_map[dt.month]
    row['is_weekend']  = int(dt.weekday() >= 5)

    # Select and fill features
    X = row[FEATURES].fillna(TRAIN_MEDIAN)
    return X, dt

def run_multi_step_forecast(hist_df, city_id, steps=2, step_hours=6):
    """
    Iteratively predict step_hours ahead, `steps` times.
    Returns list of dicts: {forecast_dt, class, proba}
    """
    results = []
    df = hist_df.copy()

    for step in range(steps):
        X, last_dt = build_forecast_features(df, city_id)
        pred_class = int(model.predict(X)[0])
        pred_proba = model.predict_proba(X)[0]
        forecast_dt = last_dt + timedelta(hours=step_hours)

        results.append({
            'step':        step + 1,
            'forecast_dt': forecast_dt,
            'class':       pred_class,
            'proba':       pred_proba,
        })

        # Build a synthetic next row using last known values + future weather
        last_row = df.iloc[-1].copy()
        next_row = last_row.copy()
        next_row['datetime'] = forecast_dt

        # Use future weather from the fetched data if available
        future_wx = hist_df[hist_df['datetime'] == forecast_dt]
        if not future_wx.empty:
            for col in WTHR_COLS:
                next_row[col] = future_wx.iloc[0][col]

        # Pollutants: use rolling mean as best estimate
        for col in POLL_COLS:
            next_row[col] = df[col].tail(6).mean()

        df = pd.concat([df, pd.DataFrame([next_row])], ignore_index=True)

    return results

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.title("🌬️ AirGuard UK")
st.markdown("**12-Hour Air Quality Forecast & Health Alert System**  |  SDG 11: Sustainable Cities")
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.header("📍 City Forecast")
st.sidebar.markdown(
    "Select a city and click **Run Forecast**. "
    "AirGuard fetches the last 24 hours of real pollution and weather data, "
    "then predicts the DAQI band **6 hours from now**."
)

selected_city = st.sidebar.selectbox(
    "City",
    options=list(CITIES.keys()),
    format_func=lambda x: CITIES[x]['name']
)

run_btn = st.sidebar.button("🔍 Run Forecast", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown(
    "1. Fetches last 24h of PM10, PM2.5, NO2, O3, SO2, temperature, "
    "humidity, wind and pressure from Open-Meteo\n"
    "2. Builds 105 lag and rolling-average features\n"
    "3. XGBoost model predicts DAQI band 6 hours ahead\n"
    "4. Health advice issued based on forecast band"
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Open-Meteo Air Quality API · Open-Meteo Weather API")
st.sidebar.caption("Model trained on DEFRA AURN 2021–2024")


# ─────────────────────────────────────────────────────────────
# RUN FORECAST
# ─────────────────────────────────────────────────────────────

if 'city_predictions' not in st.session_state:
    st.session_state.city_predictions = {}
if 'city_history' not in st.session_state:
    st.session_state.city_history = {}

if run_btn:
    with st.spinner(f"Fetching live data for {CITIES[selected_city]['name']}..."):
        hist_df, err = fetch_last_24h(selected_city)

    if err:
        st.error(f"Could not fetch data: {err}")
    else:
        X, last_dt = build_forecast_features(hist_df, selected_city)
        steps = run_multi_step_forecast(hist_df, selected_city, steps=2, step_hours=6)

        st.session_state.city_predictions[selected_city] = {
            'class':       steps[0]['class'],   # t+6h (primary, used for map colour)
            'proba':       steps[0]['proba'],
            'last_dt':     last_dt,
            'forecast_dt': steps[0]['forecast_dt'],
            'steps':       steps,               # full 12h timeline
        }
        st.session_state.city_history[selected_city] = hist_df


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────

col_map, col_detail = st.columns([1.4, 1])

# ── MAP ──────────────────────────────────────────────────────
with col_map:
    st.subheader("🗺️ UK Cities — 12h DAQI Forecast")

    m = folium.Map(location=[54.0, -2.5], zoom_start=6, tiles='CartoDB positron')

    for city_id, info in CITIES.items():
        if city_id in st.session_state.city_predictions:
            pred   = st.session_state.city_predictions[city_id]
            band   = DAQI_BANDS[pred['class']]
            colour = band['colour']
            label  = band['label']
            conf   = f"{max(pred['proba'])*100:.0f}% confidence"
        else:
            colour = '#94A3B8'
            label  = 'No forecast yet'
            conf   = ''

        folium.CircleMarker(
            location=[info['lat'], info['lon']],
            radius=18,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            tooltip=f"{info['name']}: {label} {conf}",
            popup=f"<b>{info['name']}</b><br>12h Forecast: {label}<br>{conf}"
        ).add_to(m)

        folium.Marker(
            location=[info['lat'] + 0.3, info['lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;'
                     f'color:#1E293B">{info["name"]}</div>'
            )
        ).add_to(m)

    st_folium(m, width=520, height=480)

    st.markdown("**DAQI Forecast Colour Key:**")
    cols = st.columns(3)
    for i, (cls, band) in enumerate(DAQI_BANDS.items()):
        cols[i].markdown(
            f'<div style="background:{band["colour"]};color:white;padding:6px 10px;'
            f'border-radius:6px;text-align:center;font-weight:bold">'
            f'{band["label"]}</div>',
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
        hist  = st.session_state.city_history[selected_city]

        # Forecast timestamp
        if 'steps' in pred:
            st.markdown("### 🕐 12-Hour Forecast Timeline")
            for s in pred['steps']:
                b = DAQI_BANDS[s['class']]
                st.markdown(
                f'<div style="background:{b["colour"]};color:white;padding:10px;'
                f'border-radius:8px;margin-bottom:8px;display:flex;justify-content:space-between">'
                f'<span>+{s["step"]*6}h — {s["forecast_dt"].strftime("%H:%M")}</span>'
                f'<span><b>{b["label"]}</b> ({max(s["proba"])*100:.0f}% confidence)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("")

        # DAQI badge
        st.markdown(
            f'<div style="background:{band["colour"]};color:white;padding:16px;'
            f'border-radius:10px;text-align:center;margin-bottom:16px">'
            f'<div style="font-size:13px;opacity:0.9">{city_name} — 6h Forecast</div>'
            f'<div style="font-size:36px;font-weight:bold">{band["label"]}</div>'
            f'<div style="font-size:13px;opacity:0.9">'
            f'Confidence: {max(proba)*100:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Probability bars
        st.markdown("**Forecast Confidence by Band**")
        prob_df = pd.DataFrame({
            'Probability %': [round(proba[i]*100, 1) for i in range(3)]
        }, index=['Low','Moderate','Dangerous'])
        st.bar_chart(prob_df)

        # Health advice
        st.markdown("---")
        st.markdown("### 🏥 Health Advice — Next 6 Hours")
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

        # Last 24h pollution trend
        st.markdown("---")
        st.markdown("**Last 24h Pollution Trend (inputs to model)**")
        chart_df = hist[['datetime','PM10','PM2.5','NO2','O3']].set_index('datetime').tail(24)
        st.line_chart(chart_df)

        # Latest readings table
        st.markdown("**Most Recent Hourly Reading**")
        latest = hist.iloc[-1]
        st.dataframe(pd.DataFrame({
            'Pollutant/Variable': ['NO2','PM2.5','PM10','O3','SO2',
                                   'Temp','Humidity','Wind','Pressure'],
            'Value': [
                latest['NO2'], latest['PM2.5'], latest['PM10'],
                latest['O3'],  latest['SO2'],   latest['temp'],
                latest['humidity'], latest['wind_speed'], latest['pressure']
            ],
            'Unit': ['µg/m³','µg/m³','µg/m³','µg/m³','µg/m³',
                     '°C','%','km/h','hPa']
        }), hide_index=True, use_container_width=True)

    else:
        st.markdown(f"### {city_name}")
        st.info("👈 Click **Run Forecast** to fetch live data and predict air quality 6 hours ahead.")
        st.markdown("""
        **What AirGuard UK does:**
        - Fetches real pollution and weather data from the past 24 hours
        - Builds 105 lag and rolling-average features
        - XGBoost model forecasts DAQI band 6 hours into the future
        - Issues targeted health advice by risk group

        **Run forecasts for all 5 cities to populate the map.**
        """)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:13px'>"
    "AirGuard UK  |  Data: DEFRA AURN · Open-Meteo  |  "
    "Model: XGBoost with 105 lag features  |  "
    "SDG 11: Sustainable Cities  |  AI Tool Development Challenge 2026"
    "</div>",
    unsafe_allow_html=True
)