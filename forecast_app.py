import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import requests
import shap
import plotly.graph_objects as go
from streamlit_folium import st_folium
from datetime import datetime, timedelta, timezone
import os
import json

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

ALERT_TIERS = {
    'low':       {'label': 'Low',           'colour': '#009900', 'emoji': '🟢'},
    'moderate':  {'label': 'Moderate',      'colour': '#FF9900', 'emoji': '🟡'},
    'watch':     {'label': 'Watch',         'colour': '#EAB308', 'emoji': '⚡'},
    'elevated':  {'label': 'Elevated Risk', 'colour': '#F97316', 'emoji': '🟠'},
    'dangerous': {'label': 'Dangerous',     'colour': '#CC0000', 'emoji': '🔴'},
}

DAQI_BANDS = {
    0: {'label': 'Low',       'colour': '#009900'},
    1: {'label': 'Moderate',  'colour': '#FF9900'},
    2: {'label': 'Dangerous', 'colour': '#CC0000'},
}

HEALTH_ADVICE = {
    'low': {
        'headline': 'Air quality forecast is good. No action needed.',
        'general':  'Enjoy outdoor activities as normal.',
        'asthma':   'No additional precautions needed.',
        'elderly':  'No additional precautions needed.',
        'children': 'No restrictions on outdoor play.',
    },
    'moderate': {
        'headline': 'Moderate air quality forecast. At-risk groups take care.',
        'general':  'Sensitive individuals should consider reducing prolonged outdoor exertion.',
        'asthma':   'Keep your reliever inhaler with you. Watch for symptoms.',
        'elderly':  'If you experience sore eyes, cough or sore throat, consider reducing activity.',
        'children': 'Children with asthma should keep inhalers nearby.',
    },
    'watch': {
        'headline': '⚡ Pollution signals detected. At-risk groups should monitor conditions.',
        'general':  'Consider limiting prolonged outdoor activity, especially near traffic.',
        'asthma':   'Carry your reliever inhaler. Reduce outdoor exercise if you feel any symptoms.',
        'elderly':  'Limit strenuous outdoor activity. Monitor for respiratory discomfort.',
        'children': 'Limit extended outdoor exercise. Keep inhalers accessible.',
    },
    'elevated': {
        'headline': '🟠 Elevated pollution risk forecast. Reduce outdoor exposure.',
        'general':  'Avoid prolonged strenuous outdoor activity.',
        'asthma':   'Use your reliever inhaler if needed. Avoid outdoor exercise.',
        'elderly':  'Limit time outdoors. Keep windows closed when indoors.',
        'children': 'Keep outdoor activity brief. Contact GP if breathing difficulties occur.',
    },
    'dangerous': {
        'headline': '🔴 Dangerous air quality forecast. Minimise outdoor exposure.',
        'general':  'Avoid all strenuous outdoor activity.',
        'asthma':   'Use your reliever inhaler if needed. Stay indoors.',
        'elderly':  'Remain indoors. Keep windows closed.',
        'children': 'Keep children indoors. Contact your GP if breathing difficulties occur.',
    },
}

WHO_LIMITS = {
    'NO2':   40.0,
    'PM2.5': 15.0,
    'PM10':  45.0,
    'O3':    100.0,
}

POLL_COLS    = ['NO2', 'PM2.5', 'PM10', 'O3', 'SO2']
WTHR_COLS    = ['temp', 'humidity', 'wind_speed', 'pressure']
ALL_COLS     = POLL_COLS + WTHR_COLS
LAG_HOURS    = [1, 2, 3, 6, 12, 24, 48]
ROLLING_WINS = [6, 12, 24, 48, 72]

FORECAST_LOG = "models/forecast_log.csv"


# ─────────────────────────────────────────────────────────────
# DATETIME HELPER — always naive UTC, never tz-aware
# ─────────────────────────────────────────────────────────────

def _now_utc():
    """Current UTC time as a tz-naive datetime — safe to compare with pandas Timestamps."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS + FEATURES
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    model_6h  = joblib.load("models/xgboost_6h_best.pkl")
    model_12h = joblib.load("models/xgboost_12h_best.pkl")

    # Force CPU inference — models were trained on CUDA but Streamlit runs on CPU
    model_6h.set_params(device='cpu')
    model_12h.set_params(device='cpu')

    features = pd.read_csv("models/feature_list.csv")['0'].tolist()
    median   = pd.read_csv("models/train_median.csv", index_col=0).squeeze()

    with open("models/thresholds.json") as f:
        thresholds = json.load(f)

    explainer_6h  = shap.TreeExplainer(model_6h)
    explainer_12h = shap.TreeExplainer(model_12h)

    return model_6h, model_12h, features, median, thresholds, explainer_6h, explainer_12h

(model_6h, model_12h, FEATURES, TRAIN_MEDIAN,
 THRESHOLDS, explainer_6h, explainer_12h) = load_models()


# ─────────────────────────────────────────────────────────────
# TIERED CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_tiered(proba, horizon):
    p2     = proba[2]
    thresh = THRESHOLDS[horizon]   # 0.10 for 6h, 0.04 for 12h

    if p2 >= 0.30:
        return 'dangerous'
    elif p2 >= 0.10:
        return 'elevated'
    elif p2 >= thresh:
        return 'watch'
    else:
        return 'moderate' if proba[1] >= 0.45 else 'low'


# ─────────────────────────────────────────────────────────────
# FETCH DATA FROM OPEN-METEO
# ─────────────────────────────────────────────────────────────

def fetch_last_24h(city_id):
    info = CITIES[city_id]
    now  = _now_utc()

    start = (now - timedelta(hours=73)).strftime('%Y-%m-%d')
    end   = now.strftime('%Y-%m-%d')

    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide"
        f"&start_date={start}&end_date={end}"
        "&timezone=Europe%2FLondon"
    )

    # Use past_days instead of start/end — more reliable on the forecast endpoint
    wx_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={info['lat']}&longitude={info['lon']}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
        f"&past_days=3&forecast_days=1"
        "&timezone=Europe%2FLondon"
    )

    try:
        aq_resp = requests.get(aq_url, timeout=10).json()
        wx_resp = requests.get(wx_url, timeout=10).json()

        if 'hourly' not in aq_resp:
            return None, f"AQ API error: {aq_resp.get('reason', str(aq_resp))}"
        if 'hourly' not in wx_resp:
            return None, f"Weather API error: {wx_resp.get('reason', str(wx_resp))}"

        aq_r = aq_resp['hourly']
        wx_r = wx_resp['hourly']

        aq_df = pd.DataFrame({
            'datetime': pd.to_datetime(aq_r['time']),
            'NO2':      aq_r['nitrogen_dioxide'],
            'PM2.5':    aq_r['pm2_5'],
            'PM10':     aq_r['pm10'],
            'O3':       aq_r['ozone'],
            'SO2':      aq_r['sulphur_dioxide'],
        })
        wx_df = pd.DataFrame({
            'datetime':   pd.to_datetime(wx_r['time']),
            'temp':       wx_r['temperature_2m'],
            'humidity':   wx_r['relative_humidity_2m'],
            'wind_speed': wx_r['wind_speed_10m'],
            'pressure':   wx_r['surface_pressure'],
        })

        df = pd.merge(aq_df, wx_df, on='datetime', how='inner')

        for col in POLL_COLS:
            df[col] = df[col].clip(lower=0)
        df['SO2'] = df['SO2'].fillna(0)
        for col in ALL_COLS:
            df[col] = df[col].fillna(df[col].median())

        df = df.sort_values('datetime').tail(73).reset_index(drop=True)

        if len(df) < 50:
            return None, f"Insufficient data ({len(df)} rows). Try again shortly."

        return df, None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# BUILD LAG FEATURES
# ─────────────────────────────────────────────────────────────

def build_forecast_features(df, city_id):
    df       = df.copy().reset_index(drop=True)
    new_cols = {}

    for col in ALL_COLS:
        for lag in LAG_HOURS:
            new_cols[f'{col}_lag{lag}h'] = df[col].shift(lag)
        for win in ROLLING_WINS:
            new_cols[f'{col}_mean{win}h'] = df[col].shift(1).rolling(win).mean()
        for win in [6, 12]:
            new_cols[f'{col}_max{win}h'] = df[col].shift(1).rolling(win).max()

    for col in POLL_COLS:
        new_cols[f'{col}_trend_3h'] = df[col].shift(1) - df[col].shift(4)
        new_cols[f'{col}_trend_6h'] = df[col].shift(1) - df[col].shift(7)
        new_cols[f'{col}_vs_24h']   = df[col] - df[col].shift(24)

    for col in ['PM10', 'PM2.5', 'NO2']:
        new_cols[f'{col}_roc_3h'] = (
            (df[col] - df[col].shift(3)) / (df[col].shift(3) + 1e-6)
        )

    lag_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    row    = lag_df.iloc[[-1]].copy()

    dt         = row['datetime'].iloc[0]
    season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    row['city_code']   = CITIES[city_id]['code']
    row['hour']        = dt.hour
    row['day_of_week'] = dt.weekday()
    row['month']       = dt.month
    row['season']      = season_map[dt.month]
    row['is_weekend']  = int(dt.weekday() >= 5)

    X = row[FEATURES].fillna(TRAIN_MEDIAN)
    return X, dt


# ─────────────────────────────────────────────────────────────
# FORECAST ACCURACY TRACKER
# ─────────────────────────────────────────────────────────────

def _to_naive_utc(series):
    """Parse datetime strings to naive UTC regardless of format or tz suffix."""
    return pd.to_datetime(series, format='mixed', utc=True).dt.tz_localize(None)


def load_forecast_log():
    if os.path.exists(FORECAST_LOG):
        log = pd.read_csv(FORECAST_LOG)

        for col in ['predicted_at', 'forecast_for']:
            if col in log.columns:
                log[col] = _to_naive_utc(log[col])

        if 'horizon' not in log.columns:
            log['horizon'] = '6h'
        if 'predicted_tier' not in log.columns:
            log['predicted_tier'] = log['predicted_class'].map(
                {0: 'low', 1: 'moderate', 2: 'dangerous'}
            )

        # Prevent FutureWarning when assigning strings into float columns
        for col in ['actual_label', 'predicted_label', 'predicted_tier', 'horizon']:
            if col in log.columns:
                log[col] = log[col].astype(object)

        return log

    return pd.DataFrame(columns=[
        'predicted_at', 'forecast_for', 'city', 'city_name', 'horizon',
        'predicted_class', 'predicted_label', 'predicted_tier',
        'actual_class', 'actual_label', 'correct'
    ])


def save_forecast_entry(city_id, pred_class, pred_tier, forecast_dt, horizon):
    log = load_forecast_log()
    new_row = {
        'predicted_at':    _now_utc(),
        'forecast_for':    forecast_dt,
        'city':            city_id,
        'city_name':       CITIES[city_id]['name'],
        'horizon':         horizon,
        'predicted_class': pred_class,
        'predicted_label': DAQI_BANDS[pred_class]['label'],
        'predicted_tier':  pred_tier,
        'actual_class':    None,
        'actual_label':    None,
        'correct':         None,
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    os.makedirs("models", exist_ok=True)
    log.to_csv(FORECAST_LOG, index=False)


def compute_daqi_from_readings(no2, pm25, pm10, o3, so2):
    def sub_index(val, bands):
        if val is None or np.isnan(val): return 1
        for i, t in enumerate(bands):
            if val <= t: return i + 1
        return 10
    score = max(
        sub_index(no2,  [67,134,200,267,334,400,467,534,600]),
        sub_index(pm25, [11,23,35,41,47,53,58,64,70]),
        sub_index(pm10, [16,33,50,58,66,75,83,91,100]),
        sub_index(o3,   [33,66,100,120,140,160,187,213,240]),
        sub_index(so2,  [88,177,266,354,443,532,710,887,1064]),
    )
    if score <= 3: return 0
    if score <= 6: return 1
    return 2


def verify_past_forecasts():
    log = load_forecast_log()
    if log.empty:
        return log

    now     = _now_utc()
    updated = False

    for idx, row in log.iterrows():
        if pd.notna(row['actual_class']):
            continue
        if pd.isna(row['forecast_for']):
            continue

        forecast_dt = pd.to_datetime(row['forecast_for'])
        if forecast_dt.tzinfo is not None:
            forecast_dt = forecast_dt.replace(tzinfo=None)
        if forecast_dt > now:
            continue

        city_id  = row['city']
        info     = CITIES[city_id]
        date_str = forecast_dt.strftime('%Y-%m-%d')

        try:
            aq_url = (
                "https://air-quality-api.open-meteo.com/v1/air-quality"
                f"?latitude={info['lat']}&longitude={info['lon']}"
                f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide"
                f"&start_date={date_str}&end_date={date_str}"
                "&timezone=Europe%2FLondon"
            )
            resp = requests.get(aq_url, timeout=10).json()
            if 'hourly' not in resp:
                continue

            aq_r        = resp['hourly']
            times       = pd.to_datetime(aq_r['time'])
            target_hour = forecast_dt.replace(minute=0, second=0, microsecond=0)
            idx_match   = np.where(times == target_hour)[0]

            if len(idx_match) > 0:
                i            = idx_match[0]
                actual_class = compute_daqi_from_readings(
                    aq_r['nitrogen_dioxide'][i],
                    aq_r['pm2_5'][i],
                    aq_r['pm10'][i],
                    aq_r['ozone'][i],
                    aq_r['sulphur_dioxide'][i] if aq_r['sulphur_dioxide'][i] else 0,
                )
                log.at[idx, 'actual_class'] = actual_class
                log.at[idx, 'actual_label'] = DAQI_BANDS[actual_class]['label']
                log.at[idx, 'correct']      = int(actual_class == row['predicted_class'])
                updated = True
        except Exception:
            pass

    if updated:
        log.to_csv(FORECAST_LOG, index=False)

    return log


# ─────────────────────────────────────────────────────────────
# POLLUTION TREND CHART
# ─────────────────────────────────────────────────────────────

def make_pollution_chart(hist_df):
    fig     = go.Figure()
    colours = {'NO2': '#3B82F6', 'PM2.5': '#F59E0B', 'PM10': '#10B981', 'O3': '#8B5CF6'}

    for col, colour in colours.items():
        fig.add_trace(go.Scatter(
            x=hist_df['datetime'].tail(24),
            y=hist_df[col].tail(24),
            mode='lines+markers', name=col,
            line=dict(color=colour, width=2), marker=dict(size=4),
        ))

    for pollutant, limit in WHO_LIMITS.items():
        if pollutant in colours:
            fig.add_hline(
                y=limit, line_dash='dash',
                line_color=colours[pollutant], opacity=0.5,
                annotation_text=f"WHO {pollutant}: {limit}µg/m³",
                annotation_position="bottom right", annotation_font_size=10,
            )

    fig.update_layout(
        title='Last 24h Pollution Trend — dashed lines = WHO guidelines',
        xaxis_title='Time', yaxis_title='µg/m³',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350, margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SHAP CHART
# ─────────────────────────────────────────────────────────────

def make_shap_chart(X, pred_class, tier, explainer):
    explanation = explainer(X)
    sv          = explanation.values

    if sv.ndim == 3:
        shap_vals = sv[0, :, pred_class]
    elif sv.ndim == 2:
        shap_vals = sv[0, :]
    else:
        shap_vals = np.zeros(len(FEATURES))

    shap_df = pd.DataFrame({'feature': FEATURES, 'shap': shap_vals})
    shap_df = shap_df.reindex(
        pd.Series(np.abs(shap_vals)).sort_values(ascending=False).index
    ).head(10).iloc[::-1]

    tier_colour = ALERT_TIERS[tier]['colour']
    tier_label  = ALERT_TIERS[tier]['label']
    bar_colours = [tier_colour if v > 0 else '#009900' for v in shap_df['shap']]

    fig = go.Figure(go.Bar(
        x=shap_df['shap'], y=shap_df['feature'], orientation='h',
        marker_color=bar_colours,
        text=[f"{v:.3f}" for v in shap_df['shap']], textposition='outside',
    ))
    fig.update_layout(
        title=f'Why this forecast? — Top 10 SHAP drivers (predicting: {tier_label})',
        xaxis_title=(
            f'SHAP value  ({ALERT_TIERS[tier]["emoji"]} = pushes toward {tier_label}, '
            f'🟢 = toward Low)'
        ),
        height=380, margin=dict(l=160, r=60, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='#334155', zeroline=True, zerolinecolor='#64748B'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.title("🌬️ AirGuard UK")
st.markdown(
    "**6-Hour & 12-Hour Air Quality Forecast & Health Alert System**  "
    "|  SDG 13 : Climate Action"
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.header("📍 City Forecast")

horizon = st.sidebar.radio(
    "Forecast horizon", options=["6h", "12h"], horizontal=True,
    help="Each horizon uses a separately trained XGBoost model.",
)

st.sidebar.markdown(
    "Select a city or forecast all 5 at once. "
    "AirGuard fetches the last 73 hours of real pollution and weather data "
    f"and predicts the DAQI band **{horizon} from now**."
)

selected_city = st.sidebar.selectbox(
    "City", options=list(CITIES.keys()),
    format_func=lambda x: CITIES[x]['name'],
)

run_btn     = st.sidebar.button("🔍 Run Forecast",        use_container_width=True)
run_all_btn = st.sidebar.button("🌍 Forecast All Cities", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Alert tiers explained:**")
for key, tier in ALERT_TIERS.items():
    st.sidebar.markdown(
        f'<span style="color:{tier["colour"]};font-weight:bold">'
        f'{tier["emoji"]} {tier["label"]}</span>',
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown(
    "1. Fetches last 73h of PM10, PM2.5, NO2, O3, SO2, "
    "temperature, humidity, wind and pressure from Open-Meteo\n"
    "2. Builds 150 lag and rolling-average features\n"
    "3. XGBoost model predicts DAQI band 6h or 12h ahead\n"
    "4. Tiered alert level derived from Dangerous class probability\n"
    "5. SHAP values explain which features drove the prediction\n"
    "6. Health advice issued based on alert tier"
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Open-Meteo Air Quality API · Open-Meteo Weather API")
st.sidebar.caption("Model trained on DEFRA AURN 2021–2024")


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────

if 'city_predictions' not in st.session_state:
    st.session_state.city_predictions = {}
if 'city_history' not in st.session_state:
    st.session_state.city_history = {}
if 'city_shap' not in st.session_state:
    st.session_state.city_shap = {}


# ─────────────────────────────────────────────────────────────
# FORECAST FUNCTION
# ─────────────────────────────────────────────────────────────

def run_forecast_for(city_id):
    hist_df, err = fetch_last_24h(city_id)
    if err:
        return False, err

    X, last_dt = build_forecast_features(hist_df, city_id)

    proba_6h     = model_6h.predict_proba(X)[0]
    cls_6h       = int(model_6h.predict(X)[0])
    tier_6h      = classify_tiered(proba_6h, '6h')
    forecast_6h  = last_dt + timedelta(hours=6)

    proba_12h    = model_12h.predict_proba(X)[0]
    cls_12h      = int(model_12h.predict(X)[0])
    tier_12h     = classify_tiered(proba_12h, '12h')
    forecast_12h = last_dt + timedelta(hours=12)

    st.session_state.city_predictions[city_id] = {
        '6h':  {'class': cls_6h,  'proba': proba_6h,  'tier': tier_6h,
                'last_dt': last_dt, 'forecast_dt': forecast_6h},
        '12h': {'class': cls_12h, 'proba': proba_12h, 'tier': tier_12h,
                'last_dt': last_dt, 'forecast_dt': forecast_12h},
    }
    st.session_state.city_history[city_id] = hist_df
    st.session_state.city_shap[city_id]    = X

    save_forecast_entry(city_id, cls_6h,  tier_6h,  forecast_6h,  '6h')
    save_forecast_entry(city_id, cls_12h, tier_12h, forecast_12h, '12h')
    return True, None


# ─────────────────────────────────────────────────────────────
# RUN BUTTONS
# ─────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Fetching live data for {CITIES[selected_city]['name']}..."):
        ok, err = run_forecast_for(selected_city)
    if not ok:
        st.error(f"Could not fetch data: {err}")

if run_all_btn:
    progress = st.progress(0, text="Starting forecasts...")
    for i, city_id in enumerate(CITIES.keys()):
        progress.progress(i / len(CITIES), text=f"Fetching {CITIES[city_id]['name']}...")
        ok, err = run_forecast_for(city_id)
        if not ok:
            st.warning(f"Could not fetch {CITIES[city_id]['name']}: {err}")
    progress.progress(1.0, text="All cities forecasted ✓")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab_map, tab_accuracy = st.tabs(["🗺️ Forecast Map", "📊 Forecast Accuracy"])


# ─────────────────────────────────────────────────────────────
# TAB 1 — FORECAST MAP
# ─────────────────────────────────────────────────────────────

with tab_map:
    col_map, col_detail = st.columns([1.4, 1])

    with col_map:
        st.subheader(f"UK Cities — {horizon} DAQI Forecast")

        m = folium.Map(location=[54.0, -2.5], zoom_start=6, tiles='CartoDB positron')

        for city_id, info in CITIES.items():
            if city_id in st.session_state.city_predictions:
                pred   = st.session_state.city_predictions[city_id][horizon]
                tier   = ALERT_TIERS[pred['tier']]
                colour = tier['colour']
                label  = f"{tier['emoji']} {tier['label']}"
                conf   = f"D-prob: {pred['proba'][2]*100:.1f}%"
            else:
                colour, label, conf = '#94A3B8', 'No forecast yet', ''

            folium.CircleMarker(
                location=[info['lat'], info['lon']],
                radius=18, color=colour,
                fill=True, fill_color=colour, fill_opacity=0.85,
                tooltip=f"{info['name']}: {label}  {conf}",
                popup=f"<b>{info['name']}</b><br>{horizon} Forecast: {label}<br>{conf}",
            ).add_to(m)

            folium.Marker(
                location=[info['lat'] + 0.3, info['lon']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:11px;font-weight:bold;'
                         f'color:#1E293B">{info["name"]}</div>'
                ),
            ).add_to(m)

        st_folium(m, width=520, height=480)

        st.markdown("**Alert Tier Colour Key:**")
        cols = st.columns(len(ALERT_TIERS))
        for i, (key, tier) in enumerate(ALERT_TIERS.items()):
            cols[i].markdown(
                f'<div style="background:{tier["colour"]};color:white;'
                f'padding:6px 4px;border-radius:6px;text-align:center;'
                f'font-weight:bold;font-size:12px">'
                f'{tier["emoji"]} {tier["label"]}</div>',
                unsafe_allow_html=True,
            )

    with col_detail:
        city_name = CITIES[selected_city]['name']

        if selected_city in st.session_state.city_predictions:
            pred      = st.session_state.city_predictions[selected_city][horizon]
            cls       = pred['class']
            proba     = pred['proba']
            tier_key  = pred['tier']
            tier_info = ALERT_TIERS[tier_key]
            hist      = st.session_state.city_history[selected_city]
            X_row     = st.session_state.city_shap[selected_city]
            explainer = explainer_6h if horizon == '6h' else explainer_12h

            st.markdown(
                f"<div style='color:#94A3B8;font-size:13px'>"
                f"📡 Data up to: {pred['last_dt'].strftime('%d %b %Y %H:%M')}  |  "
                f"🔮 Forecast for: {pred['forecast_dt'].strftime('%d %b %Y %H:%M')}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            st.markdown(
                f'<div style="background:{tier_info["colour"]};color:white;padding:16px;'
                f'border-radius:10px;text-align:center;margin-bottom:16px">'
                f'<div style="font-size:13px;opacity:0.9">{city_name} — {horizon} Forecast</div>'
                f'<div style="font-size:36px;font-weight:bold">'
                f'{tier_info["emoji"]} {tier_info["label"]}</div>'
                f'<div style="font-size:13px;opacity:0.9">'
                f'Dangerous signal: {proba[2]*100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Forecast Confidence by Band**")
            st.bar_chart(pd.DataFrame(
                {'Probability %': [round(proba[i]*100, 1) for i in range(3)]},
                index=['Low', 'Moderate', 'Dangerous']
            ))

            st.markdown("---")
            st.markdown(f"### 🏥 Health Advice — Next {horizon}")
            advice = HEALTH_ADVICE[tier_key]
            st.info(advice['headline'])
            with st.expander("👤 General public"):
                st.write(advice['general'])
            with st.expander("🫁 Asthma & respiratory conditions"):
                st.write(advice['asthma'])
            with st.expander("👴 Elderly"):
                st.write(advice['elderly'])
            with st.expander("👶 Children"):
                st.write(advice['children'])

            st.markdown("---")
            st.plotly_chart(make_pollution_chart(hist), width='stretch')

            st.markdown("---")
            with st.spinner("Computing SHAP explanation..."):
                shap_fig = make_shap_chart(X_row, cls, tier_key, explainer)
            st.plotly_chart(shap_fig, width='stretch')
            st.caption(
                f"{tier_info['emoji']} bars = features pushing toward {tier_info['label']}. "
                "🟢 bars = features pushing toward Low. "
                "Bar length = strength of influence."
            )

            st.markdown("---")
            st.markdown("**Most Recent Hourly Reading**")
            latest = hist.iloc[-1]
            st.dataframe(pd.DataFrame({
                'Pollutant/Variable': [
                    'NO2','PM2.5','PM10','O3','SO2',
                    'Temp','Humidity','Wind','Pressure'
                ],
                'Value': [
                    latest['NO2'],      latest['PM2.5'],    latest['PM10'],
                    latest['O3'],       latest['SO2'],       latest['temp'],
                    latest['humidity'], latest['wind_speed'], latest['pressure'],
                ],
                'Unit': ['µg/m³','µg/m³','µg/m³','µg/m³','µg/m³',
                         '°C','%','km/h','hPa'],
                'WHO Guideline': [
                    '40 µg/m³','15 µg/m³','45 µg/m³','100 µg/m³','—',
                    '—','—','—','—',
                ],
            }), hide_index=True, width='stretch')

        else:
            st.markdown(f"### {city_name}")
            st.info("👈 Click **Run Forecast** or **Forecast All Cities** to begin.")
            st.markdown("""
            **What AirGuard UK does:**
            - Fetches real pollution & weather data from the past 73 hours
            - Builds 150 lag and rolling-average features
            - Two XGBoost models forecast DAQI band 6h and 12h into the future
            - Tiered alert system: Low → Moderate → Watch → Elevated Risk → Dangerous
            - SHAP values explain exactly which features drove the prediction
            - WHO guideline lines shown on the pollution trend chart
            - Every forecast is logged and verified against actual readings

            **Click "Forecast All Cities" to populate the entire map at once.**
            """)


# ─────────────────────────────────────────────────────────────
# TAB 2 — FORECAST ACCURACY TRACKER
# ─────────────────────────────────────────────────────────────

with tab_accuracy:
    st.subheader("📊 Forecast Accuracy Tracker")
    st.markdown(
        "Every forecast made by AirGuard is logged here. "
        "After the forecast horizon elapses, the actual DAQI is fetched from Open-Meteo "
        "and compared to the prediction automatically."
    )

    with st.spinner("Checking past forecasts against actuals..."):
        log = verify_past_forecasts()

    if log.empty:
        st.info("No forecasts logged yet. Run a forecast to start tracking accuracy.")
    else:
        verified = log[log['actual_class'].notna()].copy()
        pending  = log[log['actual_class'].isna()].copy()

        if not verified.empty:
            accuracy              = verified['correct'].mean() * 100
            n_verified            = len(verified)
            n_correct             = int(verified['correct'].sum())
            n_dangerous_predicted = int((verified['predicted_class'] == 2).sum())
            n_dangerous_caught    = int(
                ((verified['predicted_class'] == 2) &
                 (verified['actual_class']    == 2)).sum()
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy",    f"{accuracy:.1f}%")
            c2.metric("Forecasts Verified",  f"{n_correct}/{n_verified}")
            c3.metric("Dangerous Predicted", str(n_dangerous_predicted))
            c4.metric("Dangerous Caught",    str(n_dangerous_caught))

            st.markdown("---")
            st.markdown("**Verified Forecasts**")
            display = verified[[
                'predicted_at','city_name','horizon','forecast_for',
                'predicted_tier','predicted_label','actual_label','correct'
            ]].copy()
            display['predicted_at'] = pd.to_datetime(
                display['predicted_at']).dt.strftime('%d %b %H:%M')
            display['forecast_for'] = pd.to_datetime(
                display['forecast_for']).dt.strftime('%d %b %H:%M')
            display['correct'] = display['correct'].map({1: '✅', 0: '❌'})
            display.columns = [
                'Predicted At','City','Horizon','Forecast For',
                'Alert Tier','Model Band','Actual','Correct'
            ]
            st.dataframe(display, hide_index=True, width='stretch')

            if len(verified) >= 5:
                st.markdown("---")
                st.markdown("**Accuracy by City**")
                city_acc = (
                    verified.groupby('city_name')['correct']
                    .agg(['mean','count']).reset_index()
                )
                city_acc.columns = ['City','Accuracy','Forecasts']
                city_acc['Accuracy'] = (
                    city_acc['Accuracy']*100).round(1).astype(str) + '%'
                st.dataframe(city_acc, hide_index=True, width='stretch')

        else:
            st.info(
                f"**{len(pending)} forecast(s) pending verification.** "
                "Actuals are checked automatically after the forecast horizon elapses."
            )

        if not pending.empty:
            st.markdown("---")
            st.markdown("**Pending Verification** (forecast time not yet reached)")
            pend = pending[[
                'predicted_at','city_name','horizon',
                'forecast_for','predicted_tier','predicted_label'
            ]].copy()
            pend['predicted_at'] = pd.to_datetime(
                pend['predicted_at']).dt.strftime('%d %b %H:%M')
            pend['forecast_for'] = pd.to_datetime(
                pend['forecast_for']).dt.strftime('%d %b %H:%M')
            pend.columns = [
                'Predicted At','City','Horizon',
                'Forecast For','Alert Tier','Model Band'
            ]
            st.dataframe(pend, hide_index=True, width='stretch')

        st.markdown("---")
        if st.button("🗑️ Clear forecast log"):
            if os.path.exists(FORECAST_LOG):
                os.remove(FORECAST_LOG)
            st.success("Forecast log cleared.")
            st.rerun()


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:13px'>"
    "AirGuard UK  |  Data: DEFRA AURN · Open-Meteo  |  "
    "Model: Dual XGBoost with 150 lag features  |  SHAP Explainability  |  "
    "SDG 13 : Climate Action  |  AI Tool Development Challenge 2026"
    "</div>",
    unsafe_allow_html=True,
)