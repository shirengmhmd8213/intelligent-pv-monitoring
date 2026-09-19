# -*- coding: utf-8 -*-
"""
=========================================================================================
SOLAR PHOTOVOLTAIC POWER MONITORING SYSTEM
Professional Scientific Telemetry & Solar Radiation Instrumentation Dashboard
+ AI Fault Detection & 3-Day Power Forecast (Open-Meteo Live, Wh Display)
=========================================================================================
"""

import streamlit as st
import plotly.graph_objects as go
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
import pytz
import jdatetime
import math
import os
import json
import html
import threading
import queue
import joblib
from collections import deque, Counter

# =========================================================================================
# 1. PAGE CONFIGURATION & TIMEZONE SETUP
# =========================================================================================
st.set_page_config(
    page_title="Solar PV Power Monitoring",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

tehran_tz = pytz.timezone('Asia/Tehran')
utc_tz = pytz.utc

if 'chart_expanded' not in st.session_state:
    st.session_state['chart_expanded'] = False

if 'chart_timeframe' not in st.session_state:
    st.session_state['chart_timeframe'] = "10 Min"

# =========================================================================================
# 2. AI MODEL LOADING
# =========================================================================================
@st.cache_resource
def load_ai_models():
    """لود مدل‌های هوش مصنوعی در startup"""
    models = {
        'fault_model': None, 'fault_features': None, 'fault_classes': None,
        'forecast_model': None, 'forecast_data': None,
        'loaded': False, 'error': None
    }
    try:
        models['fault_model'] = joblib.load('fault_detector.pkl')
        models['fault_features'] = joblib.load('fault_features.pkl')
        models['fault_classes'] = joblib.load('fault_classes.pkl')
        models['loaded'] = True
        print("✅ Fault detection model loaded successfully")
    except Exception as e:
        models['error'] = f"Fault model load error: {e}"
        print(f"❌ {models['error']}")

    try:
        models['forecast_model'] = joblib.load('pv_hourly_model.pkl')
        models['forecast_data'] = pd.read_excel('predict.xlsx')
        print("✅ Power forecast model loaded successfully")
    except Exception as e:
        print(f"⚠️ Forecast model load error: {e}")

    return models

AI_MODELS = load_ai_models()

FAULT_MESSAGES = {
    'PV_Healthy_dataset':      {'msg': '✅ The panel runs correctly.',                                  'color': '#10b981', 'level': 'success'},
    'PV_10_soiling_dataset':   {'msg': '⚠️ Mild soiling (10%) detected. Cleaning recommended.',         'color': '#f59e0b', 'level': 'warning'},
    'PV_20_soiling_dataset':   {'msg': '⚠️ Heavy soiling (20%) detected. Cleaning required.',           'color': '#f97316', 'level': 'warning'},
    'PV_DAMEGE_20_dataset':    {'msg': '🔶 Minor damage (20%) detected. Inspection advised.',           'color': '#f97316', 'level': 'warning'},
    'PV_DAMEGE_30_dataset':    {'msg': '🔴 Moderate damage (30%) detected. Maintenance needed.',        'color': '#ef4444', 'level': 'danger'},
    'PV_DAMEGE_40_dataset':    {'msg': '🔴 Severe damage (40%) detected. Immediate action required!',   'color': '#dc2626', 'level': 'danger'},
    'PV_open_circuit_dataset': {'msg': '🚨 Open circuit fault! Check wiring immediately.',              'color': '#991b1b', 'level': 'critical'},
    'PV_short_circuit_dataset':{'msg': '🚨 Short circuit fault! Emergency shutdown advised.',           'color': '#991b1b', 'level': 'critical'},
}

def build_fault_features(voltage, current, power, irradiance, temperature):
    """ساخت ۹ فیچر مدل از ۵ مقدار خام"""
    irr_safe = irradiance + 1e-6
    return {
        'Voltage_V': voltage, 'Current_A': current, 'Power': power,
        'Irradiance_Wm2': irradiance, 'Temperature_C': temperature,
        'Power_to_Irr': power / irr_safe, 'V_to_Irr': voltage / irr_safe,
        'I_to_Irr': current / irr_safe, 'Efficiency': power / irr_safe,
    }

if 'fault_vote_buffer' not in st.session_state:
    st.session_state['fault_vote_buffer'] = deque(maxlen=5)

# =========================================================================================
# 3. ICON SYSTEM
# =========================================================================================
def get_icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    icons = {
        'sun': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>',
        'sun-dim': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 4h.01"/><path d="M20 12h.01"/><path d="M12 20h.01"/><path d="M4 12h.01"/><path d="m17.657 6.343.01.01"/><path d="m17.657 17.657.01.01"/><path d="m6.343 17.657.01.01"/><path d="m6.343 6.343.01.01"/></svg>',
        'zap': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        'activity': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        'gauge': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
        'thermometer': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg>',
        'battery-charging': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 7h1a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-1"/><path d="M6 7H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h1"/><line x1="11" x2="13" y1="11" y2="13"/><line x1="13" x2="11" y1="13" y2="15"/></svg>',
        'sliders': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/></svg>',
        'wifi': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h.01"/><path d="M2 8.82a15 15 0 0 1 20 0"/><path d="M5 12.859a10 10 0 0 1 14 0"/><path d="M8.5 16.429a5 5 0 0 1 7 0"/></svg>',
        'shield-check': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>',
        'shield-alert': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>',
        'clock': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        'database': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
        'download': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>',
        'trending-up': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        'cloud': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>',
        'cloud-sun': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="M20 12h2"/><path d="m19.07 4.93-1.41 1.41"/><path d="M15.947 12.65a4 4 0 0 0-5.925-4.128"/><path d="M13 22H7a5 5 0 1 1 4.9-6H13a3 3 0 0 1 0 6Z"/></svg>',
        'cloud-rain': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 14v6"/><path d="M8 14v6"/><path d="M12 16v6"/></svg>',
        'compass': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
        'refresh-cw': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>',
        'brain': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/></svg>',
        'calendar': f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/></svg>',
    }
    return icons.get(name, f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="8"/></svg>')

# =========================================================================================
# 4. EPHEMERIS & OPEN-METEO WEATHER ENGINE
# =========================================================================================
WMO_CODES = {
    0: ("Clear", "sun"), 1: ("Mainly Clear", "sun"), 2: ("Partly Cloudy", "cloud-sun"),
    3: ("Overcast", "cloud"), 45: ("Foggy", "cloud"), 48: ("Rime Fog", "cloud"),
    51: ("Light Drizzle", "cloud-rain"), 53: ("Drizzle", "cloud-rain"),
    55: ("Dense Drizzle", "cloud-rain"), 61: ("Slight Rain", "cloud-rain"),
    63: ("Moderate Rain", "cloud-rain"), 65: ("Heavy Rain", "cloud-rain"),
    71: ("Slight Snow", "cloud"), 73: ("Moderate Snow", "cloud"), 75: ("Heavy Snow", "cloud"),
    80: ("Rain Showers", "cloud-rain"), 81: ("Moderate Showers", "cloud-rain"),
    82: ("Violent Showers", "cloud-rain"), 95: ("Thunderstorm", "zap"),
}

@st.cache_data(ttl=900)
def get_tehran_weather_data():
    url = "https://api.open-meteo.com/v1/forecast?latitude=35.6892&longitude=51.3890&current=temperature_2m,cloud_cover,weather_code&hourly=cloud_cover,weather_code&timezone=Asia%2FTehran"
    try:
        resp = requests.get(url, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            curr = data.get("current", {})
            hourly = data.get("hourly", {})
            wcode = curr.get("weather_code", 0)
            cloud_pct = curr.get("cloud_cover", 0)
            temp = curr.get("temperature_2m", None)
            sky_cond, icon_name = WMO_CODES.get(wcode, ("Overcast", "cloud"))
            schedule = []
            h_times = hourly.get("time", [])
            h_codes = hourly.get("weather_code", [])
            for t_str, c_code in zip(h_times, h_codes):
                c_lbl, _ = WMO_CODES.get(c_code, ("Overcast", "cloud"))
                schedule.append({'time_str': t_str, 'condition': c_lbl})
            return {'available': True, 'temp': temp,
                    'temp_str': f"{temp:.1f}°C" if temp is not None else "N/A",
                    'cloud_pct': cloud_pct, 'sky_condition': sky_cond,
                    'icon_name': icon_name, 'hourly_schedule': schedule}
    except Exception:
        pass
    return {'available': False, 'temp': None, 'temp_str': "N/A", 'cloud_pct': None,
            'sky_condition': None, 'icon_name': "sun", 'hourly_schedule': []}

def calculate_solar_elevation(lat=35.6892, lon=51.3890, dt=None):
    if dt is None:
        dt = datetime.now(tehran_tz)
    dt_utc = dt.astimezone(utc_tz)
    day_of_year = dt_utc.timetuple().tm_yday
    decimal_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    gamma = (2 * math.pi / 365.0) * (day_of_year - 1 + (decimal_hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    time_offset = eqtime + 4 * lon
    tst = (decimal_hour * 60 + time_offset) % 1440
    ha = (tst / 4 - 180) * math.pi / 180
    lat_rad = lat * math.pi / 180
    sin_elev = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

def get_solar_progress(dt=None):
    if dt is None:
        dt = datetime.now(tehran_tz)
    minutes = dt.hour * 60 + dt.minute
    sunrise_m = 6 * 60
    sunset_m = 18 * 60 + 30
    if minutes < sunrise_m or minutes > sunset_m:
        return None
    return (minutes - sunrise_m) / (sunset_m - sunrise_m)

def calculate_solar_variability(df_window: pd.DataFrame) -> dict:
    if df_window.empty or len(df_window) < 5 or 'irradiance_W_m2' not in df_window.columns:
        return {'label': 'Stable', 'color': '#10b981', 'cv': 0.0}
    irr_vals = df_window['irradiance_W_m2'].values
    mean_irr = float(irr_vals.mean())
    std_irr = float(irr_vals.std())
    if mean_irr < 15.0:
        return {'label': 'Stable (Low Light)', 'color': '#10b981', 'cv': 0.0}
    cv = std_irr / mean_irr
    if cv < 0.10:
        return {'label': 'Stable', 'color': '#10b981', 'cv': cv}
    elif cv < 0.25:
        return {'label': 'Moderate Variation', 'color': '#f59e0b', 'cv': cv}
    else:
        return {'label': 'High Variation', 'color': '#ef4444', 'cv': cv}

def build_weather_bands(t_min, t_max, hourly_schedule, default_condition):
    if not hourly_schedule:
        if default_condition:
            return pd.DataFrame([{'start': t_min, 'end': t_max, 'condition': default_condition}])
        return pd.DataFrame()
    blocks = []
    tz = t_min.tzinfo
    for entry in hourly_schedule:
        try:
            t_block_start = pd.to_datetime(entry['time_str'])
            if tz is not None:
                if t_block_start.tzinfo is None:
                    t_block_start = t_block_start.tz_localize(tz)
                else:
                    t_block_start = t_block_start.tz_convert(tz)
            t_block_end = t_block_start + timedelta(hours=1)
            overlap_start = max(t_min, t_block_start)
            overlap_end = min(t_max, t_block_end)
            if overlap_end > overlap_start:
                cond = entry.get('condition', default_condition)
                blocks.append({'start': overlap_start, 'end': overlap_end, 'condition': cond})
        except Exception:
            continue
    if not blocks:
        return pd.DataFrame()
    merged = []
    cur = blocks[0]
    for b in blocks[1:]:
        if b['condition'] == cur['condition'] and b['start'] <= cur['end']:
            cur['end'] = max(cur['end'], b['end'])
        else:
            merged.append(cur)
            cur = b
    merged.append(cur)
    return pd.DataFrame(merged)

# =========================================================================================
# 5. 3-DAY POWER FORECAST
# =========================================================================================
@st.cache_data(ttl=1800)
def fetch_forecast_weather_from_openmeteo():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=35.6892&longitude=51.3890"
        "&hourly=temperature_2m,shortwave_radiation"
        "&forecast_days=7"
        "&timezone=Asia%2FTehran"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            radiation = hourly.get("shortwave_radiation", [])
            if not times:
                return None
            df = pd.DataFrame({
                'datetime': pd.to_datetime(times),
                'temp': temps,
                'solarradiation': radiation
            })
            df = df.dropna().reset_index(drop=True)
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize(
                    'Asia/Tehran', ambiguous='NaT', nonexistent='shift_forward'
                )
            df = df.dropna(subset=['datetime']).reset_index(drop=True)
            return df
    except Exception as e:
        print(f"Open-Meteo forecast fetch error: {e}")
    return None


@st.cache_data(ttl=1800)
def compute_3day_forecast():
    try:
        model = AI_MODELS['forecast_model']
        if model is None:
            return None

        df_new = fetch_forecast_weather_from_openmeteo()
        data_source = "Open-Meteo Live"

        if df_new is None or df_new.empty:
            df_raw = AI_MODELS['forecast_data']
            if df_raw is None:
                return None
            df_new = df_raw.iloc[:, [1, 2, 17]].copy()
            df_new.columns = ['datetime', 'temp', 'solarradiation']
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            data_source = "predict.xlsx"

        df_new = df_new.sort_values('datetime').reset_index(drop=True)

        today = datetime.now(tehran_tz).date()
        df_future = df_new[df_new['datetime'].dt.date >= today].copy().reset_index(drop=True)
        if df_future.empty:
            df_use = df_new.head(72).copy().reset_index(drop=True)
        else:
            df_use = df_future.head(72).copy().reset_index(drop=True)

        if df_use.empty:
            return None

        night_mask = (df_use['datetime'].dt.hour < 6) | (df_use['datetime'].dt.hour >= 18)
        df_use.loc[night_mask, ['temp', 'solarradiation']] = 0
        df_use['temp'] = df_use['temp'] + (df_use['solarradiation'] * 0.0225)

        df_use['Hour'] = df_use['datetime'].dt.hour
        df_use['Hour_sin'] = np.sin(2 * np.pi * df_use['Hour'] / 24)
        df_use['Hour_cos'] = np.cos(2 * np.pi * df_use['Hour'] / 24)

        last_power = 0
        predictions = []
        for i in range(len(df_use)):
            input_data = pd.DataFrame({
                'Irradiance_Wm2': [df_use.loc[i, 'solarradiation']],
                'Temperature_C': [df_use.loc[i, 'temp']],
                'Hour_sin': [df_use.loc[i, 'Hour_sin']],
                'Hour_cos': [df_use.loc[i, 'Hour_cos']],
                'Power_lag1': [last_power]
            })
            pred = model.predict(input_data)[0]
            if night_mask.iloc[i]:
                pred = 0
            predictions.append(pred)
            last_power = pred

        df_use['predicted_power'] = predictions
        df_use['date'] = df_use['datetime'].dt.date

        daily = df_use.groupby('date').agg(
            total_wh=('predicted_power', 'sum'),
            peak_w=('predicted_power', 'max'),
        ).reset_index()
        daily['source'] = data_source

        return daily.head(3)
    except Exception as e:
        import traceback
        print(f"Forecast error: {e}")
        print(traceback.format_exc())
        return None

# =========================================================================================
# 6. ASTRONOMICAL THEME
# =========================================================================================
now_tehran = datetime.now(tehran_tz)
solar_elev = calculate_solar_elevation(dt=now_tehran)
sun_prog = get_solar_progress(dt=now_tehran)

if solar_elev > 0:
    bg_gradient = "linear-gradient(180deg, #f8fafc 0%, #f1f5f9 55%, #e2e8f0 100%)"
    if sun_prog is not None:
        sun_x = 10.0 + sun_prog * 80.0
        sun_y = 6.0 + (1.0 - math.sin(sun_prog * math.pi)) * 16.0
        glow_size = int(220 + math.sin(sun_prog * math.pi) * 160)
        sun_opacity = round(0.40 + math.sin(sun_prog * math.pi) * 0.35, 2)
        sun_color = f"rgba(251, 191, 36, {sun_opacity})"
        sun_orb_html = f"""
        <div id="solar-orb-wrapper" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: -1; overflow: hidden;">
            <div style="position: absolute; left: {sun_x:.1f}%; top: {sun_y:.1f}%; width: {glow_size}px; height: {glow_size}px; transform: translate(-50%, -50%); border-radius: 50%; background: radial-gradient(circle, {sun_color} 0%, rgba(253, 230, 138, 0.22) 45%, rgba(254, 243, 199, 0) 75%); filter: blur(30px); transition: all 1.5s ease;"></div>
        </div>
        """
    else:
        sun_orb_html = ""
elif solar_elev > -6:
    bg_gradient = "linear-gradient(180deg, #fff7ed 0%, #ffedd5 60%, #fed7aa 100%)"
    sun_orb_html = """
    <div id="solar-orb-wrapper" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: -1; overflow: hidden;">
        <div style="position: absolute; left: 88%; top: 22%; width: 250px; height: 250px; transform: translate(-50%, -50%); border-radius: 50%; background: radial-gradient(circle, rgba(249, 115, 22, 0.40) 0%, rgba(253, 186, 116, 0.18) 50%, rgba(254, 215, 170, 0) 75%); filter: blur(32px);"></div>
    </div>
    """
else:
    bg_gradient = "linear-gradient(180deg, #090d16 0%, #0f172a 50%, #1e293b 100%)"
    sun_orb_html = ""

# =========================================================================================
# 7. CSS STYLING
# =========================================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: {bg_gradient} !important;
        background-attachment: fixed !important;
        color: #0f172a !important;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; height: 0px !important; }}
    .block-container {{
        padding-top: 1rem !important; padding-bottom: 2rem !important;
        padding-left: 2rem !important; padding-right: 2rem !important;
        max-width: 1750px !important;
    }}
    .tabular-val {{ font-variant-numeric: tabular-nums !important; font-feature-settings: "tnum" 1 !important; }}
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(226, 232, 240, 0.85) !important;
        border-radius: 12px !important;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.04) !important;
        padding: 12px 14px !important;
        transition: box-shadow 0.2s ease !important;
    }}
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06) !important;
    }}
    .formal-header-bar {{
        display: flex; align-items: center; justify-content: space-between;
        background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(14px);
        border: 1px solid rgba(226, 232, 240, 0.85); border-radius: 12px;
        padding: 10px 18px; margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }}
    .header-left-group {{ display: flex; align-items: center; gap: 12px; }}
    .header-title-text {{ font-size: 20px; font-weight: 700; color: #0f172a; letter-spacing: -0.2px; }}
    .header-subtitle-text {{ font-size: 11px; font-weight: 500; color: #64748b; margin-top: 1px; }}
    .header-pills-group {{ display: flex; align-items: center; gap: 8px; }}
    .header-pill {{
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(248, 250, 252, 0.90); border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 8px; padding: 5px 10px; font-size: 11.5px; font-weight: 500; color: #334155;
    }}
    .hero-kpi-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(254, 243, 199, 0.35));
        border: 1px solid rgba(251, 191, 36, 0.45); border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 4px 14px rgba(245, 158, 11, 0.08); margin-bottom: 12px;
    }}
    .hero-kpi-title {{
        font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
        color: #b45309; display: flex; align-items: center; gap: 6px; margin-bottom: 6px;
    }}
    .hero-kpi-val {{ font-size: 40px; font-weight: 700; color: #0f172a; line-height: 1.1; font-variant-numeric: tabular-nums; }}
    .hero-kpi-unit {{ font-size: 18px; font-weight: 600; color: #d97706; margin-left: 4px; }}
    .metric-grid-2x3 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 12px; }}
    .metric-cell {{
        background: rgba(248, 250, 252, 0.90); border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 10px; padding: 10px 12px; transition: all 0.2s ease;
    }}
    .metric-cell:hover {{ background: #ffffff; border-color: #cbd5e1; }}
    .metric-cell-lbl {{
        font-size: 10.5px; font-weight: 600; color: #64748b; text-transform: uppercase;
        display: flex; align-items: center; gap: 5px; margin-bottom: 4px;
    }}
    .metric-cell-val {{ font-size: 21px; font-weight: 700; color: #0f172a; font-variant-numeric: tabular-nums; }}
    .metric-cell-unit {{ font-size: 11.5px; font-weight: 500; color: #64748b; margin-left: 3px; }}
    .dashboard-card {{
        background: rgba(255, 255, 255, 0.96); border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;
    }}
    .card-title {{
        font-size: 12px; font-weight: 600; color: #0f172a; text-transform: uppercase;
        letter-spacing: 0.4px; display: flex; align-items: center; gap: 7px;
        padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid rgba(226, 232, 240, 0.75);
    }}
    .status-row {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 4px 0; font-size: 11.5px; border-bottom: 1px solid rgba(241, 245, 249, 0.9);
    }}
    .status-row:last-child {{ border-bottom: none; }}
    .status-label {{ color: #475569; font-weight: 500; display: flex; align-items: center; gap: 6px; }}
    .status-val {{ font-weight: 600; color: #0f172a; font-variant-numeric: tabular-nums; }}
    .ai-alert-banner {{
        display: flex; align-items: center; gap: 14px;
        padding: 14px 20px; border-radius: 12px; margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        border-left: 6px solid;
    }}
    .ai-alert-icon {{
        display: flex; align-items: center; justify-content: center;
        width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
    }}
    .ai-alert-body {{ flex: 1; }}
    .ai-alert-title {{
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; opacity: 0.75; margin-bottom: 2px;
    }}
    .ai-alert-msg {{ font-size: 15.5px; font-weight: 700; line-height: 1.3; }}
    .ai-alert-meta {{ font-size: 11px; font-weight: 500; opacity: 0.7; margin-top: 3px; font-variant-numeric: tabular-nums; }}
    .forecast-card {{
        background: rgba(255, 255, 255, 0.96); border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;
    }}
    .forecast-row {{
        display: grid; grid-template-columns: auto 1fr auto;
        align-items: center; gap: 10px;
        padding: 8px 6px; border-radius: 8px; margin-bottom: 4px;
        background: rgba(248, 250, 252, 0.85);
        border: 1px solid rgba(226, 232, 240, 0.7);
    }}
    .forecast-day {{ font-size: 11.5px; font-weight: 600; color: #475569; }}
    .forecast-date {{ font-size: 11px; color: #94a3b8; }}
    .forecast-kwh {{ font-size: 17px; font-weight: 700; color: #0f172a; font-variant-numeric: tabular-nums; }}
    div[data-testid="stRadio"] > div {{
        display: flex !important; flex-direction: row !important; gap: 6px !important;
        background: rgba(255, 255, 255, 0.94) !important;
        border: 1px solid rgba(226, 232, 240, 0.85) !important;
        border-radius: 20px !important; padding: 3px 6px !important;
        width: fit-content !important; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 8px !important;
    }}
    div[data-testid="stRadio"] label {{
        background: transparent !important; border: none !important;
        border-radius: 14px !important; padding: 4px 12px !important;
        font-size: 11.5px !important; font-weight: 600 !important;
        color: #475569 !important; cursor: pointer !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stRadio"] label:hover {{ color: #0f172a !important; }}
    .chart-card-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 6px; margin-bottom: 6px;
        border-bottom: 1px solid rgba(226, 232, 240, 0.75);
    }}
    .chart-header-title {{ font-size: 14px; font-weight: 600; color: #0f172a; display: inline-flex; align-items: center; gap: 6px; }}
    .chart-header-subtitle {{ font-size: 11px; font-weight: 500; color: #64748b; margin-left: 6px; }}
    .chart-empty-state {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: rgba(248, 250, 252, 0.7); border: 1px dashed rgba(203, 213, 225, 0.85);
        border-radius: 8px; text-align: center;
    }}
    .chart-empty-state.main-empty {{ height: 280px; }}
    .chart-empty-state.sec-empty {{ height: 175px; }}
    .empty-state-title {{ font-size: 12px; font-weight: 500; color: #64748b; margin-top: 6px; }}
    .snapshot-bar {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 8px; margin-bottom: 14px; }}
    .snapshot-card {{
        background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 10px; padding: 9px 12px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
    }}
    .snapshot-lbl {{
        font-size: 10.5px; font-weight: 600; color: #64748b; text-transform: uppercase;
        display: flex; align-items: center; gap: 4px; margin-bottom: 2px;
    }}
    .snapshot-val {{ font-size: 17.5px; font-weight: 700; color: #0f172a; font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)

if sun_orb_html:
    st.markdown(sun_orb_html, unsafe_allow_html=True)

# =========================================================================================
# 8. DATA STORE & CSV PERSISTENCE
# =========================================================================================
CSV_BACKUP_FILE = 'solar_backup.csv'
PERSISTENT_LOG_INTERVAL_DAY_SEC = 300
PERSISTENT_LOG_INTERVAL_NIGHT_SEC = 3600

def is_daytime(dt=None, watts: float = 0.0, lux: float = 0.0) -> bool:
    try:
        elev = calculate_solar_elevation(dt=dt)
        if elev > 0.0:
            return True
    except Exception:
        pass
    return (watts > 5.0) or (lux > 10.0)

def save_point_to_csv(record_dict):
    df_new = pd.DataFrame([record_dict])
    if not os.path.exists(CSV_BACKUP_FILE):
        df_new.to_csv(CSV_BACKUP_FILE, index=False)
    else:
        df_new.to_csv(CSV_BACKUP_FILE, mode='a', header=False, index=False)

def load_data_from_csv():
    if os.path.exists(CSV_BACKUP_FILE):
        try:
            df = pd.read_csv(CSV_BACKUP_FILE)
            if not df.empty:
                if 'power_W' not in df.columns:
                    if 'power_mW' in df.columns:
                        df['power_W'] = df['power_mW'] / 1000.0
                    else:
                        df['power_W'] = (df['voltage_V'] * df['current_mA']) / 1000.0
                if 'energy_kWh' not in df.columns:
                    if 'energy_mWh' in df.columns:
                        df['energy_kWh'] = df['energy_mWh'] / 1_000_000.0
                    else:
                        df['energy_kWh'] = 0.0
                if 'temperature_C' in df.columns:
                    df = df[df['temperature_C'] < 100.0]
                return df.tail(1500).to_dict('records')
        except Exception:
            pass
    return []

def _telemetry_background_worker(state):
    hist_q = state['hist_queue']
    csv_q = state['csv_queue']
    lock = state['lock']
    while True:
        try:
            hist_item = hist_q.get(timeout=0.1)
            if hist_item:
                payload_str, now_iso = hist_item
                try:
                    p_json = json.loads(payload_str)
                    data_str = p_json.get("data", "")
                    if data_str:
                        parts = [p.strip() for p in data_str.split(",")]
                        if len(parts) >= 8:
                            seq = int(parts[0])
                            hist_ts_raw = parts[1]
                            hist_v = float(parts[2])
                            hist_i = float(parts[3])
                            hist_p_mw = float(parts[4])
                            hist_p_w = hist_p_mw / 1000.0
                            hist_temp = float(parts[5])
                            hist_lux = float(parts[6])
                            hist_irradiance = float(parts[7])
                            if hist_ts_raw.startswith("U"):
                                hist_iso = now_iso
                                hist_display = "Hist (Uptime)"
                            else:
                                clean_ts = hist_ts_raw.replace("T", " ")
                                hist_iso = clean_ts
                                try:
                                    hist_dt = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
                                    hist_display = f"Hist {hist_dt.strftime('%H:%M:%S')}"
                                except Exception:
                                    hist_display = hist_ts_raw[-8:]
                            hist_record = {
                                'sequence': seq, 'timestamp': hist_iso, 'time_display': hist_display,
                                'voltage_V': round(hist_v, 2), 'current_mA': round(hist_i, 2),
                                'power_mW': round(hist_p_mw, 2), 'power_W': round(hist_p_w, 4),
                                'energy_kWh': round(state['total_energy_kwh'], 6),
                                'temperature_C': round(hist_temp, 2),
                                'illuminance_lux': round(hist_lux, 1),
                                'irradiance_W_m2': round(hist_irradiance, 2),
                                'is_historical': True
                            }
                            with lock:
                                if hist_iso not in state['seen_timestamps']:
                                    state['seen_timestamps'].add(hist_iso)
                                    state['log_records'].append(hist_record)
                                    if len(state['log_records']) > 1500:
                                        popped = state['log_records'].pop(0)
                                        popped_ts = popped.get('timestamp')
                                        if popped_ts:
                                            state['seen_timestamps'].discard(str(popped_ts))
                                    t_now = datetime.now(tehran_tz).strftime("%H:%M:%S")
                                    state['events'].append({
                                        'time': t_now, 'level': 'info',
                                        'text': f"Recovered offline packet #{seq} ({hist_display})"
                                    })
                                    if len(state['events']) > 40:
                                        state['events'].pop(0)
                                    if state['logging_active']:
                                        try:
                                            csv_q.put_nowait(hist_record)
                                        except queue.Full:
                                            pass
                except Exception as e_hist:
                    with lock:
                        t_now = datetime.now(tehran_tz).strftime("%H:%M:%S")
                        state['events'].append({'time': t_now, 'level': 'warning', 'text': f"Hist parse error: {e_hist}"})
                        if len(state['events']) > 40:
                            state['events'].pop(0)
                hist_q.task_done()
        except queue.Empty:
            pass
        try:
            csv_rec = csv_q.get_nowait()
            if csv_rec:
                save_point_to_csv(csv_rec)
                csv_q.task_done()
        except queue.Empty:
            pass

@st.cache_resource
def get_sensor_data():
    initial_records = load_data_from_csv()
    initial_energy_kwh = 0.0
    initial_last_persisted = None
    if initial_records:
        last_rec = initial_records[-1]
        if 'energy_kWh' in last_rec:
            initial_energy_kwh = float(last_rec['energy_kWh'])
        try:
            if 'timestamp' in last_rec and last_rec['timestamp']:
                initial_last_persisted = datetime.strptime(str(last_rec['timestamp']), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tehran_tz)
        except Exception:
            initial_last_persisted = None
    initial_seen_ts = {str(r.get('timestamp')) for r in initial_records if r.get('timestamp')}
    persistent_lock = threading.RLock()
    hist_queue = queue.Queue(maxsize=3000)
    csv_queue = queue.Queue(maxsize=3000)
    store = {
        'lock': persistent_lock,
        'hist_queue': hist_queue, 'csv_queue': csv_queue,
        'voltage': 0.0, 'current': 0.0, 'power_w': 0.0, 'power_mw': 0.0,
        'watts': 0.0, 'lux': 0.0, 'temp': 0.0,
        'total_energy_kwh': initial_energy_kwh,
        'last_energy_calc_time': None, 'last_update_time': None,
        'last_persisted_time': initial_last_persisted,
        'mqtt_connected': False, 'logging_active': True,
        'reconnect_count': 0, 'msg_count': 0,
        'events': [], 'log_records': initial_records,
        'seen_timestamps': initial_seen_ts, 'current_cycle': None,
        'ai_fault_history': deque(maxlen=50),
    }
    worker_thread = threading.Thread(target=_telemetry_background_worker, args=(store,), daemon=True)
    worker_thread.start()
    return store

solar_data = get_sensor_data()
data_lock = solar_data['lock']

def add_event(level: str, text: str):
    t_now = datetime.now(tehran_tz).strftime("%H:%M:%S")
    with data_lock:
        solar_data['events'].append({'time': t_now, 'level': level, 'text': text})
        if len(solar_data['events']) > 40:
            solar_data['events'].pop(0)

# =========================================================================================
# 9. SIDEBAR
# =========================================================================================
with st.sidebar:
    st.markdown(f"### {get_icon('sliders', size=17, color='#0f172a')} Dashboard Controls")
    live_update = st.toggle("Live Telemetry Stream", value=True)
    st.markdown("---")
    st.markdown(f"### {get_icon('wifi', size=17, color='#0f172a')} Telemetry Broker")
    broker_ip = st.text_input("Primary MQTT Host", value="broker.emqx.io")
    broker_port = st.number_input("Port Number", value=1883, min_value=1, max_value=65535)
    fallback_broker = st.text_input("Fallback Broker Host", value="broker.hivemq.com")
    base_topic = st.text_input("Base Topic Tree", value="my_powerplant")
    st.markdown("---")
    st.markdown(f"### {get_icon('zap', size=17, color='#0f172a')} Hardware Parsing")
    incoming_power_unit = st.selectbox(
        "MQTT Power Source Unit",
        options=["Milliwatts (mW)", "Watts (W)"], index=0,
        help="Explicitly defines the hardware unit on the power topic."
    )
    st.markdown("---")
    if st.button("Reset Telemetry Session", use_container_width=True):
        with data_lock:
            solar_data['voltage'] = 0.0
            solar_data['current'] = 0.0
            solar_data['power_w'] = 0.0
            solar_data['power_mw'] = 0.0
            solar_data['watts'] = 0.0
            solar_data['lux'] = 0.0
            solar_data['temp'] = 0.0
            solar_data['total_energy_kwh'] = 0.0
            solar_data['last_energy_calc_time'] = None
            solar_data['last_update_time'] = None
            solar_data['log_records'].clear()
            solar_data['seen_timestamps'].clear()
            solar_data['current_cycle'] = None
            solar_data['msg_count'] = 0
            solar_data['ai_fault_history'].clear()
        st.session_state['fault_vote_buffer'].clear()
        add_event("info", "Telemetry session reset by operator.")

# =========================================================================================
# 10. MQTT ENGINE
# =========================================================================================
@st.cache_resource
def start_mqtt_client(broker: str, port: int, topic: str, fallback_host: str):
    client = mqtt.Client(client_id=f"SolarPV_AI_{int(time.time())}", clean_session=True)

    def on_connect(c, userdata, flags, rc):
        with data_lock:
            if rc == 0:
                solar_data['mqtt_connected'] = True
                c.subscribe(f"{topic}/#")
                add_event("success", f"Connected to broker: {broker}")
            else:
                solar_data['mqtt_connected'] = False
                add_event("warning", f"Broker refused rc={rc}")

    def on_disconnect(c, userdata, rc):
        with data_lock:
            solar_data['mqtt_connected'] = False
            solar_data['reconnect_count'] += 1
            add_event("warning", f"Disconnected (rc={rc}). Reconnect #{solar_data['reconnect_count']}")

    def on_message(c, userdata, msg):
        try:
            payload_str = msg.payload.decode('utf-8', errors='ignore').strip()
            topic_str = msg.topic.strip()
            now_dt = datetime.now(tehran_tz)
            now_iso = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            now_display = now_dt.strftime("%H:%M:%S")

            is_history_topic = topic_str.endswith('/history')
            has_data_envelope = payload_str.startswith('{"data":') or '"data"' in payload_str

            if is_history_topic or has_data_envelope:
                try:
                    solar_data['hist_queue'].put_nowait((payload_str, now_iso))
                except queue.Full:
                    pass
                return

            is_packet_complete = False
            completed_cycle = None

            with data_lock:
                if topic_str.endswith('/voltage'):
                    if solar_data['current_cycle'] is not None:
                        add_event("warning", "Incomplete cycle discarded")
                    solar_data['current_cycle'] = {
                        'start_dt': now_dt, 'start_iso': now_iso, 'start_display': now_display,
                        'voltage': float(payload_str), 'current': None, 'power_w': None,
                        'power_mw': None, 'temp': None, 'lux': None, 'watts': None
                    }
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/current'):
                    if solar_data['current_cycle'] is not None:
                        solar_data['current_cycle']['current'] = float(payload_str)
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/power_mw'):
                    raw_p = float(payload_str)
                    if solar_data['current_cycle'] is not None:
                        solar_data['current_cycle']['power_mw'] = raw_p
                        solar_data['current_cycle']['power_w'] = raw_p / 1000.0
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/power_w'):
                    raw_p = float(payload_str)
                    if solar_data['current_cycle'] is not None:
                        solar_data['current_cycle']['power_w'] = raw_p
                        solar_data['current_cycle']['power_mw'] = raw_p * 1000.0
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/power'):
                    raw_p = float(payload_str)
                    if solar_data['current_cycle'] is not None:
                        if incoming_power_unit == "Watts (W)":
                            solar_data['current_cycle']['power_w'] = raw_p
                            solar_data['current_cycle']['power_mw'] = raw_p * 1000.0
                        else:
                            solar_data['current_cycle']['power_mw'] = raw_p
                            solar_data['current_cycle']['power_w'] = raw_p / 1000.0
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/temperature'):
                    if solar_data['current_cycle'] is not None:
                        solar_data['current_cycle']['temp'] = float(payload_str)
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/lux'):
                    if solar_data['current_cycle'] is not None:
                        solar_data['current_cycle']['lux'] = float(payload_str)
                    solar_data['msg_count'] += 1
                    return
                elif topic_str.endswith('/watts'):
                    solar_data['msg_count'] += 1
                    if solar_data['current_cycle'] is not None:
                        elapsed_sec = (now_dt - solar_data['current_cycle']['start_dt']).total_seconds()
                        if 0 <= elapsed_sec <= 10.0:
                            solar_data['current_cycle']['watts'] = float(payload_str)
                            completed_cycle = solar_data['current_cycle']
                            solar_data['current_cycle'] = None
                            is_packet_complete = True
                        else:
                            solar_data['current_cycle'] = None
                            add_event("warning", "Cycle expired (timeout)")
                            return
                    else:
                        add_event("warning", "Stale /watts received")
                        return
                else:
                    try:
                        p_json = json.loads(payload_str)
                        if 'watts' in p_json:
                            raw_p_val = None
                            if 'power_W' in p_json:
                                raw_p_val = float(p_json['power_W'])
                            elif 'power_mW' in p_json:
                                raw_p_val = float(p_json['power_mW']) / 1000.0
                            elif 'power' in p_json:
                                p_in = float(p_json['power'])
                                raw_p_val = p_in if incoming_power_unit == "Watts (W)" else p_in / 1000.0
                            completed_cycle = {
                                'start_dt': now_dt, 'start_iso': now_iso, 'start_display': now_display,
                                'voltage': float(p_json['voltage']) if 'voltage' in p_json else None,
                                'current': float(p_json['current']) if 'current' in p_json else None,
                                'power_w': raw_p_val,
                                'power_mw': (raw_p_val * 1000.0) if raw_p_val is not None else None,
                                'temp': float(p_json['temperature']) if 'temperature' in p_json else None,
                                'lux': float(p_json['lux']) if 'lux' in p_json else None,
                                'watts': float(p_json['watts'])
                            }
                            is_packet_complete = True
                    except Exception:
                        pass

                if not is_packet_complete or completed_cycle is None:
                    return

                c_v = completed_cycle['voltage']
                c_i = completed_cycle['current']
                c_pw = completed_cycle['power_w']
                c_pmw = completed_cycle['power_mw']
                c_temp = completed_cycle['temp']
                c_lux = completed_cycle['lux']
                c_watts = completed_cycle['watts']

                if any(x is None for x in [c_v, c_i, c_pw, c_pmw, c_temp, c_lux, c_watts]):
                    add_event("warning", "Incomplete cycle discarded")
                    return

                solar_data['voltage'] = c_v
                solar_data['current'] = c_i
                solar_data['power_w'] = c_pw
                solar_data['power_mw'] = c_pmw
                solar_data['temp'] = c_temp
                solar_data['lux'] = c_lux
                solar_data['watts'] = c_watts

                cycle_dt = completed_cycle['start_dt']
                cycle_iso = completed_cycle['start_iso']
                cycle_display = completed_cycle['start_display']

                if solar_data['last_energy_calc_time'] is not None:
                    dt_sec = (cycle_dt - solar_data['last_energy_calc_time']).total_seconds()
                    if 0 < dt_sec < 180:
                        delta_kwh = (max(c_pw, 0.0) * (dt_sec / 3600.0)) / 1000.0
                        solar_data['total_energy_kwh'] += delta_kwh

                solar_data['last_energy_calc_time'] = cycle_dt
                solar_data['last_update_time'] = cycle_dt

                record = {
                    'timestamp': cycle_iso, 'time_display': cycle_display,
                    'voltage_V': round(c_v, 2), 'current_mA': round(c_i, 2),
                    'power_W': round(c_pw, 4), 'power_mW': round(c_pmw, 2),
                    'energy_kWh': round(solar_data['total_energy_kwh'], 6),
                    'temperature_C': round(c_temp, 2),
                    'illuminance_lux': round(c_lux, 1),
                    'irradiance_W_m2': round(c_watts, 2)
                }

                solar_data['log_records'].append(record)
                solar_data['seen_timestamps'].add(cycle_iso)
                if len(solar_data['log_records']) > 1500:
                    popped = solar_data['log_records'].pop(0)
                    popped_ts = popped.get('timestamp')
                    if popped_ts:
                        solar_data['seen_timestamps'].discard(str(popped_ts))

                # ====================================================
                # AI FAULT DETECTION (با دیباگ کامل)
                # ====================================================
                try:
                    print(f"🔍 AI Check: loaded={AI_MODELS['loaded']}, c_watts={c_watts:.1f}, V={c_v:.2f}, I={c_i:.2f}")

                    if AI_MODELS['loaded'] and c_watts > 5.0:
                        feats = build_fault_features(
                            voltage=c_v, current=c_i / 1000.0,
                            power=c_pw, irradiance=c_watts,
                            temperature=c_temp
                        )
                        print(f"📊 Features: {feats}")

                        X_row = pd.DataFrame([feats])[AI_MODELS['fault_features']]
                        print(f"📊 X_row shape: {X_row.shape}")

                        pred_class = AI_MODELS['fault_model'].predict(X_row)[0]
                        proba_arr = AI_MODELS['fault_model'].predict_proba(X_row)[0]
                        proba = proba_arr.max()

                        print(f"✅ Prediction: {pred_class} ({proba*100:.1f}%)")
                        print(f"📊 Buffer size before: {len(st.session_state['fault_vote_buffer'])}")

                        st.session_state['fault_vote_buffer'].append(pred_class)
                        votes = Counter(st.session_state['fault_vote_buffer'])
                        final_class, vote_count = votes.most_common(1)[0]

                        print(f"📊 Votes: {dict(votes)} → Final: {final_class}")

                        solar_data['ai_fault_history'].append({
                            'timestamp': cycle_iso,
                            'class': final_class,
                            'confidence': float(proba),
                            'votes': vote_count,
                            'total': len(st.session_state['fault_vote_buffer'])
                        })
                    else:
                        if not AI_MODELS['loaded']:
                            print(f"❌ AI_MODELS not loaded - skipping")
                        elif c_watts <= 5.0:
                            print(f"⏸️ AI paused: c_watts={c_watts:.1f} (need > 5.0)")
                except Exception as e_ai:
                    print(f"❌ AI Error: {e_ai}")
                    import traceback
                    print(traceback.format_exc())
                    add_event("warning", f"AI Error: {str(e_ai)[:100]}")

                if solar_data['logging_active']:
                    is_day = is_daytime(cycle_dt, c_watts, c_lux)
                    req_interval = PERSISTENT_LOG_INTERVAL_DAY_SEC if is_day else PERSISTENT_LOG_INTERVAL_NIGHT_SEC
                    last_persisted = solar_data.get('last_persisted_time')
                    should_persist = False
                    if last_persisted is None:
                        should_persist = True
                    else:
                        elapsed_sec = (cycle_dt - last_persisted).total_seconds()
                        if elapsed_sec >= req_interval:
                            should_persist = True
                    if should_persist:
                        solar_data['last_persisted_time'] = cycle_dt
                        try:
                            solar_data['csv_queue'].put_nowait(record)
                        except queue.Full:
                            pass

        except Exception as ex:
            add_event("warning", f"MQTT error: {str(ex)}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        client.connect_async(broker, port, keepalive=60)
        client.loop_start()
    except Exception as e:
        add_event("warning", f"Primary broker failed: {e}. Trying fallback...")
        try:
            client.connect_async(fallback_host, 1883, keepalive=60)
            client.loop_start()
        except Exception as ex:
            add_event("danger", f"Both brokers failed: {ex}")
    return client

mqtt_client = start_mqtt_client(broker_ip, int(broker_port), base_topic, fallback_broker)

# =========================================================================================
# 11. HEALTH STATUS
# =========================================================================================
now_cur = datetime.now(tehran_tz)
weather_info = get_tehran_weather_data()
tehran_temp = weather_info['temp_str']
sky_icon_name = weather_info['icon_name']

with data_lock:
    mqtt_conn = solar_data['mqtt_connected']
    last_upd = solar_data['last_update_time']
    cur_watts = solar_data['watts']
    cur_lux = solar_data['lux']
    ai_history_list = list(solar_data['ai_fault_history'])
    solar_data['ai_fault_history'] = deque(ai_history_list, maxlen=50)

if not mqtt_conn:
    health_status = "Disconnected"
    health_badge_color = "#ef4444"
    freshness_txt = "Offline"
elif last_upd is None:
    health_status = "Waiting for Telemetry"
    health_badge_color = "#f59e0b"
    freshness_txt = "Waiting for first packet..."
else:
    age_sec = (now_cur - last_upd).total_seconds()
    is_currently_day = is_daytime(now_cur, cur_watts, cur_lux)
    fresh_threshold = 30 if is_currently_day else 90
    if age_sec < fresh_threshold:
        health_status = "System Normal"
        health_badge_color = "#10b981"
        freshness_txt = f"{int(age_sec)}s ago"
    else:
        health_status = "Stale Telemetry"
        health_badge_color = "#f59e0b"
        freshness_txt = f"{int(age_sec)}s ago" if age_sec < 60 else f"{int(age_sec/60)}m ago"

mqtt_badge_txt = "Connected" if mqtt_conn else "Disconnected"

# =========================================================================================
# 12. HEADER
# =========================================================================================
jalali_now = jdatetime.datetime.now()
jalali_str = jalali_now.strftime("%Y/%m/%d")
time_str = now_cur.strftime("%H:%M:%S")

st.markdown(f"""
<div class="formal-header-bar">
    <div class="header-left-group">
        <span style="display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 9px; background: rgba(234, 88, 12, 0.1);">
            {get_icon('sun', size=22, color='#ea580c')}
        </span>
        <div>
            <div class="header-title-text">SOLAR PHOTOVOLTAIC POWER MONITORING</div>
            <div class="header-subtitle-text">AI-Powered Fault Detection & Telemetry Analytics</div>
        </div>
    </div>
    <div class="header-pills-group">
        <div class="header-pill">{get_icon(sky_icon_name, size=15, color='#0284c7')} <span>Tehran: <b>{tehran_temp}</b></span></div>
        <div class="header-pill">{get_icon('compass', size=15, color='#eab308')} <span>Solar Alt: <b>{solar_elev:.1f}°</b></span></div>
        <div class="header-pill">{get_icon('clock', size=15, color='#475569')} <span>{jalali_str} | {time_str}</span></div>
        <div class="header-pill"><span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {'#10b981' if mqtt_conn else '#ef4444'};"></span> <span>MQTT: <b>{mqtt_badge_txt}</b></span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================================================
# 13. AI FAULT ALERT BANNER
# =========================================================================================
current_fault_class = None
current_fault_conf = 0.0
current_fault_votes = "0/0"

if st.session_state['fault_vote_buffer']:
    votes = Counter(st.session_state['fault_vote_buffer'])
    current_fault_class, current_fault_votes_count = votes.most_common(1)[0]
    current_fault_votes = f"{current_fault_votes_count}/{len(st.session_state['fault_vote_buffer'])}"
    if ai_history_list:
        current_fault_conf = ai_history_list[-1].get('confidence', 0.0)

if current_fault_class is None:
    st.markdown(f"""
    <div class="ai-alert-banner" style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-left-color: #94a3b8;">
        <div class="ai-alert-icon" style="background: rgba(148, 163, 184, 0.15);">
            {get_icon('brain', size=22, color='#64748b')}
        </div>
        <div class="ai-alert-body">
            <div class="ai-alert-title" style="color: #64748b;">AI FAULT DETECTION</div>
            <div class="ai-alert-msg" style="color: #475569;">⏳ Waiting for sufficient solar radiation to start fault analysis...</div>
            <div class="ai-alert-meta" style="color: #94a3b8;">Analysis begins when irradiance &gt; 5 W/m²</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    info = FAULT_MESSAGES.get(current_fault_class, {'msg': f'Unknown: {current_fault_class}', 'color': '#64748b', 'level': 'info'})
    if info['level'] == 'success':
        bg = "linear-gradient(135deg, #ecfdf5, #d1fae5)"
        border_color = "#10b981"
        icon_bg = "rgba(16, 185, 129, 0.15)"
    elif info['level'] == 'warning':
        bg = "linear-gradient(135deg, #fffbeb, #fef3c7)"
        border_color = "#f59e0b"
        icon_bg = "rgba(245, 158, 11, 0.15)"
    elif info['level'] == 'danger':
        bg = "linear-gradient(135deg, #fef2f2, #fee2e2)"
        border_color = "#ef4444"
        icon_bg = "rgba(239, 68, 68, 0.15)"
    else:
        bg = "linear-gradient(135deg, #fef2f2, #fecaca)"
        border_color = "#991b1b"
        icon_bg = "rgba(153, 27, 27, 0.18)"
    icon_name = 'shield-check' if info['level'] == 'success' else 'shield-alert'
    st.markdown(f"""
    <div class="ai-alert-banner" style="background: {bg}; border-left-color: {border_color};">
        <div class="ai-alert-icon" style="background: {icon_bg};">
            {get_icon(icon_name, size=24, color=info['color'])}
        </div>
        <div class="ai-alert-body">
            <div class="ai-alert-title" style="color: {info['color']};">AI FAULT DETECTION · LIVE</div>
            <div class="ai-alert-msg" style="color: {border_color};">{info['msg']}</div>
            <div class="ai-alert-meta" style="color: {info['color']};">
                Confidence: {current_fault_conf*100:.1f}% · Consensus: {current_fault_votes} · Class: {current_fault_class}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================================================
# 14. MAIN DATA PROCESSING
# =========================================================================================
with data_lock:
    records_copy = list(solar_data['log_records'])
    current_power_w = solar_data['power_w']
    current_energy_kwh = solar_data['total_energy_kwh']
    current_voltage = solar_data['voltage']
    current_current = solar_data['current']
    current_temp = solar_data['temp']
    reconnect_val = solar_data['reconnect_count']
    msg_count_val = solar_data['msg_count']

df_raw = pd.DataFrame(records_copy)
if not df_raw.empty and 'timestamp' in df_raw.columns:
    df_raw['dt'] = pd.to_datetime(df_raw['timestamp'], errors='coerce', format='mixed')
    df_raw = df_raw.dropna(subset=['dt']).copy()
    if not df_raw.empty:
        if df_raw['dt'].dt.tz is None:
            df_raw['dt'] = df_raw['dt'].dt.tz_localize(tehran_tz, ambiguous='NaT', nonexistent='shift_forward')
        else:
            df_raw['dt'] = df_raw['dt'].dt.tz_convert(tehran_tz)
        df_raw = df_raw.dropna(subset=['dt'])
        df_raw = df_raw.sort_values('dt').drop_duplicates(subset=['dt']).reset_index(drop=True)

active_timeframe = st.session_state['chart_timeframe']
if df_raw.empty:
    df_plot = pd.DataFrame()
else:
    t_max = df_raw['dt'].max()
    if active_timeframe == "10 Min":
        cutoff = t_max - timedelta(minutes=10)
        df_plot = df_raw[df_raw['dt'] >= cutoff].copy()
    elif active_timeframe == "1 Hour":
        cutoff = t_max - timedelta(hours=1)
        df_plot = df_raw[df_raw['dt'] >= cutoff].copy()
    elif active_timeframe == "Today":
        today_start = t_max.replace(hour=0, minute=0, second=0, microsecond=0)
        df_plot = df_raw[df_raw['dt'] >= today_start].copy()
    else:
        df_plot = df_raw.copy()

variability_info = calculate_solar_variability(df_plot)
has_chart_data = not df_plot.empty and len(df_plot) >= 2

def format_power_w(val_w: float) -> str:
    if abs(val_w) < 10:
        return f"{val_w:.3f}"
    return f"{val_w:.2f}"

def format_energy_kwh(val_kwh: float) -> str:
    if val_kwh == 0.0:
        return "0.0000"
    elif abs(val_kwh) < 0.001:
        return f"{val_kwh:.6f}"
    elif abs(val_kwh) < 0.1:
        return f"{val_kwh:.5f}"
    return f"{val_kwh:.4f}"

power_trend_html = ""
if not df_raw.empty and 'dt' in df_raw.columns and len(df_raw) >= 4:
    t_latest = df_raw['dt'].max()
    t_10m_prior = t_latest - timedelta(minutes=10)
    df_prior_10m = df_raw[(df_raw['dt'] >= t_10m_prior) & (df_raw['dt'] < t_latest)]
    if len(df_prior_10m) >= 2:
        avg_p_10m = float(df_prior_10m['power_W'].mean())
        if avg_p_10m > 0.02:
            pct_change = ((current_power_w - avg_p_10m) / avg_p_10m) * 100.0
            if pct_change > 0.5:
                trend_sym, trend_color, sign = "↑", "#10b981", "+"
            elif pct_change < -0.5:
                trend_sym, trend_color, sign = "↓", "#ef4444", ""
            else:
                trend_sym, trend_color, sign = "→", "#64748b", ""
            power_trend_html = f'<div style="font-size: 11.5px; font-weight: 500; color: {trend_color}; margin-top: 4px;"><span style="font-weight: 700;">{trend_sym}</span> <span>{sign}{pct_change:.1f}% vs 10-min avg</span></div>'

scatter_cls = go.Scattergl if (has_chart_data and len(df_plot) > 3000) else go.Scatter

# =========================================================================================
# FULLSCREEN MODE
# =========================================================================================
if st.session_state['chart_expanded']:
    banner_c1, banner_c2 = st.columns([8, 2])
    with banner_c1:
        st.markdown(f'<div style="font-size:16.5px;font-weight:600;color:#0f172a;">{get_icon("sun", size=22, color="#ea580c")} FULLSCREEN TELEMETRY INSPECTION</div>', unsafe_allow_html=True)
    with banner_c2:
        if st.button("Exit Fullscreen", key="btn_exit_fs", type="primary", use_container_width=True):
            st.session_state['chart_expanded'] = False
            st.rerun()

    with st.container(border=True):
        timeframe_labels = ["10 Min", "1 Hour", "Today", "All Time"]
        new_tf = st.radio("Timeframe (Fullscreen)", options=timeframe_labels,
                          index=timeframe_labels.index(active_timeframe),
                          key="fs_chart_tf", horizontal=True, label_visibility="collapsed")
        if new_tf != active_timeframe:
            st.session_state['chart_timeframe'] = new_tf
            st.rerun()

        if has_chart_data:
            t_min, t_max = df_plot['dt'].min(), df_plot['dt'].max()
            time_fmt = '%H:%M:%S' if active_timeframe in ["10 Min", "1 Hour"] else '%H:%M'
            fig_fs = go.Figure()
            if weather_info['available'] and weather_info['hourly_schedule']:
                bands_df = build_weather_bands(t_min, t_max, weather_info['hourly_schedule'], weather_info['sky_condition'])
                for _, b_row in bands_df.iterrows():
                    c_name = str(b_row['condition']).lower()
                    if 'clear' in c_name:
                        band_fill = 'rgba(254, 240, 138, 0.13)'
                    elif 'partly' in c_name or 'mainly' in c_name:
                        band_fill = 'rgba(186, 230, 253, 0.13)'
                    elif 'rain' in c_name or 'drizzle' in c_name:
                        band_fill = 'rgba(191, 219, 254, 0.15)'
                    else:
                        band_fill = 'rgba(203, 213, 225, 0.15)'
                    delta_m = (b_row['end'] - b_row['start']).total_seconds() / 60.0
                    ann_text = str(b_row['condition']) if delta_m >= 15 else ""
                    fig_fs.add_vrect(x0=b_row['start'], x1=b_row['end'], fillcolor=band_fill, layer='below',
                                     line_width=0, annotation_text=ann_text, annotation_position="top left",
                                     annotation=dict(font_size=10, font_color="#94a3b8", font_family="Inter"))
            fig_fs.add_trace(scatter_cls(
                x=df_plot['dt'], y=df_plot['irradiance_W_m2'], mode='lines', name='Irradiance',
                line=dict(color='#f59e0b', width=2.5, shape='linear'),
                fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.05)',
                hovertemplate='Irradiance: <b>%{y:.1f} W/m²</b><extra></extra>'
            ))
            fig_fs.update_layout(
                height=650, margin=dict(l=45, r=20, t=30, b=30),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified', dragmode='zoom', uirevision=active_timeframe,
                xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                           tickformat=time_fmt, hoverformat='%H:%M:%S',
                           tickfont=dict(size=11, color='#64748b', family='Inter'),
                           showspikes=True, spikethickness=1, spikedash='dot', spikemode='across'),
                yaxis=dict(title=dict(text='Irradiance (W/m²)', font=dict(size=13, color='#ea580c', weight=600, family='Inter')),
                           showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                           tickfont=dict(size=11, color='#64748b', family='Inter'))
            )
            st.plotly_chart(fig_fs, use_container_width=True, config={'responsive': True, 'displaylogo': False})
        else:
            st.markdown('<div class="chart-empty-state" style="height: 500px;"><div class="empty-state-title">Waiting for telemetry...</div></div>', unsafe_allow_html=True)
    if live_update:
        time.sleep(3.5)
        st.rerun()
    st.stop()

# =========================================================================================
# MAIN 3-ZONE LAYOUT
# =========================================================================================
col_left, col_center, col_right = st.columns([3.2, 5.8, 3.0], gap="medium")

# -----------------------------------------------------------------------------------------
# ZONE 1 (LEFT)
# -----------------------------------------------------------------------------------------
with col_left:
    st.markdown(f"""
    <div class="hero-kpi-card">
        <div class="hero-kpi-title">{get_icon('zap', size=16, color='#d97706')} CURRENT GENERATED POWER</div>
        <div class="hero-kpi-val tabular-val">{format_power_w(current_power_w)}<span class="hero-kpi-unit">W</span></div>
        {power_trend_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid-2x3">
        <div class="metric-cell">
            <div class="metric-cell-lbl">{get_icon('gauge', size=13, color='#0284c7')} Voltage</div>
            <div class="metric-cell-val tabular-val">{current_voltage:.2f}<span class="metric-cell-unit">V</span></div>
        </div>
        <div class="metric-cell">
            <div class="metric-cell-lbl">{get_icon('activity', size=13, color='#06b6d4')} Current</div>
            <div class="metric-cell-val tabular-val">{current_current:.1f}<span class="metric-cell-unit">mA</span></div>
        </div>
        <div class="metric-cell">
            <div class="metric-cell-lbl">{get_icon('sun', size=13, color='#ea580c')} Irradiance</div>
            <div class="metric-cell-val tabular-val">{cur_watts:.1f}<span class="metric-cell-unit">W/m²</span></div>
        </div>
        <div class="metric-cell">
            <div class="metric-cell-lbl">{get_icon('sun-dim', size=13, color='#f59e0b')} Illuminance</div>
            <div class="metric-cell-val tabular-val">{cur_lux:,.0f}<span class="metric-cell-unit">Lux</span></div>
        </div>
        <div class="metric-cell">
            <div class="metric-cell-lbl">{get_icon('thermometer', size=13, color='#ef4444')} Temperature</div>
            <div class="metric-cell-val tabular-val">{current_temp:.1f}<span class="metric-cell-unit">°C</span></div>
        </div>
        <div class="metric-cell">
            <div class="metric-cell-lbl">{get_icon('battery-charging', size=13, color='#10b981')} Energy</div>
            <div class="metric-cell-val tabular-val">{format_energy_kwh(current_energy_kwh)}<span class="metric-cell-unit">kWh</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df_raw.empty and 'dt' in df_raw.columns:
        t_max_all = df_raw['dt'].max()
        today_midnight = t_max_all.replace(hour=0, minute=0, second=0, microsecond=0)
        df_today = df_raw[df_raw['dt'] >= today_midnight]
        peak_today_w = df_today['power_W'].max() if not df_today.empty else current_power_w
        e_today_kwh = (df_today['energy_kWh'].iloc[-1] - df_today['energy_kWh'].iloc[0]) if len(df_today) >= 2 else 0.0
    else:
        peak_today_w, e_today_kwh = current_power_w, 0.0

    if not mqtt_conn or last_upd is None:
        gen_state_label, gen_state_color = "Waiting", "#f59e0b"
    elif current_power_w > 0.02:
        gen_state_label, gen_state_color = "Producing", "#10b981"
    elif solar_elev <= -5:
        gen_state_label, gen_state_color = "Night Idle", "#6366f1"
    else:
        gen_state_label, gen_state_color = "Standby", "#0284c7"

    sky_status_txt = weather_info['sky_condition'] if weather_info['available'] and weather_info['sky_condition'] else "Unavailable"

    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">{get_icon('trending-up', size=16, color='#ea580c')} SOLAR PRODUCTION STATUS</div>
        <div class="status-row">
            <span class="status-label">{get_icon('zap', size=14, color='#ea580c')} Current Power</span>
            <span class="status-val tabular-val">{format_power_w(current_power_w)} W</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('activity', size=14, color='#0284c7')} Peak Today</span>
            <span class="status-val tabular-val">{format_power_w(peak_today_w)} W</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('battery-charging', size=14, color='#10b981')} Energy Today</span>
            <span class="status-val tabular-val">{format_energy_kwh(e_today_kwh)} kWh</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon(sky_icon_name, size=14, color='#0284c7')} Sky Condition</span>
            <span class="status-val">{sky_status_txt}</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('activity', size=14, color='#ea580c')} Variability</span>
            <span class="status-val" style="color: {variability_info['color']};">● {variability_info['label']}</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('shield-check', size=14, color='#0284c7')} Gen. State</span>
            <span class="status-val" style="color: {gen_state_color};">● {gen_state_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------------------
# ZONE 2 (CENTER)
# -----------------------------------------------------------------------------------------
with col_center:
    tf_col, btn_col = st.columns([7, 3])
    with tf_col:
        timeframe_labels = ["10 Min", "1 Hour", "Today", "All Time"]
        sel_tf = st.radio("Chart Window", options=timeframe_labels,
                          index=timeframe_labels.index(active_timeframe),
                          key="main_chart_tf", horizontal=True, label_visibility="collapsed")
        if sel_tf != active_timeframe:
            st.session_state['chart_timeframe'] = sel_tf
            st.rerun()
    with btn_col:
        if st.button("Expand Chart", key="btn_enter_fs", use_container_width=True):
            st.session_state['chart_expanded'] = True
            st.rerun()

    with st.container(border=True):
        st.markdown(f"""
        <div class="chart-card-header">
            <div>
                <span class="chart-header-title">{get_icon('sun', size=16, color='#ea580c')} SOLAR IRRADIANCE</span>
                <span class="chart-header-subtitle">Solar Radiation</span>
            </div>
            <span style="font-size: 11px; font-weight: 600; color: #ea580c;">● Current: {cur_watts:.1f} W/m²</span>
        </div>
        """, unsafe_allow_html=True)

        if has_chart_data:
            t_min, t_max = df_plot['dt'].min(), df_plot['dt'].max()
            time_fmt = '%H:%M:%S' if active_timeframe in ["10 Min", "1 Hour"] else '%H:%M'
            fig_main = go.Figure()
            if weather_info['available'] and weather_info['hourly_schedule']:
                bands_df = build_weather_bands(t_min, t_max, weather_info['hourly_schedule'], weather_info['sky_condition'])
                for _, b_row in bands_df.iterrows():
                    c_name = str(b_row['condition']).lower()
                    if 'clear' in c_name:
                        band_fill = 'rgba(254, 240, 138, 0.13)'
                    elif 'partly' in c_name or 'mainly' in c_name:
                        band_fill = 'rgba(186, 230, 253, 0.13)'
                    elif 'rain' in c_name or 'drizzle' in c_name:
                        band_fill = 'rgba(191, 219, 254, 0.15)'
                    else:
                        band_fill = 'rgba(203, 213, 225, 0.15)'
                    delta_m = (b_row['end'] - b_row['start']).total_seconds() / 60.0
                    ann_text = str(b_row['condition']) if delta_m >= 20 else ""
                    fig_main.add_vrect(x0=b_row['start'], x1=b_row['end'], fillcolor=band_fill, layer='below',
                                       line_width=0, annotation_text=ann_text, annotation_position="top left",
                                       annotation=dict(font_size=9, font_color="#94a3b8", font_family="Inter"))
            fig_main.add_trace(scatter_cls(
                x=df_plot['dt'], y=df_plot['irradiance_W_m2'], mode='lines', name='Irradiance',
                line=dict(color='#f59e0b', width=2.5, shape='linear'),
                fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.05)',
                hovertemplate='Irradiance: <b>%{y:.1f} W/m²</b><extra></extra>'
            ))
            fig_main.update_layout(
                height=280, margin=dict(l=40, r=20, t=30, b=30),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified', dragmode='zoom', uirevision=active_timeframe,
                xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                           tickformat=time_fmt, hoverformat='%H:%M:%S',
                           tickfont=dict(size=10, color='#64748b', family='Inter'),
                           showspikes=True, spikethickness=1, spikedash='dot', spikemode='across'),
                yaxis=dict(title=dict(text='Irradiance (W/m²)', font=dict(size=11, color='#ea580c', weight=600, family='Inter')),
                           showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                           tickfont=dict(size=10, color='#64748b', family='Inter')),
                showlegend=False
            )
            st.plotly_chart(fig_main, use_container_width=True, config={'responsive': True, 'displaylogo': False})
        else:
            st.markdown(f'<div class="chart-empty-state main-empty">{get_icon("sun", size=28, color="#94a3b8")}<div class="empty-state-title">Waiting for telemetry packets...</div></div>', unsafe_allow_html=True)

    def render_sparkline_chart(df, col_name, label, color_code, unit_str):
        if not df.empty and len(df) >= 2:
            fig_sub = go.Figure()
            fig_sub.add_trace(scatter_cls(
                x=df['dt'], y=df[col_name], mode='lines',
                line=dict(color=color_code, width=1.8, shape='linear'),
                hovertemplate=f'<b>%{{x|%H:%M:%S}}</b><br>{label}: <b>%{{y:.2f}} {unit_str}</b><extra></extra>'
            ))
            fig_sub.update_layout(
                height=175, margin=dict(l=38, r=14, t=18, b=22),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x', dragmode='zoom', uirevision=active_timeframe,
                xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False, nticks=4,
                           tickformat="%H:%M", tickfont=dict(size=9, color='#94a3b8', family='Inter')),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False,
                           tickfont=dict(size=9, color='#94a3b8', family='Inter')),
                showlegend=False
            )
            st.plotly_chart(fig_sub, use_container_width=True, config={'responsive': True, 'displaylogo': False})
        else:
            st.markdown('<div class="chart-empty-state sec-empty"><div class="empty-state-title">Waiting for data...</div></div>', unsafe_allow_html=True)

    r1_col1, r1_col2, r1_col3 = st.columns(3, gap="small")
    with r1_col1:
        with st.container(border=True):
            st.markdown(f'<div class="chart-card-header"><span class="chart-header-title">{get_icon("gauge", size=14, color="#0284c7")} Voltage (V)</span></div>', unsafe_allow_html=True)
            render_sparkline_chart(df_plot, 'voltage_V', 'Voltage', '#0284c7', 'V')
    with r1_col2:
        with st.container(border=True):
            st.markdown(f'<div class="chart-card-header"><span class="chart-header-title">{get_icon("activity", size=14, color="#06b6d4")} Current (mA)</span></div>', unsafe_allow_html=True)
            render_sparkline_chart(df_plot, 'current_mA', 'Current', '#06b6d4', 'mA')
    with r1_col3:
        with st.container(border=True):
            st.markdown(f'<div class="chart-card-header"><span class="chart-header-title">{get_icon("thermometer", size=14, color="#ef4444")} Temp (°C)</span></div>', unsafe_allow_html=True)
            render_sparkline_chart(df_plot, 'temperature_C', 'Temp', '#ef4444', '°C')

    r2_col1, r2_col2, r2_col3 = st.columns(3, gap="small")
    with r2_col1:
        with st.container(border=True):
            st.markdown(f'<div class="chart-card-header"><span class="chart-header-title">{get_icon("sun-dim", size=14, color="#eab308")} Lux</span></div>', unsafe_allow_html=True)
            render_sparkline_chart(df_plot, 'illuminance_lux', 'Lux', '#eab308', 'Lux')
    with r2_col2:
        with st.container(border=True):
            st.markdown(f'<div class="chart-card-header"><span class="chart-header-title">{get_icon("zap", size=14, color="#f97316")} Power (W)</span></div>', unsafe_allow_html=True)
            render_sparkline_chart(df_plot, 'power_W', 'Power', '#f97316', 'W')
    with r2_col3:
        with st.container(border=True):
            st.markdown(f'<div class="chart-card-header"><span class="chart-header-title">{get_icon("battery-charging", size=14, color="#10b981")} Energy (kWh)</span></div>', unsafe_allow_html=True)
            render_sparkline_chart(df_plot, 'energy_kWh', 'Energy', '#10b981', 'kWh')

# -----------------------------------------------------------------------------------------
# ZONE 3 (RIGHT)
# -----------------------------------------------------------------------------------------
with col_right:
    if current_fault_class:
        info = FAULT_MESSAGES.get(current_fault_class, {'msg': 'Unknown', 'color': '#64748b', 'level': 'info'})
        class_short = current_fault_class.replace('PV_', '').replace('_dataset', '')
        st.markdown(f"""
        <div class="dashboard-card" style="border-left: 4px solid {info['color']};">
            <div class="card-title">{get_icon('brain', size=16, color=info['color'])} AI FAULT STATUS</div>
            <div class="status-row">
                <span class="status-label">Detected Class</span>
                <span class="status-val" style="color: {info['color']};">{class_short}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Confidence</span>
                <span class="status-val tabular-val">{current_fault_conf*100:.1f}%</span>
            </div>
            <div class="status-row">
                <span class="status-label">Consensus Votes</span>
                <span class="status-val tabular-val">{current_fault_votes}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">{get_icon('brain', size=16, color='#64748b')} AI FAULT STATUS</div>
            <div style="font-size: 11.5px; color: #94a3b8; padding: 8px 0; text-align: center;">
                Waiting for telemetry...
            </div>
        </div>
        """, unsafe_allow_html=True)

    forecast_df = compute_3day_forecast()
    st.markdown(f"""
    <div class="forecast-card">
        <div class="card-title">{get_icon('calendar', size=16, color='#0284c7')} 3-DAY POWER FORECAST</div>
    """, unsafe_allow_html=True)

    if forecast_df is not None and not forecast_df.empty:
        if 'source' in forecast_df.columns:
            source_label = forecast_df['source'].iloc[0]
            source_color = "#10b981" if "Open-Meteo" in source_label else "#f59e0b"
            st.markdown(f'''
            <div style="font-size: 10px; color: {source_color}; margin-bottom: 6px; display: flex; align-items: center; gap: 4px;">
                <span>●</span> Source: <b>{source_label}</b>
            </div>
            ''', unsafe_allow_html=True)

        for _, row in forecast_df.iterrows():
            d = row['date']
            try:
                jd = jdatetime.date.fromgregorian(date=d)
                date_str = jd.strftime("%Y/%m/%d")
                weekday = jd.strftime("%A")
            except Exception:
                date_str = str(d)
                weekday = ""
            wh = float(row['total_wh'])
            if wh > 4000.0:
                status_icon, status_color = "☀️", "#10b981"
            elif wh > 2500.0:
                status_icon, status_color = "⛅", "#f59e0b"
            elif wh > 500.0:
                status_icon, status_color = "🌤️", "#f59e0b"
            else:
                status_icon, status_color = "☁️", "#64748b"
            st.markdown(f"""
            <div class="forecast-row">
                <div>
                    <div class="forecast-day">{weekday}</div>
                    <div class="forecast-date tabular-val">{date_str}</div>
                </div>
                <div style="text-align: center; font-size: 18px;">{status_icon}</div>
                <div class="forecast-kwh" style="color: {status_color};">{wh:.2f} <span style="font-size: 11px; color:#64748b;">Wh</span></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size: 11.5px; color: #94a3b8; padding: 8px 0; text-align: center;">Forecast data unavailable.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">{get_icon('shield-check', size=16, color='#0284c7')} SYSTEM STATUS</div>
        <div class="status-row">
            <span class="status-label">{get_icon('shield-check', size=14, color='#0284c7')} Health</span>
            <span class="status-val" style="color: {health_badge_color};">● {health_status}</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('wifi', size=14, color='#0284c7')} MQTT</span>
            <span class="status-val" style="color: {'#10b981' if mqtt_conn else '#ef4444'};">● {mqtt_badge_txt}</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('clock', size=14, color='#0284c7')} Last Packet</span>
            <span class="status-val tabular-val">{freshness_txt}</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('refresh-cw', size=14, color='#0284c7')} Reconnects</span>
            <span class="status-val tabular-val">{reconnect_val}</span>
        </div>
        <div class="status-row">
            <span class="status-label">{get_icon('activity', size=14, color='#0284c7')} Packets</span>
            <span class="status-val tabular-val">{msg_count_val}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    irr_cur = cur_watts
    gauge_pct = min(1.0, max(0.0, irr_cur / 1000.0))
    angle = -90 + (gauge_pct * 180)
    rad = math.radians(angle)
    nx = 100 + 65 * math.cos(rad)
    ny = 95 + 65 * math.sin(rad)
    tier_txt = "Strong" if irr_cur >= 600 else ("Moderate" if irr_cur >= 250 else "Low / Diffuse")
    tier_color = "#10b981" if irr_cur >= 600 else ("#f59e0b" if irr_cur >= 250 else "#64748b")
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">{get_icon('sun', size=16, color='#ea580c')} SOLAR INTENSITY</div>
        <div style="display: flex; flex-direction: column; align-items: center; padding: 4px 0;">
            <svg width="200" height="115" viewBox="0 0 200 120">
                <path d="M 25 95 A 75 75 0 0 1 175 95" fill="none" stroke="#e2e8f0" stroke-width="14" stroke-linecap="round"/>
                <path d="M 25 95 A 75 75 0 0 1 75 35" fill="none" stroke="#93c5fd" stroke-width="14" stroke-linecap="round"/>
                <path d="M 75 35 A 75 75 0 0 1 125 35" fill="none" stroke="#fcd34d" stroke-width="14"/>
                <path d="M 125 35 A 75 75 0 0 1 175 95" fill="none" stroke="#f97316" stroke-width="14" stroke-linecap="round"/>
                <circle cx="100" cy="95" r="7" fill="#0f172a"/>
                <line x1="100" y1="95" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#0f172a" stroke-width="3.5" stroke-linecap="round"/>
            </svg>
            <div style="font-size: 17px; font-weight: 700; color: #0f172a; margin-top: -8px;" class="tabular-val">{irr_cur:.1f} <span style="font-size: 11px; font-weight: 500; color: #64748b;">W/m²</span></div>
            <div style="font-size: 11px; font-weight: 600; color: {tier_color}; margin-top: 2px;">● {tier_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with data_lock:
        ev_copy = list(solar_data['events'])
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">{get_icon('activity', size=16, color='#0284c7')} RECENT EVENTS</div>
        <div style="max-height: 140px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px;">
    """, unsafe_allow_html=True)
    if ev_copy:
        for ev in reversed(ev_copy[-5:]):
            ev_color = '#10b981' if ev['level']=='success' else ('#f59e0b' if ev['level']=='warning' else '#0284c7')
            clean_txt = html.escape(str(ev['text']))
            st.markdown(f"""
            <div style="font-size: 11px; color: #334155; display: flex; align-items: center; justify-content: space-between; background: rgba(248, 250, 252, 0.9); padding: 5px 8px; border-radius: 6px;">
                <span style="display: flex; align-items: center; gap: 5px;"><span style="color: {ev_color};">●</span> {clean_txt}</span>
                <span style="color: #94a3b8; font-size: 10px;" class="tabular-val">{ev['time']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size: 11px; color: #94a3b8; text-align: center; padding: 12px 0;">No events.</div>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    if not df_raw.empty:
        csv_data = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"solar_telemetry_{now_cur.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================================================================================
# 15. BOTTOM: SNAPSHOT & AUDIT TABLE
# =========================================================================================
st.markdown("---")

if not df_plot.empty:
    peak_p = df_plot['power_W'].max()
    peak_v = df_plot['voltage_V'].max()
    peak_c = df_plot['current_mA'].max()
    avg_p = df_plot['power_W'].mean()
    avg_irr = df_plot['irradiance_W_m2'].mean()
else:
    peak_p, peak_v, peak_c, avg_p, avg_irr = 0.0, 0.0, 0.0, 0.0, 0.0

st.markdown(f"""
<div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
    {get_icon('trending-up', size=13, color='#64748b')} PERFORMANCE SNAPSHOT · {active_timeframe.upper()}
</div>
<div class="snapshot-bar">
    <div class="snapshot-card">
        <div class="snapshot-lbl">{get_icon('zap', size=12, color='#d97706')} Peak Power</div>
        <div class="snapshot-val tabular-val">{format_power_w(peak_p)} W</div>
    </div>
    <div class="snapshot-card">
        <div class="snapshot-lbl">{get_icon('gauge', size=12, color='#0284c7')} Peak Voltage</div>
        <div class="snapshot-val tabular-val">{peak_v:.2f} V</div>
    </div>
    <div class="snapshot-card">
        <div class="snapshot-lbl">{get_icon('activity', size=12, color='#06b6d4')} Peak Current</div>
        <div class="snapshot-val tabular-val">{peak_c:.1f} mA</div>
    </div>
    <div class="snapshot-card">
        <div class="snapshot-lbl">{get_icon('trending-up', size=12, color='#10b981')} Avg Power</div>
        <div class="snapshot-val tabular-val">{format_power_w(avg_p)} W</div>
    </div>
    <div class="snapshot-card">
        <div class="snapshot-lbl">{get_icon('sun', size=12, color='#ea580c')} Avg Irradiance</div>
        <div class="snapshot-val tabular-val">{avg_irr:.1f} W/m²</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    tbl_h1, tbl_h2 = st.columns([7, 3])
    with tbl_h1:
        st.markdown(f'<span style="font-size: 13.5px; font-weight: 600; color: #0f172a;">{get_icon("database", size=15, color="#0284c7")} TELEMETRY AUDIT LOG</span>', unsafe_allow_html=True)
    with tbl_h2:
        row_count = st.selectbox("Rows", options=[15, 30, 50, 100], index=0, label_visibility="collapsed")
    if not df_raw.empty:
        df_table = df_raw.copy()
        display_cols = ['time_display', 'voltage_V', 'current_mA', 'power_W', 'energy_kWh', 'temperature_C', 'illuminance_lux', 'irradiance_W_m2']
        avail_cols = [c for c in display_cols if c in df_table.columns]
        df_disp = df_table[avail_cols].tail(row_count).iloc[::-1].copy()
        rename_map = {
            'time_display': 'Time', 'voltage_V': 'Voltage (V)', 'current_mA': 'Current (mA)',
            'power_W': 'Power (W)', 'energy_kWh': 'Energy (kWh)', 'temperature_C': 'Temp (°C)',
            'illuminance_lux': 'Lux', 'irradiance_W_m2': 'Irradiance (W/m²)'
        }
        df_disp = df_disp.rename(columns=rename_map)
        st.dataframe(df_disp, use_container_width=True, height=210)
    else:
        st.info("Waiting for telemetry packets...")

# =========================================================================================
# 16. AUTO-RERUN
# =========================================================================================
if live_update:
    time.sleep(3.5)
    st.rerun()