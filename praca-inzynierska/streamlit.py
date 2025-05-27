import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from pathlib import Path


@st.cache_resource(show_spinner=False)
def load_artifacts():
    models_dir = Path("data/06_models")
    model_basic = joblib.load(models_dir / "model.pkl")
    model_bpm = joblib.load(models_dir / "model_with_BPM.pkl")
    scaler_basic = joblib.load(models_dir / "scaler.pkl")
    scaler_bpm = joblib.load(models_dir / "scaler_with_BPM.pkl")
    return model_basic, model_bpm, scaler_basic, scaler_bpm

MODEL_BASIC, MODEL_BPM, SCALER_BASIC, SCALER_BPM = load_artifacts()

# -----------------------------------------------------------------------------
# Stałe wyestymowane z historii treningów
# -----------------------------------------------------------------------------
KCAL_INTERCEPT = 12.5880558
KCAL_SLOPE = 0.06824024
SPEED_INTERCEPT = 10.9396933
SPEED_SLOPE = 0.070385768
ELEV_PER_KM = 2.07
DEFAULT_WEATHER = {"Temperatura": 19.06, "Wilgotnosc": 59.4, "Predkosc wiatru": 3.55, "Cisnienie": 1002.19}
API_KEY = "88798f884c281555b0143f4a12c7ba29"

# Średnia prędkość (km/h) w zależności od intensywności
SPEED_BY_INTENSITY = {
    "lekki": 17.6,
    "umiarkowany": 19.5,
    "średni": 21.4,
    "ciężki": 22.2,
    "bardzo ciężki": 23.6,
}

INTENSITY_MAP = {
    "lekki": {"avg_hr": 120, "zones": {"<135": 1.00, "136-149": 0.00, "150-163": 0.00, "164-177": 0.00, ">178": 0.00}},
    "umiarkowany": {"avg_hr": 138, "zones": {"<135": 0.90, "136-149": 0.10, "150-163": 0.00, "164-177": 0.00, ">178": 0.00}},
    "średni": {"avg_hr": 147, "zones": {"<135": 0.70, "136-149": 0.25, "150-163": 0.05, "164-177": 0.00, ">178": 0.00}},
    "ciężki": {"avg_hr": 162, "zones": {"<135": 0.50, "136-149": 0.40, "150-163": 0.05, "164-177": 0.05, ">178": 0.00}},
    "bardzo ciężki": {"avg_hr": 175, "zones": {"<135": 0.20, "136-149": 0.50, "150-163": 0.15, "164-177": 0.10, ">178": 0.05}},
}

# Kompletna lista features zgodna z modelem
FEATURES = [
    'Dystans', 'Kcal (aktywnosc)', 'Przewyzszenie (w metrach)', 
    'Srednia szybkosc', 'Srednie tetno',
    'Czas <135BPM' ,'Czas 136-149BPM', 'Czas 150-163BPM', 'Czas 164-177BPM','Czas > 178BPM',
    'Temperatura', 'Wilgotnosc', 'Predkosc wiatru', 'Cisnienie', 'month', 'day_of_week', 'hour', 'kcal/dystans'
]

BASIC_FEATURES = [
    "Dystans", "Kcal (aktywnosc)", "Przewyzszenie (w metrach)",
    "Srednia szybkosc", "Srednie tetno",
    "Temperatura", "Wilgotnosc", "Predkosc wiatru", "Cisnienie",
    "month", "day_of_week", "hour", "kcal/dystans"
]

BPM_COLS = ["Czas <135BPM", "Czas 136-149BPM", "Czas 150-163BPM", "Czas 164-177BPM", "Czas > 178BPM"]

# -----------------------------------------------------------------------------
# Weather API Functions
# -----------------------------------------------------------------------------

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_weather_data(city_name: str):
    """Pobiera dane pogodowe z OpenWeatherMap API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city_name,
            "appid": API_KEY,
            "units": "metric"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather_data = {
            "Temperatura": data["main"]["temp"],
            "Wilgotnosc": data["main"]["humidity"],
            "Predkosc wiatru": data["wind"].get("speed", 0),  # pozostaw w m/s
            "Cisnienie": data["main"]["pressure"]
        }
        return weather_data, None
    except requests.exceptions.RequestException as e:
        return None, f"Błąd połączenia z API: {str(e)}"
    except KeyError as e:
        return None, f"Błąd w strukturze danych API: {str(e)}"
    except Exception as e:
        return None, f"Nieoczekiwany błąd: {str(e)}"

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def hhmmss(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def derive_basic_features(dist_km: float, intensity: str, date: datetime, weather_data: dict, hour: int) -> pd.DataFrame:
    hr = INTENSITY_MAP[intensity]["avg_hr"]
    kcal = (KCAL_INTERCEPT + KCAL_SLOPE * hr) * dist_km
    elev = ELEV_PER_KM * dist_km
    speed = SPEED_INTERCEPT + SPEED_SLOPE * hr
    month = date.month
    dow = date.weekday()
    kcal_per_km = kcal / dist_km if dist_km > 0 else 0

    data = {
        "Dystans": dist_km,
        "Kcal (aktywnosc)": kcal,
        "Przewyzszenie (w metrach)": elev,
        "Srednia szybkosc": speed,
        "Srednie tetno": hr,
        "Temperatura": weather_data["Temperatura"],
        "Wilgotnosc": weather_data["Wilgotnosc"],
        "Predkosc wiatru": weather_data["Predkosc wiatru"],
        "Cisnienie": weather_data["Cisnienie"],
        "month": month,
        "day_of_week": dow,
        "hour": hour,
        "kcal/dystans": kcal_per_km,
    }
    return pd.DataFrame([data])[BASIC_FEATURES]

def _zone_key_from_col(col: str) -> str:
    return col.replace("Czas", "").replace("BPM", "").strip().replace("  ", " ").replace(" ", "")

def derive_full_features(basic_df: pd.DataFrame, zone_minutes: dict, intensity: str, custom_speed: float = None) -> pd.DataFrame:
    full_df = basic_df.copy()
    # Używamy custom speed jeśli podana, inaczej domyślną dla intensywności
    speed = custom_speed if custom_speed is not None else SPEED_BY_INTENSITY[intensity]
    full_df.loc[:, "Srednia szybkosc"] = speed
    
    # Dodaj kolumny BPM
    for col in BPM_COLS:
        full_df[col] = zone_minutes.get(_zone_key_from_col(col), 0.0)
    
    # Przelicz kcal/dystans jeśli zmieniła się prędkość
    if 'kcal/dystans' in full_df.columns and 'Kcal (aktywnosc)' in full_df.columns and 'Dystans' in full_df.columns:
        full_df.loc[:, 'kcal/dystans'] = full_df['Kcal (aktywnosc)'] / full_df['Dystans']
    
    return full_df[FEATURES]

def predict_basic(df: pd.DataFrame) -> float:
    return float(MODEL_BASIC.predict(SCALER_BASIC.transform(df))[0])

def predict_final(df: pd.DataFrame) -> float:
    return float(MODEL_BPM.predict(SCALER_BPM.transform(df))[0])

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

# Initialize session state for weather data
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = DEFAULT_WEATHER.copy()
if 'last_city' not in st.session_state:
    st.session_state.last_city = ""

st.title("🚴‍♂️ Predykcja czasu przejazdu z danymi pogodowymi")

st.markdown("""
Aplikacja przewiduje czas treningu na podstawie:
1. Parametrów treningu (dystans, intensywność)
2. Aktualnych danych pogodowych (API lub własne)
3. Szczegółowych stref tętna (opcjonalnie)
""")

# Podstawowe parametry treningu
col1, col2 = st.columns(2)
with col1:
    dist_km = st.number_input("Dystans [km]", min_value=1.0, value=20.0, step=1.0)
    intensity = st.select_slider("Natężenie treningu", options=list(INTENSITY_MAP.keys()), value="lekki")

with col2:
    ride_date = st.date_input("Data treningu", value=datetime.today())
    hour = st.number_input("Godzina rozpoczęcia", min_value=0, max_value=23, value=12)

# Sekcja danych pogodowych
st.markdown("### 🌤️ Dane pogodowe")
weather_option = st.radio(
    "Wybierz źródło danych pogodowych:",
    ["Pobierz z API", "Wprowadź własne"]
)

if weather_option == "Pobierz z API":
    city = st.text_input("Nazwa miasta", value="Kraków", placeholder="np. Warszawa, Kraków, Gdańsk")
    
    # Auto-fetch weather when city changes
    if city != st.session_state.last_city and city.strip():
        with st.spinner("Pobieranie danych pogodowych..."):
            api_weather, error = get_weather_data(city)
            if api_weather:
                st.session_state.weather_data = api_weather
                st.session_state.last_city = city
                st.success(f"✅ Pobrano dane pogodowe dla miasta: {city}")
            else:
                st.error(f"❌ {error}")
                st.info("Używam domyślnych wartości pogodowych.")
                st.session_state.weather_data = DEFAULT_WEATHER.copy()
    
    # Manual refresh button
    if st.button("🔄 Odśwież dane pogodowe"):
        with st.spinner("Pobieranie danych pogodowych..."):
            api_weather, error = get_weather_data(city)
            if api_weather:
                st.session_state.weather_data = api_weather
                st.success(f"✅ Odświeżono dane pogodowe dla miasta: {city}")
            else:
                st.error(f"❌ {error}")
                st.info("Używam domyślnych wartości pogodowych.")
    
    # Display current weather data
    st.markdown("**Aktualne dane pogodowe:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperatura", f"{st.session_state.weather_data['Temperatura']:.1f}°C")
    with col2:
        st.metric("Wilgotność", f"{st.session_state.weather_data['Wilgotnosc']:.0f}%")
    with col3:
        st.metric("Wiatr", f"{st.session_state.weather_data['Predkosc wiatru']:.1f} m/s")
    with col4:
        st.metric("Ciśnienie", f"{st.session_state.weather_data['Cisnienie']:.0f} hPa")
    
    weather_data = st.session_state.weather_data
    
else:
    st.markdown("#### Wprowadź własne dane pogodowe:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temp = st.number_input("Temperatura [°C]", value=19.0, step=1.0)
    with col2:
        humidity = st.number_input("Wilgotność [%]", min_value=0, max_value=100, value=60)
    with col3:
        wind = st.number_input("Prędkość wiatru [m/s]", min_value=0.0, value=3.5, step=0.5)
    with col4:
        pressure = st.number_input("Ciśnienie [hPa]", min_value=950, max_value=1050, value=1013)
    
    weather_data = {
        "Temperatura": temp,
        "Wilgotnosc": humidity,
        "Predkosc wiatru": wind,
        "Cisnienie": pressure
    }

# Zaawansowane opcje treningu
st.markdown("### ⚙️ Zaawansowane opcje")
use_custom_training = st.checkbox("Dostosuj parametry intensywności")

custom_speed = None
custom_zones = None
custom_hr = None

if use_custom_training:
    st.markdown("#### Własne parametry treningu:")
    
    col1, col2 = st.columns(2)
    with col1:
        custom_speed = st.number_input("Średnia prędkość [km/h]", min_value=10.0, max_value=50.0, 
                                     value=SPEED_BY_INTENSITY[intensity], step=0.5)
        custom_hr = st.number_input("Średnie tętno [bpm]", min_value=80, max_value=200, 
                                  value=INTENSITY_MAP[intensity]["avg_hr"])
    
    with col2:
        st.markdown("**Rozkład czasu w strefach tętna (%):**")
        zone_135 = st.slider("< 135 BPM", 0.0, 1.0, INTENSITY_MAP[intensity]["zones"]["<135"], step=0.05)
        zone_136_149 = st.slider("136-149 BPM", 0.0, 1.0, INTENSITY_MAP[intensity]["zones"]["136-149"], step=0.05)
        zone_150_163 = st.slider("150-163 BPM", 0.0, 1.0, INTENSITY_MAP[intensity]["zones"]["150-163"], step=0.05)
        zone_164_177 = st.slider("164-177 BPM", 0.0, 1.0, INTENSITY_MAP[intensity]["zones"]["164-177"], step=0.05)
        zone_178 = st.slider("> 178 BPM", 0.0, 1.0, INTENSITY_MAP[intensity]["zones"][">178"], step=0.05)
        
        total_zones = zone_135 + zone_136_149 + zone_150_163 + zone_164_177 + zone_178
        if abs(total_zones - 1.0) > 0.01:
            st.warning(f"⚠️ Suma stref wynosi {total_zones:.2f}. Powinna wynosić 1.0")
        
        custom_zones = {
            "<135": zone_135,
            "136-149": zone_136_149,
            "150-163": zone_150_163,
            "164-177": zone_164_177,
            ">178": zone_178
        }

# Obliczenia
if st.button("🔮 Oblicz przewidywany czas"):
    # Model bazowy
    basic_df = derive_basic_features(dist_km, intensity, ride_date, weather_data, hour)
    
    # Jeśli użytkownik podał własne tętno, zaktualizuj
    if custom_hr is not None:
        basic_df.loc[:, "Srednie tetno"] = custom_hr
        # Przelicz kcal na podstawie nowego tętna
        new_kcal = (KCAL_INTERCEPT + KCAL_SLOPE * custom_hr) * dist_km
        basic_df.loc[:, "Kcal (aktywnosc)"] = new_kcal
        basic_df.loc[:, "kcal/dystans"] = new_kcal / dist_km
    
    prelim_sec = predict_basic(basic_df)

    # Rozkład na strefy
    zones_to_use = custom_zones if custom_zones is not None else INTENSITY_MAP[intensity]["zones"]
    zone_sec = {z: zones_to_use[z] * prelim_sec for z in zones_to_use}
    zone_min = {z: s / 60.0 for z, s in zone_sec.items()}

    # Model pełny
    full_df = derive_full_features(basic_df, zone_min, intensity, custom_speed)
    final_sec = predict_final(full_df)

    # Wyniki
    st.success(f"⏱️ **Przewidywany czas jazdy: {hhmmss(final_sec)}**")
    
    # Statystyki
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dystans", f"{dist_km} km")
        st.metric("Średnia prędkość", f"{full_df['Srednia szybkosc'].iloc[0]:.1f} km/h")
    with col2:
        st.metric("Spalanie kalorii", f"{full_df['Kcal (aktywnosc)'].iloc[0]:.0f} kcal")
        st.metric("Średnie tętno", f"{full_df['Srednie tetno'].iloc[0]:.0f} bpm")
    with col3:
        st.metric("Przewyższenie", f"{full_df['Przewyzszenie (w metrach)'].iloc[0]:.0f} m")
        st.metric("Temperatura", f"{weather_data['Temperatura']:.1f}°C")

    
    # Debug info
    with st.expander("🔍 Szczegóły techniczne"):
        st.write("**Dane pogodowe:**")
        st.json(weather_data)
        st.write("**Cechy modelu bazowego:**")
        st.dataframe(basic_df)
        st.write("**Wszystkie cechy (model pełny):**")
        st.dataframe(full_df)
