import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

"""
Streamlit app — dwustopniowa predykcja czasu przejazdu
1⃣ Model bazowy (bez BPM) → wstępny czas
2⃣ Rozkład czasu na strefy BPM (w MINUTACH) + średnia prędkość wg intensywności
3⃣ Model rozszerzony (z BPM & speed) → wynik końcowy
Użytkownik podaje tylko dystans i intensywność.
"""

# -----------------------------------------------------------------------------
# Artefakty
# -----------------------------------------------------------------------------

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
SPEED_INTERCEPT = 10.9396933      # używane tylko w modelu bazowym
SPEED_SLOPE = 0.070385768         # używane tylko w modelu bazowym
ELEV_PER_KM = 2.07  # m/km
DEFAULT_WEATHER = {"Temperatura": 19.06, "Wilgotnosc": 59.4, "Predkosc wiatru": 3.55, "Cisnienie": 1002.19}

# Średnia prędkość (km/h) w zależności od intensywności — obliczona na podstawie datasetu
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

BASIC_FEATURES = [
    "Dystans", "Kcal (aktywnosc)", "Przewyzszenie (w metrach)",
    "Srednia szybkosc", "Srednie tetno",
    "Temperatura", "Wilgotnosc", "Predkosc wiatru", "Cisnienie",
    "month", "day_of_week",
]

BPM_COLS = ["Czas <135BPM", "Czas 136-149BPM", "Czas 150-163BPM", "Czas 164-177BPM", "Czas > 178BPM"]

FULL_FEATURES = BASIC_FEATURES[:5] + BPM_COLS + BASIC_FEATURES[5:]

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def hhmmss(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def derive_basic_features(dist_km: float, intensity: str, date: datetime) -> pd.DataFrame:
    hr = INTENSITY_MAP[intensity]["avg_hr"]
    kcal = (KCAL_INTERCEPT + KCAL_SLOPE * hr) * dist_km
    elev = ELEV_PER_KM * dist_km
    speed = SPEED_INTERCEPT + SPEED_SLOPE * hr  # tylko bazowy model
    month = date.month
    dow = date.weekday()

    data = {
        "Dystans": dist_km,
        "Kcal (aktywnosc)": kcal,
        "Przewyzszenie (w metrach)": elev,
        "Srednia szybkosc": speed,
        "Srednie tetno": hr,
        "Temperatura": DEFAULT_WEATHER["Temperatura"],
        "Wilgotnosc": DEFAULT_WEATHER["Wilgotnosc"],
        "Predkosc wiatru": DEFAULT_WEATHER["Predkosc wiatru"],
        "Cisnienie": DEFAULT_WEATHER["Cisnienie"],
        "month": month,
        "day_of_week": dow,
    }
    return pd.DataFrame([data])[BASIC_FEATURES]


def _zone_key_from_col(col: str) -> str:
    return col.replace("Czas", "").replace("BPM", "").strip().replace("  ", " ").replace(" ", "")


def derive_full_features(basic_df: pd.DataFrame, zone_minutes: dict, intensity: str) -> pd.DataFrame:
    full_df = basic_df.copy()
    # Nadpisujemy średnią prędkość odpowiednią dla intensywności
    full_df.loc[:, "Srednia szybkosc"] = SPEED_BY_INTENSITY[intensity]
    # BPM w minutach
    for col in BPM_COLS:
        full_df[col] = zone_minutes.get(_zone_key_from_col(col), 0.0)
    return full_df[FULL_FEATURES]


def predict_basic(df: pd.DataFrame) -> float:
    return float(MODEL_BASIC.predict(SCALER_BASIC.transform(df))[0])


def predict_final(df: pd.DataFrame) -> float:
    return float(MODEL_BPM.predict(SCALER_BPM.transform(df))[0])

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("🚴‍♂️ Predykcja czasu przejazdu — 2‑etapowy model (BPM w minutach)")

st.markdown("""
Podaj **dystans** oraz **natężenie treningu**. 
Aplikacja:
1. Szacuje czas (model bazowy, speed∼HR)
2. Rozkłada czas na strefy, **konwertuje je na minuty** i ustawia średnią prędkość
3. Prognozuje ostateczny czas (model z BPM)
""")

dist_km = st.number_input("Dystans [km]", min_value=1.0, value=30.0, step=1.0)
intensity = st.select_slider("Natężenie treningu", options=list(INTENSITY_MAP.keys()), value="lekki")
ride_date = st.date_input("Data treningu", value=datetime.today())

if st.button("🔮 Oblicz"):
    # --- model bazowy ---
    basic_df = derive_basic_features(dist_km, intensity, ride_date)
    prelim_sec = predict_basic(basic_df)

    # --- rozkład czasu na strefy ---
    pct = INTENSITY_MAP[intensity]["zones"]
    zone_sec = {z: pct[z] * prelim_sec for z in pct}
    zone_min = {z: s / 60.0 for z, s in zone_sec.items()}  # konwersja!

    # --- model z BPM + speed ---
    full_df = derive_full_features(basic_df, zone_min, intensity)
    final_sec = predict_final(full_df)

    # --- prezentacja ---
    st.success(f"⏱️ Szacowany czas jazdy: **{hhmmss(final_sec)}**")

    st.markdown("### Rozkład czasu w strefach tętna")
    zone_disp = pd.DataFrame({
        "Strefa": list(zone_sec.keys()),
        "Czas (HH:MM:SS)": [hhmmss(s) for s in zone_sec.values()],
        "Minuty": [round(m, 2) for m in zone_min.values()],
    })
    st.dataframe(zone_disp, hide_index=True)
    st.bar_chart(pd.Series(zone_min))

    with st.expander("🔍 Dane debug"):
        st.write("Cechy bazowe:")
        st.write(basic_df)
        st.write("Cechy pełne (BPM w minutach, prędkość nadpisana):")
        st.write(full_df)

st.caption("Kolumny BPM są przekazywane w minutach zgodnie z formatem Twojego modelu.")
