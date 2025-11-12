# SR Ingenieria - Calculos Electricos PRO (Streamlit)
# Version 1 Base

import io, json, math, os
import numpy as np, pandas as pd, altair as alt, streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

st.set_page_config(page_title="SR Ingenieria - Calculos Electricos PRO", layout="wide")
BRAND = "SR Ingenieria - Calculos Electricos PRO v1.0"
st.title("⚡ " + BRAND)
st.caption("Aplicacion tecnica base. No reemplaza la ingenieria de detalle ni las normas aplicables.")

def load_licenses():
    try:
        with open("licenses.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"keys": {}, "trial_keys": ["SR-TRIAL-2025"]}

def check_license(email, key, licdb):
    if not email or not key:
        return False
    email = email.strip().lower()
    if key in licdb.get("trial_keys", []):
        return True
    if licdb.get("keys", {}).get(email) == key:
        return True
    return False

def license_gate():
    if "licensed" not in st.session_state:
        st.session_state.licensed = False
    if st.session_state.licensed:
        return True
    with st.sidebar:
        st.header("🔐 Activacion de licencia")
        email = st.text_input("Correo")
        key = st.text_input("Clave", type="password")
        if st.button("Activar"):
            licdb = load_licenses()
            if check_license(email, key, licdb):
                st.session_state.licensed = True
                st.success("Licencia activada correctamente.")
            else:
                st.error("Clave invalida. Use SR-TRIAL-2025 para modo demo.")
        st.caption("Clave de prueba: SR-TRIAL-2025")
    return st.session_state.licensed

if not license_gate():
    st.stop()

# Base de datos simplificada
ohm_km_data = [(1.5, 12.1, 19.5, 0.080), (2.5, 7.41, 12.2, 0.078)]
ohm_df = pd.DataFrame(ohm_km_data, columns=["mm2", "R_cu_ohm_km", "R_al_ohm_km", "X_ohm_km"])

with st.sidebar:
    st.header("⚙️ Parametros globales")
    company = st.text_input("Cliente", value="Cliente S.A.")
    engineer = st.text_input("Elaborado por", value="Sergio Rivera")
    st.session_state["company"] = company
    st.session_state["engineer"] = engineer

tabs = st.tabs(["Motor", "Linea", "Acerca de"])

with tabs[0]:
    st.subheader("Datos del motor")
    pot = st.number_input("Potencia (kW)", value=75.0)
    V = st.number_input("Tension (V_ll)", value=400.0)
    pf = st.number_input("FP", value=0.88)
    eff = st.number_input("Eficiencia", value=0.95)
    I = (pot * 1000) / (np.sqrt(3) * V * pf * eff)
    st.metric("Corriente nominal (A)", f"{I:.1f}")

with tabs[1]:
    st.subheader("Datos de linea")
    L = st.number_input("Longitud (m)", value=50.0)
    mat = st.selectbox("Material", ["Cu", "Al"], index=0)
    mm2 = st.selectbox("Seccion", list(ohm_df["mm2"]), index=1)
    st.write("Parametros listos para calculo.")

with tabs[2]:
    st.subheader("Acerca de")
    st.markdown("**SR Ingenieria - Calculos Electricos PRO v1.0**")
(c) 2025 SR Ingenieria. Todos los derechos reservados.")
