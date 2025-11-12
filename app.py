import streamlit as st
import math, json, os, io
import pandas as pd
import numpy as np
import altair as alt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

st.set_page_config(page_title="SR Ingeniería – Cálculos Eléctricos PRO", layout="wide")

def load_licenses():
    try:
        with open("licenses.json","r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"keys":{}, "trial_keys":[]}

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
    st.sidebar.header("🔐 Activación de licencia")
    email = st.sidebar.text_input("Correo", key="lic_email")
    key = st.sidebar.text_input("Clave", type="password", key="lic_key")
    if st.sidebar.button("Activar"):
        licdb = load_licenses()
        if check_license(email, key, licdb):
            st.session_state.licensed = True
            st.success("Licencia activada. ¡Bienvenido/a!")
        else:
            st.error("Licencia inválida. Use la clave de prueba: SR-TRIAL-2025")
    st.sidebar.caption("Clave de prueba: SR-TRIAL-2025")
    return st.session_state.licensed

def build_pdf_report(summary_dict):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    W, H = A4
    margin = 2*cm
    y = H - margin
    c.setTitle("Reporte – Cálculos Eléctricos PRO")

    def line(txt, size=11, dy=14, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(margin, y, txt)
        y -= dy

    line("SR Ingeniería – Cálculos Eléctricos PRO", size=16, dy=22, bold=True)
    line("Reporte técnico resumido (no reemplaza la ingeniería de detalle).", size=10, dy=16)
    c.line(margin, y, W-margin, y); y -= 12

    for section, items in summary_dict.items():
        line(f"▸ {section}", size=13, dy=16, bold=True)
        for k,v in items.items():
            txt = f"- {k}: {v}"
            c.setFont("Helvetica", 11)
            c.drawString(margin+10, y, txt[:110])
            y -= 14
            if y < margin+40:
                c.showPage()
                y = H - margin
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

BRAND = "SR Ingeniería – Cálculos Eléctricos PRO v1.0"
st.title("⚡ " + BRAND)
st.caption("Versión comercial con licencia. © 2025 SR Ingeniería.")

if not license_gate():
    st.info("Ingrese su licencia en la barra lateral para continuar.")
    st.stop()

ohm_km_data = [
    (1.5, 12.1, 19.5, 0.080),(2.5, 7.41, 12.2, 0.078),(4,4.61,7.61,0.076),(6,3.08,5.06,0.075),
    (10,1.83,3.08,0.073),(16,1.15,1.91,0.072),(25,0.727,1.20,0.071),(35,0.524,0.868,0.070),
    (50,0.387,0.641,0.069),(70,0.268,0.443,0.068),(95,0.193,0.320,0.067),(120,0.153,0.253,0.066),
    (150,0.124,0.206,0.065),(185,0.0991,0.164,0.064),(240,0.0754,0.125,0.063),(300,0.0601,0.100,0.062),
    (400,0.0470,0.078,0.061),(500,0.0366,0.062,0.060),(630,0.0283,0.048,0.059),
]
ohm_cols = ["mm2","R_cu_ohm_km","R_al_ohm_km","X_ohm_km"]
ampacity_base = [
    (1.5, 19),(2.5, 26),(4, 34),(6, 44),(10, 61),(16, 82),(25, 109),(35, 134),
    (50, 167),(70, 207),(95, 247),(120, 284),(150, 325),(185, 368),(240, 438),(300, 503),(400, 579),(500, 659),(630, 750)
]
amp_cols = ["mm2","I_base_A"]

ohm_df = pd.DataFrame(ohm_km_data, columns=ohm_cols)
amp_df = pd.DataFrame(ampacity_base, columns=amp_cols)

with st.sidebar:
    st.header("⚙️ Parámetros globales")
    company = st.text_input("Nombre cliente (para PDF)", value="Cliente S.A.")
    engineer = st.text_input("Elaborado por", value="Sergio Rivera")
    st.session_state["company"] = company
    st.session_state["engineer"] = engineer

tabs = st.tabs([
    "Potencia y corrientes",
    "Caída de tensión",
    "Dimensionamiento",
    "Cortocircuito",
    "Arranque motor",
    "Corrección FP",
    "Costeo energía",
    "Reportes",
    "Acerca de & Licencia"
])

with tabs[0]:
    st.subheader("Potencia trifásica y corrientes")
    col = st.columns(3)
    with col[0]:
        Vll = st.number_input("V_ll (V)", value=400.0, min_value=1.0)
        kW = st.number_input("Potencia (kW)", value=55.0, min_value=0.0)
        pf = st.number_input("Factor de potencia", value=0.88, min_value=0.0, max_value=1.0, step=0.01)
    with col[1]:
        eff = st.number_input("Eficiencia η", value=0.95, min_value=0.01, max_value=1.0, step=0.01)
        phases = st.selectbox("Sistema", ["3φ", "1φ"], index=0)
        Hz = st.selectbox("Frecuencia", [50,60], index=1)
    with col[2]:
        st.write(" ")
    if phases == "3φ":
        S_kVA = kW / max(eff,1e-6) / max(pf,1e-6)
        I_line = (S_kVA*1000) / (np.sqrt(3)*max(Vll,1e-6))
        Q_kvar = S_kVA*np.sqrt(max(0.0,1-pf**2))
    else:
        S_kVA = kW / max(eff,1e-6) / max(pf,1e-6)
        I_line = (S_kVA*1000) / max(Vll,1e-6)
        Q_kvar = S_kVA*np.sqrt(max(0.0,1-pf**2))
    st.metric("S (kVA)", f"{S_kVA:,.2f}")
    st.metric("I línea (A)", f"{I_line:,.1f}")
    st.metric("Q (kvar)", f"{Q_kvar:,.1f}")
    st.session_state["potencia"] = {"V_ll":Vll,"kW":kW,"pf":pf,"η":eff,"S_kVA":round(S_kVA,2),"I(A)":round(I_line,1),"Q(kvar)":round(Q_kvar,1)}

with tabs[1]:
    st.subheader("Caída de tensión")
    col = st.columns(4)
    with col[0]:
        phases_vd = st.selectbox("Sistema", ["3φ","1φ"], index=0, key="vd_sys")
        material = st.selectbox("Material conductor", ["Cu","Al"], index=0)
        mm2 = st.selectbox("Sección (mm²)", list(ohm_df["mm2"]), index=9)
    with col[1]:
        length = st.number_input("Longitud unidireccional (m)", value=100.0, min_value=1.0, step=1.0)
        I_load = st.number_input("Corriente de carga (A)", value=100.0, min_value=0.0)
        Vll_vd = st.number_input("V_ll (V)", value=400.0, min_value=1.0)
    with col[2]:
        pf_vd = st.number_input("Factor de potencia", value=0.9, min_value=0.0, max_value=1.0, step=0.01)
        temp = st.number_input("Temp conductor (°C)", value=70.0, min_value=20.0)
    with col[3]:
        st.write(" ")
    row = ohm_df[ohm_df["mm2"]==mm2].iloc[0]
    R = row["R_cu_ohm_km"] if material=="Cu" else row["R_al_ohm_km"]
    X = row["X_ohm_km"]
    alpha = 0.00393 if material=="Cu" else 0.00403
    R_T = R*(1+alpha*(temp-20))
    L_km = length/1000.0
    sinphi = np.sqrt(max(0.0,1-pf_vd**2))
    if phases_vd=="3φ":
        dV = np.sqrt(3)*I_load*(R_T*pf_vd + X*sinphi)*L_km
        pct = dV / max(Vll_vd,1e-6) * 100
    else:
        dV = 2*I_load*(R_T*pf_vd + X*sinphi)*L_km
        V_phase = Vll_vd/np.sqrt(3)
        pct = dV / max(V_phase,1e-6) * 100
    st.metric("ΔV (V)", f"{dV:,.2f}")
    st.metric("ΔV (%)", f"{pct:,.2f}%")
    st.session_state["caida"] = {"Sistema":phases_vd,"Material":material,"Sección":mm2,"Long(m)":length,"I(A)":I_load,"ΔV(V)":round(float(dV),2),"ΔV(%)":round(float(pct),2)}

with tabs[2]:
    st.subheader("Dimensionamiento de cable (simplificado)")
    ampacity_base = [
        (1.5, 19),(2.5, 26),(4, 34),(6, 44),(10, 61),(16, 82),(25, 109),(35, 134),
        (50, 167),(70, 207),(95, 247),(120, 284),(150, 325),(185, 368),(240, 438),(300, 503),(400, 579),(500, 659),(630, 750)
    ]
    amp_df = pd.DataFrame(ampacity_base, columns=["mm2","I_base_A"])
    col = st.columns(4)
    with col[0]:
        I_req = st.number_input("Ib (A)", value=160.0, min_value=1.0)
    with col[1]:
        temp_amb = st.number_input("Temp ambiente (°C)", value=40.0, min_value=0.0)
        n_cables = st.number_input("Circuitos agrupados", value=1, min_value=1, step=1)
    with col[2]:
        instal = st.selectbox("Instalación", ["Bandeja/Aire", "Tubería/Conducto"], index=0)
        material_dim = st.selectbox("Material", ["Cu","Al"], index=0, key="mat_dim")
    with col[3]:
        st.write(" ")
    def f_temp(t):
        if t <= 30: return 1.0
        elif t <= 35: return 0.96
        elif t <= 40: return 0.91
        elif t <= 45: return 0.87
        elif t <= 50: return 0.82
        else: return 0.75
    def f_group(n):
        if n <= 1: return 1.0
        elif n==2: return 0.85
        elif n==3: return 0.80
        elif n<=5: return 0.75
        else: return 0.70
    f_t = f_temp(temp_amb)
    f_g = f_group(int(n_cables))
    f_inst = 1.0 if instal=="Bandeja/Aire" else 0.9
    f_mat = 1.0 if material_dim=="Cu" else 0.8
    f_total = f_t * f_g * f_inst * f_mat
    candidate = None
    for _, row2 in amp_df.sort_values("mm2").iterrows():
        if row2["I_base_A"] * f_total >= I_req:
            candidate = row2
            break
    if candidate is None:
        st.error("No hay sección suficiente en la tabla.")
    else:
        st.success(f"Sección: {candidate['mm2']} mm² (I admisible ≈ {candidate['I_base_A']*f_total:.1f} A)")
        st.caption(f"Factores: f_t={f_t:.2f}, f_g={f_g:.2f}, f_inst={f_inst:.2f}, f_mat={f_mat:.2f} → f_total={f_total:.2f}")
        st.session_state["dimensionamiento"] = {"Ib(A)":I_req,"Sección(mm²)":candidate["mm2"],"I_adm(A)":round(candidate["I_base_A"]*f_total,1)}

with tabs[3]:
    st.subheader("Cortocircuito aproximado")
    col = st.columns(4)
    with col[0]:
        S_tr = st.number_input("Trafo (kVA)", value=1000.0, min_value=1.0)
        Vll_sc = st.number_input("V secundaria (V_ll)", value=400.0, min_value=1.0)
    with col[1]:
        Zpct = st.number_input("Z% trafo", value=6.0, min_value=1.0, max_value=20.0, step=0.1)
        length_sc = st.number_input("Long alimentador (m)", value=50.0, min_value=0.0)
    with col[2]:
        mm2_sc = st.selectbox("Sección alimentador (mm²)", list(ohm_df["mm2"]), index=9, key="mm2_sc")
        material_sc = st.selectbox("Material", ["Cu","Al"], index=0, key="mat_sc")
    with col[3]:
        st.write(" ")
    I_base = (S_tr*1000)/(np.sqrt(3)*max(Vll_sc,1e-6))
    Isc_tr = I_base*(100.0/max(Zpct,1e-6))
    row = ohm_df[ohm_df["mm2"]==mm2_sc].iloc[0]
    R = row["R_cu_ohm_km"] if material_sc=="Cu" else row["R_al_ohm_km"]
    X = row["X_ohm_km"]
    L_km = length_sc/1000.0
    Z_line = complex(R*L_km, X*L_km)
    V_phase = Vll_sc/np.sqrt(3)
    Z_tr_mag = V_phase/max(Isc_tr,1e-6)
    XR = 3.0
    R_tr = Z_tr_mag / np.sqrt(1+XR**2)
    X_tr = R_tr*XR
    Z_tr = complex(R_tr, X_tr)
    Z_total = Z_tr + Z_line
    Icc = V_phase/abs(Z_total)
    st.metric("Icc en barra (kA)", f"{Icc/1000:,.2f}")
    st.caption(f"Icc trafo sin línea ≈ {Isc_tr/1000:,.2f} kA")
    st.session_state["cortocircuito"] = {"Trafo(kVA)":S_tr,"V_ll(V)":Vll_sc,"Z%":Zpct,"Long(m)":length_sc,"Sección(mm²)":mm2_sc,"Icc(kA)":round(Icc/1000,2)}

with tabs[4]:
    st.subheader("Arranque de motor y ΔV")
    col = st.columns(4)
    with col[0]:
        kWm = st.number_input("Motor (kW)", value=90.0, min_value=0.0)
        Vllm = st.number_input("V_ll (V)", value=400.0, min_value=1.0)
        pfm = st.number_input("PF nominal", value=0.88, min_value=0.1, max_value=1.0, step=0.01)
    with col[1]:
        effm = st.number_input("Eficiencia", value=0.95, min_value=0.1, max_value=1.0, step=0.01)
        Istart_mult = st.number_input("Multiplicador I_arranque", value=6.0, min_value=1.0, step=0.5)
    with col[2]:
        feeder_len = st.number_input("Long alimentador (m)", value=80.0, min_value=0.0)
        feeder_mm2 = st.selectbox("Sección (mm²)", list(ohm_df["mm2"]), index=8, key="mm2_m")
        feeder_mat = st.selectbox("Material", ["Cu","Al"], index=0, key="mat_m")
    with col[3]:
        st.write(" ")
    S = kWm/max(effm*pfm,1e-6)
    In = (S*1000)/(np.sqrt(3)*max(Vllm,1e-6))
    Istart = Istart_mult*In
    row = ohm_df[ohm_df["mm2"]==feeder_mm2].iloc[0]
    R = row["R_cu_ohm_km"] if feeder_mat=="Cu" else row["R_al_ohm_km"]
    X = row["X_ohm_km"]
    L_km = feeder_len/1000.0
    sinphi = np.sqrt(max(0.0,1-pfm**2))
    dV = np.sqrt(3)*Istart*(R*pfm + X*sinphi)*L_km
    pct = dV/max(Vllm,1e-6)*100
    st.metric("I nominal (A)", f"{In:,.1f}")
    st.metric("I arranque (A)", f"{Istart:,.0f}")
    st.metric("ΔV arranque (%)", f"{pct:,.1f}%")
    st.session_state["arranque"] = {"kW":kWm,"V_ll":Vllm,"PF":pfm,"Istart_mult":Istart_mult,"I_nom(A)":round(In,1),"I_arr(A)":round(Istart,0),"ΔV(%)":round(pct,1)}

with tabs[5]:
    st.subheader("Corrección de factor de potencia")
    col = st.columns(3)
    with col[0]:
        kW_pf = st.number_input("kW", value=500.0, min_value=0.0)
        pf1 = st.number_input("PF actual", value=0.78, min_value=0.1, max_value=1.0, step=0.01)
    with col[1]:
        pf2 = st.number_input("PF objetivo", value=0.95, min_value=0.1, max_value=1.0, step=0.01)
    with col[2]:
        st.write(" ")
    phi1 = np.arccos(np.clip(pf1,0,1))
    phi2 = np.arccos(np.clip(pf2,0,1))
    kvar = kW_pf*(np.tan(phi1)-np.tan(phi2))
    st.metric("kvar requeridos", f"{kvar:,.1f} kvar")
    st.session_state["fp"] = {"kW":kW_pf,"PF1":pf1,"PF2":pf2,"kvar":round(kvar,1)}

with tabs[6]:
    st.subheader("Costeo básico de energía")
    col = st.columns(3)
    with col[0]:
        kwh_mes = st.number_input("kWh/mes", value=250000.0, min_value=0.0, step=1000.0)
    with col[1]:
        precio_kwh = st.number_input("Precio energía (USD/kWh)", value=0.12, min_value=0.0, step=0.01, format="%.3f")
        demanda_kw = st.number_input("Demanda (kW)", value=800.0, min_value=0.0)
    with col[2]:
        precio_kw = st.number_input("Precio demanda (USD/kW)", value=8.0, min_value=0.0, step=0.1)
    costo_energia = kwh_mes*precio_kwh
    costo_demanda = demanda_kw*precio_kw
    total = costo_energia + costo_demanda
    st.metric("Costo energía (USD)", f"{costo_energia:,.0f}")
    st.metric("Costo demanda (USD)", f"{costo_demanda:,.0f}")
    st.metric("Total mensual (USD)", f"{total:,.0f}")
    st.session_state["costo"] = {"kWh":kwh_mes,"USD/kWh":precio_kwh,"Demanda(kW)":demanda_kw,"USD/kW":precio_kw,"Total(USD)":round(total,0)}

with tabs[7]:
    st.subheader("Generar Reporte PDF")
    st.caption("Se compilan los datos calculados en cada módulo.")
    if st.button("📄 Generar PDF"):
        summary = {
            "Cliente": {"Razón social": st.session_state.get("company",""), "Ingeniero": st.session_state.get("engineer","")},
            "Potencia": st.session_state.get("potencia",{}),
            "Caída de tensión": st.session_state.get("caida",{}),
            "Dimensionamiento": st.session_state.get("dimensionamiento",{}),
            "Cortocircuito": st.session_state.get("cortocircuito",{}),
            "Arranque": st.session_state.get("arranque",{}),
            "Corrección FP": st.session_state.get("fp",{}),
            "Costeo": st.session_state.get("costo",{}),
        }
        pdf = build_pdf_report(summary)
        st.download_button("Descargar Reporte.pdf", data=pdf, file_name="Reporte_Calculos_Electricos_PRO.pdf", mime="application/pdf")

with tabs[8]:
    st.subheader("Acerca de & Licencia")
    st.markdown("""  """)
# --- 9) Acerca de & Licencia
with tabs[8]:
    st.subheader("Acerca de & Licencia")
    st.markdown("""
**SR Ingeniería - Cálculos Eléctricos PRO v1.0**  
© 2025 SR Ingeniería. Todos los derechos reservados.

Esta herramienta implementa modelos **aproximados** para pre-dimensionamiento y verificación rápida.  
No reemplaza la ingeniería de detalle ni el cumplimiento normativo aplicable (IEC/IEEE/NTC/NFPA).

**Licenciamiento**  
- Uso por usuario/empresa según clave de activación.  
- Distribución no autorizada prohibida.  
- Librerías open-source: Streamlit, Pandas, Numpy, Altair, ReportLab.

**Soporte y personalización**  
Integración de normas IEC 60364/60909, IEEE 141, límites ΔV por tipo de circuito, tablas de ampacidad completas, reportes corporativos (PDF), y más.
""")
 
# --- 9) Acerca de & Licencia
with tabs[8]:
    st.subheader("Acerca de & Licencia")
   # --- 9) Acerca de & Licencia
with tabs[8]:
    st.subheader("Acerca de & Licencia")
    st.markdown(
        """
**SR Ingenieria - Calculos Electricos PRO v1.0**  
(c) 2025 SR Ingenieria. Todos los derechos reservados.

Esta herramienta implementa modelos aproximados para pre-dimensionamiento y verificacion rapida.  
No reemplaza la ingenieria de detalle ni el cumplimiento normativo aplicable (IEC/IEEE/NTC/NFPA).

**Licenciamiento**  
- Uso por usuario o empresa segun clave de activacion.  
- Distribucion no autorizada prohibida.  
- Librerias open-source: Streamlit, Pandas, Numpy, Altair, ReportLab.

**Soporte y personalizacion**  
Integracion de normas IEC 60364/60909, IEEE 141, limites de caida de tension, tablas de ampacidad, reportes PDF y mas.
        """
    )

Modelos **aproximados** para pre-dimensionamiento y verificación rápida.  
No reemplaza ingeniería de detalle ni cumplimiento normativo (IEC/IEEE/NTC/NFPA).

Licenciamiento: uso por usuario/empresa según clave de activación.  
Distribución no autorizada prohibida.  
Librerías open-source: Streamlit, Pandas, Numpy, Altair, ReportLab.
\"\"\")
