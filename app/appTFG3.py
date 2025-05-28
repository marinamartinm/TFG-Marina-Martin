import streamlit as st
import pandas as pd
import joblib
import numpy as np
import qrcode
import io
from fpdf import FPDF
from datetime import datetime
import os
import base64
import matplotlib.pyplot as plt
import shap

# ===========================================
# CONFIGURACIÓN GENERAL
# ===========================================

st.set_page_config(
    page_title="Predicción de Trastornos del Sueño",
    page_icon="logoApp.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("# 💤 Predicción de Trastornos del Sueño")
st.markdown("""
Introduce tus datos clínicos y de estilo de vida en el panel izquierdo.
Pulsa el botón para generar un informe detallado.
""")
st.markdown("---")

# ===== CÓDIGO QR EN LA BARRA LATERAL =====
def generar_qr(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

url_app = "https://tfg-marina-martin.streamlit.app/"
qr_bytes = generar_qr(url_app)
st.sidebar.subheader("📱 Acceso desde el móvil")
st.sidebar.image(qr_bytes, caption="Escanea el QR", use_container_width=True)
st.sidebar.markdown(f"[Haz clic aquí para acceder]({url_app})")
st.sidebar.markdown("---")

# ===== CARGA DEL MODELO Y SCALER =====
try:
    model = joblib.load("modelo_rf_seleccionado.pkl")
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

try:
    scaler, scaler_columns = joblib.load("scaler.pkl")
    st.success("✅ Scaler y columnas cargados correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el scaler: {e}")
    st.stop()

if not hasattr(model, 'predict'):
    st.error("❌ El modelo cargado no es válido.")
    st.stop()

# ===========================================
# CLASE Y FUNCIÓN PARA PDF
# ===========================================

class InformePDF(FPDF):
    def header(self):
        if os.path.exists("logoApp.png"):
            self.image("logoApp.png", x=8, y=6, w=20)
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "INFORME CLÍNICO - PREDICCIÓN DEL SUEÑO", ln=True, align="C")
        self.line(10, 20, 200, 20)
        self.ln(5)

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 11)

    def key_value(self, key, value):
        self.set_font("Arial", "", 11)
        self.cell(0, 8, f"{key}: {value}", ln=True)

def generar_informe_estetico(datos_usuario, prediccion_clase, probas, clases):
    pdf = InformePDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y - %H:%M')}", ln=True)
    pdf.ln(5)
    pdf.section_title("PACIENTE")
    pdf.key_value("Edad", f"{datos_usuario.get('Age')} años")
    sexo = "Masculino" if datos_usuario.get("Gender_Male", 0) == 1 else "Femenino"
    pdf.key_value("Sexo", sexo)
    ocupaciones = [k.replace("Occupation_", "") for k in datos_usuario if k.startswith("Occupation_") and datos_usuario[k] == 1]
    ocupacion = ocupaciones[0] if ocupaciones else "No especificada"
    pdf.key_value("Ocupación", ocupacion)
    pdf.key_value("IMC", datos_usuario.get("BMI Category"))
    pdf.key_value("Frecuencia cardíaca", f"{datos_usuario.get('Heart Rate')} bpm")
    pdf.key_value("Presión arterial", f"{datos_usuario.get('Systolic BP')} / {datos_usuario.get('Diastolic BP')} mmHg")
    pdf.ln(3)
    pdf.section_title("HÁBITOS")
    pdf.key_value("Sueño", f"{datos_usuario.get('Sleep Duration')} h")
    pdf.key_value("Calidad del sueño", f"{datos_usuario.get('Quality of Sleep')} / 10")
    pdf.key_value("Nivel de estrés", datos_usuario.get("Stress Level"))
    pdf.key_value("Pasos diarios", datos_usuario.get("Daily Steps"))
    pdf.ln(3)
    pdf.section_title("PREDICCIÓN")
    pdf.key_value("Trastorno detectado", prediccion_clase)
    for clase, p in zip(clases, probas):
        pdf.cell(0, 8, f"- {clase}: {p * 100:.1f}%", ln=True)
    pdf.output("informe_prediccion_estetico.pdf")
    return "informe_prediccion_estetico.pdf"

# ===========================================
# FORMULARIO PARA INTRODUCCIÓN DE DATOS
# ===========================================

with st.form("formulario"):
    st.header("📝 Datos del paciente")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Género", ["Female", "Male"])
        age = st.number_input("Edad", 1, 120, 30)
        occupation = st.selectbox("Ocupación", ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"])
        weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Altura (cm)", 100.0, 220.0, 170.0)
        sleep_duration = st.slider("Horas de sueño", 0.0, 12.0, 7.0, 0.1)
        quality_sleep = st.slider("Calidad del sueño (1-10)", 1, 10, 7)
    with col2:
        sport_days = st.selectbox("Días de deporte/semana", list(range(8)))
        stress_level = st.selectbox("Nivel de estrés", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"])
        bp = st.text_input("Presión arterial (ej: 120/80)", "120/80")
        heart_rate = st.number_input("Frecuencia cardíaca (bpm)", 30, 200, 72)
        steps = st.number_input("Pasos diarios", 0, 30000, 8000)
    submit = st.form_submit_button("🔄 Realizar predicción")

# ===========================================
# PREDICCIÓN Y RESULTADOS
# ===========================================

if submit:
    try:
        # ===== 1. Cálculo de variables
        imc = weight / (height / 100)**2
        imc_cat = "Underweight" if imc < 18.5 else "Normal" if imc < 25 else "Overweight" if imc < 30 else "Obese"
        deporte_pct = [0, 20, 40, 70, 80, 90, 95, 100][sport_days]
        stress_map = {"Muy bajo": 1, "Bajo": 3, "Moderado": 5, "Alto": 7, "Muy alto": 9}
        stress = stress_map[stress_level]
        sys, dia = map(int, bp.split("/"))
        bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}

        # ===== 2. Crear DataFrame de entrada
        data = {
            "Age": float(age),
            "Sleep Duration": float(sleep_duration),
            "Quality of Sleep": float(quality_sleep),
            "Physical Activity Level": float(deporte_pct),
            "Stress Level": float(stress),
            "BMI Category": float(bmi_map[imc_cat]),
            "Heart Rate": float(heart_rate),
            "Daily Steps": float(steps),
            "Systolic BP": float(sys),
            "Diastolic BP": float(dia),
            "Gender_Male": 1.0 if gender == "Male" else 0.0
        }

        for occ in ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"]:
            data[f"Occupation_{occ}"] = 1.0 if occupation == occ else 0.0

        df = pd.DataFrame([data])

        # ===== 3. Asegurar que df tiene TODAS las columnas que el modelo espera
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0.0

        # Ordenar columnas como espera el modelo
        df_final = df[model.feature_names_in_]

        # ===== 4. Predicción
        prediction = model.predict(df_final)[0]
        prediction_proba = model.predict_proba(df_final)[0]
        clases = model.classes_

        # ===== 5. Mostrar resultados
        st.header("📊 Resultado de la predicción")
        colA, colB, colC = st.columns(3)
        for i, col in enumerate([colA, colB, colC]):
            col.metric(clases[i], f"{prediction_proba[i] * 100:.1f}%", delta=None)

        # ===== 6. PDF
        pdf_path = generar_informe_estetico(df.iloc[0].to_dict(), prediction, prediction_proba, clases)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Descargar informe PDF", f, file_name="informe_sueno.pdf")

        # ===== 7. SHAP GLOBAL (corregido para Pipeline + Random Forest)

        st.subheader("📌 Importancia global de las variables")

        # Accedemos al modelo real (Random Forest) dentro del pipeline
        modelo_final = model.named_steps['clf']

        # Accedemos a qué columnas han sido seleccionadas
        variables_seleccionadas_idx = model.named_steps['select'].get_support(indices=True)
        df_final_seleccionado = df_final.iloc[:, variables_seleccionadas_idx]

        # TreeExplainer para Random Forest
        explainer = shap.TreeExplainer(modelo_final)
        shap_values = explainer.shap_values(df_final_seleccionado)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df_final_seleccionado, plot_type="bar", show=False)
        st.pyplot(fig)

        # ===== 8. SHAP INDIVIDUAL (corregido también)

        st.subheader("🔍 Explicación individual")
        instance = df_final_seleccionado.iloc[0:1]
        shap_values_instance = explainer.shap_values(instance)
        shap.force_plot(explainer.expected_value[0], shap_values_instance[0], instance, matplotlib=True, show=False)
        st.pyplot(plt.gcf())


    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
