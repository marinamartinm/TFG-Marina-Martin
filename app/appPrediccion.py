#=================================
#   1. CONFIGURACIÓN GENERAL
#=================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

import qrcode
import io
from fpdf import FPDF
from datetime import datetime
import base64
import shap
import matplotlib.pyplot as plt
import os

# --- Configuración de la página ---
st.set_page_config(
    page_title="Predicción de Trastornos del Sueño", 
    page_icon="logoApp.png",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("# 💤 Predicción de Trastornos del Sueño")
st.markdown("""
Bienvenido a esta herramienta interactiva desarrollada como parte del Trabajo de Fin de Grado en Ingeniería de la Salud.

Esta aplicación permite predecir la probabilidad de padecer **insomnio**, **apnea del sueño** o no presentar trastornos, a partir de información clínica, personal y de hábitos de vida.

Introduce tus datos a continuación y pulsa el botón para obtener el resultado.
""")
st.markdown("---")

#============================
#    2. CÓDIGO QR
#============================
def generar_qr(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill="black", back_color="white")
    img_bytes = io.BytesIO()
    qr_img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

app_url = "https://tfg_marina-martin.streamlit.app"
qr_img = generar_qr(app_url)

st.sidebar.subheader("📱 Acceso desde el móvil")
st.sidebar.image(qr_img, caption="Escanea el QR para acceder", use_container_width=True)
st.sidebar.write(f"O copia y pega este enlace: [Haz clic aquí]({app_url})")
st.sidebar.markdown("---")

st.title("📊 Análisis de Apnea del Sueño")

#=========================================
#     3. CARGA DEL MODELO
#=========================================
try:
    model_path = os.path.join(os.path.dirname(__file__), "random_forest_model.joblib")
    model = joblib.load(model_path)
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

if not hasattr(model, 'predict'):
    st.error("El modelo cargado no es válido.")
    st.stop()

#==================================
# 4. DEFINICIÓN DEL PDF
#==================================

class InformeClinico(FPDF):
    def header(self):
        if os.path.exists("logoApp.png"):
            self.image("logoApp.png", x=10, y=8, w=25)
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "INFORME CLÍNICO DE SUEÑO", ln=True, align="C")
        self.set_font("Arial", "", 11)
        self.cell(0, 10, f"Fecha: {datetime.now().strftime('%d/%m/%Y - %H:%M')}", ln=True, align="R")
        self.ln(10)

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, f" {title}", ln=True, fill=True)
        self.ln(2)

    def add_table(self, data: dict):
        self.set_font("Arial", "", 11)
        col_width = 50
        for key, value in data.items():
            self.cell(col_width, 8, key, border=1)
            self.cell(0, 8, str(value), border=1, ln=True)
        self.ln(3)

    def add_prediction(self, clase, probas, clases):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, f"Trastorno detectado: {clase}", ln=True)
        self.set_font("Arial", "", 11)
        for c, p in zip(clases, probas):
            self.cell(0, 8, f"{c}: {p*100:.1f}%", ln=True)
        self.ln(3)
        self.set_font("Arial", "I", 10)
        self.multi_cell(0, 6, "Nota: Este informe ha sido generado automáticamente como parte de una herramienta de ayuda para la predicción de trastornos del sueño. No sustituye el diagnóstico clínico profesional.")

def generar_informe_estetico(datos_usuario, prediccion_clase, probas, clases):
    pdf = InformeClinico()
    pdf.add_page()

    pdf.section_title("DATOS PERSONALES")
    sexo = "Masculino" if datos_usuario.get("Gender_Male", 0) == 1 else "Femenino"
    ocupaciones = [k.replace("Occupation_", "") for k in datos_usuario if k.startswith("Occupation_") and datos_usuario[k] == 1]
    ocupacion = ocupaciones[0] if ocupaciones else "No especificada"
    datos_personales = {
        "Edad": f"{datos_usuario.get('Age')} años",
        "Sexo": sexo,
        "Ocupación": ocupacion,
        "IMC (categoría)": ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"][int(datos_usuario.get("BMI Category", 1))],
        "Frecuencia cardíaca": f"{datos_usuario.get('Heart Rate')} bpm",
        "Presión arterial": f"{datos_usuario.get('Systolic BP')} / {datos_usuario.get('Diastolic BP')} mmHg"
    }
    pdf.add_table(datos_personales)

    pdf.section_title("HÁBITOS DE VIDA")
    datos_habitos = {
        "Duración del sueño": f"{datos_usuario.get('Sleep Duration')} h",
        "Calidad del sueño": f"{datos_usuario.get('Quality of Sleep')} / 10",
        "Nivel de estrés (escala 1-9)": datos_usuario.get("Stress Level"),
        "Pasos diarios": datos_usuario.get("Daily Steps")
    }
    pdf.add_table(datos_habitos)

    pdf.section_title("RESULTADOS DE LA PREDICCIÓN")
    pdf.add_prediction(prediccion_clase, probas, clases)

    ruta_pdf = "informe_prediccion_estetico.pdf"
    pdf.output(ruta_pdf)
    return ruta_pdf

#============================================
#      5. ENTRADA DE DATOS
#===========================================

st.header("📝 Datos del paciente")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Género", ["Female", "Male"])
        age = st.number_input("Edad", min_value=1, max_value=120, value=30)
        occupation = st.selectbox("Ocupación", [
            "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
            "Sales Representative", "Salesperson", "Scientist", 
            "Software Engineer", "Teacher"])
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
        altura = st.number_input("Altura (cm)", min_value=100.0, max_value=220.0, value=170.0)
        sleep_duration = st.slider("Duración del sueño (horas)", 0.0, 12.0, 7.0, 0.1)
        quality_of_sleep = st.slider("Calidad del sueño (1-10)", 1, 10, 7)
    with col2:
        deporte_dias = st.selectbox("¿Cuántos días a la semana haces deporte?", list(range(8)))
        stress_nivel = st.selectbox("¿Cómo calificarías tu nivel de estrés?", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"])
        blood_pressure = st.text_input("Presión arterial (ej: 120/80)", "120/80")
        heart_rate = st.number_input("Frecuencia cardíaca (bpm)", 30, 200, 72)
        daily_steps = st.number_input("Pasos diarios", 0, step=100, value=8000)
        polysomnography = st.selectbox("¿Datos de polisomnografía?", ["Sí", "No"], index=1)
    submitted = st.form_submit_button("🔄 Realizar predicción")

#===============================================
# 6. PROCESAMIENTO, PREDICCIÓN, SHAP + PDF
#===============================================
if submitted and model is not None:
    try:
        imc = peso / (altura / 100) ** 2
        if imc < 18.5:
            bmi_category = "Underweight"
        elif imc < 25:
            bmi_category = "Normal"
        elif imc < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        actividad_fisica_map = {i: pct for i, pct in enumerate([0, 20, 40, 70, 80, 90, 95, 100])}
        physical_activity = actividad_fisica_map.get(deporte_dias, 0)

        stress_map = {"Muy bajo": 1, "Bajo": 3, "Moderado": 5, "Alto": 7, "Muy alto": 9}
        stress_level = stress_map.get(stress_nivel, 5)

        try:
            systolic, diastolic = map(int, blood_pressure.split('/'))
        except:
            st.error("Presión arterial inválida. Usa el formato 120/80.")
            st.stop()

        bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}

        processed_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'BMI Category': bmi_mapping[bmi_category],
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic BP': systolic,
            'Diastolic BP': diastolic,
            'Gender_Male': 1 if gender == 'Male' else 0
        }

        for occ in [
            "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
            "Sales Representative", "Salesperson", "Scientist", 
            "Software Engineer", "Teacher"]:
            processed_data[f"Occupation_{occ}"] = 1.0 if occ == occupation else 0.0

        df = pd.DataFrame([processed_data])
        datos_usuario = processed_data

        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[model.feature_names_in_]

        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[0]
        prediccion_clase = prediction[0]

        st.subheader("Resultado de la predicción")
        colA, colB, colC = st.columns(3)
        clases = model.classes_
        for i, col in zip(range(len(clases)), [colA, colB, colC]):
            col.metric(clases[i], f"{prediction_proba[i]*100:.1f}%")

        ruta_pdf = generar_informe_estetico(datos_usuario, prediccion_clase, prediction_proba, clases)
        with open(ruta_pdf, "rb") as f:
            st.download_button("📄 Descargar informe PDF", f, file_name="informe_prediccion.pdf")

        # SHAP
        base_model = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(df)

        st.subheader("Importancia global de las variables")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df, plot_type="bar", show=False)
        st.pyplot(fig)

        st.subheader("🔎 Explicación individual de la predicción")
        instance = df.iloc[[0]]
        shap_values_instance = explainer.shap_values(instance)
        shap.force_plot(explainer.expected_value[0], shap_values_instance[0], instance, matplotlib=True, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
