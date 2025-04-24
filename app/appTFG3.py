#=================================
#   1. CONFIGURACIÓN GENERAL
#=================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import qrcode
import streamlit as st
import io
from fpdf import FPDF 
from datetime import datetime
import base64
import os
import matplotlib.pyplot as plt


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

Introduce tus datos en el panel lateral izquierdo y pulsa el botón para obtener el resultado.
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

# URL de la aplicación ACTUALIZAR CUANDO ESTE SUBIDA
app_url = "https://tfg-marina-martin.streamlit.app/"

# Generar imagen QR
qr_img = generar_qr(app_url)

# Mostrar en la barra lateral
st.sidebar.subheader("📱 Acceso desde el móvil")
st.sidebar.image(qr_img, caption="Escanea el QR para acceder", use_container_width=True)
st.sidebar.write(f"O copia y pega este enlace: [Haz clic aquí]({app_url})")
st.sidebar.markdown("---")


st.title("📊 Análisis de Apnea del Sueño")

#=========================================
#     3. CARGA DEL MODELO
#=========================================

try:
    model = joblib.load('modelo_random_forest_balanceado_calibrado.pkl')
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Verificación adicional (opcional, si lo necesitas)
if not hasattr(model, 'predict'):
    st.error("❌ El modelo cargado no es válido.")
    st.stop()


#============================================
#      4. ENTRADA DE DATOS
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
        deporte_dias = st.selectbox("¿Cuántos días a la semana haces deporte?", [0, 1, 2, 3, 4, 5, 6, 7])
        stress_nivel = st.selectbox("¿Cómo calificarías tu nivel de estrés?", [
            "Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"])
        blood_pressure = st.text_input("Presión arterial (ej: 120/80)", "120/80")
        heart_rate = st.number_input("Frecuencia cardíaca (bpm)", 30, 200, 72)
        daily_steps = st.number_input("Pasos diarios", 0, step=100, value=8000)
        #polysomnography = st.selectbox("¿Datos de polisomnografía?", ["Sí", "No"], index=1)

    submitted = st.form_submit_button("🔄 Realizar predicción")

#=============================================================
#      5. PROCESAMIENTO Y PREDICCIÓN
#=============================================================
if submitted and model is not None:
    try:
        # Calcular IMC
        imc = peso / (altura / 100) ** 2
        if imc < 18.5:
            bmi_category = "Underweight"
        elif imc < 25:
            bmi_category = "Normal"
        elif imc < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        # Mapeo de deporte a porcentaje
        actividad_fisica_map = {
            0: 0, 1: 20, 2: 40, 3: 70, 4: 80, 5: 90, 6: 95, 7: 100
        }
        physical_activity = actividad_fisica_map.get(deporte_dias, 0)

        # Mapeo de estrés a escala numérica
        stress_map = {
            "Muy bajo": 1, "Bajo": 3, "Moderado": 5,
            "Alto": 7, "Muy alto": 9
        }
        stress_level = stress_map.get(stress_nivel, 5)

        # Separar presión arterial
        systolic, diastolic = map(int, blood_pressure.split('/'))

        # Mapeos generales
        bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
        #polysomnography_mapping = {"Sí": 1, "No": 0}

        processed_data = {
            'Age': float(age),
            'Sleep Duration': float(sleep_duration),
            'Quality of Sleep': float(quality_of_sleep),
            'Physical Activity Level': float(physical_activity),
            'Stress Level': float(stress_level),
            'BMI Category': float(bmi_mapping[bmi_category]),
            'Heart Rate': float(heart_rate),
            'Daily Steps': float(daily_steps),
            'Systolic BP': float(systolic),
            'Diastolic BP': float(diastolic),
            'Gender_Male': float(1 if gender == 'Male' else 0),
        }

        for occ in ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
                    "Sales Representative", "Salesperson", "Scientist", 
                    "Software Engineer", "Teacher"]:
            processed_data[f'Occupation_{occ}'] = 1.0 if occupation == occ else 0.0

        df = pd.DataFrame([processed_data])

        # Adaptar columnas al modelo
        if hasattr(model, 'feature_names_in_'):
            expected_cols = list(model.feature_names_in_)
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected_cols]

        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[0]

        st.subheader("Resultado de la predicción")
        colA, colB, colC = st.columns(3)
        clases = model.classes_
        for i, col in zip(range(len(clases)), [colA, colB, colC]):
            clase = clases[i]
            porcentaje = prediction_proba[i] * 100
            color = "#ff4b4b" if clase == prediction[0] else "#1f77b4"
            col.metric(clase, f"{porcentaje:.1f}%")

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")


import shap
import matplotlib.pyplot as plt

# =============================
#    6. EXPLICABILIDAD SHAP
# =============================
base_model = model.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(df)

st.subheader("📌 Importancia global de las variables")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, df, plot_type="bar", show=False)
st.pyplot(fig)

st.subheader("🔎 Explicación individual de la predicción")
if len(df) > 1:
    index_to_explain = st.slider("Selecciona el índice del paciente a explicar:", 0, len(df)-1, 0)
else:
    index_to_explain = 0
instance = df.iloc[[index_to_explain]]
shap_values_instance = explainer.shap_values(instance)
shap.force_plot(explainer.expected_value[0], shap_values_instance[0], instance, matplotlib=True, show=False)
fig = plt.gcf()
st.pyplot(fig)

#==================================
# 7. GENERACIÓN DE INFORME PDF
#==================================
# Función para generar el PDF
def generar_pdf(datos_usuario, prediccion_clase, probas, clases, qr_img_bytes):
    # Crear gráfico de barras de probabilidades
    fig, ax = plt.subplots(figsize=(5, 2))
    bars = ax.bar(clases, [p*100 for p in probas], color=["#ff4b4b" if clase == prediccion_clase else "#1f77b4" for clase in clases])
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title("Distribución de predicción")
    plt.tight_layout()
    plot_path = "temp_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)

    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "INFORME DE PREDICCIÓN DE TRASTORNOS DEL SUEÑO", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(200, 10, f"Fecha y hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "📋 Datos del paciente", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in datos_usuario.items():
        pdf.cell(200, 8, f"- {k}: {v}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "🧠 Resultado de la predicción", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(200, 8, f"→ Trastorno predicho: {prediccion_clase}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "📊 Probabilidades por clase", ln=True)
    pdf.set_font("Arial", "", 11)
    for clase, p in zip(clases, probas):
        pdf.cell(200, 8, f"- {clase}: {p*100:.2f}%", ln=True)
    pdf.ln(5)

    pdf.image(plot_path, x=40, w=130)
    os.remove(plot_path)

    # Incluir QR
    if qr_img_bytes:
        temp_qr_path = "temp_qr.png"
        with open(temp_qr_path, "wb") as f:
            f.write(qr_img_bytes)
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "🔗 Escanea el QR para volver a la aplicación:", ln=True)
        pdf.image(temp_qr_path, x=80, w=50)
        os.remove(temp_qr_path)

    return pdf

# Función para generar el enlace de descarga
def convertir_pdf_a_link(pdf):
    pdf.output("informe_prediccion.pdf")
    with open("informe_prediccion.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="informe_prediccion.pdf">📄 Descargar informe en PDF</a>'
    return href

datos_dict = df.iloc[0].to_dict()
predicted_class = model.predict(df)[0]
probas = model.predict_proba(df)

# Asegurar que sea una lista de probabilidades
if hasattr(probas, "__getitem__") and len(probas.shape) == 2:
    probas = probas[0].tolist()
else:
    probas = [float(probas)]  # en caso de que venga un único valor


#pdf = generar_pdf(datos_dict, predicted_class, probas)
pdf = generar_pdf(datos_dict, predicted_class, probas, list(model.classes_), qr_img)
st.markdown(convertir_pdf_a_link(pdf), unsafe_allow_html=True)