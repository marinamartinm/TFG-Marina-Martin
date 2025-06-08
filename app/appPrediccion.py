# 1. LIBRERÍAS Y CONFIGURACIÓN
import streamlit as st
import pandas as pd
import joblib
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

import qrcode, io, shap, base64, os
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Predicción de Trastornos del Sueño", page_icon="logoApp.png", layout="wide")
st.markdown("# 💤 Predicción de Trastornos del Sueño")

# 2. CÓDIGO QR
def generar_qr(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

app_url = "https://tfg-marina-martin.streamlit.app"
st.sidebar.subheader("📱 Acceso desde el móvil")
st.sidebar.image(generar_qr(app_url), caption="Escanea el QR", use_container_width=True)
st.sidebar.markdown(f"[Accede aquí]({app_url})")

# 3. CARGA DE MODELO Y SCALER
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    columnas_entrenamiento = joblib.load("columnas_entrenamiento.joblib")
    st.success("Modelo y scaler cargados correctamente")
except Exception as e:
    st.error(f"Error al cargar modelo o scaler: {e}")
    st.stop()

# 4. PDF CLÍNICO
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

    def add_table(self, data):
        self.set_font("Arial", "", 11)
        for key, value in data.items():
            self.cell(60, 8, str(key), border=1)
            self.cell(0, 8, str(value), border=1, ln=True)
        self.ln(3)

    def add_prediction(self, clase, probas, clases):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, f"Trastorno detectado: {clase}", ln=True)
        self.set_font("Arial", "", 11)
        for c, p in zip(clases, probas):
            self.cell(0, 8, f"{c}: {p*100:.1f}%", ln=True)
        self.ln(3)

def generar_informe_estetico(datos_usuario, prediccion_clase, probas, clases):
    pdf = InformeClinico()
    pdf.add_page()
    pdf.section_title("DATOS PERSONALES")
    sexo = "Masculino" if datos_usuario.get("Gender_Male") == 1 else "Femenino"
    ocupaciones = [k.replace("Occupation_", "") for k in datos_usuario if k.startswith("Occupation_") and datos_usuario[k] == 1]
    datos_personales = {
        "Edad": f"{datos_usuario.get('Age')} años",
        "Sexo": sexo,
        "Ocupación": ocupaciones[0] if ocupaciones else "No especificada",
        "IMC": ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"][datos_usuario.get("BMI Category", 1)],
        "Frecuencia cardíaca": f"{datos_usuario.get('Heart Rate')} bpm",
        "Presión arterial": f"{datos_usuario.get('Systolic BP')} / {datos_usuario.get('Diastolic BP')} mmHg"
    }
    pdf.add_table(datos_personales)

    pdf.section_title("HÁBITOS")
    datos_habitos = {
        "Horas de sueño": datos_usuario.get("Sleep Duration"),
        "Calidad del sueño": datos_usuario.get("Quality of Sleep"),
        "Nivel de estrés": datos_usuario.get("Stress Level"),
        "Pasos diarios": datos_usuario.get("Daily Steps")
    }
    pdf.add_table(datos_habitos)

    pdf.section_title("PREDICCIÓN")
    pdf.add_prediction(prediccion_clase, probas, clases)

    ruta = "informe_prediccion.pdf"
    pdf.output(ruta)
    return ruta

# 5. FUNCIÓN DE PREPROCESAMIENTO
def preparar_input_con_scaler(datos_dict, columnas_entrenamiento, scaler):
    df = pd.DataFrame([datos_dict])
    df = pd.get_dummies(df)
    for col in columnas_entrenamiento:
        if col not in df.columns:
            df[col] = 0.0
    df = df[columnas_entrenamiento]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_scaled

# 6. FORMULARIO DE ENTRADA
st.header("📝 Datos del paciente")
with st.form("formulario"):
    col1, col2 = st.columns(2)
    with col1:
        genero = st.radio("Género", ["Female", "Male"])
        edad = st.number_input("Edad", 1, 100, 30)
        ocupacion = st.selectbox("Ocupación", ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
                                               "Sales Representative", "Salesperson", "Scientist",
                                               "Software Engineer", "Teacher"])
        peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        altura = st.number_input("Altura (cm)", 140.0, 220.0, 170.0)
        sueno = st.slider("Duración del sueño (h)", 0.0, 12.0, 7.0, 0.1)
        calidad = st.slider("Calidad del sueño (1-10)", 1, 10, 7)
    with col2:
        deporte = st.selectbox("Días de ejercicio a la semana", list(range(8)))
        estres = st.selectbox("Nivel de estrés", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"])
        presion = st.text_input("Presión arterial (ej: 120/80)", "120/80")
        frecuencia = st.number_input("Frecuencia cardíaca (bpm)", 30, 180, 72)
        pasos = st.number_input("Pasos diarios", 0, 30000, 8000)
    enviado = st.form_submit_button("🔄 Realizar predicción")

# 7. PROCESAMIENTO Y PREDICCIÓN
if enviado:
    try:
        imc = peso / (altura / 100)**2
        if imc < 18.5:
            bmi = "Underweight"
        elif imc < 25:
            bmi = "Normal"
        elif imc < 30:
            bmi = "Overweight"
        else:
            bmi = "Obese"
        bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}

        nivel_estres = {"Muy bajo": 1, "Bajo": 3, "Moderado": 5, "Alto": 7, "Muy alto": 9}
        deporte_map = {i: p for i, p in enumerate([0, 20, 40, 60, 70, 80, 90, 100])}

        sistolica, diastolica = map(int, presion.split("/"))

        datos_usuario = {
            "Age": edad,
            "Sleep Duration": sueno,
            "Quality of Sleep": calidad,
            "Physical Activity Level": deporte_map[deporte],
            "Stress Level": nivel_estres[estres],
            "Heart Rate": frecuencia,
            "Daily Steps": pasos,
            "Systolic BP": sistolica,
            "Diastolic BP": diastolica,
            "BMI Category": bmi_mapping[bmi],
            "Gender_Male": 1 if genero == "Male" else 0
        }
        for o in ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
                  "Sales Representative", "Salesperson", "Scientist",
                  "Software Engineer", "Teacher"]:
            datos_usuario[f"Occupation_{o}"] = 1.0 if o == ocupacion else 0.0

        df = preparar_input_con_scaler(datos_usuario, columnas_entrenamiento, scaler)
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        clases = model.classes_

        st.subheader("Resultado de la predicción")
        colA, colB, colC = st.columns(3)
        for i, col in zip(range(len(clases)), [colA, colB, colC]):
            col.metric(clases[i], f"{proba[i]*100:.1f}%")

        ruta = generar_informe_estetico(datos_usuario, pred, proba, clases)
        with open(ruta, "rb") as f:
            st.download_button("📄 Descargar informe PDF", f, file_name="informe_prediccion.pdf")

        # SHAP
        base_model = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(df)

        st.subheader("Importancia global de las variables")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df, plot_type="bar", show=False)
        st.pyplot(fig)

        st.subheader("Explicación individual")
        shap.force_plot(explainer.expected_value[0], shap_values[0][0], df.iloc[0], matplotlib=True, show=False)
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"Error: {e}")
