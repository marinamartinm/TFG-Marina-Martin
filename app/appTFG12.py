import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de trastornos del sueño", layout="centered")
st.title("🧠 Predicción de trastornos del sueño")

st.markdown("""
Esta aplicación predice el riesgo de trastornos del sueño basándose en tus datos personales y hábitos diarios. Proporciona una estimación de riesgo para **insomnio**, **apnea del sueño** o **ninguno**.
""")

model = joblib.load("modelo_random_forest_balanceado_calibrado.pkl")

st.subheader("📋 Introduce tus datos:")

with st.form("formulario"):
    edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
    genero = st.selectbox("Género", ["Femenino", "Masculino"])
    ocupacion = st.selectbox("Ocupación", ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"])
    duracion_sueno = st.slider("Horas de sueño por noche", 0.0, 24.0, 8.0, step=0.5)
    calidad_sueno = st.slider("Calidad del sueño (1-10)", 1, 10, 7)
    nivel_actividad = st.slider("Nivel de actividad física (0-100)", 0, 100, 70)
    nivel_estres = st.slider("Nivel de estrés (0-10)", 0, 10, 4)
    categoria_imc = st.selectbox("Categoría de IMC", ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"])
    pasos_diarios = st.number_input("Número de pasos diarios", min_value=0, max_value=30000, value=5000)
    pas = st.number_input("Presión arterial sistólica (mmHg)", min_value=80, max_value=200, value=120)
    pad = st.number_input("Presión arterial diastólica (mmHg)", min_value=40, max_value=130, value=80)
    enviar = st.form_submit_button("Predecir")

if enviar:
    # Validación de inputs ya se aplica desde los widgets, pero aquí hacemos una doble comprobación defensiva
    errores = []

    if not 18 <= edad <= 100:
        errores.append("Edad fuera de rango")
    if not 0 <= duracion_sueno <= 24:
        errores.append("Horas de sueño no válidas")
    if not 1 <= calidad_sueno <= 10:
        errores.append("Calidad del sueño debe estar entre 1 y 10")
    if not 0 <= nivel_actividad <= 100:
        errores.append("Nivel de actividad física no válido")
    if not 0 <= nivel_estres <= 10:
        errores.append("Nivel de estrés fuera de rango")
    if not 0 <= pasos_diarios <= 30000:
        errores.append("Número de pasos no válido")
    if not 80 <= pas <= 200:
        errores.append("Presión sistólica fuera de rango")
    if not 40 <= pad <= 130:
        errores.append("Presión diastólica fuera de rango")

    if errores:
        st.error("Errores en la entrada de datos:")
        for e in errores:
            st.warning(f"- {e}")
    else:
        # Preparar el input para el modelo
        datos = {
            "Age": edad,
            "Sleep Duration": duracion_sueno,
            "Quality of Sleep": calidad_sueno,
            "Physical Activity Level": nivel_actividad,
            "Stress Level": nivel_estres,
            "BMI Category": ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"].index(categoria_imc),
            "Heart Rate": 0,
            "Daily Steps": pasos_diarios,
            "Gender_Male": 1 if genero == "Masculino" else 0,
        }

        ocupaciones = [
            "Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative",
            "Salesperson", "Scientist", "Software Engineer", "Teacher"
        ]
        for oc in ocupaciones:
            datos[f"Occupation_{oc}"] = 1 if ocupacion == oc else 0

        datos["Systolic BP"] = pas
        datos["Diastolic BP"] = pad

        df_input = pd.DataFrame([datos])
        pred_probs = model.predict_proba(df_input)[0]
        etiquetas = model.classes_ if hasattr(model, 'classes_') else ["Insomnio", "Ninguno", "Apnea del sueño"]

        st.subheader("🔮 Resultado de la predicción")
        for etiqueta, prob in zip(etiquetas, pred_probs):
            st.markdown(f"**{etiqueta}**: {prob*100:.1f}%")
