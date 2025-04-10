import streamlit as st
import pandas as pd
import joblib
import numpy as np
import qrcode
import streamlit as st
import io


# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Apnea del Sueño", 
    layout="wide", 
    page_icon="😴",
    initial_sidebar_state="expanded"
)


# Función para generar código QR
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
app_url = "http://192.168.1.133:8501"

# Generar imagen QR
qr_img = generar_qr(app_url)

# Mostrar en la barra lateral
st.sidebar.subheader("📱 Acceso desde el móvil")
st.sidebar.image(qr_img, caption="Escanea el QR para acceder", use_column_width=True)
st.sidebar.write(f"O copia y pega este enlace: [Haz clic aquí]({app_url})")
st.sidebar.markdown("---")


st.title("📊 Análisis de Apnea del Sueño")

# --- Cargar el modelo ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_random_forest_balanceado_calibrado.pkl')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# --- Entrada de datos del paciente ---
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
        polysomnography = st.selectbox("¿Datos de polisomnografía?", ["Sí", "No"], index=1)

    submitted = st.form_submit_button("🔄 Realizar predicción")

# --- Procesamiento y predicción ---
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
        polysomnography_mapping = {"Sí": 1, "No": 0}

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

        st.subheader("🔮 Resultado de la predicción")
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

# Explicabilidad del modelo con SHAP
st.header("🔍 Explicación del modelo (SHAP)")

# Solo para modelos tree-based (como Random Forest)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df)

# Mostrar gráfico resumen de importancia de variables
st.subheader("📌 Importancia global de las variables")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values, df, plot_type="bar", show=False)
st.pyplot(fig_summary)

# Mostrar gráfica de explicación individual
st.subheader("🔎 Explicación individual de la predicción")
fig_force = shap.plots._force.force_plot(explainer.expected_value[np.argmax(prediction_proba)],
                                          shap_values[np.argmax(prediction_proba)][0,:],
                                          df.iloc[0], matplotlib=True, show=False)
shap.save_html("force_plot.html", fig_force)
st.components.v1.html(open("force_plot.html").read(), height=300)

# Nota: SHAP puede ser pesado para Streamlit, se recomienda probar primero localmente
