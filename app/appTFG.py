import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Apnea del Sueño", 
    layout="wide", 
    page_icon="😴",
    initial_sidebar_state="expanded"
)

st.title("📊 Análisis de Apnea del Sueño")

# --- Cargar el modelo ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# Mostrar información del modelo si se cargó correctamente
if model is not None:
    st.sidebar.success("✅ Modelo cargado correctamente")
    st.sidebar.write(f"Algoritmo: {type(model).__name__}")

# --- Sección de entrada de datos ---
st.header("📝 Datos del paciente")

# Organización en columnas
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Género", ["Hombre", "Mujer"], index=0)
    age = st.number_input("Edad", min_value=1, max_value=120, step=1, value=30)
    occupation = st.selectbox("Ocupación", ["Doctor", "Ingeniero", "Profesor", "Vendedor", "Otro"], index=2)
    sleep_duration = st.slider("Duración del sueño (horas)", min_value=0.0, max_value=12.0, step=0.1, value=7.0)
    quality_of_sleep = st.slider("Calidad del sueño (1-10)", min_value=1, max_value=10, step=1, value=5)
    
with col2:
    physical_activity = st.slider("Nivel de actividad física (1-100)", min_value=1, max_value=100, step=1, value=50)
    stress_level = st.slider("Nivel de estrés (1-10)", min_value=1, max_value=10, step=1, value=5)
    bmi_category = st.selectbox("Categoría de IMC", ["Bajo peso", "Normal", "Sobrepeso", "Obeso"], index=1)
    blood_pressure = st.text_input("Presión arterial (Ej: 120/80)", value="120/80")
    heart_rate = st.number_input("Frecuencia cardíaca (bpm)", min_value=30, max_value=200, step=1, value=72)

daily_steps = st.number_input("Pasos diarios", min_value=0, step=100, value=5000)

# --- Procesamiento de datos ---
def preprocess_data(input_data):
    # Mapear categorías a valores numéricos
    bmi_mapping = {"Bajo peso": 0, "Normal": 1, "Sobrepeso": 2, "Obeso": 3}
    
    # Convertir datos para el modelo
    processed_data = {
        'Gender': 1 if input_data['gender'] == 'Hombre' else 0,
        'Age': input_data['age'],
        'Sleep Duration': input_data['sleep_duration'],
        'Quality of Sleep': input_data['quality_of_sleep'],
        'Physical Activity Level': input_data['physical_activity'],
        'Stress Level': input_data['stress_level'],
        'BMI Category': bmi_mapping[input_data['bmi_category']],
        'Heart Rate': input_data['heart_rate'],
        'Daily Steps': input_data['daily_steps']
    }
    
    # Procesar presión arterial
    try:
        systolic, diastolic = map(int, input_data['blood_pressure'].split('/'))
        processed_data['Systolic BP'] = systolic
        processed_data['Diastolic BP'] = diastolic
    except:
        st.error("Formato de presión arterial incorrecto. Use formato como 120/80")
        return None
    
    return pd.DataFrame([processed_data])

# --- Sección de resultados ---
st.header("🔍 Resultados")

if st.button("📊 Ver datos ingresados"):
    data = {
        "Género": gender,
        "Edad": age,
        "Ocupación": occupation,
        "Duración del sueño": sleep_duration,
        "Calidad del sueño": quality_of_sleep,
        "Actividad física": physical_activity,
        "Estrés": stress_level,
        "IMC": bmi_category,
        "Presión arterial": blood_pressure,
        "Frecuencia cardíaca": heart_rate,
        "Pasos diarios": daily_steps
    }
    df = pd.DataFrame([data])
    st.write("Datos ingresados:")
    st.dataframe(df.style.highlight_max(axis=0, color='#fffd75'))

if st.button("🔄 Realizar predicción", type="primary"):
    if model is None:
        st.error("El modelo no está disponible. Por favor, contacte al administrador.")
    else:
        with st.spinner('Analizando datos...'):
            # Preparar datos
            input_data = {
                'gender': gender,
                'age': age,
                'sleep_duration': sleep_duration,
                'quality_of_sleep': quality_of_sleep,
                'physical_activity': physical_activity,
                'stress_level': stress_level,
                'bmi_category': bmi_category,
                'blood_pressure': blood_pressure,
                'heart_rate': heart_rate,
                'daily_steps': daily_steps
            }
            
            processed_df = preprocess_data(input_data)
            
            if processed_df is not None:
                try:
                    # Hacer predicción
                    prediction = model.predict(processed_df)
                    prediction_proba = model.predict_proba(processed_df)
                    
                    # Mostrar resultados
                    st.subheader("🔮 Resultado de la predicción")
                    
                    # Personaliza según las clases de tu modelo
                    if prediction[0] == 1:  # Asumiendo 1 es apnea
                        st.error("⚠️ **Riesgo de apnea del sueño detectado**")
                        st.metric("Probabilidad", f"{prediction_proba[0][1]*100:.1f}%")
                        
                        # Recomendaciones específicas
                        with st.expander("📌 Recomendaciones"):
                            st.write("""
                            - **Consulta especializada**: Programe una cita con un neumólogo o especialista en sueño
                            - **Estudio del sueño**: Considere realizarse una polisomnografía
                            - **Cambios de estilo de vida**:
                              - Mantenga un horario regular de sueño
                              - Evite alcohol y sedantes antes de dormir
                              - Duerma de lado en lugar de boca arriba
                              - Pierda peso si tiene sobrepeso
                            - **Dispositivos**: En casos severos, puede necesitar CPAP
                            """)
                    else:
                        st.success("✅ **No se detectó riesgo significativo de apnea del sueño**")
                        st.metric("Probabilidad", f"{prediction_proba[0][1]*100:.1f}%")
                        
                        with st.expander("💡 Consejos para mantener un sueño saludable"):
                            st.write("""
                            - Mantenga rutinas regulares de sueño
                            - Realice actividad física regular
                            - Limite el consumo de cafeína por la tarde/noche
                            - Cree un ambiente oscuro y tranquilo para dormir
                            """)
                    
                    # Gráfico de probabilidades
                    prob_df = pd.DataFrame({
                        'Categoría': ['Sin apnea', 'Apnea'],
                        'Probabilidad': [prediction_proba[0][0], prediction_proba[0][1]]
                    })
                    st.bar_chart(prob_df.set_index('Categoría'))
                    
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")

# --- Sección informativa ---
st.header("ℹ️ Información sobre apnea del sueño")
with st.expander("Ver más información"):
    st.write("""
    **¿Qué es la apnea del sueño?**
    
    La apnea del sueño es un trastorno común donde la respiración se detiene o se hace superficial durante el sueño. 
    Estas pausas pueden durar desde segundos hasta minutos y ocurrir 30+ veces por hora.
    
    **Síntomas comunes:**
    - Ronquidos fuertes
    - Pausas observadas en la respiración
    - Jadeos al respirar durante el sueño
    - Despertar con la boca seca
    - Dolor de cabeza matutino
    - Somnolencia diurna excesiva (hipersomnia)
    
    **Factores de riesgo:**
    - Sobrepeso
    - Cuello grueso
    - Vías respiratorias estrechas
    - Ser hombre
    - Edad avanzada
    - Historial familiar
    - Consumo de alcohol/sedantes
    - Fumar
    """)

# --- Pie de página ---
st.markdown("---")
st.caption("Aplicación desarrollada para TFG de Ingeniería de la Salud - © 2023")