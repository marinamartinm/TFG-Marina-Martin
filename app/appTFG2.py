import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Apnea del Sueño", 
    layout="wide", 
    page_icon="🛌",
    initial_sidebar_state="expanded"
)

st.title("📊 Análisis de Apnea del Sueño")

# --- Cargar el modelo ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.joblib")
        if hasattr(model, "feature_names_in_"):
            st.sidebar.write("Variables esperadas por el modelo:")
            st.sidebar.write(list(model.feature_names_in_))
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# --- Sección de entrada de datos ---
st.header("📝 Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Género", ["Female", "Male"], index=0)
    age = st.number_input("Edad", min_value=1, max_value=120, value=30)
    occupation = st.selectbox("Ocupación", [
        "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
        "Sales Representative", "Salesperson", "Scientist", 
        "Software Engineer", "Teacher"
    ], index=1)
    sleep_duration = st.slider("Duración del sueño (horas)", 0.0, 12.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Calidad del sueño (1-10)", 1, 10, 7)

with col2:
    physical_activity = st.slider("Nivel de actividad física (1-100)", 1, 100, 70)
    stress_level = st.slider("Nivel de estrés (1-10)", 1, 10, 4)
    bmi_category = st.selectbox("Categoría de IMC", ["Underweight", "Normal", "Overweight", "Obese"], index=1)
    blood_pressure = st.text_input("Presión arterial (ej: 120/80)", "120/80")
    heart_rate = st.number_input("Frecuencia Cardíaca (bpm)", 30, 200, 72)

daily_steps = st.number_input("Pasos diarios", 0, step=100, value=8000)

# --- Preprocesamiento de datos ---
def preprocess_data(input_data):
    bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}

    try:
        systolic, diastolic = map(int, input_data['blood_pressure'].split('/'))
    except:
        st.error("Formato incorrecto de la presión arterial. Usa el formato 120/80")
        return None

    processed_data = {
        'Age': float(input_data['age']),
        'Sleep Duration': float(input_data['sleep_duration']),
        'Quality of Sleep': float(input_data['quality_of_sleep']),
        'Physical Activity Level': float(input_data['physical_activity']),
        'Stress Level': float(input_data['stress_level']),
        'BMI Category': float(bmi_mapping[input_data['bmi_category']]),
        'Heart Rate': float(input_data['heart_rate']),
        'Daily Steps': float(input_data['daily_steps']),
        'Systolic BP': float(systolic),
        'Diastolic BP': float(diastolic),
        'Gender_Male': float(1 if input_data['gender'] == 'Male' else 0)
    }

    # Dummy encoding para ocupaciones
    occupations = [
        'Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse',
        'Sales Representative', 'Salesperson', 'Scientist',
        'Software Engineer', 'Teacher'
    ]
    for occ in occupations:
        key = f"Occupation_{occ}"
        processed_data[key] = float(1 if input_data['occupation'] == occ else 0)

    df = pd.DataFrame([processed_data])

    if hasattr(model, 'feature_names_in_'):
        expected_columns = list(model.feature_names_in_)
        missing_cols = set(expected_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0.0
        df = df[expected_columns]

    return df

# --- Predicción ---
if st.button("🔄 Realizar predicción", type="primary"):
    if model is None:
        st.error("Modelo no disponible")
    else:
        input_data = {
            'gender': gender,
            'age': age,
            'occupation': occupation,
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
                prediction = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)[0]

                st.subheader("🔮 Resultado de la predicción")
                classes = model.classes_ if hasattr(model, "classes_") else ["Clase 0", "Clase 1", "Clase 2"]
                results = dict(zip(classes, prediction_proba))
                for label, prob in results.items():
                    st.metric(label=label, value=f"{prob*100:.1f}%")

            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
                if hasattr(model, 'feature_names_in_'):
                    st.write("Variables esperadas:", list(model.feature_names_in_))
                st.write("Variables enviadas:", list(processed_df.columns))
                st.write("Valores enviados:", processed_df.iloc[0].to_dict())
