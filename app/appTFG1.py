import streamlit as st
import pandas as pd
import joblib
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
        if hasattr(model, 'feature_names_in_'):
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
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    occupation = st.selectbox("Occupation", [
        "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
        "Sales Representative", "Salesperson", "Scientist", 
        "Software Engineer", "Teacher"
    ], index=1)
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Sleep Quality (1-10)", 1, 10, 7)
    
with col2:
    physical_activity = st.slider("Physical Activity Level (1-100)", 1, 100, 70)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 4)
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"], index=1)
    blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)", "120/80")
    heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 72)

daily_steps = st.number_input("Daily Steps", 0, step=100, value=8000)
polysomnography = st.selectbox("Polysomnography Data", ["Yes", "No"], index=1)

# --- Procesamiento de datos ---
def preprocess_data(input_data):
    bmi_mapping = {
        "Underweight": 0, "Normal": 1, 
        "Overweight": 2, "Obese": 3
    }

    try:
        systolic, diastolic = map(int, input_data['blood_pressure'].split('/'))
    except:
        st.error("Incorrect blood pressure format. Use format like 120/80")
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

        'Gender_Male': float(1 if input_data['gender'] == 'Male' else 0),
        'Occupation_Doctor': float(1 if input_data['occupation'] == 'Doctor' else 0),
        'Occupation_Engineer': float(1 if input_data['occupation'] == 'Engineer' else 0),
        'Occupation_Lawyer': float(1 if input_data['occupation'] == 'Lawyer' else 0),
        'Occupation_Manager': float(1 if input_data['occupation'] == 'Manager' else 0),
        'Occupation_Nurse': float(1 if input_data['occupation'] == 'Nurse' else 0),
        'Occupation_Sales Representative': float(1 if input_data['occupation'] == 'Sales Representative' else 0),
        'Occupation_Salesperson': float(1 if input_data['occupation'] == 'Salesperson' else 0),
        'Occupation_Scientist': float(1 if input_data['occupation'] == 'Scientist' else 0),
        'Occupation_Software Engineer': float(1 if input_data['occupation'] == 'Software Engineer' else 0),
        'Occupation_Teacher': float(1 if input_data['occupation'] == 'Teacher' else 0)
    }

    df = pd.DataFrame([processed_data])

    # Asegurar que las columnas estén en el orden correcto
    if hasattr(model, 'feature_names_in_'):
        expected_columns = list(model.feature_names_in_)
        missing_cols = set(expected_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0.0  # Agregar las columnas faltantes con valor 0
        df = df[expected_columns]  # Ordenar correctamente las columnas
    
    return df


# --- Sección de resultados ---
if st.button("🔄 Realizar predicción", type="primary"):
    if model is None:
        st.error("Model not available")
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
            'daily_steps': daily_steps,
            'polysomnography': polysomnography
        }
        
        processed_df = preprocess_data(input_data)
        
        if processed_df is not None:
            try:
                prediction = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)
                
                st.subheader("🔮 Prediction Result")
                if prediction[0] == 1:
                    st.error("⚠️ **Sleep Apnea Risk Detected**")
                    st.metric("Probability", f"{prediction_proba[0][1]*100:.1f}%")
                else:
                    st.success("✅ **No Significant Sleep Apnea Risk Detected**")
                    st.metric("Probability", f"{prediction_proba[0][1]*100:.1f}%")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                if hasattr(model, 'feature_names_in_'):
                    st.write("Expected features:", list(model.feature_names_in_))
                st.write("Sent features:", list(processed_df.columns))
                st.write("Sent values:", processed_df.iloc[0].to_dict())