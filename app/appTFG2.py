import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Apnea del Sueño", layout="wide")

# --- Cargar el modelo ---
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.joblib')
    return model

model = load_model()


if hasattr(model, 'feature_names_in_'):
    st.write("Variables esperadas por el modelo:", model.feature_names_in_)

if hasattr(model, 'feature_importances_'):
    import matplotlib.pyplot as plt
    
    importance = model.feature_importances_
    features = model.feature_names_in_
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(features, importance)
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Características")
    ax.set_title("Importancia de las Características en la Predicción")
    
    st.pyplot(fig)



# --- Función de preprocesamiento IDÉNTICA al notebook ---
def preprocess_input(input_data):
    # 1. Convertir a DataFrame
    df = pd.DataFrame([input_data])
    
    # 2. Mapeo de categorías (debe ser IDÉNTICO al notebook)
    bmi_mapping = {"Bajo peso": 0, "Normal": 1, "Sobrepeso": 2, "Obeso": 3}
    df['BMI Category'] = df['BMI Category'].map(bmi_mapping)
    
    # 3. One-hot encoding para ocupación
    occupations = ['Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse', 
                  'Sales Representative', 'Salesperson', 'Scientist',
                  'Software Engineer', 'Teacher']
    
    for occ in occupations:
        df[f'Occupation_{occ}'] = (df['Occupation'] == occ).astype(int)
    
    # 4. Asegurar todas las columnas que el modelo espera
    missing_cols = set(model.feature_names_in_) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Rellenar con 0 las columnas faltantes
    
    # 5. Ordenar columnas exactamente como el modelo espera
    return df[model.feature_names_in_]

# --- Interfaz de usuario ---
st.title("Predictor de Apnea del Sueño")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=1, max_value=100, value=45)
        sleep_duration = st.slider("Duración del sueño (horas)", 0.0, 12.0, 6.5)
        quality = st.slider("Calidad del sueño (1-10)", 1, 10, 7)
        occupation = st.selectbox("Ocupación", [
            'Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse',
            'Sales Representative', 'Salesperson', 'Scientist',
            'Software Engineer', 'Teacher'
        ])
        
    with col2:
        activity = st.slider("Nivel de actividad física (1-100)", 1, 100, 65)
        stress = st.slider("Nivel de estrés (1-10)", 1, 10, 5)
        bmi = st.selectbox("Categoría IMC", ["Bajo peso", "Normal", "Sobrepeso", "Obeso"])
        gender = st.radio("Género", ["Mujer", "Hombre"])
        systolic, diastolic = st.columns(2)
        with systolic:
            systolic_bp = st.number_input("PA Sistólica", value=120)
        with diastolic:
            diastolic_bp = st.number_input("PA Diastólica", value=80)
    
    submitted = st.form_submit_button("Predecir")

# --- Procesamiento y predicción ---
if submitted:
    input_data = {
        'Age': float(age),
        'Sleep Duration': float(sleep_duration),
        'Quality of Sleep': float(quality),
        'Physical Activity Level': float(activity),
        'Stress Level': float(stress),
        'BMI Category': bmi,
        'Occupation': occupation,
        'Gender': gender,
        'Systolic BP': float(systolic_bp),
        'Diastolic BP': float(diastolic_bp),
        # Añade aquí todos los campos necesarios
    }
    
    # Transformación adicional para género
    input_data['Gender_Male'] = 1 if input_data['Gender'] == 'Hombre' else 0
    
    try:
        # Preprocesamiento
        processed_data = preprocess_input(input_data)
        
        # Predicción
        proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]
        
        # Mostrar resultados
        st.success("Predicción completada")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicción", prediction)
            
            # Gráfico de probabilidades
            prob_df = pd.DataFrame({
                'Clase': model.classes_,
                'Probabilidad': proba
            })
            st.bar_chart(prob_df.set_index('Clase'))
            
        with col2:
            st.write("Probabilidades:")
            for clase, prob in zip(model.classes_, proba):
                st.write(f"- {clase}: {prob:.1%}")
                
        # Debug (opcional)
        with st.expander("Detalles técnicos (debug)"):
            st.write("Datos procesados:", processed_data.iloc[0].to_dict())
            st.write("Features esperados:", model.feature_names_in_)
            
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        st.write("Por favor verifica que todos los campos estén completos.")