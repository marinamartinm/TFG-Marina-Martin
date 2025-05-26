import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis O2Ring", layout="wide")
st.title("📊 Análisis de datos del O2Ring")

st.markdown("Sube un archivo CSV exportado desde el dispositivo O2Ring para visualizar y analizar los datos.")

# Carga de archivo
archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente.")

    # Mostrar datos brutos
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Verificar columnas comunes (adaptar si varían los nombres)
    columnas_esperadas = ["Time", "SpO2", "Heart Rate"]
    if all(col in df.columns for col in columnas_esperadas):
        st.subheader("Resumen estadístico")
        st.write(df[columnas_esperadas].describe())

        st.subheader("Gráfica de saturación de oxígeno (SpO2)")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["Time"], df["SpO2"], label="SpO2 (%)", color='blue')
        ax1.set_xlabel("Tiempo")
        ax1.set_ylabel("SpO2 (%)")
        ax1.set_title("Evolución de SpO2")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("Gráfica de frecuencia cardíaca")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Time"], df["Heart Rate"], label="Frecuencia Cardíaca (bpm)", color='red')
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("BPM")
        ax2.set_title("Evolución de Frecuencia Cardíaca")
        ax2.legend()
        st.pyplot(fig2)

        # Análisis de desaturaciones
        st.subheader("Eventos de desaturación (< 90%)")
        desats = df[df["SpO2"] < 90]
        st.write(f"Número de eventos: {len(desats)}")
        if not desats.empty:
            st.dataframe(desats)
    else:
        st.error("El archivo debe contener las columnas: 'Time', 'SpO2', 'Heart Rate'")
