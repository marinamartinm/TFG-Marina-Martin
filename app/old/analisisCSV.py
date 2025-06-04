import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis O2Ring", layout="wide")
st.title("Análisis de datos del O2Ring")

st.markdown("Sube un archivo CSV exportado desde el dispositivo O2Ring para visualizar y analizar los datos.")

archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    df = df.dropna(axis=1, how='all')  # Eliminar columnas vacías
    df.rename(columns={"SpO2(%)": "SpO2", "Pulse Rate(bpm)": "Heart Rate"}, inplace=True)

    st.success("Archivo cargado correctamente.")
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    columnas_necesarias = ["Time", "SpO2", "Heart Rate"]
    if all(col in df.columns for col in columnas_necesarias):
        st.subheader("Resumen estadístico")
        st.write(df[["SpO2", "Heart Rate"]].describe())

        st.subheader("Evolución de la SpO₂")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["Time"], df["SpO2"], label="SpO₂ (%)", color="blue")
        ax1.set_xlabel("Tiempo")
        ax1.set_ylabel("SpO₂ (%)")
        ax1.set_title("SpO₂ durante el registro")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        st.subheader("Evolución de la Frecuencia Cardíaca")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Time"], df["Heart Rate"], label="Frecuencia Cardíaca", color="red")
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("BPM")
        ax2.set_title("Frecuencia Cardíaca durante el registro")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        st.subheader("Eventos de desaturación (< 90%)")
        desats = df[df["SpO2"] < 90]
        st.write(f"Número de eventos detectados: {len(desats)}")
        if not desats.empty:
            st.dataframe(desats)
    else:
        st.error("El archivo no contiene las columnas necesarias: Time, SpO2(%), Pulse Rate(bpm)")
