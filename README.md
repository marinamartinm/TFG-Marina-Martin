# Análisis de la Calidad del Sueño (TFG)
## Alumna: Marina Martín Marijuán
## Grado en Ingeniería de la Salud, Universidad de Burgos. Curso 2024/2025
La calidad del sueño es un factor fundamental para la salud y el bienestar general, ya que influye directamente en el rendimiento cognitivo, la salud física, el estado emocional y la calidad de vida. Un sueño adecuado contribuye a la recuperación física y mental, mientras que su deficiencia se ha asociado con problemas graves como enfermedades cardiovasculares, diabetes, trastornos del estado de ánimo y déficits de concentración o memoria. Se estima que entre el 20% y el 30% de la población mundial sufre de trastornos del sueño, como insomnio o apnea obstructiva, lo que representa un desafío importante para los sistemas de salud pública. 

Este Trabajo de Fin de Grado (TFG) se centra en la predicción y clasificación de la calidad del sueño utilizando técnicas de machine learning. Se han evaluado diversos modelos para predecir trastornos del sueño basándose en características individuales y hábitos de vida. Además, se ha explorado el impacto de técnicas de balanceo de datos (SMOTE) y el uso de distintas estrategias de normalización en el rendimiento de los modelos.

# Tecnologías Utilizadas
- **Python** – `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`
- **Machine Learning** – Regresión logística, Árboles de decisión, Random Forest, SVM, MLP
- **Balanceo de clases** – SMOTE
- **Evaluación de modelos** – Validación cruzada, matriz de confusión, curva ROC, métricas por clase
- **Optimización de hiperparámetros** – `GridSearchCV`, `RandomizedSearchCV`
- **Despliegue web** – Streamlit
- **Generación de informes PDF** – `fpdf2`, `qrcode`

# Estructura del Repositorio
- Carpeta app: contiene el código de la aplicación en Streamlit.
- Carpeta artículos:
- Carpeta datos: carpeta que recoge todos los datos utilizados para la realización del TFG.
   - O2Ring: datos crudos del dispositivo O2Ring.
- Carpeta notebooks:
- Carpeta OverLeaf-LaTex: contiene los documentos y memoria en LaTex.
- README.md: archivo de presentación del repositorio de GitHub y explicación del proyecto.

# Acceso a la aplicación web

Puedes probar la aplicación directamente desde el siguiente enlace:

🔗 [Aplicación Streamlit desplegada](https://tfg-marina-martin.streamlit.app)

La aplicación permite:
- Introducir datos clínicos simulados
- Ejecutar la predicción
- Visualizar una explicación mediante gráficos SHAP
- Descargar un informe personalizado en PDF

# Memoria del proyecto

La memoria completa del TFG, junto con los anexos y resultados, está disponible en el repositorio en formato LaTeX dentro de `OverLeaf-LaTeX/`. Puede consultarse como documentación técnica y científica del sistema.

# Autoría y agradecimientos

Proyecto desarrollado por **Marina Martín Marijuán**, tutorizado por **Telmo Miguel Medina** en la **Universidad de Burgos** (Grado en Ingeniería de la Salud, curso 2024/2025).

# Licencia

Este proyecto se distribuye bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.


# Autores y Reconocimientos
Proyecto desarrollado por **Marina Martín Marijuán** bajo la supervisión de **Telmo Miguel Medina**
Universidad de Burgos, Ingeniería de la Salud (2024/2025)
