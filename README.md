# Aplicación web para la predicción de trastornos del sueño con apoyo del dispositivo O2Ring
## Alumna: Marina Martín Marijuán
## Grado en Ingeniería de la Salud, Universidad de Burgos. Curso 2024/2025

La calidad del sueño es un factor clave para la salud física y mental, y su deterioro se asocia con problemas como insomnio, apnea del sueño o fatiga crónica. A pesar de su relevancia, los mecanismos de cribado temprano y diagnóstico accesible siguen siendo limitados en muchos entornos clínicos o poblacionales.

Este Trabajo de Fin de Grado plantea el desarrollo de un sistema predictivo que clasifica el tipo de trastorno del sueño en tres categorías (insomnio, apnea, ninguno), a partir de variables clínicas y de estilo de vida. La solución combina el uso de modelos de aprendizaje automático (Random Forest, MLP, SVM, entre otros) con técnicas de interpretabilidad como SHAP y una interfaz visual desarrollada en Streamlit.

El objetivo es ofrecer una herramienta accesible, explicable y fácilmente desplegable para su uso en contextos clínicos, educativos o de investigación. La aplicación final permite introducir datos simulados o reales, obtener una predicción y visualizar un informe explicativo en formato PDF.

## Tecnologías Utilizadas
- **Python** – `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`.
- **Machine Learning** – Regresión logística, Árboles de decisión, Random Forest, SVM, MLP.
- **Balanceo de clases** – SMOTE.
- **Evaluación de modelos** – Validación cruzada, matriz de confusión, curva ROC, métricas por clase.
- **Optimización de hiperparámetros** – `GridSearchCV`, `RandomizedSearchCV`.
- **Interfaz web** – Streamlit.
- **Generación de informes PDF** – `fpdf2`, `qrcode`.
- Adquisición de datos - **O2 Insight Pro** (software propietario del dispositivo O₂Ring, Viatom).

## Estructura del Repositorio
- Carpeta app: contiene el código de la aplicación en Streamlit y los archivos necesarios.
- Carpeta artículos: recopila los artículos científicos más relevantes utilizados durante el desarrollo del trabajo.
- Carpeta datos: carpeta que recoge todos los datos utilizados para la realización del TFG.
   - O2Ring: datos crudos del dispositivo O2Ring.
- Carpeta diagramas: contiene los diagramas de clases, casos de uso y estructuras incluidas en la documentación.
- Carpeta imágenes: carpeta con todos los recursos gráficos utilizados en la memoria y anexos.
- Carpeta memoria: carpeta que contiene la versión final de la memoria y los anexos en formato PDF.
- Carpeta notebooks: cuadernos Jupyter utilizados durante la fase de análisis exploratorio, entrenamiento del modelo, evaluación y visualización de resultados.
- Carpeta OverLeaf-LaTex: contiene la documentación de la memoria y anexos compatible con Overleaf (proyecto completo).
- README.md: archivo de presentación del repositorio de GitHub y explicación del proyecto.
- requirements.txt: listado de librerías necesarias para la ejecución de la aplicación localmente.
- runtime.txt: archivo auxiliar usado para indicar la versión de Python en entornos de despliegue.

## Acceso a la aplicación web

Puedes probar la aplicación directamente desde el siguiente enlace:

🔗 [Aplicación Streamlit desplegada](https://tfg-marina-martin.streamlit.app)

La aplicación permite:
- Introducir datos clínicos simulados
- Ejecutar la predicción
- Visualizar una explicación mediante gráficos SHAP
- Descargar un informe personalizado en PDF

## Memoria del proyecto

La memoria completa del TFG, junto con los anexos y resultados, está disponible en el repositorio en formato LaTeX dentro de `OverLeaf-LaTeX/`. Puede consultarse como documentación técnica y científica del sistema.

## Licencia

Este proyecto se distribuye bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.


# Autores y Reconocimientos
Proyecto desarrollado por **Marina Martín Marijuán** bajo la supervisión de **Telmo Miguel Medina**
Universidad de Burgos, Ingeniería de la Salud (2024/2025)
