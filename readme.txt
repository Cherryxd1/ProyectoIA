    Sistema de Predicción de Congestión Vehicular - Chillán

**Proyecto de Inteligencia Artificial | Universidad del Bío-Bío**

Este proyecto implementa un sistema basado en Inteligencia Artificial para predecir niveles de congestión vehicular en puntos críticos de la ciudad de Chillán. Utiliza un modelo de Red Neuronal (Perceptrón Multicapa - MLP) para realizar estimaciones basadas en variables temporales, climáticas y geográficas.

Autores
* Diego Loyola
* Catalina Toro
* Valentina Zúñiga

Descripción del Proyecto

El objetivo principal es proporcionar una herramienta visual para anticipar el estado del tráfico. El sistema procesa datos históricos  para clasificar el flujo vehicular en tres categorías: Fluido, Moderado y Congestionado.

Características Principales

* Modelo Predictivo: Implementación de un MLPRegressor (Scikit-Learn) con arquitectura de cuatro capas ocultas (256, 128, 64, 32 neuronas).
* Visualización Geográfica: Mapas interactivos renderizados con Plotly sobre OpenStreetMap, mostrando indicadores de congestión por segmento.
* Interfaz de Usuario: Panel de control desarrollado en Streamlit que permite ajustar parámetros como hora, día y condiciones climáticas en tiempo real.
* Manejo de Datos: Módulo para la generación de datasets sintéticos basados en la topología real de Chillán en caso de ausencia de datos históricos.
* Métricas de Evaluación: Cálculo automático de R², MAE, RMSE y exactitud de clasificación.

Requisitos del Sistema

* Python 3.8 o superior.
* Las dependencias están listadas en el archivo `requirements.txt`.

Instalación y Ejecución

Para ejecutar el proyecto en un entorno local, siga los siguientes pasos:
python -m streamlit run app.py para runear se debe estar en la carpeta
ANTES DE TODO DEBE EJECUTAR EL ARCHIVO REPARAR_DATASET.PY (python reparar_dataset.py   )   
1. Preparación del entorno:
   Asegúrese de que los archivos `app.py`, `reparar_dataset.py` y `requirements.txt` se encuentren en el mismo directorio.

2. Instalación de dependencias:
   Se recomienda utilizar un entorno virtual. Ejecute el siguiente comando en la terminal 
   : // bash

   pip install -r requirements.txt
