# TDL-Acoustic Hub 🎙️
### **Edge AI para el Diagnóstico Fonológico y Perfilado Clínico Multitarea**
---
**Proyecto desarrollado para la Hackathon iabiomed 2026**

TDL-Acoustic Hub es una plataforma avanzada de Inteligencia Artificial diseñada para el análisis acústico clínico inmediato. Utilizando muestras de voz digitalizadas, el sistema es capaz de extraer marcadores fonológicos (reconocimiento de palabras), perfiles demográficos (origen geográfico/acentos) y variantes biológicas (sexo), sirviendo como una herramienta de cribado masivo, ágil y privada para entornos hospitalarios o de investigación.

---

## 🚀 Características Principales

* **Red Neuronal Multicabeza (Multi-Head NN):** Un único tronco común de extracción de características acústicas alimenta tres cabezas de clasificación independientes (Palabra, Origen y Sexo), optimizando el rendimiento computacional.
* **Procesamiento en el Borde (Edge AI):** Toda la inferencia y extracción de características se realiza localmente en el dispositivo. No se envían datos biométricos de voz a servidores externos, garantizando el cumplimiento estricto de la privacidad del paciente.
* **Pipeline de Datos Automático:** Sistema robusto de emparejamiento ("Matcher Inteligente") capaz de mapear audios recursivamente en subcarpetas y sincronizar metadatos automáticamente.
* **Diccionario Dinámico Escalable:** Uso de serialización de objetos (`LabelEncoder`) que permite expandir el vocabulario de la IA sin necesidad de modificar el código de la interfaz gráfica.

---

## 🛠️ Stack Tecnológico

* **Análisis de Audio:** `Librosa` (Extracción de MFCCs, Deltas y Delta-Deltas a 16kHz).
* **Core de Inteligencia Artificial:** `TensorFlow` / `Keras` (Redes neuronales densas funcionales).
* **Procesamiento de Datos:** `Pandas`, `NumPy`, `Scikit-learn`.
* **Serialización:** `Joblib`.
* **Interfaz de Usuario:** `Streamlit`.

---

## 📁 Estructura del Repositorio

El proyecto está organizado en módulos independientes que componen el pipeline completo de Machine Learning:

* `etiquetado.py`: Escanea de forma recursiva la carpeta de audios y autogenera el dataset limpio (`dataset_etiquetado.csv`), deduciendo etiquetas a partir del nombre del archivo físico.
* `procesar_audios.py`: Lee el mapa de archivos generado y extrae los coeficientes cepstrales en las frecuencias de Mel (MFCCs), deltas y delta-deltas.
* `preprocesamiento_datos.py`: Consolida y normaliza la matriz de características acústicas mediante técnicas de escalado.
* `modelo_entrenamiento.py`: Construye, compila y entrena la red neuronal multicabeza, exportando el modelo final (`modelo_final.keras`) y sus diccionarios de traducción (`*.pkl`).
* `app2.py`: Dashboard de control clínico interactivo para subir muestras de audio y visualizar diagnósticos en tiempo real.

---

## 📋 Configuración e Instalación

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd TDL
pip install -r requirements.txt

### 2.Ejecutar Reentrenamiento.bat si se ingresan nuevasmuestras

### 3. Ejecutar Iniciador_TDL.bat

*Arrastar o seleccionar el audio

*Esperar a obtener los resultados
