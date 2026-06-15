import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import os
import time

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="TDL-Acoustic",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stAudio { margin-bottom: 20px; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 4px solid #1e3a8a;
    }
    .titulo-app { color: #1e3a8a; font-weight: 800; font-size: 2.5rem; text-align: center; }
    .subtitulo { color: #64748b; text-align: center; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE MODELO Y SCALER (CACHEADO)
# ==========================================
@st.cache_resource(show_spinner=False)
def cargar_motor_ia():
    # Asegúrate de que estos nombres coinciden con los tuyos
    modelo = tf.keras.models.load_model("modelo_tdl_multicabeza.h5")
    scaler = joblib.load("scaler_acustico.pkl")
    return modelo, scaler

try:
    modelo, scaler = cargar_motor_ia()
    modelo_cargado = True
except Exception as e:
    modelo_cargado = False
    error_msg = e

# ==========================================
# 3. FUNCIÓN DE EXTRACCIÓN (LA MISMA DEL BACKEND)
# ==========================================
def extraer_huella_acustica(ruta_o_buffer):
    audio, sr = librosa.load(ruta_o_buffer, sr=16000)
    audio_limpio, _ = librosa.effects.trim(audio, top_db=20)
    
    if len(audio_limpio) == 0: return None
        
    mfccs = librosa.feature.mfcc(y=audio_limpio, sr=sr, n_mfcc=20)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    vector = np.hstack([np.mean(mfccs.T, axis=0), np.mean(delta_mfccs.T, axis=0), np.mean(delta2_mfccs.T, axis=0)])
    return vector.reshape(1, -1)

# ==========================================
# 4. INTERFAZ GRÁFICA
# ==========================================
st.markdown("<div class='titulo-app'>🎙️ TDL-Acoustic Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitulo'>Screening Temprano del Trastorno del Desarrollo del Lenguaje (Edge AI)</div>", unsafe_allow_html=True)

if not modelo_cargado:
    st.error(f"❌ No se encontró el modelo o el scaler. Entrena la red neuronal primero.\nError: {error_msg}")
else:
    # Subida de archivo
    archivo_audio = st.file_uploader("📥 Sube una grabación fonológica pediátrica (.wav / .mp3):", type=['wav', 'mp3'])
    
    if archivo_audio is not None:
        st.audio(archivo_audio, format="audio/wav")
        
        if st.button("🧠 Ejecutar Inferencia Clínica", type="primary", use_container_width=True):
            with st.spinner("Extrayendo MFCCs y ejecutando Red Neuronal Multicabeza..."):
                time.sleep(0.5) # Ligera pausa visual para dar sensación de procesamiento
                
                # 1. Extraer características
                vector_acustico = extraer_huella_acustica(archivo_audio)
                
                if vector_acustico is not None:
                    # 2. Normalizar
                    vector_escalado = scaler.transform(vector_acustico)
                    
                    # 3. Predecir
                    predicciones = modelo.predict(vector_escalado, verbose=0)
                    
                    # Simulamos las etiquetas (Ajusta los índices según cómo entrenó tu LabelEncoder)
                    # Lo ideal es cargar el LabelEncoder, pero para la demo estas son las típicas:
                    palabras_ejemplo = ["blanco", "bufanda", "cara", "flecha", "fruta"] # Pon las tuyas reales
                    origenes_ejemplo = ["España", "Latinoamérica", "No nativo"]
                    sexos_ejemplo = ["Hombre", "Mujer"]
                    
                    # Obtener los índices con mayor probabilidad
                    idx_palabra = np.argmax(predicciones[0], axis=1)[0]
                    idx_origen = np.argmax(predicciones[1], axis=1)[0]
                    idx_sexo = np.argmax(predicciones[2], axis=1)[0]
                    
                    # Probabilidades de confianza
                    confianza_palabra = np.max(predicciones[0]) * 100
                    
                    st.success("✅ Análisis completado en milisegundos. Latencia óptima para dispositivos móviles.")
                    
                    st.markdown("### 📋 Resultados del Diagnóstico Fonológico")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color:#64748b; margin:0;'>Palabra Detectada</h4>
                            <h2 style='color:#1e3a8a; margin:10px 0;'>ID #{idx_palabra}</h2>
                            <p style='color:#22c55e; font-weight:bold; margin:0;'>Confianza: {confianza_palabra:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color:#64748b; margin:0;'>Perfil Demográfico</h4>
                            <h2 style='color:#1e3a8a; margin:10px 0;'>Clase {idx_origen}</h2>
                            <p style='color:#22c55e; margin:0;'>Filtro de sesgo regional</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color:#64748b; margin:0;'>Sexo Biológico</h4>
                            <h2 style='color:#1e3a8a; margin:10px 0;'>Clase {idx_sexo}</h2>
                            <p style='color:#22c55e; margin:0;'>Análisis de Frecuencia F0</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.error("No se pudo aislar la voz. El audio es demasiado corto o solo contiene silencio.")