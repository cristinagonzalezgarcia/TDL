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
# 3. FUNCIÓN DE EXTRACCIÓN
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
st.markdown("<div class='subtitulo'>Screening Temprano del TDL (Edge AI)</div>", unsafe_allow_html=True)

if not modelo_cargado:
    st.error(f"❌ Error al cargar modelo/scaler: {error_msg}")
else:
    archivo_audio = st.file_uploader("📥 Sube una grabación fonológica (.wav / .mp3):", type=['wav', 'mp3'])
    
    if archivo_audio is not None:
        st.audio(archivo_audio, format="audio/wav")
        
        if st.button("🧠 Ejecutar Inferencia Clínica", type="primary", use_container_width=True):
            with st.spinner("Analizando..."):
                vector_acustico = extraer_huella_acustica(archivo_audio)
                
                if vector_acustico is not None:
                    vector_escalado = scaler.transform(vector_acustico)
                    predicciones = modelo.predict(vector_escalado, verbose=0)
                    
                    # --- TRADUCTOR DE IA A HUMANO ---
                    # Lista alfabética (necesaria para el LabelEncoder)
                    clases_palabras = [
                        'Autobus', 'Barco', 'Blanco', 'Bolso', 'Bufanda', 'Cara', 'Chaqueta', 'Cielo', 
                        'Clase', 'Cristal', 'Diente', 'Espada', 'Estrella', 'Flecha', 'Fruta', 'Fuego', 
                        'Globo', 'Gorro', 'Jabón', 'Lapiz', 'Libro', 'Mosca', 'Negro', 'Niño', 
                        'Peine', 'Piedra', 'Plancha', 'Rojo', 'Silla', 'Tambor', 'Taza', 'Tres'
                    ]
                    
                    idx_pal = np.argmax(predicciones[0])
                    nombre_completo = clases_palabras[idx_pal]
                    resultado_palabra = nombre_completo.split('_')[0] # Limpieza del nombre
                    confianza_palabra = np.max(predicciones[0]) * 100
                    
                    resultado_origen = ['España', 'Latinoamérica', 'No Nativo'][np.argmax(predicciones[1])]
                    resultado_sexo = 'Mujer' if predicciones[2][0][0] > 0.5 else 'Hombre'
                    
                    st.success("✅ Análisis completado.")
                    st.markdown("### 📋 Resultados del Diagnóstico Clínico")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='metric-card'><h4>Palabra</h4><h2>{resultado_palabra}</h2><p>Conf: {confianza_palabra:.1f}%</p></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-card'><h4>Región</h4><h2>{resultado_origen}</h2></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='metric-card'><h4>Sexo</h4><h2>{resultado_sexo}</h2></div>", unsafe_allow_html=True)
                else:
                    st.error("No se pudo aislar la voz. El audio es demasiado corto.")
