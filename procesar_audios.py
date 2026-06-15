import os
import librosa
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

DIRECTORIO_AUDIOS = "./audios"
ARCHIVO_ETIQUETAS = "dataset_etiquetado.csv"
ARCHIVO_SALIDA = "dataset_procesado.csv"

def extraer_caracteristicas(ruta_archivo):
    try:
        audio, sr = librosa.load(ruta_archivo, sr=16000)
        audio_limpio, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio_limpio) == 0: return None
        
        mfccs = librosa.feature.mfcc(y=audio_limpio, sr=sr, n_mfcc=20)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        return np.hstack([np.mean(mfccs.T, axis=0), np.mean(delta_mfccs.T, axis=0), np.mean(delta2_mfccs.T, axis=0)])
    except Exception as e:
        print(f"❌ Error al procesar {ruta_archivo}: {e}")
        return None

def procesar_desde_etiquetas():
    print("🎙️ Iniciando extracción acústica inteligente...")
    try:
        df_etiquetas = pd.read_csv(ARCHIVO_ETIQUETAS, sep=';')
    except FileNotFoundError:
        print(f"❌ Falta el archivo '{ARCHIVO_ETIQUETAS}'. Ejecuta etiquetado.py primero.")
        return

    datos = []
    # Usamos las rutas exactas del CSV como un mapa del tesoro
    for _, fila in df_etiquetas.iterrows():
        ruta_relativa = fila['id_audio']
        ruta_completa = os.path.join(DIRECTORIO_AUDIOS, ruta_relativa)
        
        features = extraer_caracteristicas(ruta_completa)
        if features is not None:
            fila_datos = [ruta_relativa] + features.tolist()
            datos.append(fila_datos)
            print(f"✓ Procesado: {ruta_relativa}")

    columnas = ["id_audio"] + [f"mfcc_{i}" for i in range(1, 21)] + \
               [f"delta_{i}" for i in range(1, 21)] + \
               [f"delta2_{i}" for i in range(1, 21)]
               
    df_procesado = pd.DataFrame(datos, columns=columnas)
    df_procesado.to_csv(ARCHIVO_SALIDA, index=False, sep=';')
    print(f"\n✅ Extracción completada. Guardado en '{ARCHIVO_SALIDA}' con forma {df_procesado.shape}.")

if __name__ == "__main__":
    procesar_desde_etiquetas()
