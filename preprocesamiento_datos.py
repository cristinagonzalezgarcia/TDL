import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

ARCHIVO_ENTRADA = "dataset_procesado.csv"
ARCHIVO_ETIQUETAS = "dataset_etiquetado.csv" # El CSV que cruza id_audio con [Palabra, Origen, Sexo]
ARCHIVO_SALIDA = "datos_preparados.csv"
SCALER_PATH = "scaler_acustico.pkl"

def preprocesar_y_fusionar():
    print("⚙️ Iniciando normalización del dataset...")
    
    try:
        # 1. Cargar las características acústicas
        df_features = pd.read_csv(ARCHIVO_ENTRADA, sep=';')
        
        # 2. Cargar las etiquetas clínicas/demográficas (Asegúrate de que este archivo existe)
        # Debe tener columnas: id_audio, palabra, origen, sexo
        df_labels = pd.read_csv(ARCHIVO_ETIQUETAS, sep=';')
        
        # 3. Fusionar datos (Inner join para asegurar que todo audio tiene su etiqueta)
        df_final = pd.merge(df_features, df_labels, on="id_audio", how="inner")
        
        if df_final.empty:
            print("❌ Error: La fusión resultó en un dataset vacío. Revisa que los 'id_audio' coincidan en ambos CSV.")
            return

        # 4. Separar X (Características) de Y (Etiquetas) y variables ID
        columnas_features = [col for col in df_final.columns if col.startswith(('mfcc', 'delta'))]
        X = df_final[columnas_features]
        
        # 5. Estandarización Z-Score (Media=0, Varianza=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Guardar el scaler para usarlo en producción (la App móvil necesitará escalar los audios nuevos igual)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✓ Objeto Scaler guardado en '{SCALER_PATH}'")
        
        # 6. Reconstruir el DataFrame final limpio
        df_scaled = pd.DataFrame(X_scaled, columns=columnas_features)
        
        # Añadimos de vuelta el ID y las etiquetas
        df_scaled['id_audio'] = df_final['id_audio']
        df_scaled['palabra'] = df_final['palabra']
        df_scaled['origen'] = df_final['origen']
        df_scaled['sexo'] = df_final['sexo']
        
        # Reordenar para que las etiquetas estén al principio
        columnas_ordenadas = ['id_audio', 'palabra', 'origen', 'sexo'] + columnas_features
        df_scaled = df_scaled[columnas_ordenadas]
        
        # 7. Exportar
        df_scaled.to_csv(ARCHIVO_SALIDA, index=False, sep=';')
        print(f"✅ Preprocesamiento completado. Matriz limpia y normalizada guardada en '{ARCHIVO_SALIDA}'. Forma: {df_scaled.shape}")

    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}. Ejecuta primero procesar_audios.py y asegúrate de tener el archivo de etiquetas.")
    except Exception as e:
        print(f"❌ Error inesperado durante el preprocesamiento: {e}")

if __name__ == "__main__":
    preprocesar_y_fusionar()
