import os
import pandas as pd
from pathlib import Path

ARCHIVO_SALIDA = "dataset_etiquetado.csv"
DIRECTORIO_AUDIOS = "./audios"

def generar_etiquetas_emergencia():
    print("🚨 Iniciando MODO EMERGENCIA: Búsqueda recursiva en subcarpetas...")
    
    ruta_base = Path(DIRECTORIO_AUDIOS)
    if not ruta_base.exists():
        print(f"❌ ERROR: No existe la carpeta '{DIRECTORIO_AUDIOS}'.")
        return
        
    # Buscar recursivamente TODOS los .wav y .mp3 sin importar lo profundo que estén
    archivos_reales = list(ruta_base.rglob("*.wav")) + list(ruta_base.rglob("*.mp3")) + \
                      list(ruta_base.rglob("*.WAV")) + list(ruta_base.rglob("*.MP3"))
    
    if not archivos_reales:
        print("❌ ERROR: No se ha encontrado ningún .wav/.mp3 ni en la carpeta ni en las subcarpetas.")
        return
        
    datos = []
    
    for archivo_path in archivos_reales:
        # Guardamos la ruta relativa (ej: 'subcarpeta/audio.wav') para que el siguiente script sepa dónde ir
        ruta_relativa = str(archivo_path.relative_to(ruta_base)).replace('\\', '/')
        
        # Analizamos solo el nombre final para deducir la etiqueta
        nombre_base = archivo_path.stem.lower()
        partes = nombre_base.split('_')
        
        palabra = partes[0].strip()
        origen = "España" 
        sexo = "Mujer"
        
        if len(partes) >= 2:
            if "lat" in partes[1]: origen = "Latinoamérica"
            elif "no" in partes[1] or "nat" in partes[1]: origen = "No nativo"
            
        if len(partes) >= 3:
            if partes[2] in ['h', 'hombre', 'm', 'male']: sexo = "Hombre"
            
        datos.append([ruta_relativa, palabra, origen, sexo])
        
    df = pd.DataFrame(datos, columns=['id_audio', 'palabra', 'origen', 'sexo'])
    df.to_csv(ARCHIVO_SALIDA, index=False, sep=';', encoding='utf-8')
    print(f"✅ ¡Hackathon salvado! Archivo '{ARCHIVO_SALIDA}' creado.")
    print(f"💾 Se han encontrado y etiquetado {len(df)} audios escondidos en las subcarpetas.")

if __name__ == "__main__":
    generar_etiquetas_emergencia()
