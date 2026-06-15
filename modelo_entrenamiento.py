import os
import sys

# 1. Bloqueo inmediato de alertas y optimizaciones conflictivas de CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Mensaje de vida instantáneo para la terminal
print("\n🎬 Script iniciado correctamente. Cargando el motor de Inteligencia Artificial...")
print("⏳ Cargando TensorFlow y dependencias matemáticas en memoria (Espera unos 15 segundos)...")
sys.stdout.flush()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

ARCHIVO_DATOS = "datos_preparados.csv"
MODELO_SALIDA = "modelo_tdl_multicabeza.h5"

def construir_red_multicabeza(input_dim, num_palabras, num_origenes, num_sexos):
    entradas = Input(shape=(input_dim,), name='entrada_acustica')
    
    # Tronco común de la red
    x = Dense(128, activation='relu')(entradas)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Cabeza 1: Palabra
    cabeza_palabra = Dense(32, activation='relu')(x)
    salida_palabra = Dense(num_palabras, activation='softmax', name='salida_palabra')(cabeza_palabra)
    
    # Cabeza 2: Origen
    cabeza_origen = Dense(16, activation='relu')(x)
    salida_origen = Dense(num_origenes, activation='softmax', name='salida_origen')(cabeza_origen)
    
    # Cabeza 3: Sexo
    cabeza_sexo = Dense(16, activation='relu')(x)
    salida_sexo = Dense(num_sexos, activation='softmax', name='salida_sexo')(cabeza_sexo)
    
    modelo = Model(inputs=entradas, outputs=[salida_palabra, salida_origen, salida_sexo])
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'salida_palabra': 'categorical_crossentropy',
            'salida_origen': 'categorical_crossentropy',
            'salida_sexo': 'categorical_crossentropy'
        },
        loss_weights={
            'salida_palabra': 2.0,  
            'salida_origen': 1.0,
            'salida_sexo': 1.0
        },
        metrics={
            'salida_palabra': 'accuracy',
            'salida_origen': 'accuracy',
            'salida_sexo': 'accuracy'
        }
    )
    return modelo

def entrenar_y_evaluar():
    print("\n🧠 Librerías cargadas. Inicializando análisis del dataset...")
    
    try:
        df = pd.read_csv(ARCHIVO_DATOS, sep=';')
    except FileNotFoundError:
        print(f"❌ Error: No se encontró '{ARCHIVO_DATOS}'. Ejecuta el preprocesamiento primero.")
        return

    columnas_features = [col for col in df.columns if col.startswith(('mfcc', 'delta'))]
    X = df[columnas_features].values
    
    le_palabra = LabelEncoder()
    le_origen = LabelEncoder()
    le_sexo = LabelEncoder()
    
    y_palabra = to_categorical(le_palabra.fit_transform(df['palabra']))
    y_origen = to_categorical(le_origen.fit_transform(df['origen']))
    y_sexo = to_categorical(le_sexo.fit_transform(df['sexo']))
    
    X_train, X_test, yp_train, yp_test, yo_train, yo_test, ys_train, ys_test = train_test_split(
        X, y_palabra, y_origen, y_sexo, test_size=0.3, random_state=42, stratify=df['palabra']
    )
    
    modelo = construir_red_multicabeza(
        input_dim=X.shape[1], 
        num_palabras=y_palabra.shape[1], 
        num_origenes=y_origen.shape[1], 
        num_sexos=y_sexo.shape[1]
    )
    
    print("🚀 Datos listos. Iniciando épocas de entrenamiento...\n")
    sys.stdout.flush()

    historial = modelo.fit(
        X_train, 
        {'salida_palabra': yp_train, 'salida_origen': yo_train, 'salida_sexo': ys_train},
        validation_data=(X_test, {'salida_palabra': yp_test, 'salida_origen': yo_test, 'salida_sexo': ys_test}),
        epochs=80, 
        batch_size=16, 
        verbose=1 
    )
    
    modelo.save(MODELO_SALIDA)
    print(f"\n💾 Modelo exportado con éxito a '{MODELO_SALIDA}'")
    # --- AÑADE ESTO: GUARDAR LOS DICCIONARIOS ---
    joblib.dump(le_palabra, 'diccionario_palabras.pkl')
    joblib.dump(le_origen, 'diccionario_origen.pkl')
    joblib.dump(le_sexo, 'diccionario_sexo.pkl')
    print("💾 Diccionarios de traducción (LabelEncoders) guardados.")
    
    print("\n" + "="*50)
    print("📊 REPORTE DE AUDITORÍA CLÍNICA (CONJUNTO DE TEST)")
    print("="*50)
    
    predicciones = modelo.predict(X_test, verbose=0)
    pred_palabra = np.argmax(predicciones[0], axis=1)
    pred_origen = np.argmax(predicciones[1], axis=1)
    pred_sexo = np.argmax(predicciones[2], axis=1)
    
    true_palabra = np.argmax(yp_test, axis=1)
    true_origen = np.argmax(yo_test, axis=1)
    true_sexo = np.argmax(ys_test, axis=1)
    
    print("\n--- 🗣️ DIAGNÓSTICO FONOLÓGICO (Palabra) ---")
    print(classification_report(true_palabra, pred_palabra, target_names=le_palabra.classes_, zero_division=0))
    
    print("\n--- 🌍 PERFILADO DEMOGRÁFICO (Origen) ---")
    print(classification_report(true_origen, pred_origen, target_names=le_origen.classes_, zero_division=0))
    
    print("\n--- 🧬 IDENTIFICACIÓN BIOLÓGICA (Sexo) ---")
    print(classification_report(true_sexo, pred_sexo, target_names=le_sexo.classes_, zero_division=0))

    generar_graficos(historial)

def generar_graficos(historial):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(historial.history['loss'], label='Pérdida Entrenamiento', color='#1e3a8a', linewidth=2)
    plt.plot(historial.history['val_loss'], label='Pérdida Validación', color='#22c55e', linewidth=2)
    plt.title('📉 Curva de Convergencia (Pérdida Total)', fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(historial.history['salida_palabra_accuracy'], label='Accuracy Palabra (Train)', color='#1e3a8a')
    plt.plot(historial.history['val_salida_palabra_accuracy'], label='Accuracy Palabra (Val)', color='#22c55e')
    plt.title('🎯 Precisión Diagnóstica (Fonemas)', fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('convergencia_entrenamiento.png', dpi=300)
    print("\n📸 Gráfica de entrenamiento guardada como 'convergencia_entrenamiento.png'")

if __name__ == "__main__":
    entrenar_y_evaluar()
