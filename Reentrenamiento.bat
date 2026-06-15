@echo off
title Pipeline de Entrenamiento TDL
color 0A

echo ========================================================
echo      🚀 INICIANDO PIPELINE DE ENTRENAMIENTO TDL 🚀
echo ========================================================
echo.

echo [1/4] Ejecutando Matcher Inteligente y Etiquetado...
python etiquetado.py
echo.

echo [2/4] Extrayendo caracteristicas acusticas (MFCCs)...
python procesar_audios.py
echo.

echo [3/4] Normalizando matriz de datos (Z-Score)...
python preprocesamiento_datos.py
echo.

echo [4/4] Entrenando Red Neuronal Multicabeza...
python modelo_entrenamiento.py
echo.

echo ========================================================
echo   ✅ ENTRENAMIENTO COMPLETADO. YA PUEDES LANZAR LA APP
echo ========================================================
pause