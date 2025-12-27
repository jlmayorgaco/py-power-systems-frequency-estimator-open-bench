#!/bin/bash

# Nombre del archivo de salida
OUTPUT="arquitectura_proyecto_pmu.txt"

echo "Generando reporte de arquitectura para auditoría Q1..." > $OUTPUT
echo "==========================================================" >> $OUTPUT
echo "PROYECTO: Benchmarking Dynamic Frequency Estimators" >> $OUTPUT
echo "FECHA: $(date)" >> $OUTPUT
echo "==========================================================" >> $OUTPUT
echo "" >> $OUTPUT

# 1. Mostrar la estructura de carpetas (ignorando lo innecesario)
echo "ESTRUCTURA DE DIRECTORIOS:" >> $OUTPUT
if command -v tree >/dev/null 2>&1; then
    tree -I "__pycache__|.git|results*|figures*|*.pyc" >> $OUTPUT
else
    find . -maxdepth 3 -not -path '*/.*' | grep -v "__pycache__" >> $OUTPUT
fi
echo "" >> $OUTPUT
echo "==========================================================" >> $OUTPUT
echo "CONTENIDO DE ARCHIVOS TÉCNICOS:" >> $OUTPUT
echo "==========================================================" >> $OUTPUT

# 2. Iterar sobre archivos .py y .json de configuración
# Excluimos carpetas de resultados para no saturar el archivo
find . -type f \( -name "*.py" -o -name "config.json" \) \
    -not -path "*/.*" \
    -not -path "*__pycache__*" \
    -not -path "*results*" \
    -not -path "*figures*" | while read -r file; do
    
    echo "" >> $OUTPUT
    echo "----------------------------------------------------------" >> $OUTPUT
    echo "ARCHIVO: $file" >> $OUTPUT
    echo "----------------------------------------------------------" >> $OUTPUT
    cat "$file" >> $OUTPUT
    echo "" >> $OUTPUT
done

echo "Reporte finalizado en $OUTPUT"