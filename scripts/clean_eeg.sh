#!/bin/bash

# -------------------------------------------------------
# Script: clean_eeg.sh
# Uso:
#   ./clean_eeg.sh <directorio_base> [--dry-run]
# Ejemplo:
#   ./clean_eeg.sh R1_L100_bdf
#   ./clean_eeg.sh R2_L100_bdf --dry-run
# utiliza [--dry-run] para obtener el nombre de los archivos que boraras, pero sin borrarlo.
# -------------------------------------------------------

BASE_DIR="$1"
DRY_RUN=false

# Verificar argumentos
if [ -z "$BASE_DIR" ]; then
  echo "Uso: $0 <directorio_base> [--dry-run]"
  exit 1
fi

if [ "$2" == "--dry-run" ]; then
  DRY_RUN=true
fi

# Validar directorio base
if [ ! -d "$BASE_DIR" ]; then
  echo "❌ El directorio '$BASE_DIR' no existe."
  exit 1
fi

echo "🔍 Procesando: $BASE_DIR"
$DRY_RUN && echo "💡 Modo simulación activado (no se eliminará nada)."

cd "$BASE_DIR" || exit 1

# Buscar subdirectorios que empiecen con "sub-"
for subdir in sub-*; do
  eeg_dir="$subdir/eeg"

  if [ -d "$eeg_dir" ]; then
    echo "📁 Revisando $eeg_dir"

    if $DRY_RUN; then
      # Mostrar archivos que se eliminarían
      find "$eeg_dir" -type f ! -iname "*RestingState*" -print
    else
      # Eliminar archivos que NO contengan 'RestingState'
      find "$eeg_dir" -type f ! -iname "*RestingState*" -print -delete
    fi

  else
    echo "⚠️  No se encontró el directorio 'eeg' en $subdir"
  fi
done

echo "✅ Proceso finalizado en $BASE_DIR."
