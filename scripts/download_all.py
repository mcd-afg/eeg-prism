# ==========================================================
# 🔧 CONFIGURACIÓN INICIAL
# ==========================================================
import os
import gc
import math
import random
import pickle
import subprocess
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

# ==========================================================
# ⚙️ LIBRERÍAS DE DEEP LEARNING Y EEG
# ==========================================================
import torch
from torch import optim
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader

from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import (
    EEGWindowsDataset,
    BaseConcatDataset,
    BaseDataset
)
from braindecode.models import EEGNeX
from eegdash import EEGChallengeDataset

# ==========================================================
# 💻 CONFIGURACIÓN DE DISPOSITIVO (CPU / GPU)
# ==========================================================
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"

print(f"GPUs disponibles: {torch.cuda.device_count()}")
print(f"CUDA disponible: {cuda_available}")

if cuda_available:
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    # Permite a cuDNN seleccionar la mejor configuración para tu hardware
    torch.backends.cudnn.benchmark = True

print(f"Usando dispositivo: {device}")

# ==========================================================
# 📂 CARGA DE DATOS Y ARCHIVOS AUXILIARES
# ==========================================================
# Cargar lista de sujetos "caution" para excluir del procesamiento
with open("nombres_caution.pkl", "rb") as f:
    caution_subjects = pickle.load(f)

print("Sujetos marcados como 'caution':")
print(caution_subjects)

"""Descargo las bases de datos"""
# ====== CONFIGURACIÓN GENERAL ======
DATA_DIR = Path("mne_data/eeg2025_competition")
DATA_DIR.mkdir(parents=True, exist_ok=True)

release_list = [f"R{i}" for i in range(1, 12)]
print(f"Releases a procesar: {release_list}")

# Lista para guardar los datasets (solo los metadatos)
all_datasets_list = []

# 🔒 Lista para registrar carpetas ya procesadas
used_release_dirs = set()

# ====== PROCESAMIENTO POR RELEASE ======
for release in release_list:
    print(f"\n==============================")
    print(f"🔽 Procesando release: {release}")
    print(f"==============================")

    # --- 1. Descargar metadatos y raws ---
    dataset = EEGChallengeDataset(
        release=release,
        task="RestingState",
        mini=False,
        description_fields=["subject", "task", "age", "sex", "p_factor"],
        cache_dir=DATA_DIR,
    )
    print(f"✅ Dataset {release} cargado con {len(dataset.datasets)} sujetos.")

    # --- 2. Descargar los raws (para caché local, no en RAM persistente) ---
    _ = Parallel(n_jobs=os.cpu_count())(
        delayed(lambda d: d.raw)(d) for d in dataset.datasets
    )
    print(f"📦 Raws descargados para {release}.")

    # --- 3. Buscar la carpeta correspondiente a este release ---
    release_dirs = [p for p in DATA_DIR.glob("ds*-bdf") if p not in used_release_dirs]

    if len(release_dirs) > 0:
        # Tomar la primera carpeta no usada
        release_path = release_dirs[0]
        print(f"🧹 Limpiando archivos no 'RestingState' en {release_path}")

        subprocess.run(["python3", "clean_eeg.py", str(release_path)], check=False)

        # Marcar esta carpeta como usada
        used_release_dirs.add(release_path)
    else:
        print(f"⚠️ No se encontró un nuevo directorio disponible para {release}")

    # --- 4. Guardar metadatos del dataset (sin raws en memoria) ---
    all_datasets_list.append(dataset)

    # --- 5. Liberar memoria pesada ---
    del _
    gc.collect()
    print(f"🧠 Memoria liberada tras procesar {release}\n")

# ====== CONCATENAR TODOS LOS DATASETS ======
print("🔗 Concatenando todos los datasets...")
all_datasets = BaseConcatDataset(all_datasets_list)
print(f"✅ all_datasets creado con {len(all_datasets.datasets)} sujetos totales.")
