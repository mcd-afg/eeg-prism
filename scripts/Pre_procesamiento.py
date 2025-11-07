# ==========================================================
# 🧠 PREPROCESAMIENTO DE EEG PARA EEG CHALLENGE
# ==========================================================

# === Importaciones principales ===
import os
import math
import random
import json
import pickle
from pathlib import Path
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss

import mne
import pandas as pd
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
    create_fixed_length_windows,
)
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNeX
from eegdash import EEGChallengeDataset


# ==========================================================
# ⚙️ CONFIGURACIÓN INICIAL
# ==========================================================

print(f"GPUs disponibles: {torch.cuda.device_count()}")
print(f"Está disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nombre: {torch.cuda.get_device_name(0)}")


# ==========================================================
# 📦 CARGA DE DATOS BASE
# ==========================================================

# Lista de sujetos con advertencia ("caution")
with open("nombres_caution.pkl", "rb") as f:
    caution_subjects = pickle.load(f)

# Datasets originales
with open("all_datasets.pkl", "rb") as f:
    all_datasets = pickle.load(f)


# ==========================================================
# 🧩 CLASE AUXILIAR PARA DATOS CON CROP ALEATORIO
# ==========================================================

class DatasetWrapper(BaseDataset):
    def __init__(
        self,
        dataset: EEGWindowsDataset,
        crop_size_samples: int,
        target_name: str = "p_factor_category",
        seed=None,
    ):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.target_name = target_name
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        target = float(self.dataset.description[self.target_name])

        # Información adicional
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", ""),
            "run": self.dataset.description.get("run", ""),
        }

        # Recorte aleatorio de señal
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start += start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, (i_window_in_trial, i_start, i_stop), infos


# ==========================================================
# 🧹 FILTRADO DE DATASETS
# ==========================================================

SFREQ = 100  # Hz

all_datasets = BaseConcatDataset(
    [
        ds
        for ds in all_datasets.datasets
        if not ds.description.subject in caution_subjects
        and ds.raw.n_times >= 4 * SFREQ
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description["p_factor"])
    ]
)


# ==========================================================
# 🔄 PREPROCESAMIENTO Y GUARDADO INDIVIDUAL
# ==========================================================

OUTPUT_DIR = Path("preprocessed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

for count, ds in enumerate(all_datasets.datasets, start=1):
    # if count > 10:
    #     break  # Limitar a 10 sujetos para pruebas rápidas

    # Eliminar canal Cz
    if "Cz" in ds.raw.info["ch_names"]:
        ds.raw.drop_channels(["Cz"])

    # Estandarización exponencial
    data = ds.raw.get_data()
    data_std = exponential_moving_standardize(data, factor_new=0.001, init_block_size=1000)
    ds.raw._data = data_std  # reemplaza los datos originales

    # Guardar señal preprocesada
    subj_id = ds.description.subject
    ds.raw.save(OUTPUT_DIR / f"{subj_id}_standardized_eeg.fif", overwrite=True)

    # === Categorización del p-factor ===
    p_factor_value = ds.description.p_factor
    if p_factor_value < 0.44:
        p_factor_category = "bajo"
    elif p_factor_value < 0.946:
        p_factor_category = "medio"
    else:
        p_factor_category = "alto"

    # Agregar nueva categoría
    ds.description["p_factor_category"] = p_factor_category

    # Guardar descripción en JSON
    with open(OUTPUT_DIR / f"{subj_id}_description.json", "w") as f:
        json.dump(ds.description.to_dict(), f)


# ==========================================================
# 📚 RECONSTRUCCIÓN DEL CONJUNTO PREPROCESADO
# ==========================================================

datasets_list = []

for fif_path in OUTPUT_DIR.glob("*_standardized_eeg.fif"):
    subj_id = fif_path.stem.split("_standardized_eeg")[0]
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    with open(OUTPUT_DIR / f"{subj_id}_description.json", "r") as f:
        description = json.load(f)
    ds = BaseDataset(raw=raw, description=description)
    datasets_list.append(ds)

all_datasets_preproc = BaseConcatDataset(datasets_list)

#Imprimo el número de sujetos preprocesados
print(f"Número de sujetos preprocesados: {len(all_datasets_preproc.datasets)}")

#Guardo el conjunto preprocesado completo
with open("all_datasets_preproc.pkl", "wb") as f:
    pickle.dump(all_datasets_preproc, f)

# ==========================================================
# 🪟 CREACIÓN DE VENTANAS Y WRAP FINAL
# ==========================================================

windows_ds = create_fixed_length_windows(
    all_datasets_preproc,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)

# Envolver cada dataset con el wrapper definido
windows_ds = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_ds.datasets]
)

#Guardo el conjunto de ventanas
with open("windows_ds.pkl", "wb") as f:
    pickle.dump(windows_ds, f)