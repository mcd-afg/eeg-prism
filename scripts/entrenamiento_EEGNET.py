# ==============================================================
# 🧠 ENTRENAMIENTO DE EEGNet EN EL EEGChallengeDataset
# ==============================================================

# === 1️⃣ LIBRERÍAS E IMPORTS ===
import os
import math
import random
import json
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import mne
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import l1_loss

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
from braindecode.models import EEGNet, EEGNeX
from eegdash import EEGChallengeDataset

import matplotlib.pyplot as plt


# ==============================================================
# ⚙️ CONFIGURACIÓN DE DISPOSITIVO Y CUDA
# ==============================================================

print(f"GPUs disponibles: {torch.cuda.device_count()}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # mejora eficiencia al fijar tamaños


# ==============================================================
# 📦 CLASE WRAPPER DE DATASET
# ==============================================================

class DatasetWrapper(BaseDataset):
    """
    Envuelve un EEGWindowsDataset y aplica un recorte aleatorio
    para aumentar la variabilidad temporal.
    """
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
        target = int(self.dataset.description[self.target_name])

        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", ""),
            "run": self.dataset.description.get("run", ""),
        }

        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"

        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start += start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        return X, target, (i_window_in_trial, i_start, i_stop), infos


# ==============================================================
# 📁 CARGA DE ARCHIVOS Y CREACIÓN DE DATASETS
# ==============================================================

data_path = Path("preprocessed_data")
OUTPUT_DIR = Path("preprocessed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mostrar número de archivos disponibles
all_files = list(data_path.glob("*.fif"))
print(f"Número de archivos .fif en 'preprocessed_data': {len(all_files)}")

# Mapeo de categorías de texto a enteros
mapping = {"bajo": 0, "medio": 1, "alto": 2}

datasets_list = []
SFREQ = 100  # Hz

for fif_path in OUTPUT_DIR.glob("*_standardized_eeg.fif"):
    subj_id = fif_path.stem.split("_standardized_eeg")[0]
    raw = mne.io.read_raw_fif(fif_path, preload=False)

    with open(OUTPUT_DIR / f"{subj_id}_description.json", "r") as f:
        description = json.load(f)

    cat = description.get("p_factor_category", None)
    if isinstance(cat, str):
        description["p_factor_category"] = mapping[cat]

    ds = BaseDataset(raw=raw, description=description)
    datasets_list.append(ds)

all_datasets_preproc = BaseConcatDataset(datasets_list)


# ==============================================================
# ✂️ DIVISIÓN DE DATOS (TRAIN / VALID / TEST)
# ==============================================================

descriptions = all_datasets_preproc.description
p_factors = descriptions["p_factor_category"]

train_idx, test_idx = train_test_split(
    range(len(all_datasets_preproc.datasets)),
    test_size=0.2,
    stratify=p_factors,
    random_state=42,
)

train_set = BaseConcatDataset([all_datasets_preproc.datasets[i] for i in train_idx])
test_set = BaseConcatDataset([all_datasets_preproc.datasets[i] for i in test_idx])

print(f"Train: {len(train_set.datasets)} sujetos")
print(f"Test:  {len(test_set.datasets)} sujetos")

# --- División interna de train en train/valid ---
p_factors_train = train_set.description["p_factor_category"]
train_idx_inner, valid_idx = train_test_split(
    range(len(train_set.datasets)),
    test_size=0.2,
    stratify=p_factors_train,
    random_state=42,
)

train_set_inner = BaseConcatDataset([train_set.datasets[i] for i in train_idx_inner])
valid_set = BaseConcatDataset([train_set.datasets[i] for i in valid_idx])

print(f"Train interno: {len(train_set_inner.datasets)} sujetos")
print(f"Validación: {len(valid_set.datasets)} sujetos")


# ==============================================================
# 🪟 CREACIÓN DE VENTANAS TEMPORALES
# ==============================================================

windows_train = create_fixed_length_windows(
    train_set_inner,
    window_size_samples=4 * SFREQ,   # 4s
    window_stride_samples=2 * SFREQ, # 2s
    drop_last_window=True,
)
windows_valid = create_fixed_length_windows(
    valid_set,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)
windows_test = create_fixed_length_windows(
    test_set,
    window_size_samples=4 * SFREQ,
    window_stride_samples=2 * SFREQ,
    drop_last_window=True,
)

# --- Aplicar el DatasetWrapper para recorte aleatorio ---
windows_train = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_train.datasets]
)
windows_valid = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_valid.datasets]
)
windows_test = BaseConcatDataset(
    [DatasetWrapper(ds, crop_size_samples=2 * SFREQ) for ds in windows_test.datasets]
)


# ==============================================================
# 📤 DATALOADERS
# ==============================================================

batch_size = 512
num_workers = 4

train_loader = DataLoader(
    windows_train, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True
)
valid_loader = DataLoader(
    windows_valid, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)
test_loader = DataLoader(
    windows_test, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)


# ==============================================================
# 🧩 MODELO, OPTIMIZADOR Y FUNCIÓN DE PÉRDIDA
# ==============================================================

n_chans = 128
n_outputs = 3   # bajo, medio, alto

model = EEGNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_times=2 * SFREQ
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()


# ==============================================================
# 🚀 ENTRENAMIENTO DEL MODELO
# ==============================================================

n_epochs = 200
history = []

for epoch in range(n_epochs):
    # --- MODO TRAIN ---
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch_idx, batch in enumerate(train_loader):
        X, y, _, _ = batch
        X = X.to(torch.float32).to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, preds = y_pred.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    # --- MODO VALIDACIÓN ---
    model.eval()
    valid_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y, *_ in valid_loader:
            X = X.to(torch.float32).to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            valid_loss += loss.item() * X.size(0)
            _, preds = y_pred.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_valid_loss = valid_loss / len(valid_loader.dataset)
    valid_acc = correct / total

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "valid_loss": avg_valid_loss,
        "train_accuracy": train_acc,
        "valid_accuracy": valid_acc,
    })

    print(f"Epoch {epoch+1:02d} | Train Acc={train_acc:.3f} | Val Acc={valid_acc:.3f} | "
          f"Train Loss={avg_train_loss:.4f} | Val Loss={avg_valid_loss:.4f}")

# --- Guardar historia ---
history_df = pd.DataFrame(history)
torch.save(model.state_dict(), "eegnet_model.pth")
history_df.to_csv("training_history.csv", index=False)
