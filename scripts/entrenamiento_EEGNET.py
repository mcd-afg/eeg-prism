"""
EEGNet training script with cropped decoding using Braindecode + Skorch.

Pipeline:
1. GPU check and environment setup
2. Load preprocessed EEG data (.fif + .json)
3. Dataset construction and filtering
4. Model definition (EEGNet) and cropped setup
5. Windowing (fixed-length windows)
6. Train / validation / test split
7. Training with CosineAnnealingLR
8. Logging, saving metrics and model state
"""

# ==============================================================
# 1. GPU CHECK & BASIC ENVIRONMENT SETUP
# ==============================================================
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mne

#Nuevo
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

print(f"GPUs disponibles: {torch.cuda.device_count()}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU en uso: {torch.cuda.get_device_name(0)}")

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

# ==============================================================
# 2. LIBRERÍAS BRAINDCODE / SKORCH
# ==============================================================
from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import EEGNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss

from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, PrintLog

# ==============================================================
# 3. RANDOM SEEDS (REPRODUCIBILITY)
# ==============================================================
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

if cuda:
    torch.backends.cudnn.benchmark = True

# ==============================================================
# 4. LOAD DATASETS (.fif + .json)
# ==============================================================
data_path = Path("preprocessed_data")
mapping = {"bajo": 0, "medio": 1, "alto": 2}

datasets_list = []

for fif_path in data_path.glob("*_standardized_eeg.fif"):
    subj = fif_path.stem.split("_standardized_eeg")[0]
    raw = mne.io.read_raw_fif(fif_path, preload=False)

    with open(data_path / f"{subj}_description.json", "r") as f:
        desc = json.load(f)

    # if isinstance(desc["p_factor_category"], str):
    #     desc["p_factor_category"] = mapping[desc["p_factor_category"]]

    if not isinstance(desc["p_factor"], float):
        desc["p_factor"] = float(desc["p_factor"])

    ds = BaseDataset(
        raw=raw,
        description=desc,
        target_name="p_factor"
    )
    datasets_list.append(ds)

# Concatenate subjects
full_dataset = BaseConcatDataset(datasets_list)
print(f"Sujetos cargados: {len(full_dataset.datasets)}")

# ==============================================================
# 5. FILTER SUBJECTS BY MINIMUM RECORDING LENGTH
# ==============================================================
SFREQ = 100
MIN_SECONDS = 30

full_dataset = BaseConcatDataset([
    ds for ds in full_dataset.datasets
    if ds.raw.n_times >= MIN_SECONDS * SFREQ
])

print(f"Sujetos tras filtrado: {len(full_dataset.datasets)}")

# ==============================================================
# 6. MODEL DEFINITION (EEGNet)
# ==============================================================
n_times = 3000
n_outputs = 1
# classes = list(range(n_classes))

n_chans = full_dataset[0][0].shape[0]

model = EEGNet(
    n_chans,
    n_outputs,
    n_times=n_times,
    final_conv_length="auto",
)

if cuda:
    model.cuda()

# Convert to dense prediction (cropped decoding)
model.to_dense_prediction_model()

n_preds_per_input = model.get_output_shape()[2]
print(f"Predicciones por ventana: {n_preds_per_input}")

# ==============================================================
# 7. CREATE FIXED-LENGTH WINDOWS
# ==============================================================
windows_dataset = create_fixed_length_windows(
    full_dataset,
    window_size_samples=n_times,
    window_stride_samples=250,
    drop_last_window=False,
    preload=False,
)

# ==============================================================
# 8. TRAIN / VALID / TEST SPLIT (STRATIFIED)
# ==============================================================
from sklearn.model_selection import train_test_split

all_idx = np.arange(len(windows_dataset.datasets))
labels = windows_dataset.description["p_factor_category"].to_numpy()

# Train (80%) / Test (20%)
train_full_idx, test_idx = train_test_split(
    all_idx,
    test_size=0.2,
    random_state=42
)

# Train (80%) / Valid (20%) sobre train_full
train_idx, valid_idx = train_test_split(
    train_full_idx,
    test_size=0.2,
    random_state=42
)

train_set = BaseConcatDataset([windows_dataset.datasets[i] for i in train_idx])
valid_set = BaseConcatDataset([windows_dataset.datasets[i] for i in valid_idx])
test_set  = BaseConcatDataset([windows_dataset.datasets[i] for i in test_idx])

print(f"Train: {len(train_set)} ventanas")
print(f"Valid: {len(valid_set)} ventanas")
print(f"Test : {len(test_set)} ventanas")

# ==============================================================
# 9. TRAINING CONFIGURATION
# ==============================================================
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 200

from braindecode import EEGRegressor
from skorch.callbacks import EpochScoring
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

reg = EEGRegressor(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    train_split=predefined_split(valid_set),
    iterator_train__shuffle=True,
    #Nuevo
    iterator_train__num_workers=8,
    iterator_valid__num_workers=8,
    iterator_train__pin_memory=True,
    iterator_valid__pin_memory=True,
    #Nuevo
    batch_size=batch_size,
    callbacks=[
    EpochScoring(
        scoring=mean_squared_error,
        name="train_mse",
        on_train=True,
        lower_is_better=True,
    ),
    EpochScoring(
        scoring=mean_squared_error,
        name="valid_mse",
        on_train=False,
        lower_is_better=True,
    ),
    EpochScoring(
        scoring=mean_absolute_error,
        name="valid_mae",
        on_train=False,
        lower_is_better=True,
    ),
    ("lr_scheduler", LRScheduler(
        "CosineAnnealingLR",
        T_max=n_epochs - 1
    )),
    PrintLog(),
],
    device=device,
)

# Reduce memory spikes
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# ==============================================================
# 10. MODEL TRAINING
# ==============================================================
reg.fit(train_set, y=None, epochs=n_epochs)

# ==============================================================
# 11. SAVE TRAINING LOGS
# ==============================================================
results_columns = [
    "epoch",
    "train_loss",
    "valid_loss",
    "train_mse",
    "valid_mse",
    "train_mae",
    "valid_mae",
    "train_r2",
    "valid_r2",
    "event_lr",
]

rows = []

for row in reg.history:
    rows.append({col: row.get(col, np.nan) for col in results_columns})

df = pd.DataFrame(rows).set_index("epoch")

df.to_csv("training_log_EEGNET_regression.csv", index_label="epoch")
# ==============================================================
# 12. SAVE MODEL STATE (RESTARTABLE TRAINING)
# ==============================================================
reg.save_params(
    f_params="model_params.pt",
    f_optimizer="optimizer.pt",
    f_history="history.json",
)

print("Entrenamiento finalizado y modelo guardado.")
