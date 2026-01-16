# ==============================================================
# EVALUACIÓN Y MATRIZ DE CONFUSIÓN - MODELO EEGNET
# ==============================================================
# Este script evalúa un modelo EEGNet previamente entrenado,
# genera la matriz de confusión y guarda las probabilidades
# predichas para cada ventana del conjunto de validación.
# ==============================================================

# --------------------------------------------------------------
# 1. IMPORTAR LIBRERÍAS
# --------------------------------------------------------------
import numpy as np
import os
import json
from pathlib import Path
import pandas as pd
import torch
import mne
from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.util import set_random_seeds
from braindecode.models import EEGNet
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

# --------------------------------------------------------------
# 2. CONFIGURACIÓN DE PYTORCH Y DISPOSITIVO
# --------------------------------------------------------------
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

print(f"GPUs disponibles: {torch.cuda.device_count()}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU en uso: {torch.cuda.get_device_name(0)}")

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

# --------------------------------------------------------------
# 3. REPRODUCIBILIDAD (RANDOM SEEDS)
# --------------------------------------------------------------
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

if cuda:
    torch.backends.cudnn.benchmark = True

# --------------------------------------------------------------
# 4. CARGAR DATOS PREPROCESADOS (.fif + .json)
# --------------------------------------------------------------
data_path = Path("preprocessed_data")
mapping = {"bajo": 0, "medio": 1, "alto": 2}

datasets_list = []

# Iterar sobre archivos .fif
for fif_path in data_path.glob("*_standardized_eeg.fif"):
    subj = fif_path.stem.split("_standardized_eeg")[0]
    
    # Cargar datos EEG
    raw = mne.io.read_raw_fif(fif_path, preload=False)
    
    # Cargar metadata del sujeto
    with open(data_path / f"{subj}_description.json", "r") as f:
        desc = json.load(f)
    
    # Convertir categoría a numérico
    if isinstance(desc["p_factor_category"], str):
        desc["p_factor_category"] = mapping[desc["p_factor_category"]]
    
    # Crear dataset individual
    ds = BaseDataset(
        raw=raw,
        description=desc,
        target_name="p_factor_category"
    )
    datasets_list.append(ds)

# Concatenar todos los sujetos
full_dataset = BaseConcatDataset(datasets_list)
print(f"Sujetos cargados: {len(full_dataset.datasets)}")

# --------------------------------------------------------------
# 5. FILTRAR SUJETOS POR DURACIÓN MÍNIMA DE GRABACIÓN
# --------------------------------------------------------------
SFREQ = 100
MIN_SECONDS = 30

full_dataset = BaseConcatDataset([
    ds for ds in full_dataset.datasets
    if ds.raw.n_times >= MIN_SECONDS * SFREQ
])

print(f"Sujetos tras filtrado: {len(full_dataset.datasets)}")

# --------------------------------------------------------------
# 6. DEFINICIÓN DEL MODELO (EEGNet)
# --------------------------------------------------------------
n_times = 3000  # Tamaño de ventana en muestras
n_classes = 3   # bajo, medio, alto
classes = list(range(n_classes))

# Número de canales EEG
n_chans = full_dataset[0][0].shape[0]

# Crear modelo EEGNet
model = EEGNet(
    n_chans,
    n_classes,
    n_times=n_times,
    final_conv_length="auto",
)

if cuda:
    model.cuda()

# Convertir a modelo de predicción densa (cropped decoding)
model.to_dense_prediction_model()

n_preds_per_input = model.get_output_shape()[2]
print(f"Predicciones por ventana: {n_preds_per_input}")

# --------------------------------------------------------------
# 7. CREAR VENTANAS DE LONGITUD FIJA
# --------------------------------------------------------------
windows_dataset = create_fixed_length_windows(
    full_dataset,
    window_size_samples=n_times,
    window_stride_samples=250,
    drop_last_window=False,
    preload=False,
)

# --------------------------------------------------------------
# 8. DIVISIÓN TRAIN / VALID / TEST (ESTRATIFICADA)
# --------------------------------------------------------------
all_idx = np.arange(len(windows_dataset.datasets))
labels = windows_dataset.description["p_factor_category"].to_numpy()

# Train (80%) / Test (20%)
train_full_idx, test_idx = train_test_split(
    all_idx,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Train (64%) / Valid (16%) sobre el conjunto completo
train_idx, valid_idx = train_test_split(
    train_full_idx,
    test_size=0.2,
    stratify=labels[train_full_idx],
    random_state=42
)

# Crear conjuntos de datos
train_set = BaseConcatDataset([windows_dataset.datasets[i] for i in train_idx])
valid_set = BaseConcatDataset([windows_dataset.datasets[i] for i in valid_idx])
test_set = BaseConcatDataset([windows_dataset.datasets[i] for i in test_idx])

print(f"Train: {len(train_set)} ventanas")
print(f"Valid: {len(valid_set)} ventanas")
print(f"Test : {len(test_set)} ventanas")

# --------------------------------------------------------------
# 9. CONFIGURACIÓN DEL CLASIFICADOR
# --------------------------------------------------------------
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 200

# ==============================================================
# 9. CALCULAR CLASS WEIGHTS (INVERSAMENTE PROPORCIONAL)
# ==============================================================
from sklearn.utils.class_weight import compute_class_weight

# Extraer labels del conjunto de entrenamiento
train_labels = windows_dataset.description.iloc[train_idx]["p_factor_category"].to_numpy()

# Calcular pesos inversos
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Convertir a tensor y mover al device
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

def weighted_cross_entropy(preds, targets):
    """Cross entropy con class weights para clases desbalanceadas"""
    return torch.nn.functional.cross_entropy(preds, targets, weight=class_weights_tensor)

# Crear clasificador EEG
clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=weighted_cross_entropy,
    optimizer=torch.optim.AdamW,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    train_split=None,  # No usar validación interna
    batch_size=batch_size,
    device=device,
    classes=classes,
)

# Inicializar y cargar parámetros previamente entrenados
clf.initialize()
clf.load_params(
    f_params="model_params.pt",
    f_optimizer="optimizer.pt",
    f_history="history.json",
)

print("Modelo cargado exitosamente para evaluación.")

# ==============================================================
# 10. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
# ==============================================================

# Obtener etiquetas verdaderas y predicciones
y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

# Generar matriz de confusión
confusion_mat = confusion_matrix(y_true, y_pred)

# Visualizar matriz de confusión
import matplotlib.pyplot as plt

plot_confusion_matrix(confusion_mat, class_names=['bajo', 'medio', 'alto'])
plt.savefig('Matriz_confusion_ventanas.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Matriz de confusión guardada en 'Matriz_confusion_ventanas.png'")


# Inspeccionar columnas de metadata
meta = valid_set.get_metadata()
print("\nColumnas de metadata:")
print(meta.columns)

# --------------------------------------------------------------
# 11. GUARDAR PROBABILIDADES PREDICHAS
# --------------------------------------------------------------
# Obtener metadata limpia
meta = valid_set.get_metadata().reset_index(drop=True)

# Nombres de clases
classes_names = ['bajo', 'medio', 'alto']

# Obtener probabilidades predichas
y_proba = clf.predict_proba(valid_set)

# Crear DataFrame con resultados
proba_df = pd.DataFrame(y_proba, columns=classes_names)
proba_df["subject"] = meta["subject"]
proba_df["y_true"] = meta["target"]
proba_df["y_pred"] = y_pred

# Guardar a CSV
proba_df.to_csv("valid_proba_df_weights.csv", index=False)

print("\n✓ Probabilidades guardadas en 'valid_proba_df_weights.csv'")
print(f"✓ Total de predicciones: {len(proba_df)}")


