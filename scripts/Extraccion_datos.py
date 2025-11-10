from pathlib import Path
from eegdash import EEGChallengeDataset
import pandas as pd

"""Extraigo los datos que acompañan la señal de cada sujeto para hacer análisis exploratorio de estos"""

# Carpeta local para cachear metadatos (no descarga señales EEG)
DATA_DIR = Path("mne_data/eeg2025_competition")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Lista de releases disponibles (según la documentación oficial)
release_list = [f"R{i}" for i in range(1, 12)]  # R1 a R11

# Lista para almacenar conteos
release_info = []

#Creo el df vacío que voy a llenar con la info de todos los releases
all_dfs = []

for release in release_list:
    try:
        dataset = EEGChallengeDataset(
            release=release,
            task="RestingState",  # o el task que desees explorar
            mini=False,           # incluye todos los sujetos del release
            description_fields=["subject"],
            cache_dir=DATA_DIR
        )

        subjects = dataset.description["subject"].unique()
        release_info.append({
            "release": release,
            "n_subjects": len(subjects)
        })

        datos_release = dataset.description
        datos_release["release"] = release
        all_dfs.append(datos_release)

        print(f"{release}: {len(subjects)} sujetos")

    except Exception as e:
        print(f"Error en {release}: {e}")

#Concateno la lista de dfs que tengo
datos_todos = pd.concat(all_dfs, ignore_index=True)

# Convertir a DataFrame
df_releases = pd.DataFrame(release_info)
print("\nResumen:")
print(df_releases)

# Guardar resultados
df_releases.to_csv("subjects_per_release.csv", index=False)
print("\nArchivo guardado como 'subjects_per_release.csv'")

#Guardo los datos de todos
datos_todos.to_csv("Info_sujetos_challenge.csv", index=False)
