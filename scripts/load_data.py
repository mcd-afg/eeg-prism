from pathlib import Path
from src.data.load_releases import create_ccd_datasets


# 1. Define tus entradas
DATA_DIR = Path("data")

# Crea carpeeta si es que no existe
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Modifica RELEASE para determinar cuales releases cargaras
#RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
RELEASES = ["R1", "R2"]

# 3. Llama a la función
dataset_ccd = create_ccd_datasets(cache_path=DATA_DIR, release_list=RELEASES, mini = False)
