import os
from pathlib import Path
from joblib import Parallel, delayed
from src.data.load_releases import create_ccd_datasets, load_raw_verbose


# 1. Define tus entradas
DATA_DIR = Path("data")

# Modifica RELEASE para determinar cuales releases cargaras
#RELEASES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
RELEASES = ["R1", "R2"]

# 3. Llama a la función
all_datasets = create_ccd_datasets(cache_path=DATA_DIR, release_list=RELEASES, mini = True)

print("-----------------------------------------------------------------")
print("-----------------------------------------------------------------")
print(".........Descargando releases.........")

Parallel(n_jobs=os.cpu_count(), verbose=30)(
    delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
)

# Parallel(n_jobs=os.cpu_count())(
#     # En lugar de 'lambda', llamamos a nuestra nueva función
#     delayed(load_raw_verbose)(d) for d in all_datasets.datasets
# )
print("Script finalizado. Releases descargados exitosamente!")