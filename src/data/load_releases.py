from pathlib import Path
from typing import List
from eegdash.dataset import EEGChallengeDataset # Asegúrate de que esta importación sea correcta
from braindecode.datasets.base import BaseConcatDataset


def create_ccd_datasets(cache_path: Path, release_list: List[str], mini: bool) -> List[EEGChallengeDataset]:
    """
    Crea una lista de objetos EEGChallengeDataset para la tarea 'RestingState'.

    Esta función también se asegura de que el directorio de caché (cache_path) exista
    antes de intentar crear los datasets.

    Args:
        cache_path: Un objeto Path que apunta al directorio donde se guardará la caché.
        release_list: Una lista de strings, donde cada string es el nombre de un release (ej. "R1", "R2").

    Returns:
        Una lista de objetos EEGChallengeDataset inicializados.
    """
    
    # 1. Asegurar que el directorio de caché exista
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Definir los campos específicos para esta tarea
    # Los mantenemos dentro de la función ya que son específicos de "ccd"
    description_fields = [
        "subject",
        "session",
        "run",
        "task",
        "age",
        "gender",
        "sex",
        "p_factor",
    ]

# 3. Crear una lista vacía para almacenar los resultados
    datasets_list = []
    
    print(f"Iniciando la carga de {len(release_list)} releases...")

    # 4. Iterar sobre la lista de releases con un bucle 'for'
    for release in release_list:
        
        # Crear la instancia del dataset
        dataset_obj = EEGChallengeDataset(
            release=release,
            task="RestingState",
            mini=mini,
            description_fields=description_fields,
            cache_dir=cache_path,
        )
        
        # Añadir la instancia a nuestra lista
        datasets_list.append(dataset_obj)
        
        # ¡Aquí está el print que solicitaste!
        print(f"  ✅ Cargado exitosamente: {release}")
            
    # 5. Devolver la lista completa
    print("Carga de todos los releases completada.")
    return datasets_list