import os
import pickle

# Cargar la lista de sujetos a eliminar
with open("/Users/rocioperez/Documents/Projects/AFG/Git_Group_Project/eeg-prism/data/subjects/nombres_caution.pkl", "rb") as f:
    caution_subjects = pickle.load(f)

# Directorios principales a revisar
# directories = ["ds005505-bdf-mini", "ds005506-bdf-mini", "ds005507-bdf-mini", "ds005507-bdf-mini", "ds005508-bdf-mini", "ds005509-bdf-mini", "ds005510-bdf-mini",
#                "ds005511-bdf-mini", "ds005512-bdf-mini", "ds005513-bdf-mini", "ds005514-bdf-mini", "ds005515-bdf-mini", "ds005516-bdf-mini"]
directories = ["ds005505-bdf", "ds005506-bdf", "ds005507-bdf", "ds005507-bdf", "ds005508-bdf", "ds005509-bdf", "ds005510-bdf",
               "ds005511-bdf", "ds005512-bdf", "ds005513-bdf", "ds005514-bdf", "ds005515-bdf", "ds005516-bdf"]
# Ruta base
base_path = "/Users/rocioperez/Documents/Projects/AFG/Git_Group_Project/eeg-prism/data"

# Lista para guardar coincidencias
matches = []

# Recorrer los directorios principales
for dir_name in directories:
    dir_path = os.path.join(base_path, dir_name)

    # Verificar si el directorio existe
    if not os.path.exists(dir_path):
        print(f"⚠️ El directorio {dir_path} no existe, se omite.")
        continue

    # Recorrer los subdirectorios
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)

        # Verificar si es un directorio y si su nombre contiene alguno de los caution_subjects
        if os.path.isdir(subdir_path):
            for subject in caution_subjects:
                if subject in subdir:
                    matches.append(subdir_path)
                    break  # Evita repeticiones

# Mostrar resultados
if matches:
    print("\n📂 Directorios que serían eliminados:\n")
    for path in matches:
        print(" -", path)
    print(f"\nTotal: {len(matches)} directorios encontrados.")
else:
    print("✅ No se encontraron coincidencias.")