import pandas as pd
from pathlib import Path
# Directorios principales a revisar
# directories = ["ds005505-bdf-mini", "ds005506-bdf-mini", "ds005507-bdf-mini", "ds005507-bdf-mini", "ds005508-bdf-mini", "ds005509-bdf-mini", "ds005510-bdf-mini",
#                "ds005511-bdf-mini", "ds005512-bdf-mini", "ds005513-bdf-mini", "ds005514-bdf-mini", "ds005515-bdf-mini", "ds005516-bdf-mini"]
directories = ["ds005505-bdf", "ds005506-bdf", "ds005507-bdf", "ds005508-bdf", "ds005509-bdf", "ds005510-bdf",
               "ds005511-bdf", "ds005512-bdf", "ds005513-bdf", "ds005514-bdf", "ds005515-bdf", "ds005516-bdf"]
# Ruta base
base_path = Path("/Users/rocioperez/Documents/Projects/AFG/Git_Group_Project/eeg-prism/data")

dfs = []

for d in directories:
    participants_path = base_path / d / "participants.tsv"
    
    if not participants_path.exists():
        print(f"⚠️ No se encontró: {participants_path}")
        continue
    
    df_part = pd.read_csv(participants_path, sep="\t")
    
    # Opcional: agregar columna para saber de qué dataset viene cada fila
    df_part["dataset"] = d
    
    dfs.append(df_part)

# Concatenar todos en un solo DataFrame
if dfs:
    participants_all = pd.concat(dfs, ignore_index=True)
else:
    participants_all = pd.DataFrame()

# Solo para revisar
print(participants_all.shape)
print(participants_all.head())

participants_all.to_csv("complete_participants.tsv", sep="\t", index=False)