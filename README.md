# Guia de Documentacion

- [Configuración del Entorno de Desarrollo 🐍](https://github.com/mcd-afg/eeg-prism?tab=readme-ov-file#configuraci%C3%B3n-del-entorno-de-desarrollo-)
- [Descargar los releases del EEG Challenge](https://github.com/mcd-afg/eeg-prism?tab=readme-ov-file#descargar-los-releases-del-eeg-challenge)
- [Eliminar Tasks que no sean RestingState](https://github.com/mcd-afg/eeg-prism?tab=readme-ov-file#eliminar-tasks-que-no-sean-restingstate)

## Configuración del Entorno de Desarrollo 🐍

Este proyecto requiere un entorno virtual para mantener aisladas sus dependencias.  
Sigue los pasos a continuación para configurar el entorno y comenzar a trabajar.

---

### 🚀 Requisitos Previos

Asegúrate de tener instalado:

- **Python 3.10+**
- **pip** (gestor de paquetes de Python)

---

### 🧱 Crear el entorno virtual

Ejecuta el siguiente comando en la raíz del proyecto para crear un entorno virtual:

```bash
python3.10 -m venv .venv
```

---

### 🔑 Activar el entorno virtual

- **En macOS / Linux:**

  ```bash
  source .venv/bin/activate
  ```

- **En Windows (PowerShell):**

  ```bash
  .venv\Scripts\Activate.ps1
  ```

Cuando el entorno esté activado, deberías ver `(.venv)` al inicio de tu terminal.

---

### 📦 Instalar dependencias

Con el entorno virtual activo, instala las dependencias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

Si no resulta, utiliza python3 antes

```bash
python3 -m pip install -r requirements.txt
```

---

### 🧹 Desactivar el entorno virtual

Cuando termines de trabajar, puedes desactivar el entorno con:

```bash
deactivate
```

---

### 💡 Notas

- Si instalas nuevos paquetes, recuerda actualizar el archivo de requerimientos:

- Si deseas eliminar el entorno virtual, simplemente borra la carpeta `.venv`.

---

### ✅ Listo

Tu entorno está configurado correctamente.  

## Descargar los releases del EEG Challenge

Para cargar los datos en nuestra maquina, debemos  correr el script `scripts/donwload_data.py`.
Dado que los sripts estan escritos para ser corridos desde el terminnal, deben anteponer `-m` y utilizar puntos en vez de slashes.

```bash
python3 -m scripts.download_data
```

Con esto descargaras los releases del EEG Challenge. Una vez que has descargado los releases, no es necesario descargarlos de nuevo, y en cambio, para utilizarlos debemos utilizar el metodo `load_ccd_dataset()` que se encuentra en `src/data/load_releases.py`

## Eliminar tasks que no sean RestingState

Para eliminar los tasks descargados por `EEGChallengeDataset` a traves del script `scripts/downnload_data.py` podemos utilizar el script a traves de bash (MacOS/Linux) o python (cualquier OS)

### Bash usando clean_eeg.sh

Otorga permisos para ejecucion:

```bash
chmod +x scripts/clean_eeg.sh
```

Ahora puedes ejecutarlo, y solo nnecesitas especificar la ruta del directorio que contiene los datos. Por ejemplo, si descargaste los datos a partir del script `scripts/downnload_data.py`, la ruta del release 1 seria asi:

```bash
./scripts/clean_eeg.ph data/ds005506-bdf-mini
```

### Python usando clean_eeg.py

Otorga permisos para ejecucion:

```bash
chmod +x scripts/clean_eeg.py
```

Ahora puedes ejecutarlo, y solo nnecesitas especificar la ruta del directorio que contiene los datos. Por ejemplo, si descargaste los datos a partir del script `scripts/downnload_data.py`, la ruta del release 1 seria asi:

```bash
./scripts/clean_eeg.py data/ds005506-bdf-mini
```

### Ejecucion --dry-run

Tanto el script de python como el bash tienen la opcion para ejecutarse --dry-run. En este formato te traera los datos que seran borrados, pero sin borrarlos aun. Para borrarlos debes sacarle el --dry-run.

```bash
./scripts/clean_eeg.py data/ds005506-bdf-mini --dry-run

```

```bash
./scripts/clean_eeg.sh data/ds005506-bdf-mini --dry-run

```
