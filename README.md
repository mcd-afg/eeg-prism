# Guia de Documentacion

- Configuración del Entorno de Desarrollo 🐍
- Descargar los releases del EEG Challenge

## Configuración del Entorno de Desarrollo 🐍

Este proyecto requiere un entorno virtual para mantener aisladas sus dependencias.  
Sigue los pasos a continuación para configurar el entorno y comenzar a trabajar.

---

### 🚀 Requisitos Previos

Asegúrate de tener instalado:

- **Python 3.10+**
- **pip** (gestor de paquetes de Python)

Puedes verificar las versiones con:

```bash
python3 --version
pip --version
```

---

### 🧱 Crear el entorno virtual

Ejecuta el siguiente comando en la raíz del proyecto para crear un entorno virtual:

```bash
python3 -m venv .venv
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

  ```bash
  pip freeze > requirements.txt
  ```

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
