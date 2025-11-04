#!/usr/bin/env python3
import os
import sys
import argparse

def limpiar_eeg(base_dir: str, dry_run: bool = False):
    if not os.path.isdir(base_dir):
        print(f"❌ El directorio '{base_dir}' no existe.")
        sys.exit(1)

    print(f"🔍 Procesando: {base_dir}")
    if dry_run:
        print("💡 Modo simulación activado (no se eliminará nada).")

    # Buscar subdirectorios que comiencen con "sub-"
    for subdir in os.listdir(base_dir):
        if not subdir.startswith("sub-"):
            continue

        eeg_dir = os.path.join(base_dir, subdir, "eeg")

        if not os.path.isdir(eeg_dir):
            print(f"⚠️  No se encontró el directorio 'eeg' en {subdir}")
            continue

        print(f"📁 Revisando {eeg_dir}")

        for root, _, files in os.walk(eeg_dir):
            for file in files:
                if "RestingState" not in file:
                    file_path = os.path.join(root, file)
                    if dry_run:
                        print(f"→ Se eliminaría: {file_path}")
                    else:
                        try:
                            os.remove(file_path)
                            print(f"🗑️ Eliminado: {file_path}")
                        except Exception as e:
                            print(f"⚠️ Error al eliminar {file_path}: {e}")

    print(f"✅ Proceso finalizado en {base_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Limpia los directorios 'eeg' eliminando archivos que no contengan 'RestingState' en su nombre."
    )
    parser.add_argument("base_dir", help="Directorio base (por ejemplo: R1_L100_bdf)")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué archivos se eliminarían sin borrarlos")

    args = parser.parse_args()
    limpiar_eeg(args.base_dir, args.dry_run)


if __name__ == "__main__":
    main()