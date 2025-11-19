import pandas as pd
import numpy as np
import mne
from pathlib import Path
import glob

"""Código para recorrer las carpetas"""
from pathlib import Path
import mne
import matplotlib
import gc

#Uso del backend que me eprmite gráficos interactivos
matplotlib.use("Qt5Agg")

# Ruta donde están tus carpetas ds...
root = Path("eeg")

#Canales a usar, estos vienen de usar pares frontales y centrales que son de interés en las medidas
#que se van a extraer
#Van en este orden: Fp1, Fp2, F3, F4, F7, F8, C3, C4
canales_usar = ["E22", "E9", "E24", "E124", "E33", "E122", "E36", "E104"]

#Creo los nombres de los canales
nombres_canales = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4"]

#Nombres de los sufijos para TBR
sufijos_tbr = ["_TBR_open", "_TBR_closed"]

#Nombres columnas para TBR
columnas_tbr = []

#Creo los nombres de las columnas del excel en la parte TBR
for n_canales, canal in enumerate(nombres_canales):
    for sufijo in sufijos_tbr:
        columnas_tbr.append(f"{canal}{sufijo}")

#Canales FAA
canales_FAA = ["Fp2-Fp1", "F4-F3", "F8-F7"]

#Nombres de los sufijos para FAA
sufijos_faa = ["_FAA_open", "_FAA_closed"]

#Nombres columnas para FAA
columnas_faa = []
#Creo los nombres de las columnas del excel en la parte FAA
for n_canales, canal in enumerate(canales_FAA):
    for sufijo in sufijos_faa:
        columnas_faa.append(f"{canal}{sufijo}")

#Combino las columnas de TBR y FAA
columnas_excel = columnas_tbr + columnas_faa

#Creo el dataframe vacío para ir llenando los datos
datos_eeg = pd.DataFrame(columns=columnas_excel)


#Lista de sujetos que voy a remover porque no tienen todos los canales que quiero usar
sujetos_rem = []

# Recorrer todas las carpetas que comienzan por "ds"
for ds_folder in root.glob("ds*"):
    if ds_folder.is_dir():
        print(f"📁 Carpeta ds encontrada: {ds_folder}")

        # Recorrer carpetas "sub"
        for n_sujeto, sub_folder in enumerate(ds_folder.glob("sub-*")):
            # #Pongo una condición de parada para explorar
            # if n_sujeto >= 1:
            #     break

            if sub_folder.is_dir():
                print(f"   📂 Subcarpeta: {sub_folder}")

                eeg_dir = sub_folder / "eeg"

                # Asegurar que existe la carpeta eeg
                if eeg_dir.exists():
                    # Obtener archivos .bdf, .tsv y .json
                    bdf_files = list(eeg_dir.glob("*.bdf"))
                    tsv_files = list(eeg_dir.glob("*.tsv"))
                    json_files = list(eeg_dir.glob("*.json"))


                    if len(bdf_files) == 0:
                        print("      ⚠️ No hay archivos .bdf en esta carpeta")
                        continue

                    # Leer EEG con MNE
                    raw = mne.io.read_raw_bdf(bdf_files[0], preload=True)

                    #Leo el tsv de los canales
                    channels_tsv = pd.read_csv(tsv_files[0], sep="\t")

                    #Leo el tsv de lso eventos
                    events_tsv = pd.read_csv(tsv_files[1], sep="\t")

                    #Leo la descripción de los archivos json
                    json_description = pd.read_json(json_files[0], typ='series')

                    #VOy a ver si los canales de interés tienen status "good"
                    if not channels_tsv[channels_tsv['name'].isin(canales_usar)]['status'].eq('good').all():
                        sujetos_rem.append(sub_folder.name)
                        continue

                    #Filtro el raw a solo los canales de interes
                    raw = raw.pick_channels(canales_usar)
                    gc.collect()

                    #Voy a poner eventos al archivo raw usando los eventos del tsv
                    #Inicio del resting state
                    idx_start = events_tsv.index[events_tsv["event_code"] == "90"][0]

                    #ültima instrucción ojos abiertos o cerrados
                    idx_end = events_tsv.index[events_tsv["event_code"].isin(["20", "30"])][-1]

                    #Filtro a los eventos de interes
                    events_filtered = events_tsv.loc[idx_start:idx_end].reset_index(drop=True)

                    #Creo las anotaciones del raw
                    annotations = mne.Annotations(
                        onset=events_filtered["onset"].astype(float).values,
                        duration=np.nan_to_num(events_filtered["duration"].astype(float).values, nan=0.0),
                        description=events_filtered["value"].astype(str).values
                    )

                    #Agrego las anotaciones al raw
                    raw.set_annotations(annotations)

                    #Creo un diccionario de eventos
                    event_dict = {
                        'instructed_toOpenEyes': 1,
                        'instructed_toCloseEyes': 2,
                        "resting_start": 3
                    }

                    #Convierto las anotaciones a eventos
                    events, event_id = mne.events_from_annotations(raw, event_id=event_dict)

                    #Creo los eventos de ojos cerrados y abiertos
                    events_open = events[events[:, 2] == event_id['instructed_toOpenEyes']]
                    events_closed = events[(events[:, 2] == event_id['instructed_toCloseEyes']) |
                                           (events[:, 2] == event_id['resting_start'])]

                    event_id_open = {'Instructed_toOpenEyes': 1}
                    event_id_closed = {'Instructed_toCloseEyes': 2, 'Resting_start': 3}

                    #Creo los epochs por condición
                    epochs_open = mne.Epochs(raw, events_open, event_id=event_id_open, tmin=0, tmax=20,
                                             baseline=None, reject_by_annotation=True)
                    epochs_closed = mne.Epochs(raw, events_closed, event_id=event_id_closed, tmin=0, tmax=40,
                                               baseline=None, reject_by_annotation=True)

                    # Defino las bandas de frecuencia
                    bandas = {
                        "theta": (4, 8),
                        "alpha": (8, 13),
                        "beta": (13, 30),
                    }

                    #Calculo la PSD para ambos segmentos
                    psd_open = epochs_open.compute_psd(
                        method="welch",
                        fmin=0.5,
                        fmax=40,
                        n_fft=256
                    )

                    psd_closed = epochs_closed.compute_psd(
                        method="welch",
                        fmin=0.5,
                        fmax=40,
                        n_fft=256
                    )

                    #Obtengo los valores de psd y frecuencia
                    psds_open, freqs_open = psd_open.get_data(return_freqs=True)
                    psds_closed, freqs_closed = psd_closed.get_data(return_freqs=True)

                    #Creo el diccionario de potencias
                    band_powers = {}

                    #Recorro las bandas para calcular la potencia en cada una
                    for banda, (fmin, fmax) in bandas.items():

                        #Defino los límite de frecueencia de cada banda
                        band_mask_open = (freqs_open >= fmin) & (freqs_open <= fmax)
                        band_mask_closed = (freqs_closed >= fmin) & (freqs_closed <= fmax)

                        #Hago el promedio de potencia de la banda
                        band_powers[f"{banda}_open"] = psds_open[:, :, band_mask_open].mean(axis=-1)
                        band_powers[f"{banda}_closed"] = psds_closed[:, :, band_mask_closed].mean(axis=-1)

                    #Voy a calcular el FAA y TBR para cada condición
                    valores_tbr_open = band_powers["theta_open"] / band_powers["beta_open"]
                    valores_tbr_closed = band_powers["theta_closed"] / band_powers["beta_closed"]

                    valores_asimetria_open = np.column_stack([np.log(band_powers["alpha_open"][:, 1]) - np.log(band_powers["alpha_open"][:, 0]),
                                                        np.log(band_powers["alpha_open"][:, 3]) - np.log(band_powers["alpha_open"][:, 2]),
                                                        np.log(band_powers["alpha_open"][:, 5]) - np.log(band_powers["alpha_open"][:, 4])
                                                        ])
                    valores_asimetria_closed = np.column_stack([np.log(band_powers["alpha_closed"][:, 1]) - np.log(band_powers["alpha_closed"][:, 0]),
                                                          np.log(band_powers["alpha_closed"][:, 3]) - np.log(band_powers["alpha_closed"][:, 2]),
                                                          np.log(band_powers["alpha_closed"][:, 5]) - np.log(band_powers["alpha_closed"][:, 4])
                                                          ])
                    #Ahora promedio por epochs (promedio de la columna)
                    tbr_open_mean = np.nanmean(valores_tbr_open, axis=0)
                    tbr_closed_mean = np.nanmean(valores_tbr_closed, axis=0)
                    asimetria_open_mean = np.nanmean(valores_asimetria_open, axis=0)
                    asimetria_closed_mean = np.nanmean(valores_asimetria_closed, axis=0)

                    #Uno los valores en un solo array
                    datos_sujeto = np.concatenate([tbr_open_mean, tbr_closed_mean,
                                                  asimetria_open_mean, asimetria_closed_mean])

                    #Agrego los datos al dataframe
                    datos_eeg.loc[sub_folder.name] = datos_sujeto



#Voy a abrir una rchov tsv para ver su estructura
# canales = pd.read_csv("eeg\\\ds005505-bdf\\sub-NDARAC904DMU\\eeg\\sub-NDARAC904DMU_task-RestingState_channels.tsv", sep="\t")
