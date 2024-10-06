import os
from obspy import read, Stream
from obspy.signal.invsim import cosine_taper
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Rutas a las carpetas de entrenamiento y prueba
directorio_train = "C:/Python/pruebasMaestria/proyectoMaestria/hackaton/data/space_apps_2024_seismic_detection/data/mars/training/data/"  # Reemplaza esto con la ruta a tu carpeta de entrenamiento
directorio_test = "C:/Python/pruebasMaestria/proyectoMaestria/hackaton/data/space_apps_2024_seismic_detection/data/mars/test/data/"    # Reemplaza esto con la ruta a tu carpeta de prueba

# Función para cargar archivos .mseed de un directorio
def cargar_trazas(directorio):
    trazas_totales = []
    for archivo in os.listdir(directorio):
        if archivo.endswith(".mseed"):
            trazas = read(os.path.join(directorio, archivo))
            trazas_totales.extend(trazas)
    return trazas_totales

# Cargar datos de entrenamiento y prueba
trazas_entrenamiento = cargar_trazas(directorio_train)
trazas_prueba = cargar_trazas(directorio_test)

print(f"Se han cargado {len(trazas_entrenamiento)} trazas de archivos de entrenamiento.")
print(f"Se han cargado {len(trazas_prueba)} trazas de archivos de prueba.")


# Función para detectar errores en los datos (valores atípicos)
def detectar_valores_atipicos(traza, umbral=5):
    media = np.mean(traza.data)
    desviacion_std = np.std(traza.data)
    indices_atipicos = np.where(np.abs(traza.data - media) > umbral * desviacion_std)[0]
    return indices_atipicos
def manejar_datos_faltantes(traza):
    if np.any(np.isnan(traza.data)):
        print("Datos faltantes detectados. Se procede a la imputación.")
        traza.data = np.interp(np.arange(len(traza.data)),
                               np.arange(len(traza.data))[~np.isnan(traza.data)],
                               traza.data[~np.isnan(traza.data)])
    return traza
def filtrar_traza(traza, frec_baja, frec_alta, orden=4):
    b, a = butter(orden, [frec_baja, frec_alta], btype='bandpass', fs=traza.stats.sampling_rate)
    traza.data = filtfilt(b, a, traza.data)
    return traza
def preprocesar_trazas(trazas):
    for traza in trazas:
        indices_atipicos = detectar_valores_atipicos(traza)
        if len(indices_atipicos) > 0:
            print(f"Valores atípicos detectados en la traza {traza.id}: {indices_atipicos}")
            for idx in indices_atipicos:
                traza.data[idx] = np.mean(traza.data)
        traza = manejar_datos_faltantes(traza)
        traza = filtrar_traza(traza, frec_baja=0.1, frec_alta=1.0)  # Ajusta las frecuencias según tus necesidades

preprocesar_trazas(trazas_entrenamiento)
preprocesar_trazas(trazas_prueba)

for traza in trazas_entrenamiento:
    plt.figure(figsize=(12, 4))
    plt.plot(traza.times(), traza.data, label='Traza Preprocesada')
    plt.title(f'Traza Preprocesada: {traza.id}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.show()
def visualizar_trazas(trazas):
    for traza in trazas:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(traza.times(), traza.data, label='Señal Sísmica')
        plt.title(f'Visualización de la Traza: {traza.id}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.specgram(traza.data, NFFT=256, Fs=traza.stats.sampling_rate, noverlap=128, cmap='viridis')
        plt.title(f'Espectrograma de la Traza: {traza.id}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')

        plt.tight_layout()
        plt.show()

def estadisticas_descriptivas(trazas):
    for traza in trazas:
        media = np.mean(traza.data)
        mediana = np.median(traza.data)
        desviacion_std = np.std(traza.data)
        print(
            f'Traza: {traza.id} | Media: {media:.2f}, Mediana: {mediana:.2f}, Desviación Estándar: {desviacion_std:.2f}')

visualizar_trazas(trazas_entrenamiento)
visualizar_trazas(trazas_prueba)
print("Estadísticas Descriptivas para las Trazas de Entrenamiento:")
estadisticas_descriptivas(trazas_entrenamiento)

print("\nEstadísticas Descriptivas para las Trazas de Prueba:")
estadisticas_descriptivas(trazas_prueba)

###################################################################3
def calcular_energia(segmento_datos):
    # Convertir el segmento de memoryview a numpy array
    datos_numpy = np.array(segmento_datos)
    return np.sum(datos_numpy ** 2)  # Sumar los valores al cuadrado para obtener la energía

def detectar_eventos_sismicos(traza, ventana_corta, ventana_larga):
    n = len(traza.data)
    energias_sta = np.zeros(n)
    energias_lta = np.zeros(n)
    for i in range(ventana_corta, n):
        energias_sta[i] = calcular_energia(traza.data[i - ventana_corta:i]) / ventana_corta
    for i in range(ventana_larga, n):
        energias_lta[i] = calcular_energia(traza.data[i - ventana_larga:i]) / ventana_larga
    relacion_sta_lta = energias_sta / (energias_lta + 1e-10)  # Añadir un pequeño valor para evitar división por cero
    return relacion_sta_lta

def extraer_caracteristicas(traza, relacion_sta_lta, umbral_sta_lta):
    n = len(traza.data)
    eventos = []
    energia_total = 0
    frecuencia_dominante = 0
    duracion_evento = 0
    amplitud_maxima = 0

    for i in range(n):
        if relacion_sta_lta[i] > umbral_sta_lta:
            ventana_evento = traza.data[i:i + 100]
            energia_evento = calcular_energia(ventana_evento)
            energia_total += energia_evento
            amplitud_maxima = max(amplitud_maxima, np.max(np.abs(ventana_evento)))
            fft_resultado = np.abs(fft(ventana_evento))
            frec_dominante = np.argmax(fft_resultado)
            frecuencia_dominante = max(frecuencia_dominante, frec_dominante)
            duracion_evento += 1
    caracteristicas = {
        'energia_total': energia_total,
        'frecuencia_dominante': frecuencia_dominante,
        'duracion_evento': duracion_evento,
        'amplitud_maxima': amplitud_maxima
    }
    return caracteristicas

def visualizar_caracteristicas(caracteristicas):
    plt.figure(figsize=(10, 6))
    etiquetas = ['Energía Total', 'Frecuencia Dominante', 'Duración Evento', 'Amplitud Máxima']
    valores = [
        caracteristicas['energia_total'],
        caracteristicas['frecuencia_dominante'],
        caracteristicas['duracion_evento'],
        caracteristicas['amplitud_maxima']
    ]
    plt.bar(etiquetas, valores, color=['blue', 'green', 'red', 'purple'])
    plt.title("Características de los Eventos Sísmicos")
    plt.ylabel("Valor")
    plt.show()

ventana_corta = 10
ventana_larga = 100
umbral_sta_lta = 3

relacion_sta_lta = detectar_eventos_sismicos(traza, ventana_corta, ventana_larga)
caracteristicas = extraer_caracteristicas(traza, relacion_sta_lta, umbral_sta_lta)

print("Características extraídas:")
print(f"Energía total: {caracteristicas['energia_total']}")
print(f"Frecuencia dominante: {caracteristicas['frecuencia_dominante']}")
print(f"Duración del evento: {caracteristicas['duracion_evento']}")
print(f"Amplitud máxima: {caracteristicas['amplitud_maxima']}")

##########################################################################33
