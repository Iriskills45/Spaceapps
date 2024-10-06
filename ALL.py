import os
from obspy import read
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Rutas a las carpetas de entrenamiento y prueba
directorio_train = "C:/Python/pruebasMaestria/proyectoMaestria/hackaton/data/space_apps_2024_seismic_detection/data/mars/training/data/"
directorio_test = "C:/Python/pruebasMaestria/proyectoMaestria/hackaton/data/space_apps_2024_seismic_detection/data/mars/test/data/"


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


# Funciones para preprocesamiento
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


# Visualizar trazas
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


visualizar_trazas(trazas_entrenamiento)
visualizar_trazas(trazas_prueba)


# Funciones para detección de eventos sísmicos
def calcular_energia(segmento_datos):
    datos_numpy = np.array(segmento_datos)
    return np.sum(datos_numpy ** 2)


def detectar_eventos_sismicos(traza, ventana_corta, ventana_larga):
    n = len(traza.data)
    energias_sta = np.zeros(n)
    energias_lta = np.zeros(n)
    for i in range(ventana_corta, n):
        energias_sta[i] = calcular_energia(traza.data[i - ventana_corta:i]) / ventana_corta
    for i in range(ventana_larga, n):
        energias_lta[i] = calcular_energia(traza.data[i - ventana_larga:i]) / ventana_larga
    relacion_sta_lta = energias_sta / (energias_lta + 1e-10)
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


# Entrenamiento de Modelos de Aprendizaje Automático
def preparar_datos_modelo(trazas, umbral_sta_lta):
    X = []
    y = []
    for traza in trazas:
        relacion_sta_lta = detectar_eventos_sismicos(traza, ventana_corta=10, ventana_larga=100)
        caracteristicas = extraer_caracteristicas(traza, relacion_sta_lta, umbral_sta_lta)
        X.append([caracteristicas['energia_total'], caracteristicas['frecuencia_dominante'],
                  caracteristicas['duracion_evento'], caracteristicas['amplitud_maxima']])
        y.append(1 if caracteristicas['energia_total'] > 0 else 0)  # Ajusta según la lógica de tu modelo
    return np.array(X), np.array(y)


X_train, y_train = preparar_datos_modelo(trazas_entrenamiento, umbral_sta_lta=3)
X_test, y_test = preparar_datos_modelo(trazas_prueba, umbral_sta_lta=3)

# División de Datos
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Selección de Modelos y Entrenamiento
modelos = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Red Neuronal": MLPClassifier(max_iter=500)
}

for nombre, modelo in modelos.items():
    print(f"Entrenando modelo: {nombre}")
    modelo.fit(X_train, y_train)

    # Evaluación del Rendimiento
    y_pred = modelo.predict(X_val)
    print(classification_report(y_val, y_pred))

    # Curvas ROC
    y_scores = None
    if len(set(y_val)) > 1:  # Solo si hay más de una clase
        y_scores = modelo.predict_proba(X_val)[:, 1]
    else:
        # Asignar valores ficticios o de ejemplo
        y_scores = np.full(len(y_val), 0.5)  # Todos los valores en 0.5

    # Calcular y graficar la curva ROC
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {nombre}')
    plt.legend(loc='lower right')
    plt.show()

