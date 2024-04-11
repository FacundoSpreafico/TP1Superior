import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import correlate

# Función para calcular los coeficientes de la serie de Fourier discreta
def calcular_coeficientes(datos):
    N = len(datos)  # Número de puntos en la serie de tiempo
    T = datos[1][0] - datos[0][0]  # Paso de tiempo

    # Calcular la Transformada Discreta de Fourier (DFT) usando FFT
    y = [p[1] for p in datos]
    yf = np.fft.fft(y)
    
    # Calcular las frecuencias
    xf = np.fft.fftfreq(N, T)[:N//2]

    # Calcular los coeficientes de la serie de Fourier
    coeficientes = 2.0/N * np.abs(yf[:N//2])

    return xf, coeficientes

# Función para cargar los datos desde un archivo
def cargar_datos(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        datos = [linea.strip().split('\t') for linea in archivo.readlines()]
        datos = [(float(punto[0]), float(punto[1])) for punto in datos]
    return datos

# Función para suavizar la señal
def suavizar_senal(datos, ventana):
    y = [p[1] for p in datos]
    y_suavizado = convolve(y, ventana, mode='same') / sum(ventana)
    datos_suavizados = [(datos[i][0], y_suavizado[i]) for i in range(len(datos))]
    return datos_suavizados

def graficar_senal_y_espectro(x, y, xf, coeficientes, titulo):
    # Graficar la señal
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.title('Señal de ' + titulo)
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')

    # Graficar el espectro de frecuencias
    plt.subplot(2, 1, 2)
    plt.plot(xf, coeficientes)
    plt.title('Espectro de frecuencias de ' + titulo)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.tight_layout()
    plt.show()

# Función para determinar las frecuencias más afectadas
def frecuencias_mas_afectadas(xf, coeficientes, titulo):
    # Encontrar el índice del coeficiente máximo
    indice_max = np.argmax(coeficientes)
    frecuencia_afectada = xf[indice_max]
    coef_max = coeficientes[indice_max]

    print("En la señal de", titulo + ", la frecuencia más afectada es:", frecuencia_afectada, "Hz con un coeficiente de:", coef_max)

# Función para calcular la segunda derivada (aceleración) de una señal
def calcular_aceleracion(datos, T):
    y = [p[1] for p in datos]
    aceleracion = np.gradient(np.gradient(y, T), T)
    return aceleracion

# Método 1: Diferencia entre los coeficientes de Fourier
def frecuencia_mas_acelerada_method1(xf1, coeficientes1, xf2, coeficientes2):
    # Igualar las longitudes de los coeficientes
    min_length = min(len(coeficientes1), len(coeficientes2))
    coeficientes1_padded = np.pad(coeficientes1[:min_length], (0, max(0, min_length - len(coeficientes1))), mode='constant')
    coeficientes2_padded = np.pad(coeficientes2[:min_length], (0, max(0, min_length - len(coeficientes2))), mode='constant')

    diferencia_coeficientes = np.abs(coeficientes1_padded - coeficientes2_padded)
    indice_max_dif = np.argmax(diferencia_coeficientes)
    frecuencia_mas_acelerada = xf1[indice_max_dif]
    return frecuencia_mas_acelerada

# Método 2: Comparación directa de las señales en el dominio del tiempo
def frecuencia_mas_acelerada_method2(datos1, datos2):
    T = datos1[1][0] - datos1[0][0]  # Paso de tiempo
    aceleracion1 = calcular_aceleracion(datos1, T)
    aceleracion2 = calcular_aceleracion(datos2, T)

    # Encontrar la frecuencia dominante para cada señal
    indice_max_acel1 = np.argmax(np.abs(aceleracion1))
    indice_max_acel2 = np.argmax(np.abs(aceleracion2))
    
    # Obtener las frecuencias correspondientes a los índices dominantes
    frecuencia_mas_acelerada1 = 1 / (xf1[indice_max_acel1] * 2 * np.pi)
    frecuencia_mas_acelerada2 = 1 / (xf2[indice_max_acel2] * 2 * np.pi)
    
    return frecuencia_mas_acelerada1, frecuencia_mas_acelerada2

def calcular_correlacion(signal1, signal2):
    # Normalizar las señales
    signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) * len(signal1))
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    # Calcular la correlación cruzada
    correlacion = correlate(signal1, signal2, mode='full')
    # Encontrar el desplazamiento con mayor correlación
    max_correlacion_index = np.argmax(correlacion)
    return correlacion, max_correlacion_index

# Nombre de los archivos
archivo1 = 'terremoto1.txt'
archivo2 = 'terremoto2.txt'

# Cargar los datos desde los archivos
datos1 = cargar_datos(archivo1)
datos2 = cargar_datos(archivo2)

# Calcular coeficientes y TFD para los datos 1
xf1, coeficientes1 = calcular_coeficientes(datos1)

# Calcular coeficientes y TFD para los datos 2
xf2, coeficientes2 = calcular_coeficientes(datos2)

# Ventana para suavizar (por ejemplo, una ventana de media móvil)
ventana = np.ones(10) / 10  # Una ventana de longitud 10 para suavizar

# Suavizar las señales
datos1_suavizados = suavizar_senal(datos1, ventana)
datos2_suavizados = suavizar_senal(datos2, ventana)

# Calcular coeficientes y TFD para las señales suavizadas
xf1_suavizados, coeficientes1_suavizados = calcular_coeficientes(datos1_suavizados)
xf2_suavizados, coeficientes2_suavizados = calcular_coeficientes(datos2_suavizados)

# Imprimir los resultados
print("Coeficientes de la serie de Fourier para", archivo1, ":", coeficientes1)
print("Frecuencias para", archivo1, ":", xf1)
print("\nCoeficientes de la serie de Fourier para", archivo2, ":", coeficientes2)
print("Frecuencias para", archivo2, ":", xf2)

# Graficar los datos de los archivos originales
print("Gráficos de los datos originales:")
graficar_senal_y_espectro([p[0] for p in datos1], [p[1] for p in datos1], xf1, coeficientes1, archivo1)
graficar_senal_y_espectro([p[0] for p in datos2], [p[1] for p in datos2], xf2, coeficientes2, archivo2)

# Graficar los datos suavizados
print("Gráficos de los datos suavizados:")
graficar_senal_y_espectro([p[0] for p in datos1_suavizados], [p[1] for p in datos1_suavizados], xf1_suavizados, coeficientes1_suavizados, archivo1 + " Suavizado")
graficar_senal_y_espectro([p[0] for p in datos2_suavizados], [p[1] for p in datos2_suavizados], xf2_suavizados, coeficientes2_suavizados, archivo2 + " Suavizado")

# Calcular las frecuencias más afectadas antes del filtrado
print("Frecuencias más afectadas antes del filtrado:")
frecuencias_mas_afectadas(xf1, coeficientes1, archivo1)
frecuencias_mas_afectadas(xf2, coeficientes2, archivo2)

# Calcular las frecuencias más afectadas después del filtrado
print("\nFrecuencias más afectadas después del filtrado:")
frecuencias_mas_afectadas(xf1_suavizados, coeficientes1_suavizados, archivo1 + " Suavizado")
frecuencias_mas_afectadas(xf2_suavizados, coeficientes2_suavizados, archivo2 + " Suavizado")

'''Tiene problemas para solucionarlo, especialemente para arreglar los errores de compilacion
# Calcular la frecuencia que más se aceleró usando ambos métodos
frecuencia_mas_acelerada_method1 = frecuencia_mas_acelerada_method1(xf1, coeficientes1, xf2, coeficientes2)
frecuencia_mas_acelerada_method2_1, frecuencia_mas_acelerada_method2_2 = frecuencia_mas_acelerada_method2(datos1, datos2)

# Imprimir resultados
print("Frecuencia que más se aceleró (Método 1):", frecuencia_mas_acelerada_method1, "Hz")
print("Frecuencia que más se aceleró para la señal 1 (Método 2):", frecuencia_mas_acelerada_method2_1, "Hz")
print("Frecuencia que más se aceleró para la señal 2 (Método 2):", frecuencia_mas_acelerada_method2_2, "Hz")
'''

# Cargar los datos de terremoto3.txt
datos3 = np.loadtxt('terremoto3.txt')

# Calcular coeficientes y TFD para los datos 3
xf3, coeficientes3 = calcular_coeficientes(datos3)

# Calcular la correlación cruzada entre los coeficientes de Fourier de terremoto3.txt y los otros dos detectores
correlacion1, _ = calcular_correlacion([p[1] for p in datos3], [p[1] for p in datos1])
correlacion2, _ = calcular_correlacion([p[1] for p in datos3], [p[1] for p in datos2])

# Imprimir los resultados
print("Correlación cruzada con el detector 1:", np.max(correlacion1))
print("Correlación cruzada con el detector 2:", np.max(correlacion2))

# Determinar cuál de los detectores está más próximo
if np.max(correlacion1) > np.max(correlacion2):
    print("El detector 1 está más próximo a la ubicación de terremoto3.txt.")
else:
    print("El detector 2 está más próximo a la ubicación de terremoto3.txt.")