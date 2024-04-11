import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sc
from scipy.signal import convolve

'''
    .       ⣀⣠⣤⠶⠦⠤⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀ ⠀⠀
⠀⠀    ⠀⢀⣤⠾⠋⠁⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢦⡀⠀⠀⠀⠀ 
⠀⠀⠀⠀⣰⠟⠁⠀⠀⠀⠀⠀⠀⠀⣀⢀⠀⠀⠀⠀⠀⠙⢦⠀⠀⠀
 ⠀⠀⢀⡼⠃⠀⠀⠀⢀⣤⣴⣴⣾⣿⣿⣷⣷⣦⡀⠀⠀⠀⠸⣇⠀⠀
 ⠀⢀⡞⠁⠀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠄⠀⠀⠀⢹⠀
⠀ ⣴⠋⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠃⠀⠀⠀⠀⢸⡄
⠀ ⣿⠀⠀⠀⣾⠋⣉⣉⡉⠙⢻⣿⣿⡏⠉⣠⡤⠤⡄⠀⠀⠀⠀⠘⡇
⠀ ⠙⣯⠁⠄⣿⣿⠁⢉⠉⣒⣬⣿⡟⢁⣪⣍⣁⡀⣐⣾⠀⠀⠀⠰⠇
⠀ ⠰⣿⣶⡇⢿⣿⣿⣿⣿⣿⣿⣿⡷⢘⣿⣿⣿⣿⣿⣿⠀⠀⠳⣳⠀
⠀ ⠠⠯⣼⣿⢸⣿⣿⣿⣿⣿⣿⣿⣷⡌⡿⢿⣿⣿⣿⠇⠀⠈⠠⡁⠀
 ⠀⠀⠈⠿⢸⣿⣿⣿⣿⠯⠉⠛⠛⠁⠀⠺⠿⢿⣿⢸⡆⠀⠀⠀⠀⠀ 
⠀      ⣿⣿⣍⣌⣬⣿⣿⣿⣭⣭⡤⣠⣤⣿⣼⡇⠀⠀⠀⠀⠀ 
        ⣿⣿⣿⣿⣯⣿⣿⣿⣿⣧⣶⣿⣿⢳⡟⠀⠀⠀⠀⠀⠀ 
        ⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠇⠏⠀⠀⠀⠀⠀⠀⠀ 
          ⠈⠻⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⢀⡀⠀⢀⣀⣀⠀
 ⠀⣀⣤⣤⣤⣴⣇⢰⣶⣤⣉⠙⠁⠋⠉⠀⠀⢀⣤⣾⣧⣠⣼⣿⣿⣮
 ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
'''

# Función para calcular la TFD a partir de los coeficientes de la SFD
def calcular_tfd(coeficientes_sfd):
    N = len(coeficientes_sfd)
    return coeficientes_sfd * N                 #Calcula la TFD multiplicando los coeficientes de la serie de Fourier por N

# Funcion para calcular los coeficientes de la serie de Fourier de la senal
def calcular_coeficientes(datos):
    N = len(datos)                         #Calcula la cantidad de entradas dentro del archivo
    n = np.arange(N)                            #Hace un arreglo con valores desde 0 a N-1
    k = n.reshape((N, 1))                       #Hace el arreglo n como un vector columna
    M = np.exp(-2j * np.pi * k * n / N)         #Calcula el numero exponencial que tendra que ser multiplicado por la entrada x[n] (Datos)
    return np.dot(M, [dato[1] for dato in datos]) / N

# Funcion para filtrar frecuencias
def suavizar_altas_frecuencias(datos):
    ventana = np.Hann(datos)  # Utilizar una ventana de tipo Hann
    datos_suavizados = np.convolve(datos, ventana, mode='same')# Aplicar convolución
    return datos_suavizados

#Funcion para filtrar frecuencias (GPT)
def filtrar(datos):
    ventana = np.ones(10)/10            #ventana movil de long 10
    y = [p[1] for p in datos]
    y_suavizado = convolve(y, ventana, mode='same') / sum(ventana)
    datos_suavizados = [(datos[i][0], y_suavizado[i]) for i in range(len(datos))]
    return datos_suavizados

def graficar_coeficientes_sfd(coeficientes_sfd):
    n = len(coeficientes_sfd)
    #frecuencias = np.fft.fftfreq(n)
    plt.figure(figsize=(10, 6))
    #plt.plot(frecuencias, np.abs(coeficientes_sfd), label='Magnitud')
    frecuencias = np.fft.fftfreq(n)[:100]
    plt.stem(frecuencias, np.abs(coeficientes_sfd)[:100], linefmt='b-', markerfmt='bo', basefmt='k-') 
    #plt.stem(frecuencias, np.abs(coeficientes_sfd), linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.title('Coeficientes de la Serie de Fourier Discreta')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_espectro(coeficientes_sfd, titulo):  # Graficar un bastón para cada step frecuencias
    n = len(coeficientes_sfd)
    frecuencias = np.fft.fftfreq(n)
    plt.stem(frecuencias[::40], np.abs(coeficientes_sfd)[::40], linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.title(titulo)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.show()    

def graficar_tfd(tfd):
    n = len(tfd)
    frecuencias = np.fft.fftfreq(n)
    plt.figure(figsize=(10, 6))
    plt.plot(frecuencias, np.abs(tfd), label='Magnitud')
    plt.title('Transformada de Fourier Discreta (TFD)')
    plt.xlabel('Frecuencia')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_senal(datos, datos_suavizados):
    # Graficar la señal original y la señal suavizada
    plt.figure(figsize=(10, 6))
    plt.plot(datos, label='Señal original')
    plt.plot(datos_suavizados, label='Señal suavizada')
    plt.title('Suavizado de altas frecuencias con ventana de tipo Hann')
    plt.xlabel('Tiempo')
    plt.ylabel('Aceleracion')
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_senal_2(datos, datos_suavizados):
    # Extraer componentes de tiempo y aceleración de los datos originales
    tiempo_datos = [par[0] for par in datos]
    aceleracion_datos = [par[1] for par in datos]
    
    # Extraer componentes de tiempo y aceleración de los datos suavizados
    tiempo_suavizado = [par[0] for par in datos_suavizados]
    aceleracion_suavizada = [par[1] for par in datos_suavizados]

    # Graficar la señal original y la señal suavizada
    plt.figure(figsize=(10, 6))
    plt.plot(tiempo_datos, aceleracion_datos, label='Señal original')
    plt.plot(tiempo_suavizado, aceleracion_suavizada, label='Señal suavizada')
    plt.title('Suavizado de altas frecuencias')
    plt.xlabel('Tiempo')
    plt.ylabel('Aceleracion')
    plt.grid(True)
    plt.legend()
    plt.show()

def cargar_datos(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        datos = [linea.strip().split('\t') for linea in archivo.readlines()]
        datos = [(float(punto[0]), float(punto[1])) for punto in datos]
    return datos

''' Cargado de datos viejo.
# Cargar los datos desde los archivos txt
datos_terremoto1 = np.loadtxt("terremoto1.txt", dtype=float)[:, 1]  # Solo cargamos la segunda columna
datos_terremoto2 = np.loadtxt("terremoto2.txt", dtype=float)[:, 1]  # Solo cargamos la segunda columna
'''

#Cargado de datos 2.0 (hace un array de pares (tiempo, aceleracion))
datos_terremoto1 = cargar_datos('terremoto1.txt')
datos_terremoto2 = cargar_datos('terremoto2.txt')

#Calcular los coeficientes de la SFD para cada conjunto de datos
coeficientes_sfd_terremoto1 = calcular_coeficientes(datos_terremoto1)
coeficientes_sfd_terremoto2 = calcular_coeficientes(datos_terremoto2)

#Imprimir los coeficientes de Fourier
print("Coeficientes de la serie de Fourier para terremoto1: ", coeficientes_sfd_terremoto1)
print("\nCoeficientes de la serie de Fourier para terremoto2: ", coeficientes_sfd_terremoto2)

#Calcular la TFD para cada conjunto de coeficientes de la SFD
tfd_terremoto1 = calcular_tfd(coeficientes_sfd_terremoto1)
tfd_terremoto2 = calcular_tfd(coeficientes_sfd_terremoto2)

#Imprimir TFD
print("\nFrecuencias para terremoto1: ", tfd_terremoto1)
print("\nFrecuencias para terremoto2: ", tfd_terremoto2)

#Filtrado de onda (revisar es el metodo viejo)
#graficar_espectro(tfd_terremoto1, "Transformada de Fourier Discreta del terremoto 1")
#graficar_espectro(filtrado(tfd_terremoto1, 0.5), "Filtrado de la onda del terremoto 1")

# Suavizar las altas frecuencias de los terremotos y graficarlo
datos_filtrados1 = filtrar(datos_terremoto1)
graficar_senal_2(datos_terremoto1, datos_filtrados1)
datos_filtrados2 = filtrar(datos_terremoto2)
graficar_senal_2(datos_terremoto2, datos_filtrados2)