import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sc
from scipy.signal import convolve

# Función para calcular la TFD a partir de los coeficientes de la SFD
def calcular_tfd(coeficientes_sfd):
    N = len(coeficientes_sfd)
    return coeficientes_sfd * N                 #Calcula la TFD multiplicando los coeficientes de la serie de Fourier por N

def calcular_coeficientes(datos):
    N = datos.shape[0]                          #Calcula la cantidad de entradas dentro del archivo
    n = np.arange(N)                            #Hace un arreglo con valores desde 0 a N-1
    k = n.reshape((N, 1))                       #Hace el arreglo n como un vector columna
    M = np.exp(-2j * np.pi * k * n / N)         #Calcula el numero exponencial que tendra que ser multiplicado por la entrada x[n] (Datos)
    return np.dot(M, datos) / N

def filtrado(transformada, porcentaje):
    im = []
    for i in transformada:
        im.append(np.sqrt(i.real**2+i.imag**2))
        maximo_modulo=max(im)
        for i in range (0, im.__len__()):
            if(im[i]<porcentaje*maximo_modulo): F[i]=0
        return sc.ifft(transformada)

def graficar_coeficientes_sfd(coeficientes_sfd):
    n = len(coeficientes_sfd)
    #frecuencias = np.fft.fftfreq(n)
    plt.figure(figsize=(10, 6))
   # plt.plot(frecuencias, np.abs(coeficientes_sfd), label='Magnitud')
    frecuencias = np.fft.fftfreq(n)[:100]
    plt.stem(frecuencias, np.abs(coeficientes_sfd)[:100], linefmt='b-', markerfmt='bo', basefmt='k-') 
  #  plt.stem(frecuencias, np.abs(coeficientes_sfd), linefmt='b-', markerfmt='bo', basefmt='k-')
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

# Cargar los datos desde los archivos txt
datos_terremoto1 = np.loadtxt("terremoto1.txt", dtype=float)[:, 1]  # Solo cargamos la segunda columna
datos_terremoto2 = np.loadtxt("terremoto2.txt", dtype=float)[:, 1]  # Solo cargamos la segunda columna

#Calcular los coeficientes de la SFD para cada conjunto de datos
coeficientes_sfd_terremoto1 = calcular_coeficientes(datos_terremoto1)
coeficientes_sfd_terremoto2 = calcular_coeficientes(datos_terremoto2)

#Calcular la TFD para cada conjunto de coeficientes de la SFD
tfd_terremoto1 = calcular_tfd(coeficientes_sfd_terremoto1)
tfd_terremoto2 = calcular_tfd(coeficientes_sfd_terremoto2)

#Filtrado de onda
#graficar_espectro(tfd_terremoto1, "Transformada de Fourier Discreta del terremoto 1")
#graficar_espectro(filtrado(tfd_terremoto1, 0.5), "Filtrado de la onda del terremoto 1")


def suavizar_altas_frecuencias(datos):
    ventana = np.kaiser(len(datos),60)  # Utilizar una ventana de tipo Hann
    datos_suavizados = np.convolve(datos, ventana, mode='same')# Aplicar convolución
    return datos_suavizados

# Suavizar las altas frecuencias de los datos de terremoto1
datos_suavizados = suavizar_altas_frecuencias(datos_terremoto1)

# Graficar la señal original y la señal suavizada
plt.figure(figsize=(10, 6))
plt.plot(datos_terremoto1, label='Señal original')
plt.plot(datos_suavizados, label='Señal suavizada')
plt.title('Suavizado de altas frecuencias con ventana de tipo Hann')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()







'''
#Imprimir los resultados o usarlos según sea necesario
print("TFD del terremoto 1:", tfd_terremoto1)
print("TFD del terremoto 2:", tfd_terremoto2)

# Generar datos ruidosos (señal original)
t = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * t) + np.random.normal(0, 0.2, 1000)

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

# Aplicar la convolución para suavizar la señal
smoothed_signal = np.convolve(signal, gaussian_kernel(21), mode='same')

# Graficar la señal original y la señal suavizada
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Señal Original')
plt.plot(t, smoothed_signal, label='Señal Suavizada', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Suavizado de Ruido de Alta Frecuencia con Convolución')
plt.legend()
plt.grid(True)
plt.show()
'''