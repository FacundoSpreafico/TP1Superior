import numpy as np
import matplotlib.pyplot as plt

# Función para calcular la TFD a partir de los coeficientes de la SFD
def calcular_tfd(coeficientes_sfd):
<<<<<<< HEAD
    N = len(coeficientes_sfd)
    return coeficientes_sfd * N
=======
    tfd = np.fft.ifft(coeficientes_sfd)
    return tfd
>>>>>>> 7f244c56fdd625b101a1363e9c8c6b05a5b90845

# Función para calcular los coeficientes de la SFD
def calcular_coeficientes_sfd(datos):
    n = len(datos)
    coeficientes_sfd = (np.fft.fft(datos) / n)
    return coeficientes_sfd

def calcular_coeficientes(datos):
    N = datos.shape[0]                          #Calcula la cantidad de entradas dentro del archivo
    n = np.arange(N)                            #Hace un arreglo con valores desde 0 a N-1
    k = n.reshape((N, 1))                       #Hace el arreglo n como un vector columna
    M = np.exp(-2j * np.pi * k * n / N)         #Calcula el numero exponencial que tendra que ser multiplicado por la entrada x[n] (Datos)
    return np.dot(M, datos) / N

def graficar_coeficientes_sfd(coeficientes_sfd):
    n = len(coeficientes_sfd)
    frecuencias = np.fft.fftfreq(n)
    plt.figure(figsize=(10, 6))
    plt.plot(frecuencias, np.abs(coeficientes_sfd), label='Magnitud')
    plt.title('Coeficientes de la Serie de Fourier Discreta')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.legend()
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

# Cargar los datos desde los archivos txt
datos_terremoto1 = np.loadtxt("terremoto1.txt", dtype=float)[:, 1]  # Solo cargamos la segunda columna
datos_terremoto2 = np.loadtxt("terremoto2.txt", dtype=float)[:, 1]  # Solo cargamos la segunda columna

#Calcular los coeficientes de la SFD para cada conjunto de datos
coeficientes_sfd_terremoto1 = calcular_coeficientes(datos_terremoto1)
coeficientes_sfd_terremoto2 = calcular_coeficientes(datos_terremoto2)

for i in range(10):
    print (i)
    print(coeficientes_sfd_terremoto1[i])


<<<<<<< HEAD
'''Grafico de los coeficientes
graficar_coeficientes_sfd(coeficientes_sfd_terremoto1)
graficar_coeficientes_sfd(coeficientes_sfd_terremoto2)
'''

''' '''
#Calcular la TFD para cada conjunto de coeficientes de la SFD
tfd_terremoto1 = calcular_tfd(coeficientes_sfd_terremoto1)
tfd_terremoto2 = calcular_tfd(coeficientes_sfd_terremoto2)

=======
# Calcular la TFD para cada conjunto de coeficientes de la SFD
tfd_terremoto1 = calcular_tfd(calcular_coeficientes(datos_terremoto1))
#tfd_terremoto2 = calcular_tfd(coeficientes_sfd_terremoto2)


#graficar_coeficientes(calcular_coeficientes(datos_terremoto1))

graficar_tfd(tfd_terremoto1)
#graficar_tfd(tfd_terremoto2)
>>>>>>> 7f244c56fdd625b101a1363e9c8c6b05a5b90845

'''Grafico de la TFD
graficar_tfd(tfd_terremoto1)
graficar_tfd(tfd_terremoto2)
'''

'''
#Imprimir los resultados o usarlos según sea necesario
print("TFD del terremoto 1:", tfd_terremoto1)
print("TFD del terremoto 2:", tfd_terremoto2)
'''