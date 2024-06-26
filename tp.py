import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sc
from scipy.signal import convolve
from scipy.signal import correlate

# Función para calcular la TFD a partir de los coeficientes de la SFD
def calcular_tfd(coeficientes_sfd):
    N = len(coeficientes_sfd)
    return coeficientes_sfd * N                 #Calcula la TFD multiplicando los coeficientes de la serie de Fourier por N

# Funcion para calcular los coeficientes de la serie de Fourier de la senal
def calcular_coeficientes(datos):
    N = len(datos)                              #Calcula la cantidad de entradas dentro del archivo
    #print("\nLargo de los datos: ", N)
    n = np.arange(N)                            #Hace un arreglo con valores desde 0 a N-1
    k = n.reshape((N, 1))                       #Hace el arreglo n como un vector columna
    M = np.exp(-2j * np.pi * k * n / N)         #Calcula el numero exponencial que tendra que ser multiplicado por la entrada x[n] (Datos)
    return np.dot(M, [dato[1] for dato in datos]) * (2/N)

#Funcion para sacar las frecuencias de la TFD
def frecuencias_tfd(frecuencia_muestreo, datos):
    N=len(datos)
    frecuencias = np.fft.fftfreq(N, 1/frecuencia_muestreo)
    return frecuencias

#Funcion que filtra frecuencias. convolucion con hanning
def filtrar(datos):
    ventana = np.hanning(50)            #ventana hanning
    y = [p[1] for p in datos]
    y_suavizado = convolve(y, ventana, mode='same') / sum(ventana)
    datos_suavizados = [(datos[i][0], y_suavizado[i]) for i in range(len(datos))]
    return datos_suavizados

#Funcion para determinar el nivel de correlacion entre dos senales
def nivel_de_correlacion(senal1, senal2):
    aux = correlate(senal1, senal2, mode='full')
    return aux

#Funcion para acortar el terremoto 2 a 4000 puntos
def acortar_terr2(datos):
    res = []
    for i in range(len(datos)):
        if(i % 2 == 0):
            res.append(datos[i])
    #print(len(res))
    return res

#Encuentra la freq maxima para una lista de coeficientes dados
def freq_max(freqs, coef):
    coef = abs(coef)
    indice = np.argmax(coef)
    return np.abs(freqs[indice])

#suma termino a termino dos tfds
def suma_tfd(tfd1, tfd2):
    return tfd1 + tfd2

#prod punto de dos tfds
def prod_punto_tfd(tfd1, tfd2):
    return tfd1 * tfd2

def graficar_espectro_continuo(freq, coeficientes, titulo):
    positive_freq = freq[freq >= 0]
    positive_coef = coeficientes[freq >= 0]
    positive_coef = np.abs(positive_coef)

    plt.plot(positive_freq, positive_coef)
    plt.title('Espectro de frecuencias de ' + titulo)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.tight_layout()
    plt.show()

def graficar_senal(datos, datos_suavizados):
    # Extraer componentes de tiempo y aceleración de los datos originales
    tiempo_datos = [par[0] for par in datos]
    aceleracion_datos = [par[1] for par in datos]
    # Extraer componentes de tiempo y aceleración de los datos suavizados
    tiempo_suavizado = [par[0] for par in datos_suavizados]
    aceleracion_suavizada = [par[1] for par in datos_suavizados]
    # Graficar la señal original y la señal suavizada
    plt.figure(figsize=(10, 6))
    plt.plot(tiempo_datos, aceleracion_datos, label='Señal original')
    plt.plot(tiempo_suavizado, aceleracion_suavizada, label='Señal filtrada')
    plt.title('Filtrado de altas frecuencias')
    plt.xlabel('Tiempo')
    plt.ylabel('Aceleracion')
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_senal_singular(datos, tiempo):
    plt.figure(figsize=(10, 6))
    plt.plot(tiempo, datos, label='Señal original')
    plt.xlabel('Tiempo')
    plt.ylabel('Aceleracion')
    plt.grid(True)
    plt.legend()
    plt.show()

def graficar_espectros_filtrados(freq1, tfd1, freq2, tfd2):
    pos_freq1 = freq1[freq1 >=0]
    pos_tfd1 = tfd1[freq1 >=0]
    pos_tfd1=np.abs(pos_tfd1)

    pos_freq2 = freq2[freq2 >=0]
    pos_tfd2 = tfd2[freq2 >=0]
    pos_tfd2=np.abs(pos_tfd2)

    plt.figure(figsize=(10, 6))
    plt.plot(pos_freq1, pos_tfd1, label='Espectro original')
    plt.plot(pos_freq2, pos_tfd2, label='Espectro suavizado')
    plt.title('Filtrado de altas frecuencias')
    plt.xlabel('Frecuencias (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.legend()
    plt.show()

def cargar_datos(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        datos = [linea.strip().split('\t') for linea in archivo.readlines()]
        datos = [(float(punto[0]), float(punto[1])) for punto in datos]
    return datos

def cargar_datos_terr3(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        datos=np.loadtxt(nombre_archivo)
    return datos

#Cargado de datos 2.0 (hace un array de pares (tiempo, aceleracion))
datos_terremoto1 = cargar_datos('terremoto1.txt')
datos_terremoto2 = cargar_datos('terremoto2.txt')

#Calcular los coeficientes de la SFD para cada conjunto de datos
coeficientes_sfd_terremoto1 = calcular_coeficientes(datos_terremoto1)
coeficientes_sfd_terremoto2 = calcular_coeficientes(datos_terremoto2)

#Calcula las frecuencias de la TFD
frecuencias_terremoto1 = frecuencias_tfd(100, datos_terremoto1)
frecuencias_terremoto2 = frecuencias_tfd(200, datos_terremoto2)

#Imprimir los coeficientes de Fourier
print("Coeficientes de la serie de Fourier para terremoto1: ", coeficientes_sfd_terremoto1)
print("\nCoeficientes de la serie de Fourier para terremoto2: ", coeficientes_sfd_terremoto2)

#Calcular la TFD para cada conjunto de coeficientes de la SFD
tfd_terremoto1 = calcular_tfd(coeficientes_sfd_terremoto1)
tfd_terremoto2 = calcular_tfd(coeficientes_sfd_terremoto2)

#Imprimir TFD
print("\nFrecuencias para terremoto1: ", tfd_terremoto1)
print("\nFrecuencias para terremoto2: ", tfd_terremoto2)

# Suavizar las altas frecuencias de los terremotos y graficarlo
datos_filtrados1 = filtrar(datos_terremoto1)
graficar_senal(datos_terremoto1, datos_filtrados1)
datos_filtrados2 = filtrar(datos_terremoto2)
graficar_senal(datos_terremoto2, datos_filtrados2)

#Frecuencia mas afectada
frecuencias_afectadas1 = freq_max(frecuencias_terremoto1, coeficientes_sfd_terremoto1)

coeficientes_sfd_terremoto1_suavizado = calcular_coeficientes(datos_filtrados1)
frecuencias_afectadadas1_suavizadas = freq_max(frecuencias_terremoto1, coeficientes_sfd_terremoto1_suavizado)

frecuencias_afectadas2 = freq_max(frecuencias_terremoto2, coeficientes_sfd_terremoto2)

coeficientes_sfd_terremoto2_suavizado = calcular_coeficientes(datos_filtrados2)
frecuencias_afectadadas2_suavizadas = freq_max(frecuencias_terremoto2, coeficientes_sfd_terremoto2_suavizado)

print("\nLa frecuencia mas afectada en el terremoto1 antes del filtrado fue de ", frecuencias_afectadas1, " Hz")
print("\nLa frecuencia mas afectada en el terremoto1 despues del filtrado fue de ", frecuencias_afectadadas1_suavizadas, " Hz")
print("\nLa frecuencia mas afectada en el terremoto2 antes del filtrado fue de ", frecuencias_afectadas2, " Hz")
print("\nLa frecuencia mas afectada en el terremoto2 despues del filtrado fue de ", frecuencias_afectadadas2_suavizadas, " Hz")

#graficado de la transformada
graficar_espectro_continuo(frecuencias_terremoto1, tfd_terremoto1, "terremoto 1")
graficar_espectro_continuo(frecuencias_terremoto2, tfd_terremoto2, "terremoto 2")
graficar_espectro_continuo(frecuencias_terremoto1, calcular_tfd(coeficientes_sfd_terremoto1_suavizado), "terremoto 1")
graficar_espectro_continuo(frecuencias_terremoto2, calcular_tfd(coeficientes_sfd_terremoto2_suavizado), "terremoto 2")

#comparacion de espectros
graficar_espectros_filtrados(frecuencias_terremoto1, tfd_terremoto1, frecuencias_terremoto1, calcular_tfd(coeficientes_sfd_terremoto1_suavizado))
graficar_espectros_filtrados(frecuencias_terremoto2, tfd_terremoto2, frecuencias_terremoto2, calcular_tfd(coeficientes_sfd_terremoto2_suavizado))

datos_terremoto3 = cargar_datos_terr3('terremoto3.txt')
correlacion_con1 = nivel_de_correlacion([p[1] for p in datos_terremoto1], [p[1] for p in datos_terremoto3])
correlacion_con2 = nivel_de_correlacion([p[1] for p in datos_terremoto2], [p[1] for p in datos_terremoto3])

if(np.max(correlacion_con1) > np.max(correlacion_con2)):
    print("\nEl detector 3 esta mas proximo al detector 1")
else:
    print("\nEl detector 3 esta mas proximo al detector 2")


#Primera solucion freq mas acelerada (SUMAS)
tfd_terremoto1_suavizados = calcular_tfd(coeficientes_sfd_terremoto1_suavizado)
datos_filtrados2_acortados= acortar_terr2(datos_filtrados2)
tfd_terremoto2_suavizados_acortado = calcular_tfd(calcular_coeficientes(datos_filtrados2_acortados))

#graficar_senal(aux, datos_filtrados2)

resultado_suma_tfd = suma_tfd(tfd_terremoto1_suavizados, tfd_terremoto2_suavizados_acortado)

graficar_espectro_continuo(frecuencias_terremoto1, resultado_suma_tfd, "Espectro de frecuencias de suma de tfds")

maximo_1 = freq_max(frecuencias_terremoto1, resultado_suma_tfd)
print("\nLa frecuencia mas acelerada de ambas senales es de: ", maximo_1, " Hz en la primera solucion")

#Segunda solucion freq mas acelerada (SUMAS)
suma_senales = []
for i in range(len(datos_filtrados1)):
    suma_senales.append(datos_filtrados1[i][1] + datos_filtrados2_acortados[i][1])

tiempo_suma = np.arange(len(suma_senales)) / 100
datos_suma = list(zip(tiempo_suma, suma_senales))
graficar_senal_singular(suma_senales, tiempo_suma)

coeficientes_sfd_suma = calcular_coeficientes(datos_suma)
#print("\nArreglo de coeficientes: ", coeficientes_sfd_convolucion)
#print("\nLong de arreglo de coeficientes: ", len(coeficientes_sfd_convolucion))
tfd_suma = calcular_tfd(coeficientes_sfd_suma)
graficar_espectro_continuo(frecuencias_terremoto1, tfd_suma, "Espectro de la suma de senales")

maximo_2 = freq_max(frecuencias_terremoto1, tfd_suma)
print("\nLa frecuencia mas acelerada de ambas senales es de: ", maximo_2, " Hz en la segunda solucion")

'''
#Primera solucion freq mas acelerada PRODUCTO
tfd_terremoto1_suavizados = calcular_tfd(coeficientes_sfd_terremoto1_suavizado)
datos_filtrados2_acortados= acortar_terr2(datos_filtrados2)
tfd_terremoto2_suavizados_acortado = calcular_tfd(calcular_coeficientes(datos_filtrados2_acortados))

resultado_prod_punto_tfd = prod_punto_tfd(tfd_terremoto1_suavizados, tfd_terremoto2_suavizados_acortado)

graficar_espectro_continuo(frecuencias_terremoto1, resultado_prod_punto_tfd, "Espectro de frecuencias de prod punto de tfds")
maximo_1 = freq_max(frecuencias_terremoto1, resultado_prod_punto_tfd)
print("\nLa frecuencia mas acelerada de ambas senales es de: ", maximo_1, " Hz en la primera solucion")

#Segunda solucion freq mas acelerada PRODUCTO
convolucion_senales = np.convolve([p[1] for p in datos_filtrados2_acortados], [p[1] for p in datos_filtrados1], mode='full')
convolucion_senales = np.concatenate(([0], convolucion_senales))
tiempo_convolucion = np.arange(len(convolucion_senales)) / 200
datos_convolucion = list(zip(tiempo_convolucion, convolucion_senales))
graficar_senal_singular(convolucion_senales, tiempo_convolucion)

coeficientes_sfd_convolucion = calcular_coeficientes(datos_convolucion)
tfd_convolucion = calcular_tfd(coeficientes_sfd_convolucion)
graficar_espectro_continuo(frecuencias_terremoto1, tfd_convolucion, "Espectro de la convolucion de senales")

maximo_2 = freq_max(frecuencias_terremoto1, tfd_convolucion)
print("\nLa frecuencia mas acelerada de ambas senales es de: ", maximo_2, " Hz en la segunda solucion")

'''