#graficos terremoto 1 y 2
import matplotlib.pyplot as plt
import numpy as np
# Cargar los datos desde el archivo txt
datos = np.loadtxt("terremoto2.txt")

# Extraer las coordenadas x e y de los datos
x = datos[:, 0]  # Primer columna
y = datos[:, 1]  # Segunda columna

plt.plot(x, y)
plt.title('Gráfico de terremoto2')
plt.xlabel('Tiempo')
plt.ylabel('Aceleración')
plt.grid(True)
plt.show()