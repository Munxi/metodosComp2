import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
import re

with open('hysteresis.dat', 'r') as archivo:
    lineas = archivo.readlines()

with open('datos_procesados.txt', 'w') as nuevo_archivo:
    for linea in lineas:
        nueva_linea = linea.replace('-', ' ')
        nueva_linea=re.sub(r'(\d+\.\d{3})(?=\S)', r'\1 ', nueva_linea)
        nuevo_archivo.write(nueva_linea)

datos=np.loadtxt('datos_procesados.txt')

t = datos[:, 0]
B = datos[:, 1]
H = datos[:, 2]

with PdfPages('histerico.pdf') as pdf:

    # Primer gráfico: H vs t
    plt.figure()
    plt.plot(t, H, label='H vs t', color='b', marker='o')
    plt.xlabel('t')
    plt.ylabel('H')
    plt.title('Gráfico: H vs t')
    plt.legend()
    pdf.savefig()  # Guardar la gráfica en el PDF
    plt.close()

    # Segundo gráfico: B vs t
    plt.figure()
    plt.plot(t, B, label='B vs Columna t', color='r', marker='x')
    plt.xlabel('t')
    plt.ylabel('B')
    plt.title('Gráfico: B vs t')
    plt.legend()
    pdf.savefig()
    plt.close()

picos, _ = find_peaks(B)
t_picos = t[picos]
periodos = np.diff(t_picos)
periodo_medio = np.mean(periodos)
frequency = 1 / (periodo_medio / 1000)

print(f" Frecuencia de la señal: {frequency:.2f} Hz")
print("""

1. Utiliza la función `find_peaks(B)` para detectar los índices de los picos en el arreglo `B`. Los índices de los picos se almacenan en `picos`.
   
2. Luego, usando esos índices (`picos`), se seleccionan los tiempos correspondientes de los picos en el arreglo `t`, y se almacenan en `t_picos`.

3. Con la función `np.diff(t_picos)`, se calculan las diferencias entre los tiempos de los picos consecutivos, obteniendo así los períodos (intervalos de tiempo) entre los picos.

4. El período medio se obtiene con `np.mean(periodos)`, calculando el promedio de los períodos obtenidos en el paso anterior.

5. Finalmente, la frecuencia se calcula tomando el recíproco del período medio (convirtiéndolo a segundos dividiendo entre 1000), usando la fórmula `frecuencia = 1 / (periodo_medio / 1000)`.

""")