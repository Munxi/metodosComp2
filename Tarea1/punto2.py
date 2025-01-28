import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re

with open('hysteresis.dat', 'r') as archivo:
    lineas = archivo.readlines()

with open('datos_procesados.txt', 'w') as nuevo_archivo:
    for linea in lineas:
        nueva_linea = linea
        nueva_linea=re.sub(r'(-?\d+\.\d{3})(?=\S)', r'\1 ', nueva_linea)
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


def calcular_frecuencia_dominante(x, y):
    
    fft_result = np.fft.fft(y)
    frecuencias = np.fft.fftfreq(len(y), d=(x[1] - x[0]))
    magnitud = np.abs(fft_result)

    frecuencias_positivas = frecuencias[:len(frecuencias)//2]
    magnitud_positiva = magnitud[:len(magnitud)//2]

    indice_pico = np.argmax(magnitud_positiva)
    frecuencia_dominante = frecuencias_positivas[indice_pico]

    plt.figure(figsize=(10, 6))
    plt.plot(frecuencias_positivas, magnitud_positiva, 'b-')
    plt.title('Espectro de Frecuencias (FFT)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid()
    plt.show()

    return frecuencia_dominante

# Calcular la frecuencia dominante para los datos cargados
frecuencia_dominante_B = calcular_frecuencia_dominante(t,B)
print(f"La frecuencia dominante de B vs t es:  {frecuencia_dominante_B:.3f} Hz")

