import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import re
import pandas as pd

def gauss(x, A, B):
    return A * np.exp(-0.5*(x)**2/B**2)
def gauss2(x, A, B,C):
    return A * np.exp(-0.5 * (x-C) ** 2 / B ** 2)
def punto1():
    fig_a, ax_a = plt.subplots(1,2,figsize=(10,5))
    ax_a[0].set_title("Rhodium Xray data")
    ax_a[0].set_xlabel("Wavelength (pm)")
    ax_a[0].set_ylabel("Intensity (mJy)")
    ax_a[1].set_title("Rhodium Xray data filtered")
    ax_a[1].set_xlabel("Wavelength (pm)")
    ax_a[1].set_ylabel("Intensity (mJy)")
    def cargaDatos():
        x,y = np.genfromtxt('Rhodium.csv',dtype = float,delimiter=',',skip_header=1,unpack=True)
        promx = 0
        for i in range(1,len(x)):
            promx+=x[i]-x[i-1]
        promx = promx/(len(x)-1)
        ax_a[0].scatter(x,y,s=5)
        return x,y,promx
    def limpiezaDatos():
        j = 0
        x,y,promx = cargaDatos()
        x1 = x[:3]
        y1 = y[:3]
        totalx = len(x)
        for i in range(2,totalx-2):
            dx1 = x[i-1] - x[i-2]
            dy1 = y[i-1] - y[i-2]
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            dx2 = x[i+2] - x[i+1]
            dy2 = y[i+2] - y[i+1]
            if (abs(dy1/dx1) + abs(dy2/dx2))*dx/2 >= abs(dy/dx)*promx/7:
                x1 = np.append(x1,[x[i]])
                y1 = np.append(y1,[[y[i]]])
            else:
                j+=1
        ax_a[1].scatter(x1,y1,s=5)
        print(f'1.a) Número de datos eliminados: {j}')
        fig_a.savefig("limpieza.pdf", format="pdf", bbox_inches="tight")
        return x1,y1
    def dosPicos():
        x1,y1 = limpiezaDatos()
        imax = np.argmax(y1)
        i = imax
        while((y1[i+1]-y1[i-1])/(x1[i+1]-x1[i-1])<0):
            i+=1
        imax2 = np.argmax(y1[i:]) + i
        i2 = imax2
        while((y1[i2-1]-y1[i2+1])/abs((x1[i2-1]-x1[i2+1]))<0):
            i2-=1
        avg = 0
        for j in range(i,i2):
            dy = y1[j] - y1[j-1]
            dx = x1[j] - x1[j-1]
            avg += abs(dy / dx) if dx !=0 else 0
        avg /= i2-i
        right = True
        left = True
        r = imax2
        l = imax
        while(right or left):
            if(right):
                r+=1
                dy = y1[r+1] - y1[r]
                dx = x1[r+1] - x1[r]
                right = True if abs(dy/dx)>=avg/4 else False
            if(left):
                l-=1
                dy = y1[l+1] - y1[l]
                dx = x1[l+1] - x1[l]
                left = True if abs(dy/dx) >= avg/7 else False
        fig_b, ax_b = plt.subplots()
        ax_b.set_title("Rhodium Xray peaks")
        ax_b.set_xlabel("Wavelength (pm)")
        ax_b.set_ylabel("Intensity (mJy)")
        ax_b.scatter(x1[l:i + 1], y1[l:i + 1], s=5)
        ax_b.scatter(x1[i2:r + 1], y1[i2:r + 1], s=5)
        fig_b.savefig("picos.pdf", format="pdf", bbox_inches="tight")
        print('1.b) Método: Tolerancia valor de las derivadas, partiendo de los picos')
        return i,i2,imax,imax2,r,l,x1,y1
    def localizacion():
        i,i2,imax,imax2,r,l,x1,y1 = dosPicos()
        ###pico1:
        a = np.min(y1[l:i+1])
        b = x1[imax]
        para,covariance = curve_fit(gauss, x1[l:i + 1]-b, y1[l:i + 1]-a)
        ###pico2:
        a2 = np.min(y1[i2:r+1])
        b2 = x1[imax2]
        para2, covariance = curve_fit(gauss, x1[i2:r + 1] - b2, y1[i2:r + 1] - a2)
        ###fondo:
        ifmax = np.argmax(y1[:l])
        h = (y1[ifmax] - y1[0]) / 2
        for j in range(0, ifmax):
            w1 = j if j == 0 or abs(y1[w1] - h) > abs(y1[j] - h) else w1
        for k in range(r, len(x1)):
            w2 = k if k == r or abs(y1[w2] - h) > abs(y1[k] - h) else w2
        wfondo = x1[w2] - x1[w1]
        wpico1 = 2*np.sqrt(2*np.log(2))*para[1]
        wpico2 = 2*np.sqrt(2*np.log(2))*para2[1]
        print("1.c)")
        table = [["Fenomeno","Posición x maximo(pm)", "FWHM(pm)"],["Pico de la izquierda",x1[imax],round(float(wpico1),4)],["Pico de la derecha",x1[imax2],round(float(wpico2),4)]
                 ,["Fondo",x1[ifmax],round(float(wfondo),4)]]
        print(tabulate(table,headers="firstrow"))
        return x1,y1
    def integracion():
        x1,y1 = localizacion()
        x1 = x1*10**-12
        y1= y1*10**-29
        up = y1*(1.002)
        down = y1*(0.998)
        E = integrate.simpson(y1,x1)
        print("1.d) E = (",round(E*10**43,4),"+/-",round((E - integrate.simpson(down,x1))*10**43,4),")x10^-43 J/m")
    integracion()
def punto2():
    with open('hysteresis.dat', 'r') as archivo:
        lineas = archivo.readlines()

    with open('datos_procesados.txt', 'w') as nuevo_archivo:
        for linea in lineas:
            nueva_linea = linea
            nueva_linea = re.sub(r'(-?\d+\.\d{3})(?=\S)', r'\1 ', nueva_linea)
            nuevo_archivo.write(nueva_linea)

    datos = np.loadtxt('datos_procesados.txt')

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

        frecuencias_positivas = frecuencias[:len(frecuencias) // 2]
        magnitud_positiva = magnitud[:len(magnitud) // 2]

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
    frecuencia_dominante_B = calcular_frecuencia_dominante(t, B)
    print(f"2.b) La frecuencia dominante de B vs t es:  {frecuencia_dominante_B:.3f} Hz")
def punto2c():
    def procesamiento_archivo_histeresis(nombre_archivo):
        datos_corregidos = []
        with open(nombre_archivo, 'r') as archivo:
            for linea in archivo:
                coincidencias = re.findall(r'[-+]?\d*\.?\d+', linea)
                if len(coincidencias) == 3:  # (t,B,H)
                    datos_corregidos.append([float(valor) for valor in coincidencias])
        return pd.DataFrame(datos_corregidos,
                            columns=["Tiempo (ms)", "Campo externo (B) (mT)", "Densidad campo interno (H) (A/m)"])

    nombre_archivo = 'hysteresis.dat'

    df = procesamiento_archivo_histeresis(nombre_archivo)

    energy_loss = abs(np.trapezoid(df["Densidad campo interno (H) (A/m)"], df["Campo externo (B) (mT)"]))

    # Crear el gráfico de H como función de B
    plt.figure(figsize=(10, 6))
    plt.plot(df["Campo externo (B) (mT)"], df["Densidad campo interno (H) (A/m)"], label="Ciclo de histéresis",
             linewidth=2)
    plt.xlabel("Campo externo (B) (mT)", fontsize=12)
    plt.ylabel("Densidad campo interno (H) (A/m)", fontsize=12)
    plt.title("Ciclo de histéresis", fontsize=14)
    plt.grid(True)
    plt.legend()

    # Guardar el gráfico como PDF (Colab)
    plt.savefig("energy.pdf")

    plt.show()

    print(f"2.c) Energía perdida por unidad de volumen: {energy_loss:.6f} J/m³")

punto1()
punto2()
punto2c()

