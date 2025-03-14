# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lpSefBSJid64W23JGNGvCtFuIG0Tlj_O
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import scipy
from scipy.ndimage import label, binary_dilation, generate_binary_structure
import pandas as pd
import cv2


def punto1():
    def datos_prueba(t_max: float, dt: float, amplitudes: NDArray[float], frecuencias: NDArray[float],
                     ruido: float = 0.0) -> NDArray[float]:
        ts = np.arange(0., t_max, dt)
        ys = np.zeros_like(ts, dtype=float)
        for A, f in zip(amplitudes, frecuencias):
            ys += A * np.sin(2 * np.pi * f * ts)
            ys += np.random.normal(loc=0, size=len(ys), scale=ruido) if ruido else 0
        return ts, ys

    def Fourier(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
        N = len(t)
        f_hat = []
        for i, f_k in enumerate(f):
            f_hat.append(np.sum(np.multiply(y, np.exp(-2j * np.pi * f_k * t)), dtype=complex))
        return f_hat
    def puntoa():
        t_total = 5
        t_diff = 0.01
        amplitudes = [10, 20, 15]
        frecuencias = [4, np.pi, np.sqrt(2)]
        fs = np.arange(0, 1 / (2 * t_diff), 1 / (10 * t_total))
        t_clean, y_clean = datos_prueba(t_total, t_diff, np.array(amplitudes), np.array(frecuencias))
        t_noise, y_noise = datos_prueba(t_total, t_diff, np.array(amplitudes), np.array(frecuencias), 20)
        #1.a)
        fig, axs = plt.subplots(2, figsize=(20, 10))
        axs[0].set_title('Clean DFT')
        axs[1].set_title('Noisy DFT')
        for i in range(2):
            axs[i].axis([0, 4.5, None, None])
            axs[i].axvline(4, color='green', label="frequence = 4")
            axs[i].axvline(np.sqrt(2), color='red', label="frequence = sqrt(2)")
            axs[i].axvline(np.pi, color='yellow', label="frequence = pi")
        axs[0].plot(fs, np.abs(Fourier(t_clean, y_clean, fs)) ** 2)
        axs[1].plot(fs, np.abs(Fourier(t_noise, y_noise, fs)) ** 2)
        axs[0].legend(loc='best', bbox_to_anchor=(1, 1))
        axs[1].legend(loc='best', bbox_to_anchor=(1, 1))
        fig.savefig('1.a.pdf')
        print("1.a) Los picos correctos presentan un corrimiento, ensanchamiento y la transformada se distorsiona")
    def puntob():
        def func(x, a, c):
            return 1 + c * np.exp(-a * x / t_diff)
        var = np.arange(10, 300)
        widths = np.zeros_like(var, dtype=np.double)
        t_diff = 0.01
        for i, t in enumerate(var):
            t_b, y_b = datos_prueba(t, t_diff, np.array([5]), np.array([2]))
            fs = np.arange(0, 3, t_diff)
            f_bhat = np.abs(Fourier(t_b, y_b, fs)) ** 2
            widths[i] = scipy.signal.peak_widths(f_bhat, [np.argmax(f_bhat)], rel_height=0.5)[0][0]
        popt, pcov = scipy.optimize.curve_fit(func, var, widths, p0=(2e-4,2e1))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        yy = func(var, *popt)
        ax.set_title("FWHM en función del tiempo máximo(espaciado:0.05)(N: tamaño muestra)")
        ax.set_ylabel("FWHM")
        ax.set_xlabel("t_max")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.scatter(var, widths, c='blue',s=3 ,label="calculated")
        ax.scatter(var, yy, c='cyan', s=1, label="fit: 1+c*exp(-a*N)")
        ax.legend()
        fig.savefig('1.b.pdf')
    def puntoc():
        t, y, sigma = np.genfromtxt('punto1.dat').T
        f_nyq = 1/(2*np.average(np.diff(t)))
        y = y - np.average(y)
        fs = np.arange(0, 9, 1 / (5 * (t[-1] - t[0])))
        f_hat = Fourier(t, y, fs)
        freq = fs[np.argmax(np.abs(f_hat) ** 2)]
        phi = np.mod(freq * t, 1)
        fig = plt.figure()
        print("1.c) f Nyquist: {:.3f}".format(f_nyq))
        print("1.c) f true: {:.3f}".format(freq))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("y vs phi")
        ax.set_ylabel("y")
        ax.set_xlabel("phi")
        ax.scatter(phi, y, c='red', s=3)
        fig.savefig('1.c.pdf')
    puntoa()
    puntob()
    puntoc()
def punto2():
    # Función para extrapolar evaluando la serie de Fourier
    def extrapolate_fourier(X_trunc, t_eval, N, M):
        # Índices de los armónicos: 0 a M y los correspondientes negativos
        pos_indices = np.arange(0, M+1)
        neg_indices = np.arange(N - (M-1), N) if M > 1 else np.array([], dtype=int)
        k_indices = np.concatenate((pos_indices, neg_indices))
        X_reduced = X_trunc[k_indices]
        # y(t) = (1/N) * sum_{k in k_indices} X[k] exp(2πi * k * t / N)
        exponent = 2j * np.pi * np.outer(t_eval, k_indices) / N
        y_extrap = np.dot(np.exp(exponent), X_reduced) / N
        return y_extrap.real

    def load_sunspot_data(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Buscar la línea donde empieza la tabla (se asume que la primera línea con "Year" es el encabezado)
        start_index = next(i for i, line in enumerate(lines) if line.strip().startswith("Year"))
        columns = ["Year", "Month", "Day", "SSN"]
        data_list = []
        for line in lines[start_index+1:]:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    data_list.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
                except ValueError:
                    continue
        df = pd.DataFrame(data_list, columns=columns)
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df = df.sort_values("Date")
        return df

    def transformada_fourier(t, y, f):
            return np.sum(y * np.exp(-2j * np.pi * f * t))

    def puntoa():
        datos = pd.read_csv('H_field.csv')
        datos.columns = ['tiempo', 'campo']

        tiempo = datos['tiempo'].values
        campo = datos['campo'].values

        delta_t = np.mean(np.diff(tiempo))
        n = len(tiempo)

        n_fft = 100 * n  # Incremento de resolución al 400%
        fft_campo = np.fft.rfft(campo, n=n_fft)
        frecuencias = np.fft.rfftfreq(n_fft, delta_t)

        indice_max = np.argmax(np.abs(fft_campo[1:])) + 1
        f_rapida = frecuencias[indice_max]

        rango_frec = np.linspace(0, frecuencias[-1], 1000)
        magnitudes = np.array([np.abs(transformada_fourier(tiempo, campo, f)) for f in rango_frec])
        indice_general = np.argmax(magnitudes)
        f_general = rango_frec[indice_general]

        print(f"2.a) f_fast = {f_rapida:.5f}; f_general = {f_general:.5f}")

        fase_rapida = np.mod(f_rapida * tiempo, 1)
        fase_general = np.mod(f_general * tiempo, 1)

        plt.figure(figsize=(10,5))
        plt.scatter(fase_rapida, campo, label='Campo vs Fase rápida', alpha=0.5)
        plt.scatter(fase_general, campo, label='Campo vs Fase general', alpha=0.5)
        plt.xlabel('Fase')
        plt.ylabel('Campo')
        plt.legend()
        plt.title('Comparación de Campo vs Fase')
        plt.grid()
        plt.savefig('2.a.pdf')

    def puntob():
        # Configuraciones
        data_file = "list_aavso-arssn_daily.txt"
        fecha_limite_str = "2010-10-10"       # Fecha hasta la que se usan los datos históricos
        fecha_extrapolacion_str = "2025-02-12"  # Fecha hasta la que se quiere extrapolar

        # Cargar y filtrar datos
        data = load_sunspot_data(data_file)
        fecha_limite = pd.to_datetime(fecha_limite_str)
        data = data[data["Date"] <= fecha_limite]

        # Extraer la serie de tiempo
        N = len(data)
        t = np.arange(N)
        y = data["SSN"].values

        # Aplicar FFT completa
        X_fft = np.fft.fft(y)

        # Calcular la frecuencia dominante (ignorando la componente DC) y su período en años
        freqs_full = np.fft.fftfreq(N, d=1)  # en ciclos por día
        dominant_index = np.argmax(np.abs(X_fft[1:N//2])) + 1
        dominant_freq = np.abs(freqs_full[dominant_index])
        P_solar = 1 / dominant_freq / 365  # conversión de días a años
        print(f'2.b.a) {{P_solar = {P_solar:.2f} años}}')

        # Truncar los coeficientes: usar los primeros M armónicos
        M = 50
        X_fft_trunc = np.zeros_like(X_fft, dtype=complex)
        # Se conserva el componente DC y las M frecuencias positivas
        X_fft_trunc[:M+1] = X_fft[:M+1]
        # Se conservan las correspondientes frecuencias negativas (excepto DC)
        if M > 1:
            X_fft_trunc[-(M-1):] = X_fft[-(M-1):]

        # Reconstruir la señal ajustada en el intervalo original
        y_fit = np.fft.ifft(X_fft_trunc).real

        # Preparar extrapolación
        fecha_inicio_extrapolacion = data["Date"].max()
        fecha_extrapolacion = pd.to_datetime(fecha_extrapolacion_str)
        # Se suma 1 para incluir el último día en el rango de extrapolación
        dias_extrapolacion = (fecha_extrapolacion - fecha_inicio_extrapolacion).days + 1

        t_future = np.arange(N, N + dias_extrapolacion)
        y_extrap = extrapolate_fourier(X_fft_trunc, t_future, N, M)

        # Predicción del número de manchas solares en la fecha final
        n_manchas_hoy = y_extrap[-1]
        print(f'2.b.b) {{n_manchas_hoy = {n_manchas_hoy:.2f}}}')

        # Crear rango de fechas para la extrapolación
        future_dates = pd.date_range(start=fecha_inicio_extrapolacion, periods=dias_extrapolacion, freq="D")

        # Graficar: datos originales, ajuste en el intervalo conocido y extrapolación
        plt.figure(figsize=(20, 5), dpi=100)
        plt.scatter(data["Date"], y, s=2, color='green', alpha=0.6, label="Datos originales")
        plt.plot(data["Date"], y_fit, color='black', linewidth=2, label="Ajuste FFT en datos históricos")
        plt.plot(future_dates, y_extrap, color="red", linewidth=2, linestyle='dashed',
                label=f"Predicción FFT hasta {fecha_extrapolacion.strftime('%Y-%m-%d')}")
        # Asegurarse de que el eje x abarque desde la primera fecha hasta la fecha final de extrapolación
        plt.xlim(data["Date"].min(), fecha_extrapolacion)
        plt.xlabel("Fecha", fontsize=14)
        plt.ylabel("Número de manchas solares", fontsize=14)
        plt.title(f"Extrapolación del ciclo solar con FFT hasta {fecha_extrapolacion.strftime('%Y-%m-%d')}", fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("2.b.pdf")
    puntoa()
    puntob()
def punto3():
    # Definir el filtro gaussiano
    def gaussian_filter(f, alpha):
        return np.exp(-(f * alpha)**2)

    def detectar_picos_de_ruido_adaptativo(espectro, proporcion_umbral=0.6, iteraciones_dilatacion=2):
        """Detecta picos de ruido en el espectro de frecuencia usando umbralización adaptativa y dilatación."""
        max_val = np.max(espectro)
        mascara_binaria = espectro > (proporcion_umbral * max_val)

        estructura = generate_binary_structure(2, 2)
        mascara_dilatada = binary_dilation(mascara_binaria, structure=estructura, iterations=iteraciones_dilatacion)

        etiquetado, num_caracteristicas = label(mascara_dilatada)
        return np.argwhere(etiquetado > 0), mascara_dilatada

    def eliminar_ruido_periodico_adaptativo(img, proporcion_umbral=0.6, radio_notch=10, iteraciones_dilatacion=2,
                                            radio_central_x=30, radio_central_y=10, angulo=15, extra_elipse=False,
                                            radio_extra_x=20, radio_extra_y=15, angulo_extra=45):
        """Elimina el ruido periódico de una imagen mediante filtrado notch adaptativo, excluyendo el centro."""
        filas, columnas = img.shape
        centro = (columnas // 2, filas // 2)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        espectro_magnitud = np.log1p(np.abs(fshift))

        picos_de_ruido, mascara_dilatada = detectar_picos_de_ruido_adaptativo(espectro_magnitud, proporcion_umbral, iteraciones_dilatacion)
        mascara_notch = np.ones((filas, columnas), np.float32)

        for pico in np.argwhere(mascara_dilatada):
            cv2.circle(mascara_notch, tuple(pico[::-1]), radio_notch, 0, -1)

        # Primera elipse protectora
        cv2.ellipse(mascara_notch, centro, (radio_central_x, radio_central_y), angulo, 0, 360, 1, -1)

        # Segunda elipse adicional para la imagen del gato
        if extra_elipse:
            cv2.ellipse(mascara_notch, centro, (radio_extra_x, radio_extra_y), angulo_extra, 0, 360, 1, -1)

        fshift_filtrado = fshift * mascara_notch
        f_ishift = np.fft.ifftshift(fshift_filtrado)
        img_filtrada = np.abs(np.fft.ifft2(f_ishift))

        return img_filtrada, espectro_magnitud, np.log1p(np.abs(fshift_filtrado)), mascara_notch
    def procesar_imagen(ruta_imagen,nombre, proporcion_umbral, radio_notch, iteraciones_dilatacion, radio_central_x, radio_central_y, angulo=0, extra_elipse=False, radio_extra_x=20, radio_extra_y=15, angulo_extra=45):
        """Procesa una imagen para eliminar el ruido periódico."""
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
            return

        img_filtrada, espectro_original, espectro_filtrado, mascara_notch = eliminar_ruido_periodico_adaptativo(
            img, proporcion_umbral, radio_notch, iteraciones_dilatacion, radio_central_x, radio_central_y, angulo, extra_elipse, radio_extra_x, radio_extra_y, angulo_extra
        )

        fig, ejes = plt.subplots(2, 3, figsize=(18, 12))
        ejes[0, 0].imshow(img, cmap='gray')
        ejes[0, 0].set_title("Imagen Original")
        ejes[0, 1].imshow(espectro_original, cmap='inferno')
        ejes[0, 1].set_title("Espectro de Magnitud (Original)")
        ejes[0, 2].imshow(mascara_notch, cmap='gray')
        ejes[0, 2].set_title("Máscara Notch con Exclusión Central")
        ejes[1, 0].imshow(espectro_filtrado, cmap='inferno')
        ejes[1, 0].set_title("Espectro Filtrado")
        ejes[1, 1].imshow(img_filtrada, cmap='gray')
        ejes[1, 1].set_title("Imagen Filtrada")
        ejes[1, 2].axis('off')

        for eje in ejes.flat:
            eje.axis("off")
        plt.savefig(nombre)

    def puntoa():
        # Cargar datos
        data = pd.read_csv("list_aavso-arssn_daily.txt", sep='\s+', skiprows=2, names=['Year', 'Month', 'Day', 'SSN'])

        # Convertir a formato de fecha
        data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
        data.set_index('Date', inplace=True)

        # Convertir SSN a numérico y eliminar valores no válidos
        data['SSN'] = pd.to_numeric(data['SSN'], errors='coerce')
        data = data.dropna(subset=['SSN'])

        # Obtener señal
        signal = data['SSN'].values
        N = len(signal)
        dt = 1  # Intervalo de tiempo diario

        # Calcular FFT de la señal
        fft_signal = np.fft.fft(signal)

        # Calcular frecuencias correspondientes a la FFT
        freqs = np.fft.fftfreq(N, dt)

        # Aplicar fftshift para centrar la FFT
        freqs_shifted = np.fft.fftshift(freqs)
        fft_signal_shifted = np.fft.fftshift(fft_signal)

        # Valores de alpha a probar
        alphas = [0.5, 1.5, 2.5, 3.5, 4.5]

        # Crear figura para los subplots
        fig, axs = plt.subplots(len(alphas), 2, figsize=(12, 6 * len(alphas)))

        # Iterar sobre los valores de alpha
        for i, alpha in enumerate(alphas):
            # Aplicar el filtro gaussiano en el dominio de la frecuencia
            filter_values = gaussian_filter(freqs, alpha)

            # Multiplicar la señal por el filtro
            filtered_signal = np.fft.ifft(fft_signal_shifted * filter_values)

            # Graficar la señal original y la señal filtrada en el dominio del tiempo
            axs[i, 0].plot(data.index, signal, label='Señal original')
            axs[i, 0].plot(data.index, filtered_signal, label=f'Señal filtrada (α={alpha})')
            axs[i, 0].set_title(f'Señal en el dominio del tiempo (α={alpha})')
            axs[i, 0].legend()

            # Graficar la FFT de la señal original y de la señal filtrada (en escala logarítmica)

            axs[i, 1].plot(freqs_shifted, np.abs(fft_signal_shifted), label='FFT señal original')
            axs[i, 1].plot(freqs_shifted, np.abs(np.fft.fft(filtered_signal)), label=f'FFT señal filtrada (α={alpha})')
            axs[i, 1].set_title(f'FFT de la señal (α={alpha})')
            axs[i, 1].legend()
            axs[i, 1].set_xlabel('Frecuencia')
            axs[i, 1].set_ylabel('Magnitud')
        plt.tight_layout()
        plt.savefig('3.1.pdf')
    def puntob():
        # Procesar la imagen del gato con la nueva elipse adicional configurada
        procesar_imagen("catto.png","3.b.a.png" ,proporcion_umbral=0.65, radio_notch=1, iteraciones_dilatacion=1, radio_central_x=100, radio_central_y=10, angulo=20, extra_elipse=True, radio_extra_x=200, radio_extra_y=9, angulo_extra=50)

        # Procesar la imagen del castillo sin cambios
        procesar_imagen("Noisy_Smithsonian_Castle.jpg","3.b.b.png", proporcion_umbral=0.67, radio_notch=1, iteraciones_dilatacion=2, radio_central_x=70, radio_central_y=15, angulo=0, extra_elipse=False)
    puntoa()
    puntob()


punto1()
punto2()
punto3()