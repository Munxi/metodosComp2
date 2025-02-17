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
punto1()

data = pd.read_csv("H_field.csv")
t = data.iloc[:, 0].values
H = data.iloc[:, 1].values

dt = np.mean(np.diff(t))  # Intervalo de tiempo promedio

H_fft = np.fft.rfft(H)
freqs = np.fft.rfftfreq(len(H), dt)

f_fast = freqs[np.argmax(np.abs(H_fft))]

def Fourier(t, y, f):
    return np.sum(y * np.exp(-2j * np.pi * f * t))

f_general = freqs[np.argmax([np.abs(Fourier(t, H, f)) for f in freqs])]

phi_fast = (f_fast * t) % 1
phi_general = (f_general * t) % 1

plt.figure(figsize=(8, 6))
plt.scatter(phi_fast, H, s=5, label="ϕ_fast")
plt.scatter(phi_general, H, s=5, label="ϕ_general", alpha=0.6)
plt.xlabel("Fase")
plt.ylabel("H")
plt.legend()
plt.title("Comparación de fases con FFT y Transformada General")
plt.savefig("2.a.pdf")

print(f"2.a) {f_fast = :.5f}; {f_general = :.5f}")

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

def transformada_fourier(t, y, f):
    return np.sum(y * np.exp(-2j * np.pi * f * t))

rango_frec = np.linspace(0, frecuencias[-1], 1000)
magnitudes = np.array([np.abs(transformada_fourier(tiempo, campo, f)) for f in rango_frec])
indice_general = np.argmax(magnitudes)
f_general = rango_frec[indice_general]

print(f"2.a) f_rapida = {f_rapida:.5f}; f_general = {f_general:.5f}")

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
plt.show()

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

def procesar_imagen(ruta_imagen, proporcion_umbral, radio_notch, iteraciones_dilatacion, radio_central_x, radio_central_y, angulo=0, extra_elipse=False, radio_extra_x=20, radio_extra_y=15, angulo_extra=45):
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
    plt.show()

# Procesar la imagen del gato con la nueva elipse adicional configurada
procesar_imagen("catto.png", proporcion_umbral=0.65, radio_notch=1, iteraciones_dilatacion=1, radio_central_x=100, radio_central_y=10, angulo=20, extra_elipse=True, radio_extra_x=200, radio_extra_y=9, angulo_extra=50)

# Procesar la imagen del castillo sin cambios
procesar_imagen("Noisy_Smithsonian_Castle.jpg", proporcion_umbral=0.67, radio_notch=1, iteraciones_dilatacion=2, radio_central_x=70, radio_central_y=15, angulo=0, extra_elipse=False)

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

# Definir el filtro gaussiano
def gaussian_filter(f, alpha):
    return np.exp(-(f * alpha)**2)

# Valores de alpha a probar
alphas = [0, 5, 10, 15, 20, 25]

# Crear figura para los subplots
fig, axs = plt.subplots(len(alphas), 2, figsize=(12, 6 * len(alphas)))

# Iterar sobre los valores de alpha
for i, alpha in enumerate(alphas):
    # Aplicar el filtro gaussiano en el dominio de la frecuencia
    filter_values = gaussian_filter(freqs, alpha)

    # Multiplicar la señal por el filtro
    filtered_signal = signal * filter_values

    # Calcular la FFT de la señal filtrada
    fft_filtered =  np.fft.fft(filtered_signal)

    # Graficar la señal original y la señal filtrada en el dominio del tiempo
    axs[i, 0].plot(data.index, signal, label='Señal original')
    axs[i, 0].plot(data.index, filtered_signal, label=f'Señal filtrada (α={alpha})')
    axs[i, 0].set_title(f'Señal en el dominio del tiempo (α={alpha})')
    axs[i, 0].legend()

    # Graficar la FFT de la señal original y de la señal filtrada (en escala logarítmica)

    #axs[i, 1].plot(freqs_shifted, np.abs(fft_signal_shifted), label='FFT señal original')
    #axs[i, 1].plot(freqs_shifted, np.abs(np.fft.fftshift(fft_filtered)), label=f'FFT señal filtrada (α={alpha})')
    axs[i, 1].semilogy(freqs_shifted, np.abs(fft_signal_shifted), label='FFT señal original')
    axs[i, 1].semilogy(freqs_shifted, np.abs(np.fft.fftshift(fft_filtered)), label=f'FFT señal filtrada (α={alpha})')
    axs[i, 1].set_title(f'FFT de la señal (α={alpha})')
    axs[i, 1].legend()
    axs[i, 1].set_xlabel('Frecuencia')
    axs[i, 1].set_ylabel('Magnitud')

plt.tight_layout()
plt.savefig('3.1.pdf')
plt.show()
     