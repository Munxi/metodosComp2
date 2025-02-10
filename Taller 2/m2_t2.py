import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("H_field.csv")
t = data.iloc[:, 0].values  # Suponemos que la primera columna es el tiempo
H = data.iloc[:, 1].values  # Suponemos que la segunda columna es el campo H

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

file_path = "list_aavso-arssn_daily.txt"  # Asegúrate de tener el archivo en la misma carpeta

with open(file_path, 'r') as f:
    lines = f.readlines()

start_index = next(i for i, line in enumerate(lines) if line.strip().startswith("Year"))

columns = ["Year", "Month", "Day", "SSN"]
data = []
for line in lines[start_index+1:]:
    parts = line.strip().split()
    if len(parts) == 4:
        data.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])

df = pd.DataFrame(data, columns=columns)

df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

df = df[df["Date"] < "2012-01-01"]

df = df.sort_values("Date")

t = np.arange(len(df))
y = df["SSN"].values

Y = rfft(y)
frequencies = rfftfreq(len(y), d=1)  # Frecuencias en ciclos por día

dominant_idx = np.argmax(np.abs(Y[1:])) + 1  # Omitir f_0
f_solar = frequencies[dominant_idx]
P_solar = 1 / f_solar  # Periodo en días
P_solar_years = P_solar / 365.25  #años

print(f"2.b.a) {P_solar_years:.2f} años")

plt.figure(figsize=(8, 6))
plt.loglog(frequencies[1:], np.abs(Y[1:]), label="Transformada de Fourier")
plt.axvline(f_solar, color='r', linestyle='--', label=f'Periodo Solar: {P_solar_years:.2f} años')
plt.xlabel("Frecuencia (ciclos por día)")
plt.ylabel("Magnitud")
plt.title(f"Transformada de Fourier de las manchas solares\nPeriodo Solar: {P_solar_years:.2f} años")
plt.legend()
plt.grid()
plt.savefig("2.b.a.pdf")
plt.show()

future_days = (pd.Timestamp("2025-02-10") - df["Date"].iloc[-1]).days
t_future = np.arange(len(y), len(y) + future_days)

harmonics = 10
Y_filtered = np.zeros_like(Y, dtype=complex)
Y_filtered[:harmonics] = Y[:harmonics]  #primeros 10 armónicos

y_pred = irfft(Y_filtered, n=len(y) + future_days)

n_manchas_hoy = y_pred[-1]

print(f"2.b.b) {n_manchas_hoy:.2f} manchas solares")

plt.figure(figsize=(10, 6))
plt.plot(df["Date"], y, label="Datos Históricos", alpha=0.7)
plt.plot(pd.date_range(df["Date"].iloc[-1], periods=future_days, freq='D'),
         y_pred[len(y):], label="Predicción (10 armónicos)", linestyle='--', color='red')
plt.xlabel("Fecha")
plt.ylabel("Número de manchas solares")
plt.title(f"Predicción de manchas solares\nPredicción para 10/Feb/2025: {n_manchas_hoy:.2f} manchas solares")
plt.legend()
plt.grid()
plt.savefig("2.b.pdf")
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation, generate_binary_structure

def detectar_picos_de_ruido_adaptativo(espectro, proporcion_umbral=0.6, iteraciones_dilatacion=2):
    """Detecta picos de ruido en el espectro de frecuencia usando umbralización adaptativa y dilatación."""
    max_val = np.max(espectro)
    mascara_binaria = espectro > (proporcion_umbral * max_val)

    estructura = generate_binary_structure(2, 2)
    mascara_dilatada = binary_dilation(mascara_binaria, structure=estructura, iterations=iteraciones_dilatacion)

    etiquetado, num_caracteristicas = label(mascara_dilatada)
    return np.argwhere(etiquetado > 0), mascara_dilatada

def eliminar_ruido_periodico_adaptativo(img, proporcion_umbral=0.6, radio_notch=10, iteraciones_dilatacion=2, radio_central_x=30, radio_central_y=10, angulo=15):
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

    cv2.ellipse(mascara_notch, centro, (radio_central_x, radio_central_y), angulo, 0, 360, 1, -1)

    fshift_filtrado = fshift * mascara_notch
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_filtrada = np.abs(np.fft.ifft2(f_ishift))

    return img_filtrada, espectro_magnitud, np.log1p(np.abs(fshift_filtrado)), mascara_notch

def procesar_imagen(ruta_imagen, proporcion_umbral, radio_notch, iteraciones_dilatacion, radio_central_x, radio_central_y, angulo=0):
    """Procesa una imagen para eliminar el ruido periódico."""
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    img_filtrada, espectro_original, espectro_filtrado, mascara_notch = eliminar_ruido_periodico_adaptativo(
        img, proporcion_umbral, radio_notch, iteraciones_dilatacion, radio_central_x, radio_central_y, angulo
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

procesar_imagen("catto.png", proporcion_umbral=0.65, radio_notch=1, iteraciones_dilatacion=1, radio_central_x=70, radio_central_y=10, angulo=15)  # Elipse inclinada
procesar_imagen("Noisy_Smithsonian_Castle.jpg", proporcion_umbral=0.67, radio_notch=1, iteraciones_dilatacion=2, radio_central_x=50, radio_central_y=15, angulo=0)  # Círculo
