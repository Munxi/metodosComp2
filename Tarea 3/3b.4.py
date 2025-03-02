!pip install numba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from IPython.display import HTML

# Parámetros
x_max = 1.0  # Ancho del tanque (metros)
y_max = 2.0  # Longitud del tanque (metros)
Nx = 100  # Número de puntos en el eje x
Ny = 200  # Número de puntos en el eje y
dx = x_max / Nx  # Espaciado en x
dy = y_max / Ny  # Espaciado en y
c = 0.5  # Velocidad de las ondas (m/s)
c_lente = c / 5  # Velocidad de las ondas en la abertura (m/s)
dt = 0.001  # Paso de tiempo (segundos)
T = 2.0  # Tiempo final de la simulación (segundos)
M = int(T / dt)  # Número total de pasos de tiempo

# Coordenadas del dominio
x = np.linspace(0, x_max, Nx)
y = np.linspace(0, y_max, Ny)
X, Y = np.meshgrid(x, y)

# Definir la velocidad de las ondas en el dominio (optimizada con Numba)
@njit
def velocidad_onda(x, y, x_max, y_max, c, c_lente):
    C = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Pared en el centro (y = 1 metro) con un ancho de 4 cm
            pared = (np.abs(y[i, j] - y_max / 2) < 0.02)  # 4 cm de ancho
            # Abertura en la pared (x ∈ [0.3, 0.7] metros)
            abertura = (x[i, j] >= 0.3) & (x[i, j] <= 0.7)
            # Velocidad de las ondas
            if pared and not abertura:
                C[i, j] = 0.0  # Pared (velocidad cero)
            elif pared and abertura:
                C[i, j] = c_lente  # Abertura (velocidad reducida)
            else:
                C[i, j] = c  # Resto del tanque (velocidad normal)
    return C

C = velocidad_onda(X, Y, x_max, y_max, c, c_lente)

# Condiciones iniciales
u = np.zeros((Ny, Nx))  # Amplitud de la onda
u_prev = np.zeros((Ny, Nx))  # Amplitud de la onda en el paso anterior

# Fuente de la onda (sinusoidal a 10 Hz)
frecuencia = 10  # Hz
@njit
def fuente(t):
    return 0.01 * np.sin(2 * np.pi * frecuencia * t)

# Esquema de diferencias finitas para la ecuación de onda (optimizado con Numba)
@njit
def step(u, u_prev, C, dx, dy, dt):
    u_next = np.zeros_like(u)
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            # Aproximación de las derivadas espaciales
            d2u_dx2 = (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dx**2
            d2u_dy2 = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dy**2
            # Actualización temporal
            u_next[i, j] = 2 * u[i, j] - u_prev[i, j] + (C[i, j] * dt)**2 * (d2u_dx2 + d2u_dy2)
    return u_next

# Simulación
u_history = []  # Lista para guardar los estados de u
for n in range(M):
    # Aplicar la fuente en el punto (0.5, 0.5)
    fuente_idx = (int(Ny / 4), int(Nx / 2))  # Índice correspondiente a (0.5, 0.5)
    u[fuente_idx] += fuente(n * dt)
    
    # Avanzar un paso de tiempo
    u_next = step(u, u_prev, C, dx, dy, dt)
    u_prev, u = u, u_next
    
    # Guardar cada 100 pasos de tiempo
    if n % 100 == 0:
        u_history.append(u.copy())

# Rotar la animación 180 grados (para que quede en el formato mostrado en el PDF de la tarea)
u_history_rotated = [np.flipud(np.transpose(u)) for u in u_history]

# Crear la animación
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(u_history_rotated[0], extent=[0, y_max, 0, x_max], cmap='viridis', vmin=-0.01, vmax=0.01)
plt.colorbar(im, label='Amplitud (m)')
ax.set_title('Evolución de la onda en el tanque')

def update(frame):
    im.set_data(u_history_rotated[frame])
    return im,

# Crear la animación
ani = FuncAnimation(fig, update, frames=len(u_history_rotated), interval=50, blit=True)

# Guardar la animación como un archivo de video
ani.save('4.a_rotated.mp4', writer='ffmpeg', fps=20)

# Mostrar la animación
plt.close()
from IPython.display import HTML
HTML(ani.to_html5_video())