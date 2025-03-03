import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Parámetros
delta = 0.022  # Parámetro de dispersión
L = 2.0  # Longitud del dominio espacial
N = 100  # Número de puntos espaciales
h = L / N  # Paso espacial
k = 0.001  # Paso temporal
T = 2000  # Tiempo final
M = int(T / k)  # Número total de pasos de tiempo

# Condiciones iniciales
x = np.linspace(0, L, N, endpoint=False)  # Dominio espacial
u = np.cos(np.pi * x)  # Condición inicial

# Inicialización de la solución
u_new = np.copy(u)
u_old = np.copy(u)

# Almacenamiento para cantidades conservadas
masa_history = []
momento_history = []
energia_history = []

# Función para avanzar en el tiempo
@njit
def step(u, u_old, N, h, k, delta):
    u_new = np.zeros_like(u)
    for i in range(N):
        ip1 = (i + 1) % N  # Condición de frontera periódica
        im1 = (i - 1) % N  # Condición de frontera periódica
        ip2 = (i + 2) % N  # Condición de frontera periódica
        im2 = (i - 2) % N  # Condición de frontera periódica
        
        # Aproximación de las derivadas
        du_dx = (u[ip1] - u[im1]) / (2 * h)  # Primera derivada
        d3u_dx3 = (u[ip2] - 2 * u[ip1] + 2 * u[im1] - u[im2]) / (2 * h**3)  # Tercera derivada
        
        # Actualización temporal (esquema de Zabusky y Kruskal)
        u_new[i] = u_old[i] - 2 * k * (u[i] * du_dx + delta**2 * d3u_dx3)
    return u_new

# Función para calcular las cantidades conservadas
@njit
def conserved_quantities(u, h, delta):
    masa = np.sum(u) * h  # Masa
    momento = np.sum(u**2) * h  # Momento
    
    # Aproximación de la derivada espacial para la energía
    du_dx = np.zeros_like(u)
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        du_dx[i] = (u[ip1] - u[im1]) / (2 * h)
    
    energia = np.sum(u**3 - delta**2 * du_dx**2) * h  # Energía
    return masa, momento, energia

# Simulación
u_history = []  # Lista para guardar los estados de u

# Inicializar u_old (j-1) usando un paso de Euler hacia atrás
for i in range(N):
    ip1 = (i + 1) % N
    im1 = (i - 1) % N
    ip2 = (i + 2) % N
    im2 = (i - 2) % N
    
    du_dx = (u[ip1] - u[im1]) / (2 * h)
    d3u_dx3 = (u[ip2] - 2 * u[ip1] + 2 * u[im1] - u[im2]) / (2 * h**3)
    u_old[i] = u[i] - k * (u[i] * du_dx + delta**2 * d3u_dx3)

# Avanzar en el tiempo
for n in range(M):
    u_new = step(u, u_old, N, h, k, delta)  # Avanzar un paso de tiempo
    
    if np.any(np.isnan(u_new)):  # Terminar la simulación si hay NaN
        print(f"Simulación terminada en el paso {n} debido a valores NaN.")
        break
    
    if n % 100 == 0:  # Guardar cada 100 pasos de tiempo
        u_history.append(u_new.copy())
        masa, momento, energia = conserved_quantities(u_new, h, delta)
        masa_history.append(masa)
        momento_history.append(momento)
        energia_history.append(energia)
    
    # Actualizar u_old y u para el siguiente paso
    u_old = u.copy()
    u = u_new.copy()

# Graficar las cantidades conservadas
time_points = np.arange(0, len(masa_history) * k * 100, k * 100) / 1000  # Tiempo escalado por 1/1000
plt.figure(figsize=(12, 8))

# Masa
plt.subplot(3, 1, 1)
plt.plot(time_points, masa_history, label='Masa', color='blue')
plt.xlabel('Tiempo (t / 1000)')
plt.ylabel('Masa')
plt.title('Masa en función del tiempo')
plt.legend()

# Momento
plt.subplot(3, 1, 2)
plt.plot(time_points, momento_history, label='Momento', color='green')
plt.xlabel('Tiempo (t / 1000)')
plt.ylabel('Momento')
plt.title('Momento en función del tiempo')
plt.legend()

# Energía
plt.subplot(3, 1, 3)
plt.plot(time_points, energia_history, label='Energía', color='red')
plt.xlabel('Tiempo (t / 1000)')
plt.ylabel('Energía')
plt.title('Energía en función del tiempo')
plt.legend()

plt.tight_layout()
plt.savefig('3.b.pdf')  # Guardar la gráfica en un archivo PDF

# Crear la animación
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u_history[0], color='b')
ax.set_xlabel('Tiempo (t)')
ax.set_ylabel('u(x, t)')
ax.set_title('Evolución de u(x, t) en el tiempo')
ax.set_ylim(-1.5, 2.5)

# Función para actualizar la animación
def update(frame):
    line.set_ydata(u_history[frame])
    return line,

# Crear la animación
ani = FuncAnimation(fig, update, frames=len(u_history), interval=50, blit=True)
# Guardar la animación como un archivo de video
ani.save('3.a.mp4', writer='ffmpeg', fps=20)

# Mostrar la animación
plt.show()