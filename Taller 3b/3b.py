import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def punto1():
    @njit("f8[:,:](f8[:,:],i8[:],i8[:],f8[:,:],f8,f8,i8)")
    def finite_differences(rho, row_indices, col_indices, U, h, tolerance, iterations):
        u_new = np.copy(U)
        num = len(row_indices)
        for _ in range(iterations):
            for k in range(num):
                i = row_indices[k]
                j = col_indices[k]
                u_new[i, j] = 0.25 * (U[i + 1, j] + U[i - 1, j] +
                                      U[i, j + 1] + U[i, j - 1] -
                                      h ** 2 * 4 * np.pi * rho[i, j])
            if np.trace(np.abs(U - u_new)) < tolerance:
                break
            U[...] = u_new
        return U
    h = 0.01
    x = np.arange(-1, 1, h)
    y = np.arange(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    binaryBounds = R >= 0.95
    U = np.random.rand(*X.shape) * 0.2
    theta = np.arctan2(Y, X)
    U[binaryBounds] = np.sin(7 * theta[binaryBounds])
    rho = -X - Y
    poissonRegion = ~binaryBounds
    row_indices, col_indices = np.where(poissonRegion)
    U = finite_differences(rho, row_indices, col_indices, U, h, 1e-4, 15000)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    im = ax.imshow(U, extent=(-1, 1, -1, 1), origin='lower', cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    surf = ax2.plot_surface(X, Y, U, cmap='jet', edgecolor='none', linewidth=0, antialiased=True)
    ax2.view_init(elev=55,azim=240)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.axis('off')
    fig.colorbar(surf, ax=ax, shrink=0.3, aspect=10)
    fig.savefig("1.png")
def punto2():
    @njit("f8[:,:](f8[:,:],f8)")
    def dirichlet(u_dir,C):
        ver = len(u_dir[0])
        hor = len(u_dir[:,0])
        for n in range(2,hor):
            for j in range(1,ver-1):
                u_dir[n,j] =  2*u_dir[n-1,j]-u_dir[n-2,j] + (u_dir[n-1,j+1]-2*u_dir[n-1,j]+u_dir[n-1,j-1])*C**2
        return u_dir
    @njit("f8[:,:](f8[:,:],f8)")
    def neumann(u_neu,C):
        ver = len(u_neu[0])
        hor = len(u_neu[:,0])
        for n in range(2,hor):
            for j in range(1,ver-1):
                u_neu[n,j] =  2*u_neu[n-1,j]-u_neu[n-2,j] + (u_neu[n-1,j+1]-2*u_neu[n-1,j]+u_neu[n-1,j-1])*C**2
            u_neu[n,0] = u_neu[n,1]
            u_neu[n,-1] = u_neu[n,-2]
        return u_neu
    @njit("f8[:,:](f8[:,:],f8)")
    def periodic(u_per,C):
        ver = len(u_per[0])
        hor = len(u_per[:,0])
        for n in range(2,hor):
            for j in range(1,ver-1):
                u_per[n,j] =  2*u_per[n-1,j]-u_per[n-2,j] + (u_per[n-1,j+1]-2*u_per[n-1,j]+u_per[n-1,j-1])*C**2
            u_per[n,0] = u_per[n,-2]
            u_per[n,-1] = u_per[n,1]
        return u_per
    dx = 2e-2
    dt = 1e-2
    C =  dt/dx
    t_anim =10
    x = np.arange(0,2+dx,dx)
    t = np.arange(0,t_anim+dt,dt)
    u = np.zeros((int(t_anim/dt)+2,int(2/dx)+1))
    for i in range(len(u[0])):
        u[0:2,i] = np.exp(-125*(i*dx-1/2)**2)
    u_dir = np.copy(u)
    u_dir[0:2,0] = 0
    u_dir[0:2,-1] = 0
    u_dir = dirichlet(u_dir,C)
    u_neu = np.copy(u)
    u_neu[0:2,0] = u_neu[0:2,1]
    u_neu[0:2,-1] = u_neu[0:2,-2]
    u_neu = neumann(u_neu,C)
    u_per = np.copy(u)
    u_per[0:2,0] = u_per[0:2,-2]
    u_per[0:2,-1] = u_per[0:2,1]
    u_per = periodic(u_per,C)
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),sharey=True)
    fig.subplots_adjust(hspace=0.5)
    ax1.set_ylim(-1.3,1.3)
    ax2.set_xlim(0,2)
    ax1.set_xlabel("x(m)")
    ax2.set_xlabel("x(m)")
    ax3.set_xlabel("x(m)")
    ax1.set_title("1D Wave Dirichlet conditions")
    ax2.set_title("1D Wave Neumann conditions")
    ax3.set_title("1D Wave Periodic conditions")
    ax1.set_ylabel("u(x,t)")
    ax2.set_ylabel("u(x,t)")
    ax3.set_ylabel("u(x,t)")
    line_dir, = ax1.plot(x, u_dir[0, :], lw=2)  # Línea de la onda
    line_neu, = ax2.plot(x, u_neu[0, :], lw=2)
    line_per, = ax3.plot(x, u_per[0, :], lw=2)

    def update(frame):
        line_dir.set_ydata(u_dir[int(frame/dt), :])
        line_neu.set_ydata(u_neu[int(frame/dt), :])
        line_per.set_ydata(u_per[int(frame/dt), :])  # Actualizar la onda
        return line_dir,line_per,line_neu
    nframes = 100
    frames = np.linspace(0,t_anim,nframes)
    ani = FuncAnimation(fig, update, frames=frames
                        , interval=7000/nframes, blit=True)
    ani.save('2.mp4', writer='ffmpeg', fps=10)
def punto3():
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
            d3u_dx3 = (u[ip2] - 2 * u[ip1] + 2 * u[im1] - u[im2]) / (2 * h ** 3)  # Tercera derivada

            # Actualización temporal (esquema de Zabusky y Kruskal)
            u_new[i] = u_old[i] - 2 * k * (u[i] * du_dx + delta ** 2 * d3u_dx3)
        return u_new

    # Función para calcular las cantidades conservadas
    @njit
    def conserved_quantities(u, h, delta):
        masa = np.sum(u) * h  # Masa
        momento = np.sum(u ** 2) * h  # Momento

        # Aproximación de la derivada espacial para la energía
        du_dx = np.zeros_like(u)
        for i in range(N):
            ip1 = (i + 1) % N
            im1 = (i - 1) % N
            du_dx[i] = (u[ip1] - u[im1]) / (2 * h)

        energia = np.sum(u ** 3 - delta ** 2 * du_dx ** 2) * h  # Energía
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
        d3u_dx3 = (u[ip2] - 2 * u[ip1] + 2 * u[im1] - u[im2]) / (2 * h ** 3)
        u_old[i] = u[i] - k * (u[i] * du_dx + delta ** 2 * d3u_dx3)

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
def punto4():
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
                d2u_dx2 = (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dx ** 2
                d2u_dy2 = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dy ** 2
                # Actualización temporal
                u_next[i, j] = 2 * u[i, j] - u_prev[i, j] + (C[i, j] * dt) ** 2 * (d2u_dx2 + d2u_dy2)
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
    ani = FuncAnimation(fig, update, frames=len(u_history_rotated), interval=int(20000/len(u_history)), blit=True)

    # Guardar la animación como un archivo de video
    ani.save('4.a.mp4', writer='ffmpeg', fps=1)

    # Mostrar la animación
    plt.close()
punto1()
punto2()
punto3()
punto4()


