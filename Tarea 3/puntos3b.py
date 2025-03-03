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

    # Mostrar la animación
    ani.save(filename="1.a.gif")
punto1()
punto2()