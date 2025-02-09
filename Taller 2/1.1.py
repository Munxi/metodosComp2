import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import scipy

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