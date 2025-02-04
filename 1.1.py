import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float], frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
    ts = np.arange(0.,t_max,dt)
    ys = np.zeros_like(ts,dtype=float)
    for A,f in zip(amplitudes,frecuencias):
        ys += A*np.sin(2*np.pi*f*ts)
        ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
    return ts,ys

def Fourier(t:NDArray[float], y:NDArray[float], f:NDArray[float]) -> NDArray[complex]:
    return

