import numpy as np
import matplotlib.pyplot as plt

def g_x(x, n=10, alpha=4/5):
    return sum(np.exp(-(x - k)**2 * k) / (k**alpha) for k in range(1, n+1))

def metropolis_hastings(g, samples=500000, n=10, alpha=4/5):
    x = 0 
    samples_list = []
    for _ in range(samples):
        x_new = x + np.random.normal(scale=1.0)
        acceptance_ratio = g(x_new, n, alpha) / g(x, n, alpha)
        if np.random.rand() < acceptance_ratio:
            x = x_new
        samples_list.append(x)
    return np.array(samples_list)

samples = metropolis_hastings(g_x)

plt.hist(samples, bins=200, density=True, alpha=0.6, color='b')
plt.xlabel("x")
plt.ylabel("Frecuencia")
plt.title("Histograma de muestras - Metrópolis-Hastings")
plt.savefig("1.a.pdf")
plt.close()

f_x = lambda x: np.exp(-x**2)
g_values = np.array([g_x(xi) for xi in samples])
f_values = np.array([f_x(xi) for xi in samples])

A_est = np.sqrt(np.pi) / np.mean(f_values / g_values)
std_A = np.sqrt(np.var(f_values / g_values) / len(samples))

print(f"1.b) A estimado: {A_est:.6f} ± {std_A:.6f}")