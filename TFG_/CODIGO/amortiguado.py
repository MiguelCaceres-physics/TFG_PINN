import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
omega0 = 2.0          # frecuencia natural
t = np.linspace(0, 8, 1000)

# Casos de amortiguamiento
beta_sub = 0.25       # subamortiguado: beta < omega0
beta_crit = omega0    # crítico: beta = omega0
beta_over = 3.0       # sobreamortiguado: beta > omega0

# Condiciones iniciales
x0 = 1.0
v0 = 0.0

# Caso subamortiguado
omega_d = np.sqrt(omega0**2 - beta_sub**2)
C1 = x0
C2 = (v0 + beta_sub * x0) / omega_d
x_sub = np.exp(-beta_sub * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))

# Caso críticamente amortiguado
C1 = x0
C2 = v0 + beta_crit * x0
x_crit = (C1 + C2 * t) * np.exp(-beta_crit * t)

# Caso sobreamortiguado
r1 = -beta_over + np.sqrt(beta_over**2 - omega0**2)
r2 = -beta_over - np.sqrt(beta_over**2 - omega0**2)

A = np.array([[1, 1], [r1, r2]])
b = np.array([x0, v0])
C1, C2 = np.linalg.solve(A, b)

x_over = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

# Representación
plt.figure(figsize=(8, 5))
plt.plot(t, x_sub, label=r"Subamortiguado $(\beta < \omega_0)$")
plt.plot(t, x_crit, label=r"Crítico $(\beta = \omega_0)$")
plt.plot(t, x_over, label=r"Sobreamortiguado $(\beta > \omega_0)$")

plt.xlabel(r"Tiempo $t$")
plt.ylabel(r"Desplazamiento $x(t)$")
plt.title("Regímenes del oscilador armónico amortiguado")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("regimenes_oscilador_amortiguado.png", dpi=300)
plt.show()