import simpy
import random
import numpy as np
from math import exp, factorial
from scipy.stats import norm


# 1️ DISTRIBUCIÓN DE POISSON: CENTRO DE LLAMADAS

LAMBDA1 = 4  # llamadas por hora
TIEMPO1 = 1  # 1 hora
EXPERIMENTOS = 100000  # número de simulaciones

def proceso_poisson(env, tasa, eventos):
    """Genera llegadas con tiempo entre eventos exponencial."""
    while True:
        tiempo = random.expovariate(tasa)
        yield env.timeout(tiempo)
        eventos.append(env.now)

def simular_poisson(tasa, tiempo_total):
    """Simula un proceso Poisson durante cierto tiempo."""
    env = simpy.Environment()
    eventos = []
    env.process(proceso_poisson(env, tasa, eventos))
    env.run(until=tiempo_total)
    return len(eventos)

# Simulación repetida
resultados1 = [simular_poisson(LAMBDA1, TIEMPO1) for _ in range(EXPERIMENTOS)]
p_x6_sim = np.mean(np.array(resultados1) == 6)
p_x6_teo = (exp(-LAMBDA1) * (LAMBDA1**6)) / factorial(6)

print("=== Ejercicio 1: Llamadas (Poisson λ=4) ===")
print(f"Probabilidad simulada P(X=6): {p_x6_sim:.4f}")
print(f"Probabilidad teórica P(X=6):  {p_x6_teo:.4f}\n")


# 2️ DISTRIBUCIÓN DE POISSON: CLIENTES POR HORA

LAMBDA2 = 10  # clientes por hora
TIEMPO2 = 1   # 1 hora

# Simulación repetida
resultados2 = [simular_poisson(LAMBDA2, TIEMPO2) for _ in range(EXPERIMENTOS)]
p_x_le2_sim = np.mean(np.array(resultados2) <= 2)

# Teórico: P(X ≤ 2) = P(0) + P(1) + P(2)
p_x_le2_teo = sum((exp(-LAMBDA2) * (LAMBDA2**k)) / factorial(k) for k in range(3))

print("=== Ejercicio 2: Clientes (Poisson λ=10) ===")
print(f"Probabilidad simulada P(X≤2): {p_x_le2_sim:.6f}")
print(f"Probabilidad teórica P(X≤2):  {p_x_le2_teo:.6f}\n")


# 3️ DISTRIBUCIÓN NORMAL: ESTATURAS

MU1 = 170  # media
SIGMA1 = 8  # desviación estándar
N = 1000000  # cantidad de simulaciones

# Generamos muestras
muestras_estaturas = np.random.normal(MU1, SIGMA1, N)
p_entre_sim = np.mean((muestras_estaturas >= 165) & (muestras_estaturas <= 180))

# Probabilidad teórica con tabla Z
z1 = (165 - MU1) / SIGMA1
z2 = (180 - MU1) / SIGMA1
p_entre_teo = norm.cdf(z2) - norm.cdf(z1)

print("=== Ejercicio 3: Estaturas (Normal μ=170, σ=8) ===")
print(f"Probabilidad simulada P(165≤X≤180): {p_entre_sim:.4f}")
print(f"Probabilidad teórica P(165≤X≤180):  {p_entre_teo:.4f}\n")


# 4️ DISTRIBUCIÓN NORMAL: PUNTAJES DE ESTUDIANTES

MU2 = 75
SIGMA2 = 12

# Generamos muestras
muestras_puntajes = np.random.normal(MU2, SIGMA2, N)
p_ge90_sim = np.mean(muestras_puntajes >= 90)

# Probabilidad teórica con tabla Z
z = (90 - MU2) / SIGMA2
p_ge90_teo = 1 - norm.cdf(z)

print("=== Ejercicio 4: Puntajes (Normal μ=75, σ=12) ===")
print(f"Probabilidad simulada P(X≥90): {p_ge90_sim:.4f}")
print(f"Probabilidad teórica P(X≥90):  {p_ge90_teo:.4f}")
print(f"Porcentaje: {p_ge90_teo*100:.2f}%\n")
