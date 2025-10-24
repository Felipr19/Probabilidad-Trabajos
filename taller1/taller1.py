# taller3_simulacion.py
import simpy
import random
import numpy as np
from math import pow


# Parámetros de simulación

NUM_EXPERIMENTOS = 200000  # ajustar si quieres más precisión
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Problema 1: f(x) = k x^2 en (0,6)

# Calculo analítico
# ∫_0^6 k x^2 dx = 1  => k*(6^3)/3 = 1 => k*(216/3)=k*72 => k = 1/72
k = 1.0 / 72.0

def F_analytic(x):
    """CDF analítica F(x) = x^3 / 216 para 0<=x<=6 (0 si x<=0, 1 si x>=6)."""
    if x <= 0:
        return 0.0
    if x >= 6:
        return 1.0
    return (x**3) / 216.0

def sample_from_f_by_inverse(u):
    """Usando la inversa de F: F(x)=x^3/216 => x = (216*u)^(1/3)"""
    return (216.0 * u) ** (1.0/3.0)

# Usaremos SimPy para lanzar muchos "procesos" que generan una muestra cada uno.
def simpy_sample_continuous(env, out_list):
    # este proceso produce una sola muestra y termina; lo lanzamos muchas veces
    u = random.random()
    x = sample_from_f_by_inverse(u)
    out_list.append(x)
    yield env.timeout(0)

def simulate_continuous(num_experiments=NUM_EXPERIMENTOS):
    env = simpy.Environment()
    samples = []
    for _ in range(num_experiments):
        env.process(simpy_sample_continuous(env, samples))
    env.run()
    return np.array(samples)

# Probabilidades que pide el enunciado (según la hoja): P(1 < X < 5)
def analytic_prob_interval(a, b):
    return F_analytic(b) - F_analytic(a)


# Problema 2: Variable discreta uniforme {1,...,6}
# Dos variables X1, X2 independientes (como dos lanzamientos)

# Analítico:
# E(X1) = E(X2) = (1+2+...+6)/6 = 3.5
# Var(X1) = E(X^2) - E(X)^2, E(X^2) = (1^2+...+6^2)/6 = 91/6 -> Var = 91/6 - (7/2)^2 = 91/6 - 49/4 = 35/12
# S = X1 + X2 => E(S)=7, Var(S)=2 * 35/12 = 35/6

def analytic_discrete_moments():
    E1 = sum(range(1,7)) / 6.0
    E2 = sum(i*i for i in range(1,7)) / 6.0
    Var1 = E2 - E1**2
    ES = 2 * E1
    VarS = 2 * Var1  # independencia
    return {
        'E1': E1, 'E2sq': E2, 'Var1': Var1,
        'ES': ES, 'VarS': VarS
    }

# Simulación con SimPy: cada proceso simula un lanzamiento de dos "dados"
def simpy_trial_two_dice(env, out_list):
    d1 = random.randint(1, 6)
    d2 = random.randint(1, 6)
    out_list.append((d1, d2))
    yield env.timeout(0)

def simulate_two_dice(num_experiments=NUM_EXPERIMENTOS):
    env = simpy.Environment()
    observed = []
    for _ in range(num_experiments):
        env.process(simpy_trial_two_dice(env, observed))
    env.run()
    arr = np.array(observed)  # shape (N,2)
    sums = arr.sum(axis=1)
    return arr[:,0], arr[:,1], sums


# Ejecutar simulaciones y comparar

if __name__ == "__main__":
    print("=== Taller 3 de 4: verificación por simulación ===\n")

    # --- Problema 1 (continuo) ---
    print("Problema 1: f(x) = k x^2 en (0,6)")
    print(f"Constante analítica k = 1/72 = {k:.8f}")
    # muestra
    muestras = simulate_continuous(num_experiments=NUM_EXPERIMENTOS)
    # estimaciones:
    # 1) Estimar CDF en varios puntos y comparar con analítico
    puntos = [0.0, 1.0, 4.0, 5.0, 6.0]
    print("\nComparación CDF en puntos (simulada vs analítica):")
    for p in puntos:
        emp_cdf = np.mean(muestras <= p)
        an_cdf = F_analytic(p)
        print(f"  F({p}) simulada = {emp_cdf:.6f}   analítica = {an_cdf:.6f}")

    # 2) Probabilidad P(1 < X < 5)
    p_sim = np.mean((muestras > 1.0) & (muestras < 5.0))
    p_ana = analytic_prob_interval(1.0, 5.0)
    print(f"\nP(1 < X < 5) simulada = {p_sim:.6f}")
    print(f"P(1 < X < 5) analítica = {p_ana:.6f}  (={124}/216 ≈ {124/216:.6f})")

    # También mostrar la CDF completa en un par de valores aleatorios para inspección (opcional)
    # --- Problema 2 (discreto/dados) ---
    print("\n---\nProblema 2: Variables discretas uniformes en {1..6} (dos lanzamientos independientes)")
    moments = analytic_discrete_moments()
    print("Momentos analíticos:")
    print(f"  E(X1) = {moments['E1']:.6f}")
    print(f"  E(X^2) = {moments['E2sq']:.6f}")
    print(f"  Var(X1) = {moments['Var1']:.6f}")
    print(f"  E(S) = {moments['ES']:.6f}")
    print(f"  Var(S) = {moments['VarS']:.6f}\n")

    # Simular
    x1s, x2s, sums = simulate_two_dice(num_experiments=NUM_EXPERIMENTOS)
    print("Resultados simulados (N = {})".format(NUM_EXPERIMENTOS))
    print(f"  E_hat(X1) = {np.mean(x1s):.6f}")
    print(f"  E_hat(X2) = {np.mean(x2s):.6f}")
    print(f"  E_hat(S)  = {np.mean(sums):.6f}")
    print(f"  Var_hat(X1) = {np.var(x1s, ddof=0):.6f}")
    print(f"  Var_hat(S)  = {np.var(sums, ddof=0):.6f}")

    print("\nFIN - Si quieres puedo:")
    print("- aumentar NUM_EXPERIMENTOS para mayor precisión")
    print("- generar tablas/prints más explícitos (p. ej. distribución completa de S)")
    print("- preparar un notebook con gráficos y celdas ejecutables")
