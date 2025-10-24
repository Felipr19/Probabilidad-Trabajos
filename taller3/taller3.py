# taller2_simulacion.py
import numpy as np
import math
from scipy import integrate
from scipy.stats import rv_continuous
import simpy
import random
from math import comb


# A) Distribución conjunta continua
# f(x,y) = (2/5)*(2x + 3y)  en 0<x<1, 0<y<1  (0 fuera)

def f_xy(x, y):
    """Densidad conjunta"""
    # acepta arrays x,y (numpy broadcasting)
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.zeros_like(np.broadcast(x, y), dtype=float)
    mask = (x > 0) & (x < 1) & (y > 0) & (y < 1)
    out[mask] = 2/5 * (2*x[mask] + 3*y[mask])
    return out

# 1) Verificar normalización (integral sobre [0,1]x[0,1])
def analytic_normalization():
    # integral double: ∫_{x=0..1} ∫_{y=0..1} (2/5)(2x+3y) dy dx
    # lo podemos calcular con integrales simbólicas/numericas; aquí usamos cálculo analítico manual:
    # ∫x=0..1 ∫y=0..1 (2/5)(2x + 3y) dy dx
    # = (2/5)*( ∫x=0..1 ( 2x*(1) + 3*(1/2)*1^2 ) dx )
    # = (2/5)*( ∫0..1 (2x + 3/2) dx ) = (2/5)*( [x^2]_0^1 + (3/2)[x]_0^1 ) = (2/5)*(1 + 3/2) = (2/5)*(5/2)=1
    return 1.0

# numeric check by quadrature (sanity)
def numeric_normalization():
    val, err = integrate.dblquad(lambda yy, xx: (2/5)*(2*xx + 3*yy),
                                 0, 1,             # x from 0..1
                                 lambda x: 0, lambda x: 1)  # y from 0..1
    return val, err

# 2) Calcular P{(x,y) in R} para un rectángulo R = [x0,x1] x [y0,y1] intersectado con dominio
def analytic_prob_rectangle(x0, x1, y0, y1):
    # asumimos 0<=x0<x1<=1, 0<=y0<y1<=1 (si exceden, se recortan)
    xa = max(0.0, x0); xb = min(1.0, x1)
    ya = max(0.0, y0); yb = min(1.0, y1)
    if xb <= xa or yb <= ya:
        return 0.0
    # integral (2/5) ∫_{x=xa..xb} ∫_{y=ya..yb} (2x + 3y) dy dx
    # compute inner integral in y: ∫ (2x + 3y) dy = 2x*y + 3/2*y^2
    # then integrate in x.
    def inner_integral_x(x):
        return (2/5) * (2*x*(yb - ya) + 1.5*(yb**2 - ya**2))
    # integrate inner_integral_x from xa..xb
    # integral of 2*x*(yb-ya) dx = (yb-ya) * (x^2)|_xa^xb
    term1 = (yb - ya) * (xb**2 - xa**2)
    term2 = 1.5 * (yb**2 - ya**2) * (xb - xa)
    res = (2/5) * (term1 + term2)
    return res

# 3) Monte Carlo sampling from f(x,y) by rejection sampling and estimate probability in rectangle
def sample_from_f(n_samples=200000):
    # Rejection: bounding function: max f on unit square
    # f(x,y)=2/5(2x+3y) max occurs at x=1, y=1 -> f_max = 2/5*(2+3)=2/5 *5 =2
    f_max = 2.0
    samples = []
    cnt = 0
    while len(samples) < n_samples:
        # propose uniform in [0,1]^2
        u = random.random()
        v = random.random()
        # accept with probability f(u,v)/f_max
        if random.random() <= (2/5)*(2*u + 3*v) / f_max:
            samples.append((u, v))
        cnt += 1
        # safety: avoid infinite loop
        if cnt > n_samples * 20:
            break
    arr = np.array(samples)
    return arr


# B) Selección discreta (combinatoria)
# Salon: 3 sistemas, 2 electronica, 3 industrial  -> total 8
# Seleccionan 2 estudiantes sin reemplazo. Definir X = # de sistemas seleccionados, Y = # electronica seleccionados.

total = 8
nsys = 3
nele = 2
nind = 3
draw = 2

def theoretical_joint_table():
    denom = comb(total, draw)
    table = {}
    for x in range(0, draw+1):
        for y in range(0, draw+1):
            if x + y <= draw:
                z = draw - x - y  # from industrial
                if z <= nind:
                    numer = comb(nsys, x) * comb(nele, y) * comb(nind, z)
                    table[(x, y)] = numer / denom
    return table

# Simulación con SimPy: cada "experimento" es tomar 2 alumnos sin reemplazo y contar (X,Y)
def simpy_simulation(num_experiments=200000):
    results = {}
    def trial(env, out_counts):
        # pick 2 without replacement from list of labels
        pool = (['S']*nsys) + (['E']*nele) + (['I']*nind)
        picks = random.sample(pool, draw)
        x = picks.count('S')
        y = picks.count('E')
        out_counts.append((x, y))
        yield env.timeout(0)  # no time evolution needed, but using simpy per request

    env = simpy.Environment()
    observed = []
    # create processes that run instantly to generate many trials
    for _ in range(num_experiments):
        env.process(trial(env, observed))
    env.run()  # runs all processes
    # count frequencies
    freq = {}
    for (x, y) in observed:
        freq[(x, y)] = freq.get((x, y), 0) + 1
    for k in list(freq.keys()):
        freq[k] /= num_experiments
    return freq


# Ejecutar todo y mostrar resultados

if __name__ == "__main__":
    print("=== A) Densidad continua f(x,y) ===")
    print("Verificación analítica de normalización:", analytic_normalization())
    val, err = numeric_normalization()
    print(f"Verificación numérica (dblquad) = {val:.8f} (err ~ {err:.1e})")
    # Ejemplos de rectángulos para calcular P(R):
    rects = [
        (0, 0.5, 0.25, 0.5),   # ejemplo: x in [0,0.5], y in [0.25,0.5]
        (0.25, 0.5, 0.0, 0.5), # ejemplo observado en apuntes posible
        (0.25, 0.5, 0.25, 0.5),
    ]
    for (x0, x1, y0, y1) in rects:
        p_analytic = analytic_prob_rectangle(x0, x1, y0, y1)
        print(f"P({x0}<=X<={x1}, {y0}<=Y<={y1}) analítica = {p_analytic:.6f}")
    # Muestreo por rejection sampling
    samples = sample_from_f(n_samples=200000)
    for (x0, x1, y0, y1) in rects:
        # estimate
        mask = (samples[:,0] >= x0) & (samples[:,0] <= x1) & (samples[:,1] >= y0) & (samples[:,1] <= y1)
        p_est = mask.mean()
        print(f"P({x0}<=X<={x1}, {y0}<=Y<={y1}) simulada = {p_est:.6f}")

    print("\n=== B) Selección discreta (combinatoria) ===")
    table_th = theoretical_joint_table()
    print("Probabilidades teóricas P(X=x,Y=y):")
    for k in sorted(table_th.keys()):
        print(f"  P{str(k)} = {table_th[k]:.6f}")
    # Simulación con SimPy
    sim_freq = simpy_simulation(num_experiments=100000)
    print("\nEstimaciones (SimPy) de P(X,Y) por simulación:")
    # print both theoretical and simulated for comparison
    for k in sorted(table_th.keys()):
        sim_val = sim_freq.get(k, 0.0)
        print(f"  (x,y)={k}: teórica={table_th[k]:.6f}, simulada={sim_val:.6f}")
