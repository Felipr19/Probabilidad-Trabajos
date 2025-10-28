# =============================================================================
# 1. DEFINICIÓN DE DISTRIBUCIONES - CORREGIDO
# =============================================================================

import openturns as ot

print("=" * 50)
print("1. DEFINICIÓN DE DISTRIBUCIONES")
print("=" * 50)

# Distribuciones para parámetros de la viga
dist_E = ot.Normal(210e9, 10e9)      # Módulo elástico (Pa)
dist_E.setName("Módulo Elástico")

dist_I = ot.LogNormal(1e-4, 2e-5)    # Momento de inercia (m⁴)
dist_I.setName("Momento de Inercia")

dist_L = ot.Uniform(2.0, 3.0)        # Longitud de la viga (m)
dist_L.setName("Longitud")

dist_P = ot.WeibullMin(1000, 2.5, 0) # Carga aplicada (N)
dist_P.setName("Carga")

# Mostrar información de las distribuciones
print("✓ Distribuciones definidas:")
print(f"  - {dist_E.getName()}: Media = {dist_E.getMean()[0]:.2e} Pa")
print(f"  - {dist_I.getName()}: Media = {dist_I.getMean()[0]:.2e} m⁴")
print(f"  - {dist_L.getName()}: Rango = [{dist_L.getA()}, {dist_L.getB()}] m")  # ¡CORREGIDO!
print(f"  - {dist_P.getName()}: Media = {dist_P.getMean()[0]:.1f} N")

# Generar muestras
sample_size = 1000
sample_E = dist_E.getSample(sample_size)
sample_I = dist_I.getSample(sample_size)
sample_L = dist_L.getSample(sample_size)
sample_P = dist_P.getSample(sample_size)

print(f"\n✓ Generadas {sample_size} muestras para cada variable")

# Guardar muestras para usar en otros archivos
import pickle
with open('muestras.pkl', 'wb') as f:
    pickle.dump({
        'sample_E': sample_E,
        'sample_I': sample_I, 
        'sample_L': sample_L,
        'sample_P': sample_P,
        'distribuciones': [dist_E, dist_I, dist_L, dist_P]
    }, f)

print("✓ Muestras guardadas en 'muestras.pkl'")