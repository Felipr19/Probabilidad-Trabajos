# =============================================================================
# 4. ANÁLISIS ESTADÍSTICO
# =============================================================================

import numpy as np
import pickle

print("=" * 50)
print("4. ANÁLISIS ESTADÍSTICO")
print("=" * 50)

# Cargar resultados
with open('resultados.pkl', 'rb') as f:
    resultados = pickle.load(f)

with open('muestras.pkl', 'rb') as f:
    muestras = pickle.load(f)

deflection_values = resultados['deflection_values']

# Estadísticas básicas
mean_deflection = np.mean(deflection_values)
std_deflection = np.std(deflection_values)
quantile_95 = np.quantile(deflection_values, 0.95)
max_deflection = np.max(deflection_values)
min_deflection = np.min(deflection_values)

print(" ESTADÍSTICAS DE DEFLEXIÓN:")
print(f"  • Media: {mean_deflection:.6f} m")
print(f"  • Desviación estándar: {std_deflection:.6f} m") 
print(f"  • Mínimo: {min_deflection:.6f} m")
print(f"  • Máximo: {max_deflection:.6f} m")
print(f"  • Percentil 95%: {quantile_95:.6f} m")
print(f"  • Coeficiente de variación: {(std_deflection/mean_deflection)*100:.2f}%")

# Análisis en mm para mejor interpretación
print(f"\n EN MILÍMETROS:")
print(f"  • Media: {mean_deflection*1000:.2f} mm")
print(f"  • Desviación: {std_deflection*1000:.2f} mm")
print(f"  • Rango: [{min_deflection*1000:.2f}, {max_deflection*1000:.2f}] mm")
print(f"  • P95: {quantile_95*1000:.2f} mm")

# Guardar estadísticas
with open('estadisticas.pkl', 'wb') as f:
    pickle.dump({
        'mean': mean_deflection,
        'std': std_deflection,
        'quantile_95': quantile_95,
        'max': max_deflection,
        'min': min_deflection
    }, f)

print("✓ Estadísticas guardadas en 'estadisticas.pkl'")