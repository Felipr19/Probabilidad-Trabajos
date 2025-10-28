# =============================================================================
# 5. ANÁLISIS DE RIESGO
# =============================================================================

import numpy as np
import pickle

print("=" * 50)
print("5. ANÁLISIS DE RIESGO")
print("=" * 50)

# Cargar datos
with open('resultados.pkl', 'rb') as f:
    resultados = pickle.load(f)

deflection_values = resultados['deflection_values']

# Límite de diseño
limite_deflexion = 0.015  # 1.5 cm

# Calcular probabilidad de falla
fallas = np.sum(deflection_values > limite_deflexion)
probabilidad_falla = fallas / len(deflection_values)

print(" ANÁLISIS DE RIESGO:")
print(f"  • Límite de diseño: {limite_deflexion:.3f} m ({limite_deflexion*1000:.1f} mm)")
print(f"  • Casos que exceden límite: {fallas}/{len(deflection_values)}")
print(f"  • Probabilidad de falla: {probabilidad_falla:.6f}")
print(f"  • Probabilidad de falla: {probabilidad_falla*100:.4f}%")
print(f"  • Confiabilidad: {(1-probabilidad_falla)*100:.4f}%")

# Evaluación del diseño
print(f"\ EVALUACIÓN DEL DISEÑO:")
if probabilidad_falla < 0.001:
    estado = "✅ EXCELENTE"
    recomendacion = "Diseño muy robusto"
elif probabilidad_falla < 0.01:
    estado = "✅ ACEPTABLE" 
    recomendacion = "Diseño adecuado"
elif probabilidad_falla < 0.05:
    estado = "⚠️  MARGINAL"
    recomendacion = "Considerar mejoras"
else:
    estado = "❌ INACEPTABLE"
    recomendacion = "Rediseñar urgentemente"

print(f"  • Estado: {estado}")
print(f"  • Recomendación: {recomendacion}")

# Guardar análisis de riesgo
with open('riesgo.pkl', 'wb') as f:
    pickle.dump({
        'probabilidad_falla': probabilidad_falla,
        'fallas': fallas,
        'total_casos': len(deflection_values),
        'estado': estado,
        'recomendacion': recomendacion
    }, f)

print("✓ Análisis de riesgo guardado en 'riesgo.pkl'")