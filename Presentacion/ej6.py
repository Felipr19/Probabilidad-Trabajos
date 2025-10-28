# =============================================================================
# 6. VISUALIZACIÓN COMPLETA
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pickle

print("=" * 50)
print("6. VISUALIZACIÓN DE RESULTADOS")
print("=" * 50)

# Cargar todos los datos
with open('resultados.pkl', 'rb') as f:
    resultados = pickle.load(f)

with open('muestras.pkl', 'rb') as f:
    muestras = pickle.load(f)

with open('estadisticas.pkl', 'rb') as f:
    stats = pickle.load(f)

with open('riesgo.pkl', 'rb') as f:
    riesgo = pickle.load(f)

deflection_values = resultados['deflection_values']
sample_E = muestras['sample_E']
sample_I = muestras['sample_I']
sample_L = muestras['sample_L']
sample_P = muestras['sample_P']

# Crear visualizaciones
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Análisis Completo de Incertidumbre - Diseño de Viga', fontsize=16, fontweight='bold')

# 1. Histograma de deflexiones
axes[0, 0].hist(deflection_values * 1000, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
axes[0, 0].axvline(0.015 * 1000, color='red', linestyle='--', linewidth=2, label='Límite (15 mm)')
axes[0, 0].axvline(stats['mean'] * 1000, color='green', linestyle='-', linewidth=2, label=f'Media ({stats["mean"]*1000:.1f} mm)')
axes[0, 0].set_xlabel('Deflexión (mm)')
axes[0, 0].set_ylabel('Densidad de Probabilidad')
axes[0, 0].set_title('Distribución de Deflexiones')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribución de cargas
carga_values = np.array(sample_P).flatten()
axes[0, 1].hist(carga_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
axes[0, 1].set_xlabel('Carga (N)')
axes[0, 1].set_ylabel('Densidad')
axes[0, 1].set_title('Distribución de Cargas')
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribución de longitudes
longitud_values = np.array(sample_L).flatten()
axes[0, 2].hist(longitud_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black', density=True)
axes[0, 2].set_xlabel('Longitud (m)')
axes[0, 2].set_ylabel('Densidad')
axes[0, 2].set_title('Distribución de Longitudes')
axes[0, 2].grid(True, alpha=0.3)

# 4. Análisis de riesgo
labels_riesgo = [f'Falla\n{riesgo["probabilidad_falla"]*100:.2f}%', 
                f'Seguro\n{(1-riesgo["probabilidad_falla"])*100:.2f}%']
colors_riesgo = ['#ff6b6b', '#51cf66']
axes[1, 0].pie([riesgo['probabilidad_falla'], 1-riesgo['probabilidad_falla']], 
               labels=labels_riesgo, colors=colors_riesgo, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Probabilidad de Estado')

# 5. Comparación métodos
E_mean = 210e9
I_mean = 1e-4
L_mean = 2.5
P_mean = 1000
def_determinista = (P_mean * L_mean**3) / (3 * E_mean * I_mean)

metodos = ['Monte Carlo', 'Determinista']
valores = [stats['mean'] * 1000, def_determinista * 1000]
bars = axes[1, 1].bar(metodos, valores, color=['lightblue', 'orange'])
axes[1, 1].set_ylabel('Deflexión (mm)')
axes[1, 1].set_title('Monte Carlo vs Determinista')
axes[1, 1].grid(True, alpha=0.3)

# Añadir valores en las barras
for bar, valor in zip(bars, valores):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{valor:.2f} mm', ha='center', va='bottom')

# 6. Resumen ejecutivo
axes[1, 2].axis('off')
texto_resumen = f"""
RESUMEN EJECUTIVO:

• Deflexión media: {stats['mean']*1000:.2f} mm
• Desviación: {stats['std']*1000:.2f} mm  
• P95: {stats['quantile_95']*1000:.2f} mm
• Prob. falla: {riesgo['probabilidad_falla']*100:.4f}%
• Estado: {riesgo['estado']}
"""
axes[1, 2].text(0.1, 0.9, texto_resumen, transform=axes[1, 2].transAxes, 
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('resultados_analisis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualizaciones generadas exitosamente")
print("✓ Gráfico guardado como 'resultados_analisis.png'")