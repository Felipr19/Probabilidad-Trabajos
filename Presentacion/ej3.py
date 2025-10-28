# =============================================================================
# 3. PROPAGACIÓN DE INCERTIDUMBRE
# =============================================================================

import openturns as ot
import numpy as np
import pickle

print("=" * 50)
print("3. PROPAGACIÓN DE INCERTIDUMBRE")
print("=" * 50)

# Cargar datos guardados
with open('muestras.pkl', 'rb') as f:
    datos = pickle.load(f)

with open('funcion.pkl', 'rb') as f:
    viga_function = pickle.load(f)

sample_E = datos['sample_E']
sample_I = datos['sample_I']
sample_L = datos['sample_L'] 
sample_P = datos['sample_P']
sample_size = sample_E.getSize()

print(f"✓ Cargadas {sample_size} muestras y función del modelo")

# Crear muestra de entrada combinada
input_sample = ot.Sample(sample_size, 4)
for i in range(sample_size):
    input_sample[i] = [sample_E[i][0], sample_I[i][0], sample_L[i][0], sample_P[i][0]]

print("✓ Muestra de entrada combinada creada")

# Aplicar función para propagar incertidumbre
output_sample = viga_function(input_sample)

print(f"✓ Tamaño de muestra de salida: {output_sample.getSize()}")

# Convertir a numpy para análisis
deflection_values = np.array(output_sample).flatten()

# Guardar resultados
with open('resultados.pkl', 'wb') as f:
    pickle.dump({
        'input_sample': input_sample,
        'output_sample': output_sample,
        'deflection_values': deflection_values
    }, f)

print("✓ Resultados guardados en 'resultados.pkl'")