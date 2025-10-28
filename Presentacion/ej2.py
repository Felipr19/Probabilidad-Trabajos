# =============================================================================
# 2. FUNCIÓN DEL MODELO 
# =============================================================================

import openturns as ot
import numpy as np

print("=" * 50)
print("2. FUNCIÓN DEL MODELO")
print("=" * 50)

class VigaModel:
    def __init__(self):
        pass
    
    def _exec(self, X):
        """
        Maneja un solo punto (lista de 4 valores)
        """
        E = X[0]  # Módulo elástico
        I = X[1]  # Momento de inercia  
        L = X[2]  # Longitud
        P = X[3]  # Carga
        
        # Fórmula de deflexión: δ = PL³/(3EI)
        # Evitar división por cero
        if abs(E) < 1e-20 or abs(I) < 1e-20:
            deflection = 0.0
        else:
            deflection = (P * L**3) / (3 * E * I)
        
        return [deflection]
    
    def _exec_sample(self, X):
        """
        Maneja múltiples puntos (lista de listas)
        """
        results = []
        for point in X:
            results.append(self._exec(point))
        return results

# Crear instancia de la clase
viga_model_instance = VigaModel()

# Crear función OpenTURNS usando la clase
viga_function = ot.PythonFunction(4, 1, viga_model_instance._exec)
viga_function.setInputDescription(['E', 'I', 'L', 'P'])
viga_function.setOutputDescription(['Deflexion'])

print("Función de viga definida CORRECTAMENTE")

# Probar la función con un punto de prueba
punto_prueba = ot.Point([210e9, 1e-4, 2.5, 1000])  # Usar Point de OpenTURNS
deflexion_prueba = viga_function(punto_prueba)
print(f"✓ Deflexión para punto de prueba: {deflexion_prueba[0]:.6f} m")

# Probar con múltiples puntos
puntos_prueba = ot.Sample([[210e9, 1e-4, 2.5, 1000], [200e9, 1.1e-4, 2.6, 1100]])
deflexiones_prueba = viga_function(puntos_prueba)
print(f"✓ Deflexiones para 2 puntos: {[d[0] for d in deflexiones_prueba]}")

# Guardar función para usar en otros archivos
import pickle
with open('funcion.pkl', 'wb') as f:
    pickle.dump(viga_function, f)

print("✓ Función guardada en 'funcion.pkl'")

# También guardar la instancia para referencia
with open('modelo_viga.pkl', 'wb') as f:
    pickle.dump(viga_model_instance, f)

print("✓ Modelo guardado en 'modelo_viga.pkl'")