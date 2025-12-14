import numpy as np

# 1 Dimensiones 
num_participantes = 8
num_muestras_por_participante = 80
num_caracteristicas = 96

# 2. Tensor de Datos (X) 
dimensiones = (num_participantes, num_muestras_por_participante, num_caracteristicas)
X_data_simulada = np.zeros(dimensiones, dtype=np.int8)

# Rellenar con 15% de actividad simulada
porcentaje_unos = 0.15
indices_unos = np.random.rand(*dimensiones) < porcentaje_unos
X_data_simulada[indices_unos] = 1

# 3. Generación de Etiquetas (y) 
# 40 'Sí' (1) y 40 'No' (0) por participante
y_etiquetas_por_participante = np.concatenate([
    np.ones(40, dtype=np.int8),  # 40 muestras 'Sí'
    np.zeros(40, dtype=np.int8) # 40 muestras 'No'
])

# --- 4. GUARDAR DATOS EN EL DISCO ---
print("Guardando datos simulados en formato .npy...")

np.save('X_data_simulada.npy', X_data_simulada)
print(f"Archivo X guardado: X_data_simulada.npy ({X_data_simulada.shape})")

np.save('y_etiquetas_por_participante.npy', y_etiquetas_por_participante)
print(f"Archivo Y guardado: y_etiquetas_por_participante.npy ({y_etiquetas_por_participante.shape})")

print("\n¡Ejecute este script primero, luego ejecute ejecutar_ag.py!")
