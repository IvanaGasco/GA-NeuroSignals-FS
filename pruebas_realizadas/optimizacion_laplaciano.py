import numpy as np
import time

# --- 1. TU FUNCIÓN ORIGINAL (Basada en bucles) ---
def laplacian_original(senal):
    samples, channels, trials = senal.shape
    laplacian = np.empty((samples, trials), np.float64) 
    for i in range(trials):
        for k in range(samples):
            sample_sum = 0
            # Suma desde el canal 1 hasta el final
            for j in range(1, channels):
                sample_sum += senal[k, j, i] 
            # Resta: Canal 0 - Promedio de los otros
            laplacian[k, i] = senal[k, 0, i] - (sample_sum / (channels - 1))  
    return laplacian

# --- 2. MI FUNCIÓN OPTIMIZADA (Vectorizada) ---
def laplacian_vectorizado(senal):
    # np.mean(axis=1) calcula el promedio a través de los canales.
    # senal[:, 1:, :] selecciona todas las muestras, canales del 1 al final, y todos los trials.
    promedio_vecinos = np.mean(senal[:, 1:, :], axis=1)
    
    # Resta: Canal 0 - Promedio
    laplacian = senal[:, 0, :] - promedio_vecinos
    return laplacian

# --- 3. LA PRUEBA DE FUEGO ---

# Creamos datos aleatorios simulando tu estructura
# (512 muestras, 5 canales, 50 trials)
datos_fake = np.random.rand(512, 5, 50)

print("Calculando con función original...")
start = time.time()
res_original = laplacian_original(datos_fake)
end = time.time()
print(f"Tiempo Original: {end - start:.5f} segundos")

print("\nCalculando con función vectorizada...")
start = time.time()
res_vectorizado = laplacian_vectorizado(datos_fake)
end = time.time()
print(f"Tiempo Vectorizado: {end - start:.5f} segundos")

# --- 4. COMPARACIÓN MATEMÁTICA ---

# np.allclose verifica si dos arrays son iguales elemento por elemento
# Usamos esto en lugar de '==' porque los decimales flotantes pueden variar en 0.000000001
son_iguales = np.allclose(res_original, res_vectorizado)

print("\n----------------RESULTADO----------------")
print(f"¿Los valores son matemáticamente idénticos? -> {son_iguales}")

if son_iguales:
    print("✅ Puedes usar la versión optimizada con total confianza.")
else:
    diff = np.max(np.abs(res_original - res_vectorizado))
    print(f"❌ Hay diferencias. Error máximo encontrado: {diff}")
