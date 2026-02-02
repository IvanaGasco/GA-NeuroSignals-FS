import numpy as np
import pygad
import matplotlib.pyplot as plt
from modulos.modulo1 import *


####################   VARIABLES DE EJECUCION   ########################

fs = 1000  # freq muestreo (Hz)
duracion = 2  # Duracion en segundos
t = np.linspace(0, duracion, int(fs * duracion), endpoint=False) # Vector tiempo
componentes_reales = [ #senoidales que van a conformar la señal
    [5, 1.0, 0],      
    [12, 0.8, 0],     
    [25, 0.6, 0],     
    [40, 0.4, 0],     
    [60, 0.3, 0],     
]
senal_limpia = np.zeros_like(t) # Suma de senoidales
for freq, amp, fase in componentes_reales:
    senal_limpia += amp * np.sin(2 * np.pi * freq * t + fase)
nivel_ruido = 0.4 # Agrego ruido
ruido = np.random.normal(0, nivel_ruido, len(t))
senal_con_ruido = senal_limpia + ruido
mejores_soluciones = []

#Config del AG solo busco las frecuencias ampl y fase salen de DFT
num_senoidales = 5 
num_genes = num_senoidales # solo un gen por senoidal (frecuencias) en cada individuo
gene_space = [{'low': 0.5, 'high': 100} for _ in range(num_senoidales)] # busca en freq de 0 a 100
freq_bins, fft_compleja, N = calcular_dft_completa(senal_con_ruido, fs)

###########################################

# Ejecucion algoritmo AG
print("\n" + "="*60)
print("EJECUTANDO - Buscando frecuencias...")
print("="*60)

ga_instance = pygad.GA(
    num_generations=250,         # mas generaciones --> mejor busqueda
    num_parents_mating=12,       
    sol_per_pop=60,              # Poblacion grande --> explorar mejor
    num_genes=num_genes,         # 5 genes cantidad de frecuencias
    gene_space=gene_space,       
    gene_type=float,             # freq continuas
    
    fitness_func=funcion_fitness(t,num_senoidales, senal_con_ruido,freq_bins,fft_compleja,N),   # usa DFT 
    on_generation=funcionGeneracion(mejores_soluciones),    
    parent_selection_type="sss",
    keep_parents=6,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20,   
)

ga_instance.run()

##############################################

best_solution, best_fitness, _ = ga_instance.best_solution() # resultados

print(f"\nMejor fitness: {best_fitness:.6f}")
print(f"MSE final: {-best_fitness:.6f}")

frecuencias_encontradas = sorted(best_solution) # identificar freq encontradas

senoidales_separadas = []

for i, freq in enumerate(frecuencias_encontradas): # obtengo amplitus y fase de las freq encontradas
    amp, fase = extraer_parametros_de_dft(freq, freq_bins, fft_compleja, N)
    senoidales_separadas.append([freq, amp, fase])
    print(f"Senoidal {i+1}: {freq:.2f} Hz  (Amp={amp:.3f}, Fase={fase:.3f} de la DFT)")

##########################################################

senal_reconstruida_total = np.zeros_like(t) # recontruye señales individuales
senoidales_individuales = []

for freq, amp, fase in senoidales_separadas:
    senoidal = amp * np.sin(2 * np.pi * freq * t + fase)
    senoidales_individuales.append(senoidal)
    senal_reconstruida_total += senoidal

#################################################33

#comparar resueltos

print("\n" + "="*60)
print("COMPARACION CON FRECUENCIAS REALES:")
print("="*60)

freqs_reales = sorted([f for f, _, _ in componentes_reales])
freqs_encontradas = sorted(frecuencias_encontradas)

print("\n{:<15} {:<15} {:<15}".format("Real (Hz)", "Encontrada (Hz)", "Error (Hz)"))
print("-" * 45)
for i in range(len(freqs_reales)):
    real = freqs_reales[i]
    encontrada = freqs_encontradas[i]
    error = abs(real - encontrada)
    print("{:<15.2f} {:<15.2f} {:<15.3f}".format(real, encontrada, error))

print("\n" + "="*60)
print("SENOIDALES SEPARADAS")
print("El AG identifico las frecuencias usando solo la DFT")
print("="*60)