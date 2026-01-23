import time
import pygad
import numpy as np
from modulos_01 import cargar_datos_por_participante, crear_funcion_fitness, crear_callback_monitoreo

def ejecutar_ag():
    SOL_PER_POP = 100 # Ajustado para balance velocidad/precisión
    NUM_GENERATIONS = 50 
    WEIGHT_FACTOR = 0.02 # Mayor penalización para forzar menos características
    
    try:
        data_all = cargar_datos_por_participante()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Selección interactiva (Simplificada para ejemplo: procesa el primero o 'all')
    user_input = input("Ingrese ID (ej: S01) o 'all': ").strip().upper()
    participantes = data_all if user_input == 'ALL' else [p for p in data_all if p['id'] == user_input]

    for p in participantes:
        print(f"\n>>> Optimizando {p['id']}...")
        
        fitness_func = crear_funcion_fitness(p['X'], p['y'], p['N_features'], WEIGHT_FACTOR)
        
        ga_instance = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=10, # Mayor diversidad
            fitness_func=fitness_func,
            sol_per_pop=SOL_PER_POP,
            num_genes=p['N_features'], 
            gene_type=np.int8,
            gene_space=[0, 1],
            keep_elitism=2, # No perder la mejor solución
            mutation_percent_genes=5,
            on_generation=crear_callback_monitoreo(WEIGHT_FACTOR)
        )

        ga_instance.run()
        
        solution, fitness, _ = ga_instance.best_solution()
        indices = np.where(solution == 1)[0]
        print(f"FINALIZADO {p['id']}: Caract. seleccionadas: {len(indices)} | Fitness: {fitness:.4f}")

if __name__ == "__main__":
    ejecutar_ag()