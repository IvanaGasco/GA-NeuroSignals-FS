import time
import pygad
import numpy as np
import pandas as pd
import os
from modulos_01 import cargar_datos_por_participante, crear_funcion_fitness, crear_callback_monitoreo

def ejecutar_ag():
    SOL_PER_POP = 70 
    NUM_GENERATIONS = 50 
    WEIGHT_FACTOR = 0.5 
    
    try:
        data_all = cargar_datos_por_participante()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Selección interactiva 
    user_input = input("Ingrese ID (ej: S01) o 'all': ").strip().upper()
    participantes = data_all if user_input == 'ALL' else [p for p in data_all if p['id'] == user_input]

    print(f"IDs disponibles en la carga: {[p['id'] for p in data_all]}")
    
    participantes = data_all if user_input == 'ALL' else [p for p in data_all if p['id'] == user_input]
    
    print(f"Participantes a procesar: {[p['id'] for p in participantes]}")

    resumen_final = []

    for p in participantes:
        t_inicio = time.time()
        print(f"\n{'='*50}\n>>> Optimizando {p['id']}...")
        
        fitness_func = crear_funcion_fitness(p['X'], p['y'], WEIGHT_FACTOR)
        
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
            on_generation=crear_callback_monitoreo(WEIGHT_FACTOR),
            stop_criteria="saturate_15" # Se detiene si no mejora en 15 generaciones
        )

        ga_instance.run()

        t_final = time.time()
        solution, fitness, _ = ga_instance.best_solution()
        indices = np.where(solution == 1)[0]

        # MOSTRAR EN PANTALLA
        print(f"{p['id']} FINALIZADO: {len(indices)} características seleccionadas | Fitness: {fitness:.4f}")

        # GUARDAR EN LA LISTA
        resumen_final.append({
            'Participante': p['id'],
            'Fitness': round(fitness, 4),
            'Num_Features': len(indices),
            'Indices': str(indices.tolist()) # Texto para que el CSV sea legible
        })

        df = pd.DataFrame(resumen_final)
        ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_ag.csv")
        df.to_csv(ruta_csv, index=False)
        print(f"Guardado actualizado en: {ruta_csv}")

if __name__ == "__main__":
    ejecutar_ag()