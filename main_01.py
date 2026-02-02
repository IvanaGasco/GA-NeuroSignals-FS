import time
import pygad
import numpy as np
import pandas as pd
import os
from datetime import datetime

from src.utils.loaders import cargar_datos_por_participante
from src.optimizers.fitness import crear_funcion_fitness
from src.utils.callbacks import crear_callback_monitoreo
from src.classifiers.models import evaluar_svm_rapido  # <--- IMPORTAMOS EL MOTOR
from src.utils.logger import iniciar_registro_experimento


def ejecutar_ag():
    # --- CONFIGURACIÓN CENTRALIZADA ---
    config = {
        "algoritmo": "Genético (PyGAD)",
        "clasificador": "SVM (Linear)",
        "extraccion": "Wavelets bior4.4",
        "params_ag": {
            "sol_per_pop": 70,
            "num_generations": 50,
            "mutation_percent": 10,
            "weight_factor": 0.5
        },
        "fecha": str(datetime.now())
    }
    
    run_path = iniciar_registro_experimento(config)

    try:
        data_all = cargar_datos_por_participante()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Selección interactiva 
    print(f"IDs disponibles: {[p['id'] for p in data_all]}")
    user_input = input("Ingrese ID (ej: S01), varios (ej: S01, S02) o 'all': ").strip().upper()

    if user_input == 'ALL':
        participantes = data_all
    else:
        ids_buscados = [s.strip() for s in user_input.split(',')]
        participantes = [p for p in data_all if p['id'] in ids_buscados]

    if not participantes:
        print(f"No se encontró ningún participante que coincida con: {user_input}")
        return

    print(f"Participantes a procesar ({len(participantes)}): {[p['id'] for p in participantes]}")

    resumen_final = []
    pa = config["params_ag"]

    for p in participantes:
        
        print(f"\n{'='*50}\n>>> Optimizando {p['id']}...")
        
        fitness_func = crear_funcion_fitness(
            X_data=p['X'], 
            y_labels=p['y'],
            eval_func=evaluar_svm_rapido, 
            weight_factor=float(pa["weight_factor"])
        )
    
        ga_instance = pygad.GA(
            num_generations=int(pa["num_generations"]),
            sol_per_pop=int(pa["sol_per_pop"]),
            mutation_percent_genes=float(pa["mutation_percent"]),
            num_parents_mating=10, # Mayor diversidad
            fitness_func=fitness_func,
            num_genes=int(p['N_features']), 
            gene_type=np.int8,
            gene_space=[0, 1],
            keep_elitism=2, # No perder la mejor solución
            on_generation=crear_callback_monitoreo(pa["weight_factor"]),
            stop_criteria="saturate_15" # Se detiene si no mejora en 15 generaciones
        )

        ga_instance.run()

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        solution, fitness, _ = ga_instance.best_solution()
        indices = np.where(solution == 1)[0]

        print(f"{p['id']} FINALIZADO: {len(indices)} características seleccionadas | Fitness: {fitness:.4f}")

        registro_completo = {
            'Fecha_Hora': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Participante': p['id'],
            'Fitness': round(float(fitness), 4),
            'Num_Features': len(indices),
            'Indices': ",".join(map(str, indices.tolist())),
            'Algoritmo': config["algoritmo"],
            'Sol_per_Pop': pa["sol_per_pop"],
            'Num_Generations': pa["num_generations"],
            'Mutation_Pct': pa["mutation_percent"],
            'Weight_Factor': pa["weight_factor"]
        }
        
        resumen_final.append(registro_completo)

        df = pd.DataFrame(resumen_final)
        ruta_csv = os.path.join(run_path, "experimento_completo.csv")
        df.to_csv(ruta_csv, index=False)
        
        print(f"Sujeto {p['id']} integrado al archivo final en {run_path}")
    
if __name__ == "__main__":
    ejecutar_ag()