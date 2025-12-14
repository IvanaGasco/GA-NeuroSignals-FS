import time
import pygad
import numpy as np

# Importamos todas las funciones de nuestro módulo
from MODULOS import cargar_y_preparar_datos, crear_funcion_fitness, crear_callback_monitoreo

# =========================
# === FUNCIÓN PRINCIPAL ===
# =========================

def ejecutar_ag():
    """Define los parámetros, configura la instancia del AG y la ejecuta."""
    
    # A. PARÁMETROS GLOBALES
    NUM_PARTICIPANTES = 8
    
    # Parámetros del AG
    SOL_PER_POP = 500 # 30 100 POR LO MENOS GENERACION ALEATORIA MINIMO 500 O 1000 (10% DE LA CANTIDAD DE POSIILIDADES SEGUN LAS CARACTERISTICAS)
    NUM_GENERATIONS = 100 # ARRANCO CON 100 Y VPOY PROBANDO SI NO GENERA BUENO RESULTADOS AUMENTO EXPERIMENTALMENTE
    
    # Parámetros del Fitness
    NUM_FOLDS = 5 #--> validacion cruzada 
    C_PENALTY = 1.0 # --> regula C del SMV
    WEIGHT_FACTOR = 0.010 #--> penaliza complejidad

    # B. CARGA Y PREPARACIÓN DE DATOS
    try:
        X_plano, y_plano, num_caracteristicas_totales = cargar_y_preparar_datos(NUM_PARTICIPANTES)
    except FileNotFoundError as e:
        print(f"\n Error Crítico: {e}")
        print("Por favor, asegúrese de que 'modulos.py' está en la misma carpeta y los archivos .npy existen.")
        return

    # C. FUNCIÓN DE FITNESS 
    fitness_func_ag = crear_funcion_fitness(
        X_plano, y_plano, num_caracteristicas_totales, NUM_FOLDS, C_PENALTY, WEIGHT_FACTOR
    )
    
    # D. CONFIGURACIÓN PARÁMETROS DEL AG
    #GUARDAR MEJOR RESULTADO DE CADA GENERACION 
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=2,#10 20% DE LA POBKACION INICIAL DESP CRUZA Y MUTACIONES  SELECCIONAS DOS REPEIS MIL VECES PUEDE QUE OPTENGAS LO MISMO Y DE AHI CRUZA Y MUTACION (que se recombinen de a 2)
        fitness_func=fitness_func_ag,
        sol_per_pop=SOL_PER_POP,# poblacion mas grende --> mejores resultados --> mayor costo
        num_genes=num_caracteristicas_totales,
        gene_type=np.int8,
        gene_space=[0, 1],
        parent_selection_type="sss", # Explorar estas opciones
        crossover_type="single_point", # Esto tmb
        mutation_type="random", # y esto ir cambiando y llevar registro
        mutation_percent_genes=10,
        on_generation=crear_callback_monitoreo(WEIGHT_FACTOR),
    )

    # --- E. EJECUCIÓN Y RESULTADOS ---
    t_inicio = time.time()
    print("\nComenzando la ejecución del Algoritmo Genético...")
    ga_instance.run() 
    t_fin = time.time()
    
    # Resultados finales
    best_solution, best_fitness, _ = ga_instance.best_solution()
    features_indices = np.where(best_solution == 1)[0]

    print(f"\n--- OPTIMIZACIÓN FINALIZADA ---")
    print(f"Tiempo total de ejecución: {t_fin - t_inicio:.2f} segundos.")
    print("\n Mejor Solución Encontrada:")
    print(f"  Número final de características seleccionadas: {len(features_indices)}")
    print(f"  Índices de las características seleccionadas: {features_indices}")

# ===============================
# === PUNTO DE ENTRADA (MAIN) ===
# ===============================

if __name__ == "__main__":
    ejecutar_ag()