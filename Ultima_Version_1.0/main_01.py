import time
import pygad
import numpy as np
import pandas as pd 

from modulos_01 import ( 
    cargar_datos_por_participante, 
    crear_funcion_fitness, 
    crear_callback_monitoreo
)

# =========================
# === FUNCIÓN PRINCIPAL ===
# =========================

def ejecutar_ag():
    """Define los parámetros, configura la instancia del AG y la ejecuta por participante(s) seleccionado(s)."""
    
    # A. PARÁMETROS GLOBALES DEL ALGORITMO GENÉTICO
    SOL_PER_POP = 500  
    NUM_GENERATIONS = 100 
    WEIGHT_FACTOR = 0.010 # Parámetro de Penalización por Complejidad
    
    # B. CARGA Y PREPARACIÓN DE DATOS (Lista de participantes)
    t_inicio_carga = time.time()
    try:
        # data_participantes es una lista de diccionarios con TODOS los participantes
        data_participantes_all = cargar_datos_por_participante()
    except Exception as e:
        print(f"\n Error Crítico en la carga de datos: {e}")
        return
    t_fin_carga = time.time()
    print(f"\nCarga de datos finalizada en {t_fin_carga - t_inicio_carga:.2f} segundos.")
    
    
    # --------------------------------------------------------------------------
    # SELECCIÓN INTERACTIVA DE PARTICIPANTES 
    # --------------------------------------------------------------------------
    
    # 1. Obtener IDs disponibles y crear un mapa para fácil acceso
    available_ids = [p['id'] for p in data_participantes_all]
    available_ids_sorted = sorted(available_ids)
    id_map = {p['id']: p for p in data_participantes_all}
    
    print("\n" + "="*80)
    print("SELECCIÓN DE PARTICIPANTES PARA ENTRENAMIENTO AG")
    print("="*80)
    print("Participantes disponibles:")
    print(", ".join(available_ids_sorted))
    print("\nOpciones de Selección:")
    print(" - Escriba 'all' para procesar a todos.")
    print(" - Escriba el ID de un participante (ej: S01).")
    print(" - Escriba una lista de IDs separados por coma (ej: S01,S05,S10).")
    
    # Capturar la entrada del usuario
    user_input = input("\nIngrese su selección: ").strip().upper()
    
    participantes_a_procesar = []
    
    if user_input == 'ALL':
        # Opción 1: Todos
        participantes_a_procesar = data_participantes_all
        print("-> Seleccionados todos los participantes.")
    else:
        # Opción 2: Lista específica
        selected_ids = [id.strip() for id in user_input.split(',') if id.strip()]
        
        for sel_id in selected_ids:
            if sel_id in id_map:
                participantes_a_procesar.append(id_map[sel_id])
            else:
                print(f"-> Advertencia: El ID '{sel_id}' no es válido y será ignorado.")

    if not participantes_a_procesar:
        print("\nError: No se seleccionó ningún participante válido. Terminando la ejecución.")
        return
    
    # --------------------------------------------------------------------------
    # --- FIN DE LA SELECCIÓN ---
    # --------------------------------------------------------------------------
    
    # Almacenamiento de resultados
    resultados_ag_por_participante = {}

    # ============================================
    # === BUCLE DE EJECUCIÓN POR PARTICIPANTE ===
    # ============================================
    
    print("\n" + "="*80)
    print("INICIANDO - Algoritmo Genético de Selección de Características (AG-FS)")
    print(f"-> Procesando {len(participantes_a_procesar)} participante(s) seleccionado(s).")
    print("="*80)
    
    t_inicio_total = time.time()
    
    for participante in participantes_a_procesar:
        
        pid = participante['id']
        X_data = participante['X']
        y_labels = participante['y']
        N_features = participante['N_features']
        
        print("\n" + "#"*40)
        print(f"--- EJECUTANDO AG para Participante: {pid} ---")
        print(f"--- Genes totales (Características): {N_features} ---")
        print("#"*40)

        # C. FUNCIÓN DE FITNESS (Específica para este participante)
        fitness_func_ag = crear_funcion_fitness(
            X_data, y_labels, N_features, WEIGHT_FACTOR
        )
        
        # D. CONFIGURACIÓN E INICIALIZACIÓN DEL AG
        ga_instance = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=2,
            fitness_func=fitness_func_ag,
            sol_per_pop=SOL_PER_POP,
            num_genes=N_features, 
            gene_type=np.int8,
            gene_space=[0, 1],
            parent_selection_type="sss", 
            crossover_type="single_point",
            mutation_type="random", 
            mutation_percent_genes=10,
            on_generation=crear_callback_monitoreo(WEIGHT_FACTOR),
        )

        # E. EJECUCIÓN Y RESULTADOS
        t_inicio_ag = time.time()
        ga_instance.run() 
        t_fin_ag = time.time()
        
        # Resultados finales del AG para el participante
        best_solution, best_fitness, _ = ga_instance.best_solution()
        
        # Cálculo de precisión sin penalización para el reporte
        num_caracteristicas_seleccionadas = np.sum(best_solution)
        if num_caracteristicas_seleccionadas > 0:
            penalizacion = (num_caracteristicas_seleccionadas / N_features) * WEIGHT_FACTOR
            precision_estimada = best_fitness + penalizacion
        else:
             precision_estimada = 0.0

        features_indices = np.where(best_solution == 1)[0]
        
        print("\n--- RESULTADOS AG PARA EL PARTICIPANTE ---")
        print(f"  Tiempo de ejecución del AG: {t_fin_ag - t_inicio_ag:.2f} segundos.")
        print(f"  Mejor Fitness (con penalización): {best_fitness:.4f}")
        print(f"  Precisión CV Estimada (sin penalización): {precision_estimada:.4f}")
        print(f"  Características seleccionadas: {len(features_indices)}/{N_features}")
        
        # Almacenar resultados
        resultados_ag_por_participante[pid] = {
            'Best_Fitness_Penalizado': best_fitness,
            'Accuracy_Estimada': precision_estimada,
            'Num_Features_Selected': len(features_indices),
            'Selected_Features_Indices': features_indices.tolist()
        }

    # =========================================
    # === RESUMEN FINAL DE LOS PARTICIPANTES SELECCIONADOS ===
    # =========================================
    
    t_fin_total = time.time()
    
    print("\n" + "="*80)
    print(f"PROCESO COMPLETO. Tiempo total: {t_fin_total - t_inicio_total:.2f} segundos.")
    print("RESUMEN DE SELECCIÓN DE CARACTERÍSTICAS POR PARTICIPANTE:")
    print("="*80)
    
    # Reporte tabular
    print("{:<15} {:<15} {:<15} {:<15}".format("Participante", "Acc. Estimada", "Fitness Final", "# Caract. Sel."))
    print("-" * 60)
    all_accs = []
    for vol_id, res in resultados_ag_por_participante.items():
        print("{:<15} {:<15.4f} {:<15.4f} {:<15}".format(
            vol_id, 
            res['Accuracy_Estimada'], 
            res['Best_Fitness_Penalizado'], 
            res['Num_Features_Selected'])
        )
        all_accs.append(res['Accuracy_Estimada'])

    print("-" * 60)
    if all_accs:
        print(f"PROMEDIO DE ACCURACY GLOBAL (N={len(all_accs)}): {np.mean(all_accs):.4f}")

# ===============================
# === PUNTO DE ENTRADA (MAIN) ===
# ===============================

if __name__ == "__main__":
    ejecutar_ag()