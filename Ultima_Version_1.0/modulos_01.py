import numpy as np
import pandas as pd
import os
from Caracteristicas_Walavet import cross_valid_svm

# --- Módulos de Clasificación para Fitness (requeridos por cross_valid_svm) ---
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score 

# --- CARPETA DONDE SE ESPERAN LOS CSV GENERADOS ---
CSV_FOLDER = "CARACTERISTICAS_CSV" 

# ====================================
# === 1. MÓDULO DE FUNCIÓN DE FITNESS
# ====================================

def crear_funcion_fitness(X_data_participante, y_labels_participante, num_caracteristicas_totales, weight_factor):
    """
    Crea la función de fitness (closure) para un PARTICIPANTE específico.
    """
    
    def funcion_fitness_svm_cv(ga_instance, solution, solution_idx):
        # 'solution' es el genotipo (vector de 0s y 1s)
        
        # 1. Definición de caracteristicas_seleccionadas_idx
        caracteristicas_seleccionadas_idx = np.where(solution == 1)[0]
        
        if len(caracteristicas_seleccionadas_idx) == 0:
            # Penalización máxima por no seleccionar nada
            return -1.0 

        
        # PASO 1: DEFINICIÓN DE X_subconjunto
        # Filtra la matriz de datos original (X_data_participante) para incluir SÓLO las columnas (características)
        # donde el valor de 'solution' (el genotipo) es 1.
        X_subconjunto = X_data_participante[:, caracteristicas_seleccionadas_idx]
        
        # PASO 2: Evaluación con SVM (obteniendo la precisión)
        # Llama a la función de clasificación con el subconjunto de datos.
        # cross_valid_svm_AG debe retornar (acc_mean, acc_std, tpr_mean, tpr_std)
        precision_media_cv, _, _, _,_ = cross_valid_svm(
            X_subconjunto, 
            y_labels_participante
        )
        
        # PASO 3: Cálculo de la Penalización
        num_caracteristicas_usadas = len(caracteristicas_seleccionadas_idx)
        # weight_factor y num_caracteristicas_totales se usan desde el scope exterior (closure)
        penalizacion = (num_caracteristicas_usadas / num_caracteristicas_totales) * weight_factor
        
        # PASO 4: DEFINICIÓN DE fitness_final
        # Aplica la fórmula: Fitness = Precisión - Penalización.
        fitness_final = precision_media_cv - penalizacion
        
        return fitness_final 
    
    return funcion_fitness_svm_cv

# ==============================================
# === 2. MÓDULO DE CARGA Y PREPARACIÓN DE DATOS 
# ==============================================

def cargar_datos_por_participante():
    """
    Busca y carga todas las matrices de características binarias (80x96) 
    de la carpeta CSV_FOLDER, una por participante.
    
    Retorna una lista de diccionarios con los datos (X, y) de cada matriz.
    """
    data_participantes = []
    
    if not os.path.exists(CSV_FOLDER):
        raise FileNotFoundError(f"Error: La carpeta '{CSV_FOLDER}' no fue encontrada. Asegúrese de ejecutar 'Caracteristicas_Walavet.py' primero.")
        
    feature_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]

    if not feature_files:
        raise FileNotFoundError(f"Error: No se encontraron archivos CSV binarios en '{CSV_FOLDER}'.")

    for filename in feature_files:
        filepath = os.path.join(CSV_FOLDER, filename)
        
        try:
            df = pd.read_csv(filepath)
            
            # X_data: Todas las columnas excepto la última ('Label')
            X_data = df.iloc[:, :-1].values.astype(np.float64) 

            # Y_labels: La última columna ('Label')
            Y_labels = df['Label'].values.astype(np.int64)

            # Extraemos el ID del participante
            participante_id_match = filename.split('_')[3] if len(filename.split('_')) > 3 else filename.split('.')[0]
            
            data_participantes.append({
                'id': participante_id_match,
                'filename': filename,
                'X': X_data,
                'y': Y_labels,
                'N_features': X_data.shape[1]
            })

            print(f"Cargado: {participante_id_match} | Shape X: {X_data.shape}, Shape y: {Y_labels.shape}")

        except Exception as e:
            print(f"Advertencia: No se pudo cargar o procesar el archivo {filename}. Error: {e}")
            continue

    if not data_participantes:
         raise ValueError("Error: No se pudo cargar ninguna matriz de características.")

    return data_participantes

# =================================================
# === 3. MÓDULO DE MONITOREO 
# =================================================

def crear_callback_monitoreo(weight_factor):
    def on_generation_callback(ga_instance):
        """Función de monitoreo ejecutada después de cada generación."""
        
        gen = ga_instance.generations_completed
        best_solution, best_fitness, _ = ga_instance.best_solution()
        num_caracteristicas_totales = ga_instance.num_genes
        
        num_caracteristicas_seleccionadas = np.sum(best_solution)

        if num_caracteristicas_seleccionadas > 0:
            penalizacion = (num_caracteristicas_seleccionadas / num_caracteristicas_totales) * weight_factor
            precision_estimada = best_fitness + penalizacion
        else:
            precision_estimada = 0.0

        if gen % 20 == 0 or gen == 1:
            print("--------------------------------------------------")
            print(f"Generación: {gen:02d} | Fitness Final (con penal.): {best_fitness:.4f}")
            print(f"  Precisión CV Estimada: {precision_estimada:.4f}")
            print(f"  Características usadas: {num_caracteristicas_seleccionadas}/{num_caracteristicas_totales}")

    return on_generation_callback