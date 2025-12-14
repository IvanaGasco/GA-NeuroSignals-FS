import numpy as np
import pygad
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
import time

# ==============================================
# === 1. MÓDULO DE CARGA Y PREPARACIÓN DE DATOS
# ==============================================

#TRABAJAR CON DICCIONARIO --> DATOS

def cargar_y_preparar_datos(num_participantes):
    """
    Carga los archivos .npy, verifica las formas y transforma el tensor 3D 
    (Participantes x Muestras x Características) a la matriz 2D.
    """
    try:
        # Cargar archivos guardados por el script generar_datos_eeg.py
        X_data_simulada = np.load('X_data_simulada.npy') 
        y_etiquetas_por_participante = np.load('y_etiquetas_por_participante.npy')
    
    except FileNotFoundError:
        # Si no se encuentran los archivos, se lanza un error y se detiene la ejecución
        raise FileNotFoundError(
            "Error: Los archivos 'X_data_simulada.npy' o 'y_etiquetas_por_participante.npy' no se encontraron. "
            "Asegúrese de haber ejecutado el script 'generar_datos_eeg.py' primero."
        )

    # 1. Transformación 3D a 2D NO ES NECESARIO TRABAJO CON LAS METRICES POR SEPARADO DE CADA PARTICIPANTE
    num_caracteristicas = X_data_simulada.shape[2]
    
    # X_plano: de (8, 80, 96) a (640, 96) --> Aplanamiento
    X_plano = X_data_simulada.reshape(-1, num_caracteristicas) 

    # Y_plano: de (80,) a (640,) repitiendo el patrón 8 veces
    y_plano = np.tile(y_etiquetas_por_participante, num_participantes)

    print(f"Datos cargados y aplanados. X_plano forma: {X_plano.shape}")
    
    return X_plano, y_plano, num_caracteristicas

# ====================================
# === 2. MÓDULO DE FUNCIÓN DE FITNESS  Uno con smv y otro con lda
# ====================================

def crear_funcion_fitness(X_plano, y_plano, num_caracteristicas_totales, num_folds, c_penalty, weight_factor):
    cv_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=42) # MODIFICAR POR LOS PARAMETROS
    #SACO PARAMETROS DE SVM POR CUADRÍCULA --> VER EL EL DOC
    svm_model = SVC(kernel='linear', C=c_penalty, random_state=42)

    def funcion_fitness_svm_cv(ga_instance, solution, solution_idx):
        caracteristicas_seleccionadas_idx = np.where(solution == 1)[0]
        
        if len(caracteristicas_seleccionadas_idx) == 0:
            return 0.0

        X_subconjunto = X_plano[:, caracteristicas_seleccionadas_idx]
        
        scores = cross_val_score(svm_model, X_subconjunto, y_plano, cv=cv_splitter, scoring='accuracy', n_jobs=-1)
        precisión_media_cv = scores.mean()

        num_caracteristicas_usadas = len(caracteristicas_seleccionadas_idx)
        penalizacion = (num_caracteristicas_usadas / num_caracteristicas_totales) * weight_factor
        
        fitness_final = precisión_media_cv - penalizacion
        return fitness_final

    return funcion_fitness_svm_cv

# =================================================
# === 3. MÓDULO DE MONITOREO Y EJECUCIÓN PRINCIPAL
# =================================================

# 💡 Nueva Función Fábrica para crear el callback
def crear_callback_monitoreo(weight_factor):
    """
    Función que crea y devuelve el callback on_generation, 
    encapsulando el weight_factor.
    """

    def on_generation_callback(ga_instance):
        """Función de monitoreo ejecutada después de cada generación."""
        
        gen = ga_instance.generations_completed
        best_solution, best_fitness, _ = ga_instance.best_solution()
        num_caracteristicas_totales = ga_instance.num_genes
        
        #'weight_factor' encapsulado
        num_caracteristicas_seleccionadas = np.sum(best_solution)

        if num_caracteristicas_seleccionadas > 0:
            # Fórmula para revertir el fitness y obtener la precisión estimada
            penalizacion = (num_caracteristicas_seleccionadas / num_caracteristicas_totales) * weight_factor
            precision_estimada = best_fitness + penalizacion
        else:
            precision_estimada = 0.0

        if gen % 10 == 0 or gen == 1:
            print("--------------------------------------------------")
            print(f"Generación: {gen:02d} | Fitness Final (con penal.): {best_fitness:.4f}")
            print(f"  Precisión CV Estimada: {precision_estimada:.4f}")
            print(f"  Características usadas: {num_caracteristicas_seleccionadas}/{num_caracteristicas_totales}")

    return on_generation_callback