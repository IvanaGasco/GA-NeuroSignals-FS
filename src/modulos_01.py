import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CSV_FOLDER = "data/processed/CARACTERISTICAS_CSV" 

#######################################################################################

def cargar_datos_por_participante():
    data_participantes = []

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    CSV_FOLDER = os.path.join(base_path, "data", "processed", "CARACTERISTICAS_CSV")

    print(f"Buscando en: {CSV_FOLDER}")

    if not os.path.exists(CSV_FOLDER):
        raise FileNotFoundError(f"Carpeta {CSV_FOLDER} no encontrada.")
        
    for filename in [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]:
        filepath = os.path.join(CSV_FOLDER, filename)
        df = pd.read_csv(filepath)
        
        X = df.iloc[:, :-1].values.astype(np.float64) 
        y = df['Label'].values.astype(np.int64)
    
        try:
            partes = filename.split('_')
            pid = "Desconocido"
            for p in partes:
                if p.startswith('S') and len(p) <= 3: 
                    pid = p
                    break
        except Exception:
            pid = filename
            
        data_participantes.append({'id': pid, 'X': X, 'y': y, 'N_features': X.shape[1]})
        
    print(f"--> Cargados {len(data_participantes)} participantes.")
    return data_participantes

########################################################################################################

def evaluar_svm_rapido(X, y):
    """
    Usa un SVM con parámetros por defecto y CV simple.
    Es 100 veces más rápido, ideal para calcular el fitness miles de veces.
    """
    # Pipeline: Escalado -> SVM
    # Usamos kernel='linear' o 'rbf' con C=1 por defecto 
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', random_state=42))
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return np.mean(scores)

def evaluar_svm_final(X, y, n_repeats=10):
    """
    Usa GridSearchCV y RepeatedStratifiedKFold.
    Solo se llama UNA VEZ al final del AG.
    """
    # Pipeline base
    pipe = make_pipeline(StandardScaler(), SVC(random_state=42))
    
    param_grid = {
        'svc__C': [0.1, 1, 10, 100], 
        'svc__gamma': [0.001, 0.01, 0.1, 1, 'scale'],
        'svc__kernel': ['rbf'] # O 'linear' para probar ambos
    }
    
    # Búsqueda de hiperparámetros 
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=4, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    print(f"   Mejores params: {grid_search.best_params_}")
    
    cv_final = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=42)
    
    acc_scores = cross_val_score(best_model, X, y, cv=cv_final, scoring='accuracy', n_jobs=-1) 
    tpr_scores = cross_val_score(best_model, X, y, cv=cv_final, scoring='recall', n_jobs=-1)
    
    return (np.mean(acc_scores), np.std(acc_scores), np.mean(tpr_scores), np.std(tpr_scores), best_model)


#############################################################################################################################

def crear_funcion_fitness(X_data, y_labels, weight_factor):
    # Pre-calculamos el número total para evitar recalculés
    n_total_feats = X_data.shape[1]

    def fitness_func(ga_instance, solution, solution_idx):
        # 1. Decodificar cromosoma (binario -> índices)
        indices = np.where(solution == 1)[0]
        
        # 2. Penalización dura si hay muy pocas características
        if len(indices) < 2:
            return 0.001 # Valor muy bajo para descartar esta solución
        
        # 3. Filtrar columnas
        X_subset = X_data[:, indices]
        
        # 4. Evaluación RÁPIDA (Aquí estaba el cuello de botella)
        acc = evaluar_svm_rapido(X_subset, y_labels)
        
        # 5. Cálculo del Fitness con penalización por cantidad de características
        # Objetivo: Maximizar Accuracy minimizando (ratio de features * peso)
        ratio_features = len(indices) / n_total_feats
        fitness = acc - (ratio_features * weight_factor)
        
        return fitness
    
    return fitness_func

#######################################################################

def crear_callback_monitoreo(weight_factor):
    def on_generation(ga_instance):
        gen = ga_instance.generations_completed
        # Imprimir cada 5 generaciones o en la primera
        if gen % 5 == 0 or gen == 1:
            best_sol, best_fit, _ = ga_instance.best_solution()
            
            # --- AQUÍ ESTÁ EL TRUCO: Contamos los '1' en el cromosoma ---
            n_caract = int(np.sum(best_sol))
            
            print(f"  > Gen {gen:02d}: Fitness = {best_fit:.4f} | Caractersticas: {n_caract}")
            
    return on_generation