import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from collections import defaultdict 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score 

CSV_FOLDER = "CARACTERISTICAS_CSV" 

def cargar_datos_por_participante():
    data_participantes = []
    if not os.path.exists(CSV_FOLDER):
        raise FileNotFoundError(f"Carpeta {CSV_FOLDER} no encontrada.")
        
    for filename in [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]:
        df = pd.read_csv(os.path.join(CSV_FOLDER, filename))
        X = df.iloc[:, :-1].values.astype(np.float64) 
        y = df['Label'].values.astype(np.int64)
        pid = filename.split('_')[3]
        data_participantes.append({'id': pid, 'X': X, 'y': y, 'N_features': X.shape[1]})
    return data_participantes

#Validación cruzada SVM (más adelante puedo aplicar herencia y polimorfismo con LDA y demas clasificadores)
def cross_valid_svm(X, y, n_repeats=10):
    svm = SVC(kernel='rbf', random_state=42)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']}
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=4, scoring='accuracy', n_jobs=-1) 
    grid_search.fit(X, y)
    clf = grid_search.best_estimator_
    
    cv_final = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=42)
    acc_scores = cross_val_score(clf, X, y, cv=cv_final, scoring='accuracy', n_jobs=-1) 
    tpr_scores = cross_val_score(clf, X, y, cv=cv_final, scoring='recall')
    return (np.mean(acc_scores), np.std(acc_scores), np.mean(tpr_scores), np.std(tpr_scores), clf)


def crear_funcion_fitness(X_data_participante, y_labels_participante, num_caracteristicas_totales, weight_factor):
    def funcion_fitness_svm_cv(ga_instance, solution, solution_idx):
        indices = np.where(solution == 1)[0]
        
        # SEGURIDAD: El SVM requiere al menos 2 características para ser estable
        if len(indices) < 2:
            return 0.0001 
        
        X_sub = X_data_participante[:, indices]
        
        # Evaluación rápida para el AG (n_repeats bajo para velocidad)
        precision_media_cv, _, _, _, _ = cross_valid_svm(X_sub, y_labels_participante, n_repeats=2)
        
        penalizacion = (len(indices) / num_caracteristicas_totales) * weight_factor
        return precision_media_cv - penalizacion
    
    return funcion_fitness_svm_cv


def crear_callback_monitoreo(weight_factor):
    def on_generation(ga_instance):
        gen = ga_instance.generations_completed
        if gen % 10 == 0 or gen == 1:
            _, fit, _ = ga_instance.best_solution()
            print(f"Generación {gen}: Mejor Fitness = {fit:.4f}")
    return on_generation