import numpy as np

def crear_funcion_fitness(X_data, y_labels, eval_func, weight_factor):
    """
    X_data, y_labels: Tus matrices de datos y etiquetas.
    eval_func: Aquí recibimos la FUNCIÓN (ej: evaluar_svm_rapido).
    weight_factor: El peso para la penalización de características.
    """
    n_total_feats = X_data.shape[1]

    def fitness_func(ga_instance, solution, solution_idx):
        indices = np.where(solution == 1)[0]
        
        if len(indices) < 2: # Tu lógica original de penalización dura
            return 0.001 
        
        X_subset = X_data[:, indices]
        
        acc = eval_func(X_subset, y_labels)
        
        ratio_features = len(indices) / n_total_feats
        fitness = acc - (ratio_features * weight_factor)
        
        return fitness
    
    return fitness_func
