import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def evaluar_svm_rapido(X, y):
    """
    Tu código original: Pipeline: Escalado -> SVM.
    Ideal para el cálculo de fitness miles de veces.
    """
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale', random_state=42))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return np.mean(scores)

def evaluar_svm_final(X, y, n_repeats=10):
    """
    Tu código original: Usa GridSearchCV para encontrar los mejores parámetros.
    Solo se llama UNA VEZ al final del AG.
    """
    pipe = make_pipeline(StandardScaler(), SVC(random_state=42))
    param_grid = {
        'svc__C': [0.1, 1, 10, 100], 
        'svc__gamma': [0.001, 0.01, 0.1, 1, 'scale'],
        'svc__kernel': ['rbf'] 
    }
    
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=4, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    print(f"    Mejores params: {grid_search.best_params_}")
    
    cv_final = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=42)
    
    acc_scores = cross_val_score(best_model, X, y, cv=cv_final, scoring='accuracy', n_jobs=-1) 
    tpr_scores = cross_val_score(best_model, X, y, cv=cv_final, scoring='recall', n_jobs=-1)
    
    return (np.mean(acc_scores), np.std(acc_scores), np.mean(tpr_scores), np.std(tpr_scores), best_model)

def evaluar_lda(X, y):
    pass 
    
