import numpy as np
import pandas as pd
import pywt
import os
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from collections import defaultdict 

# -------------------------------------------------------------
# --- CONFIGURACIÓN Y CARGA DE DATOS ---
# -------------------------------------------------------------

# --- CARPETA DE SALIDA ---
OUTPUT_FOLDER = "CARACTERISTICAS_CSV" 

# --- UMBRAL PARA BINARIZACIÓN ---
THRESHOLD_PERCENTAGE = 0.05 
# --------------------------

dictionary = np.load('dictionary_512_500ms.npy', allow_pickle=True).item() 
df_info = pd.read_excel('Database_Info.xlsx')
lower_dominant = (df_info["Lower Dominant Laterality"]).tolist()

# -------------------------------------------------------------
# --- FUNCIONES AUXILIARES ---
# -------------------------------------------------------------

#(EXTRAIDA DE "Wavelet_T1_volunteers.py")
def laplacian_filter(senal):
  samples, channels, trials = senal.shape
  laplacian = np.empty((samples, trials), np.float64) 
 
  for i in range(trials):
      for k in range(samples):
          sample_sum = 0
          for j in range(1, channels):
              sample_sum += senal[k, j, i] 
          laplacian[k, i] = senal[k, 0, i] - (sample_sum / (channels - 1))  
  return laplacian

# --- FUNCIÓN WAVELET --- (EXTRAIDA DE "Wavelet_T1_volunteers.py")
def Wavelet(signals, mother, fs=512, nperseg=512):
    trials, samples = signals.shape
    wavelet_values = []
    for trial in range(trials):
        # Se utiliza level=5 para obtener cD5 y cD4 (banda 8-30 Hz)
        cA5,cD5,cD4,cD3,cD2,cD1 = pywt.wavedec(signals[trial], mother, mode='periodization',level=5)
        # Concatenación: 20 (D5) + 2* 38 (D4) = 96 coeficientes reales
        wavelet_values.append(np.concatenate((cD5,cD4))) 

    wavelet_values = np.array(wavelet_values)
    return wavelet_values

#(EXTRAIDA DE "Wavelet_T1_volunteers.py")
def cross_valid_svm(X, y, n_repeats=10): 
    # 1. Búsqueda de los mejores hiperparámetros C y gamma (GridSearch)
    svm = SVC(kernel='rbf', random_state=42)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']}
    # Usamos cv=4 para la búsqueda rápida
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=4, scoring='accuracy', n_jobs=-1) 
    grid_search.fit(X, y)
    clf = grid_search.best_estimator_
    
    # 2. Evaluación final con Repeated Stratified K-Fold 
    # Usa el parámetro n_repeats
    cv_final = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=42)
    
    acc_scores = cross_val_score(clf, X, y, cv=cv_final, scoring='accuracy', n_jobs=-1) 
    tpr_scores = cross_val_score(clf, X, y, cv=cv_final, scoring='recall')

    # Retorna resultados completos
    return (np.mean(acc_scores), np.std(acc_scores),
            np.mean(tpr_scores), np.std(tpr_scores),
            clf)

# --- Implemento FUNCIÓN DE GUARDADO (Binario 80x96) ---
def save_features_matrix(volunteer_id, mother_wavelet, X_features, Y_labels):
    """
    Guarda la matriz BINARIA (ceros y unos) de 80x96 y etiquetas (80) en un archivo CSV.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Nombre del archivo reflejando que es BINARIO y tiene 96 columnas
    filename = f"features_binary_96cols_{volunteer_id}_{mother_wavelet}.csv" 
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    df_features = pd.DataFrame(X_features)
    df_features['Label'] = Y_labels
    df_features.to_csv(filepath, index=False) 
    
    print(f"[GUARDADO CSV] Matriz BINARIA (80x96) de {volunteer_id} en {filepath}")


# -------------------------------------------------------------
# --- BLOQUE PRINCIPAL ---
# -------------------------------------------------------------

mother_wavelet = 'bior4.4'
mother = mother_wavelet

svm_results_summary = defaultdict(list)
    
voluntarios = [1,2,4,5,6,7,8,9,10,11,14,15,16,17,18,19,21,23,24,26,27,28,29,30]

print(f"--- INICIANDO PROCESO con Wavelet: {mother_wavelet} y BINARIZACIÓN ---")
print(f"--- Dimensiones finales: {80} filas x {96} columnas (D5+D4) ---")
print(f"--- Los archivos se guardarán en la carpeta: {OUTPUT_FOLDER} ---")
print("-" * 50)

for volunteer in voluntarios:
    
    if volunteer < 10:
        ini = 'S0'
    else:
        ini = 'S'
    
    vol_id = ini + str(volunteer)

    X = dictionary[vol_id]['Lower']['Task1']['X']

    ### 1. Extracción de Wavelet (valores REALES 80x96)
    laplace_left = laplacian_filter(X['Left'])
    laplace_right = laplacian_filter(X['Right'])
    laplace_rest = laplacian_filter(X['Rest'])

    Wavelet_left = Wavelet(laplace_left, mother=mother)
    Wavelet_right = Wavelet(laplace_right, mother=mother)
    Wavelet_rest = Wavelet(laplace_rest, mother=mother)

    ### 2. Concatenación de ensayos (valores REALES 80x96)
    if lower_dominant[volunteer-1] == 'Right':
        index = np.random.choice(Wavelet_rest.shape[0], 40, replace=False) 
        selection = Wavelet_rest[index]
        X_real = np.concatenate((Wavelet_right, selection), axis=0) 
    else:
        index = np.random.choice(Wavelet_rest.shape[0], 40, replace=False) 
        selection = Wavelet_rest[index]
        X_real = np.concatenate((Wavelet_left, selection), axis=0) 

    y_final = np.concatenate((np.ones(40),np.zeros(40)), axis=0) 

    # --- 3. BINARIZACIÓN (Presencia o Ausencia) ---
    T = THRESHOLD_PERCENTAGE * np.max(np.abs(X_real)) 
    X_final = np.where(np.abs(X_real) > T, 1, 0) # X_final es la matriz binaria (80x58)
    
    # 4. GUARDAR CARACTERÍSTICAS BINARIAS (80x96) y ETIQUETAS (80) EN CSV
    save_features_matrix(vol_id, mother, X_final, y_final)

    # 5. EVALUAR CON SVM (usando X_final BINARIA)
    acc_m, acc_std, tpr_m, tpr_std, clf = cross_valid_svm(X_final, y_final)
    
    svm_results_summary[vol_id] = {
        'Accuracy_mean': acc_m, 
        'Accuracy_std': acc_std, 
        'Recall_mean': tpr_m, 
        'Recall_std': tpr_std
    }
    
    print(f"  [SVM] {vol_id}: Acc={acc_m:.4f} (+/-{acc_std:.4f}), Recall={tpr_m:.4f} (+/-{tpr_std:.4f})")

print("-" * 50)
print("PROCESO TERMINADO. Las matrices BINARIAS (80x58) están en la carpeta.")

avg_acc = np.mean([r['Accuracy_mean'] for r in svm_results_summary.values()])
avg_tpr = np.mean([r['Recall_mean'] for r in svm_results_summary.values()])
print("-" * 50)
print(f"PROMEDIO TOTAL (N={len(voluntarios)}): Accuracy={avg_acc:.4f}, Recall={avg_tpr:.4f}")
