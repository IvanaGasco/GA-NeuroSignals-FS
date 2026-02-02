import numpy as np
import pandas as pd
import pywt
import os

OUTPUT_FOLDER = "CARACTERISTICAS_CSV" 


# aislar las señales de un electrodo (cz)
def laplacian_filter(senal):
    """
    senal shape: (samples, channels, trials)
    Asume que el canal 0 es el central (ej. CZ) y los demás (1:) son los vecinos.
    """
    # Calculamos el promedio de los canales vecinos (índices 1 en adelante)
    # axis=1 opera sobre los canales.
    # Resultado shape: (samples, trials)
    promedio_vecinos = np.mean(senal[:, 1:, :], axis=1)
    
    # Canal central (índice 0)
    canal_central = senal[:, 0, :]
    
    # Resta directa de matrices
    laplacian = canal_central - promedio_vecinos
    
    return laplacian

def Wavelet(signals, mother, level=5):
    trials, samples = signals.shape
    wavelet_values = []
    for trial in range(trials):
        # Descomposición
        coeffs = pywt.wavedec(signals[trial], mother, mode='periodization', level=level)
        # Concatenar cD5 (coeffs[1]) y cD4 (coeffs[2])
        # coeffs[0] es cA5
        wavelet_values.append(np.concatenate((coeffs[1], coeffs[2]))) 
    return np.array(wavelet_values)

if __name__ == "__main__":
    # 1. Configuración inicial
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Carpeta creada: {OUTPUT_FOLDER}")


# BLOQUE DE EJECUCIÓN PROTEGIDO --> esto va a ser interfaz parcialmente
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    np.random.seed(42) 

    mother = 'bior4.4'
    dictionary = np.load('dictionary_512_500ms.npy', allow_pickle=True).item() 
    df_info = pd.read_excel('Database_Info.xlsx')
    lower_dominant = df_info["Lower Dominant Laterality"].tolist()
    
    voluntarios = [1,2,4,5,6,7,8,9,10,11,14,15,16,17,18,19,21,23,24,26,27,28,29,30]

    for volunteer in voluntarios:
        ini = 'S0' if volunteer < 10 else 'S'
        vol_id = ini + str(volunteer)
        
        # 1. Cargar y Procesar
        X_raw = dictionary[vol_id]['Lower']['Task1']['X']
        
        L = Wavelet(laplacian_filter(X_raw['Left']), mother)
        R = Wavelet(laplacian_filter(X_raw['Right']), mother)
        Rest = Wavelet(laplacian_filter(X_raw['Rest']), mother)

        # 2. Identificar la clase ACTIVA y contar cuántas hay (n_active)
        if lower_dominant[volunteer-1] == 'Right':
            Active = R
        else:
            Active = L
            
        n_active = Active.shape[0] 
        
        # 3. Seleccionar la misma cantidad de REPOSO
        n_rest_total = Rest.shape[0]

        #VER QUE ONDA ESTE CONTROL
      
        if n_rest_total >= n_active:
            # Caso normal: Tomamos n_active muestras al azar de Rest
            idx_rest = np.random.choice(n_rest_total, n_active, replace=False)
            Selected_Rest = Rest[idx_rest]
            
            # La clase activa se queda tal cual
            Selected_Active = Active
            
            n_final = n_active # La cantidad final por clase
        else:
            print(f"Aviso en {vol_id}: Pocas muestras de Reposo ({n_rest_total}). Recortando Activa.")
            Selected_Rest = Rest # Usamos todo el reposo disponible
            Selected_Active = Active[:n_rest_total] # Recortamos la activa
            
            n_final = n_rest_total

        # 4. Concatenar
        X_real = np.concatenate((Selected_Active, Selected_Rest), axis=0)
        
        # 5. Etiquetas Dinámicas (n_final unos, n_final ceros)
        y_final = np.concatenate((np.ones(n_final), np.zeros(n_final)), axis=0)

        # 6. Guardar
        filename = f"features_{vol_id}_{mother}.csv"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        df_features = pd.DataFrame(X_real)
        df_features['Label'] = y_final
        
        df_features.to_csv(filepath, index=False)
        print(f"Sujeto {vol_id}: {n_final} vs {n_final} -> Total {X_real.shape[0]} filas.")

    
