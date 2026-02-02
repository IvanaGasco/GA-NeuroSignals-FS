import numpy as np
import pandas as pd
import os

def cargar_datos_por_participante():
    """
    Busca los archivos CSV en la carpeta de procesados y carga X e y.
    """
    data_participantes = []

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
