import os
import json
from datetime import datetime

def iniciar_registro_experimento(config):
    # 1. Definir carpeta base
    base_dir = "runs"
    
    # 2. Crear nombre único con fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # 3. Crear la carpeta físicamente
    os.makedirs(run_dir, exist_ok=True)
    
    # 4. Guardar los parámetros en un JSON dentro de esa carpeta
    ruta_config = os.path.join(run_dir, "config.json")
    with open(ruta_config, "w") as f:
        json.dump(config, f, indent=4)
        
    # Devolvemos la ruta para que el Main sepa dónde guardar los resultados
    return run_dir
