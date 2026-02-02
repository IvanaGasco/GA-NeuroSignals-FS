import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_analisis_visual(ruta_csv, n_total_features=None):
    # 1. Cargar de datos
    df = pd.read_csv(ruta_csv)
    
    # 2. Convertir'Indices' de string a listas de enteros
    # Ejemplo: "3,6,10" -> [3, 6, 10]
    df['Indices_List'] = df['Indices'].apply(lambda x: [int(i) for i in str(x).split(',')] if pd.notnull(x) else [])

    # 3. Determinar el número total de características
    if n_total_features is None:
        todos_los_indices = [idx for sublist in df['Indices_List'] for idx in sublist]
        n_total_features = max(todos_los_indices) + 1

    # 4. Crear la Matriz Binaria (Sujetos x Características)
    matrix = np.zeros((len(df), n_total_features))
    
    for i, lista_indices in enumerate(df['Indices_List']):
        for idx in lista_indices:
            if idx < n_total_features:
                matrix[i, idx] = 1

    # Convertir a DataFrame para graficar mejor
    df_matrix = pd.DataFrame(matrix, index=df['Participante'], columns=range(n_total_features))

    # --- VISUALIZACIÓN 1: MAPA DE CALOR ---
    plt.figure(figsize=(15, 8))
    sns.heatmap(df_matrix, cmap="YlGnBu", cbar_kws={'label': 'Seleccionada (1) / No (0)'}, linewidths=0.1)
    plt.title("Mapa de Calor: Características Seleccionadas por Sujeto", fontsize=15)
    plt.xlabel("Índice de la Característica", fontsize=12)
    plt.ylabel("Participante", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(ruta_csv), "heatmap_caracteristicas.png"))
    plt.show()

    # --- VISUALIZACIÓN 2: FRECUENCIA DE SELECCIÓN ---
    frecuencias = df_matrix.sum(axis=0)
    plt.figure(figsize=(15, 5))
    frecuencias.plot(kind='bar', color='skyblue')
    plt.title("Frecuencia de Aparición de cada Característica", fontsize=15)
    plt.xlabel("Índice de la Característica", fontsize=12)
    plt.ylabel("Cantidad de Sujetos que la eligieron", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(ruta_csv), "frecuencia_caracteristicas.png"))
    plt.show()

    # --- RESUMEN POR CONSOLA ---
    print("\n" + "="*40)
    print("TOP 25 CARACTERÍSTICAS MÁS IMPORTANTES")
    print("="*40)
    top_10 = frecuencias.sort_values(ascending=False).head(25)
    for idx, count in top_10.items():
        print(f"Característica #{idx:<3} | Elegida por {int(count):>2} sujetos")

