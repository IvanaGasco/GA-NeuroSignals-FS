import numpy as np

def extraer_parametros_de_dft(frecuencia, freq_bins, fft_compleja, N): # partiendo de freq busca ampl y fase
    
    idx = np.argmin(np.abs(freq_bins - frecuencia)) # bin mas cercano a la freq
    
    magnitud = np.abs(fft_compleja[idx]) / N * 2  # extraigo datos de espectro el 2 es x simetria
    fase = np.angle(fft_compleja[idx])
    
    return magnitud, fase

def calcular_dft_completa(senal, fs):
    N = len(senal)
    fft_vals = np.fft.fft(senal) 
    fft_freq = np.fft.fftfreq(N, 1/fs) ## vector de frecuencias
    
    mask = fft_freq >= 0 # freq positivas
    return fft_freq[mask], fft_vals[mask], N # valores positivos de freq y ver amplitud


def calcular_dft_completa(senal, fs):
    N = len(senal)
    fft_vals = np.fft.fft(senal) 
    fft_freq = np.fft.fftfreq(N, 1/fs) ## vector de frecuencias
    
    mask = fft_freq >= 0 # freq positivas
    return fft_freq[mask], fft_vals[mask], N # valores positivos de freq y ver amplitud

def extraer_parametros_de_dft(frecuencia, freq_bins, fft_compleja, N): # partiendo de freq busca ampl y fase
    
    idx = np.argmin(np.abs(freq_bins - frecuencia)) # bin mas cercano a la freq
    
    magnitud = np.abs(fft_compleja[idx]) / N * 2  # extraigo datos de espectro el 2 es x simetria
    fase = np.angle(fft_compleja[idx])
    
    return magnitud, fase

def funcion_fitness(t, num_senoidales, senal_con_ruido, freq_bins, fft_compleja, N):

    def fitness_func(ga_instance, solution, solution_idx): # funcion fitness con la señal final 
        
        senal_reconstruida = np.zeros_like(t) #arreglo de ceros del mismo tamaño
        
        for i in range(num_senoidales):
            freq = solution[i] #freq propuestas por AG   
            amp, fase = extraer_parametros_de_dft(freq, freq_bins, fft_compleja, N) # extrae datos
            senal_reconstruida += amp * np.sin(2 * np.pi * freq * t + fase) # recontruye esta componente de la señal
        
        mse = np.mean((senal_con_ruido - senal_reconstruida) ** 2) # saca el error de la reconstruccion
        
        fitness = -mse # actualiza fitness
        return fitness
        
    return fitness_func

def funcionGeneracion(mejores_soluciones):

    def on_generation(ga_instance):
        gen = ga_instance.generations_completed
        solution, fitness, _ = ga_instance.best_solution()
        mejores_soluciones.append((gen, fitness, solution.copy()))
        
        if gen % 30 == 0 or gen == 1:
            print(f"Gen {gen}: Fitness={fitness:.6f}, MSE={-fitness:.6f}")
            # Mostrar frecuencias encontradas
            freqs = sorted(solution)
            print(f"  Frecuencias: {[f'{f:.1f}' for f in freqs]}")
    return on_generation
