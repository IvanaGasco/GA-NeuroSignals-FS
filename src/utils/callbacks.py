import numpy as np

def crear_callback_monitoreo(weight_factor):
    """
    Crea una función que PyGAD ejecutará en cada generación para informar el progreso.
    """
    def on_generation(ga_instance):
        gen = ga_instance.generations_completed
   
        if gen % 5 == 0 or gen == 1:
            best_sol, best_fit, _ = ga_instance.best_solution()
            
            n_caract = int(np.sum(best_sol))
            
            print(f"  > Gen {gen:02d}: Fitness = {best_fit:.4f} | Características seleccionadas: {n_caract}")
            
    return on_generation
