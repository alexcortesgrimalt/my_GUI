import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analizar_perfil_m1(csv_filename):
    # 1. Cargar los datos
    # Si tu CSV no tiene nombres de columnas en la primera fila, añade: header=None
    df = pd.read_csv(csv_filename)
    
    # 2. Extraer los vectores (iloc usa índices: 0 es la 1ª columna, 2 es la 3ª columna)
    x = df.iloc[:, 0].values
    M1 = df.iloc[:, 2].values
    
    # 3. Generar la gráfica
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Gráfica solo de M1 original
    ax.plot(x, M1, color='black', linewidth=2, label='M1 Original')
    
    # Configuración estética de la gráfica
    ax.set_title('Current Profile in NW1 (M1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Position x (um)', fontsize=12)
    ax.set_ylabel('Current (nA)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # --- TIP PARA TU PRESENTACIÓN ---
    # Como comentamos, para presentar resultados de mapas EBIC/EBAC, 
    # necesitas mostrar variables clave. Puedes añadirlas como texto en la gráfica:
    # ax.text(0.05, 0.95, 'V_acc = X kV\nI_beam = Y pA\nHigh/Low/GND config', 
    #         transform=ax.transAxes, verticalalignment='top', 
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# --- PARA EJECUTARLO ---
# Cambia 'Perpendicular_NW_1.csv' por el nombre real de tu archivo exportado de la GUI
if __name__ == "__main__":
    analizar_perfil_m1('Perpendicular_NW_1.csv')