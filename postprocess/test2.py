import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analizar_media_nw2(csv_filename):
    # 1. Cargar los datos
    # Si tu CSV no tiene nombres de columnas en la primera fila, añade: header=None
    df = pd.read_csv(csv_filename)
    
    # 2. Extraer los vectores (asumiendo la misma estructura: x en col 0, I en col 2)
    x = df.iloc[:, 0].values
    I_nw2 = df.iloc[:, 2].values
    
    # 3. Calcular la media de la corriente
    media_I = np.mean(abs(I_nw2))
    print(f"--> La corriente media en el NW2 es: {media_I:.4f} (en las unidades de tu CSV)")
    
    # 4. Generar la gráfica
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Gráfica del perfil original
    ax.plot(x, I_nw2, color='black', linewidth=2, label='Corriente NW2')
    
    # Línea horizontal indicando la media
    ax.axhline(media_I, color='red', linestyle='--', linewidth=2, 
               label=f'Media: {media_I:.4f}')
    
    # Configuración estética de la gráfica
    ax.set_title('Current Profile and Mean in NW2', fontsize=14, fontweight='bold')
    ax.set_xlabel('Position x (unidades)', fontsize=12)
    ax.set_ylabel('Current (nA / pA)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    # --- RECORDATORIO DE PRESENTACIÓN ---
    # Descomenta esto para incluir los parámetros obligatorios de tu experimento 
    # en la imagen final que le muestres a tu jefe:
    # ax.text(0.05, 0.95, 'V_acc = 5 kV\nI_beam = 100 pA\nConfig: Tip-Au-InAs...', 
    #         transform=ax.transAxes, verticalalignment='top', 
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Devolvemos el valor por si lo quieres usar en cálculos posteriores en Python
    return media_I

# --- PARA EJECUTARLO ---
# Cambia 'Perpendicular_NW_2.csv' por el nombre real de tu archivo
if __name__ == "__main__":
    valor_medio = analizar_media_nw2('Perpendicular_NW_2.csv')