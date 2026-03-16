import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import csv

def analyze_iv_curve(csv_filepath):
    """
    Lee un archivo CSV de curva I-V, extrae metadatos, grafica los datos 
    y calcula la resistencia mediante un ajuste lineal.
    """
    metadata = {}
    data_start_line = 0
    
    # 1. Analizar el archivo para extraer metadatos y encontrar el inicio de los datos
    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # Limpiar celdas vacías de la fila
            clean_row = [cell.strip() for cell in row if cell.strip()]
            
            if not clean_row:
                continue
                
            # Extraer metadatos interesantes
            if clean_row[0] == "Sample Name" and len(clean_row) > 1:
                metadata["Sample Name"] = clean_row[1]
            elif clean_row[0] == "Executed" and len(clean_row) > 1:
                metadata["Date"] = clean_row[1]
            elif clean_row[0] == "Start/Bias" and len(clean_row) > 1:
                metadata["Start V"] = clean_row[1]
            elif clean_row[0] == "Stop" and len(clean_row) > 1:
                metadata["Stop V"] = clean_row[1]
            elif clean_row[0] == "Compliance" and len(clean_row) > 1:
                metadata["Compliance (A)"] = clean_row[1]
                
            # Detectar la cabecera de las columnas de datos
            if clean_row[0].startswith("I DUT") or clean_row[0].startswith("I1"):
                data_start_line = i + 1
                # Seguimos iterando para asegurarnos de saltar ambas posibles cabeceras ('I1,V1' y 'I DUT, V DUT')
                pass
            
            # Si vemos un número con formato científico o decimal en la primera columna, ¡hemos llegado a los datos!
            # y salimos del bucle.
            if data_start_line > 0 and i > data_start_line:
                try:
                    float(clean_row[0])
                    data_start_line = i # Esta es la fila real donde empiezan los números
                    break
                except ValueError:
                    pass

    # 2. Cargar los datos numéricos usando pandas
    # Usamos usecols=[0, 1] para ignorar las comas vacías al final de las líneas
    df = pd.read_csv(csv_filepath, skiprows=data_start_line, header=None, usecols=[0, 1], names=['Current_A', 'Voltage_V'])
    
    # Limpiar posibles filas NaN si el CSV tiene líneas en blanco al final
    df = df.dropna()

    I = df['Current_A'].values
    V = df['Voltage_V'].values

    # 3. Calcular la Resistencia (Ajuste lineal I vs V)
    # Según la Ley de Ohm: I = (1/R) * V + offset
    # Si ajustamos I vs V, la pendiente 'm' es la conductancia G (1/R)
    slope, intercept, r_value, p_value, std_err = linregress(V, I)
    
    # Resistencia es la inversa de la pendiente
    resistance_ohms = 1.0 / slope
    
    # Determinar si mostrar en Ohms, kOhms o MOhms para mayor claridad
    if resistance_ohms >= 1e6:
        res_str = f"{resistance_ohms/1e6:.2f} MΩ"
    elif resistance_ohms >= 1e3:
        res_str = f"{resistance_ohms/1e3:.2f} kΩ"
    else:
        res_str = f"{resistance_ohms:.2f} Ω"

    # 4. Generar la gráfica exhaustiva
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Scatter plot de los datos crudos
    ax.scatter(V, I * 1e6, color='#1f77b4', label='Measured Data', s=15, alpha=0.7)
    
    # Línea del ajuste
    fit_I = (slope * V + intercept) * 1e6 # Pasado a microAmperios para la gráfica
    ax.plot(V, fit_I, color='red', linestyle='--', linewidth=2, 
            label=f'Linear Fit (R² = {r_value**2:.4f})')

    # Configuración de los ejes
    ax.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Current (μA)', fontsize=12, fontweight='bold')
    
    # Títulos y metadatos
    sample_name = metadata.get("Sample Name", "Unknown Sample")
    ax.set_title(f"I-V Curve: {sample_name}", fontsize=14, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Caja de información con los parámetros integrados
    info_text = (
        f"◆ Calculated Resistance: {res_str}\n"
        f"◆ Fit Equation: I = V/{res_str} + {intercept*1e6:.2f} μA\n"
        f"◆ Sweep: {metadata.get('Start V', 'N/A')}V to {metadata.get('Stop V', 'N/A')}V\n"
        f"◆ Compliance: {metadata.get('Compliance (A)', 'N/A')} A\n"
        f"◆ Date: {metadata.get('Date', 'N/A')}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#cccccc')
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    plt.tight_layout()
    
    # Guardar y mostrar
    output_filename = f"IV_Fit_{sample_name.replace(' ', '_')}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Analysis complete! Resistance = {res_str}")
    print(f"✅ Plot saved as: {output_filename}")
    
    plt.show()
    
    return resistance_ohms, metadata

# ==========================================
# CÓMO USAR EL SCRIPT
# Reemplaza 'tus_datos.csv' con el nombre de tu archivo
# ==========================================
if __name__ == "__main__":
    csv_file = "IV_Curve_4H_3L.csv" 
    R, meta = analyze_iv_curve(csv_file)
    pass