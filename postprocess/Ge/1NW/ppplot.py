import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np # Importante para el ajuste lineal

# 1. Path configuration
path = './*.csv' 
files = glob.glob(path)

if not files:
    print("No .csv files found in the folder.")

plot_data = []
results = [] # Para guardar d y R

for file in files:
    try:
        filename = os.path.basename(file)
        # Ajustado para tu nomenclatura: OXF008_S29_L_IV_devNW{n}G{d}.csv
        # Si el nombre es solo el número, d_value = float(d_str) funciona.
        # Si no, usamos una extracción rápida:
        d_str = filename.split('G')[-1].replace('.csv', '')
        d_value = float(d_str)

        skip = 0
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if 'I1,V1' in line:
                    skip = i + 2 
                    break
        
        df = pd.read_csv(file, skiprows=skip, header=None, usecols=[0, 1], names=['I', 'V'])
        df = df.dropna()

        # --- CÁLCULO DE RESISTENCIA ---
        # polyfit(x, y, grado 1) -> y = mx + c. Aquí I = m*V + c
        # La pendiente m es la conductancia (1/R)
        slope, intercept = np.polyfit(df['V'], df['I'], 1)
        resistance = 1 / slope
        
        results.append({'d': d_value, 'R': resistance})
        plot_data.append((d_value, df, resistance))
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Ordenar de menor a mayor d para la tabla
results.sort(key=lambda x: x['d'])

# Imprimir resultados para tu tabla de LaTeX
print("\n--- Resultados para la tabla ---")
print(f"{'d (um)':<10} | {'R (Ohm)':<20}")
print("-" * 35)
for res in results:
    print(f"{res['d']:<10} | {res['R']:<20.2f}")

# 3. Plotting
plt.figure(figsize=(10, 6))

# Ordenar para el plot (opcional, aquí de mayor a menor como tenías)
plot_data.sort(key=lambda x: x[0], reverse=True)

for d, df, r in plot_data:
    plt.plot(df['V'], df['I'], label=f'{d} µm ($R = {r/1e3:.1f}\ k\Omega$)')

plt.title('I-V Characteristic Curves - InAs NWs')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(title="Antenna Gap (d)", loc='best')
plt.tight_layout()
plt.show()