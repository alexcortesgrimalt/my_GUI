import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# ==========================================
# 0. PLOT SETTINGS (LATEX STYLE & HUGE SIZES)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern (LaTeX look)
    # 'text.usetex': True,     # <-- UNCOMMENT THIS IF YOU HAVE MIKTEX/TEXLIVE INSTALLED
    'axes.labelsize': 28,      # Huge axis labels
    'xtick.labelsize': 20,     # Huge X numbers
    'ytick.labelsize': 20,     # Huge Y numbers
    'legend.fontsize': 16,     # Tamaño ajustado para la leyenda
    'legend.title_fontsize': 18 # Tamaño para el título de la leyenda
})

# ==========================================
# 1. METADATA EXTRACTION / DATA PROCESSING
# ==========================================
path = './*.csv' 
files = glob.glob(path)

if not files:
    print("No .csv files found in the folder.")

plot_data = []
results = [] # Para guardar d y R

for file in files:
    try:
        filename = os.path.basename(file)
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
        # El cálculo se hace con los datos originales (Amperios) para que la R sea correcta
        slope, intercept = np.polyfit(df['V'], df['I'], 1)
        resistance = 1 / slope
        
        results.append({'d': d_value, 'R': resistance})
        plot_data.append((d_value, df, resistance))
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

results.sort(key=lambda x: x['d'])

print("\n--- Resultados para la tabla ---")
print(f"{'d (um)':<10} | {'R (Ohm)':<20}")
print("-" * 35)
for res in results:
    print(f"{res['d']:<10} | {res['R']:<20.2f}")

# ==========================================
# 2. PLOTTING
# ==========================================
plt.figure(figsize=(9, 6))

# Ordenar de mayor a menor para el plot
plot_data.sort(key=lambda x: x[0], reverse=True)

for d, df, r in plot_data:
    label_str = fr'${d} \ \mu\mathrm{{m}} \ (R = {r/1e3:.1f} \ \mathrm{{k}}\Omega)$'
    
    # Multiplicamos la corriente por 1e6 para mostrarla en microamperios
    plt.plot(df['V'], df['I'] * 1e6, linewidth=2.5, label=label_str)

# Ejes con formato MathText estricto (etiqueta Y actualizada a microamperios)
plt.xlabel(r'$V \ (\mathrm{V})$')
plt.ylabel(r'$I \ (\mu\mathrm{A})$')

# Reducir el número de valores/marcas en el eje X
plt.locator_params(axis='x', nbins=5)

# Cuadrícula y layout
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title=r"$\mathbf{Antenna \ Gap \ (d)}$", loc='best')
plt.tight_layout()
plt.xlim(-2, 2)
plt.ylim(-100, 100)
plt.show()