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
    # 'text.usetex': True,     # <-- Uncomment if you have TeX installed
    'axes.labelsize': 28,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

# ==========================================
# 1. METADATA EXTRACTION / DATA PROCESSING
# ==========================================
path = './*.csv'
files = glob.glob(path)

if not files:
    print("No .csv files found in the folder.")

plot_data = []
results = []

for file in files:
    try:
        filename = os.path.basename(file)

        # Extraer d desde nombres tipo "...G8.csv"
        d_str = filename.split('G')[-1].replace('.csv', '')
        d_value = float(d_str)

        skip = 0
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if 'I1,V1' in line:
                    skip = i + 2
                    break

        df = pd.read_csv(
            file,
            skiprows=skip,
            header=None,
            usecols=[0, 1],
            names=['I', 'V']
        )

        df = df.dropna()

        # Ajuste lineal para calcular R
        slope, intercept = np.polyfit(df['V'], df['I'], 1)
        resistance = 1 / slope

        results.append({
            'd': d_value,
            'R': resistance
        })

        plot_data.append((d_value, df, resistance))

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ==========================================
# 2. PRINT TABLE
# ==========================================
results.sort(key=lambda x: x['d'])

print("\n--- Resultados para la tabla ---")
print(f"{'d (um)':<10} | {'R (Ohm)':<20}")
print("-" * 35)

for res in results:
    print(f"{res['d']:<10} | {res['R']:<20.2f}")

# ==========================================
# 3. PLOTTING
# ==========================================
plt.figure(figsize=(9, 6))

# Ordenar de mayor a menor
plot_data.sort(key=lambda x: x[0], reverse=True)

for d, df, r in plot_data:

    label_str = (
        fr'${d:g}\,\mu\mathrm{{m}}'
        fr'\ (R={r/1e3:.1f}\,\mathrm{{k}}\Omega)$'
    )

    # Curva especial: d = 8 um
    if np.isclose(d, 8):

        # Ajuste lineal en microamperios
        m, b = np.polyfit(df['V'], df['I'] * 1e6, 1)

        # Malla muy densa para que el corte sea exacto
        V_dense = np.linspace(-2.2, 2.2, 5000)
        I_dense = m * V_dense + b

        # Mostrar SOLO la parte central [-2, 2]
        mask = (V_dense >= -2.0) & (V_dense <= 2.0)

        plt.plot(
            V_dense[mask],
            I_dense[mask],
            linewidth=2.5,
            label=label_str
        )

    else:

        plt.plot(
            df['V'],
            df['I'] * 1e6,
            linewidth=2.5,
            label=label_str
        )

# ==========================================
# 4. AXES & STYLE
# ==========================================
plt.xlabel(r'$V\;(\mathrm{V})$')
plt.ylabel(r'$I\;(\mu\mathrm{A})$')

plt.locator_params(axis='x', nbins=5)

plt.grid(True, linestyle='--', alpha=0.4)

plt.legend(
    title=r'$\mathbf{Antenna\ Gap\ (d)}$',
    loc='best'
)

plt.xlim(-2.2, 2.2)
plt.ylim(-100, 100)

plt.tight_layout()
plt.show()