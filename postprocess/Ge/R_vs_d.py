import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ==========================================
# 0. PLOT SETTINGS (LATEX STYLE & HUGE SIZES)
# ==========================================
paper_rc = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern (LaTeX look)
    'axes.labelsize': 28,      # Huge axis labels
    'xtick.labelsize': 20,     # Huge X numbers
    'ytick.labelsize': 20,     # Huge Y numbers
    'legend.fontsize': 16,     # Tamaño ajustado para la leyenda
    'axes.linewidth': 1.5,     # Bordes de los gráficos un poco más gruesos
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',   # Ticks hacia adentro (estilo paper clásico)
    'ytick.direction': 'in',
    'axes.titlesize': 22       # Título principal grande
}

# 1. Preparación de los datos
# Datos para n = 1 (Rojo)
d_n1 = [8, 24]
R_insitu_n1 = [-20,-20] #Not reAL
R_exsitu_n1 = [60.70, 111.94]


# Datos para n = 5 (Negro)
d_n5 = [8, 16, 24]
R_insitu_n5 = [24.0, 33.1, 49.2]
R_exsitu_n5 = [20.48, 29.96, 36.30]

# 2. Creación del gráfico
# Usamos el rc_context para aplicar el estilo LaTeX a este plot
with plt.rc_context(paper_rc):
    plt.figure(figsize=(9, 6))

    # Plots para n = 1 (Rojo) - Se engrosan las líneas y los marcadores
    plt.plot(d_n1, R_insitu_n1, 'o-', color='red', lw=2.5, markersize=10, 
             label=r'$R_{\mathrm{in-situ}} \ (n=1)$')
    plt.plot(d_n1, R_exsitu_n1, 's--', color='red', lw=2.5, markersize=10, alpha=0.7, 
             label=r'$R_{\mathrm{ex-situ}} \ (n=1)$')

    # Plots para n = 5 (Negro)
    plt.plot(d_n5, R_insitu_n5, 'o-', color='black', lw=2.5, markersize=10, 
             label=r'$R_{\mathrm{in-situ}} \ (n=5)$')
    plt.plot(d_n5, R_exsitu_n5, 's--', color='black', lw=2.5, markersize=10, alpha=0.7, 
             label=r'$R_{\mathrm{ex-situ}} \ (n=5)$')

    # 3. Estética y etiquetas (textos formateados con LaTeX puro)
    plt.title(r'$\mathbf{Resistance \ vs \ Antenna \ Gap \ (Ge \ NWs)}$', pad=15)
    plt.xlabel(r'$d \ (\mu\mathrm{m})$')
    plt.ylabel(r'$R \ (\mathrm{k}\Omega)$')

    # Reducir el número de marcas (ticks) en el eje X para que no se saturen
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Cuadrícula y leyenda
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)

    # Opcional: ajustar límites para ver mejor los datos
    plt.xlim(5, 30)
    plt.ylim(0, 130)

    plt.tight_layout()
    plt.show()