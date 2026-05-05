import matplotlib.pyplot as plt

# 1. Preparación de los datos (basado en tu tabla)
# Datos para n = 1 (Rojo)
d_n1 = [8, 16, 24, 32]
R_insitu_n1 = [35.9, 122.4, 266.9, 285.6]
R_exsitu_n1 = [53.99, 123.68, 247.54, 347.31]

# Datos para n = 5 (Negro)
d_n5 = [16, 24, 32]
R_insitu_n5 = [23.3, 46.9, 71.4]
R_exsitu_n5 = [22.77, 39.76, 63.34]

# 2. Creación del gráfico
plt.figure(figsize=(9, 6))

# Plots para n = 1 (Rojo)
plt.plot(d_n1, R_insitu_n1, 'o-', color='red', label='$R_{in-situ}$ ($n=1$)')
plt.plot(d_n1, R_exsitu_n1, 's--', color='red', alpha=0.7, label='$R_{ex-situ}$ ($n=1$)')

# Plots para n = 5 (Negro)
plt.plot(d_n5, R_insitu_n5, 'o-', color='black', label='$R_{in-situ}$ ($n=5$)')
plt.plot(d_n5, R_exsitu_n5, 's--', color='black', alpha=0.7, label='$R_{ex-situ}$ ($n=5$)')

# 3. Estética y etiquetas
plt.title('Resistance vs Antenna Gap (InAs NWs)', fontsize=14)
plt.xlabel('Distance $d$ ($\mu$m)', fontsize=12)
plt.ylabel('Resistance $R$ ($k\Omega$)', fontsize=12)

plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True)

# Opcional: ajustar límites para ver mejor los datos
plt.xlim(5, 35)
plt.ylim(0, 400)

plt.tight_layout()
plt.show() 