import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar los datos desde el archivo CSV
# Asegúrate de que el archivo d4_NWdata.csv esté en la misma carpeta que este script
archivo_csv = 'd4_NWdata.csv'
df = pd.read_csv(archivo_csv)

# 2. Extraer las variables (X = Distancia en micras, Y = Corriente cruda en nA)
x = df['Distance']
y = df['I_raw']

# 3. Realizar el ajuste lineal (Polinomio de grado 1: y = m*x + b)
# m es la pendiente (dI/dx) y b es la intersección
m, b = np.polyfit(x, y, 1)

# Crear la línea del ajuste usando la ecuación y = mx + b
ajuste_y = m * x + b

# 4. Mostrar los resultados numéricos por consola
print("=== Resultados del Ajuste Lineal (EBAC) ===")
print(f"Pendiente (dI/dx) : {m:.4f} nA/um")
print(f"Intersección (b)  : {b:.4f} nA")
print(f"Ecuación del fit  : I(x) = {m:.4f} * x + {b:.4f}")
print("===========================================")

# 5. Graficar los datos y el ajuste
plt.figure(figsize=(10, 6))

# Plotear los datos originales como puntos o línea sutil
plt.plot(x, y, label='Datos Originales (I_raw)', color='blue', alpha=0.6, linewidth=2)

# Plotear la línea de tendencia del ajuste lineal
plt.plot(x, ajuste_y, label=f'Ajuste Lineal ($dI/dx$ = {m:.2f} nA/$\mu$m)', color='red', linestyle='--', linewidth=2)

# Estética del gráfico
plt.title('Perfil de Corriente y Ajuste Lineal (EBAC a -4V)', fontsize=14)
plt.xlabel('Distancia a lo largo del Nanohilo ($\mu$m)', fontsize=12)
plt.ylabel('Corriente Medida (nA)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

# Guardar la imagen de alta calidad para tu reporte
plt.savefig('Ajuste_Lineal_EBAC_NW.png', dpi=300)

# Mostrar la gráfica
plt.show()