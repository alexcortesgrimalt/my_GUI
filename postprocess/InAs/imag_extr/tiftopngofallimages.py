import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ==========================================
# 1. ENRUTAMIENTO PARA IMPORTAR TU CÓDIGO
# ==========================================
ruta_actual = os.path.dirname(os.path.abspath(__file__))
# Cambia '../../' según la posición de image_handler.py
ruta_objetivo = os.path.abspath(os.path.join(ruta_actual, "../../../"))

if ruta_objetivo not in sys.path:
    sys.path.insert(0, ruta_objetivo)

from image_handler import SEMDataManager

# ==========================================
# 2. ESTILO CIENTÍFICO (LaTeX)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False
})

# ==========================================
# 3. FUNCIÓN DE EXPORTACIÓN
# ==========================================
def export_ebic_batch_clean_white(folder_path, cmap_name='plasma'):
    search_pattern = os.path.join(folder_path, "*.[tT][iI][fF]*")
    tif_files = sorted([f for f in glob.glob(search_pattern) if f.lower().endswith(('.tif', '.tiff'))])
    
    if not tif_files:
        print(f"❌ No se encontraron archivos .tif en: {folder_path}")
        return

    print(f"📁 Procesando {len(tif_files)} archivos con escala 90% y texto blanco puro...")
    
    manager = SEMDataManager()
    
    for file in tif_files:
        try:
            success = manager.load_file(file)
            if success and manager.current_map is not None:
                data = np.copy(manager.current_map).astype(float)
                
                local_min = np.nanmin(data)
                local_max = np.nanmax(data)
                
                base_name = os.path.splitext(file)[0]
                out_name = f"{base_name}.png"
                
                h, w = data.shape
                
                # Crear figura sin márgenes
                fig = plt.figure(figsize=(w/100, h/100), dpi=300)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                
                # Dibujar mapa de corriente
                im = ax.imshow(data, cmap=cmap_name, vmin=local_min, vmax=local_max, aspect='auto')
                
                

                # Guardar PNG
                plt.savefig(out_name, format='png', dpi=300)
                plt.close(fig)
                
                print(f"  ✅ Guardado: {os.path.basename(out_name)} ({local_min:.2f} a {local_max:.2f} nA)")
                
            else:
                print(f"⚠️ Error en frame EBIC: {os.path.basename(file)}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ejecuta en la carpeta donde esté el script
    export_ebic_batch_clean_white(ruta_actual, cmap_name='gray')