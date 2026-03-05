import numpy as np
import scipy.ndimage as ndi
from scipy.signal import find_peaks

class NWDetector:
    def __init__(self, pixel_size_m=1e-6):
        self.pixel_size_m = pixel_size_m

    def detect_and_track(self, ebic_data, p0_px, p1_px, length_px, rel_prominence=0.2, search_width_px=15, step_px=2):
        """
        Detecta y rastrea nanohilos iterativamente, culminando en un ajuste lineal.
        """
        c0, r0 = p0_px
        c1, r1 = p1_px

        # Vector de la línea base (perpendicular manual)
        v_c = c1 - c0
        v_r = r1 - r0
        L_px = np.hypot(v_c, v_r)
        
        # --- CORRECCIÓN AQUÍ: Devolver dos listas vacías ---
        if L_px < 2: 
            return [], []

        u_c = v_c / L_px
        u_r = v_r / L_px
        
        # Dirección de propagación aproximada (normal a la manual)
        n_c = -u_r
        n_r = u_c

        # 1. SEMILLAS: Buscar máximos en la línea trazada
        N = int(np.ceil(L_px))
        c_vals = c0 + u_c * np.linspace(0, L_px, N)
        r_vals = r0 + u_r * np.linspace(0, L_px, N)

        ebic_profile = ndi.map_coordinates(ebic_data, [r_vals, c_vals], order=1, mode='nearest')
        prominence = np.ptp(ebic_profile) * rel_prominence
        peaks, _ = find_peaks(ebic_profile, prominence=prominence)

        nw_lines = []
        all_tracked_points = [] # <-- Lista para guardar el historial
        half_l = length_px / 2.0

        for p in peaks:
            seed_c = c_vals[p]
            seed_r = r_vals[p]
            tracked_points = [(seed_c, seed_r)]
            
            # 2. CAMINATA ITERATIVA (Hacia arriba y hacia abajo del NW)
            for direction in [1, -1]:
                curr_c, curr_r = seed_c, seed_r
                curr_nc, curr_nr = n_c * direction, n_r * direction
                
                traveled = 0
                while traveled < half_l:
                    # Dar un paso en la dirección de propagación
                    curr_c += curr_nc * step_px
                    curr_r += curr_nr * step_px
                    traveled += step_px
                    
                    # Definir ventana de búsqueda transversal
                    cs_c = curr_c + u_c * np.linspace(-search_width_px/2, search_width_px/2, int(search_width_px))
                    cs_r = curr_r + u_r * np.linspace(-search_width_px/2, search_width_px/2, int(search_width_px))
                    
                    # Extraer EBIC en la ventana y buscar el máximo local
                    cs_profile = ndi.map_coordinates(ebic_data, [cs_r, cs_c], order=1, mode='nearest')
                    max_idx = np.argmax(cs_profile)
                    
                    # Actualizar a la coordenada real de máxima corriente
                    curr_c = cs_c[max_idx]
                    curr_r = cs_r[max_idx]
                    tracked_points.append((curr_c, curr_r))

            # 3. AJUSTE LINEAL ORTOGONAL (SVD)
            tracked_points_arr = np.array(tracked_points)
            X = tracked_points_arr[:, 0]
            Y = tracked_points_arr[:, 1]
            
            # Centro de masa del nanohilo
            X_mean, Y_mean = np.mean(X), np.mean(Y)
            
            # Matriz de covarianza y SVD
            centered = np.vstack((X - X_mean, Y - Y_mean)).T
            if len(centered) > 2:
                _, _, Vh = np.linalg.svd(centered, full_matrices=False)
                dir_c, dir_r = Vh[0] 
            else:
                dir_c, dir_r = n_c, n_r 
            
            # Calcular extremos
            start_c = X_mean - dir_c * half_l
            start_r = Y_mean - dir_r * half_l
            end_c = X_mean + dir_c * half_l
            end_r = Y_mean + dir_r * half_l
            
            nw_lines.append(((start_c, start_r), (end_c, end_r)))
            all_tracked_points.append(tracked_points)

        # --- CORRECCIÓN AQUÍ: Devolver ambas listas ---
        return nw_lines, all_tracked_points