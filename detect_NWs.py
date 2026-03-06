import numpy as np
import scipy.ndimage as ndi
from scipy.signal import find_peaks

class NWDetector:
    def __init__(self, pixel_size_m=1e-6):
        self.pixel_size_m = pixel_size_m

    def detect_and_track(self, ebic_data, p0_px, p1_px, length_px, rel_prominence=0.2, search_width_px=15, step_px=2, expected_nw_count=0):
        """
        Detecta y rastrea nanohilos iterativamente. 
        Si expected_nw_count > 0, se queda estrictamente con los N picos de mayor corriente.
        """
        c0, r0 = p0_px
        c1, r1 = p1_px

        v_c = c1 - c0
        v_r = r1 - r0
        L_px = np.hypot(v_c, v_r)
        
        if L_px < 2: 
            return [], []

        u_c = v_c / L_px
        u_r = v_r / L_px
        
        n_c = -u_r
        n_r = u_c

        # 1. SEMILLAS: Buscar máximos en la línea trazada
        N = int(np.ceil(L_px))
        c_vals = c0 + u_c * np.linspace(0, L_px, N)
        r_vals = r0 + u_r * np.linspace(0, L_px, N)

        ebic_profile = ndi.map_coordinates(ebic_data, [r_vals, c_vals], order=1, mode='nearest')
        prominence = np.ptp(ebic_profile) * rel_prominence
        
        # Obtenemos también las alturas (heights) para poder ordenarlos
        peaks, properties = find_peaks(ebic_profile, prominence=prominence, height=-np.inf)

        # --- NUEVO FILTRO: Número exacto de NWs ---
        if expected_nw_count > 0 and len(peaks) > expected_nw_count:
            peak_heights = properties['peak_heights']
            # Obtener los índices de los N picos más altos
            top_indices = np.argsort(peak_heights)[-expected_nw_count:]
            # Reordenar espacialmente para que se dibujen en orden geométrico
            peaks = np.sort(peaks[top_indices])

        nw_lines = []
        all_tracked_points = []
        half_l = length_px / 2.0

        for p in peaks:
            seed_c = c_vals[p]
            seed_r = r_vals[p]
            tracked_points = [(seed_c, seed_r)]
            
            # 2. CAMINATA ITERATIVA
            for direction in [1, -1]:
                curr_c, curr_r = seed_c, seed_r
                curr_nc, curr_nr = n_c * direction, n_r * direction
                
                traveled = 0
                while traveled < half_l:
                    curr_c += curr_nc * step_px
                    curr_r += curr_nr * step_px
                    traveled += step_px
                    
                    cs_c = curr_c + u_c * np.linspace(-search_width_px/2, search_width_px/2, int(search_width_px))
                    cs_r = curr_r + u_r * np.linspace(-search_width_px/2, search_width_px/2, int(search_width_px))
                    
                    cs_profile = ndi.map_coordinates(ebic_data, [cs_r, cs_c], order=1, mode='nearest')
                    max_idx = np.argmax(cs_profile)
                    
                    curr_c = cs_c[max_idx]
                    curr_r = cs_r[max_idx]
                    tracked_points.append((curr_c, curr_r))

            # 3. AJUSTE LINEAL ORTOGONAL (SVD)
            tracked_points_arr = np.array(tracked_points)
            X = tracked_points_arr[:, 0]
            Y = tracked_points_arr[:, 1]
            
            X_mean, Y_mean = np.mean(X), np.mean(Y)
            
            centered = np.vstack((X - X_mean, Y - Y_mean)).T
            if len(centered) > 2:
                _, _, Vh = np.linalg.svd(centered, full_matrices=False)
                dir_c, dir_r = Vh[0] 
            else:
                dir_c, dir_r = n_c, n_r 
            
            # --- NUEVO: Forzar dirección de Izquierda a Derecha ---
            # Si el vector apunta hacia la izquierda (dir_c negativo), lo invertimos.
            # Si es exactamente vertical (dir_c == 0), forzamos que apunte hacia arriba o abajo según prefieras (aquí hacia arriba).
            if dir_c < 0:
                dir_c = -dir_c
                dir_r = -dir_r
            elif dir_c == 0 and dir_r < 0:
                dir_r = -dir_r
            
            start_c = X_mean - dir_c * half_l
            start_r = Y_mean - dir_r * half_l
            end_c = X_mean + dir_c * half_l
            end_r = Y_mean + dir_r * half_l
            
            nw_lines.append(((start_c, start_r), (end_c, end_r)))
            all_tracked_points.append(tracked_points)

        # Empaquetar variables en juego para el mapa final
        run_parameters = {
            'p0_px': p0_px, 'p1_px': p1_px, 'length_px': length_px,
            'rel_prominence': rel_prominence, 'search_width_px': search_width_px,
            'step_px': step_px, 'expected_nw_count': expected_nw_count
        }

        return nw_lines, all_tracked_points, run_parameters