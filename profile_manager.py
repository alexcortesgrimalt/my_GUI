import numpy as np
from matplotlib.lines import Line2D

class InteractiveProfile:
    """Clase para manejar un único perfil interactivo."""
    def __init__(self, ax, p0, p1, t, d1, d2, idx, color='#ff7f0e'):
        self.ax = ax
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.t = t      # Posición paramétrica (0 a 1) sobre la línea principal
        self.d1 = d1    # Longitud hacia arriba (dirección normal)
        self.d2 = d2    # Longitud hacia abajo (dirección -normal)
        self.idx = idx  # Número del perfil
        self.color = color
        
        # Calcular vectores de dirección y normal
        v = self.p1 - self.p0
        self.length = np.linalg.norm(v)
        self.u = v / self.length if self.length > 0 else np.array([1, 0])
        self.n = np.array([-self.u[1], self.u[0]]) # Perpendicular
        
        # Crear los artistas gráficos (Línea discontinua con marcadores en los extremos y centro para arrastrar)
        self.line = Line2D([], [], color=self.color, linestyle='--', linewidth=2, 
                           marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        
        self.text = self.ax.text(0, 0, str(self.idx), color=self.color, fontsize=12, fontweight='bold',
                                 ha='center', va='center', zorder=5,
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        self.ax.add_line(self.line)
        self.update_positions()

    def update_positions(self):
        """Actualiza la geometría de la línea basada en t, d1 y d2."""
        self.t = np.clip(self.t, 0.0, 1.0) # Restringir al segmento de la línea original
        B = self.p0 + self.t * (self.p1 - self.p0) # Centro
        
        P1 = B + self.d1 * self.n # Extremo superior
        P2 = B - self.d2 * self.n # Extremo inferior
        
        # Data de la línea: P1 -> B -> P2
        self.line.set_data([P1[0], B[0], P2[0]], [P1[1], B[1], P2[1]])
        
        # Posición del texto (ligeramente desplazado del extremo P1)
        offset = max(self.length * 0.03, (self.d1 + self.d2) * 0.05)
        self.text.set_position((P1[0] + self.n[0] * offset, P1[1] + self.n[1] * offset))

    def remove(self):
        """Limpia los artistas del gráfico."""
        try: self.line.remove()
        except: pass
        try: self.text.remove()
        except: pass


class ProfileManager:
    """Gestor general para múltiples perfiles interactivos."""
    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self.profiles = []
        self.active_handle = None # (idx_perfil, tipo_handle) -> 0: P1, 1: Centro, 2: P2
        self.p0 = None
        self.p1 = None

    def generate_profiles(self, p0, p1, num_profiles, default_length):
        """Crea N perfiles equiespaciados."""
        self.clear()
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        
        if num_profiles == 1:
            t_vals = [0.5]
        else:
            t_vals = np.linspace(0, 1, num_profiles)
            
        for i, t in enumerate(t_vals):
            prof = InteractiveProfile(self.ax, self.p0, self.p1, t, default_length, default_length, i + 1)
            self.profiles.append(prof)
            
        self.canvas.draw()

    def clear(self):
        """Elimina todos los perfiles actuales."""
        for p in self.profiles: p.remove()
        self.profiles.clear()
        self.active_handle = None
        self.canvas.draw()

    def get_closest_handle(self, x_px, y_px, max_dist_px=15):
        """Encuentra el tirador (handle) más cercano al clic del ratón."""
        if x_px is None or y_px is None: return None
        
        best_dist = float('inf')
        best_match = None

        for i, prof in enumerate(self.profiles):
            x_data, y_data = prof.line.get_data() # Índices: 0=P1, 1=Centro, 2=P2
            for j in range(3):
                # Convertir coordenadas de los datos a píxeles de la pantalla para calcular distancia real
                disp_pt = self.ax.transData.transform((x_data[j], y_data[j]))
                dist = np.hypot(disp_pt[0] - x_px, disp_pt[1] - y_px)
                
                if dist < best_dist and dist < max_dist_px:
                    best_dist = dist
                    best_match = (i, j)
                    
        return best_match

    # --- EVENTOS INTERACTIVOS MATPLOTLIB ---
    def on_press(self, event):
        if not self.profiles or event.inaxes != self.ax: return False
        
        handle = self.get_closest_handle(event.x, event.y)
        if handle:
            self.active_handle = handle
            return True # Evento consumido
        return False

    def on_drag(self, event):
        if not self.active_handle or event.inaxes != self.ax: return False
        if event.xdata is None or event.ydata is None: return False

        prof_idx, handle_idx = self.active_handle
        prof = self.profiles[prof_idx]
        mouse_pt = np.array([event.xdata, event.ydata])
        min_len = prof.length * 0.02 # Longitud mínima para que no se invierta la línea

        if handle_idx == 1: # Arrastrando el CENTRO (mover a lo largo de la línea base)
            v = prof.p1 - prof.p0
            v_norm = np.dot(v, v)
            if v_norm > 0:
                t_new = np.dot(mouse_pt - prof.p0, v) / v_norm
                prof.t = t_new

        elif handle_idx == 0: # Arrastrando P1 (Extremo superior)
            B = prof.p0 + prof.t * (prof.p1 - prof.p0)
            d_new = np.dot(mouse_pt - B, prof.n)
            prof.d1 = max(d_new, min_len) # Evitar que sea menor de 0

        elif handle_idx == 2: # Arrastrando P2 (Extremo inferior)
            B = prof.p0 + prof.t * (prof.p1 - prof.p0)
            d_new = np.dot(mouse_pt - B, -prof.n)
            prof.d2 = max(d_new, min_len)

        prof.update_positions()
        self.canvas.draw()
        return True

    def on_release(self, event):
        if self.active_handle:
            self.active_handle = None
            return True
        return False