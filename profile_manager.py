import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PyQt6.QtWidgets import (QMainWindow, QToolBar, QFileDialog, QMessageBox, QDialog,
                             QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from perpendicular_fitting import PerpendicularFitter

def gradient_with_window(x, y, window=9):
    """
    Compute gradient using a moving window with linear fit.
    This provides smoother derivatives than np.gradient by fitting
    a line to a local window around each point.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    n = len(x)
    k = window // 2
    dy = np.full(n, np.nan)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        xi = x_arr[lo:hi]
        yi = y_arr[lo:hi]
        if xi.size >= 2:
            p = np.polyfit(xi, yi, 1)
            dy[i] = p[0]
    return dy

class ProfilePlotWindow(QMainWindow):
    def __init__(self, prof_idx, dist, sem, ebic, vc, selected_keys, unit_label="\u03BCm"):
        super().__init__()
        
        # --- NUEVO: ESTILO CIENTÍFICO / LATEX ---
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
            "mathtext.fontset": "cm",      # Usa fuente tipo LaTeX para las matemáticas
            "axes.formatter.use_mathtext": True,
            "axes.linewidth": 0.8,         # Bordes de los gráficos más finos
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",       # Ticks hacia adentro (estilo clásico paper)
            "ytick.direction": "in",
            "font.size": 11,               # Tamaño base de la fuente
            "axes.titlesize": 12,
            "axes.labelsize": 11
        })
        # ----------------------------------------

        self.setWindowTitle(f"Perpendicular {prof_idx} Data")
        self.resize(700, 200 * len(selected_keys) + 100)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        self.fig, self.axes = plt.subplots(len(selected_keys), 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.setCentralWidget(self.canvas)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.addToolBar(self.toolbar)
        
        # Barra de herramientas para exportar
        toolbar = QToolBar("Export Tools")
        self.addToolBar(toolbar)
        
        save_action = QAction("Save Plot (.png)", self)
        save_action.triggered.connect(self.save_plot)
        toolbar.addAction(save_action)
        
        csv_action = QAction("Save Data (.csv)", self)
        csv_action.triggered.connect(self.save_csv)
        toolbar.addAction(csv_action)
        
        # Barra de herramientas para fitting
        fitting_toolbar = QToolBar("Fitting Tools")
        self.addToolBar(fitting_toolbar)
        
        param_action = QAction("Configure Fit Parameters", self)
        param_action.triggered.connect(self._on_configure_parameters)
        fitting_toolbar.addAction(param_action)
        
        fit_action = QAction("Fit & Extract", self)
        fit_action.triggered.connect(self._on_fit_properties)
        fitting_toolbar.addAction(fit_action)
        
        save_props_action = QAction("Save Properties (.csv)", self)
        save_props_action.triggered.connect(self.save_properties)
        fitting_toolbar.addAction(save_props_action)
        
        # Asegurarnos de que axes sea iterable
        if len(selected_keys) == 1:
            ax_list = [self.axes]
        else:
            ax_list = self.axes
            
        # Cálculos de datos
        self.dist = dist
        self.sem_norm = (sem - np.min(sem)) / (np.ptp(sem) + 1e-12)
        self.i = ebic
        self.abs_i = np.abs(ebic)
        self.vc = vc # <--- GUARDAMOS LOS DATOS DE VOLTAJE AQUÍ
        
        if unit_label == "\u03BCm":
            dist_um_array = self.dist
        elif unit_label == "nm":
            dist_um_array = self.dist * 1e-3
        elif unit_label == "mm":
            dist_um_array = self.dist * 1e3
        else:
            dist_um_array = self.dist


        if len(self.dist) > 1:
            self.deriv_i = gradient_with_window(dist_um_array, ebic, window=9)
        else:
            self.deriv_i = np.zeros_like(self.dist)

        # --- CÁLCULO DE RESISTIVIDAD PUNTO A PUNTO ---
        # 1. Suavizado y logaritmo para la corriente
        pos = self.abs_i[self.abs_i > 0]
        floor = max(np.min(pos) * 0.1, 1e-12) if pos.size > 0 else 1e-12
        self.ln_i = np.log(np.maximum(self.abs_i, floor))
        
        # 2. Derivada de ln(I) (ya lo tenías)
        if len(dist) > 1: self.deriv = gradient_with_window(dist, self.ln_i, window=9)
        else: self.deriv = np.zeros_like(self.ln_i)

        if self.vc is not None:
            # Convertimos I(nA) a Amperios. Usamos un límite inferior (1 fA) para evitar dividir por 0
            i_amp = np.maximum(self.abs_i, 1e-6) * 1e-9  
            # R = |V| / |I| (en Ohmios)
            self.resistance = np.abs(self.vc) / i_amp
        else:
            self.resistance = np.zeros_like(self.dist)
            
        # 3. Derivada del Voltaje y Resistividad Local
        if len(dist) > 1 and self.vc is not None:
            # Factor de escala temporal para derivadas espaciales (asumiendo dist en µm)
            factor_cm = 1e-4 if unit_label == "\u03BCm" else 1e-7 if unit_label == "nm" else 1e-1
            dist_cm = self.dist * factor_cm 
            
            self.deriv_vc = gradient_with_window(dist_cm, self.vc, window=9)
            
            # Conversiones físicas
            # Prevención de divisiones por cero (limitamos corriente mínima a 1 fA para el cálculo)
            i_amp = np.maximum(self.abs_i, 1e-6) * 1e-9  
        else:
            self.deriv_vc = np.zeros_like(self.dist)
            self.resistivity = np.zeros_like(self.dist)
        
        # Storage for fit results
        self.properties = None
        
        pos = self.abs_i[self.abs_i > 0]
        floor = max(np.min(pos) * 0.1, 1e-12) if pos.size > 0 else 1e-12
        self.ln_i = np.log(np.maximum(self.abs_i, floor))
        
        if len(dist) > 1:
            self.deriv = gradient_with_window(dist, self.ln_i, window=9)
        else:
            self.deriv = np.zeros_like(self.ln_i)
            
        # Dibujar cada gráfico solicitado
        for idx, key in enumerate(selected_keys):
            ax = ax_list[idx]
            if key == 'sem':
                ax.plot(dist, self.sem_norm, color='#1f77b4', lw=1)
                ax.set_ylabel(r"SEM norm", color='#1f77b4')
            elif key == 'i':
                ax.plot(dist, self.i, color='#9467bd', lw=1)
                ax.set_ylabel(r"$I$ [nA]", color='#9467bd')
                ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.7)
            elif key == 'abs_i':
                ax.plot(dist, self.abs_i, color='#d62728', lw=1)
                ax.set_ylabel(r"$|I|$ [nA]", color='#d62728')
            elif key == 'ln_i':
                ax.plot(dist, self.ln_i, color='#ff7f0e', lw=1)
                ax.set_ylabel(r"$\ln |I|$", color='#ff7f0e')
            elif key == 'deriv':
                ax.plot(dist, self.deriv, color='#2ca02c', lw=1)
                ax.set_ylabel(r"d$\ln(I)$/dx [1/" + unit_label + "]", color='#2ca02c')
                ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.7)
            elif key == 'vc': 
                if self.vc is not None:
                    ax.plot(dist, self.vc, color='#17becf', lw=1)
                    ax.set_ylabel(r"Voltage [V]", color='#17becf')
                    ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.7)
            elif key == 'deriv_vc':
                if self.vc is not None:
                    ax.plot(dist, self.deriv_vc, color='#e377c2', lw=1)
                    ax.set_ylabel(r"d$V$/dx [V/cm]", color='#e377c2')
                    ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.7)
            elif key == 'r': 
                ax.plot(dist, self.resistance, color='#8c564b', lw=1)
                ax.set_ylabel(r"Resistance [$\Omega$]", color='#8c564b')
            elif key == 'deriv_i':
                ax.plot(dist, self.deriv_i, color='#ff7f0e', lw=1)
                ax.set_ylabel(r"d$I$/dx [nA/" + unit_label + "]", color='#ff7f0e')
                ax.axhline(0, color='gray', linestyle=':', lw=0.8, alpha=0.7)
                
            # Grid más sutil, tipo paper
            ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        ax_list[-1].set_xlabel(f"Distance ({unit_label})")
        self.fig.suptitle(f"Profile Extraction: Perpendicular {prof_idx}", fontsize=13)
        self.fig.tight_layout()
        
        # Find and store the ln_i axis for plotting fits
        self._find_ln_i_axis(selected_keys)
        
    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", f"Perpendicular_{self.windowTitle().split()[1]}.png", "PNG (*.png)")
        if path: self.fig.savefig(path, dpi=300, bbox_inches='tight')
        
    
    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"Perpendicular_{self.windowTitle().split()[1]}.csv", "CSV (*.csv)")
        if path:
            # --- HEADER MODIFICADO ---
            header = "Distance,SEM_norm,I_raw,abs_I,ln_abs_I,deriv_ln_I,Voltage_V,dV_dx_V_cm,Resistance_Ohm"
            vc_data = self.vc if self.vc is not None else np.full_like(self.dist, np.nan)
            deriv_vc_data = self.deriv_vc if hasattr(self, 'deriv_vc') else np.full_like(self.dist, np.nan)
            
            # --- DATA STACK MODIFICADO ---
            data = np.column_stack([self.dist, self.sem_norm, self.i, self.abs_i, 
                                    self.ln_i, self.deriv, vc_data, deriv_vc_data, self.resistance])
            np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.6e')

    def _find_ln_i_axis(self, selected_keys):
        """Find and store the axis that displays ln(I) for fit plotting."""
        self.ln_i_axis = None
        self.ln_i_axis_idx = None
        
        if len(selected_keys) == 1:
            ax_list = [self.axes]
        else:
            ax_list = self.axes
        
        for idx, key in enumerate(selected_keys):
            if key == 'ln_i':
                self.ln_i_axis = ax_list[idx]
                self.ln_i_axis_idx = idx
                break
    
    def _on_configure_parameters(self):
        """Callback to show the parameter configuration dialog."""
        if self.show_fitting_parameters_dialog():
            QMessageBox.information(self, 'Parameters Updated',
                                  'Fitting parameters have been updated.\nNew settings will be used for the next fit.')
    
    def _on_fit_properties(self):
        """Callback to run fitting with current parameters."""
        self.fit_properties()
    
    def show_fitting_parameters_dialog(self):
        """Show dialog for adjusting fitting parameters."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Fitting Parameters')
        dialog.setGeometry(100, 100, 400, 350)
        
        layout = QVBoxLayout()
        
        # Min points
        layout.addWidget(QLabel("Minimum points for fit:"))
        spin_min_pts = QSpinBox()
        spin_min_pts.setMinimum(3)
        spin_min_pts.setMaximum(50)
        spin_min_pts.setValue(getattr(self, '_fit_min_points', 6))
        layout.addWidget(spin_min_pts)
        
        # Skip near junction
        layout.addWidget(QLabel("Points to skip near junction:"))
        spin_skip = QSpinBox()
        spin_skip.setMinimum(0)
        spin_skip.setMaximum(20)
        spin_skip.setValue(getattr(self, '_fit_skip_near', 3))
        layout.addWidget(spin_skip)
        
        # SNR threshold
        layout.addWidget(QLabel("SNR threshold for tail truncation:"))
        spin_snr = QDoubleSpinBox()
        spin_snr.setMinimum(0.5)
        spin_snr.setMaximum(10.0)
        spin_snr.setSingleStep(0.5)
        spin_snr.setValue(getattr(self, '_fit_snr_threshold', 3.0))
        layout.addWidget(spin_snr)
        
        # SCR width
        layout.addWidget(QLabel("SCR width estimate (µm, 0=auto):"))
        spin_scr = QDoubleSpinBox()
        spin_scr.setMinimum(0.0)
        spin_scr.setMaximum(5.0)
        spin_scr.setSingleStep(0.1)
        spin_scr.setValue(getattr(self, '_fit_scr_width', 0.0))
        layout.addWidget(spin_scr)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Fit with These Parameters")
        btn_cancel = QPushButton("Cancel")
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        # Connect buttons
        def on_ok():
            self._fit_min_points = spin_min_pts.value()
            self._fit_skip_near = spin_skip.value()
            self._fit_snr_threshold = spin_snr.value()
            scr_w = spin_scr.value()
            self._fit_scr_width = scr_w if scr_w > 0 else None
            dialog.accept()
        
        btn_ok.clicked.connect(on_ok)
        btn_cancel.clicked.connect(dialog.reject)
        
        dialog.setLayout(layout)
        return dialog.exec() == QDialog.DialogCode.Accepted
    
    def fit_properties(self, min_points=6, skip_near=3, snr_threshold=3.0, 
                      scr_width=None, fit_method='linear_log', pixel_size=1.0):
        """Run PerpendicularFitter on this window's profile with adjustable parameters.
        
        Parameters can be overridden via instance attributes set by dialog.
        Plots the fit lines on the ln(I) axis if available.
        """
        # Use stored parameters if they exist
        min_points = getattr(self, '_fit_min_points', min_points)
        skip_near = getattr(self, '_fit_skip_near', skip_near)
        snr_threshold = getattr(self, '_fit_snr_threshold', snr_threshold)
        scr_width = getattr(self, '_fit_scr_width', scr_width)
        fit_method = getattr(self, '_fit_method', fit_method)
        pixel_size = getattr(self, '_fit_pixel_size', pixel_size)
        
        try:
            fitter = PerpendicularFitter(
                min_points=min_points,
                skip_near_junction=skip_near,
                snr_threshold=snr_threshold,
                scr_width_estimate=scr_width,
                fit_method=fit_method,
                pixel_size_um=pixel_size
            )
            res = fitter.fit_profile(self.dist, self.abs_i, show_debug=False)
            self.properties = res
            
            if self.ln_i_axis is not None:
                self._plot_fits(res)
            
            left = res.get('left')
            right = res.get('right')
            msg = f"Junction @ {res.get('junction_pos', 0):.6g} µm\n"
            msg += f"Estimated SCR width: {res.get('scr_width', 0):.4g} µm\n\n"
            
            if left:
                msg += f"Left side:\n"
                msg += f"  Slope: {left.get('slope'):.4g}\n"
                msg += f"  R² = {left.get('r2'):.3f}\n"
                msg += f"  Diffusion length: {left.get('inv_length'):.4g} µm\n"
            else:
                msg += "Left fit: insufficient data\n"
            
            if right:
                msg += f"\nRight side:\n"
                msg += f"  Slope: {right.get('slope'):.4g}\n"
                msg += f"  R² = {right.get('r2'):.3f}\n"
                msg += f"  Collection length: {right.get('inv_length'):.4g} µm\n"
            else:
                msg += "\nRight fit: insufficient data\n"
            
            if res.get('depletion_width') is not None:
                msg += f"\nEstimated depletion width: {res.get('depletion_width'):.4g} µm"
            
            QMessageBox.information(self, 'Fit Results', msg)
            return res
        except Exception as e:
            QMessageBox.critical(self, 'Fit Error', f'Fitting failed:\n{str(e)}')
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_fits(self, result):
        """Plot the linear fits on the ln(I) axis."""
        if self.ln_i_axis is None:
            return
        
        junction_pos = result.get('junction_pos', 0)
        left = result.get('left')
        right = result.get('right')
        
        if left is not None and left.get('x_fit') is not None:
            x_fit_rel = left['x_fit']
            y_fit = left.get('y_fit')
            x_fit_abs = junction_pos - x_fit_rel
            if y_fit is not None:
                self.ln_i_axis.plot(x_fit_abs, y_fit, '--', color='blue', lw=2.5, alpha=0.85, label='Left fit', zorder=5)
        
        if right is not None and right.get('x_fit') is not None:
            x_fit_rel = right['x_fit']
            y_fit = right.get('y_fit')
            x_fit_abs = junction_pos + x_fit_rel
            if y_fit is not None:
                self.ln_i_axis.plot(x_fit_abs, y_fit, '--', color='red', lw=2.5, alpha=0.85, label='Right fit', zorder=5)
        
        self.ln_i_axis.axvline(junction_pos, color='green', linestyle=':', lw=2, alpha=0.7, label='Junction', zorder=4)
        
        handles, labels = self.ln_i_axis.get_legend_handles_labels()
        if len(set(labels)) > 0:
            self.ln_i_axis.legend(loc='best', fontsize=9, framealpha=0.9)
        
        self.canvas.draw()
    
    def save_properties(self):
        """Save last-fit properties to CSV; if none, run fit first."""
        if not hasattr(self, 'properties') or self.properties is None:
            self.fit_properties()
        
        props = getattr(self, 'properties', None)
        if props is None:
            QMessageBox.warning(self, 'Warning', 'No properties to save. Run fit first.')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, 
            'Save Properties', 
            f'Perpendicular_{self.windowTitle().split()[1]}_props.csv', 
            'CSV (*.csv)'
        )
        if not path:
            return
        
        def _fmt_side(d):
            if d is None:
                return (np.nan, np.nan, np.nan)
            return (d.get('slope', np.nan), d.get('r2', np.nan), d.get('inv_length', np.nan))
        
        l_slope, l_r2, l_inv = _fmt_side(props.get('left'))
        r_slope, r_r2, r_inv = _fmt_side(props.get('right'))
        
        header = 'junction_pos,left_slope,left_r2,left_inv_length,right_slope,right_r2,right_inv_length,left_start,right_start,depletion_width'
        data = np.array([[
            props.get('junction_pos', np.nan),
            l_slope, l_r2, l_inv,
            r_slope, r_r2, r_inv,
            props.get('left_start', np.nan),
            props.get('right_start', np.nan),
            props.get('depletion_width', np.nan)
        ]])
        
        np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.6e')
        QMessageBox.information(self, 'Saved', f'Properties saved to:\n{path}')


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
        
        v = self.p1 - self.p0
        self.length = np.linalg.norm(v)
        self.u = v / self.length if self.length > 0 else np.array([1, 0])
        self.n = np.array([-self.u[1], self.u[0]])
        
        self.line = Line2D([], [], color=self.color, linestyle='--', linewidth=2, 
                           marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        
        self.text = self.ax.text(0, 0, str(self.idx), color=self.color, fontsize=12, fontweight='bold',
                                 ha='center', va='center', zorder=5,
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        self.ax.add_line(self.line)
        self.update_positions()

    def update_positions(self):
        self.t = np.clip(self.t, 0.0, 1.0) 
        B = self.p0 + self.t * (self.p1 - self.p0)
        
        P1 = B + self.d1 * self.n
        P2 = B - self.d2 * self.n 
        
        self.line.set_data([P1[0], B[0], P2[0]], [P1[1], B[1], P2[1]])
        offset = max(self.length * 0.03, (self.d1 + self.d2) * 0.05)
        self.text.set_position((P1[0] + self.n[0] * offset, P1[1] + self.n[1] * offset))

    def remove(self):
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
        self.active_handle = None
        self.p0 = None
        self.p1 = None

    def generate_profiles(self, p0, p1, num_profiles, default_length):
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
        for p in self.profiles: p.remove()
        self.profiles.clear()
        self.active_handle = None
        self.canvas.draw()

    def get_closest_handle(self, x_px, y_px, max_dist_px=15):
        if x_px is None or y_px is None: return None
        best_dist = float('inf')
        best_match = None

        for i, prof in enumerate(self.profiles):
            x_data, y_data = prof.line.get_data()
            for j in range(3):
                disp_pt = self.ax.transData.transform((x_data[j], y_data[j]))
                dist = np.hypot(disp_pt[0] - x_px, disp_pt[1] - y_px)
                if dist < best_dist and dist < max_dist_px:
                    best_dist = dist
                    best_match = (i, j)
        return best_match

    def on_press(self, event):
        if not self.profiles or event.inaxes != self.ax: return False
        handle = self.get_closest_handle(event.x, event.y)
        if handle:
            self.active_handle = handle
            return True
        return False

    def on_drag(self, event):
        if not self.active_handle or event.inaxes != self.ax: return False
        if event.xdata is None or event.ydata is None: return False

        prof_idx, handle_idx = self.active_handle
        prof = self.profiles[prof_idx]
        mouse_pt = np.array([event.xdata, event.ydata])
        min_len = prof.length * 0.02 

        if handle_idx == 1: 
            v = prof.p1 - prof.p0
            v_norm = np.dot(v, v)
            if v_norm > 0:
                t_new = np.dot(mouse_pt - prof.p0, v) / v_norm
                prof.t = t_new

        elif handle_idx == 0:
            B = prof.p0 + prof.t * (prof.p1 - prof.p0)
            d_new = np.dot(mouse_pt - B, prof.n)
            prof.d1 = max(d_new, min_len)

        elif handle_idx == 2:
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