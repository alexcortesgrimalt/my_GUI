from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QToolBar, QFrame, QLabel, 
                             QPushButton, QStatusBar, QButtonGroup, QSlider,
                             QComboBox, QMenu, QToolButton, QTabWidget,
                             QDoubleSpinBox, QCheckBox, QMessageBox, QSpinBox,
                             QScrollArea, QGroupBox, QInputDialog)
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg 
import numpy as np
import scipy.ndimage as ndi
import os
import glob

# Import the data manager and analyzers
from image_handler import SEMDataManager
from junction_analyzer import JunctionAnalyzer
from profile_manager import ProfileManager, ProfilePlotWindow, EBIC3DWindow
from detect_NWs import NWDetector
import matplotlib.pyplot as plt

class CorrelationGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Operations Master")
        self.resize(1400, 900)

        # --- DATA MANAGER ---
        self.data_manager = SEMDataManager()
        
        # --- SWEEP DATA ---
        self.sweep_data_manager = SEMDataManager()
        self.sweep_dx = 0.0
        self.sweep_dy = 0.0

        # --- VISUAL STATE ---
        self.img_width = 0
        self.img_height = 0

        # --- FOLDER NAVIGATION ---
        self.current_folder = ""
        self.folder_files = []
        self.current_file_index = -1
        
        # Dynamic physical variables
        self.pixel_size_phys = 1.0
        self.width_phys = 0.0
        self.height_phys = 0.0
        self.unit_label = "\u03BCm" 
        self.unit_factor = 1e6     
        
        self.mode = "view" 
        self.show_overlay = False 
        self.opacity = 0.5        

        # --- GRAPHIC OBJECTS ---
        self.layer_sem = None
        self.layer_ebic = None
        self.cbar = None 
        
        # Scale bar data
        self.scale_bar_length_phys = None
        self.scale_bar_label = None
        self.scale_bar_line = None
        self.scale_bar_text = None
        
        self.colormaps = ['plasma', 'viridis', 'inferno', 'magma', 'cividis', 'rainbow', 'jet', 'gray']
        self.current_cmap = 'jet'

        # Interactive variables
        self.pan_start = None 
        self.mmb_pan_start = None
        self.line_start_point = None
        self.current_line_artist = None 
        self.stored_lines = [] 
        self.junction_line_artist = None
        
        # NWs Data Structures
        self.nw_artists = []
        self.nw_arrows = []   
        self.nw_texts = []    
        self.detected_nws_data = [] # Stores: [((sx, sy), (ex, ey)), ...] physically

        # --- OPTIMIZACIÓN: Variables de Blitting (Super smooth drag) ---
        self.dragging_nw_idx = None
        self.dragging_nw_end = None # 'start' o 'end'
        self.blit_bg_cache = None   # Caché del fondo estático
        
        # Profile Data Memory
        self.plot_windows = []
        
        # --- FRAME NAVIGATION ---
        self.frames_list = []
        self.current_frame_idx = 0

        # --- UI SETUP ---
        self.setup_ui()
        self.profile_manager = ProfileManager(self.ax, self.canvas)

        # --- MATPLOTLIB EVENTS ---
        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # --- START: INSTRUCTION SCREEN ---
        self.show_placeholder()

    # --- COORDINATE CONVERSION (Vectorized for safety) ---
    def phys_to_px(self, px, py):
        c = px / self.pixel_size_phys
        r = (self.height_phys - py) / self.pixel_size_phys
        return c, r
    
    def px_to_phys(self, c, r):
        px = c * self.pixel_size_phys
        py = self.height_phys - (r * self.pixel_size_phys)
        return px, py

    # ==========================================================
    # --- MOUSE EVENTS & INTERACTIVITY ---
    # ==========================================================
    def on_mouse_press(self, event):
        if event.inaxes != self.ax: return
        
        # --- NUEVO: PANNING CON BOTÓN CENTRAL ---
        if event.button == 2: # 2 es el click central / rueda
            self.mmb_pan_start = (event.xdata, event.ydata)
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.mode == 'view':
            # 1. Comprobar si tocamos líneas de perfil manuales (ProfileManager)
            if self.profile_manager.on_press(event): return 

            # 2. Detectar agarre de puntas de NW (Blitting Setup)
            if event.xdata and event.ydata and self.detected_nws_data:
                xlim = self.ax.get_xlim()
                tol = (xlim[1] - xlim[0]) * 0.015 
                
                for i, ((sx, sy), (ex, ey)) in enumerate(self.detected_nws_data):
                    is_start = np.hypot(event.xdata - sx, event.ydata - sy) < tol
                    is_end = np.hypot(event.xdata - ex, event.ydata - ey) < tol
                    
                    if is_start or is_end:
                        self.dragging_nw_idx = i
                        self.dragging_nw_end = 'start' if is_start else 'end'
                        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
                        
                        selected_arrow = self.nw_arrows[i]
                        selected_text = self.nw_texts[i]
                        selected_arrow.set_visible(False)
                        selected_text.set_visible(False)
                        self.canvas.draw() 
                        
                        self.blit_bg_cache = self.canvas.copy_from_bbox(self.ax.bbox)
                        
                        selected_arrow.set_visible(True)
                        selected_text.set_visible(True)
                        return
                        
        try:
            if self.mode == 'pan':
                self.pan_start = (event.xdata, event.ydata)
                self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.mode == 'line':
                if len(self.stored_lines) >= 1:
                    for l in self.stored_lines: l.remove()
                    self.stored_lines.clear()
                    self.profile_manager.clear()
                    
                self.line_start_point = (event.xdata, event.ydata)
                self.current_line_artist = Line2D([event.xdata, event.xdata], 
                                                  [event.ydata, event.ydata], 
                                                  color='red', linewidth=2)
                self.ax.add_line(self.current_line_artist)
                self.canvas.draw()
            elif self.mode == 'manual_nw':
                if not event.xdata or not event.ydata: return
                
                self.manual_nw_points.append((event.xdata, event.ydata))
                
                # Dibujar un punto temporal rojo donde hicimos clic
                dot, = self.ax.plot(event.xdata, event.ydata, 'ro', markersize=4)
                self.temp_manual_dots.append(dot)
                self.canvas.draw_idle()

                if len(self.manual_nw_points) == 1:
                    self.status_bar.showMessage("Manual NW: Click on the END point of the Nanowire.", 10000)
                elif len(self.manual_nw_points) == 2:
                    self._create_manual_nw() # Completa la creación del NW con los 2 puntos seleccionados
        except: pass

    def on_mouse_move(self, event):
        # Actualizar coordenadas en status bar (siempre)
        try:
            if event.inaxes:
                c, r = self.phys_to_px(event.xdata, event.ydata)
                px_x, px_y = int(c), int(r)
                px_x = max(0, min(px_x, self.img_width - 1))
                px_y = max(0, min(px_y, self.img_height - 1))
                self.lbl_coords.setText(f"Coordinates (Px): X {px_x}, Y {px_y}")
            else:
                self.lbl_coords.setText("Coordinates (Px): - , -")
                return
        except: pass

        # --- NUEVO: LÓGICA DE MOVIMIENTO CON BOTÓN CENTRAL ---
        if getattr(self, 'mmb_pan_start', None):
            if not event.xdata or not event.ydata: return
            dx = event.xdata - self.mmb_pan_start[0]
            dy = event.ydata - self.mmb_pan_start[1]
            
            new_xlim = [self.ax.get_xlim()[0] - dx, self.ax.get_xlim()[1] - dx]
            new_ylim = [self.ax.get_ylim()[0] - dy, self.ax.get_ylim()[1] - dy]

            if new_xlim[0] < 0: new_xlim = [0, new_xlim[1] - new_xlim[0]]
            elif new_xlim[1] > self.width_phys: new_xlim = [self.width_phys - (new_xlim[1] - new_xlim[0]), self.width_phys]
                
            if new_ylim[0] < 0: new_ylim = [0, new_ylim[1] - new_ylim[0]]
            elif new_ylim[1] > self.height_phys: new_ylim = [self.height_phys - (new_ylim[1] - new_ylim[0]), self.height_phys]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.draw_scale_bar()
            self.canvas.draw_idle()
            return # Bloquea el resto de acciones mientras haces pan con el botón central

        if self.mode == 'view':
            if self.dragging_nw_idx is not None and self.blit_bg_cache is not None:
                if not event.xdata or not event.ydata: return
                
                i = self.dragging_nw_idx
                (sx, sy), (ex, ey) = self.detected_nws_data[i]
                
                if self.dragging_nw_end == 'start':
                    fix_x, fix_y = ex, ey
                    move_x, move_y = sx, sy
                else:
                    fix_x, fix_y = sx, sy
                    move_x, move_y = ex, ey
                    
                vx, vy = move_x - fix_x, move_y - fix_y
                L_orig = np.hypot(vx, vy)
                if L_orig == 0: return 
                ux, uy = vx / L_orig, vy / L_orig
                
                wx, wy = event.xdata - fix_x, event.ydata - fix_y
                proj_len = wx * ux + wy * uy
                
                W_phys = self.width_phys
                H_phys = self.height_phys
                min_len = self.pixel_size_phys
                
                t_candidates = []
                if abs(ux) > 1e-12: 
                    t0x = (0 - fix_x) / ux
                    twx = (W_phys - fix_x) / ux
                    if t0x > min_len: t_candidates.append(t0x)
                    if twx > min_len: t_candidates.append(twx)
                    
                if abs(uy) > 1e-12: 
                    t0y = (0 - fix_y) / uy
                    thy = (H_phys - fix_y) / uy
                    if t0y > min_len: t_candidates.append(t0y)
                    if thy > min_len: t_candidates.append(thy)
                
                if t_candidates:
                    max_allowed_len = min(t_candidates)
                else:
                    max_allowed_len = proj_len 

                proj_len = max(min_len, min(proj_len, max_allowed_len))
                    
                new_x = fix_x + proj_len * ux
                new_y = fix_y + proj_len * uy
                
                if self.dragging_nw_end == 'start':
                    sx, sy = new_x, new_y
                else:
                    ex, ey = new_x, new_y
                self.detected_nws_data[i] = ((sx, sy), (ex, ey))
                
                self.canvas.restore_region(self.blit_bg_cache)
                self.nw_arrows[i].xy = (ex, ey)          
                self.nw_arrows[i].set_position((sx, sy)) 
                self.nw_texts[i].set_position((sx, sy))  
                self.ax.draw_artist(self.nw_arrows[i])
                self.ax.draw_artist(self.nw_texts[i])
                self.canvas.blit(self.ax.bbox)
                return
            
            if self.profile_manager.on_drag(event): return

        try:
            if self.mode == 'pan' and self.pan_start:
                dx = event.xdata - self.pan_start[0]
                dy = event.ydata - self.pan_start[1]
                
                new_xlim = [self.ax.get_xlim()[0] - dx, self.ax.get_xlim()[1] - dx]
                new_ylim = [self.ax.get_ylim()[0] - dy, self.ax.get_ylim()[1] - dy]

                if new_xlim[0] < 0: new_xlim = [0, new_xlim[1] - new_xlim[0]]
                elif new_xlim[1] > self.width_phys: new_xlim = [self.width_phys - (new_xlim[1] - new_xlim[0]), self.width_phys]
                    
                if new_ylim[0] < 0: new_ylim = [0, new_ylim[1] - new_ylim[0]]
                elif new_ylim[1] > self.height_phys: new_ylim = [self.height_phys - (new_ylim[1] - new_ylim[0]), self.height_phys]

                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.draw_scale_bar()
                self.canvas.draw_idle()
                
            elif self.mode == 'line' and self.line_start_point and self.current_line_artist:
                self.current_line_artist.set_data([self.line_start_point[0], event.xdata], 
                                                  [self.line_start_point[1], event.ydata])
                self.canvas.draw_idle()
        except: pass

    def on_mouse_release(self, event):
        # --- NUEVO: LIBERAR BOTÓN CENTRAL ---
        if event.button == 2:
            self.mmb_pan_start = None
            # Restaurar el cursor correcto según la herramienta actual
            if self.mode == 'pan': self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.mode == 'line': self.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else: self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if self.mode == 'view':
            if getattr(self, 'dragging_nw_idx', None) is not None:
                self.dragging_nw_idx = None
                self.dragging_nw_end = None
                self.blit_bg_cache = None
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
                self.canvas.draw_idle() 
                return
                
            if self.profile_manager.on_release(event): return

        try:
            if self.mode == 'pan':
                self.pan_start = None
                self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.mode == 'line':
                if self.current_line_artist:
                    self.stored_lines.append(self.current_line_artist)
                    self.current_line_artist = None
                    self.line_start_point = None
        except: pass

    # ==========================================================
    # --- GUI SETUP & VISUALS (Rest of the class) ---
    # ==========================================================
    def setup_ui(self):
        # 1. Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        upload_action = QAction("Load Multi-Frame .TIF", self)
        upload_action.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        upload_action.triggered.connect(self.upload_image)
        toolbar.addAction(upload_action)

        self.action_prev_file = QAction("⏮ Prev File", self)
        self.action_prev_file.triggered.connect(lambda: self.navigate_folder(-1))
        self.action_prev_file.setEnabled(False)
        toolbar.addAction(self.action_prev_file)

        self.action_next_file = QAction("Next File ⏭", self)
        self.action_next_file.triggered.connect(lambda: self.navigate_folder(1))
        self.action_next_file.setEnabled(False)
        toolbar.addAction(self.action_next_file)
        
        toolbar.addSeparator() # Un separador visual

        correlate_action = QAction("Correlate Maps (M1 ± M2)", self)
        correlate_action.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogDetailedView))
        correlate_action.triggered.connect(self.action_correlate_maps)
        toolbar.addAction(correlate_action)

        save_button = QToolButton()
        save_button.setText("Save")
        save_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        save_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        save_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        save_menu = QMenu()
        menu_sem = QMenu("Save SEM image as ", self)
        menu_sem.addAction(".tif", lambda: self.save_data("sem", "tif"))
        menu_sem.addAction(".png", lambda: self.save_data("sem", "png"))
        save_menu.addMenu(menu_sem)
        menu_ebic = QMenu("Save Current Map as ", self)
        menu_ebic.addAction(".tif", lambda: self.save_data("ebic_img", "tif"))
        menu_ebic.addAction(".png", lambda: self.save_data("ebic_img", "png"))
        save_menu.addMenu(menu_ebic)
        menu_overlay = QMenu("Save SEM + Current Map as ", self)
        menu_overlay.addAction(".tif", lambda: self.save_data("overlay", "tif"))
        menu_overlay.addAction(".png", lambda: self.save_data("overlay", "png"))
        save_menu.addMenu(menu_overlay)
        menu_screen = QMenu("Save screen as ", self)
        menu_screen.addAction(".tif", lambda: self.save_data("screen", "tif"))
        menu_screen.addAction(".png", lambda: self.save_data("screen", "png"))
        menu_screen.addSeparator() # <--- AÑADIR
        menu_screen.addAction("Figure for a paper (.png)", self.save_paper_figure) # <--- AÑADIR
        save_menu.addMenu(menu_screen)
        save_menu.addSeparator()
        save_menu.addAction("Save SEM map (.csv)", lambda: self.save_data("sem", "csv"))
        save_menu.addAction("Save Current Map (.csv)", lambda: self.save_data("ebic", "csv"))

        save_button.setMenu(save_menu)
        toolbar.addWidget(save_button)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_coords = QLabel("Coordinates (Px): - , -")
        self.status_bar.addPermanentWidget(self.lbl_coords)

        # 2. Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # -- LEFT PANEL --
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(70) 
        self.left_panel.setStyleSheet("background-color: #e0e0e0; border-right: 1px solid #c0c0c0;")
        
        self.tools_layout = QVBoxLayout(self.left_panel)
        self.tools_layout.setContentsMargins(5, 10, 5, 10)
        self.tools_layout.setSpacing(15)

        self.btn_home = self.create_tool_button("", "Reset Zoom and Lines")
        self.btn_home.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DirHomeIcon))
        self.btn_home.clicked.connect(self.action_home_reset) 
        
        self.btn_pan = self.create_tool_button("✋", "Pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(lambda: self.set_mode("pan"))

        self.btn_line = self.create_tool_button("📏", "Draw Line")
        self.btn_line.setCheckable(True)
        self.btn_line.clicked.connect(lambda: self.set_mode("line"))
        
        self.btn_grid = self.create_tool_button("▦", "Toggle Grid")
        self.btn_grid.setCheckable(True)
        self.btn_grid.clicked.connect(self.toggle_grid)
        
        self.btn_overlay = self.create_tool_button("OL", "Toggle Overlay")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setStyleSheet("QPushButton { font-weight: bold; color: purple; }")
        self.btn_overlay.clicked.connect(self.toggle_overlay)

        self.btn_3d_map = self.create_tool_button("3D", "Show 3D EBIC Map")
        self.btn_3d_map.setStyleSheet("QPushButton { font-weight: bold; background-color: #e2e3e5; padding: 6px; margin-top: 5px; }")
        self.btn_3d_map.clicked.connect(self.action_show_3d_ebic)

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_line)

        self.tools_layout.addWidget(self.btn_home)
        self.tools_layout.addWidget(self.btn_pan)
        self.tools_layout.addWidget(self.btn_line)
        self.tools_layout.addWidget(self.btn_grid)
        self.tools_layout.addSpacing(20)
        self.tools_layout.addWidget(self.btn_overlay)
        self.tools_layout.addWidget(self.btn_3d_map)
        self.tools_layout.addStretch() 

        self.main_layout.addWidget(self.left_panel)

        # -- CENTER --
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0,0,0,0)
        
        self.fig = Figure(figsize=(8, 6), facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.center_layout.addWidget(self.canvas)
        
        self.frame_nav_layout = QHBoxLayout()
        self.frame_nav_layout.setContentsMargins(10, 5, 10, 15)
        
        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setFixedWidth(100)
        self.btn_prev_frame.clicked.connect(lambda: self.change_frame(-1))
        
        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setFixedWidth(100)
        self.btn_next_frame.clicked.connect(lambda: self.change_frame(1))
        
        self.lbl_frame_info = QLabel("Frame: - / -")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_info.setStyleSheet("font-weight: bold; font-size: 13px;")
        
        self.frame_nav_layout.addStretch()
        self.frame_nav_layout.addWidget(self.btn_prev_frame)
        self.frame_nav_layout.addWidget(self.lbl_frame_info)
        self.frame_nav_layout.addWidget(self.btn_next_frame)
        self.frame_nav_layout.addStretch()
        
        self.center_layout.addLayout(self.frame_nav_layout)
        self.main_layout.addWidget(self.center_panel)

        # -- RIGHT PANEL (Tabs) --
        self.right_panel = QFrame()
        self.right_panel.setFixedWidth(290)
        self.right_panel.setStyleSheet("background-color: #f0f0f0; border-left: 1px solid #dcdcdc;")
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(5, 5, 5, 5)

        self.tabs_right = QTabWidget()
        
        # TAB 1: VISUALIZATION
        self.tab_vis = QWidget()
        vis_layout = QVBoxLayout(self.tab_vis)
        lbl_props = QLabel("Visualization")
        lbl_props.setStyleSheet("font-weight: bold; font-size: 14px;")
        vis_layout.addWidget(lbl_props)
        vis_layout.addSpacing(15)

        # --- NUEVO: Selector de Overlay ---
        lbl_overlay_type = QLabel("Overlay Mode:")
        self.combo_overlay = QComboBox()
        self.combo_overlay.addItems(["EBIC (Current)", "Voltage Contrast"])
        self.combo_overlay.currentTextChanged.connect(self.toggle_overlay)
        vis_layout.addWidget(lbl_overlay_type)
        vis_layout.addWidget(self.combo_overlay)
        vis_layout.addSpacing(15)
        # ---------------------------------

        lbl_cmap = QLabel("Color Palette:")
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(self.colormaps)
        self.combo_cmap.setCurrentText(self.current_cmap)
        self.combo_cmap.currentTextChanged.connect(self.update_layer_props)
        vis_layout.addWidget(lbl_cmap)
        vis_layout.addWidget(self.combo_cmap)
        vis_layout.addSpacing(20)

        # Hacer la etiqueta genérica
        self.lbl_opacity = QLabel(f"Overlay Intensity: {int(self.opacity*100)}%")
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setMinimum(0)
        self.slider_opacity.setMaximum(100)
        self.slider_opacity.setValue(int(self.opacity*100))
        self.slider_opacity.valueChanged.connect(self.update_layer_props)
        vis_layout.addWidget(self.lbl_opacity)
        vis_layout.addWidget(self.slider_opacity)
        vis_layout.addStretch()
        
        # TAB 2: JUNCTION
        self.tab_junc = QWidget()
        junc_layout = QVBoxLayout(self.tab_junc)
        
        lbl_junc_title = QLabel("Junction Detection")
        lbl_junc_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        junc_layout.addWidget(lbl_junc_title)
        junc_layout.addSpacing(10)
        
        lbl_inst_1 = QLabel("1. Draw a single line near\n the junction (tool 📏).")
        lbl_inst_1.setWordWrap(True)
        junc_layout.addWidget(lbl_inst_1)
        
        junc_layout.addSpacing(10)
        lbl_inst_2 = QLabel("2. Half-width of the junction (\u03BCm):")
        self.spin_hw = QDoubleSpinBox()
        self.spin_hw.setRange(0.01, 100.0)
        self.spin_hw.setSingleStep(0.1)
        self.spin_hw.setValue(1.0)
        junc_layout.addWidget(lbl_inst_2)
        junc_layout.addWidget(self.spin_hw)

        junc_layout.addSpacing(10)
        lbl_ebic_weight = QLabel("3. EBIC Weight (for Detection):")
        self.spin_ebic_weight = QDoubleSpinBox()
        self.spin_ebic_weight.setRange(0.0, 1000.0)
        self.spin_ebic_weight.setSingleStep(1.0)
        self.spin_ebic_weight.setValue(10.0) 
        junc_layout.addWidget(lbl_ebic_weight)
        junc_layout.addWidget(self.spin_ebic_weight)
        
        junc_layout.addSpacing(15)
        lbl_plots = QLabel("3. Select outputs:")
        lbl_plots.setStyleSheet("font-weight: bold;")
        junc_layout.addWidget(lbl_plots)
        
        self.chk_a = QCheckBox("a) SEM ROI - EBIC / Current ROI")
        self.chk_b = QCheckBox("b) Junction Detection Comparison")
        self.chk_c = QCheckBox("c) Raw EBIC & Filtered EBIC")
        self.chk_d = QCheckBox("d) Canny (Filtered, Spline) - General")
        self.chk_e = QCheckBox("e) Draw Detected Junction (Over Main)")
        self.chk_e.setChecked(True) 
        self.chk_f = QCheckBox("f) Observe Junction Profile (1D Plot)")
        self.chk_f.setChecked(False)

        junc_layout.addWidget(self.chk_a)
        junc_layout.addWidget(self.chk_b)
        junc_layout.addWidget(self.chk_c)
        junc_layout.addWidget(self.chk_d)
        junc_layout.addWidget(self.chk_e)
        junc_layout.addWidget(self.chk_f) 
        
        junc_layout.addSpacing(15)
        self.btn_run_junction = QPushButton("OK")
        self.btn_run_junction.setStyleSheet("QPushButton { font-weight: bold; background-color: #d1e7dd; padding: 8px; }")
        self.btn_run_junction.clicked.connect(self.run_junction_detection)
        junc_layout.addWidget(self.btn_run_junction)
        junc_layout.addStretch()

        # TAB 3: PROFILES
        self.tab_prof = QWidget()
        prof_layout = QVBoxLayout(self.tab_prof)
        
        lbl_prof_title = QLabel("Multiple Profiles Extraction")
        lbl_prof_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        prof_layout.addWidget(lbl_prof_title)
        prof_layout.addSpacing(10)

        lbl_prof_source = QLabel("1. Select baseline source:")
        prof_layout.addWidget(lbl_prof_source)
        
        self.combo_baseline_source = QComboBox()
        self.combo_baseline_source.addItems(["Manual Line", "Detected Junction"])
        prof_layout.addWidget(self.combo_baseline_source)

        prof_layout.addSpacing(10)
        lbl_prof_count = QLabel("Number of profiles:")
        self.spin_prof_count = QSpinBox()
        self.spin_prof_count.setRange(1, 100)
        self.spin_prof_count.setValue(5)
        prof_layout.addWidget(lbl_prof_count)
        prof_layout.addWidget(self.spin_prof_count)

        lbl_prof_length = QLabel("Initial length (\u03BCm):")
        self.spin_prof_length = QDoubleSpinBox()
        self.spin_prof_length.setRange(0.01, 500.0)
        self.spin_prof_length.setSingleStep(0.5)
        self.spin_prof_length.setValue(2.0)
        prof_layout.addWidget(lbl_prof_length)
        prof_layout.addWidget(self.spin_prof_length)

        self.btn_gen_profiles = QPushButton("Generate Profiles")
        self.btn_gen_profiles.setStyleSheet("QPushButton { font-weight: bold; background-color: #fff3cd; padding: 6px; }")
        self.btn_gen_profiles.clicked.connect(self.generate_profiles_action)
        prof_layout.addWidget(self.btn_gen_profiles)

        prof_layout.addSpacing(10)
        lbl_prof_inst2 = QLabel("2. You can manually tweak endpoints.")
        lbl_prof_inst2.setStyleSheet("font-style: italic; color: #555555;")
        prof_layout.addWidget(lbl_prof_inst2)

        prof_layout.addSpacing(10)
        lbl_prof_outs = QLabel("3. Select outputs per profile:")
        lbl_prof_outs.setStyleSheet("font-weight: bold;")
        prof_layout.addWidget(lbl_prof_outs)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_widget)
        prof_layout.addWidget(self.scroll_area)
        
        self.profile_checkboxes = {}

        prof_layout.addSpacing(15)
        self.btn_extract_profiles = QPushButton("Extract Data")
        self.btn_extract_profiles.setStyleSheet("QPushButton { font-weight: bold; background-color: #d1e7dd; padding: 8px; }")
        self.btn_extract_profiles.clicked.connect(self.extract_profiles_data)
        prof_layout.addWidget(self.btn_extract_profiles)

        # TAB 4: SWEEP
        self.tab_sweep = QWidget()
        sweep_layout = QVBoxLayout(self.tab_sweep)
        
        lbl_sweep_title = QLabel("Sweep / Drift Correction")
        lbl_sweep_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        sweep_layout.addWidget(lbl_sweep_title)
        sweep_layout.addSpacing(10)
        
        lbl_sweep_inst1 = QLabel("1. Load image at different voltage:")
        sweep_layout.addWidget(lbl_sweep_inst1)
        
        self.btn_load_sweep = QPushButton("Load Sweep TIF")
        self.btn_load_sweep.clicked.connect(self.load_sweep_image)
        sweep_layout.addWidget(self.btn_load_sweep)
        
        self.lbl_sweep_status = QLabel("Status: No sweep image loaded")
        self.lbl_sweep_status.setStyleSheet("color: #555;")
        sweep_layout.addWidget(self.lbl_sweep_status)
        
        self.lbl_sweep_preview = QLabel("No image preview")
        self.lbl_sweep_preview.setFixedSize(260, 200)
        self.lbl_sweep_preview.setStyleSheet("background-color: #dcdcdc; border: 1px solid #aaa;")
        self.lbl_sweep_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sweep_layout.addWidget(self.lbl_sweep_preview)

        sweep_layout.addSpacing(15)

        lbl_sweep_inst2 = QLabel("2. Run Image Cross-Correlation:")
        sweep_layout.addWidget(lbl_sweep_inst2)
        
        self.btn_detect_sweep = QPushButton("Detect Sweep")
        self.btn_detect_sweep.setStyleSheet("QPushButton { font-weight: bold; background-color: #ffeeba; padding: 6px; }")
        self.btn_detect_sweep.clicked.connect(self.detect_sweep)
        sweep_layout.addWidget(self.btn_detect_sweep)
        sweep_layout.addSpacing(10)
        
        self.btn_check_sweep = QPushButton("Check Sweep (Visualize)")
        self.btn_check_sweep.setStyleSheet("QPushButton { font-weight: bold; background-color: #d1e7dd; padding: 6px; }")
        self.btn_check_sweep.clicked.connect(self.check_sweep)
        self.btn_check_sweep.setEnabled(False)
        sweep_layout.addWidget(self.btn_check_sweep)
        
        sweep_layout.addStretch()

        # TAB 5: NANO WIRES DETECTION
        self.tab_nws = QWidget()
        nws_layout = QVBoxLayout(self.tab_nws)
        
        lbl_nws_title = QLabel("Nanowires (NWs) Detection")
        lbl_nws_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        nws_layout.addWidget(lbl_nws_title)
        
        lbl_nws_inst1 = QLabel("1. Draw a baseline (📏).\n2. Drag cyan arrow ends to modify length.")
        lbl_nws_inst1.setWordWrap(True)
        nws_layout.addWidget(lbl_nws_inst1)
        nws_layout.addSpacing(10)
        
        lbl_nws_expected = QLabel("Expected NWs (0 = Auto-detect all):")
        self.spin_nw_expected = QSpinBox()
        self.spin_nw_expected.setRange(0, 100)
        self.spin_nw_expected.setValue(0)
        nws_layout.addWidget(lbl_nws_expected)
        nws_layout.addWidget(self.spin_nw_expected)

        lbl_nws_prom = QLabel("Peak prominence threshold (0.01 - 1.0):")
        self.spin_nw_prom = QDoubleSpinBox()
        self.spin_nw_prom.setRange(0.01, 1.0)
        self.spin_nw_prom.setSingleStep(0.05)
        self.spin_nw_prom.setValue(0.20)
        nws_layout.addWidget(lbl_nws_prom)
        nws_layout.addWidget(self.spin_nw_prom)

        lbl_nws_len = QLabel("Initial Length (\u03BCm):")
        self.spin_nw_len = QDoubleSpinBox()
        self.spin_nw_len.setRange(0.1, 100.0)
        self.spin_nw_len.setSingleStep(1.0)
        self.spin_nw_len.setValue(5.0)
        nws_layout.addWidget(lbl_nws_len)
        nws_layout.addWidget(self.spin_nw_len)
        
        lbl_nws_search = QLabel("Tracking search width (Px):")
        self.spin_nw_search = QSpinBox()
        self.spin_nw_search.setRange(5, 100)
        self.spin_nw_search.setValue(15)
        nws_layout.addWidget(lbl_nws_search)
        nws_layout.addWidget(self.spin_nw_search)

        self.chk_nw_inverse = QCheckBox("Inverse Detection (Find Minima)")
        self.chk_nw_inverse.setChecked(False)
        nws_layout.addWidget(self.chk_nw_inverse)
        nws_layout.addSpacing(10)

        self.btn_detect_nws = QPushButton("Detect NWs")
        self.btn_detect_nws.setStyleSheet("QPushButton { font-weight: bold; background-color: #ffeeba; padding: 6px; }")
        self.btn_detect_nws.clicked.connect(self.action_detect_nws)
        nws_layout.addWidget(self.btn_detect_nws)

        self.btn_manual_nw = QPushButton("Detect 1 NW Manually (2 Clicks)")
        self.btn_manual_nw.setStyleSheet("QPushButton { font-weight: bold; background-color: #cce5ff; padding: 6px; }")
        self.btn_manual_nw.clicked.connect(self.action_manual_nw)
        nws_layout.addWidget(self.btn_manual_nw)
        
        nws_layout.addSpacing(15)

        # =========================================================
        # --- NUEVO: CHECKBOXES PARA SELECCIONAR SALIDAS EN NWs ---
        # =========================================================
        lbl_nw_outs = QLabel("3. Select outputs per NW:")
        lbl_nw_outs.setStyleSheet("font-weight: bold;")
        nws_layout.addWidget(lbl_nw_outs)

        # Layout horizontal para dividir en dos columnas
        h_nw_layout = QHBoxLayout()
        
        col1_nw = QVBoxLayout()
        self.chk_nw_sem = QCheckBox("SEM norm")
        self.chk_nw_i = QCheckBox("Current (I)")
        self.chk_nw_deriv_i = QCheckBox("dI / dx") # <--- NUEVA SALIDA
        self.chk_nw_r = QCheckBox("Resistance (R)")
        col1_nw.addWidget(self.chk_nw_sem)
        col1_nw.addWidget(self.chk_nw_i)
        col1_nw.addWidget(self.chk_nw_deriv_i)
        col1_nw.addWidget(self.chk_nw_r)

        col2_nw = QVBoxLayout()
        self.chk_nw_vc = QCheckBox("Voltage (V)")
        self.chk_nw_deriv_vc = QCheckBox("dV / dx")
        self.chk_nw_abs_i = QCheckBox("abs(I)")
        self.chk_nw_deriv = QCheckBox("d ln(I)/dx")
        col2_nw.addWidget(self.chk_nw_vc)
        col2_nw.addWidget(self.chk_nw_deriv_vc)
        col2_nw.addWidget(self.chk_nw_abs_i)
        col2_nw.addWidget(self.chk_nw_deriv)

        h_nw_layout.addLayout(col1_nw)
        h_nw_layout.addLayout(col2_nw)
        nws_layout.addLayout(h_nw_layout)

        # Selecciones por defecto (igual que antes)
        self.chk_nw_sem.setChecked(False)
        self.chk_nw_i.setChecked(True)
        self.chk_nw_deriv_i.setChecked(False) 
        self.chk_nw_r.setChecked(True)
        self.chk_nw_vc.setChecked(True)
        self.chk_nw_deriv_vc.setChecked(True)
        self.chk_nw_abs_i.setChecked(False)
        self.chk_nw_deriv.setChecked(False)
        # =========================================================

        nws_layout.addSpacing(10)

        # --- CONTINÚA CON EL BOTÓN EXISTENTE ---
        self.btn_extract_nws = QPushButton("Extract Current Profiles (1D)")
        self.btn_extract_nws.setStyleSheet("QPushButton { font-weight: bold; background-color: #d1e7dd; padding: 6px; }")
        self.btn_extract_nws.clicked.connect(self.action_extract_nws_profiles)
        nws_layout.addWidget(self.btn_extract_nws)
        
        nws_layout.addStretch()
        # Add tabs
        self.tabs_right.addTab(self.tab_vis, "Vis")
        self.tabs_right.addTab(self.tab_junc, "Junction")
        self.tabs_right.addTab(self.tab_prof, "Profiles")
        self.tabs_right.addTab(self.tab_sweep, "Sweep")
        self.tabs_right.addTab(self.tab_nws, "NWs")
        
        self.right_layout.addWidget(self.tabs_right)
        self.main_layout.addWidget(self.right_panel)

    def create_tool_button(self, text, tooltip):
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(50, 50)
        return btn

    def set_mode(self, mode):
        try:
            if self.mode == mode:
                self.mode = "view"
                self.tool_group.setExclusive(False)
                self.btn_pan.setChecked(False)
                self.btn_line.setChecked(False)
                self.tool_group.setExclusive(True)
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.mode = mode
                if mode == 'pan': self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
                elif mode == 'line': self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        except Exception as e: print(e)

    def show_placeholder(self):

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        self.ax.axis('off') 
        
        instructions_text = (
            "Welcome to Map Operations Master\n\n"
            "This software correlates scanning electron microscopy (SEM) images\n"
            "with EBIC current measurements, allowing to carry out many map operations!.\n\n"
            "LOAD INSTRUCTIONS:\n"
            "Please load a multi-page .tif file containing:\n"
            "  • 1st Image: SEM topography image.\n"
            "  • 2nd Image: EBIC current map.\n"
            "  • Metadata: Embedded XML with physical dimensions and conversion parameters."
        )
        
        self.ax.text(0.5, 0.5, instructions_text, 
                     transform=self.ax.transAxes,
                     ha='center', va='center', 
                     fontsize=11, color='#333333',
                     bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', edgecolor='#cccccc'))
        self.ax.set_title("") 
        
        self.btn_prev_frame.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        self.lbl_frame_info.setText("Frame: - / -")
        
        self.canvas.draw()

    # ==========================================================
    # --- LOAD & FOLDER NAVIGATION LOGIC ---
    # ==========================================================
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load TIF", self.current_folder, "TIF Files (*.tif *.tiff)")
        if file_path:
            # 1. Actualizar la lista de archivos de la carpeta
            self.current_folder = os.path.dirname(file_path)
            search_pattern = os.path.join(self.current_folder, "*.[tT][iI][fF]*")
            # Ordenar alfabéticamente para que la navegación tenga sentido
            self.folder_files = sorted([f for f in glob.glob(search_pattern) if f.lower().endswith(('.tif', '.tiff'))])
            
            # 2. Encontrar el índice del archivo seleccionado
            try:
                self.current_file_index = self.folder_files.index(os.path.normpath(file_path))
            except ValueError:
                self.current_file_index = -1
                
            # 3. Cargar el archivo
            self._load_specific_file(file_path)

    def navigate_folder(self, direction):
        """Avanza o retrocede en la lista de archivos de la carpeta actual."""
        if not self.folder_files: return
        
        new_idx = self.current_file_index + direction
        
        # Comprobar límites para no salirnos de la lista
        if 0 <= new_idx < len(self.folder_files):
            self.current_file_index = new_idx
            next_file = self.folder_files[self.current_file_index]
            self._load_specific_file(next_file)

    def _load_specific_file(self, file_path):
        """Lógica interna para cargar un TIF y actualizar la interfaz."""
        self.reset_entire_state()
        success = self.data_manager.load_file(file_path)
        
        if success:
            # --- Protección contra imágenes RGB (3D) ---
            if self.data_manager.sem_data.ndim == 3:
                self.data_manager.sem_data = self.data_manager.sem_data[:, :, 0]
            
            if self.data_manager.current_map is not None and self.data_manager.current_map.ndim == 3:
                self.data_manager.current_map = self.data_manager.current_map[:, :, 0]
            # --------------------------------------------------

            self.img_height, self.img_width = self.data_manager.sem_data.shape
            
            # Extraer solo el nombre del archivo para los títulos de los gráficos
            self.current_filename = os.path.basename(file_path)
            
            self.initialize_plot()
            
            # Actualizar estado de los botones de navegación
            total_files = len(self.folder_files)
            self.action_prev_file.setEnabled(self.current_file_index > 0)
            self.action_next_file.setEnabled(self.current_file_index < total_files - 1)
            
            msg = f"Loaded: {self.current_filename} ({self.current_file_index + 1}/{total_files} in folder)"
            self.status_bar.showMessage(msg, 5000)
        else:
            self.show_placeholder()
            self.status_bar.showMessage("Error: Failed to load image.", 5000)

    # ==========================================================
    # --- CORRELATE MAPS LOGIC ---
    # ==========================================================
    def action_correlate_maps(self):
        # 1. Verificar que M1 y M2 están en memoria
        if self.data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "M1 is not loaded. Please load the main map first.")
            return
        if self.sweep_data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "M2 is not loaded. Go to the 'Sweep' tab, load the second map, and run 'Detect Sweep' first.")
            return

        # 2. Verificar que el EBIC/EBAC existe en ambos
        if self.data_manager.current_map is None or self.sweep_data_manager.current_map is None:
            QMessageBox.warning(self, "Error", "Both M1 and M2 must contain current maps (EBIC) to correlate.")
            return

        # 3. Advertencia si no se ha calculado el drift (Sweep = 0,0)
        if self.sweep_dx == 0.0 and self.sweep_dy == 0.0:
            reply = QMessageBox.question(self, "Warning", "Calculated drift is 0.0. Did you forget to click 'Detect Sweep' in the Sweep tab?", 
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No: return

        # 4. Seleccionar operación
        items = [
            "(M1 + M2) / 2   [Symmetric / EBAC]", 
            "(M1 - M2) / 2   [Asymmetric / Pure EBIC]"
        ]
        item, ok = QInputDialog.getItem(self, "Select Operation", "Choose correlation operation:", items, 0, False)
        
        if ok and item:
            # 5. ALINEACIÓN FÍSICA (Drift Correction) PRE-OPERACIÓN
            ebic_m1 = self.data_manager.current_map.astype(float)
            ebic_m2 = self.sweep_data_manager.current_map.astype(float)
            sem_m1 = self.data_manager.sem_data.astype(float)
            sem_m2 = self.sweep_data_manager.sem_data.astype(float)
            
            # Desplazamos M2 usando la inversa del drift detectado (interpolación nearest para no crear artefactos en bordes)
            ebic_m2_aligned = ndi.shift(ebic_m2, shift=(-self.sweep_dy, -self.sweep_dx), mode='nearest')
            sem_m2_aligned = ndi.shift(sem_m2, shift=(-self.sweep_dy, -self.sweep_dx), mode='nearest')
            
            # 6. Operación Matemática
            if "+" in item:
                new_ebic = (ebic_m1 + ebic_m2_aligned) / 2.0
                op_name = "Sum M1+M2"
            else:
                new_ebic = (ebic_m1 - ebic_m2_aligned) / 2.0
                op_name = "Diff M1-M2"
                
            # Sobrescribimos el mapa actual de M1 con el resultado
            self.data_manager.current_map = new_ebic
            # Promediamos el SEM de M1 y el M2 alineado para mejorar la relación señal-ruido del fondo
            self.data_manager.sem_data = (sem_m1 + sem_m2_aligned) / 2.0
            
            # 7. Trazabilidad: Guardamos las variables y parámetros aplicados para el título del gráfico
            drift_x_str = f"{-self.sweep_dx:.2f}"
            drift_y_str = f"{-self.sweep_dy:.2f}"
            self.current_filename = f"Correlated ({op_name}) | Shift: X={drift_x_str}px, Y={drift_y_str}px"
            
            # 8. Resetear y dibujar el nuevo estado
            self.action_home_reset() 
            self.initialize_plot()   
            self.status_bar.showMessage(f"Maps correlated perfectly. Drift corrected: X={drift_x_str}, Y={drift_y_str}.", 6000)

    # --- FRAME NAVIGATION ---
    def update_frame_ui(self):
        total = len(self.frames_list)
        if total > 0:
            self.lbl_frame_info.setText(f"Frame: {self.current_frame_idx + 1} / {total}")
            self.btn_prev_frame.setEnabled(self.current_frame_idx > 0)
            self.btn_next_frame.setEnabled(self.current_frame_idx < total - 1)
        else:
            self.lbl_frame_info.setText("Frame: - / -")
            self.btn_prev_frame.setEnabled(False)
            self.btn_next_frame.setEnabled(False)

    def change_frame(self, delta):
        if not self.frames_list: return
        
        new_idx = self.current_frame_idx + delta
        if 0 <= new_idx < len(self.frames_list):
            self.current_frame_idx = new_idx
            new_data = self.frames_list[self.current_frame_idx]
            self.layer_sem.set_data(new_data)
            
            vmin = np.nanmin(new_data)
            vmax = np.nanmax(new_data)
            if vmin == vmax: vmax += 1.0 
            self.layer_sem.set_clim(vmin, vmax)
            
            # --- CAMBIAR LA LÓGICA DEL TÍTULO ---
            file_str = f" | {self.current_filename}" if hasattr(self, 'current_filename') and self.current_filename else ""
            overlay_str = " + EBIC Overlay" if self.show_overlay else ""
            
            if self.current_frame_idx == 0:
                self.ax.set_title(f"SEM View (Frame 0){overlay_str}{file_str}")
            elif self.current_frame_idx == 1:
                self.ax.set_title(f"Raw EBIC View (Frame 1){overlay_str}{file_str}")
            
            self.update_frame_ui()
            self.canvas.draw()

    # --- AUTOMATIC SCALE BAR CALCULATION ---
    def draw_scale_bar(self):
        if self.width_phys == 0: return
        
        if self.scale_bar_line:
            try: self.scale_bar_line.remove()
            except: pass
        if self.scale_bar_text:
            try: self.scale_bar_text.remove()
            except: pass

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        vis_width = abs(xlim[1] - xlim[0])
        vis_height = abs(ylim[1] - ylim[0])
        
        target_width_phys = vis_width * 0.20
        target_width_m = target_width_phys / self.unit_factor
        
        scales = [
            (10e-9, "10 nm"), (100e-9, "100 nm"),
            (1e-6, "1 \u03BCm"), (10e-6, "10 \u03BCm"), (100e-6, "100 \u03BCm"),
            (1e-3, "1 mm")
        ]
        
        best_scale_m, best_scale_label = min(scales, key=lambda x: abs(x[0] - target_width_m))
        
        self.scale_bar_length_phys = best_scale_m * self.unit_factor
        self.scale_bar_label = best_scale_label
        
        margin_x = vis_width * 0.05
        margin_y = vis_height * 0.05
        
        x1 = max(xlim) - margin_x
        x0 = x1 - self.scale_bar_length_phys
        y0 = min(ylim) + margin_y 
        
        self.scale_bar_line = Line2D([x0, x1], [y0, y0], color='#00FF00', linewidth=1)
        self.ax.add_line(self.scale_bar_line)
        self.scale_bar_text = self.ax.text(x0 + self.scale_bar_length_phys/2, y0 + (vis_height*0.02), 
                                           self.scale_bar_label, color='#00FF00', fontsize=11, 
                                           fontweight='normal', ha='center', va='bottom')

    # --- SAVE LOGIC ---
    def save_data(self, content_type, file_format):
        if self.data_manager.sem_data is None:
            self.status_bar.showMessage("Error: No data loaded to save.", 3000)
            return

        filter_str = f"{file_format.upper()} Files (*.{file_format})"
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save as {file_format.upper()}", "", filter_str)
        
        if not file_path: return 
        if not file_path.lower().endswith(f".{file_format}"): file_path += f".{file_format}"

        try:
            if content_type == "sem" and file_format == "csv":
                np.savetxt(file_path, self.data_manager.sem_data, delimiter=",")
            elif content_type == "ebic" and file_format == "csv":
                if self.data_manager.current_map is not None:
                    np.savetxt(file_path, self.data_manager.current_map, delimiter=",")
                else: return

            elif content_type == "sem" and file_format in ["tif", "png"]:
                mpimg.imsave(file_path, self.data_manager.sem_data, cmap='gray')

            elif content_type == "ebic_img" and file_format in ["tif", "png"]:
                if self.data_manager.current_map is not None:
                    mpimg.imsave(file_path, self.data_manager.current_map, cmap=self.current_cmap)
                else: return

            elif content_type == "screen":
                self.fig.savefig(file_path, format=file_format, bbox_inches='tight')

            elif content_type == "overlay":
                temp_fig = Figure(figsize=(self.img_width/100, self.img_height/100), dpi=100)
                temp_ax = temp_fig.add_subplot(111)
                temp_ax.axis('off') 
                
                extent_physical = [0, self.width_phys, 0, self.height_phys]
                
                temp_ax.imshow(self.data_manager.sem_data, cmap='gray', aspect='equal', extent=extent_physical)
                if self.data_manager.current_map is not None:
                    temp_ax.imshow(self.data_manager.current_map, cmap=self.current_cmap, 
                                   alpha=self.opacity, aspect='equal', extent=extent_physical,
                                   vmin=np.nanmin(self.data_manager.current_map), 
                                   vmax=np.nanmax(self.data_manager.current_map))
                
                if self.scale_bar_length_phys:
                    target_m = (self.width_phys / self.unit_factor) * 0.20
                    scales = [(10e-9, "10 nm"), (100e-9, "100 nm"), (1e-6, "1 \u03BCm"), (10e-6, "10 \u03BCm"), (100e-6, "100 \u03BCm"), (1e-3, "1 mm")]
                    best_m, best_lbl = min(scales, key=lambda x: abs(x[0] - target_m))
                    sb_len_phys = best_m * self.unit_factor
                    
                    margin_x = self.width_phys * 0.05
                    margin_y = self.height_phys * 0.05
                    x1 = self.width_phys - margin_x
                    x0 = x1 - sb_len_phys
                    y0 = margin_y 
                    
                    temp_ax.add_line(Line2D([x0, x1], [y0, y0], color='#00FF00', linewidth=1))
                    temp_ax.text(x0 + sb_len_phys/2, y0 + (self.height_phys*0.02), 
                                 best_lbl, color='#00FF00', fontsize=14, 
                                 fontweight='normal', ha='center', va='bottom')
                
                temp_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                temp_fig.savefig(file_path, format=file_format, pad_inches=0)

            self.status_bar.showMessage(f"Successfully saved to: {file_path}", 5000)

        except Exception as e: print(e)

    def save_paper_figure(self):
        if self.data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "No image loaded to save.")
            return

        # 1. Preguntar por el Título
        title, ok_title = QInputDialog.getText(self, "Paper Figure", "Enter Title (Bold):")
        if not ok_title: return

        # 2. Preguntar si se quieren Ejes o solo Scalebar
        axes_opts = ["Show Axes", "Only Scalebar"]
        ax_mode, ok_ax = QInputDialog.getItem(self, "Axes Visibility", "Select axes mode:", axes_opts, 0, False)
        if not ok_ax: return

        # 3. Preguntar por el Modo de la barra de color
        modes = ["Inside", "Outside", "None"]
        cb_mode, ok_mode = QInputDialog.getItem(self, "Colorbar Mode", "Select placement mode:", modes, 0, False)
        if not ok_mode: return

        # 4. Preguntar por la Posición de la barra de color
        cb_pos = "Bottom"
        if cb_mode != "None":
            positions = ["Top", "Bottom", "Left", "Right"]
            cb_pos, ok_pos = QInputDialog.getItem(self, "Colorbar Position", "Select side:", positions, 0, False)
            if not ok_pos: return

        # 5. Pedir ruta de guardado
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Paper Figure", "", "PNG Files (*.png);;TIF Files (*.tif *.tiff)")
        if not file_path: return

        # --- INICIO DE TRANSFORMACIÓN PARA PUBLICACIÓN ---
        from matplotlib.ticker import MaxNLocator, AutoLocator

        original_title = self.ax.get_title()
        original_cbar_vis = self.cbar.ax.get_visible() if self.cbar else False
        original_cbar_v_vis = getattr(self, 'cbar_voltage', None) and self.cbar_voltage.ax.get_visible()

        if self.cbar: self.cbar.ax.set_visible(False)
        if getattr(self, 'cbar_voltage', None): self.cbar_voltage.ax.set_visible(False)

        # Configuración base de la fuente LaTeX
        paper_rc = {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }

        with plt.rc_context(paper_rc):
            # Título
            if title:
                self.ax.set_title(title, fontweight='bold', fontsize=28, pad=20)
            else:
                self.ax.set_title("")

            # Forzar el símbolo \mu matemático para que Computer Modern lo reconozca siempre
            unit_tex = r"$\mu m$" if self.unit_label == "\u03BCm" else rf"${self.unit_label}$"
            
            # Limitar la cantidad de números en los ejes para que no se solapen al ser tan grandes
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            self.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Control de Ejes
            if ax_mode == "Only Scalebar":
                self.ax.axis('off')
            else:
                self.ax.axis('on')
                self.ax.set_xlabel(rf"[{unit_tex}]", fontsize=28, fontfamily='serif', fontname='cmr10')
                self.ax.set_ylabel(rf"[{unit_tex}]", fontsize=28, fontfamily='serif', fontname='cmr10')
                self.ax.tick_params(axis='both', which='major', labelsize=24)
                
                for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                    label.set_fontfamily('serif')
                    label.set_fontname('cmr10')

            # Scalebar más gorda, con texto gigante y el símbolo LaTeX inyectado temporalmente
            original_sb_fontsize = None
            original_sb_lw = None
            original_sb_text = None
            if self.scale_bar_text:
                original_sb_text = self.scale_bar_text.get_text()
                # Reemplazamos el carácter problemático por código LaTeX puro
                new_text = original_sb_text.replace("\u03BCm", r"$\mu m$")
                self.scale_bar_text.set_text(new_text)
                
                original_sb_fontsize = self.scale_bar_text.get_fontsize()
                self.scale_bar_text.set_fontsize(26)
                self.scale_bar_text.set_fontfamily('serif')
                self.scale_bar_text.set_fontname('cmr10')
            if self.scale_bar_line:
                original_sb_lw = self.scale_bar_line.get_linewidth()
                self.scale_bar_line.set_linewidth(4.5)

            # Preparar datos de la barra
            active_im = None
            label_text = ""
            if self.show_overlay:
                mode_ui = getattr(self, 'combo_overlay', None)
                active_mode = mode_ui.currentText() if mode_ui else "EBIC (Current)"
                if active_mode == "EBIC (Current)" and self.layer_ebic:
                    active_im = self.layer_ebic
                    label_text = r"Current [$nA$]"
                elif active_mode == "Voltage Contrast" and getattr(self, 'layer_voltage', None):
                    active_im = self.layer_voltage
                    label_text = r"Voltage [$V$]"

            # Lógica de Colorbar (90% de ancho/alto)
            temp_cbar = None
            cax = None
            
            if active_im and cb_mode != "None":
                is_horiz = cb_pos in ["Top", "Bottom"]
                
                if cb_mode == "Outside":
                    # Si va fuera, Matplotlib automáticamente pone los números hacia el exterior
                    temp_cbar = self.fig.colorbar(active_im, ax=self.ax, location=cb_pos.lower(), shrink=0.9, pad=0.06)
                    temp_cbar.set_label(label_text, fontsize=28, labelpad=15, fontfamily='serif', fontname='cmr10')
                    temp_cbar.ax.tick_params(labelsize=24)
                    
                    ticks_axis = temp_cbar.ax.xaxis if is_horiz else temp_cbar.ax.yaxis
                    ticks_axis.set_major_locator(MaxNLocator(nbins=4))
                    
                    for label in ticks_axis.get_ticklabels():
                        label.set_fontfamily('serif')
                        label.set_fontname('cmr10')
                
                elif cb_mode == "Inside":
                    # Si va dentro, empujamos los números hacia el interior de la imagen
                    if cb_pos == "Top":
                        cax = inset_axes(self.ax, width="90%", height="4%", loc='upper center', borderpad=1.5)
                        temp_cbar = self.fig.colorbar(active_im, cax=cax, orientation='horizontal')
                        temp_cbar.ax.xaxis.set_ticks_position('bottom')
                        temp_cbar.ax.xaxis.set_label_position('bottom')
                    elif cb_pos == "Bottom":
                        cax = inset_axes(self.ax, width="90%", height="4%", loc='lower center', borderpad=1.5)
                        temp_cbar = self.fig.colorbar(active_im, cax=cax, orientation='horizontal')
                        temp_cbar.ax.xaxis.set_ticks_position('top')
                        temp_cbar.ax.xaxis.set_label_position('top')
                    elif cb_pos == "Left":
                        cax = inset_axes(self.ax, width="4%", height="90%", loc='center left', borderpad=1.5)
                        temp_cbar = self.fig.colorbar(active_im, cax=cax, orientation='vertical')
                        temp_cbar.ax.yaxis.set_ticks_position('right')
                        temp_cbar.ax.yaxis.set_label_position('right')
                    elif cb_pos == "Right":
                        cax = inset_axes(self.ax, width="4%", height="90%", loc='center right', borderpad=1.5)
                        temp_cbar = self.fig.colorbar(active_im, cax=cax, orientation='vertical')
                        temp_cbar.ax.yaxis.set_ticks_position('left')
                        temp_cbar.ax.yaxis.set_label_position('left')

                    temp_cbar.set_label(label_text, color='white', fontsize=28, labelpad=10, fontfamily='serif', fontname='cmr10')
                    temp_cbar.ax.tick_params(colors='white', labelsize=24)
                    
                    ticks_axis = temp_cbar.ax.xaxis if is_horiz else temp_cbar.ax.yaxis
                    ticks_axis.set_major_locator(MaxNLocator(nbins=4))

                    for label in ticks_axis.get_ticklabels():
                        label.set_color('white')
                        label.set_fontfamily('serif')
                        label.set_fontname('cmr10')
                        
                    temp_cbar.outline.set_edgecolor('white')
                    temp_cbar.outline.set_linewidth(1.5)

            # Renderizar y Guardar
            self.canvas.draw()
            self.fig.savefig(file_path, bbox_inches='tight', dpi=300)

            # --- RESTAURACIÓN ---
            self.ax.axis('on') 
            self.ax.xaxis.set_major_locator(AutoLocator())
            self.ax.yaxis.set_major_locator(AutoLocator())

            self.ax.set_title(original_title, fontweight='normal', fontsize=12) 
            self.ax.set_xlabel(f"Distance ({self.unit_label})", fontfamily='sans-serif', fontsize=10)
            self.ax.set_ylabel(f"Distance ({self.unit_label})", fontfamily='sans-serif', fontsize=10)
            self.ax.tick_params(axis='both', which='major', labelsize=10)
            
            for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                label.set_fontfamily('sans-serif')

            # Restauramos el texto original de la escala (con el símbolo unicode normal)
            if self.scale_bar_text and original_sb_text is not None:
                self.scale_bar_text.set_text(original_sb_text)
                self.scale_bar_text.set_fontsize(original_sb_fontsize)
                self.scale_bar_text.set_fontfamily('sans-serif')
            if self.scale_bar_line and original_sb_lw:
                self.scale_bar_line.set_linewidth(original_sb_lw)

            # Borramos la barra de color (que a su vez borra el cax)
            if temp_cbar: 
                try: temp_cbar.remove()
                except Exception: pass
            
            # Por si acaso Matplotlib no borró el cax, intentamos borrarlo con cuidado
            if cax: 
                try: cax.remove()
                except Exception: pass

            if self.cbar and original_cbar_vis: self.cbar.ax.set_visible(True)
            if getattr(self, 'cbar_voltage', None) and original_cbar_v_vis: self.cbar_voltage.ax.set_visible(True)

            self.canvas.draw_idle()

        self.status_bar.showMessage(f"Paper Figure saved successfully to: {file_path}", 6000)
    # --------------------------------------------------------
    def reset_entire_state(self):
        self.layer_sem = None
        self.layer_ebic = None
        self.mmb_pan_start = None
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None

        if getattr(self, 'cbar_voltage', None):
            try: self.cbar_voltage.remove()
            except: pass
            self.cbar_voltage = None

        self.stored_lines = []
        self.current_line_artist = None
        if self.junction_line_artist:
            try: self.junction_line_artist.remove()
            except: pass
            self.junction_line_artist = None
            
        if hasattr(self, 'profile_manager'):
            self.profile_manager.clear()
            
        if hasattr(self, 'scroll_layout'):
            for i in reversed(range(self.scroll_layout.count())):
                widget = self.scroll_layout.itemAt(i).widget()
                if widget: widget.deleteLater()
            self.profile_checkboxes = {}

        # Limpiar NWs
        if hasattr(self, 'nw_artists'):
            for artist in self.nw_artists:
                try: artist.remove()
                except: pass
            self.nw_artists.clear()
        if hasattr(self, 'nw_arrows'): self.nw_arrows.clear()
        if hasattr(self, 'nw_texts'): self.nw_texts.clear()
        if hasattr(self, 'detected_nws_data'): self.detected_nws_data.clear()
        
        # Limpiar variables de arrastre
        self.dragging_nw_idx = None
        self.dragging_nw_end = None
        self.blit_bg_cache = None
        
        self.scale_bar_length_phys = None
        self.scale_bar_label = None
        self.scale_bar_line = None
        self.scale_bar_text = None
        
        self.frames_list = []
        self.current_frame_idx = 0
        
        self.mode = "view"
        self.opacity = 0.5
        self.show_overlay = False
        
        self.slider_opacity.blockSignals(True)
        self.slider_opacity.setValue(50)
        self.slider_opacity.blockSignals(False)
        self.lbl_opacity.setText("EBIC Weight: 50%")
        
        self.btn_overlay.blockSignals(True)
        self.btn_overlay.setChecked(False)
        self.btn_overlay.blockSignals(False)

        self.btn_grid.blockSignals(True)
        self.btn_grid.setChecked(False)
        self.btn_grid.blockSignals(False)
        
        self.tool_group.setExclusive(False)
        self.btn_pan.setChecked(False)
        self.btn_line.setChecked(False)
        self.tool_group.setExclusive(True)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        self.sweep_data_manager = SEMDataManager()
        self.sweep_dx = 0.0
        self.sweep_dy = 0.0
        self.lbl_sweep_status.setText("Status: No sweep image loaded")
        self.btn_check_sweep.setEnabled(False)

        self.lbl_sweep_preview.clear()
        self.lbl_sweep_preview.setText("No image preview")

        self.current_filename = None
        self.setWindowTitle("Map Operations Master")

        self.show_placeholder()

    def initialize_plot(self):
        self.fig.clear()  # Destruye todo rastro de ejes y barras de color viejas
        self.ax = self.fig.add_subplot(111)

        self.profile_manager = ProfileManager(self.ax, self.canvas)
       
        self.ax.axis('on')
        
        self.frames_list = []
        if self.data_manager.sem_data is not None:
            self.frames_list.append(self.data_manager.sem_data)
        if self.data_manager.current_map is not None:
            self.frames_list.append(self.data_manager.current_map)
            
        self.current_frame_idx = 0
        self.update_frame_ui()
        
        px_size = self.data_manager.pixel_size if self.data_manager.pixel_size > 0 else 1e-6
        fov_m = px_size * self.img_width
        
        if fov_m < 1e-6:
            self.unit_factor = 1e9
            self.unit_label = "nm"
        elif fov_m < 1e-3:
            self.unit_factor = 1e6
            self.unit_label = "\u03BCm"
        else:
            self.unit_factor = 1e3
            self.unit_label = "mm"
            
        self.pixel_size_phys = px_size * self.unit_factor
        self.width_phys = self.img_width * self.pixel_size_phys
        self.height_phys = self.img_height * self.pixel_size_phys
        
        extent_physical = [0, self.width_phys, 0, self.height_phys]

        base_data = self.frames_list[0] if self.frames_list else np.zeros((self.img_height, self.img_width))
        self.layer_sem = self.ax.imshow(base_data, cmap='gray', interpolation='nearest', aspect='equal', extent=extent_physical)
        
        vmin = np.nanmin(base_data)
        vmax = np.nanmax(base_data)
        if vmin != vmax: self.layer_sem.set_clim(vmin, vmax)

        if self.data_manager.current_map is not None:
            data_ebic = self.data_manager.current_map
            vmin = np.nanmin(data_ebic)
            vmax = np.nanmax(data_ebic)

            self.layer_ebic = self.ax.imshow(data_ebic, cmap=self.current_cmap, alpha=self.opacity, 
                                             interpolation='nearest', aspect='equal', extent=extent_physical,
                                             vmin=vmin, vmax=vmax)
            self.cbar = self.fig.colorbar(self.layer_ebic, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label('Current (nA)', rotation=270, labelpad=15)
            self.layer_ebic.set_visible(False)
            self.cbar.ax.set_visible(False)

        self.layer_voltage = None
        self.cbar_voltage = None

        if hasattr(self.data_manager, 'voltage_map') and self.data_manager.voltage_map is not None:
            data_vc = self.data_manager.voltage_map
            vmin_vc = np.nanmin(data_vc)
            vmax_vc = np.nanmax(data_vc)
            
            self.layer_voltage = self.ax.imshow(data_vc, cmap=self.current_cmap, alpha=self.opacity, 
                                                 interpolation='nearest', aspect='equal', extent=extent_physical,
                                                 vmin=vmin_vc, vmax=vmax_vc)
            self.cbar_voltage = self.fig.colorbar(self.layer_voltage, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar_voltage.set_label('Voltage (V)', rotation=270, labelpad=15)
            self.layer_voltage.set_visible(False)
            self.cbar_voltage.ax.set_visible(False)
        
        self.ax.set_xlabel(f"Distance ({self.unit_label})")
        self.ax.set_ylabel(f"Distance ({self.unit_label})")
        file_str = f" | {self.current_filename}" if hasattr(self, 'current_filename') and self.current_filename else ""
        self.ax.set_title(f"SEM View (Frame 0){file_str}")
        
        self.ax.grid(self.btn_grid.isChecked())
        
        self.ax.set_xlim(0, self.width_phys)
        self.ax.set_ylim(0, self.height_phys)
        
        self.draw_scale_bar()
        self.canvas.draw()

    # --- JUNCTION DETECTION FUNCTIONALITY ---
    def run_junction_detection(self):
        if self.data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "No SEM image loaded.")
            return
            
        if len(self.stored_lines) != 1:
            QMessageBox.warning(self, "Error", "Please draw exactly ONE line on the image.")
            return

        line = self.stored_lines[0]
        xdata, ydata = line.get_data()
        x0_phys, y0_phys = xdata[0], ydata[0]
        x1_phys, y1_phys = xdata[1], ydata[1]

        c0, r0 = self.phys_to_px(x0_phys, y0_phys)
        c1, r1 = self.phys_to_px(x1_phys, y1_phys)

        v_c = c1 - c0
        v_r = r1 - r0
        L_px = np.sqrt(v_c**2 + v_r**2)
        
        if L_px == 0:
            QMessageBox.warning(self, "Error", "The drawn line is too short.")
            return

        u_c = v_c / L_px
        u_r = v_r / L_px
        n_c = -u_r
        n_r = u_c

        half_width_um = self.spin_hw.value()
        half_width_m = half_width_um * 1e-6
        pixel_size_m = self.data_manager.pixel_size if self.data_manager.pixel_size > 0 else 1e-6
        HW_px = int(np.ceil(half_width_m / pixel_size_m))
        
        W = int(np.ceil(L_px))
        H = 2 * HW_px + 1

        c_grid = c0 + u_c * np.arange(W).reshape(1, W) + n_c * (np.arange(H) - HW_px).reshape(H, 1)
        r_grid = r0 + u_r * np.arange(W).reshape(1, W) + n_r * (np.arange(H) - HW_px).reshape(H, 1)

        sem_data_float = self.data_manager.sem_data.astype(float)
        roi_sem = ndi.map_coordinates(sem_data_float, [r_grid, c_grid], order=1, mode='nearest')
        
        roi_ebic = None
        if self.data_manager.current_map is not None:
            ebic_data_float = self.data_manager.current_map.astype(float)
            roi_ebic = ndi.map_coordinates(ebic_data_float, [r_grid, c_grid], order=1, mode='nearest')

        manual_line_px = np.column_stack([c0 + u_c * np.arange(W), r0 + u_r * np.arange(W)])
        ebic_w = self.spin_ebic_weight.value()
        analyzer = JunctionAnalyzer(pixel_size_m=pixel_size_m)
        results = analyzer.detect(
            roi_sem, 
            manual_line_px, 
            roi_current=roi_ebic, 
            weight_current=ebic_w,
            plot_a=self.chk_a.isChecked(),
            plot_b=self.chk_b.isChecked(),
            plot_c=self.chk_c.isChecked()
        )

        if not results:
            QMessageBox.warning(self, "Error", "Detection failed.")
            return

        name, detected_coords_px, metrics = results[0]

        if self.chk_d.isChecked():
            analyzer.visualize_results(self.data_manager.sem_data, manual_line_px, results)
            
        if self.chk_f.isChecked() and detected_coords_px is not None:
            self.observe_junction_profile(detected_coords_px)
            
        if self.chk_e.isChecked() and detected_coords_px is not None:
            self.stored_lines[0].remove()
            self.stored_lines.clear()
            
            if self.junction_line_artist:
                try: self.junction_line_artist.remove()
                except: pass
            
            phys_x = detected_coords_px[:, 0] * self.pixel_size_phys
            phys_y = self.height_phys - (detected_coords_px[:, 1] * self.pixel_size_phys)
            
            self.junction_line_artist = Line2D(phys_x, phys_y, color='#00FF00', linewidth=2.5)
            self.ax.add_line(self.junction_line_artist)
            self.canvas.draw()
            
            msg = f"Junction detected.\nMean Dev: {metrics[0]:.2f} µm\nStd Dev: {metrics[1]:.2f} µm"
            self.status_bar.showMessage(msg, 10000)

    def observe_junction_profile(self, coords_px):
        if coords_px is None or len(coords_px) == 0:
            return

        active_windows = []
        for w in self.plot_windows:
            try:
                if w.isVisible():
                    active_windows.append(w)
            except RuntimeError:
                pass 
        self.plot_windows = active_windows

        xs = coords_px[:, 0]
        ys = coords_px[:, 1]

        sem_data = self.data_manager.sem_data.astype(float)
        cur_map = self.data_manager.current_map

        sem_vals = ndi.map_coordinates(sem_data, [ys, xs], order=1, mode='nearest')
        
        cur_vals = None
        if cur_map is not None:
            cur_map_float = cur_map.astype(float)
            cur_vals = ndi.map_coordinates(cur_map_float, [ys, xs], order=1, mode='nearest')

        diffs = np.sqrt(np.sum(np.diff(coords_px, axis=0) ** 2, axis=1))
        dists_px = np.concatenate(([0.0], np.cumsum(diffs)))
        dists_phys = dists_px * self.pixel_size_phys

        sem_norm = (sem_vals - np.min(sem_vals)) / (np.ptp(sem_vals) + 1e-12)

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(dists_phys, sem_norm, color='tab:blue', linewidth=2, label='SEM (norm)')
        ax1.set_xlabel(f'Distance along junction ({self.unit_label})')
        ax1.set_ylabel('SEM (norm)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        if cur_vals is not None:
            ax2 = ax1.twinx()
            ax2.plot(dists_phys, cur_vals, color='tab:red', linewidth=1.5, label='Current (nA)')
            ax2.set_ylabel('Current (nA)', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

        ax1.set_title("Profile Along Detected Junction")
        ax1.grid(True)
        fig.tight_layout()
        fig.show()

        self.plot_windows.append(fig.canvas.manager.window)

    # --- PROFILES FUNCTIONALITY ---
    def generate_profiles_action(self):
        if self.data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "No SEM image loaded.")
            return
            
        source = self.combo_baseline_source.currentText()
        
        if source == "Manual Line":
            if len(self.stored_lines) != 1:
                QMessageBox.warning(self, "Error", "Please draw exactly ONE manual baseline on the image.")
                return
            line = self.stored_lines[0]
            xdata, ydata = line.get_data()
            p0 = (xdata[0], ydata[0])
            p1 = (xdata[-1], ydata[-1])
            
        else: 
            if self.junction_line_artist is None:
                QMessageBox.warning(self, "Error", "No Detected Junction found. Please run the Junction Detection first.")
                return
            xdata, ydata = self.junction_line_artist.get_data()
            p0 = (xdata[0], ydata[0])
            p1 = (xdata[-1], ydata[-1])
        
        num_profiles = self.spin_prof_count.value()
        length_m = self.spin_prof_length.value() * 1e-6
        plot_length = length_m * self.unit_factor 

        self.profile_manager.generate_profiles(
            p0=p0,
            p1=p1,
            num_profiles=num_profiles,
            default_length=plot_length / 2.0 
        )
        self.set_mode("view") 

        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget: 
                widget.setParent(None)
                widget.deleteLater()
        self.profile_checkboxes.clear()

        for i in range(num_profiles):
            gb = QGroupBox(f"Perpendicular {i+1}")
            gb.setCheckable(True)
            gb.setChecked(True)
            
            vbox = QVBoxLayout(gb)
            cb_sem = QCheckBox("a) SEM norm")
            cb_abs = QCheckBox("b) abs(I)")
            cb_ln = QCheckBox("c) ln abs(I)")
            cb_deriv = QCheckBox("d) d ln(abs(I)) / dx")
            cb_vc = QCheckBox("e) Voltage Contrast") # <--- AÑADIDO
            
            cb_sem = QCheckBox("a) SEM norm")
            cb_abs = QCheckBox("b) abs(I)")
            cb_ln = QCheckBox("c) ln abs(I)")
            cb_deriv = QCheckBox("d) d ln(abs(I)) / dx")
            cb_vc = QCheckBox("e) Voltage Contrast")
            cb_deriv_vc = QCheckBox("f) dV/dx (Electric Field)") # <--- AÑADIDO
            cb_r = QCheckBox("g) Resistance (V/I)")              # <--- AÑADIDO (Ya que lo tienes en la clase Profile)
            
            cb_sem.setChecked(True)
            cb_abs.setChecked(True)
            cb_ln.setChecked(True)
            cb_deriv.setChecked(True)
            cb_vc.setChecked(False)
            cb_deriv_vc.setChecked(False) # <--- AÑADIDO
            cb_r.setChecked(False)        # <--- AÑADIDO
            
            vbox.addWidget(cb_sem)
            vbox.addWidget(cb_abs)
            vbox.addWidget(cb_ln)
            vbox.addWidget(cb_deriv)
            vbox.addWidget(cb_vc)
            vbox.addWidget(cb_deriv_vc)   # <--- AÑADIDO
            vbox.addWidget(cb_r)          # <--- AÑADIDO
            
            self.scroll_layout.addWidget(gb)
            
            self.profile_checkboxes[i+1] = {
                'group': gb,
                'sem': cb_sem,
                'abs_i': cb_abs,
                'ln_i': cb_ln,
                'deriv': cb_deriv,
                'vc': cb_vc,
                'deriv_vc': cb_deriv_vc,  # <--- AÑADIDO
                'r': cb_r                 # <--- AÑADIDO
            }

            self.canvas.draw_idle()
            

    def extract_profiles_data(self):
        if not self.profile_manager.profiles:
            QMessageBox.warning(self, "Error", "No profiles generated yet.")
            return

        # Limpieza segura de ventanas fantasma
        active_windows = []
        for w in self.plot_windows:
            try:
                if w.isVisible():
                    active_windows.append(w)
            except RuntimeError:
                pass 
        self.plot_windows = active_windows

        for prof in self.profile_manager.profiles:
            ui_elements = self.profile_checkboxes.get(prof.idx)
            
            if not ui_elements or not ui_elements['group'].isChecked():
                continue
                
            selected_keys = []
            if ui_elements['sem'].isChecked(): selected_keys.append('sem')
            if ui_elements['abs_i'].isChecked(): selected_keys.append('abs_i')
            if ui_elements['ln_i'].isChecked(): selected_keys.append('ln_i')
            if ui_elements['deriv'].isChecked(): selected_keys.append('deriv')
            if ui_elements['vc'].isChecked(): selected_keys.append('vc')
            if ui_elements.get('deriv_vc') and ui_elements['deriv_vc'].isChecked(): selected_keys.append('deriv_vc') # <--- AÑADIDO
            if ui_elements.get('r') and ui_elements['r'].isChecked(): selected_keys.append('r') # <--- AÑADIDO
            
            if not selected_keys: 
                continue
                
            x_data, y_data = prof.line.get_data()
            P1x, P1y = x_data[0], y_data[0]
            P2x, P2y = x_data[2], y_data[2]
            
            c1, r1 = self.phys_to_px(P1x, P1y)
            c2, r2 = self.phys_to_px(P2x, P2y)
            
            N = int(np.ceil(np.hypot(c2 - c1, r2 - r1)))
            if N < 2: N = 2
            
            c_vals = np.linspace(c1, c2, N)
            r_vals = np.linspace(r1, r2, N)
            
            sem_data = self.data_manager.sem_data.astype(float)
            if self.data_manager.current_map is not None:
                ebic_data = self.data_manager.current_map.astype(float)
            else:
                ebic_data = np.zeros_like(sem_data)

            if getattr(self.data_manager, 'voltage_map', None) is not None:
                vc_data = self.data_manager.voltage_map.astype(float)
            else:
                vc_data = np.zeros_like(sem_data)
                
            sem_prof = ndi.map_coordinates(sem_data, [r_vals, c_vals], order=1, mode='nearest')
            ebic_prof = ndi.map_coordinates(ebic_data, [r_vals, c_vals], order=1, mode='nearest')
            vc_prof = ndi.map_coordinates(vc_data, [r_vals, c_vals], order=1, mode='nearest')
            
            dist_um = np.linspace(0, np.hypot(P2x - P1x, P2y - P1y), N)
            
            win = ProfilePlotWindow(prof.idx, dist_um, sem_prof, ebic_prof, vc_prof, selected_keys, self.unit_label)
            win.show()
            self.plot_windows.append(win)

    # --- ACTIONS ---
    def action_home_reset(self):
        if self.width_phys == 0: return 

        self.ax.set_xlim(0, self.width_phys)
        self.ax.set_ylim(0, self.height_phys)

        for line in self.stored_lines: line.remove()
        self.stored_lines.clear()
        
        if self.junction_line_artist:
            try: self.junction_line_artist.remove()
            except: pass
            self.junction_line_artist = None
            
        if hasattr(self, 'nw_artists'):
            for artist in self.nw_artists:
                try: artist.remove()
                except: pass
            self.nw_artists.clear()
        if hasattr(self, 'nw_arrows'): self.nw_arrows.clear()
        if hasattr(self, 'nw_texts'): self.nw_texts.clear()
        if hasattr(self, 'detected_nws_data'): self.detected_nws_data.clear()

        self.profile_manager.clear() 
        for i in reversed(range(self.scroll_layout.count())):
            w = self.scroll_layout.itemAt(i).widget()
            if w: w.deleteLater()
        self.profile_checkboxes.clear()
            
        self.current_line_artist = None
        self.line_start_point = None

        self.set_mode("view")
        self.draw_scale_bar()
        self.canvas.draw_idle()

    def toggle_grid(self):
        if self.width_phys > 0:
            self.ax.grid(self.btn_grid.isChecked())
            self.canvas.draw_idle()

    def toggle_overlay(self, _=None):
        self.show_overlay = self.btn_overlay.isChecked()
        mode = getattr(self, 'combo_overlay', None)
        active_mode = mode.currentText() if mode else "EBIC (Current)"
        
        # 1. Apagar ambas capas gráficamente por defecto
        if self.layer_ebic:
            self.layer_ebic.set_visible(False)
            if self.cbar: self.cbar.ax.set_visible(False)
        if getattr(self, 'layer_voltage', None):
            self.layer_voltage.set_visible(False)
            if self.cbar_voltage: self.cbar_voltage.ax.set_visible(False)
            
        # 2. Encender la seleccionada si el botón global está activado
        overlay_str = ""
        if self.show_overlay:
            op_str = f"Op: {int(self.opacity*100)}%"
            
            if active_mode == "EBIC (Current)" and self.layer_ebic:
                self.layer_ebic.set_visible(True)
                if self.cbar: self.cbar.ax.set_visible(True)
                overlay_str = f" + EBIC Overlay ({op_str})"
                
            elif active_mode == "Voltage Contrast" and getattr(self, 'layer_voltage', None):
                self.layer_voltage.set_visible(True)
                if self.cbar_voltage: self.cbar_voltage.ax.set_visible(True)
                overlay_str = f" + Voltage Contrast Overlay ({op_str})"

        # 3. Presentación de parámetros en juego
        file_str = f" | {self.current_filename}" if hasattr(self, 'current_filename') and self.current_filename else ""
        base_title = "SEM View (Frame 0)" if self.current_frame_idx == 0 else "Raw EBIC View (Frame 1)"
        self.ax.set_title(f"{base_title}{overlay_str}{file_str}")
        
        self.canvas.draw_idle()

    def update_layer_props(self, _=None):
        val = self.slider_opacity.value()
        self.opacity = val / 100.0
        self.lbl_opacity.setText(f"Overlay Intensity: {val}%")
        
        cmap_name = self.combo_cmap.currentText()
        
        if self.layer_ebic:
            self.layer_ebic.set_alpha(self.opacity)
            self.layer_ebic.set_cmap(cmap_name)
        if getattr(self, 'layer_voltage', None):
            self.layer_voltage.set_alpha(self.opacity)
            self.layer_voltage.set_cmap(cmap_name)
            
        # Llamar al toggle actualiza el título forzosamente con el nuevo valor %
        self.toggle_overlay()

    def action_show_3d_ebic(self):
        """Genera y muestra un mapa de superficie 3D del EBIC actual."""
        if self.data_manager.current_map is None:
            QMessageBox.warning(self, "Error", "An EBIC map is required to generate the 3D surface.")
            return

        file_name = self.current_filename if hasattr(self, 'current_filename') else "Unknown file"
        px_val = f"{self.data_manager.pixel_size * self.unit_factor:.2f}" if self.data_manager.pixel_size else "N/A"
        
        params = (f"File: {file_name} | Palette: {self.current_cmap}\n"
                  f"Dimensions: {self.width_phys:.1f} x {self.height_phys:.1f} {self.unit_label} "
                  f"| Pixel size: {px_val} {self.unit_label}/px")

        active_windows = []
        for w in getattr(self, 'plot_windows', []):
            try:
                if w.isVisible():
                    active_windows.append(w)
            except RuntimeError:
                pass
                
        self.plot_windows = active_windows

        # --- NUEVO: Extraer los límites de color del mapa 2D actual ---
        current_vmin, current_vmax = self.layer_ebic.get_clim()

        # Crear y mostrar la ventana 3D pasando los límites
        win3d = EBIC3DWindow(
            ebic_data=self.data_manager.current_map,
            cmap_name=self.current_cmap,
            width_phys=self.width_phys,
            height_phys=self.height_phys,
            unit_label=self.unit_label,
            title_params=params,
            vmin=current_vmin,  # <--- PASAMOS EL VMIN
            vmax=current_vmax   # <--- PASAMOS EL VMAX
        )
        win3d.show()
        self.plot_windows.append(win3d)

    # ==========================================================
    # --- NANOWIRES DETECTION LOGIC ---
    # ==========================================================

    def set_mode(self, mode):
        try:
            # Limpieza de puntos temporales si abortas la creación manual
            if hasattr(self, 'temp_manual_dots'):
                for dot in self.temp_manual_dots:
                    try: dot.remove()
                    except: pass
                self.temp_manual_dots.clear()
            self.manual_nw_points = []

            if self.mode == mode:
                self.mode = "view"
                self.tool_group.setExclusive(False)
                self.btn_pan.setChecked(False)
                self.btn_line.setChecked(False)
                self.tool_group.setExclusive(True)
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.mode = mode
                if mode == 'pan': self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
                elif mode == 'line': self.canvas.setCursor(Qt.CursorShape.CrossCursor)
                # --- NUEVO MODO ---
                elif mode == 'manual_nw': self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        except Exception as e: print(e)

    def action_detect_nws(self):
        if self.data_manager.current_map is None:
            QMessageBox.warning(self, "Error", "An EBIC map is required for NW detection.")
            return
            
        if len(self.stored_lines) != 1:
            QMessageBox.warning(self, "Error", "Please draw exactly ONE baseline across the NWs.")
            return

        for line in self.nw_artists:
            try: line.remove()
            except: pass

        self.nw_artists.clear()
        self.nw_arrows.clear()
        self.nw_texts.clear()
        self.detected_nws_data.clear()

        line = self.stored_lines[0]
        xdata, ydata = line.get_data()
        c0, r0 = self.phys_to_px(xdata[0], ydata[0])
        c1, r1 = self.phys_to_px(xdata[1], ydata[1])

        pixel_size_m = self.data_manager.pixel_size if self.data_manager.pixel_size > 0 else 1e-6
        
        length_px = (self.spin_nw_len.value() * 1e-6) / pixel_size_m
        prominence = self.spin_nw_prom.value()
        expected_nws = self.spin_nw_expected.value()
        search_width = self.spin_nw_search.value()

        detector = NWDetector(pixel_size_m)
        ebic_map = self.data_manager.current_map.astype(float)

        is_inverse = self.chk_nw_inverse.isChecked()
        detect_map = -ebic_map if is_inverse else ebic_map
        
        # Ejecutar detección original
        nw_lines_px, tracked_points_px, run_parameters = detector.detect_and_track(
            detect_map, 
            (c0, r0), 
            (c1, r1), 
            length_px, 
            rel_prominence=prominence,
            search_width_px=search_width, 
            step_px=2,
            expected_nw_count=expected_nws
        )

        run_parameters['inverse_detection'] = is_inverse
        self.last_nw_run_parameters = run_parameters

        if not nw_lines_px:
            QMessageBox.information(self, "No NWs", "No Nanowires detected. Try lowering the peak prominence threshold.")
            return

        for i, ((sc, sr), (ec, er)) in enumerate(nw_lines_px):
            sx, sy = self.px_to_phys(sc, sr)
            ex, ey = self.px_to_phys(ec, er)
            
            pts_c, pts_r = zip(*tracked_points_px[i]) 
            pts_x, pts_y = self.px_to_phys(np.array(pts_c), np.array(pts_r))
            
            # 1. Puntos rastreados (caminata)
            track_dots = Line2D(pts_x, pts_y, color='red', marker='.', markersize=3, linewidth=0, alpha=0.5)
            self.ax.add_line(track_dots)
            self.nw_artists.append(track_dots)

            # 2. Dibujar flecha indicando la dirección del perfil
            arrow = self.ax.annotate(
                '', 
                xy=(ex, ey),       # Punta de la flecha
                xytext=(sx, sy),   # Base de la flecha
                # --- MEJORA: Evitar que desaparezca al hacer zoom ---
                annotation_clip=False, 
                arrowprops=dict(
                    arrowstyle="->", 
                    color="cyan", 
                    lw=2, 
                    mutation_scale=15
                )
            )
            self.nw_artists.append(arrow)
            self.nw_arrows.append(arrow) # Guardamos referencia para moverla
            
            # 3. Etiqueta numérica en el origen de la flecha
            txt = self.ax.text(sx, sy, f" NW{i+1}", color='white', 
                               fontsize=10, fontweight='bold', ha='right', va='bottom',
                               # --- MEJORA: Evitar recorte agresivo ---
                               clip_on=False) 
            self.nw_artists.append(txt)
            self.nw_texts.append(txt) # Guardamos referencia
            
            self.detected_nws_data.append(((sx, sy), (ex, ey)))

        self.canvas.draw_idle()
        
        if expected_nws > 0 and len(nw_lines_px) < expected_nws:
            self.status_bar.showMessage(f"Warning: Only found {len(nw_lines_px)} NWs, but {expected_nws} were expected.", 7000)
        else:
            self.status_bar.showMessage(f"Detected {len(nw_lines_px)} Nanowires. Drag cyan arrow ends to modify length.", 5000)

    def action_manual_nw(self):
        """Activa el modo para dibujar un NW haciendo 2 clics."""
        if self.data_manager.current_map is None:
            QMessageBox.warning(self, "Error", "An EBIC map is required to extract data.")
            return
            
        self.set_mode("manual_nw")
        self.manual_nw_points = []
        self.temp_manual_dots = []
        self.status_bar.showMessage("Manual NW: Click on the START point of the Nanowire.", 10000)

    def _create_manual_nw(self):
        """Dibuja el NW y lo añade a la base de datos tras el segundo clic."""
        # 1. Limpiar los puntos temporales rojos
        for dot in self.temp_manual_dots:
            try: dot.remove()
            except: pass
        self.temp_manual_dots.clear()

        # 2. Extraer coordenadas físicas
        sx, sy = self.manual_nw_points[0]
        ex, ey = self.manual_nw_points[1]
        
        # 3. Registrar el índice para nombrar el NW (NW1, NW2...)
        i = len(self.detected_nws_data)

        # 4. Dibujar la flecha Cyan
        arrow = self.ax.annotate(
            '', 
            xy=(ex, ey),       # Punta de la flecha (End)
            xytext=(sx, sy),   # Base de la flecha (Start)
            annotation_clip=False, 
            arrowprops=dict(arrowstyle="->", color="cyan", lw=2, mutation_scale=15)
        )
        self.nw_artists.append(arrow)
        self.nw_arrows.append(arrow)
        
        # 5. Añadir la etiqueta de texto
        txt = self.ax.text(sx, sy, f" NW{i+1}", color='white', 
                           fontsize=10, fontweight='bold', ha='right', va='bottom',
                           clip_on=False) 
        self.nw_artists.append(txt)
        self.nw_texts.append(txt)
        
        # 6. Guardar en la estructura de datos que usa 'Extract Current Profiles'
        self.detected_nws_data.append(((sx, sy), (ex, ey)))

        # 7. Restaurar estado visual
        self.set_mode("view")
        self.canvas.draw_idle()
        self.status_bar.showMessage(f"Manual NW {i+1} added. You can extract data or drag its ends to adjust.", 6000)

    def action_extract_nws_profiles(self):
        if not self.detected_nws_data:
            QMessageBox.warning(self, "Error", "No NWs detected yet. Click 'Detect NWs' first.")
            return
        
        selected_keys = []
        if self.chk_nw_sem.isChecked(): selected_keys.append('sem')
        if self.chk_nw_i.isChecked(): selected_keys.append('i')
        if self.chk_nw_abs_i.isChecked(): selected_keys.append('abs_i')
        if self.chk_nw_deriv.isChecked(): selected_keys.append('deriv')
        if self.chk_nw_deriv_i.isChecked(): selected_keys.append('deriv_i') # La nueva clave
        if self.chk_nw_vc.isChecked(): selected_keys.append('vc')
        if self.chk_nw_deriv_vc.isChecked(): selected_keys.append('deriv_vc')
        if self.chk_nw_r.isChecked(): selected_keys.append('r')

        if not selected_keys:
            QMessageBox.warning(self, "Warning", "Please select at least one output to plot.")
            return

        # Limpieza segura de ventanas fantasma
        active_windows = []
        for w in self.plot_windows:
            try:
                if w.isVisible():
                    active_windows.append(w)
            except RuntimeError:
                pass 
        self.plot_windows = active_windows

        sem_data = self.data_manager.sem_data.astype(float)
        ebic_data = self.data_manager.current_map.astype(float) if self.data_manager.current_map is not None else np.zeros_like(sem_data)
        vc_data = self.data_manager.voltage_map.astype(float) if getattr(self.data_manager, 'voltage_map', None) is not None else np.zeros_like(sem_data)
        


        # Preparamos la cadena de parámetros para presentar todas las variables en juego
        # Preparamos la cadena de parámetros para presentar todas las variables en juego
        param_str = ""
        if hasattr(self, 'last_nw_run_parameters'):
            p = self.last_nw_run_parameters
            inv_str = " | Mode: Inverse (Minima)" if p.get('inverse_detection', False) else " | Mode: Normal (Maxima)"
            param_str = f" | Prom: {p['rel_prominence']}, Search: {p['search_width_px']}px{inv_str}"

        for idx, ((sx, sy), (ex, ey)) in enumerate(self.detected_nws_data):
            c1, r1 = self.phys_to_px(sx, sy)
            c2, r2 = self.phys_to_px(ex, ey)
            
            N = int(np.ceil(np.hypot(c2 - c1, r2 - r1)))
            if N < 2: N = 2
            
            c_vals = np.linspace(c1, c2, N)
            r_vals = np.linspace(r1, r2, N)
            
            sem_prof = ndi.map_coordinates(sem_data, [r_vals, c_vals], order=1, mode='nearest')
            ebic_prof = ndi.map_coordinates(ebic_data, [r_vals, c_vals], order=1, mode='nearest')
            vc_prof = ndi.map_coordinates(vc_data, [r_vals, c_vals], order=1, mode='nearest') 
            
            dist_um = np.linspace(0, np.hypot(ex - sx, ey - sy), N)
            
            win_title = f"NW_{idx+1}{param_str}"
            # Añadimos 'deriv_vc' a la lista para mostrar el gradiente de potencial
            win = ProfilePlotWindow(win_title, dist_um, sem_prof, ebic_prof, vc_prof, selected_keys, self.unit_label)
            win.show()
            self.plot_windows.append(win)

    def zoom_fun(self, event):
        if self.width_phys == 0: return
        try:
            base_scale = 1.2
            if event.inaxes != self.ax: return
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            xdata, ydata = event.xdata, event.ydata
            
            scale_factor = 1/base_scale if event.button == 'up' else base_scale
            
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            
            new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
            new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]

            # Bounding Box zoom safety
            if new_xlim[0] < 0 or new_xlim[1] > self.width_phys: new_xlim = [0, self.width_phys]
            if new_ylim[0] < 0 or new_ylim[1] > self.height_phys: new_ylim = [0, self.height_phys]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.draw_scale_bar()
            self.canvas.draw_idle()
        except: pass

    # ==========================================================
    # --- SWEEP LOGIC (Cross-Correlation) ---
    # ==========================================================
    def load_sweep_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Sweep TIF", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            success = self.sweep_data_manager.load_file(file_path)
            if success:
                # --- NUEVO: Protección contra imágenes RGB (3D) ---
                if self.sweep_data_manager.sem_data is not None and self.sweep_data_manager.sem_data.ndim == 3:
                    self.sweep_data_manager.sem_data = self.sweep_data_manager.sem_data[:, :, 0]
                # --------------------------------------------------

                self.lbl_sweep_status.setText(f"Status: Loaded -> {file_path.split('/')[-1]}")
                self.btn_check_sweep.setEnabled(False)
                
                # --- PREVISUALIZACIÓN DE IMAGEN ---
                if self.sweep_data_manager.sem_data is not None:
                    data = self.sweep_data_manager.sem_data
                    vmin = np.nanmin(data)
                    vmax = np.nanmax(data)
                    if vmax > vmin:
                        norm_data = (255 * (data - vmin) / (vmax - vmin)).astype(np.uint8)
                    else:
                        norm_data = np.zeros_like(data, dtype=np.uint8)
                    
                    height, width = norm_data.shape
                    bytes_per_line = width
                    q_img = QImage(norm_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    self.lbl_sweep_preview.setPixmap(pixmap.scaled(
                        self.lbl_sweep_preview.size(), 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    ))
            else:
                QMessageBox.warning(self, "Error", "Failed to load sweep image.")

    def _estimate_translation(self, ref, img):
        """Estimate (dx, dy) pixel shift using FFT cross-correlation"""
        def make_composite(sem_img):
            a = np.asarray(sem_img, dtype=float)
            return (a - np.mean(a)) / (np.std(a) + 1e-12)

        A = make_composite(ref)
        B = make_composite(img)

        if A.shape != B.shape:
            minr = min(A.shape[0], B.shape[0])
            minc = min(A.shape[1], B.shape[1])
            A = A[:minr, :minc]
            B = B[:minr, :minc]

        try:
            import cv2
            A32 = np.float32(A)
            B32 = np.float32(B)
            shift, _ = cv2.phaseCorrelate(A32, B32)
            return float(shift[0]), float(shift[1])
        except Exception:
            # Fallback: FFT cross-correlation
            fa = np.fft.fft2(A - np.mean(A))
            fb = np.fft.fft2(B - np.mean(B))
            cross = np.fft.ifft2(fa * np.conj(fb))
            cross_abs = np.abs(cross)
            shift_y, shift_x = np.unravel_index(np.argmax(cross_abs), cross_abs.shape)
            
            if shift_x > cross.shape[1] // 2: shift_x -= cross.shape[1]
            if shift_y > cross.shape[0] // 2: shift_y -= cross.shape[0]
            return float(shift_x), float(shift_y)

    def detect_sweep(self):
        if self.data_manager.sem_data is None or self.sweep_data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "Load both Base and Sweep images first.")
            return
        self.status_bar.showMessage("Computing global pixel shift using FFT... Please wait.", 5000)
        
        try:
            dx, dy = self._estimate_translation(
                self.data_manager.sem_data, 
                self.sweep_data_manager.sem_data
            )
            self.sweep_dx = dx
            self.sweep_dy = dy
            self.btn_check_sweep.setEnabled(True)
            self.status_bar.showMessage(f"Sweep Detection Complete! Shift: dx={dx:.2f}, dy={dy:.2f}", 5000)
            QMessageBox.information(self, "Success", f"Shift found: X={dx:.2f}px, Y={dy:.2f}px.\nYou can now check the alignment.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute shift:\n{e}")

    def check_sweep(self):
        active_windows = []
        for w in self.plot_windows:
            try:
                if w.isVisible(): active_windows.append(w)
            except RuntimeError: pass 
        self.plot_windows = active_windows

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        base_img = self.data_manager.sem_data
        sweep_img = self.sweep_data_manager.sem_data
        
        axes[0].imshow(base_img, cmap='gray')
        axes[0].set_title("1. Base Image (Reference)")
        
        axes[1].imshow(sweep_img, cmap='gray')
        axes[1].set_title("2. Sweep Image (Drifted)")
        
        shifted_sweep = ndi.shift(sweep_img, shift=(-self.sweep_dy, -self.sweep_dx), mode='nearest')
        axes[2].imshow(shifted_sweep, cmap='gray')
        axes[2].set_title(f"3. Sweep Corrected (Shift: {-self.sweep_dx:.1f}, {-self.sweep_dy:.1f})")

        fig.tight_layout()
        fig.show()
        self.plot_windows.append(fig.canvas.manager.window)