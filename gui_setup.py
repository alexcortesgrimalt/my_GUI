from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QToolBar, QFrame, QLabel, 
                             QPushButton, QStatusBar, QButtonGroup, QSlider,
                             QComboBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

# Importamos el data manager
from image_handler import SEMDataManager

class CorrelationGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Correlación SEM-EBIC - Overlay Master")
        self.resize(1400, 900)

        # --- DATA MANAGER ---
        self.data_manager = SEMDataManager()

        # --- ESTADO VISUAL ---
        self.img_width = 0
        self.img_height = 0
        
        self.mode = "view" 
        self.show_overlay = False 
        self.opacity = 0.5        

        # --- OBJETOS GRÁFICOS ---
        self.layer_sem = None
        self.layer_ebic = None
        self.cbar = None 
        
        self.colormaps = ['plasma', 'viridis', 'inferno', 'magma', 'cividis', 'rainbow', 'jet', 'gray']
        self.current_cmap = 'plasma'

        # Variables interactivas
        self.pan_start = None 
        self.line_start_point = None
        self.current_line_artist = None 
        self.stored_lines = [] 

        # --- UI SETUP ---
        self.setup_ui()
        
        # --- MATPLOTLIB EVENTOS ---
        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # --- INICIO: PANTALLA EN BLANCO ---
        self.show_placeholder()

    def setup_ui(self):
        # 1. Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        upload_action = QAction("Cargar .TIF Multi-Frame", self)
        upload_action.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        upload_action.triggered.connect(self.upload_image)
        toolbar.addAction(upload_action)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_coords = QLabel("Coordenadas: - , -")
        self.status_bar.addPermanentWidget(self.lbl_coords)

        # 2. Main Layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # -- PANEL IZQUIERDO --
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(70) 
        self.left_panel.setStyleSheet("background-color: #e0e0e0; border-right: 1px solid #c0c0c0;")
        
        self.tools_layout = QVBoxLayout(self.left_panel)
        self.tools_layout.setContentsMargins(5, 10, 5, 10)
        self.tools_layout.setSpacing(15)

        # Botón HOME
        self.btn_home = self.create_tool_button("", "Resetear Zoom y Líneas")
        self.btn_home.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DirHomeIcon))
        self.btn_home.clicked.connect(self.action_home_reset) 
        
        self.btn_pan = self.create_tool_button("✋", "Mover (Pan)")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(lambda: self.set_mode("pan"))

        self.btn_line = self.create_tool_button("📏", "Dibujar Línea")
        self.btn_line.setCheckable(True)
        self.btn_line.clicked.connect(lambda: self.set_mode("line"))
        
        self.btn_overlay = self.create_tool_button("OL", "Activar Overlay")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setStyleSheet("QPushButton { font-weight: bold; color: purple; }")
        self.btn_overlay.clicked.connect(self.toggle_overlay)

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_line)

        self.tools_layout.addWidget(self.btn_home)
        self.tools_layout.addWidget(self.btn_pan)
        self.tools_layout.addWidget(self.btn_line)
        self.tools_layout.addSpacing(20)
        self.tools_layout.addWidget(self.btn_overlay)
        self.tools_layout.addStretch() 

        self.main_layout.addWidget(self.left_panel)

        # -- CENTRO --
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0,0,0,0)
        
        self.fig = Figure(figsize=(8, 6), facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.center_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.center_panel)

        # -- PANEL DERECHO --
        self.right_panel = QFrame()
        self.right_panel.setFixedWidth(260)
        self.right_panel.setStyleSheet("background-color: #f0f0f0; border-left: 1px solid #dcdcdc;")
        
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(15, 20, 15, 20)
        
        lbl_props = QLabel("Visualización")
        lbl_props.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.right_layout.addWidget(lbl_props)
        self.right_layout.addSpacing(20)

        lbl_cmap = QLabel("Paleta de Color:")
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(self.colormaps)
        self.combo_cmap.setCurrentText(self.current_cmap)
        self.combo_cmap.currentTextChanged.connect(self.update_layer_props)
        
        self.right_layout.addWidget(lbl_cmap)
        self.right_layout.addWidget(self.combo_cmap)
        self.right_layout.addSpacing(20)

        self.lbl_opacity = QLabel(f"Intensidad EBIC: {int(self.opacity*100)}%")
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setMinimum(0)
        self.slider_opacity.setMaximum(100)
        self.slider_opacity.setValue(int(self.opacity*100))
        self.slider_opacity.valueChanged.connect(self.update_layer_props)
        self.right_layout.addWidget(self.lbl_opacity)
        self.right_layout.addWidget(self.slider_opacity)
        
        self.right_layout.addStretch()
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

    # --- PANTALLA EN BLANCO ---
    def show_placeholder(self):
        """Limpia la gráfica y la deja totalmente blanca."""
        self.ax.clear()
        self.ax.axis('off') # Ocultar ejes y reglas
        self.ax.set_title("") 
        self.canvas.draw()

    # --- LÓGICA DE CARGA ---

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar TIF", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            # 1. Reset total (pone pantalla blanca y limpia variables)
            self.reset_entire_state()
            
            # 2. Intentar cargar
            success = self.data_manager.load_file(file_path)
            if success:
                self.img_height, self.img_width = self.data_manager.sem_data.shape
                # Si carga bien, inicializamos el plot
                self.initialize_plot()
            else:
                # Si falla, aseguramos que se vea blanco
                self.show_placeholder()

    def reset_entire_state(self):
        """Limpieza profunda"""
        self.layer_sem = None
        self.layer_ebic = None
        if self.cbar:
            try:
                self.cbar.remove()
            except: pass
            self.cbar = None

        self.stored_lines = []
        self.current_line_artist = None
        
        self.mode = "view"
        self.opacity = 0.5
        self.show_overlay = False
        
        # Reset UI
        self.slider_opacity.blockSignals(True)
        self.slider_opacity.setValue(50)
        self.slider_opacity.blockSignals(False)
        self.lbl_opacity.setText("Intensidad EBIC: 50%")
        
        self.btn_overlay.blockSignals(True)
        self.btn_overlay.setChecked(False)
        self.btn_overlay.blockSignals(False)
        
        self.tool_group.setExclusive(False)
        self.btn_pan.setChecked(False)
        self.btn_line.setChecked(False)
        self.tool_group.setExclusive(True)
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        # Mostrar pantalla en blanco al resetear
        self.show_placeholder()

    def initialize_plot(self):
        # ax.clear() borra cualquier estado anterior
        self.ax.clear()
        self.ax.axis('on') # Reactivar ejes para la imagen
        
        extent_fixed = [0, self.img_width, self.img_height, 0]

        # 1. SEM
        self.layer_sem = self.ax.imshow(self.data_manager.sem_data, 
                                        cmap='gray', 
                                        interpolation='nearest',
                                        aspect='equal',
                                        extent=extent_fixed)

        # 2. EBIC
        if self.data_manager.current_map is not None:
            data_ebic = self.data_manager.current_map
            vmin = np.nanmin(data_ebic)
            vmax = np.nanmax(data_ebic)

            self.layer_ebic = self.ax.imshow(data_ebic,
                                             cmap=self.current_cmap,
                                             alpha=self.opacity, 
                                             interpolation='nearest',
                                             aspect='equal',
                                             extent=extent_fixed,
                                             vmin=vmin, vmax=vmax)
            
            self.cbar = self.fig.colorbar(self.layer_ebic, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label('Corriente (nA)', rotation=270, labelpad=15)
            
            self.layer_ebic.set_visible(False)
            self.cbar.ax.set_visible(False)
        
        self.ax.set_title("Vista SEM")
        self.canvas.draw()

    # --- ACCIONES ---

    def action_home_reset(self):
        """Reset de vista (zoom y líneas) para la imagen actual."""
        if self.img_width == 0: return 

        self.ax.set_xlim(0, self.img_width)
        self.ax.set_ylim(self.img_height, 0)

        for line in self.stored_lines:
            line.remove()
        self.stored_lines.clear()
        self.current_line_artist = None
        self.line_start_point = None

        self.set_mode("view")
        self.canvas.draw()

    def toggle_overlay(self):
        self.show_overlay = self.btn_overlay.isChecked()
        if self.layer_ebic:
            self.layer_ebic.set_visible(self.show_overlay)
            if self.cbar:
                self.cbar.ax.set_visible(self.show_overlay)
            
            self.ax.set_title("Vista SEM" + (" + EBIC" if self.show_overlay else ""))
            self.canvas.draw()

    def update_layer_props(self, _=None):
        val = self.slider_opacity.value()
        self.opacity = val / 100.0
        self.lbl_opacity.setText(f"Intensidad EBIC: {val}%")
        
        if self.layer_ebic:
            self.layer_ebic.set_alpha(self.opacity)
            cmap_name = self.combo_cmap.currentText()
            self.layer_ebic.set_cmap(cmap_name)
            self.canvas.draw()

    # --- EVENTOS RATÓN ---

    def on_mouse_press(self, event):
        if event.inaxes != self.ax: return
        try:
            if self.mode == 'pan':
                self.pan_start = (event.xdata, event.ydata)
                self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.mode == 'line':
                self.line_start_point = (event.xdata, event.ydata)
                self.current_line_artist = Line2D([event.xdata, event.xdata], 
                                                  [event.ydata, event.ydata], 
                                                  color='red', linewidth=2)
                self.ax.add_line(self.current_line_artist)
                self.canvas.draw()
        except: pass

    def on_mouse_release(self, event):
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

    def on_mouse_move(self, event):
        try:
            if event.inaxes:
                txt = f"X: {int(event.xdata)}, Y: {int(event.ydata)}"
                self.lbl_coords.setText(txt)
            else:
                self.lbl_coords.setText("Coordenadas: - , -")
                return

            if self.mode == 'pan' and self.pan_start:
                dx = event.xdata - self.pan_start[0]
                dy = event.ydata - self.pan_start[1]
                self.ax.set_xlim(self.ax.get_xlim() - dx)
                self.ax.set_ylim(self.ax.get_ylim() - dy)
                self.canvas.draw()
            elif self.mode == 'line' and self.line_start_point and self.current_line_artist:
                self.current_line_artist.set_data([self.line_start_point[0], event.xdata], 
                                                  [self.line_start_point[1], event.ydata])
                self.canvas.draw()
        except: pass

    def zoom_fun(self, event):
        if self.img_width == 0: return
        try:
            base_scale = 1.2
            if event.inaxes != self.ax: return
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            xdata, ydata = event.xdata, event.ydata
            scale_factor = 1/base_scale if event.button == 'up' else base_scale
            
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            if new_width > self.img_width * 3: return

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            
            self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
            self.canvas.draw()
        except: pass