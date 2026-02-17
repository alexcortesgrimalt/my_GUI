from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QToolBar, QFrame, QLabel, 
                             QPushButton, QStatusBar, QButtonGroup, QSlider,
                             QComboBox, QMenu, QToolButton)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.image as mpimg 
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
        
        # --- NAVEGACIÓN DE FRAMES ---
        self.frames_list = []
        self.current_frame_idx = 0

        # --- UI SETUP ---
        self.setup_ui()
        
        # --- MATPLOTLIB EVENTOS ---
        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # --- INICIO: PANTALLA CON INSTRUCCIONES ---
        self.show_placeholder()

    def setup_ui(self):
        # 1. Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # --- BOTÓN CARGAR ---
        upload_action = QAction("Cargar .TIF Multi-Frame", self)
        upload_action.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        upload_action.triggered.connect(self.upload_image)
        toolbar.addAction(upload_action)

        # --- BOTÓN GUARDAR / EXPORTAR ---
        save_button = QToolButton()
        save_button.setText("Guardar Exportar...")
        save_button.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        save_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        save_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        save_menu = QMenu()
        
        # 1. Submenú: Save SEM image
        menu_sem = QMenu("Save SEM image", self)
        menu_sem.addAction("as .tif", lambda: self.save_data("sem", "tif"))
        menu_sem.addAction("as .png", lambda: self.save_data("sem", "png"))
        save_menu.addMenu(menu_sem)

        # 2. Submenú: Save Current Map (NUEVO)
        menu_ebic = QMenu("Save Current Map", self)
        menu_ebic.addAction("as .tif", lambda: self.save_data("ebic_img", "tif"))
        menu_ebic.addAction("as .png", lambda: self.save_data("ebic_img", "png"))
        save_menu.addMenu(menu_ebic)

        # 3. Submenú: Save SEM + Current Map
        menu_overlay = QMenu("Save SEM + Current Map", self)
        menu_overlay.addAction("as .tif", lambda: self.save_data("overlay", "tif"))
        menu_overlay.addAction("as .png", lambda: self.save_data("overlay", "png"))
        save_menu.addMenu(menu_overlay)

        # 4. Submenú: Save screen (Guarda ejes, zoom, lineas, etc)
        menu_screen = QMenu("Save screen", self)
        menu_screen.addAction("as .tif", lambda: self.save_data("screen", "tif"))
        menu_screen.addAction("as .png", lambda: self.save_data("screen", "png"))
        save_menu.addMenu(menu_screen)

        save_menu.addSeparator()
        
        # 5. Guardado CSV
        save_menu.addAction("Save SEM map (.csv)", lambda: self.save_data("sem", "csv"))
        save_menu.addAction("Save Current Map (.csv)", lambda: self.save_data("ebic", "csv"))

        save_button.setMenu(save_menu)
        toolbar.addWidget(save_button)

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

        # -- CENTRO (Matplotlib + Controles de Frame) --
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0,0,0,0)
        
        # Figura Matplotlib
        self.fig = Figure(figsize=(8, 6), facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.center_layout.addWidget(self.canvas)
        
        # --- BARRA DE NAVEGACIÓN DE FRAMES ---
        self.frame_nav_layout = QHBoxLayout()
        self.frame_nav_layout.setContentsMargins(10, 5, 10, 15)
        
        self.btn_prev_frame = QPushButton("◀ Anterior")
        self.btn_prev_frame.setFixedWidth(100)
        self.btn_prev_frame.clicked.connect(lambda: self.change_frame(-1))
        
        self.btn_next_frame = QPushButton("Siguiente ▶")
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

    # --- PANTALLA DE INSTRUCCIONES ---
    def show_placeholder(self):
        self.ax.clear()
        self.ax.axis('off') 
        
        texto_instrucciones = (
            "Bienvenido a Overlay Master\n\n"
            "Este software correlaciona imágenes de microscopía electrónica (SEM)\n"
            "con mediciones de corriente EBIC, mapeando la intensidad en cada punto.\n\n"
            "INSTRUCCIONES DE CARGA:\n"
            "Por favor, cargue un archivo en formato .tif multipágina que contenga:\n"
            "  • 1ª Imagen (Frame 0): Imagen de topografía SEM.\n"
            "  • 2ª Imagen (Frame 1): Mapa de corriente EBIC."
        )
        
        self.ax.text(0.5, 0.5, texto_instrucciones, 
                     transform=self.ax.transAxes,
                     ha='center', va='center', 
                     fontsize=11, color='#333333',
                     bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', edgecolor='#cccccc'))
        
        self.ax.set_title("") 
        
        # Desactivar botones de frame
        self.btn_prev_frame.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        self.lbl_frame_info.setText("Frame: - / -")
        
        self.canvas.draw()

    # --- LÓGICA DE CARGA ---

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar TIF", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.reset_entire_state()
            success = self.data_manager.load_file(file_path)
            if success:
                self.img_height, self.img_width = self.data_manager.sem_data.shape
                self.initialize_plot()
            else:
                self.show_placeholder()

    # --- NAVEGACIÓN DE FRAMES ---
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
            
            if self.current_frame_idx == 0:
                self.ax.set_title("Vista SEM (Frame 0)" + (" + EBIC" if self.show_overlay else ""))
            elif self.current_frame_idx == 1:
                self.ax.set_title("Vista EBIC Raw (Frame 1)" + (" + EBIC Overlay" if self.show_overlay else ""))
            
            self.update_frame_ui()
            self.canvas.draw()

    # --- LÓGICA DE GUARDADO MODIFICADA ---
    def save_data(self, content_type, file_format):
        if self.data_manager.sem_data is None:
            self.status_bar.showMessage("Error: No hay datos cargados para guardar.", 3000)
            return

        filter_str = f"{file_format.upper()} Files (*.{file_format})"
        file_path, _ = QFileDialog.getSaveFileName(self, f"Guardar como {file_format.upper()}", "", filter_str)
        
        if not file_path:
            return 
            
        if not file_path.lower().endswith(f".{file_format}"):
            file_path += f".{file_format}"

        try:
            # Opción 1: CSV de SEM o EBIC
            if content_type == "sem" and file_format == "csv":
                np.savetxt(file_path, self.data_manager.sem_data, delimiter=",")
            
            elif content_type == "ebic" and file_format == "csv":
                if self.data_manager.current_map is not None:
                    np.savetxt(file_path, self.data_manager.current_map, delimiter=",")
                else:
                    self.status_bar.showMessage("Error: No hay datos EBIC disponibles.", 3000)
                    return

            # Opción 2: Guardar solo imagen SEM base
            elif content_type == "sem" and file_format in ["tif", "png"]:
                mpimg.imsave(file_path, self.data_manager.sem_data, cmap='gray')

            # Opción 3: Guardar solo el Current Map (EBIC en colores usando el cmap actual)
            elif content_type == "ebic_img" and file_format in ["tif", "png"]:
                if self.data_manager.current_map is not None:
                    mpimg.imsave(file_path, self.data_manager.current_map, cmap=self.current_cmap)
                else:
                    self.status_bar.showMessage("Error: No hay datos EBIC disponibles.", 3000)
                    return

            # Opción 4: Screen Completa
            elif content_type == "screen":
                self.fig.savefig(file_path, format=file_format, bbox_inches='tight')

            # Opción 5: Overlay limpio a resolución original
            elif content_type == "overlay":
                temp_fig = Figure(figsize=(self.img_width/100, self.img_height/100), dpi=100)
                temp_ax = temp_fig.add_subplot(111)
                temp_ax.axis('off') 
                
                extent_fixed = [0, self.img_width, self.img_height, 0]
                
                temp_ax.imshow(self.data_manager.sem_data, cmap='gray', aspect='equal', extent=extent_fixed)
                
                if self.data_manager.current_map is not None:
                    temp_ax.imshow(self.data_manager.current_map, cmap=self.current_cmap, 
                                   alpha=self.opacity, aspect='equal', extent=extent_fixed,
                                   vmin=np.nanmin(self.data_manager.current_map), 
                                   vmax=np.nanmax(self.data_manager.current_map))
                
                temp_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                temp_fig.savefig(file_path, format=file_format, pad_inches=0)

            self.status_bar.showMessage(f"Guardado exitosamente en: {file_path}", 5000)

        except Exception as e:
            self.status_bar.showMessage(f"Error al guardar: {str(e)}", 5000)
            print(f"Excepción al guardar: {e}")

    # --------------------------------------------------------

    def reset_entire_state(self):
        self.layer_sem = None
        self.layer_ebic = None
        if self.cbar:
            try:
                self.cbar.remove()
            except: pass
            self.cbar = None

        self.stored_lines = []
        self.current_line_artist = None
        
        self.frames_list = []
        self.current_frame_idx = 0
        
        self.mode = "view"
        self.opacity = 0.5
        self.show_overlay = False
        
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

        self.show_placeholder()

    def initialize_plot(self):
        self.ax.clear()
        self.ax.axis('on')
        
        self.frames_list = []
        if self.data_manager.sem_data is not None:
            self.frames_list.append(self.data_manager.sem_data)
        if self.data_manager.current_map is not None:
            self.frames_list.append(self.data_manager.current_map)
            
        self.current_frame_idx = 0
        self.update_frame_ui()
        
        extent_fixed = [0, self.img_width, self.img_height, 0]

        # 1. Capa Base (Inicialmente SEM Frame 0)
        base_data = self.frames_list[0] if self.frames_list else np.zeros((self.img_height, self.img_width))
        self.layer_sem = self.ax.imshow(base_data, 
                                        cmap='gray', 
                                        interpolation='nearest',
                                        aspect='equal',
                                        extent=extent_fixed)
        
        vmin = np.nanmin(base_data)
        vmax = np.nanmax(base_data)
        if vmin != vmax: self.layer_sem.set_clim(vmin, vmax)

        # 2. Capa EBIC (Overlay - Siempre usa current_map)
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
        
        self.ax.set_title("Vista SEM (Frame 0)")
        self.canvas.draw()

    # --- ACCIONES ---

    def action_home_reset(self):
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
            
            titulo_base = "Vista SEM (Frame 0)" if self.current_frame_idx == 0 else "Vista EBIC Raw (Frame 1)"
            self.ax.set_title(titulo_base + (" + EBIC Overlay" if self.show_overlay else ""))
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