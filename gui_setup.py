from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QToolBar, QFrame, QLabel, 
                             QPushButton, QStatusBar, QButtonGroup, QSlider,
                             QComboBox, QMenu, QToolButton, QTabWidget,
                             QDoubleSpinBox, QCheckBox, QMessageBox, QSpinBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.image as mpimg 
import numpy as np
import scipy.ndimage as ndi

# Import the data manager and analyzers
from image_handler import SEMDataManager
from junction_analyzer import JunctionAnalyzer
from profile_manager import ProfileManager

class CorrelationGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Operations Master")
        self.resize(1400, 900)

        # --- DATA MANAGER ---
        self.data_manager = SEMDataManager()

        # --- VISUAL STATE ---
        self.img_width = 0
        self.img_height = 0
        
        # Dynamic physical variables (can be nm, um, or mm)
        self.pixel_size_phys = 1.0
        self.width_phys = 0.0
        self.height_phys = 0.0
        self.unit_label = "\u03BCm" # Default
        self.unit_factor = 1e6     # Conversion factor from meters
        
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
        self.current_cmap = 'plasma'

        # Interactive variables
        self.pan_start = None 
        self.line_start_point = None
        self.current_line_artist = None 
        self.stored_lines = [] 
        self.junction_line_artist = None
        
        # --- FRAME NAVIGATION ---
        self.frames_list = []
        self.current_frame_idx = 0

        # --- UI SETUP ---
        self.setup_ui()
        
        # --- INITIALIZE MANAGERS ---
        self.profile_manager = ProfileManager(self.ax, self.canvas)

        # --- MATPLOTLIB EVENTS ---
        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # --- START: INSTRUCTION SCREEN ---
        self.show_placeholder()

    def setup_ui(self):
        # 1. Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        upload_action = QAction("Load Multi-Frame .TIF", self)
        upload_action.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        upload_action.triggered.connect(self.upload_image)
        toolbar.addAction(upload_action)

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

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_pan)
        self.tool_group.addButton(self.btn_line)

        self.tools_layout.addWidget(self.btn_home)
        self.tools_layout.addWidget(self.btn_pan)
        self.tools_layout.addWidget(self.btn_line)
        self.tools_layout.addWidget(self.btn_grid)
        self.tools_layout.addSpacing(20)
        self.tools_layout.addWidget(self.btn_overlay)
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
        vis_layout.addSpacing(20)

        lbl_cmap = QLabel("Color Palette:")
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(self.colormaps)
        self.combo_cmap.setCurrentText(self.current_cmap)
        self.combo_cmap.currentTextChanged.connect(self.update_layer_props)
        vis_layout.addWidget(lbl_cmap)
        vis_layout.addWidget(self.combo_cmap)
        vis_layout.addSpacing(20)

        self.lbl_opacity = QLabel(f"EBIC Intensity: {int(self.opacity*100)}%")
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
        
        junc_layout.addSpacing(15)
        lbl_plots = QLabel("3. Select outputs:")
        lbl_plots.setStyleSheet("font-weight: bold;")
        junc_layout.addWidget(lbl_plots)
        
        self.chk_a = QCheckBox("a) SEM ROI - EBIC / Current ROI")
        self.chk_b = QCheckBox("b) Junction Detection Comparison")
        self.chk_c = QCheckBox("c) Raw EBIC & Filtered EBIC")
        self.chk_d = QCheckBox("d) Canny (Filtered, Spline) - General")
        self.chk_e = QCheckBox("e) Observe Junction (Over Main)")
        self.chk_e.setChecked(True) 
        
        junc_layout.addWidget(self.chk_a)
        junc_layout.addWidget(self.chk_b)
        junc_layout.addWidget(self.chk_c)
        junc_layout.addWidget(self.chk_d)
        junc_layout.addWidget(self.chk_e)
        
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
        lbl_prof_inst2 = QLabel("2. Adjust points by dragging:\n   - Center: Slide along baseline.\n   - Ends: Extend/Shrink.")
        lbl_prof_inst2.setStyleSheet("font-style: italic; color: #555555;")
        lbl_prof_inst2.setWordWrap(True)
        prof_layout.addWidget(lbl_prof_inst2)

        prof_layout.addSpacing(15)
        lbl_prof_outs = QLabel("3. Select outputs:")
        lbl_prof_outs.setStyleSheet("font-weight: bold;")
        prof_layout.addWidget(lbl_prof_outs)

        self.chk_prof_a = QCheckBox("a) Extracted 1D Data Profiles")
        self.chk_prof_b = QCheckBox("b) Cross-Section View")
        self.chk_prof_c = QCheckBox("c) Signal Overlay")
        self.chk_prof_d = QCheckBox("d) Export Matrix directly")
        
        self.chk_prof_a.setChecked(True)

        prof_layout.addWidget(self.chk_prof_a)
        prof_layout.addWidget(self.chk_prof_b)
        prof_layout.addWidget(self.chk_prof_c)
        prof_layout.addWidget(self.chk_prof_d)

        prof_layout.addSpacing(15)
        self.btn_extract_profiles = QPushButton("Extract Data")
        self.btn_extract_profiles.setStyleSheet("QPushButton { font-weight: bold; background-color: #d1e7dd; padding: 8px; }")
        self.btn_extract_profiles.clicked.connect(self.extract_profiles_data)
        prof_layout.addWidget(self.btn_extract_profiles)

        prof_layout.addStretch()


        # Add tabs
        self.tabs_right.addTab(self.tab_vis, "Vis")
        self.tabs_right.addTab(self.tab_junc, "Junction")
        self.tabs_right.addTab(self.tab_prof, "Profiles")
        
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
        self.ax.clear()
        self.ax.axis('off') 
        
        instructions_text = (
            "Welcome to Map Operations Master\n\n"
            "This software correlates scanning electron microscopy (SEM) images\n"
            "with EBIC current measurements, allowing to carry out many map operations!.\n\n"
            "LOAD INSTRUCTIONS:\n"
            "Please load a multi-page .tif file containing:\n"
            "  • 1st Image: SEM topography image.\n"
            "  • 2nd Image: EBIC current map."
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

    # --- LOAD LOGIC ---
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load TIF", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.reset_entire_state()
            success = self.data_manager.load_file(file_path)
            if success:
                self.img_height, self.img_width = self.data_manager.sem_data.shape
                self.initialize_plot()
            else:
                self.show_placeholder()

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
            
            if self.current_frame_idx == 0:
                self.ax.set_title("SEM View (Frame 0)" + (" + EBIC Overlay" if self.show_overlay else ""))
            elif self.current_frame_idx == 1:
                self.ax.set_title("Raw EBIC View (Frame 1)" + (" + EBIC Overlay" if self.show_overlay else ""))
            
            self.update_frame_ui()
            self.canvas.draw()

    # --- AUTOMATIC SCALE BAR CALCULATION (DYNAMIC WITH ZOOM) ---
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

    # --------------------------------------------------------
    def reset_entire_state(self):
        self.layer_sem = None
        self.layer_ebic = None
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None

        self.stored_lines = []
        self.current_line_artist = None
        if self.junction_line_artist:
            try: self.junction_line_artist.remove()
            except: pass
            self.junction_line_artist = None
            
        if hasattr(self, 'profile_manager'):
            self.profile_manager.clear()
        
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
        
        self.ax.set_xlabel(f"Distance ({self.unit_label})")
        self.ax.set_ylabel(f"Distance ({self.unit_label})")
        self.ax.set_title("SEM View (Frame 0)")
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

        c0 = x0_phys / self.pixel_size_phys
        r0 = (self.height_phys - y0_phys) / self.pixel_size_phys
        c1 = x1_phys / self.pixel_size_phys
        r1 = (self.height_phys - y1_phys) / self.pixel_size_phys

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

        analyzer = JunctionAnalyzer(pixel_size_m=pixel_size_m)
        results = analyzer.detect(
            roi_sem, 
            manual_line_px, 
            roi_current=roi_ebic, 
            weight_current=10.0,
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

    # --- PROFILES FUNCTIONALITY ---
    def generate_profiles_action(self):
        if self.data_manager.sem_data is None:
            QMessageBox.warning(self, "Error", "No SEM image loaded.")
            return
            
        source = self.combo_baseline_source.currentText()
        
        # Determinar de dónde sacar los puntos de la línea base (p0 y p1)
        if source == "Manual Line":
            if len(self.stored_lines) != 1:
                QMessageBox.warning(self, "Error", "Please draw exactly ONE manual baseline on the image.")
                return
            line = self.stored_lines[0]
            xdata, ydata = line.get_data()
            p0 = (xdata[0], ydata[0])
            p1 = (xdata[-1], ydata[-1])
            
        else:  # source == "Detected Junction"
            if self.junction_line_artist is None:
                QMessageBox.warning(self, "Error", "No Detected Junction found. Please run the Junction Detection first.")
                return
            xdata, ydata = self.junction_line_artist.get_data()
            # Cogemos el primer y último punto de la curva/línea ajustada para establecer la dirección
            p0 = (xdata[0], ydata[0])
            p1 = (xdata[-1], ydata[-1])
        
        num_profiles = self.spin_prof_count.value()
        
        # Escalar el valor de la longitud en um a las unidades actuales del canvas
        length_um = self.spin_prof_length.value()
        length_m = length_um * 1e-6
        plot_length = length_m * self.unit_factor 

        self.profile_manager.generate_profiles(
            p0=p0,
            p1=p1,
            num_profiles=num_profiles,
            default_length=plot_length / 2.0  # La mitad hacia arriba y la mitad hacia abajo
        )
        self.set_mode("view") # Desactiva el modo línea para poder arrastrar libremente

    def extract_profiles_data(self):
        """Función placeholder para la futura extracción de datos a), b), c)..."""
        if not self.profile_manager.profiles:
            QMessageBox.warning(self, "Error", "No profiles generated yet.")
            return
            
        QMessageBox.information(self, "Working on it", "Data extraction logic will be implemented here.")

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
            
        self.profile_manager.clear() # Limpiar los perfiles generados también
            
        self.current_line_artist = None
        self.line_start_point = None

        self.set_mode("view")
        self.draw_scale_bar()
        self.canvas.draw()

    def toggle_grid(self):
        if self.width_phys > 0:
            self.ax.grid(self.btn_grid.isChecked())
            self.canvas.draw()

    def toggle_overlay(self):
        self.show_overlay = self.btn_overlay.isChecked()
        if self.layer_ebic:
            self.layer_ebic.set_visible(self.show_overlay)
            if self.cbar:
                self.cbar.ax.set_visible(self.show_overlay)
            
            base_title = "SEM View (Frame 0)" if self.current_frame_idx == 0 else "Raw EBIC View (Frame 1)"
            self.ax.set_title(base_title + (" + EBIC Overlay" if self.show_overlay else ""))
            self.canvas.draw()

    def update_layer_props(self, _=None):
        val = self.slider_opacity.value()
        self.opacity = val / 100.0
        self.lbl_opacity.setText(f"EBIC Intensity: {val}%")
        
        if self.layer_ebic:
            self.layer_ebic.set_alpha(self.opacity)
            cmap_name = self.combo_cmap.currentText()
            self.layer_ebic.set_cmap(cmap_name)
            self.canvas.draw()

    # --- MOUSE EVENTS ---
    def on_mouse_press(self, event):
        if event.inaxes != self.ax: return
        
        # 1. Si estamos en modo "view", chequeamos si el usuario intenta interactuar con un Perfil
        if self.mode == 'view' and self.profile_manager.on_press(event):
            return 
        
        try:
            if self.mode == 'pan':
                self.pan_start = (event.xdata, event.ydata)
                self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.mode == 'line':
                # Remove any existing line and profiles to ensure only 1 is drawn
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
        except: pass

    def on_mouse_release(self, event):
        # Soltar la interacción con los Perfiles
        if self.mode == 'view' and self.profile_manager.on_release(event): return

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
        # Actualizar posición de las coordenadas en la barra
        try:
            if event.inaxes:
                px_x = int(event.xdata / self.pixel_size_phys)
                px_y = int((self.height_phys - event.ydata) / self.pixel_size_phys)
                px_x = max(0, min(px_x, self.img_width - 1))
                px_y = max(0, min(px_y, self.img_height - 1))
                self.lbl_coords.setText(f"Coordinates (Px): X {px_x}, Y {px_y}")
            else:
                self.lbl_coords.setText("Coordinates (Px): - , -")
                return
        except: pass

        # Arrastrar perfiles si estamos sobre uno en modo "view"
        if self.mode == 'view' and self.profile_manager.on_drag(event): return

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
                self.canvas.draw()
                
            elif self.mode == 'line' and self.line_start_point and self.current_line_artist:
                self.current_line_artist.set_data([self.line_start_point[0], event.xdata], 
                                                  [self.line_start_point[1], event.ydata])
                self.canvas.draw()
        except: pass

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

            if new_xlim[0] < 0 or new_xlim[1] > self.width_phys: new_xlim = [0, self.width_phys]
            if new_ylim[0] < 0 or new_ylim[1] > self.height_phys: new_ylim = [0, self.height_phys]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.draw_scale_bar()
            self.canvas.draw()
        except: pass