import numpy as np
from PIL import Image
from metadata_loader import Metadata, extract_xmp_metadata

class SEMDataManager:
    def __init__(self):
        self.sem_data = None
        self.current_map = None # EBIC data en nA
        self.voltage_map = None # <-- NUEVO: Almacenará el mapa de Voltage Contrast
        self.metadata = None
        self.pixel_size = 1.0

    def load_file(self, file_path):
        try:
            # 1. Extraer Metadata
            xml_str = extract_xmp_metadata(file_path)
            if xml_str:
                self.metadata = Metadata(xml_str).data
                self.pixel_size = self.metadata.get('PixelSizeX', 1e-6)
            else:
                print("Advertencia: No se encontró metadata XML.")
                self.metadata = None

            # 2. Cargar Imagenes (Frames)
            img = Image.open(file_path)
            
            # Frame 0: SEM (Pixel Map)
            img.seek(0)
            self.sem_data = np.array(img).astype(np.float64)

            # --- NUEVO: Generar el Voltage Map de la capa SEM ---
            if self.metadata is not None:
                self._compute_voltage_map()
            else:
                self.voltage_map = None
            # ----------------------------------------------------

            # Frame 1: EBIC (Current Map) - Si existe
            try:
                img.seek(1)
                raw_ebic = np.array(img).astype(np.float64)
                if self.metadata:
                    self.current_map = self._compute_current(raw_ebic)
                else:
                    self.current_map = raw_ebic # Fallback sin conversion
            except EOFError:
                print("El archivo solo tiene 1 frame. No hay mapa de corriente.")
                self.current_map = None

            return True

        except Exception as e:
            print(f"Error cargando archivo: {e}")
            return False

    def _compute_current(self, pixels):
        """Aplica la formula de conversion de pixeles a nA."""
        m = self.metadata
        
        # Usamos .get() para mayor robustez si falta alguna etiqueta
        C = m.get('Contrast', 1.0)
        G = m.get('EffectiveAmpGain', 1e6)
        O = m.get('OutputOffset', 0.0)
        I = m.get('InputOffset', 0.0)
        inv = m.get('InverseEnabled', False)
        
        scale = 1  # mV
        offset = -0.5  # mV
        
        # Normalizar voltaje (16-bits)
        voltage = (pixels / 65535.0) * scale + offset

        # Calcular corriente
        if inv:
            current = (((voltage - O) / C) + I) / G * -1e9
        else:
            current = (((voltage - O) / C) - I) / G * +1e9
            
        return current

    # --- NUEVA FUNCIÓN ---
    def _compute_voltage_map(self):
        """Calcula el contraste de voltaje basándose en la imagen SEM y la metadata."""
        V_offset = self.metadata.get('OutputOffset', 0.0)
        V_contrast = self.metadata.get('Contrast', 1.0)
        V_bias = self.metadata.get('BiasVoltage', 0.0)
        
        # Normalizamos usando la escala máxima del digitalizador (16-bits = 65535) 
        # para mantener coherencia con la conversión de corriente
        sem_norm = self.sem_data / 65535.0
        
        # Cálculo físico: Escala aplicada al porcentaje de señal + Offsets
        self.voltage_map = (sem_norm * V_contrast) + V_offset # + V_bias