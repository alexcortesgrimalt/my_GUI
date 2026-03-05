import numpy as np
from PIL import Image
from metadata_loader import Metadata, extract_xmp_metadata

class SEMDataManager:
    def __init__(self):
        self.sem_data = None
        self.current_map = None # EBIC data en nA
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
        
        C = m['Contrast']
        G = m['EffectiveAmpGain']
        O = m['OutputOffset']
        I = m['InputOffset']
        inv = m['InverseEnabled']
        # Bias logic commented out as per your snippet
        
        scale = 1  # mV
        offset = -0.5  # mV
        
        # Normalizar voltaje
        voltage = (pixels / 65535) * scale + offset

# 1. Extraer Metadata
        xml_str = extract_xmp_metadata("../NW_2024-06-18_15-30-00_SEM_EBIC.tif")
            
           # --- AÑADE ESTO TEMPORALMENTE ---
        print("\n--- INICIO DEL XML RAW ---")
        print(xml_str)
        print("--- FIN DEL XML RAW ---\n")
            # --------------------------------

   


        # Calcular corriente
        if inv:
            current = (((voltage - O) / C) + I) / G * -1e9
        else:
            current = (((voltage - O) / C) - I) / G * +1e9
            
        return current