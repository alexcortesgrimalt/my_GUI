import sys
from PyQt6.QtWidgets import QApplication
from gui_setup import CorrelationGui

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Opcional: Estilo Fusion para que se vea igual en Windows/Mac/Linux
    app.setStyle("Fusion") 
    
    window = CorrelationGui()
    window.show()
    sys.exit(app.exec())