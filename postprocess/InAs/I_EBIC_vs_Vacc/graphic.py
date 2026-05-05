import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

# ==========================================
# 0. PLOT SETTINGS (LATEX STYLE & HUGE SIZES)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern (LaTeX look)
    # 'text.usetex': True,     # <-- UNCOMMENT THIS IF YOU HAVE MIKTEX/TEXLIVE INSTALLED
    'axes.labelsize': 28,      # Huge axis labels
    'xtick.labelsize': 20,     # Huge X numbers
    'ytick.labelsize': 20      # Huge Y numbers
})

# ==========================================
# 1. METADATA EXTRACTION
# ==========================================
def extract_xmp_metadata(file_path):
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        pattern = re.compile(rb'<\?xpacket begin=.*?\?>.*?<\?xpacket end=.*?\?>', re.DOTALL)
        match = pattern.search(content)
        if match:
            return match.group(0).decode("utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"Error extracting XMP: {e}")
    return None

class Metadata:
    def __init__(self, xml_content):
        self.xml_content = xml_content
        self.data = self._parse_metadata()

    def _parse_metadata(self):
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'image': 'http://ns.pointelectronic.com/Image/1.0/',
            'efa': 'http://ns.pointelectronic.com/EFA/1.0/',
        }
        try:
            root = ET.fromstring(self.xml_content)
            desc = root.find('.//rdf:Description', ns)
        except ET.ParseError:
            return {}

        if desc is None:
            return {}

        def get_nested_value(tag, default=0.0):
            elem = desc.find(tag, ns)
            if elem is not None:
                val = elem.find('rdf:value', ns)
                if val is not None and val.text:
                    try: return float(val.text.strip())
                    except: pass
                if elem.text:
                    try: return float(elem.text.strip())
                    except: pass
            return float(default)

        data = {
            'PixelSizeX': float(desc.findtext('image:PixelSizeX', '1e-6', namespaces=ns)),
            'Contrast': get_nested_value('efa:Contrast', 1.0),
            'EffectiveAmpGain': get_nested_value('efa:EffectiveAmpGain', 1e6),
            'OutputOffset': get_nested_value('efa:OutputOffset', 0.0),
            'InputOffset': get_nested_value('efa:InputOffset', 0.0),
            'InverseEnabled': bool(int(desc.findtext('efa:InverseEnabled', '0', namespaces=ns))),
            'BiasEnabled': bool(int(desc.findtext('efa:BiasEnabled', '0', namespaces=ns))),
            'BiasVoltage': get_nested_value('efa:Bias', 0.0)
        }
        return data

# ==========================================
# 2. CURRENT CONVERSION
# ==========================================
def compute_current_nA(pixel_matrix, m_data):
    C = m_data.get('Contrast', 1.0)
    G = m_data.get('EffectiveAmpGain', 1e6)
    O = m_data.get('OutputOffset', 0.0)
    I = m_data.get('InputOffset', 0.0)
    inv = m_data.get('InverseEnabled', False)
    
    scale = 1.0 
    offset = -0.5 
    
    if pixel_matrix.dtype.kind in 'ui':
        bit_depth_max = float(np.iinfo(pixel_matrix.dtype).max)
    else:
        bit_depth_max = 65535.0
        
    pixels_float = pixel_matrix.astype(np.float64)
    voltage = (pixels_float / bit_depth_max) * scale + offset

    if inv:
        current = (((voltage - O) / C) + I) / G * -1e9
    else:
        current = (((voltage - O) / C) - I) / G * 1e9
        
    return current

# ==========================================
# 3. MAIN ROUTINE
# ==========================================
image_folder = '.'  
search_pattern = os.path.join(image_folder, '*(1).tif')
files = glob.glob(search_pattern)

vacc_values = []
iebic_values = []

print(f"Scanning directory: {os.path.abspath(image_folder)}")
print(f"Found {len(files)} valid files (frame 1).\n")

for file_path in files:
    filename = os.path.basename(file_path)
    match = re.search(r'([\d]+)d([\d]+)\(1\)\.tif', filename)
    
    if match:
        vacc = float(f"{match.group(1)}.{match.group(2)}")
        
        try:
            xml_str = extract_xmp_metadata(file_path)
            if not xml_str:
                continue
            
            m_data = Metadata(xml_str).data
            
            img = Image.open(file_path)
            img.seek(1)
            raw_ebic_img = np.array(img)
            
            current_map_nA = compute_current_nA(raw_ebic_img, m_data)
            
            i_max = np.max(current_map_nA)
            i_min = np.min(current_map_nA)
            i_ebic_nA = 0.5 * abs(i_max - i_min)
            
            vacc_values.append(vacc)
            iebic_values.append(i_ebic_nA)
            print(f"Processed: {filename} | Vacc: {vacc} kV | I_EBIC: {i_ebic_nA:.2f} nA")
            
        except EOFError:
            pass
        except Exception as e:
            print(f"[{filename}] Error: {e}")

# ==========================================
# 4. PLOTTING
# ==========================================
if not vacc_values:
    raise ValueError("No valid data extracted.")

vacc_sorted, iebic_sorted = zip(*sorted(zip(vacc_values, iebic_values)))

plt.figure(figsize=(9, 6))

# Plot line
plt.plot(vacc_sorted, iebic_sorted, marker='o', markersize=10, 
         linestyle='-', linewidth=2.5, color='black')

# Axes labels forced into MathText for strict LaTeX rendering
plt.xlabel(r'$V_{acc} \ (\mathrm{kV})$')
plt.ylabel(r'$I_{EBIC} \ (\mathrm{nA})$')

# Operational parameters box 
#params_text = (
#    r"$\mathbf{Parameters:}$" + "\n"
#    r"$I_{beam} = \mathrm{XX \ pA}$" + "\n"
#    r"$\mathrm{Dwell \ Time} = \mathrm{XX \ \mu s}$" + "\n"
#    r"$\mathrm{Magnification} = \mathrm{XX \ kX}$"
#)

#plt.text(0.05, 0.95, params_text, transform=plt.gca().transAxes,
#         fontsize=16, verticalalignment='top',
#         bbox=dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.9, edgecolor='black'))

# Layout adjustments
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.show()