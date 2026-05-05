import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

# ==========================================
# 0. PLOT SETTINGS (LATEX STYLE & SIZES)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  
    'axes.labelsize': 24,      
    'xtick.labelsize': 18,     
    'ytick.labelsize': 18,
    'legend.fontsize': 16
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
# 3. READ MAPS (TIFF)
# ==========================================
image_folder = '.'  
search_pattern = os.path.join(image_folder, 'HL_Vb*.tif')
files = glob.glob(search_pattern)

vbias_maps = []
imean_maps = []

print(f"Scanning maps in: {os.path.abspath(image_folder)}")

for file_path in files:
    filename = os.path.basename(file_path)
    # Extract bias (e.g., HL_Vb0d01 -> 0.01)
    match = re.search(r'HL_Vb(\d+(?:d\d+)?)', filename)
    
    if match:
        bias_str = match.group(1).replace('d', '.')
        vbias = float(bias_str)
        
        try:
            xml_str = extract_xmp_metadata(file_path)
            if not xml_str:
                continue
            
            m_data = Metadata(xml_str).data
            
            img = Image.open(file_path)
            # Try to read the secondary frame (where EBIC signal usually is), otherwise read frame 0
            try:
                img.seek(1)
            except EOFError:
                img.seek(0)
                
            raw_ebic_img = np.array(img)
            current_map_nA = compute_current_nA(raw_ebic_img, m_data)
            
            # Extract mean current from the map
            i_mean_nA = np.mean(current_map_nA)
            
            vbias_maps.append(vbias)
            imean_maps.append(i_mean_nA)
            print(f"Processed: {filename} | Vbias: {vbias} V | Mean Current: {i_mean_nA:.2f} nA")
            
        except Exception as e:
            print(f"[{filename}] Error: {e}")

# ==========================================
# 4. READ IV CURVE (CSV)
# ==========================================
csv_file = "IV_curve.csv"
v_csv = []
i_csv = []

if os.path.exists(csv_file):
    print(f"\nReading CSV file: {csv_file}")
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
    data_started = False
    for line in lines:
        if "I Terminal 1,V Terminal 1" in line:
            data_started = True
            continue
        if data_started:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    # CSV: I is in column 0, V is in column 1
                    v_val = float(parts[1])
                    
                    # Filter for only positive voltages
                    if v_val >= 0 and v_val <= 0.045:
                        i_val = float(parts[0]) * 1e9 # Convert A to nA to match scales
                        i_csv.append(i_val)
                        v_csv.append(v_val)
                except ValueError:
                    pass
else:
    print("\nWarning: IV_curve.csv not found")

# ==========================================
# 5. PLOTTING
# ==========================================
plt.figure(figsize=(10, 7))

# CSV Curve
if v_csv and i_csv:
    plt.plot(v_csv, i_csv, linestyle='-', linewidth=2.5, color='royalblue', 
             label='I(V) with Kethley', zorder=1)

# Mean points from TIFF maps
if vbias_maps:
    vbias_sorted, imean_sorted = zip(*sorted(zip(vbias_maps, imean_maps)))
    plt.scatter(vbias_sorted, imean_sorted, marker='o', s=120, color='crimson', 
                edgecolor='black', label='Mean($I_{EBIC}(x,y)$)', zorder=2)

plt.xlabel(r'$V_{bias} \ (\mathrm{V})$')
plt.ylabel(r'$I_{mean} \ (\mathrm{nA})$')

# Operational parameters box 
#params_text = (
#    r"$\mathbf{Acquisition \ Parameters}$" + "\n"
#    r"$V_{acc} = \mathrm{XX \ kV}$" + "\n"
#    r"$I_{beam} = \mathrm{XX \ pA}$" + "\n"
#    r"$\mathrm{Dwell \ Time} = \mathrm{XX \ \mu s}$" + "\n"
#    r"$\mathrm{Magnification} = \mathrm{XX \ kX}$"
#)

#plt.text(0.03, 0.96, params_text, transform=plt.gca().transAxes,
#         fontsize=14, verticalalignment='top',
#         bbox=dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.9, edgecolor='black'))

plt.legend(loc='lower right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.show()