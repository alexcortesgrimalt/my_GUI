import xml.etree.ElementTree as ET
import re

class Metadata:
    def __init__(self, xml_content):
        self.xml_content = xml_content
        self.data = self._parse_metadata()

    def _parse_metadata(self):
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'image': 'http://ns.pointelectronic.com/Image/1.0/',
            'efa': 'http://ns.pointelectronic.com/EFA/1.0/',
            'cdev': 'http://ns.pointelectronic.com/CommonDevice/1.0/',
            'd6sp': 'http://ns.pointelectronic.com/DISS6/1.0/types/ScanParameters#',
            'photoshop': 'http://ns.adobe.com/photoshop/1.0/'
        }

        try:
            root = ET.fromstring(self.xml_content)
            desc = root.find('.//rdf:Description', ns)
        except ET.ParseError:
            print("Error parseando XML")
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
            'Contrast': get_nested_value('efa:Contrast', 0.0),
            'EffectiveAmpGain': get_nested_value('efa:EffectiveAmpGain', 1e6),
            'OutputOffset': get_nested_value('efa:OutputOffset', 0.0),
            'InputOffset': get_nested_value('efa:InputOffset', 0.0),
            'InverseEnabled': bool(int(desc.findtext('efa:InverseEnabled', '0', namespaces=ns))),
            'BiasEnabled': bool(int(desc.findtext('efa:BiasEnabled', '0', namespaces=ns))),
            'BiasVoltage': get_nested_value('efa:Bias', 0.0),
           
            'BeamCurrent': get_nested_value('cdev:BeamCurrent', 0.0),
            'AccelerationVoltage': get_nested_value('cdev:HV', 0.0),
            'Magnification': float(desc.findtext('cdev:Mag', '0.0', namespaces=ns)),
            'WorkingDistance_mm': get_nested_value('cdev:WD', 0.0),
            'ScanRotation_deg': get_nested_value('cdev:ScanRotAngle', 0.0),
            'BiasVoltage': get_nested_value('efa:Bias', 0.0),
            'TimeConstant': get_nested_value('efa:TimeConstant', 0.0),
            'DwellTime_us': float(desc.findtext('d6sp:ScanParameters/d6sp:AcquisitionTime', '0us', namespaces=ns).replace('us','')),
            'DateCreated': desc.findtext('photoshop:DateCreated', 'Unknown', namespaces=ns)
        }
        return data

def extract_xmp_metadata(file_path):
    """Extrae el bloque XMP raw del archivo TIFF binario."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Regex para encontrar el paquete XMP
        pattern = re.compile(rb'<\?xpacket begin=.*?\?>.*?<\?xpacket end=.*?\?>', re.DOTALL)
        match = pattern.search(content)
        
        if match:
            return match.group(0).decode("utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"Error extrayendo XMP: {e}")
    return None