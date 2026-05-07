import xml.etree.ElementTree as ET
import re

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
        print(f"❌ Error leyendo el archivo TIF: {e}")
    return None

def find_compensation_current(xml_content):
    """Busca la corriente de compensación y lista todos los parámetros EFA."""
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'efa': 'http://ns.pointelectronic.com/EFA/1.0/'
    }

    try:
        root = ET.fromstring(xml_content)
        desc = root.find('.//rdf:Description', ns)
    except ET.ParseError:
        print("❌ Error parseando la estructura XML del TIF.")
        return

    if desc is None:
        print("⚠️ No se encontró el bloque <rdf:Description> en los metadatos.")
        return

    # Función interna para extraer el valor (soporta el anidado de pointelectronic)
    def get_value(element):
        if element is None: return None
        val = element.find('rdf:value', ns)
        if val is not None and val.text:
            return val.text.strip()
        if element.text:
            return element.text.strip()
        return None

    # 1. Búsqueda específica de etiquetas de compensación
    target_tags = ['efa:Compensation', 'efa:CompensationCurrent', 'efa:InputOffset', 'efa:Offset']
    
    print("\n--- BÚSQUEDA DE COMPENSATION CURRENT ---")
    found_any = False
    for tag in target_tags:
        elem = desc.find(tag, ns)
        val = get_value(elem)
        if val is not None:
            found_any = True
            try:
                # Intentamos pasarlo a formato científico para leerlo mejor
                val_float = float(val)
                print(f"✅ Encontrado '{tag}': {val_float:.3e} Amperios")
            except ValueError:
                print(f"✅ Encontrado '{tag}': {val}")

    if not found_any:
        print("❌ No se encontró ninguna etiqueta estándar de Compensation.")

    # 2. Volcado de todos los parámetros EFA para inspección manual
    print("\n--- TODOS LOS PARÁMETROS DEL AMPLIFICADOR (EFA) ENCONTRADOS ---")
    for child in desc:
        # Filtramos para mostrar solo los tags que pertenecen al namespace del amplificador
        if 'EFA/1.0/' in child.tag or 'efa' in child.tag:
            tag_name = child.tag.split('}')[-1] # Limpiamos el namespace para leerlo mejor
            val = get_value(child)
            print(f"  • {tag_name}: {val}")


# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    # Sustituye esto por la ruta de tu mapa EBIC real
    ARCHIVO_TIF = "NWs_InAs_3H_4L_0Vb.tif" 
    
    print(f"Analizando: {ARCHIVO_TIF} ...")
    xml_data = extract_xmp_metadata(ARCHIVO_TIF)
    
    if xml_data:
        find_compensation_current(xml_data)
    else:
        print("❌ No se pudieron extraer metadatos XMP del archivo.")