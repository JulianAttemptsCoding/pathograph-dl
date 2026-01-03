import requests
import sys
from xml.etree import ElementTree as ET

def main():
    url = "https://api.imf.org/external/sdmx/2.1/datastructure/IMF.STA/DSD_IMTS/1.0.0"
    print(f"Fetching {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        
        # Namespaces
        ns = {
            'mes': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
            'str': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
            'com': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
        }
        
        # Find Dimensions
        # Path: Structure > Structures > DataStructures > DataStructure > DataStructureComponents > DimensionList > Dimension
        dims = root.findall('.//str:Dimension', ns)
        
        print(f"Found {len(dims)} dimensions.")
        for d in dims:
            did = d.get('id')
            # LocalRep > Enumeration > Ref
            # or CoreRep?
            
            # Check LocalRepresentation
            enum_ref_id = "N/A"
            local_rep = d.find('str:LocalRepresentation', ns)
            if local_rep is not None:
                enum = local_rep.find('str:Enumeration', ns)
                if enum is not None:
                    ref = enum.find('Ref', ns) # Ref is usually in base NS or just Ref tag? No, Ref is distinct.
                    # In SDMX 2.1 it's com:Ref usually, or Ref inside Enumeration
                    # Let's try finding any 'Ref' child
                    # Actually, often it is <Ref id="..." />
                    if ref is None:
                        # Try finding namespaced Ref
                        ref = enum.find('Ref') or enum.find('str:Ref', ns) or enum.find('com:Ref', ns)
                    
                    if ref is not None:
                        enum_ref_id = ref.get('id')

            print(f"Dim: {did} -> Codelist: {enum_ref_id}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
