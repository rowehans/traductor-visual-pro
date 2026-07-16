import requests
import base64
import fitz  # PyMuPDF

# Create a simple test PDF with text
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 100), "Hola Mundo - Test PDF", fontsize=24)
page.insert_text((50, 150), "Esto es una prueba de traducción", fontsize=18)
page.insert_text((50, 200), "KONO SUBARASHII SEKAI", fontsize=16)

pdf_bytes = doc.tobytes()
doc.close()

# Encode to base64
b64 = base64.b64encode(pdf_bytes).decode()
data_url = 'data:application/pdf;base64,' + b64

print('Testing PDF upload and translation...')
resp = requests.post('http://127.0.0.1:5174/api/process-page', 
                    json={'image': data_url, 'target': 'es', 'source': 'auto'},
                    timeout=120)
print('Status:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    print('Blocks:', len(data.get('blocks', [])))
    for b in data.get('blocks', []):
        print(f'  Source: {b["source"]} -> Translated: {b["translated"]}')
else:
    print('Error:', resp.text[:500])