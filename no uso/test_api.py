import requests
import base64
import cv2
import numpy as np

# Create test image
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.putText(img, 'TEST MANGA', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
_, buf = cv2.imencode('.png', img)
b64 = base64.b64encode(buf).decode()
data_url = 'data:image/png;base64,' + b64

print('Testing /api/process-page...')
resp = requests.post('http://127.0.0.1:5174/api/process-page', 
                    json={'image': data_url, 'target': 'es', 'source': 'ja'},
                    timeout=120)
print('Status:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    print('Blocks:', len(data.get('blocks', [])))
    for b in data.get('blocks', []):
        print('  Source:', b['source'], '-> Translated:', b['translated'])
else:
    print('Error:', resp.text[:200])