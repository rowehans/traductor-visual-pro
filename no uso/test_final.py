import requests
import base64
import cv2
import numpy as np

img = np.zeros((600, 800, 3), dtype=np.uint8)
cv2.putText(img, 'KONO SUBARASHII', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
cv2.putText(img, 'SEKAI NI SHUKUFUKU WO', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3)
cv2.putText(img, 'CHAPTER 1', (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

_, buf = cv2.imencode('.png', img)
b64 = base64.b64encode(buf).decode()
data_url = 'data:image/png;base64,' + b64

print('Testing /api/process-page with Japanese-style text...')
resp = requests.post('http://127.0.0.1:5174/api/process-page', 
                    json={'image': data_url, 'target': 'es', 'source': 'ja'},
                    timeout=120)
print('Status:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    print('Blocks:', len(data['blocks']))
    for b in data['blocks']:
        print('  Source:', b['source'], '-> Translated:', b['translated'])
else:
    print('Error:', resp.text[:200])