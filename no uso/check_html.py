import requests
import re

r = requests.get('http://127.0.0.1:5174/')
html = r.text

# Find all script src
scripts = re.findall(r'<script[^>]*src=["\']([^"\']*)["\']', html)
print('Script sources found:')
for s in scripts:
    print('  ', s)

# Check for tesseract anywhere
if 'tesseract' in html.lower():
    print('FOUND tesseract in HTML!')
else:
    print('NO tesseract in HTML')

# Check inline scripts
inline = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
for i, scr in enumerate(inline):
    if 'tesseract' in scr.lower() or 'createWorker' in scr or 'ocrCanvas' in scr:
        print(f'INLINE script {i} has tesseract/ocr:', scr[:200])

print('HTML length:', len(html))