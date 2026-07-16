import requests
import re

r = requests.get('http://127.0.0.1:5174/')
html = r.text

# Check for service worker
if 'serviceWorker' in html or 'navigator.serviceWorker' in html:
    print('FOUND service worker registration')
else:
    print('NO service worker registration')

# Check for image.png references
png_refs = re.findall(r'[\'"][^\'"]*image\.png[^\'"]*[\'"]', html)
if png_refs:
    print('Found image.png references:', png_refs)
else:
    print('NO image.png references in HTML')

# Check for manifest
if 'manifest' in html.lower():
    print('Found manifest reference')
else:
    print('NO manifest')

# Check all external resources
resources = re.findall(r'(?:src|href)=[\'"]([^\'"]+)[\'"]', html)
print('\nAll external resources:')
for res in resources:
    print('  ', res)