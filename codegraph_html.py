import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract IDs
ids = re.findall(r'id="([^"]+)"', content)
print('=== index.html ===')
print(f'  IDs ({len(ids)}): {ids}')

# Extract scripts
scripts = re.findall(r'<script[^>]*src="([^"]+)"', content)
print(f'  Scripts: {scripts}')

# Extract CSS links
css = re.findall(r'<link[^>]*href="([^"]+)"[^>]*rel="stylesheet"', content)
print(f'  Stylesheets: {css}')

# Meta tags
meta = re.findall(r'<meta[^>]*>', content)
print(f'  Meta tags: {len(meta)}')