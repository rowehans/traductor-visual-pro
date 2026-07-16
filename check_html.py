import re
with open(r'D:\crear traductor\dist\index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Check for type=module
for m in re.finditer(r'type\s*=\s*["\']module["\']', html):
    print(f'type=module at {m.start()}: {html[max(0,m.start()-50):m.end()+50]}')

# Check for module keyword in script tags
for m in re.finditer(r'<script\b[^>]*>', html):
    if 'module' in m.group():
        print(f'module in script tag at {m.start()}: {m.group()}')