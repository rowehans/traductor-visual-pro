import re
with open(r'D:\crear traductor\dist\app.min.js', 'r', encoding='utf-8') as f:
    code = f.read()
lines = code.split('\n')
for i, line in enumerate(lines, 1):
    for m in re.finditer(r'\bmodule\b', line):
        before = line[:m.start()]
        if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
            print(f'Line {i+1}: {line[:200]}')