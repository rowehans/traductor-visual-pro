import requests

r = requests.get('http://127.0.0.1:5174/', timeout=5)
print('=== FINAL VERIFICATION ===')
print('Status:', r.status_code)
print('HTML size:', len(r.text), 'chars')

checks = [
    ('noai meta', 'noai' in r.text.lower()),
    ('noimageai meta', 'noimageai' in r.text.lower()),
    ('CSP meta', 'Content-Security-Policy' in r.text),
    ('Brave detection', 'brave' in r.text.lower()),
    ('Theme toggle', 'initTheme' in r.text),
    ('Toast system', 'showToast' in r.text),
    ('Keyboard shortcuts', 'initKeyboardShortcuts' in r.text),
    ('Error interceptors', 'Cannot read' in r.text and 'image.png' in r.text),
]

print()
for name, result in checks:
    status = 'OK' if result else 'MISSING'
    print('  ' + status + ': ' + name)

print()
print('Security headers:')
r = requests.get('http://127.0.0.1:5174/', timeout=5)
for h in ['Content-Security-Policy', 'X-Content-Type-Options', 'X-Frame-Options', 'Permissions-Policy']:
    val = r.headers.get(h)
    if val:
        print('  ' + h + ': PRESENT - ' + val[:80])
    else:
        print('  ' + h + ': MISSING')