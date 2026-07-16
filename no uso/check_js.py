import requests
r = requests.get('http://127.0.0.1:5174/app.js?v=20260714', timeout=5)
content = r.text
checks = [
    ('initTheme', 'initTheme' in content),
    ('toggleTheme', 'toggleTheme' in content),
    ('showToast', 'showToast' in content),
    ('initKeyboardShortcuts', 'initKeyboardShortcuts' in content),
    ('themeToggle button creation', 'themeToggle' in content),
    ('toastContainer creation', 'toastContainer' in content),
    ('keyboard shortcuts for T', 'autoTranslateCurrentPage' in content and 'autoTranslateAllPages' in content),
    ('keyboard shortcuts for arrow keys', 'arrowleft' in content.lower() or 'arrowright' in content.lower()),
]
for name, result in checks:
    print(f'{name}: {"OK" if result else "MISSING"}')