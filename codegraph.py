import ast
import os

def get_file_info(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        tree = ast.parse(content)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    imports.append(alias.name)
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    for alias in n.names:
                        imports.append(f"{n.module}.{alias.name}")
        return {'classes': classes, 'functions': functions, 'imports': imports}
    except Exception as e:
        return {'classes': [], 'functions': [], 'imports': [], 'error': str(e)}

def get_js_functions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    import re
    # Find function declarations and arrow functions
    functions = re.findall(r'function\s+(\w+)|const\s+(\w+)\s*=\s*\(|let\s+(\w+)\s*=\s*\(|async\s+function\s+(\w+)', content)
    return [f for group in functions for f in group if f]

# Python files
py_files = ['server.py', 'app.py', 'check_html.py', 'check_html2.py', 'check_js.py', 'final_check.py', 'test_api.py', 'test_api2.py', 'test_final.py', 'test_pdf_api.py']
for f in py_files:
    if os.path.exists(f):
        info = get_file_info(f)
        print(f'\n=== {f} ===')
        if info['classes']:
            print(f'  Classes: {info["classes"]}')
        if info['functions']:
            print(f'  Functions: {info["functions"][:30]}' + ('...' if len(info['functions']) > 30 else ''))
        if info['imports']:
            print(f'  Imports: {info["imports"][:20]}' + ('...' if len(info['imports']) > 20 else ''))

# JS files
js_files = ['app.js']
for f in js_files:
    if os.path.exists(f):
        funcs = get_js_functions(f)
        print(f'\n=== {f} ===')
        print(f'  Functions: {funcs[:50]}' + ('...' if len(funcs) > 50 else ''))