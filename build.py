"""
build.py — Script de build para produccion.
Minifica JS/CSS, deshabilita source maps, genera version optimizada.
"""
import os, re, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "dist")

def minify_js():
    src = os.path.join(ROOT, "app.js")
    dst = os.path.join(DIST, "app.min.js")
    os.makedirs(DIST, exist_ok=True)
    print(f"[build] Copiando JS sin minificar (preserva ES6) {src} -> {dst}")
    shutil.copy2(src, dst)
    size_kb = os.path.getsize(dst) / 1024
    print(f"[build] JS copiado: {size_kb:.1f} KB")

def minify_css():
    src = os.path.join(ROOT, "styles.css")
    dst = os.path.join(DIST, "styles.min.css")
    print(f"[build] Minificando {src} -> {dst}")
    with open(src, "r", encoding="utf-8") as f:
        data = f.read()
    from rcssmin import cssmin
    minified = cssmin(data)
    os.makedirs(DIST, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(minified)
    size_kb = os.path.getsize(dst) / 1024
    print(f"[build] CSS minificado: {size_kb:.1f} KB")

def copy_index():
    src = os.path.join(ROOT, "index.html")
    dst = os.path.join(DIST, "index.html")
    with open(src, "r", encoding="utf-8") as f:
        html = f.read()
    # Reemplazar referencias a archivos fuente por minificados
    html = html.replace("styles.css", "styles.min.css")
    html = html.replace("app.js", "app.min.js")
    os.makedirs(DIST, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[build] index.html copiado con referencias minificadas")

def copy_assets():
    """Copia recursos estaticos, incluidos los modulos ES importados por app."""
    assets_dir = os.path.join(ROOT, "assets")
    if os.path.isdir(assets_dir):
        dst = os.path.join(DIST, "assets")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(assets_dir, dst)
        print(f"[build] Assets copiados")

    # app.min.js conserva los imports ES6 de ./js/*.js. Si esta carpeta falta,
    # el navegador carga el entrypoint pero falla antes de inicializar la UI.
    js_dir = os.path.join(ROOT, "js")
    if os.path.isdir(js_dir):
        dst = os.path.join(DIST, "js")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(js_dir, dst)
        print(f"[build] Modulos JS copiados")

    icon_path = os.path.join(ROOT, "icon.ico")
    if os.path.isfile(icon_path):
        shutil.copy2(icon_path, os.path.join(DIST, "icon.ico"))
        print(f"[build] Icono copiado")

if __name__ == "__main__":
    print("[build] === INICIANDO BUILD ===")
    minify_js()
    minify_css()
    copy_index()
    copy_assets()
    print("[build] === BUILD COMPLETADO ===")