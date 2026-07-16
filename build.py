"""
build.py — Script de build para produccion.
Minifica JS/CSS, deshabilita source maps, genera version optimizada.
"""
import os, re, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "dist")

def minify_js():
    import os
    import re
    import shutil
    from jsmin import jsmin

    src = os.path.join(ROOT, "app.js")
    dst = os.path.join(DIST, "app.min.js")

    print(f"[build] Minificando {src} -> {dst}")

    # 1. Asegurar la existencia del directorio antes de escribir
    os.makedirs(DIST, exist_ok=True)

    try:
        with open(src, "r", encoding="utf-8") as f:
            data = f.read()

        # Eliminar comentarios // y /* */
        data = re.sub(r'//.*$', '', data, flags=re.MULTILINE)
        data = re.sub(r'/\*.*?\*/', '', data, flags=re.DOTALL)
        # Eliminar console.log
        data = re.sub(r'console\.log\(.*?\);?', '', data)

        # Minificar código
        minified = jsmin(data)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(minified)

        size_kb = os.path.getsize(dst) / 1024
        print(f"[build] JS minificado: {size_kb:.1f} KB")

    except Exception as e:
        print(f"[build] Error en minificación, aplicando copia de respaldo: {e}")
        # Copia de seguridad si la minificación falla por sintaxis ES6+
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"[build] JS copiado (sin minificar): {size_kb:.1f} KB")

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
    """Copia assets estaticos que no se minifican (imagenes, fuentes, etc.)"""
    assets_dir = os.path.join(ROOT, "assets")
    if os.path.isdir(assets_dir):
        dst = os.path.join(DIST, "assets")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(assets_dir, dst)
        print(f"[build] Assets copiados")

if __name__ == "__main__":
    print("[build] === INICIANDO BUILD ===")
    minify_js()
    minify_css()
    copy_index()
    copy_assets()
    print("[build] === BUILD COMPLETADO ===")