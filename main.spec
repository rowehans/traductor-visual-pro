# -*- mode: python ; coding: utf-8 -*-
"""
main.spec — PyInstaller spec para Traductor Visual Pro.

El .exe empaqueta solo el código fuente Python y el frontend.
Las dependencias pesadas (EasyOCR, torch, ArgosTranslate, etc.)
se cargan desde el entorno virtual 'env/' junto al proyecto.
"""

import os
import sys
from pathlib import Path

# ─── Raíz del proyecto ─────────────────────────────────
PROJECT_ROOT = Path(r"D:\crear traductor")
DIST = PROJECT_ROOT / "dist"

# ─── Datos a incluir en el bundle ──────────────────────
# Formato: (ruta_origen, destino_en_bundle)
DATAS = [
    # Frontend (desde la raíz del proyecto)
    (str(PROJECT_ROOT / "index.html"), "."),
    (str(PROJECT_ROOT / "app.js"), "."),
    (str(PROJECT_ROOT / "styles.css"), "."),

    # Módulos Python (desde la raíz)
    (str(PROJECT_ROOT / "server.py"), "."),
    (str(PROJECT_ROOT / "config.py"), "."),
    (str(PROJECT_ROOT / "translator.py"), "."),
    (str(PROJECT_ROOT / "ocr_utils.py"), "."),
    (str(PROJECT_ROOT / "ocr_ctd_fallback.py"), "."),
    (str(PROJECT_ROOT / "cache.py"), "."),
    (str(PROJECT_ROOT / "models.py"), "."),
    (str(PROJECT_ROOT / "ratelimit.py"), "."),
    # Paquete routes/ (blueprints Flask)
    (str(PROJECT_ROOT / "routes"), "routes"),

    # Módulos CTD (detección de texto artístico, ~100KB)
    (str(PROJECT_ROOT / "ctd_lib"), "ctd_lib"),

    # Modelo CTD (ComicTextDetector) — descargado ya en local
    (str(PROJECT_ROOT / "models" / "ctd"), "models/ctd"),
]

# ─── Hidden imports (módulos que PyInstaller no detecta) ─
# Los imports dinámicos (dentro de funciones) no se detectan automáticamente.
# Los módulos ligeros van en el bundle; los pesados (torch, easyocr)
# se cargan desde env/ en tiempo de ejecución vía _fix_cwd() en main.py.
HIDDEN_IMPORTS = [
    # Web framework
    "flask",
    "flask_sqlalchemy",
    "flask_limiter",
    # Database
    "sqlalchemy",
    # Translation — Argos
    "argostranslate",
    "argostranslate.package",
    "argostranslate.translate",
    "deep_translator",
    "langdetect",
    # CTranslate2, transformers, sentencepiece, huggingface_hub NO se empaquetan.
    # Se cargan desde env/Lib/site-packages en runtime vía _fix_cwd() en main.py.
    # Esto reduce el .exe de ~2.6GB a ~200MB y acelera el arranque de 25s a ~3s.
    # Image processing
    "cv2",
    "PIL",
    "PIL.Image",
    "numpy",
    # CTD dependencies (imports condicionales, PyInstaller no los detecta)
    "pyclipper",
    "shapely",
    "einops",
    # Network / utils
    "requests",
    "urllib3",
    "psutil",
    "tqdm",
]

# ─── Excludes (reducir tamaño) ──────────────────────────
EXCLUDES = [
    # ── Módulos pesados que se cargan desde env/ en runtime ──
    # torch (1.2GB CUDA), transformers (500MB), easyocr (200MB),
    # ctranslate2 (300MB CUDA), sentencepiece, huggingface_hub.
    # Todos se importan dinámicamente dentro de funciones:
    #   - translator.py: _get_ct2_translator() -> import ctranslate2, torch, transformers
    #   - ocr_utils.py: _get_ocr_reader() -> import easyocr, torch
    # En runtime, _fix_cwd() en main.py agrega env/Lib/site-packages a sys.path.
    "torch",
    "easyocr",
    "ctranslate2",
    "transformers",
    "sentencepiece",
    "huggingface_hub",
    # ── Dependencies no críticas ──
    "tkinter", "matplotlib", "pandas", "scipy",
    "notebook", "jupyter", "IPython",
    "pytest", "sphinx", "docutils",
    "setuptools", "pip", "wheel",
]

block_cipher = None

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="main",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX deshabilitado: muy lento con modelos grandes
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI mode — sin ventana CMD
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,          # UPX deshabilitado: muy lento, poco beneficio en bundles modernos
    name="main",
)
