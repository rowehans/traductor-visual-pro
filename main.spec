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
    # Módulos JS (ES module imports desde app.js: ./js/config.js, utils.js, etc.)
    # ⚠️ CRÍTICO: Sin esta línea, los imports del módulo ES fallan con 404,
    # app.js nunca ejecuta, initOpenCv() nunca se llama, y el badge se queda
    # en "Cargando OpenCV..." para siempre.
    (str(PROJECT_ROOT / "js"), "js"),

    # Módulos Python (desde la raíz)
    (str(PROJECT_ROOT / "server.py"), "."),
    (str(PROJECT_ROOT / "config.py"), "."),
    (str(PROJECT_ROOT / "translator.py"), "."),
    (str(PROJECT_ROOT / "ocr_utils.py"), "."),
    (str(PROJECT_ROOT / "ocr_engine.py"), "."),
    (str(PROJECT_ROOT / "runtime_diagnostics.py"), "."),
    (str(PROJECT_ROOT / "translation_memory.py"), "."),
    (str(PROJECT_ROOT / "cache.py"), "."),
    (str(PROJECT_ROOT / "models.py"), "."),
    (str(PROJECT_ROOT / "ratelimit.py"), "."),
    # Daemon U-OCR: uocr_client.py se importa dinámicamente desde routes/api.py
    # y server.py; uocr_daemon.py se LANZA como subproceso (venv env_uocr_gpu),
    # así que ambos deben existir como archivos junto al .exe.
    (str(PROJECT_ROOT / "uocr_client.py"), "."),
    (str(PROJECT_ROOT / "uocr_daemon.py"), "."),
    # Paquete routes/ (blueprints Flask)
    (str(PROJECT_ROOT / "routes"), "routes"),

    # Icono del ejecutable
    (str(PROJECT_ROOT / "icon.ico"), "."),
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
    # NOTA: ocr_engine.py / uocr_client.py / uocr_daemon.py NO van en hiddenimports.
    # Se importan DINÁMICAMENTE (dentro de funciones) desde routes/api.py y server.py,
    # y viajan en DATAS como .py (como ocr_utils.py). Si se añaden aquí, PyInstaller
    # analiza sus imports estáticos (ocr_engine -> ocr_utils -> ultralytics/torch)
    # y el .exe crece de ~360MB a ~2.4GB (torch CUDA, paddle, polars, spacy, ...).
    # Image processing
    "cv2",
    "PIL",
    "PIL.Image",
    "numpy",
    # Network / utils
    "requests",
    "urllib3",
    "psutil",
    # Stdlib module that PyInstaller may miss — pickletools es requerido por EasyOCR
    # pero PyInstaller no lo detecta automaticamente porque se importa dinamicamente.
    "pickletools",
    # unittest.mock es importado por torch (env/Lib/site-packages/torch/utils/_config_module.py)
    # al cargarse desde env/ en runtime. En modo frozen, el stdlib vive en base_library.zip
    # y PyInstaller no incluye unittest por defecto -> "No module named 'unittest.mock'"
    # rompe la carga de EasyOCR/CT2/YOLO. Hay que empaquetarlo explícitamente.
    "unittest",
    "unittest.mock",
    # Más stdlib que se importa desde env/ en runtime y falta en base_library.zip:
    #  - modulefinder: lo importa torch (carga EasyOCR) -> "No module named 'modulefinder'"
    #  - plistlib: lo importa ultralytics/utils/logger.py (carga YOLO)
    #  - tqdm.auto: huggingface_hub (desde env/) importa tqdm.auto; el tqdm del
    #    bundle no incluye el submodulo auto -> "No module named 'tqdm.auto'"
    "modulefinder",
    "plistlib",
    # filecmp/shelve: stdlib que transformers importa y no viaja en base_library.zip
    # (transformers.models.auto.tokenization_auto -> "No module named 'filecmp'")
    "filecmp",
    "shelve",
    # PIL.ImageEnhance no lo recoge el hook de PIL (falta en PYZ) y EasyOCR lo
    # importa al cargarse desde env/ -> "cannot import name 'ImageEnhance' from 'PIL'"
    "PIL.ImageEnhance",
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
    # ── ML pesado que se carga desde env/ en runtime (vía _fix_cwd) ──
    # Si algo del grafo los importa (ej. ocr_utils -> ultralytics), excluirlos
    # evita que PyInstaller meta torch CUDA / paddle / polars / spacy en el .exe.
    "ultralytics", "paddle", "polars", "spacy", "thinc", "blis",
    "torchvision", "timm", "onnx", "onnxruntime", "rapidocr_onnxruntime",
    "accelerate", "peft", "functorch", "thop", "opt_einsum", "stanza",
    "safetensors",
    # tqdm se excluye del bundle: huggingface_hub (desde env/) importa
    # submodulos tqdm.auto/tqdm.contrib.* y el bundle parcial rompe la carga
    # ("No module named 'tqdm.contrib.concurrent'"). Se carga completo desde env/.
    "tqdm",
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
    icon=str(PROJECT_ROOT / "icon.ico"),
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
