"""
config.py — Constantes, patrones de ruido y configuración global.

Extraído de server.py para mantener el archivo principal más ligero.
Todas las constantes se importan desde aquí a los módulos que las necesitan.
"""

import os
import re
import sys
from pathlib import Path
from typing import Final


# ─── Paths (compatible with PyInstaller frozen mode) ──────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    ROOT: Path = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parent
DIST: Final[Path] = ROOT / "dist"
IS_PRODUCTION: bool = DIST.exists() and (DIST / "index.html").exists()

# ─── Server constants ────────────────────────────────────────────
MAX_WORKERS: Final[int] = min(8, (os.cpu_count() or 4))
REQUEST_TIMEOUT: Final[int] = 20  # seconds for external API calls
MAX_IMAGE_DIMENSION: Final[int] = 4096  # max width/height for OCR processing
APP_VERSION: Final[str] = "20260715"


# ─── Timeouts (single source of truth — shared via /api/config) ────
TIMEOUT_OPENCV_INIT_MS: Final[int] = 15000       # app.js: OpenCV init + poll timeout
TIMEOUT_PDFJS_CDN_MS: Final[int] = 10000          # app.js: PDF.js CDN load (UMD)
TIMEOUT_PDFJS_ES_MODULE_MS: Final[int] = 10000    # app.js: PDF.js ES module import
TIMEOUT_PDF_RENDER_MS: Final[int] = 60000          # app.js: PDF page render (60s para PDFs escaneados pesados)
TIMEOUT_TRANSLATE_MS: Final[int] = 30000           # app.js: translate single request
TIMEOUT_TRANSLATE_BATCH_MS: Final[int] = 60000     # app.js: translate batch request
TIMEOUT_PROCESS_PAGE_MS: Final[int] = 120000       # app.js: server process-page
TIMEOUT_INPAINTED_IMAGE_MS: Final[int] = 15000     # app.js: inpainted image decode
TIMEOUT_EXPORT_REVOKE_MS: Final[int] = 10000       # app.js: export URL.revokeObjectURL
TIMEOUT_CDN_LOAD_MS: Final[int] = 8000             # index.html: __loadCdn default

LANGUAGES: Final[dict[str, str]] = {
    "es": "spanish", "en": "english", "pt": "portuguese",
    "fr": "french",  "de": "german",  "it": "italian",
    "ja": "japanese","ko": "korean",  "zh": "chinese (simplified)",
    "zh-cn": "chinese (simplified)", "zh-tw": "chinese (traditional)",
    "auto": "auto",
}


# ─── Diccionario de corrección PRE-OCR (Errores comunes de OCR) ──
# Versión raw (strings de regex) para compatibilidad hacia atrás.
GLOSARIO_PRE: Final[dict[str, str]] = {
    r"\belscon\b": "el scan",
    r"\bIo web\b": "la web",
    r"\bJPuede\b": "¿Puede",
    r"\bSIEMPPE\b": "SIEMPRE",
    r"\bEMPPEDECIBLE\b": "IMPREDECIBLE",
    r"\bPELACIONADO\b": "RELACIONADO",
    r"\bTRAJCÓN\b": "TRAICIÓN",
    r"\bTRAIQÓN\b": "TRAICIÓN",  # OCR: 'C' misread as 'Q'
    r"\b@NCO\b": "CINCO",
    r"\bLaaYUDa\b": "La ayuda",
    r"\bCavBrE\b": "Cabre",
    r"\bHANTENÍA\b": "MANTENÍA",
    r"\bNEŒSITABA\b": "NECESITABA",
    r"\bNEŒSTO\b": "NECESITO",
    r"\bMucha\s+DOdaS\b": "Muchas dudas",
    r"\bUn\s+Scomumano\b": "Un ser humano",
    r"\bcnar\b": "criar",  # OCR confusion: 'ri' misread as 'n'
    r"\bccrrettimerte\b": "correctamente",  # OCR: stylized font compression
    r"\baiar\b": "criar",  # OCR: 'ri' misread as 'ai'
    r"\bMUESIRA\b": "MUESTRA",  # OCR: 'T' misread as 'I'
    r"\bPaDRiNO\b": "PADRINO",  # OCR: mixed case
    r"\bScomumano\b": "ser humano",  # OCR: fused words
    r"\bConFguroción\b": "Configuración",  # OCR: 'nfig' misread as 'Fgur'
    r"\bconfguroción\b": "configuración",  # OCR: lowercase variant
    r"\b0\b": "a",  # OCR: standalone 'a' misread as digit '0'
    r"\bmuesira\b": "muestra",  # OCR: 'T' misread as 'I' (lowercase)
    r"\bScomunano\b": "ser humano",  # OCR: fusion n-variant
    r"\bshInel\b": "Shinel",  # OCR: username handle mixed case
    r"\b@\b": "",              # OCR: @ solitario (ruido de escaneo)
    r"@": "",                  # OCR: @ en cualquier posición (ruido decorativo)
}

# Versión pre-compilada (evita re.compile() implícito en cada llamada)
GLOSARIO_REGEX: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in GLOSARIO_PRE.items()
]


# ─── Patrones de ruido en márgenes (fecha/hora/numeración) ───────
MARGIN_NOISE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}'),
    re.compile(r'\d{1,2}[/.\-]\d{1,2}1?\d{2}\b'),
    re.compile(r'\d{1,2}:\d{2}\s*([ap]\.?\s?m\.?)?', re.IGNORECASE),
    re.compile(r'^\d{1,4}\s*/\s*\d{1,4}$'),
    re.compile(r'\b\d{1,4}\s+de\s+\d{1,4}\b', re.IGNORECASE),
    re.compile(r'\bp[aá]g(?:ina)?\.?\s?\d{1,4}\b', re.IGNORECASE),
    re.compile(r'\b(?:p[aá]g(?:ina)?|page)\s+\d+\s+(?:de|of)\s+\d+\b', re.IGNORECASE),
    # Metadatos de exportación y timestamps en márgenes (ej. 20260713-11032519C, 13726, 458pm)
    re.compile(r'\b\d{8,14}[A-Za-z0-9_\-]*\b'),
    re.compile(r'\b\d{3,6}\s*,?\s*\d{3,4}\s*p\.?m\.?\b', re.IGNORECASE),
    re.compile(r'\b\d{3,4}\s*p\.?m\.?\b', re.IGNORECASE),
]

# Patrones de marcas de agua globales (sellos de grupos de escaneo)
# NOTA: Nombres de grupos de scanlation ("olympus", "scanlation") NO se incluyen
# porque aparecen legítimamente en títulos de capítulo. Solo se filtran patrones
# específicos de sellos/watermarks.
WATERMARK_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r'zonaolympus[\s-]?com', re.IGNORECASE),
    re.compile(r'\b1\s*[\s-]?c\s*[\s-]?2\s*[\s-]?e\b', re.IGNORECASE),
    # Broken "http://" — OCR mangles "https://" into "htps fo", "htp ://", "htpsjj" etc.
    re.compile(r'\bhtps?\s*[:\s/\'"\\\\]', re.IGNORECASE),
    # Domain with underscore/apostrophe instead of dot before TLD: "xyz_com", "site'com"
    re.compile(r'[a-z]+[_\'"\s]\s*(?:com|net|org|xyz|io)\b', re.IGNORECASE),
]


# ─── Glosario post-traducción para corregir salidas literales de CT2 ──
# CT2 (OPUS-MT) tiende a traducciones literales. Este glosario aplica
# correcciones específicas para términos comunes de manga.
# Formato: (patron_regex, reemplazo) — aplicado con re.IGNORECASE
GLOSARIO_POST: Final[list[tuple[str, str]]] = [
    # Términos de capítulos/episodios
    (r"\bTEMPORARY\s+(\d+)\b", r"SEASON \1"),          # TEMPORADA 7 → SEASON 7
    (r"\bTEMPORARILY\s+(\d+)\b", r"SEASON \1"),         # variante
    # Términos de scanlation
    (r"\bSCAN\b", r"scan"),                                # normalizar mayúsculas
    (r"\bSCANLATION\b", r"scanlation"),
    # Términos de configuración/página
    (r"\bCONFIGURATION\b", r"Settings"),
    (r"\bPAGE\b", r"page"),
]


# ─── Security headers (CSP, Brave Leo opt-out) ───────────────────
CSP_POLICY: Final[str] = (
    "default-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com "
    "https://fonts.googleapis.com https://fonts.gstatic.com "
    "data: blob:; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
    "worker-src 'self' blob: https://cdnjs.cloudflare.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' http://127.0.0.1:5174 https://cdnjs.cloudflare.com https://cdn.jsdelivr.net data:;"
    "frame-ancestors 'none'; "
    "form-action 'self';"
)
