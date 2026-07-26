# Traductor Visual Pro

<p align="center">
  <a href="https://github.com/rowehans/traductor-visual-pro/releases/latest">
    <img src="https://img.shields.io/github/v/release/rowehans/traductor-visual-pro?label=%C3%9Altima%20versi%C3%B3n&color=brightgreen" alt="Latest release">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/releases/tag/v0.1.37">
    <img src="https://img.shields.io/badge/Descargar-Instalador%20(373%20MB)-blue?logo=windows" alt="Download">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/releases">
    <img src="https://img.shields.io/github/downloads/rowehans/traductor-visual-pro/total?label=Descargas&color=success&logo=download" alt="Total downloads">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/rowehans/traductor-visual-pro/ci.yml?label=CI&logo=github&color=blueviolet" alt="CI status">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/actions/workflows/release.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/rowehans/traductor-visual-pro/release.yml?label=Build&logo=githubactions&color=purple" alt="Build status">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro">
    <img src="https://img.shields.io/github/stars/rowehans/traductor-visual-pro?style=social" alt="Stars">
  </a>
</p>

App local para traducir manga, cómics y documentos en PDF e imagen. Backend Python con EasyOCR + OpenCV + ArgosTranslate, frontend JavaScript con canvas interactivo y editor de burbujas.

---

## Descargar

**Opción recomendada — Instalador profesional:**

[![Download Installer](https://img.shields.io/badge/Descargar_TraductorVisual_Setup_20260725.exe-373MB-blue?style=for-the-badge&logo=windows)](https://github.com/rowehans/traductor-visual-pro/releases/download/v0.1.37/TraductorVisual_Setup_20260725.exe)

El instalador incluye:
- Ejecutable `main.exe` (200 MB)
- Modelo CTD (ComicTextDetector) para detección de texto artístico
- Script `setup.ps1` para configuración automatizada del entorno
- Descarga bajo demanda de modelos OCR y CT2 durante la instalación

**Requisitos:** Windows 10/11 (64-bit), Python 3.8+, 8 GB RAM, GPU NVIDIA con CUDA (opcional)

---

## Iniciar la app

**Opción 1 — Script PowerShell:**
```powershell
.\start-app.ps1
```

**Opción 2 — Directo:**
```powershell
.\env\Scripts\python.exe server.py
```

La app abre en: http://127.0.0.1:5174/

## Arquitectura

```
server.py         → Entry point Flask (puerto 5174)
config.py         → Constantes, patrones de ruido, CSP
translator.py     → Detección de idioma + traducción (Argos → Google → CT2)
ocr_utils.py      → OCR EasyOCR + inpainting OpenCV + filtros
routes/api.py     → Endpoints REST
routes/main.py    → Rutas estáticas
app.js            → Frontend canvas + UI (~2514 líneas)
index.html        → Estructura HTML
styles.css        → Estilos premium dark/light
cache.py          → Caché de traducciones filesystem
models.py         → Modelos SQLAlchemy
ratelimit.py      → Rate limiting
```

## Funciones principales

- **Carga PDF, PNG, JPG, WebP** — renderizado vía pdf.js (ESM v4.10 con fallback UMD v3.11)
- **OCR automático** — EasyOCR en servidor, GPU→CPU fallback
- **Traducción** — ArgosTranslate (offline) → Google Translate (online) → CT2 OPUS-MT es→en
- **Detección de idioma** — langdetect + heurística española (acentos, verbos, diccionario)
- **Filtros de ruido** — 9 filtros post-OCR + 2 pre-OCR (morfología OpenCV)
- **Inpainting** — OpenCV INPAINT_NS con máscara adaptativa (glifos vs rectángulo)
- **Editor de burbujas** — dibujar, mover, redimensionar, estilizar texto
- **Exportación** — PNG individual, PDF página actual, PDF completo
- **Temas** — oscuro/claro con toggle, persistencia en localStorage
- **Atajos de teclado** — D/V modos, Ctrl+T traducir, Ctrl+E/P exportar, flechas navegar

## Requisitos

- Python 3.10+
- Entorno virtual en `env/` con todas las dependencias
- OpenCV.js carga desde CDN jsDelivr (con callback nativo `onRuntimeInitialized`)
- Conexión a internet para primera descarga de modelos EasyOCR y ArgosTranslate

## Notas técnicas

- **OpenCV.js**: No más polling. Usa `cv['onRuntimeInitialized']` con 3 casos de carga y timeout 15s.
- **Caché**: Traducciones repetidas se cachean en `cache/translations/` por 7 días (5000 entradas máx).
- **Rate limiting**: 200 req/día, 50 req/hora global. Endpoints sensibles: 30/min (translate), 20/min (batch), 5/min (process-page).
- **CSP**: Política estricta inyectada en todas las respuestas. Permite CDNs específicos.
- **Seguridad**: Protección contra path traversal en rutas estáticas.

## Tests

```powershell
.\env\Scripts\python.exe test_ci.py
```

## CI

- **Pre-commit hook**: Syntax check + test_ci.py en archivos modificados
- **GitHub Actions**: Syntax check + tests en push/PR con cambios en server.py o routes/
- **CI local completa**: `.\run_ci.ps1` (rápido) o `.\run_ci.ps1 -Full` (con stress test)
