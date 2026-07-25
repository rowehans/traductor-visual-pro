# Traductor Visual Pro

App local para traducir manga, cómics y documentos en PDF e imagen. Backend Python con EasyOCR + OpenCV + ArgosTranslate, frontend JavaScript con canvas interactivo y editor de burbujas.

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
