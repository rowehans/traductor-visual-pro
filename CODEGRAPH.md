# CODEGRAPH — Traductor Visual Pro

## Arquitectura modular (post-refactor Julio 2026)

```
┌─────────────────────────────────────────────────────────┐
│                      server.py                          │
│  Entry point Flask (port 5174, ~217 líneas)             │
│  ┌─ Flask app, static routes, DB init, cache init       │
│  ├─ **CT2 preload en background** (hilo daemon):        │
│  │   precarga modelos es→en y en→es al arrancar.       │
│  │   Evita cold start de ~21.5s en 1ra traducción.     │
│  └─ _translate_one wrapper (inyecta cache en módulo)    │
├─────────────────────────────────────────────────────────┤
│                     config.py                            │
│  ┌─ ROOT, DIST, IS_PRODUCTION, APP_VERSION              │
│  ├─ LANGUAGES, MAX_WORKERS, REQUEST_TIMEOUT             │
│  ├─ CSP_POLICY, GLOSARIO_PRE                            │
│  └─ MARGIN_NOISE_PATTERNS, WATERMARK_PATTERNS           │
├─────────────────────────────────────────────────────────┤
│                    translator.py                         │
│  ┌─ _detect_language_simple / _robust (ES/JA/KO/ZH)    │
│  ├─ _translate_one → 3 motores en PARALELO              │
│  │    ┌─ CT2 (CTranslate2 int8, ~53ms, 10 pares)        │
│  │    ├─ Argos (~2.8s, lock global, 100% cobertura)      │
│  │    └─ Google (~1.1s, HTTP pool singleton)             │
│  │    **Executor compartido** (4 threads, lazy init):   │
│  │    _get_translate_engine_executor() con double-check │
│  │    locking. Antes: ThreadPoolExecutor nuevo por      │
│  │    llamada (~1.857 ciclos para 619 bloques). Ahora:  │
│  │    threads siempre vivos, sin shutdown, retorno      │
│  │    inmediato. Tiempo efectivo: ~53ms (el más rápido) │
│  │    ├─ Fallback: si TODOS fallan → Google retry con    │
│  │    │   backoff progresivo (5s, 15s, 30s). Resetea     │
│  │    │   rate limit entre reintentos (fix SIN_TRAD).    │
│  ├─ _es_ocr_noise (detecta OCR ruidoso, salta Argos)    │
│  ├─ _es_traduccion_valida (6 validaciones anti-basura)  │
│  ├─ _corregir_ct2 (glosario POST, 6 reglas)             │
│  └─ _aplicar_glosario (correccion PRE-OCR)              │
├─────────────────────────────────────────────────────────┤
│                     ocr_utils.py                         │
│  ┌─ _get_ocr_reader (EasyOCR lazy, GPU→CPU fallback)    │
│  ├─ _pre_filter_image (morfologia: 4% strips + lineas)  │
│  ├─ _detect_and_ocr (text_threshold=0.18, mag_ratio=1.2,│
│  │                   canvas_size=2500)                   │
│  │    ├─ 2 niveles de fallback:                          │
│  │    │    Tier 1: EasyOCR directo (GPU, ~0.88s)        │
│  │    │    Tier 2: CLAHE+sharpen → EasyOCR (~1s)        │
│  │    ├─ allow_fallback=False: desactiva tier 2         │
│  │    │   (modo easyocr, default)                       │
│  ├─ _group_and_merge_blocks (9 filtros post-merge)      │
│  ├─ _is_inside_speech_bubble (deteccion de globos)      │
│  ├─ _build_glyph_mask_for_bubble (mascara solo-glifos)  │
│  ├─ _build_inpaint_mask (rectangular o por glifos)     │
│  ├─ _inpaint_image (OpenCV INPAINT_NS, radio adaptativo)│
│  ├─ _sample_bg_color (perimetro del bloque)             │
│  ├─ _base64_to_cv2 / _cv2_to_base64 (conversion)       │
│  └─ _filter_watermarks_from_blocks (pre-filtro rapido)  │
│  **CTD eliminado** (Julio 2026): dependencia externa    │
│  frágil (~84MB) reemplazada por EasyOCR GPU (~0.88s/pág)│
├─────────────────────────────────────────────────────────┤
│                     routes/api.py                        │
│  ┌─ GET /api/health (estado: memoria, modelos, cache)   │
│  ├─ POST /api/translate (texto individual)              │
│  ├─ POST /api/translate-batch (multiples textos)        │
│  └─ POST /api/process-page (OCR + inpainting + trad.)   │
│       ├─ ocr_mode (default "easyocr"): "easyocr" | "auto"            │
│       │   easyocr: solo EasyOCR GPU (default, ~7.1 min 128 págs)    │
│       │   auto:    EasyOCR + CLAHE fallback (~72 min)                │
│       ├─ Modos CTD eliminados: choices ahora ["easyocr", "auto"]     │
├─────────────────────────────────────────────────────────┤
│                     routes/main.py                       │
│  └─ GET /, GET /<path> (estaticos con path traversal    │
│                           protection)                   │
├─────────────────────────────────────────────────────────┤
│                      app.js (~2533 lineas)               │
│  State: kind/pdf|image, pdf, page/pageCount, scale=1.8  │
│         boxesByPage:Map, selectedId, cvLoaded,          │
│         inpaintedBgByPage:Map, abortTranslation         │
│  Boot:  loadPdfJs (Promise.any 4 CDNs en paralelo)      │
│         initOpenCv (callback onRuntimeInitialized +      │
│          deferred 100ms check para race condition)       │
│         initTheme, initKeyboardShortcuts, initToast      │
│  PDF:   renderPage -> pdf.js @scale -> cleanBgCanvas     │
│         -> updateErasedBg -> renderBoxes                  │
│  OCR:   serverProcessPage (envia cleanBgCanvas->base64) │
│  Editor: renderBoxes, fitTextLayout (CJK char/latin     │
│          word), selectBox, draw/move/resize              │
│  Text:  **drawTextOnCanvas()** compartida (unifica      │
│          renderBoxes + drawProfessionalText)             │
│          Orden dibujo: relleno → glow → shadow → stroke  │
│          → fill                                          │
│  Glow:  glowToggle (checkbox + atajo G), glowColor       │
│         (color picker), glowBlur (range 0-30),           │
│         preview hover (mouseenter/mouseleave),           │
│         panel classes has-active-glow (ambar)            │
│  FillOpacity: range slider 0-100% → 0-1,                 │
│               panel class has-active-fill (cyan)         │
|  Filters: MARGIN_NOISE_PATTERNS (sincronizado con       │
│           config.py)                                     │
│           GLOBAL_NOISE_PATTERNS (sincronizado con        │
│           config.py)                                     │
│           filterPageBlocks (8% margin)                  │
│  Export: renderEditedCanvas -> PNG / jsPDF / PDF full   │
│  Shortcuts: D/V modes, Ctrl+T/E/P/S, arrows, Del, N/I/B,│
│             G (glow toggle)                              │
├─────────────────────────────────────────────────────────┤
│                      index.html                          │
│  43 IDs: fileInput, prevPage, pageNumber, pageTotal,    │
│          sourceLang, targetLang, drawMode, moveMode,    │
│          eraseMode, coverOriginal, bubbleColor,         │
│          textColor, strokeColor, fontFamily, fontSize,  │
│          btnItalic, btnBold, sourceText, translateBtn,  │
│          translatedText, placeManualBtn, deleteBox,     │
│          clearPageBoxes, exportName, exportPng,         │
│          exportPdf, exportAllPdf, docName, status,      │
│          mobileMenuBtn, opencvBadge, fitPage, printPage,│
│          stageWrap, stage, pdfCanvas, overlay,          │
│          emptyState, dismiss-leo-warning                │
│  Scripts: __loadCdn() generico con Promise.any()        │
│           jsPDF (jsDelivr + cdnjs fallback),            │
│           OpenCV.js (carga SECUENCIAL con IIFE:         │
│            tryCdn + tryNextCdn, evita doble carga WASM, │
│            jsDelivr @techstark + unpkg fallback),       │
│           app.js (defer, local)                         │
│  CSP: connect-src 'self' http://127.0.0.1:5174          │
│       https://cdnjs.cloudflare.com https://cdn.jsdelivr │
│       default-src/script-src incluyen docs.opencv.org   │
│       frame-ancestors solo via HTTP header (config.py)  │
├─────────────────────────────────────────────────────────┤
│              styles.css (~1318 lineas)                   │
│  Tokens: --bg-app #040406, --accent #10b981,            │
│          --radius-md 12px, --transition 200ms            │
│  Layout: .app grid minmax(240px,25%) 1fr                │
│  Effects: glassmorphism, gradient-text, bg-patterns      │
│  Responsive: <=1024px sidebar drawer                     │
├─────────────────────────────────────────────────────────┤
│              cache.py / models.py / ratelimit.py         │
│  cache.py: Filesystem con TTL 7d, MAX 5000, LRU        │
│  models.py: SQLAlchemy (User, Project, Page, TextBlock) │
│  ratelimit.py: Flask-Limiter (200/dia, 50/hora)         │
└─────────────────────────────────────────────────────────┘
```

## Flujo de datos: /api/process-page

```
Cliente (app.js)                    Servidor
      │                                 │
      ├─ canvas.toDataURL("image/png") ─┤
      │    (base64 de cleanBgCanvas)    │
      │                                 ├─ _base64_to_cv2()
      │                                 ├─ _detect_and_ocr()
      │                                 │    ├─ Tier 1: EasyOCR directo (~1s)
      │                                 │    ├─ Tier 2: CLAHE+sharpen (~1s)
      │                                 │    └─ Tier 3: CTD detecta + EasyOCR lee (~3-5s)
      │                                 ├─ _detect_language_robust()
      │                                 ├─ _build_inpaint_mask()
      │                                 │    ├─ _is_inside_speech_bubble()
      │                                 │    └─ _build_glyph_mask_for_bubble()
      │                                 ├─ _inpaint_image()
      │                                 ├─ ThreadPool: _translate_one() x N
      │                                 ├─ _sample_bg_color() x N
      │                                 └─ _cv2_to_base64(inpainted)
      ├─ {inpainted_image, blocks} ─────┤
      │                                 │
      ├─ loadBase64IntoCanvas()         │
      ├─ filterPageBlocks()             │
      └─ makeAutoTextBox() -> render    │
```

## Lanzamiento

### Modo script (desarrollo)
```
start-app.ps1 -> env\Scripts\python.exe server.py -> http://127.0.0.1:5174
start-app.ps1 también limpia procesos zombie en el puerto 5174 antes de arrancar.
```

### Modo ejecutable (.exe)
```
main.spec -> PyInstaller -> dist/main/main.exe
                               │
                               ├─ main.py (entry point)
                               │    ├─ _hide_console() → oculta CMD
                               │    ├─ _fix_cwd() → enlaza env/Lib/site-packages
                               │    └─ run_server() → importa server.py, app.run()
                               │
                               ├─ server.py + módulos Python (empaquetados)
                               ├─ index.html, app.js, styles.css (empaquetados)
                               └─ env/ → Dependencias en tiempo de ejecución
                                    ├─ easyocr, torch, cv2, numpy
                                    ├─ Flask, SQLAlchemy
                                    ├─ argostranslate, deep-translator, langdetect
                                    └─ ... (NO van dentro del .exe, se cargan desde env/)

### Cómo recompilar el .exe

```powershell
cd D:\crear traductor
.\env\Scripts\python.exe -m PyInstaller main.spec --clean --noconfirm
# Output: dist/main/main.exe (200MB, antes 2.6GB)
```

**Importante**:
- Usar `onedir` (no `--onefile`) — las dependencias pesadas se cargan desde `env/`
- Si se añaden nuevos archivos, actualizar `main.spec: DATAS` y `HIDDEN_IMPORTS`
- `main.py` acepta `--server` para modo servidor sin launcher: `main.exe --server`
- Módulos pesados (torch, transformers, ct2, easyocr) **excluidos** del .exe vía `EXCLUDES`. Se cargan desde `env/Lib/site-packages` en runtime via `_fix_cwd()`. Esto reduce el .exe de 2.6GB a 200MB y acelera el arranque de 25s a ~3s.
- `upx=False` en COLLECT (UPX compression deshabilitada — el cuello de botella era UPX en binarios grandes, no la compilación en sí). Build time: ~3.75 min (antes 10+ min).

## Pipeline CI (Integración Continua)

Flujo de verificación que se ejecuta después de cada cambio importante:

```
Cambio en código
      │
      ▼
┌──────────────────────────────────────┐
│  1. Syntax check (py_compile)        │  ← 13 archivos Python
│     ├─ server.py / routes/*.py       │
│     ├─ config.py / translator.py     │
│     ├─ ocr_utils.py / models.py      │
│     └─ cache.py / ratelimit.py / main.py / process_all_pages.py
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  2. test_ci.py (detección de idioma) │  ← ~400ms, standalone
│     └─ Detecta ES/JA/KO/ZH/EN       │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  3. Iniciar servidor Flask           │  ← 5174, 12s timeout
│     ├─ GET /api/health → db, cache   │
│     ├─ POST /api/translate → CT2     │
│     ├─ POST /api/translate-batch     │
│     └─ GET /api/config / app.js /css │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  4. analisis_calidad.py              │  ← calidad vs corpus
│     └─ BUENA / LITERAL / OCR_GARBAGE │
└──────────────────────────────────────┘
      │
      ▼  (con -Full flag)
┌──────────────────────────────────────┐
│  5. stress_test_memory.py (50 págs)  │  ← ~10 minutos
│     └─ Memoria, fugas, rendimiento   │
└──────────────────────────────────────┘
      │
      ▼  (si todo OK)
┌──────────────────────────────────────┐
│  Compilar .exe (PyInstaller)         │  ← main.spec
│     └─ dist/main/main.exe (200MB)    │
└──────────────────────────────────────┘
```

### Cobertura real del pipeline de traducción (benchmark 723 bloques, PDF 128 páginas)

| Motor | Cobertura | Tiempo promedio | Modo | Notas |
|---|---|---|---|---|
| CT2 (CTranslate2 int8) | 55% | **53ms** | Paralelo 🚀 | 10 pares de idiomas, offline, más rápido |
| ArgosTranslate | 100% ⚠️ | 2826ms | Paralelo 🚀 | Produce basura en OCR-ruidoso, filtrado por validación |
| Google Translate | 56% | 1111ms | Paralelo 🚀 | Requiere internet, más natural |
| **Pipeline combinado** | **~86%** traduce ≠ original | **~53ms** | **Paralelo** | **3 motores simultáneos** + Google retry con backoff (fix SIN_TRAD) |

#### Calidad real de traducción (analisis_calidad.py sobre 723 bloques)

| Categoría | % | Bloques | Significado |
|:----------|:-:|:-------:|:------------|
| ✅ **BUENA** | 47.6% | 344 | Traducción correcta y natural |
| 📖 **LITERAL** | 25.4% | 184 | Correcta pero palabra-por-palabra |
| ⚠️ **OCR_NOISY** | 2.6% | 19 | OCR ruidoso pero traducción aceptable |
| 🟢 **Aceptable** | **75.8%** | **547** | **Traducciones útiles** |
| ❌ **OCR_GARBAGE** | 15.9% | 115 | OCR tan ruidoso que la traducción es basura |
| ❓ **UNTRANSLATED** | 8.3% | 60 | Texto sin traducir (OCR ilegible) |
| 🔊 **ONOMATOPOEIA** | 0.1% | 1 | Efecto de sonido, correctamente sin traducir |

> **Nota**: La cobertura del pipeline (~86%) mide si el texto de salida es DIFERENTE al original (logró traducir algo). La calidad (~76% aceptable) es más estricta: requiere que la traducción sea útil (natural, literal aceptable, o comprensible a pesar de ruido OCR). La diferencia principal son 115 bloques de OCR_GARBAGE que el pipeline tradujo pero produjeron basura (principalmente running headers con fecha/hora/metadatos de página).

### Perfilado de Rendimiento (2026-07-25)

Métricas obtenidas con profiling standalone directo (sin overhead HTTP) sobre el PDF de prueba (128 págs, manga español→inglés).

| Etapa | Modo CTD | Modo EasyOCR (CPU) | % del pipeline |
|:------|:--------:|:------------------:|:--------------:|
| **OCR** (EasyOCR GPU) | **0.88s**/pág | 70% |
| **Inpainting** (OpenCV) | 0.15-0.22s/pág | 15% |
| **Traducción CT2** (GPU int8) | 0.02-0.17s/bloque | 10% |
| **Merge + filtros** | <0.01s/pág | 0% |
| **Overhead serialización** | ~0.1s/pág | 5% |
| **Total por página** | **~1.2s** | 100% |

**Bottleneck principal:** EasyOCR GPU consume **70%** del tiempo/página. Traducción en GPU es casi instantánea (0.048s/bloque).

**Benchmark real 128 páginas (workers=4, easyocr GPU, Jul 2026):**
```
Tiempo total: 425s (7.1 min)
Promedio/pág: 3.3s (incluye overhead de workers + render)
Cobertura:    128/128 (100%)
Traducción:   519/623 (83.3%)
Errores:      0
```

**Jerarquía de bottlenecks (modo easyocr, default):**

| # | Bottleneck | Impacto | Estado |
|:-:|:-----------|:-------:|:-------|
| 1 | **EasyOCR GPU** | 0.88s/pág (70%) | 🟢 Aceptable: GTX 1050 Ti, 128MB VRAM |
| 2 | **Carga de modelos** (1ra llamada) | 1.6-8.8s one-time | 🟢 Resuelto: preload background |
| 3 | **Semáforo OCR** (concurrencia) | Colas entre workers | 🟡 Mitigado: workers=4 punto óptimo |
| 4 | **Inpainting bloques grandes** | ~0.3s | 🟢 Aceptable |
| 5 | **Overhead HTTP** | ~0.1s/pág | 🟢 Aceptable |

**Benchmark GPU (GTX 1050 Ti) — EasyOCR en GPU vs CPU:**

| Métrica | CPU | GPU | Mejora |
|:--------|:--:|:---:|:------:|
| EasyOCR avg/página | 5.06s | **0.88s** | **5.7x** |
| Pipeline total/página | ~5.5s | **~1.2s** | **4.6x** |
| 128 páginas (workers=4) | ~18-35 min | **7.1 min** | **3-5x** |

> **Nota**: Ambos motores en GPU simultáneamente ✅. Se eliminó `force_cpu=True` — CT2 ahora auto-detecta CUDA. Orden precarga crítico: **EasyOCR primero** (inicializa torch.cuda/cuDNN), luego **CT2** (auto-detecta `cuda`). Verificado: GTX 1050 Ti, sin crash. Pipeline total: **7.1 min/128 págs**.

### Herramientas CI y scripts de procesamiento
| Archivo | Propósito |
|---|---|
| `run_ci.py` | **CI unificado** — syntax check + test_ci + servidor + calidad + stress (opcional). `python run_ci.py --full` |
| `run_ci.ps1` | Script CI legacy (syntax + tests + servidor + calidad) |
| `test_ci.py` | Test standalone de detección de idioma (sin modelos) |
| `analisis_calidad.py` | Auditoría de calidad de traducción contra corpus de 221 textos |
| `stress_test_memory.py` | Test de estrés **paralelo** (50 páginas, 4 workers, ~5 min con -Full) |
| `process_all_pages.py` | **Procesamiento completo de PDF en paralelo** — 128 páginas, `--workers N` (default 4, punto óptimo), `--ocr-mode easyocr|auto` (default `easyocr`, ~7.1 min), checkpoint cada 10 páginas, productor-consumidor (1 render + N API) |
| `.github/workflows/ci.yml` | CI en GitHub Actions |
| `main.spec` | Configuración de PyInstaller para build del .exe |

## Dependencias clave

- `env/` (venv completo con EasyOCR, OpenCV, Flask, ArgosTranslate, deep-translator, langdetect, torch, ctranslate2, transformers, sentencepiece)
- `requirements.txt` (dependencias pineadas)
- CDNs: jsPDF (jsDelivr + cdnjs fallback), OpenCV.js (jsDelivr @techstark + unpkg fallback secuencial — **no** Promise.any: evita doble ejecución WASM), PDF.js (Promise.any: cdnjs ESM v4 / cdnjs UMD v3 / jsDelivr / unpkg)
- Carga paralela con `Promise.any()` via `__loadCdn()` generico en index.html (solo jsPDF; OpenCV.js usa carga secuencial con IIFE para evitar BindingError por doble inicialización de bindings WASM)
