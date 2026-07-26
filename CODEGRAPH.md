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
│  │    ├─ 3 niveles de fallback:                          │
│  │    │    Tier 1: EasyOCR directo (rápido, ~1s)        │
│  │    │    Tier 2: CLAHE+sharpen → EasyOCR (~1s)        │
│  │    │    Tier 3: CTD detecta → EasyOCR reconoce (~3-5s)│
│  │    ├─ use_ctd_only=True: salta tier 1 y 2, va directo│
│  │    │   a CTD (modo ctd en API)                       │
│  │    ├─ allow_fallback=False: desactiva tiers 2 y 3    │
│  │    │   (modo easyocr en API)                         │
│  ├─ _group_and_merge_blocks (9 filtros post-merge)      │
│  ├─ _is_inside_speech_bubble (deteccion de globos)      │
│  ├─ _build_glyph_mask_for_bubble (mascara solo-glifos)  │
│  ├─ _build_inpaint_mask (rectangular o por glifos)     │
│  ├─ _inpaint_image (OpenCV INPAINT_NS, radio adaptativo)│
│  ├─ _sample_bg_color (perimetro del bloque)             │
│  ├─ _base64_to_cv2 / _cv2_to_base64 (conversion)       │
│  └─ _filter_watermarks_from_blocks (pre-filtro rapido)  │
├─────────────────────────────────────────────────────────┤
│                  ocr_ctd_fallback.py                      │
│  Fallback OCR con CTD (ComicTextDetector).               │
│  ┌─ Modelo: comictextdetector.pt (~76MB, ConvNeXt)      │
│  ├─ _download_ctd_model (descarga lazy con .ctd_ready)  │
│  ├─ preload_ctd (carga thread-safe con lock)            │
│  └─ ctd_fallback_ocr (detecta regiones, EasyOCR lee)   │
├─────────────────────────────────────────────────────────┤
│                     routes/api.py                        │
│  ┌─ GET /api/health (estado: memoria, modelos, cache)   │
│  ├─ POST /api/translate (texto individual)              │
│  ├─ POST /api/translate-batch (multiples textos)        │
│  └─ POST /api/process-page (OCR + inpainting + trad.)   │
│       ├─ ocr_mode (default "ctd"): "ctd" | "auto" | "easyocr"          │
│       │   ctd:     solo CTD, salta EasyOCR completo (más rápido)    │
│       │   auto:    3 niveles (directo→CLAHE→CTD)                     │
│       │   easyocr: solo EasyOCR, sin fallbacks                       │
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
│         initOpenCv (callback onRuntimeInitialized)       │
│         initTheme, initKeyboardShortcuts, initToast      │
│  PDF:   renderPage -> pdf.js @scale -> cleanBgCanvas     │
│         -> updateErasedBg -> renderBoxes                  │
│  OCR:   serverProcessPage (envia cleanBgCanvas->base64) │
│  Editor: renderBoxes, fitTextLayout (CJK char/latin     │
│          word), selectBox, draw/move/resize              │
|  Filters: MARGIN_NOISE_PATTERNS (sincronizado con       │
│           config.py — sin capítulo/cómo criar/how to    │
│           raise para no filtrar títulos legítimos)      │
│           GLOBAL_NOISE_PATTERNS (sincronizado con       │
│           config.py — solo zonaolympus.com + 1 C 2 E)  │
│           filterPageBlocks (8% margin)                  │
│  Export: renderEditedCanvas -> PNG / jsPDF / PDF full   │
│  Shortcuts: D/V modes, Ctrl+T/E/P/S, arrows, Del, N/I/B│
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
│           OpenCV.js (jsDelivr @techstark + docs.opencv),│
│           app.js (defer, local)                         │
│  CSP: connect-src 'self' http://127.0.0.1:5174          │
│       https://cdnjs.cloudflare.com https://cdn.jsdelivr │
│       default-src/script-src incluyen docs.opencv.org   │
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
│     ├─ ocr_ctd_fallback.py            │
│     └─ cache.py / ratelimit.py / main.py
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
│     └─ dist/main/main.exe (321MB)    │
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
| **Detección de texto** (OCR/CTD) | **0.21-0.25s**/pág | **5.06s**/pág | CTD: 30%, EasyOCR: 93% |
| **Inpainting** (OpenCV) | 0.15-0.22s/pág | 0.15-0.22s/pág | 27% |
| **Traducción CT2** (GPU int8) | 0.02-0.17s/bloque | 0.02-0.17s/bloque | 20% |
| **Merge + filtros** | <0.01s/pág | <0.01s/pág | 0% |
| **Overhead serialización** | ~0.1s/pág | ~0.1s/pág | 13% |
| **Total por página** | **~0.7s** | **~5.5s** | 100% |

**Bottleneck principal:** EasyOCR en CPU consume **93%** del tiempo en modo `auto`. CTD (modo `ctd`, el default) es **13-24x más rápido** y reduce el tiempo total a ~0.7s/página.

**Proyección 128 páginas (3 workers, CTD):**
```
CTD:        0.25s × 128 / 3  =  10.7s
Inpainting: 0.22s × 128 / 3  =   9.4s
Traducción: 0.10s × 128 / 3  =   4.3s
Overhead:   0.05s × 128 / 3  =   2.1s
Trabajo puro (sin contención): ~27s
+ colas + warmup + render         3-5 min   ← benchmark real
```

**Jerarquía de bottlenecks:**

| # | Bottleneck | Impacto | Estado |
|:-:|:-----------|:-------:|:-------|
| 1 | **EasyOCR en CPU** | 5s/pág cuando se usa | 🟢 **Resuelto: ambos en GPU** (EasyOCR 0.88s + CT2 0.048s). Orden precarga: EasyOCR → CUDA, luego CT2. Sin crash cuDNN. |
| 2 | **Carga de modelos** (1ra llamada) | 1.6-8.8s one-time | Resuelto: preload CT2 en background al arrancar |
| 3 | **CTD en páginas vacías** | ~0.3s inútil | Mitigado: semáforo OCR limita concurrencia |
| 4 | **Inpainting bloques grandes** | ~0.3s | Aceptable: radio adaptativo ya optimizado |
| 5 | **Overhead HTTP** | ~0.1s/pág | Aceptable: sesión persistente reutiliza conexiones |

**Benchmark GPU (GTX 1050 Ti) — EasyOCR en GPU vs CPU:**

| Métrica | CPU | GPU | Mejora |
|:--------|:--:|:---:|:------:|
| EasyOCR avg/página | 5.06s | **0.88s** | **5.7x** |
| Pipeline total/página | ~5.5s | **0.7s** | **7.9x** |
| 128 páginas (estimado, 3 workers) | ~18-35 min | **~8 min** | ~3x |

> **Nota actualizada (2026-07-25)**: **Ambos motores en GPU simultáneamente** ✅. Se eliminó `force_cpu=True` — CT2 ahora auto-detecta CUDA y carga en GPU. Orden de precarga crítico: **EasyOCR primero** (inicializa torch.cuda/cuDNN), luego **CT2** (auto-detecta `cuda`). Verificado: GTX 1050 Ti, 128MB VRAM, sin crash cuDNN. EasyOCR: 0.88s/pág (5.7x), CT2: 0.048s/trad (~6x). Pipeline total: **~0.7s/pág** en GPU vs ~5.5s en CPU.

### Herramientas CI y scripts de procesamiento
| Archivo | Propósito |
|---|---|
| `run_ci.py` | **CI unificado** — syntax check + test_ci + servidor + calidad + stress (opcional). `python run_ci.py --full` |
| `run_ci.ps1` | Script CI legacy (syntax + tests + servidor + calidad) |
| `test_ci.py` | Test standalone de detección de idioma (sin modelos) |
| `analisis_calidad.py` | Auditoría de calidad de traducción contra corpus de 221 textos |
| `stress_test_memory.py` | Test de estrés **paralelo** (50 páginas, 4 workers, ~5 min con -Full) |
| `process_all_pages.py` | **Procesamiento completo de PDF en paralelo** — 128 páginas, `--workers N` (default 3, punto óptimo), `--ocr-mode ctd|auto|easyocr` (default `ctd`, 2x más rápido), checkpoint cada 10 páginas, productor-consumidor (1 render + N API) |
| `.github/workflows/ci.yml` | CI en GitHub Actions |
| `main.spec` | Configuración de PyInstaller para build del .exe |

## Dependencias clave

- `env/` (venv completo con EasyOCR, OpenCV, Flask, ArgosTranslate, deep-translator, langdetect, torch, ctranslate2, transformers, sentencepiece)
- `requirements.txt` (dependencias pineadas)
- CDNs: jsPDF (jsDelivr + cdnjs fallback), OpenCV.js (jsDelivr @techstark + docs.opencv.org fallback), PDF.js (Promise.any: cdnjs ESM v4 / cdnjs UMD v3 / jsDelivr / unpkg)
- Carga paralela con `Promise.any()` via `__loadCdn()` generico en index.html
