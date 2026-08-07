# CODEGRAPH — Traductor Visual Pro

## Arquitectura modular (post-refactor Julio 2026)

```
┌─────────────────────────────────────────────────────────┐
│                      server.py                          │
│  Entry point Flask (port 5174, ~255 líneas)             │
│  ┌─ Flask app, static routes, DB init, cache init       │
│  ├─ **CT2 preload en background** (hilo daemon):        │
│  │   precarga modelos es→en y en→es al arrancar.       │
│  │   Evita cold start de ~21.5s en 1ra traducción.     │
│  ├─ **Logs en tiempo real**:                             │
│  │   sys.stdout.reconfigure(line_buffering=True) +      │
│  │   PYTHONUNBUFFERED=1 + python -u. Todos los prints   │
│  │   [translate], [OCR], [CT2] se escriben inmediata-   │
│  │   mente en server_output.log, sin buffer 8KB.        │
│  ├─ **Preload daemon U-OCR** (_preload_unlimited_daemon,│
│  │   Ago 2026): lanza uocr_daemon.py en hilo propio     │
│  │   (paso 3 del preload, independiente de EasyOCR/CT2) │
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
│  ├─ _translate_one → pipeline SECUENCIAL optimizado     │
│  │    ├─ 1. CT2 síncrono (CTranslate2 int8, ~0.02-0.12s)│
│  │    │      10 pares de idiomas, GPU auto-detect        │
│  │    ├─ 2. Google fallback síncrono (~2s)              │
│  │    └─ 3. SIN_TRAD inmediato (sin timeout 30-80s)     │
│  │    **Eliminado**: pipeline paralelo (causaba timeouts │
│  │    de 30s + retry 50s = 80s/bloque). Eliminado:      │
│  │    _get_translate_engine_executor, _probar_motor,     │
│  │    executor compartido (~80 líneas de código muerto). │
│  ├─ _es_ocr_noise (detecta OCR ruidoso)                  │
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
│  ├─ _run_rapidocr (RapidOCR ONNX, lazy-load con        │
│  │                  semáforo + timeout guard)            │
│  ├─ _fusionar_blocks (merge EasyOCR+RapidOCR por        │
│  │                  texto normalizado)                   │
│  ├─ _get_rapid_engine (RapidOCR lazy, CPU ~1.1-1.5s)   │
│  ├─ _get_yolo_engine (Fase 6: ultralytics lazy DINÁMICO│
│  │    → models/comic-speech-bubble-detector.pt, 52MB;  │
│  │    sin librería/modelo degrada a [] seguro)          │
│  ├─ _detect_text_regions_in_page (Fase 6: YOLO predict  │
│  │    → regiones {x,y,w,h,label,cls_conf}; filtra por   │
│  │    clase keyword + área mínima; device auto: GPU "0"  │
│  │    si CUDA libre Y daemon no infiere + _gpu_lock no-  │
│  │    bloqueante (serializa con EasyOCR GPU, Fase 6.5)  │
│  ├─ _preprocess_rapid (pre_filter + enhance para Rapid) │
│  ├─ _is_inside_speech_bubble (deteccion de globos)      │
│  ├─ _build_glyph_mask_for_bubble (mascara solo-glifos)  │
│  ├─ _build_inpaint_mask (rectangular o por glifos)     │
│  ├─ _inpaint_image (OpenCV INPAINT_NS, radio adaptativo)│
│  ├─ _sample_bg_color (perimetro del bloque)             │
│  ├─ _base64_to_cv2 / _cv2_to_base64 (conversion)       │
│  └─ _filter_watermarks_from_blocks (pre-filtro rapido)  │
│  **CTD eliminado** (Julio 2026): dependencia externa    │
│  frágil (~84MB) reemplazada por EasyOCR GPU (~0.88s/pág)│
│  **Fase 2**: _run_rapidocr acepta box_thresh/unclip_    │
│  ratio/text_score (params SIEMPRE explícitos: la        │
│  librería muta postprocess_op). _fusionar_blocks_multi  │
│  pondera la votación por type semántico (Fase 3).       │
├─────────────────────────────────────────────────────────┤
│                  ocr_engine.py (NUEVO, 2026-08-05)      │
│  OCRManager: orquesta los 3 motores en UNA clase        │
│  ├─ run_ocr() dispatcher (easyocr/auto/fusion/unlimited)│
│  ├─ _compute_trigger (trigger v4.2 aislado/testeable)   │
│  ├─ _reforzar_con_unlimited (+ Ruta C re-OCR globos,    │
│  │    cls rotación 180° via TextClassifier RapidOCR)    │
│  ├─ _ruta_c_yolo (Fase 6): YOLO detecta globos/cartelas/│
│  │    títulos como objetos → Ruta C (upscale 3.5× +     │
│  │    rotation_info 0/90/180/270). Gate heurístico      │
│  │    (<3 bloques o conf<0.35); disable_uocr lo apaga.  │
│  ├─ _reforzar_con_rapid_agresivo (Fase 2, pre-VLM:      │
│  │    reintento CPU ~1.5s antes del daemon 2-8 min)     │
│  ├─ cache decisiones negativas §8.4.1 (firma de página, │
│  │    TTL 1800s, LRU 256)                               │
│  └─ Acceso a ocr_utils/routes.api EN RUNTIME (mocks de  │
│       pytest sin romper: self.ou.<fn>, import dentro)   │
├─────────────────────────────────────────────────────────┤
│             uocr_daemon.py + uocr_client.py (NUEVO)     │
│  Daemon persistente (puerto 5177, venv env_uocr_gpu,    │
│  modelo 3B 4-bit NF4 bitsandbytes, ~2.25GB VRAM):       │
│  ├─ POST /ocr        (1 página, crop_mode + re-OCR arte)│
│  ├─ POST /ocr-batch  (Fase 1: _model.infer_multi(),     │
│  │                    1-4 págs, separador <PAGE> antes   │
│  │                    de cada página, split [1:])        │
│  ├─ Salida: <|det|>type [x,y,w,h]<|/det|> por stdout    │
│  └─ uocr_client: process_page()/process_batch()/health()│
├─────────────────────────────────────────────────────────┤
│                     routes/api.py                        │
│  ┌─ GET /api/health (estado: memoria, modelos, cache,   │
│  │       unlimited_ocr, uocr_load_s)                    │
│  ├─ POST /api/translate (texto individual)              │
│  ├─ POST /api/translate-batch (multiples textos)        │
│  └─ POST /api/process-page (OCR + inpainting + trad.)   │
│       ├─ DELEGA en OCRManager (ocr_engine.py) — el      │
│       │   bloque OCR inline de ~110 líneas se movió     │
│       ├─ ocr_mode: "easyocr" | "auto" | "fusion" |      │
│       │   "unlimited" (+ flags force_uocr/disable_uocr/ │
│       │   pure_easyocr de benchmark)                    │
│       ├─ fusion (default de trabajo): híbrido siempre +  │
│       │   U-OCR solo con trigger v4.2 + reintento Fase 2│
│       └─ _ocr_with_unlimited: propaga type semántico    │
│           (Fase 3) + conf heurística + image_panels     │
├─────────────────────────────────────────────────────────┤
│                     routes/main.py                       │
│  └─ GET /, GET /<path> (estaticos con path traversal    │
│                           protection)                   │
├─────────────────────────────────────────────────────────┤
│                      app.js (~2898 lineas)               │
│  State: kind/pdf|image, pdf, page/pageCount, scale=1.8  │
│         boxesByPage:Map, selectedId, cvLoaded,          │
│         inpaintedBgByPage:Map, abortTranslation,        │
│         translationPaused (Pausa/Reanudar toggle)       │
│  Boot:  loadPdfJs (Promise.any 4 CDNs en paralelo)      │
│         initOpenCv (callback onRuntimeInitialized +      │
│          deferred 100ms check para race condition)       │
│         initTheme, initKeyboardShortcuts, initToast      │
│  PDF:   renderPage -> pdf.js @scale -> cleanBgCanvas     │
│         -> updateErasedBg -> renderBoxes                  │
│  OCR:   serverProcessPage (envia cleanBgCanvas->base64) │
│  Motor: #ocrEngine selector (Automático=fusion default) │
│         badge daemon #ocrEngineStatus (polling 20s)     │
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
│  Progress: showProgress() — barra [⏸ Pausar] [🛑 Cancelar]│
│           state.translationPaused toggle                 │
│           autoTranslateAllPages: while loop 500ms        │
│  LangWarn: checkLanguageWarning() — aviso origen==destino│
│           isoToSelector ("es"→"spa", "en"→"eng", etc.)   │
│           source="auto" → aviso condicional              │
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
│              styles.css (~1811 lineas)                   │
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

### Benchmarks del pipeline híbrido (Julio 2026)

| Pipeline | Tiempo 128 págs | Promedio/pág | Cobertura | Sin traducir |
|:---------|:---------------:|:------------:|:---------:|:------------:|
| EasyOCR GPU solo | ~15-20 min | 3.3s | 100% | 0 págs |
| EasyOCR + CLAHE | ~37 min | 17.5s | 100% | 0 págs |
| **EasyOCR + RapidOCR** | **11.7 min** | **5.5s** | **100%** | **0 págs** |

**Traducción**: 351/400 bloques (87.8%). El 12.2% restante son fragmentos OCR ruidosos que ningún motor pudo descifrar.

**Nota CPU vs GPU**: `onnxruntime-gpu` instalado y probado en GTX 1050 Ti. **GPU NO acelera RapidOCR** (~1.0x speedup) porque los modelos PP-OCRv4 son pequeños (~6.5MB total). El overhead de transferencia PCIe anula cualquier ganancia. `onnxruntime` (CPU) es suficiente y se recomienda como dependencia.

> **Nota 2026-08-03**: la tabla anterior y los benchmarks de Julio usan el PDF viejo (128 págs). El PDF de prueba actual es `Capítulo 43 de Cómo criar villanos correctamente.pdf` (53 págs) — ver `benchmark_overhead_results.json` y AGENTS.md §4 para métricas del PDF nuevo.

## Flujo de datos: /api/process-page

```
Cliente (app.js)                    Servidor
      │                                 │
      ├─ canvas.toDataURL("image/png") ─┤
      │    (base64 de cleanBgCanvas)    │
      │                                 ├─ _base64_to_cv2()
      │                                 ├─ OCRManager().run_ocr()  (ocr_engine.py)
      │                                 │    ├─ _run_hybrid: _detect_and_ocr()
      │                                 │    │    ├─ Tier 1: EasyOCR directo (~0.88s)
      │                                 │    │    ├─ Tier 2: CLAHE+sharpen (~1s)
      │                                 │    │    └─ Tier 3: RapidOCR ONNX CPU (~1.1-1.5s)
      │                                 │    ├─ _compute_trigger (v4.2): 0 bloques |
      │                                 │    │    <3 Y conf<0.2 | panel image>15% | force
      │                                 │    ├─ Fase 2: _reforzar_con_rapid_agresivo
      │                                 │    │    (box_thresh .30/unclip 2.2/text .40,
      │                                 │    │     CPU ~1.5s — evita el VLM si resuelve)
│                                 │    ├─ run_ocr_batch() (Fase 1 ✅): /process-page-batch
│                                 │    │    híbrido+trigger+Fase 2 por página → las que
│                                 │    │    requieren VLM van en UN /ocr-batch (infer_multi)
│                                 │    │    → Ruta C + fusión por página
│                                 │    └─ _reforzar_con_unlimited → uocr_client
      │                                 │         → daemon 5177 (U-OCR 3B 4-bit) + Ruta C
      │                                 ├─ _detect_language_robust()
      │                                 ├─ _build_inpaint_mask()
      │                                 │    ├─ _is_inside_speech_bubble()
      │                                 │    └─ _build_glyph_mask_for_bubble()
      │                                 ├─ _inpaint_image()
      │                                 ├─ ThreadPool: _translate_one() x N
      │                                 ├─ _sample_bg_color() x N
      │                                 └─ _cv2_to_base64(inpainted)
      ├─ {inpainted_image, blocks+type} ─┤
      │                                 │
      ├─ loadBase64IntoCanvas()         │
      ├─ filterPageBlocks()             │
      └─ makeAutoTextBox() -> render    │
```

**Nota**: el diagrama muestra el pipeline fusion completo; el modo `easyocr`/`auto`
solo ejecuta `_run_hybrid` (sin daemon). En modo `fusion` el daemon U-OCR solo
se consulta si el trigger v4.2 lo decide (y el reintento agresivo Fase 2 no
resuelve la página). El estado del daemon se expone en `/api/health`
(`unlimited_ocr`, `uocr_load_s`).

### Fase 5 validada (2026-08-04) — capítulo 53 págs en fusion + batch

**Run real**: `--ocr-mode fusion --batch-window 4` → **~22.5 min de pared** (vs ~47 min
estimados sin batch, **2.1x más rápido**), 53/53 páginas, **0 errores**, 47 páginas con
texto. Solo 3 lotes disparan U-OCR (p5 y lotes 39-42, 51-53): con `infer_multi` el
prefill del VLM se comparte (lote de 4 páginas ≈ 671s ≈ **168s/pág** vs 366-592s/pág
individuales). Páginas normales: ~5-15s (p3 12.2s, p11 14.9s, p12 9.2s).

**Bug fijado durante la validación**: las páginas 19-22 daban 500 (`http_500`) por la
race window de §8.4.4 en el camino de página COMPLETA: `_run_ocr_on_image` degradaba a
RapidOCR (dicts) pero `_ocr_results_to_blocks` solo desempaquetaba tuplas
`(bbox,text,conf)` → `"too many values to unpack"` (un dict itera sus keys). Fix:
`isinstance(res, dict)` → convierte directo a bloque (filtro `conf<0.08` con paridad,
code review). El mismo patrón dict/tupla ya existía en la Ruta C; ahora es central en
`_ocr_results_to_blocks` y cubre tier 1, retry mag_ratio y tier 2 CLAHE. 4 tests nuevos.

**Reporte**: `generate_fusion_report.py` → `reporte_fusion.html` (datos reales del run:
5.1-331.7s por página artística, CER vs ground truth, distribución de tiempos, imágenes
con bloques resaltados). Nota: la tasa de traducción (25.2%) fue baja porque el par CT2
es→en no se descargó (sin internet a HuggingFace en ese run) — la cobertura de detección
(47/53, 0 errores) es el dato clave.

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
                               ├─ **js/** → 5 módulos ES (empaquetados)
                               │    ├─ config.js, utils.js, toast.js,
                               │    ├─ theme.js, filters.js
                               │    ⚠️ CRÍTICO: Si falta js/ en el bundle,
                               │       app.js (ES module) no importa,
                               │       initOpenCv() nunca se ejecuta,
                               │       badge queda "Cargando OpenCV...".
                               └─ env/ → Dependencias en tiempo de ejecución
                                    ├─ easyocr, torch, cv2, numpy
                                    ├─ Flask, SQLAlchemy
                                    ├─ argostranslate, deep-translator, langdetect
                                    └─ ... (NO van dentro del .exe, se cargan desde env/)

### Cómo recompilar el .exe

```powershell
cd D:\crear traductor
.\env\Scripts\python.exe -m PyInstaller main.spec --clean --noconfirm
# Output: dist/main/main.exe (360MB, antes 2.6GB)
```

**Importante**:
- Usar `onedir` (no `--onefile`) — las dependencias pesadas se cargan desde `env/`
- Si se añaden nuevos archivos, actualizar `main.spec: DATAS` y `HIDDEN_IMPORTS`
- `main.py` acepta `--server` para modo servidor sin launcher: `main.exe --server`
- Módulos pesados (torch, transformers, ct2, easyocr) **excluidos** del .exe vía `EXCLUDES`. Se cargan desde `env/Lib/site-packages` en runtime via `_fix_cwd()`. Esto reduce el .exe de 2.6GB a 200MB (360MB con los modelos ONNX de RapidOCR incluidos) y acelera el arranque de 25s a ~3s.
- `upx=False` en COLLECT (UPX compression deshabilitada — el cuello de botella era UPX en binarios grandes, no la compilación en sí). Build time: ~3.75 min (antes 10+ min).

## Pipeline CI (Integración Continua)

Flujo de verificación que se ejecuta después de cada cambio importante:

```
Cambio en código
      │
      ▼
┌──────────────────────────────────────┐
│  1. Syntax check (py_compile)        │  ← 16 archivos Python
│     ├─ server.py / routes/*.py       │
│     ├─ config.py / translator.py     │
│     ├─ ocr_utils.py / models.py      │
│     ├─ ocr_engine.py / uocr_daemon.py│
│     │   uocr_client.py (Ago 2026)    │
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
│     └─ dist/main/main.exe (360MB)    │
└──────────────────────────────────────┘
```

### Cobertura real del pipeline de traducción (benchmark 723 bloques, PDF 128 páginas)

| Motor | Cobertura | Tiempo promedio | Modo | Notas |
|---|---|---|---|---|
| CT2 (CTranslate2 int8) | 55% | **53ms** | 1º (síncrono) | 10 pares de idiomas, offline, más rápido |
| ArgosTranslate | 100% ⚠️ | 2826ms | — | Produce basura en OCR-ruidoso, filtrado por validación (ya no se usa en el pipeline) |
| Google Translate | 56% | 1111ms | 2º (fallback) | Requiere internet, más natural |
| **Pipeline secuencial** | **~86%** traduce ≠ original | **~53ms** | **CT2→Google→SIN_TRAD** | **Desde sesión 68 (2026-07-29) el pipeline es SECUENCIAL** — los porcentajes de motor son del benchmark de 723 bloques (Julio), la orquestación cambió (3 motores en paralelo → CT2 síncrono primero, Google fallback, SIN_TRAD inmediato) |

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

| Etapa | EasyOCR GPU | % del pipeline |
|:------|:-----------:|:--------------:|
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

**Jerarquía de bottlenecks (modo easyocr — ya no es el default desde Fase 4, Ago 2026):**

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
| `process_all_pages.py` | **Procesamiento completo de PDF en paralelo** — `--workers N` (default 3, punto óptimo), `--ocr-mode auto|easyocr|fusion` (default `fusion` desde Fase 4 — 2026-08-06), timeout 1800s (páginas U-OCR en cola), checkpoint cada 10 páginas, productor-consumidor (1 render + N API) |
| `ocr_engine.py` | **OCRManager** (Ago 2026) — orquesta los 3 motores con trigger v4.2, reintento Fase 2 y cache §8.4.1 |
| `uocr_daemon.py` + `uocr_client.py` | **Daemon Unlimited-OCR persistente** (Ago 2026) — `POST /ocr` y `POST /ocr-batch` (Fase 1, `infer_multi`) en 127.0.0.1:5177 |
| `benchmark_fusion_overhead.py` | Benchmark de overhead del merge (modos disable_uocr/pure_easyocr sobre el PDF nuevo de 53 págs) |
| `INVESTIGACION_3_OCR.md` / `PLAN_CODIGO_INTERNO_3_OCR.md` | Investigación del código interno de los 3 motores + plan de fusión por fases |
| `.github/workflows/ci.yml` | CI en GitHub Actions |
| `main.spec` | Configuración de PyInstaller para build del .exe |

## Dependencias clave

- `env/` (venv completo con EasyOCR, OpenCV, Flask, ArgosTranslate, deep-translator, langdetect, torch, ctranslate2, transformers, sentencepiece)
- `requirements.txt` (dependencias pineadas)
- CDNs: jsPDF (jsDelivr + cdnjs fallback), OpenCV.js (jsDelivr @techstark + unpkg fallback secuencial — **no** Promise.any: evita doble ejecución WASM), PDF.js (Promise.any: cdnjs ESM v4 / cdnjs UMD v3 / jsDelivr / unpkg)
- Carga paralela con `Promise.any()` via `__loadCdn()` generico en index.html (solo jsPDF; OpenCV.js usa carga secuencial con IIFE para evitar BindingError por doble inicialización de bindings WASM)
