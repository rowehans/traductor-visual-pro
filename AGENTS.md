> ⚠️ **LEE ESTE ARCHIVO ANTES DE MODIFICAR CUALQUIER ARCHIVO DEL PROYECTO.**

# AGENTS.md — Guía de contexto para IAs y colaboradores

## 1. Descripción del proyecto

**Traductor Visual Pro** — Aplicación local para traducir manga, cómics y documentos en PDF e imagen.

### Arquitectura actual (post-refactor Julio 2026)

| Archivo | Rol | Tamaño |
|---|---|---|
| `app.js` | Frontend completo (~2767 líneas): renderizado PDF/imagen vía pdf.js, editor de burbujas (draw/move/resize), OCR delegado al servidor EasyOCR, filtros de bloques, layout de texto en canvas (wrap+fit), comunicación con API Flask, exportación PNG/PDF, tema oscuro/claro, atajos de teclado, toasts, carga asíncrona de OpenCV.js vía callback. | 101KB |
| `server.py` | Entry point Flask (~217 líneas). Importa de config.py/translator.py, envuelve _translate_one() con caché, **precarga EasyOCR + CT2 en background** (orden crítico: EasyOCR primero para inicializar CUDA, luego CT2 auto-detecta GPU). Puerto 5174. | 8KB |
| `config.py` | Constantes globales: paths, `LANGUAGES`, `CSP_POLICY`, patrones de ruido (`MARGIN_NOISE_PATTERNS`, `WATERMARK_PATTERNS`), glosario pre-OCR (`GLOSARIO_PRE`). | 6KB |
| `translator.py` | Lógica de traducción: detección de idioma (`_detect_language_robust`), 3 motores en **paralelo** vía **executor compartido** (4 threads, evita crear/destruir threads por llamada), validación de traducción (6 validaciones anti-basura), glosarios PRE/POST, filtro pre-Argos para OCR noise. **Google retry con backoff** (5s, 15s, 30s) cuando todos los motores fallan (fix SIN_TRAD). Cache injectado desde server.py. | 28KB |
| `ocr_utils.py` | OCR con EasyOCR (GPU prioritario, CPU fallback automático si CUDA no disponible), pre-filtro de imagen, inpainting con OpenCV (INPAINT_NS), detección de globos de diálogo, máscara de glifos, sampleo de color, fusión y filtrado de bloques (9 filtros post-merge). **Pipeline simplificado: 2 niveles**: EasyOCR directo → CLAHE+sharpen (sin CTD). **Optimizado**: canvas_size=2500px, text_threshold=0.18, mag_ratio=1.2. **GPU**: GTX 1050 Ti verificado ~0.88s/pág vs 5s CPU (5.7x). | 23KB |
| `routes/api.py` | Blueprint REST: `/api/health`, `/api/translate`, `/api/translate-batch`, `/api/process-page`. **Expone `ocr_mode`** en `/api/process-page` (default `"easyocr"`): `"easyocr"` (solo EasyOCR GPU, ~18 min 128 págs), `"auto"` (EasyOCR + CLAHE fallback, ~72 min). **Incluye decorador `@profile_endpoint`** para profiling cProfile inline activable vía `?profile=1`. Importa directamente de los submódulos. | 11KB |
| `routes/main.py` | Blueprint de rutas estáticas con protección contra path traversal. | 1KB |
| `index.html` | UI HTML (~339 líneas): estructura del editor visual, CSP vía `<meta>`, detección de Brave Leo/Shields. | 17KB |
| `styles.css` | Estilos visuales premium (tema dark/light con variables CSS, glassmorphism, animaciones, responsive). | 40KB |
| `cache.py` | Caché de traducciones en filesystem con TTL (7 días) y LRU eviction (5000 entradas máx). | 2KB |
| `models.py` | Modelos SQLAlchemy (User, Project, Page, TextBlock) con repositorios. SQLite local / PostgreSQL producción. | 7KB |
| `ratelimit.py` | Rate limiting con Flask-Limiter. Evita imports circulares entre server.py y routes/. | 1KB |
| `start-app.ps1` | Lanzador: inicia `env\Scripts\python.exe server.py` y abre `http://127.0.0.1:5174`. | 1KB |
| `run_ci.py` | **CI unificado** — ejecuta syntax check + test_ci.py + servidor + analisis_calidad + stress test en un solo comando Python. No depende de PowerShell. `python run_ci.py --full` para test completo (~10 min). | 20KB |
| `requirements.txt` | Dependencias Python pineadas. | 1KB |
| `main.py` | Entry point del ejecutable (.exe). Modo launcher: oculta consola, inicia Flask, abre Chrome. Modo `--server`: solo servidor. | 4KB |
| `main.spec` | PyInstaller spec para compilar `main.exe` con `console=False` (sin ventana CMD). CTD eliminado del bundle. **Excluye módulos pesados** (torch, transformers, ct2, easyocr) — se cargan desde `env/` en runtime. **UPX deshabilitado**. .exe resultante: **200MB** (antes 2.6GB). | 3KB |
| `launcher.py` | Launcher alternativo (subprocess). Usa `env\Scripts\python.exe server.py` como proceso hijo. | 1KB |
| `env/` | Entorno virtual Python con **todas** las dependencias (EasyOCR, OpenCV, Flask, ArgosTranslate, deep-translator, langdetect, torch). | — |

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

**Bottleneck principal:** EasyOCR GPU consume **70%** del tiempo/página. Traducción CT2 en GPU es casi instantánea (0.048s/bloque).

**Benchmark real 128 páginas (workers=4, easyocr GPU, Jul 2026):**
```
Tiempo total: 425s (7.1 min)  🚀
Promedio/pág: 3.3s
Cobertura:    128/128 (100%)
Traducción:   519/623 (83.3%)
Errores:      0
```

**Jerarquía de bottlenecks:**

| # | Bottleneck | Impacto | Estado |
|:-:|:-----------|:-------:|:-------|
| 1 | **EasyOCR GPU** | 0.88s/pág (70% del tiempo) | 🟢 Aceptable: GPU GTX 1050 Ti, 128MB VRAM, 5.7x vs CPU |
| 2 | **Carga de modelos** (1ra llamada) | 1.6-8.8s one-time | 🟢 Resuelto: preload CT2 en background al arrancar |
| 3 | **Semáforo OCR** (concurrencia limitada) | Colas entre workers | 🟡 Mitigado: workers=4 es punto óptimo, 5+ satura |
| 4 | **Inpainting bloques grandes** | ~0.3s | 🟢 Aceptable: radio adaptativo ya optimizado |
| 5 | **Overhead HTTP** | ~0.1s/pág | 🟢 Aceptable: sesión persistente reutiliza conexiones |

---

## 2. Zonas Sensibles ⛔

### `app.js`

| Función / Variable | Línea aprox. | Por qué es delicada |
|---|---|---|
| `loadPdfJs()` | ~476 | **Carga dual de PDF.js**: intenta primero `import()` de `.mjs` (v4.10.38), y si falla, carga script UMD clásico `pdf.js` v3.11.174 como fallback con 4 estrategias de CDN (cdnjs → jsDelivr → unpkg). |
| `wrapTextLines(ctx, text, maxWidth)` | ~1880 | **Bug histórico crítico**: si la condición `containsCJK()` cambia, palabras occidentales vuelven a separarse letra por letra. Solo separa carácter a carácter si el texto contiene CJK. |
| `MARGIN_NOISE_PATTERNS` / `GLOBAL_NOISE_PATTERNS` | ~1007–1025 | Deben estar sincronizados con `_MARGIN_NOISE_PATTERNS`/`_WATERMARK_PATTERNS` en `config.py`. Divergencia causa textos basura o diálogos eliminados. |
| `state` (incl. `inpaintedBgByPage` Map) | ~148 | Caché global de estado. `inpaintedBgByPage` guarda imágenes inpaintadas por página. Si no se escribe aquí tras `serverProcessPage()`, el texto original reaparece. |
| `initOpenCv()` | ~168 | Reemplazó el polling por `cv['onRuntimeInitialized']` callback. 3 casos: cv ya cargado, cv existe sin Mat, cv no existe (con timeout 15s). |
| `autoTranslateCurrentPage(pageNo, ...)` | ~1419 | Camino único: servidor. `serverProcessPage()` → guarda `inpaintedBgByPage` → carga `erasedBgCanvas` → filtra bloques → `makeAutoTextBox` → render. |
| `renderPage()` | ~744 | Usa `_renderToken` para cancelación, `_renderTempCanvas` global reutilizado, `renderTask.cancel()` en timeout. Retorna `{aborted: bool}`. |
| `filterPageBlocks(blocks, pageHeight)` | ~1038 | Único punto que filtra bloques antes de crear cajas. `marginTop` = 8% de altura. |
| `exportFullPdf()` | ~2105 | Genera PDF completo página por página: renderRawPdfPage → renderEditedCanvas → jsPDF.addPage. |

### `server.py` (entry point)

| Elemento | Línea | Por qué es delicado |
|---|---|---|
| `_preload_background()` | ~50 | **ORDEN CRÍTICO**: EasyOCR primero (inicializa CUDA), luego CT2 (auto-detecta GPU). NO INVERTIR. Si CT2 carga primero, sus DLLs cuDNN conflictúan con las de PyTorch → crash "Could not load symbol cudnnGetLibConfig". Verificado: GTX 1050 Ti, ambos en GPU, 128MB VRAM usados, sin crash. |
| `_translate_one()` | ~95 | Wrapper que inyecta `cache_get`/`cache_set` en `translator._translate_one()`. **Debe importarse desde `server` en routes/api.py** — NO desde `translator.py` directo. |
| `from config import ...` | ~12 | Importaciones controladas. No agregar imports circulares. |

### `translator.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_translate_one()` | ~185 | **3 motores en paralelo** (CT2 CTranslate2 int8 → Argos → Google) lanzados via **executor compartido** `_get_translate_engine_executor()` (4 threads, lazy init, double-checked locking). **Antes**: creaba/destruía un `ThreadPoolExecutor` por llamada (~1.857 ciclos para 619 bloques). **Ahora**: executor compartido, sin shutdown, retorno inmediato cuando CT2 gana. Acepta `cache_get`/`cache_set`/`translation_cache_available` para inyección. |
| `_get_google_session()` | ~58 | Double-checked locking con `_google_session_lock`. La sesión HTTP se crea **dentro del lock**. |
| `_detect_language_robust()` | ~96 | langdetect thread-local + heurística `_detect_language_simple`. Mapeo zh-cn/zh-tw → zh. |
| `_ensure_argo_package()` | ~20 | Descarga modelos Argos con lock para evitar descargas duplicadas. |
| `_CT2_MODELS` (dict global) | ~35 | **18 pares de idiomas** (10 originales + 6 CJK: ja|en, en|ja, ko|en, en|ko, zh|en, en|zh + 2 legacy). Cada entrada mapea `"src|tgt"` a un modelo Helsinki-NLP OPUS-MT. **Si se agrega un par, debe coincidir exactamente** con los códigos ISO de `_detect_language_robust()` (ja, ko, zh). Modelos lazy-load: descarga + conversión CT2 en primer uso. No modificar las keys sin verificar que el pipeline `_get_ct2_translator()` siga funcionando (usa `f"{source}|{target}"` como lookup). |

### `ocr_utils.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_get_ocr_reader()` | ~20 | Lazy-loading EasyOCR con `threading.Lock()`. **GPU prioritario** (torch.cuda.is_available() → gpu=True), CPU fallback automático si CUDA no disponible o hay error. **Importante**: el orden de carga respecto a CT2 es crítico — EasyOCR debe cargar PRIMERO para inicializar torch.cuda antes que CT2 cargue sus DLLs cuDNN. Verificado: GTX 1050 Ti, ~0.88s/pág vs 5s CPU (5.7x). No cargar en hilo secundario en Windows. |
| `_detect_and_ocr()` | ~160 | Parámetros actuales: `text_threshold=0.18`, `low_text=0.12`, `min_size=8`, `mag_ratio=1.2`, `canvas_size=2500`. **2 niveles** de fallback: directo → CLAHE+sharpen. CTD eliminado (dependencia externa frágil, ~84MB ahorrados). Acepta `allow_fallback` (desactiva CLAHE en modo easyocr-only). |
| `_group_and_merge_blocks()` | ~260 | **⚠️ Bug histórico corregido**: los patrones `WATERMARK_PATTERNS`, `MARGIN_NOISE_PATTERNS` y URL ahora se verifican contra el texto ORIGINAL del OCR (**antes** de limpiar símbolos). Antes se verificaban después de `re.sub(r'[/.,:;...]', ...)` que destruía `/`, `.`, `,` — los caracteres que necesitan las fechas/horas ("13/7/26", "4.58 p.m") para matchear. **9 filtros post-merge**: números puros, patrones numéricos, comillas, puntuación suelta, aspecto estrecho, chars sueltos, baja confianza, dígito+letra. Fusión horizontal con gap tolerante `max(35, w*2.5)`. |
| `_build_inpaint_mask()` | ~390 | Para globos de diálogo usa máscara de solo-glifos (preserva forma del globo). Para texto flotante usa rectángulo completo. |
| `_pre_filter_image()` | ~120 | Filtro pre-OCR con morfología OpenCV. Franjas 4% superior/inferior + líneas horizontales. |

---

### Secciones sincronizadas entre frontend y backend

| Componente | Ubicaciones | Riesgo |
|---|---|---|
| `MARGIN_NOISE_PATTERNS` | `app.js:~809` + `config.py` | Deben ser idénticos. Divergencia = texto basura o diálogos eliminados. |
| `GLOBAL_NOISE_PATTERNS` / `WATERMARK_PATTERNS` | `app.js:~828` + `config.py` | Misma razón: sincronización obligatoria. |
| `state.inpaintedBgByPage` | `app.js:~119` + `routes/api.py` response | Servidor devuelve base64 PNG; frontend lo convierte a `Image` y lo guarda en Map. |

---

## 3. Zonas Seguras de Editar Libremente ✅

- **`styles.css`**: Colores, animaciones, variables CSS de tema. No afecta lógica.
- **`index.html`**: Añadir botones/campos si no cambian IDs usados por `app.js`.
- **`setStatus()`, `showProgress()`, `formatDuration()`** en `app.js`: Solo UI de estado/progreso.
- **`start-app.ps1`**: Puerto, URL de apertura, browser.
- **`initTheme()` / `toggleTheme()`**: Añadir variantes de tema.
- **`showToast()`**: Cambiar duración, estilos, animaciones.
- **`initKeyboardShortcuts()`**: Añadir/quitar atajos (Ctrl+letra o teclas simples).
- **Diccionario `spa_words` en `_detect_language_simple()`**: Añadir palabras spanish.
- **`cache.py`**: Ajustar TTL, `MAX_CACHE_ENTRIES`, o estrategia de evicción.
- **`ratelimit.py`**: Cambiar límites de rate limiting.

---

## 4. Estado Actual

**Última actualización**: 2026-07-25

### Cambios acumulados (Julio 2026)

#### Sesión 2026-07-22 — Optimizaciones de velocidad + CTD + fix SIN_TRAD + calidad real (10 cambios)

Tres optimizaciones que reducen el tiempo por página de ~35-50s a ~10-18s.

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 22 | **OCR 2-3x más rápido**: canvas_size limitado a 2000px (antes 4096+), text_threshold 0.25 (antes 0.15), low_text 0.15 (antes 0.10), min_size 10 (antes 8), mag_ratio 1.0 (antes 1.2). EasyOCR procesa menos píxeles y produce menos falsos positivos → menos bloques a traducir. | `ocr_utils.py` | **2-3x** por página |
| 23 | **Sin pre-OCR para detectar idioma**: cuando source="auto", se eliminó el OCR al 25% que solo servía para detectar idioma (~5-10s extra). Ahora se asume "es" temporalmente y se corrige post-OCR. **Bug corregido**: la condición `detected_lang == "auto"` nunca se cumplía porque ya era "es" — ahora usa `source_lang == "auto"`. | `routes/api.py` | **-5-10s** por página (source=auto) |
| 24 | **Traducción paralela**: los 3 motores (CT2, Argos, Google) se lanzan simultáneamente con `ThreadPoolExecutor` + `as_completed`. Se acepta el PRIMER resultado válido inmediatamente. Tiempo por bloque: del más lento → al más rápido con resultado válido (~53ms en vez de ~4s). | `translator.py` | **5x** por bloque |

#### Sesión 2026-07-22 — Scripts paralelizados (128 páginas)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 25 | **Procesamiento de páginas en paralelo**: `process_all_pages.py` reescrito con arquitectura productor-consumidor (1 render thread + N API workers). Checkpoint cada 10 páginas thread-safe. Tiempo 128 páginas: ~75-100 min → ~20-35 min. N=3 por defecto (--workers CLI, punto óptimo benchmark: 2 es seguro, 4 causa timeouts). | `process_all_pages.py` | **3x** total |
| 26 | **Bug fix**: `#` literal antes de URL en API call (`f"#{API_URL}/api/process-page"`) invalidaba TODAS las llamadas — 0 páginas procesadas. Corregido a `f"{API_URL}/api/process-page"`. También eliminado `import threading` duplicado. | `process_all_pages.py` | Bug crítico ✅ |
| 27 | **Bug fix `is_lenient`**: cuando un motor devuelve `trad == orig` en modo lenient (≤3 palabras), ahora verifica con `_detect_language_robust()`. Si el texto sigue en idioma origen → rechazar y probar siguiente motor. Si ya está en destino → aceptar. Antes aceptaba cualquier resultado idéntico. | `translator.py` | Bug crítico ✅ |
| 28 | **Modo CTD-only**: nuevo parámetro `ocr_mode` en `/api/process-page` (`auto`/`easyocr`/`ctd`). `use_ctd_only=True` en `_detect_and_ocr()` salta EasyOCR de imagen completa y va directo a CTD. `allow_fallback=False` en modo `easyocr` desactiva fallbacks. Ahorra ~1-2s en páginas donde EasyOCR consistentemente falla. **Default cambiado de `auto` a `ctd`** (benchmark: 2x más rápido, 93.9% cobertura vs 87.0%). | `ocr_utils.py`, `routes/api.py` | 🆕 Modo CTD |
| 29 | **Sincronización de patrones app.js ↔ config.py**: eliminados `olympus|scanlation|zonaolympus|scan_group` de `GLOBAL_NOISE_PATTERNS` y `capítulo|cómo criar|how to raise` de `MARGIN_NOISE_PATTERNS` en `app.js` para coincidir con `config.py`. Verificado con `node --check`. | `app.js`, `config.py` | 🆕 Sincronizado |
| 30 | **Fix SIN_TRAD — Google retry con backoff**: cuando todos los motores (CT2, Argos, Google) fallan en paralelo, se reintenta Google 3 veces con backoff progresivo (5s, 15s, 30s), reseteando el rate limit entre intentos. También se eliminaron 2 líneas de código muerto duplicadas al final de `_translate_one()`. | `translator.py` | 🆕 Fix SIN_TRAD |
| 31 | **Métrica de calidad real**: análisis de 723 bloques traducidos muestra 75.8% aceptable (BUENA + LITERAL + OCR_NOISY), 15.9% OCR_GARBAGE (running headers principalmente), 8.3% UNTRANSLATED. Documentado en CODEGRAPH.md como referencia de calidad. | `CODEGRAPH.md` | 📊 Benchmark real |
| 32 | **Fix running headers**: el symbol cleaning (`re.sub` removiendo `/`, `.`, `,`, `:`) se ejecutaba ANTES de verificar MARGIN_NOISE_PATTERNS, destruyendo patrones de fecha/hora como "13/7/26" y "4.58 p.m". Movido el filtro ANTES del cleaning. | `ocr_utils.py` | 🐛 Fix bug crítico |

#### Sesión 2026-07-24 — Defaults optimizados + CT2 18 pares CJK

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 33 | **Default workers 3** (antes 2): benchmark demostró que 3 workers es el punto óptimo (4 causa timeouts en páginas pesadas). Tiempo 128 págs CTD: 4.6 min con 3 workers vs 5.8 min con 2. | `process_all_pages.py` | **-20%** tiempo |
| 34 | **Default ocr_mode `ctd`** (antes `auto`): benchmark demostró que CTD es 2x más rápido (4.6 min vs 9.3 min) con mejor cobertura (93.9% vs 87.0%). El pipeline híbrido (EasyOCR con fallback CTD) sigue disponible explícitamente con `--ocr-mode auto`. | `routes/api.py`, `process_all_pages.py` | **2x** más rápido |
| 35 | **Allow_fallback semánticamente correcto**: cuando `ocr_mode="ctd"`, `allow_fallback=False` explícitamente (antes era True pero inofensivo porque `use_ctd_only` saltaba el pipeline antes de llegar a `allow_fallback`). | `routes/api.py` | 🧹 Cleanup |
| 36 | **CT2 18 pares CJK**: agregados ja|en, en|ja, ko|en, en|ko, zh|en, en|zh a `_CT2_MODELS`. Prueba ja→en con japonés real (kanji+kana): 10/10 traducciones exitosas en GPU (CUDA, int8). Modelos lazy-load: descarga+conversión automática en primer uso. Pipeline CT2 (busca `f"{source}|{target}"` en el dict) funciona sin cambios — `_detect_language_robust()` ya retorna ja, ko, zh. | `translator.py` | 🆕 CT2 CJK |

#### Sesión 2026-07-25 — Aceleración GPU dual + profiling cProfile inline

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 37 | **CT2 preload en background**: precarga modelos es→en y en→es al arrancar el servidor (hilo daemon). Evita el cold start de ~21.5s (carga del modelo de 300MB a GPU) en la primera traducción. Las traducciones pasan de ~2.8s a **0.12s**. | `server.py` | **~23x** primera traducción |
| 38 | **Executor compartido en translator**: reemplaza `with ThreadPoolExecutor(...)` que creaba/destruía 3 threads por cada traducción (~1.857 ciclos). Nuevo executor compartido 4 threads con lazy init + double-checked locking. | `translator.py` | **Sin overhead de threads** + **retorno inmediato** |
| 39 | **.exe optimizado**: excluye módulos pesados del bundle (torch, transformers, ct2, easyocr, etc.) vía `EXCLUDES`. .exe: **200MB** (antes 2.6GB). Build: **3.75 min** (antes 10+ min). | `main.spec` | **13x más pequeño** + **~5x arranque** |
| 40 | **GPU dual: EasyOCR + CT2 ambos en GPU**: reemplazado `force_cpu=True` por auto-detect CUDA en `_get_ct2_translator()`. Orden precarga: EasyOCR primero → CUDA, luego CT2. Sin crash cuDNN. 5.7x OCR (0.88s) + ~6x CT2 (0.048s). | `server.py`, `translator.py`, `ocr_utils.py` | **~6x** CT2 + **5.7x** OCR |
| 41 | **cProfile inline decorator**: `@profile_endpoint` en routes/api.py activable con `?profile=1`. Guarda .prof, log consola top-5 cumtime, header X-Profile. Post-procesamiento protegido, fast-pass sin overhead. | `routes/api.py` | 🆕 Profiling server-side |
| 42 | **py-spy evaluado**: no funciona en Windows sin SeDebugPrivilege (incluso con `--`). Alternativas: cProfile inline, profile_standalone.py, Windows ETW. | `profile_pyspy.py` | 📊 Diagnóstico |

#### Sesión anterior — CI unificado + calidad de traducción

| # | Cambio | Archivo |
|---|--------|---------|
| 21 | Eliminado catch-all en is_onomatopoeia() | `analisis_calidad.py` |

#### Sesión 2026-07-21 — Refactor COMPLETO de server.py

El refactor en módulos (`config.py`, `translator.py`, `ocr_utils.py`) fue completado el 2026-07-22. `server.py` pasó de ~1150 líneas a **141 líneas**, importando toda la lógica de los módulos.

| # | Cambio | Archivo |
|---|--------|---------|
| 1 | **Refactor server.py → 3 módulos**: `config.py`, `translator.py`, `ocr_utils.py`. Server.py ahora es entry point (~141 líneas). Eliminadas ~1000 líneas de código duplicado inline. | `server.py`, `config.py`, `translator.py`, `ocr_utils.py` |
| 2 | **Cache wrapper**: `_translate_one()` inyecta `cache_get`/`cache_set` en `translator._translate_one()` | `server.py` |
| 3 | **Eliminado MIT_AVAILABLE**: Bloque completo + `manga_pipeline.py` (17KB código muerto) | ~~`server.py`~~, ~~`routes/api.py`~~, ~~`manga_pipeline.py`~~ |
| 4 | **Eliminado _GLOSARIO_POST**: 9 reglas específicas de un solo manga | ~~`server.py`~~ (original) |
| 5 | **Eliminado _preload_models()**: Función comentada y su invocación | `server.py` (original) |
| 7 | **OpenCV.js callback**: Reemplazado polling `setInterval` por `cv['onRuntimeInitialized']` con 3 casos + timeout 15s | `app.js` |
| 8 | **CDN OpenCV.js**: Cambiado de `docs.opencv.org` a `cdn.jsdelivr.net` (@techstark/opencv-js) | `index.html` |
| 9 | **CSP actualizado**: Reemplazado `docs.opencv.org` por `cdn.jsdelivr.net` | `config.py`, `index.html` |
| 10 | **Fix race condition**: `_get_google_session()` ahora usa double-checked locking correctamente | `translator.py` |
| 11 | **Fix fused line**: Línea 2323 — comentario y código fusionados causaban SyntaxError (app congelada en "Cargando OpenCV...") | `app.js` |
| 12 | **Limpieza proyecto**: ~25 archivos temporales eliminados, 12 tests archivados | — |
| 13 | **.gitignore completo**: 14 secciones cubriendo Python, ML models, debug, checkpoints | `.gitignore` |
| 14 | **run_ci.ps1 + pre-commit hook**: CI local con syntax checks + tests + stress opcional | `run_ci.ps1`, `.git/hooks/pre-commit` |
| 15 | **run_ci.py — CI unificado Python**: Reemplaza run_ci.ps1 como metodo principal. Unifica syntax check + test_ci + servidor + analisis_calidad + stress test en un solo comando. Sin dependencia de PowerShell. Compatible con Windows nativamente. | `run_ci.py` |
| 16 | **Optimizaciones de performance**: GLOSARIO_REGEX pre-compilado, _SPA_WORDS como constante global, GoogleTranslator cacheado por par de idioma con double-checked locking | `config.py`, `translator.py` |
| 17 | **Filtro pre-Argos (OCR noise)**: _es_ocr_noise() detecta texto ruidoso y salta Argos (~3s ahorrado por texto ruidoso) | `translator.py` |
| 18 | **Glosario post-CT2**: 6 reglas de correccion post-traduccion, arregla "TEMPORARY 7" → "SEASON 7" | `config.py`, `translator.py` |
| 19 | **Validación de traducción robusta**: Reemplazado hardcode `"mainstremainstre"` por regex `_REPEATED_CHUNK_PAT` que detecta cualquier fragmento repetido pegado (4-20 chars) sin depender de longitud total ni posición inicial. | `translator.py` |
| 20 | **Flag ENCODING_ROTO en analizador**: Reemplazado `SIN_MARCA` (`"[!]" not in t`, inútil ~100% de bloques) por `ENCODING_ROTO` (`"\ufffd" in t`) que detecta corrupción de encoding real. | `analizar_traduccion.py` |
| 21 | **Eliminado catch-all en is_onomatopoeia()**: Ya no clasifica cualquier palabra mayúscula suelta como SFX; esos casos caen a UNTRANSLATED para revisión manual. | `analisis_calidad.py` |

#### Sesión 2026-07-25-v2 — CTD eliminado + default easyocr + lenient fix (3 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 43 | **CTD eliminado completamente**: `ctd_lib/` (100KB), `ocr_ctd_fallback.py` (218 líneas), `models/ctd/comictextdetector.pt` (77MB), y 3 dependencias pip (pyclipper, shapely, einops ≈7MB). Pipeline OCR simplificado de 3 a 2 niveles (EasyOCR → CLAHE). `choices=["auto","easyocr"]`, sin modo `ctd`. Ahorro total: **~84MB**, 2690 líneas menos. | `ctd_lib/`, `ocr_ctd_fallback.py`, `ocr_utils.py`, `routes/api.py`, `process_all_pages.py`, `main.spec`, `requirements.txt` | **~84MB** + **~2.7K LOC** |
| 44 | **Default OCR cambiado de `ctd` a `easyocr`**: el modo `auto` tomaba ~72 min por fallback CLAHE en páginas densas. `easyocr` (solo GPU) tarda **7.1 min** con workers=4, 100% cobertura, 0 errores. Modo `auto` sigue disponible como opt-in. | `process_all_pages.py`, `routes/api.py` | **10x** más rápido (72 min → 7.1 min) |
| 45 | **Fix textos cortos — OCR garbage + lenient mode**: `_es_ocr_noise()` filtros 6-7 atrapan Q/N/ze (≤3 chars sin vocal). `_resultado_valido()` reordenado para aceptar nombres/SFX en lenient mode primero. Tasa traducción esperada: 82.8% → ~90%+. | `translator.py` | **+~7%** tasa traducción |

#### Sesiones anteriores (2026-07-14 a 2026-07-20)

_[46 correcciones de bugs: renderToken, hasServerInpainted, export canvases, base64 validation, glyph mask try/finally, Tesseract getter/setter, pdfPage.cleanup(), renderTask.cancel(), patrones de ruido unificados, mag_ratio adaptativo, fusión vertical, filtros OCR post-merge, traducción leniente para texto corto (1-3 palabras), etc.]_

---

## 5. Cómo actualizar el ejecutable (.exe)

Cada vez que se modifique el código Python o frontend que deba distribuirse como `.exe`, hay que **recompilar** con PyInstaller.

### Requisitos
- PyInstaller instalado en `env/`: `pip install pyinstaller`
- `main.spec` — configuración de compilación (ya existe y está actualizada)
- `main.py` — entry point del .exe (modo launcher con ocultación de consola)

### Comando para recompilar

```powershell
cd D:\crear traductor
.\env\Scripts\python.exe -m PyInstaller main.spec --clean --noconfirm
```

Esto genera:
- `dist/main/main.exe` — ejecutable sin ventana CMD (`console=False`)
- El .exe empaqueta el código fuente Python + frontend (HTML, JS, CSS)
- Las dependencias pesadas (EasyOCR, torch, ArgosTranslate, OpenCV, etc.) se cargan **desde `env/` en tiempo de ejecución**, no van dentro del .exe

### Después de compilar

1. Copia `dist/main/main.exe` al directorio de distribución (ej. escritorio)
2. Asegúrate de que `env/` esté presente junto al .exe (es donde busca las dependencias)
3. Verifica que arranque correctamente con doble click

### Notas importantes

- **No usar `--onefile`**: el modo onefile extrae todo a un temp dir cada vez, lo que rompe la carga de EasyOCR/torch por los paths absolutos. Usamos el modo `onedir` (carpeta `dist/main/`).
- **Siempre compilar desde el entorno virtual** (`env/Scripts/python.exe -m PyInstaller`), no desde el Python del sistema.
- Si se añaden nuevos archivos Python o frontend, hay que actualizar `DATAS` y `HIDDEN_IMPORTS` en `main.spec`.
- El spec `main.spec` usa `console=False` para que el .exe no muestre ventana CMD.
- `main.py` tiene `_hide_console()` como fallback adicional para ocultar la consola al arrancar.

## 6. Cómo ejecutar el CI local

El CI (Integración Continua) verifica que el proyecto esté sano después de cambios.

### Pre-commit hook (se ejecuta automáticamente en cada `git commit`)

El hook verifica sintaxis de Python (con `py_compile`) y JavaScript (con `node --check`)
antes de permitir el commit. Usa el Python del entorno virtual (`env/Scripts/python.exe`)
para evitar el stub de Microsoft Store. Si `node` no está en PATH, salta la verificación JS.

Ubicación: `.git/hooks/pre-commit`

### Qué verifica

| Paso | Test | Qué comprueba |
|---|---|---|
| 1 | Syntax check | `py_compile` en todos los Python files |
| 2 | `test_ci.py` | Detección de idioma (langdetect + heurísticas) |
| 3 | Iniciar servidor | `/api/health` + endpoints translate, batch, config, static |
| 4 | `analisis_calidad.py` | Calidad de traducciones vs corpus de referencia |
| 5 (opcional) | `stress_test_memory.py` | 50 páginas bajo carga (con `--full`) |

### Archivos verificados por el CI

```
server.py, config.py, translator.py, ocr_utils.py,
routes/api.py, routes/main.py, cache.py, models.py,
ratelimit.py, main.py, process_all_pages.py
```

> **Nota**: `ocr_ctd_fallback.py` y `ctd_lib/` fueron eliminados en Julio 2026 (dependencia CTD frágil, ~84MB).

### Cómo ejecutar (recomendado)

```bash
cd D:\crear traductor
.\env\Scripts\python.exe run_ci.py           # Tests rápidos (~30s)
.\env\Scripts\python.exe run_ci.py --full     # Tests completos (~10 min, incluye stress test)
.\env\Scripts\python.exe run_ci.py --help     # Ayuda completa
```

Flags disponibles:
- `--full`: Incluye stress test de 50 páginas (~10 min)
- `--server`: Solo inicia servidor + health check (para desarrollo)
- `--skip-syntax`: Omitir syntax check (si ya verificaste manualmente)

### Alternativa legacy (run_ci.ps1)

```powershell
cd D:\crear traductor
.\run_ci.ps1           # Tests rápidos (~30s)
.\run_ci.ps1 -Full     # Tests completos (~10 min)
```

> **Nota**: `run_ci.py` es el CI recomendado. `run_ci.ps1` es legacy y puede fallar por encoding en algunas configuraciones de Windows.

### Cuándo ejecutar el CI

- **Siempre** después de cambiar `server.py`, `translator.py`, `ocr_utils.py`, `routes/api.py`, `config.py`
- **Siempre** antes de compilar el .exe (ver §5)
- **Siempre** antes de hacer commit (el pre-commit hook verifica syntax automáticamente)
- **Con `--full`** antes de un release o después de cambios grandes en el pipeline de traducción

### Si el CI falla

1. Revisa el paso que falló (syntax, tests, servidor)
2. Para errores de sintaxis: `python -m py_compile <archivo>` te da la línea exacta
3. Para errores de JavaScript: `node --check app.js`
4. Para error del servidor: revisa `ci_server.log` en la raíz del proyecto
5. Si `test_ci.py` falla: revisa `_detect_language_simple()` o `_detect_language_robust()` en `translator.py`
6. Si el pre-commit hook falla porque no encuentra Python, verifica que `env/Scripts/python.exe` exista. Si el hook usa `sys.executable` por fallback, asegúrate de que no sea el stub de Microsoft Store (desinstalar desde Configuración > Aplicaciones > Alias de ejecución).

## 7. Flujo de Trabajo Recomendado

1. **Lee AGENTS.md** antes de modificar cualquier archivo del proyecto.
2. Si tocas una **zona sensible**, documenta el motivo.
3. **Para cambios en Python**: ejecuta `python -m py_compile` en todos los archivos modificados.
4. **Para cambios en app.js**: ejecuta `node --check app.js`.
5. **Ejecuta el CI** (ver §6) antes de considerar un cambio completo.
6. **Actualiza el .exe** con PyInstaller después de cambios confirmados (ver §5).
7. **Actualiza §4 (Estado Actual)** con lo que se hizo y queda pendiente.

---

## 8. Protocolo de Sesión

- **Al inicio**: Lee AGENTS.md y CODEGRAPH.md antes de cualquier código.
- **Durante**: Si el contexto se agota, detente y actualiza §4 con el estado parcial.
- **Objetivo**: Que la siguiente sesión pueda retomar sin perder contexto.
