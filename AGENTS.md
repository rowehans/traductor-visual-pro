> ⚠️ **LEE ESTE ARCHIVO ANTES DE MODIFICAR CUALQUIER ARCHIVO DEL PROYECTO.**

# AGENTS.md — Guía de contexto para IAs y colaboradores

## 1. Descripción del proyecto

**Traductor Visual Pro** — Aplicación local para traducir manga, cómics y documentos en PDF e imagen.

### Arquitectura actual (post-refactor Julio 2026)

| Archivo | Rol | Tamaño |
|---|---|---|
| `app.js` | Frontend completo (~2900 líneas): renderizado PDF/imagen vía pdf.js, editor de burbujas (draw/move/resize), OCR delegado al servidor, filtros de bloques (importados de js/filters.js), **drawTextOnCanvas() compartida** (unifica renderBoxes + drawProfessionalText), **glow exterior** neón configurable, **relleno semitransparente** (fillOpacity), layout de texto en canvas (wrap+fit), comunicación con API Flask (ES6 modules), exportación PNG/PDF, tema oscuro/claro, atajos de teclado (incl. G para glow), toasts, carga asíncrona de OpenCV.js vía callback. **Botón ⏸️ Pausar/▶️ Reanudar** en barra de progreso (toggle `state.translationPaused`). **Aviso ⚠️ origen==destino** (función `checkLanguageWarning()` con mapa `isoToSelector`). **Selector Motor OCR** (`#ocrEngine`) con modo Automático (fusion) y **badge de estado del daemon U-OCR** (`#ocrEngineStatus` + `updateOcrEngineStatus()` con polling 20s). **JS modularizado**: `import` desde js/config.js, js/theme.js, js/toast.js, js/filters.js, js/utils.js. | 112KB |
| `server.py` | Entry point Flask (~255 líneas). Importa de config.py/translator.py, envuelve _translate_one() con caché, **precarga EasyOCR + CT2 en background** (orden crítico: EasyOCR primero para inicializar CUDA, luego CT2 auto-detecta GPU) y **lanza el daemon U-OCR** (`_preload_unlimited_daemon`). **Logs en tiempo real**: `sys.stdout.reconfigure(line_buffering=True)` + `PYTHONUNBUFFERED=1` + `python -u` para que `print()` se escriba inmediatamente en `server_output.log`. Puerto 5174. | 12KB |
| `config.py` | Constantes globales: paths, `LANGUAGES`, `CSP_POLICY`, patrones de ruido (`MARGIN_NOISE_PATTERNS`, `WATERMARK_PATTERNS`), glosario pre-OCR (`GLOSARIO_PRE`), constantes de fusión (`UOCR_TRIGGER_*`, `OCR_ENGINE_WEIGHTS`, `RAPID_AGGRESSIVE_PARAMS`, `FUSION_TYPE_*`, `UOCR_CACHE_*`). | 16KB |
| `translator.py` | Lógica de traducción: detección de idioma (`_detect_language_robust`), **pipeline secuencial CT2→Google→SIN_TRAD** (antes: 3 motores en paralelo con 30s timeout + 50s retry = hasta 80s por texto. Ahora: CT2 síncrono ~0.02s en GPU, Google fallback ~2s, SIN_TRAD inmediato). Validación de traducción (6 validaciones anti-basura), glosarios PRE/POST, filtro pre-Argos para OCR noise. Cache injectado desde server.py. ~80 líneas de código muerto eliminadas (executor compartido, función `_probar_motor`, `translation_fns`). | 44KB |
| `ocr_utils.py` | OCR con EasyOCR (GPU prioritario, CPU fallback automático si CUDA no disponible), **corrector ortográfico automático con pyspellchecker** (86K palabras, 0 mantenimiento manual, reemplaza _OCR_DICT manual de 600 palabras), pre-filtro de imagen, inpainting con OpenCV (INPAINT_NS), detección de globos de diálogo, máscara de glifos, sampleo de color, fusión y filtrado de bloques (9 filtros post-merge), **RapidOCR** (tier híbrido), **fusión multi-motor** (`_fusionar_blocks_multi`), firma de página (`_page_signature`) y re-OCR de globos (`_recover_regions_with_easyocr`). **Pipeline híbrido 3 niveles**: EasyOCR directo → reintento mag_ratio=1.8 con conf baja → RapidOCR (siempre, complementa) → CLAHE+sharpen como último fallback. **Optimizado**: canvas_size=2500px, text_threshold=0.15, mag_ratio=1.3 (adaptativo a 1.8). **GPU**: GTX 1050 Ti verificado ~0.88s/pág vs 5s CPU (5.7x). | 76KB |
| `ocr_engine.py` | **OCRManager** (Ago 2026): orquesta los 3 motores en una clase única — `run_ocr()` dispatcher, `_compute_trigger` (v4.2), `_reforzar_con_rapid_agresivo` (Fase 2), `_reforzar_con_unlimited` + Ruta C, cache de decisiones negativas §8.4.1. Acceso a ocr_utils/routes.api EN RUNTIME (los mocks de pytest funcionan). | 20KB |
| `uocr_daemon.py` + `uocr_client.py` | Daemon persistente Unlimited-OCR (puerto 5177, venv `env_uocr_gpu`, modelo 3B 4-bit NF4 ~2.25GB VRAM): `POST /ocr` (1 página) y `POST /ocr-batch` (Fase 1, `infer_multi` con separador `<PAGE>`), salida `<|det|>type [x,y,w,h]<|/det|>` por stdout. Cliente stdlib puro: `health()`, `wait_ready()`, `process_page()`, `process_batch()`. | 24KB + 12KB |
| `routes/api.py` | Blueprint REST: `/api/health`, `/api/translate`, `/api/translate-batch`, `/api/process-page`. **Expone `ocr_mode`** en `/api/process-page` (default `"fusion"` desde Ago 2026 — Fase 4): `"fusion"` (EasyOCR+RapidOCR siempre + Unlimited-OCR solo si la página es difícil, vía `OCRManager` de `ocr_engine.py`), `"unlimited"` (daemon forzado), `"easyocr"` (solo EasyOCR GPU), `"auto"` (legacy, EasyOCR + CLAHE fallback). El bloque OCR de `process_page` delega en `OCRManager().run_ocr(...)`. **Incluye decorador `@profile_endpoint`** para profiling cProfile inline activable vía `?profile=1`. Importa directamente de los submódulos. | 40KB |
| `routes/main.py` | Blueprint de rutas estáticas con protección contra path traversal. | 1KB |
| `index.html` | UI HTML (~454 líneas): estructura del editor visual, CSP vía `<meta>`, detección de Brave Leo/Shields, selector Motor OCR (`#ocrEngine`) y badge de estado del daemon (`#ocrEngineStatus`). | 24KB |
| `styles.css` | Estilos visuales premium (tema dark/light con variables CSS, glassmorphism, animaciones, responsive, clases del badge `.ocr-engine-status`). | 48KB |
| `cache.py` | Caché de traducciones en filesystem con TTL (7 días) y LRU eviction (5000 entradas máx). | 2KB |
| `models.py` | Modelos SQLAlchemy (User, Project, Page, TextBlock) con repositorios. SQLite local / PostgreSQL producción. | 7KB |
| `ratelimit.py` | Rate limiting con Flask-Limiter. Evita imports circulares entre server.py y routes/. | 1KB |
| `start-app.ps1` | Lanzador: inicia `env\Scripts\python.exe server.py` y abre `http://127.0.0.1:5174`. | 1KB |
| `run_ci.py` | **CI unificado** — ejecuta syntax check + test_ci.py + servidor + analisis_calidad + stress test en un solo comando Python. No depende de PowerShell. `python run_ci.py --full` para test completo (~10 min). | 20KB |
| `requirements.txt` | Dependencias Python pineadas. | 1KB |
| `main.py` | Entry point del ejecutable (.exe). Modo launcher: oculta consola, inicia Flask, abre Chrome. Modo `--server`: solo servidor. | 4KB |
| `main.spec` | PyInstaller spec para compilar `main.exe` con `console=False` (sin ventana CMD). **Incluye carpeta `js/`** (5 módulos ES importados por `app.js`: config.js, utils.js, toast.js, theme.js, filters.js). CTD eliminado del bundle. **Incluye `ocr_engine.py`, `uocr_client.py`, `uocr_daemon.py` en DATAS** (imports dinámicos) — NUNCA en hiddenimports (el .exe crece a 2.4GB). **Excluye módulos pesados** (torch, transformers, ct2, easyocr, ultralytics, tqdm, onnxruntime, rapidocr_onnxruntime, etc.) — se cargan desde `env/` en runtime. **Stdlib frozen** (unittest.mock, modulefinder, plistlib, filecmp, shelve, PIL.ImageEnhance) en hiddenimports. **UPX deshabilitado**. .exe resultante: **343MB**. | 4KB |
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
| `loadPdfJs()` | ~493 | **Carga dual de PDF.js**: intenta primero `import()` de `.mjs` (v4.10.38), y si falla, carga script UMD clásico `pdf.js` v3.11.174 como fallback con 4 estrategias de CDN (cdnjs → jsDelivr → unpkg). |
| `wrapTextLines(ctx, text, maxWidth)` | ~1910 | **Bug histórico crítico**: si la condición `containsCJK()` cambia, palabras occidentales vuelven a separarse letra por letra. Solo separa carácter a carácter si el texto contiene CJK. |
| `MARGIN_NOISE_PATTERNS` / `GLOBAL_NOISE_PATTERNS` | **js/filters.js:12/30** (app.js:7 los importa) | Deben estar sincronizados con `MARGIN_NOISE_PATTERNS`/`WATERMARK_PATTERNS` en `config.py`. Divergencia causa textos basura o diálogos eliminados. |
| `state` (incl. `inpaintedBgByPage` Map) | ~121 | Caché global de estado. `inpaintedBgByPage` guarda imágenes inpaintadas por página. Si no se escribe aquí tras `serverProcessPage()`, el texto original reaparece. |
| `initOpenCv()` | ~143 | **Race condition fix (Jul 2026)**: check diferido de 100ms tras asignar `onRuntimeInitialized` en Caso 2. Cubre el escenario donde el WASM de OpenCV.js ya se inicializó antes de que se asignara el callback. Guard `state.cvLoaded` evita doble llamada. 3 casos: cv ya cargado con Mat, cv existe sin Mat (con callback + deferred check), cv no existe (polling 200ms + timeout 15s). |
| `autoTranslateCurrentPage(pageNo, ...)` | ~1363 | Camino único: servidor. `serverProcessPage()` → guarda `inpaintedBgByPage` → carga `erasedBgCanvas` → filtra bloques → `makeAutoTextBox` → render. |
| `renderPage()` | ~648 | Usa `_renderToken` para cancelación, `_renderTempCanvas` global reutilizado, `renderTask.cancel()` en timeout. Retorna `{aborted: bool}`. |
| `filterPageBlocks(blocks, pageHeight)` | **js/filters.js:47** (app.js la usa en ~893/1416) | Único punto que filtra bloques antes de crear cajas. `marginTop` = 7% de altura (desde la modularización vive en `js/filters.js`, no en app.js). |
| `drawTextOnCanvas(ctx, text, box, layout)` | ~1696 | **Función compartida** que unifica el dibujo de texto entre `renderBoxes()` y `drawProfessionalText()`. Orden de dibujo: (1) relleno semitransparente, (2) glow exterior con `shadowColor` + `fillStyle="transparent"`, (3) sombra de legibilidad (si no hay glow), (4) contorno (`strokeWidth * 2`), (5) texto principal. Cualquier cambio visual debe hacerse aquí y se refleja en pantalla y exportación. |
| `exportFullPdf()` | ~2418 | Genera PDF completo página por página: renderRawPdfPage → renderEditedCanvas → jsPDF.addPage. |
| `getSelectedOcrMode()` / `updateOcrEngineStatus()` | ~1231 / ~1247 | Lee el selector `#ocrEngine` (default "fusion") y actualiza el badge del daemon con polling 20s mientras el modelo carga. `_uocrStatusTimer` global; `_stopUocrPolling()` para pararlo. |

### `server.py` (entry point)

| Elemento | Línea | Por qué es delicado |
|---|---|---|
| `_preload_background()` | ~76 | **ORDEN CRÍTICO**: EasyOCR primero (inicializa CUDA), luego CT2 (auto-detecta GPU). NO INVERTIR. Si CT2 carga primero, sus DLLs cuDNN conflictúan con las de PyTorch → crash "Could not load symbol cudnnGetLibConfig". Verificado: GTX 1050 Ti, ambos en GPU, 128MB VRAM usados, sin crash. También lanza `_preload_unlimited_daemon` (daemon U-OCR, hilo independiente). |
| `_translate_one()` | ~178 | Wrapper que inyecta `cache_get`/`cache_set` en `translator._translate_one()`. **Debe importarse desde `server` en routes/api.py** — NO desde `translator.py` directo. |
| `from config import ...` | ~19 | Importaciones controladas. No agregar imports circulares. |

### `translator.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_translate_one()` | ~802 | **Pipeline secuencial optimizado** (CT2 síncrono → Google fallback → SIN_TRAD inmediato). **Antes**: 3 motores en paralelo con `as_completed(timeout=30s)` + Google retry 5+15+30s = hasta 80s por texto. Cuando todos los motores fallaban (timeout por contienda de workers), SIN_TRAD devolvía el original. **Ahora**: CT2 primero (síncrono, ~0.02s en GPU). Si CT2 devuelve traducción válida → retorno inmediato. Si falla → Google fallback (~2s). Sin esperas, sin colas de workers. Acepta `cache_get`/`cache_set`/`translation_cache_available` para inyección. **Código muerto eliminado**: `_get_translate_engine_executor()`, `_probar_motor()`, `translation_fns`, lógica `_es_ocr_noise()` para construcción de lista de motores. |
| `_get_google_session()` | ~91 | Double-checked locking con `_google_session_lock`. La sesión HTTP se crea **dentro del lock**. |
| `_detect_language_robust()` | ~249 | langdetect thread-local + heurística `_detect_language_simple`. Mapeo zh-cn/zh-tw → zh. |
| `_ensure_argo_package()` | ~37 | Descarga modelos Argos con lock para evitar descargas duplicadas. |
| `_CT2_MODELS` (dict global) | ~617 | **18 pares de idiomas** (10 originales + 6 CJK: ja|en, en|ja, ko|en, en|ko, zh|en, en|zh + 2 legacy). Cada entrada mapea `"src|tgt"` a un modelo Helsinki-NLP OPUS-MT. **Si se agrega un par, debe coincidir exactamente** con los códigos ISO de `_detect_language_robust()` (ja, ko, zh). Modelos lazy-load: descarga + conversión CT2 en primer uso. No modificar las keys sin verificar que el pipeline `_get_ct2_translator()` siga funcionando (usa `f"{source}|{target}"` como lookup). |

### `ocr_utils.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_get_ocr_reader()` | ~45 | Lazy-loading EasyOCR con `threading.Lock()`. **GPU prioritario** (torch.cuda.is_available() → gpu=True), CPU fallback automático si CUDA no disponible o hay error. **Importante**: el orden de carga respecto a CT2 es crítico — EasyOCR debe cargar PRIMERO para inicializar torch.cuda antes que CT2 cargue sus DLLs cuDNN. Verificado: GTX 1050 Ti, ~0.88s/pág vs 5s CPU (5.7x). No cargar en hilo secundario en Windows. |
| `_get_spellchecker()` / `_ocr_spellcheck()` | ~413 / ~470 | **Corrector ortográfico post-OCR con pyspellchecker** (86K palabras, lazy-load thread-safe). Reemplaza el antiguo `_OCR_DICT` manual (600 palabras, mantenimiento infinito). `_get_spellchecker()` con double-checked locking, carga 16 palabras de dominio manga con alta frecuencia (`wf.add(word, 1000000)`). `_ocr_spellcheck()` llama a `sp.correction()` una vez fuera del loop. **Sin mantenimiento manual**. Fallback a `_levenshtein()` + `_FALLBACK_DICT` (~20 palabras) si pyspellchecker no está instalado. |
| `_group_and_merge_blocks()` | ~1261 | **⚠️ Bug histórico corregido**: los patrones `WATERMARK_PATTERNS`, `MARGIN_NOISE_PATTERNS` y URL ahora se verifican contra el texto ORIGINAL del OCR (**antes** de limpiar símbolos). Antes se verificaban después de `re.sub(r'[/.,:;...]', ...)` que destruía `/`, `.`, `,` — los caracteres que necesitan las fechas/horas ("13/7/26", "4.58 p.m") para matchear. **9 filtros post-merge**: números puros, patrones numéricos, comillas, puntuación suelta, aspecto estrecho, chars sueltos, baja confianza, dígito+letra. Fusión horizontal con gap tolerante `max(35, w*2.5)`. |
| `_build_inpaint_mask()` | ~1543 | Para globos de diálogo usa máscara de solo-glifos (preserva forma del globo). Para texto flotante usa rectángulo completo. |
| `_pre_filter_image()` | ~662 | Filtro pre-OCR con morfología OpenCV. Franjas 4% superior/inferior + líneas horizontales. |
| `_recover_regions_with_easyocr()` | ~1027 | Ruta C: re-OCR de globos con upscale 3.5×. **§8.4.4**: chequea `_uocr_inferring` ANTES de `_get_ocr_reader()` — si el daemon infiere degrada a RapidOCR CPU; maneja formato mixto dict/tupla (race window). |
| `_page_signature()` | ~1177 | Firma de layout (grid 8×8 de oscuridad + dark_ratio cuantizado) — la clave del cache de decisiones §8.4.1 del OCRManager. Calibrada: umbral de celda 0.05. |

---

### Secciones sincronizadas entre frontend y backend

| Componente | Ubicaciones | Riesgo |
|---|---|---|
| `MARGIN_NOISE_PATTERNS` | `js/filters.js:12` + `config.py` | Deben ser idénticos. Divergencia = texto basura o diálogos eliminados. |
| `GLOBAL_NOISE_PATTERNS` / `WATERMARK_PATTERNS` | `js/filters.js:30` + `config.py` | Misma razón: sincronización obligatoria. |
| `state.inpaintedBgByPage` | `app.js:135` + `routes/api.py` response | Servidor devuelve base64 PNG; frontend lo convierte a `Image` y lo guarda en Map. |

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

**Última actualización**: 2026-08-13

### Cambios acumulados (Agosto 2026)

#### Sesión 2026-08-06-v7 — Fix NameError en process_all_pages + --max-pages/--checkpoint-file + smoke test fusion CT2 (1 fix + 2 CLI)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 109 | **Fix NameError silencioso en `procesar_pagina`**: el refactor a `_registrar_resultado` (sesión 98) perdió la línea `data = resp.json()` en el camino SINGLE (batch-window=1) — la función usaba `data.get("blocks")` con `data` indefinido → el worker crasheaba en silencio y el script reportaba 0 páginas aunque el servidor SÍ procesaba (5 requests 200 OK en el log, checkpoint vacío). El camino batch (`procesar_lote`) sí tenía `data = resp.json()`, por eso F5/F6 con `--batch-window 4` funcionaban. Restaurada la línea + comentario explicativo. | `process_all_pages.py` | 🐛 **Fix: modo single ahora registra resultados** |
| 110 | **Nuevos CLI `--max-pages N` y `--checkpoint-file FILE`**: permite smoke tests cortos sin recorrer el capítulo y sin pisar el checkpoint del run F6 (`resultados_progreso.json` con 53/53 páginas). Defaults preservan el comportamiento previo. | `process_all_pages.py` | 🆕 Testeabilidad |

**Smoke test 5 páginas (fusion, workers=2)**: 23 bloques, 13 traducidos, 0 errores, 2.4 min. **Todas las traducciones reales vía `[ctranslate2 OK (fast path)]`** (6/6; los SIN_TRAD son ruido OCR no traducible como `Non-Text`/`REDUCIBLE`/URLs, esperado). **0 errores de red en todo el log** (sin `couldn't connect`, sin `[CT2] Error`). CT2 cargado desde `hf_cache` con `local_files_only=True` (confirmado el fix de la sesión 108 en servidor real). Se respaldó y restauró la caché de traducciones (1895 entradas) y el checkpoint F6 quedó intacto.

#### Sesión 2026-08-06-v10 — Fix métrica de tiempos: 'Tiempo total' ahora es pared real (no suma de elapsed heredados) (1 fix + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 115 | **Fix métrica de tiempos en `process_all_pages.py`**: el 'Tiempo total' del REPORTE FINAL pasaba de `sum(page_times)` (que en modo batch heredaba el elapsed COMPLETO de cada lote a CADA página — un lote de 4 págs de 587s sumaba 4×587s, inflando ~4-8x el total y confundiendo comparaciones batch-vs-single) a un **tiempo de pared real medido en `main()`** (`t_wall_start` antes de lanzar los workers → `wall_time` al final). Además `procesar_lote` ahora **reparte** el elapsed del lote entre sus páginas (`per_page_elapsed = elapsed/len(valid)`) para que `page_times`/checkpoint reflejen tiempos honestos, e imprime `LOTE [...] N págs en Xs (Ys/pág)`. El promedio usa `wall_time / páginas nuevas de esta corrida` (`n_paginas_iniciales` capturado tras cargar el checkpoint; `max(...,1)` evita div/0 si el checkpoint ya cubre todo). 3 tests nuevos (lote de 2 págs con time.time mockeado → `page_times == [5.0, 5.0]`, suma == elapsed del lote; lote de 1 pág; reporte final contiene 'pared real' y NO 'suma'). **19 tests** en verde. Los checkpoints viejos con page_times inflados no afectan al reporte (ya no se usa `sum(page_times)`). | `process_all_pages.py`, `tests/test_process_all_pages.py` | 🐛 **Métrica honesta** |

#### Sesión 2026-08-06-v11 — No-determinismo del trigger v4.2 ELIMINADO: device YOLO fijado por proceso + cache de decisión del trigger por firma (2 fixes + 10 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 116 | **Política determinista del trigger v4.2 (doble)**: la p4 disparaba U-OCR en single pero NO en batch porque `YOLO_DEVICE="auto"` resolvía GPU/CPU **por llamada** según el estado dinámico de `_gpu_lock`/`_uocr_inferring` (si EasyOCR de otro worker tenía el lock, YOLO degradaba a CPU; GPU vs CPU en ultralytics dan detecciones marginalmente distintas que cruzan el umbral 0.25 distinto → distinta Ruta C → distinto blocks/avg_conf → distinto trigger). Fix 1 — **`_resolver_device_yolo()` en `ocr_utils.py`**: resuelve el device UNA sola vez por proceso (módulo-global `_yolo_device` + lock), y con device `"0"` YOLO adquiere `_gpu_lock` de forma **BLOQUEANTE** (espera ~0.9-2s a EasyOCR de otro worker, timeout 30s) en vez de degradar a CPU por llamada — el device es SIEMPRE el mismo. Fix 2 — **cache de decisión del trigger por firma de layout** (`_trigger_dec_cache` en `OCRManager`, TTL 1800s + LRU 256): `_trigger_con_cache()` consulta la decisión (disparar/no disparar VLM) por `_page_signature` ANTES de `_compute_trigger` y cachea **positivas Y negativas** (a diferencia del §8.4.1 que solo cachea negativas) — misma imagen → misma firma → misma decisión entre corridas aunque cuDNN varíe el híbrido. No aplica con `force_uocr`/`disable_uocr` (benchmark). `clear_decision_cache()` limpia ambos caches. **Fixes del code review**: (a) el resolver NO consulta `_uocr_inferring` (si dependiera del flag en el primer call, un proceso que arrancara con el daemon infiriendo resolvería CPU y otro GPU → no-determinismo; la sesión 103 ya verificó que YOLO GPU coexiste con el daemon en VRAM); (b) salvaguarda de calidad — una decisión NEGATIVA cacheada NO suprime el VLM en una página GEMELA con el mismo layout pero detección MUCHO más débil (la firma es de layout, no de contenido; si `len(blocks)<n_c` Y `avg_conf<conf_c*0.8` se recomputa). | `config.py`, `ocr_utils.py`, `ocr_engine.py`, `tests/test_ocr_engine.py`, `tests/test_ocr_utils.py` | 🎯 **Trigger determinista** |
| 117 | **Validación con 2 corridas idénticas (págs 38-42, la p4 conflictiva)**: batch (`--batch-window 4`) y single produjeron **resultados IDÉNTICOS** (10 bloques / 5 trad / mismas páginas / mismas traducciones). Log del servidor: **10 hits** `[trigger] sesión 116: decisión cacheada por firma`, `[YOLO] device fijado UNA vez por proceso: 0`, 0 llamadas al daemon en las corridas de validación (el VLM solo se disparó en el run previo de warm-up: 2 `req_multi` → ahí nacieron las decisiones y los negativos §8.4.1 que las corridas limpias reutilizaron). Antes: p4 disparaba U-OCR en single (108s) pero no en batch. | — | ✅ **Determinismo validado** |

**Nota**: el cache de trigger es POR PROCESO (class var de `OCRManager`) — la garantía entre 2 corridas en servidores separados la da el device YOLO fijado por CUDA + firma estable; el cache añade determinismo DENTRO del proceso (páginas gemelas del mismo capítulo) y bloquea la deriva cuDNN entre páginas del mismo run. Riesgo aceptado (documentado): dos PDFs con layout parecido podrían compartir firma dentro de la ventana TTL de 30 min — misma limitación que el §8.4.1 existente.

#### Sesión 2026-08-06-v12 — Tests unitarios de procesar_lote (camino batch Fase 1) (5 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 118 | **Clase `TestProcesarLote` en `tests/test_process_all_pages.py`** (5 tests del camino batch): (1) `test_registra_todas_las_paginas_del_lote` — mock de `POST /api/process-page-batch` que verifica URL exacta, payload `images` en el MISMO orden de entrada, y registro de las N páginas con stats agregados (OK + VACIO); (2) `test_lote_con_b64_none_no_envia_esa_pagina` — una página con b64=None (render previo fallido) se registra como `render_error` (con su `render_t`, no el elapsed) y NO viaja en el payload del batch; (3) `test_lote_respuesta_con_menos_resultados_que_paginas` — el servidor devuelve menos resultados que páginas → las faltantes se registran como `missing_result` (sin perder páginas); (4) `test_lote_resultado_no_dict_registra_bad_result` — un resultado no-dict → `bad_result` sin romper el resto del lote; (5) `test_batch_registra_sin_nameerror` — regresión del fix de la sesión 109: el camino batch usa `data = resp.json()` (que el refactor sí conservó) y registra sin excepción (status `SIN_TRAD` correcto para bloque con orig==trad). Fix code review: asserts de `(page, status)` en vez de solo status por índice (los tests 3 y 4). **24 tests** de `test_process_all_pages.py` en verde; suite completa **428 tests** en verde. | `tests/test_process_all_pages.py` | ✅ Cobertura batch |

#### Sesión 2026-08-06-v13 — Refactor tests de procesar_lote: helper compartido _mock_batch_post (1 refactor)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 119 | **Helper compartido `_mock_batch_post(monkeypatch, tmp_path, results, argv_extra=None, status_code=200)`** en `tests/test_process_all_pages.py`: extrae el boilerplate repetido en los 9 tests del camino batch (carga del módulo con `_load_module(_argv_isolado(...))` + `captured={}` + `fake_post` + `monkeypatch.setattr`) en una sola función que devuelve `(mod, captured)` con la captura de `url`/`json` del POST. Refactorizados: los 5 tests de `TestProcesarLote` + `test_force_uocr_true/false_por_defecto_batch` + `test_lote_reparte_elapsed_entre_paginas`/`test_lote_single_suma_elapsed` (estos dos siguen mockeando `mod.time.time` DESPUÉS del helper, setup específico que pertenece al caller). **Sin cambios de aserciones** — 24 tests en verde. Fix code review: el branch `callable` del parámetro `results` era código muerto (los 9 tests pasan listas) → reemplazado por un kwarg `status_code=200` más descubrible (simula errores HTTP sin romper la captura del payload). | `tests/test_process_all_pages.py` | 🧹 Sin duplicación |

#### Sesión 2026-08-06-v14 — Fix branches de error de procesar_lote: elapsed repartido + NameError conn_error (2 fixes + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 120 | **Branches de error de `procesar_lote` alineados con el fix de la sesión 115**: los 3 branches (timeout / excepción genérica conn_error / HTTP != 200) registraban el elapsed COMPLETO del lote en CADA página (un lote de 4 págs fallido de 587s metía 4×587s en `results[].time` del checkpoint). Ahora calculan `per_page_elapsed = elapsed/max(len(valid),1)` y lo registran — consistente con el camino de éxito. **Fix NameError latente**: el branch `conn_error` usaba `elapsed` sin definirlo cuando la PRIMERA excepción no era Timeout (solo se computaba en `except requests.Timeout`) → NameError silencioso que mataba el worker y perdía el lote entero sin registrar nada. Ahora `elapsed`/`per_page_elapsed` se computan al inicio de cada except (y en cada reintento). | `process_all_pages.py` | 🐛 **Fix NameError + tiempos honestos** |
| 121 | **3 tests nuevos en `TestProcesarLote`** (branches de error): (a) `test_lote_timeout_definitivo_registra_timeout` — requests.Timeout en todos los intentos (MAX_RETRIES=0) → TODAS las páginas como `timeout` con time==5.0 (elapsed 10s / 2 páginas, time.time mockeado 100→110), pages_done=={1,2}, pages_error==2; (b) `test_lote_conn_error_registra_conn_error_sin_nameerror` — requests.ConnectionError en la PRIMERA llamada → **regresión del NameError** (antes crasheaba), todas `conn_error` con time repartido; (c) `test_lote_http_error_registra_http_status` — usa el kwarg `status_code=500` del helper `_mock_batch_post` → todas `http_500` con time repartido. **27 tests** de `test_process_all_pages.py` en verde; suite completa **431 tests** en verde. | `tests/test_process_all_pages.py` | ✅ Cobertura errores |

#### Sesión 2026-08-06-v15 — Tests de integración de main() en modo batch: acumulación de lotes + centinela re-insertado (2 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 122 | **Clase `TestMainBatchWindow` en `tests/test_process_all_pages.py`**: integración de `main()` COMPLETO (render thread + API workers + checkpoint + reporte) con `--batch-window 2` sobre un PDF fake. El mock de POST verifica que la URL es SIEMPRE `/api/process-page-batch` (nunca el endpoint single — 2 asserts defensivos) y devuelve una entrada de resultado por imagen enviada. Test 1 (4 págs): **2 lotes de 2 imágenes** `[2,2]` (bucle de acumulación), `renders==4`, checkpoint `pages_done==[1,2,3,4]` en orden, 4 bloques/4 traducidos/0 errores. Test 2 (3 págs): **lote parcial** `[1,2]` + `[3]` (el centinela `(None,None,0)` se encuentra como `extra` dentro del while interno y se RE-INSERTA en `_rendered_queue` — sin eso, la iteración externa espera 60s de stall) → guard de tiempo `wall < 10s` falla si el centinela no se re-inserta; checkpoint `[1,2,3]`. **Fix code review (race real)**: el assert `[len(images)] == [2,1]` del test 2 era RACY — los lotes van a un `ThreadPoolExecutor(MAX_WORKERS=3)` y el lote `[3]` puede postear ANTES que `[1,2]` (no hay garantía de orden de threads) → cambiado a order-independent `sorted(...) == [1,2]`. Verificado 10/10 corridas estables. **29 tests** de `test_process_all_pages.py` en verde; suite completa **433 tests** en verde. | `tests/test_process_all_pages.py` | ✅ Integración batch |

#### Sesión 2026-08-06-v16 — Benchmark determinismo: capítulo completo 53 págs × 2 corridas con la política de la sesión 116 (1 hallazgo + 1 tool)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 123 | **Benchmark de determinismo real (53 págs × 2 corridas, fusion, workers=2, caché de traducciones limpia en ambas, servidor REINICIADO entre corridas para que los caches de trigger/§8.4.1 —por proceso— partan vacíos)**: **Run A 214 bloques/122 trad (57.0%) vs Run B 205 bloques/107 trad (52.2%)**. **15/53 páginas difieren** en bloques/status/textos — la política determinista de la sesión 116 NO logró determinismo perfecto entre corridas separadas. Causas del residuo: (1) **cuDNN no-determinismo de EasyOCR GPU** — el híbrido produce inputs distintos del trigger entre procesos (ej. p5: en A el híbrido detectó débil → disparó U-OCR 92s y recuperó el diálogo artístico `...ERA UNA PROPUESTA`; en B detectó 3 bloques conf 0.75 → trigger NO disparó (2.4s) y **el diálogo artístico se perdió**); (2) **YOLO degradó a CPU 5× en B** (`_gpu_lock` ocupado por EasyOCR de otro worker) — la política bloqueante espera 30s pero con workers=2 un worker puede tener EasyOCR activo → degradación GPU→CPU reintroduce variación de detección; (3) **el cache de decisión es POR PROCESO** — entre corridas separadas no puede congelar decisiones. **Lo que SÍ garantizó la sesión 116**: dentro de UNA corrida, las páginas con la MISMA firma comparten decisión (10 hits de cache de trigger en B, 3 skips §8.4.1) → el determinismo intra-run es real. **Comparación vs pre-fix (F5)**: la tasa de traducción **SUBÓ** de 25.2% (143 bloq/36 trad) a 57.0%/52.2% — el cache de decisión NO suprimió VLM legítimo a nivel agregado (más bloques: 214 vs 143; más págs con texto: 53/53 vs 47/53). **PERO el caso p5 es la advertencia**: la pérdida del diálogo artístico en B es exactamente el riesgo del no-determinismo — el cache de decisión NEGATIVA cacheado de una página gemela con la MISMA firma de layout puede suprimir el VLM de una página artística (el cache guarda `0.2:000000000000…` — firma principal compartida por ~10 páginas del capítulo, verificada en la calibración de la sesión 90). Conclusión práctica: la política 116 da determinismo intra-corrida + sin regresión de calidad, pero para determinismo ENTRE corridas haría falta (a) persistir el cache de decisión en disco (producido por 1 corrida, consumido por la 2ª) y/o (b) `torch.backends.cudnn.deterministic=True` (costo de rendimiento) y/o (c) YOLO bloqueante SIN degradación a CPU (esperar hasta 180s en vez de 30s). Herramienta `tools/compare_det_runs.py` creada: compara 2 checkpoints (stats, tasa, diferencias por página con textos) con encoding seguro para CJK. | AGENTS.md, `tools/compare_det_runs.py` (nuevo) | 📊 **Determinismo intra-run real; inter-run NO** |

#### Sesión 2026-08-06-v17 — Medición de colisión de _page_signature entre PDFs: riesgo cross-PDF ALTO entre mangas (1 hallazgo + 1 tool)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 124 | **Medición real de colisión de `_page_signature` entre PDFs** (herramienta `tools/medir_colision_firmas.py`: renderiza cada página con ZOOM=1.2 como el pipeline real y computa firmas exactas + Hamming). **Resultado: el riesgo de interferencia cross-PDF es ALTO entre mangas de la misma serie** — cap 43 (53 págs) vs `47.pdf` de Descargas (54 págs, cap 47 de la misma serie): **39 de 42 firmas únicas del 43 están también en el 47 → 50/53 páginas (94%) colisionan EXACTAMENTE** (firma principal `0.2:0000000000000000…` compartida por ~9-14 páginas en cada capítulo). Con los caches de decisión viviendo en el PROCESO del servidor (trigger sesión 116 + §8.4.1 negativas, TTL 30 min), procesar el cap 47 justo después del 43 en el MISMO servidor hace que las páginas del 47 hereden las decisiones del 43 — interferencia real: un refuerzo U-OCR que NO recuperó nada en el 43 (negativa §8.4.1) suprimiría el VLM de una página artística gemela del 47. **Control no-manga**: cap 43 vs artículo científico (19 págs) → **0 colisiones exactas** (layout totalmente distinto). **Conclusión**: la firma de layout discrimina bien entre manga y texto, pero NO entre capítulos de la misma serie (paneles/trama similares → misma cuadrícula de oscuridad). **Mitigación propuesta (no implementada)**: escopear el cache por PDF — añadir un identificador de documento/sesión a la clave de la firma (el servidor recibe la imagen, no el PDF; habría que pasar el scope desde `/api/process-page` o generar un hash de página-documento en el caller), o subir `grid` a 16×16 (256 bits) + dark_ratio 2 decimales. El cache persistido en disco (sugerencia sesión 123) AMPLIFICARÍA este problema si no se escopea por PDF. | `tools/medir_colision_firmas.py` (nuevo) | ⚠️ **Riesgo cross-PDF 94% entre mangas** |

#### Sesión 2026-08-06-v18 — Cache de decisión del trigger persistido en disco: determinismo ENTRE servidores (1 feature + 5 tests + validación 2 restarts)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 125 | **Persistencia en disco de `_trigger_dec_cache`**: `OCRManager` ahora escribe `cache/ocr_decision_cache.json` (`{version, trigger: {firma: [ts, n_blocks, avg_conf, decision]}}`) tras cada `_trigger_cache_put` y lo carga en `__init__` una sola vez por proceso (`_cargar_cache_disco` + `_cache_cargado`/`_cache_load_lock`). Escritura atómica (`.tmp` + `os.replace`) bajo `_DISK_LOCK`; TTL 1800s podado en la carga (mismo TTL/LRU que en memoria); archivo corrupto → unlink + partir de cero (nunca rompe el OCR); `clear_decision_cache` también elimina el archivo. **Determinismo ENTRE procesos**: el servidor B (recién arrancado) carga las decisiones de A y las honra → 2 corridas en servidores SEPARADOS toman las mismas decisiones de trigger (la sesión 116 solo garantizaba intra-proceso). **Decisión de diseño (code review)**: el cache §8.4.1 de NEGATIVAS NO se persiste — su consulta es ciega (sin salvaguarda `mucho_mas_debil` como el trigger) y la sesión 124 midió 94% de colisión de firma entre capítulos de la misma serie; persistir negativas amplificaría la supresión cross-PDF. El trigger cache por sí solo da decisiones idénticas (las negativas siguen en memoria, optimización intra-corrida; un servidor nuevo re-corre el VLM en páginas con decisión positiva — dirección segura). **Tests**: fixture autouse en `tests/conftest.py` que redirige `_DECISION_CACHE_PATH` a `tmp_path` (toda la suite queda aislada del archivo real) + clase `TestPersistenciaDisco` (5 tests: put→recarga en proceso nuevo honra la decisión; negativa NO persiste; expiradas podadas en la carga; corrupto degrada sin crash + se elimina; clear elimina el archivo). **Validación real (2 restarts del servidor, págs 38-42, fusion workers=2)**: Run A computa y persiste 5 decisiones (2 VLM `True`, 3 `False`; 2 llamadas al daemon); servidor matado; Run B en proceso NUEVO → **5 hits `[trigger] sesión 116: decisión cacheada por firma` con stats EXACTAS del disco (conf 0.81/0.43/0.00/0.66/0.17 — imposibles de reproducir por azar)**, VLM en las mismas 2 firmas (2 req_* nuevos), 0 decisiones divergentes. Bloques por página difieren ±1 (cuDNN del híbrido — exactamente la variación que el cache neutraliza a nivel de decisión); traducciones 16 vs 14 (consecuencia de los textos distintos, no de las decisiones). | `ocr_engine.py`, `tests/conftest.py`, `tests/test_ocr_engine.py` | 🆕 **Determinismo inter-proceso** |

#### Sesión 2026-08-06-v19 — Scope por documento (doc_id): el cache de decisiones ya NO puede cruzar capítulos (1 feature + 5 archivos + tests + re-medición 0 colisiones)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 126 | **Scope por documento del cache de decisiones** (fix del riesgo cross-PDF medido en la sesión 124, amplificado por la persistencia de la 125): los caches de decisión del trigger **y** §8.4.1 de negativas se clavean ahora por `"doc_id:firma"` en vez de la firma bruta. `OCRManager.run_ocr()`/`run_ocr_batch()` aceptan `doc_id: str = ""` y el helper `_firma_documento(doc_id, firma)` devuelve `firma` sin cambio si `doc_id` o `firma` van vacíos (los callers no migrados conservan el comportamiento previo exacto — no hay footgun nuevo). Los endpoints `POST /api/process-page` y `/api/process-page-batch` leen `doc_id` opcional del payload y lo **sanitizan** (`re.sub(r"[^A-Za-z0-9_]", "")[:64]` — un `:` en doc_id rompería el prefijo). **Callers migrados**: `process_all_pages.py` (deriva DOC_ID como md5 del nombre del PDF `[:12]` en ambos payloads single/batch), `app.js` (`getDocId()` = hash FNV-1a del nombre del archivo, prefijo `ui`, en los 2 fetch de `/api/process-page`), `reprocess_failed.py`/`stress_test_memory.py`/`gestor.py` (mismo hash del capítulo 43). **Bump de versión del archivo persistido** (v1→v2): la carga descarta por mismatch de versión y ELIMINA el archivo viejo (las claves v1 sin prefijo nunca matchearían los lookups escopeados — clean start). **Re-medición con `tools/medir_colision_firmas.py --scoped`**: firma bruta sigue colisionando 94% (39/42 firmas únicas del 43 en el 47), pero la clave REAL `doc_id:firma` da **0 colisiones efectivas** — el cap 47 ya no puede heredar decisiones del 43, ni en memoria ni en disco. **Tests** (6 nuevos): `TestPersistenciaDisco.test_archivo_version_v1_sin_scope_se_descarta` (v1 → descartada + archivo eliminado) + test del scope en `test_ocr_engine.py` (misma firma con doc_id distinto → claves distintas; vacío → clave bruta), endpoints en `test_api.py` (single y batch pasan `doc_id` a `OCRManager`), payload en `test_process_all_pages.py` (DOC_ID presente y estable en ambos payloads). **449 tests en verde** (117 api + 65 ocr_engine + 115 ocr_utils + 87 translator + 31 process_all_pages + 23 ocr_functions + 11 uocr_daemon), `py_compile` + `node --check app.js` OK. | `ocr_engine.py`, `routes/api.py`, `process_all_pages.py`, `app.js`, `reprocess_failed.py`, `stress_test_memory.py`, `gestor.py`, `tools/medir_colision_firmas.py`, `tests/` | 🆕 **0 interferencia cross-PDF** |

#### Sesión 2026-08-06-v20 — Flag UOCR_NEG_CACHE_PERSIST: persistencia OPCIONAL de las negativas §8.4.1 (1 flag + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 127 | **`UOCR_NEG_CACHE_PERSIST: Final[bool] = False` en config.py** (sesión 127): activa opcionalmente la persistencia de las negativas §8.4.1 junto al trigger en `cache/ocr_decision_cache.json` (clave `"neg"`, firma → timestamp, TTL 1800s + LRU 256 como en memoria). Default en memoria (comportamiento de la sesión 125 intacto). En `ocr_engine.py`: `_persistir_cache()` incluye `data["neg"]` solo si el flag está activo; `_cargar_cache_disco()` carga la sección `"neg"` (podada por TTL + capped LRU) solo si el flag está activo — un proceso con el flag apagado NO la carga aunque el archivo la contenga (re-corre el VLM: dirección segura, solo recupera); `_registrar_decision_negativa()` persiste tras registrar solo si el flag está activo. La reescritura completa del archivo con flag=False purga naturalmente la sección `"neg"` dejada por una corrida con flag=True. **Trade-off documentado en config.py**: Pro — determinismo de EJECUCIÓN completo entre servidores (las decisiones de trigger Y los saltos §8.4.1 de páginas repetitivas se congelan en disco → 2 corridas en procesos separados hacen EXACTAMENTE las mismas llamadas VLM). Contra — la consulta de negativas es CIEGA (sin salvaguarda `mucho_mas_debil` como `_trigger_con_cache`) y la sesión 124 midió 94% de colisión de firma entre capítulos de la misma serie; el scope por doc_id (sesión 126) limita el daño a páginas del MISMO documento pero no lo elimina dentro de un capítulo con layout repetido. **Code review**: sin deadlocks — `_persistir_cache` se llama FUERA del `with _uocr_cache_lock` en `_registrar_decision_negativa` y el orden de locks (`_trigger_dec_lock` → `_uocr_cache_lock`, secuenciales y nunca anidados al revés) es consistente; monkeypatch de `ocr_engine.UOCR_NEG_CACHE_PERSIST` (no `config.*`) es el target correcto; el LRU de carga (sorted-slice) evicta las mismas entradas que el de memoria (while-min); sin bump de versión (la clave opcional `"neg"` no rompe el esquema v2). **Tests** (3 nuevos en `TestPersistenciaDisco`): flag=True → la negativa se escribe en `"neg"` y un proceso nuevo la honra; negativas expiradas se podan en la carga con el flag activo; con el flag apagado un proceso nuevo NO carga las negativas del archivo. **452 tests en verde** (3 nuevos), `py_compile` OK. | `config.py`, `ocr_engine.py`, `tests/test_ocr_engine.py` | 🆕 **Determinismo de ejecución completo (opt-in)** |

#### Sesión 2026-08-06-v21 — Refresh de timestamp en hits del cache de decisiones (TTL deslizante) + validación capítulo completo (1 fix + 3 tests + 1 benchmark)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 128 | **Refresh en hits (touch LRU / TTL deslizante)**: `_trigger_cache_get()` y `_is_decision_negativa_vigente()` ahora REARMAN el timestamp (`time.time()`) en cada hit dentro del TTL — la ventana de 30 min se cuenta desde la ÚLTIMA consulta, no desde el guardado. Una corrida larga (>30 min) donde una firma reaparece cada pocas páginas ya NO expira decisiones a mitad de capítulo. El refresh también se PERSISTE (`_persistir_cache()` fuera del lock) para que un servidor nuevo honre la ventana extendida — sin esto, un restart a mitad de corrida recargaría el ts viejo y la decisión expiraría igual. El refresh solo toca el timestamp, NO los stats (n_blocks/avg_conf) — la salvaguarda `mucho_mas_debil` de la sesión 116 sigue comparando contra los inputs de la PRIMERA decisión (intactos). En el §8.4.1 se persiste solo si `UOCR_NEG_CACHE_PERSIST` (sesión 127). **Code review**: sin deadlocks (el persist se llama FUERA del lock de memoria en ambos caches; `_persistir_cache` adquiere trigger→uocr secuencialmente, nunca anidado al revés con `_cache_load_lock`); el flag `refrescado` se simplificó (siempre True en el camino de retorno). **3 tests nuevos**: hit con ts envejecido a TTL-5s refresca y sigue devolviendo la decisión; el refresh se persiste y un proceso nuevo la carga como vigente; la negativa §8.4.1 refresca igual. **455 tests en verde**, `py_compile` OK. | `ocr_engine.py`, `tests/test_ocr_engine.py` | 🛡️ **Sin expiración a mitad de capítulo** |
| 129 | **Validación real — capítulo completo (53 págs, fusion, workers=2, cache de decisiones LIMPIO y servidor reiniciado como la sesión 123)**: **53/53 páginas, 0 errores, 209 bloques / 121 traducidos = 57.9% de tasa** — la MÁS ALTA medida en el capítulo, por encima de la sesión 123 (Run A 57.0%, Run B 52.2%) y muy por encima del baseline F5 pre-fix (25.2%). **La persistencia + scope + refresh NO sacrificaron recuperación de diálogo artístico**: el trigger siguió disparando el VLM (11 llamadas al daemon en la corrida) y 10 hits de `decisión cacheada` con stats CONSISTENTES (6 bloq conf 0.88 → VLM ×3, 3 bloq conf 0.75 → no VLM ×4, 2 bloq conf 0.42 → no VLM ×3) — la misma firma produjo la misma decisión dentro de la corrida. **Caveat honesto (mismo fenómeno de la sesión 123)**: la p5 (diálogo artístico `...ERA UNA PROPUESTA`) NO se recuperó en esta corrida (el híbrido la detectó 3 bloques conf 0.42 → decisión negativa cacheada `no VLM` ×3), mientras que la corrida anterior del turno (proceso huérfano del turno anterior, que completó 53/53 con 278 bloques/186 trad = 66.9% ANTES de que mi duplicado lo pisara) sí la recuperó — la variación cuDNN del híbrido entre procesos sigue existiendo Y el cache de decisión la CONGELA por firma (una vez negativa, la gemela repite la negativa dentro del TTL). El refresh de la sesión 128 NO resuelve ese caso (no es un problema de expiración, es de decisión congelada) — mitigación natural: el scope por doc_id (sesión 126) ya impide que sea cross-PDF; intra-documento sigue siendo el trade-off aceptado del determinismo. **Nota de proceso**: al reiniciar Freebuff a mitad de validación quedaron DOS procesos `process_all_pages.py` escribiendo al MISMO checkpoint (el huérfano del turno anterior siguió vivo y completó 53/53; mi relanzamiento lo pisó con su progreso parcial). Lección: verificar `wmic process` ANTES de relanzar y usar checkpoints con nombre único. | `run_det128_run1.json` (checkpoint), AGENTS.md | 📊 **57.9% — mejor tasa del capítulo** |

#### Sesión 2026-08-06-v22 — Salvaguarda mucho_mas_debil en las negativas §8.4.1 + UOCR_NEG_CACHE_PERSIST default True (1 feature + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 130 | **Salvaguarda `mucho_mas_debil` en la consulta de negativas §8.4.1** (misma que `_trigger_con_cache`): el formato del cache de negativas pasa de `{firma: ts}` a `{firma: (ts, n_blocks, avg_conf)}` — cada negativa guarda los stats de detección del híbrido en el momento de registrarla. `_is_decision_negativa_vigente(firma, n_blocks, avg_conf)` ahora devuelve **False (ignora la negativa y RE-DISPARA el VLM)** si la página actual se detecta MUCHO más débil que la que registró la negativa (`n_blocks < n_c AND avg_conf < conf_c * 0.8`) — la firma es de layout, no de contenido: una página gemela con el mismo layout pero diálogo artístico que el híbrido ahora pierde es justo el que el VLM podría recuperar. Defaults `(0, 0.0)` en la consulta y el registro: sin stats → `n_blocks < 0` nunca se cumple → comportamiento honra (viejo), compatibilidad para callers que no pasan stats; docstring advierte que SIEMPRE hay que pasarlos (con defaults la salvaguarda queda desactivada para esa entrada). **Callers actualizados**: batch Fase A (`len(blocks)/avg_conf`), batch Fase B (nueva lista `per_page_avg_conf` guardada en Fase A — con comentario de que `avg_conf` refleja el híbrido, no se recomputa tras el merge YOLO: par consistente entre consulta y registro, misma convención que single), `_run_fusion` y `_reforzar_con_unlimited`. **Persistencia**: el archivo `neg` pasa a formato `[ts, n_blocks, avg_conf]` y `_DECISION_CACHE_VERSION` sube a **v3** (los archivos v2 — negativas como ts plano — se descartan en la carga; mensaje de mismatch generalizado a "formato desactualizado"). **`UOCR_NEG_CACHE_PERSIST` default False → True** (config.py): con la salvaguarda la supresión ya NO es ciega → el determinismo de ejecución completo entre servidores (también los saltos §8.4.1, no solo el trigger) ya no sacrifica recuperación artística. Trade-off residual documentado: la salvaguarda solo cubre "detección actual mucho más débil"; páginas con detección COMPARABLE siguen honrando la negativa (determinismo) y el scope por doc_id (sesión 126) impide el cross-PDF. **Code review**: sin deadlocks (release del lock antes de persistir), factor 0.8 idéntico al trigger, carga con try/except + LRU por `entry[0]`, consisitencia post-YOLO verificada; nits aplicados (mensaje de mismatch genérico, docstring del footgun de defaults, comentario avg_conf-YOLO). **3 tests nuevos**: much_mas_debil ignora la negativa y re-dispara el VLM (integración con daemon mockeado); detección comparable honra la negativa (0 llamadas); la salvaguarda sobrevive a la recarga desde disco. Tests de persistencia actualizados al default True y al formato [ts, n, c]. **458 tests en verde** (3 nuevos), `py_compile` OK. | `ocr_engine.py`, `config.py`, `tests/test_ocr_engine.py` | 🛡️ **Supresión de VLM ya no es ciega** |

#### Sesión 2026-08-06-v23 — Benchmark overhead de persistencia: el flag activo cuesta 9ms por capítulo — batching NO merece la pena (1 hallazgo + 1 tool)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 131 | **Medición del overhead de escritura de `UOCR_NEG_CACHE_PERSIST=True`** (herramienta `tools/medir_overhead_persistencia.py`: combina eventos REALES de la corrida run_det128_run1 — hits de trigger, saltos §8.4.1, fusiones VLM, llamadas al daemon por `find req_*` en la ventana de la corrida — con microbenchmark del costo real de `_persistir_cache()`: json.dumps + write .tmp + os.replace). **Resultado: el overhead del flag es DESPRECIABLE**. Eventos reales del capítulo (53 págs): 42 firmas únicas (42 puts de trigger) + 10 hits de trigger (refresh) = 52 escrituras con flag OFF; con flag ON se añaden 6 registros de negativa + 3 hits de negativa = **61 escrituras totales (9 más)**. Microbenchmark: **~1.0 ms por escritura** con el tamaño real del archivo (5.7 KB) y **~1.7 ms** en el peor caso (512 entradas, 21 KB — el límite LRU de 256 por sección). **Tiempo total añadido por el flag: ~9 ms por capítulo = 0.0004% del tiempo de pared (40 min)**; incluso en el peor caso de cache lleno: 104 ms = 0.0043%. **La regla "2 por decisión" es correcta para una página VLM sin recuperación** (1 put de trigger + 1 registro de negativa), pero a ~1 ms cada una es irrelevante frente a los 130-500s de la inferencia VLM. **Decisión documentada: NO implementar batching de la persistencia** — escribir tras cada mutación es simple, robusto y el costo total es < 1% del capítulo (el umbral del benchmark). Si algún día el archivo creciera a miles de entradas (no ocurre: LRU cap 256), el costo por escritura subiría linealmente y ahí sí valdría batching. | `tools/medir_overhead_persistencia.py` (nuevo), AGENTS.md | 📊 **9ms/capítulo → sin batching** |

#### Sesión 2026-08-07-v24 — Validación en vivo del flag UOCR_NEG_CACHE_PERSIST=True: negativas del disco honradas por un proceso nuevo (1 validación)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 132 | **Validación en vivo de la persistencia de negativas §8.4.1 entre procesos** (2 servidores separados, `UOCR_NEG_CACHE_PERSIST=True` por defecto, págs 14-17 del capítulo — las 4 con VLM real según `page_times` del checkpoint run_det128_run1). **Run A** (servidor fresco, cache limpio): 3 llamadas VLM (p15, p16, p17; p17 = 1 bloque conf 0.53, el VLM corrió 123s y no recuperó nada → registró negativa `f1cf2cc03ad4:0.2:000…` con stats `[ts, 1, 0.53]` en `cache/ocr_decision_cache.json` v3). **Servidor reiniciado (proceso B, memoria vacía)**: arranca y carga las negativas del disco. **Run B** (mismo seed, checkpoint gemelo): **1 salto §8.4.1 explícito en el log** (`[fusion] §8.4.1: firma … repetitiva (U-OCR no recuperó antes) — salto de refuerzo`), **solo 2 llamadas VLM** (p15, p16 — las que sí recuperan) y la **p17 saltó el VLM produciendo el MISMO resultado** (`persona→Person`, 1 bloque, 84.2s vs 123s). El timestamp de la negativa en disco pasó a 00:13:12 = el **touch LRU del hit en Run B** (sesión 128: cada consulta de negativa refresca y re-persiste) — evidencia de que la entrada cargada del disco se consultó y honró en el proceso nuevo. **Conclusiones**: (a) la persistencia de negativas funciona end-to-end entre servidores (el determinismo de ejecución completo prometido en la sesión 129 es real); (b) la salvaguarda mucho_mas_debil (sesión 129) no interfirió porque la detección de p17 fue idéntica (1/0.53) a la registrada; (c) el ahorro fue 1 llamada VLM (~40-120s) en solo 4 páginas — el flag paga su costo con creces. | AGENTS.md | ✅ **Negativas entre procesos** |

#### Sesión 2026-08-07-v26 — Salvaguarda de detección débil en las negativas §8.4.1: el caso p5 ya no se congela (1 feature + 8 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 134 | **Salvaguarda de detección débil en las negativas §8.4.1 (caso p5)**: si la negativa se registró con una detección híbrida DEMASIADO pobre (`n_c < UOCR_NEG_WEAK_MAX_BLOCKS` O `conf_c < UOCR_NEG_WEAK_MIN_CONF`), la página GEMELA con detección comparable (que el much_mas_debil de la sesión 129 no libera) puede **RE-DISPARAR el VLM hasta `UOCR_NEG_MAX_REINTENTOS` veces por firma** antes de congelarse. P5 registró la negativa con 2-3 bloques conf ~0.42 (y p17 con 1 bloque conf 0.53): la variación cuDNN del híbrido hace que esas detecciones pobres fluctúen y la negativa congelada mataba el diálogo artístico que el VLM sí leería. **Formato de la entrada**: `(ts, n_blocks, avg_conf)` → `(ts, n_blocks, avg_conf, re_disparos)` — el contador viaja por la persistencia (`_DECISION_CACHE_VERSION` 3 → 4; archivos v3 descartados en la carga, mensaje "formato desactualizado" ya generalizado en la sesión 129) para mantener el determinismo entre servidores: un proceso nuevo respeta los re-disparos ya consumidos. **Semántica del contador**: `_is_decision_negativa_vigente` — tras el check much_mas_debil (que NO consume contador; su cadena está acotada porque cada re-registro baja los stats y termina en 0 bloques donde la salvaguarda débil consume el contador), si la negativa es débil y `re_disparos < MAX` → incrementa y devuelve False (re-disparo); al agotar → congela (el VLM ya tuvo su oportunidad). La mutación se persiste FUERA del lock (mismo patrón que la sesión 128). **`_registrar_decision_negativa` PRESERVA el contador** de una entrada previa — evita el bucle re-dispara→falla→re-registra→re-dispara infinito: tras un re-disparo fallido, la firma queda congelada. **`_limpiar_decision_negativa(firma)`** (nuevo): borra la negativa cuando el refuerzo RECUPERA algo (single `_reforzar_con_unlimited` + batch Fase B) — la recuperación refuta la negativa obsoleta y las gemelas posteriores vuelven a intentar el VLM (sin esto, el re-disparo exitoso dejaría la negativa estale congelando a las siguientes). **Calibración documentada** (config.py): el ejemplo del usuario era "<2 bloques o conf <0.3" pero p5 registró 2-3 bloques conf 0.42 — se usan `<3 bloques` (cubre 0/1/2) O `conf <0.45` (cubre 0.42); una página con 3 bloques conf 0.9 (detección real) se congela como antes. **Efecto amplio acotado**: TODA negativa con <3 bloques (incluidas las de texto débil del trigger v4.2) califica como débil → ~1 inferencia VLM extra por firma débil por TTL, nunca infinitas. **Code review**: sin deadlocks (persist fuera del lock, sin anidación nueva); el return-False-tras-mutación dentro del lock se evitó (la mutación cae al persist del final); carga/parseo de 4-tuplas en sync con el formato persistido; 3 nits aplicados (comentario del bound de la cadena much_mas_debil, nota del efecto amplio en config.py, test del clearing en batch Fase B). **8 tests nuevos**: unit contador (débil re-dispara 1 vez y congela), fuerte congela sin re-disparo, debilidad por conf, coexistencia con much_mas_debil (no consume contador), integración 3 gemelas (call 1→2→2), recuperación exitosa limpia (single), recuperación exitosa limpia (batch Fase B), contador persistido con reload que congela. 1 test base §8.4.1 actualizado (ahora usa 5 bloques conf 0.7 + panel grande para no caer en la salvaguarda débil — su intención de "repetitiva no re-dispara" se preserva). **469 tests en verde** (8 nuevos), `py_compile` OK. | `config.py`, `ocr_engine.py`, `tests/test_ocr_engine.py` | 🛡️ **El caso p5 ya no se congela** |

#### Sesión 2026-08-07-v27 — Benchmark del coste real de la salvaguarda débil (sesión 134): 0 inferencias VLM extra en 53 págs (1 hallazgo + 1 tool)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 135 | **Benchmark del coste real de la salvaguarda de detección débil en un capítulo completo (53 págs, fusion, workers=2, cache fresco)**: la salvaguarda (sesión 134) costó **0 inferencias VLM extra** — las 3 negativas registradas quedaron con `re_disparos=0` y la ÚNICA negativa débil de la corrida (13:09:02, 1 bloque conf 0.53 — el caso p17/p5) **nunca volvió a ser consultada** (ningún trigger posterior con su clave exacta `…38383838`, que es única en el capítulo — solo comparte el prefijo `f1cf2cc03ad4`). **Firmas débiles**: 19/42 (45%) de las firmas de trigger del capítulo califican como débiles (<3 bloques o conf <0.45), pero de las 3 negativas §8.4.1 registradas solo 1 era débil — el contador=1 es un seguro que NO se cobra cuando la gemela no aparece dentro del TTL. **Beneficio potencial intacto**: el mecanismo queda armado (la negativa débil puede re-disparar UNA vez si una página con su misma firma exacta aparece en la ventana de 30 min); el coste máximo por capítulo es ~1 inferencia VLM (~2-8 min) por firma débil registrada — en esta corrida, cero. **Tasa de traducción**: 158/245 (64.5%) vs run1 121/209 (57.9%) — la mejora NO es atribuible a la salvaguarda (que no hizo nada), sino al no-determinismo del trigger (más páginas VLM: 14 vs 11, 151.7 min vs 29.6 min de pared) por cache fresco + device YOLO. **Herramienta** `tools/medir_coste_salvaguarda.py`: analiza el cache persistido v4 (formato `[ts, n, conf, re_disparos]`), las claves trigger y las ventanas temporales de los req del daemon para reconstruir la historia de re-disparos offline — se puede re-correr tras cualquier corrida con `--cache cache/ocr_decision_cache.json`. **Caveat de medición**: el log del servidor NO imprime los re-disparos de la salvaguarda (es silenciosa — retorna False y el caller lanza el VLM), así que la fuente de verdad es el contador persistido. | `tools/medir_coste_salvaguarda.py` (nuevo), cache persistido | 📊 **El seguro no se cobró: 0 VLM extra** |

#### Sesión 2026-08-07-v28 — Salvaguarda de detección débil en el cache de decisión del TRIGGER: la gemela artística ya no honra el "no VLM" a ciegas (1 feature + 8 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 136 | **Salvaguarda de detección débil en el cache de decisión del TRIGGER (`_trigger_dec_cache`)** — espejo de la sesión 134 (negativas §8.4.1) en el cache que decidía si disparar el VLM. **Problema**: una decisión NEGATIVA de trigger ("no VLM") cacheada por firma se honraba en páginas GEMELAS con detección COMPARABLE aunque la gemela fuera artística — el much_mas_debil de la sesión 116 solo libera gemelas detectadas MUCHO más débiles (menos bloques Y conf < 80%), así que una gemela con detección comparable (p.ej. 2 bloq conf 0.42 cacheados vs 2 bloq conf 0.15 actuales — el híbrido la pierde distinto por cuDNN) quedaba congelada sin VLM a pesar de tener diálogo artístico. **Formato de la entrada**: `(ts, n_blocks, avg_conf, decision)` → `(ts, n_blocks, avg_conf, decision, re_computes)` — el contador de recomputes viaja por la persistencia (`_DECISION_CACHE_VERSION` 4 → 5; archivos v4 descartados en la carga, mensaje "formato desactualizado" ya generalizado). **Semántica**: en `_trigger_con_cache`, las decisiones POSITIVAS (VLM) se honran SIEMPRE sin consumir contador (nada que suprimir); las negativas con much_mas_debil recomputan sin contador (sesión 116, va primero); SOLO una negativa con detección COMPARABLE consulta `_consumir_recompute_salvaguarda(firma)` (nuevo, sesión 136): si los stats CACHEADOS son débiles (`n_c < UOCR_NEG_WEAK_MAX_BLOCKS` O `conf_c < UOCR_NEG_WEAK_MIN_CONF`) y `re_computes < UOCR_NEG_MAX_REINTENTOS` → consume el contador y devuelve True (el caller recomputa el trigger — si la página actual cruza el umbral v4.2, el VLM dispara); al agotar → congela. Re-lee la entrada almacenada BAJO el lock (fuente de verdad, race-safe con otro worker) y persiste FUERA del lock (mismo patrón 128/134). **`_trigger_cache_put` PRESERVA el contador** de una entrada previa — evita el bucle recomputa→negativo→recomputa infinito; `_trigger_cache_get` lo preserva en el touch LRU. **Casos límite**: entrada evictada/expirada entre get y consumir → True (sin negativa que honrar, recomputar); `has_big_panel` NO está en la tupla → una negativa FUERTE cacheada + gemela con panel grande sigue honrando "no VLM" (limitación pre-existente del determinismo, fuera del scope "débil" — documentada). **Código**: el branch `if decision_c: pass/elif` se simplificó a `recomputar = (not decision_c and (mucho_mas_debil or self._consumir_recompute_salvaguarda(firma)))` (code review). **8 tests nuevos** (en `TestTriggerDecisionCache`): contador 0→1 y congela (unit con spy de `_compute_trigger`); recompute que flipea negativo→positivo y dispara el VLM (el caso de valor); fuerte no recomputa; débil por conf; positivo nunca consume; much_mas_debil coexiste sin consumir; integración 3 gemelas con daemon mockeado (call 0→1→2: la 3ª honra la decisión ya positiva); contador persistido con reload que congela. Tests existentes actualizados a la 5-tupla (inyecciones en memoria y archivos de disco de la clase `TestPersistenciaDisco`). **477 tests en verde** (8 nuevos), `py_compile` OK. | `ocr_engine.py`, `config.py`, `tests/test_ocr_engine.py` | 🛡️ **El "no VLM" débil ya no se congela** |

#### Sesión 2026-08-07-v29 — Validación EN VIVO de la salvaguarda de detección débil: Run A → negativa débil, Run B → re-disparo (contador 0→1), Run C → congelación (1 validación E2E)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 137 | **Validación en vivo del ciclo completo de la salvaguarda débil §8.4.1 (caso p17, 1 bloque conf 0.53)** sobre las págs 14-17 del capítulo 43 (fusion, workers=2, seed 1-13, cache v4 descartado al arrancar → baseline v5 fresco). **Secuencia de 3 corridas con el MISMO servidor y cache persistido entre ellas** (la "gemela" = la misma página reprocesada): **Run A** (15:55, cache fresco): p17 corre VLM real (549.8s — `[process-page] OCR (fusion): 1 bloques en 549.8s`), el refuerzo no recupera nada → negativa §8.4.1 registrada como DÉBIL `[ts, 1, 0.53, re_disparos=0]` (1 bloque < UOCR_NEG_WEAK_MAX_BLOCKS=3). **Run B** (16:14): p17 → decisión de trigger cacheada (1 bloq conf 0.53 → VLM) → `_is_decision_negativa_vigente` encuentra la negativa débil con contador disponible → **consume (0→1) y re-dispara el VLM** (638.6s — `[process-page] OCR (fusion): 1 bloques en 638.6s`); al no recuperar, se re-registra con el contador PRESERVADO =1 (sin esto, bucle infinito). **Run C** (16:24): p17 → trigger cacheado (VLM) → la negativa con `re_disparos=1` agotado → **CONGELA**: log del servidor línea exacta `[fusion] §8.4.1: firma f1cf2cc03ad4:0.2:000 repetitiva (U-OCR no recuperó antes) — salto de refuerzo`, OCR en 303.1s (espera de `_gpu_lock` detrás del VLM de p16 en paralelo, no inferencia VLM propia) y **0 llamadas al daemon para p17**. **Evidencia con timestamps del daemon**: en la ventana de Run C (16:24:01-16:39:29) los dirs `req_*` del daemon muestran EXACTAMENTE 2 llamadas — `req_1786137844495` (16:24:04, p15) y `req_1786138464911` (16:34:24, p16) — ninguna atribuible a p17; las fusiones del log lo corroboran (`Fusión: 6 híbrido + 7 U-OCR → 7` para p15, `7 híbrido + 5 U-OCR → 10` para p16, y p17 SIN línea de fusión U-OCR). **Cache final v5**: negativa `f1cf2cc03ad4:0.2:000…3838383838383838` = `[ts, 1, 0.532, re_disparos=1]` (agotada); entrada del trigger de p17 `[ts, 1, 0.532, True, re_computes=0]` (touch LRU en el hit preserva decision=True de Run B). **Conclusión**: el ciclo completo funciona — negativa débil → 1 re-disparo permitido → congelación al agotar, con el coste acotado a ~1 inferencia VLM por firma débil por TTL. El 303.1s de Run C p17 es contención de GPU (p17 esperó el `_gpu_lock`/daemon de p16), NO una llamada VLM — la congelación ahorró ~5-8 min de inferencia. Entorno limpio tras la validación (servidor 5174 parado, daemon 5177 vivo). Logs conservados: `run_136_a/b/c.log` (output de process_all_pages) y `run_136_server.log` (eventos del servidor). | `run_136_a/b/c.log`, `run_136_server.log`, `cache/ocr_decision_cache.json` | ✅ **Salvaguarda validada E2E** |

#### Sesión 2026-08-07-v30 — Print explícito del recompute de la salvaguarda débil en el log del trigger (1 feature + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 138 | **El recompute de la salvaguarda débil del trigger ya se loguea**: antes era silencioso (la sesión 135 documentó el caveat — la fuente de verdad era solo el contador persistido). Ahora `_consumir_recompute_salvaguarda` devuelve `int` en vez de `bool` — `>=1` = recompute consumido (contador+1), `0` = permitido sin consumo (entrada evictada/expirada, no hay negativa que honrar), `-1` = congelado (contador agotado) — y `_trigger_con_cache` usa `recompute_n >= 0` para decidir recomputar y `>= 1` para imprimir `[trigger] sesión 136: salvaguarda débil — recompute {n}/{UOCR_NEG_MAX_REINTENTOS} de firma {firma[:16]}…` (p.ej. `recompute 1/1`). El print SOLO se emite cuando el contador se consume de verdad (no en el camino "sin entrada", que devuelve 0, ni en el mucho_mas_debil de la sesión 116, que no consume); al congelarse, el log muestra la línea `sesión 116: decisión cacheada ... no VLM` como siempre. `UOCR_NEG_MAX_REINTENTOS=1` → formato `1/1`. **Tests**: 2 nuevos con `capsys` en `TestTriggerDecisionCache` — (1) la 1ª gemela débil imprime la línea `recompute 1/1`, la 2ª (congelada) NO imprime recompute y sí la de sesión 116 con `no VLM`; (2) el camino sin entrada cacheada no loguea `salvaguarda débil` (sin falsos positivos). **479 tests en verde** (2 nuevos), `py_compile` OK. *Nota de entorno: `pytest tests/` (directorio) devuelve salida vacía por un quirk PRE-EXISTENTE — pytest 9 desciende a `tests/archive/` y recolecta `_run_ci_test.py` (CI archivado que ejecuta `py_compile` en bucle y rompe la colección silenciosamente, exit 0 sin resumen); el proyecto siempre invoca pytest con LISTA EXPLÍCITA de archivos (`run_ci.py` línea 223: `test_files` = translator, ocr_utils, ocr_functions, api, ocr_engine, uocr_daemon + process_all_pages), que es la forma canónica y la que pasa los 479 tests.* | `ocr_engine.py`, `tests/test_ocr_engine.py` | 🪵 **Recomputes visibles en el log** |

#### Sesión 2026-08-07-v31 — Verificación batch: la salvaguarda débil del trigger aplica igual en run_ocr_batch Fase A (1 test de integración)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 139 | **Confirmado por test de integración: `run_ocr_batch` (Fase A) aplica la salvaguarda débil del trigger IDÉNTICA al single** — el código ya la heredaba porque Fase A llama al MISMO `_trigger_con_cache` (ocr_engine.py ~501) que `_run_fusion`; faltaba la cobertura. Nuevo `test_batch_salvaguarda_debil_trigger_recomputa_y_congela` (en `TestRunOcrBatch`): siembra 2 negativas de trigger DÉBILES cacheadas (2 bloq conf 0.42 vía `_trigger_cache_put`), agota el contador de la 2ª (inyección directa de la 5-tupla `(ts, n, conf, decision, re_computes=1)` bajo el lock), mockea `_page_signature` con `side_effect=[firma1, firma2]` (1 imagen → 1 firma en Fase A, exactamente 2 llamadas), `_detect_and_ocr` → 2 bloq conf 0.15 (cruza el umbral v4.2 si recomputa), YOLO (`_detect_text_regions_in_page` → []) y Fase 2 (`_run_rapidocr` → []) sin aportar, `_fusionar_blocks_multi` concatenación y `routes.api._ocr_with_unlimited_batch` devolviendo 1 página. Aserciones: (1) el daemon batch se llama UNA vez con SOLO la imagen 1 (`len(batch_imgs)==1`); (2) página 1 `['easyocr+rapid','unlimited-batch']` con el bloque recuperado, página 2 solo `['easyocr+rapid']` (congelada, 0 VLM); (3) cache final: firma1 positiva con `re_computes=1` (recompute 0→1 consumido, put preservó el contador), firma2 sigue negativa con `re_computes=1` intacto; (4) log con `capsys`: línea `sesión 136: salvaguarda débil — recompute 1/1 de firma 0.400:b1…` (solo para la 1) + `sesión 116: decisión cacheada` (la 2ª congelada). **Code review**: nit aplicado — `TestRunOcrBatch` NO tiene `setup_method` y el cache de decisión es de CLASE (el put preserva el contador previo; una entrada heredada con re_computes=1 congelaría la página 1 y rompería el test silenciosamente) → `OCRManager.clear_decision_cache()` al inicio del test (patrón de la sesión 136) + comentario de la relación 1:1 imagen→firma del `side_effect`. **480 tests en verde** (1 nuevo), suite canónica completa. | `tests/test_ocr_engine.py` | ✅ **Batch cubierto por integración** |

#### Sesión 2026-08-07-v32 — Benchmark del capítulo completo con la salvaguarda del trigger activa: 0 recomputes, tasa 60.5% (1 medición real)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 140 | **Benchmark real de la salvaguarda del trigger (sesión 136) en el capítulo completo (53 págs, fusion, workers=2, cache fresco, daemon caliente)**. Corrida 17:51→18:26:24 = **35.0 min de pared**, **tasa de traducción 144/238 = 60.5%** (49/53 págs con texto traducido). **Respuestas a las preguntas del benchmark**: (1) **decisiones negativas débiles**: en el cache de decisión del TRIGGER final (v5, 42 firmas únicas) hay **14 negativas débiles** (<3 bloq o conf <0.45), 18 negativas fuertes y 10 positivas; en el cache §8.4.1, **2 de 4 negativas son débiles** (la p17 1 bloq conf 0.53 y una nueva 1 bloq conf 0.147). (2) **recomputes consumidos por el contador=1: CERO** — 0 líneas `[trigger] sesión 136: salvaguarda débil — recompute 1/1` en el log (print de la sesión 138, fuente de verdad en vivo) y `re_computes=0` en las 42 firmas; los 11 hits de decisión cacheada (3 VLM + 8 no VLM) fueron todos HONRADOS sobre la firma principal NO débil (6 bloq conf 0.52/0.88 — la de ~10 páginas del capítulo), ninguna consulta a una negativa débil con detección comparable (cada firma débil es única en el capítulo). (3) **flips a positivo que dispararon VLM: CERO** — la salvaguarda no generó NI UNA inferencia VLM extra. (4) **tasa de traducción vs baseline**: subió vs run1 (57.9%, 121/209) a 60.5% pero bajó vs run_det135 (64.5%, 158/245) — la diferencia NO es de la salvaguarda (que no hizo nada) sino del no-determinismo del trigger: esta corrida disparó **13 VLM** (13 páginas con OCR >60s: máx 930/929s, 8 con recuperación de bloques → líneas de fusión, 4 negativas §8.4.1 registradas + 1 sin registro) vs 11 (run1) y 14 (run_det135); run_det135 tenía más páginas VLM (más bloques recuperados) y pagó 151.7 min, esta corrida pagó 35.0 min. **Artefacto a notar**: una página devolvió `4 híbrido + 197 U-OCR → 7` — el VLM devolvió 197 bloques (misparse/ruido) que el merge colapsó a 7 (sin impacto en el resultado final). **Conclusión**: igual que la salvaguarda §8.4.1 (sesión 135), el seguro del trigger no se cobró en este capítulo — el beneficio queda armado para cuando una firma débil SÍ tenga gemela comparable dentro del TTL (coste máximo ~1 recompute/inferencia por firma débil por TTL, aquí 0). **Herramienta** `tools/medir_salvaguarda_trigger.py` (nuevo): cuenta negativas débiles/consumos/flips desde el cache v5 + correlaciona las líneas del log del servidor (sesión 138) + tasa desde `stats` del checkpoint. Cache conservado para re-medición; backup del previo en `cache/ocr_decision_cache_backup_s139.json`. | `tools/medir_salvaguarda_trigger.py` (nuevo), `run_139_server.log`, `run_139_out.log`, `resultados_progreso_20260807_1751.json` | 📊 **El seguro del trigger tampoco se cobró: 0 recomputes** |

#### Sesión 2026-08-07-v33 — Recompilación del .exe con todo el trabajo de Agosto + smoke test E2E completo (1 build + 1 validación)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 141 | **`.exe` recompilado con PyInstaller 6.21.0** (`env/Scripts/python.exe -m PyInstaller main.spec --clean --noconfirm`, ~10 min): `dist/main/main.exe` **108.6 MB** (18:49, antes 113 MB de una foto del 5-ago sin OCRManager). El spec ya incluía `ocr_engine.py`, `uocr_client.py`, `uocr_daemon.py` en DATAS + `js/` + fixes de stdlib en HIDDEN_IMPORTS (commiteado en la sesión de build). **Bundle verificado**: `dist/main/_internal/` contiene `js/` (5 módulos), `ocr_engine.py`, `uocr_client.py`, `uocr_daemon.py`, `routes/`. **Smoke test E2E completo del .exe** (`--server`, log `exe_smoke.log`): (1) arranque en **12s** + adopción del daemon U-OCR (5177) + preload EasyOCR GPU 40.6s + CT2 es→en/en→es (11.4s, CUDA int8) + YOLO (0.7s) — todo desde `env/` (localización por subida de directorios intacta); (2) `/api/translate` por CT2 fast path: `"HOLA, COMO ESTAS HOY" → "HELLO, HOW ARE YOU TODAY?"` (el primer intento devolvió el texto sin cambios porque CT2 aún no había terminado de precargar — segundo intento OK); (3) `/api/process-page` sobre la pág 5 (artística) en fusion: `engines_used=['easyocr+rapid','unlimited']` (el trigger disparó el VLM en el .exe), **3/3 bloques traducidos** vía CT2 fast path incluyendo el diálogo artístico `'...ERA UNA PROPUESTA' → '...IT WAS A PROPOSAL.'` (conf 1.00/0.95) — la recuperación artística de todo el trabajo de Agosto funciona en el binario. 80-243s/pág (VLM disparado). Entorno limpio al final (server 5174 parado, daemon vivo). **Nota para el usuario**: el acceso directo del escritorio (`Traductor Visual Pro.lnk`) apunta a `dist/main/main.exe` — sigue funcionando sin tocarlo. | `dist/main/main.exe` (recompilado), `exe_smoke.log` | 🚀 **.exe al día + validado E2E** |

#### Sesión 2026-08-07-v34 — Verificación del .exe en modo LAUNCHER (doble clic) + flujo UI completo (1 validación + 1 hallazgo)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 142 | **Verificación del modo launcher (sin `--server`) con interacción real de UI vía preview** (Chrome abierto por el propio launcher con `--app=http://127.0.0.1:5174 --window-size=1400,900`, confirmado por CommandLine). **Flujo UI completo validado en el .exe**: (1) arranque launcher en ~12s, servidor 5174 (PID 3456) + adopción del daemon U-OCR (badge `U-OCR: listo (carga 713s)` visible en la UI) + preloads completos (EasyOCR GPU, CT2 es→en/en→es, YOLO); (2) **carga de PDF real desde la UI** (input file alimentado vía fetch del mismo servidor, evento change nativo → `openFile`): "Página 1 de 53 lista."; (3) **traducción manual de página desde la UI**: p3 → `POST /api/process-page` 200 en **5.1s**, **4 bloques traducidos por CT2 fast path** (`'FROM THE GODFATHER OF SEOLLLANG.'`, `'AND THAT'S WHY'`, `'I WAS HOPING TO MAKE THE MOST OF IT.'`), 4 burbujas renderizadas en el canvas, status `Página 3 traducida: 4 bloques.`, `ocr_engine=fusion` en la respuesta, logs frontend sin errores. La p5 (artística) también completó server-side: fusión `4 híbrido + 4 U-OCR → 3` en 165.3s con los 3 diálogos artísticos traducidos. **HALLAZGO REAL (limitación conocida)**: la p5 tardó 165.3s pero `TIMEOUT_PROCESS_PAGE_MS=120000` (config.py/js/config.js, servido por `/api/config`) aborta el fetch del frontend con `signal is aborted without reason` → la UI muestra "Error traduciendo página" AUNQUE el servidor completó y tradujo (el usuario puede reintentar y recibe el resultado del cache). En p3 (5.1s) no ocurre. **Recomendación**: subir `TIMEOUT_PROCESS_PAGE_MS` a ≥300s para páginas VLM (process_all_pages ya usa 900s) — no urgente, la página normal funciona, pero la artística da error de UI en el primer intento. **Cierre**: servidor de prueba parado, daemon 5177 vivo, `launcher_test.log` eliminado, sin cambios de código (solo AGENTS.md). | — (sin cambios de código) | ✅ **Launcher + UI validados; hallazgo: timeout 120s < páginas VLM** |

#### Sesión 2026-08-11-v35 — PLAN_MANGA_OCR Paso 2: tier comic-text-detector (ONNX CPU) para extracción de texto de manga (1 feature + 10 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 143 | **Nuevo tier de detección de texto de cómic (Tier 3.6) para el tool de extracción `input_manga → output_texto`** (PLAN_MANGA_OCR, compatible con el proyecto). **Paso 1 (entorno) hecho en sesión anterior**: carpetas `input_manga/`/`output_texto/` creadas, `models/comic-text-detector.onnx` descargado (94.7 MB, port ONNX de mayocream del dmMaze comic-text-detector, GPL-3.0; firma verificada: IN `images[1,3,1024,1024]` → OUT `blk[1,64512,7]` + `seg[1,1,1024,1024]` + `det[1,2,1024,1024]`), CUDA verificado. **Paso 2 implementado**: (1) `config.py` — 9 flags `COMIC_DETECTOR_*` (ENABLED, MODEL_PATH, CONF_THRESH 0.4, NMS_THRESH 0.35, MASK_THRESH 0.3, LINE_SCORE_THRESH 0.6, UNCLIP_RATIO 1.5, MAX_REGIONS 60, MIN_AREA_RATIO 0.0005 — defaults de dmMaze); (2) `ocr_utils.py` — `_get_comic_detector_engine` (lazy-load onnxruntime CPU + thread-safe, degrada a `[]`), `_comic_detector_letterbox` (letterbox YOLOv5 1024²/stride 64/relleno 114), `_comic_detector_nms` (NMS por clase como yolov5_utils), `_comic_detector_box_score` (DBNet box_score_fast), `_comic_detector_map_box` (**inversa EXACTA del letterbox**: resta el padding y escala por el contenido real — el inference.py de dmMaze omite el padding y desplaza las cajas ~15-20% en mangas verticales), y los 3 decoders `_comic_detector_{blk,det,seg}_regions` + `_detect_text_regions_comic_detector` (fusión de los 3 heads en regiones del formato de la Ruta C, `source='ctd'`, labels `ctd_eng/ja/line/mask`). Post-proceso replicado de dmMaze: blk → conf=obj*cls + NMS; det (DBNet) → binarizar + contornos + minAreaRect + unclip 1.5 + score > 0.6 (sin pyclipper/shapely, unclip axis-aligned — los crops de la Ruta C son axis-aligned); seg (UNet) → binarizar + blobs cuyo centro NO cae en una región blk/det previa. Corre 100% CPU → **0 VRAM extra**, batch=1 estricto. (3) `tests/test_ocr_utils.py` — **10 tests unitarios con onnxruntime mockeado** (mapeo exacto del letterbox a coords de página, NMS por clase eng/ja, regiones de línea DBNet, máscara no cubierta, blob BGR CHW normalizado, cap MAX_REGIONS, degradaciones sin modelo/error, carga lazy cacheada con `CPUExecutionProvider`). **Validación**: 125/125 en test_ocr_utils.py + **490 passed** en la suite canónica (`--ignore=tests/archive`) + **smoke EN VIVO con el modelo real** sobre la pág 5 del capítulo 43 (render 2x, 1684×1190): **1.36s CPU totales (incl. carga del modelo), 12 regiones (3 blk + 9 líneas), conf 0.81-0.94** — estructura correcta (las líneas anidadas dentro de sus bloques: el bloque (499,863,195,142) contiene líneas en y=871/905/938/968). **Hallazgo pre-existente**: `pytest tests/` (directorio) crashea silenciosamente (exit 1, 0 líneas) por `tests/archive/` (scripts CI stale colectados) — la suite canónica debe correrse con `--ignore=tests/archive` o lista explícita de archivos. | `config.py`, `ocr_utils.py`, `tests/test_ocr_utils.py`, `PLAN_MANGA_OCR.md` | 🚀 **Tier 3.6 listo y validado en vivo (CPU, 0 VRAM)** |

#### Sesión 2026-08-11-v36 — PLAN_MANGA_OCR Paso 3: CLI manga_ocr.py (input_manga → output_texto) + gate UOCR_ENABLED (1 feature + 14 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 144 | **CLI `manga_ocr.py`** (Paso 3 de PLAN_MANGA_OCR): extracción pura de texto de manga — escanea `input_manga/` (PDF/imágenes), procesa cada archivo con `OCRManager.run_ocr()` en modo fusion (EasyOCR GPU + RapidOCR + YOLO de globos), **batch=1 estricto** (una página a la vez, VRAM 4GB) y escribe `output_texto/<archivo>.json` + `.txt` con el schema del plan (`texto`, `bbox [x0,y0,x1,y1]`, `conf`, `motores`, `detector`). Flags: `--input/--output/--zoom` (render fitz, default 2.0)/`--ocr-mode`/`--lang`/`--vlm`/`--pages A-B` (rango 1-indexado, render solo del rango)/`--force`. **Escritura incremental por página** (un crash no pierde lo procesado), `doc_id` = md5 del nombre del archivo [:12] (mismo esquema que process_all_pages → caches de decisión escopeados por documento, sesión 126), texto_plano ordenado por lectura (y, x). **Nuevo gate `UOCR_ENABLED` en `config.py`** (default True = histórico) leído en runtime por `_reforzar_con_unlimited` (ocr_engine.py): anula SOLO el refuerzo VLM (el daemon de 2-8 min/pág) sin apagar YOLO/Ruta C/cls de rotación (a diferencia de `disable_uocr` que apaga todo el pipeline de recuperación) — manga_ocr lo pone a False por defecto y `--vlm` lo deja activo (requiere daemon 5177). **Tests**: 13 nuevos en `tests/test_manga_ocr.py` (schema de bloques por origen hibrido/yolo/ctd/vlm, orden de lectura, escaneo/rango, escritura JSON+TXT, integración de main con OCRManager mockeado: salida real, gate apagado sin --vlm / activo con --vlm, omisión de archivos ya procesados sin --force, `doc_id` pasado) + 1 gate en `TestUnlimited` de test_ocr_engine.py (False → ni daemon ni negativa §8.4.1; True → flujo histórico). **Validación**: **504 passed** (490 + 14) + **smoke EN VIVO**: 3 págs del capítulo 43 → **33.8s** (incl. carga EasyOCR GPU), bloques 4/6/7, la p1 usó `yolo+rutac` (Fase 6 activa), **0 llamadas VLM** (gate funcionando: la p2 detectó panel grande pero no disparó el daemon), JSON+TXT verificados (schema completo, texto_plano ordenado). | `manga_ocr.py` (nuevo), `config.py`, `ocr_engine.py`, `tests/test_manga_ocr.py` (nuevo), `tests/test_ocr_engine.py` | 🚀 **CLI de extracción listo y validado** |

#### Sesión 2026-08-11-v37 — PLAN_MANGA_OCR Paso 5: benchmark del tier CTD en 5 páginas reales — 0 VRAM extra, +17% detección (1 medición)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 145 | **Benchmark de extracción con el tier comic-text-detector** (Paso 5 de PLAN_MANGA_OCR): `tools/benchmark_extraccion_ctd.py` mide las MISMAS 5 páginas reales (2-6 del capítulo 43, zoom 2.0) en el MISMO proceso con y sin el tier — pase ADITIVO idéntico al del Paso 4 (regiones CTD → Ruta C `_recover_regions_with_easyocr` upscale 3.5× → `_fusionar_blocks_multi`), sampler de VRAM vía nvidia-smi en hilo (0.4s), refuerzo VLM apagado (gate UOCR_ENABLED). **Resultados**: **(1) VRAM — el tier añade 0 MiB** (ΔCTD = 0 en las 5 páginas; pico total 3863 MiB = 94% de los 4GB con el daemon cargado, idle daemon 2262 MiB): confirmado medido que la decisión de diseño "comic-text-detector 100% CPU" cumple (los modelos pequeños en CPU no compiten por VRAM, mismo hallazgo que el benchmark de RapidOCR). **(2) Tiempos por página (steady-state)**: pipeline base fusion 3.7-4.4s; detección CTD **0.75-0.89s CPU** (el smoke de la sesión 143 medía 1.36s incluyendo la carga del modelo); re-OCR Ruta C de los crops 2.0-4.0s → el pase CTD completo añade ~3-5s/pág (~+90% del tiempo de página: el coste está en el re-OCR de crops, no en la detección). **(3) Tasa de detección**: 29 → 34 bloques únicos (**+5, +17.2%**); 85 regiones CTD → 21 bloques recuperados por Ruta C → solo 5 NUEVOS tras el merge (16 colisionan con la cobertura YOLO/híbrido ya existente); la p4 dio +0 (página bien cubierta → todo duplicado). **Implicación para el Paso 4** (integración en OCRManager): la detección CTD es barata (0.8s) pero el re-OCR de crops cuesta 2-4s y la mayoría de regiones DUPLICA las de YOLO — el Paso 4 debe (a) gate tipo YOLO (correr CTD solo en páginas débilmente detectadas, <3 bloques o conf<0.35) y/o (b) deduplicar regiones CTD vs YOLO por IoU ANTES de la Ruta C, para cobrar el +17% sin pagar el re-OCR duplicado. Artefacto: `benchmark_ctd_results.json` (git-ignored, medición). | `tools/benchmark_extraccion_ctd.py` (nuevo), `benchmark_ctd_results.json` | 📊 **CTD: 0 VRAM, +17.2% detección, coste 0.8s detección + 2-4s re-OCR** |

#### Sesión 2026-08-11-v38 — PLAN_MANGA_OCR Paso 4: integración del tier comic-text-detector en OCRManager — Ruta C + gate en cascada + dedup vs YOLO (1 feature + 9 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 146 | **Integración del tier comic-text-detector en OCRManager** (Paso 4 de PLAN_MANGA_OCR, con las lecciones del benchmark del Paso 5 incorporadas). **(a) `_ruta_c_yolo` ahora retorna `(yolo_blocks, regiones_utilizadas)`** — las regiones YOLO usadas se pasan al tier CTD para deduplicar por overlap ANTES de su propia Ruta C. **(b) Nuevo método `_ruta_c_ctd(img, ocr_lang, blocks, avg_conf, yolo_regiones)`** (Fase 6.5): gate en CASCADA evaluado con los bloques POST-YOLO (`COMIC_DETECTOR_GATE_MIN_BLOCKS` = 3 = umbral del trigger, `COMIC_DETECTOR_GATE_MAX_CONF` = 0.35; avg_conf pre-YOLO — convención documentada: si YOLO ya resolvió la página, CTD no corre; si la página sigue débil en alguna dimensión, corre); **dedup 1** de regiones CTD vs `yolo_regiones` por `_overlap_ratio` > `COMIC_DETECTOR_DEDUP_IOU` (0.40 — misma categoría de detección: YOLO ya va a re-OCRear esa zona; lección del benchmark: 85 regiones → 21 recuperadas → solo 5 nuevas); **dedup 2** vs bloques existentes (>0.5, mismo patrón que YOLO: solo diálogo perdido); cada región restante → `_recover_regions_with_easyocr` (upscale 3.5×) → `_fusionar_blocks_multi` con pesos `[easyocr, yolo]` (consistente con el merge YOLO) → `engines_used.append("ctd+rutac")`. **(c) Integración en AMBOS caminos, ANTES del trigger v4.2** (los bloques recuperados pueden elevar la página y evitar el VLM — misma posición que YOLO): `_run_fusion` (single) y Fase A de `run_ocr_batch` (batch). **(d) `disable_uocr` apaga el tier** vía nuevo evento `_ctd_disabled` en ocr_utils.py (mismo patrón que `_yolo_disabled`, chequeado en el detector YOLO-compat y en `_ruta_c_ctd`). Degradación segura en todos los caminos (sin modelo/onnxruntime/error → []). **Tests**: 9 nuevos en `TestRutaCCTD` de test_ocr_engine.py — recupera y fusiona en single (sin VLM: 1 bloque conf 0.6 no dispara), cascada YOLO→CTD (ambos tiers conviven), gate bloquea en página bien detectada (4 bloques conf 0.7 → ni YOLO ni CTD corren), disable_uocr apaga, degradación sin modelo, dedup 1 (región CTD duplicada de YOLO → el detector CTD corre pero NO re-OCRea: `recover.call_count == 1`), dedup 2 (región cubierta por bloque → sin re-OCR), error del detector no tumba la página, y batch Fase A (recupera en run_ocr_batch). **Validación**: **513 passed** (504 + 9; suite canónica con `--ignore=tests/archive`) + **smoke EN VIVO** sobre págs 5-8 del capítulo 43 (VLM apagado, gate UOCR_ENABLED): págs 5-7 bien detectadas (4-6 bloques, conf 0.79-0.98) → gate bloquea CTD correctamente; **pág 8 débil (2 bloques) → YOLO 5 regiones → 0 recuperados; CTD 17 regiones (blk=6, línea=11) → 4 útiles tras dedup → 2 bloques recuperados → merge `2 híbrido+YOLO + 2 CTD → 4` → `engines=['easyocr+rapid','ctd+rutac']`** — el tier aporta en vivo exactamente donde el benchmark lo predijo (páginas que híbrido y YOLO detectan débilmente). 0 llamadas VLM en el smoke. **Code review auto** (sin agente en este build): gate/dedup/pesos/trigger/degradaciones verificados — único detalle cosmético: el print interno de `_ruta_c_ctd` usa `[process-page]` también en batch. | `ocr_engine.py`, `ocr_utils.py`, `config.py`, `tests/test_ocr_engine.py` | 🚀 **Tier CTD aporta al pipeline real (single + batch)** |

#### Sesión 2026-08-11-v39 — PLAN_MANGA_OCR Paso 7: destilación VLM→YOLO — teacher, entrenamiento y loop de corrección (1 feature + 18 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 147 | **Destilación: el daemon VLM (teacher) enseña al detector YOLO de globos (student)** — el flujo completo de "darle clases" al modelo con tu hardware (GTX 1050 Ti). **(a) `tools/etiquetar_con_vlm.py`** (teacher): corre el daemon U-OCR (5177) sobre páginas reales del PDF (fitz zoom 2.0), toma sus bloques como etiquetas de regiones y escribe dataset YOLO (`train_data/vlm/{train,val}/images+labels` + `manifest.json` con trazabilidad y split determinista por semilla 42). **Clases**: `text_bubble=0`/`text_free=1` (las del modelo ogkalu) — cada bloque del teacher se clasifica con un ORÁCULO: si solapa una detección del ogkalu actual (IoU>0.3) hereda su clase (el student aprende la semántica del pretrained); si no, por el type semántico del VLM (title/header → free, text → bubble). Filtros de calidad (tamaño mínimo, área máx 60% de página, dedup por IoU conservando la mayor). **`--append`**: añade páginas al dataset sin reorganizar lo ya etiquetado (split previo preservado desde el manifest — el val se mantiene estable para A/B honesto). **(b) `tools/entrenar_detector.py`** (student): fine-tune del YOLOv8m ogkalu con ultralytics — **anti-olvido**: arranca de los pesos ogkalu, `freeze=10/20` (congela backbone), `lr0=1e-5..3e-5` bajo, epochs modestos, y **nunca sobrescribe el original**: escribe `models/comic-speech-bubble-detector-finetuned.pt` (swap solo con `--swap`, con backup `.bak` reversible). A/B rápido integrado (detecciones/conf/clases sobre val) + `--device` para forzar cpu. **Validado en vivo**: entrenamiento en GPU de la 1050 Ti a **~2 GB VRAM, 25-40 épocas en 3-4 min** (el daemon se detiene durante el entrenamiento: ocupa 2.25 GB residentes — recarga ~80s). **(c) Loop de CORRECCIÓN ("darle clases" a mano)**: `tools/exportar_anotaciones.py` exporta las pseudo-etiquetas a un workspace YOLO plano (`train_data/corregir/` con `images/`, `labels/`, `classes.txt`, `data.yaml` y `LEEME.txt` en español) listo para abrir en **X-AnyLabeling** (SAM: un clic por globo, autosave, atajos D/A/Ctrl+Alt+K); `tools/fusionar_correcciones.py` fusiona las correcciones con las pseudo-etiquetas — solo las páginas tocadas cambian (`.txt` ausente = no corregida → conserva el teacher; `.txt` vacío = página sin texto → ejemplo negativo), respeta el split original del manifest, valida clases/coords, y con `--train` re-entrena de un comando. **(d) Resultado honesto del primer ciclo**: teacher sobre 53 págs → **37 páginas (31 train + 6 val), 121 etiquetas (115 bubble / 6 free — imbalance real del capítulo)**; fine-tune 25 épocas freeze=20 lr=3e-5: **detecta MÁS regiones (34 vs 27) pero el A/B riguroso por IoU contra las etiquetas del teacher en val da recall 56% vs 62.5% de ogkalu** — con 37 páginas aún no supera al pretrained (y la clase free 6/121 no es aprendible); **NO se activa el modelo** (sin swap) — el camino es el loop: corregir en X-AnyLabeling (oro) → merge → retrain → A/B, iterando por capítulo. Lección del primer run: fine-tune agresiva (40 épocas lr=1e-4 freeze=10) COLAPSA el modelo en datasets pequeños (mAP 0.995→0.077); la receta conservadora no rompe. **Tests**: 18 nuevos en `tests/test_correccion_detector.py` (clasificación del teacher con oráculo, normalización/filtros/dedup, export del workspace, merge preservando split + página nueva → train + vaciada → negativo + filtro de inválidas + layouts de labels, `--train` lanza el entrenador, data.yaml, swap reversible). **Validación**: **531 passed** (513 + 18). .gitignore: `input_manga/`, `output_texto/`, `train_data/`. | `tools/etiquetar_con_vlm.py` (nuevo), `tools/entrenar_detector.py` (nuevo), `tools/exportar_anotaciones.py` (nuevo), `tools/fusionar_correcciones.py` (nuevo), `tests/test_correccion_detector.py` (nuevo), `.gitignore` | 🎓 **El pipeline de "darle clases" está armado y validado E2E (teacher→train→corregir→retrain)** |

#### Sesión 2026-08-11-v41 — "comienza a enseñarle": teacher sobre 3 series nuevas en .webp + 4 retrains — ninguno supera a ogkalu con el umbral real (1 feature + 4 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 149 | **Teacher extendido a carpetas (`--carpeta`)**: el usuario puso 865 mangas en .webp en `input_manga/` (sesión 148) y pidió seguir enseñando al detector. `tools/etiquetar_con_vlm.py` ahora acepta **`--carpeta <dir>`** (además de `--pdf`): las imágenes se leen en orden natural (0.webp, 1.webp, …, 10.webp), se exige exactamente una fuente (argparse.error si ambas/ninguna), y cada documento usa un **PREFIJO propio en los nombres de página** (`{prefijo}_pNNN`, stem sanitizado a 24 chars) para que capítulos de series distintas no colisionen en el manifest ni en train/val — el append puede acumular varias series sin reorganizar lo etiquetado (split previo conservado, mismo patrón que antes). | `tools/etiquetar_con_vlm.py` | 📚 **El teacher ya come los .webp anidados** |
| 150 | **Primer lote de enseñanza con variedad real**: 3 series DIFERENTES de `input_manga/` (29458/cap `1490498` 5 págs → 2 train+1 val, 25 etiquetas; 21016/cap `1457338` 7 págs → 1 train+1 val, 5 etiquetas; 29158/cap `1103524` 14 págs → 7 train+1 val, 14 etiquetas). Dataset final `train_data/vlm`: **63 páginas (54 train + 9 val), 200 etiquetas (177 bubble / 23 free)** — las series nuevas aportan 13 páginas de estilos distintos al corpus del cap 43/47 (50 págs). El val ahora mezcla 3 series (más difícil que el val de una sola serie de la sesión 147 — ogkalu cae de 62.5% a 55.6% recall en este val nuevo). | `train_data/vlm/` | 📊 **63 págs / 200 etiquetas de 5 series** |
| 151 | **4 retrains probados, NINGUNO supera a ogkalu con el umbral real del pipeline (0.25)**: (a) **v3** 40 épocas freeze=20 lr=3e-5 + 200 sintéticas (`--extra-data synth`): mAP50 val 0.044, A/B conf=0.10 recall 63.0% vs 55.6% (¡parecía ganar!) pero **a conf>=0.25 colapsa a 7.4% (2/27)** — detecta más cajas pero con conf media 0.21, y el pipeline las descarta. (b) **v4** 10 épocas freeze=20 lr=1e-5 + sintéticas: 6 detecciones vs 30, recall val 4%. (c) **v5** 20 épocas freeze=20 lr=1e-5 SIN sintéticas (54 reales): 1 detección, mAP50 0.0155. (d) **v6** 10 épocas freeze=10 lr=1e-4 SIN sintéticas: 3 detecciones, mAP50 0.0085. **Lección clave**: con 63 páginas (mayoría de UNA serie) el fine-tune sobre-entrena a la serie del train y DES-CALIBRA la confianza — el recall bruto a conf 0.10 puede subir (+7.4 pts en v3) pero la confianza colapsa por debajo del umbral del pipeline (0.25), así que el modelo real traduciría PEOR. **Decisión: NO se activa ningún finetuned** (sin swap; `models/comic-speech-bubble-detector-finetuned.pt` queda con el mejor, v3, para referencia). El camino correcto sigue siendo el loop de CORRECCIÓN (X-AnyLabeling con oro → retrain) que ya está armado; más pseudo-etiquetas del teacher no desbloquea el salto de calidad con 4 GB. | `tools/entrenar_detector.py` (sin cambios), `models/`, `train_data/` | 🎯 **Pseudo-etiquetas solas ya no escalan: toca corregir a mano** |

**Tests**: 4 nuevos en `tests/test_correccion_detector.py` (`TestEtiquetarConVlm`): `_paginas_de_carpeta` orden natural y filtro de extensiones, `_prefijo_documento` saneado/truncado a 24 chars (stem vacío → 'doc'), `main --carpeta` etiqueta con prefijo por documento (daemon mockeado; 3 webp generados con cv2 → 2 train + 1 val con nombres `{prefijo}_pNNN`, manifest con `fuente` = carpeta, clase oráculo bubble), y `main --append` con DOS carpetas distintas no colisiona (nombres separados por prefijo, split previo conservado: [train,val,train,val]). **Validación**: **541 passed** (537 + 4). **Smoke en vivo** de la integración completa: teacher sobre los 3 capítulos reales (daemon 5177, 26 páginas procesadas en ~50 min — páginas sin diálogo saltadas solas), retrain v3-v6 en GPU (~2.5-3.9 GB VRAM, el daemon se paró durante el entrenamiento y se relanzó con `uocr_client.spawn_daemon()` al final). Artefactos de medición: `teacher_lote{1,2,3}.log`, `train_v{3,4,5,6}.log`, `_tmp/eval_ab_v3.py` (git-ignored). | `tools/etiquetar_con_vlm.py`, `tests/test_correccion_detector.py`, `AGENTS.md` | 🧪 **El ciclo se repite pero el salto requiere oro** |

#### Sesión 2026-08-11-v42 — entrenar_detector.py: flag --conf + gate de swap por el umbral real del pipeline (1 feature + 4 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 152 | **`--conf` en `tools/entrenar_detector.py` + gate del swap por el umbral real**: la lección de la sesión 151 (v3 detectaba MÁS a conf 0.10 pero colapsaba a 0.25 — el umbral que usa el pipeline) quedó mecanizada en el trainer. **(a)** `_eval_rapida` ahora recibe `conf` y `labels_dir` y mide **recall por IoU (>0.3) contra las GT del val al umbral `--conf`** — el A/B post-entrenamiento se hace con el MISMO umbral de producción (`config.YOLO_CONF_THRESH` = 0.25 por defecto). **(b) `--swap` SOLO surte efecto si el fine-tuned GANA el recall a ese umbral**; si pierde imprime `⛔ --swap IGNORADO: des-calibrado o sin ganancia` y no toca `YOLO_MODEL_PATH` — imposible activar un modelo des-calibrado por accidente. **(c) Fix de métrica**: el recall ya no salta las páginas donde un modelo no detecta nada (`boxes is None → continue` saltaba el conteo de GT — un modelo que falla una página no pagaba su recall); ahora `boxes=[]` → 0 hits sobre su GT, el A/B es honesto para ambos modelos. **Tests** (4 nuevos en `TestEntrenarDetector` de test_correccion_detector.py): `_eval_rapida` pasa el umbral a predict y calcula recall contra las GT del val (detección conf 0.9 cubre la GT), una detección conf 0.15 < 0.25 NO cubre la GT (recall 0.0, el fake simula el filtrado de ultralytics), y los dos gates del swap en `main` — perdedor a `--conf` ignora `--swap` (no copia el finetuned sobre el original), ganador activa con backup `.bak` reversible. **Validación**: **546 passed** (541 + 4/5 nuevos; suite canónica con los 10 archivos EXPLÍCITOS — `pytest tests/` sigue crasheando por `tests/archive/` obsoleto, pre-existente). | `tools/entrenar_detector.py`, `tests/test_correccion_detector.py`, `AGENTS.md` | 🛡️ **Nunca se activa un modelo des-calibrado: el swap exige ganar al umbral real** |

#### Sesión 2026-08-11-v43 — retrain SOLO sintéticas: el dominio sintético NO des-calibra (aislamiento del experimento v3-v6) + 2 fixes (1 experimento + 2 fixes)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 153 | **Aislamiento del dominio sintético**: retrain con las 200 sintéticas SOLAS (train = `train_data/synth`, val = las 9 páginas REALES del VLM — dataset `train_data/synth_solo` armado con `_tmp/armar_synth_solo.py`), receta de la sesión 151 (freeze=20 lr=3e-5, 40 épocas, imgsz 512, GPU). **Resultado decisivo**: el A/B por IoU contra las GT del val real con `_eval_rapida` da **conf>=0.10 → synth_only recall 63.0% vs ogkalu 55.6% (166 det vs 41, conf media 0.242); conf>=0.25 (el umbral REAL del pipeline) → EMPATE 55.6% vs 55.6%, pero con conf media 0.387 y cobertura de texto libre MUY superior (16 free vs 2 de ogkalu)**. Conclusión: a diferencia de v3 (VLM+synth, que colapsaba a 7.4% a 0.25), las sintéticas SOLAS **no** des-calibran la confianza en manga real — el daño venía de la mezcla con el train VLM de una sola serie, no del fondo plano sintético. El mAP50 de entrenamiento en val real fue 0 durante toda la corrida (el modelo genera muchos falsos positivos de conf baja que hunden la precisión mAP pero el recall por IoU sigue siendo sano). No se activa el swap (empate, no victoria — el gate `--conf` de la sesión 152 funciona como diseño). Modelo de referencia guardado en `models/comic-speech-bubble-detector-finetuned-synth.pt`. | `tools/entrenar_detector.py`, `models/comic-speech-bubble-detector-finetuned-synth.pt` (nuevo), `train_data/synth_solo/` (nuevo), `_tmp/` | 🧪 **Las sintéticas solas NO dañan la calibración — el train VLM era el problema** |
| 154 | **Fix de aislamiento en tests del swap**: `_mocks_main` usaba `mocker.spy(mod, "_swap_model")` — un spy EJECUTA la función real, y `main()` la llama con la ruta REAL de `--weights` (default de argparse), así que `test_main_swap_ganador_activa` copió el `best.pt` falso del test (4 bytes `b"BEST"`) sobre `models/comic-speech-bubble-detector.pt` de producción (corrupción detectada al intentar el retrain de esta sesión: `pickle data was truncated`). Fix: `mocker.patch.object(mod, "_swap_model", return_value=...)` (mock puro, nunca toca disco) + aserción de que el mock recibió el modelo real de producción como origen. Restaurado el modelo base desde `comic-speech-bubble-detector.pt.bak` (52 MB, el backup de la sesión 147). | `tests/test_correccion_detector.py`, `models/comic-speech-bubble-detector.pt` | 🛡️ **Los tests ya no pueden corromper el modelo de producción** |
| 155 | **Fix del quirk de `project` en `_entrenar`**: ultralytics `get_save_dir` redirige `project` RELATIVO bajo `SETTINGS['runs_dir']` (`runs/detect/models/<name>/` en vez de `models/<name>/`), así que el trainer terminaba las 40 épocas y luego fallaba `FileNotFoundError: No se generó best.pt`. Fix: `project = Path(project).resolve()` (absoluto) antes de `model.train(...)` — best.pt aparece donde el trainer lo espera. | `tools/entrenar_detector.py` | 🔧 **El retrain ya no pierde el best.pt por resolución de rutas** |

**Validación**: **546 passed** (suite canónica de 10 archivos explícitos; `tests/archive/` obsoleto sigue crasheando la recolección por directorio, pre-existente). Daemon reiniciado y `ready` (2.25 GB VRAM) tras el entrenamiento. Artefactos: `train_data/synth_solo/` (git-ignored), `_tmp/armar_synth_solo.py`, `_tmp/eval_ab_synth.py`, `/tmp/train_synth_solo.log`. | `tests/test_correccion_detector.py`, `tools/entrenar_detector.py`, `AGENTS.md` | 🧪 **Experimento aislado + 2 bugs corregidos de paso** |

#### Sesión 2026-08-11-v44 — oro + sintéticas: el oro REAL (X-AnyLabeling) también des-calibra a 0.25 (respuesta a "¿más capítulos mejora?") (1 experimento)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 156 | **Experimento oro+synth (sesión 156): el oro corregido a mano también des-calibra la confianza a 0.25**. El usuario SÍ corrigió en X-AnyLabeling: verificado 33/63 páginas de `train_data/corregir/` perdieron ≥1 caja del teacher (IoU>0.8 mismo cls — oro real, no copia). Dataset `train_data/oro_synth`: train = 200 sintéticas + 54 oro (excluyendo las 9 val para no contaminar), val = las 9 reales. Retrain receta synth_solo (freeze=20 lr=3e-5, 40 épocas, GPU). **A/B por IoU sobre el val real completo a los DOS umbrales**: conf>=0.10 → oro_synth recall 63.0% (empata con synth_solo) pero con SOLO 59 det (vs 166 de synth_solo, conf media 0.206) — más selectivo y preciso; **conf>=0.25 (pipeline real) → COLAPSA a 7.4% (2/27), igual que v3 (VLM+synth)**. Conclusión clave: **el patrón no es del VLM ni del oro — es del dato REAL de una sola serie en el train**. Las 200 sintéticas SOLAS mantienen la calibración (55.6% a 0.25, sesión 153) porque su dominio neutral no sesga la distribución de confianza; cualquier inyección de páginas reales de pocas series (VLM 54, oro 54) re-sesga el modelo hacia esa serie y des-calibra la confianza por debajo de 0.25. **Respuesta a la pregunta del usuario "¿si pongo más capítulos mejorará?"**: más capítulos → más pseudo-etiquetas → NO (ya medido, sesión 151); más capítulos corregidos → sigue sin escalar con la receta actual de fine-tune completo (54 oro ≈ 54 VLM ≈ mismo colapso). El salto requiere O BIEN oro de MUCHAS series (cientos de páginas variadas, el val mixto de la sesión 150 ya hizo caer a ogkalu de 62.5%→55.6% — el modelo necesita esa variedad para generalizar sin sobreajustar a UNA serie), O BIEN un enfoque de calibración distinto (entrenar solo la cabeza de clasificación, o usar el detector a conf baja con re-escaleo — investigar en la próxima sesión). El gate `--conf` de la sesión 152 volvió a bloquear el swap correctamente. | `tools/entrenar_detector.py`, `models/finetune_oro_synth/weights/best.pt` (nuevo), `train_data/oro_synth/` (nuevo), `_tmp/armar_oro_synth.py`, `_tmp/eval_ab_oro.py` | 🧪 **El oro real tampoco supera a ogkalu con la receta actual — el cuello es la variedad de series, no la calidad de etiquetas** |

#### Sesión 2026-08-11-v45 — test del fix de project en _entrenar (quirk runs/detect, sesión 153) (1 test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 157 | **Test de regresión del quirk de `project` en `_entrenar`**: `test_entrenar_resuelve_project_a_absoluto` verifica que `_entrenar` pasa `project` ABSOLUTO a `model.train()` — el fix de la sesión 153 que impide que ultralytics redirija el `project` relativo bajo `SETTINGS['runs_dir']` (`runs/detect/<project>/...`). Patrón: FakeYOLO que captura `kwargs['project']` y simula ultralytics escribiendo `best.pt` en su `save_dir`; `monkeypatch.chdir(tmp_path)` para que el `project` relativo (`mi_proyecto_relativo`) se resuelva DENTRO del tmp (el test no escribe en el repo — verificado, no queda ningún dir suelto). Aserciones: `os.path.isabs(project_visto)` es True, el valor coincide con `str((tmp_path / "mi_proyecto_relativo").resolve())`, y `best.pt` aparece exactamente en `project/name/weights/best.pt` (sin la redirección a `runs/detect/`). **Validación**: **547 passed** (546 + 1). | `tests/test_correccion_detector.py` | 🧪 **El fix de la sesión 153 queda blindado contra regresiones** |

#### Sesión 2026-08-11-v53 — Desglose por clase en el calificador + grid con colores por clase: la comparación visual synth_solo vs ogkalu sobre el texto libre (1 feature + 1 test + 1 medición)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 165 | **El calificador ahora desglosa el recall por clase** (0=text_bubble, 1=text_free): `_leer_gt` conserva la clase, `_evaluar_pagina`/`_evaluar_workspace` agregan `recall_bubble`/`recall_free` (y contadores `gt_/hit_bubble/free`) a la ronda, y el reporte imprime `Por clase: globos X/Y (%) | texto libre X/Y (%)`. **El grid distingue clases con color**: globos del oro en verde, **texto libre del oro en magenta**, detecciones del modelo en rojo (leyenda actualizada). Fix de robustez: el nombre del grid pasa a resolución de **segundos** (`%Y%m%d_%H%M%S`) porque dos corridas en el mismo minuto se sobrescribían (pasó en esta sesión: ogkalu y synth_solo corrieron a la vez → mismo grid). **Comparativa real sobre las 63 páginas (mismo marcado, imgsz 640 CPU)**: ogkalu → globos 115/187 (61%) | **texto libre 4/23 (17%)**; synth_solo → globos 121/187 (65%) | **texto libre 6/23 (26%)** — el synth_solo encuentra **50% más texto libre** (6 vs 4 de 23) y +4 globos, confirmando visual y numéricamente que detecta mejor el free (aunque ambos siguen débiles en free, 17-26%). Grids con colores por clase: `train_data/calificaciones_grid_ogkalu_cls.png` y `train_data/calificaciones_grid_20260811_2350.png` (ambos 960×1502). Historial limpiado a 4 rondas distintas (baseline ogkalu, puntual, ogkalu cls, synth cls). **Validación**: **557 passed** (556 + 1: `_evaluar_pagina` desglosa clases — globo encontrado + free perdido → bubble 1/1, free 0/1). | `tools/calificar_detector.py`, `tests/test_correccion_detector.py`, `train_data/calificaciones.json`, `train_data/calificaciones_grid_ogkalu_cls.png`, `train_data/calificaciones_grid_20260811_2350.png` | 🎨 **La comparación visual por clase responde: synth_solo detecta 50% más texto libre que ogkalu** |

#### Sesión 2026-08-11-v54 — Modo comparativa [ORO | MODELO] lado a lado + preview visual en el escritorio (1 feature + 1 test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 166 | **Nuevo modo `--comparativa` en el calificador**: genera un montaje lado a lado donde **cada fila es una página con texto perdido** — izquierda el oro (globos en verde, texto libre en magenta), derecha las detecciones del modelo (rojo), con cabecera `página hits/gt recall%` y franja de leyenda. Selecciona las páginas con fns>0 (peor recall primero, máx `--comparativa-paginas` 8) y reusa `_predecir` + `_par_comparativa` (letterbox extraído a helper). El path del PNG se guarda en la ronda del historial (clave `comparativa`). **Comparativa real sobre las 63 páginas** (ogkalu producción, imgsz 640 CPU): `train_data/comparativa_oro_dl.png` (600×3666, 8 filas: 1490498_p004 0/9, 1457338_p001 0/4, p009 0/2, p036 0/2, 1103524_p014 0/1, …). **Preview en el escritorio**: la imagen se empaquetó en `train_data/comparativa_preview.html` (base64 embebida, escalada a 900px) y se registró en el tab Preview de Freebuff Desktop — el usuario ve oro vs DL sin abrir la imagen en disco. Hallazgo lateral: `pytest tests/` (directorio completo) **mata el proceso sin output** porque `tests/archive/test_endpoint.py` (no trackeado) hace `sys.exit(1)` a nivel de módulo al ser colectado — la suite canónica se ejecuta con los 10 archivos trackeados explícitos (558 passed). **Validación**: **558 passed** (557 + 1: `_generar_comparativa` incluye la página con fns en el montaje y escribe el PNG; sin páginas perdidas devuelve None). | `tools/calificar_detector.py`, `tests/test_correccion_detector.py`, `train_data/calificaciones.json`, `train_data/comparativa_oro_dl.png`, `train_data/comparativa_preview.html` | 👁 **El usuario ve de un vistazo qué páginas tienen diálogo que el modelo no encuentra — el bucle marcar→calificar→corregir se vuelve visual** |

| 169 | **Fusión del oro corregido al dataset + re-entrenamiento + A/B** (respuesta a "fusiona las correcciones al dataset y re-entrena para ver el A/B vs el modelo original"). El usuario corrigió el oro completo a mano con el corrector de la sesión v56: **63/63 páginas revisadas, 210 → 374 cajas (333 globos + 41 texto libre), 46 cajas gigantes (>25% área) → 0** (quedan 8 tiras anchas-finas marcadas por la heurística w>0.5, todas con área ≤10% — legítimas). `fusionar_correcciones.py` aplicó el oro a `train_data/vlm` (diffs por página p. ej. p023: 23→5 — el teacher metía cajas de más). **Entrenamiento**: `entrenar_detector.py` receta estándar (desde ogkalu, freeze=10, lr0=1e-4, imgsz 512, batch 4, +synth en train, `--name finetune_oro`). **Detalle operativo**: el primer intento vía `nohup` murió sin traceback en la época 16 (probable cierre del proceso por el harness); relanzado como proceso desacoplado con PowerShell `Start-Process` + `PYTHONUTF8=1` y stdout/stderr separados (`_tmp/train2.log`/`.err.log`) → completó. **Early stopping sano**: patience=15 cortó en la época 16 (best = época 1, mAP50 0.445). **A/B a conf≥0.25 sobre 8 págs val (50 GT): ogkalu 66.0% (30 det, conf 0.806, bubble 28/free 2) vs fine-tuned 72.0% (37 det, conf 0.418, bubble 26/free 11) — el fine-tuned GANA +6 pts y multiplica x5.5 la detección de texto libre (2→11), efecto directo del oro corregido con más cajas libres**. SIN `--swap` (el modelo de producción queda intacto; para activarlo: `entrenar_detector.py --swap` solo surte efecto si gana el A/B, que es el caso). Artefactos: `models/comic-speech-bubble-detector-finetuned.pt` (best de la época 1), `models/finetune_oro/`. El usuario preguntó por correr el entrenamiento en la nube gratis — se respondió (Colab T4 / Kaggle P100 / Ultralytics HUB). | `tools/fusionar_correcciones.py` (uso), `tools/entrenar_detector.py` (uso), `models/comic-speech-bubble-detector-finetuned.pt`, `models/finetune_oro/`, `train_data/vlm` (oro aplicado), `train_data/vlm_aug` (reconstruido), `_tmp/train2.log` | 🎯 **El fine-tuned con tu oro supera a ogkalu a conf 0.25 (72% vs 66%) y detecta 5× más texto libre** |

#### Sesión 2026-08-13-v77 — SFX CJK repetitivos sin ampliar los modelos (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 195 | **Detección de SFX CJK inequívocos**: repeticiones del mismo glifo en kana, jamo/hangul o hanzi/kanji (`ゴゴゴ`, `哈哈哈`, `ㅋㅋㅋ`) se preservan; palabras CJK normales no se clasifican como SFX. | `translator.py`, `tests/test_translator.py` | **Evita traducir onomatopeyas claras como si fueran diálogo** |

**Validación**: regresión CJK en verde y `run_ci.py --skip-cov` completado: **644/644 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v78 — Detección conservadora de bloques mixtos sin modelos nuevos (1 mejora + tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 196 | **Code-switching dentro del mismo globo**: `_detect_mixed_languages()` identifica solo evidencia fuerte de los idiomas actuales (es/en/pt/fr/de/it/ja/ko/zh), mantiene el idioma dominante para memoria/diagnóstico y expone `source_langs` + `mixed_source` por bloque. Si el idioma dominante ya es el destino —caso que antes retornaba intacto— se prueba el traductor Google existente con `source=auto`; si falla o devuelve lo mismo, se conserva el original. Kanji acompañado de kana se trata como japonés; una palabra inglesa aislada no dispara el gate. No se agregan idiomas, diccionarios de usuario ni modelos OCR/traducción. | `translator.py`, `routes/api.py`, `tests/test_translator.py`, `tests/test_api.py` | **Traduce frases incrustadas como `No puedo go home` sin corromper nombres, préstamos aislados ni CJK** |

**Validación**: **15 tests** de mezcla/API en verde y `run_ci.py --skip-cov` completado: **650/650 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (372 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v79 — Gates finales de calidad y robustez del contrato OCR (2 mejoras + tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 197 | **Barrera de traducción en idioma origen**: antes de aceptar CT2/Google, se rechazan únicamente salidas con evidencia fuerte de que siguen en el idioma origen (`the house is big` para es, o kana/hangul/hanzi inequívocos). No se aplica a nombres cortos, préstamos ni textos con evidencia insuficiente; el fallback existente sigue funcionando. | `translator.py`, `tests/test_translator.py` | **Reduce traducciones falsas que parecían válidas y evita que el texto original se presente como resultado** |
| 198 | **Gate OCR para basura débil y payloads malformados**: los bloques no-CJK de baja confianza que coinciden con `_es_ocr_noise()` se descartan antes de inpainting; CJK queda protegido. Además, confianzas inválidas de dict/tupla se omiten sin tumbar el batch ni devolver 500. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Menos falsos positivos/inpainting innecesario y más estabilidad ante respuestas RapidOCR/EasyOCR incompletas** |

**Validación**: `run_ci.py --skip-cov` completado: **657/657 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v80 — Fallback de idioma por bloque aislado (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 199 | **Detección multilingüe resistente a fallos parciales**: si falla el detector sobre el texto combinado de la página, se conserva el fallback solo para los bloques cuyo detector individual también falle; ya no se fuerza `es` para toda la página. | `routes/api.py`, `tests/test_api.py` | **Evita contaminar japonés/inglés/chino con el idioma español por un error puntual del detector y mejora la consistencia entre globos** |

**Validación**: `run_ci.py --skip-cov` completado: **658/658 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (372 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v81 — NMS de fusión consciente del texto (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 200 | **NMS final del OCR sensible al contenido**: el descarte ya no depende únicamente de un solapamiento espacial; solo elimina una caja solapada cuando su texto es idéntico o suficientemente parecido. Dos líneas distintas con cajas altas que se cruzan parcialmente se conservan, mientras la deduplicación de resultados equivalentes sigue ocurriendo. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Reduce falsos negativos de OCR sin reintroducir duplicados equivalentes; mejora cobertura de diálogos multilingües y SFX** |

**Validación**: `run_ci.py --skip-cov` completado: **659/659 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v82 — Validación de entradas cacheadas y nombres CJK (2 mejoras + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 201 | **El caché respeta los gates de traducción**: una respuesta antigua pasa ahora por la misma validación de idioma origen y anti-basura que CT2/Google; si falla, se reintenta con los motores existentes y se evita propagar el error entre páginas. | `translator.py`, `tests/test_translator.py` | **Elimina traducciones incorrectas persistentes y mejora la consistencia entre páginas** |
| 202 | **Gate CJK basado en escritura real**: para rechazar una salida no traducida se exige kana, hangul o hanzi según el idioma origen; el contexto de detección ya no basta para rechazar nombres romanizados como `Tanaka` ni romper honoríficos cacheados. | `translator.py`, `tests/test_translator.py` | **Reduce falsos positivos en nombres propios y mantiene la calidad multilingüe** |

**Validación**: `run_ci.py --skip-cov` completado: **660/660 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v83 — Memoria de página con gates y nombres CJK preservados (3 mejoras + tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 203 | **La memoria por documento valida sus entradas existentes**: tanto el endpoint manual como el flujo de páginas comprueban idioma origen y anti-basura antes de reutilizar una traducción; las entradas inválidas se descartan y se reintenta con los motores actuales. | `routes/api.py`, `translation_memory.py`, `tests/test_api.py` | **Evita que un error antiguo se propague por todo el capítulo y mantiene consistencia entre páginas** |
| 204 | **Invalidación exacta de memoria**: se añade `discard()` thread-safe para retirar una entrada que ya no cumple los gates, sin limpiar el resto del documento. | `translation_memory.py` | **Corrección localizada y segura de memoria contaminada** |
| 205 | **Gate CJK proporcional al script**: una frase completa en kana/hangul/hanzi sigue bloqueándose si permanece en origen, pero un nombre aislado conservado dentro de una traducción no invalida el resultado. | `translator.py`, `tests/test_translator.py` | **Menos falsos positivos en nombres propios y honoríficos multilingües** |

**Validación**: `run_ci.py --skip-cov` completado: **662/662 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (363 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v84 — Preservación de marcadores SFX/pensamiento durante OCR (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 206 | **Los textos reconocidos como SFX antes de normalizar se conservan sin limpiar sus marcadores**: `*sigh*` y `*thinking*` ya no pierden asteriscos ni semántica antes de traducir; filtros de URL, fechas y números siguen ejecutándose primero. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Mejor detección y preservación de pensamiento/onomatopeyas con menos traducciones incorrectas** |

**Validación**: `run_ci.py --skip-cov` completado: **663/663 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v85 — Batch OCR seguro para lotes internos >4 páginas (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 207 | **`OCRManager.run_ocr_batch()` procesa todas las páginas pendientes en ventanas de cuatro**: la API conserva su límite 1-4 por request y el manager directo ya no deja páginas 5+ sin refuerzo U-OCR. Cada ventana mantiene la serialización y el presupuesto de VRAM de la GTX 1050 Ti. | `ocr_engine.py`, `tests/test_ocr_engine.py` | **Evita pérdidas silenciosas de OCR en integraciones internas, scripts y futuros endpoints batch** |

**Validación**: `run_ci.py --skip-cov` completado: **664/664 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v86 — Gate de idioma origen al aprender memoria (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 208 | **La memoria solo aprende resultados que no conservan claramente el idioma origen**: `_learn_translation_if_valid()` reutiliza el mismo gate aplicado a cache, CT2 y Google, evitando almacenar respuestas parciales o sin traducir aunque el validador estructural las considere formalmente aceptables. | `routes/api.py`, `tests/test_api.py` | **Evita contaminar futuras páginas con errores de traducción y mejora estabilidad a largo plazo** |

**Validación**: `run_ci.py --skip-cov` completado: **665/665 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v87 — Wrapping híbrido para texto CJK/latino en frontend (1 mejora)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 209 | **El wrapping del editor distingue tramos CJK de palabras latinas**: una página mixta como `No puedo こんにちは go home` ya no rompe `go home` carácter por carácter ni lo parte por la presencia de japonés; los tramos CJK siguen ajustándose por glifo y el comportamiento latino puro se conserva. | `app.js` | **Mejor maquetación y legibilidad en páginas con español/inglés mezclado con japonés, chino o coreano, sin cambiar modelos ni contrato de OCR** |

**Validación**: smoke test Node con Unicode real para casos mixtos y CJK puro; `run_ci.py --skip-cov` completado: **665/665 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v88 — Detección de code-switching con frases naturales (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 210 | **El detector multilingüe amplía sus señales internas para frases frecuentes**: reconoce casos como `Mi amigo says hello` como español + inglés, sin convertir un préstamo aislado en página mixta. No añade modelos ni exige que el usuario mantenga un glosario. | `translator.py`, `tests/test_translator.py` | **Mejor manejo de globos con cambio de idioma real y menos riesgo de enviar un bloque mixto al modelo equivocado** |

**Validación**: `run_ci.py --skip-cov` completado: **666/666 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v89 — Sincronización de filtros de metadatos entre backend y frontend (1 mejora)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 211 | **El filtro JS incorpora timestamps y metadatos de margen ya presentes en `config.py`**: fechas compactas de exportación y horas como `458pm` se eliminan solo en los márgenes, mientras el texto normal se conserva. | `js/filters.js` | **Reduce falsos positivos visibles y evita que basura de escaneo reaparezca en importaciones o reprocesados del frontend** |

**Validación**: smoke test Node del filtro con metadatos y diálogo normal; `run_ci.py --skip-cov` completado: **666/666 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v90 — Guard de VRAM para nuevos pares CT2 (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 212 | **La carga de un par CT2 nuevo respeta el presupuesto observable de la GPU**: si la GTX 1050 Ti tiene poca memoria libre, ese par se carga en CPU; los pares ya cargados no se desalojan y el comportamiento CPU-only permanece compatible. | `translator.py`, `config.py`, `tests/test_translator.py` | **Evita OOM y cuelgues después de usar varios idiomas/modelos, protegiendo EasyOCR y U-OCR sin añadir modelos** |

**Validación**: `run_ci.py --skip-cov` completado: **667/667 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v91 — Traducción segura de bloques con code-switching (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 213 | **`source=auto` usa Google en modo automático cuando un bloque contiene varios idiomas y el destino difiere del idioma dominante**: se evita enviar una frase mixta a CT2 como si fuera monolingüe; si el resultado no supera los gates, continúan los fallbacks actuales. | `translator.py`, `tests/test_translator.py` | **Mejor calidad en globos con español/inglés u otras combinaciones actuales, conservando frases ya escritas en el idioma destino** |

**Validación**: `run_ci.py --skip-cov` completado: **668/668 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (372 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v92 — Code-switching alemán/inglés con señales internas (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 214 | **El detector reconoce cambios breves alemán/inglés como `Ich liebe you`**: añade señales frecuentes y una regla conservadora para una palabra inglesa distintiva cuando el idioma dominante ya tiene evidencia fuerte. Los préstamos aislados comunes siguen sin activar mezcla. | `translator.py`, `tests/test_translator.py` | **Amplía la detección de páginas mixtas a todos los idiomas actuales sin glosario manual ni modelos nuevos** |

**Validación**: `run_ci.py --skip-cov` completado: **669/669 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v93 — Code-switching romance/inglés en idiomas actuales (1 mejora + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 215 | **El detector amplía las señales internas a francés, portugués e italiano**: frases breves como `Je aime you`, `Eu amo you` e `Io amo you` activan mezcla solo cuando el idioma dominante tiene evidencia suficiente y la palabra inglesa es distintiva. | `translator.py`, `tests/test_translator.py` | **Cobertura multilingüe más uniforme y menor riesgo de usar un modelo monolingüe incorrecto en globos mixtos** |

**Validación**: `run_ci.py --skip-cov` completado: **672/672 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v101 — Decodificación Base64 estricta en imágenes (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 224 | **`_base64_to_cv2` usa `base64.b64decode(..., validate=True)`**: caracteres inválidos ya no se ignoran silenciosamente antes de pasar los bytes a OpenCV; se conserva el soporte para data URI y los formatos existentes. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Rechaza payloads de imagen corruptos de forma determinista y evita OCR/inpainting sobre bytes distintos de los enviados; sin coste apreciable para la GTX 1050 Ti** |

**Validación**: prueba RED confirmada (un Base64 válido seguido de `!` era aceptado), pruebas focalizadas **9/9** en verde. El CI completo de la sesión pasó con **693/693** tests, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (371 MB).

#### Sesión 2026-08-13-v102 — Guardia de píxeles antes de decodificar imágenes (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 225 | **Se inspeccionan las dimensiones de la cabecera antes de `cv2.imdecode()`** mediante Pillow lazy y un límite de 64M píxeles; se conservan formatos que Pillow no reconoce usando el fallback histórico de OpenCV. | `config.py`, `ocr_utils.py`, `tests/test_ocr_utils.py` | **Evita reservas de memoria desproporcionadas por imágenes comprimidas con dimensiones abusivas y mantiene páginas largas legítimas hasta 4096×16000, protegiendo la estabilidad de RAM/VRAM** |

**Validación**: RED confirmado con cabecera simulada de 100000×100000 píxeles; conversión Base64 focalizada **9/9** en verde. El CI completo de la sesión pasó con **694/694** tests, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (372 MB).

#### Sesión 2026-08-13-v103 — Allowlist de rutas del daemon U-OCR (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 226 | **El daemon valida las rutas de `/ocr` y `/ocr-batch` contra la raíz del proyecto y el temporal del sistema**: resuelve rutas reales antes de comparar, bloquea archivos inexistentes/fuera de scope y evita incluir rutas rechazadas en las respuestas de error. El flujo normal del servidor usa exactamente una de esas dos raíces. | `uocr_daemon.py`, `tests/test_uocr_daemon.py` | **Reduce la superficie de lectura/procesamiento arbitrario de archivos por el servicio loopback sin cambiar el pipeline ni la VRAM** |

**Validación**: RED confirmado con helper inexistente; guard focalizado **4/4** en verde. El CI completo de la sesión pasó con **695/695** tests, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (371 MB).

#### Sesión 2026-08-13-v104 — Lanzador sin terminación global de procesos (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 227 | **`start-app.ps1` deja de inspeccionar/matar PIDs por puerto**: si el servidor local ya responde, reutiliza esa sesión; si el proceso nuevo termina porque el puerto está ocupado, informa y sale sin tocar al proceso ajeno. El daemon U‑OCR queda para la adopción segura de `uocr_client`. | `start-app.ps1`, `tests/test_packaging.py` | **Evita pérdida de trabajo, cierres de software ajeno y conflictos de VRAM al reiniciar la aplicación** |

**Validación**: RED confirmado por la presencia de terminación global; tests de packaging **4/4** en verde. El CI completo de la sesión pasó con **697/697** tests, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (371 MB).

#### Sesión 2026-08-13-v105 — CI valida todos los módulos ES6 (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 228 | **El syntax check del CI incluye los módulos frontend importados** (`js/config.js`, `js/filters.js`, `js/theme.js`, `js/toast.js`, `js/utils.js`) además de `app.js`, mediante una lista centralizada `_js_syntax_files()`. | `run_ci.py`, `tests/test_run_ci.py` | **Evita que cambios de timeout, filtros o utilidades rompan la carga del editor aunque `app.js` siga siendo sintácticamente válido** |

**Validación**: RED confirmado por helper inexistente; módulos ES6 focalizados en verde. El CI completo de la sesión pasó con **698/698** tests, syntax Python/JS correcto, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (372 MB).

#### Sesión 2026-08-13-v106 — Allowlist U-OCR compatible con temporales en otra unidad (1 fix + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 229 | **La allowlist del daemon continúa comprobando raíces cuando `commonpath()` encuentra unidades distintas**: en Windows, el proyecto puede estar en `D:` y `%TEMP%` en `C:`; la unidad incompatible se omite y la raíz temporal se evalúa correctamente. | `uocr_daemon.py`, `tests/test_uocr_daemon.py` | **Evita bloquear todas las inferencias U‑OCR del flujo normal después de endurecer la seguridad de rutas** |

**Validación**: RED confirmado con proyecto `D:` y temporal `C:` simulados; guards focalizados **5/5** en verde. El CI completo de la sesión pasó con **699/699** tests, syntax Python/JS correcto, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (403 MB).

#### Sesión 2026-08-13-v107 — Launchers idempotentes con servidor activo (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 230 | **`main.py` y `launcher.py` comprueban 127.0.0.1:5174 antes de arrancar Flask**: si la sesión existente responde, abren la UI y reutilizan el servidor; solo crean el proceso nuevo cuando el puerto no está activo. | `main.py`, `launcher.py`, `tests/test_packaging.py` | **Evita errores de bind, instancias duplicadas y arranques confusos al abrir la app varias veces** |

**Validación**: RED confirmado por ausencia de guard en ambos launchers; tests de packaging **5/5** y `py_compile` en verde. El CI completo de la sesión pasó con **700/700** tests, syntax Python/JS correcto, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (362 MB).

#### Sesión 2026-08-13-v108 — Code-switching seguro con idioma origen explícito (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 231 | **Los bloques mixtos usan Google con `source=auto` aunque el usuario haya fijado el idioma dominante**: una frase como `No puedo go home` ya no entra como si fuera completamente `es→en` a CT2; los bloques monolingües conservan el camino CT2 rápido. | `translator.py`, `tests/test_translator.py` | **Mejora la calidad en imágenes mixtas y evita deformar frases incrustadas en otro idioma, sin agregar modelos ni glosario** |

**Validación**: RED confirmado porque CT2 devolvía `translated` antes del fallback mixto; tests de code-switching focalizados **7/7** en verde. El CI completo de la sesión pasó con **701/701** tests, syntax Python/JS correcto, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (371 MB).

#### Sesión 2026-08-13-v109 — Mensajes de toast sin interpretación HTML (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 232 | **`showToast()` renderiza el mensaje con `textContent`** en vez de interpolarlo dentro de `innerHTML`; los iconos y controles siguen siendo markup estático. Se añade una regresión estática que impide volver a insertar `${message}` directamente. | `js/toast.js`, `tests/test_packaging.py` | **Evita XSS/HTML inyectado desde errores de API, OCR o excepciones mostradas en la interfaz local, sin coste de rendimiento ni cambio visual esperado** |

**Validación**: RED confirmado; tests de packaging **6/6** y `node --check js/toast.js` en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v110 — Presupuesto acumulado de píxeles en OCR batch (1 mejora + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 233 | **`/process-page-batch` limita la suma de píxeles decodificados a 96M** y pasa el presupuesto restante a `_base64_to_cv2(max_pixels=...)`; la comprobación ocurre antes de `cv2.imdecode()` cuando Pillow puede leer la cabecera y también se repite tras decodificar para formatos alternativos. | `config.py`, `ocr_utils.py`, `routes/api.py` | **Evita OOM de RAM al enviar varias páginas comprimidas o largas juntas; mantiene el límite individual de 64M y no cambia el flujo normal de manga** |

**Validación**: RED confirmado en el helper y en el endpoint; pruebas focalizadas **2/2** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v111 — Code-switching con marcador extranjero distintivo (1 mejora + 5 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 234 | **La detección de mezcla reconoce un único marcador extranjero distintivo cuando el idioma dominante tiene al menos dos evidencias** (`I love gracias`, `Ich liebe danke`, `Je veux danke`). Los préstamos ambiguos (`amigo`, `casa`, `non`) quedan fuera para no disparar Google innecesariamente. | `translator.py`, `tests/test_translator.py` | **Reduce falsos negativos en imágenes con frases mayoritarias de un idioma y una palabra insertada de otro, manteniendo el gate conservador y sin nuevos modelos** |

**Validación**: RED confirmado con los casos nuevos; detección mixta focalizada **13/13** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v112 — Aislamiento por usuario en repositorios de páginas (1 mejora + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 235 | **`PageRepository` y `TextBlockRepository` exigen `user_id` y validan la relación `Page → Project → User`** antes de leer o escribir; se bloquean enumeración, lectura, creación y guardado de bloques sobre recursos ajenos. | `models.py`, `tests/test_models.py` | **Corrige una brecha de aislamiento de datos preparada para futuros endpoints CRUD, sin cambiar el flujo OCR actual que no usa estos repositorios** |

**Validación**: RED confirmado por llamadas sin scope; pruebas SQLite en memoria **2/2** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v113 — Validación efectiva de `allow_source_auto` (1 fix + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 236 | **`_validate_lang_params()` pasa `allow_source_auto` a `_validate_lang_code()`**; el modo estricto ya rechaza realmente `source=auto` y conserva el mensaje de idiomas permitidos. | `routes/api.py`, `tests/test_api.py` | **Elimina una opción de validación ignorada y evita que futuros endpoints acepten detección automática cuando exijan origen explícito** |

**Validación**: RED confirmado (`auto` era aceptado con el flag falso); prueba focalizada **1/1** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v114 — Falsos positivos SFX en palabras CAPS multilingües (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 237 | **`_es_sfx()` excluye palabras comunes de los idiomas actuales** reutilizando `_MIXED_LANGUAGE_MARKERS`; `STOP`, `WHAT`, `BONJOUR`, `DANKE`, `CIAO` y `OBRIGADO` ya no se preservan como onomatopeyas, mientras `NARUTO`, `BANG` y `DON` mantienen el comportamiento existente. | `translator.py`, `tests/test_translator.py` | **Evita perder diálogos cortos en mayúsculas y reduce falsos positivos sin agregar modelos ni afectar SFX reconocibles** |

**Validación**: RED confirmado con `STOP`; pruebas de SFX **13/13** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v115 — Interjecciones alargadas fuera del gate SFX (1 mejora + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 238 | **`_es_sfx()` colapsa repeticiones consecutivas para reconocer interjecciones de diálogo** (`NOOOOO`, `AAAAH`, `HELP`, `WAIT`) y las excluye; los SFX repetitivos inequívocos como `GRRRR` siguen preservados. | `translator.py`, `tests/test_translator.py` | **Evita que exclamaciones visuales de diálogo queden sin traducir, sin abrir el gate para ruido repetitivo** |

**Validación**: RED confirmado con `NOOOOO!`; pruebas de SFX **15/15** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v116 — Selector UI occidental compuesto aceptado por la API (1 fix + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 239 | **`_validate_lang_code()` normaliza `eng+spa+fra+deu` a `auto`**, manteniendo la detección por bloque existente; el backend ya no rechaza el valor que ofrece `#sourceLang` ni fuerza una cadena compuesta a un modelo CT2. | `routes/api.py`, `tests/test_api.py` | **Repara el flujo de imágenes con varios idiomas occidentales desde la UI, reduce errores de API y conserva el comportamiento seguro con los modelos actuales** |

**Validación**: RED confirmado porque el alias devolvía `None`; prueba focalizada **1/1** en verde. Pendiente ejecutar el CI completo de la sesión.

#### Sesión 2026-08-13-v100 — Normalización de cajas devueltas por U-OCR (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 223 | **Cajas del daemon acotadas a la página**: `_parse_daemon_blocks` descarta tamaños no positivos/diminutos y recorta coordenadas negativas o fuera de los límites antes de enviarlas a fusión, inpainting y frontend. | `routes/api.py`, `tests/test_api.py` | **Evita falsos positivos visuales, máscaras corruptas y errores por detecciones VLM malformadas** |

**Validación**: `run_ci.py --skip-cov` completado: **692/692 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v99 — Normalización segura de flags OCR en la API (1 mejora + 8 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 222 | **Flags booleanos normalizados explícitamente**: `prefilter`, `force_uocr`, `disable_uocr` y `pure_easyocr` aceptan booleanos y tokens claros (`true/false`, `1/0`, etc.); cadenas arbitrarias ya no se convierten implícitamente en `True`. | `routes/api.py`, `tests/test_api.py` | **Evita activar accidentalmente U‑OCR o desactivar rutas por payloads mal tipados, reduciendo latencia, consumo de VRAM y resultados no reproducibles** |

**Validación**: `run_ci.py --skip-cov` completado: **691/691 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (372 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v98 — Desempate estable del idioma dominante por página (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 221 | **El empate de idiomas usa el detector del texto combinado**: cuando dos idiomas tienen el mismo número de bloques, `_finalize_page_blocks` prioriza `page_fallback_lang`; si no es candidato, mantiene el primer candidato de la lista original. Reordenar globos ya no cambia el idioma dominante ni las claves de memoria/diagnóstico. | `routes/api.py`, `tests/test_api.py` | **Más consistencia de traducción entre páginas y ejecuciones, especialmente en páginas mixtas** |

**Validación**: `run_ci.py --skip-cov` completado: **683/683 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v97 — Rate limiting correcto para redes privadas (1 mejora + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 220 | **La exención del rate limiter queda restringida a loopback**: se elimina la excepción para `192.168.*`; la UI local (`127.0.0.1`, `::1`, `localhost` y loopback IPv4-mapeado) sigue sin límite, pero una conexión LAN conserva los límites configurados. | `ratelimit.py`, `tests/test_ratelimit.py`, `run_ci.py` | **Evita que una futura exposición LAN/proxy deje sin protección los endpoints caros de OCR y traducción** |

**Validación**: `run_ci.py --skip-cov` completado: **682/682 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (372 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v96 — Propiedad segura del proceso U-OCR (1 mejora + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 219 | **El reinicio del daemon ya no mata PIDs por puerto**: se elimina `netstat` + `taskkill /F` sobre cualquier proceso en 5177. El cliente solo termina el `Popen` que él mismo conserva en `_proc`; si el estado de error pertenece a una sesión vieja o a otro servicio, retorna fallback seguro sin destruirlo. | `uocr_client.py`, `tests/test_uocr_client.py`, `run_ci.py` | **Evita terminar software ajeno y reduce una fuente de inestabilidad local; Bandit baja de 11 a 6 findings LOW** |

**Validación**: `run_ci.py --skip-cov` completado: **680/680 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v95 — Límites y limpieza segura del daemon U-OCR (2 mejoras + tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 217 | **Límite de entrada JSON del daemon**: `_read_json_body` rechaza cuerpos mayores de 64 KB, JSON no objeto y lecturas incompletas antes de acceder a `.get()`. El endpoint interno solo necesita rutas y parámetros pequeños, por lo que no se reduce una capacidad válida del pipeline. | `uocr_daemon.py`, `tests/test_uocr_daemon.py` | **Evita consumo de memoria y errores 500 ante payloads malformados; mejora estabilidad del proceso persistente** |
| 218 | **Limpieza de artefactos completa**: `_cleanup_old_out_dirs` conserva como máximo 20 directorios combinados `req_*` y `art_*`; antes los recortes de re-OCR artístico quedaban acumulados indefinidamente. | `uocr_daemon.py`, `tests/test_uocr_daemon.py` | **Evita crecimiento progresivo del disco durante capítulos con muchas páginas artísticas** |

**Validación**: `tests/test_uocr_daemon.py` **14/14** y `run_ci.py --skip-cov` completado: **678/678 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v94 — Validación final de traducciones antes del render (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 216 | **Barrera final en `_finalize_page_blocks`**: las traducciones devueltas por workers pasan una segunda validación estructural y de idioma origen antes de llegar al frontend. Las salidas basura o claramente no traducidas vuelven al texto OCR original. Se mantienen excepciones estrechas y explícitas para SFX y nombres propios de una sola palabra en mayúsculas (`NARUTO → Naruto`), sin abrir la puerta a frases repetidas o basura. | `routes/api.py`, `tests/test_api.py` | **Evita que una respuesta defectuosa de cualquier motor termine renderizada o se guarde en el resultado de página; mejora estabilidad y calidad final sin cambiar los modelos** |

**Validación**: `run_ci.py --skip-cov` completado: **675/675 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v76 — Cobertura italiana con el recognizer latino existente (1 mejora + test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 193 | **EasyOCR automático incluye `it`**: el código ya acepta italiano y dispone de pares de traducción, pero el lector latino omitía `it`. Se añade a la misma lista de recognizer latino; no se incorpora un modelo nuevo ni se cambia el alcance de idiomas. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Mejor detección italiana en páginas actuales y mixtas** |
| 194 | **Aislamiento correcto del test de EasyOCR**: PyTorch se importa antes de parchear solo el módulo `easyocr`; así la restauración del fixture no desmonta parcialmente el runtime nativo y no contamina los tests posteriores de YOLO/daemon. | `tests/test_ocr_utils.py` | **Suite estable y reproducible** |

**Validación**: regresión de idiomas actuales en verde y `run_ci.py --skip-cov` completado: **643/643 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v75 — Fallback CJK selectivo en Ruta C y merge sin márgenes de página (2 mejoras + tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 191 | **Ruta C conserva globos de borde**: el merge final de recortes validados ya no reaplica filtros de margen de página; se evitan falsos negativos en globos pegados al borde superior/inferior. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Más cobertura sin abrir el filtro global de páginas** |
| 192 | **Fallback CJK de baja confianza**: si RapidOCR aporta kana/hangul/hanzi pero no supera el umbral de texto usable, Ruta C selecciona el lector EasyOCR específico en CPU (solo con `source=auto`), en lugar del lector latino `auto`/GPU. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Recupera texto mixto y protege la VRAM de la GTX 1050 Ti** |

**Validación**: 3 regresiones nuevas en verde y `run_ci.py --skip-cov` completado: **642/642 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v74 — Corrector OCR seguro para idiomas actuales y frases mixtas (1 mejora + tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 190 | **Spellcheck OCR consciente del idioma**: el corrector español se omite para inglés, CJK y otros idiomas actuales; además compara tokens desconocidos con los diccionarios ya disponibles de en/pt/fr/de/it para no alterar frases mixtas como `Quiero go home`. No se añaden idiomas ni modelos OCR/traducción; los diccionarios extranjeros solo se cargan bajo demanda como barrera de falsos positivos. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Conserva texto multilingüe correcto y reduce corrupciones antes de traducir** |

**Validación**: 4 regresiones nuevas en verde y `run_ci.py --skip-cov` completado: **640/640 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/11 LOW**, servidor/API/estáticos correctos (362 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-13-v73 — OCR multilingüe selectivo, honoríficos automáticos y filtros de recortes (3 mejoras + tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 187 | **Páginas mixtas dentro de los modelos actuales**: `source=auto` se conserva hasta `OCRManager`; RapidOCR identifica kana/hangul/hanzi y activa como máximo dos lectores EasyOCR específicos en CPU. No se agregan idiomas ni modelos externos: es/en/pt/fr/de/it/ja/ko/zh siguen siendo el alcance válido y la GTX 1050 Ti no recibe lectores GPU duplicados. | `ocr_utils.py`, `routes/api.py`, `tests/test_ocr_utils.py`, `tests/test_api.py` | **Mejor recuperación de manga con varios idiomas sin ampliar la confusión ni la VRAM** |
| 188 | **Consistencia automática de nombres**: se preservan `-san`, `-chan`, `-kun`, `-sama`, `-senpai`, `-sensei` y sufijos coreanos frecuentes en cajas aisladas, para todos los destinos actuales; no se fuerza el marcador en frases completas. | `translator.py`, `tests/test_translator.py` | **Menos variación entre páginas sin mantener glosario manual** |
| 189 | **Ruta C sin falsos negativos de margen**: RapidOCR recibe `filter_page_margins=False` en crops de globos; los bordes del crop ya no se interpretan como márgenes de página. | `ocr_utils.py`, `tests/test_ocr_utils.py` | **Más cobertura de texto en globos detectados por YOLO/CTD** |

**Validación**: `run_ci.py --skip-cov` → **636/636 tests**, sintaxis Python/JS correcta, Bandit 0 HIGH/0 MEDIUM, servidor/API/estáticos correctos. El reporte de calidad mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semánticos.

#### Sesión 2026-08-12-v72 — Prueba de la TRADUCCIÓN REAL del pipeline caja por caja contra el oro (respuesta a "prueba la traducción real y haz una comparativa de uno x uno") (1 medición + 1 visual)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 186 | **Prueba de la traducción real del pipeline (server 5174) caja por caja contra el oro** (respuesta a "prueba la traducción real y haz una comparativa de uno x uno"). Levanté el server (OCR fusion + CT2 es↔en GPU) y pasé 5 páginas con oro de `corregir/` (las de más cajas GT, 1 por serie: s29458_1161158_p10 45, s54739_1968563_p7 31, s24279_2751240_p16 23, s29458_1161160_p3 22, s54739_1968562_p3 21 → 142 cajas en total) por `/api/process-page` con target=en (las páginas de corregir ya están en español — scanlation — así que es→en es la traducción REAL visible; con target=es sería identidad). `_tmp/prueba_traduccion_real.py`: por cada caja GT, match con el bloque del pipeline (overlap>0.3, la métrica canónica) → fila con texto detectado + traducido + conf. **Resultado: el pipeline real cubre solo 21/142 cajas oro (14.8%)** — el OCR a página completa (fusion easyocr+rapid) no lee la mayoría de los globos (p. ej. s54739_1968563_p7: 2/31). De las 21 cubiertas, muchas son basura o parciales: "DEUN ERUDITO DU SE RETIRA → DEUN ERUDITO DU SE RETIRE", "reveler her MASTER Wiilreture → Reveal her MASTER Wiilreture", "MERCENARY → MERCURY" (el CT2 "corrige" mal un nombre propio), "IANSUI", "GOD0 GTO PARADIS BLACKFIELL" — la confianza es alta (0.9+) pero el texto base está mal leído, así que la traducción es basura pulida. Montaje visual por página (cajas oro numeradas verde/magenta + bloques pipeline rojo) + tabla uno-a-uno (detectado vs traducido, conf, overlap) → `train_data/comparativa_traduccion_real.html` (742 KB, preview registrado). | `_tmp/prueba_traduccion_real.py`, `_tmp/traduccion_real_uno1.json`, `train_data/comparativa_traduccion_real.html` | 👁 **El usuario ve la traducción real globo por globo: el cuello de botella no es el traductor sino el OCR a página completa (14.8% de cobertura) — confirma el valor de la Ruta C por globo + el hallazgo v69 (RapidOCR lee mejor que EasyOCR)** |

#### Sesión 2026-08-12-v71 — Fusión del LOTE 3 al dataset + paquete de Colab v3 (respuesta a "cuando termine de corregir el lote 4, fusiona y prepara Colab v3") (1 merge + 1 paquete)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 185 | **Fusión del lote 3 al dataset + paquete Colab v3** (respuesta a "cuando termine de corregir el lote 4, fusiona las correcciones al dataset y prepara el paquete de Colab v3"). El lote 4 (320 páginas) AÚN no está corregido (0 revisadas), pero el lote 3 (384 páginas, 3.263 cajas) sí estaba listo y **nunca se había fusionado** (verificado: 0 páginas `1388631/1019622/...` en `vlm`). `fusionar_correcciones.py --data train_data/vlm` → **822 train** (438 + 384 nuevas, 6.348 etiquetas: 3.011 globos + 3.337 libres) + 9 val (el lote 4 sin etiquetas se ignora: no toca nada). `vlm_aug` reconstruido (respaldo temporal de las 200 sintéticas `syn_*` → `copytree(vlm)` → re-añadidas) → **1.022 train + 9 val**. Paquete Colab v3 (`_tmp/colab/preparar_paquete_v3.py`): `dataset_vlm_aug.zip` **392 MB** (1.022 train + 9 val, yaml portable `path: .`, sin caches, verificado por zipfile) + `pesos_ogkalu.pt` = **modelo de producción actual (v2)** — el que ganó el A/B, no ogkalu ni el v1 (el notebook espera el nombre "ogkalu" pero el contenido es el v2, para que el v3 parta del mejor modelo). LEEME_COLAB.md actualizado (pasos + nota: al terminar el lote 4 se re-fusiona y re-empaqueta como v4). | `tools/fusionar_correcciones.py` (uso), `train_data/vlm` (822 train), `train_data/vlm_aug` (1.022 train), `_tmp/colab/preparar_paquete_v3.py`, `_tmp/colab/dataset_vlm_aug.zip`, `_tmp/colab/pesos_ogkalu.pt`, `_tmp/colab/LEEME_COLAB.md` | 🚀 **El oro del lote 3 (3.263 cajas) ya está en el dataset y el paquete v3 de Colab está listo para entrenar en la nube desde el mejor modelo** |

#### Sesión 2026-08-12-v70 — Comparativa OCR vs DL sobre 5 páginas del LOTE 4 (sin oro): el DL detecta 1.6× regiones que el OCR lee texto, 57% de las regiones DL sin texto (candidatas a Ruta C) (1 medición + 1 visual)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 184 | **Comparativa OCR real vs DL sobre el lote 4 (respuesta a "genera la comparativa sobre las 320 nuevas para ver estilos no vistos")**. Las 320 del lote 4 NO tienen oro (sin cajas GT), así que la cobertura contra GT no aplica — `_tmp/comparar_ocr_vs_dl_l4.py` mide el **acuerdo entre los dos sistemas** sobre 5 páginas nuevas (1 por serie, sin label): `_detect_and_ocr` (híbrido Easy+Rapid) vs `_detect_text_regions_in_page` (YOLO v2 producción). Métrica: centro de cada bloque OCR dentro/fuera de regiones DL. **Resultado**: 19 bloques OCR vs 30 regiones DL → **89% del texto OCR cae dentro de regiones DL** (el DL no se salta lo que el OCR lee), **57% de las regiones DL no tienen texto leído dentro** (17/30 — el DL encuentra 1.6× regiones de las que el OCR lee, candidatas a la Ruta C), solo 11% del texto OCR queda fuera del DL. Por página: s21016 3 OCR/11 DL, s27854 3/9, s29158 1/0 (página sin DL — el único caso donde el OCR supera al DL), s29458 1/2, s54739 11/8 (la única donde el OCR lee más que el DL). Montaje 2 columnas [OCR REAL naranja+texto | DL rojo] → `train_data/comparativa_ocr_vs_dl_l4.png` (920×3445, 4.0 MB) + `.html` (5.3 MB, preview registrado). | `_tmp/comparar_ocr_vs_dl_l4.py`, `train_data/comparativa_ocr_vs_dl_l4.png`, `train_data/comparativa_ocr_vs_dl_l4.html` | 👁 **En estilos no vistos el DL sigue ubicando más regiones de las que el OCR lee — consistente con el estudio con oro; las regiones sin texto son donde la Ruta C debería recuperar diálogo** |

#### Sesión 2026-08-12-v69 — Calidad del texto de la Ruta C por globo en las mismas 5 páginas: EasyOCR lee basura (49% solo números, 32% decente) mientras RapidOCR lee el diálogo real (1 medición + hallazgo)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 183 | **Medición de CALIDAD del texto de la Ruta C por globo** (respuesta a "mide la calidad del texto (CER/accuracy) que extrae la Ruta C por globo"). Reto: el oro solo tiene cajas, sin texto de referencia (los benchmarks del proyecto tienen GT solo para págs. 3/11/12) → CER real imposible sin transcripción; se mide con proxies + cruce de motores. `_tmp/ruta_c_texto.py`: por cada globo ORO de las 5 páginas del estudio, recorte + upscale 3.5× + cls 180 + EasyOCR con rotation_info (la Ruta C literal, SIN merge cruzado entre globos — el bug de `_group_and_merge_blocks` fusionaba bloques entre globos y el mismo texto aparecía en 7) y RapidOCR sobre el mismo crop como motor independiente. **Resultado**: 129 globos oro → EasyOCR produce "texto" en 113 (88%) pero **55 son solo números/símbolos (49%) y solo 41 globos (32%) tienen texto con ≥2 letras**; agreement Easy-vs-Rapid ~8%; confianza media engañosa (0.58-0.83 — alta pero con texto basura). **Hallazgo principal**: en estas páginas **RapidOCR lee el diálogo real** ("experiencia, pero... Seria bueno", "ESCLAVOSY LO SABEN TRAFICANTE DE EL OPONENTE") donde EasyOCR lee "AIEURc", "WHRSRROF", "1", "8" — la Ruta C (EasyOCR) es el eslabón débil del texto por globo. Descartado rotation_info como causa (probado con/sin: resultados casi idénticos). Mi harness inicial llamaba mal a RapidOCR (`_preprocess_rapid` lo mataba — la degradación real lo llama directo); corregido. Resultados en `_tmp/ruta_c_texto.json`. | `_tmp/ruta_c_texto.py`, `_tmp/ruta_c_texto.json` | 🎯 **La Ruta C podría mejorar mucho el texto por globo usando RapidOCR como motor principal o fusionándolos — candidato claro de optimización** |

#### Sesión 2026-08-12-v68 — Comparativa OCR real vs Deep Learning sobre 5 páginas con oro: el DL (YOLO v2) cubre 65.9% de las cajas vs 38.0% del OCR a página completa (1 medición + 1 visual)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 182 | **Comparación directa OCR real vs Deep Learning** (respuesta a "compara el ocr real con el deep learning"). `_tmp/comparar_ocr_vs_dl.py`: sobre 5 páginas con oro (1 por serie, las de más cajas GT de `train_data/corregir/`), corre **el OCR real** (`_detect_and_ocr`: híbrido EasyOCR GPU + RapidOCR CPU, extrae TEXTO) contra **el deep learning** (`_detect_text_regions_in_page`: YOLO producción = v2, detecta REGIONES) y mide cobertura sobre el oro (overlap_ratio > 0.3, la métrica canónica). **Resultado**: OCR real **49/129 cajas cubiertas (38.0%)** con 47 bloques vs **DL 85/129 (65.9%)** con 75 regiones. Por página: s29458 45 GT (OCR 20, DL 31), s54739 31 GT (OCR 14, DL 24), s24279 23 GT (OCR 11, DL 20), s29158 16 GT (OCR 4, DL 4), s27854 14 GT (OCR 0, DL 6). **Lectura**: el OCR a página completa pierde ~2/3 del texto (globos artísticos); el DL ubica más regiones pero no las LEE — de ahí el diseño del pipeline (YOLO encuentra la región → Ruta C la re-OCRea con upscale 3.5× → recupera el texto que el OCR a página completa pierde). Montaje visual 3 columnas [ORO | OCR REAL (texto naranja) | DL (regiones rojas)] → `train_data/comparativa_ocr_vs_dl.png` (1260×3160, 5.2 MB) + `comparativa_ocr_vs_dl.html` (7.0 MB, preview registrado). | `_tmp/comparar_ocr_vs_dl.py`, `train_data/comparativa_ocr_vs_dl.png`, `train_data/comparativa_ocr_vs_dl.html` | 👁 **El usuario ve de un vistazo que el DL ubica casi el doble de cajas que el OCR solo, y dónde falla cada uno** |

#### Sesión 2026-08-12-v67 — Tercer lote grande para corregir: 320 páginas más de capítulos NUEVOS, saltando capítulos usados en CUALQUIER lote (704 en la app) (1 feature en tool existente)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 180 | **Importador ahora salta capítulos usados en cualquier lote, no solo en la app** (respuesta a "importa otro lote grande sin duplicar los ya importados"). `_caps_ya_usados()` solo miraba `destino` (`corregir/images`) — re-ejecutar podía re-picar capítulos del lote 2 que viven en `terminadas/`. Ahora barre **también `terminadas/images`**: un capítulo representado en cualquier lote (corregir O terminadas) no se re-pica. **Resultado**: 320 páginas importadas (5 series × 64; la serie 24279 no aportó — sus capítulos ya están todos usados), 0 ya existían, 0 rotas → `corregir/images` pasa a **704 páginas**. Verificado: 0 duplicados de nombre, API `/api/estado` → `{total: 704}`. | `tools/importar_paginas.py` (mejora: `_caps_ya_usados` escanea corregir + terminadas), `train_data/corregir/images/` (384→704), `train_data/corregir/LEEME.txt` | 🎯 **Cada ejecución avanza a capítulos nuevos aunque los lotes viejos se hayan movido a terminadas/ — el oro potencial crece sin repetir nada** |
| 181 | **Las 384 del lote 3 (3,263 cajas) se marcaron como revisadas** para que el filtro "solo sin revisar" muestre únicamente las 320 nuevas (igual que en v62): se escribió `revisadas.json` con `true` para todas las imágenes que tienen archivo de label. API → `{total: 704, revisadas: 384, cajas: 3263}`. El servidor (PID 5744) lee la carpeta en cada petición — sin reiniciar. | `train_data/corregir/revisadas.json` | 🎯 **El usuario ve solo las 320 páginas nuevas del lote 4: flujo de barrido limpio sin mezclar con lo hecho** |

#### Sesión 2026-08-12-v66 — Comparativa visual regenerada con montaje ANTES/DESPUÉS: [ORO | ogkalu | v2] sobre las mismas 8 páginas (1 feature en script tmp)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 179 | **Comparativa regenerada con el modelo v2 (producción actual) como montaje de 3 columnas** (respuesta a "regenera la comparativa con el modelo nuevo para ver el antes/después"). `_tmp/comparativa_v2.py`: reutiliza la ronda guardada en `calificaciones.json` (2026-08-11, 63 págs → selección fns>0, 8 peores) — **las MISMAS páginas de la comparativa anterior** — y construye por fila `[ORO (verde/magenta) | ANTES (ogkalu original, .bak) | DESPUÉS (v2, producción)]` vía `_celda_grid`/`_predecir`/`_leer_gt` de `calificar_detector`, imgsz 1280 (el del pipeline), conf 0.25, GPU. Cabeceras con `hits/gt` por modelo (p. ej. `ANTES ogkalu 0/9` vs `DESPUES v2 9/9`). Nota: ultralytics rechaza sufijo `.bak` (`acceptable suffix is {'.pt'}`) → el ogkalu se copia a `models/_ogkalu_tmp.pt` para predecir y se borra al terminar. Artefactos: `comparativa_oro_dl.png` **1440×5726** (4.2 MB, antes 960×5726) y `comparativa_preview.html` (5.6 MB) con la leyenda honesta; preview registrado. Verificado: los 3 colores presentes en el PNG (92,743 px verde / 44,427 magenta / 118,157 rojo). | `_tmp/comparativa_v2.py`, `train_data/comparativa_oro_dl.png`, `train_data/comparativa_preview.html` | 👁 **El usuario ve en las mismas páginas cómo el v2 recupera el diálogo que ogkalu perdía — antes/después en una sola imagen** |

#### Sesión 2026-08-12-v65 — A/B fino v1 vs v2 a conf 0.15/0.25/0.35 sobre las 384 páginas corregidas: v2 gana en LOS TRES umbrales (1 medición)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 178 | **A/B multi-umbral** (respuesta a "A/B más fino a varios umbrales"). Replica `_eval_rapida` (imgsz 512, IoU>0.3) reutilizando los modelos cargados, sobre las 384 páginas nuevas corregidas (3,263 GT). Modelos: **v1** = `finetuned.pt` (Colab v1, la producción previa) vs **v2** = `finetuned-v2.pt` (producción actual). Resultados (recall IoU): conf 0.15 → v1 **85.1%** (3,763 det, 2,107 b/1,656 f) vs v2 **86.7%** (4,432 det, 1,992 b/2,440 f); conf 0.25 → v1 77.1% (2,864 det, 1,971 b/893 f) vs v2 **79.6%** (3,189 det, 1,886 b/1,303 f); conf 0.35 → v1 69.2% (2,366 det) vs v2 **70.9%** (2,493 det). **Lectura**: v2 gana en los 3 umbrales (no es un pico de un solo conf — no hay des-calibración), el gap crece a conf baja (la receta ampliada baja el umbral efectivo del detector), y mantiene su 2.4–2.8× más texto libre a conf de producción. El pipeline corre a conf 0.25 → operativamente v2 = 79.6% vs v1 77.1%. Resultados en `_tmp/ab_umbrales_resultado.json`. | `_tmp/ab_umbrales.py`, `_tmp/ab_umbrales_resultado.json` | 🎯 **Confirmado: el v2 activo es mejor en todo el rango de confianza útil, no solo a 0.25** |

#### Sesión 2026-08-12-v64 — A/B decisivo del modelo Colab v2 contra las 384 páginas NUEVAS corregidas (lote 3): gana 79.6% vs 70.5% recall y se ACTIVA como producción (1 medición + 1 deploy)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 176 | **A/B del v2 contra el lote 3 (el veredicto que importa)** (respuesta a "hazlo" tras el empate de 76% vs 76% sobre el val de 8 páginas). El val pequeño (9 páginas, 50 GT) no separaba a los modelos; se repitió `_eval_rapida` (imgsz 512, conf 0.25 — el umbral real del pipeline) sobre **las 384 páginas nuevas ya corregidas** (`train_data/corregir/`, 3,263 cajas GT) que el v2 NUNCA vio en entrenamiento. Resultado: **producción (Colab v1) 70.5% recall (2,687 det, conf 0.604, 2,202 bubble / 485 free) vs v2 79.6% recall (3,189 det, conf 0.616, 1,886 bubble / 1,303 free)**. +9.1 pts de recall y 2.7× más texto libre — el v2 sí es mejor; el empate anterior era un artefacto del val diminuto. | — (medición) | 🎯 **Veredicto con 3,263 GT reales: el oro ampliado sí mejoró el detector, sobre todo en texto libre** |
| 177 | **SWAP ACTIVADO: producción = modelo Colab v2** (vía `_swap_model`: backup `.bak` ya existía = ogkalu original, intacto; el v1 queda conservado como `models/comic-speech-bubble-detector-finetuned.pt`). `comic-speech-bubble-detector.pt` (lo que usa el pipeline real) ahora es el v2 entrenado con el oro ampliado (638 train). Reversible: restaurar el `.bak` o el `finetuned.pt`. | `models/comic-speech-bubble-detector.pt` | 🎯 **El traductor de manga en producción detecta +9.1 pts de recall y 2.7× más texto libre que hace una sesión** |

#### Sesión 2026-08-12-v63 — Las 384 del lote 2 se MUEVEN a terminadas/: el corrector queda solo con las 384 del lote 3 (1 reorganización)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 175 | **Las 384 páginas del lote 2 (las etiquetadas, 2,764 cajas) se movieron a `terminadas/`** (respuesta a "como la anterior saca lo que ya está hecho, guárdalo y deja solo lo nuevo"). Se MOVIERON (no copiaron): 384 imágenes + 384 labels de `corregir/images` y `corregir/labels` → `terminadas/images` y `terminadas/labels` (ahora consolida **447 páginas**: 63 del lote 1 + 384 del lote 2). `revisadas.json` se limpió de las movidas (queda vacío). `corregir/` queda con **solo las 384 del lote 3** (0 labels, 0 revisadas, sin banner de gigantes). **Seguridad**: el oro ya está fusionado en `train_data/vlm` desde la v64/fusión anterior — el fusionador itera `corregir/images/` y las ausentes se saltan conservando su estado; `calificar_detector.py --workspace train_data/corregir/terminadas` permite evaluar los lotes guardados; `labels/_original/` sigue como respaldo histórico. **Verificado**: servidor del 8789 sigue vivo (PID 5744, sin reiniciar — lee la carpeta en cada petición); API `/api/estado` → `{total: 384, revisadas: 0, cajas: 0, sin gigantes}`; en el navegador: `Página 1/384 (384 total) · 0 revisadas · 0 cajas oro · sin cajas gigantes ✓`, desplegable solo con las `s*` del lote 3. LEEME.txt actualizado. | `train_data/corregir/images/` (768→384), `train_data/corregir/labels/` (384→0), `train_data/corregir/terminadas/` (63→447), `train_data/corregir/revisadas.json` (384→0), `train_data/corregir/LEEME.txt` | 🎯 **La app queda solo con las 384 nuevas del lote 3: el usuario corrige desde cero sin mezclar con lo hecho** |

#### Sesión 2026-08-12-v62 — Segundo lote grande para corregir: 384 páginas más, ahora de capítulos NUEVOS (768 en la app) (1 feature en tool existente)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 174 | **Importador de páginas ahora salta capítulos ya usados** (respuesta a "importa otro lote grande sin duplicar las 384"). `importar_paginas.py` tenía selección determinista sobre los primeros capítulos de cada serie — re-ejecutarlo habría re-picado las mismas páginas (ya existían). Se añadió `_caps_ya_usados()` (lee `destino/` con el patrón `s{serie}_{cap}_p{pagina}.jpg`, parte por el último `_p`) y `_paginas_de_serie()` ahora recibe `destino` y **salta capítulos que ya tienen páginas en el corrector** — cada ejecución avanza a capítulos nuevos en vez de repetir los primeros. **Resultado**: 384 páginas importadas nuevas (64 por serie, 0 ya existían, 0 rotas) → `train_data/corregir/images` pasa a **768 páginas** (las 384 previas conservan sus 2,764 cajas oro; las 384 nuevas sin etiquetar). Verificado: servidor del 8789 sigue vivo sin reiniciar (lee la carpeta en cada petición) — API `/api/estado` → `{total: 768, revisadas: 0, cajas: 2764}`, 0 duplicados de nombre. LEEME.txt actualizado (sección "LOTE 3"). Nota: el entrenamiento local (train3, dataset ampliado 638) ya terminó — logs sin proceso activo; el A/B queda pendiente de revisar. | `tools/importar_paginas.py` (mejora: salta capítulos usados), `train_data/corregir/images/` (384→768), `train_data/corregir/LEEME.txt` | 🎯 **El corrector suma otras 384 páginas de capítulos distintos: el oro potencial se duplica a 768 sin repetir nada** |

#### Sesión 2026-08-12-v61 — Las 63 corregidas se MUEVEN fuera de la app a terminadas/: el corrector queda solo con las 384 páginas nuevas (1 reorganización)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 173 | **Las 63 páginas corregidas se movieron de `corregir/` a `terminadas/`** (respuesta a "quiero que quites las que ya fueron corregidas y solo queden las nuevas"). Se MOVIERON (no copiaron): las 63 imágenes + sus 63 labels salieron de `corregir/images` y `corregir/labels` hacia `terminadas/` (que ya tenía copias de la sesión v59 — quedan consolidadas ahí, sin duplicados). `corregir/` queda con **solo las 384 nuevas** (0 labels, 0 revisadas). **Seguridad verificada**: (a) el oro corregido ya está fusionado en `train_data/vlm` desde la v57 — la próxima fusión no lo toca (el fusionador itera `corregir/images/`, las páginas ausentes se saltan y vlm conserva su estado corregido); (b) `calificar_detector.py` filtra por "imagen con etiqueta" y degrada con aviso ("Sin páginas con marcado") en vez de romperse; para evaluar el lote viejo existe `--workspace train_data/corregir/terminadas` (documentado en LEEME.txt); (c) `labels/_original/` se conserva como respaldo histórico. **Operativo**: el servidor del 8789 murió al liberar el preview anterior (reemplazo fallido) — relanzado limpio (PID 15420) y preview re-registrado; verificado en el navegador: `Página 1/384 (384 total) · 0 revisadas · 0 cajas oro · sin cajas gigantes ✓`, desplegable solo con las `s*`, sin banner. LEEME.txt actualizado. | `train_data/corregir/images/` (63→384), `train_data/corregir/labels/` (63→0), `train_data/corregir/terminadas/` (63 consolidados), `train_data/corregir/LEEME.txt` | 🎯 **La app queda solo con las 384 páginas nuevas: el usuario corrige desde cero sin mezclar con lo hecho** |

#### Sesión 2026-08-12-v60 — Filtro "solo sin revisar" en el corrector: al entrar ya no se ven las páginas corregidas — la lista arranca en la primera nueva y al guardar una desaparece (1 feature UX)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 172 | **Filtro "solo sin revisar" en el corrector** (respuesta a "sigo viendo las que ya corregí cuando entro al link"). Con 447 páginas (63 corregidas + 384 nuevas), la app abría en la página 0 (corregida) y el desplegable mezclaba todo. Cambios en `tools/corrector_oro.html`: (a) array maestro `todas` + vista filtrada `paginas` (mismas referencias de objeto — guardar muta la revisada en ambas); (b) checkbox **"solo sin revisar" activado por defecto** — al cargar solo se listan las no revisadas y `irPagina(0)` cae en la primera nueva; si no queda ninguna (todo revisado) el filtro se auto-desactiva y muestra todo; (c) al guardar con revisada ✓ y filtro activo, la página **se quita de la lista y avanza a la siguiente** (con o sin "avanzar al guardar") — flujo de barrido limpio; (d) cabecera ahora muestra `Página X/Y (Z total) · W revisadas` (contadores globales sobre `todas`, posición sobre la vista); (e) el salto del banner gigante se recalcula contra la vista (si la página gigante está oculta por el filtro, va a la primera visible). Se eliminó `pintarEstadoGlobal` (fundida en `pintarEstado`). Servidor reiniciado limpio (PID 15760) y preview re-registrado: verificado en el navegador — `Página 1/384 (447 total) · 63 revisadas`, desplegable solo con páginas nuevas empezando en `s21016_1388605_p2`, checkbox activo, consola sin errores, 8 tests del corrector en verde. | `tools/corrector_oro.html` | 🎯 **El usuario entra directo a las páginas nuevas: ve 384 por corregir, cada guardado las va quitando de la lista** |

#### Sesión 2026-08-12-v59 — Lote nuevo de páginas para corregir: las 63 corregidas quedan respaldadas en terminadas/ y se importan 384 páginas reales nuevas (6 series) al corrector (1 feature + 1 tool nueva)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 171 | **Lote nuevo de páginas para el corrector de oro** (respuesta a "pasame muchas mas paginas para corregir y las que ya corregi guardalas en una carpeta"). (a) **Backup de las 63 corregidas**: copiadas a `train_data/corregir/terminadas/` (images/ + labels/ + revisadas.json) — se COPIA, no se mueve, porque `fusionar_correcciones.py` itera `corregir/images/` y si faltara la imagen perdería la corrección en la próxima fusión. (b) **Nueva tool `tools/importar_paginas.py`**: importa páginas reales desde `input_manga/BookDownloads/BookDownloads` al workspace del corrector — convierte webp→jpg (el corrector y el pipeline solo aceptan jpg/png), salta portadas/contraportadas (2 primeras/últimas de cada capítulo), excluye las series ya corregidas (1103524/1457338/1490498), nombres únicos `s{serie}_{capitulo}_p{pagina}.jpg` idempotentes (re-ejecutar no duplica) y reparto uniforme entre series. (c) **Hallazgo de datos**: de las 169 carpetas de serie, solo 6 tienen imágenes reales (13.294 páginas): 21016 (3673), 29458 (6429), 27854 (1547), 29158 (828), 54739 (540), 24279 (277) — el resto son descargas fallidas vacías; las copias "(N)" son duplicados. **Importadas 384 páginas** (--max 360 --por-serie 60 → 336 nuevas + 48 del primer intento), el corrector pasa de 63 → **447 páginas** (63 revisadas + 384 nuevas), todas reales (dimensiones verificadas 800×1133, 1114×1600…). (d) **UX para el lote grande**: nuevo botón "○ sin revisar" + tecla `N` en `corrector_oro.html` (salta a la primera página sin revisar). (e) **Operativo**: había 2 servidores en el 8789 (el viejo del 19996 sobrevivió al reinicio de Freebuff sirviendo HTML stale) — se mataron ambos y se levantó uno limpio con `--no-browser` (PID 2548), preview re-registrado y verificado (447 páginas, 63 revisadas, botón nuevo presente, salto a la primera sin revisar funciona). LEEME.txt actualizado. | `tools/importar_paginas.py` (nuevo), `tools/corrector_oro.html` (botón N), `train_data/corregir/terminadas/` (nuevo, 63 respaldadas), `train_data/corregir/images/` (63→447), `train_data/corregir/LEEME.txt` | 🎯 **El oro pasa de 63 a 447 páginas corregibles (6× más datos reales) con las ya hechas a salvo — el material para superar el 76% de recall** |

#### Sesión 2026-08-12-v58 — Entrenamiento en Colab (nube gratis, T4) + A/B + SWAP ACTIVADO: el fine-tuned de Colab gana a ogkalu a conf 0.25 (76% vs 66% recall) y queda como modelo de producción (1 medición + 1 deploy)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 170 | **Entrenamiento en Colab + A/B + swap del YOLO de producción** (respuesta a "ya tengo el best.pt"). El usuario entrenó en Google Colab (T4) con el paquete de la sesión v57 (`_tmp/colab/`: notebook `entrenar_colab.ipynb` con la receta exacta local — ultralytics 8.4.115, epochs=40, imgsz=512, batch=4, freeze=10, lr0=1e-4, lrf=0.05, patience=15 —, `dataset_vlm_aug.zip` y `pesos_ogkalu.pt`; `data.yaml` corregido a `path: .` portátil). Descargó `best.pt` (51.98 MB, validado: checkpoint ultralytics OK, train_args confirman `/content/vlm_aug/data.yaml`, clases `{0: text_bubble, 1: text_free}`). **Verificación**: no es descarga cortada (tamaño correcto), carga con ultralytics y tiene las 2 clases. Copiado a `models/comic-speech-bubble-detector-finetuned.pt`. **A/B** con `_eval_rapida` (mismos params que main: imgsz 512, conf 0.25, val real de `train_data/vlm` — el oro corregido): **ogkalu 66.0% (30 det, conf 0.806, bubble 28/free 2) vs Colab fine-tuned 76.0% (60 det, conf 0.505, bubble 55/free 5) sobre 8 págs val (50 GT) — GANA +10 pts**. **`_swap_model` ejecutado**: `models/comic-speech-bubble-detector.pt` (YOLO_MODEL_PATH) ahora es el fine-tuned de Colab; ogkalu original intacto en `comic-speech-bubble-detector.pt.bak` (el .bak previo ya existía y no se tocó). Verificado que el modelo activo carga con las 2 clases. Sin cambios de código — solo medición + deploy del modelo. | `models/comic-speech-bubble-detector.pt` (swap), `models/comic-speech-bubble-detector-finetuned.pt`, `C:/Users/roweh/Downloads/best.pt` (origen) | 🎯 **El traductor de manga ya usa el modelo entrenado en la nube con tu oro corregido (76% recall a conf 0.25, +10 pts sobre ogkalu)** |

#### Sesión 2026-08-12-v57 — Oro corregido fusionado + re-entrenamiento + A/B: el fine-tuned supera a ogkalu a conf 0.25 (72% vs 66% recall) y detecta 5× más texto libre (1 medición)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|

#### Sesión 2026-08-12-v56 — Corrector interactivo del ORO en el navegador: dibuja/corrige las cajas de globos y texto libre y cada Guardar escribe el oro real (1 feature + 8 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|

#### Sesión 2026-08-12-v55 — La comparativa ya se VE: rellenos translúcidos + trazo grueso + tiles grandes + leyenda honesta (fix de renderizado, 1 feature + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 167 | **Fix de renderizado de la comparativa y el grid** (respuesta al feedback del usuario: "el globo del oro y el texto libre no sirve, no se ve ni la calidad ni las correcciones"). Diagnóstico real: (1) el PNG SÍ dibujaba los colores pero en tiles pequeños (comparativa 600px, grid 240px) con trazo de 2px → cajas casi invisibles al escalar; (2) el **oro son 46/210 cajas GIGANTES** (>25% de la página, varias w=1.00 del teacher) que tapaban el arte y no dejaban ver ni el globo real ni las detecciones rojas dentro; (3) **el preview HTML prometía un color amarillo 'Coincidencia oro∩modelo' que nunca se dibujó** — leyenda con un color fantasma. Cambios en `calificar_detector.py`: `_dibujar_cajas` acepta `relleno` (0-1) y pinta un **relleno TRANSLÚCIDO** (el arte se ve a través de la caja) + `ancho_pag`/`alto_pag` para **RECORTAR las cajas a la página** (las regiones w=1.00 ya no pintan sobre la banda letterbox gris); `_letterbox` devuelve las dimensiones de página escaladas; `_celda_grid`/`_par_comparativa` usan relleno 0.18 (oro) / 0.12 (modelo) y **grosor escalado** `max(2, tile_w/120)`; tiles default más grandes (grid 240→320, comparativa 300→480); cabeceras con **desglose por clase `G globos h/g | L libre h/g`** (nuevo `_titulo_cabecera`); leyenda del PNG aclara que el relleno marca la región cubierta. **`--preview-html`** (nuevo): empaqueta la comparativa en HTML autocontenido con la **leyenda honesta de 3 colores** (nuevo `_escribir_html_preview` — sin el amarillo fantasma). **Artefactos regenerados** desde la ronda guardada (sin re-evaluar 63 páginas, GPU libre): `comparativa_oro_dl.png` 960×5726 (antes 600×3666), `calificaciones_grid_mejorada.png`, `comparativa_preview.html` (leyenda corregida) — el tab Preview se actualizó. **2 tests nuevos**: el relleno translúcido cubre más área que el borde (la caja ya no tapa el arte) y el HTML no contiene `#f1c40f`/`Coincidencia` (sin color fantasma). **Validación**: **558 passed** (556 + 2), `py_compile` OK. | `tools/calificar_detector.py`, `tests/test_correccion_detector.py`, `train_data/comparativa_oro_dl.png`, `train_data/calificaciones_grid_mejorada.png`, `train_data/comparativa_preview.html`, `_tmp/regenerar_comparativa.py` | 🎨 **La comparativa muestra el oro sin tapar el arte, el trazo se ve, y la leyenda dice solo lo que se dibuja** |

| 168 | **Corrector interactivo del ORO (`tools/corrector_oro.py` + `tools/corrector_oro.html`)** (respuesta directa a "quiero que yo lo corrija y les diga dónde está, y tú pones que es un oro — lo que han hecho hasta ahora está mal"). El oro son los `labels/*.txt` YOLO de `train_data/corregir` (los leen `calificar_detector.py` y `fusionar_correcciones.py`); el teacher los generó con **56 de 63 páginas con cajas gigantes** (w=1.00, 210 cajas totales). La herramienta: mini-servidor stdlib (`ThreadingHTTPServer`) que sirve una **app web en el navegador** con (a) dibujar cajas nuevas arrastrando (clase activa G=globo/L=libre), (b) mover arrastrando desde dentro, (c) redimensionar por esquinas, (d) Supr/clic derecho borra, doble clic cambia clase, (e) **las cajas gigantes se marcan en rojo discontinuo** + banner "⚠ N páginas tienen cajas gigantes" con botón para saltar a la primera, (f) números identificadores por caja para referirse a ellas en el chat, (g) checkbox "✓ Revisada" + `revisadas.json` para llevar progreso entre sesiones. **Cada Guardar (Ctrl+S) escribe el .txt de la página como ORO**: normaliza a [0,1], descarta degeneradas, y una página SIN cajas escribe un `.txt` VACÍO (semántica clave: `fusionar_correcciones.py` la trata como "sin texto" y NO conserva las pseudo-etiquetas malas). **Backup del oro original** en `labels/_original/` (una sola vez) como red de seguridad. Endpoints: `GET /api/paginas`, `GET /api/estado`, `GET /api/img/<n>`, `POST /api/guardar` (valida el nombre contra las imágenes del workspace — sin path traversal). **Verificado**: 8 tests nuevos (parseo/escritura YOLO, gigante, normalización, backup único, página vacía, listar) — `pytest tests/test_corrector_oro.py tests/test_correccion_detector.py` → **49 passed** (41 previos + 8 nuevos), `py_compile` OK, smoke del flujo HTTP completo contra un workspace temporal (listar→guardar→disco→revisadas→backup) y el preview del escritorio registrado en `http://127.0.0.1:8789` con la página 0 renderizando (4 cajas, gigante #2 marcado, rellenos translúcidos verificados por muestreo de píxeles). `LEEME.txt` actualizado para recomendar el corrector propio sobre X-AnyLabeling. | `tools/corrector_oro.py` (nuevo), `tools/corrector_oro.html` (nuevo), `tests/test_corrector_oro.py` (nuevo), `train_data/corregir/LEEME.txt` | ✏️ **Tú corriges a mano y cada Guardar ES el nuevo oro — con backup del original y cajas gigantes señaladas** |

#### Sesión 2026-08-11-v52 — Calificación comparativa synth_solo vs ogkalu sobre las 63 páginas corregidas (1 medición)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 164 | **Primera calificación del modelo synth_solo contra el MISMO marcado de 63 páginas** (`--model models/comic-speech-bubble-detector-finetuned-synth.pt`, imgsz 640 CPU). Resultado vs ogkalu en el mismo historial: **synth_solo nota 60/100 (recall 60.5%, 127/210) vs ogkalu 57/100 (56.7%, 119/210)** — el synth_solo encuentra **8 cajas más (+3.8 pts recall)** y **recupera 3 páginas completas de diálogo** que ogkalu perdía (p002, p012, 1457338_p001), a cambio de **perder 1 página** (1103524_p005, 1 caja, que ogkalu sí encontraba). **Pero a costa de ruido**: 401 detecciones extra vs 269 (precisión 24.1% vs 42.2%, F1 34.4% vs 48.4%) — el patrón conocido del synth_solo (recall alto a conf baja con muchas cajas falsas). Lectura honesta para el usuario: si la meta es "encontrar letras que faltan" (recall) el synth_solo gana; si la meta es no ensuciar el OCR con cajas falsas (precisión/F1), ogkalu sigue ganando. El grid comparativo quedó en `train_data/calificaciones_grid_20260811_2346.png` (960×1502, 16 miniaturas). Nota de entorno: el auto-open del visor falla en este shell (ShellExecute WinError 2 con exists=True — sin dispatch de shell; el archivo es válido, se abre a doble clic). | `train_data/calificaciones.json` (ronda 4), `train_data/calificaciones_grid_20260811_2346.png` | ⚖️ **synth_solo gana recall (+3.8 pts, 3 páginas recuperadas) pero ogkalu gana precisión/F1 — trade-off claro entre encontrar más y ensuciar menos** |

#### Sesión 2026-08-11-v51 — `--paginas` en calificar_detector.py: calificar páginas concretas + tabla de recall por página en el reporte (1 feature + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 163 | **`calificar_detector.py` acepta `--paginas 'p002,1490498_p001'`** para calificar SOLO las páginas pedidas del marcado (stems exactos separados por coma): filtra las imágenes antes de evaluar, avisa con `⚠ Páginas pedidas sin marcado: …` si alguna no existe (y si ninguna existe, sale sin guardar ronda), y la nota/historial/grid se refieren solo al subconjunto. **Con el flag activo, el reporte imprime la tabla `RECALL POR PÁGINA (peor primero)`** — página, hits/gt, perdidas, recall y precisión de cada página evaluada en la misma pasada (en vez del top-5 de peores). **2 tests** (`TestCalificarDetector`): `--paginas img1` → solo 1 página evaluada, nota 100, tabla con `2/2` en la salida y `img2` ausente; `--paginas p999` → avisa `Páginas pedidas sin marcado` y NO guarda ronda. **Smoke en vivo**: `--paginas 'p002,1490498_p001'` → 2 páginas, 11 GT, tabla con `1490498_p001 0/9` y `p002 0/2` (precisión 100% — detecta 2 cajas pero ninguna coincide con el oro gigante). **Validación**: **556 passed** (554 + 2). | `tools/calificar_detector.py`, `tests/test_correccion_detector.py` | 🎯 **Calificación puntual de páginas con su tabla por página** |

#### Sesión 2026-08-11-v50 — Montaje visual del calificador: grid de las páginas con texto perdido con el oro (verde) y las detecciones del modelo (rojo) superpuestas (1 feature + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 162 | **`calificar_detector.py` ahora genera el montaje visual** al terminar la calificación: `_generar_grid` selecciona las páginas con texto perdido (fns>0, peores primero, cap `--grid-max-paginas` default 16), re-corre el modelo sobre ellas y compone un grid de miniaturas con **el marcado del usuario en verde y las detecciones del modelo en rojo** (franja de cabecera por celda con página + hits/gt + recall, letterbox al ratio manga, leyenda de colores arriba). Default: `train_data/calificaciones_grid_YYYYMMDD_HHMM.png`; `--grid <ruta>`, `--grid-cols`, `--grid-width`, `--no-grid` para desactivar. La ruta del PNG se guarda en la ronda del historial (`ronda["grid"]`) y se imprime con el reporte. Refactor: `_predecir` (detección compartida entre evaluación y grid) y `_evaluar_workspace` ahora devuelve la lista completa de páginas. **3 tests** (`TestCalificarDetector`): el grid se crea e incluye la página perdida (img2 0/2) con PNG no vacío; sin pérdidas devuelve None sin crear archivo; y `--no-grid` en main() no crea PNG pero sí guarda la ronda. **Validación en vivo**: 63 páginas → nota 57/100 (119/210) y montaje `train_data/calificaciones_grid_20260811_2341.png` (960×1502, 16 miniaturas 240px, 4 cols) — la p004/p001 de `1490498` y `p002/p009` se ven sin rojo dentro del verde (texto que el modelo no encontró). **Validación**: **554 passed** (551 + 3). | `tools/calificar_detector.py`, `tests/test_correccion_detector.py`, `train_data/calificaciones_grid_*.png` | 🖼 **El fallo del detector ahora se VE de un vistazo** |

#### Sesión 2026-08-11-v49 — tools/calificar_detector.py: la NOTA del bucle corregir→calificar→reentrenar + fix de métrica (IoU→overlap_ratio) (1 feature + 3 tests + 1 fix)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 161 | **Nueva tool que responde al pedido del usuario ("te mando un marcado y le das una calificación… hasta que tenga alta calificación")**: `calificar_detector.py` puntúa cuánto del texto REAL (marcado manual en X-AnyLabeling) encuentra el modelo actual. Corre el modelo sobre el workspace corregido a conf del pipeline (0.25), empareja por **`_overlap_ratio` (inter/min_area, la métrica canónica que usan `ocr_utils` y `_eval_rapida`)** con umbral 0.3, y reporta: nota 0-100 (recall), hits/fns (texto que NO encontró)/fps (extra), **páginas con diálogo perdido** (GT>0 y 0 hits — las que hay que corregir primero), peores páginas ordenadas, y **% de cajas gigantes** (>25% de la página: el score mide cobertura de regiones, no de globos individuales). Cada ronda se guarda en `train_data/calificaciones.json` (historial con progreso 6→57). Flags: `--workspace --model --conf --imgsz --device (auto|cpu) --overlap --historia --etiqueta`; `--device cpu` evita pelear por VRAM con el daemon. **3 tests** (`TestCalificarDetector`): conteo hits/fns/fps, nota+páginas perdidas+historial acumulado, y main() con workspace vacío. **Fix de métrica durante la validación en vivo**: la 1ª versión usaba IoU plano (inter/unión) → nota 6/100, porque el oro del teacher son regiones GIGANTES (100% de las 210 cajas >25% de la página, muchas desbordan los bordes: p005 0/2 dentro, 1490498_p005 0/7 con w=1.00) y una caja enorme nunca empata por IoU con un globo apretado aunque lo contenga. Cambiado a `_overlap_ratio` (la métrica del repo) → **nota 57/100 (56.7% recall, 119/210; 8 páginas con diálogo 100% perdido)** — consistente con el 63.6% de la sesión 158 (que ya usaba overlap, no IoU). Diagnóstico de contención: 60% de las regiones del oro tienen ≥1 detección dentro. **Validación**: **551 passed** (548 + 3). | `tools/calificar_detector.py` (nuevo), `tests/test_correccion_detector.py`, `train_data/calificaciones.json` (nuevo), `_tmp/diag_contencion.py` | 🎯 **El bucle de mejora del usuario ya tiene su nota** — y la 1ª nota expuso que el oro del teacher son regiones gigantes, no globos apretados |

#### Sesión 2026-08-11-v46 — A/B ampliado: el synth_solo SÍ supera a ogkalu a conf 0.25 en 19 páginas reales (el empate del val de 9 era artefacto de tamaño) (1 medición)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 158 | **Medición ampliada del synth_solo sobre 19 páginas REALES del cap 43 (p30-p50, 66 GT del teacher) — set independiente que el modelo NUNCA vio** (train del synth_solo = solo 200 sintéticas + las 9 val del VLM; verificado 0 contaminación de p30-p50 en su train). A/B con `_eval_rapida` a los DOS umbrales: **conf>=0.10 → ogkalu 66.7% vs synth_solo 74.2% (317 det, conf 0.259); conf>=0.25 (pipeline real) → ogkalu 63.6% vs synth_solo 69.7% (+6.1 pts) con 139 det vs 80 y cobertura de texto libre 28 vs 1**. La conf media del synth_solo a 0.25 es 0.387 (sana, no des-calibrada). **Conclusión**: el empate 55.6% del val de 9 páginas (sesión 153) era un artefacto del tamaño — con 2x más páginas reales el synth_solo GANA a ogkalu a conf 0.25, manteniendo la calibración. El oro_synth confirma su colapso fuera del val pequeño (6.1% — no era artefacto, era el sesgo real de una sola serie). **Por primera vez un modelo fine-tuned supera a ogkalu en el umbral REAL del pipeline con datos que no vio** — el candidato al swap existe (el gate `--conf` de la sesión 152 lo permitiría: recall 69.7% > 63.6%). | `models/comic-speech-bubble-detector-finetuned-synth.pt` (candidato al swap), `_tmp/eval_ab_p3050.py` | 🎯 **El synth_solo supera a ogkalu en el umbral real — el empate era del val pequeño** |

#### Sesión 2026-08-11-v47 — experimento de VARIEDAD de series (balance 50/50 oro+series nuevas): la variedad mejora la detección bruta pero no la calibración a 0.25 (1 experimento)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 159 | **Experimento de variedad de series con control de volumen (respuesta a "¿cuántas páginas corregidas de series DIFERENTES hacen falta?")**. **Inventario previo del oro real**: el usuario corrigió 27 págs del cap 43/47 (train) + 1 de `1103524`; las series `1457338`/`1490498` tienen **0 oro** — no existe oro multi-serie todavía, así que el experimento literal es imposible. **Proxy medible**: dataset `train_data/balance_series` = 200 synth + **10 oro cap + 10 series nuevas (pseudo) — train real BALANCEADO 50/50 entre 2 bloques de series** (vs oro_synth de la sesión 156 que era 44/10 = 81% una serie), mismo volumen real total (20 págs vs 54, menos porque el oro es limitado), val = las 9 reales sin contaminar. Retrain receta estándar (freeze=20 lr=3e-5, 40 épocas). **A/B por IoU sobre 19 páginas reales p30-p50 (66 GT, set que NINGÚN modelo vio en train)**: conf>=0.10 → **balance 84.9% vs synth 74.2% vs ogkalu 66.7%** (268 det) — la variedad de series SÍ mejora la detección bruta (+10.7 pts sobre synth_solo); conf>=0.25 (pipeline real) → **balance 34.8% vs synth 69.7% vs ogkalu 63.6%** — el balance sigue des-calibrado (conf media 0.310, pierde ~60% de sus detecciones al subir el umbral). **Conclusión**: la variedad de series importa para la detección (balance gana a 0.10 con la MITAD del volumen real que oro_synth) PERO no resuelve la calibración a 0.25 — cualquier dato real en el train (balanceado o no) des-calibra la confianza; las sintéticas solas siguen siendo el único train que mantiene la confianza sana. Implicación práctica: el camino no es solo "más oro variado" — es oro variado + un enfoque de calibración (entrenar solo la cabeza de cls, o umbral más bajo con el modelo balance/synth). | `tools/entrenar_detector.py`, `models/finetune_balance/weights/best.pt` (nuevo), `train_data/balance_series/` (nuevo), `_tmp/armar_balance_series.py`, `_tmp/eval_ab_4modelos.py` | 🧪 **Variedad > volumen para detectar (84.9% a 0.10), pero la calibración a 0.25 sigue rota con dato real** |

#### Sesión 2026-08-11-v48 — Test de regresión del swap: `_swap_model` es mock PURO y el disco de producción queda intacto (1 test)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 160 | **Regresión dedicada de la corrupción de la sesión 152**: `test_main_swap_no_toca_disco_de_produccion` verifica que (1) `main()` con `--swap` ganador le pasa a `_swap_model` el modelo REAL de producción (`models/comic-speech-bubble-detector.pt`, el default de `--weights`) como origen; (2) `_swap_model` es un **mock puro** (`mocker.patch`, nunca se ejecuta el `copy2` real); y (3) **el estado del dir de producción queda byte-idéntico tras main()** — snapshot de existencia+bytes del `.pt` y del `.bak` antes y después. La aserción del `.bak` se ajustó al detectar que `models/…pt.bak` es un artefacto LEGÍTIMO preexistente (4/ago, de un swap real anterior): el test compara antes/después en vez de exigir que no exista. El bug de la sesión 152 (un `mocker.spy` EJECUTABA la función real y copiaba el `best.pt` falso del test sobre producción) queda cubierto de forma explícita y permanente. | `tests/test_correccion_detector.py` | 🛡️ **El disco de producción no se puede corromper desde los tests** |

#### Sesión 2026-08-11-v40 — manga_ocr: escaneo recursivo por carpetas + --solo — soporte de mangas en .webp anidados (1 feature + 6 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 148 | **El usuario puso muchos mangas en `input_manga/` en `.webp` anidados** (`input_manga/BookDownloads/…/<serie>/<capítulo>/N.webp` — 865 capítulos, 12,754 páginas, 2 GB). El CLI NO los veía: `_escaneo_archivos` usaba `iterdir()` de UN solo nivel y trataba cada imagen como documento suelto. **Fix**: (a) `_escaneo_documentos` recursivo con `_Documento` (dataclass: pdf | carpeta | imagen): PDF/imagen suelta en el nivel superior → 1 documento cada uno (compatibilidad total); **carpeta que contiene imágenes directamente → UN documento con N páginas** (patrón walk: los contenedores de serie sin imágenes propias se recorren, cada capítulo se agrupa en su propia carpeta, y si una carpeta tiene imágenes Y subcarpetas aporta ambas). El **orden natural** (`_orden_natural`: 0.webp < 1.webp < 2.webp < 10.webp, no léxico) respeta el orden real de páginas del capítulo. **Colisiones entre series**: dos capítulos con el mismo stem → el nombre de salida se prefija con la serie (`serie_capitulo`) en vez de sobrescribirse (verificado: 0 colisiones pendientes en el dataset real). (b) **`--solo '<subcadena>'`** filtra documentos por nombre (procesar un capítulo concreto por su ID sin recorrer 2 GB); `--pages` sigue aplicando al documento. (c) `meta` del JSON añade `tipo` (pdf/carpeta/imagen). **Validado**: suite **537 passed** (531 + 6 nuevos/actualizados: filtro+orden de documentos, agrupación de carpeta anidada en orden natural, carpetas sin imágenes ignoradas, orden natural, --solo en main) + **smoke en vivo**: `--solo 1388605 --pages 1-2` sobre el dataset real → 1 documento, 2 páginas en orden, JSON agrupado con `tipo: carpeta` (3 bloques/pág en 27.0s/7.7s, con el tier CTD activo). | `manga_ocr.py`, `tests/test_manga_ocr.py` | 📚 **Los 865 mangas en .webp ya son procesables** |

#### Sesión 2026-08-07-v25 — Checkpoint default con sufijo temporal: cada corrida usa su propio archivo (1 fix + 3 tests)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 133 | **`--checkpoint-file` default `"resultados_progreso.json"` → `None` + sufijo temporal `resultados_progreso_YYYYMMDD_HHMM.json`**: el nombre se genera a nivel de módulo (`time.strftime('%Y%m%d_%H%M')` tras `parse_known_args`) solo cuando no se pasa el flag. **Motivo**: la lección operativa de la sesión 129 — dos procesos `process_all_pages.py` compitiendo por el MISMO checkpoint fijo se pisaban el archivo y se perdió una corrida completa de 53 págs. Ahora cada corrida escribe su PROPIO archivo; el solapamiento entre procesos/corridas es imposible salvo dos lanzamientos en el MISMO minuto (granularidad HHMM, tal como pidió el usuario — se puede subir a `%H%M%S` si se quiere eliminar ese caso). **Resume intacto**: `--checkpoint-file` explícito se usa tal cual (verificado por `test_checkpoint_resume_salta_paginas_ya_hechas` que ya existía + `test_checkpoint_file_explicito_usa_nombre_exacto` nuevo). **Consumidores verificados**: `reprocess_failed.py` tiene su PROPIO default fijo (`CHECKPOINT_FILE = "resultados_progreso.json"` en su línea 29 — herramienta aparte que lee/escribe su archivo, no importa el default de process_all_pages) y `run_ci.py` usa el nombre fijo solo como CORPUS de análisis de calidad con `[SKIP]` si no existe — ninguno se rompe. **Tests** (3 nuevos en `TestCheckpointDefaultTemporal`): default matchea regex `resultados_progreso_\d{8}_\d{4}\.json` y NO es el nombre fijo antiguo; `--checkpoint-file` explícito se usa exacto; dos corridas generan nombres DIFERENTES (determinista: `time.strftime` global mockeado a valores distintos por carga — se parchea el módulo `time` COMPARTIDO antes de `_load_module`, porque process_all_pages importa el mismo objeto). **Code review**: (a) el primer test de "dos corridas" era casi vacuo (early-return en el caso normal) → reescrito determinista con monkeypatch; (b) el `import re` local movido al tope del archivo; (c) comentario obsoleto `CHECKPOINT_FILE puede sobrescribirse… (default aquí)` eliminado (el default ya vive en el bloque de la sesión 133). **461 tests en verde** (3 nuevos; la suite completa de 7 archivos se corre con archivos EXPLÍCITOS — `pytest tests/` recursivo recoge `tests/archive/` que contiene tests obsoletos que crashean la recolección con exit 1 sin salida, problema PRE-EXISTENTE no relacionado con este cambio), `py_compile` OK. | `process_all_pages.py`, `tests/test_process_all_pages.py` | 🛡️ **Sin checkpoint pisado entre corridas** |

#### Sesión 2026-08-06-v9 — Benchmark batch-vs-single págs 38-42 con trigger forzado: infer_multi NO gana con daemon caliente (1 hallazgo + 1 CLI)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 113 | **Hallazgo de benchmark (págs 38-42, fusion, workers=2, daemon caliente, `--force-uocr` nuevo flag)**: **el modo single (270s pared) fue MÁS RÁPIDO que `--batch-window 4` (587s pared)** — la ventaja teórica de infer_multi compartiendo prefill NO se materializó con el daemon caliente. Daemon puro: single 5 llamadas individuales = 261.9s (45.4+54.5+42.4+42.9+76.7) vs batch 1×infer_multi de 4 págs = 570.9s (142.7s/pág) + 1 pág suelta 9.7s. Explicación probable: con la serialización GPU §82 y la degradación CPU §82 ya activas, las llamadas individuales ya no sufren contención (antes 366-592s en F5 por pelea VRAM); el ahorro de prefill compartido es pequeño frente al coste de generar N páginas en un solo contexto. **Cualidad**: batch recuperó más bloques (19 vs 15) y mejor tasa (78.9% vs 46.7%) — el batch aún gana en detección en páginas artísticas, pero su ventaja de tiempo solo aplica cuando las páginas disparan U-OCR MASIVO (p5, 39-42, 51-53 de F5). Se deja el default en `--batch-window 1` para texto normal. | `process_all_pages.py`, AGENTS.md | 📊 Benchmark real |
| 114 | **Nuevo flag CLI `--force-uocr`** en `process_all_pages.py`: pasa `force_uocr=true` en los payloads de `/api/process-page` y `/api/process-page-batch` (skip del trigger v4.2). Permite benchmarks deterministas (mide la ventaja de infer_multi sin no-determinismo del trigger). **Fixes del code review**: (a) docstring/help de `--batch-window` actualizados con el hallazgo medido (batch-window 4 NO es más rápido con daemon caliente — se recomienda default 1); (b) warning en `main()` si `--force-uocr` con daemon `loading` (evita runs silenciosos 100% http_503); (c) 5 tests nuevos del flag (payload single/batch True/False + warning daemon no-ready). **16 tests** de `test_process_all_pages.py` en verde. | `process_all_pages.py`, `tests/test_process_all_pages.py` | 🆕 Benchmark determinista |

#### Sesión 2026-08-06-v8 — Refactor process_all_pages.py testable + tests unitarios (1 refactor + 1 test file)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 111 | **Refactor para testear `process_all_pages.py` en aislamiento**: el flujo completo (health check, apertura del PDF, límite `--max-pages`, carga de checkpoint, render thread, worker loop, reporte) se movió a `main()` bajo `if __name__ == "__main__"`, y el estado (`pages_done`/`results`/`page_times`/`stats`/`total_pages`/`_pages_processed_since_checkpoint`) pasó a defaults a nivel de módulo. `_registrar_resultado` protege la división por `total_pages` (0 en aislamiento → pct=0). Los side effects de import (`os.environ['PYTHONIOENCODING']`, `sys.stdout.reconfigure`) se movieron a `main()` con guardas (los tests importan el módulo y no deben tocar el stdout/entorno del proceso pytest). Global muerto `_total_pages` eliminado. `try/finally` alrededor de `wait()` para `api_workers.shutdown()`/`doc.close()`. | `process_all_pages.py` | 🧹 Importable + testeable |
| 112 | **`tests/test_process_all_pages.py` (11 tests)**: (a) `procesar_pagina` con `requests` mockeado (`_http_session.post`) verifica que registra resultados SIN NameError — regresión del fix 109 (comprobado: el test FALLA si se quita `data = resp.json()`) — y los statuses VACIO/SIN_TRAD/render_error/timeout/http_500; (b) `main()` con fitz/requests mockeados sobre un PDF fake de 53 páginas verifica que `--max-pages 5` limita el render a 5 (checkpoint resultante con 5 páginas), que sin `--max-pages` procesa las 53, que `--max-pages 999` se limita al tamaño del PDF, y que un checkpoint previo coincidente hace saltar las páginas ya hechas (solo re-renderiza las pendientes). El módulo se carga en fresco por test (importlib con nombre único + `sys.argv` controlado y restaurado). | `tests/test_process_all_pages.py` (nuevo) | ✅ Cobertura del fix 109 + --max-pages |

**Nota infraestructura (pre-existente, no de esta sesión)**: correr los 7 archivos de `tests/` juntos en un solo proceso falla de forma INTERMITENTE con crash nativo silencioso (0 bytes de output, exit 1) — se reproduce sin los archivos de esta sesión y en frío; los 403 tests pasan por separado y en combinaciones parciales. Causa probable: race de VRAM (test_api.py importa server.py, que lanza/adopta el daemon U-OCR de ~2.25GB VRAM, mientras otros tests cargan torch/CUDA en la GTX 1050 Ti de 4GB). No afecta al CI (`run_ci.py` no corre la suite completa en un proceso).

**Benchmark smoke easyocr vs fusion (5 páginas, workers=2, misma caché limpia, capítulo 53 págs)**: easyocr **74s (1.2 min, 14.9s/pág)** vs fusion **144s (2.4 min, 28.8s/pág)** = **1.9× más rápido** (págs 1-2 incluyen cold start ~33s de EasyOCR/CT2; steady-state easyocr 2.1-3.9s/pág). **easyocr detecta 16 bloques (13 trad, 81.2%)** vs **fusion 23 bloques (13 trad, 56.5%)**: MISMO texto real traducido (13), pero fusion recupera ~7 bloques extra vía YOLO/Ruta C que en estas páginas eran mayormente ruido no traducible (Non-Text/URLs/fragmentos), por eso su tasa es menor. Detalle: en easyocr 6 fragmentos de ruido ('MN', 'Mn', 'HEH. PAORINO...') cayeron a Google fallback (CT2 los descartó como "mismo texto"); fusion tuvo 0 Google. 0 errores en ambos. **Lectura**: para capítulos de texto normal el modo rápido es ~2× más barato con salida equivalente; fusion solo paga la pena en páginas artísticas donde el híbrido pierde diálogo real (caso "ERA UNA PROPUESTA" de la p5 del PDF de 128 págs).

**Benchmark batch vs single (8 páginas, fusion, workers=2, caché limpia, capítulo 53 págs)** — valida que el camino batch sigue funcionando tras el refactor/main() y el fix del NameError (sesión 111-112): batch `--batch-window 4` → 8/8 páginas registradas en 2 lotes (`/api/process-page-batch` 200 OK), 0 errores, **~100s de pared** (ambos lotes en paralelo; cada lote ~100s). Single (`--batch-window 1`) → 8/8 páginas, 0 errores, **~111s de pared** (00:56:09→00:58:00). ⚠️ **Comparación confundida por no-determinismo del trigger v4.2**: en batch, la p4 fue resuelta por YOLO (engines `easyocr+rapid`,`yolo+rutac`, 0 llamadas U-OCR); en single, la p4 disparó el refuerzo U-OCR (108s) y recuperó diálogo real ("...Y SI HUBIESE ACEPTADO..." → "AND IF I HAD ACCEPTED...") — por eso single detectó 41 bloq/30 trad (73.2%) vs batch 32/18 (56.2%). La ventaja real del batch (infer_multi compartiendo prefill) solo se materializa cuando VARIAS páginas del lote disparan U-OCR (F5: 2.1×). **Ojo métrica**: el "Tiempo total (suma)" del script NO es comparable entre modos — en batch cada página hereda el elapsed del lote (801s de suma para ~100s de pared). Usar timestamps del access log.

#### Sesión 2026-08-06-v6 — Caché HF dentro del proyecto: HF_HOME=hf_cache en server.py (1 cambio)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 108 | **Caché HF dentro del proyecto**: `server.py` Y `translator.py` definen `HF_HOME`/`TRANSFORMERS_CACHE`/`HF_HUB_CACHE` = `ROOT/hf_cache` (regla "no tocar C:", mismo patrón que `uocr_daemon.py:48-50`). Sin esto, las descargas de tokenizers OPUS-MT de CT2 (sesión 104) caían en `~/.cache/huggingface` del usuario y no viajaban con el proyecto. Se puso en AMBOS módulos porque `translator.py` se usa directamente (tests, `tools/`) sin pasar por `server.py`. `setdefault` respeta un HF_HOME ya definido por el entorno. **Migración de la caché**: los 15 repos OPUS (`models--Helsinki-NLP--opus-mt-*`) + 3 modelos de conversión (~1.5GB) se MOVIERON de `~/.cache/huggingface/hub` a `hf_cache/hub/` (mismo dir donde ya vive el modelo Unlimited-OCR del daemon) — la caché de usuario quedó vacía. ⚠️ Ojo: los `snapshots/` de la caché HF usan symlinks relativos (`../../blobs/...`) que `mv` no recrea entre unidades (C:→D:) — hubo que copiar 3 repos con `cp -rL` (dereference). **Verificado**: `import server` y `import translator` directos setean las 3 vars a `D:\crear traductor\hf_cache[...]`, CT2 carga tokenizer desde `hf_cache` y traduce, 87 tests de `test_translator.py` en verde. | `server.py`, `translator.py` | 🆕 Caché portable en el proyecto |

#### Sesión 2026-08-06-v5 — Rebuild .exe: spec actualizado + stdlib frozen completo (1 cambio)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 107 | **`.exe` recompilado con el fix de CT2** + `main.spec` corregido: (1) **DATAS**: agregados `ocr_engine.py`, `uocr_client.py`, `uocr_daemon.py` (se importan/ejecutan dinámicamente y faltaban en el bundle → el exe anterior crasheaba con `ModuleNotFoundError`). (2) **NO van en HIDDEN_IMPORTS**: añadirlos ahí hacía que PyInstaller analizara `ocr_engine → ocr_utils → ultralytics` y el .exe crecía de ~360MB a **2.4GB** (torch CUDA 2.4G, paddle, polars, spacy). Viajan como .py en DATAS (mismo patrón que ocr_utils.py). (3) **EXCLUDES ampliado** con todo el ML pesado que debe cargarse de `env/` en runtime: ultralytics, paddle, polars, spacy, onnxruntime, rapidocr_onnxruntime, tqdm, torchvision, timm, accelerate, peft, etc. (4) **stdlib faltante en modo frozen** agregado a HIDDEN_IMPORTS (base_library.zip no incluye todo): `unittest.mock` (lo importa torch `_config_module.py`), `modulefinder` (torch/easyocr), `plistlib` (ultralytics logger), `filecmp`/`shelve` (transformers tokenization_auto), `tqdm.auto`/`tqdm.contrib.*` (huggingface_hub — resuelto excluyendo tqdm del bundle), `PIL.ImageEnhance` (el hook de PIL no lo recoge; EasyOCR lo necesita). (5) **Fix code review**: `uocr_client._resolve_root()` — en modo frozen `__file__` apunta a `_MEIPASS`, donde NO existe `env_uocr_gpu/` → el daemon no se podía lanzar desde el .exe (solo adoptaba uno ya vivo). Ahora resuelve la raíz subiendo desde `sys.executable` buscando `env_uocr_gpu/Scripts/python.exe` (mismo patrón que `_fix_cwd()` de main.py). **Verificado end-to-end con `main.exe --server`**: el exe LANZA su propio daemon (PID propio en 5177), EasyOCR GPU 5.3s, CT2 es|en + en|es en GPU, YOLO 0.4s, `/api/translate` y `/api/translate-batch` vía `[ctranslate2 OK (fast path)]`, y **`/api/process-page` completo** (EasyOCR detecta `ola estas como` conf 0.82 → CT2 traduce → inpainted_image devuelto). **Tamaño final: 343MB**. | `main.spec`, `uocr_client.py`, `dist/main/` | 🐛 **Fix: .exe funcional con OCR+CT2+YOLO+daemon** |

#### Sesión 2026-08-06-v4 — Descarga de modelos HF faltantes: CT2 offline reparado (3 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 104 | **Tokenizers OPUS descargados a la caché HF** (`tools/download_hf_tokenizers.py`): los 16 repos Helsinki-NLP OPUS-MT de `_CT2_MODELS` NO tenían sus tokenizers cacheados en `~/.cache/huggingface/hub` (el server principal NO define HF_HOME; solo el daemon U-OCR usa `hf_cache/`). Por eso CT2 fallaba SIEMPRE con `We couldn't connect to 'https://huggingface.co'... couldn't find it in the cached files` en todos los runs F5/F6 (el tokenizer se carga con `local_files_only=True`). Se descargaron SOLO los archivos del tokenizer (config.json, tokenizer_config.json, vocab.json, *.spm — ~5-20MB c/u, sin los pesos de cientos de MB que no se necesitan porque las conversiones CT2 ya existen). | `~/.cache/huggingface/hub/`, `tools/download_hf_tokenizers.py` (nuevo) | 🐛 **Fix crítico: CT2 es|en y en|es vuelven a traducir offline** |
| 105 | **Nombres de repos corregidos en `_CT2_MODELS`**: `en|ja` apuntaba a `Helsinki-NLP/opus-mt-en-ja` y `en|ko` a `opus-mt-en-ko` — **esos repos NO existen en HuggingFace** (401). Corregidos a los reales: `opus-mt-en-jap` y `opus-mt-tc-big-en-ko`. | `translator.py` | 🐛 Fix de pares muertos |
| 106 | **Conversión CT2 de los 3 pares reverso CJK que faltaban** (`en|ja`, `en|ko`, `en|zh` — solo existían los directos ja/ko/zh→en): `tools/convert_ct2_missing.py`. **Workaround necesario**: `TransformersConverter` de ctranslate2 4.8.1 pasa el kwarg `dtype=` a `MarianMTModel.from_pretrained`, incompatible con transformers 4.48.3 (`unexpected keyword argument 'dtype'`) — la conversión se hace manualmente con `_MODEL_LOADERS['MarianConfig']` + `spec.validate()/optimize(int8)/save()` + centinela `.ct2_conversion_ok` + checksums SHA256 (mismo formato que las 13 conversiones previas). | `models/ct2/en-{ja,ko,zh}/`, `tools/convert_ct2_missing.py` (nuevo) | 🆕 16/16 pares CT2 listos |

**Verificación**: 16/16 pares de `_CT2_MODELS` cargan y traducen (es|en → "Hey, how are you today?", en|es → "Hola, ¿cómo estás hoy?", ja|en/ko|en/zh|en → correcto; los reverso CJK en→ja/ko traducen pero con calidad OPUS limitada, esperado). Los 87 tests de `tests/test_translator.py` en verde. Los pares es|en/en|es del capítulo (español→inglés) ya no dependen de Google. **Migración posterior (sesión 108)**: `server.py` ahora define `HF_HOME`/`TRANSFORMERS_CACHE`/`HF_HUB_CACHE` = `ROOT/hf_cache` (mismo patrón que el daemon U-OCR, regla "no tocar C:") — los tokenizers descargados se MOVIERON de `~/.cache/huggingface/hub` a `hf_cache/hub/` (15 repos OPUS + 3 modelos de conversión, ~1.5GB), y la caché de usuario quedó vacía. Al recompilar el .exe, la caché HF NO va dentro del bundle — `hf_cache/` debe viajar junto al proyecto (config.py ya resuelve ROOT al proyecto en modo frozen).

#### Sesión 2026-08-06-v3 — Fase 1 integrada en OCRManager: batch multi-página con infer_multi (4 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 95 | **`run_ocr_batch()` en OCRManager** (Fase 1 completa): procesa N páginas en paralelo lógico — híbrido + trigger v4.2 + Fase 2 por página, acumula las que necesitan VLM y las envía en **UN solo `_ocr_with_unlimited_batch`** (daemon `/ocr-batch` con `_model.infer_multi()`, las N imágenes comparten el prefill del modelo). Luego Ruta C + fusión por página. Protege el límite de 4 (VRAM daemon), guarda la firma §8.4.1 por índice en Fase A (no recomputa el grid 8×8 en Fase B) y degrada a híbrido por página si el daemon cae (RuntimeError). | `ocr_engine.py` | 🚀 **~1.2-2x** páginas artísticas |
| 96 | **Refactor de parseo/pipeline compartido en `routes/api.py`**: `_parse_daemon_blocks()` (convierte bloques crudos del daemon → bloques del server con type/confianza/fontSize; usado por single y batch) y `_finalize_page_blocks()` (pipeline post-OCR: filtro watermarks → idioma → inpaint → traducción → armado de respuesta; usado por `/process-page` y `/process-page-batch`). El endpoint single refactorizado a delegar — mismo orden de operaciones y watchdog zombie intacto. | `routes/api.py` | 🧹 Cero duplicación |
| 97 | **Nuevo endpoint `POST /api/process-page-batch`**: acepta 1-4 imágenes, valida tamaño/dimensiones por imagen, delega en `OCRManager.run_ocr_batch()` y devuelve `{results: [{inpainted_image, blocks, ocr_engine, engines_used}], t_total}` en el mismo orden. `_ocr_with_unlimited_batch()` con la misma serialización GPU (`_uocr_inferring` + `_gpu_lock`) que el single. | `routes/api.py` | 🆕 API batch |
| 98 | **`process_all_pages.py --batch-window N`** (default 1): agrupa páginas contiguas en un solo request `/api/process-page-batch`. `procesar_lote()` con retry del lote completo y `_registrar_resultado()` compartido (stats/checkpoint únicos). Fix code review: re-insertar el centinela al final para evitar stall de 60s. | `process_all_pages.py` | 🆕 Batch CLI |
| 99 | **Fase 3 punto 3 — TextClassifier de RapidOCR en la Ruta C**: `_classify_rotate_crop()` en `ocr_utils.py` usa el Cls PP-OCRv4 (ONNX CPU, `cls_thresh=0.9`) para detectar globos rotados 180° y rotarlos ANTES del re-OCR con EasyOCR (que no detecta texto girado). La librería ya devuelve la imagen rotada internamente; la función devuelve `(img, se_roto, score)` con degradación segura (sin engine/error → crop original). `_recover_regions_with_easyocr` refactorizado: mapeo unificado dict/tupla a coords upscale + **des-rotación 180°** (`x→W-x-w, y→H-y-h`) antes de mapear a página. Flags: `RUTA_C_CLS_ENABLED`/`RUTA_C_CLS_THRESH` en config.py; `disable_uocr` (benchmark) también apaga el cls vía Event `_ruta_c_cls_disabled` (mismo patrón que `_uocr_inferring`). 8 tests nuevos (cls 180/0, score bajo, sin engine, flag off, fallo ONNX, integración des-rotación, manager disable_uocr). | `ocr_utils.py`, `config.py`, `ocr_engine.py`, `tests/test_ocr_utils.py`, `tests/test_ocr_engine.py` | 🆕 Globos girados detectados |
| 100 | **INVESTIGACION_MEJORA_DETECCION_MANGA.md** (documento, sin código): búsqueda exhaustiva de técnicas para mejorar la detección de manga, priorizadas por ratio eficiencia/costo. Top: (1) YOLOv11n-seg bubble detector → regiones de la Ruta C (recupera globos que los OCR no ven como texto, 200-400ms CPU, mAP@50 98-99%); (2) Real-ESRGAN anime 6B en crops pre-re-OCR; (3) CTD ONNX como complemento (ya probado y eliminado en el proyecto en su forma PyTorch pesada, sesión 43). Descartados por evidencia: SWT/MSER (la trama de manga confunde el grosor de trazo) y binarización dura (dañina para deep learning). Incluye datasets (Manga109, DCM772, PopManga), herramientas de referencia (manga-image-translator, BallonsTranslator con ysgyolo) y roadmap F6-F10. Verificado contra el código: `_detect_bubble_regions_in_panel` (ocr_utils.py:961) es heurística OpenCV (luminancia>200 + roundness) — la brecha exacta que YOLO cierra; `rotation_info` de EasyOCR NO está activado. | `INVESTIGACION_MEJORA_DETECCION_MANGA.md` (nuevo) | 📊 Investigación priorizada |
| 102 | **Fase 6 — detector YOLO de regiones de texto (globos/cartelas/títulos)**: `_detect_text_regions_in_page()` en `ocr_utils.py` usa un YOLO fine-tuned (ogkalu `comic-speech-bubble-detector.pt`, 52MB en `models/`) cargado DINÁMICAMENTE (`_get_yolo_engine()` importa ultralytics en runtime; degrada a `[]` si falta, el .exe no se infla). Detecta regiones como OBJETOS y las envía a la Ruta C existente (`_recover_regions_with_easyocr`, upscale 3.5×) vía `OCRManager._ruta_c_yolo()` — integrada en `_run_fusion` y `run_ocr_batch` Fase A ANTES del trigger v4.2 (que NO se modifica). **Gate heurístico** (code review): solo corre en páginas débilmente detectadas (<3 bloques o conf <0.35) — el re-OCR de hasta 40 crops no cuesta en páginas normales. `disable_uocr` (benchmark) apaga YOLO vía Event `_yolo_disabled` (mismo patrón que `_ruta_c_cls_disabled`). Preload en `server.py` (paso 3 de `_preload_background`). **`rotation_info` en la Ruta C**: `EASYOCR_ROTATION_INFO=(0,90,180,270)` — enteros NO strings (easyocr pasa el ángulo a `scipy.ndimage.rotate` y con numpy 2.5/scipy 1.17 un string rompe el casting `cosdg`; verificado empíricamente). EasyOCR rota los CROPS internamente y devuelve cajas en coords del crop original (verificado en `make_rotated_img_list`), así el mapeo ÷upscale no cambia. Tier 1 de página completa NO usa rotation (costo ~4x). Peso `OCR_ENGINE_WEIGHTS["yolo"]=0.9`. ultralytics==8.4.115 en requirements.txt. 13 tests nuevos (mapeo/filtro clase/área/degradación/rotation/gate/disable/batch). **Prueba real**: p3 detecta 4 globos (conf 0.81-0.93), p12 3 — donde OpenCV blobs da 0. | `config.py`, `ocr_utils.py`, `ocr_engine.py`, `server.py`, `requirements.txt`, `tests/` | 🆕 **Tier 3.5 YOLO** |
| 101 | **Fix race window en página completa (bug http_500 pág 19-22)**: `_ocr_results_to_blocks()` ahora acepta dicts en formato interno (degradación v4.2: `_run_ocr_on_image` devuelve bloques RapidOCR como dicts si `_uocr_inferring` se setea entre el check de `_detect_and_ocr` y la ejecución) además de tuplas `(bbox,text,conf)` — antes el unpacking explotaba con "too many values to unpack" (un dict itera sobre sus keys) → 500. Filtro `conf < 0.08` con paridad en ambos caminos (code review). Cubre los 3 callers (tier 1, retry mag_ratio, tier 2 CLAHE). 4 tests nuevos. **Fase 5 ejecutada**: capítulo 53 págs con `--ocr-mode fusion --batch-window 4` → ~22.5 min de pared (vs ~47 min sin batch, 2.1x), 0 errores, 47/53 págs con texto; solo 3 lotes disparan U-OCR (p5, 39-42, 51-53; infer_multi ~168s/pág vs 366-592s individual). Reporte `reporte_fusion.html` regenerado con datos reales. | `ocr_utils.py`, `tests/test_ocr_utils.py`, `generate_fusion_report.py` | 🐛 **Fix 500 + 🚀 2.1x** |


| 103 | **Fase 6.5 — YOLO device `auto` (GPU cuando el daemon duerme)**: `YOLO_DEVICE` pasa de `"cpu"` a `"auto"` — `_detect_text_regions_in_page` resuelve a GPU `"0"` SOLO si `torch.cuda.is_available()` Y `_uocr_inferring` NO está seteado (el daemon VLM no infiere); si no, `"cpu"`. **Fix code review (gap real)**: YOLO GPU ahora adquiere `_gpu_lock` NO-bloqueante — serializa con EasyOCR GPU del MISMO proceso (un worker leyendo otra página lo tiene → YOLO degrada a CPU en vez de competir por VRAM). No usa `_gpu_lock.locked()` (RLock solo refleja el hilo actual). Device explícito != cpu se respeta pero degrada si CUDA falta o el daemon infiere; `except` → cpu. **Benchmark real (GTX 1050 Ti, yolov8m imgsz=1280, 827x1170, 4 págs)**: CPU **~730ms** vs GPU **~130ms** por página = **5.6× speedup** — y la GPU funcionó con el daemon vivo (2.25GB + YOLO caben en 4GB). E2E confirmado: server loguea `[YOLO] 4 regiones (device=0)`, engines=`['easyocr+rapid','yolo+rutac']`. 6 tests nuevos (device auto: daemon infiere→cpu, sin CUDA→cpu, CUDA libre→'0', error CUDA→cpu, gpu_lock ocupado→cpu, gpu_lock libre→'0'+se libera) + fixture autouse que limpia `_uocr_inferring`/`_yolo_disabled` tras cada test. | `config.py`, `ocr_utils.py`, `tests/test_ocr_utils.py` | 🚀 **YOLO 5.6× más rápido en GPU** |

#### Sesión 2026-08-06-v2 — Fase 4: fusion como default + selector Automático + badge daemon (3 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 94 | **Default `fusion` en el endpoint y el script de capítulo**: `/api/process-page` pasa de `auto` a `fusion` (híbrido EasyOCR+RapidOCR siempre; Unlimited-OCR solo con trigger v4.2 + reintento Fase 2). `process_all_pages.py` pasa de `easyocr` a `fusion` (docstring/help actualizados con tiempos estimados del capítulo de 53 págs). `stress_test_memory.py` fija `ocr_mode='easyocr'` explícito (mide memoria, no calidad — no debe disparar el daemon). `reprocess_failed.py`/`gestor.py` dependen del default del endpoint → fusion (comportamiento deseado). | `routes/api.py`, `process_all_pages.py`, `stress_test_memory.py` | 🚀 **Mejor calidad por defecto** |
| 93 | **Selector UI simplificado a 'Automático'**: el `#ocrEngine` queda con `fusion` → "Automático (recomendado)" (default), `unlimited` → "Fuerza Unlimited-OCR (preciso)" y `easyocr` → "EasyOCR (rápido)". Eliminada la opción legacy `auto` (EasyOCR+CLAHE) — el endpoint sigue aceptándola para retrocompat (scripts), pero la UI ya no la expone. El badge de estado del daemon (`#ocrEngineStatus` + `updateOcrEngineStatus()` con polling 20s, clases `ready`/`loading`/`offline`) ya existía de una sesión anterior — se conserva. | `index.html` | 🆕 UX simplificada |
| 92 | **Tests del nuevo default**: `test_default_ocr_mode_es_fusion` (POST sin ocr_mode → `ocr_engine='fusion'`, `engines_used==['easyocr+rapid']`, daemon no llamado); `test_ocr_no_blocks_returns_empty` mockea el camino fusion completo (Fase 2 + daemon caído); `test_ocr_blocks_returned` mockea `_page_has_large_image_panel` (determinista bajo fusion). | `tests/test_api.py` | ✅ Cobertura default |

#### Sesión 2026-08-06 — Fase 3: tipos semánticos del VLM propagados y ponderados en la fusión (2 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 87 | **Ponderación por tipo en la votación de la fusión**: `_fusionar_blocks_multi` refuerza la confianza del ganador con `FUSION_TYPE_REINFORCE` (title +0.20, header +0.18, text +0.15) cuando 2+ motores coinciden — usa el type del bloque entrante, o el del sobreviviente, o `text` por defecto. `_block_score` multiplica por `FUSION_TYPE_WEIGHTS` (title 1.15, header 1.05; sin type → 1.0, sin cambio para EasyOCR/RapidOCR que nunca llevan type). Constantes en `config.py`. | `ocr_utils.py`, `config.py` | 🆕 Votación por tipo |
| 86 | **Type semántico propagado a los bloques U-OCR**: `_ocr_with_unlimited` perdía el campo `type` (text/title/header) del daemon — lo calculaba para la heurística pero no lo incluía en el dict. Ahora los bloques llevan `type` hasta la fusión, y el payload de `/api/process-page` lo expone (`block.get('type', 'text')`) para que el frontend pueda filtrar por tipo. Caminos ya cubiertos: daemon single/batch y re-OCR de arte. La Ruta C produce bloques EasyOCR/RapidOCR sin tipo semántico (default text). | `routes/api.py` | 🆕 Tipo en respuesta |

#### Sesión 2026-08-05 — Fase 2: reintento agresivo de RapidOCR pre-VLM + batch multi-página del daemon (2 features)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 85 | **Fase 2 — reintento agresivo de RapidOCR antes del VLM**: `_reforzar_con_rapid_agresivo` en `OCRManager` corre `_run_rapidocr` con `box_thresh 0.30 / unclip_ratio 2.2 / text_score 0.40` (CPU ~1.5s) cuando la conf media del híbrido es baja y NO hay panel image grande; si el merge resuelve la página con margen (>=3 bloques Y conf >=0.30, por encima del trigger 0.2), se **evita la inferencia VLM (~2-8 min)**. Skipeado con `force_uocr`/`disable_uocr`. `_run_rapidocr` ahora pasa SIEMPRE los params explícitos (defaults 0.5/1.6/0.5 si no se indican) — la librería muta `postprocess_op` en la primera llamada con kwargs y no debe filtrarse una llamada agresiva a una posterior. Constantes `RAPID_AGGRESSIVE_PARAMS`, `RAPID_RETRY_MAX_CONF` (guarda defensiva, inalcanzable con el trigger v4.2), `RAPID_RETRY_SALVADO_CONF` en `config.py`. 10 tests nuevos (6 manager + 3 params + 1 frontera). | `ocr_engine.py`, `ocr_utils.py`, `config.py`, `tests/` | 🚀 **Menos disparos VLM** |
| 84 | **Fase 1 — batch multi-página del daemon U-OCR**: `POST /ocr-batch` en `uocr_daemon.py` usa `_model.infer_multi()` (N páginas en UNA inferencia VLM, separadas por `<PAGE>` — semántica oficial `outputs.split('<PAGE>')[1:]` del modelo: el marcador va ANTES de cada página). `_parse_blocks_multi` (sin filtrar secciones vacías — bug corregido: una página sin texto desalineaba las posteriores), `_map_multi_blocks_to_page` (640x640 → página), `_recover_art_dialogue` por página con try/except aislado. `process_batch()` en `uocr_client.py` (timeout proporcional). Validación 1-4 imágenes + paths existentes (400). 11 tests con `_FakeModel` stub. | `uocr_daemon.py`, `uocr_client.py`, `tests/test_uocr_daemon.py` | 🆕 Inferencia compartida |

#### Sesión 2026-08-05-v2 — OCRManager + cache §8.4.1 + degradación §8.4.4 + docs de investigación (4 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 91 | **OCRManager (ocr_engine.py)**: refactor de la orquestación de los 3 motores en una clase única. `run_ocr()` dispatcher (easyocr/auto/fusion/unlimited), `_compute_trigger` (trigger v4.2 aislado y testeable), `_reforzar_con_unlimited`, `_ruta_c_globos`, `_has_big_panel`. `routes/api.py`: el bloque OCR inline de `process_page` (~110 líneas) → `OCRManager().run_ocr(...)`. **Acceso a las funciones EN RUNTIME vía módulo** (`self.ou._detect_and_ocr`, `import routes.api as ra` dentro de `_unlimited_ocr`) para que los mocks existentes (`ocr_utils._detect_and_ocr`, `routes.api._ocr_with_unlimited`) sigan funcionando — los 108 tests de test_api.py pasaron intactos sin tocarlos. Sin dependencias circulares. `tests/test_ocr_engine.py` (18 tests) añadido al CI. 312 tests en verde. | `ocr_engine.py` (nuevo), `routes/api.py`, `tests/test_ocr_engine.py`, `run_ci.py` | 🧹 Refactor sin cambio de comportamiento |
| 90 | **Cache de decisiones negativas §8.4.1 por firma de página**: `_page_signature` en ocr_utils (grid 8×8 de oscuridad + dark_ratio cuantizado; calibrado empíricamente — 10 páginas del capítulo comparten la firma principal). `OCRManager._uocr_neg_cache` (class vars + lock + TTL 1800s + eviction LRU 256). Solo se cachean resultados donde el refuerzo U-OCR **no recuperó nada** → las páginas repetitivas del capítulo con la MISMA firma no re-disparan la inferencia VLM (~2-8 min). Las decisiones positivas NO se cachean (una página gemela con diálogo distinto debe poder re-disparar). Constantes `UOCR_CACHE_TTL_S`/`UOCR_CACHE_MAX_ENTRIES` en `config.py`. | `ocr_utils.py`, `ocr_engine.py`, `config.py` | 🚀 **Sin VLM redundante en páginas repetitivas** |
| 89 | **Degradación §8.4.4 de la Ruta C a RapidOCR CPU**: `_recover_regions_with_easyocr` chequea `_uocr_inferring.is_set()` ANTES de `_get_ocr_reader()` — si el daemon infiere, el re-OCR de globos usa `_run_rapidocr` sobre el crop upscaleado (mapeo ÷upscale + offset, `engine='rapidocr-region'`) en vez de esperar el `_gpu_lock` sin timeout. **Fix race window del code review**: `_run_ocr_on_image` puede degradar internamente a RapidOCR devolviendo dicts en vez de tuplas (bbox,text,conf) — normalización de ambos formatos + test dedicado. 315 tests en verde. | `ocr_utils.py`, `tests/test_ocr_utils.py` | 🧵 **CPU mientras el daemon infiere** |
| 88 | **Docs de investigación interna**: `INVESTIGACION_3_OCR.md` (nuevo) — análisis a nivel de función del código interno de los 3 motores: EasyOCR (reader lazy, detector CRAFT, recognizer CRNN, `detect()`/`recognize()` separables), RapidOCR (PP-OCRv4 DBNet+Cls+rec ONNX, `__call__(**kwargs)` ajusta `box_thresh`/`unclip_ratio` sin recargar), Unlimited-OCR (VLM DeepSeek-V2-Lite 3B, `infer()`/`infer_multi()`, `TPSTextStreamer` por stdout, `re_match`) + capa de fusión + tabla de locks GPU. `PLAN_CODIGO_INTERNO_3_OCR.md` (nuevo) — arquitectura a nivel de paquete con referencias exactas (easyocr `craft.py:30`, rapidocr `main.py:66`, `modeling_unlimitedocr.py:1139`) + veredicto de complementariedad + **5 fases de implementación** (batch `infer_multi` → params agresivos → tipos semánticos + rotación → fusion default → validación). **Descubrimiento clave**: `infer_multi()` ya existe en el modelo y soporta N imágenes por inferencia — la base de la Fase 1 (batch). | `INVESTIGACION_3_OCR.md`, `PLAN_CODIGO_INTERNO_3_OCR.md` | 📚 Base del roadmap |

#### Sesión 2026-08-04-v2 — Benchmark overhead puro de la fusión: el merge NO cuesta nada (2 hallazgos + 2 flags)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 83 | **Flags de benchmark**: `disable_uocr` (anula el refuerzo U-OCR en fusion → solo EasyOCR+RapidOCR+merge) y `pure_easyocr` (desactiva el tier híbrido RapidOCR → EasyOCR GPU puro) en `/api/process-page`. `benchmark_fusion_overhead.py` mide los 3 modos sobre el PDF nuevo completo (53 págs). | `routes/api.py`, `benchmark_fusion_overhead.py` | 🆕 Benchmark overhead |

**Hallazgo 1 — el "modo easyocr" de la app YA es híbrido**: `_detect_and_ocr(use_hybrid=True)` por defecto corre EasyOCR+RapidOCR. El benchmark reveló que EasyOCR GPU puro solo detecta **22 bloques** en 53 págs, mientras el híbrido detecta **225** (RapidOCR aporta ~90% de la detección). Los benchmarks anteriores de "EasyOCR solo" (7.1 min, 623 bloques) eran en realidad EasyOCR+RapidOCR.

**Hallazgo 2 — el overhead del merge de fusión es ~0**: medido con `t_ocr` del log del servidor (sin sesgo de caché de traducción): fusion+disable_uocr **6.91s/pág** vs easyocr híbrido **7.25s/pág** (-0.34s, ruido) y 224/225 textos idénticos. La fusión no cuesta nada sin U-OCR; el 100% del overhead del modo fusion (150 min vs 7.1 min) está en la inferencia del daemon U-OCR (~130-250s/página), NO en la fusión de bloques. El merge `_fusionar_blocks_multi` solo se activa cuando U-OCR dispara. | `benchmark_overhead_results.json` | 📊 Dato clave |

#### Sesión 2026-08-04 — Optimizaciones v4.2 del modo fusion: trigger selectivo + serialización GPU + degradación CPU (3 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 80 | **Trigger U-OCR selectivo v4.2**: `UOCR_TRIGGER_CONF` 0.25 → **0.20**. El refuerzo U-OCR solo dispara con `image > 15%` O (`< 3 bloques` Y `conf < 0.2`) — antes bastaba <3 bloques a secas y disparaba 41/128 páginas (32%) = 95% del tiempo del capítulo. Benchmark 9 págs del PDF nuevo: 7/9 normales NO disparan (8-29s c/u). | `config.py` | 🎯 **Menos refuerzos inútiles** |
| 81 | **Serialización GPU**: `_gpu_lock` (RLock) compartido en `ocr_utils.py` — EasyOCR (`_run_ocr_on_image`) y daemon U-OCR (`_ocr_with_unlimited` en `routes/api.py`) se serializan: un solo motor GPU a la vez. Evita la contención VRAM que pasaba al daemon de 83s a 140-1439s. | `ocr_utils.py`, `routes/api.py` | 🧵 **GPU no se pelea** |
| 82 | **Degradación CPU mientras el daemon infiere**: flag `_uocr_inferring` (Event global en `ocr_utils.py`). `_ocr_with_unlimited` lo setea durante la llamada al daemon; `_detect_and_ocr` lo consulta **antes de `_get_ocr_reader()`** y, si está activo, degrada a **RapidOCR CPU puro** (no carga EasyOCR a VRAM ni toca la GTX) en vez de esperar el mutex → los workers de otras páginas avanzan en paralelo con la inferencia VLM. **Race-window fix (code review)**: `_run_ocr_on_image` re-chequea el flag tras adquirir el semáforo — si el daemon empezó a inferir justo después del chequeo inicial, degrada a CPU en vez de bloquearse en `_gpu_lock` sin timeout durante 2-8 min (lo que provocaba timeouts de 120s en los demás workers). Benchmark: p15 247.7→152.0s (-39%), p5 497.8→132.4s (-73%). | `ocr_utils.py`, `routes/api.py` | 🚀 **-39% a -73%** en páginas U-OCR |

**Benchmark v4.2 (2026-08-04, 9 páginas del PDF nuevo, modo fusion)**: trigger selectivo funciona (solo páginas con panel image>15% disparan U-OCR); degradación CPU real (p15 -39%, p5 -73%); calidad preservada — pág. artística p5 recupera `ERA UNA PROPUESTA` (conf 1.0) que EasyOCR solo pierde. Estimación capítulo 53 págs: ~44 min (vs 150.6 min originales, 3.4× menos) con workers=2. Detalle completo en `PLAN_FUSION_OCR.md §3.6`. | `benchmark_v42.py` | 📊 Benchmark |

#### Sesión 2026-08-03-v2 — PDF de prueba actualizado: el usuario puso uno nuevo en Descargas (1 cambio)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 79 | **PDF de prueba nuevo**: reemplazado `Capítulo 43 de Cómo criar villanos correctamente _ Olympus Scanlation_compressed.pdf` (128 págs, ~50MB) por `Capítulo 43 de Cómo criar villanos correctamente.pdf` (53 págs, 2.87MB) copiado de Descargas. Actualizadas las 10 referencias al nombre del PDF (process_all_pages.py, run_ci.py, gestor.py, stress_test_memory.py, reprocess_failed.py, benchmark_ocr_tiers.py, benchmark_artistic_pages.py, benchmark_ruta_c_v2.py, benchmark_unlimited_ocr.py, contexto_para_ia.md). `process_all_pages.py` lee `len(doc)` dinámicamente → el checkpoint viejo (total_pages=128) se descarta solo. `benchmark_page{3,11,12}.png` regenerados del PDF nuevo. E2E validado: página 5 → 4 bloques en 23.7s (el diálogo artístico "ERA UNA PROPUESTA" aparece ahora en pág. 5 del PDF nuevo, no en la 12). | 10 archivos .py/.md | 🆕 PDF actualizado |

#### Sesión 2026-08-03 — Modo fusion: fusión de los 3 OCRs implementada + benchmark de capítulo completo (2 features)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 74 | **Fusión multi-motor implementada (Ruta B)**: `_estimate_confidence_heuristic()` (confianza por tipo de bloque + calidad de texto + consenso + fontSize, pesos EasyOCR×1.0/RapidOCR×0.9/U-OCR×1.1), `_fusionar_blocks_multi()` (dedup texto + IoU + alineación Levenshtein + votación ponderada + NMS calibrado) en `ocr_utils.py`. Constantes de fusión en `config.py` (`FUSION_TRIGGER_*`, `ENGINE_WEIGHTS`). | `ocr_utils.py`, `config.py` | 🆕 Fusión de bloques |
| 75 | **`ocr_mode="fusion"` en `/api/process-page`**: cascada B+C — EasyOCR+RapidOCR siempre; dispara U-OCR (daemon) solo si <3 bloques o conf<0.25 o bloque image>15%; merge con `_fusionar_blocks_multi`; `engines_used` en respuesta. Import local de `_estimate_confidence_heuristic` en `_ocr_with_unlimited` (bug de scope corregido). | `routes/api.py` | 🆕 Modo fusion |
| 76 | **Frontend + script**: opción `fusion` en selector OCR (default) y fallback en `app.js`; `--ocr-mode fusion` soportado en `process_all_pages.py` (timeout subido a 900s por páginas U-OCR). Test API extendido con mock del daemon (cv2.imwrite con MagicMock corregido). | `index.html`, `app.js`, `process_all_pages.py`, `tests/test_api.py` | 🆕 UI + script |
| 77 | **Fix PIL en daemon**: `uocr_daemon.py::_recover_art_dialogue` usaba `cv2` que `env_uocr_gpu` NO tiene (bug preexistente — el venv GPU solo cargaba torch/transformers) → HTTP 500 en toda página con panel image. Reescrito con PIL puro (`Image.open`+`crop`+`resize(LANCZOS)`+`paste`+`save`). | `uocr_daemon.py` | 🐛 **Fix crítico: re-OCR artístico crasheaba siempre** |
| 78 | **Benchmark capítulo completo (128 págs, modo fusion)**: 128/128 páginas, 0 errores, 590 bloques (408 traducidos = 69.2%), **9038s = 150.6 min** (vs EasyOCR solo 7.1 min, auto 25.3 min). 41 páginas dispararon U-OCR → 95% del tiempo (promedio 210s/pág, máx 1439s por contención GPU EasyOCR+daemon). **Recuperación de diálogo artístico real**: pág. 3 SFX `INCREIBLE REALMENTE` → fusion lee `NCREIBLE` (CER 0.819, EasyOCR solo leía 0/2 palabras); pág. 12 globo en panel recuperado (CER 0.717). **Veredicto**: cobertura igual, CER artístico mejor, pero 21× más lento por trigger U-OCR demasiado agresivo + contención GPU. Optimizaciones v4.2 documentadas en PLAN_FUSION_OCR.md §3.6 (trigger selectivo + serializar GPU). | `PLAN_FUSION_OCR.md` | 📊 Benchmark real |

#### Sesión 2026-08-03 — Daemon persistente Unlimited-OCR (GPU 4-bit) con preload en background (1 feature)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 70 | **Daemon persistente Unlimited-OCR**: nuevo `uocr_daemon.py` (proceso HTTP en 127.0.0.1:5177, corre en el venv dedicado `env_uocr_gpu` con torch cu126 + bitsandbytes). Carga el modelo 4-bit NF4 **UNA sola vez** (~8 min, 2.25 GB VRAM) y sirve `GET /health` (estado loading/ready/error) y `POST /ocr` (image_path + max_length → text, blocks con coordenadas `<|det|>`). `uocr_client.py` (stdlib puro) expone `spawn_daemon()` (con adopción de daemon ya vivo, verificación post-lanzamiento y auto-reinicio en estado error), `health()`, `wait_ready()`, `process_page()`. | `uocr_daemon.py`, `uocr_client.py` | 🆕 **Elimina los 494s de carga one-shot por página** |
| 71 | **Preload en background en server.py**: el daemon se lanza en un **hilo propio e independiente** (`_preload_unlimited_daemon`, paso 3 del preload) — NO bloquea ni depende de EasyOCR/CT2. La primera página ya no espera 8 min: benchmark real 63s → 41s por página (vs 494s carga + 84s CPU antes). | `server.py` | 🚀 Primera página sin espera |
| 72 | **`ocr_mode="unlimited"` en `/api/process-page`**: usa el daemon (helper `_ocr_with_unlimited`), filtra ruido de página (footer/page_number/image), estima fontSize desde altura del bloque. 503 con mensaje claro si el modelo aún carga. `/api/health` reporta `unlimited_ocr` y `uocr_load_s`. | `routes/api.py` | 🆕 Motor OCR alternativo |
| 73 | **Robustez**: limpieza de daemon zombi en `start-app.ps1` (puerto 5177 + 5174), timeout de lock de inferencia (1800s), validación max_length, limpieza de `uocr_daemon_out/req_*` (máx 20). | `uocr_daemon.py`, `start-app.ps1` | 🛡️ Sin zombies |

**NOTA hardware**: el daemon U-OCR consume ~2.25 GB VRAM permanente + EasyOCR ~0.13 GB → caben en la GTX 1050 Ti de 4 GB, pero no ejecutar OCR U-OCR y EasyOCR simultáneamente en páginas densas.

### Cambios acumulados (Julio 2026)

#### Sesión 2026-07-29 — Fix timeout de traducción + logs en tiempo real (2 fixes)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 68 | **Timeout de traducción eliminado**: reemplazado el pipeline paralelo (CT2+Argos+Google con `as_completed(timeout=30s)` + Google retry 5+15+30s = hasta 80s por texto) por flujo secuencial CT2→Google→SIN_TRAD. CT2 primero (síncrono, ~0.02-0.12s en GPU). Google fallback (~2s). SIN_TRAD inmediato sin esperas. **Resultado**: "CAPITULO 43" → "CHAPTER 43" en 0.03s, "TEMPORADA 1" → "SEASON 1" en 0.02s, "Como Criar Villanos Correctamente" → "Like raising villains correctly" en 0.07s. **Código muerto eliminado**: `_get_translate_engine_executor()`, `_probar_motor()`, `translation_fns`, lógica `_es_ocr_noise()`. | `translator.py` | 🚀 **~1000x más rápido** (0.03s vs 30-80s) |
| 69 | **Logs en tiempo real**: `sys.stdout.reconfigure(line_buffering=True)` en server.py + `PYTHONUNBUFFERED=1` + `python -u` al iniciar. Antes: Python buferizaba stdout en bloques de 8KB cuando se redirigía a archivo → los `print("[translate] ...")` NUNCA aparecían en `server_output.log` hasta que el servidor terminara. Ahora: cada `\n` vacía el buffer inmediatamente. | `server.py` | 🐛 **Fix: logs ahora visibles en tiempo real** |

#### Sesión 2026-08-01 — Diccionario manual eliminado: reemplazado por pyspellchecker (0 mantenimiento)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 67 | **Diccionario manual `_OCR_DICT` eliminado** (~50 líneas, 600 palabras). Reemplazado por **pyspellchecker** (86,158 palabras pre-cargadas) con lazy-load thread-safe. **Cero mantenimiento**: el usuario nunca tendrá que agregar palabras. 16 palabras de dominio manga cargadas con `wf.add(word, 1000000)` para forzar frecuencia alta (villano, villanos, manga, manhwa, scanlation, capítulo, etc.). `_ocr_spellcheck()` refactorizada para llamar `_get_spellchecker()` una vez fuera del loop. Fallback `_levenshtein()` + `_FALLBACK_DICT` si pyspellchecker no está instalado. `pyspellchecker==0.9.0` agregado a `requirements.txt`. | `ocr_utils.py`, `requirements.txt` | 🆕 **Cero mantenimiento** |

#### Sesión 2026-07-30 — Inpainting por glifos universal + Fix cabeceras con hora en punto (2 fixes)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 64 | **Inpainting por glifos universal**: `_build_inpaint_mask()` ahora aplica máscara por trazo de letra (glifos) en el 100% de los bloques por defecto. Eliminada la restricción de `brightness > 80` en `_is_inside_speech_bubble()` que descartaba el 99% de globos blancos. Preserva el arte del manga, fondos e ilustraciones sin destruirlos con rectángulos sólidos. | `ocr_utils.py` | 🎨 **Arte preservado** |
| 65 | **Fix de marcas de tiempo en cabecera**: Se actualizó `MARGIN_NOISE_PATTERNS` en `config.py` y `js/filters.js` para admitir horas con punto (`4.58 p.m.`) además de dos puntos. Ajustado `margin_top` de 7% a 8.5% para eliminar cabeceras de navegación de lectores web. | `config.py`, `js/filters.js`, `ocr_utils.py` | 🧹 **Márgenes limpios** |
| 66 | **Pipeline híbrido universal RapidOCR + EasyOCR**: `_detect_and_ocr()` ahora ejecuta siempre RapidOCR para complementar EasyOCR. Resuelve el problema donde EasyOCR direct ignoraba títulos estilizados en dorado o portadas (como "Cómo Criar Villanos Correctamente") devolviendo solo ruido de margen. | `ocr_utils.py` | 🎯 **Títulos detectados** |

#### Sesión 2026-07-30 — is_ocr_garbage() mejorado: 8 filtros para OCR fragments + pipeline híbrido EasyOCR+RapidOCR (2 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 62 | **`is_ocr_garbage()` mejorado con 8 filtros**: detección de OCR fragments que antes caían como UNTRANSLATED. Nuevos filtros: (5) texto 1-2 chars sin vocales → "N", "kc"; (6A) empieza minúscula + resto mayúscula → "sRESPONDERMFR"; (6B) patrón AaaAaaA con 1-2 mayús + 2+ minús → "ADelAntE."; (6C) minúsculas + 3+ mayúsculas → "saaaAALIR!"; (7) texto ≤4 chars con espacios y dígitos → "M 4"; (8) caracteres especiales (~_ - =) en bordes → "~YSILA acePtaba _". **Resultado**: OCR_GARBAGE 1.8% → **16.8%** (60 fragmentos reclasificados correctamente), UNTRANSLATED 22.8% → **17.5%** (solo texto real sin traducir). | `analisis_calidad.py` | 📊 **Clasificación más precisa** |
| 63 | **Pipeline híbrido EasyOCR+RapidOCR** en `ocr_utils.py`: nuevo motor RapidOCR (ONNX, CPU ~1.1-1.5s/pág) como tier 3. Lazy loading con `_get_rapid_engine()` + `threading.Lock()`. `_run_rapidocr()` con semáforo thread-safe. `_fusionar_blocks()` merge por texto normalizado. `_detect_and_ocr()` modificado: si confianza EasyOCR < 0.3 → RapidOCR + fusión; si 0 bloques → RapidOCR directo. **Benchmark CPU vs GPU**: GTX 1050 Ti probada con `onnxruntime-gpu` — **sin mejora medible** (~1.0x speedup) porque los modelos PP-OCRv4 son pequeños (~1.5MB detector + ~5MB recognizer) y el overhead de transferencia PCIe anula cualquier ganancia de cómputo. **RapidOCR CPU es suficiente** (~1.1-1.5s/pág). **Benchmark 128 págs**: **11.7 min** (699s) en modo `auto`, 0 páginas sin traducir. **12.2%** de bloques sin traducir (49/400) — fragmentos OCR ruidosos que ningún motor pudo descifrar. .exe compilado: **360 MB** (+160 MB por inclusión de modelos ONNX RapidOCR). `requirements.txt` actualizado con `rapidocr-onnxruntime`. | `ocr_utils.py`, `requirements.txt` | 🚀 **Pipeline híbrido** |

#### Sesión 2026-07-29 — Fix bundle PyInstaller: carpeta js/ faltante (1 fix)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 61 | **Fix bundle PyInstaller: carpeta `js/` faltante**. `app.js` (módulo ES6) importa 5 archivos de `./js/` (config.js, utils.js, toast.js, theme.js, filters.js). `main.spec` no incluía `js/` en DATAS → al ejecutar el .exe, los imports fallaban con 404 → `window.__appJsLoaded` nunca se establecía → `initOpenCv()` nunca se ejecutaba → el badge se quedaba en "Cargando OpenCV..." para **siempre** (ni el safety timeout de 25s podía cambiar el badge porque el módulo nunca cargaba). **Solución**: agregar `(str(PROJECT_ROOT / "js"), "js"),` a DATAS. .exe recompilado: `dist/main.exe` (138MB), `dist/main/_internal/js/` contiene los 5 archivos. Verificado con test_client de Flask: los 5 endpoints `/js/*.js` devuelven 200 OK con Content-Type correcto. Además, se redujo el safety timeout de OpenCV en `index.html` de 25s a **15s** (coincide con `TIMEOUT_OPENCV_INIT_MS` de app.js) para que el badge cambie más rápido si hay un error de carga. | `main.spec`, `index.html` | 🐛 **Fix crítico: "Cargando OpenCV..." infinito en .exe** |

#### Sesión 2026-07-28-v2 — Botón Pausa/Reanudar + Aviso origen==destino (2 features)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 59 | **Botón ⏸️ Pausar / ▶️ Reanudar**: nuevo botón en la barra de progreso de `showProgress()` junto a Cancelar. Togglea `state.translationPaused` (flag agregado al estado global). En `autoTranslateAllPages()`, while loop con `await new Promise(r => setTimeout(r, 500))` espera sin bloquear hasta que se reanude. Al reanudar, continúa desde la página exacta donde se pausó. Al cancelar, resetea `translationPaused`. Se resetea al iniciar nueva traducción. Estilos CSS `.progress-pause` con hover/active. **Bug corregido**: flag no se reseteaba al reiniciar traducción. | `app.js`, `styles.css` | 🆕 Control de flujo |
| 60 | **Aviso visual cuando origen == destino**: nuevo elemento `<div id="langWarning">` debajo de los selectores de idioma. Función `checkLanguageWarning()` compara `sourceLang.value` vs `targetLang.value` usando mapa `isoToSelector` ("es"→"spa", "en"→"eng", etc.). Si source=="auto", muestra aviso condicional: "si el texto está en [idioma], no se traducirá". Si source incluye destino explícitamente, muestra aviso exacto. Clases CSS `.lang-warning` con transiciones suaves (hidden→visible). **Bug corregido en code review**: `const targetCode = tgt;` estaba en temporal dead zone (referencia antes de declaración). Movido al inicio de la función. | `app.js`, `index.html`, `styles.css` | 🆕 UX preventivo |

#### Sesión 2026-07-28 — Safety timeout OpenCV + módulo signal + race condition Case 3 (3 fixes + commit + build)

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



#### Sesión 2026-07-27 — Race condition OpenCV + BindingError doble carga + CSP warning (3 fixes + commit + build)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 53 | **Race condition OpenCV.js** (`initOpenCv()` Caso 2): check diferido `setTimeout(cvReadyCheck, 100)` tras asignar `onRuntimeInitialized`. Si el WASM ya se inicializó antes de asignar el callback, el check lo detecta en 100ms en vez de esperar el timeout de 15s. Guard `state.cvLoaded` en `onOpenCvReady()` evita doble llamada. | `app.js` | 🐛 **Fix: "Cargando OpenCV..." infinito** |
| 54 | **BindingError "IntVector twice"**: `Promise.any()` con 2 URLs de CDN causaba que AMBOS scripts se ejecutaran (las promesas perdedoras NO se cancelan). Reemplazado por carga secuencial con IIFE (`tryCdn` + `tryNextCdn`), timeout de 10s por CDN, y fallback a unpkg solo si jsDelivr falla. Variable `loaded` evita doble disparo entre onerror y timeout. | `index.html` | 🐛 **Fix: doble carga WASM** |
| 55 | **CSP warning: `frame-ancestors` ignorado en `<meta>`**: la directiva solo funciona como HTTP header. Eliminada del meta tag en `index.html`. Se mantiene en `config.py` (`CSP_POLICY`) donde se envía como HTTP header vía `add_security_headers()`. Sin pérdida de protección anti-clickjacking. | `index.html`, `config.py` (sin cambios) | ⚠️ **Fix: warning eliminado** |

**Commit `bdeb46e`** + push a `origin/main` (`d72a1ae..bdeb46e main -> main`). **Build PyInstaller** exitoso: `dist/main/main.exe` (144MB, exit code 0). Logs del navegador limpios: `[OpenCV] Cargado exitosamente`, sin BindingError, sin CSP warnings.

#### Sesión 2026-07-28 — Safety timeout OpenCV + módulo signal + race condition Case 3 (3 fixes + commit + build)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 56 | **Safety timeout 25s en IIFE de OpenCV**: si el badge sigue en "loading" tras 25s, se actualiza automáticamente. Diagnóstico diferencial: si `window.__appJsLoaded` no está definido → "OpenCV Inactivo (Error Módulo)" con mensaje en consola (`app.js` nunca ejecutó por error de import). Si sí está definido → "OpenCV Inactivo" normal. Solo actúa si badge sigue en estado "loading". | `index.html` | 🛡️ **Fix: badge nunca se quedaba "Cargando..." para siempre** |
| 57 | **`window.__appJsLoaded = true` al inicio del módulo ES**: señal para que el IIFE detecte si `app.js` (ES module) ejecutó o si los imports fallaron (MIME type incorrecto, 404, etc.). Se setea sincrónicamente antes de cualquier async call. Si algún import falla, esta línea nunca se ejecuta. | `app.js` | 🛡️ **Diagnóstico: módulo vs CDN** |
| 58 | **Deferred 100ms check en `initOpenCv()` Caso 3**: mismo patrón que Caso 2. Cuando el polling encuentra `window.cv` sin `Mat`, asigna callback `onRuntimeInitialized` + check diferido `setTimeout(cvReadyCheck3, 100)`. Cubre race condition donde el WASM ya se inicializó antes de asignar el callback. Guard `!state.cvLoaded` evita doble llamada. | `app.js` | 🐛 **Fix: race condition en Caso 3** |

**Commit `3ed0924`** + push a `origin/main` (`bdeb46e..3ed0924 main -> main`). **Build PyInstaller** exitoso: `dist/main/main.exe` (138MB, exit code 0). Logs del navegador limpios: `[BOOT] app.js cargado`, `[OpenCV] Cargado exitosamente`, badge "OPENCV ACTIVO".

#### Sesión 2026-07-26-v2 — Refactor drawTextOnCanvas + glow + UI controles + tooltips (7 cambios)

| # | Cambio | Archivo | Impacto |
|---|--------|---------|---------|
| 46 | **drawTextOnCanvas() compartida**: nueva función que unifica la lógica de dibujo de texto (sombra, contorno con `strokeWidth * 2`, centrado vertical, glow, fillOpacity) entre `renderBoxes()` y `drawProfessionalText()`. ~45 líneas menos de código duplicado. Un solo punto de cambio para modificar estilos visuales. | `app.js` | **-45 líneas**, +mantenibilidad |
| 47 | **Glow exterior** (brillo neón): implementado con `shadowColor` + `fillStyle="transparent"` en posición real del texto (no off-screen). Se activa automáticamente para texto expresivo/impactante (mayúsculas, `!¡?¿`) fuera de burbujas. Color e intensidad configurables. Orden de dibujo por línea: relleno → glow → shadow → stroke → fill. | `app.js` | 🆕 Efecto visual |
| 48 | **Relleno semitransparente** (`fillOpacity`): fondo semitransparente detrás de cada línea para texto flotante sin burbuja (default 0.35). Mejora legibilidad sobre fondos complejos sin ocultar el arte. | `app.js` | 🆕 Efecto visual |
| 49 | **Panel UI 'Efectos de Texto'**: nuevo panel en sidebar con toggle glow, color picker, slider intensidad (0-30), slider fillOpacity (0-100%). Tooltips descriptivos en cada control y hint de teclado `G`. | `index.html`, `app.js`, `styles.css` | 🆕 Controles UI |
| 50 | **Range sliders estilizados**: `input[type="range"]` con thumb gradiente verde, hover/focus states, y `.range-value` display numérico. Estilo `kbd.hint` para hints de teclado. Clases `.has-active-glow`/`.has-active-fill` para feedback visual del panel. | `styles.css` | 🆕 Estilos sliders |
| 51 | **Preview hover glow + atajo G**: al pasar mouse sobre glowToggle, se muestra preview temporal del glow en el box seleccionado (mouseenter guarda estado, mouseleave restaura). Atajo de teclado **G** togglea glow con toast. Fix: `_hoverGlowPreview = null` en change handler evita que hover preview sobrescriba cambios permanentes. | `app.js` | 🆕 UX + Fix bug |
| 52 | **JS modularizado**: `app.js` ahora es un módulo ES6 que importa de `js/config.js` (CLIENT_CONFIG, fetchClientConfig), `js/theme.js` (initTheme, toggleTheme), `js/toast.js` (showToast), `js/filters.js` (filterPageBlocks, MARGIN_NOISE_PATTERNS, GLOBAL_NOISE_PATTERNS), `js/utils.js` (formatDuration, canvasToBase64, etc.). CSP actualizado con CDNs adicionales. Console.log de depuración reducidos. | `app.js`, `index.html`, `js/*.js`, `config.py` | 🧹 Modularización |

#### Sesiones anteriores (2026-07-25) — Aceleración GPU dual + profiling cProfile inline

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
#### SesiÃ³n 2026-08-13-v117 â€” MigraciÃ³n segura de confidence en SQLite (1 fix + test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 240 | **init_db() migra de forma idempotente text_blocks.confidence de INTEGER a FLOAT** cuando encuentra una base SQLite anterior: reconstruye solo la tabla dentro de una transacciÃ³n, copia las columnas del modelo, conserva los datos y restaura la tabla legacy si ocurre un error. PostgreSQL y esquemas ya correctos no se modifican. | models.py, tests/test_models.py | **Conserva la precisiÃ³n OCR decimal usada por scoring y revisiÃ³n de calidad al actualizar instalaciones existentes, sin reescritura ni cambio de datos fuera de text_blocks** |

**ValidaciÃ³n**: RED confirmado por importaciÃ³n ausente y por el conflicto transaccional inicial; prueba de migraciÃ³n y suite de modelos en verde (**3/3**). CI completo: **716/716 tests**, sintaxis Python/JS correcta, Bandit **0 HIGH/0 MEDIUM/6 LOW**, servidor/API/estÃ¡ticos correctos (371 MB). Se mantiene la advertencia esperada del corpus antiguo: 15.1% y 0% de metadatos semÃ¡nticos.
#### Sesion 2026-08-13-v118 - Fallback lexical para idiomas occidentales actuales (1 mejora + 1 test parametrizado)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 241 | **El fallback de idioma reconoce portugues, frances, aleman e italiano en bloques cortos** con dos marcadores conservadores o una palabra distintiva (Bonjour merci, Ich liebe dich, Ciao grazie, Eu amo voce, Danke, Merci, Grazie). _detect_language_robust prioriza esa evidencia cuando el texto es corto y el detector estadistico carece de contexto. | translator.py, tests/test_translator.py | **Reduce la seleccion del par CT2 equivocado en OCR breve/ruidoso y mejora la consistencia multilingue sin modelos nuevos, glosario del usuario ni falsos positivos por palabras ambiguas** |

**Validacion**: RED confirmado con 7 casos; pruebas de deteccion robusta y fallback **14/14**, suite de traduccion **135/135**. Esta ampliacion queda temporalmente desactivada por decision del usuario; el estado vigente es el de la sesion v119.

#### Sesion 2026-08-13-v119 - Desactivacion temporal de frances, aleman e italiano (1 ajuste de soporte + 4 pruebas)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 242 | **Se desactivan temporalmente FR/DE/IT de forma coherente**: se eliminan de LANGUAGES/UI/API, se rechaza el selector occidental historico `eng+spa+fra+deu`, la deteccion automatica nunca devuelve esos codigos, `_translate_one()` conserva el texto si recibe uno de ellos directamente y EasyOCR latino queda en ES/EN/PT. Los modelos CT2 de esos pares se conservan en disco para una futura reactivacion. | config.py, routes/api.py, translator.py, ocr_utils.py, index.html, app.js | **Evita que los modelos y la deteccion mezclen idiomas pausados con los activos, reduce trabajo del recognizer OCR y mantiene una ruta de reactivacion sin reinstalacion** |

**Validacion**: pruebas focalizadas **444/444** en verde. CI completo **722/722 tests**, sintaxis Python/JS correcta, servidor/API/estáticos correctos, Bandit **0 HIGH/0 MEDIUM/6 LOW**. Se mantiene la advertencia esperada del corpus antiguo/parcial: 15.1% y 0% de metadatos; el stress test de memoria se omite con `--skip-cov`.

#### Sesion 2026-08-13-v120 - Solapamiento seguro del OCR hibrido (1 optimizacion + 2 tests)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 243 | **RapidOCR se precarga junto con EasyOCR/CT2 y su trabajo CPU se inicia en paralelo a la inferencia EasyOCR GPU**. Se conserva el semaforo de RapidOCR, la fusion, los umbrales, los fallbacks y la degradacion U-OCR; solo se elimina espera secuencial entre motores. | server.py, ocr_utils.py, tests/test_ocr_utils.py, tests/test_packaging.py | **Menor tiempo por pagina sin consumir VRAM adicional ni cambiar los bloques fusionados. En paginas de referencia ya calientes: 5.45 -> 4.13 s, 4.74 -> 3.92 s y 4.36 -> 3.50 s; cobertura observada sin cambios: 7, 9 y 4 bloques.** |

**Validacion**: prueba RED/GREEN del solapamiento, OCR **158/158**, CI rapido y completo **724/724 tests**, sintaxis Python/JS correcta, servidor/API/estaticos correctos y Bandit **0 HIGH/0 MEDIUM/6 LOW**. El stress procesó **50/50 paginas** y mantuvo memoria estable (~147.5 MB al final, sin crecimiento progresivo); el parser del CI no reconoció su resumen y lo reportó como advertencia. Se mantiene la advertencia conocida del corpus antiguo/parcial: 15.1% de aceptacion y 0% de metadatos.
#### Sesion 2026-08-13-v121 - Solapamiento de inpainting y traduccion (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 244 | **El inpainting y la codificacion de la imagen inpainted se ejecutan en un hilo mientras el executor traduce los bloques**. Antes ambos tramos eran secuenciales. Se espera el resultado antes de muestrear colores o responder; timeout y fallback conservan la imagen original. | routes/api.py, tests/test_api.py | **Reduce la latencia post-OCR sin cambiar bloques, traducciones, coordenadas, colores ni el contrato de la API. La concurrencia queda limitada a un hilo de inpainting por pagina y no usa VRAM adicional.** |

**Validacion**: pruebas RED/GREEN de solapamiento y metrica, suite API **152/152**, CI completo **726/726 tests**, sintaxis Python/JS correcta, servidor/API/estaticos correctos, Bandit **0 HIGH/0 MEDIUM/6 LOW** y stress **50/50 paginas** sin errores ni leak detectado (promedio 4.2 s/pagina, +70.7 MB de memoria estable). Se mantiene la advertencia del corpus antiguo/parcial: 15.1% de aceptacion y 0% de metadatos.

#### Sesion 2026-08-13-v122 - Eliminacion de render PDF duplicado en traduccion masiva (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 245 | **`autoTranslateAllPages()` deja de renderizar cada pagina antes de llamar a `autoTranslateCurrentPage()`**. `serverProcessPage()` ya hace el render limpio obligatorio inmediatamente antes de codificar y enviar la imagen; se conserva ese unico render canonico y el retry existente. | app.js, tests/test_packaging.py | **Evita un render PDF completo redundante por pagina en traducciones masivas, reduciendo CPU/tiempo de interfaz sin cambiar la imagen enviada, OCR, traduccion ni exportacion.** |

**Validacion**: prueba RED/GREEN del flujo masivo, `node --check app.js` correcto y CI rapido **727/727 tests**, servidor/API/estaticos correctos, Bandit **0 HIGH/0 MEDIUM/6 LOW**. El stress completo queda cubierto por la validacion v121; se mantiene la advertencia conocida del corpus antiguo/parcial.

#### Sesion 2026-08-13-v123 - Eliminacion de copia de canvas antes de base64 (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 246 | **`serverProcessPage()` codifica directamente `cleanBgCanvas` y elimina el canvas auxiliar, `drawImage()` y la asignacion duplicada de todos los pixeles**. `toDataURL()` es sincrono y captura el contenido antes de cualquier suspension del flujo. | app.js, tests/test_packaging.py | **Menor CPU, memoria temporal y latencia por pagina en el navegador, sin cambiar el PNG enviado ni la respuesta del servidor.** |

**Validacion**: prueba RED/GREEN, `node --check app.js` correcto y CI rapido **728/728 tests**, servidor/API/estaticos correctos, Bandit **0 HIGH/0 MEDIUM/6 LOW**. El stress completo sigue cubierto por la validacion v121; se mantiene la advertencia conocida del corpus antiguo/parcial.

#### Sesion 2026-08-13-v124 - Conversion BGR diferida tras semaforo EasyOCR (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 247 | **`_run_ocr_on_image()` convierte BGR a RGB solo despues de adquirir el semaforo OCR y el lock GPU**. Las llamadas que esperan, expiran o se degradan a RapidOCR no crean antes una copia RGB innecesaria. | ocr_utils.py, tests/test_ocr_utils.py | **Reduce copias temporales y CPU bajo concurrencia, especialmente en GTX 1050 Ti cuando varios workers esperan el unico lector EasyOCR; los parametros y resultados de OCR permanecen iguales.** |

**Validacion**: prueba RED/GREEN y suite OCR **159/159**. CI rapido **729/729 tests**, sintaxis Python/JS correcta, servidor/API/estaticos correctos y Bandit **0 HIGH/0 MEDIUM/6 LOW**. El stress completo sigue cubierto por la validacion v121; se mantiene la advertencia conocida del corpus antiguo/parcial.

#### Sesion 2026-08-13-v125 - Precarga del corrector ortografico OCR (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 248 | **`_get_spellchecker()` se ejecuta durante la precarga de servidor, despues de RapidOCR y antes de YOLO**. Si la dependencia no esta disponible, se conserva el fallback existente y el servidor continua arrancando. | server.py, tests/test_packaging.py | **La primera pagina no paga la carga del diccionario de ~86K palabras; no consume VRAM ni cambia el postprocesado OCR.** |

**Validacion**: prueba RED/GREEN y CI rapido **730/730 tests**, sintaxis Python/JS correcta, servidor/API/estaticos correctos y Bandit **0 HIGH/0 MEDIUM/6 LOW**. El stress completo sigue cubierto por la validacion v121; se mantiene la advertencia conocida del corpus antiguo/parcial.

#### Sesion 2026-08-13-v126 - Reutilizacion del pre-filtro para RapidOCR (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 249 | **`_preprocess_rapid()` acepta una imagen ya prefiltrada y `_detect_and_ocr()` la reutiliza en el hilo de RapidOCR**. EasyOCR y RapidOCR ya no ejecutan dos veces la misma limpieza morfologica cuando `prefilter=True`; callers con imagen cruda conservan el comportamiento anterior. | ocr_utils.py, tests/test_ocr_utils.py | **Menos CPU/copia temporal por pagina sin cambiar realce, parametros, fusion ni resultados OCR.** |

#### Sesion 2026-08-13-v127 - Cache LRU de deteccion de idioma (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 250 | **`_detect_language_robust()` usa una LRU de 4096 textos**. Las frases repetidas entre bloques/paginas no vuelven a ejecutar `langdetect`; el resultado determinista y los fallbacks conservadores se mantienen. | translator.py, tests/test_translator.py | **Menos CPU y mas consistencia en paginas con nombres/frases repetidas, con memoria acotada y sin modelos nuevos.** |

#### Sesion 2026-08-13-v128 - Gate de ruido OCR antes de Google (1 mejora de calidad + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 251 | **Si CT2 no produce una salida valida y el bloque es ruido OCR evidente, `_translate_one()` conserva el texto y no activa Google**. El gate excluye fuentes CJK para no romper el pivote ja/ko/zh -> en -> es; el texto valido y el camino CT2 no cambian. | translator.py, tests/test_translator.py | **Evita falsas traducciones de fragmentos rotos y elimina una peticion de red innecesaria por bloque.** |

#### Sesion 2026-08-13-v129 - Read-through en memoria para cache de traducciones (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 252 | **La cache persistente mantiene un read-through LRU acotado a 5000 entradas y 2 MB de texto**. Los hits repetidos se sirven en memoria; escrituras atomicas, TTL, eviccion y `clear()` siguen sincronizados con disco. | cache.py, tests/test_cache.py | **Reduce aperturas/parseos JSON en frases repetidas entre paginas sin modificar traducciones ni eliminar persistencia ni introducir presion de RAM.** |

**Validacion**: pruebas RED/GREEN de las cuatro optimizaciones; suites focalizadas **447/447**, CI rapido y completo **734/734 tests**, sintaxis Python/JS correcta, servidor/API/estaticos correctos, Bandit **0 HIGH/0 MEDIUM/6 LOW** y stress **50/50 paginas**, promedio **2.9s/pagina**, memoria **+71.1MB estable**, sin leak detectado. Se mantiene la advertencia conocida del corpus antiguo/parcial: 15.1% de aceptacion y 0% de metadatos.

#### Sesion 2026-08-13-v130 - Miss de cache sin syscall `exists()` redundante (1 optimizacion + 1 test)

| # | Cambio | Archivo | Impacto |
|---|---|---|---|
| 253 | **`cache.get()` intenta leer directamente el JSON y trata `FileNotFoundError` como miss**, eliminando el `exists()` previo. | cache.py, tests/test_cache.py | **Reduce una llamada al sistema por cada texto nuevo/no cacheado sin cambiar TTL, validacion, persistencia ni el resultado de los hits.** |

**Validacion adicional**: cache **4/4** en verde; CI rapido **735/735 tests**, sintaxis Python/JS correcta, servidor/API/estaticos correctos y Bandit **0 HIGH/0 MEDIUM/6 LOW**. La advertencia del corpus antiguo/parcial permanece sin cambios.
