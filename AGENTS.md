> ⚠️ **LEE ESTE ARCHIVO ANTES DE MODIFICAR CUALQUIER ARCHIVO DEL PROYECTO.**

# AGENTS.md — Guía de contexto para IAs y colaboradores

## 1. Descripción del proyecto

**Traductor Visual Pro** — Aplicación local para traducir manga, cómics y documentos en PDF e imagen.

### Arquitectura actual (post-refactor Julio 2026)

| Archivo | Rol | Tamaño |
|---|---|---|
| `app.js` | Frontend completo (~2767 líneas): renderizado PDF/imagen vía pdf.js, editor de burbujas (draw/move/resize), OCR delegado al servidor EasyOCR, filtros de bloques, layout de texto en canvas (wrap+fit), comunicación con API Flask, exportación PNG/PDF, tema oscuro/claro, atajos de teclado, toasts, carga asíncrona de OpenCV.js vía callback. | 101KB |
| `server.py` | Entry point Flask (~188 líneas). Importa de config.py/translator.py, envuelve _translate_one() con caché, registra blueprints. Puerto 5174. | 7KB |
| `config.py` | Constantes globales: paths, `LANGUAGES`, `CSP_POLICY`, patrones de ruido (`MARGIN_NOISE_PATTERNS`, `WATERMARK_PATTERNS`), glosario pre-OCR (`GLOSARIO_PRE`). | 6KB |
| `translator.py` | Lógica de traducción: detección de idioma (`_detect_language_robust`), 3 motores en **paralelo** (CT2 CTranslate2 int8 → ArgosTranslate → Google Translate), validación de traducción (6 validaciones anti-basura), glosarios PRE/POST, filtro pre-Argos para OCR noise. **Google retry con backoff** (5s, 15s, 30s) cuando todos los motores fallan (fix SIN_TRAD). Cache injectado desde server.py. | 28KB |
| `ocr_utils.py` | OCR con EasyOCR (GPU→CPU fallback), pre-filtro de imagen, inpainting con OpenCV (INPAINT_NS), detección de globos de diálogo, máscara de glifos, sampleo de color, fusión y filtrado de bloques (9 filtros post-merge). **3 niveles de fallback**: directo → CLAHE+sharpen → CTD (ComicTextDetector). **Modo CTD-only** (`use_ctd_only=True`) salta EasyOCR completo para páginas donde EasyOCR consistentemente falla. **Optimizado**: canvas_size=2500px, text_threshold=0.18, mag_ratio=1.2. | 24KB |
| `routes/api.py` | Blueprint REST: `/api/health`, `/api/translate`, `/api/translate-batch`, `/api/process-page`. **Expone `ocr_mode`** en `/api/process-page`: `"auto"` (3 niveles), `"easyocr"` (solo EasyOCR), `"ctd"` (solo CTD). Importa directamente de los submódulos. | 11KB |
| `routes/main.py` | Blueprint de rutas estáticas con protección contra path traversal. | 1KB |
| `ocr_ctd_fallback.py` | Fallback OCR con **CTD (ComicTextDetector)** para texto artístico. Detecta regiones de texto vía modelo ConvNeXt (76MB), luego EasyOCR reconoce en regiones recortadas. **Filtro post-detección**: área mínima 400px², altura mín 8px, aspect ratio 0.4-20, máximo 15 regiones. Usado como tier 3 en modo `auto`, o como único motor en modo `ctd`. Incluye `preload_ctd()` para carga lazy thread-safe. | 13KB |
| `index.html` | UI HTML (~339 líneas): estructura del editor visual, CSP vía `<meta>`, detección de Brave Leo/Shields. | 17KB |
| `styles.css` | Estilos visuales premium (tema dark/light con variables CSS, glassmorphism, animaciones, responsive). | 40KB |
| `cache.py` | Caché de traducciones en filesystem con TTL (7 días) y LRU eviction (5000 entradas máx). | 2KB |
| `models.py` | Modelos SQLAlchemy (User, Project, Page, TextBlock) con repositorios. SQLite local / PostgreSQL producción. | 7KB |
| `ratelimit.py` | Rate limiting con Flask-Limiter. Evita imports circulares entre server.py y routes/. | 1KB |
| `start-app.ps1` | Lanzador: inicia `env\Scripts\python.exe server.py` y abre `http://127.0.0.1:5174`. | 1KB |
| `run_ci.py` | **CI unificado** — ejecuta syntax check + test_ci.py + servidor + analisis_calidad + stress test en un solo comando Python. No depende de PowerShell. `python run_ci.py --full` para test completo (~10 min). | 20KB |
| `requirements.txt` | Dependencias Python pineadas. | 1KB |
| `main.py` | Entry point del ejecutable (.exe). Modo launcher: oculta consola, inicia Flask, abre Chrome. Modo `--server`: solo servidor. | 4KB |
| `main.spec` | PyInstaller spec para compilar `main.exe` con `console=False` (sin ventana CMD). Incluye CTD model como DATA. | 3KB |
| `launcher.py` | Launcher alternativo (subprocess). Usa `env\Scripts\python.exe server.py` como proceso hijo. | 1KB |
| `env/` | Entorno virtual Python con **todas** las dependencias (EasyOCR, OpenCV, Flask, ArgosTranslate, deep-translator, langdetect, torch). | — |

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
| `_translate_one()` | ~95 | Wrapper que inyecta `cache_get`/`cache_set` en `translator._translate_one()`. **Debe importarse desde `server` en routes/api.py** — NO desde `translator.py` directo. |
| `from config import ...` | ~12 | Importaciones controladas. No agregar imports circulares. |

### `translator.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_translate_one()` | ~185 | **3 motores en paralelo** (CT2 CTranslate2 int8 → Argos → Google) lanzados via `ThreadPoolExecutor` + `as_completed`. Acepta el PRIMER resultado válido inmediatamente. Fallback: si ningún motor pasa validación, usa el primero que devolvió algo. Acepta `cache_get`/`cache_set`/`translation_cache_available` para inyección. |
| `_get_google_session()` | ~58 | Double-checked locking con `_google_session_lock`. La sesión HTTP se crea **dentro del lock**. |
| `_detect_language_robust()` | ~96 | langdetect thread-local + heurística `_detect_language_simple`. Mapeo zh-cn/zh-tw → zh. |
| `_ensure_argo_package()` | ~20 | Descarga modelos Argos con lock para evitar descargas duplicadas. |

### `ocr_utils.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_get_ocr_reader()` | ~20 | Lazy-loading EasyOCR con `threading.Lock()`. GPU→CPU fallback. No cargar en hilo secundario en Windows. |
| `_detect_and_ocr()` | ~160 | Parámetros actuales: `text_threshold=0.18`, `low_text=0.12`, `min_size=8`, `mag_ratio=1.2`, `canvas_size=2500`. 3 niveles de fallback: directo → CLAHE+sharpen → CTD. Acepta `use_ctd_only` (salta EasyOCR en imagen completa) y `allow_fallback` (desactiva fallbacks en modo easyocr-only). |
| `_group_and_merge_blocks()` | ~260 | Aplica 9 filtros post-merge: números puros, patrones numéricos, comillas, puntuación suelta, aspecto estrecho, chars sueltos, baja confianza, dígito+letra. Fusión horizontal con gap tolerante `max(35, w*2.5)`. |
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

**Última actualización**: 2026-07-22

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
| 25 | **Procesamiento de páginas en paralelo**: `process_all_pages.py` reescrito con arquitectura productor-consumidor (1 render thread + N API workers). Checkpoint cada 10 páginas thread-safe. Tiempo 128 páginas: ~75-100 min → ~20-35 min. N=2 por defecto (--workers CLI) para evitar saturar semáforo OCR. | `process_all_pages.py` | **3x** total |
| 26 | **Bug fix**: `#` literal antes de URL en API call (`f"#{API_URL}/api/process-page"`) invalidaba TODAS las llamadas — 0 páginas procesadas. Corregido a `f"{API_URL}/api/process-page"`. También eliminado `import threading` duplicado. | `process_all_pages.py` | Bug crítico ✅ |
| 27 | **Bug fix `is_lenient`**: cuando un motor devuelve `trad == orig` en modo lenient (≤3 palabras), ahora verifica con `_detect_language_robust()`. Si el texto sigue en idioma origen → rechazar y probar siguiente motor. Si ya está en destino → aceptar. Antes aceptaba cualquier resultado idéntico. | `translator.py` | Bug crítico ✅ |
| 28 | **Modo CTD-only**: nuevo parámetro `ocr_mode` en `/api/process-page` (`auto`/`easyocr`/`ctd`). `use_ctd_only=True` en `_detect_and_ocr()` salta EasyOCR de imagen completa y va directo a CTD. `allow_fallback=False` en modo `easyocr` desactiva fallbacks. Ahorra ~1-2s en páginas donde EasyOCR consistentemente falla. | `ocr_utils.py`, `routes/api.py` | 🆕 Modo CTD |
| 29 | **Sincronización de patrones app.js ↔ config.py**: eliminados `olympus|scanlation|zonaolympus|scan_group` de `GLOBAL_NOISE_PATTERNS` y `capítulo|cómo criar|how to raise` de `MARGIN_NOISE_PATTERNS` en `app.js` para coincidir con `config.py`. Verificado con `node --check`. | `app.js`, `config.py` | 🆕 Sincronizado |
| 30 | **Fix SIN_TRAD — Google retry con backoff**: cuando todos los motores (CT2, Argos, Google) fallan en paralelo, se reintenta Google 3 veces con backoff progresivo (5s, 15s, 30s), reseteando el rate limit entre intentos. También se eliminaron 2 líneas de código muerto duplicadas al final de `_translate_one()`. | `translator.py` | 🆕 Fix SIN_TRAD |
| 31 | **Métrica de calidad real**: análisis de 723 bloques traducidos muestra 75.8% aceptable (BUENA + LITERAL + OCR_NOISY), 15.9% OCR_GARBAGE (running headers principalmente), 8.3% UNTRANSLATED. Documentado en CODEGRAPH.md como referencia de calidad. | `CODEGRAPH.md` | 📊 Benchmark real |
| 32 | **Fix running headers**: el symbol cleaning (`re.sub` removiendo `/`, `.`, `,`, `:`) se ejecutaba ANTES de verificar MARGIN_NOISE_PATTERNS, destruyendo patrones de fecha/hora como "13/7/26" y "4.58 p.m". Movido el filtro ANTES del cleaning. | `ocr_utils.py` | 🐛 Fix bug crítico |

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

### Qué verifica

| Paso | Test | Qué comprueba |
|---|---|---|
| 1 | Syntax check | `py_compile` en todos los Python files |
| 2 | `test_ci.py` | Detección de idioma (langdetect + heurísticas) |
| 3 | Iniciar servidor | `/api/health` + endpoints translate, batch, config, static |
| 4 | `analisis_calidad.py` | Calidad de traducciones vs corpus de referencia |
| 5 (opcional) | `stress_test_memory.py` | 50 páginas bajo carga (con `--full`) |

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
- **Siempre** antes de hacer commit
- **Con `--full`** antes de un release o después de cambios grandes en el pipeline de traducción

### Si el CI falla

1. Revisa el paso que falló (syntax, tests, servidor)
2. Para errores de sintaxis: `python -m py_compile <archivo>` te da la línea exacta
3. Para error del servidor: revisa `ci_server.log` en la raíz del proyecto
4. Si `test_ci.py` falla: revisa `_detect_language_simple()` o `_detect_language_robust()` en `translator.py`

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
