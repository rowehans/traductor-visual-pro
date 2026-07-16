> ⚠️ **LEE ESTE ARCHIVO ANTES DE TOCAR `app.js` O `server.py`.**

# AGENTS.md — Guía de contexto para IAs y colaboradores

## 1. Descripción del proyecto

**Traductor Visual Pro** — Aplicación local para traducir manga, cómics y documentos en PDF e imagen.

| Archivo | Rol |
|---|---|
| `app.js` | Frontend completo (~2230 líneas): renderizado PDF/imagen vía pdf.js, editor de burbujas (draw/move/resize), OCR delegado al servidor EasyOCR, filtros de bloques, layout de texto en canvas (wrap+fit), comunicación con API Flask, exportación PNG/PDF, tema oscuro/claro, atajos de teclado, toasts. |
| `server.py` | Backend Flask (puerto 5174, ~770 líneas): OCR con EasyOCR + GPU→CPU fallback, inpainting con OpenCV (INPAINT_NS), traducción con ArgosTranslate + Google fallback, filtros de ruido/marcas de agua, batch translation y process-page endpoint. |
| `index.html` | UI HTML (~290 líneas): estructura del editor visual, CSP via `<meta>`, detección de Brave Leo/Shields. |
| `styles.css` | Estilos visuales (tema dark/light con variables CSS). |
| `start-app.bat` / `start-app.ps1` | Launchers: inician `env\Scripts\python.exe server.py` y abren `http://127.0.0.1:5174`. |
| `env/` | Entorno virtual Python con **todas** las dependencias (EasyOCR, OpenCV, Flask, ArgosTranslate, deep-translator, langdetect, torch). `.venv/` es un venv **vacío/incompleto** — siempre usar `env/`. |
| `requirements.txt` | Dependencias Python pineadas. |

---

## 2. Zonas Sensibles ⛔ — No tocar sin entender primero por qué están así

### `app.js`

| Función / Variable | Línea aprox. | Por qué es delicada |
|---|---|---|
| `loadPdfJs()` | ~470 | **Carga dual de PDF.js**: intenta primero `import()` de `.mjs` (v4.10.38, más rápido), y si falla (CORS/Brave/navegador viejo), carga script UMD clásico `pdf.js` v3.11.174 como fallback. Ambos métodos configuran `GlobalWorkerOptions.workerSrc` inmediatamente. Cambiar el orden de fallback o quitar una estrategia puede romper la carga de PDF en ciertos navegadores. |
| `wrapTextLines(ctx, text, maxWidth)` | ~1457 | **Bug histórico crítico**: si la condición `containsCJK()` cambia, las palabras occidentales vuelven a separarse letra por letra (ej. "C O R R E C T L Y"). Solo separa carácter a carácter si el texto contiene escritura CJK. Latino siempre por palabras completas. |
| `containsCJK(text)` | ~1433 | Rango Unicode expandido para detectar CJK (hiragana, katakana, hangul, hanzi, fullwidth). Si se reduce, textos japonés/coreano/chino se rompen por palabra. |
| `fitTextLayout(text, box)` | ~1377 | Depende de `wrapTextLines`. Ajusta tamaño decreciente hasta que el texto quepa verticalmente (maxWidth = 75% del ancho, padding generoso para burbujas manga). |
| `MARGIN_NOISE_PATTERNS` / `GLOBAL_NOISE_PATTERNS` | ~809–833 | Regex calibradas para detectar cabeceras/pies impresos por navegador (fecha, hora, numeración "3/128") y sellos de escaneo (ZonaOlympus, "1 C 2 E", olympus). Tocar sin probar puede filtrar diálogo real o dejar metadatos. |
| `filterPageBlocks(blocks, pageHeight)` | ~842 | Único punto que filtra bloques antes de crear cajas. Se aplica a PDF nativo, OCR local y resultados del servidor. `marginTop` = 5% de altura. |
| `state.inpaintedBgByPage` (Map) | ~119, ~1211 | Caché de imágenes inpaintadas por página. Si no se escribe aquí tras `serverProcessPage()`, el texto original reaparece al interactuar con el canvas. |
| `updateErasedBg()` | ~616 | Lee `inpaintedBgByPage` primero. Si se cambia el orden, el fondo inpaintado se sobrescribe con la imagen original. Si `hasServerInpainted` es true, **no hace borrado local** (evita doble inpainting). |
| `makeAutoTextBox(block, translated, serverData)` | ~1012 | Crea objetos de caja. Con `serverData` usa Comic Sans MS + peso 400 + estilo normal (look manga profesional). Sin serverData usa UI state (italic/bold configurable). `eraseMode: "none"` si server ya borró. |
| `renderBoxes()` | ~1276 | Dibuja texto directamente en canvas con clipping, sombra, centrado vertical y stroke. El overlay div es transparente (solo maneja clicks). Z-order: fondo → shadow → stroke → fill. |
| `drawProfessionalText(ctx, text, box)` | ~1822 | Versión "profesional" para exportación. Debe **coincidir visualmente** con `renderBoxes()` (misma sombra, mismo centrado, mismo `lineWidth * 2`). |
| `eraseWithInpainting(canvas, boxes)` | ~1661 | Máscara OpenCV con rectángulos + `inset 3%`. Radio `15` con `INPAINT_TELEA`. `Mat.delete()` en `finally` obligatorio (memoria WASM). |
| `markGlyphMask(cv, sourceCanvas, mask, box)` | ~1715 | Detección de glifos por contraste de color (`threshold = 40` vs fondo de borde). Cambiar umbral marca demasiados o muy pocos píxeles. |
| `fallbackEraseBox(ctx, box)` | ~1769 | Borrado básico. `sampleEraseColor` muestrea SIEMPRE de `cleanBgCanvas`, nunca de `pdfCanvas`. Usa borde exterior 10% con mediana de luminosidad. |
| `renderEditedCanvas(pageNo, rawCanvas)` | ~1866 | Canvas de exportación. Orden: (1) copiar raw canvas, (2) inpainting/fallback, (3) drawProfessionalText. Invertir = texto original tapa al traducido. |
| `autoTranslateCurrentPage(pageNo, ...)` | ~1176 | **Camino único**: servidor. `serverProcessPage()` → guarda `inpaintedBgByPage` → carga `erasedBgCanvas` → filtra bloques → `makeAutoTextBox` con serverData → render. Si se salta algún paso, el canvas queda inconsistente. |
| `serverProcessPage(pageNo)` | ~1131 | Envía `cleanBgCanvas` como base64 a `/api/process-page`. Timeout 120s. Devuelve `{inpainted_image, blocks}`. |
| `state.boxesByPage` (Map) | ~105 | Caché de cajas por página. `boxesByPage.set(page, array)` es el único método seguro. `getPageBoxes()` inicializa array vacío si no existe. |
| `initTheme()` / `toggleTheme()` | ~148–211 | Tema dark/light con `data-theme` en `<html>` y localStorage. Afecta variables CSS globales. |
| `initKeyboardShortcuts()` | ~306 | Atajos: D=dibujar, V=mover, Ctrl+T=traducir, Ctrl+E=exportar PNG, Ctrl+Shift+P=PDF página, flechas=navegar, Supr=borrar burbuja. |
| `canvasToBase64(canvas)` / `loadBase64IntoCanvas(b64, targetCanvas)` | ~1111–1128 | Conversión bidireccional canvas↔base64. Usado por serverProcessPage y para cargar inpainted_image. |
| `openFile(file)` | ~505 | **Case-insensitive**: usa `/\.pdf$/i.test(file.name)`. Logging detallado en consola. Limpia `inpaintedBgByPage` y `boxesByPage` al cambiar archivo. |
| `exportFullPdf()` | ~1946 | Genera PDF completo página por página: renderRawPdfPage → renderEditedCanvas → jsPDF.addPage. |

### `server.py`

| Función | Línea aprox. | Por qué es delicada |
|---|---|---|
| `_get_ocr_reader(lang)` | ~80 | Lazy-loading de EasyOCR con `threading.Lock()`. GPU→CPU fallback automático. Cargar EasyOCR en hilo secundario en Windows causa crash de cuDNN. No mover la carga. |
| `_detect_and_ocr(img_bgr, lang_hint)` | ~322 | Parámetros calibrados: `text_threshold=0.45`, `min_conf=0.25` (línea 355), `mag_ratio=1.5`. Subir `text_threshold` = textos estilizados invisibles. |
| `_group_and_merge_blocks(blocks, img_h)` | ~419 | Aplica `_WATERMARK_PATTERNS` (global), `_MARGIN_NOISE_PATTERNS` (solo 5% márgenes), URLs, números sueltos, ruido estrecho (aspect < 0.4). Fusión SOLO horizontal. |
| `_WATERMARK_PATTERNS` / `_MARGIN_NOISE_PATTERNS` | ~393–416 | Patrones compilados con `re.compile()`. Deben estar sincronizados con `GLOBAL_NOISE_PATTERNS`/`MARGIN_NOISE_PATTERNS` de app.js. |
| `_build_inpaint_mask(img_bgr, blocks)` | ~524 | Máscara binaria con rectángulos completos + padding 8%/12% + dilatación 5x5. No marcar píxeles individuales (causa ghosting). |
| `_inpaint_image(img_bgr, mask, blocks)` | ~554 | Radio adaptativo (`avg_height * 0.6`, clip 5–30). Usa `INPAINT_NS` (Navier-Stokes, mejor para tramas/texturas manga). |
| `_translate_one(text, source, target)` | ~263 | Pipeline: detectar idioma → ArgosTranslate directo → pivoteo `src→en→es` (doble paso Argos) → Google fallback. Si se simplifica, pierde traducción offline. |
| `_detect_language_robust(text)` | ~243 | `langdetect` thread-local + heurística `_detect_language_simple` con diccionario español. Mapeo `zh-cn/zh-tw → zh`. |
| `_ensure_argo_package(src, tgt)` | ~157 | `_argo_lock` evita descargas duplicadas del mismo modelo. Sin lock = race condition en disco/memoria. |
| `process_page()` endpoint `/api/process-page` | ~672 | **Endpoint principal**. Orden: decodificar → OCR → detección idioma → máscara → inpainting → traducción batch paralela → muestrear bgColor → armar respuesta `{inpainted_image, blocks}`. Cambiar orden = resultados inconsistentes. |
| `translate_batch()` endpoint `/api/translate-batch` | ~645 | Traducción paralela con `ThreadPoolExecutor` compartido. `as_completed()` respeta orden original. |
| `_cv2_to_base64(img, fmt)` | ~314 | PNG compresión 3 (`IMWRITE_PNG_COMPRESSION, 3`). Cambiar compresión afecta calidad de inpainted_image. |
| `CSP_POLICY` / `add_security_headers()` | ~53–74 | CSP inyectado en TODAS las responses. Permite CDNs, data:, blob:, `connect-src http://127.0.0.1:5174`. |
| `_sample_bg_color(img_bgr, block)` | ~584 | Muestrea franjas superior/inferior del bloque (fuera del texto). Devuelve hex RGB. |
| `_thread_local` langdetect detector | ~145–154 | Thread-local para evitar race conditions en `_detect_language_robust()` con múltiples hilos. `DetectorFactory.seed = 0`. |

---

### Secciones sincronizadas (app.js ↔ server.py)

| Componente sincronizado | Por qué es delicado |
|---|---|
| `filterPageBlocks` (app.js ~842) + `_group_and_merge_blocks` (server.py ~419) | Ambos implementan mismos filtros de margen/marcas de agua. Divergencia = comportamiento impredecible según camino. |
| `MARGIN_NOISE_PATTERNS` (app.js ~809) + `_MARGIN_NOISE_PATTERNS` (server.py ~393) | Deben ser idénticos. Si divergen: texto basura traducido o diálogo legítimo eliminado. |
| `GLOBAL_NOISE_PATTERNS` (app.js ~828) + `_WATERMARK_PATTERNS` (server.py ~412) | Misma razón: sincronización obligatoria. |
| `state.inpaintedBgByPage` (app.js ~119) + `inpainted_image` del servidor (server.py ~753) | Servidor devuelve base64 PNG; frontend lo convierte a `Image` y lo guarda en Map. Cambiar formato rompe renderizado. |
| `makeAutoTextBox` `eraseMode: "none"` (app.js ~1055) vs inpainting del servidor | Si frontend repite inpainting sobre imagen ya inpaintada, degrada el fondo. |

## 3. Zonas Seguras de Editar Libremente ✅

- **`styles.css`**: Colores, animaciones, variables CSS de tema. No afecta lógica.
- **`index.html`**: Añadir botones/campos si no cambian IDs usados por `app.js`.
- **Endpoints `/api/health`, `/api/translate`, `/api/translate-batch`** en `server.py`: Ampliar sin impacto en flujo principal.
- **`setStatus()`, `showProgress()`, `formatDuration()`** en `app.js`: Solo UI de estado/progreso.
- **`start-app.bat` / `start-app.ps1`**: Puerto, URL de apertura, browser.
- **`_preload_models()`** en `server.py`: Añadir pares de idioma para precalentar.
- **`sampleBgColorAround()`, `sampleTextColor()`** en `app.js`: Mejorar muestreo de colores para contraste.
- **`initTheme()` / `toggleTheme()`**: Añadir variantes de tema.
- **`showToast()`**: Cambiar duración, estilos, animaciones.
- **`initKeyboardShortcuts()`**: Añadir/quitar atajos (Ctrl+letra o teclas simples).
- **Diccionario `spa_words` en `_detect_language_simple()`**: Añadir palabras spanish.

---

## 4. Estado Actual / Últimos Cambios

**Fecha**: 2026-07-14

### Cambios de esta sesión (2026-07-15)

- **MIT integration**: Nuevo pipeline Manga-Image-Translator en `manga_pipeline.py`.
  - **Detección de texto**: EasyOCR reemplazado por **CTD** (Comic Text Detector) — red neuronal entrenada en manga, detecta texto estilizado (gótico, terror, rasgado) que EasyOCR perdía.
  - **OCR**: Modelo **ocr_48px** entrenado en manga (vs EasyOCR genérico). Mejor precisión con tipografías artísticas.
  - **Inpainting**: OpenCV `INPAINT_NS` reemplazado por **LaMa** fine-tuned en manga/anime — preserva la forma de globos de diálogo y texturas de trama. Sin rectángulos grises.
  - **Máscara inteligente**: `_is_inside_speech_bubble()` detecta fondos oscuros/uniformes → usa máscara solo-glifos que preserva el globo completo.
  - **Fallback automático**: si MIT no está disponible, cae al sistema legacy (EasyOCR + OpenCV sin cambios).

- **Nuevo archivo**: `manga_pipeline.py` (~200 líneas) — wrapper síncrono sobre módulos de MIT.
  - Importa solo los submódulos necesarios (detection, ocr, textline_merge, inpainting).
  - `ensure_ready()` descarga modelos (~550 MB: CTD, OCR 48px, LaMa) al primer uso.
  - `run_pipeline(img_bgr)` → `{inpainted_image: b64, blocks: [...]}`.

- **`server.py`**:
  - Línea 17: Import condicional de `manga_pipeline`. `MIT_AVAILABLE` flag.
  - `process_page()` (~858): si MIT disponible → `run_pipeline()`; si falla o no disponible → legacy EasyOCR + OpenCV.
  - `_is_inside_speech_bubble()`, `_build_glyph_mask_for_bubble()`: nuevas funciones para preservar globos oscuros incluso en legacy.
  - `_sample_bg_color()`: para bloques dentro de globos, muestrea borde interior del perímetro (negro real) en vez de franjas externas (arte rojo/sombreado).

- **`app.js`**:
  - `makeAutoTextBox()` (~1141): si `bgColor` del servidor es muy oscuro (brillo < 60), usa `bg: "transparent"` para que el canvas inpainted se vea a través.

- **Dependencias nuevas**: transformers, huggingface_hub, einops, kornia, manga-ocr, py3langid, shapely, pyclipper, omegaconf, rusty-manga-image-translator (~200MB).
- **Modelos descargados** (~550MB): CTD detector, OCR 48px, LaMa inpainter. Se almacenan en `manga-image-translator/models/`.

- **Nota**: `pydensecrf` no se pudo compilar en Windows (falta C++ Build Tools). Se creó stub en `env/lib/site-packages/pydensecrf/`. Mask refinement se salta (LaMa funciona sin CRF).

### Cambios de esta sesión (2026-07-14)

- **app.js**:
  - `loadPdfJs()` (~470): **Estrategia dual** — intenta `import()` ES module v4.10.38, con fallback a script UMD clásico v3.11.174 si falla. Ambos configuran worker inmediatamente. Logging detallado.
  - `openFile()` (~535): Detección de PDF ahora **case-insensitive** (`/\.pdf$/i.test()`). Logging detallado al detectar tipo de archivo. Error en catch muestra mensaje o "Desconocido".
  - `renderPage()` (~580): Logging de obtención de página PDF.
  - `openFile()` catch (~574): Muestra `error?.message` con fallback "Desconocido".
  - **CSP fix**: `data:` añadido a `connect-src` (coordinado con server.py e index.html) → OpenCV WASM carga sin bloqueo.
  - **SyntaxError `container` duplicado** (línea 275): eliminado `const container` extra en `showToast()`.
  - **SyntaxError `mobileMenuBtn` duplicado**: eliminada 2ª declaración.
  - **SyntaxError `detectSelected` es `null`**: eliminada `const detectSelected` + event listener movido a `autoDetectPage` (botón "Traducir Página" real).
  - **Mobile menu toggle** (≤1024px): botón ☰ en topbar + toggle `.sidebar.open` + cierre al click fuera.
  - **Layout responsive fluido**: sidebar `grid-template-columns: minmax(240px, 25%) 1fr`, `min-width: 240px`, `max-width: 30%`, padding 1.5rem. Breakpoints actualizados (1200px: 28%, 1024px: drawer).
  - **Zoom nativo (sin CSS transform)**: `fitPage` recalcula `state.scale` basado en dimensiones reales de página (scale 1.0) vs viewport disponible, llama `renderPage()` para re-render nativo PDF.js. Doble-click en canvas = reset a `state.scale = 1.8` (default).
  - **viewport meta**: `minimum-scale=0.5, maximum-scale=3, user-scalable=yes`.

- **server.py**:
  - **CSP fix**: `data:` añadido a `connect-src` en `CSP_POLICY` (línea 62) → OpenCV WASM carga sin bloqueo.

- **index.html**:
  - **CSP fix**: `data:` en `connect-src` (línea 8) → OpenCV WASM.
  - **viewport meta**: `minimum-scale=0.5, maximum-scale=3, user-scalable=yes`.
  - Botón mobile menu ☰ en topbar (hidden, shown via CSS ≤1024px).

- **styles.css**:
  - Sidebar fluida: `minmax(240px, 25%)`, `min-width: 240px`, `max-width: 30%`, padding 1.5rem.
  - Breakpoints: 1200px (28%), 1024px (drawer), 640px (stack).
  - `.stage` mantiene `transform-origin: center center` para zoom futuro si se necesitara.

- **start-app.bat**: Corregido `PYTHON=%ROOT%env\Scripts\python.exe` (antes `.venv`).
- **start-app.ps1**: Corregido `$python = Join-Path $root "env\Scripts\python.exe"` (antes `.venv`).

### Bugs corregidos anteriormente

| # | Bug | Corrección |
|---|---|---|
| 1 | `wrapTextLines` separaba palabras occidentales carácter por carácter | Solo CJK se separa por carácter; occidental por palabras completas |
| 2 | Cabeceras/pies de navegador se traducían | `filterPageBlocks` filtra 5% márgenes con regex robustas |
| 3 | Sello `ZONAOLYMPUS-COM` y `1 C 2 E` se traducían | Watermark patterns descartan en cualquier posición |
| 4 | Texto `"8"` fantasma de arte | Filtro aspect ratio (`w/h < 0.4` y `text_len <= 3`) |
| 5 | Bloques del servidor sin filtrar en cliente | `filterPageBlocks` aplicado a `serverResult.blocks` |
| 6 | Doble inpainting (servidor + cliente) | `hasServerInpainted` → skip client inpainting |
| 7 | Race condition auto-translate al cargar | `await renderPage(1)` antes de `autoTranslateAllPages()` |
| 8 | `.venv` sin dependencias vs `env/` con todo | `start-app.bat/.ps1` y AGENTS.md documentan `env/` |

### Pendiente de vigilar

- **PDF.js fallback**: Monitorear si el fallback UMD se activa frecuentemente (indicaría que `import()` de `.mjs` falla sistemáticamente). Si es así, considerar migrar definitivamente a UMD o usar `pdfjs-dist` como dependencia local.
- **Burbujas en margen extremo**: diálogo real en 5% superior/inferior podría filtrarse. Si ocurre, bajar umbral a `0.03`.
- **Sello `1 C 2 E`**: OCR variable puede leerlo distinto. Ajustar regex `\b1\s*[\s-]?c\s*[\s-]?2\s*[\s-]?e\b`.
- **Página 3 del manga de prueba**: verificar que "INCREÍBLE... REALMENTE INCREÍBLE..." se detecte completo.
- **`.venv/` vs `env/`**: `.venv/` existe pero está incompleto. No usar. `env/` es el entorno real. Si se reinstalan dependencias, siempre en `env/`.

---

## 5. Flujo de Trabajo Recomendado

1. **Lee AGENTS.md** antes de modificar `app.js` o `server.py`.
2. Si tocas una **zona sensible**, documenta el motivo.
3. **Antes de agotar contexto/tokens**: actualiza §4 (Estado Actual) con lo hecho y pendiente.
4. Reinicia el servidor Flask tras cambios en `server.py`:
   ```powershell
   $env:PYTHONIOENCODING="utf-8"; & "D:\crear traductor\env\Scripts\python.exe" "D:\crear traductor\server.py"
   ```
5. Recarga el navegador con **Ctrl + Shift + R** (bypass cache) para `app.js` actualizado.

---

## 6. Protocolo de Sesión

- **Al inicio**: Lee AGENTS.md antes de cualquier código.
- **Durante**: Vigila presupuesto de contexto. Al notar agotamiento, **detente**.
- **Prioridad inmediata**: Actualiza §4 "Estado Actual" con:
  - Qué se cambió (archivo, línea aproximada, qué)
  - Qué queda pendiente o a medio terminar
- **No pospongas**: si el espacio se agota, guarda el estado parcial.
- **Objetivo**: que la siguiente sesión pueda retomar sin perder contexto.