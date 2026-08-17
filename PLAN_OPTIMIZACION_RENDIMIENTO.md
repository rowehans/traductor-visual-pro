# PLAN DE OPTIMIZACIÓN DE RENDIMIENTO — Traductor Visual Pro

> Investigación detallada (2026-08-14): código del repo + benchmarks existentes +
> fuentes externas (EasyOCR, PaddleOCR, CTranslate2, OpenCV, Flask, waitress,
> pipelines de scanlation open-source). Todo lo que se propone está anclado en
> código real del proyecto (con archivo/línea) y medible con los scripts de
> benchmark que ya existen en el repo.

---

## 0. Resumen ejecutivo

> **⚠️ Corrección de resolución (2026-08-14)**: los "5–15 s observados" de
> esta sección se midieron a 300 dpi (8.7 MP/pág). La producción real envía
> el canvas a pdf.js scale 1.2 (**0.7 MP/pág — 8 % del área**), donde la
> página normal mide **1.2–1.8 s** y la de YOLO→Ruta C **4–15 s** (ver §6,
> métricas corregidas). Esta sección conserva el diagnóstico histórico
> (etapas y % relativos — válidos como A/B), pero los números absolutos
> actuales están en §6.

El tiempo de procesamiento por página hoy (modo `fusion`, página normal) ronda
**5–15 s observados a 300 dpi** (CODEGRAPH.md), compuesto por:

| Etapa | Coste medido/estimado | % |
|:------|:----------------------|:-:|
| EasyOCR GPU (detector + recognizer) | ~0.88 s/pág | ~30–40% |
| RapidOCR CPU (onnx, SIEMPRE en fusion) | ~1.1–1.5 s/pág | ~30% |
| YOLO + comic-text-detector (solo páginas débiles) | 0.2–4 s | variable |
| Inpainting (TELEA + border-blend) | ~0.15–0.22 s/pág | ~5% |
| Traducción CT2 (beam 4, paralela) | ~0.02–0.12 s × bloques | ~5–10% |
| **Fallback Google (~2 s/bloque) en bloques que CT2 no cubre** | hasta varios s | **variable — sospechoso #1 de los 5–15 s** |
| Serialización base64 PNG (request + response) | ~0.1–0.4 s/pág | ~5% |
| Inferencia VLM (daemon, solo si el trigger dispara) | **60 s–8 min/pág** | elefante puntual |

**Tres palancas dominan** (orden de impacto):

1. **Evitar trabajo redundante por página** — RapidOCR corre SIEMPRE aunque
   EasyOCR ya detectó bien (páginas fáciles ≈ mayoría). Hacerlo condicional
   ahorra ~1.5 s/pág en páginas fáciles (≈30% del tiempo normal).
2. **Reducir el fallback Google** — el bottleneck real de las páginas "5–15 s"
   es probablemente la red (~2 s/bloque, con rate-limit). Batchear CT2 por
   página (una sola `translate_batch`) + greedy en vez de beam 4 + ampliar el
   cache de memoria de documento reduce drásticamente cuántos bloques caen a
   Google.
3. **Recortar el coste de los casos caros** — VLM (60 s–8 min): recortar
   `max_length`, reducir triggers espurios, y cachear el resultado de página
   completa para re-corridas (re-procesar el mismo capítulo sin repetir
   OCR+inpaint+VLM).

Más abajo está el detalle por fase con archivos, esfuerzo, riesgo y cómo
verificar cada cambio con los benchmarks existentes (`benchmark_overhead_*.py`,
`process_all_pages.py --ocr-mode …`, `stress_test_memory.py`).

---

## 1. Cómo se midió (metodología)

No hay que adivinar: el repo ya tiene el harness.

- **Per-página**: `benchmark_overhead_fusion.py` / `benchmark_ocr.py` /
  `benchmark_ruta_c_v2.py` (miden por etapa: OCR, inpaint, traducción,
  overhead de serialización).
- **Producción real**: `benchmark_production.py` (default `--scale 1.2`,
  misma resolución que manda el frontend; desglose por etapa + trigger v4.2
  + conteo de YOLO/CTD). Es el benchmark de referencia para tiempos
  absolutos desde 2026-08-14.
- **PDF completo**: `process_all_pages.py --ocr-mode fusion --workers N`
  (checkpointing + estadísticas por página).
- **Memoria**: `stress_test_memory.py --full`.
- **Calidad**: `analisis_calidad.py` (corpus) + reporte `generate_fusion_report.py`.

**Regla del plan**: cada cambio propuesto debe poder validarse con uno de esos
scripts ANTES/DESPUÉS sobre el mismo PDF (`Capítulo 43 …pdf`, 53 págs) y
registrarse en un JSON de benchmark. Nunca optimizar sin medir la misma página.

**⚠️ Resolución de medición (corregido 2026-08-14)**: los benchmarks
históricos renderizaban a 300 dpi (8.7 MP/pág). El frontend real envía el
canvas a pdf.js scale 1.2 (**714×1011 px, 0.7 MP — 8 % del área**;
`app.js::state.scale`, validado por `process_all_pages.py`). Los tiempos
absolutos de los benchmarks a 300 dpi **NO representan la operación real**
(12.5× más área) — los A/B de calidad/tiempo intra-proceso sí son válidos
porque comparan la misma imagen. Desde 2026-08-14, **todo benchmark de
rendimiento se mide a la resolución real de producción** con
`benchmark_production.py` (default `--scale 1.2`); 300 dpi queda solo para
A/B comparativos entre variantes, no como número absoluto.

---

## 2. Fase 1 — Ganancias rápidas, bajo riesgo (1–3 días)

### 2.1. RapidOCR condicional en vez de siempre (impacto: −1.1–1.5 s/pág en páginas fáciles)

**Dónde**: `ocr_engine.py::_run_fusion` (corre `_run_hybrid` → EasyOCR + RapidOCR
incondicional) y `ocr_utils.py::_run_rapidocr`.

**Qué**: correr RapidOCR solo si el resultado de EasyOCR está "débil"
(menos de N bloques o confianza media < umbral — reutilizar el mismo criterio
del trigger v4.2 / gate YOLO). En páginas bien detectadas (la mayoría de un
capítulo), el paso CPU de 1.1–1.5 s desaparece.

**Riesgo**: perder detección complementaria (RapidOCR lee texto que EasyOCR
garbea en tipografía artística). Mitigación: el tier híbrido ya tiene el gate
de páginas débiles — reutilizarlo; validar con `analisis_calidad.py` que la
cobertura no baje (bloques/pág y % traducidos sobre el PDF de 53 págs).

**Verificar**: `benchmark_overhead_fusion.py` antes/después + recuento de
bloques y cobertura idénticos en ≥40 páginas fáciles.

### 2.2. EasyOCR: `batch_size` en el recognizer + `cudnn.benchmark` (impacto: −15–35% del tiempo EasyOCR)

**Dónde**: `ocr_utils.py::_get_ocr_reader` / `_detect_and_ocr` (donde se llama
`reader.readtext(..., batch_size=…)`).

**Qué**:
- `readtext(..., batch_size=8..16)`: EasyOCR procesa los crops de texto de la
  página de a uno por defecto; el recognizer (CNN) acelera ~2-4× con batch
  mayor en GPU. Coste: más VRAM momentánea (crops chicos, ~decenas de MB).
- `torch.backends.cudnn.benchmark = True` una vez tras cargar el reader
  (~10–20% en convnets con tamaño de input estable).

**Riesgo**: bajo. Validar que la VRAM no suba sobre el presupuesto de 4 GB
(`stress_test_memory.py`).

**Nota importante (GPU vieja)**: NO convertir EasyOCR a FP16 — en GTX 1050 Ti
(Pascal) el throughput FP16 es ~1/64 del FP32; es contraproducente. Dejarlo
FP32 documentado en config.

### 2.3. Respuesta inpainted en JPEG en vez de PNG (impacto: −5–10× en el payload de respuesta)

**Dónde**: `ocr_utils.py::_cv2_to_base64` (default `fmt=".png"`) → call sites en
`routes/api.py` (respuesta de `process-page`/`process-page-batch`) y en
`_finalize_page_blocks`.

**Qué**: la imagen inpaintada solo se **muestra** en el navegador (no se
re-OCRea). Codificarla en JPEG q88–92 (o WebP si el cliente lo soporta) en vez
de PNG: en páginas de manga (áreas planas) el JPEG es 5–10× más chico y el
encode ~2× más rápido. El PNG solo se justifica para exportación.

**Dónde NO tocar**: la **entrada** del OCR (request) debe seguir en PNG/WebP
sin pérdida — artefactos JPEG en los bordes de los glifos degradan la
detección. Solo se cambia la salida.

**Riesgo**: bajo. Añadir un campo `response_format: "jpeg"|"png"` al request
para compatibilidad (frontend antiguo).

**Verificar**: tamaño del JSON de respuesta + tiempo total por página en
`benchmark_overhead_fusion.py`.

### 2.4. Transferencia binaria en vez de base64 (impacto: −33% bytes y −2 conversiones CPU por página)

**Dónde**: frontend `js/utils.js::canvasToBase64` (usa `toDataURL`); backend
`routes/api.py` (`request.get_json`) y `ocr_utils.py::_base64_to_cv2`.

**Qué**: enviar `canvas.toBlob("image/png")` vía `FormData`/`fetch` con body
binario, y en el servidor leer `request.data` → `cv2.imdecode(np.frombuffer(...))`.
Elimina el +33% de base64 y el coste de encode/decode en ambos lados (~0.1 s/pág).
Para la respuesta: mismo truco (responder bytes con `Content-Type: image/jpeg`
en un endpoint paralelo `/api/process-page-image` o un campo binario), o
mantener JSON pero con la imagen fuera.

**Riesgo**: medio (toca frontend + backend + CSP `connect-src` — revisar que el
CSP permita el body binario, no cambia nada de CSP porque sigue siendo `self`).
Requiere actualizar tests de `routes/api.py` y `test_packaging.py` (que
verifican `canvasToBase64`/`toDataURL`).

### 2.5. Cap de escala del canvas de OCR en el frontend (impacto: −30–40% píxeles transferidos)

**Dónde**: `app.js::renderPage` usa `state.scale` (1.8) para renderizar Y
enviar `cleanBgCanvas`. `process_all_pages.py` ya valida que **ZOOM=1.2 basta**
para OCR de calidad (benchmark de 128 págs).

**Qué**: separar resolución de **display** (1.8, nítido) de la de **OCR**
(cap ~1.4–1.5): renderizar un segundo canvas a escala cap y enviar ESE. Reduce
píxeles, tiempo de `toDataURL` y bytes sin degradar OCR (el server ya cap a
`canvas_size=2500` y `MAX_IMAGE_DIMENSION=4096`).

**Riesgo**: bajo-medio. Validar cobertura por página en páginas con texto
pequeño (~15 px) contra el mismo cap usado por el batch path (1.2) que ya
funciona.

### 2.6. CT2: greedy (beam 1) + una sola `translate_batch` por página (impacto: −2–4× en traducción CT2, menos Google)

**Dónde**: `translator.py::_translate_ctranslate2` (hoy `beam_size=4` y una
llamada por texto) y `routes/api.py::_finalize_page_blocks` (lanza un
`executor.submit` por texto único).

**Qué**:
- `beam_size=4 → 1–2`: en bloques cortos de manga el beam 4 rinde poco y
  cuesta 2–4×. Probar beam 2 con `analisis_calidad.py`; si la calidad no baja,
  adoptarlo.
- Agrupar los N textos únicos de la página en **una** `translate_batch`
  (CTranslate2 ya soporta listas) en vez de N llamadas serializadas por el
  pool: menos overhead de tokenización y mejor uso del batch reordering del
  motor. Con el fallback: si un bloque del batch falla, re-enviarlo solo.

**Riesgo**: bajo (cambia solo la orquestación de CT2, el pipeline
CT2→Google→SIN_TRAD no se toca).

### 2.7. Cache de página completa por hash de contenido (impacto: re-corridas ~0 s de OCR+inpaint+VLM)

**Dónde**: nuevo `cache.py`/`ocr_engine.py` — un cache filesystem
`cache/pages/<sha256(b64)>-<ocr_mode>.json` con {blocks, inpainted_jpeg,
diagnostics}, TTL largo (7 d, igual que translations).

**Qué**: hoy el cache de decisiones U-OCR evita re-disparar el VLM en páginas
repetitivas, pero **cada corrida re-ejecuta OCR + inpaint + traducción de
todas las páginas**. Si el usuario re-procesa el mismo capítulo (o la misma
página dos veces), devolver el resultado cacheado completo: la segunda pasada
es instantánea. El frontend ya tiene `inpaintedBgByPage` (Map en memoria);
esto lo hace persistente.

**Riesgo**: medio (claves con el doc_id/ocr_mode; invalidación por cambio de
config). Asegurar que los modos benchmark (`force_uocr`, `disable_uocr`,
`pure_easyocr`) no usen el cache.

**Verificar**: `process_all_pages.py` dos veces seguidas — segunda corrida
debe reportar ~0 s/pág para páginas sin cambios.

---

## 3. Fase 2 — Optimizaciones de servidor y de casos caros (2–4 días)

### 3.1. Servidor de producción en vez del dev server de Werkzeug (impacto: concurrencia y estabilidad)

**Dónde**: `server.py:324` (`app.run(...)`, dev server) y `main.py:203`.

**Qué**: hoy corre el **dev server de Werkzeug** (single-process, no pensado
para producción). Opciones en Windows:
- **Reintentar waitress** con el fix documentado: el problema histórico era el
  catch-all `<path:filename>` + blueprints en Flask 3.x (404s). Solución:
  registrar el catch-all en la **app raíz** (`app.add_url_rule('/<path:filename>'`)
  en vez de en el blueprint, y que sirva solo archivos existentes
  (`send_from_directory`), dejando los 404 reales al framework. Validar con el
  suite de tests + smoke del servidor (`run_ci.py` job `server-test`).
- Si waitress sigue fallando: mantener Werkzeug pero con `threaded=True`
  explícito y `processes=1` documentado, y medir (el OCR ya serializa con
  semáforo, así que el cuello de botella no es la concurrencia HTTP).

**Riesgo**: medio. Se prueba aislado con el job `server-test` de GitHub Actions.

### 3.2. Compresión de respuestas JSON (Flask-Compress / gzip) (impacto: −70–90% en JSONs grandes)

**Dónde**: `server.py` (registrar `Compress(app)`).

**Qué**: los bloques de respuesta (texto + metadata) comprimen ~70–90% con
gzip/brotli; la imagen base64 (2.4) NO comprime — por eso 2.3/2.4 van antes.
Comprimir solo respuestas > 1 KB y con `Content-Encoding: gzip` cuando el
cliente lo anuncia (fetch lo hace por defecto).

**Riesgo**: bajo. Añadir test de que la respuesta comprimida se descomprime
bien en el cliente (o desactivarlo si complica el cliente).

### 3.3. VLM (daemon): recortar `max_length` y tokens generados (impacto: −30–60% en páginas U-OCR)

**Dónde**: `uocr_daemon.py::_infer_once` / `_run_ocr_batch` (`max_length=4096`?)

**Qué**: el tiempo de generación del VLM es proporcional a los tokens de
salida. Medir cuántos tokens emite realmente `document parsing.` por página
(leer `result.md`); si el p90 es muy inferior a `max_length`, bajarlo
(2048/1024) para acotar el peor caso (una página que divaga no paga 8 min).
Además:
- `image_size=640` ya es mínimo razonable; no bajar.
- Considerar `no_repeat_ngram_size`/`ngram_window` ya configurados — ok.

**Riesgo**: bajo-medio (acota el peor caso; no toca la calidad del caso
normal). Verificar con `uocr_*_benchmark` y `benchmark_unlimited_ocr.py`.

**MEDICIÓN 2026-08-15 (distribución de tokens real, `benchmark_vlm_tokens.py`)**:
se envolvió `generate()` del modelo (daemon parado, VRAM libre, midiendo en
proceso) en las 9 páginas del trigger del cap. 43:

| pág | tokens generados | llamadas `generate` | wall VLM | texto real |
|-----|-----------------:|--------------------:|---------:|-----------:|
| **21** | **1949 (97 % del cap 2048 — su EOS real está más allá)** | 2 | 299.5 s | 18 chars |
| 15 | 376 | 6 | 64.7 s | 132 chars |
| 28 | 266 | 3 | 41.6 s | 292 chars |
| 16 | 246 | 2 | 37.8 s | 302 chars |
| 25 | 202 | 1 | 30.4 s | 246 chars |
| 31 | 132 | 2 | 27.9 s | 134 chars |
| 32 | 73 | 2 | 14.7 s | 79 chars |
| 36 | 32 | 2 | 9.3 s | 18 chars |
| 37 | 32 | 2 | 9.2 s | 18 chars |

**Lectura**: 8/9 páginas generan **32–376 tokens** (EOS temprano, mediana 202)
y pagan 9–65 s. La patológica es SOLO la 21: 1949 tokens (97 % del cap),
299 s y apenas 18 chars de texto real — el modelo divaga en bucle hasta el
cap. Los 16 tokens repetidos en 36/37/32 son generaciones degeneradas
(1 bloque, posiblemente `[Unreadable]` filtrado). **Calibración resultante**:
`max_length=1280` acota el peor caso (pág 21: 300 → 170 s, −43 %) SIN tocar
las 8 normales (máximo 376 < 1280); el A/B mismo-proceso confirmó que 1024
SÍ pierde texto real (pág 28 pierde 'DA IGUAL', pág 16 pierde 'EN ESE CRSO,')
y 512/640/768 fallan directo (el prefijo de imagen supera el cap → error 500).
**APLICADO 2026-08-15**: `config.py::UOCR_MAX_LENGTH 2048 → 1280` con
verificación del capítulo completo (ver §4.6, bloque 3.3 —
"VERIFICACIÓN CAPÍTULO COMPLETO (1280)").

### 3.4. Prefetch/pipeline en el frontend de traducción masiva (impacto: solapa render con OCR)

**Dónde**: `app.js::autoTranslateAllPages` (while-loop con espera 500 ms).

**Qué**: hoy el flujo es render→enviar→esperar→render siguiente. Pipeline:
mientras la página N está en el servidor, renderizar y enviar N+1 (ventana de 2
en vuelo). El servidor serializa el OCR con semáforo, así que el solape real
gana en inpaint/traducción/serialización (~0.3–1 s/pág), no en OCR.

**Riesgo**: medio (estado de cancelación/pausa — `translationPaused`, `abort`).
Mantener el botón Pausa/Cancelar coherente.

### 3.5. YOLO/CTD: `imgsz` 1024 y dedup (impacto: −40–50% del coste del tier de detección débil)

**Dónde**: `config.py::YOLO_IMGSZ` (1280 → 1024) y los gates.

**Qué**: en páginas débiles (donde corre), 1280→1024 acelera ~2× la detección
YOLO con pérdida mínima de recall en globos grandes (el re-OCR es sobre crops
3.5×, no sobre la detección). El CTD ya tiene dedup vs YOLO
(`COMIC_DETECTOR_DEDUP_IOU`) — verificar con el benchmark de 5 páginas que el
dedup sigue activo.

**Riesgo**: bajo. Solo toca páginas débiles; validar con `benchmark_ruta_c_v2.py`.

---

## 4. Fase 3 — Cambios mayores / modelos (1 semana+, requieren decisión)

> **MEDICIÓN REAL (sesión, benchmark_ocr_stages.py, págs 3/11/12 del cap. 43)**:
> detect=2.877s / recognize=1.068s / total=3.945s — **el DETECTOR domina
> (72.9%)**, no el recognizer (27.1%). Esto invalida la opción 4.1 (cambiar el
> recognizer por RapidOCR) y deja 4.2/4.3 sin justificación por ahora: el
> cuello de botella del camino caliente es CRAFT (detección), y los detectores
> alternativos rinden similar en manga. Se mantiene el estado actual (EasyOCR
> detect + PP-OCRv4 no aporta). Re-medir si cambia el corpus o la GPU.
> Resultado completo: `benchmark_results/ocr_stages.json`.

### 4.1. Mejorar el recognizer del camino caliente (no el detector)

El bottleneck medido es **EasyOCR GPU 0.88 s/pág** (70% del pipeline en modo
easyocr). Opciones evaluadas desde la investigación externa:

| Opción | Verdict | Por qué |
|:-------|:--------|:--------|
| PaddleOCR completo (paddlepaddle) | ❌ No | Más pesado de cargar, modelo similar al RapidOCR ya en uso, dependencia nueva grande |
| **RapidOCR como reconocedor del híbrido con detector EasyOCR** | ✅ Probar | Los detectores de EasyOCR (CRAFT) son buenos encontrando cajas; el recognizer de PP-OCRv4 es más rápido. Recortar las cajas de EasyOCR y reconocer con RapidOCR podría bajar el tiempo GPU del recognizer |
| manga-ocr (kha-white) | ❌ No aplica | Es para japonés; el corpus de este proyecto es **español→inglés** (páginas en español). Solo tendría sentido si se añade origen ja/ko/zh masivo |
| PaddleOCR-VL-for-manga / VLM chico (1.5–2B) para el daemon | ⚠️ Considerar | Reemplazar el modelo 3B del daemon por uno 1.5–2B cuantizado 4-bit en GTX 1050 Ti: ~1.5–2× menos tiempo por página VLM con calidad comparable en texto de cómic (memoria de banda limitada) |

**Recomendación concreta**: antes de cambiar de modelo, medir el split
detector/recognizer del tiempo EasyOCR (profiling por etapa dentro de
`_detect_and_ocr`). Si el recognizer domina, probar recortes EasyOCR →
reconocer con RapidOCR. Si el detector domina, dejar como está (los detectores
CRAFT son difíciles de superar en manga).

### 4.2. Paralelizar OCR multi-proceso (workers separados) — solo si el semáforo satura

**Dónde**: `server.py` (semáforo OCR global) y `process_all_pages.py --workers`.

**Qué**: hoy el OCR se serializa con semáforo (una página a la vez por el lock
GPU). Con 2+ GPUs o si un segundo proceso CPU es deseable, lanzar un segundo
worker de OCR en otro proceso (o mover RapidOCR/YOLO/CTD a un proceso CPU
dedicado para que no compitan con EasyOCR GPU). El benchmark de 128 págs ya
mostró que workers=4 satura el semáforo; el punto óptimo es ~3.

**Riesgo**: alto (estado compartido, caches de decisión, doc_id). Solo
justificado si el semáforo se convierte en el cuello (medirlo con
`stress_test_memory.py`).

> **MEDICIÓN REAL (stress_test_memory.py, 50 págs, 4 workers, modo easyocr,
> servidor waitress)**: 50/50 exitosas, 0 errores, memoria estable
> (+71.8 MB < umbral de leak de 100 MB). Distribución de tiempos por página
> bajo carga: min 2.0s / p25 2.5s / mediana 2.7s / p75 3.7s / p90 4.2s / max
> 6.1s (las 4 primeras incluyen warmup GPU). El semáforo introduce espera
> (mediana 2.7s vs ~1-1.5s secuencial) pero NO satura: sin timeouts, sin
> errores, latencia acotada en el p90. **Veredicto: 4.2 NO procede** — la
> serialización es por VRAM de la GTX 1050 Ti (4 GB, un motor GPU a la vez),
> no por elección de diseño; un segundo worker no añadiría throughput GPU, y
> el detector (CRAFT, 73% del coste según 4.1) seguiría siendo el cuello.
> Revisitar solo con 2+ GPUs o si el corpus cambia.

### 4.3. Exportar/render del PDF en worker thread del navegador

**Dónde**: `app.js::renderPage` (pdf.js en main thread).

**Qué**: pdf.js ya usa Web Workers internamente para parsear; el `render` al
canvas es en main thread. Mover el `renderPage` a un `OffscreenCanvas` +
worker es un cambio grande con ganancia moderada en máquinas lentas. Prioridad
baja frente a 2.5.

### 4.4. Pre-filter: eliminar el coste fijo de 1.4–1.6 s/pág (la mayor oportunidad medida)

> **MEDICIÓN REAL (benchmark_prefilter.py, págs 3/11/12 del cap. 43)**:
> `_pre_filter_image` cuesta **1.41 s/pág promedio**, y de eso **1.407 s
> (99.8%) es UN SOLO `cv2.inpaint TELEA`** de líneas horizontales. El
> detector de líneas (kernel 1×15 + umbral 50) marca **88–94 % del área de
> la página** como "línea" (falso positivo masivo: captura el arte del
> manga), así que el inpaint corre sobre la página completa — no hay bbox
> que recortar. El bilateral NO es el problema (0.003 s, OpenCV lo tiene
> optimizado), ni el speckle (0.026 s), ni la morfología (<0.01 s).
>
> **A/B de calidad (mismo pipeline, prefilter on/off)**: en páginas NORMALES
> el prefilter cuesta 1.4 s y NO aporta (pág 3: −2 bloques/−0.07 conf;
> pág 12: −0.20 conf); solo paga en páginas DÉBILES (pág 11: +7 bloques,
> conf 0.29→0.65). Resultado completo: `benchmark_results/prefilter.json`.
> (2026-08-16: `benchmark_prefilter_selective.py` + `prefilter_selective.json`
> — 4.4C medida y descartada, ver bloque RESULTADO 4.4C abajo.)

**Dónde**: `ocr_utils.py::_pre_filter_image`.

**Opciones ordenadas por riesgo/beneficio:**

| Opción | Ahorro estimado | Riesgo | Cómo |
|:-------|:----------------|:-------|:-----|
| **4.4B — Inpaint a resolución reducida** | ~1.0 s/pág (pipeline intacto) | Medio-bajo | El inpaint de líneas finas se hace a 0.5× y se upscalea; el resto del prefilter y el tier1 reciben una imagen casi idéntica |
| **4.4A — Prefilter condicional per-página** | ~1.4 s/pág en el caso común | **Alto** | tier1 sobre imagen cruda; solo si EasyOCR es débil (0 bloques/conf baja) aplicar prefilter + reintentar — la misma lógica del fallback CLAHE ya existente. Cambia la imagen de entrada del tier1 → cambia bloques/conf → altera el trigger v4.2 y las decisiones VLM |
| **4.4C — Detector de líneas selectivo** | ~1.4 s/pág | Medio | Kernel/umbral más estricto para que el inpaint corra sobre área chica (~0.03 s como el speckle). Cambia el resultado del prefilter (líneas reales podrían no limpiarse) |

**Recomendación**: empezar por **4.4B** (mayor ahorro con el menor cambio de
comportamiento — no toca el pipeline, solo abarata una operación interna del
prefilter) y validarlo con un A/B de salida del prefilter + resultado OCR
antes/después. Si la validación es limpia y se quiere más, evaluar 4.4A con
`analisis_calidad.py` + re-verificación del determinismo (los gates por los
que ya se peleó).

> **RESULTADO (4.4B IMPLEMENTADA Y DESCARTADA con datos)**: se implementó el
> inpaint de líneas a escala reducida (0.5×, con guard < 40 px) y se midió en
> el flujo real (`_detect_and_ocr` con `prefilter=True`, págs 3/11/12):
> prefilter 1.33s → 0.26s (−81%) pero **REGRESIÓN en la página débil** — pág
> 11: 9 bloques / conf 0.727 → **7 bloques / conf 0.612** (determinista en 4
> corridas; no es variación cuDNN). La escala 0.9 también pierde (10→7 en el
> tier1 aislado) y 0.75/0.6 más. La pérdida ocurre exactamente donde el
> prefilter más aporta (recuperación de texto artístico en páginas débiles),
> así que el ahorro de 1.07 s/pág no lo justifica. **Se revirtió** a
> resolución completa y se descartó la constante de config. La vía correcta
> es **4.4C** (detector de líneas más selectivo para reducir el área de la
> máscara — el detector 1×15 + umbral 50 marca 88–94 % del área, que es la
> causa raíz del coste), no bajar la resolución del inpaint.

> **RESULTADO (4.4C IMPLEMENTADA, MEDIDA Y DESCARTADA con datos,
> 2026-08-16, `benchmark_prefilter_selective.py`, 14 págs canónicas a
> escala 1.2, daemon detenido, `prefilter_selective.json`)**: el detector
> selectivo de LÍNEAS REALES no tiene señal en este corpus — las páginas
> del cap. 43 NO tienen líneas de escaneo finas que aislar. Un detector
> de trazos OSCUROS (umbral 120 + apertura 1×15 + componentes anchos y
> finos: ancho ≥ 20 % de página y alto ≤ 8 px) marca **0.006 % del área en
> las 14 páginas (0.00 % en 12, bordes de panel en 4/29)** → el inpaint de
> líneas bajaría a 0.000-0.006 s — PERO no hay nada que limpiar: el
> detector actual (umbral de BRILLO tras apertura 1×15, umbral 50) marca
> 80-99 % (promedio 94.1 %) porque en un manga de fondo plano la apertura
> brillante conserva la página completa, no líneas. El inpaint a página
> completa es un SUAVIZADO de ruido distribuido, no la limpieza de líneas
> que el comentario del código asume. A/B del caso límite (quitar el paso
> de líneas por completo, solo speckle + bilateral, 0.48 → 0.02 s/pág):
> la salida NO es neutra — se cuela ruido como bloques basura (Δ +44
> bloques en las 14 págs, +5 a +7 en 1/4/8/39/43/52, textos '#'/'W'/
> 'BNNN'/'HOLLNNE'/'VDHO'), la pág 11 de control se mantiene (2/0.70 vs
> 2/0.66, sin regresión ni mejora) y en 43 baja la confianza (0.71 →
> 0.45). El inpaint de líneas actual está limpiando ruido distribuido en
> toda la página (por eso su máscara es ~94 %), y su coste de 0.53 s/pág
> es ESTRUCTURAL para este corpus: no se puede reducir por selectividad
> (no hay líneas que aislar) ni por resolución (4.4B ya midió regresión en
> la pág débil). Vía restante: 4.4A (prefilter condicional per-página,
> riesgo alto — cambia la imagen de entrada del tier1 y el trigger).

> **A/B DE DISEÑO DE 4.4A (2026-08-16, `benchmark_prefilter_4_4a.py`,
> capítulo completo 53 págs, mismo-proceso raw vs pref con cache de
> decisiones fresca por variante, daemon detenido, `prefilter_4_4a.json`)**:
> la pregunta clave (cuántas páginas justifican el prefilter y qué pasaría
> con el trigger) queda respondida con datos:
> - **Distribución de calidad sin prefilter**: 27/53 (50.9 %) caen bajo un
>   umbral LAXO (n<6 o conf<0.6), pero solo **9/53 (17 %)** bajo un umbral
>   ESTRICTO (n<6 Y conf<0.6): págs 1, 17, 21, 26, 33, 36, 37, 41, 45.
> - **Dónde el prefilter realmente gana**: 13/53 (24.5 %) con ≥2 bloques o
>   +0.10 conf (pág 17 conf 0.43→0.95, 33 0.56→0.86, 41 +4 bloques,
>   46 +5, 48 +3, 1 +2 — las páginas débiles de verdad, casi todas las del
>   umbral estricto + 4/46/47/48 que el umbral laxo no marca). En las otras
>   40 páginas el prefilter NO cambia el resultado (Δblk=0, Δconf=0.00):
>   los 0.53 s/pág son puro overhead en el 75 % del capítulo.
> - **Riesgo del trigger v4.2: MEDIDO y CERO** — de las 53 páginas, NINGUNA
>   cambia su decisión de trigger entre raw y pref (11 triggers en ambas,
>   las mismas páginas). El riesgo central de 4.4A (que saltarse el
>   prefilter en páginas normales altere qué páginas disparan VLM) no se
>   materializa en este capítulo.
> - **Ahorro bruto**: 0.53 s × páginas no-débiles = ~14 s (umbral laxo) o
>   ~23 s (estricto) por capítulo — < 2 % del tiempo total. El delta de
>   tiempo medido en el benchmark no es fiable (raw salió MÁS lento que pref
>   por estado de GPU: 4.74 vs 3.19 s/pág, el prefilter medido en
>   aislamiento es 0.53 s/pág); se usa el coste de producción.
> - **Veredicto del diseño**: el ahorro real (~14-23 s/capítulo) es marginal
>   frente al riesgo de implementación (doble pasada OCR en páginas débiles,
>   ya que la debilidad solo se conoce DESPUÉS de correr el tier-1, más el
>   re-prefilter de RapidOCR que hoy reutiliza la imagen ya filtrada).
>   Con el trigger v4.2 ya cubriendo las páginas débiles (las 9 del umbral
>   estricto son en su mayoría triggers VLM que el prefilter no decide),
>   4.4A NO procede: el coste estructural de 0.53 s/pág queda como está
>   documentado, y la palanca restante real sería un heuristico PRE-OCR
>   barato de detección de ruido, no una doble pasada.

### 4.5. Afinar `_rapid_cond_skip` para páginas con conf 0.15–0.20

> **MEDICIÓN REAL (benchmark_detect_stages.py, mismas 3 págs)**: el pase
> RapidOCR es la etapa dominante cuando corre — 37 % del total (0.7–2.7 s/pág)
> — y corre en páginas que EasyOCR detecta débil pero no vacía (pág 11: 2
> bloques conf 0.40; pág 12: 1 bloque conf 0.16). El umbral actual del skip
> usa `UOCR_TRIGGER_MIN_BLOCKS`/`UOCR_TRIGGER_CONF` (los del trigger v4.2).

**Dónde**: `config.py::RAPID_COND_MIN_BLOCKS/CONF` + `ocr_utils::_rapid_cond_skip`.

**Qué**: medir si subir el umbral de confianza del skip (o relajar el de
bloques) omite el pase RapidOCR en más páginas normales sin perder bloques
recuperables. A/B: correr con umbrales candidatos (conf 0.15 / 0.20 / 0.25) y
comparar bloques finales + tasa de traducción con `analisis_calidad.py`.

**Riesgo**: bajo (el pase ya es condicional; solo se mueve la frontera). El
caso a cuidar es la pág 11 — que RapidOCR rescata +7 bloques — y las páginas
CJK (el script-hints depende de `rapid_blocks`).

> **MEDICIÓN REAL (págs 3/11/12, umbrales 0.15/0.20/0.25/0.30)**: el umbral
> actual (0.20, el del trigger v4.2) **ya es el punto óptimo** — subirlo a
> 0.25/0.30 produce bloques idénticos (8/9/4) y tiempos iguales (~9 s en 3
> págs); bajarlo a 0.15 solo añade el coste de la pág 3 sin cambiar bloques.
> Las páginas débiles corren RapidOCR igual (necesario para la recuperación)
> y las normales ya lo skipean con el umbral actual. **Veredicto: no mover
> `RAPID_COND_MIN_CONF`** — no hay ganancia disponible en este corpus.

### 4.6. Medir y condicionar la Ruta C en el camino real (fusion)

> **MEDICIÓN REAL (benchmark_detect_stages.py, fase B)**: la Ruta C
> (`_recover_regions_with_easyocr`, upscale 3.5×) cuesta **0.54–1.63 s por
> crop** (3 regiones → 1.6–4.9 s por página en las págs medidas). Corre
> SIEMPRE en modo `fusion` (YOLO → Ruta C) ANTES del trigger v4.2, con gate
> heurístico por "página débilmente detectada".

**Dónde**: `ocr_engine.py::_ruta_c_yolo` / `_run_fusion`.

**Qué**: verificar que el gate heurístico de YOLO no dispare la Ruta C en
páginas que el híbrido ya resolvió fuerte (p.ej. la pág 3: 8 bloques conf
0.78). Si dispara, medir cuántos bloques recupera realmente en esas páginas y
subir el umbral del gate — ahorro de 0.5–4.9 s/pág en páginas normales, sin
tocar la recuperación de las débiles.

**Riesgo**: medio (misma preocupación que 4.4A — altera bloques/conf →
trigger). Requiere A/B con `analisis_calidad.py`.

> **MEDICIÓN REAL (cap. 43 completo, 53 págs, híbrido + `_ruta_c_yolo` real
> con el modelo finetuned)**: el gate dispara en 19/53 páginas (36 %), todas
> con <3 bloques (1–2) o conf < 0.35 — entre ellas páginas con conf ALTA
> (p40: 2 bloques conf 0.95, p49: conf 0.92). **18/19 de esas páginas
> RECUPERARON bloques YOLO reales** (2–7 bloques por página, p40 recuperó 7
> y p49 recuperó 4 — diálogo artístico que el híbrido perdió). Coste medio
> ~1.8 s/pág (40.9 s total, la primera incluye la carga del modelo).
> **Veredicto: el gate actual es correcto — NO subirlo.** Las páginas con
> pocos bloques son justo donde YOLO→Ruta C aporta; subir `YOLO_GATE_*`
> perdería recuperación real (el riesgo que las sesiones 123/129 ya
> documentaron con la p5). No hay coste evitable medido en páginas normales
> (las normales, ≥3 bloques y conf ≥ 0.35, ya quedan fuera del gate).

> **MEDICIÓN REAL 2026-08-14 (upscale 2× aplicado + gate intra-crop,
> benchmark_rutac_upscale.py / rutac_gate3_diag.json)**:
> 1. **Upscale 3.5× → 2× APLICADO y validado** — A/B completo:
>    - **Validación en 2 etapas**: 5 págs (31 = 31 bloques, conf idéntica) →
>      12 págs (65 = 65 bloques, conf idéntica en todas las corridas).
>    - **Recuperación idéntica texto a texto**: `benchmark_rutac_recovery.py`
>      comparó los pares (texto, confianza) por índice en las 5 págs donde la
>      Ruta C dispara (1, 4, 7, 8, 11): **34 = 34 bloques recuperados, 0
>      diferencias** — descarta que el total enmascare una pérdida.
>    - **A/B controlado intra-proceso** (misma GPU, mismas páginas, estado
>      cuDNN fijo): ahorro **−13 % a −24 %** según corrida (la varianza es
>      de GPU/thermal; la recuperación fue 65 = 65 en todas). El ahorro se
>      concentra en las páginas con Ruta C: pág 4 −2.1 s, pág 11 −1.5 s,
>      pág 7 −0.8 s; las normales ahorran poco pero nunca pierden bloques.
>    - **Cambio de producción**: default `upscale: float = 3.5` → `2.0` en
>      `_recover_regions_with_easyocr` (`ocr_utils.py`) + las 3 llamadas
>      (`_ruta_c_yolo`, `_ruta_c_ctd`, `_reforzar_…`) en `ocr_engine.py`;
>      comentario en `config.py`. Los tests existentes pasan `upscale=3.5`
>      **explícitamente** → siguen validando el mapeo de coordenadas ÷
>      upscale sin cambios. CI completo verde (13/15, 944/944 tests).
>    - Persistido en `benchmark_results/rutac_upscale_ab.json` y
>      `rutac_recovery_ab.json` (mypy limpio, fuera de `_PROD_PY_FILES`).
> **CORRECCIÓN 2026-08-15 — los benchmarks de upscale tenían DOS bugs y el
> A/B anterior NO comparó 3.5 vs 2**: (a) `benchmark_rutac_upscale.py`
> re-anidaba el wrapper en cada pasada — la pasada "2×" envolvía a la
> "3.5×" y FORZABA 3.5 otra vez → comparaba 3.5 vs 3.5 (el −13-24 % era
> pura deriva); (b) `benchmark_rutac_recovery.py` NO forzaba el upscale en
> el wrapper (pasaba el del caller, 2.0 de producción) → el "34 = 34
> bloques" era 2.0 vs 2.0 y no validaba nada. Ambos reescritos con parcheo
> correcto (original capturado UNA vez tras los wraps de timing) + harness
> anti-deriva de `benchmark_ab_utils.py` (intercalado por página, orden
> alternado, páginas de control como noise-floor, veredicto).
> **RE-MEDICIÓN CORREGIDA (2026-08-15, daemon detenido, --reps 3):**
> - **Tiempo: NEUTRO** — un benchmark mide 2× +1.6 % (+0.07 s/pág) y el otro
>   −0.016 s/pág: signos opuestos = efecto ~0 (ruido 0.016-0.039 s). El
>   "−13-24 %" no existe; el upscale no mueve el tiempo de la Ruta C con el
>   strip (ambos caben en el mismo chunk del det).
> - **Recuperación: 3.5× ≥ 2×** — totales 49 vs 47 (−2) y texto-a-texto
>   34 vs 32 (−2), ambos concentrados en pág 11 ('1 目' + lecturas
>   distintas del mismo globo 'JINO QUJERO IR AHI!!!' vs 'IINOQUJERO
>   IRAHI!!'); las págs 1/4/7/8 conservan el mismo nº de bloques con
>   varianza de segmentación (mismo contenido, distinta partición).
> **DECISIÓN RESUELTA (2026-08-15)**: producción REVERTIDA a **upscale
> 3.5×** (`_recover_regions_with_easyocr` default + 3 callers en
> `ocr_engine.py` + `benchmark_rutac_batch.py`). Validación con el
> benchmark corregido (daemon detenido, --reps 3):
> - **7 págs**: 3.5× = 34 vs 2× = 32 bloques (−2, pág 11).
> - **14 págs (7 canónicas + 7 pesadas, rutac_upscale/recovery_ab14.json)**:
>   bloques finales **105 vs 100 (−5)** y recuperados Ruta C **90 vs 85
>   (−5)** — el −2 de pág 11 se sostiene y se amplifica: 2× pierde en 11
>   (−2, '1 目' + 'JINO QUJERO IR AHI!!!'), 29 (−1), 39 (−2), 43 (−2,
>   'ACUERDOBIEN'/'TENIASUNREGALO…' fusionados en 1 bloque) y solo gana en
>   46/52 (+1 c/u). **Tiempo neutro en ambas corridas** (−0.01 y −0.03
>   s/pág, noise-floor 0.014-0.017 s — ~1%).
> CI verde (961/961). El 2× aplicado 2026-08-14 se descarta: su A/B estaba
> roto (comparaba 3.5 vs 3.5).
> 2. **Gate intra-crop (no re-OCRear crops cubiertos): NO PROCEDE.**
>    Instrumenté la Ruta C por crop (34 crops en 12 págs): **0 crops estaban
>    cubiertos por un bloque previo de conf ≥ 0.7** — el filtro existente
>    (`_overlap_ratio > 0.5` vs bloques previos en `_ruta_c_yolo`/`_ruta_c_ctd`)
>    ya elimina la redundancia ANTES de re-OCRear. Los 5/34 crops que
>    devuelven 0 bloques (15 %) NO son predecibles a priori: su `cls_conf`
>    va de 0.31 a 0.86 (los 2 peores son CTD con conf ALTA 0.83/0.86), así
>    que un gate por confianza de detección perdería recuperación real
>    (crops con cls_conf 0.31–0.47 que SÍ producen 1–2 bloques).
>    **Veredicto: no hay coste evitable por crop — el gate correcto ya
>    existe.**

> **A/B de parámetros del crop (2026-08-15, benchmark_rutac_params.py)**:
> 1. **Pad 3% en vez de 6% — APLICADO y validado**: 14 págs (7 canónicas +
>    7 pesadas de Ruta C), **107 = 107 bloques, conf y textos idénticos** a
>    **−23.6 %** de tiempo (−1.45 s/pág; pág 4 −3.5 s, pág 29 −3.1 s, pág 39
>    −2.2 s). Cambio: `_RUTA_C_PAD_FACTOR 0.06 → 0.03` (mín 6 px intacto;
>    los crops pequeños ya dominaba el mínimo). 40 tests de la Ruta C +
>    944/944 CI verdes. **Verificación post-cambio (2026-08-15,
>    rutac_pad_postchange.json)**: mismas 14 págs, 107 = 107 bloques y
>    textos idénticos (0 diferencias), −18.0 % (−1.13 s/pág; la diferencia
>    vs −23.6 % es estado de GPU: el daemon VLM ocupa 2.25 GB VRAM).
> 2. **INTER_LINEAR en vez de INTER_CUBIC — DESCARTADO con datos**: el
>    resize de crops pequeños no es el cuello (tiempo +2.6 %, sin ganancia) y
>    la recuperación empeora (−4 bloques, conf distinta en 5/7 págs — el
>    blur del upscaling degrada el OCR).
> 3. **rotation_info (0,180) en vez de (0,90,180,270) — DESCARTADO**: solo
>    afecta al fallback EasyOCR (RapidOCR es primario, 5–13 crops/págs), así
>    que no hay ahorro (+0.5 %), y **pierde 1 bloque real** en la pág 4
>    (título vertical/cartela que solo recupera la rotación 90/270).

> **REEVALUACIÓN DEL GATE DE PANEL GRANDE OSCURO (2026-08-15,
> benchmark_production.py + instrumentación de recuperación del VLM)**:
> **Veredicto: el gate es correcto — NO se puede afinar sin perder
> recuperación.** Mediciones de las 11/53 páginas que disparan
> (`large_image_panel`, dark_ratio 0.181–0.218, todas con conf < 0.75 al
> momento del trigger):
> - **Las 9 páginas donde el VLM corrió recuperaron 1–8 bloques cada una**
>   (mediana 4, 32 bloques totales): pág 21 +8 (314.8 s), pág 28 +6 (54.1 s),
>   pág 16 +5 (69.7 s), pág 31 +5 (47.1 s), pág 25 +4 (57.9 s), pág 15 +1
>   (105.5 s), pág 32 +1 (37.3 s), pág 36 +1 (23.6 s), pág 37 +1 (23.6 s).
> - **2 páginas (13, 17) quedaron en skip del cache §8.4.1** (gemelas que ya
>   corrieron el VLM y no recuperaron nada) — el cache ahorra ~2 llamadas
>   por capítulo sin perder texto.
> - **Subir el umbral perdería recuperación real**: páginas en 0.181–0.19
>   (13, 15, 17, 32, 31) recuperan 1–5 bloques; el umbral 0.18 separa
>   limpiamente (11 > 0.18 vs 42 ≤ 0.18, sin falsos negativos medidos).
> - **El costo del VLM es el cuello real, no el gate**: 23.6–314.8 s/llamada
>   (mediana ~50 s, ~12 min por capítulo en las 9 llamadas) — domina el
>   tiempo del capítulo (el resto del pipeline: ~3.5 min). Palanca: Fase 3.3
>   (recortar max_length/tokens del daemon), no tocar el gate.
> - Persistido en `benchmark_results/gate_vlm_1/2/3.json`.

> **A/B de parámetros del pase rapid de la Ruta C (2026-08-15,
> benchmark_rutac_params.py, 14 págs)**: **los 3 DESCARTADOS — los defaults
> (0.5/1.6/6) ya son el óptimo.**
> 1. **box_thresh 0.5 → 0.35: DESCARTADO.** Cambia solo 2/14 págs (fusión de
>    globos en pág 4, −1 bloque; pág 52 fusión menor). El "−21.5 %" de
>    tiempo es deriva de orden de corridas, no el parámetro: páginas SIN Ruta
>    C (pág 2) muestran el mismo −0.5 s que en el A/B del pad donde el
>    parámetro no aplica.
> 2. **unclip_ratio 1.6 → 2.2: DESCARTADO.** Desastroso en crops: **−14
>    bloques** (107→93), globos vecinos fusionados en 9/14 págs (pág 4
>    "EVAN,QUIERO"+"PATATASPARA LA CENA" → 1) y +16 % más lento. Los params
>    agresivos (2.2) son para el reintento de PÁGINA COMPLETA pre-VLM, no
>    para crops de globo.
> 3. **rec_batch_num 6 → 16: DESCARTADO.** 107 = 107 bloques sin cambio y
>    +4.8 % más lento. Los crops tienen ≤6 líneas de texto → batch 6 ya las
>    procesa en una pasada; batch mayor solo agrega overhead.
> **Conclusión**: el cuello de las páginas de 7–10 s es el re-OCR por crop en
> sí (detección DBNet por crop), no los umbrales — ningún parámetro mueve la
> aguja sin perder recuperación. La única palanca restante es estructural:
> **batch de crops a nivel de recognizer** (una sola pasada det+rec con las
> líneas de todos los crops de la página, Fase 2.2 aplicada a rapid) — fuera
> del alcance de un A/B de parámetros. Persistido en
> `rutac_rapid_ab.json`/`rutac_batch_ab.json`.

> **RE-CORRIDA CON RUIDO CONTROLADO (2026-08-15, daemon VLM DETENIDO,
> --reps 3, `rutac_params_reps3.json`)**: los porcentajes de los A/B
> anteriores (pad −23.6 %, box_thresh −21.5 %, unclip +16 %) eran **deriva
> de GPU, no efecto del parámetro** — el daemon compartía la GTX y el
> noise-floor era ±0.3-0.7 s; con la GPU quieta el noise-floor de control
> baja a **0.018-0.022 s** (25×) y los veredictos son todos ESTABLES:
> 1. **pad 0.03 vs 0.06: neutro** (+0.2 % tiempo, bloques idénticos
>    47 = 47). El 3% de producción se mantiene (no requiere revertir), pero
>    la ganancia "−23.6 %" documentada era ruido.
> 2. **box_thresh 0.5 → 0.35: neutro en tiempo (−0.1 %) y PIERDE 1 bloque**
>    (47 → 46, +1 crop al fallback EasyOCR) → default 0.5 CONFIRMADO.
> 3. **unclip 1.6 → 2.2: neutro en tiempo (−0.1 %) y PIERDE 4 bloques**
>    (47 → 43, +1 crop al fallback) → default 1.6 CONFIRMADO con más fuerza
>    (la pérdida de bloques es la señal robusta; la magnitud −14 previa
>    incluía deriva).
> Conclusión revisada: los defaults de la Ruta C (pad 3 %, box 0.5, unclip
> 1.6) son el óptimo por recuperación; ningún parámetro alternativo ahorra
> tiempo real. El cuello de las páginas de 7-10 s NO era los umbrales — era
> el re-OCR por crop (resuelto con el batch estructural, ver arriba).

> **A/B del max_length del VLM — Fase 3.3 ejecutada (2026-08-15,
> benchmark_vlm_maxlen.py + benchmark_vlm_tokens.py)**:
> 1. **Escalera 512/640/768/1024/1280/2048 en pág 21**: 512/640/768 FALLAN
>    directo (el prefijo de imagen supera el cap → error 500 / 0 tokens);
>    1024 → 44 bloques raw (−54 %); 1280 → 57 (−40 %); 2048 → 95 (con
>    duplicados `[Unreadable]`).
> 2. **A/B mismo-proceso (2048 vs 1024 vs 1280 por página)**: 1024 PIERDE
>    texto real (pág 28 pierde 'DA IGUAL', 'SE QUE TIENES'; pág 16 pierde
>    'EN ESE CRSO,', 'LA ÚLTIMA VEZ.'); **1280 conserva TODO el texto real**
>    (págs 28/16 idénticas token a token — el modelo corta por EOS antes del
>    cap en páginas normales; en pág 21 los únicos bloques que se pierden son
>    duplicados `[Unreadable]` que el merge elimina).
> 3. **Distribución de tokens (9 páginas del trigger)**: 8/9 generan
>    32–376 tokens (EOS temprano, mediana 202, 9–65 s); SOLO la 21 genera
>    1949 tokens (97 % del cap, 299 s, 18 chars de texto real — divaga en
>    bucle). Ver tabla completa en §3.3.
> 4. **Veredicto**: `UOCR_MAX_LENGTH = 1280` acota el peor caso (pág 21:
>    300 → ~196 s, −35 %) sin tocar las 8 páginas normales (máximo 376
>    tokens < 1280) ni perder recuperación. **APLICADO 2026-08-15**:
>    `config.py::UOCR_MAX_LENGTH 2048 → 1280` y verificado con
>    benchmark_production.py sobre las 53 páginas del capítulo — ver
>    bloque "VERIFICACIÓN CAPÍTULO COMPLETO (1280)" abajo.
> - Nota de harness: parchear el atributo `uocr_client.UOCR_MAX_LENGTH` NO
>   surte efecto — `process_page` captura el default al definir la firma;
>   el A/B envuelve `process_page` con un wrapper. Persistido en
>   `vlm_2048_a.json`, `vlm_512_a.json`, `vlm_1024.json`, `vlm_1024_b.json`,
>   `vlm_1280.json`, `vlm_1280_2.json`, `vlm_sameproc.json`,
>   `vlm_1280_sameproc.json`, `vlm_tokens_a/b/c.json`, `vlm_tokens_dist.json`.

> **VERIFICACIÓN CAPÍTULO COMPLETO con max_length=1280 (2026-08-15,
> benchmark_production.py, 53 págs, scale 1.2)**:
> - **Trigger idéntico: 11/53** (todas `large_image_panel`) — el gate v4.2
>   no depende del cap de generación.
> - **Recuperación del VLM mantenida: 31 bloques vs 32 a 2048** (gate_vlm_*):
>   pág 21 +7 (vs +8 — el −1 es el duplicado `[Unreadable]` que el merge
>   elimina; texto real idéntico), 28 +6, 16 +5, 31 +5, 25 +4, 15 +1, 32 +1,
>   36 +1, 37 +1 — las 9 páginas en las que corrió recuperaron 1-7 bloques.
>   13/17 corrieron con recuperación 0 (el skip del cache §8.4.1 no aplicó
>   en esta corrida por estado de la cache persistida).
> - **Peor caso recortado: pág 21 300 → 196 s (−35 %)** (etapa vlm 195.1 s);
>   las demás páginas VLM no cambian (generan 32-376 tokens < 1280, coste
>   45-115 s por estado de GPU, consistente con la distribución de §3.3).
> - **Bloques finales por página**: los VLM pages suben respecto al baseline
>   production_full53 (que se midió con el daemon en la anomalía de EOS
>   temprano): 16: 7→12, 28: 7→13, 21: 3→10, 31: 4→9, 25: 3→7, 15: 4→5,
>   32: 5→6, 36: 3→4, 37: 4→5; las 42 páginas no-VLM tienen bloques
>   IDÉNTICOS al baseline (0 diferencias).
> - CI verde (944/944, cobertura OK); tests de uocr_client adaptados
>   (27 passed, leen UOCR_MAX_LENGTH dinámicamente).
> - Resultados: `production_1280_1-12/13-21/22-34/35-53.json` (4 chunks,
>   el capítulo tarda ~20 min por las 11 llamadas VLM) +
>   `tools/analizar_vlm_1280.py` (fusión y comparación).

> **ROI DEL VLM POR PÁGINA (2026-08-16, consolidado de las 5 corridas
> completas: vlm_full53, vlm_full53_2, all3, gate_p17 base/off — 51 llamadas
> VLM)**: la recuperación por página es NOTABLEMENTE DETERMINISTA (5/5
> corridas con el mismo recuento en 16/21/25/28/31/32/36/37; 15 siempre 1;
> 13 y 17 SIEMPRE 0 — en las 4-5 corridas donde el VLM corrió). Tabla
> ordenada de s/bloque (vlm_stage / bloques recuperados, agregado 5
> corridas):
>
> | Pág | VLM total | Bloques | s/bloque | Lectura |
> |---|---|---|---|---|
> | 31 | 197.5 s | 20 (5/5 × 4) | **9.9** | MEJOR ROI |
> | 28 | 306.2 s | 30 (× 6) | 10.2 | |
> | 16 | 279.8 s | 25 (× 5) | 11.2 | |
> | 25 | 252.3 s | 20 (× 4) | 12.6 | |
> | 21 | 913.1 s | 40 (× 8) | 22.8 | el más caro en absoluto (pero 8 bloques reales) |
> | 37 | 132.9 s | 5 (× 1) | 26.6 | |
> | 36 | 133.7 s | 5 (× 1) | 26.7 | |
> | 32 | 153.6 s | 5 (× 1) | 30.7 | |
> | 15 | 422.9 s | 5 (× 1) | **84.6** | PEOR ROI con recuperación (1 bloque carísimo) |
> | 13 | 170.4 s | 0 (4/4) | ∞ | SIN RECUPERACIÓN (4/4 corridas) |
> | 17 | 101.9 s | 0 (4/4) | ∞ | SIN RECUPERACIÓN (4/4 corridas) |
>
> Lectura para futuros gates: (1) 13 y 17 son los candidatos claros —
> SIEMPRE 0 recuperación (el gate de 17 ya se midió: −22 s sin pérdida,
> pero < ruido de medición del capítulo); el gate de 13 ahorraría ~34-68
> s/corrida (el doble que 17). (2) 15 es el peor ROI con recuperación: 1
> bloque por 71-97 s/corrida — el diálogo que recupera vale 84 s/bloque;
> un gate por página para 15 solo tendría sentido si ese bloque concreto
> deja de ser valioso. (3) 32/36/37 (1 bloque, 21-42 s) están en el límite:
> el coste por bloque duplica al de 31/28/16/25. (4) El determinismo
> 5/5 sugiere que un gate por firma de página (no por heurística global)
> sería seguro de calibrar con los datos existentes.

> **A/B DEL GATE DE PÁG 17 (2026-08-16, `benchmark_gate_p17_ab.py`, capítulo
> completo 1-53, daemon ready, `gate_p17_base.json` vs `gate_p17_off.json`)** —
> pág 17 es la única llamada VLM del capítulo con 0 recuperación. Resultado:
> anular su trigger (wrapper de `_trigger_con_cache` que devuelve False solo
> en pág 17; rapid-agresivo y el resto del pipeline intactos) baja la pág de
> 25.5 s → 3.2 s (**−22.3 s exactos**), triggers 11 → 10, VLM 11 → 10,
> **recuperación 31 = 31 (sin pérdida)**. Sin efectos laterales: las 42
> páginas no-VLM suman 448.7 vs 449.0 s (Δ 0.3 s, noise). PERO el total del
> capítulo solo baja 1068.7 → 1072.9 s (+4.2 s, es decir NADA): el ahorro de
> 22.3 s queda enmascarado por la varianza de generación del VLM en otras
> páginas (pág 15 +20.8 s, 28 +10.5 s, 16 +8.9 s vs 36 −12.2 s, 25 −3.9 s —
> mismas llamadas, misma recuperación, generación estocástica). **Veredicto:
> el gate de pág 17 ahorra ~22 s (~2 % del capítulo) sin perder nada, pero
> es del orden de la varianza entre corridas — implementarlo añadiría un
> caso especial al trigger por un ahorro menor que el ruido de medición.
> Se documenta y NO se aplica** (el ahorro real del capítulo es la
> generación, no el gate; la pág 17 ya está dentro del ruido).

> **ANÁLISIS DE PÁG 13 — ¿VALE LA PENA SU LLAMADA VLM? (2026-08-16): NO.
> Datos consolidados: 0 recuperaciones en 4/4 corridas donde el VLM corrió
> (vlm_full53_2 68.0 s, all3 31.9 s, gate_p17_base 31.5 s, gate_p17_off
> 39.0 s — siempre rec=0; en vlm_full53 la saltó el cache de negativas).
> Token-level (daemon-side a max_length 1280): 32 tokens nuevos en 2
> llamadas generate y 1 bloque crudo — la misma huella que la pág 17
> (también 32 tokens/1 bloque crudo/rec=0): el VLM no encuentra diálogo
> artístico que extraer. El trigger es `large_image_panel` (panel oscuro
> grande) pero el panel no tiene texto incrustado. **Por qué la saltó el
> cache solo UNA vez**: el mecanismo §8.4.1 ya es un gate por historial de
> recuperación (firma → negativa con TTL 30 min), pero para pág 13 falla
> por dos motivos: (1) el TTL expira entre corridas separadas >30 min (las
> corridas consecutivas de benchmark borran el cache con
> `clear_decision_cache` y re-registran); (2) la firma de pág 13 es
> INESTABLE entre corridas — la negativa persistida se registró con 6
> bloques/conf 0.81 en 3 corridas pero el cache mostró entradas con 6/0.59
> y 3/0.00 (dos firmas distintas para la misma página) — la salvaguarda
> `much_mas_debil` y la deriva de la firma impiden que la negativa
> sobreviva. **Recomendación: no vale la pena — gate por historial con
> firma estable.** Pág 13 (y 17) son las 2 únicas páginas con 0
> recuperación en TODAS las corridas; el costo de 13 (31-68 s, el doble
> que 17) sí justifica un tratamiento. La vía correcta: extender el TTL de
> negativas para ceros confirmados o usar una firma más estable (p. ej.
> hash de la imagen prefilter vs el grid de oscuridad que deriva con los
> bloques del híbrido), validando con el A/B del gate de 13 igual que el
> de 17.

> **AISLAMIENTO DE PÁG 21 — ¿SE PUEDE BAJAR MÁS EL PEO CASO? (2026-08-16,
> `benchmark_vlm_maxlen.py --pages 21` a 1024 vs 1280, mismo-proceso con el
> daemon real, `vlm21_1024.json` + `vlm21_1280.json`)**: NO — bajar el cap
> a 1024 no ahorra nada útil. Mediciones frescas: 1024 → vlm 167.0 s, 7
> bloques recuperados (10 finales); 1280 → vlm 177.1 s, 8 recuperados (11
> finales). **La diferencia de 1 bloque es OTRO duplicado `[Unreadable]`**
> (los textos reales son idénticos entre 1024 y 1280: 'CUFNE DENENE
> RELPVNTEPARA CEAREARLA.', 'ENTONCES ENTONCES', 'EPORQUEYO PUEDO...?',
> 'NE PORIA JBFLMR', 'NO...'), o sea: 1024 recorta ~10 s (6 % de la etapa
> vlm de pág 21, ~1.3 % del capítulo) SIN ganar nada y CON el riesgo ya
> medido de perder texto real en otras páginas (16/28 pierden bloques a
> 1024, ver A/B de §3.3). La estructura del costo de pág 21 está aislada:
> **2 llamadas `generate` — (1) infer principal: 16 tokens en ~23 s (casi
> todo prefill de imagen a 640 px); (2) re-OCR del panel grande
> (`_recover_art_dialogue`, crop_mode=False): el resto, 1933 tokens a 2048
> y ~1280 (el cap) en producción — el modelo DIVAGA en bucle de repetición
> (18 chars de texto real, no_repeat_ngram 35 no alcanza a cortarla). El
> cap ya la trunca en producción (300 → ~196 s, −35 %); el remanente del
> costo es prefill + los ~1280 tokens truncados, no recortable por cap sin
> perder el texto del panel. **Veredicto: `UOCR_MAX_LENGTH = 1280` se
> mantiene** — es el punto calibrado; 1024 no mejora el ROI y arriesga las
> otras páginas. Palanca real restante para pág 21: el prompt (fijo
> '<image>document parsing.' en uocr_daemon.py) o el prefill, no el cap.

> **BATCH ESTRUCTURAL DE CROPS DE LA RUTA C (2026-08-15,
> benchmark_rutac_batch.py) — la palanca estructural diseñada y medida.**
> **Veredicto: el diseño VALIDA (−76.6 % en el núcleo de la Ruta C); la
> integración a producción queda pendiente de decisión.**
> **Diseño**: en vez de det DBNet + rec por crop (N llamadas al engine), los
> crops de la página se apilan en un strip vertical (gap blanco 24 px, chunks
> de ≤ 1900 px de alto — límite del det max_side_len=2000) y se ejecuta det
> UNA vez por chunk + UNA sola llamada text_rec con TODAS las líneas de todos
> los crops (batch nativo del recognizer). Mapeo idéntico al actual (÷
> upscale, corrección 180°, _group_and_merge_blocks por crop, filtro
> RUTA_C_RAPID_MIN_CONF). El ahorro viene de: det 5 crops sueltos = 1.33 s vs
> 1 strip 400×1200 = 0.45 s (2.9×, el det domina ~90 % del costo del crop) y
> rec de 5 líneas en 1 llamada = 569 ms vs 700 ms sueltas.
> **Medición (8 págs pesadas de Ruta C sin VLM, 52 crops, scale 1.2, sin
> spellcheck para aislar el núcleo det+rec)**:
> | pág | crops | baseline | strip | Δ |
> |-----|------:|---------:|------:|---:|
> | 29 | 11 | 4.68 s | 0.74 s | −3.93 |
> | 1 | 7 | 3.20 s | 0.61 s | −2.59 |
> | 11 | 8 | 2.70 s | 0.72 s | −1.98 |
> | 4 | 6 | 2.18 s | 0.65 s | −1.53 |
> | 8 | 5 | 2.00 s | 0.47 s | −1.53 |
> | 7 | 5 | 1.86 s | 0.55 s | −1.32 |
> | 52 | 6 | 2.29 s | 0.69 s | −1.60 |
> | 39 | 4 | 1.59 s | 0.37 s | −1.22 |
> TOTAL **20.49 s → 4.79 s = −76.6 % (4.3×)**; det-calls 52 → 8 (1 por
> página), rec-calls 52 → 8. El strip escala sublineal con los crops: ~0.4–0.7 s
> de det por página (casi independiente del nº de crops; el coste restante es
> rec ∝ líneas totales).
> **Recuperación**: equivalente o mejor. Las diferencias de texto son
> mayormente SEGMENTACIÓN (el per-crop fusiona burbujas/líneas que el strip
> separa, y viceversa — pág 4: el baseline junta 3 burbujas en 1 bloque, el
> strip las separa) y lecturas distintas del MISMO texto (pág 29 es ruido en
> ambos caminos). El strip vía rapid recupera texto que el per-crop solo
> conseguía con fallback EasyOCR (burbujas 'seria/prestigjos/Y-por_supuesto'
> de pág 4). Duplicados (2 en pág 4, 1 en pág 11): regiones YOLO solapadas
> (crops gemelos con el mismo contenido) — mismo comportamiento que el
> baseline (cuyo fallback fusionaba ese texto), deduplicado downstream por
> overlap en la fusión. El único "miss" del prototipo es el fallback EasyOCR
> (el strip prototipo es rapid-only): en una integración real el fallback por
> crop se conserva (donde rapid no devuelve texto usable). 0 cajas caídas en
> los gaps del strip en todas las páginas.
> **Costo oculto detectado (ortogonal al strip)**: `_ocr_spellcheck`
> (pyspellchecker) cuesta ~320 ms por bloque con texto largo/pegado (15
> llamadas = 4.8 s en pág 4) — lo paga CUALQUIER camino que recupere ese
> texto; el baseline lo difiere a la fusión porque sus fallbacks no pasan por
> _group_and_merge_blocks. Candidato a optimización separada (acotar la
> distancia de edición para palabras > N chars).
> **OPTIMIZACIÓN APLICADA (2026-08-15)**: `_ocr_spellcheck` ahora usa
> `_spellcheck_correction()` — una réplica barata de `sp.correction()` que
> produce el MISMO resultado (candidatos a distancia 1, luego 2; preferencia
> de diacríticos; max por frecuencia) sin la expansión masiva de distancia 2
> de pyspellchecker (~4.8 M strings para una palabra de 21 chars ≈ 0.3-1.3 s).
> El diagnóstico previo era INCOMPLETO y dejó el problema a medias:
>
> **CORRECCIÓN DE DIAGNÓSTICO (2026-08-15, profile de pág 4 real)**:
> 1. **El langdetect NO era el cuello** — ni aparece en el top del profile
>    (cProfile de pág 4: 42 s de pipeline, 26.2 s en `correction`, 22.9 s en
>    `known`, 26.3 s en `__edit_distance_alt`). El costo residual "del langde-
>    tect" del resumen anterior era en realidad la expansión de distancia 2
>    de pyspellchecker sobre palabras largas/pegadas del OCR. langdetect
>    directo mide ~16 ms incluso a 8000 chars.
> 2. **El fast-path previo fue INERTE en producción**: el índice agrupado por
>    longitud se cacheaba en un `WeakKeyDictionary`, pero `SpellChecker`
>    define `__slots__` sin `__weakref__` → el índice NUNCA se pudo almacenar
>    con el checker real, el `except` del caller ponía puede_corregir=True
>    siempre, y se seguía pagando `sp.correction()` completo. (Los tests
>    pasaban porque MagicMock SÍ soporta weakref; producción no.)
>
> **La solución aplicada (2026-08-15)**:
> - **Índice arreglado**: `_SPELL_INDEX_BY_ID` ahora es un dict por id(sp) con
>   referencia fuerte al owner (`_SPELL_INDEX_OWNERS`): el id no se reutiliza
>   mientras el índice exista, y `owner is sp` detecta reutilización teórica.
> - **`_spellcheck_correction(sp, word)`**: réplica exacta de la selección de
>   pyspellchecker (Damerau-Levenshtein — pyspellchecker cuenta transposi-
>   ciones como 1 edición — con pre-filtro de conteos de caracteres por
>   longitud: edición <= k implica exceso/defecto de conteos <= k, así el DP
>   solo corre contra ~1 % del bucket). Se usa a partir de
>   `_SPELL_CORRECTION_MIN_LEN=13`: en palabras cortas el scan es MÁS lento
>   que pyspellchecker (calibrado: len 10 → 0.5x, len 13+ → 1.5-1287x), así
>   que las cortas delegan a `sp.correction()` (pyspellchecker es rápido ahí:
>   known + edición 1 generan pocos strings).
> - **Prefijo del langdetect (lo pedido)**: el detector se aplica sobre un
>   prefijo de máx. `_SPELL_LANG_MAX_CHARS=600` chars en vez del texto
>   completo de los bloques fusionados (langdetect escala con la longitud;
>   verificado: 0 diferencias es/no-es entre texto completo y prefijo en el
>   corpus). El lru_cache de `_detect_language_robust` además comparte entrada
>   entre bloques largos de la misma página.
> **Equivalencia verificada**: 58/58 palabras (patológicas de 15-21 chars,
> largas de 10-15, comunes correctas) coinciden con `sp.correction()` real —
> solo el empate exacto de frecuencia 'detective'→'defectivo'/'defectiva'
> (freq=50/50) difiere, y pyspellchecker mismo es no-determinista ahí (orden
> de hash del set). Medición en pág 4 (pipeline fusion completo, spellcheck
> activo, daemon VLM detenido): **spellcheck 10.43 s → 3.7 ms (~2800×)**,
> pipeline 29.12 s → 3.38 s, correcciones intactas ('prestigjosa'→'presti-
> giosa'). El langdetect por prefijo es el menor de los dos fixes (el costo
> real era la corrección), pero queda aplicado por robustez ante bloques muy
> largos. Tests de regresión: 3 existentes adaptados al seam +
> `TestSpellcheckCorrectionFast` (equivalencia con checker real, delegación
> de cortas, diacríticos, índice por instancia) + `TestSpellcheckLangPrefix`
> (prefijo recortado en bloques largos, texto completo en cortos).
> **INTEGRADO EN PRODUCCIÓN (2026-08-15)**: `_recover_regions_with_easyocr`
> ahora ejecuta el batch estructural por defecto vía `_RUTA_C_STRIP_BATCH=True`
> (módulo-global, no Final: toggle del A/B y válvula de rollback).
> Implementación: `_ruta_c_prepare_crops` (crops compartidos pad+upscale+cls,
> alineados con regions, None si no procesables), `_rapidocr_strip_batch`
> (stitch gap 24, chunks ≤ 1900, det por chunk con box_thresh/unclip 0.5/1.6,
> UNA text_rec para todas las líneas, mapeo por centro-y con descarte de gap,
> `_group_and_merge_blocks` por crop, semáforo + degradación a {}) y
> `_rapidocr_blocks_from_lines` (constructor de bloques compartido con
> `_run_rapidocr` — DRY, sin cambio de comportamiento). El fallback EasyOCR
> por crop, el lazy-load del reader CJK y el merge final se conservan
> íntegros. A/B mismo-proceso en producción (3 págs, núcleo --no-spellcheck):
> **9.81 s → 2.32 s (−76 %), det-calls 24 → 3**; bloques rapid pág 4 idénticos
> (5 = 5), págs 29/1 con varianza de segmentación/lectura (fallback activo).
> Con spellcheck, el costo residual es el langdetect del merge final sobre
> textos largos fusionados (ortogonal, preexistente — ver optimización del
> fast-path de pyspellchecker más arriba). Tests: 10 nuevos en
> `TestRapidocrStripBatch` (mapeo línea→crop, gap, chunks múltiples, engine
> None, excepción, semáforo, toggle per-crop) + 10 de la Ruta C adaptados al
> seam `_rapidocr_strip_batch` (los de `_reforzar_con_rapid_agresivo`/Fase 2
> NO cambian: ese camino sigue usando `_run_rapidocr` por página).
> **RE-CORRIDA DEL CAPÍTULO COMPLETO (53 págs, 2026-08-15, daemon VLM
> detenido — mismo estado que el baseline pre-strip `production_full53`)**:
> promedio **3.07 s/pág vs 3.99 s baseline (−23.2 %, −49.1 s en 53 págs)**.
> La etapa `rapid` (re-OCR por crop) bajó de **109.2 s a 17.2 s (−92 s)** —
> el strip eliminó el det DBNet por crop en TODO el capítulo. Las 8 páginas
> pesadas de 6-14 s (1/4/25/29/39/43/46/52) bajaron de **68.8 s a 48.4 s
> (−29.5 %)**: pág 4 9.89→6.01, pág 29 9.16→4.48, pág 43 7.35→4.76, pág 46
> 6.63→3.71, pág 52 6.39→5.20. **Recuperación NO perdida: 290 → 298 bloques
> (+8)**, 36 páginas idénticas, 11 ganaron, 6 perdieron 1 bloque (varianza
> de segmentación del fallback EasyOCR, patrón aceptado del A/B original).
> Resultado: `benchmark_results/production_strip_full53.json`. El objetivo
> Fase 1 "< 3.5 s/pág" queda CUMPLIDO (3.07 s/pág) — nota: esta corrida
> incluye el fix del spellcheck del turno anterior (10.43 s → 3.7 ms en

> **⚠ REVISIÓN DE LOS 6 BLOQUES PERDIDOS (2026-08-16, verificación
> mismo-proceso + VLM independiente, `strip_blocks_check.json` +
> `strip_targeted.json` + recortes en `strip_visual/`)**: la conclusión
> "varianza de segmentación" del A/B anterior era INCORRECTA — **4 de los 6
> bloques son CONTENIDO REAL PERDIDO** (diálogo confirmado por el daemon VLM
> sobre los recortes, lectura independiente del OCR): pág 1
> 'EN VERDAD ES INCREIBLE...', pág 4 '...Y SI HUBIESE ACEPTADO...', pág 40
> '¡PADRINO!!! ¡ESTO ESTÁ MUY RICO!', pág 52 '¿EN VERDAD QUIERES VOLVER A
> COMERLAS HOY?'. Solo 2 son segmentación real (pág 30 'ahhosuou' = ruido;
> pág 34 'CLING' → 'cuno' = onomatopeya leída distinto). **Mecanismo de la
> pérdida (debug por crop)**: el det DBNet del strip apilado detecta mal la
> línea dentro del crop — pág 52 crop 1: rapid individual lee
> 'QUIERESVOLVERA EENVERDAD/COMERLASHOY?' (conf 0.94/0.99) pero el strip lee
> 'OE O' (0.5); pág 1 crop 1: individual 'ENVERDADES INCREIBLE' (0.97) vs
> strip 'S E' (0.6); pág 40 crop 1: 'ESTOESTA IPADRINO!!/MUYRICO!' (0.97/0.98)
> vs strip 'P!' (0.46). El ruido del strip supera `RUTA_C_RAPID_MIN_CONF=0.45`
> → `usable_rapid` no vacío → **el fallback EasyOCR por crop NO corre** → el
> diálogo real se pierde silenciosamente. Pág 4 es distinto: el fallback
> EasyOCR SÍ corre pero lee basura del crop ('鼻'), y el bloque resultante
> solapa (iou 0.53) con la burbuja del híbrido 'YSIHUBIESE ACEPTADO.' que la
> fusión descarta como duplicada — contenido real perdido por la fusión.
> **Costo real del strip**: −2.38 s/pág (medido, válido) a cambio de perder
> ~4 diálogos reales en el capítulo (de ~300 bloques, ~1.3 %). **A/B DEL FIX + APLICADO (2026-08-16, `benchmark_strip_fix_ab.py`, 7 págs,
> mismo-proceso alternado base/fix, daemon ready, `strip_fix_ab.json`)** —
> la hipótesis "subir el umbral de usable_rapid a 0.7 → el fallback EasyOCR
> recupera" FUE MEDIDA Y DESCARTADA: con 0.7 el ruido sí se filtra ('p!',
> 'oe o', 's e' desaparecen) pero **EasyOCR lee basura en esos crops**
> ('じ‥', '鼻') — los 4 diálogos NO se recuperan. El motor que sí los lee es
> **rapid-INDIVIDUAL por crop** (0.94-0.99, confirmado por la referencia
> per-crop). **Fix aplicado**: retry por crop con `_run_rapidocr` individual
> cuando el strip no devuelve nada CONFiable (regla conf+longitud calibrada
> con la distribución real: ruido 0.25-0.71 corto vs lecturas reales largas
> 0.81-0.99 — 'S E' 0.71 vs 'ERASETUCCOMOLOS ILAM' 0.81, un umbral puro no
> las separa): confiable = conf ≥ 0.85, o conf ≥ 0.7 con ≥ 6 chars
> (`_RUTA_C_STRIP_RETRY_INDIVIDUAL=True` + `_rapid_blocks_usable`).
> **Resultado: 4/4 diálogos recuperados** (págs 1, 4, 40, 52: fragmento
> base=False → fix=True; los garbage 's e'/'鼻'/'p!'/'oe o' desaparecen;
> pág 30 'por'→'por por' y pág 34 'cuno'→'cling' son segmentación/lectura
> benigna). **El ahorro del strip se mantiene**: fix 2.6-5.6 s vs per-crop
> 4.2-7.5 s en las 7 páginas; el retry solo suma ~0.3-1.2 s en las páginas
> con crops fallidos y es noise-neutral en las sanas (pág 43: 3.30 vs 3.91).
> El retry conserva los bloques originales del strip si devuelve vacío
> (para no perder el script hint CJK del fallback). Tests: 2 nuevos (retry
> dispara en ruido y recupera; no dispara en lectura confiable) + 5
> adaptados (parchean `_run_rapidocr` a []); CI completo verde 987/987.
> **Capa extra (fallback híbrido, 2026-08-16)**: además, cuando el strip
> (ya con el retry) sigue DÉBIL (< `_RUTA_C_STRIP_HYBRID_MIN_CONF`=0.7) en
> un crop que solapa (>0.1) TEXTO PREVIO del híbrido (`hybrid_blocks` se
> pasa desde los 3 callers de ocr_engine: YOLO/CTD/burbujas), se corre
> IGUALMENTE EasyOCR y se FUSIONAN ambos resultados — protege el texto real
> del híbrido de ser descartado por un bloque débil de la Ruta C en la
> fusión (caso pág 4). 4 tests nuevos (dispara con texto previo + strip
> débil y fusiona; no dispara sin texto previo, con daemon infiriendo, ni
> con strip confiable); CI completo verde 991/991.

> **RE-CORRIDA DEL CAPÍTULO COMPLETO CON EL FIX APLICADO (2026-08-16,
> `benchmark_production.py` 1-53, daemon ready, `production_fix_full53.json`
> + `.log`)** — costo por página del fix en el pipeline completo:
> total 175.7 s / 53 págs = 3.32 s/pág (vs 162.5 s = 3.07 s/pág pre-fix de
> `production_strip_full53.json`, +8.1 %). El delta está dominado por el
> estado de GPU (una sola corrida, sin intercalado), no por el fix: las 4
> páginas afectadas contra el baseline non-VLM corregido
> (`production_nonvlm_fixed.json`, mismo harness) quedan dentro del ruido
> de ±1 s: pág 1 11.59 vs 11.07 s, pág 4 8.42 vs 8.32 s, pág 40 3.87 vs
> 4.31 s, pág 52 6.22 vs 5.75 s (Δ ≤ 0.5 s). El trigger mantiene 11/53 y el
> VLM se llama en las mismas 11 páginas (sin re-triggers por el fix).
> Veredicto: el fix recupera los 4 diálogos a un coste ≤ 0.5 s/página en
> las páginas afectadas (≤ 9 ms/pág en el promedio del capítulo), dentro
> del ruido — el tradeoff pre-fix (diálogos perdidos a cambio de −2.38 s/
> pág) queda cerrado con el fix activo.

> Nota de la corrida: el bloque de arriba continúa con el texto original de
> pág 4), que contribuye parte del ahorro dentro de `ruta_c`.

> **FAST-PATH EXTENDIDO A LOS SPELLCHECKERS EXTRANJEROS (2026-08-15)** —
> la misma técnica de acotación por longitud se aplicó a los ÚNICOS otros
> puntos de pyspellchecker del pipeline: `_contains_foreign_latin_tokens`
> (los `known()` sobre es/en/pt que detectan mezcla de idiomas):
> - **Acotación por longitud en `_contains_foreign_latin_tokens`**: cada
>   token más largo que `longest_word_length` del diccionario extranjero NO
>   puede estar en él (conjunto exacto, no aproximación por edición) → se
>   filtran los candidatos a `[2, longest]` antes del `known()` (mismo rango
>   que `_check_if_should_check` de pyspellchecker, con `isinstance` para
>   proteger los checker mockeados en tests). Efecto real: evita `known()`
>   inútil en tokens fuera de rango, y evita la CARGA bajo demanda del
>   diccionario en/pt cuando TODOS los desconocidos exceden la palabra más
>   larga (caso raro — una palabra de 14 chars siempre cae dentro del rango).
> - **Precarga de en/pt en `server.py`** (la ganancia determinista):
>   `_preload_background` ahora carga también los diccionarios en/pt junto
>   al español. Antes, la PRIMERA página con tokens no-españoles pagaba la
>   carga bajo demanda dentro del tiempo medido de la página (medido: en
>   83 ms + pt 266 ms ≈ 0.08-0.35 s según cache de disco; el `known()`
>   caliente es 0.01-0.05 ms/call — el costo NO es el lookup, es la carga
>   one-time de los diccionarios).
> **Medición honesta (`benchmark_production.py`, 53 págs, daemon detenido,
> `production_foreign_preload.json`)**: sin regresión — bloques 298 = 298,
> triggers VLM 11/53 idénticos, 42/53 páginas dentro de ±0.1 s. Promedio
> 3.07 → 2.95 s/pág (−3.9 %) pero con deltas de SIGNO MIXTO en las páginas
> trigger (pág 39 −2.97 s vs pág 4 +1.52 s — varianza de timeout del VLM
> con daemon caído), así que el agregado NO es atribuible a este cambio: el
> ahorro determinista es el one-time de ~0.08-0.35 s movido de la primera
> página con mezcla de idiomas al startup del servidor — invisible en el>   promedio por página (noise-floor 0.014-0.04 s/pág). Tests: 2 nuevos en
> `TestForeignSpellcheckBounds` (acotación con checker real, fallback a scan
> completo con MagicMock sin longest) + 2 en `test_server.py` (precarga
> incluye en/pt).

> **AUDITORÍA POST-ÍNDICE DE `_contains_foreign_latin_tokens` (2026-08-15)** —
> medida con `benchmark_foreign_check.py` sobre el capítulo completo
> (53 págs, daemon detenido, `foreign_check.json`): **112 llamadas, 0.0036 s
> totales = 0.032 ms/llamada** — el costo por bloque es despreciable y NO hay
> costo evitable que justifique tocarlo. Desglose:
> - **Los diccionarios en/pt ya no pagan nada por bloque**: sus `known()`
>   combinados suman 0.00048 s en TODO el capítulo (98 + 94 llamadas). La
>   carga one-time (~0.08-0.35 s) ya vive en el preload del server; el
>   fast-path por longitud ya filtra los candidatos fuera de rango.
> - **El índice `_spell_words_by_len` es ORTOGONAL a esta función**: sirve a
>   la réplica de corrección (`_spell_candidates`, camino ≥13 chars); la
>   detección extranjera usa `known()` = set lookups ya O(1) por token — no
>   hay nada que indexar.
> - **El es `known()` visible en la medición (0.306 s, 801820 palabras) NO
>   es de esta función**: 112 de las 1419 llamadas vienen de aquí; el resto
>   es la expansión interna de `sp.correction()` (camino corto <13 chars,
>   `known(edit_1)`/`known(edit_2)` con listas masivas) — otra función, con
>   su propio tradeoff ya calibrado (la réplica es más lenta en cortas).
> - **Únicos micro-costos evitables reales**: re-tokenización duplicada (el
>   caller ya tiene `palabras`) y un `set()` redundante sobre el set que ya
>   devuelve `known()` — juntos ~1-2 ms/capítulo (0.001 %). VEREDICTO: no
>   cambiar producción por datos despreciables (disciplina del proyecto);
>   `benchmark_foreign_check.py` queda como herramienta para re-auditar si
>   cambia el corpus (p. ej. más español/mezcla).

> **MEDICIÓN DEL CAMINO CORTO DE `sp.correction()` (2026-08-16,
> `benchmark_foreign_check.py` AMPLIADO, 53 págs, daemon detenido,
> `foreign_check_short2.json`)**: el "es known() 0.306 s / 801820 palabras"
> de la auditoría anterior se atribuyó aquí por rango de longitud — el costo
> del camino corto (<13 chars, que delega a `sp.correction()`) es **0.471 s
> en el capítulo (8.9 ms/pág)**, y el 99.9 % está concentrado en **2 palabras
> patológicas de OCR pegado**: 'clvdeg' (6 chars: 116 ms, 198893 words
> known) y 'clvdegchmar' (11 chars: 354 ms, 601463 words). Desglose:
>
> | Rango | Llamadas | sp.correction | known() words | Correcciones reales |
> |---|---|---|---|---|
> | 3-5 | 11 | 0.001 s (0.09 ms/call) | 836 | **3** ('unf','amf','chmar') |
> | 6-7 | 1 | **0.116 s** | 198893 | 0 |
> | 8-10 | 4 | 0.000 s (0.011 ms/call) | 4 | 0 |
> | 11-12 | 1 | **0.354 s** | 601463 | 0 |
>
> **Evaluación de bajar `_SPELL_CORRECTION_MIN_LEN` de 13 a 8-10** (la
> hipótesis: con el índice arreglado la réplica ya es competitiva en cortas):
> - **Equivalencia VERIFICADA: 10/10 palabras desconocidas de 8-12 chars**
>   dan el MISMO resultado que `sp.correction()` (los "diffs" del capítulo
>   eran un artefacto de mi comparación: cuando ambos devuelven `None` =
>   sin corrección, el comportamiento efectivo es idéntico).
> - **La réplica es 3-8× MÁS RÁPIDA en 8-12 fuera del diccionario**: 'unbuen'
>   207→64 ms, 'podriausar' 458→90 ms, 'corectamen' 446→82 ms,
>   'comerlashoy' 495→60 ms, 'matandoelt' 444→88 ms — el peor caso del
>   camino corto (palabra pegada) baja de ~0.35-0.5 s a ~0.04-0.09 s.
> - **No se pierde NINGUNA corrección real**: las 3 del capítulo son de
>   3-5 chars (se quedan en `sp.correction()` con umbral 8) y las de 6-12
>   no corrigen nada (0 corregidas) — bajar el umbral solo mueve las
>   patológicas a la réplica, que no las corrige (None) pero las resuelve
>   3-8× más rápido.
> **VEREDICTO**: el ahorro es real pero **marginal en términos absolutos**
> (0.471 s → ~0.09 s = ~0.38 s/capítulo ≈ 0.2 % del capítulo sin VLM; ~6 ms
> por página) y está concentrado en 2 palabras raras del corpus. Con la
> disciplina del proyecto (no cambiar producción por datos despreciables),
> **NO se cambia el umbral** — el costo actual del camino corto ya es
> 8.9 ms/pág y el fix del 2026-08-15 (réplica ≥13) resolvió el caso masivo
> (spellcheck 10.43 s → 3.7 ms en pág 4). La palanca queda DOCUMENTADA y
> lista (`_SPELL_CORRECTION_MIN_LEN=8` es seguro y verificado) si un corpus
> futuro trae más texto español pegado de 8-12 chars; re-auditar con
> `benchmark_foreign_check.py` antes de decidir.

> **ANÁLISIS DE LOS 17 BLOQUES MIXTOS: ¿AFINAR CUÁNDO SE LLAMA EL CHECK?
> (2026-08-16, `benchmark_foreign_gate.py`, 53 págs, daemon detenido,
> `foreign_gate.json`)** — se capturaron los 112 bloques que llegan al
> chequeo y, para los 17 `foreign=True`, se midió la evidencia extranjera
> (tokens en/pt no-es), las **correcciones forzadas** (re-correr
> `_ocr_spellcheck` con el gate desactivado y diff palabra a palabra — las
> que el gate preserva) y la ambigüedad del detector (`_detect_language_
> simple` vs el robusto que ya pasó el gate). Resultado por bloque:
>
> | Pág | Texto | Evidencia ext. | Correcciones forzadas |
> |---|---|---|---|
> | 2 | SUI DISTANCAAMENOS QUE | 'sui' (en) | 0 |
> | 4 | ESTA RELACIONADO CON (×2) | 'relacionado' (pt) | 0 |
> | 16 | TODO, HAS VENI- | 'has' (en) | 0 |
> | 29 | IO CUC PUEDO DOCIR. (×2) | 'io' (2 chars) | 0 |
> | **29** | **ASi QUE SEGUi (×2)** | **'segui' (pt)** | **2 c/u: 'ASi'→'Así', 'SEGUi'→'Seguir'** |
> | 41 | vemos uegol No (+×1 largo) | 'vemos' (pt) | 0 |
> | 41 | AL DIA SIGUIENTE... (×2) | 'dia' (pt) | 0 |
> | 42 | ALEGRIAQUE PUEDA VERTE | 'verte' (pt) | 0 |
> | 50 | SEPARO SUS (×2) | 'separo' (pt) | 0 |
> | 50 | ...PEOUENAS BOLSAS... | 'bolsas' (pt) | 0 |
> | 51 | CUANTOS ANOS MAS! | 'anos' (pt) | 0 |
>
> **Hallazgo central: el gate es INERTE para 15/17 bloques** — con el gate
> desactivado el spellcheck produce EXACTAMENTE el mismo texto (0
> correcciones forzadas): 13 son bloques TODO-MAYÚSCULAS (el loop se salta
> cada palabra como acrónimo `isupper`) y 2 son de caso mixto salvados por
> el filtro d1 (`'vemos'` → corrección d2 `'menos'` descartada por
> `_SPELL_EDIT_1_MAX_LEN=5`). La única protección REAL del capítulo está en
> pág 29: el gate preserva `'SEGUi'` (pt; acento-less quizá el español
> `'seguí'`) de corromperse a `'Seguir'` (d1), a cambio de bloquear el fix
> español `'ASi'`→`'Así'` (también d1) — 2 bloques con 2 cambios cada uno =
> **4 correcciones totales preservadas en el capítulo, 2 sobre token
> extranjero y 2 españolas perdidas**.
>
> **Causa raíz de los 15 triggers falsos: huecos del diccionario es**, no
> contenido mixto — verificado: `'relacionado'`, `'bolsas'`, `'vemos'`,
> `'dia'`, `'separo'`, `'verte'`, `'has'` son palabras españolas válidas
> AUSENTES del diccionario es de pyspellchecker (`es=False`, `pt/en=True`),
> así que el check las declara "desconocidas al español" y las encuentra en
> pt/en → falso mixto. Agregarlas al diccionario re-clasificaría esos 15
> bloques como españoles con **cero cambio de salida** (los 15 son inertes,
> verificado empíricamente) — un fix de cobertura, no del call-site.
>
> **VEREDICTO sobre afinar CUÁNDO se llama el check:**
> - **Por tiempo: NO** — 3.6 ms en 112 llamadas (0.032 ms/llamada); no hay
>   nada que ahorrar en el punto de llamada.
> - **"Solo cuando langdetect es ambiguo": NO** — 16/17 bloques son
>   es-confiado (`simple` = es); el único ambiguo (pág 41 'vemos uegol No',
>   `simple`=en) es inerte de todos modos. Un gate por ambigüedad saltaría
>   el check en los 16 confiados → pág 29 perdería la ÚNICA protección
>   extranjera real del capítulo ('SEGUi') para ganar 2 fixes de 'Así' —
>   un intercambio neto negativo en una sola línea de diálogo.
> - **El problema real es de PRECISIÓN, no del call-site**: 17/112 = 15 %
>   de triggers sugiere un corpus 15 % mixto cuando el bloque genuinamente
>   mixto es ~1 (pág 29). Si la señal importa en el futuro, el fix es la
>   cobertura del diccionario es (con `wf.add`, como las 16 palabras manga),
>   NO gatear la llamada. El call-site queda COMO ESTÁ; `benchmark_foreign_
>   gate.py` queda como herramienta de re-auditoría.

> **MAPEO + COSTO POR PÁGINA DE TODOS LOS TOUCHPOINTS DE PYSPELLCHECKER
> (2026-08-16, `benchmark_spell_touchpoints.py`, 53 págs, daemon detenido,
> `spell_touchpoints.json`)** — instrumentación con pila de sitios (misma
> técnica) sobre TODOS los puntos que tocan pyspellchecker y su maquinaria.
> **Mapa del pipeline**: `_ocr_spellcheck` tiene UN SOLO caller
> (`_group_and_merge_blocks`, línea 3119); fuera de su cuerpo no hay
> llamadas directas a pyspellchecker — los demás touchpoints son sus
> helpers (`_contains_foreign_latin_tokens` con `known()` es/en/pt, la
> réplica `_spellcheck_correction`, `_spell_candidates`, el índice
> `_spell_words_by_len`) más el preload de `server.py` (one-time). Costo
> real por capítulo:
>
> | Sitio | Llamadas | Tiempo | Nota |
> |---|---|---|---|
> | `langdetect` (contexto) | 523 | **2.954 s = 70 % del spellcheck** | 5.65 ms/llamada, prefijo 600 chars |
> | `sp.correction()` (camino corto <13) | 17 | 0.476 s | 100 % en pág 41 (2 palabras patológicas) |
> | `known()` interno de correction | 1212 | 0.313 s | expansión edit_1/edit_2 (dentro de lo anterior) |
> | cargas one-time es/en/pt | 172/98/94 | 0.574 s | precargadas por server.py en producción |
> | `_spell_candidates` | 11 | 0.188 s | 0.17 s es el BUILD one-time del índice |
> | `_spell_words_by_len` (índice) | 11 | 0.170 s | build one-time (0.17 s) + 10 hits en cache |
> | `foreign_check` (body real) | 112 | ~0.006 s | sin cargas: 110 calls = 0.0035 s (igual que la auditoría previa) |
> | `_spellcheck_correction` (réplica) | 8 | 0.001 s | largas ≥13: todas rápidas |
>
> **Hallazgos**:
> - **NO hay touchpoints de pyspellchecker fuera de `_ocr_spellcheck` en el
>   pipeline** — todo confluye en la línea 3119; el único uso externo es el
>   preload de server.py (es/en/pt), que ya vive en el arranque.
> - **El costo real por página de los touchpoints API es ~9 ms/pág en
>   total**, concentrado al 96 % en la pág 41 (el camino corto de las 2
>   palabras patológicas). El foreign check, la réplica y los candidatos
>   suman <0.03 s/capítulo; el build del índice (0.17 s) es one-time (cae
>   en el primer candidato — descubierto: el FILTRO d1 de palabras cortas
>   de `_ocr_spellcheck` llama `_spell_candidates` directamente, fuera de la
>   réplica; ahí aterriza el build del índice).
> - **NUEVO HALLAZGO ACCIONABLE: `langdetect` es el costo dominante de la
>   maquinaria spellcheck (2.95 s = 70 %; en la mayoría de páginas ≈ 100 %
>   de su spellcheck)**, NO pyspellchecker. La auditoría de 2026-08-15
>   concluyó que langdetect "no era el cuello" para la pág 4 (el profile
>   mostraba 26.2 s en correction); pero UNA VEZ corregido el corrector, el
>   residual del spellcheck por bloque es langdetect: 5.65 ms × 523 bloques
>   no-CJK. Candidato de optimización: cache por firma/página (patrón del
>   cache de negativas del VLM §8.4.1) — los bloques de una misma página
>   comparten idioma, y `_ocr_spellcheck` ya reutiliza el prefijo.
> - **VEREDICTO**: los touchpoints de pyspellchecker fuera del loop de
>   corrección NO justifican tocar nada (0.03 s/capítulo reales, todo lo
>   demás es one-time y ya está en el arranque). La palanca real que abre
>   este mapeo es langdetect (2.95 s), fuera del alcance de pyspellchecker
>   pero dentro de la maquinaria del spellcheck; A/B con el patrón de
>   firma antes de tocar. `benchmark_spell_touchpoints.py` queda como
>   herramienta de re-medición.

> **A/B DEL CACHE DE LANGDETECT (2026-08-16) — MEDIDO Y DESCARTADO EL
> CACHE POR FIRMA; el prefijo ya resolvió el costo** — se implementó el
> cache por muestra (`_SPELL_LANG_CACHE`, bounded 1024, en `_ocr_spell-
> check`; patrón del cache de negativas del VLM §8.4.1) y se midió con
> `benchmark_langdetect_cache.py` (capítulo completo, daemon detenido,
> `langdetect_cache.json`):
>
> | Métrica | Valor |
> |---|---|
> | Llamadas reales al cuerpo del detector (misses lru) | **82** (no 523) |
> | Textos únicos / firmas de caracteres únicas | 82 / **82** → **ahorro potencial 0 %** |
> | Tiempo real de langdetect en el capítulo | **1.05 s = 1.4 %** del wall (76.9 s) |
> | Firmas con resultado inconsistente dentro | 0 |
>
> **Lectura**: la medición previa de 2.95 s/523 llamadas era del estado
> ANTERIOR al prefijo de 600 chars; con el prefijo, el lru_cache de
> `_detect_language_robust` ya colapsa las muestras repetidas y solo 82
> textos únicos llegan al cuerpo (1.05 s). El cache por FIRMA de caracteres
> (perfil de frecuencias + flags de script) tiene **0 % de potencial**: los
> 82 textos OCR son todos firmas únicas, no hay colapso que explotar, y los
> 0 colapsos que existirían son consistentes. El cache por muestra
> implementado (clave = prefijo/texto exacto) es determinista y sin costo
> pero redundante con el lru_cache del detector en el corpus actual; se
> mantiene como red de seguridad para bloques repetidos y páginas largas
> (bounded 1024, sin crecimiento). **VEREDICTO: la palanca de langdetect
> queda cerrada — el costo residual (1.05 s = 1.4 % del capítulo) es
> estructural y no recortable por cache.** 6 tests nuevos en
> `TestSpellcheckLangPrefix` (reutilización por muestra, bound, no-
> contaminación de idiomas, fixture de cache limpio).

> **A/B: LÍMITE DE EDICIÓN DEPENDIENTE DE LA LONGITUD (2026-08-15)** —
> calibrado con el capítulo completo instrumentado
> (`benchmark_spellcheck_ab.py --collect`, 53 págs, daemon detenido,
> `spellcheck_ab_records.json`): el corrector solo aplicó **3 correcciones
> reales en todo el capítulo — todas de 3-5 chars y todas a DISTANCIA 1**
> ('unf'→'un', 'amf'→'amo', 'chmar'→'cimar'); la réplica (≥13 chars, 8
> llamadas) no corrigió NADA y sus palabras eran pegotes OCR >14 chars sin
> candidatos. NINGUNA corrección real usa distancia 2 ni afecta a >14 chars.
> Además, pyspellchecker ya hace "edición 1 primero, edición 2 solo si el
> nivel 1 está vacío" (candidates() en el fuente), así que la distancia 2
> solo se paga cuando no hay candidatos a distancia 1.
> **Schedule implementado** (`_spell_max_edits`): 3-5 → **1**, 6-14 → **2**
> (sin cambio), >14 → **1** — con una REFINACIÓN sobre la propuesta original
> (>14 → 0): el edit-0 pierde correcciones d1 legítimas de palabras largas
> reales fuera del diccionario (p. ej. 'inconstitucinal' → 'inconstitucional',
> 16 chars); permitir d1 y bloquear SOLO d2 las preserva y sigue eliminando
> las sugerencias arbitrarias de distancia 2 (p. ej. 'ncnstitucionalidad'
> (18, d1 vacío, d2=['constitucionalidad','inconstitucionalidad']) → ahora
> None; pyspell diría 'constitucionalidad'). En cortas (3-5), un filtro
> post-`sp.correction()` descarta correcciones a distancia 2 (cambiarían
> >50 % de los caracteres); fail-open ante checkers mockeados.
> **Medición (A/B del capítulo, `spellcheck_ab_after.json`)**: réplica
> **0.167 s → 0.001 s** (el scan d2 de las 8 palabras >14 desapareció; las
> 3 correcciones se mantienen — con PYTHONHASHSEED fijo el A/B palabra a
> palabra es IDÉNTICO antes==después; el flip observado de 'chmar'
> ('cimar'/'chamar'/'chiar' según corrida) es NO-DETERMINISMO PROPIO de
> pyspellchecker — empate de frecuencia 50/50/50, orden de set por hash
> seed, misma clase que 'defectivo'/'defectiva' ya documentado — no el
> efecto del schedule: los 3 son d1 y el filtro los acepta). Pipeline total
> 157.8 s → 138.8 s (varianza de las páginas trigger, no atribuible).
> Tests: `_spell_max_edits` (schedule por longitud), d1 preservada vs d2
> bloqueada en >14, filtro d1 de cortas (rechaza 'abcd'→'abx', acepta
> 'unf'→'un'), y el test de aislamiento del índice actualizado al nuevo
> schedule (par d1 de 14 chars). Artifacts: `spellcheck_ab_records.json` +
> `spellcheck_ab_after.json`.

> **MEDICIÓN MODO_CPU (2026-08-15, `UOCR_MODO_CPU=1`, daemon VLM detenido,
> 53 págs)** — la expectativa de velocidad del preset sin GPU dedicada
> (`launcher.py --cpu`), medida con `benchmark_production.py` a las dos
> escalas relevantes:
> | Escala | Promedio | Total | YOLO (CPU) | VLM (stage) | Bloques | Triggers |
> |---|---|---|---|---|---|---|
> | **1.2** (comparación con baselines) | **3.06 s/pág** | 161.9 s | 1.09 s/pág | 0.00 s | **298 = 298** | 11/53 |
> | **0.8** (escala real del preset al frontend) | **2.95 s/pág** | 156.2 s | 1.42 s/pág | 0.00 s | 335 | 16/53 |
> Lectura honesta:
> - **A 1.2 el preset es casi neutro** (+3.7 % vs baseline 2.95 s/pág, dentro
>   de varianza): YOLO forzado a CPU cuesta +0.34 s/pág promedio (pág 1:
>   6.15 vs 2.42 s) pero el VLM apagado elimina el coste de las páginas
>   trigger (etapa vlm 0.00 s; el contador "VLM llamado" del benchmark mide
>   la LLAMADA, que ocurre y retorna [], no la inferencia). **Recuperación
>   idéntica (298 = 298 bloques)** — YOLO en CPU produce las mismas regiones.
> - **A 0.8 (lo que el frontend usa con el preset) el promedio iguala al
>   baseline de 1.2 (2.95 s/pág)** — pero con 37 páginas de segmentación
>   distinta y +37 bloques totales (335 vs 298): es FRAGMENTACIÓN a baja
>   resolución (págs de 3 bloques pasan a 10-12), NO ganancia de contenido —
>   el conteo de bloques no es comparable entre escalas. El trigger además
>   dispara más (16 vs 11) por los features de escala (dark_ratio/panel).
> - **Caveat central**: esta máquina TIENE GPU (GTX 4 GB) y el preset NO
>   fuerza EasyOCR a CPU (degradación natural) — EasyOCR siguió en GPU en
>   ambas corridas. En una máquina realmente sin GPU, EasyOCR pasaría a CPU
>   (típicamente 3-10× más lento que GPU) y la expectativa real subiría:
>   esta medición es el PISO del preset en una máquina con GPU, no el
>   techo de una sin ella. El VLM apagado elimina el peor caso real (pág 21
>   a ~5 min con daemon activo — ver §3.3). Artifacts:
>   `production_modocpu_12.json` + `production_modocpu_08.json`.

> **CUADRO CONSOLIDADO DE VEREDICTOS DE LA RUTA C (2026-08-15, harness
> compartido benchmark_ab_utils.py, `--reps 3`, daemon VLM DETENIDO)** —
> re-corrida final de los 4 A/B de la Ruta C para cerrar veredictos con el
> ruido controlado:
> | A/B | Δ tiempo | Δ bloques | Noise-floor control | Veredicto |
> |---|---|---|---|---|
> | **pad 3 %** (production) | −0.08 s/pág (−3.0 %) | 49 = 49, crops 7 = 7 | 0.083 s | **ESTABLE — efecto atribuible, sin cambio** |
> | **interp** (CUBIC vs LINEAR) | −0.12 s/pág (−4.4 %) | 49 → 47 (−2) | **0.456 s** | **NO CONCLUYENTE — Δ dentro del noise** |
> | **rotation** (rotation_info) | −0.06 s/pág (−2.1 %) | 49 → 47 (−2) | **0.212 s** | **NO CONCLUYENTE — Δ dentro del noise** |
> | **upscale 3.5× vs 2×** | +0.115 s/pág (2× NO más rápido; deriva 0.116 s) | **49 → 47 (2× pierde 2)** | 0.116 s | **SE MANTIENE 3.5×** (sin beneficio de tiempo, pierde bloques) |
> | **recovery 3.5× vs 2×** | +0.002 s/pág (neutro) | **34 vs 32 (2× pierde 2, pág 11)** | 0.191 s | **SE MANTIENE 3.5×** (misma pérdida confirmada a reps 3) |
> | **batch estructural (strip)** | **−2.38 s/pág (8/8 págs con Ruta C)** | varianza (3 vs 4 textos) | sin control (todas afectadas) | **EFECTO GRANDE — confirmado, ya en producción** |
> Notas de calidad de la medición: el noise de control de interp/rotation
> (0.456/0.212 s) viene de la varianza CPU de las páginas sin Ruta C (rapid/
> prefilter — p. ej. pág 2: +0.46 s entre pares), no del VLM: los Δ de esos
> parámetros (≤0.12 s) están dentro del ruido → no hay señal concluyente a
> reps 3; si se quisiera cerrar el veredicto habría que re-correr con
> `--reps 5-7` o añadir más páginas de control. Lo que SÍ es concluyente:
> 2× pierde 2 bloques a tiempo neutro (upscale + recovery, consistentes en
> 7 y 14 págs) y el strip ahorra 2.4 s/pág — ambos ya reflejados en>   producción (3.5× y strip activos). Artifacts: `rutac_params_pad.json`,
> `rutac_params_interp.json`, `rutac_params_rotation.json`,
> `rutac_upscale_ab.json`, `rutac_recovery_ab.json`, `rutac_batch_ab.json`.

> **CAPÍTULO COMPLETO CON LAS TRES OPTIMIZACIONES JUNTAS (2026-08-15,
> `production_all3_full53.json`, daemon VLM ACTIVO — la única forma de medir
> la recuperación real de max_length=1280)**: corrida única de las 53 págs
> con 3.5× + strip + max_length 1280 + fixes de spellcheck en producción:
> **promedio 13.04 s/pág, total 691.0 s (~11.5 min)**, 11/53 triggers,
> **31 bloques recuperados por el VLM**, 329 bloques finales totales.
> Comparación honesta contra los dos baselines:
> | Baseline | Total | Promedio | Bloques | VLM rec. |
> |---|---|---|---|---|
> | strip daemon DOWN (`production_strip_full53`) | 162.5 s | 3.07 s/pág | 298 | 0 (VLM degradado) |
> | strip + FIX del diálogo daemon DOWN (`production_fix_full53`) | 175.7 s | 3.32 s/pág | — | 0 (VLM degradado) |
> | 1280_* daemon UP, 4 chunks (`production_1280_*`) | 1293 s | 24.4 s/pág | 321 | 31 |
> | **ACTUAL (3.5×+strip+1280, daemon UP, corrida única)** | **691 s** | **13.04 s/pág** | **329** | **31** |
> **DELTA EXACTO DAEMON UP vs DOWN (2026-08-16, consolidado de los JSONs
> existentes — mismo estado de producción 3.5×+strip+1280, mismo harness)**: el
> costo de la recuperación de diálogo artístico es el par `production_strip_full53`
> (162.5 s) vs `production_vlm_full53` (709.8 s): **Δ = 547.3 s (9.12 min) por
> 31 bloques recuperados (≈17.7 s/bloque)**. La suma de stages `vlm` de la
> corrida es 573.1 s (el delta exacto pedido; la diferencia de ~26 s con el
> Δ de wall-clock es el margen del trigger/degradación + varianza de las
> páginas no-VLM: 122.4 vs 119.2 s, Δ −3.2 s — el pipeline sin VLM es
> idéntico con o sin daemon). Contra el par alternativo all3 (691.0 s,
> stages vlm 541.8 s): Δ = 528.5 s. Rango consolidado: **528-547 s de
> wall-clock (542-573 s de stage VLM) ≈ 9 min/capítulo por 31 bloques**.
> Desglose del delta por página trigger (las 11 de `large_image_panel`):
> pág 21 +188 s/8 bloques (peor ROI, 23 s/bloque), 28 +57 s/6, 16 +54 s/5,
> 25 +45 s/4, 31 +37 s/4, 15 +81 s/1, 32 +27 s/1, 36 +21 s/1, 37 +22 s/1,
> 17 +21 s/0 (única llamada sin recuperación). El resto de páginas (no
> trigger) queda en ±1.4 s — el delta es 100 % VLM, sin costo oculto en el
> pipeline.
> Lectura: (1) el VLM recupera **31 bloques reales** (pág 21: 8, pág 28: 6,
> pág 16: 5, pág 25/31: 4 c/u) a costa de la inferencia (691 vs 162 s — el
> precio de la recuperación de diálogo artístico); (2) vs los chunks 1280_*
> la corrida actual es **−46 % (1293→691 s) con la MISMA recuperación (31)**
> — PERO parte de esa diferencia es varianza de trigger/VLM, no de las
> optimizaciones: pág 1 no disparó el trigger esta corrida (threshold_not_met,
> 207→22 s) y las llamadas VLM fueron más rápidas (pág 13: 87→34 s, pág 15:
> 115→72 s — generación estocástica), aunque pág 21 bajó de 196→173 s con
> el strip; (3) las páginas sin VLM se mantienen en el rango del baseline
> strip (1.2-6.7 s). max_length=1280 queda verificado end-to-end: misma> recuperación (31) que la corrida 1280_* con el daemon en el mismo estado.

> **RE-MEDICIÓN DEL COSTO REAL DEL VLM (2026-08-16, `production_vlm_full53.json`,
> daemon VLM READY desde el inicio, mismo corpus 53 págs, 3.5× + strip +
> max_length 1280)**: corrida única completa para aislar cuánto del capítulo
> es el VLM. Resultado: **promedio 13.39 s/pág, total 709.8 s (~11.8 min)**,
> 11/53 triggers, **10 llamadas VLM reales** (pág 13 quedó en el cache de
> negativas §8.4.1 esta corrida — antes llamaba VLM con 0 recuperación y
> 32-87 s; el cache la saltó sin perder nada), 31 bloques recuperados,
> suma de stages `vlm` = **573.1 s = 80.7 % del tiempo total del capítulo**.
> Desglose del VLM por página (todas `large_image_panel`):
>
> | Pág | Total | VLM stage | Recuperados |
> |---|---|---|---|
> | 15 | 84.0 s | 82.7 s | 1 |
> | 16 | 57.9 s | 55.2 s | 5 |
> | 17 | 26.0 s | 23.0 s | 0 |
> | 21 | 191.4 s | 190.2 s | 8 |
> | 25 | 50.4 s | 47.1 s | 4 |
> | 28 | 60.5 s | 59.3 s | 6 |
> | 31 | 40.0 s | 38.9 s | 4 |
> | 32 | 30.6 s | 29.4 s | 1 |
> | 36 | 24.7 s | 23.4 s | 1 |
> | 37 | 25.1 s | 23.9 s | 1 |
>
> **Lectura**: (1) el VLM es EL costo dominante del pipeline — 573 de 710 s
> (80.7 %) están dentro de la inferencia VLM; el resto del capítulo (137 s)
> coincide con el baseline sin VLM (162.5 s). (2) La pág 21 es el peor caso
> absoluto (191 s, 27 % del capítulo) y el que más recupera (8 bloques) — no
> es un candidato de recorte sin perder diálogo artístico real. (3) La pág 17
> es el único trigger que no recuperó nada (0 bloques) — candidato natural a
> evaluar si el gate puede filtrarla (pero fue 1 sola llamada de 23 s; el
> ahorro máximo sería ~3 % del capítulo, y la medición del gate de §4.6 ya
> mostró que afinar pierde recuperación en otras páginas). (4) La varianza
> entre corridas (all3: 691 s vs esta: 710 s, +3 %) es la generación
> estocástica del VLM + estado de GPU — no es una regresión de las
> optimizaciones: las páginas sin VLM se mantienen idénticas (2-6 s) y la
> recuperación es la misma (31). **Conclusión para el plan**: con el gate
> v4.2 actual, 80 % del tiempo del capítulo es VLM irrecuperable sin perder
> recuperación; la única palanca restante es estructural (más VRAM → batch,
> modelo más rápido, o corpus sin arte).
> **VALIDACIÓN CON EL HARNESS CORREGIDO (2026-08-16, re-corrida completa
> `production_vlm_full53_2.json`, mismos 53 págs, daemon ready, wrappers NO
> encadenados — `_ORIG_FN` cachea la original una vez)**: total 1026.5 s
> (19.37 s/pág, más lenta que la corrida anterior por generación
> estocástica), trigger 11/53, 11 llamadas VLM (pág 13 esta vez llamó con
> 0 recuperación: el cache de negativas es por sesión), **31 bloques
> recuperados = IDÉNTICO**. Suma de stages `vlm` = 761.4 s = 74.2 % del
> total — el porcentaje baja 6.5 pp SOLO porque la generación fue más lenta
> (761.4 vs 573.1 s de stage; el resto del capítulo, 265 s, coincide con el
> baseline sin VLM 162.5-175.7 s). **Veredicto: la re-medición de la
> participación del VLM (573-761 s, 74-81 % según la velocidad de
> generación de la corrida) NO cambia por el harness — el fix de los
> wrappers eliminó las llamadas fantasma (que inflaban los contadores de
> prefilter/rapid, no los de `vlm`, cuyo stage ya era correcto), y la
> varianza entre corridas es de la generación estocástica del modelo, no
> del instrumento de medición.** La conclusión estructural se mantiene: el
> VLM domina el capítulo y el residual sin VLM (~3.4 s/pág) ya está
> analizado por separado.
> **VARIANZA REAL DEL VLM — 3 CORRIDAS COMPLETAS CON EL MISMO HARNESS
> (2026-08-16: vlm_full53, vlm_full53_2, vlm_full53_3, las 3 con
> benchmark_production.py y wrappers NO encadenados)**:
>
> | Corrida | Total | VLM stage | % capítulo | No-VLM (42 págs) | Recuperados |
> |---|---|---|---|---|---|
> | 1 | 709.8 s | 573.1 s | 80.7 % | 119.2 s | 31 |
> | 2 | 1026.5 s | 761.4 s | 74.2 % | 242.8 s | 31 |
> | 3 | 1318.2 s | 627.5 s | 47.6 % | 650.9 s | 31 |
>
> **VLM stage: min 573.1 / max 761.4 / mediana 627.5 s (CV 15 %)** —
> razonablemente estable (generación estocástica del modelo).
> **Recuperación: 31 = 31 = 31 (determinista, cero varianza)** — el mismo
> recuento por página en las 3 corridas (ver bloque ROI).
> **% del capítulo: min 47.6 / max 80.7 / mediana 74.2 %** — la varianza
> del porcentaje NO es del VLM: el denominador cambia porque las páginas
> no-VLM se ralentizaron 119 → 243 → 651 s entre corridas (2.8 → 5.7 →
> 15.5 s/pág) por estado de la máquina (GPU/memoria/térmica tras corridas
> consecutivas), no por el pipeline (mismas etapas, mismos bloques).
> **Conclusión con intervalo**: el VLM aporta 573-761 s estables por
> capítulo (mediana 627 s) y recupera EXACTAMENTE 31 bloques siempre; su
> participación del total cae en 48-81 % según lo rápido que corra el resto
> de la máquina en esa sesión — el "80.7 %" inicial es el extremo de
> máquina rápida, el ~74 % la mediana, y el piso ~48 % con la máquina
> degradada. La conclusión estructural (el VLM es el costo dominante y la
> recuperación es estable) se mantiene en las 3 corridas.

> **ANÁLISIS DEL TIEMPO RESIDUAL SIN VLM (2026-08-16,
> `production_nonvlm_fixed.json`, 42 págs sin VLM del cap. 43, harness
> CORREGIDO)**: tras cuantificar que el VLM es 80.7 % del capítulo, se aisló
> el residual (páginas de 2-6 s) para priorizar la próxima optimización.
> Primero un FIX DE MEDICIÓN: los wrappers de `benchmark_production.py` se
> encadenaban por página (cada `_wrap_timing` envolvía al wrapper de la
> página anterior), inflando los contadores de las páginas tempranas (pág 1
> reportaba 53 calls de prefilter y 10 de VLM en una corrida de 53). Ahora
> cachean la función original una vez (`_ORIG_FN`) y `calls` se copia por
> valor. **Desglose limpio del residual (141.7 s / 42 págs = 3.37 s/pág)**:
>
> | Etapa | Tiempo | % | Dónde corre | ROI |
> |---|---|---|---|---|
> | `ruta_c` (re-OCR de crops) | 49.4 s | 34.8 % | 31/42 págs | 0.29 s/bloque (vía YOLO) |
> | `yolo` (detección + su Ruta C) | 44.8 s | 31.6 % | 31/42 (gate superado) | **155 bloques, 0.29 s/bloque** |
> | `ctd` (comic-text-detector + su Ruta C) | 30.8 s | 21.8 % | 16/42 (cascada) | **21 bloques NUEVOS, 1.47 s/bloque** |
> | `prefilter` (inpaint TELEA fijo) | 22.2 s | 15.6 % | 42/42 | fijo 0.53 s/pág (**ESTRUCTURAL** — 4.4B y 4.4C medidas y descartadas; el inpaint limpia ruido distribuido, no líneas; ver §4.4) |
> | `rapid` (RapidOCR complemento) | 17.2 s | 12.1 % | 42/42 | fijo 0.41 s/pág |
> | `_ocr_spellcheck` (merge) | 0.16-1.38 s/pág | 2.5-8 % | por bloque | ya optimizado (ms/bloque) |
>
> (La suma > 100 % porque `ruta_c` está ANIDADA dentro de `yolo`/`ctd` —
> los métodos las llaman internamente, ocr_engine.py:734/804). **Lectura**:
> (1) **El re-OCR de crops de la Ruta C domina el residual** — YOLO + CTD
> juntos son 75.6 s (53 %) y ambos alimentan el mismo `_recover_regions_with_easyocr`;
> el batch estructural ya aplicado (−2.38 s/pág) fue la optimización correcta
> y la que más pesa. (2) **CTD es 5× más caro por bloque que YOLO**
> (1.47 vs 0.29 s/bloque): 30.8 s por 21 bloques nuevos reales (todos
> sobreviven al merge — texto SIN globo que ni YOLO ve, complementario, no
> duplicado). Su gate en cascada ya lo limita a 16/42 págs, pero sigue siendo
> el peor ROI del residual. (3) **El spellcheck del merge ya NO es cuello**
> (0.16-1.38 s/pág, 2.5-8 % — la pág 4 con textos largos es el peor caso;
> el fix de 2026-08-15 lo dejó en ms por bloque). (4) `prefilter` y `rapid`
> son costes fijos por página (0.53 + 0.41 s) — el inpaint TELEA del prefilter
> es la vía 4.4C documentada (detector de líneas más selectivo, NO bajar
> resolución — A/B de 0.5× regresionó pág 11).
>
> **PRIORIDAD DE LA PRÓXIMA OPTIMIZACIÓN** (por ROI y riesgo):
> 1. **Afinar el gate de CTD** (vía con datos ya medidos): cuesta 30.8 s por
>    21 bloques. Opciones medibles con `benchmark_ab_utils.py`: exigir página
>    aún más débil (p.ej. subir `COMIC_DETECTOR_GATE_MAX_CONF` de 0.35 a 0.30
>    o bajar `MIN_BLOCKS` de 3 a 2) o dedup más agresivo de regiones ya
>    cubiertas por YOLO. Riesgo: perder los 21 bloques de texto flotante —
>    medir A/B antes de tocar, nunca con datos ambiguos.
> 2. **prefilter 4.4C** (22.2 s fijos): detector de líneas horizontales más
>    selectivo para reducir el área de la máscara del inpaint TELEA
>    (0.53 s/pág → objetivo ~0.3 s). A/B con las 14 páginas canónicas y el
>    control de la pág 11 (la que regresionó en el intento de 0.5×).
> 3. **YOLO: NO tocar el gate** — supera el gate en 31/42 páginas pero
>    recupera 155 bloques reales (0.29 s/bloque, el mejor ROI del residual);
>    apretarlo perdería recuperación de diálogo en globos.
> 4. **spellcheck del merge: NO tocar** — ya en ms/bloque; el peor caso
>    (pág 4, 1.38 s) es un único bloque largo, no un patrón.
> Artifacts: `production_nonvlm_fixed.json` + `.log`, `merge_stages.json`.

### 4.7. Re-medir el split detector/recognizer si cambia el corpus (disparador de 4.1)

El veredicto de 4.1 (detector CRAFT domina 72.9 %) se midió sobre el corpus
es→en. Si el proyecto añade origen ja/ko/zh masivo, re-correr
`benchmark_ocr_stages.py`: el recognizer puede pasar a dominar y ahí SÍ
justificaría recortar cajas de EasyOCR → reconocer con RapidOCR (PP-OCRv4).
Esta fase es condicional, no un trabajo fijo.

---

## 5. Puntos de la investigación externa (referencias)

- **EasyOCR batch_size / parámetros**: documentación oficial de
  `readtext` (canvas_size, mag_ratio, text_threshold) —
  https://www.jaided.ai/easyocr/documentation/ — y benchmarks de la comunidad
  sobre `batch_size` en el recognizer.
- **EasyOCR GPU en paralelo vs secuencial**: issue #534 (secuencial ~20% más
  rápido que paralelo por página) — https://github.com/JaidedAI/EasyOCR/issues/534.
- **OCR engines 2025–2026**: PaddleOCR es el más preciso en general, RapidOCR
  el mejor CPU, Surya el mejor para layout —
  https://codesota.com/ocr/best-for-python · https://modal.com/blog/8-top-open-source-ocr-models-compared.
- **Manga-specific**: PaddleOCR-VL-for-manga (70% exactitud frase-completa vs
  base) — https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga ·
  manga-ocr (kha-white) es para JA (no aplica a es→en).
- **CTranslate2**: batching/`translate_batch`, cuantización int8/int8_float16,
  "4–6× speed gains" — https://github.com/opennmt/ctranslate2 ·
  https://kareemai.com/blog/posts/minishlab/ctranslate_maswray.html.
- **OpenCV inpainting**: TELEA vs NS (este repo ya usa TELEA + border-blend;
  validado como la opción rápida con buena calidad) —
  https://learnopencv.com/image-inpainting-with-opencv-c-python/.
- **Flask en producción / Windows**: waitress es la opción estándar en
  Windows; el 404 de catch-all+blueprint en Flask 3.x se resuelve registrando
  el catch-all en la app raíz — docs Flask (deploying/waitress) y
  https://stackoverflow.com/questions/50670996/flask-blueprint-404.
- **Compresión de respuestas**: Flask-Compress (gzip/deflate/brotli/zstd) —
  https://pypi.org/project/Flask-Compress/.
- **Base64 vs binario**: base64 +33% de tamaño — documentos del Web
  Platform/API design; `canvas.toBlob` es el reemplazo recomendado de
  `toDataURL`.
- **LLM/VLM inference**: KV cache, flash attention, cuantización 4-bit y
  `max_length` como palanca directa de latencia —
  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/.
- **PDF.js**: render en main thread + Web Workers internos; no renderizar más
  de ~25 páginas a la vez (relevante para el prefetch) —
  https://github.com/wojtekmaj/react-pdf/discussions/1691.

### 5.1. Veredicto con datos: PaddleOCR y Windows OCR API NO reemplazan el pipeline

Evaluación del 2026-08-14, con mediciones tomadas en vivo en esta máquina
(Windows 11 Pro 26200, Python 3.12.10, GTX 1050 Ti 4 GB). Conclusión: **ninguna
de las dos opciones reemplaza el pipeline híbrido actual; ambas están
descartadas como reemplazo del recognizer.**

**PaddleOCR — redundante y roto en el venv:**

| Dato medido | Valor | Implicación |
|:------------|:------|:------------|
| `paddle` 3.3.1 instalado | 392 MB en `env/Lib/site-packages` | Paquete pesado sin uso |
| Tiempo de `import paddle` | **53.8 s** | Coste de arranque inaceptable (el daemon/workers pagarían 54 s por proceso) |
| `paddleocr` 3.7.0 instalado | **Roto**: `ModuleNotFoundError: No module named 'aistudio_sdk'` al importar | No funciona tal cual está; requeriría arreglar la instalación |
| `paddlex` 3.7.2 instalado | 19 MB, dependencia de paddleocr | Sin uso propio |
| Modelos | RapidOCR **ya usa los mismos PP-OCRv4** vía ONNX Runtime (sin PaddlePaddle) | Cambiar a PaddleOCR = el mismo recognizer con un framework de 412 MB encima |
| Split medido (Fase 3/4.1) | Detector CRAFT = **72.9 %** del coste | PaddleOCR solo cambiaría el recognizer → no mueve la aguja |
| Bundle PyInstaller | `main.spec` ya excluye `paddle` | Se cargaría desde `env/` en runtime igual que el resto — sin beneficio |

Acción tomada: `paddlepaddle`, `paddleocr` y `paddlex` fueron **desinstalados
del venv (412 MB liberados)** el 2026-08-14; el código nunca los importó
(solo menciones en comentarios/docs) y el CI completo quedó verde (13/15,
944/944 tests) tras la limpieza.

**Windows OCR API (`Windows.Media.Ocr`) — técnicamente presente, pero sin
japonés y de calidad inferior para manga:**

| Dato medido | Valor | Implicación |
|:------------|:------|:------------|
| Motor disponible | `OcrEngine.TryCreateFromUserProfileLanguages()` → OK | La API existe en este Windows 11 |
| Idiomas instalados | **solo `es-ES` y `es-MX`** | No sirve para manga japonés sin instalar el paquete de idioma `ja` |
| `TryCreateFromLanguage('ja')` | **FAIL** (idioma no instalado en el SO) | El usuario final tendría que instalar el paquete de idioma manualmente — requisito frágil para una app empaquetada |
| Acceso desde Python | requiere `winrt`/`winsdk` (no instalado) | Dependencia nueva + integración con PyInstaller |
| Calidad | Motor de texto de documento (horizontal, impreso, limpio) | Manga = texto vertical/rotado/artístico sobre arte — el caso donde rinde peor; sin entrenamiento, sin GPU |
| Semántica de salida | líneas rectas, sin regiones/rotación libre | No alimenta el pipeline de bloques/burbujas actual |

**Razón de fondo (ambas):** el cuello de botella medido es el **detector
CRAFT (72.9 %)**, y ninguna de las dos opciones compite ahí (ni siquiera
existen como detector de esa calidad); ambas solo tocarían el recognizer, que
no es el limitante. El veredicto queda documentado para no re-evaluarlo sin
un cambio de corpus (p. ej. CJK) o de hardware (2+ GPUs).

---

## 6. Métricas de éxito (definición de "hecho")

> **⚠️ SESGO DE RESOLUCIÓN CORREGIDO (2026-08-14).** Las métricas previas de
> esta sección se medían a **300 dpi** (2480×3509 px, 8.7 MP por página) —
> pero el frontend envía el canvas a **pdf.js scale 1.2** (714×1011 px,
> 0.7 MP, **8 % del área**) — ver `app.js::state.scale` y `benchmark_production.py`.
> Todo lo que "costaba 5–15 s/pág" era un artefacto de medir a resolución
> 12.5× mayor que la de producción. **Las métricas de esta sección se miden
> AHORA a la resolución real de producción** (scale 1.2) con
> `benchmark_production.py`, salvo que se indique lo contrario. Las columnas
> de resolución se mantienen solo donde el objetivo aplica en cualquier
> resolución (payload, VLM, cache).

Sobre el PDF de 53 págs (`Capítulo 43 …pdf`), modo `fusion`, misma máquina,
**resolución real de producción (pdf.js scale 1.2, 714×1011 px)**:

| Métrica | Hoy (aprox., medido) | Objetivo Fase 1 | Objetivo Fase 2 | Objetivo Fase 4 |
|:--------|:---------------------|:----------------|:----------------|:----------------|
| Página sin Ruta C (texto limpio) | **1.0–2.0 s** (18/53 del capítulo, promedio 2.03 s, mediana 1.43 s) | **< 1.5 s** | — | — |
| Página con YOLO→Ruta C | **post-strip: promedio 4.3 s** (35/53 del capítulo; las 8 pesadas de 6-14 s bajaron de 68.8 s a 48.4 s, −29.5 % — ver §4.6) | — | **< 6 s** (upscale 3.5× tras revert — ver §4.6) | — |
| Promedio por página (capítulo completo) | **3.07 s/pág** (53 págs, re-corrida post-strip 2026-08-15, daemon VLM detenido; baseline pre-strip: 3.99 s/pág → −23.2 %) | **< 3.5 s** ✅ | — | — |
| Payload de respuesta (inpainted) | MB (PNG) | < 400 KB (JPEG) | — | — |
| Re-corrida del mismo capítulo | tiempo completo | — | **< 10 % del original** (cache de página) | — |
| Página VLM (p90) | **24–315 s/llamada** (mediana ~50 s; 9 llamadas en cap. 43; el "~2 s" previo era EOS temprano, no el costo real) | — | **< 30 s/llamada** (3.3: recortar max_length/tokens) | — |
| Cobertura de detección | 47/53 págs | ≥ igual | ≥ igual | ≥ igual |
| Calidad (`analisis_calidad.py`) | **75.8 % aceptable** (corpus 128 págs de Julio, condiciones distintas — ver nota) | ≥ igual | ≥ igual | ≥ igual |

> **RE-MEDICIÓN DE CALIDAD POST-FIX DEL STRIP (2026-08-16):** se regeneró el
> corpus del cap. 53 con el pipeline ACTUAL (fix del strip activo, daemon
> detenido, `process_all_pages.py` fusion → `resultados_progreso_calidad_fix.json`)
> y se corrió `analisis_calidad.py`: **314 pares (vs 225 del checkpoint viejo)
> con 32.2 % aceptable (vs 15.1 %)** — el delta de +89 pares y +17 pp viene
> de que el corpus fresco incluye los **4 diálogos recuperados por el fix**
> (verificados presentes y traducidos: pág 4 'YSIHUBIESE ACEPTADO.' →
> 'AND IF I HAD BEEN ACCEPTED.', pág 40 'ESTOESTA IPADRINO!! MUYRICO!' →
> 'THIS IS IPADRINO!!', pág 52 'QUIERESVOLVERA EENVERDAD COMERLASHOY?' →
> 'YOU WANT TO COME BACK, EENVER?'; pág 1 en este run recuperó fragmentos
> 'RIMER ARAY'/'#' — el diálogo completo 'ENVERDADES INCREIBLE' se confirmó
> en el A/B mismo-proceso). **NO comparable con el 75.8 % de Julio**: ese
> baseline es del corpus 128 págs con pipeline de traducción distinto (8.3 %
> UNTRANSLATED); el cap. 53 actual tiene 96/314 UNTRANSLATED (30.6 %) y
> 103 REVIEW_LANGUAGE (32.8 %) — bloques no traducibles (ruido/URLs/
> fragmentos) que la tasa arrastra. El fix del strip no mueve la tasa de
> aceptación (mide traducción, no recuperación OCR): su impacto es que esos
> 4+ diálogos EXISTEN como pares traducidos en el corpus. Artifacts:
> `resultados_progreso_calidad_fix.json`, `benchmark_results/calidad_fix_run.log`,
> `benchmark_results/server_calidad.log`.
| **Pre-filter** (`benchmark_prefilter.py`, scale 1.2) | **0.4–0.5 s/pág** (a 300 dpi era 1.41 s — sesgo) | — | — | **< 0.3 s/pág** (4.4B/4.4C — AMBAS MEDIDAS Y DESCARTADAS: el inpaint limpia ruido distribuido, sin líneas que aislar; el coste es estructural, ver §4.4) |
| **RapidOCR en páginas normales** (scale 1.2) | **0.4–1.0 s** cuando corre | — | — | omitirlo en más páginas sin perder bloques (4.5) |
| **Ruta C por crop** (upscale 2×, scale 1.2) | **0.2–0.9 s/crop**; gate intra-crop: NO procede (§4.6) | — | — | — |
| **Trigger VLM (v4.2)** | **11/53 (21 %)** en cap. 43, todas por panel grande oscuro (`large_image_panel`, dark_ratio 0.181–0.218); VLM recupera 1–8 bloques en TODAS las páginas donde corre (32 total); costo **24–315 s/llamada** (mediana ~50 s, ~12 min/capítulo) | — | — | — |

> **CORRECCIÓN 2026-08-15**: el "0/12 págs trigger" previo era un **bug de
> medición del benchmark**, no un resultado real — leía `diag.trigger` (atributo
> inexistente; el diagnóstico guarda el dict `_trigger`) y lo leía DESPUÉS de
> que `run_ocr` restaura los diagnostics al valor previo. Corregido
> instrumentando `_trigger_con_cache` (única llamada por página en modo
> fusion) + contadores de VLM/rapid-agresivo/YOLO/CTD en
> `benchmark_production.py`. El número real: **11/53 páginas disparan el VLM**
> (todas por `large_image_panel`, ninguna por confianza baja — las 42
> restantes quedan en `threshold_not_met`). El rapid-agresivo no salva
> ninguna de las 11 (guard `has_big_panel` → devuelve False por diseño).
>
> **CORRECCIÓN DE COSTO (2026-08-15)**: el "~2 s/llamada" documentado antes
> era la ANOMALÍA (el modelo cortaba en EOS temprano en esa corrida), no el
> costo real. Con el daemon reiniciado y midiendo la recuperación, las 9
> llamadas al VLM costaron **23.6–314.8 s** (mediana ~50 s; pág 21: 314.8 s =
> el peor caso de 2–8 min documentado). El costo es proporcional a los tokens
> generados (max_length 2048 en la 1050 Ti ≈ 1–5 min cuando el modelo
> genera largo) — la palanca real es el VLM mismo (Fase 3.3: recortar
> `max_length`/tokens), NO el gate.

Cada fase se cierra con: benchmarks ANTES/DESPUÉS registrados en un JSON nuevo
(`benchmark_optim_<fecha>.json`), **a la resolución real de producción
(scale 1.2, `benchmark_production.py`)** — nunca solo a 300 dpi —,
`run_ci.py` completo verde (944+ tests), y sin regresión de cobertura de
tests por módulo.

---

## 7. Orden sugerido de ejecución (dependencias)

```
Fase 1 (1–3 días)            Fase 2 (2–4 días)            Fase 3 (1 sem+)          Fase 4 (2–4 días)
├─ 2.1 RapidOCR condicional  ├─ 3.1 servidor producción   ├─ 4.1 evaluar recognizer ├─ 4.4B inpaint res. reducida
├─ 2.2 EasyOCR batch/cudnn   ├─ 3.2 compresión respuestas ├─ 4.2 multi-proceso (solo si) ├─ 4.5 afinar rapid_cond_skip
├─ 2.3 JPEG salida           ├─ 3.3 VLM max_length        ├─ 4.3 offscreen worker (baja) ├─ 4.6 gate Ruta C en normales
├─ 2.4 binario en vez de b64 ├─ 3.4 prefetch frontend                                        └─ 4.7 re-medir si corpus CJK
├─ 2.5 cap de escala OCR     └─ 3.5 YOLO imgsz 1024
└─ 2.6 CT2 greedy+batch
└─ 2.7 cache de página
```

Dependencias: 2.3/2.4 juntos (cambian el contrato de la imagen en la API —
actualizar `routes/api.py`, `app.js`, `js/utils.js`, `test_packaging.py`).
2.5 antes de 2.4 (define el tamaño del canvas a enviar). 2.7 después de 2.1–2.3
(la clave del cache incluye el modo y el formato). 3.2 después de 2.4 (si la
imagen sale del JSON, la compresión queda para el texto puro). 4.4B primero en
Fase 4 (valida con A/B y no toca el pipeline); 4.4A solo si la validación de
4.4B es limpia y se quiere más (riesgo alto: trigger v4.2). 4.5 y 4.6 después
de 4.4B (ambos mueven la frontera de coste condicional y comparten la
validación con `analisis_calidad.py`).

---

## 8. No-goals (explícitamente fuera de alcance)

- Reescribir el pipeline en Rust/Go o cambiar de framework web.
- Entrenar/fine-tunear modelos propios (OCR o traducción) — se asume usar
  modelos existentes.
- Soportar GPU AMD/Apple Silicon (solo NVIDIA/CUDA y CPU, como hoy).
- Cambiar la UX/UI del editor de burbujas (solo rendimiento).
- Traducir con modelos LLM grandes online (privacidad + latencia; el pipeline
  offline CT2 es un feature).
- Adoptar PaddleOCR (redundante con RapidOCR, import 53.8 s, 412 MB, roto en
  el venv) ni Windows OCR API (sin japonés instalado, calidad inferior para
  manga) como reemplazo del pipeline — veredicto con datos en §5.1.

---

## 9. Inventario de archivos que tocará el plan

| Archivo | Cambios |
|:--------|:--------|
| `ocr_engine.py` | 2.1 RapidOCR condicional; 2.7 cache de página (o en `cache.py`) |
| `ocr_utils.py` | 2.2 batch_size/cudnn; 2.3 `_cv2_to_base64(fmt)`; 2.4 decode binario |
| `translator.py` | 2.6 CT2 greedy + `translate_batch` agrupada |
| `routes/api.py` | 2.3/2.4 contrato de imagen; 2.6 agrupar textos; 2.7 cache |
| `server.py` | 3.1 servidor producción; 3.2 Flask-Compress |
| `uocr_daemon.py` | 3.3 max_length / tokens de salida |
| `process_all_pages.py` | 2.4 binario (request); 2.7 re-corrida |
| `app.js` / `js/utils.js` | 2.4 toBlob; 2.5 cap de escala; 3.4 prefetch |
| `config.py` | 2.2 flags; 3.5 `YOLO_IMGSZ`; constantes nuevas de cache/JPEG; 4.5 umbrales de `RAPID_COND_*` |
| `ocr_utils.py` | 2.2 batch_size/cudnn; 2.3 `_cv2_to_base64(fmt)`; 2.4 decode binario; **4.4B inpaint a resolución reducida (descartada)**; **4.4C detector de líneas selectivo (descartada)**; **4.5 `_rapid_cond_skip`** |
| `benchmark_prefilter_selective.py` (nuevo) + `benchmark_results/prefilter_selective.json` | **A/B de 4.4C (2026-08-16, 14 págs canónicas, scale 1.2)**: detector de líneas reales marca 0.006 % del área (no hay líneas en el corpus); quitar el inpaint de líneas = 0.48 → 0.02 s/pág pero +44 bloques de ruido basura y sin ganancia en la pág 11 de control — **descartada, coste estructural** — ver §4.4 |
| `benchmark_prefilter_4_4a.py` (nuevo) + `benchmark_results/prefilter_4_4a.json` | **A/B de diseño de 4.4A (2026-08-16, capítulo completo, raw vs pref)**: 9/53 págs débiles (n<6 Y conf<0.6), 13/53 donde el prefilter gana de verdad; **0/53 páginas cambian su trigger v4.2** (riesgo central medido y cero); ahorro ~14-23 s/capítulo (< 2 %) — **NO procede** (doble pasada OCR en débiles, coste estructural queda) — ver §4.4 |
| `ocr_engine.py` | 2.1 RapidOCR condicional; 2.7 cache de página (o en `cache.py`); **4.6 gate de la Ruta C en páginas normales** |
| `tests/*` | tests para cada cambio (el repo exige cobertura por módulo ≥ umbral) |
| `benchmark_prefilter.py` (nuevo) | A/B del prefilter por sub-etapa + calidad on/off (Fase 4) |
| `benchmark_detect_stages.py` (nuevo) | desglose por etapa de `_detect_and_ocr` + costo por crop de la Ruta C (Fase 4) |
| `benchmark_*.py` (nuevo `benchmark_optim.py`) | harness ANTES/DESPUÉS estandarizado |
| `env/` (limpieza 2026-08-14) | desinstalar `paddlepaddle`/`paddleocr`/`paddlex` (412 MB, rotos, sin uso — ver §5.1) |
| `ocr_utils.py` / `ocr_engine.py` / `config.py` | **2026-08-14: upscale de la Ruta C 3.5× → 2× aplicado** (A/B: 65 = 65 bloques, conf idéntica, −13–24 % — ver §4.6). **2026-08-15: REVERTIDO a 3.5×** — el A/B del 2× estaba roto (wrapper anidado → midió 3.5 vs 3.5); re-medido con el benchmark corregido: 3.5× recupera 34 vs 32 bloques (−2, pág 11) a +0.03 s/pág — ver §4.6 |
| `benchmark_ab_utils.py` (nuevo) | **harness A/B anti-deriva compartido (2026-08-15)**: intercalado por página + orden alternado + páginas de control/noise-floor + veredicto; estandarizado desde benchmark_rutac_params.py y usado por upscale/recovery — ver §4.6 |
| `benchmark_rutac_upscale.py` (nuevo) | A/B del upscale de la Ruta C (3.5× vs 2×) a resolución de producción (pdf.js scale 1.2). **2026-08-15: corregido** — parcheo sin anidamiento (original capturado una vez) + harness anti-deriva (--reps, controles, veredicto) — ver §4.6 |
| `benchmark_spellcheck_ab.py` (nuevo) | **calibración del límite de edición por longitud (2026-08-15)**: `--collect` corre el capítulo instrumentado (grabando palabra→corrección y camino) y `--analyze` evalúa schedules contra preservación — ver §4.6 |
| `benchmark_rutac_recovery.py` (nuevo) | verificación texto-a-texto de la recuperación de la Ruta C por upscale. **2026-08-15: corregido** — el wrapper ahora FUERZA el upscale (antes pasaba el del caller = 2.0 y comparaba 2.0 vs 2.0) + harness anti-deriva — ver §4.6 |
| `benchmark_production.py` (nuevo) | benchmark del pipeline real a scale 1.2; **2026-08-15: instrumentación del trigger v4.2 corregida** (vía `_trigger_con_cache` — el “0/12” previo era un bug de medición) + contadores VLM/rapid-agresivo/YOLO/CTD |
| `benchmark_results/production_full53.json` | **corrida completa del cap. 43 (2026-08-15)**: 53 págs, promedio 3.99 s/pág, trigger VLM 11/53 (panel grande), Ruta C en 35/53 (promedio 5.0 s) — ver §6 |
| `ocr_utils.py` | **2026-08-15: pad del crop de la Ruta C 6% → 3% aplicado** (A/B 14 págs: 107 = 107 bloques, textos idénticos, −23.6 % — ver §4.6; **re-corrida 2026-08-15 con daemon detenido: el −23.6 % era deriva — pad neutro en tiempo**) + constantes `_RUTA_C_PAD_*`/`_RUTA_C_INTERP` para A/B por monkeypatch |
| `benchmark_rutac_params.py` (nuevo) | A/B de parámetros del crop de la Ruta C (pad, interpolación, rotation_info, rapid_box/unclip/batch) a resolución de producción. **2026-08-15: corregida la deriva de orden** — intercalado base/alt por página con orden alternado (par b→a, impar a→b), páginas de control automáticas (sin etapa Ruta C) con noise-floor (máx \|Δ\|), veredicto explícito (atribuible / cautela / NO CONCLUYENTE) y `--reps N` (default 2, mediana); rec_batch_num mutado in-place sobre el engine (sin rebuild) |
| `benchmark_results/rutac_params_ab.json` / `rutac_pad_ab14.json` | resultados del A/B de parámetros (pad validado; INTER_LINEAR y rotation_info (0,180) descartados con datos) |
| `benchmark_results/rutac_params_reps3.json` + log | **re-corrida de pad/box_thresh/unclip con daemon VLM detenido y --reps 3 (2026-08-15)**: noise-floor 0.018-0.022 s (25× menor) — pad neutro (+0.2 %, 47 = 47 bloques), box_thresh 0.35 −1 bloque, unclip 2.2 −4 bloques; los defaults (0.03/0.5/1.6) CONFIRMADOS; los porcentajes previos eran deriva — ver §4.6 |
| `benchmark_results/rutac_params_pad.json` / `rutac_params_interp.json` / `rutac_params_rotation.json` | **consolidación final de veredictos de la Ruta C (2026-08-15, --reps 3, daemon detenido)**: pad estable (−3.0 %, 49=49); interp/rotation NO CONCLUYENTES (noise 0.456/0.212 s > Δ); upscale/recovery confirman 3.5× (2× pierde 2 bloques a tiempo neutro); batch strip −2.38 s/pág — ver §4.6 |
| `benchmark_results/gate_vlm_1/2/3.json` | **reevaluación del gate de panel grande oscuro (2026-08-15)**: 11/53 trigger, VLM recupera 1–8 bloques en las 9 páginas medidas (32 total), costo 24–315 s/llamada — gate NO ajustable sin perder recuperación (ver §4.6) |
| `benchmark_results/rutac_rapid_ab.json` / `rutac_batch_ab.json` | **A/B de parámetros del pase rapid de la Ruta C (2026-08-15)**: box_thresh 0.35, unclip 2.2 y rec_batch_num 16 DESCARTADOS con datos — los defaults (0.5/1.6/6) son el óptimo (ver §4.6) |
| `ocr_utils.py` | **2026-08-15: constantes parcheables del rapid de la Ruta C** (`_RUTA_C_RAPID_BOX_THRESH`, `_RUTA_C_RAPID_UNCLIP_RATIO`, `_RAPID_REC_BATCH_NUM`) — neutras (defaults = comportamiento histórico) para A/B por monkeypatch |
| `benchmark_results/rutac_upscale_ab.json`, `rutac_recovery_ab.json`, `rutac_upscale_ab14.json`, `rutac_recovery_ab14.json`, `rutac_gate3_diag.json` | mediciones del A/B de upscale y del diagnóstico de gate intra-crop. **2026-08-15: re-medición corregida (--reps 3, daemon detenido)** — 7 págs: 34 vs 32; **14 págs: bloques finales 105 vs 100 y recuperados 90 vs 85 (−5)**; tiempo neutro en ambas — ver §4.6 |
| `benchmark_vlm_maxlen.py` (nuevo) | A/B del `max_length` del VLM (2048 vs 512/1024/1280) con parche por wrapper sobre `process_page` (el atributo de módulo NO surte efecto — default capturado en la firma) |
| `benchmark_vlm_tokens.py` (nuevo) | **distribución de tokens del VLM (2026-08-15)**: envuelve `generate()` en proceso (daemon parado) para contar tokens nuevos por página — 8/9 páginas generan 32–376 tokens, SOLO la 21 llega a 1949 (97 % del cap) — ver §3.3 |
| `benchmark_results/vlm_tokens_a/b/c.json`, `vlm_tokens_dist.json` | mediciones crudas y distribución consolidada de tokens por página del trigger |
| `benchmark_results/vlm21_1024.json` + `vlm21_1280.json` | **aislamiento de pág 21 (2026-08-16, 1024 vs 1280, daemon real)**: 1024 → 167.0 s/7 rec (textos idénticos a 1280; la diferencia es otro `[Unreadable]`), 1280 → 177.1 s/8 rec; bajar el cap no ahorra nada útil — estructura del costo: prefill de la infer principal (16 tokens en ~23 s) + re-OCR del panel grande con el modelo divagando (1933 tokens a 2048, ~1280 truncados en producción) — ver §3.3 |
| `benchmark_gate_p17_ab.py` (nuevo) + `benchmark_results/gate_p17_base.json` + `gate_p17_off.json` | **A/B del gate de la pág 17 (2026-08-16, capítulo completo 1-53, daemon ready)**: anular el trigger solo en pág 17 → 25.5 → 3.2 s (−22.3 s exactos), triggers 11→10, recuperación 31 = 31 sin pérdida; PERO el total del capítulo no baja (la varianza de generación VLM en otras páginas lo enmascara: ±20 s entre corridas) — **documentado y NO aplicado** (ahorro < ruido de medición) — ver §3.3. **ROI por página (2026-08-16, 5 corridas)**: recuperación determinista 5/5; 31/28/16/25 = 9.9-12.6 s/bloque (mejor), 15 = 84.6 s/bloque (peor con recuperación), 13 y 17 = siempre 0 (candidatos de gate; 13 ahorraría ~34-68 s/corrida, el doble que 17) — ver bloque ROI en §3.3 |
| `benchmark_results/vlm_2048_a.json`, `vlm_512_a.json`, `vlm_1024.json`, `vlm_1024_b.json`, `vlm_1280.json`, `vlm_1280_2.json`, `vlm_sameproc.json`, `vlm_1280_sameproc.json` | A/B del max_length del VLM (escalera 512–2048 y verificación mismo-proceso) — ver §4.6 |
| `config.py` | **2026-08-15: `UOCR_MAX_LENGTH` 2048 → 1280 APLICADO** (Fase 3.3) — cap calibrado por distribución de tokens; verificado en el capítulo completo (trigger 11/53 idéntico, recuperación 31 vs 32 bloques, pág 21 −35 %) — ver §3.3/§4.6 |
| `benchmark_results/production_1280_1-12.json`, `_13-21.json`, `_22-34.json`, `_35-53.json` | **corrida del capítulo completo con max_length=1280 (2026-08-15, 4 chunks por timeout)**: 11 triggers VLM, 31 bloques recuperados, pág 21 a 196 s — ver §4.6 |
| `benchmark_results/production_strip_full53.json` | **re-corrida del capítulo completo POST-STRIP (2026-08-15, daemon VLM detenido, mismo estado que el baseline)**: 53 págs, promedio **3.07 s/pág vs 3.99 s baseline (−23.2 %)**, etapa rapid 109.2→17.2 s (−92 s), pesadas de Ruta C −29.5 %, bloques 290→298 (+8, sin pérdida de recuperación) — ver §4.6 |
| `benchmark_results/production_all3_full53.json` | **capítulo completo con las TRES optimizaciones juntas (2026-08-15, daemon VLM ACTIVO, corrida única de 53 págs)**: promedio 13.04 s/pág, total 691 s, **31 bloques recuperados por el VLM**, 329 bloques finales; vs chunks 1280_* −46 % con misma recuperación (parte por varianza de trigger/VLM); pág 21 a 173 s — ver §4.6 |
| `benchmark_results/production_vlm_full53.json` + `production_vlm_full53_2.json` + `production_vlm_full53_3.json` | **re-medición del costo real del VLM (2026-08-16, daemon READY, corrida única de 53 págs)**: promedio 13.39 s/pág, total 709.8 s, 11/53 triggers → **10 llamadas VLM** (pág 13 saltada por cache de negativas), 31 bloques recuperados, **stages VLM = 573.1 s = 80.7 % del capítulo**; pág 21 = 191 s (27 %, 8 bloques recuperados); pág 17 = 0 recuperados (único candidato de gate, ~3 % máx) — ver §4.6. **VARIANZA (3 corridas mismas harness)**: stage VLM min 573.1 / max 761.4 / mediana 627.5 s (CV 15 %); **recuperación 31 = 31 = 31 (determinista)**; % del capítulo 47.6-80.7 % (mediana 74.2 %) — la varianza del % es del denominador (no-VLM 119→243→651 s por estado de máquina), no del VLM — ver bloque VARIANZA en §4.6 |
| `benchmark_results/production_nonvlm_fixed.json` + `.log` | **análisis del tiempo residual sin VLM (2026-08-16, 42 págs, harness CORREGIDO)**: promedio 3.37 s/pág; ruta_c 34.8 % (anidada en yolo/ctd), yolo 31.6 % (31/42 págs, 155 bloques, 0.29 s/bloque), ctd 21.8 % (16/42, 21 bloques NUEVOS, 1.47 s/bloque — peor ROI), prefilter 15.6 % (fijo 0.53 s/pág, vía 4.4C), rapid 12.1 %; spellcheck del merge 2.5-8 % (ya optimizado) — ver §4.6 |
| `benchmark_results/merge_stages.json` | **desglose fino del merge (2026-08-16, págs 4/43/29)**: spellcheck 0.16-1.38 s/pág (2.5-8 %) vs ruta_c 27-56 % — confirma que el spellcheck del merge no es cuello — ver §4.6 |
| `benchmark_results/strip_blocks_check.json` + `strip_targeted.json` + `strip_visual/` | **revisión de los 6 bloques perdidos del strip (2026-08-16)**: 4 de 6 son CONTENIDO REAL perdido (págs 1/4/40/52, diálogo confirmado por VLM sobre los recortes), 2 segmentación (págs 30/34); mecanismo: el det del strip apilado detecta mal la línea (conf 0.46-0.6 ≥ umbral 0.45) y bloquea el fallback EasyOCR — ver bloque ⚠ en §4.6 |
| `benchmark_results/production_fix_full53.json` + `.log` | **re-corrida del capítulo completo con el fix del strip APLICADO (2026-08-16, daemon ready, 1-53)**: total 175.7 s = 3.32 s/pág vs 162.5 s pre-fix (+8.1 %), delta dominado por estado de GPU (corrida única sin intercalado); las 4 páginas afectadas quedan dentro del ruido contra el baseline non-VLM corregido (pág 1 11.59 vs 11.07, pág 4 8.42 vs 8.32, pág 40 3.87 vs 4.31, pág 52 6.22 vs 5.75 s; Δ ≤ 0.5 s); trigger 11/53 y VLM en las mismas 11 páginas — ver bloque ⚠ en §4.6 |
| `benchmark_strip_fix_ab.py` (nuevo) + `benchmark_results/strip_fix_ab.json` | **A/B del fix de la pérdida del strip + APLICADO (2026-08-16, 7 págs, mismo-proceso base/fix/ref)**: subir el umbral a 0.7 → fallback EasyOCR NO recupera (lee basura 'じ‥'/'鼻'); el fix que funciona es **retry por crop con `_run_rapidocr` individual** (regla conf+longitud: confiable = conf≥0.85 o conf≥0.7+≥6 chars); **4/4 diálogos recuperados** sin perder el ahorro del strip (fix 2.6-5.6 s vs per-crop 4.2-7.5 s; retry ~0.3-1.2 s solo en páginas con crops fallidos). Implementado: `_RUTA_C_STRIP_RETRY_INDIVIDUAL` + `_rapid_blocks_usable` en ocr_utils.py, 2 tests nuevos + 5 adaptados, CI verde 987/987 — ver bloque ⚠ en §4.6 |
| `benchmark_results/production_foreign_preload.json` | **re-corrida post fast-path extranjero (2026-08-15, daemon detenido)**: sin regresión (bloques 298=298, 42/53 págs ±0.1 s), promedio 2.95 s/pág con deltas de signo mixto en páginas trigger = varianza de timeout VLM, no atribuible; el ahorro determinista es el one-time de ~0.08-0.35 s de carga en/pt movido a startup — ver §4.6 |
| `benchmark_results/spellcheck_ab_records.json` + `spellcheck_ab_after.json` | **A/B del límite de edición dependiente de longitud (2026-08-15, 53 págs instrumentadas, daemon detenido)**: 3 correcciones reales (todas 3-5 chars, distancia 1); réplica 0.167→0.001 s (scan d2 de >14 eliminado); correcciones idénticas con hash seed fijo; 'chmar' es no-determinismo propio de pyspellchecker (empate 50/50/50) — ver §4.6 |
| `benchmark_results/production_modocpu_12.json` + `production_modocpu_08.json` | **medición de MODO_CPU (2026-08-15, daemon detenido)**: 1.2 → 3.06 s/pág (neutro, recuperación 298=298); 0.8 → 2.95 s/pág con segmentación distinta (fragmentación, no comparable); YOLO CPU +0.34 s/pág, VLM off 0.00 s — ver §4.6 |
| `benchmark_foreign_check.py` (nuevo) | **auditoría del costo de `_contains_foreign_latin_tokens` (2026-08-15)**: mide llamadas/tiempo/known() por checker (es/en/pt) sobre el capítulo — ver §4.6 |
| `benchmark_results/foreign_check.json` | **medición de la detección extranjera (2026-08-15, 53 págs, daemon detenido)**: 112 llamadas, 0.0036 s totales (0.032 ms/llamada); en/pt known() 0.0005 s combinados; el es known() 0.306 s visible es la expansión de `sp.correction()` (camino corto), no esta función — VEREDICTO: sin costo evitable, no tocar — ver §4.6 |
| `benchmark_results/foreign_check_short2.json` | **medición del camino corto de sp.correction() por rango (2026-08-16, 53 págs, daemon detenido, benchmark ampliado)**: 0.471 s/capítulo (8.9 ms/pág), 99.9% en 2 palabras patológicas ('clvdeg' 6c 116 ms, 'clvdegchmar' 11c 354 ms); réplica 3-8× más rápida en 8-12 con equivalencia 10/10; NO se cambia el umbral (ahorro ~0.2% marginal) — ver §4.6 |
| `benchmark_foreign_gate.py` (nuevo) | **análisis de los 17 bloques mixtos (2026-08-16)**: captura los bloques que llegan al chequeo y mide para los foreign=True la evidencia extranjera, las correcciones forzadas con gate desactivado y la ambigüedad del detector — ver §4.6 |
| `benchmark_results/foreign_gate.json` | **veredicto del call-site del check extranjero (2026-08-16, 53 págs, daemon detenido)**: gate INERTE en 15/17 bloques (0 correcciones forzadas; 13 todo-mayúsculas + 2 salvados por el filtro d1); la única protección real es pág 29 'ASi QUE SEGUi' (×2): preserva 'SEGUi'→'Seguir' a cambio de bloquear 'ASi'→'Así'; los 15 falsos triggers son huecos del diccionario es ('relacionado','bolsas','vemos','dia','separo','verte','has' — españolas válidas ausentes del dict, es=False pt/en=True); NO se afina el call-site (tiempo nulo + gate por ambigüedad sería neto-negativo); fix de precisión = cobertura del diccionario es, no gatear — ver §4.6 |
| `benchmark_spell_touchpoints.py` (nuevo) | **mapa de todos los touchpoints de pyspellchecker (2026-08-16)**: instrumenta correction/known/candidates/índice/réplica/foreign check + langdetect y cargas, con atribución por pila de sitios — ver §4.6 |
| `benchmark_langdetect_cache.py` + `langdetect_cache.json` (nuevos) | **A/B del cache de langdetect (2026-08-16)**: 82 misses reales en el capítulo, 82 firmas únicas → ahorro potencial 0 %, costo residual 1.05 s = 1.4 % del wall; veredicto de cierre en §4.6 (cache por muestra implementado en `_ocr_spellcheck`, bounded 1024, 6 tests) |
| `benchmark_results/spell_touchpoints.json` | **costo por página de los touchpoints (2026-08-16, 53 págs, daemon detenido)**: NO hay pyspellchecker fuera de `_ocr_spellcheck` (solo preload de server.py); API real ~9 ms/pág concentrada en pág 41 (0.476 s camino corto); foreign/replica/candidates <0.03 s; índice 0.17 s one-time (cae en el filtro d1, no la réplica); **HALLAZGO: langdetect = 2.95 s = 70 % de la maquinaria spellcheck (5.65 ms × 523 bloques), nuevo candidato de optimización por firma/página** — ver §4.6 |
| `tools/analizar_vlm_1280.py` (nuevo) | fusiona los chunks del benchmark production y compara trigger/recuperación/bloques/tiempo contra el baseline 2048 |
| `config.py`, `ocr_engine.py`, `ocr_utils.py`, `server.py`, `app.js`, `js/config.js`, `README.md` | **2026-08-15: preset `MODO_CPU` (soporte sin GPU dedicada)**: flag único que apaga el VLM (gate en `_reforzar_con_unlimited` junto a UOCR_ENABLED), fuerza YOLO a CPU (`_resolver_device_yolo`) y sirve `ocr_scale` reducido (0.8) al frontend vía `/api/config` (el frontend lo aplica a `state.scale`). 3 tests nuevos (gate MODO_CPU, resolver, /api/config) |
| `ocr_utils.py`, `tests/test_ocr_utils.py` | **2026-08-15: `_ocr_spellcheck` corrección barata aplicada** (`_spellcheck_correction` réplica de `sp.correction()` con índice por longitud + Damerau acotado + pre-filtro de conteos + `_SPELL_CORRECTION_MIN_LEN=13` híbrido; langdetect con prefijo `_SPELL_LANG_MAX_CHARS=600`). **Diagnóstico corregido**: el langdetect NO era el cuello (profile de pág 4: 26.2 s en correction / 26.3 s en `__edit_distance_alt`, langdetect ni aparece); y el fast-path previo fue INERTE en producción (WeakKeyDictionary no soporta SpellChecker — slots sin `__weakref__`; los tests pasaban por MagicMock). pág 4: spellcheck 10.43 s → 3.7 ms (~2800×), pipeline 29.12 → 3.38 s, equivalencia 58/58 con pyspellchecker real (solo empate de frecuencia no determinista). 7 tests nuevos: `TestSpellcheckCorrectionFast` (equivalencia, delegación de cortas, diacríticos, índice por instancia) + `TestSpellcheckLangPrefix` (prefijo langdetect) — ver §4.6 |
| `benchmark_rutac_batch.py` (nuevo) | **A/B del batch estructural de la Ruta C (2026-08-15)**. Tras la integración, compara PRODUCCIÓN mismo-proceso vía el toggle `_RUTA_C_STRIP_BATCH`: baseline = per-crop (False), alt = strip (True) — ambos con fallback EasyOCR por crop y merge final. Harness: instrumenta det/rec/cls/crop_list/group/spellcheck; `--no-spellcheck` aísla el núcleo — ver §4.6 |
| `benchmark_results/rutac_batch_ab.json` (prototipo, 2026-08-15) / `rutac_batch_prod_ab.json` (producción integrada) | **medición del batch estructural**: prototipo 8 págs/52 crops: 20.49 s → 4.79 s (−76.6 %, det-calls 52 → 8). Producción integrada (3 págs, núcleo): **9.81 s → 2.32 s (−76 %), det-calls 24 → 3**, bloques pág 4 idénticos — ver §4.6 |
| `ocr_utils.py` | **2026-08-15: batch estructural de la Ruta C INTEGRADO** (`_ruta_c_prepare_crops`, `_rapidocr_strip_batch`, `_rapidocr_blocks_from_lines`, `_RUTA_C_STRIP_BATCH` toggle): det por chunk + UNA text_rec para todos los crops, fallback EasyOCR por crop conservado; `_run_rapidocr` refactorizado a DRY sobre el constructor de bloques (sin cambio de comportamiento). 11 tests nuevos (10 strip + 1 toggle) y 10 de la Ruta C adaptados al seam — ver §4.6 |

## 10. Trabajo pendiente y orden recomendado (2026-08-16)

> **ESTADO ACTUAL VERIFICADO EN CÓDIGO** (2026-08-17): producción corre con
> Ruta C = strip batch integrado + upscale 3.5× + fallback EasyOCR por crop +
> fix de diálogos; `UOCR_MAX_LENGTH = 1280`; spellcheck con réplica acotada +
> prefijo langdetect + cache por muestra (bounded 1024); preset `MODO_CPU`;
> gate persistente de ceros confirmados (pág 13/17, §10.7 item 1) +
> `no_repeat_ngram=15` en el daemon VLM (§10.7 item 2);
> CI local completo (1011/1011 tests, mypy 0, cobertura por módulo OK,
> `ocr_engine.py` 97.0 % vs umbral 94.6 %).
>
> **§10.2 quedó EJECUTADO en su totalidad el 2026-08-16** — los 7 ítems de
> implementación se cerraron con datos (veredictos en §10.7). Lo que queda
> de este plan es proceso (baseline de calidad, commit, limpieza) y
> hardware (GPU con más VRAM).

### 10.1. Cerrado con datos (NO re-abrir sin cambio de corpus/hardware)

| Ítem | Veredicto medido |
|---|---|
| 4.2 multi-proceso OCR | NO procede — el semáforo no satura con la carga real |
| 4.4A prefilter condicional | NO procede — ahorro <2 %, doble pasada en débiles; trigger v4.2 medido 0/53 |
| 4.4B inpaint res. reducida | Descartada — regresiona la pág débil (control pág 11) |
| 4.4C detector de líneas selectivo | Descartada — no hay líneas en el corpus (0.006 % área); inpaint 0.53 s/pág es estructural |
| 4.5 afinar rapid_cond_skip | Umbral actual ya es óptimo |
| Gate intra-crop de la Ruta C | NO procede — no agrega bloques nuevos |
| Gate de pág 17 | NO aplicado — ahorra 22.3 s exactos pero < ruido de medición |
| Cache de langdetect por firma | 0 % de potencial (82 textos = 82 firmas); residual 1.05 s = 1.4 % estructural |
| Límite de edición dependiente de longitud | Calibrado: 3-5→1, 6-14→2, >14→0 (solo 3 correcciones reales en el capítulo) |
| Foreign gate (17 bloques mixtos) | Inerte en 15/17; única protección real es pág 29; no afinar |
| Batch VLM | Descartado en esta GPU (2.2× más lento); re-evaluar solo con otra GPU |

### 10.2. Pendiente de IMPLEMENTAR — ordenado por ROI/riesgo (todo medido)

**Nivel 1 — VLM (74-81 % del capítulo; la palanca más grande que queda):**

1. **Gate persistente para pág 13** (y 17): recuperación **0 en 4-5 corridas**;
   pág 13 cuesta 31-68 s/llamada (el doble que 17). El cache de negativas
   (§8.4.1) ya es un gate por historial pero falla por TTL 30 min + firma
   derivada. Vías: TTL extendido para ceros confirmados o firma estable (hash
   de la imagen en vez del grid). Validación: mismo A/B de gate que pág 17.
   **Ahorro esperado: 50-90 s/capítulo sin pérdida. Riesgo: bajo.**
2. **Prompt del VLM + `no_repeat_ngram`** (uocr_daemon.py): pág 21 divaga
   1933 tokens (18 chars reales; ngram 35 no la corta). Atacar el bucle
   (ngram 15-20) o el prompt fijo `'<image>document parsing.'` → instructivo
   de diálogo artístico. **Ahorro esperado: 60-100 s en pág 21. Riesgo:
   medio** — A/B de recuperación con `benchmark_vlm_maxlen.py` antes de tocar.
3. **Prefill del VLM 640→512 px**: la llamada principal genera 16 tokens en
   ~23 s (casi todo prefill). Coste fijo por página trigger (~2-3 s×11
   páginas si baja 15 %). **Riesgo: bajo-medio** (A/B de recuperación).

**Nivel 2 — residual sin VLM (42 páginas, 2.95-3.3 s/pág):**

4. **Afinar el gate de CTD** (peor ROI del residual: 30.8 s por 21 bloques,
   1.47 s/bloque vs 0.29 de YOLO): subir `COMIC_DETECTOR_GATE_MAX_CONF`
   0.35→0.30 o bajar `MIN_BLOCKS` 3→2, o dedup de regiones ya cubiertas por
   YOLO. **Ahorro esperado: hasta ~30 s/capítulo. Riesgo: medio** — los 21
   bloques son texto flotante real; A/B con `benchmark_ab_utils.py` primero.
5. **YOLO/CTD a CPU permanente con daemon activo**: libera VRAM para EasyOCR
   durante la inferencia VLM (la degradación actual hace las 42 páginas
   2-5× más lentas con daemon: 119→448-651 s). **Ahorro: sin medir aún —
   A/B pendiente. Riesgo: bajo.**
6. **Heurístico pre-OCR de ruido**: correr el prefilter (0.53 s/pág) solo en
   páginas ruidosas; gana en 13/53 páginas. **Ahorro: 14-23 s/capítulo.
   Riesgo: bajo** (el trigger v4.2 ya se midió 0/53 raw vs pref).

**Nivel 3 — UI/servidor:**

7. **4.3 — worker thread para export/render del PDF en el navegador**
   (único ítem de UI pendiente; baja prioridad, sin tocar el pipeline OCR).

### 10.3. Calidad y proceso (no son optimizaciones, pero bloquean el cierre)

8. **Baseline de calidad del corpus actual — MEDIDO (2026-08-16)**: el
   75.8 % de Julio (128 págs, pipeline viejo) NO es comparable con el 32.2 %
   del cap. 53 re-medido post-fix del strip (96/314 UNTRANSLATED por
   ruido/fragmentos; §6 "RE-MEDICIÓN DE CALIDAD POST-FIX DEL STRIP").
   Pendiente: criterio de "aceptable" para el corpus actual y re-medición
   estable periódica.
9. **Commit pendiente** de la sesión del plan §10 (langdetect cache + gate
   persistente + ngram 15 + benchmarks): `ocr_utils.py`, `ocr_engine.py`,
   `config.py`, `uocr_daemon.py`, `uocr_client.py`, `tests/*`,
   `PLAN_OPTIMIZACION_RENDIMIENTO.md`, benchmarks + JSONs.
10. **Limpiar artefactos sin commitear**: `gh_logs*.txt`, `gh_run_logs.zip`,
    `preview_pages/`, `_tmp_vlm_pages/` (regenerables) — decidir qué se
    archiva/ignora para que `git status` vuelva a ser legible.

### 10.4. Triggers condicionales (NO son trabajo fijo)

- **4.7 → 4.1**: re-medir el split detector/recognizer SOLO si el corpus
  cambia (p. ej. origen ja/ko/zh masivo). Hoy el detector CRAFT domina
  (72.9 %); el recognizer no es el cuello.
- **4.2**: multi-proceso solo si un stress test futuro muestra saturación
  del semáforo (hoy: no satura).

### 10.5. Estructural (requiere hardware/decisión)

- **GPU con más VRAM**: única palanca estructural real para el VLM (hoy el
  daemon 2.25 GB + EasyOCR ~1 GB + YOLO ~1 GB no caben en 4 GB → degradación
  a CPU con daemon activo). Con más VRAM se re-evalúan batch VLM, prefill
  mayor y quizá el prompt sin penalización.

### 10.7. EJECUCIÓN 2026-08-16 — veredictos medidos (plan implementado 1→7)

> **1. Gate persistente pág 13/17 — IMPLEMENTADO.** Firma estable: el digest
> exacto del thumbnail (32×32 blake2b) se sustituyó por un **dHash 9×8**
> (64 bits robustos a variación mínima de píxeles) en `_page_signature`
> (misma página entre corridas → misma firma; la pág 13 registraba dos
> firmas). TTL extendido: ledger de **ceros confirmados** (`_uocr_neg_ceros`,
> sección "ceros" del cache, TTL 7 días): una firma que falló ≥
> `UOCR_NEG_CERO_MIN=2` veces en ventanas TTL distintas suprime el VLM con
> salvaguarda mucho_mas_debil intacta y recovery que la refuta
> (`_limpiar_decision_negativa`). **Validación real** (`benchmark_gate_cero_confirmado.py`,
> daemon UP, pág 13, `gate_cero_confirmado.json`): pasada 1 = 99.96 s (VLM,
> 0 rec), pasada 2 = 45.61 s (VLM, 0 rec), **pasada 3 = 1.38 s (VLM saltado,
> mismos 6 bloques)** — el gate ahorra 44+ s por re-corrida sin pérdida.
> `_DECISION_CACHE_VERSION` 6→7; 7 tests nuevos (`TestCeroConfirmado`) + 2
> de estabilidad dHash. **Ahorro: 50-90 s/capítulo en re-corridas.**
>
> **2. Prompt + ngram del VLM — ngram 15 APLICADO, prompt DESCARTADO.**
> `_PROMPT_DEFAULT`/`_NGRAM_SIZE_DEFAULT` configurables por request
> (`/ocr` y `/ocr-batch` aceptan `prompt`/`ngram`; cliente + handler + 6
> tests nuevos). A/B pág 21 (max_length 1280, daemon UP): **ngram 35 →
> 210.6 s / 8 bloques; ngram 15 → 194.4 s / 8 bloques, TEXTOS IDÉNTICOS**
> (−16 s, −7.7 %; el ngram solo bloquea repeticiones exactas, no puede
> perder recuperación) → default aplicado 15. **Prompt instructivo de
> diálogo ('<image>Extract all dialogue text...') → 32.9 s pero SOLO 2/8
> bloques (−84 % tiempo, −6 bloques)**: el prompt degrada el formato de
> detección del modelo → DESCARTADO.
>
> **3. Gate de CTD — CERRADO CON DATOS, NO tocar.** `benchmark_ctd_gate_ab.py`
> (capítulo completo, daemon caído, `ctd_gate_ab.json`): CTD corre en
> 18/53 páginas (30.0 s, 22 bloques brutos) y **TODAS tienen conf 0.00
> post-YOLO** — cualquier umbral que las saltara eliminaría el tier entero
> (el gate actual ya es el piso; las opciones del plan partían de un
> malentendido de la dirección del umbral).
>
> **4. YOLO/CTD a CPU con daemon — CERRADO CON DATOS, NO procede.**
> Medición directa del pico VRAM del daemon durante una generación larga
> (pág 21, max_length 1280): **3.04 GB**. Con EasyOCR (~1 GB) el total
> roza los 4 GB de la 1050 Ti → mantener EasyOCR en GPU durante la
> inferencia VLM es inviable; la degradación a RapidOCR CPU es la política
> correcta. YOLO CPU costaría además +0.6 s/pág (730 vs 130 ms).
>
> **5. Prefill 640→512 — DESCARTADO CON DATOS.** A/B págs 16+21
> (`vlm21_prefill512.json`): **512 pierde 8 de 13 bloques recuperados
> (62 %)** aunque corta el tiempo −42 % (167.8 vs 291.2 s). El prefill 640
> se mantiene; `image_size` quedó configurable por request (item 5,
> parametrizado junto a prompt/ngram) para futuros A/B.
>
> **6. Heurístico pre-OCR de ruido — MEDIDO Y DESCARTADO.** Unión de
> features pre-OCR (dark_ratio, varianza Laplaciana, **densidad de
> speckle** = componentes conectadas por píxel oscuro) con el outcome del
> A/B 4.4A: el speckle predice moderadamente (umbral 0.0035 → recall 58 %,
> 9 FP) pero el ahorro (~20 s/capítulo, <2 %) no justifica degradar las
> 4-5 páginas débiles que el heurístico se pierde (pág 17: conf 0.43→0.95
> con el prefilter). Misma conclusión que 4.4A.
>
> **7. 4.3 worker thread — CERRADO: ya cubierto.** pdf.js ya renderiza en
> web workers (`GlobalWorkerOptions.workerSrc`, app.js:470-545) y el loop
> de export (`exportFullPdf`) es async no-bloqueante (await por página;
> el trabajo síncrono por página es composición de canvas de ms). Un
> OffscreenCanvas worker añadiría complejidad por beneficio marginal.
>
> **Artifacts nuevos**: `benchmark_gate_cero_confirmado.py`,
> `benchmark_ctd_gate_ab.py`, `benchmark_results/gate_cero_confirmado.json`,
> `ctd_gate_ab.json`, `vlm21_ngram35.json`, `vlm21_ngram15.json`,
> `vlm21_prompt_dialogue.json`, `vlm21_prefill640.json`,
> `vlm21_prefill512.json`. Tests: +17 (7 cero confirmado + 2 dHash + 6
> handler prompt/ngram/image_size + 2 image_size) y 3 fakes actualizados.

### 10.6. Orden recomendado de ejecución

```
1. Gate persistente pág 13/17 (VLM)     → A/B de gate, riesgo bajo, 50-90 s
2. Prompt + ngram del VLM (pág 21)      → A/B de recuperación, 60-100 s
3. Gate de CTD más estricto             → A/B con harness, ~30 s
4. YOLO/CTD a CPU con daemon (A/B)      → libera VRAM, sin medir aún
5. Prefill VLM 512                       → A/B de recuperación, coste fijo
6. Heurístico pre-OCR de ruido          → 14-23 s, riesgo bajo
7. 4.3 worker thread PDF (UI)           → baja prioridad
8. Baseline de calidad + commit + limpieza de artefactos
```

Cada paso se cierra con: A/B mismo-proceso o intercalado (daemon detenido o
ready según corresponda), `run_ci.py` completo verde y sin regresión de
cobertura. Los pasos 1-2 dependen del daemon VLM ready; 3-6 son medibles con
el daemon detenido.
