# 🔬 Investigación exhaustiva del código interno de los 3 motores OCR

> **Fecha**: 2026-08-04 · **Autor**: Buffy (Freebuff)
> **Propósito**: Documentar a nivel de código CÓMO funciona cada uno de los 3 motores
> (EasyOCR, RapidOCR, Unlimited-OCR), cómo se integran hoy en la app, y cuál es el plan
> para unirlos en un solo motor unificado. Este documento complementa PLAN_FUSION_OCR.md
> (estrategia/fusion) con el detalle interno de implementación.

---

## 0. Resumen ejecutivo

La app **Traductor Visual Pro** ya ejecuta 3 motores OCR distintos, cada uno con
características complementarias:

| Motor | Archivo(s) | Pipeline | Dispositivo | Velocidad | Fortaleza |
|---|---|---|---|---|---|
| **EasyOCR** | `ocr_utils.py` | CRAFT detector + CRNN recognizer (PyTorch) | GPU (CUDA) | ~0.88s/pág | Texto normal en burbujas, rápido |
| **RapidOCR** | `ocr_utils.py` | PP-OCRv4 detector + recognizer (ONNX Runtime) | CPU | ~2.4s/pág | Detecta ~90% de los bloques del manga; títulos estilizados |
| **Unlimited-OCR** | `uocr_daemon.py` + `uocr_client.py` | DeepSeek-OCR VLM 3B (4-bit NF4, bitsandbytes) | GPU | 60-500s/pág | Diálogo pintado EN el arte (págs. artísticas), lectura semántica |

Los 3 están **unidos hoy** a través del modo `fusion` (`routes/api.py`): un híbrido
EasyOCR+RapidOCR SIEMPRE, y Unlimited-OCR solo cuando un trigger heurístico decide que
la página es difícil. La fusión de bloques se hace con `_fusionar_blocks_multi()`
(dedup + Levenshtein + votación + NMS).

**El veredicto de los benchmarks (2026-08-03/04):**
- El merge de fusión cuesta **~0 segundos** (§3.7 de PLAN_FUSION_OCR.md).
- El overhead total del modo fusion está 100% en la inferencia del daemon U-OCR.
- El "modo easyocr" de la app **ya es híbrido** (corre RapidOCR por defecto).

---

## 1. Motor #1 — EasyOCR (interno)

### 1.1 Carga del reader (`_get_ocr_reader`, ocr_utils.py:44)

```python
_ocr_readers: dict[str, Any] = {}   # cache por idioma: "latin" | "ja" | "ko" | "zh"
_ocr_lock: threading.Lock            # double-checked locking
```

- **Lazy-load con cache por clave de idioma** (`latin` para español/inglés/portugués/
  francés/alemán; `ja`, `ko`, `zh` para CJK).
- **GPU prioritario**: chequea `torch.cuda.is_available()`; si falla, fallback CPU
  automático con try/except.
- **Modelos en disco**: `ROOT/ocr_models` (se descargan una vez con `download_enabled=True`).
- **⚠️ Orden crítico de carga** (documentado en server.py `_preload_background`):
  EasyOCR debe cargar PRIMERO para inicializar `torch.cuda` y cuDNN; si CT2 carga antes,
  sus DLLs cuDNN conflictúan → crash `cudnnGetLibConfig`.

### 1.2 Inferencia (`_run_ocr_on_image`, ocr_utils.py:710)

```python
with _ocr_semaphore:          # máx 1 lectura simultánea (EasyOCR no es thread-safe)
    if _uocr_inferring.is_set():   # v4.2: degradar a RapidOCR CPU (race window)
        return _run_rapidocr(...)
    with _gpu_lock:            # v4.2: serializar con daemon U-OCR (VRAM compartida)
        return reader.readtext(img_rgb, detail=1, paragraph=False,
            min_size=_OCR_MIN_SIZE,          # 6
            text_threshold=_OCR_TEXT_THRESHOLD,  # 0.15 (sensible para manga)
            low_text=_OCR_LOW_TEXT,          # 0.10
            link_threshold=0.3,
            canvas_size=min(max(dim), 2500), # _OCR_CANVAS_SIZE
            mag_ratio=mag)                   # 1.3 default, 1.8 si conf baja
```

- **Semáforo OCR** (`_ocr_semaphore = Semaphore(1)`): limita a UNA llamada EasyOCR a la vez.
- **Lock global GPU** (`_gpu_lock = RLock`): serializa EasyOCR (server) con el daemon
  U-OCR (proceso aparte). Sin esto, la contención VRAM hace al daemon pasar de 83s a
  140-1439s por página.
- **Flag de degradación** (`_uocr_inferring = Event`): si el daemon U-OCR está infiriendo,
  los workers de otras páginas degradan a RapidOCR CPU en vez de esperar el mutex.

### 1.3 Conversión a bloques (`_ocr_results_to_blocks`, ocr_utils.py:756)

El resultado crudo de EasyOCR es `list[(bbox_4pts, text, confidence)]`. Se convierte a:

```python
{
  "x": int, "y": int, "w": int, "h": int,     # caja del texto
  "text": str,
  "confidence": float,                          # confianza real del recognizer
  "fontSize": max(8, int(h * 0.75)),            # estimado desde altura
  "textColor": "#000000"|"#ffffff",             # por luminancia del ROI
}
```

Luego pasa por `_group_and_merge_blocks()` (ver §1.5).

### 1.4 Pipeline completo (`_detect_and_ocr`, ocr_utils.py:792)

El pipeline por defecto es **híbrido de 3 niveles**:

1. **EasyOCR directo** (GPU, ~1.16s) — si `prefilter=True`, aplica `_pre_filter_image`
   (limpieza morfológica) antes.
2. **mag_ratio adaptativo**: si la confianza promedio < `avg_conf_threshold` (0.15),
   re-intenta con `mag_ratio=1.8` (mejor detalle para tipografía artística).
3. **RapidOCR siempre** (si `use_hybrid=True`, que es el default) — corre `_preprocess_rapid`
   + `_run_rapidocr` y **fusiona** con `_fusionar_blocks(blocks_easy, rapid_blocks)`.
4. Si EasyOCR devolvió 0 bloques y `allow_fallback=True`: **CLAHE+sharpen** →
   `_preprocess_enhanced` → EasyOCR de nuevo.

### 1.5 Filtros y agrupación (`_group_and_merge_blocks`, ocr_utils.py:1099)

Esta función es el "cañón de limpieza" común a EasyOCR y RapidOCR:

**Pre-merge (contra el texto ORIGINAL, antes de limpiar símbolos):**
- Watermark patterns (`WATERMARK_PATTERNS` de config.py).
- Metadatos de margen (`MARGIN_NOISE_PATTERNS`: fechas "13/7/26", horas "4.58 p.m").
- Texto numérico de margen (ratio de dígitos ≥35% y ≤4 palabras).
- URLs (`https?://`, `www.`, `.com/.net/...`).

**Corrección post-OCR:**
- Glosario pre-OCR (`_aplicar_glosario` de translator.py) — "Vilianos"→"Villanos", etc.
- Limpieza de símbolos (`re.sub` de `@#$%^&*...`).
- **Corrector ortográfico pyspellchecker** (`_ocr_spellcheck`) — 86K palabras + 16
  palabras manga con frecuencia alta.

**Merge horizontal + vertical:**
- Fusión horizontal: misma línea (`abs(cy1-cy2) < max_h * 0.45`) y gap tolerante
  (`< max(35, w*2.5)`).
- Fusión vertical (2ª pasada): misma columna (x-overlap ≥ 50%) y gap vertical
  (`< 1.5× altura`).

**Post-merge (9 filtros finales):**
- Números puros, patrones numéricos, comilla+número, puntuación suelta, aspecto
  estrecho (<0.4 y ≤3 chars), char suelto (1 char conf<0.25), corto+baja confianza
  (conf<0.15 y ≤3 chars), combo dígito+letra.

---

## 2. Motor #2 — RapidOCR (interno)

### 2.1 Carga (`_get_rapid_engine`, ocr_utils.py:131)

```python
from rapidocr_onnxruntime import RapidOCR
_rapid_engine = RapidOCR()   # modelos PP-OCRv4 (ONNX Runtime, CPU)
```

- **Lazy-load singleton** con double-checked locking (`_rapid_lock`).
- **CPU puro** (ONNX Runtime): no compite por la GPU.
- Usa los MISMOS modelos PP-OCRv4 que PaddleOCR, pero sin el conflicto de
  PaddlePaddle vs PyTorch.

### 2.2 Preprocesado (`_preprocess_rapid`, ocr_utils.py:148)

```python
def _preprocess_rapid(img_bgr):
    filtered = _pre_filter_image(img_bgr)      # morfología: líneas, speckle, bordes
    enhanced = _preprocess_enhanced(filtered)  # CLAHE + gamma + bilateral + unsharp
    return enhanced
```

Reutiliza los mismos preprocesados que EasyOCR (ver §1.5).

### 2.3 Inferencia (`_run_rapidocr`, ocr_utils.py:155)

```python
with _rapid_semaphore:   # Semaphore(1): ONNX Runtime no es thread-safe
    result, _ = engine(img_bgr)   # → list[(bbox, text, conf)]
```

- Filtra `conf < 0.08` y `w/h < 3px`.
- Produce bloques con **el mismo formato interno que EasyOCR** (clave para la fusión):
  `{x, y, w, h, text, confidence, fontSize, textColor}`.
- `textColor` también por luminancia del ROI (misma heurística).
- Final: `_group_and_merge_blocks()`.

### 2.4 Rol en la fusión

En `_detect_and_ocr`, cuando `use_hybrid=True` (default), RapidOCR corre SIEMPRE y sus
bloques se fusionan con EasyOCR vía `_fusionar_blocks()` (peso 0.9). El benchmark
demostró que **RapidOCR aporta ~90% de los bloques detectados** en este manga (225
bloques híbridos vs 22 de EasyOCR puro).

---

## 3. Motor #3 — Unlimited-OCR (interno)

### 3.1 Arquitectura: daemon separado

| Pieza | Archivo | Rol |
|---|---|---|
| **Daemon** | `uocr_daemon.py` | Proceso Python en `env_uocr_gpu` (torch cu126 + bitsandbytes). Carga el modelo 4-bit UNA vez (~494s) y sirve OCR por HTTP en `127.0.0.1:5177`. |
| **Cliente** | `uocr_client.py` | Módulo stdlib-only (urllib/subprocess) que el servidor Flask importa. Lanza el daemon en background y le habla por HTTP. |
| **Modelo** | `models_unlimited_patched/` | DeepSeek-OCR VLM 3B (patched), cuantizado 4-bit NF4. |

**Por qué un proceso separado**: el venv `env_uocr_gpu` tiene torch cu126 + bitsandbytes
que NO pueden importarse dentro del proceso del servidor (conflicto de DLLs CUDA con
EasyOCR/CT2 de `env/`).

### 3.2 Carga del modelo (`_load_model`, uocr_daemon.py)

```python
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
_model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
    use_safetensors=True, torch_dtype=torch.float16,
    quantization_config=quant, device_map={"": 0}).eval()
```

- **Carga en background** (hilo daemon) — no bloquea `/health`.
- `device_map={"": 0}` fuerza a la única GPU (GTX 1050 Ti 4GB).
- Estado expuesto por `/health`: `{state: loading|ready|error, load_s, vram_gb, error}`.

### 3.3 Inferencia (`_infer_once`, uocr_daemon.py)

```python
with contextlib.redirect_stdout(stream):        # captura el stream del modelo
    _model.infer(_tokenizer,
        prompt="<image>document parsing.",
        image_file=image_path,
        output_path=out_dir,
        base_size=1024, image_size=640, crop_mode=crop_mode,
        max_length=4096,
        no_repeat_ngram_size=35, ngram_window=128,
        save_results=True)
```

- El modelo **emite por stdout** líneas con formato:
  `<|det|>type [x, y, w, h]<|/det|>text` — ahí se recuperan las coordenadas.
- `result.md` guarda el texto LIMPIO (sin tags), pero SIN coordenadas.
- `crop_mode=True` = 9-grid sobre la página completa (pase principal);
  `crop_mode=False` = una sola vista global 640x640 (usado en el re-OCR artístico).

### 3.4 Parseo de bloques (`_parse_blocks`, uocr_daemon.py)

```python
_DET_RE = re.compile(
    r"<\|det\|>(\w+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*<\|/det\|>[ \t]*([^\n]*)",
    re.MULTILINE)
```

- `[ \t]*` (no `\s*`) para NO cruzar saltos de línea: bloques sin texto (type=image)
  no absorben la línea siguiente.
- Produce `{type, x, y, w, h, text}` donde `type ∈ {text, title, header, image,
  footer, page_number, ...}`.

### 3.5 Re-OCR de diálogo artístico (`_recover_art_dialogue`, uocr_daemon.py)

**El post-procesado clave para págs. artísticas 3/12**:

1. Detecta bloques `type="image"` con área > 30% de la página (`_ART_RECOVER_MIN_AREA_RATIO`).
2. Recorta la región (con padding 8px).
3. Escala preservando aspecto + **letterbox blanco** a 640x640 (PIL LANCZOS).
4. Re-envía al modelo con `crop_mode=False`.
5. **Mapea los sub-bloques de vuelta al espacio de la página**:
   ```python
   px = x0 + (sb["x"] - ox) / s   # compensa recorte + escala + offset letterbox
   py = y0 + (sb["y"] - oy) / s
   ```
6. Marca los recuperados con `from_art_recrop: True` (peso extra en la fusión).

No recurre (solo 1 nivel) y usa PIL (no cv2 — env_uocr_gpu no lo tiene).

### 3.6 Serialización (`_run_ocr`, uocr_daemon.py)

```python
if not _infer_lock.acquire(timeout=1800):   # UNA inferencia a la vez (GPU única)
    return {"error": "timeout esperando turno de inferencia", ...}
try:
    ... pase principal + re-OCR artístico ...
finally:
    _infer_lock.release()
```

- `torch.cuda.reset_peak_memory_stats()` para medir VRAM real.
- Devuelve `{text, infer_s, vram_gb, blocks, recovered_from_art}`.

### 3.7 Confianza: heurística (NO logits)

Validado empíricamente (2026-08-03, test_uocr_stream_conf.py): el modelo **no emite
confidence** y los logits saturan ~0.997 sin discriminar. Por eso `_estimate_confidence_heuristic`
(ocr_utils.py:241) estima:

- Base por tipo: text/title 0.90, header 0.70, resto 0.80.
- Sin letras ×0.5; ≤2 chars ×0.7; ratio vocales ≥0.25 +0.05.
- `from_art_recrop`: piso 0.80.
- fontSize 10-40px (rango natural de diálogo): +0.03.

---

## 4. La capa de fusión (routes/api.py)

### 4.1 Endpoint `/api/process-page`

Flujo completo (routes/api.py):

```
1. Validar payload (image base64, target, source, ocr_mode, prefilter, force_uocr,
   disable_uocr, pure_easyocr).
2. Decodificar → BGR. Safety resize si > 4096px (con scale_x/scale_y para
   devolver coords en el espacio original).
3. Detección de idioma (source=="auto" → asume "es" temporalmente).
4. OCR según modo:
   - "easyocr": _detect_and_ocr(use_hybrid=True) → YA es híbrido E+R.
   - "unlimited": _ocr_with_unlimited() → si RuntimeError, fallback a easyocr.
   - "fusion": _detect_and_ocr (E+R) SIEMPRE + trigger U-OCR (ver §4.2).
   - "auto": _detect_and_ocr(allow_fallback=True) (con CLAHE fallback).
5. Watchdog anti-zombie (OCR <2s con 0 bloques ×3 = zombie).
6. Filtrar watermarks.
7. Re-detección de idioma post-OCR (si source=="auto").
8. Inpainting: _build_inpaint_mask (máscara de GLIFOS) + _inpaint_image
   (TELEA radio adaptativo + border-blend).
9. Traducción paralela (ThreadPoolExecutor, _translate_one con caché CT2→Google).
10. Armar respuesta con bgColor muestreado y coords escaladas.
```

### 4.2 Trigger de refuerzo U-OCR (v4.2)

```python
trigger = (not blocks
           or (len(blocks) < UOCR_TRIGGER_MIN_BLOCKS and avg_conf < UOCR_TRIGGER_CONF)
           or has_big_panel
           or force_uocr)
if disable_uocr:
    trigger = False
```

| Constante (config.py) | Valor | Significado |
|---|---|---|
| `UOCR_TRIGGER_CONF` | **0.20** | confianza media del híbrido por debajo de la cual se refuerza |
| `UOCR_TRIGGER_MIN_BLOCKS` | 3 | mínimo de bloques para NO reforzar |
| `UOCR_IMAGE_BLOCK_RATIO` | 0.15 | bloque image ≥15% de la página |
| `OCR_ENGINE_WEIGHTS` | E=1.0, R=0.9, U=1.1 | pesos de calibración por motor |

- **`has_big_panel`**: `_page_has_large_image_panel()` (ocr_utils.py:1047) — heurística
  barata (~20ms): downscale a 300px, mide ratio de píxeles oscuros (<120) y retorna
  True si >18% (arte oscuro domina la página).

### 4.3 Re-OCR a nivel de globo (Ruta C)

Cuando el trigger dispara, además de U-OCR se ejecuta la **Ruta C** (la única
granularidad que el benchmark demostró que funciona):

```python
regions = []
for panel in uimage_panels:          # globos dentro de los paneles image del daemon
    regions += _detect_bubble_regions_in_panel(img_bgr, panel)
regions += _detect_bubble_regions_in_panel(img_bgr, full_page)  # + página completa
# Descartar globos ya cubiertos por bloques híbridos (IoU > 0.5)
regions = [r for r in regions if not any(_overlap_ratio(r, b) > 0.5 for b in blocks)]
bubble_blocks = _recover_regions_with_easyocr(img_bgr, regions, ocr_lang, upscale=3.5)
```

**Detección de globos** (`_detect_bubble_regions_in_panel`, ocr_utils.py:916):
- Blobs de luminancia >200 (interior de globo) + morfología de cierre 9x9.
- Roundness `4π·area/perímetro² ≥ 0.30` (elíptico).
- Borde oscuro definido (`border_ratio ≥ 0.08`).
- Interior con tinta (`dark_ratio > 0.02`).

**Re-OCR de globo** (`_recover_regions_with_easyocr`, ocr_utils.py:987):
- Recorte con padding → upscale 3.5× (INTER_CUBIC) → EasyOCR GPU → mapear ÷upscale.
- Marcados con `engine: "easyocr-region"`.

### 4.4 La fusión de bloques (`_fusionar_blocks_multi`, ocr_utils.py:269)

```python
sources = [blocks_hibrido, ublocks + bubble_blocks]
weights = [OCR_ENGINE_WEIGHTS["easyocr"], OCR_ENGINE_WEIGHTS["unlimited"]]
merged = _fusionar_blocks_multi(sources, weights)
```

Algoritmo en 4 pasos:

1. **Dedup por texto normalizado idéntico** (lowercase + sin acentos) → gana el de mayor
   score (`_block_score = conf × min(2, max(0.5, len/5)) × weight`).
2. **Alineación Levenshtein**: bloques con `IoU > 0.4` y distancia de edición
   ≤30% de la longitud → mismo texto.
3. **Votación**: 2+ motores distintos coinciden → confianza del ganador +0.15.
4. **NMS espacial final**: `IoU > 0.40` descarta el duplicado de menor score.

---

## 5. Sincronización de recursos (semáforos y locks)

| Recurso | Tipo | Quién | Propósito |
|---|---|---|---|
| `_ocr_semaphore` | `Semaphore(1)` | EasyOCR | EasyOCR no es thread-safe |
| `_rapid_semaphore` | `Semaphore(1)` | RapidOCR | ONNX Runtime no es thread-safe |
| `_gpu_lock` | `RLock` | EasyOCR + daemon U-OCR | Serializar GPU (GTX 1050 Ti 4GB compartida) |
| `_uocr_inferring` | `Event` | routes/api → ocr_utils | Degradar a RapidOCR CPU mientras el daemon infiere |
| `_infer_lock` | `Lock` (daemon) | U-OCR | Una inferencia VLM a la vez |
| `_ocr_lock` / `_rapid_lock` | `Lock` | lazy-load readers | Double-checked locking |

**Cómo se coordina todo** (modo fusion, workers=2):
```
Worker A: página 3 (trigger: panel image grande)
  → _detect_and_ocr (E+R, ~7s)
  → trigger → _ocr_with_unlimited:
      _uocr_inferring.set()          # ← el resto del sistema lo ve
      with _gpu_lock:                # ← la GTX es del daemon ahora
          uocr_client.process_page() #   60-500s de inferencia VLM
      _uocr_inferring.clear()

Worker B: página 4 (sin trigger)
  → _detect_and_ocr:
      ¿_uocr_inferring está set? SÍ → degrada a RapidOCR CPU (~2-4s)
      → la página normal avanza en paralelo con la inferencia VLM ✅
```

---

## 6. Preload y arranque (server.py)

```python
_preload_thread = threading.Thread(target=_preload_background, daemon=True)  # EasyOCR → CT2
_uocr_thread = threading.Thread(target=_preload_unlimited_daemon, daemon=True)  # lanza daemon U-OCR
```

**Orden crítico** (server.py `_preload_background`):
1. **EasyOCR primero** (toma GPU, inicializa CUDA/cuDNN).
2. **CT2 después** (auto-detecta GPU — ya no force_cpu porque CUDA está inicializado).

El daemon U-OCR se lanza en paralelo (`_preload_unlimited_daemon`): `uocr_client.spawn_daemon()`
inicia el proceso `env_uocr_gpu/python.exe uocr_daemon.py --port 5177` y el modelo tarda
~8 min en cargar (494s medidos) mientras el servidor ya responde.

---

## 7. El runner del capítulo (process_all_pages.py)

- **Render thread** (1): PyMuPDF → `get_pixmap(zoom=1.2)` → PNG → base64 → cola.
- **API workers** (N, default 3; en fusion se usa 2): consumen la cola y hacen
  `POST /api/process-page`.
- **Checkpoint** cada 10 páginas (`resultados_progreso.json`).
- **Timeout por página**: 1800s en fusion (la cola del daemon U-OCR puede encolar
  2-3 inferencias ≈ 260-1446s).
- **Reintentos**: 2 con backoff (5s, 10s).

---

## 8. PLAN: unificar los 3 motores en "un solo OCR" (v5)

### 8.1 Objetivo

Hacer que la app use **un único motor conceptual** ("Motor Único") que internamente
orqueste los 3 pipelines existentes con triggers inteligentes — sin que el usuario
tenga que elegir modo, y manteniendo la calidad del fusion con el costo del easyocr.

### 8.2 Arquitectura propuesta: `OCRManager`

```
                    ┌─────────────────────────────┐
                    │       OCRManager (nuevo)    │
                    │  ocr_engine.py (refactor)   │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────┬───────────┼───────────────┬───────────────┐
        ▼              ▼           ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐ ┌───────────┐  ┌────────────┐  ┌──────────────┐
   │ Tier 1  │   │ Tier 2   │ │ Tier 3    │  │ Tier 3.5   │  │ Post-OCR     │
   │ EasyOCR │──▶│ RapidOCR │─▶│ U-OCR     │─▶│ Ruta C     │─▶│ merge+       │
   │ GPU     │   │ CPU      │  │ daemon    │  │ (globos)   │  │ filtros      │
   └─────────┘   └──────────┘  └───────────┘  └────────────┘  └──────────────┘
```

**Reglas de decisión (dónde vive el trigger hoy):**

| Condición (heurística barata, ~20-100ms) | Acción |
|---|---|
| `_page_has_large_image_panel()` (dark_ratio > 0.18) | Tier 3 (U-OCR) + Ruta C |
| `< 3 bloques` Y `avg_conf < 0.20` | Tier 3 (U-OCR) + Ruta C |
| 0 bloques | Tier 3 (U-OCR) + Ruta C |
| Todo lo demás | Tier 1+2 (E+R) solo — **~15-30s/página** |
| Daemon no listo / error | Tier 1+2 solo (degradación automática) |
| `_uocr_inferring` activo (otro worker) | RapidOCR CPU solo |

### 8.3 Cambios por archivo

| Archivo | Cambio | Riesgo |
|---|---|---|
| **NUEVO `ocr_engine.py`** | `OCRManager` que envuelve `_detect_and_ocr`, `_ocr_with_unlimited`, Ruta C, `_fusionar_blocks_multi` con la lógica de trigger actual. | Bajo (mueve lógica existente) |
| `routes/api.py` | `process_page` delega en `OCRManager.run(img, lang)`; el modo "fusion" pasa a ser el DEFAULT del endpoint. | Medio (cambiar default) |
| `process_all_pages.py` | Default `--ocr-mode fusion` + `--workers 2`. | Bajo |
| `ocr_utils.py` | Sin cambios de lógica; exponer constantes de trigger como parámetros del manager. | Bajo |
| `config.py` | Añadir `OCR_DEFAULT_MODE = "fusion"`; tabla de pesos en el manager. | Bajo |
| `index.html` / `app.js` | El selector de OCR pasa a "Automático (recomendado)" / "Fuerza Unlimited". | Medio (UI) |
| `uocr_daemon.py` | (Opcional) aceptar `confidence_mode` en el payload para futuras señales. | Bajo |

### 8.4 Mejoras adicionales detectadas en la investigación

1. **Cache de decisiones por página**: si la página N ya disparó U-OCR y no recuperó
   nada (0 bloques nuevos), marcar las páginas N+1..N+k con la misma firma (mismo
   patrón de oscuridad) para NO re-disparar → ahorra tiempo en capítulos repetitivos.
2. **Batch de páginas al daemon**: el daemon ya serializa con `_infer_lock`; enviar
   páginas encoladas en una sola petición con varias imágenes (si el modelo lo permite)
   o al menos ordenar la cola para que las páginas trigger vayan primero.
3. **Estimación de confianza mejorada**: el fontSize ya se estima (h*0.8); el daemon
   podría devolver el ratio de área de texto/área de bloque como señal débil adicional.
4. **Ruta C con RapidOCR** en vez de EasyOCR cuando `_uocr_inferring` esté activo:
   los globos re-OCR no necesitan GPU si usamos RapidOCR sobre el crop upscaleado.
5. **Warmup del daemon bajo demanda**: si el trigger dispara por primera vez y el daemon
   aún carga (~8 min), mostrar estado en el frontend en vez de degradar silenciosamente.

### 8.5 Métricas objetivo (del benchmark v4.2, PDF nuevo 53 págs)

| Métrica | Antes (fusion v1) | Meta (unificado v5) |
|---|---|---|
| Tiempo capítulo 53 págs | ~150 min | **~45-50 min** (trigger selectivo ya implementado) |
| Tiempo página normal | ~15s | ~15s (sin cambio) |
| Tiempo página U-OCR | 130-500s | 130-500s (costo real de la VLM) |
| Cobertura | 100% páginas | 100% (degradación automática) |
| CER págs. artísticas | 0.72-0.82 | mismo o mejor |

### 8.6 Orden de ejecución sugerido

1. Extraer `OCRManager` (sin cambiar comportamiento) + tests.
2. Hacer `fusion` el modo por defecto (backend y runner).
3. UI: selector "Automático" + badge de estado del daemon.
4. Mejoras incrementales (§8.4): cache de decisiones, batch al daemon.
5. Benchmark final capítulo completo + actualizar docs.

---

## 9. Estado actual del run en curso (2026-08-04)

- **Servidor** 5174 (PID 10928) + **daemon** U-OCR 5177 (PID 15948, ready).
- **Run del capítulo fusion**: 20/53 páginas procesadas, 0 errores, ~9,870s acumulados
  (el costo U-OCR domina). Sigue corriendo con workers=2 y timeout 1800s.
- Checkpoint en `resultados_progreso.json` (total_pages=53).
