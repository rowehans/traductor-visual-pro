# Plan: Fusión de los 3 motores OCR en uno solo

> **Estado**: PLAN v4.1 — veredicto empírico incluido (prueba real con el daemon, 2026-08-03). La confianza por logits queda **descartada**; se usa confianza heurística. Ruta ganadora: **Cascada (B) + fusión de bloques**. La Ruta C (re-OCR del bloque image completo) queda **descartada por evidencia empírica** — se redefine a nivel de globo. Además se corrigió un **bug preexistente**: el daemon usaba `cv2` (ausente en `env_uocr_gpu`) → el re-OCR artístico crasheaba siempre; ahora usa PIL.

---

## 1. Contexto: los 3 motores actuales

| Motor | Dónde vive | Fortaleza | Debilidad | Coste |
|---|---|---|---|---|
| **EasyOCR** | `ocr_utils._detect_and_ocr()` tier 1 (GPU) | Rápido, fiable en globos normales | Garblea texto artístico, ignora títulos dorados | ~1-8s/pág |
| **RapidOCR** | `ocr_utils._detect_and_ocr()` tier 3 (CPU ONNX) | Detecta títulos estilizados que EasyOCR ignora | Menos preciso en general; CPU lento como principal | ~2-4s/pág |
| **Unlimited-OCR** | Daemon `uocr_daemon.py` (GPU 4-bit NF4, DeepSeek-OCR 3B MoE ~500M activos) | Precisión casi perfecta donde detecta; emite tipos de bloque (text/title/header/image/footer/page_number); R-SWA para multi-página | Lento (20-60s/pág), clasifica paneles artísticos como `image` | ~20-60s/pág |

**Ya existe** (verificado en código):
- Fusión 2-vías `_fusionar_blocks()` (ocr_utils.py:209: dedup texto normalizado + IoU>40% con `_block_score`/`_overlap_ratio` internas).
- `_detect_and_ocr()` híbrido con mag_ratio adaptativo (1.3→1.8) y pre-filter CLAHE+sharpen.
- Daemon con `_recover_art_dialogue()` (uocr_daemon.py:175): re-OCR de bloques `image` >30% de página vía recorte + letterbox 640 + crop_mode=False, mapeo round-trip validado (±4px).
- `_ocr_with_unlimited()` (routes/api.py:391) hardcodea `confidence: 1.0` y `textColor: "#000000"`.
- Selector UI + polling de estado + toast de fallback (implementados).

---

## 2. 🔬 VEREDICTO EMPÍRICO (2026-08-03) — prueba con el daemon real

Se ejecutó `test_uocr_stream_conf.py` con el modelo 4-bit real sobre `benchmark_page11.png` (página donde U-OCR lee bien):

### Q1: ¿El modelo emite tokens confidence/fontSize en el stream? → **NO. Confirmado empíricamente.**

- Pase A (stream RAW vía `infer()`): 9 bloques `<|det|>` (2 header, 1 title, 4 text, 1 footer, 1 page_number).
- Búsqueda en el stream completo: **0 ocurrencias** de `confidence`, `fontSize`, `font_size`, `score`, `prob`.
- Los tokens `Ġconfidence`/`ĠfontSize` que existen en el tokenizer pertenecen al modelo base, no a la salida de parsing. (Coincide con la doc: la salida es solo `<|det|>type [x,y,w,h]<|/det|>texto`.)

### Q2: ¿La confianza por logits es viable y diferenciadora? → **NO. Se descarta.**

- Método: monkeypatch de `model.generate` para añadir `output_scores=True` a la llamada real de `infer()` (preprocesado 100% idéntico al Pase A). Generación completa: 300 tokens, 300 steps de scores.
- Resultado por bloque (media geométrica de p(token)):
  - title: conf=**0.9980** (22 tok)
  - text: conf=**0.9963** (19 tok)
  - **Spread total: 0.0018** — todos los bloques saturan a ~0.997.
- **Causa**: generación greedy (`do_sample=False`) siempre produce tokens argmax con probabilidad alta. La media geométrica no puede distinguir un bloque mal leído (`Y., POR.SURVESTO...JSU`) de uno bien leído (`PODRÍA UTILIZARLO...`) — ambos obtendrían ~0.99.
- Además: overhead de memoria real (scores = vocab×steps, ~150MB por 300 tokens) y complejidad de mapeo token→bloque, para un beneficio nulo.

### ✅ Decisión: **confianza HEURÍSTICA** (no logits)

Para U-OCR se estima confianza con señales que sí discriminan:

| Señal | Regla heurística |
|---|---|
| **Tipo de bloque** | `title`/`text` → 0.9 base; `header` → 0.7; `image` → 0.0 (nunca); `footer`/`page_number` → filtrados |
| **Calidad del texto** | ratio vocales ≥ 0.25 y longitud ≥ 2 → +0.05; texto todo-símbolos/1 char → ×0.5 |
| **Re-OCR artístico** | bloque con `from_art_recrop` → 0.8 base (vino de un panel recuperado) |
| **Consenso** | si EasyOCR o RapidOCR detectan el MISMO texto en la MISMA región → +0.15 (voto doble) |
| **Tamaño** | fontSize estimado 10-40 px → +0.03 (rango natural de diálogo) |

Fusión final = `conf_heuristica * peso_motor` (EasyOCR×1.0, RapidOCR×0.9, U-OCR×1.1).

---

## 3. 🔍 Exploración de 3 rutas de fusión (comparadas)

### Ruta A — Fusión de bloques completa (los 3 motores en cada página)
- Correr EasyOCR + RapidOCR + U-OCR en TODA página y fusionar.
- ➕ Máxima cobertura, modelo mental simple.
- ➖ **Latencia inaceptable**: U-OCR 20-60s/pág × 128 págs = 45-130 min por capítulo. Para una página normal bien leída por EasyOCR no aporta nada.
- ➖ Contención de VRAM (3 modelos).

### Ruta B — Cascada rápida→precisa con triggers
- EasyOCR+RapidOCR siempre (~10s). U-OCR completo **solo si** dispara el trigger: confianza media < 0.25, o < 3 bloques, o bloque image > 15% de la página.
- ➕ Páginas normales rápidas; precisas solo las difíciles.
- ➕ Patrón documentado en producción (cascades multi-OCR).
- ➖ En páginas artísticas (3/12) sigue pagando 20-60s.
- ➖ La fusión solo ocurre en páginas con trigger.

### Ruta C — Re-OCR por región (cirugía selectiva) ← **la clave del v4**
- **Hallazgo empírico de las págs. 3/12**: el diálogo perdido SIEMPRE está dentro de los bloques `image` que U-OCR ya delimitó. El fix no es mejorar el detector, es **re-examinar esas regiones**.
- En vez de re-OCR con U-OCR (lento), re-OCR con **EasyOCR GPU** (rápido, ~0.9s): recortar el panel `image`, upscale 2-3×, EasyOCR, mapear de vuelta. Es exactamente lo que ya hace `_recover_art_dialogue` pero con el motor rápido.
- ➕ **Solo paga en regiones problemáticas** (~1-3s por panel, no 60s).
- ➕ No requiere que el daemon esté listo para beneficiarse (EasyOCR ya está en el server).
- ➖ Necesita detección de regiones image: del daemon si está listo, o heurística propia si no.

### 🏆 Ruta ganadora: **B + C combinadas (Cascada + re-OCR por región)**

> ⚠️ **ACTUALIZADO con benchmark empírico (v4.1, 2026-08-03)**: la Ruta C tal como estaba diseñada (re-OCR del bloque `image` COMPLETO) **NO funciona** — ver §3.5. La Ruta C se redefine a nivel de **globo individual** (la granularidad del recorte es crítica).

```
Página →
  Paso 1: _detect_and_ocr(use_hybrid=True)  ← EasyOCR GPU + RapidOCR CPU (~10s)
          → bloques_hibridos + confianza_avg
          │
          ▼
  Paso 2: Detectar regiones problemáticas:
          a) del daemon (si ready): bloques type=image >15% página
          b) heurística local (si daemon off): zonas grandes (≥8% página)
             donde EasyOCR/RapidOCR devolvieron 0 bloques
          │
          ▼
  Paso 3: Re-OCR por región con EasyOCR GPU:
          recortar región → upscale 2-3× → _run_ocr_on_image → mapear
          (reutiliza _recover_art_dialogue como plantilla, motor rápido)
          │
          ▼
  Paso 4: _fusionar_blocks_multi([hibridos, recuperados, (uocr opcional)])
          1. Alinear por texto normalizado + Levenshtein (misma región)
          2. Votación: 2 motores coinciden → confianza +0.15
          3. NMS espacial IoU >40% con score calibrado (conf heurística × peso)
          │
          ▼
  bloques unificados → filtro watermarks → resto del pipeline
```

**El trigger del Pase B (U-OCR completo) se mantiene solo como tier opcional**: si tras B+C la página sigue con <3 bloques o confianza media <0.2, y el daemon está ready, entonces sí se corre U-OCR completo (~20-60s) y se fusiona. Así la página 11 (que U-OCR lee perfecto) también puede beneficiarse cuando los motores rápidos fallan.

---

## 3.5 🔬 RESULTADOS EMPÍRICOS — benchmark de la Ruta C (2026-08-03, págs. 3/11/12)

Se ejecutó `benchmark_ruta_c.py` (daemon real + EasyOCR GPU). Se recortó cada bloque `image` grande (>15% página) que U-OCR delimitó, se upscaleó 2×/3×, y se re-OCReó con EasyOCR GPU; se comparó con el re-OCR propio del daemon (`from_art_recrop`).

### Hallazgo #1 — Bug preexistente corregido: el daemon crasheaba con cv2

`uocr_daemon.py::_recover_art_dialogue` importaba `cv2`, pero `env_uocr_gpu` NO tiene OpenCV instalado (el venv GPU solo cargaba torch/transformers). El re-OCR artístico **crasheaba siempre** con `ModuleNotFoundError: No module named 'cv2'` → HTTP 500. Se reescribió con **PIL puro** (el venv sí tiene PIL, lo usa el propio modelo): `Image.open` + `crop()` + `resize(LANCZOS)` + `paste()` + `save()`. Sintaxis OK en ambos venvs, daemon funcional.

### Hallazgo #2 — La Ruta C a nivel de bloque image COMPLETO no funciona

| Página | Bloque image | Tamaño | EasyOCR-crop (2×/3×) | U-OCR recrop | Ref. encontrada |
|---|---|---|---|---|---|
| 3 | 1 (719×980) | 30% pág | solo ruido: `AD1`, `TA`, `D` | 0 bloques | **0/2** palabras ("INCREÍBLE REALMENTE") |
| 11 | 0 (U-OCR lee bien) | — | — | — | — |
| 12 | 1 (1015×972) | 51% pág | solo cabeceras: `13/7/26...`, `Capítulo 43...` | 2 bloques (cabeceras) | **0/8** palabras ("ERA UNA PROPUESTA...") |

**Conclusión**: recortar el panel `image` completo (que incluye arte + diálogo + a veces cabeceras del margen) y re-OCRearlo **NO recupera el diálogo** — ni con EasyOCR (hasta 3×) ni con el propio U-OCR. El diálogo pintado es una fracción minúscula del panel; al upscalear el panel entero, el texto sigue siendo demasiado pequeño/artístico. En la pág. 12 el panel image incluso incluye las cabeceras superiores (por eso recrop devolvió solo "13/7/26" y "Capítulo 43" — texto grande, no el globo).

### Hallazgo #3 — La granularidad del recorte es CRÍTICA

El análisis previo (`analizar_dialogo_artistico.py` + resumen.json) ya demostró que el recorte del **globo individual** (411×245 en pág. 12, roundness 0.739) SÍ recupera el diálogo con re-OCR (conf 0.96). El benchmark actual confirma el inverso: el recorte del panel completo (que contiene el globo como fracción pequeña) NO lo recupera. → **La Ruta C correcta necesita detección de globos/regiones de texto dentro del panel antes de recortar.**

### Recomendación revisada para la Ruta C (v4.1)

1. Dentro de cada bloque image grande, detectar **regiones de texto** (OpenCV: blobs oscuros sobre fondo claro — ya existe `_build_glyph_mask_for_bubble` / heurísticas de `_is_inside_speech_bubble` en ocr_utils.py).
2. Recortar cada región de texto a nivel de globo, upscale 3-4×, re-OCR con EasyOCR (rápido).
3. Para el caso de la pág. 3 (SFX pintado sin globo), el re-OCR regional tampoco ayuda — el SFX es irrecuperable con OCR estándar; hay que aceptarlo o usar U-OCR full-page como tier opcional.

---

## 3.6 🔬 BENCHMARK DE CAPÍTULO COMPLETO — modo `fusion` (2026-08-03, 128 págs)

Se implementó la Ruta B (fusión de bloques multi-motor + confianza heurística + cascada con trigger U-OCR) en `ocr_utils.py`/`routes/api.py`, y se procesó el capítulo completo (128 págs) con `process_all_pages.py` (workers=2, timeout 900s, trigger U-OCR: <3 bloques o conf<0.25 o image>15%).

### Resultados vs. benchmarks de línea base

| Modo | Tiempo total | Bloques | Traducidos | Tasa | Págs | Errores | Promedio/pág |
|---|---|---|---|---|---|---|---|
| **EasyOCR solo** (benchmark Jul, workers=4) | **425s (7.1 min)** | 623 | 519 | **83.3%** | 128 | 0 | 3.3s |
| **Modo auto** (checkpoint Jul 29) | 1521s (25.3 min) | 517 | 424 | 82.0% | 128 | 0 | 11.9s |
| **Modo fusion** (run completo) | 9038s (**150.6 min**) | 590 | 408 | **69.2%** | 128 | 0 | 70.6s |

### Desglose de tiempo (fusion)

- **41 páginas dispararon U-OCR** (>30s) → **8609s = 95% del tiempo total**. Promedio de esas páginas: **210s** (máx 1439s en pág. 30 por contención de GPU con EasyOCR del servidor).
- **87 páginas normales** (solo EasyOCR+fusión) → 429s = 5%, **4.9s/pág promedio** (vs 3.3s EasyOCR solo; +1.6s de overhead de fusión y trigger).

### Recuperación de diálogo artístico (CER vs ground truth)

| Página | Ground truth | Fusion src | CER |
|---|---|---|---|
| 3 | INCREIBLE REALMENTE (SFX pintado) | `TEALMENT- NCREIBLE` + `YCREIBLE..` | **0.819** |
| 12 | ERA UNA PROPUESTA QUE SOLO PODIA BENEFICIARME PERO (globo en panel) | `RA CNA PROPOESTA OUE SOLC POCA FENEHCASE. FERT.` | 0.717 |
| 11 | (U-OCR lee bien) | — | — |

### Veredicto del benchmark

1. **Cobertura**: 128/128 páginas, 0 errores — igual que los benchmarks. El modo fusion no rompe nada.
2. **CER en págs artísticas**: mejora clara vs. EasyOCR full-page. En la pág. 3 EasyOCR solo NO leía nada del SFX (0/2 palabras); fusion recupera `NCREIBLE` (parcial, CER 0.819). En la pág. 12 fusion captura la propuesta del globo (antes solo cabeceras). **La fusión SÍ recupera diálogo que EasyOCR pierde.**
3. **Tiempo: EL PROBLEMA**. 150.6 min vs 7.1 min (EasyOCR solo) = **21× más lento**. El 95% del tiempo lo consumen las 41 páginas con U-OCR disparado (~210s c/u). Dos causas: (a) el trigger U-OCR dispara demasiado (41/128 = 32% de páginas), y (b) **contención de GPU**: cuando el daemon U-OCR infiere, EasyOCR del servidor también está en GPU → el daemon se ralentiza de 83s (standalone) a 140-1439s.
4. **Tasa de traducción menor (69.2% vs 83.3%)**: contraintuitivo pero explicable — la fusión detecta más bloques problemáticos (diálogo artístico, bloques U-OCR con ruido) que no se traducen, y los bloques fusionados en páginas artísticas son más difíciles. La cobertura de texto real mejora aunque el ratio bruto baja.

### Optimizaciones requeridas (v4.2)

1. **Reducir el trigger U-OCR**: solo disparar en páginas con `image > 15%` O `< 3 bloques` con `EasyOCR conf < 0.2` — NO en cualquier página con <3 bloques. Esto corta las 41 páginas → ~10-15.
2. **Serializar GPU**: en modo fusion, pausar la inferencia de EasyOCR mientras el daemon U-OCR procesa (mutex compartido o `--workers 1` con el daemon), o usar RapidOCR (CPU) como re-OCR de paneles en vez de EasyOCR para no competir por VRAM.
3. **U-OCR opcional por página**: mover U-OCR a `--ocr-mode fusion` pero con `USE_UOCR_PER_PAGE` configurable, y documentar el costo real (~210s/pág) para que el usuario decida.
4. **Meta**: con las optimizaciones 1+2, el tiempo estimado cae a ~30-40 min (págs U-OCR serializadas ~150s × 15 = 2250s + 113 págs normales × 5s = 565s ≈ 47 min). Sigue siendo 6× más lento que EasyOCR solo, pero con recuperación de diálogo artístico real.

### Resultados v4.2 (2026-08-04, PDF nuevo de 53 págs)

Implementadas las optimizaciones 1+2+3 sobre el PDF nuevo (53 págs, 2.87MB):

| Cambio | Detalle |
|---|---|
| **Trigger selectivo** | `image > 15%` O (`< 3 bloques` Y `conf < 0.2`). `UOCR_TRIGGER_CONF` 0.25 → **0.20**. Ya NO dispara en cualquier página con <3 bloques (antes 41/128 = 32% de páginas). |
| **Mutex GPU** | `_gpu_lock` (RLock) compartido en `ocr_utils.py`: EasyOCR (`_run_ocr_on_image`) y daemon U-OCR (`_ocr_with_unlimited`) se serializan — un solo motor GPU a la vez. |
| **Degradación CPU** | Flag `_uocr_inferring` (Event): mientras el daemon infiere, los workers de otras páginas degradan a **RapidOCR CPU puro** (no tocan la GTX) en vez de esperar el mutex → las páginas normales avanzan en paralelo. |

**Benchmark (9 páginas del PDF nuevo, modo fusion):**

| Página | Tiempo | Bloques | Engines | Disparó U-OCR |
|---|---|---|---|---|
| 1 | 57.6s | 4 | easyocr+rapid | ✗ (cold start server) |
| 2 | 592.0s | 6 | easyocr+rapid | ✓ (image>15%) — primera inferencia U-OCR incl. warmup |
| 7 | 28.7s | 5 | easyocr+rapid | ✗ |
| 15 | 247.7s → **152.0s** | 5 | +unlimited | ✓ (image>15%) — **-39% con degradación CPU** |
| 30 | 11.2s | 5 | easyocr+rapid | ✗ |
| 45 | 8.0s | 2 | easyocr+rapid | ✗ |
| 3 | 8.6s | 7 | easyocr+rapid | ✗ |
| 5 | 497.8s → **132.4s** | 3 | +unlimited | ✓ (force) — **recupera `ERA UNA PROPUESTA`/`HOY, LUEGO DE`/`ME GUESTARÍA`** |
| 11 | 17.2s | 8 | easyocr+rapid | ✗ |

**Lectura de resultados:**
- **Trigger selectivo funciona**: 7/9 páginas (normales) NO disparan U-OCR → 8-29s c/u. Solo las páginas con panel image >15% (portadas/ilustraciones) pagan el costo U-OCR.
- **Degradación CPU real**: p15 247.7→152s (-39%) y p5 497.8→132.4s (-73%) al quitar la contención VRAM de EasyOCR durante la inferencia del daemon.
- **Calidad preservada**: la pág. artística p5 recupera el diálogo pintado que EasyOCR solo pierde (`ERA UNA PROPUESTA`, conf 1.0).
- **Costo residual**: cada página U-OCR paga ~130-250s (inferencia VLM 3B 4-bit en GTX 1050 Ti + warmup). Con ~10-15 págs U-OCR en el capítulo: ~10×200s = 2000s (33 min) + 43 págs normales × 15s = 645s (11 min) ≈ **~44 min** (workers=2, degradación CPU activa) — cerca de la meta de ~47 min y **3.4× menos que los 150.6 min originales**.

### §3.7 🔬 OVERHEAD PURO DE LA FUSIÓN — el merge NO cuesta nada (2026-08-04, PDF nuevo 53 págs)

Benchmark con `benchmark_fusion_overhead.py`: 53 págs × 3 modos, dpi=180, mismos parámetros:

| Modo | Total (53 págs) | t_ocr/pág | Bloques | Notas |
|---|---|---|---|---|
| **EasyOCR GPU puro** (`pure_easyocr`) | **219.6s** | **3.68s** | **22** | solo EasyOCR, sin RapidOCR — detecta muy poco |
| **EasyOCR híbrido** (default de la app) | 427.1s | 7.25s | 225 | EasyOCR+RapidOCR+_fusionar_blocks (¡ya es fusión!) |
| **Fusion sin U-OCR** (`disable_uocr`) | **403.8s** | **6.91s** | 225 | misma detección, merge idéntico |

*`t_ocr` extraído del log del servidor (tiempo puro de OCR, sin traducción/caché) para eliminar el sesgo de hits de caché de traducción entre corridas.*

**Hallazgo 1 — el "modo easyocr" ya es híbrido**: `_detect_and_ocr` corre `use_hybrid=True` por defecto. EasyOCR GPU puro solo detecta **22 bloques** vs **225** del híbrido → RapidOCR aporta ~90% de la detección en este manga (tipografía artística/densa). Los benchmarks históricos de "EasyOCR solo" (7.1 min, 623 bloques) eran en realidad EasyOCR+RapidOCR.

**Hallazgo 2 — el merge de fusión es ~0**: t_ocr fusion (6.91s) vs easyocr híbrido (7.25s) = **-0.34s/pág (-4.7%, ruido)** y **224/225 textos idénticos** (única diferencia: una capitalización en p32). El `_fusionar_blocks_multi` solo se invoca cuando U-OCR dispara; sin daemon, la fusión NO añade ningún costo perceptible.

**Conclusión práctica**: el overhead de 150 min del modo fusion completo está **100% en la inferencia del daemon U-OCR** (~130-250s/página), NO en la fusión de bloques. La estrategia correcta sigue siendo la v4.2: minimizar el nº de páginas que disparan U-OCR (trigger selectivo) y serializar/degradar GPU — el merge en sí no requiere optimización.

---

## 4. Cambios por archivo

| Archivo | Cambio |
|---|---|
| `ocr_utils.py` | `_fusionar_blocks_multi(sources: list[(blocks, peso_motor)], engine="...")` — versión N-motores de `_fusionar_blocks`: helpers `_block_score`/`_overlap_ratio` a nivel módulo + alineación Levenshtein + votación. Nueva `_estimate_confidence_heuristic(block, type)` para U-OCR. Nueva `_recover_regions_with_easyocr(img, regiones)` (recorte+upscale+re-OCR, plantilla de `_recover_art_dialogue`). `_detect_and_ocr` gana flag `recover_art=True`. |
| `routes/api.py` | Modo `fusion` (default). `_ocr_with_unlimited()` deja de hardcodear 1.0 → usa `_estimate_confidence_heuristic`. Trigger de tier opcional. `ocr_engine_used`/`engines_used` en respuesta. |
| `uocr_daemon.py` | **Sin cambios** (ya tiene `_recover_art_dialogue`). Opcional: exponer `regions` (bloques image) en `/ocr` sin re-OCR para que el server los use con EasyOCR — ahorra trabajo duplicado. |
| `config.py` | `UOCR_TRIGGER_CONF` (0.25), `UOCR_TRIGGER_MIN_BLOCKS` (3), `UOCR_IMAGE_BLOCK_RATIO` (0.15), `ART_REGION_MIN_RATIO` (0.08), `OCR_ENGINE_WEIGHTS` {easyocr:1.0, rapid:0.9, uocr:1.1}. |
| `index.html` | Selector: opción `fusion` (Fusión Inteligente) como default recomendado. |
| `app.js` | `getSelectedOcrMode()` default `"fusion"`; badge muestra motores usados; toast si degradó. |
| `tests/test_api.py` | Añadir `"fusion"` a modos válidos; test de la ruta C con mock. |
| `test_uocr_art_recover.py` | Ampliar: test de `_fusionar_blocks_multi` (dedup, Levenshtein, votación, pesos) y de `_recover_regions_with_easyocr` (round-trip coordenadas). |
| `process_all_pages.py` | `OCR_MODE` default `"fusion"`. |
| `PLAN_FUSION_OCR.md` | Este documento. |

---

## 5. Confianza heurística en detalle (reemplaza a logits)

```python
def _estimate_confidence_heuristic(block, block_type=None):
    conf = {"text": 0.90, "title": 0.90, "header": 0.70}.get(block_type, 0.80)
    text = str(block.get("text", ""))
    # Calidad del texto
    letters = [c for c in text if c.isalpha()]
    if not letters:
        conf *= 0.5
    elif len(text) <= 2:
        conf *= 0.7
    else:
        vocals = sum(c in "aeiouáéíóúAEIOUÁÉÍÓÚ" for c in letters)
        if vocals / len(letters) >= 0.25:
            conf += 0.05
    # Re-OCR artístico (vino de un panel recuperado)
    if block.get("from_art_recrop"):
        conf = max(conf, 0.80)  # el recorte+upscale ya mejoró la lectura
    # fontSize en rango de diálogo
    fs = block.get("fontSize", 0)
    if 10 <= fs <= 40:
        conf += 0.03
    return round(min(1.0, conf), 4)
```

- **Consenso en la fusión**: si `_normalize_text` de un bloque U-OCR coincide con uno EasyOCR/RapidOCR en la misma región (IoU > 0.3), se suma +0.15 al ganador y se descarta el duplicado.
- **Alineación Levenshtein**: dos bloques con IoU > 0.4 pero texto que difiere levemente (distancia ≤ 30% de la longitud) se consideran el mismo texto → gana el de mayor score calibrado.

---

## 6. Validación

1. **Unitario**: `_fusionar_blocks_multi` (dedup, Levenshtein, votación, pesos); `_estimate_confidence_heuristic` (casos: text/title/image/garbage); `_recover_regions_with_easyocr` (round-trip coordenadas con mock).
2. **Sintaxis**: `py_compile` en `env/` y `env_uocr_gpu/`; `node --check app.js`.
3. **Tests API**: suite completa por archivo (`test_api.py` 105 + `test_translator.py`/`test_ocr_functions.py`/`test_ocr_utils.py` 181).
4. **Benchmark real** (págs. 3, 11, 12 — las artísticas donde EasyOCR falla):
   - Cobertura de diálogo: ¿`fusion` recupera págs. 3/12 y mantiene 11?
   - CER vs referencia.
   - Tiempo/página: normal ~10-12s; artística ~12-15s (re-OCR regional, NO 60s).
   - Distribución de confianzas de los 3 motores (para ajustar pesos con datos).
5. **E2E**: `/api/process-page` con `ocr_mode="fusion"` (server + daemon reales).

---

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Re-OCR regional duplica bloques | Dedup texto + Levenshtein + NMS IoU (ya probado en 2-vías) |
| Región image mal delimitada → re-OCR basura | Umbral mínimo de área (8% página) + filtro post-merge existente; `conf < 0.15` se descarta |
| Escalas de confianza distintas | Confianza heurística uniforme + pesos por motor |
| Páginas normales más lentas (re-OCR regional extra) | Solo re-OCR regiones con 0 bloques y área ≥8%; normalmente no hay → overhead ~0 |
| Daemon no listo | Ruta C usa EasyOCR (ya cargado en el server) — el beneficio principal NO depende del daemon |
| U-OCR completo opcional lento | Solo se activa si tras B+C sigue pobre (<3 bloques o conf<0.2) |

---

## 8. Entregables finales

1. Modo `ocr_mode="fusion"` default: cascada B + re-OCR regional C (+ U-OCR opcional).
2. `_fusionar_blocks_multi` con alineación Levenshtein + votación + pesos y tests.
3. `_estimate_confidence_heuristic` (reemplaza el hardcode 1.0 de U-OCR) y tests.
4. `_recover_regions_with_easyocr` (recorte+upscale+re-OCR con motor rápido) y tests.
5. Constantes configurables en `config.py`.
6. Selector UI "Fusión Inteligente" + badge con motores usados.
7. Benchmark comparativo págs. 3/11/12 con distribución de confianzas.
8. Documentación en AGENTS.md (§4).

---

## 9. Orden de ejecución

1. `ocr_utils.py`: `_estimate_confidence_heuristic` + tests.
2. `ocr_utils.py`: `_fusionar_blocks_multi` (N-motores, Levenshtein, votación, pesos) + tests.
3. `ocr_utils.py`: `_recover_regions_with_easyocr` + tests (round-trip).
4. `config.py`: constantes.
5. `routes/api.py`: modo `fusion` (B+C+U-OCR opcional), `engines_used`.
6. Frontend: selector + `app.js` (default fusion, badge motores).
7. `tests/test_api.py`: modos + trigger.
8. py_compile + node --check + tests completos.
9. Benchmark págs. 3/11/12 (cobertura + tiempo + distribución confianzas).
10. Revisión de código + AGENTS.md.

---

## 10. Ideas futuras (fuera de alcance v4)

- **Parsing multi-página**: aprovechar R-SWA de U-OCR para parsear un capítulo completo en un solo request (daemon `infer_multi`).
- **LaMa para inpainting**: reemplazar el inpainting OpenCV por LaMa (SOTA manga).
- **Corrección LLM post-fusión**: el pipeline CT2 ya actúa como capa de corrección contextual.
- **Aprovechar el re-OCR de U-OCR como tercera fuente en la fusión** cuando el daemon está listo (fusionar B+C+U-OCR-regional sin correr U-OCR completo).
