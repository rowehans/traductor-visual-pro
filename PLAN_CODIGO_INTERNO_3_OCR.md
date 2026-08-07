# 🔬 Plan: Hacer funcionar el código interno de los 3 OCR juntos

> **Fecha**: 2026-08-04 · **Basado en**: lectura directa del código fuente instalado
> (`env/Lib/site-packages/easyocr/`, `env/Lib/site-packages/rapidocr_onnxruntime/`,
> `models_unlimited_patched/`) + INVESTIGACION_3_OCR.md + PLAN_FUSION_OCR.md.
> **Objetivo**: entender el funcionamiento interno de los 3 motores al nivel del
> código fuente real (no solo de las API que usa la app) y diseñar el plan para
> que trabajen juntos en un solo motor unificado.

---

## 1. Lo que dice el código fuente real (investigación a nivel de paquete)

### 1.1 EasyOCR 1.7.2 — `env/Lib/site-packages/easyocr/`

**Arquitectura interna (2 fases clásicas detección → reconocimiento):**

```
readtext()                                (easyocr.py:440)
  ├── reformat_input(image)                → (RGB, gris)
  ├── detect()                             (easyocr.py:311)
  │     ├── get_textbox(detector, ...)     → CRAFT (detection.py:92)
  │     │     └── test_net()               → run CRAFT en 2 escalas + TTA
  │     └── group_text_box()               → agrupa cajas en líneas
  └── recognize()                          (easyocr.py:353)
        ├── get_text(...)                  (recognition.py:186)
        └── CTCLabelConverter              → decoder greedy/beam
```

**Detalles clave del código fuente:**

| Pieza | Archivo | Qué hace |
|---|---|---|
| `CRAFT` (detector) | `craft.py:30` | U-Net-like con `double_conv`, salida 2 canales (region score + affinity score), backbone VGG16-BN. **Detecta caracteres y su afinidad** — no cajas directas. |
| `test_net()` | `detection.py:24` | Redimensiona a `canvas_size` con `mag_ratio`, corre CRAFT, hace TTA (reflejo), y `craft_utils.getDetBoxes` convierte mapas de calor → polígonos. |
| `get_detector()` | `detection.py:74` | Carga `craft_mlt_25k.pth`, `quantize=True` por defecto (int8 dinámico). |
| DBNet alternativo | `detection_db.py` | EasyOCR también trae DBNet (`get_detector(backbone='resnet18')`) — **la app usa CRAFT** (default). |
| Recognizer gen2 | `recognition.py:153` | Modelo `vgg_model.Model` con CTC (`CTCLabelConverter`), `latin_g2.pth`. |
| `group_text_box()` | `craft_utils.py` | Une cajas individuales en líneas usando `slope_ths`, `ycenter_ths`, `height_ths`, `width_ths`, `add_margin`. **Este es el "merge" nativo de EasyOCR.** |

**Cómo lo usa la app (ocr_utils.py):**
- `reader.readtext(img_rgb, detail=1, paragraph=False, min_size=6,
  text_threshold=0.15, low_text=0.10, link_threshold=0.3,
  canvas_size=min(max_dim,2500), mag_ratio=1.3)` — parámetros que la app afina
  para manga denso.
- `_ocr_results_to_blocks()` convierte `(bbox, text, conf)` → formato interno.

**Oportunidades de integración interna (lo que el código fuente permite):**
- `Reader.detect()` y `Reader.recognize()` son **públicos y separables** → se puede
  usar el detector de EasyOCR con el recognizer de otro motor, o viceversa.
- `group_text_box()` es reutilizable como merge de cajas alternativo al
  `_group_and_merge_blocks()` propio de la app.

---

### 1.2 RapidOCR 1.4.4 — `env/Lib/site-packages/rapidocr_onnxruntime/`

**Arquitectura interna (pipeline PP-OCR clásico, 3 etapas):**

```
RapidOCR.__call__()                       (main.py:66)
  ├── LoadImage()                         → numpy
  ├── preprocess()                        → resize con max_side_len=2000, ratio_h/w
  ├── maybe_add_letterbox()
  ├── auto_text_det() → TextDetector      (DBNet: PP-OCRv4 det ONNX)
  ├── get_crop_img_list()                 → recorta cada caja detectada
  ├── text_cls() → TextClassifier         (PP-OCRv4 cls ONNX: 0°/180°)
  └── text_rec() → TextRecognizer         (PP-OCRv4 rec ONNX: 320×48)
```

**Detalles clave del código fuente (config.yaml):**

| Parámetro | Valor | Significado |
|---|---|---|
| `max_side_len` | 2000 | límite de lado mayor en preproceso |
| `min_height` / `width_height_ratio` | 30 / 8 | filtra cajas demasiado estrechas |
| `Det.model_path` | `ch_PP-OCRv4_det_infer.onnx` | detector DBNet (736px limit_side_len) |
| `Det.thresh` / `box_thresh` | 0.3 / 0.5 | umbrales del mapa de probabilidad |
| `Det.unclip_ratio` | 1.6 | expansión de la máscara → caja |
| `Cls.model_path` | `ch_ppocr_mobile_v2.0_cls_infer.onnx` | clasificador de rotación |
| `Rec.model_path` | `ch_PP-OCRv4_rec_infer.onnx` | recognizer (48×320) |
| `text_score` | 0.5 | umbral de confianza de reconocimiento |

**Cómo lo usa la app (ocr_utils.py):**
- `engine(img_bgr)` devuelve `(result, elapse)` con `result = [[bbox, text, conf], ...]`.
- `_run_rapidocr()` convierte cada resultado → formato interno + filtros
  (`conf < 0.08`, `w/h < 3px`).
- Se ejecuta SIEMPRE en el híbrido (`use_hybrid=True` default) con `_preprocess_rapid`
  (pre-filter + CLAHE+sharpen).

**Oportunidades de integración interna:**
- **RapidOCR es 100% ONNX Runtime** → puede ejecutarse en CPU SIN tocar la GPU.
  Ideal como degradación cuando el daemon U-OCR infiere.
- `TextDetector`, `TextClassifier`, `TextRecognizer` son **clases separadas** — se
  pueden instanciar individualmente si solo se quiere el detector o el recognizer.
- `kwargs` de `__call__` (`box_thresh`, `unclip_ratio`, `text_score`) permiten
  **ajustar la detección por página** sin recargar modelos (¡barato!).
- **Tiene clasificador de rotación (Cls)** — EasyOCR no lo tiene. Útil para
  globos girados.

---

### 1.3 Unlimited-OCR (DeepSeek-OCR 3B) — `models_unlimited_patched/`

**Arquitectura interna (VLM multimodal):**

```
UnlimitedOCRForCausalLM.infer()           (modeling_unlimitedocr.py:787)
  ├── format_messages()                    → prompt "<image>document parsing."
  ├── load_pil_images()                    → lee la imagen
  ├── [crop_mode=True] dynamic_preprocess()→ recorta en grid de 640px + thumbnail
  │     ├── global_view = ImageOps.pad(image, (base_size=1024, 1024))
  │     └── local views (crops 640×640)    → hasta 32 crops
  ├── build input_ids con tokens de imagen  (image_token_id=128815)
  ├── generate()                           → DeepseekV2ForCausalLM (hidden 1280, 12 capas, MoE)
  │     └── TPSTextStreamer                → imprime tokens por stdout
  └── re_match(outputs)                    → extrae <|det|>type [x,y,w,h]<|/det|>
      └── result.md (texto limpio) + result_with_boxes.jpg
```

**Detalles clave del código fuente:**

| Pieza | Archivo | Qué hace |
|---|---|---|
| Vision encoder | `deepencoder.py:243` | `CLIPVisionEmbeddings` + atención NoTP + MlpProjector |
| LLM head | `modeling_deepseekv2.py` | DeepSeek-V2-Lite (1280 hidden, 12 capas, MoE 896) |
| `dynamic_preprocess()` | `modeling_unlimitedocr.py:175` | Recorta imagen en grid: `min_num=2, max_num=32, image_size=640` |
| `re_match()` | `modeling_unlimitedocr.py:44` | Regex de `<|det|>` y `<|ref|>` para extraer bloques |
| `TPSTextStreamer` | `modeling_unlimitedocr.py:386` | Hace `print()` de cada token decodificado → **por eso el daemon captura stdout** |
| `infer_multi()` | `modeling_unlimitedocr.py:~1150` | **¡Multi-imagen!** "Does NOT support crop mode" — sirve para batch de páginas |
| Config | `config.json` | vocab 129280, hidden 1280, 12 capas, MoE |

**Cómo lo usa la app:**
- Daemon (`uocr_daemon.py`) carga 4-bit NF4 (bitsandbytes) y captura stdout con
  `contextlib.redirect_stdout` para sacar las coordenadas `<|det|>`.
- `_parse_blocks()` (uocr_daemon.py) extrae `{type, x, y, w, h, text}` con regex.
- Re-OCR artístico: `_recover_art_dialogue()` con `crop_mode=False` (una sola vista 640×640).

**Oportunidades de integración interna (¡descubrimiento clave!):**
- **`infer_multi()` existe y soporta VARIAS imágenes por inferencia** — es la base
  para batchear páginas al daemon y amortizar el prefill del VLM.
- El VLM **emite tipos semánticos** (`text`, `title`, `header`, `image`, `footer`,
  `page_number`) — los otros 2 motores no distinguen. Útil como señal extra en la fusión.
- El VLM **lee texto en contexto visual** (entiende que "ERA UNA PROPUESTA" está
  pintado sobre el arte) — es lo que EasyOCR/RapidOCR no pueden hacer.
- Se puede pedir al modelo salida en formato JSON/estructurado cambiando el prompt
  (`format_messages` es configurable) — el prompt actual es `"<image>document parsing."`.

---

## 2. Veredicto: cómo se complementan los 3 internamente

| Capacidad | EasyOCR (CRAFT) | RapidOCR (PP-OCRv4) | Unlimited-OCR (VLM 3B) |
|---|---|---|---|
| Detección de texto normal | ✅ | ✅ (mejor en títulos) | ✅ |
| Diálogo pintado EN arte | ❌ | ❌ | ✅ (semántico) |
| Confianza real | ✅ | ✅ | ❌ (heurística) |
| Tipos semánticos | ❌ | ❌ | ✅ (text/title/header/image) |
| Rotación (Cls) | ❌ | ✅ | ✅ (implícito) |
| Velocidad | ⚡ GPU 0.88s | 🐢 CPU 2.4s | 🐌 GPU 60-500s |
| CPU-only | ❌ (GPU default) | ✅ | ❌ |
| Multimagen/batch | ❌ | ❌ | ✅ (`infer_multi`) |

**Conclusión de la investigación interna**: no hay un motor "mejor" — hay 3
**fortalezas ortogonales**: velocidad (EasyOCR), detección/detección-estilizada +
rotación (RapidOCR), y comprensión semántica del diálogo en arte (U-OCR). El
código fuente confirma que cada uno es **componible a nivel de sub-módulos**
(detector/recognizer separables en EasyOCR y RapidOCR; prompt + crops configurables
en el VLM).

---

## 3. Plan de integración: "un solo motor" por dentro

### 3.1 Arquitectura (ya esbozada en INVESTIGACION_3_OCR.md §8)

```
                    ┌─────────────────────────────────────┐
                    │        OCRManager (ocr_engine.py)   │  ← YA IMPLEMENTADO
                    │  orquesta: trigger v4.2 + tiers     │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────┬───────────┼──────────────┬────────────────┐
        ▼               ▼           ▼              ▼                ▼
   EasyOCR GPU      RapidOCR CPU  U-OCR daemon   Ruta C (globos)  Merge+post
   (CRAFT+CRNN)     (DBNet+Rec)   (VLM 3B 4bit)  (blobs OpenCV)   (_fusionar)
        │               │           │              │                │
        └─── ya fusiona ┴─── 3 vías ┴──┬───────────┴────────────────┘
```

### 3.2 Qué hacer funcionar juntos (por prioridad)

**✅ YA FUNCIONA (implementado en sesiones previas):**
1. Híbrido EasyOCR+RapidOCR en `_detect_and_ocr` (siempre, `use_hybrid=True`).
2. Modo fusion: trigger v4.2 + Ruta C + `_fusionar_blocks_multi`.
3. OCRManager (ocr_engine.py) que centraliza todo.
4. Serialización GPU (`_gpu_lock` + `_uocr_inferring` + degradación RapidOCR CPU).
5. §8.4.4: Ruta C degrada a RapidOCR CPU cuando el daemon infiere (recién hecho).

**🚀 PRÓXIMO (aprovechando el código fuente interno):**

| # | Mejora | Código fuente que la habilita | Impacto |
|---|---|---|---|
| 1 | **Batch de páginas al daemon con `infer_multi()`** | `modeling_unlimitedocr.py:~1150` — multi-imagen, sin crop mode | Amortiza prefill: N páginas en 1 llamada → ~1.5-2× más rápido |
| 2 | **Usar el Cls de RapidOCR para globos rotados** | `rapidocr_onnxruntime/main.py` — `TextClassifier` separado | EasyOCR no detecta texto girado; añade robustez |
| 3 | **Reutilizar `group_text_box()` de EasyOCR como merge alternativo** | `craft_utils.py` | Mejor agrupación de líneas que el merge propio en textos inclinados |
| 4 | **Ajustar RapidOCR por página vía kwargs** (`box_thresh`, `unclip_ratio`) | `main.py:66` `__call__(**kwargs)` | Refuerzo barato en páginas difíciles SIN recargar modelos |
| 5 | **Usar los tipos semánticos del VLM en la fusión** | `re_match()` — type ∈ {text,title,header,image,...} | Dar peso extra a `title`/`header` en la votación |
| 6 | **Prompt VLM configurable** (JSON estructurado) | `format_messages()` + `infer(prompt=...)` | Salida parseable sin regex de stdout |

### 3.3 Plan de implementación por fases

**Fase 1 — Batch multi-página al daemon (mayor impacto de tiempo)**
1. `uocr_daemon.py`: nuevo endpoint `POST /ocr-batch` que acepta `images: [paths]`
   y usa `_model.infer_multi(tokenizer, prompt, image_files, ...)`.
2. `uocr_client.py`: `process_batch(images)` con `wait_timeout`.
3. `OCRManager._reforzar_con_unlimited`: acumular páginas trigger y enviarlas en
   un solo batch al final del lote (worker del capítulo).
4. Mapear bloques por índice de imagen (infer_multi devuelve por imagen? verificar).

**Fase 2 — Reforzar RapidOCR con parámetros adaptativos**
1. En `_run_rapidocr`: aceptar `box_thresh`/`unclip_ratio` opcionales.
2. En el trigger: si confianza media baja, reintentar RapidOCR con
   `box_thresh=0.3, unclip_ratio=1.8` (más agresivo) antes de llamar al VLM.
3. Benchmark: páginas donde RapidOCR con params agresivos iguala al VLM → ahorra
   60-500s por página.

**Fase 3 — Tipos semánticos + rotación en la fusión**
1. Propagar `type` del VLM a los bloques (`_ocr_with_unlimited` ya lo usa para
   filtrar ruido; añadirlo al bloque como `block_type`).
2. En `_fusionar_blocks_multi`: votación ponderada por tipo (title/header pesan más).
3. En Ruta C: antes de EasyOCR, pasar cada globo por `TextClassifier` de RapidOCR
   para detectar rotación → rotar el crop → OCR.

**Fase 4 — Unificar el flujo en OCRManager (default fusion)**
1. `ocr_mode` default → `fusion` (endpoint + process_all_pages + UI).
2. UI: selector "Automático" en vez de exponer los 3 modos.
3. Badge de estado del daemon en el frontend.

**Fase 5 — Validación final**
1. Capítulo completo (53 págs) en fusion con batch → medir vs 44 min estimado.
2. CER en págs. artísticas (3/5/11/12) vs ground truth.
3. 314+ tests en verde + code review.

---

## 4. Riesgos y mitigaciones (del código fuente)

| Riesgo | Origen | Mitigación |
|---|---|---|
| `infer_multi` sin crop mode pierde detalle de texto pequeño | `modeling_unlimitedocr.py` — no soporta crop | Usar batch solo para páginas ya seleccionadas por el trigger; mantener crop mode para páginas individuales artísticas |
| Batch de N páginas = N× tokens → OOM en 4GB VRAM | 3B 4-bit + seq larga | `max_length` reducido por imagen; batch ≤ 4 páginas; medir VRAM |
| RapidOCR Cls añade ~0.1-0.3s por globo | CPU | Solo aplicar en globos con texto no detectado (Ruta C) |
| `group_text_box` cambia el merge actual | craft_utils | A/B test vs `_group_and_merge_blocks`; mantener el actual como default |
| El VLM manda texto a stdout (necesario para coords) | streamer | Ya capturado con redirect_stdout; `infer_multi` usa el mismo streamer |
| Race mixed-format en Ruta C | `_run_ocr_on_image` degrada internamente | YA CORREGIDO (isinstance dict, sesión actual) |

---

## 5. Referencias de código fuente exactas

| Motor | Archivo | Línea | Nota |
|---|---|---|---|
| EasyOCR | `easyocr/easyocr.py` | 30, 311, 353, 440 | Reader, detect, recognize, readtext |
| EasyOCR | `easyocr/craft.py` | 30 | CRAFT detector |
| EasyOCR | `easyocr/detection.py` | 24, 74, 92 | test_net, get_detector, get_textbox |
| EasyOCR | `easyocr/recognition.py` | 153, 186 | get_recognizer (vgg_model), get_text |
| EasyOCR | `easyocr/craft_utils.py` | — | group_text_box (merge de líneas) |
| RapidOCR | `rapidocr_onnxruntime/main.py` | 33, 66 | RapidOCR class, __call__ |
| RapidOCR | `rapidocr_onnxruntime/config.yaml` | 1-60 | det/cls/rec config + umbrales |
| U-OCR | `modeling_unlimitedocr.py` | 175, 787, 44, 386 | dynamic_preprocess, infer, re_match, TPSTextStreamer |
| U-OCR | `modeling_unlimitedocr.py` | ~1150 | **infer_multi (batch)** |
| U-OCR | `deepencoder.py` | 243 | CLIP vision encoder |
| U-OCR | `config.json` | 30-118 | DeepSeek-V2-Lite config |

---

## 6. Estado actual

- **§8.4.4 implementado y validado**: Ruta C degrada a RapidOCR CPU cuando el daemon
  infiere + fix del revisor para formato mixto (5/5 tests de Ruta C, 314 totales).
- **OCRManager activo** (ocr_engine.py) con trigger v4.2.
- Run del capítulo en curso: 20-27/53 páginas, 0 errores.
- Próximo paso sugerido: **Fase 1 (batch multi-página con infer_multi)** — el mayor
  ahorro de tiempo disponible según el código fuente.
