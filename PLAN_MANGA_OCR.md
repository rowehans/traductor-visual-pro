# PLAN — manga_ocr: Extracción de texto de manga (compatible con Traductor Visual Pro)

**Estado: Paso 1 (planificación + entorno) — 2026-08-11**
Objetivo final: un CLI `manga_ocr.py` que lea `input_manga/`, extraiga el texto de cada página (incluyendo globos sin borde, texto flotante sobre el dibujo, pensamientos y tipografías estilizadas) y lo guarde ordenado en `output_texto/`. **Reutiliza la maquinaria del proyecto** (OCRManager, config, ocr_utils, YOLO, fusión multi-motor, VLM opcional) en vez de construir algo aislado. Solo extracción: sin traducción ni inpainting.

---

## 1. Estado del entorno (verificado hoy)

CUDA funciona en la GTX 1050 Ti y **todas las librerías ya están instaladas** en `env/` — no hace falta instalar nada.

| Componente | Versión | Rol | Estado |
|---|---|---|---|
| `torch` | 2.6.0+cu124 | Backend EasyOCR (GPU) | ✅ `cuda-avail=True`, GTX 1050 Ti detectada |
| `easyocr` | 1.7.2 | OCR principal (GPU, ~0.88 s/pág validado) | ✅ |
| `rapidocr-onnxruntime` | 1.4.4 | OCR híbrido CPU (complementa estilizados) | ✅ |
| `onnxruntime` | 1.28.0 | Motor del detector de texto de cómic (CPU) | ✅ |
| `ultralytics` | 8.4.115 | YOLO ogkalu (globos de diálogo) | ✅ |
| `pymupdf` | 1.28.0 | Render PDF → PNG (patrón ya usado en process_all_pages) | ✅ |
| `opencv-python` | 4.11.0.86 | Pre/post-proceso de imagen | ✅ |
| `huggingface_hub` | 0.25.2 | Descarga de modelos | ✅ |

**Modelos:**
- `models/comic-speech-bubble-detector.pt` (52 MB — ogkalu YOLOv8m, 8K+ imágenes manga/webtoon/manhua: globos de diálogo) ✅ ya existía
- `models/comic-text-detector.onnx` (94.7 MB — dmMaze comic-text-detector, port ONNX de mayocream) ✅ **descargado hoy**
  - Firma verificada: `IN images[1,3,1024,1024] float` → `OUT blk[1,64512,7]` (cajas), `seg[1,1,1024,1024]` (máscara), `det[1,2,1024,1024]` (mapas de prob. texto/línea)
  - Corre 100% en CPU con onnxruntime → **0 VRAM extra**

---

## 2. Arquitectura

```
input_manga/  (PDF | PNG | JPG | WEBP)
     │
     ▼
manga_ocr.py  ── NUEVO entry point CLI
     │   por archivo: PDF → render fitz (patrón render_worker de process_all_pages.py)
     │                imagen → directa
     ▼
OCRManager.run_ocr()  ── REUTILIZADO (ocr_engine.py), batch=1 estricto
     │
     ├─ [T1] EasyOCR GPU ......... texto normal en globos
     ├─ [T2] RapidOCR CPU ........ complementa fuentes estilizadas
     ├─ [T3.5] YOLO ogkalu ....... globos con/sin borde, cartelas (Fase 6 existente)
     ├─ [T3.6] NUEVO: comic-text-detector ONNX (CPU) ── texto SIN globo:
     │         flotante sobre el dibujo, pensamientos, títulos, tipografía de arte
     │         → regiones alimentan Ruta C (re-OCR EasyOCR con upscale), igual que YOLO
     ├─ [_fusionar_blocks_multi] votación ponderada + 9 filtros anti-ruido (existente)
     └─ [opcional] VLM daemon 5177 (solo con --ocr-mode fusion+vlm, trigger v4.2)
     ▼
output_texto/<archivo>.json   ── NUEVO schema de extracción
output_texto/<archivo>.txt    ── texto plano legible
```

**Nuevo código** (siguiendo convenciones del proyecto):
- `manga_ocr.py` — CLI: escaneo de `input_manga/`, procesado, escritura de JSON/TXT, checkpoint/resume si aplica.
- `ocr_utils.py::_detect_text_regions_comic_detector()` — tier de detección con lazy-load thread-safe (mismo patrón que `_get_yolo_engine`), degradación a `[]` sin romper el pipeline.
- `config.py` — flags nuevas: `COMIC_DETECTOR_ENABLED`, `COMIC_DETECTOR_MODEL_PATH`, `COMIC_DETECTOR_CONF_THRESH`, `COMIC_DETECTOR_MAX_REGIONS`.
- `ocr_engine.py` — hook de las regiones CTD en la Fase 6/Ruta C (merge con regiones YOLO antes del re-OCR).
- `tests/` — tests unitarios (mock de onnxruntime) + integración con imagen sintética.

**Schema JSON de salida (diseño):**
```json
{
  "archivo": "capitulo_47.pdf",
  "generado": "2026-08-11T18:00:00",
  "ocr_mode": "fusion",
  "detectores": ["easyocr", "rapid", "yolo_globos", "comic_text_detector"],
  "paginas": [
    {
      "n": 1,
      "bloques": [
        {"texto": "¡Ya casi llego!", "bbox": [x0, y0, x1, y1], "conf": 0.92,
         "motores": ["easyocr", "rapid"], "detector": "yolo|ctd|hibrido|directo"}
      ],
      "texto_plano": "¡Ya casi llego!\n"
    }
  ]
}
```

---

## 3. Estrategia VRAM (GTX 1050 Ti, 4 GB — batch=1 estricto)

Regla de oro: **una sola imagen en el pipeline a la vez y nunca dos modelos grandes en GPU simultáneamente.**

| Consumidor | VRAM | Dónde corre |
|---|---|---|
| EasyOCR (torch) | ~1.6–2.0 GB | GPU |
| comic-text-detector | **0** | CPU (onnxruntime CPU) — por diseño: modelos pequeños pierden contra CPU por overhead PCIe (ya demostrado con RapidOCR en este proyecto) |
| YOLO ogkalu | ~0.3–0.5 GB | device resuelto UNA vez (política determinista sesión 116) |
| VLM daemon | ~2.25 GB | solo si se pide `fusion+vlm` (apagado por defecto en extracción) |

- Total extracción pura sin VLM: **< 2.5 GB de 4 GB** — margen cómodo.
- Entre páginas: `del` + `gc.collect()` + `torch.cuda.empty_cache()` (cuando corresponda).

---

## 4. Instalación (comandos — ya ejecutados en su mayoría)

```bash
cd "D:/crear traductor"

# 1) Verificar CUDA (debe imprimir: True NVIDIA GeForce GTX 1050 Ti)  ✅ VERIFICADO
env/Scripts/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2) Descargar el detector de texto de cómic (~95 MB)  ✅ EJECUTADO → models/comic-text-detector.onnx
env/Scripts/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('mayocream/comic-text-detector-onnx', 'comic-text-detector.onnx', local_dir='models')"

# 3) Crear carpetas  ✅ EJECUTADO
mkdir input_manga output_texto
```

**Re-creación del entorno desde cero (solo si algún día se borra `env/`):**
```bash
env/Scripts/python -m venv env
env/Scripts/pip install -r requirements.txt   # ya incluye torch 2.6.0+cu124 → CUDA 12.4 soporta Pascal (GTX 1050 Ti)
```

---

## 5. Plan de pasos (siguientes)

1. ✅ **Paso 1** — Planificación + entorno: carpetas creadas, modelo descargado, CUDA verificado.
2. ✅ **Paso 2** — Tier CTD en `ocr_utils.py` + flags en `config.py` + 10 tests unitarios con onnxruntime mockeado. **Validado en vivo** (sesión 143): pág 5 del capítulo 43 → **1.36s CPU (incl. carga), 12 regiones (3 blk + 9 líneas), conf 0.81-0.94, 0 VRAM**; suite canónica 490 passed.
3. ✅ **Paso 3** — `manga_ocr.py` CLI (sesión 144): escaneo `input_manga/`, render fitz (`--zoom`), `run_ocr()` fusion por página (batch=1), JSON+TXT incrementales con el schema del plan, `doc_id` escopeado. Nuevo gate `UOCR_ENABLED` (solo VLM; YOLO/Ruta C siguen activos) — default OFF en el CLI (--vlm para activarlo). 13 tests + 1 gate; **504 passed**; smoke: 3 págs en 33.8s, 0 llamadas VLM, p1 con `yolo+rutac`.
4. ✅ **Paso 4** — Integración de regiones CTD en `ocr_engine.py` (sesión 146): **`_ruta_c_ctd`** (Fase 6.5) con las lecciones del Paso 5 incorporadas — gate en CASCADA post-YOLO (`GATE_MIN_BLOCKS`=3, `GATE_MAX_CONF`=0.35; si YOLO ya resolvió la página, CTD no corre), **dedup 1** de regiones CTD vs `yolo_regiones` por `_overlap_ratio` > 0.40 (una zona que YOLO ya va a re-OCRear no se paga dos veces) y **dedup 2** vs bloques existentes (>0.5); integrado en `_run_fusion` (single) y Fase A del batch ANTES del trigger v4.2 (los bloques recuperados pueden evitar el VLM); `engines_used.append("ctd+rutac")`; `disable_uocr` lo apaga (`_ctd_disabled`). 9 tests; **513 passed**; smoke en vivo págs 5-8: p8 débil → YOLO 0 recuperados, CTD 17 regiones → 2 bloques nuevos → `ctd+rutac`. **Validado de punta a punta con la lección del benchmark: el gate evita el re-OCR duplicado en páginas bien detectadas (págs 5-7 no lo corren) y cobra el +17% en las débiles (pág 8).**
5. ✅ **Paso 5** — Benchmark real (sesión 145): 5 páginas del capítulo 43 → **CTD añade 0 MiB de VRAM** (pico 3863/4096 MiB con daemon cargado), detección CPU 0.75-0.89s/pág, re-OCR 2.0-4.0s/pág; detección **29 → 34 bloques (+17.2%)**, 85 regiones → 21 recuperadas → 5 nuevas tras merge. `tools/benchmark_extraccion_ctd.py` reutilizable.
6. ✅ **Paso 6** — Docs y hygiene: AGENTS.md (sesiones por paso), `.gitignore` con `input_manga/`, `output_texto/` y `train_data/` (sesión 147); sin dependencias nuevas (onnxruntime/ultralytics ya estaban).
7. ✅ **Paso 7 — DESTILACIÓN VLM→YOLO + loop de corrección** (sesión 147): el daemon VLM (teacher) etiqueta páginas reales (`tools/etiquetar_con_vlm.py`, `--append` por capítulo), el YOLO ogkalu se fine-tunea como student (`tools/entrenar_detector.py`: freeze/lr bajo anti-olvido, nunca pisa el original, A/B integrado, ~2 GB VRAM), y el usuario CORRIGE las pseudo-etiquetas en X-AnyLabeling (`tools/exportar_anotaciones.py` → workspace YOLO listo) para re-entrenar con un comando (`tools/fusionar_correcciones.py --train`). **Primer ciclo**: 37 págs / 121 etiquetas del capítulo 43 → fine-tune detecta más regiones (34 vs 27) pero recall IoU 56% vs 62.5% de ogkalu — aún no supera al pretrained (datos escasos y clase free 6/121); **el modelo fine-tuned NO se activa hasta que el loop de corrección acumule oro**. 18 tests; suite 531 passed.

---

## 6. Riesgos y mitigaciones

- **Firma ONNX fija 1024×1024** → letterbox obligatorio; manga es vertical estricto → padding lateral; escalar cajas al coord original con el factor correcto.
- **`blk` con 64512 filas** → post-proceso con NMS requerido (referencia: `box_utils` de dmMaze); coste CPU medido en Paso 5.
- **Solape YOLO vs CTD** → merge de regiones antes de Ruta C para no re-OCR lo mismo dos veces.
- **Sin internet en runtime** → modelo ya local en `models/` ✓.
- **VRAM** → batch=1 estricto + CTD en CPU; monitorizado con nvidia-smi en el benchmark.
