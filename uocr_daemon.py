"""
uocr_daemon.py — Daemon persistente de Unlimited-OCR (GPU 4-bit NF4).

Carga el modelo 4-bit UNA sola vez al arrancar (≈494s en esta máquina) y queda
escuchando en 127.0.0.1:5177 para servir OCR sin recargar por página.
El servidor Flask (server.py) lo lanza en background vía uocr_client.py.

Endpoints (JSON):
  GET  /health -> {"state": "loading"|"ready"|"error",
                   "load_s": float, "vram_gb": float, "error": str|null}
  POST /ocr    -> body: {"image_path": "...", "max_length": 1280}
                  resp: {"text": "...", "infer_s": float, "vram_gb": float,
                         "blocks": [{"type","x","y","w","h","text"}],
                         "recovered_from_art": int}
  POST /ocr-batch -> body: {"images": ["...", ...] (1-4), "max_length": 1280}
                  resp: {"pages": [{"blocks": [...], "recovered_from_art": int}, ...],
                         "infer_s": float, "vram_gb": float, "n_images": int}
                  Usa _model.infer_multi(): N páginas en UNA inferencia VLM
                  (prefill compartido) — Fase 1 del plan de unificación.
                  Las páginas se separan con <PAGE> y cada una se mapea de
                  vuelta al espacio de píxeles de su imagen original.

Post-procesado:
  Si el modelo devuelve un bloque type="image" que ocupa >30% del área de la
  página (caso de páginas con diálogo pintado sobre arte, pág. 3/12 del manga),
  se recorta la región, se escala a 640x640 (letterbox blanco) y se re-envía al
  modelo con crop_mode=False para intentar extraer el diálogo incrustado en el
  arte. Los bloques recuperados se mapean de vuelta al espacio de píxeles de la
  página original y se añaden a la respuesta.

Uso:
  env_uocr_gpu\\Scripts\\python.exe uocr_daemon.py [--port 5177]
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

sys.stdout.reconfigure(  # type: ignore[union-attr]
    encoding="utf-8", line_buffering=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
# Regla "no tocar C:": toda la caché HF en D:
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(ROOT, "hf_cache", "hub"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT, "hf_cache", "hub"))

MODEL_DIR = os.path.join(ROOT, "models_unlimited_patched")

# ─── Estado global ───────────────────────────────────────────────
_model: Any = None
_tokenizer: Any = None
_status: dict[str, Any] = {"state": "loading", "load_s": 0.0, "vram_gb": 0.0, "error": None}
_infer_lock = threading.Lock()  # una inferencia a la vez (GPU única)

# Formato de las líneas de detección que emite el modelo por stdout:
#   <|det|>header [40, 17, 149, 28]<|/det|>13/7/26, 4:58 p.m.
# NOTA: result.md guarda el texto LIMPIO (el post-procesado del modelo
# elimina los tags <|det|>...<|/det|>). Las coordenadas solo se pueden
# recuperar capturando el stream de stdout durante la generación.
# El separador tras <|/det|> usa [ \t]* (no \s*) para NO cruzar saltos de
# línea: los bloques sin texto (image) no deben absorber la línea siguiente.
_DET_RE = re.compile(
    r"<\|det\|>(\w+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*<\|/det\|>[ \t]*([^\n]*)",
    re.MULTILINE,
)


def _parse_blocks(text: str) -> list[dict[str, Any]]:
    blocks = []
    for m in _DET_RE.finditer(text):
        typ, x, y, w, h = (
            m.group(1), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
        text_content = m.group(6).strip()
        blocks.append({"type": typ, "x": x, "y": y, "w": w, "h": h, "text": text_content})
    return blocks


def _load_model() -> None:
    """Carga el modelo 4-bit en background (no bloquea /health)."""
    global _model, _tokenizer
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # MODEL_DIR es un directorio local del bundle; no se descarga desde la
        # petición HTTP ni desde una ruta controlada por el usuario.
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)  # nosec B615
        _model = AutoModel.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,  # nosec B615
            use_safetensors=True,
            torch_dtype=torch.float16,
            quantization_config=quant,
            device_map={"": 0},  # forzar a la única GPU (GTX 1050 Ti 4GB)
        ).eval()
        torch.cuda.reset_peak_memory_stats()
        _status.update(
            state="ready",
            load_s=round(time.time() - t0, 1),
            vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        )
        print(f"[uocr-daemon] MODELO LISTO en {_status['load_s']:.1f}s | "
              f"VRAM {_status['vram_gb']:.2f} GB", flush=True)
    except Exception as e:  # nosec — el daemon reporta el error por /health
        traceback.print_exc()
        _status.update(state="error", load_s=round(time.time() - t0, 1), error=str(e))
        print(f"[uocr-daemon] ERROR cargando modelo: {e}", flush=True)


_MAX_REQ_DIRS = 20  # máx directorios de salida por request conservados en disco
_MAX_JSON_BODY_BYTES = 64 * 1024  # solo contiene rutas y parámetros pequeños


def _is_allowed_input_path(value: object) -> bool:
    """Valida rutas de entrada del daemon contra raíces locales conocidas.

    El servidor Flask usa archivos temporales para enviar páginas y el propio
    proyecto puede contener imágenes de prueba. Resolver la ruta real antes
    de comparar bloquea escapes mediante ``..`` o symlinks y evita que el
    endpoint loopback procese archivos arbitrarios de otras carpetas.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        candidate = os.path.normcase(os.path.realpath(os.path.abspath(value)))
        if not os.path.isfile(candidate):
            return False
        roots = (ROOT, tempfile.gettempdir())
        for root in roots:
            root_real = os.path.normcase(os.path.realpath(os.path.abspath(root)))
            try:
                if os.path.commonpath((candidate, root_real)) == root_real:
                    return True
            except ValueError:
                # En Windows, D:\proyecto y C:\Temp no comparten unidad;
                # una raíz incompatible no invalida las siguientes.
                continue
    except (OSError, ValueError, TypeError):
        return False
    return False


def _cleanup_old_out_dirs() -> None:
    """Borra los directorios req_* más antiguos (cada request genera ~200KB)."""
    try:
        base = os.path.join(ROOT, "uocr_daemon_out")
        if not os.path.isdir(base):
            return
        dirs = sorted(
            (d for d in os.listdir(base) if d.startswith(("req_", "art_"))),
            key=lambda d: os.path.getmtime(os.path.join(base, d)),
        )
        for d in dirs[:-_MAX_REQ_DIRS]:
            import shutil
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    except Exception:
        pass


# Prompt y anti-repetición del VLM (plan §10.2 item 2, 2026-08-16):
# configurables por request (/ocr y /ocr-batch aceptan "prompt" y "ngram"
# opcionales) para poder A/B el prompt y el no_repeat_ngram SIN reiniciar el
# daemon (el modelo tarda ~2 min en recargar).
#
# A/B de pág 21 (2026-08-16, benchmark_vlm_maxlen.py --pages 21, max_length
# 1280): ngram 35 → 210.6 s VLM / 8 bloques; ngram 15 → 194.4 s / 8 bloques
# con TEXTOS IDÉNTICOS (−16 s, −7.7 %; el ngram solo bloquea repeticiones
# exactas, no puede perder recuperación). Prompt instructivo de diálogo
# ('<image>Extract all dialogue text...') → 32.9 s pero SOLO 2/8 bloques
# (−84 % tiempo, −6 bloques — el prompt degrada el formato de detección del
# modelo): DESCARTADO. Default de ngram aplicado: 15.
_PROMPT_DEFAULT = "<image>document parsing."
_NGRAM_SIZE_DEFAULT = 15
_NGRAM_WINDOW_DEFAULT = 128
# Plan §10.2 item 5: image_size del pase principal (prefill). El A/B
# 640 vs 512 mide si el prefill (~23 s/llamada, casi todo el coste de la
# llamada principal de 16 tokens) se puede recortar sin perder recuperación.
_IMAGE_SIZE_DEFAULT = 640


def _infer_once(image_path: str, out_dir: str, max_length: int,
                crop_mode: bool = True, image_size: int = 640,
                prompt: str | None = None,
                ngram_size: int | None = None,
                ngram_window: int | None = None) -> tuple[str, list[dict[str, Any]], float]:
    """Ejecuta UNA inferencia del modelo y devuelve (texto_md, bloques, infer_s).

    Captura stdout (el modelo emite <|det|>...<|/det|> con coordenadas por
    print()) y lee result.md (texto limpio). Reutilizada por _run_ocr y por
    el re-OCR de bloques image (crop_mode=False).

    prompt/ngram_size/ngram_window: si se pasan, sobreescriben los defaults
    del módulo (_PROMPT_DEFAULT/_NGRAM_SIZE_DEFAULT/_NGRAM_WINDOW_DEFAULT) —
    el A/B los pasa por request sin tocar el proceso del daemon.
    """
    import contextlib
    import io

    t0 = time.time()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        _model.infer(
            _tokenizer,
            prompt=prompt or _PROMPT_DEFAULT,
            image_file=image_path,
            output_path=out_dir,
            base_size=1024, image_size=image_size, crop_mode=crop_mode,
            max_length=max_length,
            no_repeat_ngram_size=(
                _NGRAM_SIZE_DEFAULT if ngram_size is None else ngram_size),
            ngram_window=(
                _NGRAM_WINDOW_DEFAULT if ngram_window is None else ngram_window),
            save_results=True,
        )
    infer_s = time.time() - t0
    text = ""
    md_path = os.path.join(out_dir, "result.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            text = f.read()
    blocks = _parse_blocks(stream.getvalue())
    return text, blocks, infer_s


# Fracción mínima del área de página para considerar un bloque type="image"
# como "panel con diálogo incrustado en arte" y disparar el re-OCR.
_ART_RECOVER_MIN_AREA_RATIO = 0.30
_ART_RECOVER_CANVAS = 640  # el modelo con crop_mode=False procesa a 640x640


def _recover_art_dialogue(image_path: str, blocks: list[dict[str, Any]],
                          max_length: int, prompt: str | None = None,
                          ngram_size: int | None = None,
                          ngram_window: int | None = None) -> tuple[list[dict[str, Any]], int]:
    """Re-OCR de bloques type="image" grandes para recuperar diálogo en arte.

    Para cada bloque image con área >30% de la página:
      1. Recortar la región (con margen) de la imagen original.
      2. Escalar preservando aspecto y letterbox blanco a 640x640.
      3. Re-enviar al modelo con crop_mode=False (una sola vista global).
      4. Mapear los bloques recuperados de vuelta al espacio de píxeles de la
         página original (compensando recorte + escala + offset de letterbox).

    Retorna (bloques_finales, n_recuperados). No recurre: solo un nivel.

    Usa PIL (no OpenCV): env_uocr_gpu no tiene cv2 instalado y PIL es la única
    dependencia de imagen garantizada (la usa el propio modelo).
    """
    import numpy as np
    from PIL import Image

    try:
        page = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[uocr-daemon] No se pudo leer {image_path}: {e}", flush=True)
        return blocks, 0
    pw, ph = page.size
    page_area = pw * ph
    big = [b for b in blocks
           if b.get("type") == "image"
           and (b.get("w", 0) * b.get("h", 0)) > _ART_RECOVER_MIN_AREA_RATIO * page_area]
    if not big:
        return blocks, 0

    recovered: list[dict[str, Any]] = []
    pad = 8
    for bi, b in enumerate(big):
        # ── 1. Recorte (clamp a la página) ────────────────────────
        x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
        x1, y1 = min(pw, b["x"] + b["w"] + pad), min(ph, b["y"] + b["h"] + pad)
        cw, ch = x1 - x0, y1 - y0
        if cw < 32 or ch < 32:
            continue
        crop = page.crop((x0, y0, x1, y1))

        # ── 2. Escala + letterbox blanco a 640x640 ────────────────
        s = _ART_RECOVER_CANVAS / max(cw, ch)  # s>1 → upscale, s<1 → downscale
        nw = max(1, int(round(cw * s)))
        nh = max(1, int(round(ch * s)))
        resized = crop.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (_ART_RECOVER_CANVAS, _ART_RECOVER_CANVAS), (255, 255, 255))
        ox, oy = (_ART_RECOVER_CANVAS - nw) // 2, (_ART_RECOVER_CANVAS - nh) // 2
        canvas.paste(resized, (ox, oy))

        tmp_dir = os.path.join(ROOT, "uocr_daemon_out",
                               f"art_{int(time.time() * 1000)}_{bi}")
        os.makedirs(tmp_dir, exist_ok=True)
        crop_path = os.path.join(tmp_dir, "crop_640.png")
        canvas.save(crop_path)

        # ── 3. Re-inferencia con crop_mode=False ──────────────────
        try:
            _, sub_blocks, sub_s = _infer_once(
                crop_path, tmp_dir, max_length,
                crop_mode=False, image_size=_ART_RECOVER_CANVAS,
                prompt=prompt, ngram_size=ngram_size,
                ngram_window=ngram_window,
            )
        except Exception as e:  # nosec — un panel fallido no debe tumbar la página
            print(f"[uocr-daemon] re-OCR panel image {bi} falló: {e}", flush=True)
            continue

        # ── 4. Mapear de vuelta al espacio de la página ───────────
        for sb in sub_blocks:
            if not sb.get("text") or sb["type"] == "image":
                continue
            px = x0 + (sb["x"] - ox) / s
            py = y0 + (sb["y"] - oy) / s
            rw = sb["w"] / s
            rh = sb["h"] / s
            if rw < 4 or rh < 4:
                continue
            recovered.append({
                "type": sb["type"],
                "x": int(round(px)), "y": int(round(py)),
                "w": int(round(rw)), "h": int(round(rh)),
                "text": sb["text"],
                "from_art_recrop": True,
            })
        print(f"[uocr-daemon] re-OCR panel image {bi}: {len(sub_blocks)} sub-bloques "
              f"({sub_s:.1f}s)", flush=True)

    if recovered:
        blocks = blocks + recovered
    return blocks, len(recovered)


def _parse_blocks_multi(stream_text: str, n_images: int) -> list[list[dict[str, Any]]]:
    """Divide el stream de stdout de infer_multi() en páginas y parsea cada una.

    infer_multi() genera TODAS las imágenes en una sola pasada y separa las
    páginas con el marcador literal '<PAGE>'. El stream de stdout (capturado
    con redirect_stdout) contiene:
        <PAGE>
        <|det|>...                          ← página 0
        <PAGE>
        <|det|>...                          ← página 1
        ...

    El modelo emite '<PAGE>' ANTES de cada página (semántica del save_results
    oficial: outputs.split('<PAGE>')[1:]). El split produce n_images+1
    secciones; la primera (antes del primer marcador) es ruido de prompt y se
    descarta. IMPORTANTE: las secciones vacías NO se filtran — una página sin
    texto emite '<PAGE>\n<PAGE>' consecutivos y filtrarla desalinearía todas
    las páginas posteriores (bug corregido).

    Args:
        stream_text: stdout completo capturado de la llamada a infer_multi().
        n_images: Número de imágenes enviadas (debe coincidir con las páginas).

    Returns:
        list[list[dict]]: una lista de bloques por imagen (len == n_images;
        se rellena con [] si el modelo generó menos páginas de las esperadas).
    """
    sections = stream_text.split("<PAGE>")
    # Defensa N=1 / modelo sin marcador inicial: si la sección pre-primer-
    # marcador contiene tags de detección, es contenido real, no ruido.
    if len(sections) > 1 and "<|det|>" in sections[0]:
        # El stream NO empezó con <PAGE>: el primer bloque pertenece a la
        # página 0 y los marcadores separan las siguientes.
        return [_parse_blocks(s) for s in sections[:n_images]] + [
            [] for _ in range(max(0, n_images - len(sections)))
        ]
    # Semántica oficial: secciones[1:] (la primera es ruido). Sin filtro de
    # vacíos para no desalinear páginas sin texto.
    usable = sections[1:]
    results: list[list[dict[str, Any]]] = []
    for sec in usable[:n_images]:
        results.append(_parse_blocks(sec))
    while len(results) < n_images:
        results.append([])
    return results


def _map_multi_blocks_to_page(blocks: list[dict[str, Any]], page_w: int,
                              page_h: int, canvas: int = 640) -> list[dict[str, Any]]:
    """Mapea bloques del espacio 640x640 (infer_multi) al espacio de página.

    infer_multi() redimensiona cada imagen a canvas×canvas (resize directo a
    640x640, sin crop), así que el modelo reporta coordenadas en ese espacio.
    El mapeo lineal px = x/canvas * page_w, py = y/canvas * page_h devuelve
    los bloques al espacio de píxeles de la página original.
    """
    mapped: list[dict[str, Any]] = []
    for b in blocks:
        mapped.append({
            "type": b.get("type", "text"),
            "x": int(round(b.get("x", 0) / canvas * page_w)),
            "y": int(round(b.get("y", 0) / canvas * page_h)),
            "w": max(1, int(round(b.get("w", 0) / canvas * page_w))),
            "h": max(1, int(round(b.get("h", 0) / canvas * page_h))),
            "text": b.get("text", ""),
        })
    return mapped


_MAX_BATCH_IMAGES = 4  # límite VRAM: 4 páginas 640x640 prefill simultáneo en GTX 4GB


def _run_ocr_batch(image_paths: list[str], max_length: int,
                    prompt: str | None = None, ngram_size: int | None = None,
                    ngram_window: int | None = None,
                    image_size: int | None = None) -> dict[str, Any]:
    """OCR de VARIAS páginas en una sola inferencia VLM (infer_multi).

    Fase 1 del plan: amortiza el prefill del modelo (el costo por página cae
    porque las N imágenes comparten una sola pasada forward + generación
    continua). Devuelve una lista de resultados por página, cada uno con sus
    bloques ya mapeados al espacio de la página original.

    Nota: el re-OCR de diálogo artístico (_recover_art_dialogue) se aplica
    POR PÁGINA individual (infer_multi no soporta crop_mode; las páginas
    artísticas del trigger mantienen el pase individual de _run_ocr). Costo
    real: 1 inferencia batch + 1 por página con panel grande (>30%) — el
    timeout del cliente (1800s * n) debe cubrir esa suma.
    """
    import contextlib
    import io
    import torch  # noqa: PLC0415 — import diferido (ya cargado por _load_model)
    from PIL import Image

    if not _infer_lock.acquire(timeout=1800):
        return {"error": "timeout esperando turno de inferencia", "pages": []}
    try:
        out_dir = os.path.join(ROOT, "uocr_daemon_out",
                               f"req_multi_{int(time.time() * 1000)}")
        os.makedirs(out_dir, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            _model.infer_multi(
                _tokenizer,
                prompt=prompt or _PROMPT_DEFAULT,
                image_files=image_paths,
                output_path=out_dir,
                image_size=image_size or _IMAGE_SIZE_DEFAULT,
                max_length=max_length,
                no_repeat_ngram_size=(
                    _NGRAM_SIZE_DEFAULT if ngram_size is None else ngram_size),
                ngram_window=(
                    _NGRAM_WINDOW_DEFAULT if ngram_window is None else ngram_window),
                save_results=True,
            )
        infer_s = time.time() - t0
        pages_blocks = _parse_blocks_multi(stream.getvalue(), len(image_paths))
        result_pages: list[dict[str, Any]] = []
        for i, (path, blocks) in enumerate(zip(image_paths, pages_blocks)):
            try:
                with Image.open(path) as im:
                    pw, ph = im.size
                mapped = _map_multi_blocks_to_page(blocks, pw, ph)
                # Re-OCR de arte por página (calidad): los paneles image
                # grandes se recortan y re-envían individualmente.
                mapped, n_rec = _recover_art_dialogue(
                    path, mapped, max_length,
                    prompt=prompt, ngram_size=ngram_size,
                    ngram_window=ngram_window)
                result_pages.append({"blocks": mapped,
                                     "recovered_from_art": n_rec})
            except Exception as e:  # nosec — una página fallida no tumba el batch
                print(f"[uocr-daemon] batch página {i} falló: {e}", flush=True)
                result_pages.append({"blocks": [], "recovered_from_art": 0,
                                     "error": str(e)})
        vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(f"[uocr-daemon] BATCH {len(image_paths)} págs hecho en {infer_s:.1f}s "
              f"({infer_s / max(len(image_paths), 1):.1f}s/pág) | "
              f"VRAM {vram:.2f} GB", flush=True)
        return {"pages": result_pages, "infer_s": round(infer_s, 2),
                "vram_gb": vram, "n_images": len(image_paths)}
    finally:
        _infer_lock.release()


def _run_ocr(image_path: str, max_length: int,
              prompt: str | None = None, ngram_size: int | None = None,
              ngram_window: int | None = None,
              image_size: int | None = None) -> dict[str, Any]:
    import torch  # noqa: PLC0415 — import diferido (ya cargado por _load_model)
    # Timeout de espera: si otra inferencia cuelga (OOM, etc.), no bloquear
    # la siguiente petición indefinidamente.
    if not _infer_lock.acquire(timeout=1800):
        return {"error": "timeout esperando turno de inferencia", "blocks": [], "infer_s": 0.0}
    try:
        out_dir = os.path.join(ROOT, "uocr_daemon_out", f"req_{int(time.time() * 1000)}")
        os.makedirs(out_dir, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # Pase principal: crop_mode=True (9-grid sobre la página completa)
        # Plan §10.2 item 5: image_size del prefill configurable por request
        # (el A/B 640 vs 512 decide si el prefill se recorta sin perder
        # recuperación).
        text, blocks, _ = _infer_once(
            image_path, out_dir, max_length, crop_mode=True,
            prompt=prompt, ngram_size=ngram_size, ngram_window=ngram_window,
            image_size=image_size or _IMAGE_SIZE_DEFAULT)
        # Post-procesado: recuperar diálogo incrustado en arte
        blocks, n_rec = _recover_art_dialogue(
            image_path, blocks, max_length,
            prompt=prompt, ngram_size=ngram_size, ngram_window=ngram_window)
        infer_s = time.time() - t0
        vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(f"[uocr-daemon] OCR hecho en {infer_s:.1f}s | {len(blocks)} bloques "
              f"({n_rec} recuperados de arte) | VRAM {vram:.2f} GB", flush=True)
        return {"text": text, "infer_s": round(infer_s, 2), "vram_gb": vram,
                "blocks": blocks, "recovered_from_art": n_rec}
    finally:
        _infer_lock.release()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # nosec — silenciar logs
        pass

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, _status)
        else:
            self._send(404, {"error": "not found"})

    def _read_json_body(self) -> dict[str, Any] | None:
        """Lee y parsea el body JSON. Retorna dict o None si es inválido."""
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n < 0 or n > _MAX_JSON_BODY_BYTES:
                return None
            raw = self.rfile.read(n) if n else b"{}"
            if len(raw) != n and n:
                return None
            body = json.loads(raw.decode("utf-8"))
            return body if isinstance(body, dict) else None
        except Exception:
            return None

    def do_POST(self) -> None:
        if self.path not in ("/ocr", "/ocr-batch"):
            self._send(404, {"error": "not found"})
            return
        if _status["state"] != "ready":
            self._send(503, {"error": f"modelo {_status['state']}", "status": _status})
            return
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "json inválido"})
            return
        try:
            from config import UOCR_MAX_LENGTH as _DEF_MAX_LEN
            max_length = int(body.get("max_length", _DEF_MAX_LEN))
        except (TypeError, ValueError):
            self._send(400, {"error": "max_length debe ser un entero"})
            return
        if not (256 <= max_length <= 65536):
            self._send(400, {"error": f"max_length fuera de rango: {max_length}"})
            return
        # Plan §10.2 item 2: prompt y no_repeat_ngram opcionales por request
        # (el A/B los varía sin reiniciar el daemon — el modelo tarda ~2 min
        # en recargar). Solo str/entero válidos; inválidos → 400 (no silenciar
        # el error de configuración).
        prompt = body.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            self._send(400, {"error": "prompt debe ser un string"})
            return
        ngram = body.get("ngram")
        if ngram is not None:
            try:
                ngram = int(ngram)
            except (TypeError, ValueError):
                self._send(400, {"error": "ngram debe ser un entero"})
                return
            if not (1 <= ngram <= 64):
                self._send(400, {"error": f"ngram fuera de rango: {ngram}"})
                return
        # Plan §10.2 item 5: image_size del prefill opcional por request.
        image_size = body.get("image_size")
        if image_size is not None:
            try:
                image_size = int(image_size)
            except (TypeError, ValueError):
                self._send(400, {"error": "image_size debe ser un entero"})
                return
            if not (256 <= image_size <= 1024):
                self._send(400, {"error": f"image_size fuera de rango: "
                                          f"{image_size}"})
                return
        try:
            _cleanup_old_out_dirs()
            if self.path == "/ocr-batch":
                images = body.get("images")
                if (not isinstance(images, list) or not images
                        or len(images) > _MAX_BATCH_IMAGES):
                    self._send(400, {"error": f"images debe ser una lista de "
                                              f"1-{_MAX_BATCH_IMAGES} paths"})
                    return
                bad = [p for p in images if not _is_allowed_input_path(p)]
                if bad:
                    self._send(400, {"error": "paths de imagen no permitidos"})
                    return
                result = _run_ocr_batch(images, max_length, prompt=prompt,
                                        ngram_size=ngram,
                                        image_size=image_size)
                self._send(200, result)
                return
            image_path = body.get("image_path")
            if not _is_allowed_input_path(image_path):
                self._send(400, {"error": "image_path no permitido"})
                return
            assert isinstance(image_path, str)  # _is_allowed_input_path ya lo validó
            result = _run_ocr(image_path, max_length, prompt=prompt,
                              ngram_size=ngram, image_size=image_size)
            self._send(200, result)
        except Exception as e:  # nosec
            traceback.print_exc()
            self._send(500, {"error": str(e)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Daemon Unlimited-OCR (GPU 4-bit)")
    ap.add_argument("--port", type=int, default=5177)
    args = ap.parse_args()

    t = threading.Thread(target=_load_model, daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"[uocr-daemon] Escuchando en 127.0.0.1:{args.port} (modelo cargando en background)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
