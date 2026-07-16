"""
manga_pipeline.py — Wrapper síncrono sobre submódulos de manga-image-translator.
Usa detección CTD + OCR 48px + LaMa inpainting.
NO usa traducción ni renderizado de MIT (se mantiene el sistema existente).
"""

import asyncio
import base64
import os
import sys
import traceback

import cv2
import numpy as np
from PIL import Image

MIT_DIR = os.path.join(os.path.dirname(__file__), "manga-image-translator")
if MIT_DIR not in sys.path:
    sys.path.insert(0, MIT_DIR)

# Importar SOLO los submódulos que necesitamos
# (NO importamos manga_translator.__init__ que carga todos los traductores)
from manga_translator.detection import dispatch as _detect_dispatch, prepare as _detect_prepare
from manga_translator.ocr import dispatch as _ocr_dispatch, prepare as _ocr_prepare
from manga_translator.textline_merge import dispatch as _merge_dispatch
from manga_translator.inpainting import dispatch as _inpaint_dispatch, prepare as _inpaint_prepare
from manga_translator.config import Detector, Ocr, Inpainter, OcrConfig, InpainterConfig, DetectorConfig

_loop = None
_ready = False

def _get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop

async def _init_models():
    """Descarga y prepara modelos (se ejecuta una vez)."""
    print("[MIT] Descargando modelos CTD + DBNet + OCR 48px + LaMa...")
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    await _detect_prepare(Detector.ctd)
    await _detect_prepare(Detector.default)  # Fallback si CTD no detecta
    await _ocr_prepare(Ocr.ocr48px, device)
    await _inpaint_prepare(Inpainter.lama_large, device)
    print("[MIT] Modelos listos.")

def ensure_ready():
    """Asegura que los modelos están descargados."""
    global _ready
    if not _ready:
        loop = _get_loop()
        loop.run_until_complete(_init_models())
        _ready = True

def _text_region_to_block(region) -> dict:
    """Convierte TextBlock de MIT al formato de server.py."""
    pts = np.array(region.lines)
    all_pts = np.vstack(pts) if len(pts) > 0 else pts
    if all_pts.size == 0:
        return None
    xs = all_pts[:, 0]
    ys = all_pts[:, 1]
    x, y = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    w, h = max(5, x2 - x), max(5, y2 - y)

    text = " ".join(region.texts) if hasattr(region, "texts") else ""

    fr, fg, fb = region.fg_color if hasattr(region, "fg_color") and region.fg_color else (0, 0, 0)
    text_color = f"#{fr:02x}{fg:02x}{fb:02x}"
    br, bg, bb = region.bg_color if hasattr(region, "bg_color") and region.bg_color else (255, 255, 255)
    bg_color = f"#{br:02x}{bg:02x}{bb:02x}"

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "text": text,
        "confidence": float(region.prob) if hasattr(region, "prob") else 1.0,
        "fontSize": max(8, int(h * 0.75)),
        "textColor": text_color,
        "bgColor": bg_color,
        "polygon": all_pts.tolist(),
    }

async def _run_pipeline_async(img_bgr: np.ndarray) -> dict:
    """
    Pipeline async: detect -> OCR -> merge -> mask -> inpaint.
    Retorna {inpainted_image_b64, blocks}.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Detección con CTD (Comic Text Detector) — intento 1: umbrales normales
    print("[MIT] Detectando texto (CTD)...")
    textlines, mask_raw, mask = await _detect_dispatch(
        Detector.ctd, img_rgb, detect_size=2048,
        text_threshold=0.5, box_threshold=0.75, unclip_ratio=2.3,
        invert=False, gamma_correct=False, rotate=False, verbose=False
    )

    # 1b. Si CTD no encontró nada, reintentar con umbrales muy bajos para
    #     fuentes estilizadas extremas (góticas, horror, rasgadas)
    if not textlines:
        print("[MIT] Reintentando detección con umbrales bajos (fuentes estilizadas)...")
        textlines, mask_raw, mask = await _detect_dispatch(
            Detector.ctd, img_rgb, detect_size=2560,
            text_threshold=0.25, box_threshold=0.5, unclip_ratio=3.0,
            invert=False, gamma_correct=False, rotate=False, verbose=False
        )

    # 1c. Si sigue sin encontrar, probar con detector default (DBNet)
    if not textlines:
        print("[MIT] Reintentando con detector default (DBNet)...")
        textlines, mask_raw, mask = await _detect_dispatch(
            Detector.default, img_rgb, detect_size=2048,
            text_threshold=0.4, box_threshold=0.6, unclip_ratio=2.5,
            invert=False, gamma_correct=False, rotate=False, verbose=False
        )

    if not textlines:
        print("[MIT] Sin texto detectado (tres intentos fallidos).")
        _, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return {"inpainted_image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(), "blocks": []}

    # 2. OCR con modelo 48px (confianza baja para fuentes estilizadas)
    print(f"[MIT] OCR sobre {len(textlines)} textlines...")
    ocr_config = OcrConfig(ocr=Ocr.ocr48px, min_text_length=1)
    textlines = await _ocr_dispatch(Ocr.ocr48px, img_rgb, textlines, config=ocr_config, device="cpu")

    # 3. Fusión en regiones de texto
    print("[MIT] Fusionando textlines...")
    text_regions = await _merge_dispatch(textlines, w, h)

    # 4. Máscara para inpainting (usar máscara del detector directamente)
    print("[MIT] Preparando máscara de inpainting...")
    if mask is None:
        mask = mask_raw.copy() if mask_raw is not None else np.zeros((h, w), dtype=np.uint8)

    # 5. Inpainting con LaMa
    print("[MIT] Inpainting con LaMa...")
    config = InpainterConfig(inpainter=Inpainter.lama_large, inpainting_size=2048)
    img_inpainted = await _inpaint_dispatch(
        Inpainter.lama_large, img_rgb, mask, config,
        inpainting_size=2048, device="cpu"
    )

    # 6. Convertir resultados
    blocks = []
    for region in text_regions:
        block = _text_region_to_block(region)
        if block:
            blocks.append(block)

    img_bgr_out = cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr_out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    inpainted_b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    print(f"[MIT] Completado: {len(blocks)} bloques detectados.")
    return {"inpainted_image": inpainted_b64, "blocks": blocks}


def run_pipeline(img_bgr: np.ndarray) -> dict:
    """
    Pipeline síncrono: detecta texto, hace OCR, inpinta y devuelve
    {inpainted_image: base64, blocks: [...]}.

    Args:
        img_bgr: numpy array BGR (OpenCV).
    """
    try:
        ensure_ready()
        loop = _get_loop()
        return loop.run_until_complete(_run_pipeline_async(img_bgr))
    except Exception as e:
        print(f"[MIT] Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    # Test rápido
    print("Inicializando MIT pipeline...")
    ensure_ready()
    print("MIT pipeline listo.")