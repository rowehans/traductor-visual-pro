"""
ocr_ctd_fallback.py — Fallback OCR usando CTD (ComicTextDetector) de
manga-image-translator, especificamente entrenado para texto de manga/comics.

Flujo:
1. EasyOCR devuelve 0 bloques -> CTD detecta regiones de texto
2. Cada region se recorta de la imagen original
3. EasyOCR reconoce el texto en cada region recortada
4. Resultados convertidos al formato interno de bloques

Dependencias:
- einops (ya instalado)
- torch con CUDA (ya instalado)
- Modelo: comictextdetector.pt (~76MB, descarga automatica)
"""

import hashlib
import os
import sys
import threading
from typing import Any

import cv2
import numpy as np
import requests
import torch
from tqdm import tqdm

# ── Módulos CTD locales (extraídos de manga-image-translator, ~100KB) ──
# Solo los módulos CTD necesarios están en ctd_lib/ (sin los 1.2GB completos).
_CTD_LIB = os.path.join(os.path.dirname(__file__), "ctd_lib")
if os.path.isdir(_CTD_LIB) and _CTD_LIB not in sys.path:
    sys.path.insert(0, _CTD_LIB)

try:
    from manga_translator.detection.ctd_utils.basemodel import TextDetBase
    from manga_translator.detection.ctd_utils.utils.db_utils import (
        SegDetectorRepresenter,
    )
    from manga_translator.detection.ctd_utils.utils.imgproc_utils import letterbox
    _CTD_IMPORTS_OK = True
except ImportError as e:
    print(f"[CTD] No se pudieron importar modulos CTD: {e}")
    import traceback
    traceback.print_exc()
    _CTD_IMPORTS_OK = False

from config import ROOT

# --- Rutas y constantes ---
_CTD_MODEL_DIR = str(ROOT / "models" / "ctd")
_CTD_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator"
    "/releases/download/beta-0.3/comictextdetector.pt"
)
_CTD_MODEL_SHA256 = "1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb"
_CTD_MODEL_PATH = os.path.join(_CTD_MODEL_DIR, "comictextdetector.pt")
_CTD_SENTINEL = os.path.join(_CTD_MODEL_DIR, ".ctd_ready")
# CTD trabaja en 1024x1024, redimensionamos imagenes grandes a esto
_CTD_INPUT_SIZE: tuple[int, int] = (1024, 1024)

# ── Filtros post-detección CTD (constantes validadas) ────────────
# Thresholds optimizados con barrido parametrico sobre pagina 125
# (peor caso: 31 detecciones) y validados en paginas artisticas 3,11,12.
# Ver test_ctd_thresholds.py (archivo temporal eliminado) para detalles.
#
#   area>=400px²  : filtra motas/patrones pequenos (< 20x20px)
#   height>=8px   : filtra lineas finas de vineta
#   aspect 0.4-20 : evita regiones extremadamente alargadas
#   max 15 reg    : evita saturacion en paginas densas (125-127 tenian 31+)
_CTD_FILTER_MIN_AREA: int = 400
_CTD_FILTER_MIN_HEIGHT: int = 8
_CTD_FILTER_ASPECT_MIN: float = 0.4
_CTD_FILTER_ASPECT_MAX: float = 20.0
_CTD_FILTER_MAX_REGIONS: int = 15

# --- Estado global (thread-safe) ---
_ctd_model: Any = None
_ctd_seg_rep: Any = None
_ctd_device: str = "cpu"
_ctd_available: bool = False
_ctd_lock: threading.Lock = threading.Lock()
_ctd_loaded: bool = False


# --- Descarga del modelo ---
def _download_ctd_model() -> bool:
    """Descarga comictextdetector.pt desde GitHub si no existe."""
    if os.path.exists(_CTD_SENTINEL) and os.path.exists(_CTD_MODEL_PATH):
        print("[CTD] Modelo ya descargado")
        return True

    os.makedirs(_CTD_MODEL_DIR, exist_ok=True)

    if os.path.exists(_CTD_MODEL_PATH):
        sha256 = hashlib.sha256()
        with open(_CTD_MODEL_PATH, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        if sha256.hexdigest().lower() == _CTD_MODEL_SHA256.lower():
            with open(_CTD_SENTINEL, "w") as f:
                f.write("ok")
            print("[CTD] Modelo verificado OK")
            return True
        print("[CTD] Hash incorrecto, redescargando...")
        os.remove(_CTD_MODEL_PATH)

    print(f"[CTD] Descargando modelo ({_CTD_MODEL_URL})...")
    try:
        resp = requests.get(_CTD_MODEL_URL, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(_CTD_MODEL_PATH, "wb") as f:
            with tqdm(
                desc="comictextdetector.pt",
                total=total,
                unit="B",
                unit_scale=True,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        # Verificar hash post-descarga
        sha256 = hashlib.sha256()
        with open(_CTD_MODEL_PATH, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        if sha256.hexdigest().lower() != _CTD_MODEL_SHA256.lower():
            print("[CTD] ERROR: Hash de descarga no coincide")
            os.remove(_CTD_MODEL_PATH)
            return False

        with open(_CTD_SENTINEL, "w") as f:
            f.write("ok")
        print("[CTD] Descarga completada y verificada")
        return True

    except Exception as e:
        print(f"[CTD] Error descargando modelo: {e}")
        if os.path.exists(_CTD_MODEL_PATH):
            os.remove(_CTD_MODEL_PATH)
        return False


# --- Carga del modelo (thread-safe con double-checked locking) ---
def _load_ctd_model() -> bool:
    """Carga comictextdetector.pt en GPU/CPU. Thread-safe."""
    global _ctd_model, _ctd_seg_rep, _ctd_device, _ctd_available, _ctd_loaded

    if _ctd_loaded:
        return True

    with _ctd_lock:
        if _ctd_loaded:
            return True

        if not _CTD_IMPORTS_OK:
            print("[CTD] Modulos de manga-image-translator no disponibles")
            return False

        if not _download_ctd_model():
            print("[CTD] No se pudo descargar el modelo")
            return False

        try:
            # Detectar dispositivo
            _ctd_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[CTD] Cargando modelo en {_ctd_device}...")

            # Cargar modelo PyTorch
            model = TextDetBase(_CTD_MODEL_PATH, device=_ctd_device, act="leaky")
            model.to(_ctd_device)
            model.eval()

            _ctd_model = model
            _ctd_seg_rep = SegDetectorRepresenter(thresh=0.3)
            _ctd_available = True
            _ctd_loaded = True
            print(f"[CTD] Modelo cargado en {_ctd_device}")
            return True

        except Exception as e:
            print(f"[CTD] Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False


# --- Preprocesamiento de imagen para CTD ---
def _ctd_preprocess(img_bgr: np.ndarray
                    ) -> tuple[torch.Tensor, int, int]:
    """
    Preprocesa imagen para entrada a CTD.
    - Redimensiona manteniendo aspect ratio, max 1024px
    - Convierte a tensor CHW normalizado
    - Retorna (tensor, dw, dh) donde dw/dh es el padding aplicado
    """
    h, w = img_bgr.shape[:2]

    # Si la imagen es mas grande que 1024, redimensionar primero
    scale = min(_CTD_INPUT_SIZE[0] / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_bgr.copy()
        scale = 1.0

    # letterbox: padding a 1024x1024 manteniendo aspect ratio
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_in, _, (dw, dh) = letterbox(
        img_rgb, new_shape=_CTD_INPUT_SIZE, auto=False, stride=64
    )

    # Convertir a tensor CHW, normalizar
    img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
    img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
    img_tensor = torch.from_numpy(img_in).to(_ctd_device)

    return img_tensor, dw, dh


# --- Deteccion CTD (sincrona) ---
def _detect_ctd_regions(
    img_bgr: np.ndarray,
    max_dim: int = 1024,
) -> list[dict[str, Any]]:
    """
    Ejecuta CTD sobre una imagen y retorna regiones de texto detectadas.
    Cada region tiene: {x, y, w, h, confidence}
    Retorna lista vacia si no se detecta nada.
    """
    global _ctd_model, _ctd_seg_rep

    if not _ctd_available:
        if not _load_ctd_model():
            return []

    try:
        h, w = img_bgr.shape[:2]

        # Redimensionar si es necesario (CTD espera ~1024px)
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            img_input = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        else:
            img_input = img_bgr
            scale = 1.0

        with torch.no_grad():
            img_tensor, dw, dh = _ctd_preprocess(img_input)

            # Inferencia
            _, mask, lines = _ctd_model(img_tensor)

            # Postprocesar mascara
            mask_np = mask.squeeze_()
            if _ctd_device != "cpu":
                mask_np = mask_np.detach().cpu()
            mask_np = (mask_np.numpy() * 255).astype(np.uint8)

            # Recortar padding
            mask_np = mask_np[:mask_np.shape[0] - dh, :mask_np.shape[1] - dw]
            lines = lines[..., :lines.shape[2] - dh, :lines.shape[3] - dw]

            # Segment detector: lineas -> cuadrilateros
            ih, iw = img_input.shape[:2]
            lines_np = lines.detach().cpu().numpy()
            result_lines, scores = _ctd_seg_rep(
                None, lines_np, height=ih, width=iw
            )

            # Filtrar por confianza
            box_thresh = 0.6
            idx = np.where(scores[0] > box_thresh)
            filtered_lines = result_lines[0][idx]
            filtered_scores = scores[0][idx]

        # Convertir a formato interno (escalando de vuelta si fue necesario)
        regions: list[dict[str, Any]] = []
        for pts, score in zip(filtered_lines, filtered_scores):
            pts_int = (pts / scale).astype(int) if scale < 1.0 else pts.astype(int)
            xs = pts_int[:, 0]
            ys = pts_int[:, 1]
            rx, ry = int(min(xs)), int(min(ys))
            rw = int(max(xs) - rx)
            rh = int(max(ys) - ry)

            if rw < 5 or rh < 5:
                continue

            # ── Filtro post-detección: eliminar falsos positivos ──
            # Thresholds optimizados con barrido parametrico sobre
            # pagina 125 (peor caso: 31 detecciones). Ver analisis:
            #   area>=400px²  : filtra motas/patrones pequenos
            #   height>=8px   : filtra lineas finas de vineta
            #   aspect 0.4-20 : evita regiones extremadamente alargadas
            #   max 15 reg    : evita saturacion en paginas densas
            area = rw * rh
            aspect = rw / max(rh, 1)

            # ── Log debug de regiones filtradas (verificar que son ruido) ──
            if area < _CTD_FILTER_MIN_AREA:
                print(f"[CTD] Filtrado por area  <{_CTD_FILTER_MIN_AREA}: ({rx},{ry}) {rw}x{rh} area={area} conf={float(score):.3f}")
                continue
            if rh < _CTD_FILTER_MIN_HEIGHT:
                print(f"[CTD] Filtrado por altura <{_CTD_FILTER_MIN_HEIGHT}: ({rx},{ry}) {rw}x{rh} area={area} aspect={aspect:.2f} conf={float(score):.3f}")
                continue
            if aspect < _CTD_FILTER_ASPECT_MIN or aspect > _CTD_FILTER_ASPECT_MAX:
                print(f"[CTD] Filtrado por aspect [{_CTD_FILTER_ASPECT_MIN}-{_CTD_FILTER_ASPECT_MAX}]: ({rx},{ry}) {rw}x{rh} area={area} aspect={aspect:.2f} conf={float(score):.3f}")
                continue

            regions.append({
                "x": max(0, rx),
                "y": max(0, ry),
                "w": min(rw, w - rx),
                "h": min(rh, h - ry),
                "confidence": float(score),
            })

        print(f"[CTD] Detectadas {len(regions)} regiones de texto (filtro: {len(filtered_lines)-len(regions)} descartadas)")

        # Ordenar por confianza descendente y limitar
        if _CTD_FILTER_MAX_REGIONS > 0:
            regions.sort(key=lambda r: r["confidence"], reverse=True)
            if len(regions) > _CTD_FILTER_MAX_REGIONS:
                print(f"[CTD] Limitando de {len(regions)} a {_CTD_FILTER_MAX_REGIONS} regiones (solo las mas confiables)")
                regions = regions[:_CTD_FILTER_MAX_REGIONS]
        return regions

    except Exception as e:
        print(f"[CTD] Error en deteccion: {e}")
        import traceback
        traceback.print_exc()
        return []


# --- OCR en regiones CTD (usando EasyOCR) ---
def _ocr_ctd_regions(
    img_bgr: np.ndarray,
    ctd_regions: list[dict[str, Any]],
    easyocr_reader: Any,
) -> list[dict[str, Any]]:
    """
    Para cada region detectada por CTD:
    1. Recortar la region de la imagen original
    2. Pasar a EasyOCR para reconocimiento
    3. Convertir al formato interno de bloques
    """
    from ocr_utils import _run_ocr_on_image, _ocr_results_to_blocks

    all_blocks: list[dict[str, Any]] = []

    for region in ctd_regions:
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]

        # Recortar region con margen
        pad = max(5, int(min(rw, rh) * 0.1))
        x1 = max(0, rx - pad)
        y1 = max(0, ry - pad)
        x2 = min(img_bgr.shape[1], rx + rw + pad)
        y2 = min(img_bgr.shape[0], ry + rh + pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        # OCR en la region recortada
        blocks = _ocr_results_to_blocks(
            _run_ocr_on_image(easyocr_reader, crop), crop
        )

        # Ajustar coordenadas al sistema de la imagen original
        for blk in blocks:
            blk["x"] += x1
            blk["y"] += y1
            all_blocks.append(blk)

    print(f"[CTD] EasyOCR reconocio {len(all_blocks)} bloques en regiones CTD")
    return all_blocks


# --- API principal ---
def ctd_fallback_ocr(
    img_bgr: np.ndarray,
    easyocr_reader: Any,
    ocr_lang: str = "es",
) -> list[dict[str, Any]]:
    """
    Punto de entrada principal para el fallback CTD.
    1. CTD detecta regiones de texto
    2. EasyOCR reconoce texto en cada region
    3. Retorna bloques en formato interno

    Retorna lista vacia si no se detecta nada o si CTD no esta disponible.
    """
    if not _CTD_IMPORTS_OK:
        print("[CTD] Modulos no disponibles, saltando fallback")
        return []

    # Paso 1: CTD detecta regiones
    ctd_regions = _detect_ctd_regions(img_bgr)
    if not ctd_regions:
        return []

    # Paso 2: EasyOCR reconoce en cada region
    blocks = _ocr_ctd_regions(img_bgr, ctd_regions, easyocr_reader)

    # Paso 3: Agrupar y mergear (reusa la logica existente)
    from ocr_utils import _group_and_merge_blocks
    return _group_and_merge_blocks(blocks, img_bgr.shape[0])


def is_ctd_available() -> bool:
    """Verifica si CTD esta descargado y listo para usar."""
    return _ctd_loaded or (
        os.path.exists(_CTD_SENTINEL) and os.path.exists(_CTD_MODEL_PATH)
    )


def preload_ctd() -> bool:
    """
    Precarga CTD en memoria (util al inicio del servidor).
    Retorna True si se cargo exitosamente.
    """
    return _load_ctd_model()
