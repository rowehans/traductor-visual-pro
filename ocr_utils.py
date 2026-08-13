"""
ocr_utils.py — OCR (EasyOCR), inpainting (OpenCV), filtros de ruido y muestreo de color.

Extraído de server.py. Depende de config.py para patrones de ruido y constantes.
"""

import base64
import re
import threading
import unicodedata
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from config import (ROOT, MARGIN_NOISE_PATTERNS, WATERMARK_PATTERNS,
                    FUSION_TYPE_REINFORCE, FUSION_TYPE_WEIGHTS)

# Type alias for OpenCV images (suppresses overly-strict ndarray checks)
_Img = np.ndarray


# ─── EasyOCR (lazy load with multi-lang support + CPU fallback) ──
_ocr_readers: dict[str, Any] = {}
_ocr_lock: threading.Lock = threading.Lock()
# Semaforo para limitar concurrencia OCR: max 1 lectura simultanea
# porque EasyOCR no es thread-safe y cada reader consume ~1-2GB VRAM/RAM
_ocr_semaphore: threading.Semaphore = threading.Semaphore(1)
_rapid_semaphore: threading.Semaphore = threading.Semaphore(1)
# Benchmark (disable_uocr): apaga el TextClassifier de la Ruta C junto con
# el VLM — mide el overhead puro de la fusión sin la clasificación de
# rotación. Mismo patrón que _uocr_inferring (Event global consultado en
# runtime, seteado por OCRManager).
_ruta_c_cls_disabled: threading.Event = threading.Event()
# Benchmark (disable_uocr): también apaga el detector YOLO de regiones (Fase
# 6) — el overhead puro de la fusión se mide sin el recuperador de regiones
# (mismo patrón que _ruta_c_cls_disabled, set/clear por request).
_yolo_disabled: threading.Event = threading.Event()
# Benchmark (disable_uocr): también apaga el detector comic-text-detector
# (Tier 3.6, Fase 6.5 — PLAN_MANGA_OCR Paso 4). Mismo patrón que
# _yolo_disabled: el overhead puro de la fusión se mide sin recuperadores.
_ctd_disabled: threading.Event = threading.Event()
# Lock global de GPU (RLock): serializa la inferencia de EasyOCR (server, GPU)
# con la del daemon U-OCR (proceso separado, mismo GPU). Sin esto, un worker
# corriendo EasyOCR compite por VRAM con el daemon mientras infiere → el daemon
# pasa de 83s a 140-1439s por página (benchmark fusion 2026-08-03, §3.6 v4.2).
_gpu_lock: threading.RLock = threading.RLock()

# Flag global de degradación (v4.2): mientras el daemon U-OCR infiere en GPU
# (proceso separado que comparte la GTX 1050 Ti de 4GB), los workers que
# procesan OTRAS páginas en paralelo degradan a RapidOCR CPU en vez de esperar
# el _gpu_lock — así el daemon tiene la VRAM completa y las páginas normales
# avanzan en CPU sin bloquearse. Se setea en routes/api._ocr_with_unlimited.
_uocr_inferring: threading.Event = threading.Event()


def _get_ocr_reader(lang: str = "auto") -> Any:
    lang_key: str = "latin"
    if lang == "ja":
        lang_key = "ja"
    elif lang == "ko":
        lang_key = "ko"
    elif lang == "zh":
        lang_key = "zh"

    global _ocr_readers  # noqa: PLW0603
    if lang_key in _ocr_readers:
        return _ocr_readers[lang_key]

    with _ocr_lock:
        if lang_key in _ocr_readers:
            return _ocr_readers[lang_key]

        langs: list[str] = ["es", "en"]
        if lang_key == "ja":
            langs = ["ja", "en"]
        elif lang_key == "ko":
            langs = ["ko", "en"]
        elif lang_key == "zh":
            langs = ["ch_sim", "en"]
        else:
            langs = ["es", "en", "pt", "fr", "de"]

        # ═══ Carga EasyOCR: GPU si CUDA disponible, CPU si no ═══════
        # El orden de carga es CRÍTICO para evitar conflicto cuDNN:
        #   - Si CT2 carga PRIMERO CUDA/cuDNN, luego PyTorch no puede
        #     cargar sus propios símbolos cuDNN → crash.
        #   - Server.py ahora carga EasyOCR PRIMERO (PyTorch toma GPU,
        #     carga cuDNN), y luego CT2 con force_cpu=True.
        #   - Si el usuario llama EasyOCR sin preload (caso de esquina),
        #     también funciona: PyTorch inicializa CUDA, CT2 detecta
        #     que CUDA está disponible pero se pasa force_cpu como fallback.
        # ════════════════════════════════════════════════════════════
        print(f"[OCR] Cargando EasyOCR para {langs}...")

        try:
            import easyocr
            # Intentar GPU primero. Si falla (CUDA no disponible, memoria insuficiente),
            # EasyOCR automáticamente cae a CPU via el try/except que sigue.
            gpu_available = True
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except Exception:
                gpu_available = False

            reader = easyocr.Reader(
                langs,
                gpu=gpu_available,
                model_storage_directory=str(ROOT / "ocr_models"),
                download_enabled=True,
                verbose=False,
            )
            _ocr_readers[lang_key] = reader
            device_str = "GPU" if gpu_available else "CPU"
            print(f"[OCR] EasyOCR para {langs} listo en {device_str}.")
        except Exception as e:
            print(f"[OCR] Error cargando EasyOCR en GPU para {langs}: {e}")
            print(f"[OCR] Reintentando en CPU...")
            try:
                import easyocr
                reader = easyocr.Reader(
                    langs,
                    gpu=False,
                    model_storage_directory=str(ROOT / "ocr_models"),
                    download_enabled=True,
                    verbose=False,
                )
                _ocr_readers[lang_key] = reader
                print(f"[OCR] EasyOCR para {langs} listo en CPU (fallback).")
            except Exception as e2:
                print(f"[OCR] Error cargando EasyOCR incluso en CPU: {e2}")
                return None
    return _ocr_readers[lang_key]


# ─── RapidOCR (lazy load with thread safety) ─────────────────────
# Usa los mismos modelos PP-OCRv4 que PaddleOCR pero via ONNX Runtime,
# sin el conflicto de PaddlePaddle vs PyTorch.
_rapid_engine: Any = None
_rapid_lock: threading.Lock = threading.Lock()


def _get_rapid_engine() -> Any:
    global _rapid_engine
    if _rapid_engine is not None:
        return _rapid_engine
    with _rapid_lock:
        if _rapid_engine is not None:
            return _rapid_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_engine = RapidOCR()
            print("[OCR] RapidOCR listo (CPU/ONNX)")
        except Exception as e:
            print(f"[OCR] Error cargando RapidOCR: {e}")
            return None
    return _rapid_engine


def _preprocess_rapid(img_bgr: _Img) -> _Img:
    """Preprocesamiento optimizado para RapidOCR (pre-filter + enhance)."""
    filtered = _pre_filter_image(img_bgr)
    enhanced = _preprocess_enhanced(filtered)
    return enhanced


# ─── YOLO detector de regiones de texto (Fase 6) ─────────────────
# Tier 3.5 de detección: un YOLO fine-tuned detecta globos/cartelas/títulos
# como OBJETOS (no como texto). Cada región alimenta la Ruta C existente
# (_recover_regions_with_easyocr). ultralytics se importa EN RUNTIME para no
# inflar el .exe ni romper el import del módulo si no está instalado.
_yolo_engine: Any = None
_yolo_lock: threading.Lock = threading.Lock()
# Device YOLO resuelto UNA sola vez por proceso (sesión 116) — la base de la
# política determinista del trigger v4.2 (ver _resolver_device_yolo).
_yolo_device: str | None = None
_yolo_device_lock: threading.Lock = threading.Lock()


def _get_yolo_engine() -> Any:
    """Lazy-load del detector YOLO (ultralytics) con thread-safety.

    Devuelve None si ultralytics no está instalado, el modelo no existe y
    YOLO_AUTODOWNLOAD=False, o la carga falla — los callers degradan a [] sin
    romper el pipeline (mismo patrón que _get_rapid_engine).
    """
    global _yolo_engine
    if _yolo_engine is not None:
        return _yolo_engine
    with _yolo_lock:
        if _yolo_engine is not None:
            return _yolo_engine
        try:
            from config import YOLO_ENABLED, YOLO_MODEL_PATH, YOLO_AUTODOWNLOAD
            if not YOLO_ENABLED:
                return None
            import os
            if not os.path.exists(YOLO_MODEL_PATH):
                if YOLO_AUTODOWNLOAD:
                    print(f"[YOLO] Modelo no existe ({YOLO_MODEL_PATH}); "
                          "descargando yolov8n.pt (requiere internet)...")
                else:
                    print(f"[YOLO] Modelo no existe ({YOLO_MODEL_PATH}) y "
                          "YOLO_AUTODOWNLOAD=False — tier YOLO degradado a []")
                    return None
            # Import DINÁMICO: ultralytics no se importa al cargar ocr_utils
            # (el .exe no lo necesita; si falta, el tier simplemente no aporta).
            from ultralytics import YOLO
            engine = YOLO(YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH)
                          else "yolov8n.pt")
            _yolo_engine = engine
            print(f"[YOLO] Detector listo (modelo={YOLO_MODEL_PATH})")
        except Exception as e:
            print(f"[YOLO] No disponible ({e}); tier YOLO degradado a []")
            return None
    return _yolo_engine


def _resolver_device_yolo() -> str:
    """Resuelve el device YOLO UNA sola vez por proceso (sesión 116).

    Política determinista del trigger v4.2: la decisión de disparar el VLM no
    puede depender del estado dinámico de _gpu_lock/_uocr_inferring en cada
    página (causa raíz del no-determinismo: la misma p4 disparaba U-OCR en
    single pero no en batch porque YOLO corría GPU o CPU según quién tuviera
    el lock, y GPU vs CPU dan detecciones marginalmente distintas que cruzan
    el umbral YOLO_CONF_THRESH de forma distinta → distinta Ruta C → distinto
    blocks/avg_conf → distinta decisión v4.2). Con la resolución única, TODAS
    las páginas del proceso usan el MISMO device → 2 corridas idénticas dan
    SIEMPRE la misma decisión de trigger por página.

    IMPORTANTE (code review sesión 116): la decisión depende SOLO de
    torch.cuda.is_available() — NO de _uocr_inferring.is_set(). Si el
    resolver consultara el flag del daemon en el primer call, un proceso que
    arrancara justo cuando el daemon infiere resolvería CPU y otro GPU →
    no-determinismo entre corridas (la misma fuente que se elimina). La
    sesión 103 ya verificó que YOLO GPU coexiste con el daemon en VRAM
    (2.25GB + ~1GB + 0.13GB < 4GB), así que el flag no aporta protección y
    sí introduce azar.
    """
    global _yolo_device
    if _yolo_device is not None:
        return _yolo_device
    with _yolo_device_lock:
        if _yolo_device is not None:
            return _yolo_device
        from config import YOLO_DEVICE
        device = YOLO_DEVICE
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "0"   # determinista: solo depende de CUDA
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"
        elif device != "cpu":
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
        _yolo_device = device
        print(f"[YOLO] device fijado UNA vez por proceso: {device}")
        return _yolo_device


def _detect_text_regions_in_page(img_bgr: _Img) -> list[dict[str, Any]]:
    """Detecta regiones de diálogo (globos, cartelas, títulos) con YOLO.

    Fase 6: reemplaza/complementa los blobs OpenCV de
    _detect_bubble_regions_in_panel con un detector entrenado. Cada detección
    se convierte a una región en el MISMO formato que los blobs OpenCV
    ({x, y, w, h, roundness, ...} en coordenadas de página) para que la Ruta C
    (_recover_regions_with_easyocr) la consuma sin cambios.

    Device: resuelto UNA sola vez por proceso por _resolver_device_yolo()
    (sesión 116 — política determinista del trigger v4.2): con YOLO_DEVICE
    "auto" se fija GPU "0" si CUDA está disponible en el arranque (si no,
    "cpu"), y TODAS las páginas del proceso usan ese MISMO device. La
    serialización GPU con EasyOCR (mismo proceso) se hace con _gpu_lock
    BLOQUEANTE (espera ~0.9-2s) en vez de degradar a CPU por llamada — el
    device nunca cambia a mitad de proceso, así que 2 corridas idénticas dan
    la misma detección y la misma decisión de trigger por página.

    Degradación segura: sin ultralytics, sin modelo, o error → [] (el tier
    simplemente no aporta; el pipeline sigue con los blobs OpenCV).
    """
    # Benchmark: disable_uocr apaga el detector YOLO igual que el cls (el
    # overhead puro de la fusión se mide sin el recuperador de regiones).
    if _yolo_disabled.is_set():
        return []
    engine = _get_yolo_engine()
    if engine is None:
        return []
    device = _resolver_device_yolo()
    try:
        from config import (YOLO_CONF_THRESH, YOLO_IOU_THRESH, YOLO_IMGSZ,
                            YOLO_MAX_REGIONS, YOLO_MIN_AREA_RATIO,
                            YOLO_CLASS_KEYWORDS, YOLO_GPU_LOCK_BLOQUEANTE,
                            YOLO_GPU_LOCK_TIMEOUT_S)
        # Serialización GPU determinista (sesión 116): el device se resuelve
        # UNA sola vez por proceso (_resolver_device_yolo) y NO vuelve a
        # depender del estado dinámico de _gpu_lock/_uocr_inferring en cada
        # llamada — la misma página produce SIEMPRE la misma detección YOLO
        # (causa raíz del no-determinismo del trigger v4.2: GPU vs CPU dan
        # detecciones marginalmente distintas que cruzaban el umbral 0.25 de
        # forma distinta → distinta Ruta C → distinto trigger). Con device
        # "0" (GPU), YOLO adquiere _gpu_lock de forma BLOQUEANTE (espera a
        # que EasyOCR de otro worker termine ~0.9-2s) en vez de degradar a
        # CPU — el device es SIEMPRE el mismo → determinista. Solo si el
        # timeout se agota (caso extremo) degrada a CPU sin romper el flujo.
        _gpu_held = False
        if device == "0":
            if YOLO_GPU_LOCK_BLOQUEANTE:
                _gpu_held = _gpu_lock.acquire(timeout=YOLO_GPU_LOCK_TIMEOUT_S)
            else:
                _gpu_held = _gpu_lock.acquire(blocking=False)
            if not _gpu_held:
                print("[YOLO] _gpu_lock no disponible: degradando a CPU")
                device = "cpu"
        try:
            results = engine.predict(
                img_bgr,
                conf=YOLO_CONF_THRESH,
                iou=YOLO_IOU_THRESH,
                imgsz=YOLO_IMGSZ,
                device=device,
                verbose=False,
            )
        finally:
            if _gpu_held:
                _gpu_lock.release()
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        names = results[0].names or {}
        h_page, w_page = img_bgr.shape[:2]
        min_area = (h_page * w_page) * YOLO_MIN_AREA_RATIO
        regions: list[dict[str, Any]] = []
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        for i, box in enumerate(xyxy):
            x0, y0, x1, y1 = [float(v) for v in box]
            x, y = int(round(x0)), int(round(y0))
            w, h = int(round(x1 - x0)), int(round(y1 - y0))
            if w < 8 or h < 8 or (w * h) < min_area:
                continue
            cls_id = int(clss[i]) if i < len(clss) else 0
            label = str(names.get(cls_id, "")).lower()
            # Filtrar por tipo: solo regiones que contienen texto (globo,
            # cartela, título, caja de texto, SFX). Las clases no relevantes
            # (person, face...) se ignoran.
            if label and not any(kw in label for kw in YOLO_CLASS_KEYWORDS):
                continue
            regions.append({
                "x": x, "y": y, "w": w, "h": h,
                "roundness": 0.0,
                "border_ratio": 0.0,
                "dark_ratio": 0.0,
                "source": "yolo",
                "label": label or "text",
                "cls_conf": float(confs[i]) if i < len(confs) else 0.0,
            })
            if len(regions) >= YOLO_MAX_REGIONS:
                break
        if regions:
            print(f"[YOLO] {len(regions)} regiones de diálogo detectadas "
                  f"(device={device})")
        return regions
    except Exception as e:
        print(f"[YOLO] Error en detección de regiones: {e}")
        return []


# ─── Detector de texto de cómic (comic-text-detector ONNX, CPU) ──
# Tier 3.6 de detección: complementa al YOLO de globos (Fase 6) con dmMaze
# comic-text-detector (port ONNX de mayocream, GPL-3.0) — detecta REGIONES de
# texto incluyendo texto SIN globo (flotante sobre el dibujo, pensamientos,
# títulos artísticos) que los OCR y el detector de globos pierden. Corre 100%
# en CPU (onnxruntime) → 0 VRAM extra; batch=1 estricto (firma fija 1024²).
#
# Post-proceso replicado de dmMaze (inference.py, db_utils.py, yolov5_utils.py):
#   blk  → conf=obj*cls + NMS por clase + xywh→xyxy (head YOLOv5)
#   det  → binarizar + contornos + minAreaRect + unclip (head DBNet)
#   seg  → binarizar + contornos NO cubiertos por blk/det (máscara UNet)
# Los 3 heads se fusionan en una lista de regiones del formato de la Ruta C.
#
# DIFERENCIA deliberada vs dmMaze: la inversa del letterbox aquí es EXACTA
# (resta el padding superior/izquierdo y escala por el CONTENIDO real). El
# inference.py original multiplica por resize_ratio sin restar el padding, lo
# que desplaza las cajas ~15-20% de la página en mangas verticales (los crops
# alimentarían a la Ruta C con coordenadas erróneas).
_comic_detector_engine: Any = None
_comic_detector_lock: threading.Lock = threading.Lock()
_COMIC_DETECTOR_IMGSZ: int = 1024
_COMIC_DETECTOR_CLS_NAMES: tuple[str, str] = ("eng", "ja")
_COMIC_DETECTOR_PAD_COLOR: int = 114


def _get_comic_detector_engine() -> Any:
    """Lazy-load del detector de texto de cómic (onnxruntime, CPU) con
    thread-safety.

    Devuelve None si onnxruntime no está instalado, el modelo no existe,
    COMIC_DETECTOR_ENABLED=False, o la carga falla — los callers degradan a []
    sin romper el pipeline (mismo patrón que _get_yolo_engine).
    """
    global _comic_detector_engine
    if _comic_detector_engine is not None:
        return _comic_detector_engine
    with _comic_detector_lock:
        if _comic_detector_engine is not None:
            return _comic_detector_engine
        try:
            from config import COMIC_DETECTOR_ENABLED, COMIC_DETECTOR_MODEL_PATH
            if not COMIC_DETECTOR_ENABLED:
                return None
            import os
            if not os.path.exists(COMIC_DETECTOR_MODEL_PATH):
                print(f"[CTD] Modelo no existe ({COMIC_DETECTOR_MODEL_PATH}); "
                      "tier comic-text-detector degradado a []")
                return None
            # Import DINÁMICO: onnxruntime no se importa al cargar ocr_utils
            # (el .exe no lo necesita si el tier no aporta; si falta, degrada).
            import onnxruntime
            session = onnxruntime.InferenceSession(
                COMIC_DETECTOR_MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            _comic_detector_engine = session
            print(f"[CTD] Detector de texto de cómic listo "
                  f"(modelo={COMIC_DETECTOR_MODEL_PATH})")
        except Exception as e:
            print(f"[CTD] No disponible ({e}); tier degradado a []")
            return None
    return _comic_detector_engine


def _comic_detector_letterbox(img_bgr: _Img) -> tuple[_Img, int, int, float, float]:
    """Letterbox YOLOv5 (auto=False, stride=64, relleno 114) a 1024×1024.

    Devuelve (imagen 1024², left, top, scale_x, scale_y): left/top es el
    padding superior-izquierdo en el espacio 1024, y scale_x/scale_y mapean el
    CONTENIDO al tamaño original — x_orig = (x_pad - left) * scale_x. Esta
    inversa EXACTA del letterbox corrige el desplazamiento que el inference.py
    de dmMaze omite (multiplica por resize_ratio sin restar el padding).
    """
    imgsz = _COMIC_DETECTOR_IMGSZ
    h, w = img_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (W', H')
    dw = (imgsz - new_unpad[0]) / 2.0
    dh = (imgsz - new_unpad[1]) / 2.0
    resized = img_bgr
    if (h, w) != (new_unpad[1], new_unpad[0]):
        resized = cv2.resize(img_bgr, new_unpad,
                             interpolation=cv2.INTER_LINEAR)
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT,
                                value=(_COMIC_DETECTOR_PAD_COLOR,) * 3)
    scale_x = w / new_unpad[0] if new_unpad[0] else 1.0
    scale_y = h / new_unpad[1] if new_unpad[1] else 1.0
    return padded, left, top, scale_x, scale_y


def _comic_detector_nms(boxes: np.ndarray, scores: np.ndarray,
                        classes: np.ndarray, iou_thresh: float) -> list[int]:
    """NMS por clase sobre cajas xyxy (mismo comportamiento que el
    non_max_suppression de yolov5_utils.py de dmMaze: solo suprime entre
    cajas de la MISMA clase). Devuelve índices en orden de confianza
    descendente."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        same_cls = classes[rest] == classes[i]
        b = boxes[i]
        r = boxes[rest]
        xx1 = np.maximum(b[0], r[:, 0])
        yy1 = np.maximum(b[1], r[:, 1])
        xx2 = np.minimum(b[2], r[:, 2])
        yy2 = np.minimum(b[3], r[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
        area_r = np.clip(r[:, 2] - r[:, 0], 0, None) * np.clip(r[:, 3] - r[:, 1], 0, None)
        iou = inter / (area_b + area_r - inter + 1e-9)
        order = rest[~(same_cls & (iou > iou_thresh))]
    return keep


def _comic_detector_box_score(prob_map: np.ndarray,
                              contour_pts: np.ndarray) -> float:
    """Media del mapa de probabilidad dentro del contorno (box_score_fast de
    db_utils.py de dmMaze): máscara binaria del polígono sobre su bbox."""
    if contour_pts.ndim != 2 or contour_pts.shape[1] != 2 or contour_pts.shape[0] < 3:
        return 0.0
    h, w = prob_map.shape[:2]
    pts = contour_pts.astype(np.float64)
    xmin = int(np.clip(np.floor(pts[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(pts[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(pts[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(pts[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = pts.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    if mask.sum() == 0:
        return 0.0
    return float(cv2.mean(prob_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0])


def _comic_detector_map_box(x0p: float, y0p: float, x1p: float, y1p: float,
                            left: int, top: int, scale_x: float,
                            scale_y: float, im_w: int, im_h: int,
                            min_area: float) -> tuple[int, int, int, int] | None:
    """Mapea una caja del espacio 1024 (padded) a coordenadas de página con la
    inversa EXACTA del letterbox, recorta a la página y filtra por tamaño
    mínimo. Devuelve (x, y, w, h) o None si la caja es inválida/diminuta."""
    x0o = int(round((x0p - left) * scale_x))
    y0o = int(round((y0p - top) * scale_y))
    x1o = int(round((x1p - left) * scale_x))
    y1o = int(round((y1p - top) * scale_y))
    x0o, y0o = max(x0o, 0), max(y0o, 0)
    x1o, y1o = min(x1o, im_w), min(y1o, im_h)
    w_ = x1o - x0o
    h_ = y1o - y0o
    if w_ < 8 or h_ < 8 or (w_ * h_) < min_area:
        return None
    return x0o, y0o, w_, h_


def _comic_detector_blk_regions(blk: np.ndarray, left: int, top: int,
                                scale_x: float, scale_y: float,
                                conf_thresh: float, nms_thresh: float,
                                max_regions: int, min_area: float,
                                im_w: int, im_h: int) -> list[dict[str, Any]]:
    """Head YOLOv5 (blk): [cx,cy,w,h,obj,cls...] → conf=obj*cls → NMS por
    clase → regiones de BLOQUE de texto en coordenadas de página. Réplica de
    postprocess_yolo + non_max_suppression de dmMaze (con la inversa del
    letterbox EXACTA)."""
    pred = blk[0] if blk.ndim == 3 else blk  # [N, 7]
    if pred.size == 0:
        return []
    obj = pred[:, 4]
    cls_scores = pred[:, 5:] if pred.shape[1] > 5 else np.zeros_like(obj)
    scores = obj * cls_scores.max(axis=1)
    cand = scores >= conf_thresh
    if not cand.any():
        return []
    p = pred[cand]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2.0, cy - h / 2.0,
                      cx + w / 2.0, cy + h / 2.0], axis=1)
    boxes = np.clip(boxes, 0, _COMIC_DETECTOR_IMGSZ)
    cls = cls_scores[cand].argmax(axis=1)
    sc = scores[cand]
    regions: list[dict[str, Any]] = []
    for i in _comic_detector_nms(boxes, sc, cls, nms_thresh):
        x0p, y0p, x1p, y1p = boxes[i]
        mapped = _comic_detector_map_box(x0p, y0p, x1p, y1p, left, top,
                                         scale_x, scale_y, im_w, im_h,
                                         min_area)
        if mapped is None:
            continue
        x0o, y0o, w_, h_ = mapped
        cls_id = int(cls[i])
        label = (_COMIC_DETECTOR_CLS_NAMES[cls_id]
                 if cls_id < len(_COMIC_DETECTOR_CLS_NAMES) else "unknown")
        regions.append({
            "x": x0o, "y": y0o, "w": w_, "h": h_,
            "roundness": 0.0, "border_ratio": 0.0, "dark_ratio": 0.0,
            "source": "ctd", "label": f"ctd_{label}",
            "cls_conf": float(sc[i]),
        })
        if len(regions) >= max_regions:
            break
    return regions


def _comic_detector_det_regions(det: np.ndarray, left: int, top: int,
                                scale_x: float, scale_y: float,
                                mask_thresh: float, score_thresh: float,
                                unclip_ratio: float, max_regions: int,
                                min_area: float, im_w: int,
                                im_h: int) -> list[dict[str, Any]]:
    """Head DBNet (det): mapa shrink → binarizar → contornos → minAreaRect →
    unclip → score medio del mapa → regiones de LÍNEA de texto. Réplica del
    SegDetectorRepresenter de dmMaze (boxes_from_bitmap + unclip con
    pyclipper), simplificado a cajas axis-aligned — los crops de la Ruta C son
    axis-aligned de todos modos, y así se evitan las dependencias
    pyclipper/shapely."""
    lines_map = det[0, 0] if det.ndim == 4 else det[0]
    binary = (lines_map > mask_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []
    for contour in contours[:1000]:
        c = contour.squeeze(1)
        if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 3:
            continue
        x0p = float(c[:, 0].min())
        y0p = float(c[:, 1].min())
        x1p = float(c[:, 0].max())
        y1p = float(c[:, 1].max())
        area = (x1p - x0p) * (y1p - y0p)
        perim = 2.0 * ((x1p - x0p) + (y1p - y0p))
        if area <= 0 or perim <= 0:
            continue
        # Unclip DBNet: expande por distance = area * ratio / perimetro
        dist = area * unclip_ratio / perim
        x0p, y0p = max(x0p - dist, 0.0), max(y0p - dist, 0.0)
        x1p = min(x1p + dist, _COMIC_DETECTOR_IMGSZ)
        y1p = min(y1p + dist, _COMIC_DETECTOR_IMGSZ)
        score = _comic_detector_box_score(lines_map, c)
        if score < score_thresh:
            continue
        mapped = _comic_detector_map_box(x0p, y0p, x1p, y1p, left, top,
                                         scale_x, scale_y, im_w, im_h,
                                         min_area)
        if mapped is None:
            continue
        x0o, y0o, w_, h_ = mapped
        regions.append({
            "x": x0o, "y": y0o, "w": w_, "h": h_,
            "roundness": 0.0, "border_ratio": 0.0, "dark_ratio": 0.0,
            "source": "ctd", "label": "ctd_line",
            "cls_conf": float(score),
        })
        if len(regions) >= max_regions:
            break
    return regions


def _comic_detector_seg_regions(seg: np.ndarray, left: int, top: int,
                                scale_x: float, scale_y: float,
                                mask_thresh: float,
                                existing: list[tuple[int, int, int, int]],
                                max_regions: int, min_area: float,
                                im_w: int, im_h: int) -> list[dict[str, Any]]:
    """Head UNet (seg): máscara de texto → binarizar → contornos → regiones de
    MÁSCARA. Solo se conservan los blobs cuyo CENTRO no cae dentro de una
    región blk/det ya detectada (la máscara es el head de mayor cobertura y
    sirve de red de seguridad para texto decorativo sin caja)."""
    mask = seg[0, 0] if seg.ndim == 4 else seg[0]
    binary = (mask > mask_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []
    for contour in contours[:1000]:
        c = contour.squeeze(1)
        if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 3:
            continue
        cx = float(c[:, 0].mean())
        cy = float(c[:, 1].mean())
        x0o = int(round((cx - left) * scale_x))
        y0o = int(round((cy - top) * scale_y))
        if any(x0 <= x0o <= x0 + w_ and y0 <= y0o <= y0 + h_
               for (x0, y0, w_, h_) in existing):
            continue
        x0p = float(c[:, 0].min())
        y0p = float(c[:, 1].min())
        x1p = float(c[:, 0].max())
        y1p = float(c[:, 1].max())
        score = _comic_detector_box_score(mask, c)
        mapped = _comic_detector_map_box(x0p, y0p, x1p, y1p, left, top,
                                         scale_x, scale_y, im_w, im_h,
                                         min_area)
        if mapped is None:
            continue
        x0o, y0o, w_, h_ = mapped
        regions.append({
            "x": x0o, "y": y0o, "w": w_, "h": h_,
            "roundness": 0.0, "border_ratio": 0.0, "dark_ratio": 0.0,
            "source": "ctd", "label": "ctd_mask",
            "cls_conf": float(score),
        })
        if len(regions) >= max_regions:
            break
    return regions


def _detect_text_regions_comic_detector(img_bgr: _Img) -> list[dict[str, Any]]:
    """Detecta regiones de texto de cómic (texto SIN globo: flotante sobre el
    dibujo, pensamientos, títulos artísticos) con comic-text-detector ONNX.

    Tier 3.6 de detección — complementa a _detect_text_regions_in_page (YOLO
    de globos, Fase 6). Corre 100% en CPU (onnxruntime) → 0 VRAM extra; una
    imagen a la vez (batch=1, firma fija 1024²).

    Usa los 3 heads del modelo (blk/seg/det) con el post-proceso de dmMaze
    (NMS por clase, DBNet unclip, umbrales por defecto) y devuelve regiones en
    el MISMO formato que la Ruta C espera ({x, y, w, h, roundness, ...},
    source='ctd') para que el re-OCR las consuma sin cambios.

    Degradación segura: sin onnxruntime, sin modelo, o error → [] (el tier
    simplemente no aporta).
    """
    # Benchmark: disable_uocr apaga el detector (mismo patrón que YOLO — se
    # chequea aquí y en _ruta_c_ctd para que los tests que mockean la función
    # del detector lo respeten).
    if _ctd_disabled.is_set():
        return []
    engine = _get_comic_detector_engine()
    if engine is None:
        return []
    from config import (COMIC_DETECTOR_CONF_THRESH, COMIC_DETECTOR_NMS_THRESH,
                        COMIC_DETECTOR_MASK_THRESH,
                        COMIC_DETECTOR_LINE_SCORE_THRESH,
                        COMIC_DETECTOR_UNCLIP_RATIO,
                        COMIC_DETECTOR_MAX_REGIONS,
                        COMIC_DETECTOR_MIN_AREA_RATIO)
    try:
        im_h, im_w = img_bgr.shape[:2]
        img1024, left, top, scale_x, scale_y = _comic_detector_letterbox(img_bgr)
        # HWC → CHW y canales BGR tal cual (dmMaze: transposición + reversa =
        # la red fue entrenada con BGR — el input NO lleva swap a RGB)
        blob = np.ascontiguousarray(img1024.transpose((2, 0, 1))[::-1])
        blob = blob[np.newaxis, ...].astype(np.float32) / 255.0
        input_name = engine.get_inputs()[0].name
        blk, seg, det = engine.run(None, {input_name: blob})
        blk = np.asarray(blk)
        seg = np.asarray(seg)
        det = np.asarray(det)
        min_area = (im_h * im_w) * COMIC_DETECTOR_MIN_AREA_RATIO
        blk_regions = _comic_detector_blk_regions(
            blk, left, top, scale_x, scale_y, COMIC_DETECTOR_CONF_THRESH,
            COMIC_DETECTOR_NMS_THRESH, COMIC_DETECTOR_MAX_REGIONS, min_area,
            im_w, im_h)
        det_regions = _comic_detector_det_regions(
            det, left, top, scale_x, scale_y, COMIC_DETECTOR_MASK_THRESH,
            COMIC_DETECTOR_LINE_SCORE_THRESH, COMIC_DETECTOR_UNCLIP_RATIO,
            COMIC_DETECTOR_MAX_REGIONS, min_area, im_w, im_h)
        existing = [(r["x"], r["y"], r["w"], r["h"])
                    for r in blk_regions + det_regions]
        seg_regions = _comic_detector_seg_regions(
            seg, left, top, scale_x, scale_y, COMIC_DETECTOR_MASK_THRESH,
            existing, COMIC_DETECTOR_MAX_REGIONS, min_area, im_w, im_h)
        regions = blk_regions + det_regions + seg_regions
        if regions:
            print(f"[CTD] {len(regions)} regiones de texto de cómic detectadas "
                  f"(blk={len(blk_regions)}, línea={len(det_regions)}, "
                  f"máscara={len(seg_regions)})")
        return regions
    except Exception as e:
        print(f"[CTD] Error en detección de regiones: {e}")
        return []


def _run_rapidocr(
    img_bgr: _Img,
    box_thresh: float | None = None,
    unclip_ratio: float | None = None,
    text_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    Ejecuta RapidOCR sobre una imagen y retorna bloques en el
    mismo formato que EasyOCR (x, y, w, h, text, confidence, ...).
    Adquiere _rapid_semaphore (ONNX Runtime no es thread-safe).

    Parámetros de detección (Fase 2 — reintento agresivo):
        box_thresh: umbral del detector DBNet (default 0.5). Menor → acepta
            cajas más débiles (texto artístico/decorativo).
        unclip_ratio: expansión de la caja tras la máscara (default 1.6).
            Mayor → recupera texto partido en glifos sueltos.
        text_score: umbral del recognizer (default 0.5). Menor → acepta
            caracteres menos confiables.
    Los valores SIEMPRE se pasan explícitos (defaults si no se indican): la
    librería muta postprocess_op.box_thresh/unclip_ratio en la primera
    llamada con kwargs, así que una llamada agresiva anterior no debe
    filtrarse a una llamada default posterior.
    """
    engine = _get_rapid_engine()
    if engine is None:
        return []
    acquired = _rapid_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        print("[OCR] Timeout adquiriendo semaforo RapidOCR (120s)")
        return []
    try:
        params = {
            "box_thresh": box_thresh if box_thresh is not None else 0.5,
            "unclip_ratio": unclip_ratio if unclip_ratio is not None else 1.6,
            "text_score": text_score if text_score is not None else 0.5,
        }
        result, _ = engine(img_bgr, **params)
        blocks: list[dict[str, Any]] = []
        if result:
            for r in result:
                try:
                    bbox, text, conf = r
                    text = str(text).strip()
                    if not text or conf < 0.08:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    if w < 3 or h < 3:
                        continue
                    font_size = max(8, int(h * 0.75))
                    cx, cy = x + w // 2, y + h // 2
                    pad = max(2, h // 6)
                    roi = img_bgr[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
                    text_color = "#ffffff"
                    if roi.size > 0:
                        mean_b = int(np.mean(roi[:, :, 0]))
                        mean_g = int(np.mean(roi[:, :, 1]))
                        mean_r = int(np.mean(roi[:, :, 2]))
                        brightness = mean_r * 0.299 + mean_g * 0.587 + mean_b * 0.114
                        text_color = "#000000" if brightness > 128 else "#ffffff"

                    blocks.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "text": text,
                        "confidence": float(conf),
                        "fontSize": font_size,
                        "textColor": text_color,
                    })
                except (ValueError, IndexError, TypeError):
                    continue
        return _group_and_merge_blocks(blocks, img_bgr.shape[0])
    except Exception as e:
        print(f"[OCR] Error en RapidOCR: {e}")
        return []
    finally:
        if acquired:
            _rapid_semaphore.release()


def _normalize_text(t: str) -> str:
    """Normaliza texto para comparacion entre OCRs: lowercase + sin acentos."""
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return t


def _block_score(b: dict[str, Any]) -> float:
    """Score de calidad de un bloque: confianza × factor de longitud × tipo.

    El multiplicador de tipo (Fase 3) aplica SOLO a bloques con "type"
    semántico del VLM (title/header): en el dedup/NMS de la fusión, un bloque
    tipado del VLM gana empatando contra un bloque sin tipo (EasyOCR/RapidOCR
    nunca llevan type → factor 1.0, sin cambio de comportamiento).
    """
    text_len = len(str(b.get("text", "")).strip())
    conf = float(b.get("confidence", 0.5))
    tw = FUSION_TYPE_WEIGHTS.get(str(b.get("type", "")).lower(), 1.0)
    return conf * tw * min(2.0, max(0.5, text_len / 5.0))


def _overlap_ratio(b1: dict[str, Any], b2: dict[str, Any]) -> float:
    """Ratio de solapamiento espacial entre dos bloques (inter/min_area)."""
    x1 = max(b1["x"], b2["x"])
    y1 = max(b1["y"], b2["y"])
    x2 = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
    y2 = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    min_area = min(b1["w"] * b1["h"], b2["w"] * b2["h"])
    return inter / float(min_area) if min_area > 0 else 0.0


def _estimate_confidence_heuristic(block: dict[str, Any], block_type: str | None = None) -> float:
    """Confianza heurística para bloques de Unlimited-OCR (el modelo no emite confianza).

    Reglas (validado empíricamente: los logits saturan ~0.997 y no diferencian):
      - Base por tipo: text/title 0.90, header 0.70, resto 0.80.
      - Calidad del texto: sin letras ×0.5; ≤2 chars ×0.7; ratio vocales ≥0.25 +0.05.
      - from_art_recrop (re-OCR artístico del daemon): piso 0.80.
      - fontSize 10-40px (rango natural de diálogo): +0.03.
    """
    conf = {"text": 0.90, "title": 0.90, "header": 0.70}.get(block_type or "", 0.80)
    text = str(block.get("text", ""))
    letters = [c for c in text if c.isalpha()]
    if not letters:
        conf *= 0.5
    elif len(text) <= 2:
        conf *= 0.7
    else:
        vocals = sum(c in "aeiouáéíóúAEIOUÁÉÍÓÚ" for c in letters)
        if vocals / len(letters) >= 0.25:
            conf += 0.05
    if block.get("from_art_recrop"):
        conf = max(conf, 0.80)
    fs = int(block.get("fontSize", 0) or 0)
    if 10 <= fs <= 40:
        conf += 0.03
    return round(min(1.0, conf), 4)


def _fusionar_blocks_multi(
    sources: list[list[dict[str, Any]]],
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Fusiona bloques de N motores OCR en una lista unificada.

    sources: una lista de bloques por motor (EasyOCR, RapidOCR, Unlimited-OCR...).
    weights: peso de calibración por motor (mismo orden; default 1.0).

    Pasos:
      1. Dedup por texto normalizado idéntico → gana el de mayor score.
      2. Alineación Levenshtein: bloques con IoU > 0.4 y texto a distancia
         de edición ≤ 30% de la longitud se consideran el mismo texto.
      3. Votación: si 2+ motores coinciden en texto/región, la confianza del
         ganador se refuerza (+0.15).
      4. NMS espacial: IoU > 0.40 descarta el duplicado de menor score.
    """
    if not sources:
        return []
    non_empty = [s for s in sources if s]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return list(non_empty[0])
    if weights is None:
        weights = [1.0] * len(sources)

    # Etiquetar cada bloque con el índice de su motor
    tagged: list[dict[str, Any]] = []
    for i, src in enumerate(sources):
        w = weights[i] if i < len(weights) else 1.0
        for b in src:
            if not str(b.get("text", "")).strip():
                continue
            tagged.append({**b, "_engine": i, "_weight": w})

    def _score(b: dict[str, Any]) -> float:
        return _block_score(b) * b.get("_weight", 1.0)

    # 1. Dedup por texto normalizado idéntico
    by_text: dict[str, dict[str, Any]] = {}
    for b in tagged:
        key = _normalize_text(b["text"])
        if key not in by_text or _score(b) > _score(by_text[key]):
            by_text[key] = b

    # 2. Alineación Levenshtein entre bloques solapados espacialmente
    candidates: list[dict[str, Any]] = list(by_text.values())
    candidates.sort(key=_score, reverse=True)
    final_result: list[dict[str, Any]] = []
    for b in candidates:
        merged = False
        for existing in final_result:
            if _overlap_ratio(b, existing) > 0.40:
                nb = _normalize_text(b["text"])
                ne = _normalize_text(existing["text"])
                max_len = max(len(nb), len(ne))
                if max_len == 0:
                    continue
                dist = _levenshtein(nb, ne)
                same_text = nb == ne or dist / max_len <= 0.30
                if same_text:
                    # Votación: motores distintos coinciden → refuerzo.
                    # Fase 3: el refuerzo se pondera por el tipo semántico
                    # del bloque implicado (solo los del VLM llevan "type"):
                    # title/header pesan más que diálogo común — el acuerdo
                    # entre motores sobre texto distintivo es más fiable.
                    if b.get("_engine") != existing.get("_engine"):
                        _t = str(b.get("type", "")).lower()
                        if _t not in FUSION_TYPE_REINFORCE:
                            _t = str(existing.get("type", "")).lower()
                        if _t not in FUSION_TYPE_REINFORCE:
                            _t = "text"
                        existing["confidence"] = min(
                            1.0, float(existing.get("confidence", 0.5))
                            + FUSION_TYPE_REINFORCE[_t])
                    merged = True
                    break
        if not merged:
            final_result.append(b)

    # 3. NMS espacial final (IoU > 0.40)
    final_result.sort(key=_score, reverse=True)
    nms: list[dict[str, Any]] = []
    for b in final_result:
        if any(_overlap_ratio(b, e) > 0.40 for e in nms):
            continue
        nms.append(b)

    # Limpiar etiquetas internas
    for b in nms:
        b.pop("_engine", None)
        b.pop("_weight", None)
    return nms


def _fusionar_blocks(
    easy_blocks: list[dict[str, Any]],
    rapid_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fusión 2-vías (EasyOCR + RapidOCR) — delega en _fusionar_blocks_multi."""
    return _fusionar_blocks_multi([easy_blocks, rapid_blocks], weights=[1.0, 0.9])


# ─── Corrector ortografico post-OCR (pyspellchecker) ────────────
# Usa pyspellchecker en lugar de un diccionario manual.
# pyspellchecker trae diccionarios completos de espanol, ingles,
# frances, aleman, portugues, etc. SIN mantenimiento manual.
# Correccion por distancia de Levenshtein (max 2).
# Fallback a _levenshtein() si pyspellchecker no esta instalado.
_OCR_SPELLCHECKER: Any = None
_OCR_SPELL_LOCK: threading.Lock = threading.Lock()


def _get_spellchecker(lang: str = "es") -> Any:
    """Retorna instancia de SpellChecker (lazy load, thread-safe)."""
    global _OCR_SPELLCHECKER
    if _OCR_SPELLCHECKER is not None:
        return _OCR_SPELLCHECKER
    with _OCR_SPELL_LOCK:
        if _OCR_SPELLCHECKER is not None:
            return _OCR_SPELLCHECKER
        try:
            from spellchecker import SpellChecker
            sp = SpellChecker(language=lang)
            wf = sp.word_frequency
            # Palabras del dominio manga que pyspellchecker podria no conocer
            # o tener baja frecuencia. Lista FIJA - no requiere mantenimiento.
            # Usamos .add(word, frequency) con alta frecuencia para que
            # pyspellchecker las prefiera sobre alternativas incorrectas.
            MANGA_WORDS = {
                'villano': 1000000, 'villanos': 1000000,
                'villana': 1000000, 'villanas': 1000000,
                'manga': 1000000, 'manhwa': 1000000,
                'webtoon': 1000000, 'scanlation': 1000000,
                'capitulo': 1000000, 'episodio': 1000000,
                'temporada': 1000000, 'personaje': 1000000,
                'protagonista': 1000000, 'antagonista': 1000000,
                'comic': 1000000, 'anime': 1000000,
            }
            for word, freq in MANGA_WORDS.items():
                wf.add(word, freq)
            _OCR_SPELLCHECKER = sp
            print(f"[OCR-spellcheck] pyspellchecker listo (lang={lang}, "
                  f"{len(wf.dictionary)} palabras + {len(MANGA_WORDS)} palabras manga)")
        except Exception as e:
            print(f"[OCR-spellcheck] No se pudo cargar pyspellchecker: {e}")
            return None
    return _OCR_SPELLCHECKER


def _levenshtein(s1: str, s2: str) -> int:
    """Distancia de Levenshtein entre dos strings (iterativa, O(n*m))."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(
                curr[j] + 1,          # delete
                prev[j + 1] + 1,      # insert
                prev[j] + cost         # substitute
            ))
        prev = curr
    return prev[-1]


def _ocr_spellcheck(text: str) -> str:
    """
    Corrector ortografico post-OCR.

    Para cada palabra no reconocida en el diccionario, busca la
    palabra mas cercana por distancia de Levenshtein (max 2).
    Si encuentra una coincidencia, la reemplaza.

    No corrige:
    - Palabras de 1-2 caracteres (muy ambiguas)
    - Palabras en mayusculas que parecen acronimos
    - Nombres propios (no estan en el diccionario, pero no los fuerza)

    Esto funciona para CUALQUIER PDF sin necesidad de mantener GLOSARIO_PRE.
    """
    if not text or len(text) <= 2:
        return text

    palabras = text.split()
    corregidas: list[str] = []
    # Obtener spellchecker UNA VEZ fuera del loop
    sp = _get_spellchecker()

    for p in palabras:
        stripped = p.strip("'\".,;:!?¡¿()[]{}")
        if not stripped or len(stripped) <= 2:
            corregidas.append(p)
            continue

        # Si parece un acronimo (todo mayusculas, >2 chars), no corregir
        if stripped.isupper() and len(stripped) > 2:
            corregidas.append(p)
            continue

        # Si contiene digitos, no corregir
        if any(c.isdigit() for c in stripped):
            corregidas.append(p)
            continue

        # Intentar correccion con pyspellchecker
        p_lower = stripped.lower()

        if sp is not None:
            # pyspellchecker.correction(word) retorna None si la palabra es correcta
            # o la palabra corregida si no lo es (distancia Levenshtein implicita)
            correccion = sp.correction(p_lower)
            if correccion is not None and correccion != p_lower:
                # Preservar capitalizacion original
                if p[0].isupper() and not p.isupper():
                    corregida = correccion.capitalize()
                elif p.isupper():
                    corregida = correccion.upper()
                else:
                    corregida = correccion
                corregidas.append(corregida)
                print(f"[OCR-spellcheck] '{p}' -> '{corregida}'")
                continue
        else:
            # Fallback: _levenshtein sobre palabras comunes clave
            # Solo un minimo necesario para que funcione sin pyspellchecker
            _FALLBACK_DICT: frozenset[str] = frozenset({
                "correctamente", "correcto", "villano", "villanos",
                "temporada", "capitulo", "como", "criar", "persona",
                "primero", "segundo", "tercero", "despues", "entonces",
                "the", "and", "for", "with", "from", "this", "that",
            })
            mejor_dist = 999
            mejor_palabra = None
            for dw in _FALLBACK_DICT:
                if abs(len(dw) - len(p_lower)) > 2:
                    continue
                dist = _levenshtein(p_lower, dw)
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_palabra = dw
                    if dist == 0:
                        break
            if mejor_palabra and 0 < mejor_dist <= 2:
                if p[0].isupper() and not p.isupper():
                    corregida = mejor_palabra.capitalize()
                elif p.isupper():
                    corregida = mejor_palabra.upper()
                else:
                    corregida = mejor_palabra
                corregidas.append(corregida)
                print(f"[OCR-spellcheck] '{p}' -> '{corregida}' (fallback, dist={mejor_dist})")
                continue

        corregidas.append(p)

    return " ".join(corregidas)


# ─── Image base64 conversion ─────────────────────────────────────
def _base64_to_cv2(b64: str) -> _Img | None:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _cv2_to_base64(img: _Img, fmt: str = ".png") -> str:
    # Mapa de extension a MIME type
    _MIME_MAP: dict[str, str] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    mime = _MIME_MAP.get(fmt, "image/png")
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if fmt in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success, buf = cv2.imencode(fmt, img, params)
    if not success:
        raise ValueError(f"No se pudo codificar la imagen en formato {fmt}")
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Preprocesamiento mejorado (fallback para texto artístico) ──
# Cuando EasyOCR no detecta texto con el pipeline normal,
# probamos técnicas más agresivas de mejora de contraste:
# 1. CLAHE en LAB: realza contraste local sin amplificar ruido
# 2. Unsharp mask: afila trazos finos (típico de texto artístico manga)
# 3. Gamma correction: ilumina sombras profundas sin quemar blancos
# 4. Bilateral filter: reduce ruido preservando bordes

def _preprocess_enhanced(img_bgr: _Img) -> _Img:
    """
    Preprocesamiento agresivo para texto artístico/decorativo que
    EasyOCR no detecta con el pipeline normal.

    Aplica CLAHE + Unsharp Mask + Gamma + Bilateral y fusiona
    para mejorar contraste local y nitidez de trazos finos.
    """
    h, w = img_bgr.shape[:2]

    # ── 1. CLAHE adaptativo en espacio LAB ───────────────────────
    # clip_limit adaptativo: imágenes más grandes necesitan más
    # contraste local. tile_grid_size proporcional al tamaño.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    tile_size = max(4, min(16, min(w, h) // 100))
    clip_limit = min(4.0, max(2.0, 3.0 * max(w, h) / 2500.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── 2. Gamma correction (ilumina sombras) ───────────────────
    # Gamma < 1 ilumina zonas oscuras (útil para texto en sombras
    # o escaneos subexpuestos). Aplicar solo si la imagen es oscura.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 100:
        gamma = 0.6 + 0.4 * (mean_brightness / 100.0)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)], dtype=np.uint8)
        img_gamma = cv2.LUT(img_bgr, table)
    else:
        img_gamma = img_bgr

    # ── 3. Bilateral filter (reduce ruido, preserva bordes) ─────
    # A diferencia de GaussianBlur, bilateral preserva bordes
    # de texto mientras suaviza ruido de fondo de escaneo.
    # d=5, sigmaColor=50, sigmaSpace=50 son valores suaves.
    img_denoised = cv2.bilateralFilter(img_gamma, d=5, sigmaColor=50, sigmaSpace=50)

    # ── 4. Unsharp mask sobre la imagen denoised ────────────────
    blurred = cv2.GaussianBlur(img_denoised, (0, 0), sigmaX=1.5)
    img_sharp = cv2.addWeighted(img_denoised, 1.8, blurred, -0.8, 0)

    # ── 5. Fusionar CLAHE + sharp ──────────────────────────────
    # CLAHE da contraste, sharp da nitidez, gamma ilumina sombras.
    enhanced = cv2.addWeighted(img_clahe, 0.5, img_sharp, 0.5, 0)

    return enhanced


# ─── Preprocesamiento morfológico (limpieza de ruido de escaneo) ─
# Elimina líneas horizontales finas, puntos de ruido aislados,
# y limpia bordes de página (común en escaneos manga).

def _pre_filter_image(img_bgr: _Img) -> _Img:
    """
    Limpieza morfológica pre-OCR.
    - Elimina líneas horizontales finas (artefactos de escaneo).
    - Remueve puntos de ruido aislados (speckle).
    - Limpia franjas 4% superior/inferior (sombras de borde).
    - Suaviza ruido de fondo de papel.
    """
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()

    # ── 1. Limpiar franjas 4% superior e inferior ───────────────
    # Los escaneos de manga suelen tener sombras o texto basura
    # en los bordes extremos de la página.
    margin_height = max(1, int(h * 0.04))
    # Superior: rellenar con el color promedio del borde
    top_strip = img_bgr[margin_height:margin_height * 2, :, :]
    if top_strip.size > 0:
        top_fill = np.median(top_strip.reshape(-1, 3), axis=0).astype(np.uint8)
        result[:margin_height, :, :] = top_fill
    # Inferior
    bot_strip = img_bgr[h - margin_height * 2:h - margin_height, :, :]
    if bot_strip.size > 0:
        bot_fill = np.median(bot_strip.reshape(-1, 3), axis=0).astype(np.uint8)
        result[h - margin_height:, :, :] = bot_fill

    # ── 2. Eliminar líneas horizontales finas (artefactos) ─────
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Detectar bordes horizontales con kernel 1x15
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detect_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    # Umbral para identificar líneas
    _, thresh_lines = cv2.threshold(detect_lines, 50, 255, cv2.THRESH_BINARY)
    # Dilatar ligeramente para cubrir la línea completa
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    line_mask = cv2.dilate(thresh_lines, kernel_dilate, iterations=1)
    # Inpainting de las líneas detectadas
    if int(line_mask.max()) > 0:
        result = cv2.inpaint(result, line_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # Re-calcular gray después del inpainting (modificó result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # ── 3. Eliminar puntos de ruido aislados (speckle) ─────────
    # OTSU separa texto oscuro (0) de fondo claro (255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # MORPH_OPEN remueve pequeños puntos blancos aislados (p. ej. speckle de
    # escaneo dentro de regiones oscuras). XOR entre binary y cleaned revela
    # EXACTAMENTE qué píxeles cambiaron de 255→0 por MORPH_OPEN — el speckle
    # real. En lugar de bitwise_and destructivo sobre el canal L completo,
    # sólo inpaintamos esos píxeles específicos, preservando texto y líneas
    # inpaintadas intactos.
    speckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, speckle_kernel, iterations=1)
    speckle_pixels = cv2.bitwise_xor(binary, cleaned)
    # Umbral mínimo de área: evitar inpainting por ruido sub-pixel
    # en páginas limpias (p.ej. bordes de imagen donde MORPH_OPEN
    # elimina 1-2 píxeles fronterizos sin beneficio real).
    if int(speckle_pixels.max()) > 0 and np.count_nonzero(speckle_pixels) > 50:
        # Dilatar 1px con kernel 3x3 para cubrir bordes de speckle
        speckle_mask = cv2.dilate(speckle_pixels, speckle_kernel, iterations=1)
        result = cv2.inpaint(result, speckle_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    # ── 4. Suavizado ligero de fondo (bilateral) ───────────────
    # Preserva bordes, reduce ruido de papel escaneado.
    result = cv2.bilateralFilter(result, d=3, sigmaColor=30, sigmaSpace=30)

    return result





# ─── OCR principal ───────────────────────────────────────────────
# ─── OCR optimizado: equilibrio velocidad/detección ──
# Canvas size 2500px: un poco mas grande que antes (2000) para no perder
# texto pequeño en burbujas de manga, pero sin llegar al original (4096+)
# que saturaba EasyOCR.
_OCR_CANVAS_SIZE: int = 2500
# Thresholds ajustados para manga denso: más sensibles para capturar
# texto pequeño en burbujas, onomatopeyas y diálogos fragmentados.
_OCR_TEXT_THRESHOLD: float = 0.15
_OCR_LOW_TEXT: float = 0.10
_OCR_MIN_SIZE: int = 6
# Factor de upscaling interno de EasyOCR. 1.3 mejora detección de
# texto muy pequeño (manga) sin saturar con ruido de fondo.
_OCR_MAG_RATIO: float = 1.3


def _run_ocr_on_image(
    reader: Any,
    img_bgr: _Img,
    mag_ratio: float | None = None,
    rotation_info: list[str] | tuple[str, ...] | None = None,
) -> list[Any]:
    """
    Ejecuta EasyOCR sobre una imagen y retorna resultados crudos.
    Adquiere el semáforo OCR (solo una llamada a la vez).

    Args:
        mag_ratio: Factor de upscaling. Si es None, usa _OCR_MAG_RATIO (1.3).
                   Valores mas altos (1.5-2.0) mejoran deteccion de texto
                   artistico pero anaden mas ruido.
        rotation_info: Fase 6 — ángulos de rotación que EasyOCR prueba y elige
            el de mejor confianza. None = comportamiento actual (sin rotación;
            el tier 1 de página completa NO lo pasa para no multiplicar ~4x el
            tiempo del camino caliente). Solo la Ruta C (crops de regiones)
            lo pasa (EASYOCR_ROTATION_INFO), donde vive el texto vertical /
            estilizado que YOLO detecta como región. Con rotation_info, EasyOCR
            devuelve las cajas ya en coordenadas del crop ORIGINAL (las rota
            internamente), así que el mapeo ÷upscale de la Ruta C no cambia.
    """
    mag = mag_ratio if mag_ratio is not None else _OCR_MAG_RATIO
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    acquired = _ocr_semaphore.acquire(blocking=True, timeout=120)
    try:
        # v4.2 race-window fix: si el daemon U-OCR empezó a inferir DESPUÉS de
        # que _detect_and_ocr chequeara el flag (check-then-act), este worker ya
        # llegó aquí. En vez de bloquearse en _gpu_lock sin timeout durante toda
        # la inferencia VLM (2-8 min) reteniendo el semáforo OCR y provocando
        # timeouts de 120s en los demás workers, degradar a RapidOCR CPU.
        if _uocr_inferring.is_set():
            print("[OCR] Daemon U-OCR infiriendo (race window): "
                  "degradando a RapidOCR CPU")
            img_rapid_cpu = _preprocess_rapid(img_bgr)
            return _run_rapidocr(img_rapid_cpu)
        # v4.2: adquirir el lock GPU para no competir por VRAM con el daemon
        # U-OCR si está infiriendo (un solo motor GPU a la vez).
        with _gpu_lock:
            kwargs: dict[str, Any] = {
                "detail": 1,
                "paragraph": False,
                "min_size": _OCR_MIN_SIZE,
                "text_threshold": _OCR_TEXT_THRESHOLD,
                "low_text": _OCR_LOW_TEXT,
                "link_threshold": 0.3,
                "canvas_size": min(max(img_bgr.shape[:2]), _OCR_CANVAS_SIZE),
                "mag_ratio": mag,
            }
            if rotation_info is not None:
                kwargs["rotation_info"] = list(rotation_info)
            return reader.readtext(img_rgb, **kwargs)
    except Exception as e:
        print(f"[OCR] Error en readtext: {e}")
        return []
    finally:
        if acquired:
            _ocr_semaphore.release()


def _ocr_results_to_blocks(results: list[Any], img_bgr: _Img) -> list[dict[str, Any]]:
    """Convierte resultados crudos de OCR al formato interno de bloques.

    Acepta AMBOS formatos, porque _run_ocr_on_image puede devolver tuplas
    crudas de EasyOCR `(bbox, text, conf)` O dicts en formato interno
    (degradación v4.2: si el daemon U-OCR empezó a inferir en la race
    window entre el check de _detect_and_ocr y la ejecución, degrada a
    RapidOCR CPU que devuelve dicts). Antes este camino solo manejaba
    tuplas → ValueError "too many values to unpack" → 500 en las páginas
    19-22 del run fusion batch (Fase 5, 2026-08-04). Mismo tratamiento
    dict/tupla que _recover_regions_with_easyocr (Ruta C).
    """
    blocks: list[dict[str, Any]] = []
    for res in results:
        if isinstance(res, dict):
            # Formato interno (RapidOCR CPU en la race window): bloques ya
            # formateados con x/y/w/h/text/confidence/fontSize/textColor.
            text = str(res.get("text", "")).strip()
            conf = float(res.get("confidence", 0.5))
            # Paridad con el camino tupla: mismo filtro de confianza mínima.
            if not text or conf < 0.08:
                continue
            x = int(res.get("x", 0))
            y = int(res.get("y", 0))
            w = int(res.get("w", 0))
            h = int(res.get("h", 0))
            if w < 3 or h < 3:
                continue
            blocks.append({
                "x": x, "y": y, "w": w, "h": h,
                "text": text,
                "confidence": conf,
                "fontSize": int(res.get("fontSize", 0) or max(8, int(h * 0.75))),
                "textColor": res.get("textColor", "#000000"),
            })
            continue
        bbox, text, conf = res
        text = str(text).strip()
        if not text or conf < 0.08:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        if w < 3 or h < 3:
            continue
        font_size = max(8, int(h * 0.75))
        cx, cy = x + w // 2, y + h // 2
        pad = max(2, h // 6)
        roi = img_bgr[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
        text_color = "#000000"
        if roi.size > 0:
            mean_b = int(np.mean(roi[:, :, 0]))
            mean_g = int(np.mean(roi[:, :, 1]))
            mean_r = int(np.mean(roi[:, :, 2]))
            brightness = mean_r * 0.299 + mean_g * 0.587 + mean_b * 0.114
            text_color = "#000000" if brightness > 128 else "#ffffff"

        blocks.append({
            "x": x, "y": y, "w": w, "h": h,
            "text": text,
            "confidence": float(conf),
            "fontSize": font_size,
            "textColor": text_color,
        })

    return _group_and_merge_blocks(blocks, img_bgr.shape[0])


def _detect_and_ocr(
    img_bgr: _Img,
    lang_hint: str = "auto",
    allow_fallback: bool = True,
    prefilter: bool = True,
    use_hybrid: bool = True,
    avg_conf_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """
    OCR con pipeline HÍBRIDO de 3 niveles EasyOCR + RapidOCR:

    1. **EasyOCR directo** (GPU, ~1.16s) — rápido para texto normal.
       Si prefilter=True, aplica limpieza morfológica antes.
    2. **CLAHE+sharpen -> EasyOCR** (~0.3s extra) — fallback para
       texto artístico/decorativo que EasyOCR no captura directamente.
    3. **RapidOCR** (CPU ONNX, ~2.4s) — fallback final usando los
       mismos modelos PP-OCRv4 que PaddleOCR pero sin conflictos
       de PaddlePaddle. Se activa solo cuando:
       - EasyOCR devuelve 0 bloques (todo falló), O
       - La confianza promedio de EasyOCR es < avg_conf_threshold
         (texto artístico detectado débilmente)

    Si ambos OCRs devuelven bloques y la confianza de EasyOCR es baja,
    los resultados se fusionan con _fusionar_blocks(): por cada texto
    detectado por ambos, se queda con el de mayor confianza.

    Args:
        allow_fallback: Si False, solo ejecuta tier 1 (EasyOCR directo).
        prefilter: Si True, aplica _pre_filter_image antes del tier 1.
        use_hybrid: Si True, activa tier 3 (RapidOCR) como fallback.
        avg_conf_threshold: Si la confianza promedio de EasyOCR es
            menor a este valor, se activa el tier hibrido (RapidOCR + fusión).
    """
    # ── Degradación v4.2: daemon U-OCR infiriendo → solo RapidOCR CPU ──
    # Si el daemon (proceso separado) está usando la GTX ahora mismo, NO
    # cargar/ejecutar EasyOCR GPU: degradar a RapidOCR CPU para no competir
    # por VRAM (la inferencia VLM del daemon tarda 2-8 min; esperar el
    # _gpu_lock bloquearía a todos los workers). El flag lo setea
    # _ocr_with_unlimited. IMPORTANTE: se chequea ANTES de _get_ocr_reader()
    # — cargar el reader cargaría los modelos de EasyOCR a VRAM mientras el
    # daemon infiere, que es exactamente la contención que v4.2 elimina.
    if _uocr_inferring.is_set():
        print("[OCR] Daemon U-OCR infiriendo: degradando a RapidOCR CPU "
              "para liberar la GTX")
        img_rapid_cpu = _preprocess_rapid(img_bgr)
        return _run_rapidocr(img_rapid_cpu)

    reader = _get_ocr_reader(lang_hint)
    if reader is None:
        return []

    # ── Pre-filter opcional (antes de tier 1) ────────────────────
    img_ocr = _pre_filter_image(img_bgr) if prefilter else img_bgr

    # ── Intento 1: EasyOCR directo ───────────────────────────────
    results = _run_ocr_on_image(reader, img_ocr)
    blocks_easy = _ocr_results_to_blocks(results, img_ocr)

    if blocks_easy:
        avg_conf = float(np.mean([b.get("confidence", 0) for b in blocks_easy]))
        print(f"[OCR] EasyOCR: {len(blocks_easy)} bloques (conf={avg_conf:.2f})")

        # ── mag_ratio adaptativo: si confianza baja, reintentar con mas resolucion ──
        # Esto corrige errores en tipografia artistica (portadas, titulos decorativos)
        # donde mag_ratio=1.3 no da suficiente detalle. Al aumentar a 1.8, EasyOCR
        # puede distinguir mejor caracteres como V vs Y, R vs Y, etc.
        if avg_conf < avg_conf_threshold:
            print(f"[OCR] Confianza baja ({avg_conf:.2f} < {avg_conf_threshold}). "
                  f"Reintentando con mag_ratio=1.8...")
            results_high = _run_ocr_on_image(reader, img_ocr, mag_ratio=1.8)
            blocks_high = _ocr_results_to_blocks(results_high, img_ocr)
            if blocks_high:
                avg_conf_high = float(np.mean([b.get("confidence", 0) for b in blocks_high]))
                print(f"[OCR] mag_ratio=1.8: {len(blocks_high)} bloques (conf={avg_conf_high:.2f})")
                # Usar el resultado de mayor mag_ratio si tiene mejor confianza O mas bloques
                if avg_conf_high > avg_conf or len(blocks_high) > len(blocks_easy):
                    blocks_easy = blocks_high
                    avg_conf = avg_conf_high

    # ── Híbrido: Ejecutar RapidOCR siempre para capturar texto estilizado/títulos ──
    if use_hybrid:
        print("[OCR] Ejecutando RapidOCR para complementar texto estilizado y títulos...")
        img_rapid = _preprocess_rapid(img_bgr)
        rapid_blocks = _run_rapidocr(img_rapid)

        if rapid_blocks:
            merged = _fusionar_blocks(blocks_easy, rapid_blocks)
            print(f"[OCR] Híbrido EasyOCR + RapidOCR: {len(merged)} bloques totales "
                  f"(Easy={len(blocks_easy)}, Rapid={len(rapid_blocks)})")
            return merged

    if blocks_easy:
        return blocks_easy

    if not allow_fallback:
        return []

    # ── Intento 2: Pre-filter + CLAHE + sharpen ──────────────────
    if not prefilter:
        print("[OCR] 0 bloques con EasyOCR. Probando pre-filter + CLAHE+sharpen...")
        img_filtered = _pre_filter_image(img_bgr)
    else:
        print("[OCR] 0 bloques con EasyOCR (pre-filter ya aplicado). Probando CLAHE+sharpen...")
        img_filtered = img_ocr
    img_enhanced = _preprocess_enhanced(img_filtered)
    results2 = _run_ocr_on_image(reader, img_enhanced)
    blocks2 = _ocr_results_to_blocks(results2, img_enhanced)

    if blocks2:
        print(f"[OCR] Pre-filter+CLAHE detecto {len(blocks2)} bloques!")
        return blocks2

    print("[OCR] Todos los fallbacks agotados")
    return []


# ─── Ruta C: re-OCR a nivel de globo dentro de paneles image ──
# El benchmark de la Ruta C (PLAN_FUSION_OCR.md §3.5) demostró que recortar
# el panel image COMPLETO y re-OCRearlo NO recupera el diálogo — ni con
# EasyOCR (hasta 3x) ni con U-OCR. La granularidad del recorte es CRÍTICA:
# el globo individual (411x245 en pág. 12, roundness 0.739) SÍ se recupera
# con re-OCR (conf 0.96). Estas funciones implementan esa granularidad.


def _detect_bubble_regions_in_panel(
    img_bgr: _Img,
    panel: dict[str, Any],
    min_area_ratio: float = 0.004,
    max_area_ratio: float = 0.60,
) -> list[dict[str, Any]]:
    """Detecta globos/regiones de texto dentro de un panel image.

    Heurística OpenCV validada en analizar_dialogo_artistico.py (págs. 3/12):
      - Blobs de luminancia >200 (interior de globo) con morfología de cierre.
      - Roundness real 4π·area/perímetro² ≥ 0.30 (elíptico, no línea).
      - Borde oscuro definido alrededor (border_ratio ≥ 0.08).
      - Interior con tinta (dark_ratio > 0.02).

    Args:
        panel: dict con x, y, w, h en coordenadas de PÁGINA.

    Returns:
        Regiones en coordenadas de PÁGINA: [{x, y, w, h, roundness, ...}].
    """
    h_page, w_page = img_bgr.shape[:2]
    px, py, pw, ph = int(panel["x"]), int(panel["y"]), int(panel["w"]), int(panel["h"])
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(w_page, px + pw), min(h_page, py + ph)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    num, labels, stats, _cents = cv2.connectedComponentsWithStats(bright)
    min_area = (h * w) * min_area_ratio
    max_area = (h * w) * max_area_ratio
    regions: list[dict[str, Any]] = []
    for i in range(1, num):
        bx, by, bw, bh, area = stats[i]
        if area < min_area or area > max_area or bw < 40 or bh < 25:
            continue
        comp = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        roundness = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        if roundness < 0.30:  # blob demasiado irregular para ser globo
            continue
        mask_region = bright[by:by + bh, bx:bx + bw]
        border_mask = cv2.dilate(mask_region, np.ones((5, 5), np.uint8)) ^ mask_region
        border_hits = cv2.countNonZero(cv2.bitwise_and(edges[by:by + bh, bx:bx + bw], border_mask))
        border_ratio = border_hits / max(cv2.countNonZero(border_mask), 1)
        if border_ratio < 0.08:
            continue
        interior = gray[by:by + bh, bx:bx + bw]
        dark_ratio = float(np.mean(interior[mask_region.astype(bool)] < 100))
        if dark_ratio < 0.02:  # interior sin tinta → no hay texto
            continue
        regions.append({
            "x": px + int(bx), "y": py + int(by),
            "w": int(bw), "h": int(bh),
            "roundness": round(float(roundness), 3),
            "border_ratio": round(float(border_ratio), 2),
            "dark_ratio": round(float(dark_ratio), 3),
        })
    return regions


def _classify_rotate_crop(up_img: _Img) -> tuple[_Img, bool, float]:
    """Clasifica la rotación (0°/180°) de un crop de texto con el Cls de
    RapidOCR (Fase 3 punto 3) y lo rota si es necesario.

    El TextClassifier de RapidOCR (PP-OCRv4 cls, ONNX CPU) devuelve por cada
    imagen un label ("0" o "180") + score. La librería YA rota internamente
    la imagen cuando detecta "180" con score > cls_thresh (0.9): solo hay
    que devolver la imagen ya rotada.

    Args:
        up_img: crop upscaleado (BGR).

    Returns:
        (img_rotada, se_roto, score): la imagen (rotada si procede), si se
        aplicó la rotación, y la confianza del clasificador. Degradación
        segura: si el engine no está o falla, devuelve (up_img, False, 0.0)
        — la Ruta C sigue con el crop original sin romper.
    """
    from config import RUTA_C_CLS_ENABLED, RUTA_C_CLS_THRESH
    if not RUTA_C_CLS_ENABLED:
        return up_img, False, 0.0
    # Benchmark: disable_uocr también apaga el cls (mide el overhead puro de
    # la fusión sin la clasificación de rotación, igual que sin el VLM).
    if _ruta_c_cls_disabled.is_set():
        return up_img, False, 0.0
    engine = _get_rapid_engine()
    if engine is None or getattr(engine, "text_cls", None) is None:
        return up_img, False, 0.0
    acquired = _rapid_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        print("[OCR] Timeout adquiriendo semaforo RapidOCR (cls, 120s)")
        return up_img, False, 0.0
    try:
        # engine.text_cls() devuelve (imgs_rotadas, cls_res, elapse); las
        # imágenes detectadas como 180° con score > thresh ya vienen rotadas.
        rotated_list, cls_res, _elapse = engine.text_cls([up_img])
        if not rotated_list or not cls_res:
            return up_img, False, 0.0
        label = str(cls_res[0][0]) if isinstance(cls_res[0], (list, tuple)) else ""
        score = float(cls_res[0][1]) if isinstance(cls_res[0], (list, tuple)) else 0.0
        if "180" in label and score > RUTA_C_CLS_THRESH:
            return rotated_list[0], True, score
        return up_img, False, score
    except Exception as e:
        print(f"[OCR] TextClassifier falló ({e}); usando crop sin rotar")
        return up_img, False, 0.0
    finally:
        _rapid_semaphore.release()


def _recover_regions_with_easyocr(
    img_bgr: _Img,
    regions: list[dict[str, Any]],
    lang_hint: str = "es",
    upscale: float = 3.5,
) -> list[dict[str, Any]]:
    """Re-OCR de regiones de texto/globos a nivel individual (Ruta C).

    Para cada región (coordenadas de página):
      1. Recortar con padding.
      2. Upscale 3-4× (INTER_CUBIC) — la granularidad que el benchmark
         demostró necesaria para el diálogo artístico.
      3. OCR sobre el recorte upscaleado.
      4. Mapear bloques de vuelta a coordenadas de página (÷ upscale).

    Motor (§8.4.4): EasyOCR GPU por defecto; pero si el daemon U-OCR está
    infiriendo (_uocr_inferring activo, v4.2), NO se carga EasyOCR a VRAM y
    se degrada a **RapidOCR CPU** sobre el crop upscaleado — así la Ruta C
    no compite por la GTX con la inferencia VLM en curso (un solo motor GPU
    a la vez). El chequeo ocurre ANTES de _get_ocr_reader(), igual que en
    _detect_and_ocr.

    Returns:
        Bloques en formato interno ({x, y, w, h, text, confidence, fontSize}).
    """
    # §8.4.4: si el daemon U-OCR está infiriendo, degradar a RapidOCR CPU.
    use_rapid = _uocr_inferring.is_set()
    reader = None if use_rapid else _get_ocr_reader(lang_hint)
    if reader is None and not use_rapid:
        return []
    if not regions:
        return []
    # Fase 6: rotation_info (0/90/180/270) para recuperar títulos verticales
    # y cartelas rotadas que YOLO detecta como región. Fuera del loop (code
    # review): import cacheado pero feo dentro de cada región.
    from config import EASYOCR_ROTATION_INFO
    if use_rapid:
        print("[OCR] Ruta C degradada a RapidOCR CPU "
              "(daemon U-OCR infiriendo)")
    h_page, w_page = img_bgr.shape[:2]
    recovered: list[dict[str, Any]] = []
    for r in regions:
        pad = max(6, int(min(r["w"], r["h"]) * 0.06))
        x0, y0 = max(0, r["x"] - pad), max(0, r["y"] - pad)
        x1, y1 = min(w_page, r["x"] + r["w"] + pad), min(h_page, r["y"] + r["h"] + pad)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        up_w, up_h = int(crop.shape[1] * upscale), int(crop.shape[0] * upscale)
        if up_w > 8000 or up_h > 8000:
            continue
        up_img = cv2.resize(crop, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        if use_rapid:
            # Degradación §8.4.4: RapidOCR CPU sobre el crop upscaleado.
            # _run_rapidocr ya devuelve bloques en formato interno.
            rapid_blocks = _run_rapidocr(up_img)
            for rb in rapid_blocks:
                text = str(rb.get("text", "")).strip()
                if not text:
                    continue
                bx = x0 + int(rb.get("x", 0) / upscale)
                by = y0 + int(rb.get("y", 0) / upscale)
                bw = int(rb.get("w", 0) / upscale)
                bh = int(rb.get("h", 0) / upscale)
                if bw < 3 or bh < 3:
                    continue
                recovered.append({
                    "x": bx, "y": by, "w": bw, "h": bh,
                    "text": text,
                    "confidence": float(rb.get("confidence", 0.5)),
                    "fontSize": max(8, int(bh * 0.75)),
                    "textColor": rb.get("textColor", "#000000"),
                    "engine": "rapidocr-region",
                })
        else:
            # Fase 3 punto 3: TextClassifier de RapidOCR (Cls PP-OCRv4) —
            # detecta si el globo está rotado 180° y lo rota ANTES del re-OCR
            # (EasyOCR no detecta texto girado; el bloque rotado se pierde).
            up_img_ocr, se_roto, cls_score = _classify_rotate_crop(up_img)
            if se_roto:
                print(f"[OCR] Ruta C: globo rotado 180° "
                      f"(score {cls_score:.2f}) — corregido")
            # Fase 6: rotation_info — EasyOCR prueba 0/90/180/270 y elige el
            # de mejor confianza. Recupera títulos verticales (tategaki) y
            # cartelas rotadas que YOLO detecta como región. EasyOCR devuelve
            # las cajas en coords del crop ORIGINAL (rota internamente), así
            # que el mapeo ÷upscale posterior no cambia. Solo se aplica en los
            # CROPS de la Ruta C, no en la página completa (costo ~4x).
            results = _run_ocr_on_image(
                reader, up_img_ocr, rotation_info=EASYOCR_ROTATION_INFO)
            for res in results:
                # Normalizar ambos formatos: si _run_ocr_on_image degradó
                # internamente a RapidOCR (race window: el daemon empezó a
                # inferir justo después del chequeo de _uocr_inferring al
                # inicio de esta función), devuelve bloques en formato interno
                # (dicts) en vez de tuplas crudas (bbox, text, conf). Tratar
                # ambos para no perder la recuperación silenciosamente.
                if isinstance(res, dict):
                    text = str(res.get("text", "")).strip()
                    if not text:
                        continue
                    bx_u = int(res.get("x", 0))
                    by_u = int(res.get("y", 0))
                    bw_u = int(res.get("w", 0))
                    bh_u = int(res.get("h", 0))
                    conf = float(res.get("confidence", 0.5))
                    color = res.get("textColor", "#000000")
                else:
                    # Formato crudo estándar de EasyOCR: (bbox, text, conf)
                    bbox, text, conf = res
                    text = str(text).strip()
                    if not text or conf < 0.10:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    bx_u = int(min(xs))
                    by_u = int(min(ys))
                    bw_u = int(max(xs) - min(xs))
                    bh_u = int(max(ys) - min(ys))
                    color = "#000000"
                # Si el cls rotó el crop 180°, las coords del bloque están en
                # el espacio rotado — des-rotarlas para mapear al crop original.
                if se_roto:
                    up_w = int(up_img.shape[1])
                    up_h = int(up_img.shape[0])
                    bx_u = up_w - bx_u - bw_u
                    by_u = up_h - by_u - bh_u
                bx = x0 + int(bx_u / upscale)
                by = y0 + int(by_u / upscale)
                bw = int(bw_u / upscale)
                bh = int(bh_u / upscale)
                if bw < 3 or bh < 3:
                    continue
                recovered.append({
                    "x": bx, "y": by, "w": bw, "h": bh,
                    "text": text,
                    "confidence": float(conf),
                    "fontSize": max(8, int(bh * 0.75)),
                    "textColor": color,
                    "engine": "easyocr-region",
                })
    if not recovered:
        return []
    return _group_and_merge_blocks(recovered, h_page)


def _page_dark_features(img_bgr: _Img) -> tuple[float, np.ndarray] | None:
    """Extrae features de oscuridad de una página (barato, ~20ms).

    Downscale a 300px, convierte a gris y cuenta el ratio de píxeles oscuros
    (<120: arte/tinta vs papel escaneado 220-255). Compartido por
    _page_has_large_image_panel (trigger v4.2) y _page_signature (cache §8.4.1)
    para que ambos usen EXACTAMENTE el mismo cómputo.

    Returns:
        (dark_ratio, gray_small) o None si la imagen no es procesable.
    """
    try:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, 300.0 / max(h, w))
        small = cv2.resize(
            img_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(gray < 120))
        return dark_ratio, gray
    except Exception:
        return None


def _page_signature(img_bgr: _Img, grid: int = 8, cell_dark_ratio: float = 0.05) -> str:
    """Firma del LAYOUT de una página por distribución espacial de oscuridad.

    Divide la página downscaleada (via _page_dark_features) en una cuadrícula
    grid×grid y marca qué celdas son "oscuras" (arte/tinta dominante). El hash
    binario resultante identifica páginas con el MISMO layout — la base del
    cache de decisiones U-OCR (§8.4.1): si una página con esta firma ya disparó
    el refuerzo y no recuperó nada, las páginas repetitivas del capítulo con la
    misma firma no deben volver a disparar la inferencia VLM.

    Diseño: la cuadrícula captura la ESTRUCTURA (paneles, márgenes, cabeceras)
    y es robusta a cambios de texto dentro de los globos — dos páginas del mismo
    capítulo con diálogo distinto comparten firma; dos layouts distintos no.

    Args:
        img_bgr: Página en BGR.
        grid: Celdas por lado (8 → 64 bits de firma).
        cell_dark_ratio: Fracción de píxeles oscuros para considerar una celda
            "oscura" (0.05 calibrado en el PDF real: paneles con arte marcan
            la celda; texto suelto no).

    Returns:
        str: firma "<dark_ratio 1 decimal>:<bits hex>" o "" si no procesable.
        El dark_ratio se CUANTIZA a 1 decimal para que páginas casi idénticas
        (mismo layout, mínima variación de tinta) compartan firma.
    """
    feats = _page_dark_features(img_bgr)
    if feats is None:
        return ""
    dark_ratio, gray = feats
    sh, sw = gray.shape
    dark = (gray < 120).astype(np.uint8)
    bits = 0
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = int(gy * sh / grid), int((gy + 1) * sh / grid)
            x0, x1 = int(gx * sw / grid), int((gx + 1) * sw / grid)
            cell = dark[y0:y1, x0:x1]
            if cell.size and float(cell.mean()) > cell_dark_ratio:
                bits |= 1 << (gy * grid + gx)
    return f"{dark_ratio:.1f}:{bits:0{grid * grid}x}"


def _page_has_large_image_panel(img_bgr: _Img, min_ratio: float = 0.15) -> bool:
    """Heurística v4.2: ¿la página contiene un panel image grande (>min_ratio)?

    Barata (~20ms): downscale a 300px, umbral de luminancia (<120 = arte oscuro,
    no papel blanco) y cuenta el ratio de píxeles oscuros. Un panel image grande
    (ilustración/arte) domina la página; una página normal de diálogo tiene el
    ratio de oscuridad bajo. Se usa en el trigger de U-OCR: solo se dispara el
    refuerzo si hay panel image grande O <3 bloques con confianza <0.2.

    Args:
        img_bgr: Página en BGR.
        min_ratio: Fracción mínima de área oscura para considerarla panel grande.

    Returns:
        True si la página parece tener un panel image grande.
    """
    feats = _page_dark_features(img_bgr)
    if feats is None:
        return False
    dark_ratio, _ = feats
    # Una página normal de manga: paneles con texto sobre papel blanco →
    # dark_ratio < 10%. Una página con panel image grande (ilustración o
    # portada) → dark_ratio > 25%. Umbral adaptativo con histéresis.
    return dark_ratio > 0.18


# ─── Filtro de marcas de agua ──────────────────────────────────
def _filter_watermarks_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []
    result: list[dict[str, Any]] = []
    for b in blocks:
        text = str(b.get("text", "")).strip()
        if any(p.search(text) for p in WATERMARK_PATTERNS):
            print(f"[filter-watermark] Bloque filtrado: '{text}'")
            continue
        result.append(b)
    return result


# ─── Agrupación y fusión de bloques ────────────────────────────
def _group_and_merge_blocks(blocks: list[dict[str, Any]], img_h: int | None = None) -> list[dict[str, Any]]:
    if not blocks:
        return []

    margin_top: float | None = img_h * 0.085 if img_h else None
    margin_bottom: float | None = img_h * 0.955 if img_h else None

    pre_filtered: list[dict[str, Any]] = []
    for b in blocks:
        text_raw = str(b["text"]).strip()
        h = b["h"]
        cy = b["y"] + h / 2

        # ── Filtros contra texto ORIGINAL (antes de limpiar simbolos) ──
        # IMPORTANTE: Los patrones de ruido (fecha, hora: "13/7/26", "4.58 p.m")
        # dependen de / y . que la limpieza de simbolos elimina. Verificar
        # contra text_raw preserva estos caracteres.
        if any(p.search(text_raw) for p in WATERMARK_PATTERNS):
            print(f"[OCR] Filtrando marca de agua pre-merge: '{text_raw[:50]}'")
            continue

        in_margin = margin_top is not None and (cy < margin_top or cy > margin_bottom)
        if in_margin:
            if any(p.search(text_raw) for p in MARGIN_NOISE_PATTERNS):
                print(f"[OCR] Filtrando metadato de margen pre-merge: '{text_raw[:50]}' en y={b['y']}")
                continue
            digit_count = sum(1 for c in text_raw if c.isdigit())
            if len(text_raw) > 0 and (digit_count / len(text_raw) >= 0.35) and len(text_raw.split()) <= 4:
                print(f"[OCR] Filtrando metadato numérico de margen pre-merge: '{text_raw[:50]}' en y={b['y']}")
                continue

        if re.search(r'https?://|www\.|\.(com|net|org|xyz|io)\b', text_raw, re.IGNORECASE):
            print(f"[OCR] Filtrando URL pre-merge: '{text_raw}'")
            continue

        # ── Ahora corregir errores de OCR con glosario, limpiar simbolos y spellcheck ──
        from translator import _aplicar_glosario
        text_corr = _aplicar_glosario(text_raw)
        text = re.sub(r'[@#$%^&*()+={}\[\]|:;<>/\\]', '', text_corr).strip()
        # Corrector ortografico post-OCR: corrige errores como V->Y, R->Y, etc.
        # SIN depender de GLOSARIO_PRE. Funciona para cualquier PDF.
        text = _ocr_spellcheck(text)
        b["text"] = text if text else text_corr

        pre_filtered.append(b)

    if not pre_filtered:
        return []

    sorted_b = sorted(pre_filtered, key=lambda b: (b["y"] + b["h"] / 2, b["x"]))
    merged: list[dict[str, Any]] = []
    used = [False] * len(sorted_b)

    for i, b in enumerate(sorted_b):
        if used[i]:
            continue
        group = [b]
        used[i] = True
        for j, b2 in enumerate(sorted_b):
            if used[j] or i == j:
                continue
            group_x2 = max(g["x"] + g["w"] for g in group)
            cy1 = group[-1]["y"] + group[-1]["h"] / 2
            cy2 = b2["y"] + b2["h"] / 2
            max_h = max(group[-1]["h"], b2["h"])
            gap_x = b2["x"] - group_x2

            if abs(cy1 - cy2) < max_h * 0.45 and -b2["w"] < gap_x < max(35, group[-1]["w"] * 2.5):
                group.append(b2)
                used[j] = True

        if len(group) == 1:
            merged.append(b)
        else:
            group.sort(key=lambda g: g["x"])
            all_x = [g["x"] for g in group]
            all_y = [g["y"] for g in group]
            all_x2 = [g["x"] + g["w"] for g in group]
            all_y2 = [g["y"] + g["h"] for g in group]
            mx = min(all_x)
            my = min(all_y)
            mw = max(all_x2) - mx
            mh = max(all_y2) - my
            merged.append({
                "x": mx, "y": my, "w": mw, "h": mh,
                "text": " ".join(g["text"] for g in group),
                "confidence": float(np.mean([g["confidence"] for g in group])),
                "fontSize": max(g["fontSize"] for g in group),
                "textColor": group[0]["textColor"],
            })

    # ─── Segunda pasada: fusión vertical (líneas de mismo globo) ───
    # Combina bloques en la misma columna y cerca verticalmente
    if len(merged) > 1:
        sorted_v = sorted(merged, key=lambda b: (b["x"], b["y"]))
        v_merged: list[dict[str, Any]] = []
        v_used = [False] * len(sorted_v)
        for i, b in enumerate(sorted_v):
            if v_used[i]:
                continue
            v_group = [b]
            v_used[i] = True
            for j, b2 in enumerate(sorted_v):
                if v_used[j] or i == j:
                    continue
                # No fusionar verticalmente cabeceras del margen superior (y < 2.5% img_h) con contenido
                if img_h and (b["y"] < img_h * 0.025) != (b2["y"] < img_h * 0.025):
                    continue
                # Mismo columna (x overlap significativo)
                x_overlap = min(b["x"] + b["w"], b2["x"] + b2["w"]) - max(b["x"], b2["x"])
                min_w = min(b["w"], b2["w"])
                if x_overlap < min_w * 0.5:
                    continue
                # Cerca verticalmente (gap < 1.5x altura)
                gap_y = abs(b2["y"] - (b["y"] + b["h"]))
                if gap_y < max(b["h"], b2["h"]) * 1.5:
                    v_group.append(b2)
                    v_used[j] = True
            if len(v_group) > 1:
                v_group.sort(key=lambda g: g["y"])
                all_x = [g["x"] for g in v_group]
                all_y = [g["y"] for g in v_group]
                all_x2 = [g["x"] + g["w"] for g in v_group]
                all_y2 = [g["y"] + g["h"] for g in v_group]
                mx = min(all_x)
                my = min(all_y)
                mw = max(all_x2) - mx
                mh = max(all_y2) - my
                v_merged.append({
                    "x": mx, "y": my, "w": mw, "h": mh,
                    "text": " ".join(g["text"] for g in v_group),
                    "confidence": float(np.mean([g["confidence"] for g in v_group])),
                    "fontSize": max(g["fontSize"] for g in v_group),
                    "textColor": v_group[0]["textColor"],
                })
            else:
                v_merged.append(b)
        merged = v_merged

    final_blocks: list[dict[str, Any]] = []
    for b in merged:
        w, h = b["w"], b["h"]
        text = str(b["text"]).strip()
        text_len = len(text)
        conf = float(b.get("confidence", 0))

        if text_len == 0:
            continue

        if re.match(r'^\d+$', text):
            print(f"[OCR] Filtrando número puro post-merge: '{text}' en [{b['x']},{b['y']}]")
            continue

        if re.match(r'^[\d\s.,;:!?%\'\"\-–—/\\]+$', text):
            print(f"[OCR] Filtrando patrón numérico post-merge: '{text}'")
            continue

        if re.match(r'^["\']+\d', text) and text_len <= 6:
            print(f"[OCR] Filtrando comilla+número post-merge: '{text}'")
            continue

        if re.match(r'^[~\'\"\-–—:;,.!¡¿?=]+$', text) and text_len <= 2:
            print(f"[OCR] Filtrando puntuación suelta post-merge: '{text}'")
            continue

        aspect = w / max(h, 1)
        if aspect < 0.4 and text_len <= 3:
            print(f"[OCR] Filtrando ruido estrecho post-merge: '{text}' aspect={aspect:.2f}")
            continue

        if text_len == 1 and conf < 0.25:
            print(f"[OCR] Filtrando carácter suelto post-merge: '{text}'")
            continue

        if conf < 0.15 and text_len <= 3:
            print(f"[OCR] Filtrando texto corto baja confianza post-merge: '{text}' conf={conf:.2f}")
            continue

        if text_len <= 3 and re.match(r'^\d+[a-zA-Z]?$', text):
            print(f"[OCR] Filtrando combo digito+letra post-merge: '{text}'")
            continue

        final_blocks.append(b)

    return final_blocks


# ─── Detección de globo de diálogo ─────────────────────────────
def _is_inside_speech_bubble(img_bgr: _Img, block: dict[str, Any]) -> bool:
    """Detecta si un bloque está sobre un fondo uniforme (burbuja o panel).

    Un globo de diálogo se caracteriza por bordes de color UNIFORME
    (std < 35 por canal), sin importar si es blanco, negro o coloreado.
    Ya NO controla la estrategia de inpainting (siempre glifos primero).
    Se usa para logging y decisiones de estilo.
    """
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]
    edge = max(3, int(min(bw, bh) * 0.15))

    samples: list[NDArray[np.float64]] = []
    for region_coords in [
        (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),
        (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),
        (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),
        (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),
    ]:
        y1, y2, x1, x2 = region_coords
        if y2 > y1 and x2 > x1:
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                mean_bgr = roi.reshape(-1, 3).mean(axis=0)
                samples.append(mean_bgr)

    if len(samples) < 3:
        return False

    # Uniformidad del borde = burbuja/panel (blanco, negro, o cualquier color)
    samples_arr = np.array(samples)
    std_per_channel = samples_arr.std(axis=0)
    max_std = float(std_per_channel.max())
    return max_std <= 35


# ─── Máscara de glifos para globos de diálogo ──────────────────
def _build_glyph_mask_for_bubble(img_bgr: _Img, block: dict[str, Any]) -> _Img:
    h, w = img_bgr.shape[:2]
    mask: _Img = np.zeros((h, w), dtype=np.uint8)
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

    edge = max(3, int(min(bw, bh) * 0.15))
    bg_pixels: list[_Img] = []
    for region_coords in [
        (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),
        (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),
        (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),
        (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),
    ]:
        y1, y2, x1, x2 = region_coords
        if y2 > y1 and x2 > x1:
            roi = img_bgr[y1:y2, x1:x2]
            bg_pixels.append(roi.reshape(-1, 3))

    pad_x = max(3, int(bw * 0.04))
    pad_y = max(3, int(bh * 0.06))
    rx1 = max(0, bx - pad_x)
    ry1 = max(0, by - pad_y)
    rx2 = min(w, bx + bw + pad_x)
    ry2 = min(h, by + bh + pad_y)

    roi = img_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return mask

    if bg_pixels:
        all_bg = np.concatenate(bg_pixels, axis=0)
        bg_mean = all_bg.mean(axis=0)
    else:
        bg_mean = roi.mean(axis=(0, 1))

    # ── Enfoque híbrido: diferencia de color + Canny ────────────
    # La diferencia de color funciona bien para texto sobre fondo
    # liso (globos de diálogo). Canny captura bordes finos que la
    # diferencia de color puede perder (glifos rotados o artísticos).

    # ── Detección de glifos basada en contraste local ───────────
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    local_median = int(np.median(roi_gray))
    diff_from_median = cv2.absdiff(roi_gray, local_median)
    
    # Píxeles de texto: aquellos que difieren de forma destacada de la mediana del ROI
    glyph_mask_roi = (diff_from_median > 45).astype(np.uint8) * 255

    # Refinar bordes con cierre ligero
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(glyph_mask_roi, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    mask[ry1:ry2, rx1:rx2] = combined
    return mask


# ─── Construcción de máscara de inpainting ─────────────────────
def _build_inpaint_mask(img_bgr: _Img, blocks: list[dict[str, Any]]) -> _Img:
    """Construye máscara de inpainting para borrar texto detectado.

    Estrategia: Usar SIEMPRE máscara de glifos basada en trazos de letra.
    No se utilizan máscaras rectangulares sólidas porque destruyen el arte
    del manga (burbujas, ilustraciones, gradientes).
    """
    h, w = img_bgr.shape[:2]
    mask: _Img = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        glyph_mask = _build_glyph_mask_for_bubble(img_bgr, block)
        glyph_pixels = int(np.sum(glyph_mask > 0))
        if glyph_pixels > 0:
            mask = cv2.bitwise_or(mask, glyph_mask)

    # Dilatación mínima (3x3) para cubrir solo bordes inmediatos de los trazos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ─── Inpainting con OpenCV (TELEA + border-blend) ──────────────
def _inpaint_image(img_bgr: _Img, mask: _Img, blocks: list[dict[str, Any]] | None = None) -> _Img:
    """
    Inpaint 2 fases: TELEA + border-blend.

    Fase 1 — TELEA (Fast Marching Method) con radio adaptativo mejorado:
        - Texto pequeño (<15px): radio 3
        - Texto normal (15-30px): radio 4
        - Texto grande (25-40px): radio 6
        - Texto muy grande (>40px): radio 8

    Fase 2 — Border-blend post-inpainting:
        Para cada bloque, extrae el color MEDIO del borde exterior del
        bloque DESDE LA IMAGEN ORIGINAL y lo mezcla suavemente con el
        relleno de TELEA (alpha=0.35). Esto elimina los parches oscuros
        o irregulares que TELEA deja en fondos complejos (portadas,
        ilustraciones con gradientes, sombras).

    El blend usa máscara dilatada + GaussianBlur para bordes suaves,
    evitando artefactos visuales en la transición.
    """
    if int(mask.max()) == 0:
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]

    # ---- Fase 1: TELEA con radio adaptativo mejorado ----
    if blocks:
        heights = [b["h"] for b in blocks if b.get("h", 0) > 0]
        if heights:
            max_h = max(heights)
            if max_h < 15:
                radius = 3
            elif max_h > 40:
                radius = 8
            elif max_h > 25:
                radius = 6
            elif max_h > 15:
                radius = 4
            else:
                radius = 3
        else:
            radius = 4
    else:
        radius = 4

    print(f"[inpaint] TELEA radio={radius}px + border-blend")
    img_telea = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    # Si no hay bloques, devolver solo TELEA
    if not blocks:
        return img_telea

    # ---- Fase 2: Post-procesamiento con color de borde ----
    # Para cada bloque, rellenar el area de la mascara con el color
    # promedio del borde circundante de la imagen ORIGINAL.
    # Esto evita parches oscuros en fondos complejos (portadas).
    result = img_telea.copy()

    for block in blocks:
        bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

        # Padding para el borde de muestreo (15% del tamano del bloque)
        pad_x = max(5, int(bw * 0.15))
        pad_y = max(5, int(bh * 0.15))

        # Area expandida para muestrear el borde
        y1_exp = max(0, by - pad_y)
        y2_exp = min(h, by + bh + pad_y)
        x1_exp = max(0, bx - pad_x)
        x2_exp = min(w, bx + bw + pad_x)

        # Extraer pixeles del borde (fuera del bloque) de la imagen ORIGINAL
        expanded_roi = img_bgr[y1_exp:y2_exp, x1_exp:x2_exp]
        if expanded_roi.size == 0:
            continue

        # Coordenadas del bloque dentro del ROI expandido
        inner_y1 = by - y1_exp
        inner_x1 = bx - x1_exp
        inner_y2 = inner_y1 + bh
        inner_x2 = inner_x1 + bw

        border_pixels = []
        # Borde superior
        if inner_y1 > 0:
            top_roi = expanded_roi[:inner_y1, :, :]
            if top_roi.size > 0:
                border_pixels.append(top_roi.reshape(-1, 3))
        # Borde inferior
        if inner_y2 < expanded_roi.shape[0]:
            bot_roi = expanded_roi[inner_y2:, :, :]
            if bot_roi.size > 0:
                border_pixels.append(bot_roi.reshape(-1, 3))
        # Borde izquierdo (solo en el rango vertical del bloque)
        if inner_x1 > 0:
            left_roi = expanded_roi[inner_y1:inner_y2, :inner_x1, :]
            if left_roi.size > 0:
                border_pixels.append(left_roi.reshape(-1, 3))
        # Borde derecho
        if inner_x2 < expanded_roi.shape[1]:
            right_roi = expanded_roi[inner_y1:inner_y2, inner_x2:, :]
            if right_roi.size > 0:
                border_pixels.append(right_roi.reshape(-1, 3))

        if not border_pixels:
            continue

        # Calcular el color MEDIO del borde (mediana es mas robusta que la media)
        all_border = np.concatenate(border_pixels, axis=0)
        border_color = np.median(all_border, axis=0).astype(np.uint8)

        # Extraer la region de la mascara para este bloque
        mask_roi = mask[by:min(h, by + bh), bx:min(w, bx + bw)]
        if mask_roi.size == 0 or int(mask_roi.max()) == 0:
            continue

        # Crear mascara de mezcla (dilatar + blur para bordes suaves)
        blend_mask = (mask_roi > 0).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blend_mask = cv2.dilate(blend_mask, k, iterations=2)
        blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), sigmaX=2)

        # Obtener el ROI del resultado para este bloque
        result_roi = result[by:min(h, by + bh), bx:min(w, bx + bw)]
        if result_roi.shape[:2] != blend_mask.shape[:2]:
            continue

        # Crear imagen de color de borde
        border_img = np.full_like(result_roi, border_color)

        # Alpha ADAPTATIVO basado en varianza del borde:
        #   - Baja varianza (fondo uniforme, burbuja blanca): alpha ~0.25
        #     -> preserva mas textura TELEA (el fondo original ya es bueno)
        #   - Alta varianza (fondo complejo, portada, gradiente, sombra): alpha ~0.50
        #     -> mas color de borde para eliminar parches oscuros/irregulares
        # La desviacion estandar del borde es un proxy directo de la
        # complejidad del fondo circundante.
        border_std = float(np.std(all_border))
        # Normalizar: std=80 (fondo muy heterogeneo) -> factor=1.0
        # std=80 es el doble del umbral de _is_inside_speech_bubble (std<=35)
        border_factor = min(1.0, border_std / 80.0)
        adaptive_alpha = 0.25 + 0.25 * border_factor  # rango [0.25, 0.50]
        if adaptive_alpha > 0.35:
            print(f"[inpaint] border-blend alpha={adaptive_alpha:.2f} "
                  f"(std={border_std:.0f}, fondo complejo)")
        alpha = adaptive_alpha * blend_mask[..., np.newaxis]
        blended = (result_roi * (1 - alpha) + border_img * alpha).astype(np.uint8)

        result[by:min(h, by + bh), bx:min(w, bx + bw)] = blended

    return result


# ---- Muestreo de color de fondo ---------------------------------
def _sample_bg_color(img_bgr: _Img, block: dict[str, Any]) -> str:
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

    if _is_inside_speech_bubble(img_bgr, block):
        edge = max(4, int(min(bw, bh) * 0.12))
        samples: list[NDArray[np.float64]] = []
        for region_coords in [
            (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),
            (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),
            (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),
            (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),
        ]:
            y1, y2, x1, x2 = region_coords
            if y2 > y1 and x2 > x1:
                region = img_bgr[y1:y2, x1:x2]
                if region.size > 0:
                    samples.append(np.mean(region.reshape(-1, 3), axis=0))

        if samples:
            mean_bgr = np.mean(samples, axis=0)
            b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
            return f"#{r:02x}{g:02x}{b:02x}"

        return "#000000"

    pad = max(5, int(min(bw, bh) * 0.2))
    samples = []
    y1, y2 = max(0, by - pad), max(0, by)
    if y2 > y1:
        region = img_bgr[y1:y2, max(0, bx):min(w, bx + bw)]
        if region.size > 0:
            samples.append(np.mean(region.reshape(-1, 3), axis=0))
    y1, y2 = min(h, by + bh), min(h, by + bh + pad)
    if y2 > y1:
        region = img_bgr[y1:y2, max(0, bx):min(w, bx + bw)]
        if region.size > 0:
            samples.append(np.mean(region.reshape(-1, 3), axis=0))

    if not samples:
        return "#ffffff"
    mean_bgr = np.mean(samples, axis=0)
    b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"
