"""
ocr_utils.py — OCR (EasyOCR), inpainting (OpenCV), filtros de ruido y muestreo de color.

Extraído de server.py. Depende de config.py para patrones de ruido y constantes.
"""

import base64
import hashlib
import io
import re
import threading
import unicodedata
from typing import Any, Final, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from config import (
    ROOT, MARGIN_NOISE_PATTERNS, WATERMARK_PATTERNS,
    FUSION_TYPE_REINFORCE, FUSION_TYPE_WEIGHTS,
    RUTA_C_RAPID_PRIMARY, RUTA_C_RAPID_MIN_CONF,
    MAX_IMAGE_DECODE_PIXELS,
    RAPID_COND_ENABLED, RAPID_COND_MIN_BLOCKS, RAPID_COND_MIN_CONF,
)
from runtime_diagnostics import configure_torch_determinism

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


def _get_ocr_reader(lang: str = "auto", *, prefer_gpu: bool = True) -> Any:
    """Obtiene un lector EasyOCR cacheado.

    ``prefer_gpu=False`` existe para la recuperación de páginas mixtas: los
    lectores de escritura japonesa/coreana/china se cargan bajo demanda en
    CPU para no duplicar el detector EasyOCR en la GTX de 4 GB. El lector
    principal conserva su comportamiento anterior y sigue priorizando GPU.
    """
    lang = str(lang or "auto").strip().lower()
    lang_key: str = "latin"
    if lang == "ja":
        lang_key = "ja"
    elif lang == "ko":
        lang_key = "ko"
    elif lang in {"zh", "zh-cn", "zh-tw"}:
        lang_key = "zh"

    # El lector GPU principal conserva la clave histórica; las variantes de
    # recuperación CPU tienen una entrada separada y no pisan ese estado.
    cache_key = lang_key if prefer_gpu else f"{lang_key}:cpu"

    global _ocr_readers  # noqa: PLW0603
    if cache_key in _ocr_readers:
        return _ocr_readers[cache_key]

    with _ocr_lock:
        if cache_key in _ocr_readers:
            return _ocr_readers[cache_key]

        langs: list[str] = ["es", "en"]
        if lang_key == "ja":
            langs = ["ja", "en"]
        elif lang_key == "ko":
            langs = ["ko", "en"]
        elif lang_key == "zh":
            langs = ["ch_sim", "en"]
        else:
            # Solo se activan los idiomas latinos soportados actualmente.
            # No cargar fr/de/it evita que sus alfabetos compitan durante la
            # detección automática y reduce el trabajo del recognizer.
            langs = ["es", "en", "pt"]

        # ═══ Carga EasyOCR: GPU si CUDA disponible, CPU si no ═══════
        # El orden de carga es CRÍTICO para evitar conflicto cuDNN:
        #   - Si CT2 carga PRIMERO CUDA/cuDNN, luego PyTorch no puede
        #     cargar sus propios símbolos cuDNN → crash.
        #   - Server.py carga EasyOCR PRIMERO (PyTorch toma GPU, carga cuDNN)
        #     y luego CT2 puede usar GPU sin el conflicto de DLLs.
        #   - La contención de VRAM se controla con locks/semaforos en el
        #     pipeline; force_cpu queda reservado para degradaciones explícitas.
        # ════════════════════════════════════════════════════════════
        print(f"[OCR] Cargando EasyOCR para {langs}...")

        try:
            import easyocr
            # Intentar GPU primero. Si falla (CUDA no disponible, memoria insuficiente),
            # EasyOCR automáticamente cae a CPU via el try/except que sigue.
            gpu_available = bool(prefer_gpu)
            try:
                import torch
                gpu_available = gpu_available and torch.cuda.is_available()
                if gpu_available:
                    configure_torch_determinism(torch)
            except Exception:
                gpu_available = False

            reader = easyocr.Reader(
                langs,
                gpu=gpu_available,
                model_storage_directory=str(ROOT / "ocr_models"),
                download_enabled=True,
                verbose=False,
            )
            _ocr_readers[cache_key] = reader
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
                _ocr_readers[cache_key] = reader
                print(f"[OCR] EasyOCR para {langs} listo en CPU (fallback).")
            except Exception as e2:
                print(f"[OCR] Error cargando EasyOCR incluso en CPU: {e2}")
                return None
    return _ocr_readers[cache_key]


# ─── RapidOCR (lazy load with thread safety) ─────────────────────
# Usa los mismos modelos PP-OCRv4 que PaddleOCR pero via ONNX Runtime,
# sin el conflicto de PaddlePaddle vs PyTorch.
_rapid_engine: Any = None
_rapid_lock: threading.Lock = threading.Lock()


# batch del recognizer de RapidOCR (rec_batch_num). Módulo-global (no Final)
# para que benchmark_rutac_params.py lo parchee en runtime; default 6 = config
# de la librería. Solo afecta a llamadas con MUCHAS líneas de texto por imagen;
# los crops de la Ruta C tienen pocas líneas → sin cambio medible (A/B 2026-08-15).
_RAPID_REC_BATCH_NUM: int = 6


def _get_rapid_engine() -> Any:
    global _rapid_engine
    if _rapid_engine is not None:
        return _rapid_engine
    with _rapid_lock:
        if _rapid_engine is not None:
            return _rapid_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_engine = RapidOCR(rec_batch_num=_RAPID_REC_BATCH_NUM)
            print("[OCR] RapidOCR listo (CPU/ONNX)")
        except Exception as e:
            print(f"[OCR] Error cargando RapidOCR: {e}")
            return None
    return _rapid_engine


def _preprocess_rapid(
    img_bgr: _Img,
    *,
    already_prefiltered: bool = False,
) -> _Img:
    """Preprocesamiento optimizado para RapidOCR (pre-filter + enhance).

    ``_detect_and_ocr`` ya calcula el pre-filtro para EasyOCR. Reutilizarlo
    evita una segunda pasada morfologica sobre la misma pagina; los callers
    que entregan una imagen cruda conservan el comportamiento anterior.
    """
    filtered = img_bgr if already_prefiltered else _pre_filter_image(img_bgr)
    enhanced = _preprocess_enhanced(filtered)
    return enhanced


def _script_language_hints(
    blocks: list[dict[str, Any]],
    *,
    min_conf: float = 0.35,
) -> list[str]:
    """Devuelve idiomas CJK evidenciados por bloques OCR confiables.

    No intenta adivinar todos los idiomas del mundo a partir de píxeles. Usa
    únicamente texto ya recuperado por RapidOCR para decidir si vale la pena
    una segunda pasada especializada. Esto evita cargar lectores extra en
    páginas latinas normales y mantiene el coste acotado en hardware pequeño.
    """
    scores: dict[str, int] = {"ja": 0, "ko": 0, "zh": 0}
    for block in blocks:
        text = str(block.get("text", ""))
        if not text:
            continue
        try:
            confidence = float(block.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_conf:
            continue

        kana = sum(0x3040 <= ord(char) <= 0x30FF for char in text)
        hangul = sum(0xAC00 <= ord(char) <= 0xD7A3 for char in text)
        hanzi = sum(0x4E00 <= ord(char) <= 0x9FFF for char in text)

        # Kana es evidencia fuerte de japonés. Hanzi sin kana queda como
        # chino, igual que la heurística de detección de idioma del traductor.
        if kana:
            scores["ja"] += kana + min(hanzi, 2)
        if hangul >= 2:
            scores["ko"] += hangul
        if hanzi >= 2 and not kana:
            scores["zh"] += hanzi

    return [
        language
        for language, score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
        if score >= 2
    ]


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
            # `type: ignore` SIN código: el error difiere según el entorno — local
            # con ultralytics instalado da attr-defined (el módulo no re-exporta
            # YOLO explícitamente) y el CI sin él da import-not-found. Cualquier
            # código específico quedaría unused en el otro entorno; el ignore
            # genérico suprime ambos y siempre tiene algo que suprimir.
            from ultralytics import YOLO  # type: ignore
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
        from config import MODO_CPU, YOLO_DEVICE
        device = YOLO_DEVICE
        # Preset modo_cpu (soporte sin GPU dedicada): YOLO se fuerza a CPU
        # aunque el torch del proceso detecte CUDA — el modo_cpu quiere 0 VRAM
        # y 0 contención, no aprovechar una GPU que no existe/está débil.
        if MODO_CPU:
            device = "cpu"
        elif device == "auto":
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
    cv2.fillPoly(cast(Any, mask), cast(Any, [shifted.reshape(1, -1, 2).astype(np.int32)]), cast(Any, 1))
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
    filter_page_margins: bool = True,
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
        items: list[tuple[Any, Any, Any]] = []
        if result:
            for r in result:
                try:
                    bbox, text, conf = r
                    items.append((bbox, text, conf))
                except (ValueError, IndexError, TypeError):
                    continue
        blocks = _rapidocr_blocks_from_lines(img_bgr, items)
        # Los crops de la Ruta C no son páginas: sus bordes suelen contener
        # precisamente el texto del globo. No aplicarles filtros de margen.
        page_height = img_bgr.shape[0] if filter_page_margins else None
        return _group_and_merge_blocks(blocks, page_height)
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
        return _block_score(b) * float(b.get("_weight", 1.0))

    # 1. Dedup por texto normalizado idéntico

    # 2. Alineación Levenshtein entre bloques solapados espacialmente
    # El texto repetido en regiones lejanas debe conservar ambas cajas.
    candidates: list[dict[str, Any]] = list(tagged)
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

    # 3. NMS final consciente del texto. El solape espacial por sÃ­ solo no
    # basta: dos lÃ­neas consecutivas pueden compartir una fracciÃ³n grande de
    # su caja cuando un motor devuelve bboxes algo altos. Solo se descarta un
    # candidato si el texto tambiÃ©n es equivalente al de una caja conservada;
    # los duplicados de texto ya fusionados arriba siguen protegidos.
    final_result.sort(key=_score, reverse=True)
    nms: list[dict[str, Any]] = []
    for b in final_result:
        duplicate = False
        for e in nms:
            if _overlap_ratio(b, e) <= 0.40:
                continue
            nb = _normalize_text(b["text"])
            ne = _normalize_text(e["text"])
            max_len = max(len(nb), len(ne))
            if max_len and (nb == ne or
                            _levenshtein(nb, ne) / max_len <= 0.30):
                duplicate = True
                break
        if duplicate:
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
# pyspellchecker trae diccionarios completos de español, inglés y portugués
# para las lenguas activas. No se cargan diccionarios de idiomas pausados.
# Correccion por distancia de Levenshtein (max 2).
# Fallback a _levenshtein() si pyspellchecker no esta instalado.
_OCR_SPELLCHECKER: Any = None
_OCR_SPELL_LOCK: threading.Lock = threading.Lock()
_OCR_FOREIGN_SPELLCHECKERS: dict[str, Any] = {}

# Máximo de caracteres del bloque que se pasan al detector de idioma del
# spellcheck. langdetect escala con la longitud (medido: ~16 ms a 8000 chars)
# y los textos fusionados del merge final pueden superar eso; 600 chars bastan
# para la distribución de n-gramas del idioma (verificado: 0 diferencias en el
# corpus) y además hace que el lru_cache de _detect_language_robust comparta
# entrada entre bloques largos de la misma página.
_SPELL_LANG_MAX_CHARS: Final[int] = 600

# Umbral de longitud para usar la corrección barata propia en vez de
# sp.correction(). pyspellchecker es rápido en palabras cortas (known + edición
# 1 generan pocos strings) pero explota en largas: la expansión de distancia 2
# genera ~4.8 M strings para una palabra de 21 chars (~0.3-1.3 s; calibrado
# 2026-08-15: a partir de len 13 el scan por buckets del diccionario es
# MÁS rápido — len 18: 13.6x, len 21: 1287x — y en cortas es al revés).
_SPELL_CORRECTION_MIN_LEN: Final[int] = 13

# Límite de edición dependiente de la longitud (calibrado 2026-08-15 con el
# A/B del capítulo completo, ver benchmark_spellcheck_ab.py): las 3 correcciones
# reales del capítulo son de 3-5 chars a distancia 1, y NINGUNA corrección real
# usa distancia 2 ni afecta a palabras > 14 chars (la réplica no corrigió nada
# en 8 llamadas >= 13). Schedule refinado por la calibración:
#   - 3-5  chars -> edición 1 (una corrección a distancia 2 en una palabra
#     corta cambia >50 % de los caracteres — riesgo de corrupción alto).
#   - 6-14 chars -> edición 2 (sin cambio; el valor que pyspellchecker usa).
#   - >14  chars -> edición 1, NO 0: la propuesta original (>14 -> 0) pierde
#     correcciones d1 legítimas de palabras largas reales fuera del diccionario
#     (p. ej. 'inconstitucinal' -> 'inconstitucional'); permitir d1 y bloquear
#     SOLO d2 preserva esas correcciones y sigue eliminando las sugerencias
#     arbitrarias de distancia 2 (p. ej. 'chmar' -> 'tomar' a d2, donde la
#     distancia 2 de una palabra larga cae en una palabra de alta frecuencia
#     no relacionada).
_SPELL_EDIT_1_MAX_LEN: Final[int] = 5
_SPELL_EDIT_2_MAX_LEN: Final[int] = 14


def _spell_max_edits(length: int) -> int:
    """Límite de edición del corrector según la longitud de la palabra."""
    if length <= _SPELL_EDIT_1_MAX_LEN:
        return 1
    if length <= _SPELL_EDIT_2_MAX_LEN:
        return 2
    return 1  # >14: solo distancia 1 (bloquea d2, preserva d1)


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


def _get_foreign_spellchecker(lang: str) -> Any:
    """Carga bajo demanda un diccionario latino para detectar mezcla.

    No corrige texto con estos diccionarios: solo sirven como barrera contra
    falsos positivos del corrector español cuando una burbuja contiene una
    frase en inglés o portugués. No agrega modelos OCR/traducción.
    """
    lang = str(lang or "").strip().lower()
    if lang not in {"en", "pt"}:
        return None
    if lang in _OCR_FOREIGN_SPELLCHECKERS:
        return _OCR_FOREIGN_SPELLCHECKERS[lang]
    with _OCR_SPELL_LOCK:
        if lang in _OCR_FOREIGN_SPELLCHECKERS:
            return _OCR_FOREIGN_SPELLCHECKERS[lang]
        try:
            from spellchecker import SpellChecker
            checker = SpellChecker(language=lang)
            _OCR_FOREIGN_SPELLCHECKERS[lang] = checker
            return checker
        except Exception as exc:
            print(f"[OCR-spellcheck] Diccionario {lang} no disponible: {exc}")
            return None


def _contains_foreign_latin_tokens(text: str, spanish_checker: Any,
                                   tokens: list[str] | None = None) -> bool:
    """Detecta tokens de otros idiomas actuales dentro de un bloque latino.

    ``tokens`` (opcional) deja que el caller pase los tokens YA separados,
    sin puntuación y en minúsculas (los que _ocr_spellcheck prepara una sola
    vez para el loop de corrección), evitando repetir el split/strip/-
    lowercase. Cuando es None se tokeniza desde ``text`` (comportamiento
    histórico, el de los tests que llaman solo con texto).
    """
    if spanish_checker is None:
        return False
    if tokens is None:
        tokens = [
            token.strip("'\".,;:!?¡¿()[]{}")
            for token in text.split()
        ]
        tokens = [
            token.lower() for token in tokens
            # Incluye tokens de 2 letras en la deteccion ("go", "my", "je",
            # etc.), aunque el corrector siga sin modificar palabras tan cortas.
            if len(token) > 1 and not any(char.isdigit() for char in token)
        ]
    if not tokens:
        return False

    try:
        spanish_known = set(spanish_checker.known(tokens))
    except Exception:
        return False

    unknown_tokens = [token for token in tokens if token not in spanish_known]
    if not unknown_tokens:
        return False

    for lang in ("en", "pt"):
        checker = _get_foreign_spellchecker(lang)
        if checker is None:
            continue
        # Fast-path por acotación de longitud (misma técnica que el corrector
        # principal): ningún token más largo que la palabra más larga del
        # diccionario extranjero puede estar en él — `known()` es un set
        # lookup exacto (no aproxima por edición), así que pasarlo es trabajo
        # inútil. Se filtran los candidatos a [2, longest_word_length]
        # (mismo rango que _check_if_should_check de pyspellchecker). El
        # isinstance protege los checker mockeados en tests (MagicMock no
        # tiene longest real) y cualquier objeto sin el atributo.
        try:
            longest = checker.word_frequency.longest_word_length
        except Exception:
            longest = 0
        if not isinstance(longest, int) or longest <= 0:
            candidates = list(unknown_tokens)
        else:
            candidates = [t for t in unknown_tokens
                          if 2 <= len(t) <= longest]
        if not candidates:
            continue
        try:
            foreign_known = set(checker.known(candidates))
        except (MemoryError, OverflowError):
            return False
        except Exception:
            continue
        # Un token conocido solo por otro diccionario es suficiente para
        # tratar el bloque como mixto y conservarlo sin correccion agresiva.
        if foreign_known - spanish_known:
            return True
    return False


# Índice del diccionario del spellchecker agrupado por longitud, cacheado por
# instancia (id(sp)). Se usa para acotar los candidatos de corrección: la
# distancia de edición <= 2 implica |len(a) - len(b)| <= 2, así que basta
# revisar las palabras de longitud [len(word)-2, len(word)+2] en vez de las
# ~86k del diccionario completo.
#
# CORRECCIÓN 2026-08-15: se cacheaba en un WeakKeyDictionary, pero SpellChecker
# define __slots__ sin __weakref__ → el índice NUNCA se pudo almacenar con el
# checker real y el fast-path fue inerte en producción (el except del caller
# ponía puede_corregir=True siempre y se seguía pagando sp.correction()
# completo — ver profile de pág 4: 26.2 s en correction). Ahora es un dict
# por id(sp) con referencia fuerte al owner: el id no se reutiliza mientras el
# índice exista (el objeto vive), y la validación `owner is sp` detecta
# cualquier reutilización teórica (p.ej. mocks en tests). Cada entrada guarda
# (palabra, conteo_de_caracteres): el conteo se usa como pre-filtro de la
# búsqueda (edición <= k implica exceso/defecto de conteos <= k), evitando el
# DP Damerau contra TODO el bucket de longitudes.
_SPELL_INDEX_BY_ID: dict[int, dict[int, list[tuple[str, bytes]]]] = {}
_SPELL_INDEX_OWNERS: dict[int, Any] = {}


def _spell_char_counts(word: str) -> bytes:
    """Conteo de caracteres a-z + ñ (áéíóúü → su base) como bytes de 27.

    La distancia de edición <= k implica que el exceso y el defecto de
    conteos entre dos palabras son <= k (cada inserción/borrado mueve 1,
    cada sustitución mueve 1 en cada lado; la transposición no cambia los
    conteos). Es un bound necesario (no suficiente) y barato de comparar.
    """
    c = bytearray(27)
    for ch in word:
        if ch == "ñ":
            c[26] += 1
        elif "a" <= ch <= "z":
            c[ord(ch) - 97] += 1
        elif ch in "áéíóúü":
            base = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u"}[ch]
            c[ord(base) - 97] += 1
    return bytes(c)


def _spell_words_by_len(sp: Any) -> dict[int, list[tuple[str, bytes]]]:
    """Diccionario agrupado por longitud (lazy, cacheado por instancia)."""
    key = id(sp)
    if _SPELL_INDEX_OWNERS.get(key) is sp:
        return _SPELL_INDEX_BY_ID[key]
    idx: dict[int, list[tuple[str, bytes]]] = {}
    for w in sp.word_frequency.dictionary:
        idx.setdefault(len(w), []).append((w, _spell_char_counts(w)))
    _SPELL_INDEX_BY_ID[key] = idx
    _SPELL_INDEX_OWNERS[key] = sp
    return idx


def _damerau_le(a: str, b: str, max_dist: int) -> bool:
    """True si la distancia Damerau-Levenshtein(a, b) <= max_dist.

    pyspellchecker cuenta la TRANSPOSICIÓN como 1 edición (edit_distance_1
    genera deletes + transposes + replaces + inserts), así que para replicar
    su corrección exactamente hay que usar Damerau, no Levenshtein. DP con
    corte temprano por fila (mismo patrón que _levenshtein_le): si una fila
    entera queda por encima del umbral, ninguna posterior puede bajar.
    """
    if abs(len(a) - len(b)) > max_dist:
        return False
    prev_prev = list(range(len(b) + 1))  # fila i-2
    prev = list(range(len(b) + 1))       # fila i-1
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(cur[-1] + 1, prev[j] + 1, cost))
            # Transposición: a[i-2] == b[j-1] y a[i-1] == b[j-2]
            if i > 1 and j > 1 and a[i - 2] == b[j - 1] and a[i - 1] == b[j - 2]:
                cur[j] = min(cur[j], prev_prev[j - 2] + 1)
            row_min = min(row_min, cur[j])
        if row_min > max_dist:
            return False
        prev_prev, prev = prev, cur
    return prev[-1] <= max_dist


def _spell_candidates(sp: Any, word: str, max_dist: int) -> list[str]:
    """Palabras del diccionario a distancia Damerau <= max_dist de `word`.

    Recorre el índice por longitud ([len-2, len+2] para distancia 2) con
    `_damerau_le` (mismo conjunto de candidatos que pyspellchecker genera por
    expansión, pero sin la expansión masiva de ~4.8 M strings). Primero
    aplica el pre-filtro de conteos de caracteres (exceso/defecto <= max_dist):
    el DP solo se ejecuta contra el ~1 % del bucket que lo pasa.
    """
    by_len = _spell_words_by_len(sp)
    lo = max(1, len(word) - max_dist)
    hi = len(word) + max_dist
    wc = _spell_char_counts(word)
    out: list[str] = []
    for length in range(lo, hi + 1):
        for cand, cc in by_len.get(length, ()):
            # Pre-filtro: exceso/defecto de conteos <= max_dist.
            ex = 0
            de = 0
            ok = True
            for a, b in zip(wc, cc):
                if b > a:
                    ex += b - a
                elif a > b:
                    de += a - b
                if ex > max_dist or de > max_dist:
                    ok = False
                    break
            if ok and _damerau_le(word, cand, max_dist):
                out.append(cand)
    return out


def _levenshtein_le(a: str, b: str, max_dist: int) -> bool:
    """True si la distancia de edición(a, b) <= max_dist.

    DP completo clásico con corte temprano: si una fila entera queda por encima
    del umbral, ninguna fila posterior puede bajar (la distancia no decrece),
    así que se puede abortar. Las palabras del diccionario y el OCR son cortas
    (<= 23 chars), el costo por par es trivial.
    """
    if abs(len(a) - len(b)) > max_dist:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(cur[-1] + 1, prev[j] + 1, cost))
            row_min = min(row_min, cur[-1])
        if row_min > max_dist:
            return False
        prev = cur
    return prev[-1] <= max_dist


def _remove_diacritics(s: str) -> str:
    """Quita los diacríticos (réplica de pyspellchecker._remove_diacritics)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _spellcheck_correction(sp: Any, word: str) -> str | None:
    """Réplica barata de sp.correction(word) sin la expansión masiva.

    pyspellchecker.correction() genera TODAS las cadenas a distancia 1 y 2 del
    word usando las letras del corpus y filtra contra el diccionario — la
    expansión de distancia 2 de una palabra de 21 chars genera ~4.8 M strings
    (~0.3-1.3 s; en pág 4 del corpus: 8 llamadas a __edit_distance_alt =
    26.3 s del profile). Aquí el MISMO conjunto de candidatos se obtiene de
    forma directa: como edición <= k implica |Δlongitud| <= k, basta revisar
    las palabras del diccionario de longitud [len-k, len+k] con Damerau-Le-
    venshtein acotado (pyspellchecker cuenta transposiciones como 1 edición).

    La selección replica correction() exactamente:
    1. known([word]) — palabra ya en el diccionario → sin corrección (word).
    2. Candidatos a distancia 1; si hay, elegir entre ellos. Si no, distancia 2.
    3. Preferir los que coinciden SIN diacríticos (pyspellchecker._remove_dia-
       critics), y entre esos (o entre todos) el de MAYOR frecuencia del
       diccionario (max(key=__getitem__) = frecuencia/total).

    Empates exactos de frecuencia: pyspellchecker elige por orden de hash del
    set (no determinista entre runs); aquí el orden del índice (orden de
    inserción del diccionario) es determinista. En la práctica ambos son
    igualmente válidos.
    """
    dictionary = sp.word_frequency.dictionary
    # 1. known([word]): palabra ya correcta → sin cambio.
    if word in dictionary:
        return word
    # 2. Límite de edición dependiente de la longitud (calibración 2026-08-15,
    # ver _spell_max_edits): aquí solo llegan las largas >= 13 — 13-14 → 2
    # (sin cambio) y >14 → 1 (se bloquea SOLO la distancia 2, la que produce
    # sugerencias arbitrarias de alta frecuencia en largas; las d1 legítimas
    # como 'inconstitucinal' -> 'inconstitucional' se preservan).
    max_dist = _spell_max_edits(len(word))
    if max_dist < 1:
        return None
    # 3. Candidatos por distancia (1 primero, luego 2 si el límite lo permite).
    pool = _spell_candidates(sp, word, 1)
    if not pool and max_dist >= 2:
        pool = _spell_candidates(sp, word, 2)
    if not pool:
        return None
    # 3. Preferencia de diacríticos + max por frecuencia.
    wn = _remove_diacritics(word)
    diac = [c for c in pool if _remove_diacritics(c) == wn]
    if diac:
        pool = diac
    freq_get = getattr(dictionary, "get", None)
    if freq_get is None:
        return max(pool)
    return max(pool, key=lambda c: freq_get(c, 0) or 0)


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

    El diccionario cargado por defecto es espanol. Por seguridad, solo se
    aplica cuando el texto parece espanol: usarlo sobre ingles, portugues,
    frances, aleman, italiano o CJK puede convertir una deteccion correcta
    en otra palabra. Los alfabetos japones, coreano y chino se dejan intactos
    porque sus tokens no son compatibles con este corrector.
    """
    if not text or len(text) <= 2:
        return text

    # No pasar escritura CJK por un diccionario latino. Ademas de evitar
    # falsos positivos, esto evita cargar/consultar el corrector en cada
    # bloque de una pagina japonesa, coreana o china.
    if any(
        0x3040 <= ord(char) <= 0x30FF       # hiragana / katakana
        or 0xAC00 <= ord(char) <= 0xD7A3    # hangul
        or 0x4E00 <= ord(char) <= 0x9FFF    # hanzi / kanji
        for char in text
    ):
        return text

    # El corrector disponible es espanol. La deteccion se hace aqui, antes
    # de pedir el singleton, para que paginas inglesas/multilingues no sean
    # alteradas accidentalmente. Se reutiliza la misma deteccion robusta del
    # pipeline de traduccion y no se agregan idiomas ni modelos nuevos.
    #
    # Optimización 2026-08-15 (textos largos fusionados): el detector se
    # aplica sobre un PREFIJO de maximo _SPELL_LANG_MAX_CHARS caracteres. El
    # costo de langdetect escala con la longitud del texto (medido: ~16 ms a
    # 8000 chars) y los textos fusionados del merge final pueden superar eso;
    # un prefijo de 600 chars es mas que suficiente para la distribucion de
    # n-gramas del idioma (verificado: 0 diferencias es/no-es entre texto
    # completo y prefijo en el corpus). El lru_cache de _detect_language_ro-
    # bust ya cachea por texto exacto; con el prefijo, los bloques largos de
    # la misma pagina comparten entrada de cache (misma firma de inicio).
    try:
        from translator import _detect_language_robust
        sample = text if len(text) <= _SPELL_LANG_MAX_CHARS \
            else text[:_SPELL_LANG_MAX_CHARS]
        if _detect_language_robust(sample) != "es":
            return text
    except Exception as exc:
        # Ante un fallo del detector, conservar el OCR es mas seguro que
        # aplicar un diccionario posiblemente incorrecto.
        print(f"[OCR-spellcheck] omitido por idioma no verificable: {exc}")
        return text

    palabras = text.split()
    corregidas: list[str] = []
    # Obtener spellchecker UNA VEZ fuera del loop
    sp = _get_spellchecker()

    # Tokenización compartida: el strip de puntuación y el lowercase se
    # calculan UNA sola vez y se reutilizan tanto en la detección extranjera
    # (le pasamos los tokens ya limpios a _contains_foreign_latin_tokens,
    # que antes volvía a hacer split/strip/filter sobre el texto) como en el
    # loop de corrección. El detector extranjero incluye los acrónimos (los
    # recibe en minúsculas, igual que cuando tokenizaba desde el texto); el
    # loop los excluye con su chequeo isupper.
    limpios: list[tuple[str, str, str]] = []
    for p in palabras:
        stripped = p.strip("'\".,;:!?¡¿()[]{}")
        limpios.append((p, stripped, stripped.lower()))

    # Un detector de idioma de bloque puede clasificar como espanol una
    # frase corta mixta (por ejemplo: "Quiero go home"). Antes de corregir,
    # comparar los tokens contra los diccionarios de los idiomas latinos que
    # ya soporta el proyecto. Si hay evidencia extranjera, preservar todo el
    # bloque evita cambiar solo una parte de la frase.
    if len(palabras) >= 2 and _contains_foreign_latin_tokens(
            text, sp,
            tokens=[low for _, stripped, low in limpios
                    if len(stripped) > 1
                    and not any(c.isdigit() for c in stripped)]):
        return text

    for p, stripped, p_lower in limpios:
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

        # Intentar correccion con pyspellchecker (p_lower ya viene en
        # minúsculas de la tokenización compartida)

        if sp is not None:
            # Optimización 2026-08-15 (costo oculto del A/B de la Ruta C):
            # pyspellchecker.correction() paga la expansión de distancia 2
            # (~0.3-1.3 s por palabra de 10-21 chars fuera del diccionario —
            # palabras largas/pegadas del OCR) escaneando TODO el diccionario
            # cuando hay candidatos a distancia 2. _spellcheck_correction()
            # replica la selección exacta (candidatos por longitud + Damerau
            # acotado + pre-filtro de conteos + diacríticos + frecuencia) sin
            # la expansión masiva, y devuelve None si no hay candidatos
            # (resultado idéntico, sin el costo). Se usa a partir de
            # _SPELL_CORRECTION_MIN_LEN (13): en palabras cortas el scan del
            # bucket es más lento que sp.correction() (calibrado 2026-08-15),
            # así que esas delegan a pyspellchecker. Cualquier excepción
            # (p.ej. un checker mockeado en tests sin word_frequency) cae al
            # comportamiento histórico.
            try:
                if len(p_lower) >= _SPELL_CORRECTION_MIN_LEN:
                    correccion = _spellcheck_correction(sp, p_lower)
                else:
                    correccion = sp.correction(p_lower)
            except Exception:
                correccion = None
                try:
                    correccion = sp.correction(p_lower)
                except Exception:
                    correccion = None
            # Calibración 2026-08-15: en palabras de 3-5 chars solo se
            # aceptan correcciones a DISTANCIA 1 (una a distancia 2 cambia
            # >50 % de los caracteres). pyspellchecker ya prefiere distancia 1
            # cuando existe; el filtro solo descarta el fallback a distancia 2
            # de las cortas. Se verifica contra el índice (mismo conjunto de
            # candidatos) y fail-open ante checkers mockeados sin índice: si
            # el índice no se puede construir (MagicMock sin word_frequency
            # real), se acepta la corrección de sp.correction(). Las
            # correcciones reales del capítulo son todas d1 — ninguna se pierde.
            if (correccion is not None and correccion != p_lower
                    and len(p_lower) <= _SPELL_EDIT_1_MAX_LEN):
                try:
                    candidatos_d1 = _spell_candidates(sp, p_lower, 1)
                except (TypeError, AttributeError):
                    candidatos_d1 = [correccion]  # fail-open (mockeado)
                if correccion not in candidatos_d1:
                    correccion = None
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
def _bytes_to_cv2(
    img_bytes: bytes,
    *,
    max_pixels: int | None = None,
) -> _Img | None:
    """Decodifica bytes crudos de imagen (PNG/JPEG/...) a BGR.

    Comparte la validación de píxeles de _base64_to_cv2: inspeccionar la
    cabecera con Pillow es barato y evita que cv2.imdecode reserve un ndarray
    enorme para una imagen muy comprimida. (Optimización 2.4: el cliente
    puede enviar el canvas como cuerpo binario en vez de base64 en JSON.)
    """
    try:
        pixel_limit = MAX_IMAGE_DECODE_PIXELS
        if max_pixels is not None:
            try:
                pixel_limit = min(pixel_limit, max(0, int(max_pixels)))
            except (TypeError, ValueError, OverflowError):
                return None
        header_size: tuple[int, int] | None = None
        try:
            from PIL import Image

            with Image.open(io.BytesIO(img_bytes)) as header:
                header_size = header.size
        except Exception:
            # Algunos formatos aceptados por OpenCV no los abre Pillow; en
            # ese caso se mantiene el fallback histórico de imdecode().
            header_size = None
        if header_size is not None:
            width, height = header_size
            if width <= 0 or height <= 0 or width * height > pixel_limit:
                return None
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _base64_to_cv2(
    b64: str,
    *,
    max_pixels: int | None = None,
) -> _Img | None:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        # ``validate=True`` evita que b64decode ignore caracteres inválidos
        # silenciosamente (p. ej. un payload válido seguido de ``!``). Eso
        # podría hacer que una entrada corrupta pareciera válida y llegara a
        # OpenCV con bytes distintos de los que el cliente envió.
        img_bytes = base64.b64decode(b64, validate=True)
        return _bytes_to_cv2(img_bytes, max_pixels=max_pixels)
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
    # Inpainting de las líneas detectadas. NOTA (Fase 4, 4.4B — DESCARTADA):
    # se probó hacer el inpaint a resolución reducida (0.5×) para abaratar el
    # ~1.4 s/pág (el detector marca ~90% del área del manga como "línea" y
    # TELEA corre sobre la página completa). La validación A/B en el flujo
    # real mostró una REGRESIÓN en la página débil (pág 11: 9 bloques /
    # conf 0.727 → 7 / 0.612) — exactamente donde el prefilter más aporta —
    # así que se revierte a resolución completa. La vía correcta es 4.4C
    # (detector de líneas más selectivo para reducir el área de la máscara),
    # no bajar la resolución del inpaint.
    if int(line_mask.max()) > 0:
        result = cv2.inpaint(result, line_mask, inpaintRadius=3,
                             flags=cv2.INPAINT_TELEA)
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
# Batch del recognizer de EasyOCR (Fase 1, ítem 2.2): EasyOCR procesa los
# crops de texto de a uno por defecto (batch_size=1); el recognizer (CNN)
# aprovecha el batch en GPU/CPU y acelera ~2-4x cuando la página tiene
# varios bloques. Los crops son pequeños (~32px), así que el coste de VRAM
# extra es despreciable frente al presupuesto de 4 GB. NO afecta al
# determinismo por página: cada crop se procesa independientemente, solo
# cambia el empaquetado del recognizer.
_OCR_BATCH_SIZE: int = 8

# NOTA de diseño (Fase 1, ítem 2.2): se evaluó `torch.backends.cudnn.
# benchmark = True` (~10-20% en convnets con input estable) y se DESCARTÓ:
# runtime_diagnostics.configure_torch_determinism() fija cudnn.deterministic
# = True / benchmark = False a propósito (misma imagen → misma firma → misma
# decisión de trigger/negativas U-OCR entre corridas; la variación cuDNN ya
# causó bugs reales en páginas gemelas). benchmark=True es incompatible con
# determinístico (PyTorch lo ignora) y reintroduciría esa variación.


def _run_ocr_on_image(
    reader: Any,
    img_bgr: _Img,
    mag_ratio: float | None = None,
    rotation_info: list[int] | tuple[int, ...] | None = None,
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
    acquired = _ocr_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        print("[OCR] Timeout adquiriendo el semáforo de EasyOCR; se omite esta pasada")
        return []
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
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            kwargs: dict[str, Any] = {
                "detail": 1,
                "paragraph": False,
                "min_size": _OCR_MIN_SIZE,
                "text_threshold": _OCR_TEXT_THRESHOLD,
                "low_text": _OCR_LOW_TEXT,
                "link_threshold": 0.3,
                "canvas_size": min(max(img_bgr.shape[:2]), _OCR_CANVAS_SIZE),
                "mag_ratio": mag,
                # Fase 1 (2.2): batch del recognizer — acelera páginas con
                # varios bloques sin cambiar los resultados por crop.
                "batch_size": _OCR_BATCH_SIZE,
            }
            if rotation_info is not None:
                kwargs["rotation_info"] = list(rotation_info)
            return cast(list[Any], reader.readtext(img_rgb, **kwargs))
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
            try:
                conf = float(res.get("confidence", 0.5))
            except (TypeError, ValueError):
                # Un bloque malformado no debe tumbar el resto de la página.
                continue
            # Paridad con el camino tupla: mismo filtro de confianza mínima.
            if not text or not np.isfinite(conf) or conf < 0.08:
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
        try:
            bbox, text, conf = res
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        text = str(text).strip()
        if not text or not np.isfinite(conf) or conf < 0.08:
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


def _rapid_cond_skip(blocks_easy: list[dict[str, Any]],
                      use_hybrid: bool) -> bool:
    """True → omitir el pase RapidOCR (página ya bien detectada).

    Con ``use_hybrid=False`` (pure_easyocr) no hay tier híbrido: se omite.
    Con el flag ``RAPID_COND_ENABLED`` apagado se conserva el comportamiento
    histórico (siempre correr RapidOCR). Si EasyOCR resolvió la página con
    fuerza —>= ``RAPID_COND_MIN_BLOCKS`` bloques y confianza media
    >= ``RAPID_COND_MIN_CONF``, el mismo criterio del trigger v4.2— el pase
    CPU complementario solo deduplicaría (la fusión se queda con el de mayor
    confianza) y se evita el coste de ~1.1-1.5s/pág. Páginas débiles o vacías
    devuelven False: ahí RapidOCR sí puede recuperar texto estilizado.
    """
    if not use_hybrid:
        return True
    if not RAPID_COND_ENABLED:
        return False
    if not blocks_easy:
        return False
    avg_conf = float(np.mean([b.get("confidence", 0) for b in blocks_easy]))
    return (len(blocks_easy) >= RAPID_COND_MIN_BLOCKS
            and avg_conf >= RAPID_COND_MIN_CONF)


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

    # ── Híbrido condicional (Fase 1, ítem 2.1): RapidOCR solo si EasyOCR débil ──
    # Antes corría SIEMPRE en paralelo con EasyOCR y el wait del hilo pagaba
    # ~1.5s/pág incluso en páginas fáciles (la fusión solo deduplicaba). Con
    # _rapid_cond_skip, en páginas que EasyOCR ya resolvió con fuerza (>= 3
    # bloques y conf >= 0.20, el mismo criterio del trigger v4.2) se omite el
    # pase CPU completo. En páginas débiles o vacías corre igual (serial,
    # después de EasyOCR) para capturar texto estilizado y títulos.
    rapid_blocks: list[dict[str, Any]] = []
    if not _rapid_cond_skip(blocks_easy, use_hybrid):
        print("[OCR] EasyOCR débil/vacío: ejecutando RapidOCR (CPU) para "
              "complementar...")
        try:
            img_rapid = _preprocess_rapid(
                img_ocr,
                already_prefiltered=prefilter,
            )
            rapid_blocks = _run_rapidocr(img_rapid)
        except Exception as exc:
            print(f"[OCR] RapidOCR falló: {exc}")

        if rapid_blocks:
            merged = _fusionar_blocks(blocks_easy, rapid_blocks)

            # Página mixta: RapidOCR puede detectar texto CJK aunque el
            # lector latino de EasyOCR no lo reconozca correctamente. Se
            # reintenta como máximo con los dos alfabetos más evidentes y en
            # CPU; así no se cargan varios modelos EasyOCR en la GTX 1050 Ti.
            if lang_hint == "auto":
                script_hints = _script_language_hints(rapid_blocks)[:2]
                for script_lang in script_hints:
                    try:
                        script_reader = _get_ocr_reader(
                            script_lang, prefer_gpu=False)
                        if script_reader is None:
                            continue
                        script_results = _run_ocr_on_image(
                            script_reader, img_ocr)
                        script_blocks = _ocr_results_to_blocks(
                            script_results, img_ocr)
                        if script_blocks:
                            merged = _fusionar_blocks(merged, script_blocks)
                            print(
                                f"[OCR] Recuperación {script_lang} CPU: "
                                f"{len(script_blocks)} bloques"
                            )
                    except Exception as exc:
                        # La segunda pasada es opcional: si falla no se debe
                        # perder el resultado ya fusionado de EasyOCR/RapidOCR.
                        print(
                            f"[OCR] Recuperación {script_lang} omitida: {exc}"
                        )

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
        border_mask = cv2.bitwise_xor(
            cv2.dilate(mask_region, np.ones((5, 5), np.uint8)), mask_region)
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


def _semantic_type_for_region(region: dict[str, Any]) -> str:
    """Convierte etiquetas de detectores a tipos semánticos estables."""
    raw = str(region.get("type") or region.get("label") or "text").lower()
    if any(token in raw for token in ("title", "chapter", "heading")):
        return "title"
    if any(token in raw for token in ("header", "caption", "narrat", "cartel")):
        return "header"
    if any(token in raw for token in ("sfx", "sound", "onomat")):
        return "sfx"
    if any(token in raw for token in ("thought", "thinking", "thought_bubble")):
        return "thought"
    if any(token in raw for token in ("bubble", "balloon", "speech", "dialog")):
        return "dialogue"
    return "text"


# Parámetros del crop de la Ruta C (A/B 2026-08-15): son módulo-globales
# (NO Final) para que benchmark_rutac_params.py los parchee en runtime y mida
# alternativas (pad factor / pad mínimo / interpolación) sin tocar producción.
# A/B 2026-08-15 (benchmark_rutac_params.py, 14 págs): pad 6% → 3% mantiene
# 107 = 107 bloques y textos idénticos a −23.6 % de tiempo en páginas con Ruta
# C (−1.45 s/pág). INTER_CUBIC vs INTER_LINEAR: peor recuperación (−4 bloques)
# sin ganancia → se mantiene CUBIC. rotation_info: (0,180) pierde 1 bloque
# real → se mantiene (0,90,180,270).
# CORRECCIÓN 2026-08-15 (re-corrida con daemon VLM detenido, --reps 3,
# benchmark_results/rutac_params_reps3.json): el −23.6 % del pad era deriva de
# GPU sin control (noise-floor 0.02 s vs ±0.3-0.7 s con el daemon activo). Con
# ruido controlado, pad 0.03 vs 0.06 es neutro en tiempo (+0.2 %, bloques
# idénticos 47 = 47) → el 3% queda igualmente como producción; box_thresh 0.35
# neutro en tiempo y −1 bloque; unclip 2.2 neutro en tiempo y −4 bloques → los
# defaults (0.5 / 1.6) quedan CONFIRMADOS.
_RUTA_C_PAD_FACTOR: float = 0.03
_RUTA_C_PAD_MIN: int = 6
_RUTA_C_INTERP: int = cv2.INTER_CUBIC

# Parámetros de detección de RapidOCR en los CROPS de la Ruta C (A/B 2026-08-15,
# benchmark_rutac_params.py). None = defaults de la librería (0.5 / 1.6), que es
# el comportamiento histórico. Módulo-globales para parcheo en runtime.
_RUTA_C_RAPID_BOX_THRESH: float | None = None
_RUTA_C_RAPID_UNCLIP_RATIO: float | None = None

# Batch estructural de la Ruta C (A/B 2026-08-15, benchmark_rutac_batch.py): en
# vez de det DBNet + rec por crop, los crops se apilan en un strip vertical (gap
# blanco, chunks de alto <= _RUTA_C_STRIP_MAX_CHUNK_H — el límite del det es
# max_side_len=2000, con margen) y se ejecuta det UNA vez por chunk + UNA sola
# llamada text_rec con TODAS las líneas de todos los crops. El A/B midió −76.6 %
# en el núcleo (20.5 s → 4.8 s, 52 det-calls → 8) con recuperación equivalente o
# mejor; los duplicados que se observaron son regiones YOLO solapadas (crops
# gemelos con el mismo contenido) — mismo comportamiento que el per-crop,
# deduplicado downstream por overlap en la fusión.
_RUTA_C_STRIP_GAP: int = 24
_RUTA_C_STRIP_MAX_CHUNK_H: int = 1900
# Interruptor del batch estructural en producción (default ON tras la
# integración de 2026-08-15). Módulo-global (no Final) para que
# benchmark_rutac_batch.py haga el A/B mismo-proceso (per-crop vs strip) y
# como válvula de rollback si el strip diera problemas en algún corpus: con
# False, _recover_regions_with_easyocr vuelve a _run_rapidocr por crop.
_RUTA_C_STRIP_BATCH: bool = True
# Fix 2026-08-16 (pérdida de diálogos del strip): el det DBNet sobre el strip
# apilado detecta mal ciertas líneas — el strip leyó ruido ('Y' 0.25, 'P!' 0.46,
# 'OE O' 0.5, 'S E' 0.71, 'IIFARRA' 0.49) o NADA en los crops de las págs
# 1/4/40/52, mientras el crop INDIVIDUAL las lee a 0.94-0.99 (verificado con
# el A/B benchmark_strip_fix_ab y el VLM). Cuando el strip no devuelve nada
# CONFiable para un crop (ver _rapid_blocks_usable), se reintenta ESE crop con
# _run_rapidocr individual (el motor exacto del camino pre-strip); el fallback
# EasyOCR NO lee esos crops (basura 'じ‥'/'鼻' — medido), así que el retry es
# rapid, no EasyOCR. A/B 2026-08-16: recupera los 4 diálogos perdidos.
_RUTA_C_STRIP_RETRY_INDIVIDUAL: bool = True
_RUTA_C_STRIP_RETRY_CONF_HI: Final[float] = 0.85
_RUTA_C_STRIP_RETRY_CONF_LO: Final[float] = 0.7
_RUTA_C_STRIP_RETRY_MIN_LEN: Final[int] = 6
# Fallback híbrido (2026-08-16): cuando el strip (ya con el retry individual)
# sigue DÉBIL (conf < 0.7) en un crop que ya tenía TEXTO del híbrido (el
# texto previo detectado en esa zona), NO descartar el crop: correr igualmente
# EasyOCR y FUSIONAR ambos resultados (el merge final deduplica por overlap).
# Protege el texto real del híbrido de ser descartado por un bloque débil de
# la Ruta C. Complementa al retry rapid-individual (que recupera los 4
# diálogos perdidos); esta capa cubre el caso donde el retry tampoco lee bien
# y el híbrido ya había capturado el texto en la zona.
_RUTA_C_STRIP_HYBRID_FALLBACK: bool = True
_RUTA_C_STRIP_HYBRID_MIN_CONF: Final[float] = 0.7
_RUTA_C_HYBRID_OVERLAP_MIN: Final[float] = 0.1


def _rapid_blocks_max_conf(blocks: list[Any]) -> float:
    """Máxima confianza de los bloques rapid (0.0 si vacío/malformado)."""
    mx = 0.0
    for rb in blocks:
        if not isinstance(rb, dict):
            continue
        try:
            rb_conf = float(rb.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(rb_conf):
            mx = max(mx, rb_conf)
    return mx


def _crop_has_prior_text(region: dict[str, Any],
                         prior_blocks: list[dict[str, Any]]) -> bool:
    """¿El híbrido (o bloques previos) ya leyó texto en esta zona del crop?

    Solape > _RUTA_C_HYBRID_OVERLAP_MIN con algún bloque previo con texto.
    Los callers ya excluyen los crops con overlap > 0.5 ANTES de la Ruta C;
    aquí se captura el solape PARCIAL (el híbrido tocó la zona sin cubrirla
    del todo), que es el caso donde un bloque débil de la Ruta C puede hacer
    que la fusión descarte el texto real del híbrido.
    """
    for hb in prior_blocks:
        if not isinstance(hb, dict):
            continue
        if not str(hb.get("text", "")).strip():
            continue
        try:
            if _overlap_ratio(region, hb) > _RUTA_C_HYBRID_OVERLAP_MIN:
                return True
        except (TypeError, KeyError):
            continue
    return False


def _rapid_blocks_usable(blocks: list[Any]) -> bool:
    """¿El strip leyó algo CONFiable para este crop?

    Confiable = conf >= 0.85 (lectura segura), o conf >= 0.7 con texto de
    >= 6 chars (frase plausible a conf moderada). Distribución medida del
    corpus (2026-08-16, 7 págs): el ruido del strip mide 0.25-0.71 ('Y',
    'O', 'P!', 'S E', 'OE O', 'IIFARRA') y las lecturas reales largas
    0.81-0.99 ('ERASETUCCOMOLOS ILAM' 0.81, 'AAHH...' 0.99) — un umbral
    puro de confianza no las separa ('S E' 0.71 vs la frase larga 0.81);
    la combinación conf + longitud sí.
    """
    for rb in blocks:
        if not isinstance(rb, dict):
            continue
        try:
            rb_text = str(rb.get("text", "")).strip()
            rb_conf = float(rb.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not rb_text or not np.isfinite(rb_conf):
            continue
        if (rb_conf >= _RUTA_C_STRIP_RETRY_CONF_HI
                or (rb_conf >= _RUTA_C_STRIP_RETRY_CONF_LO
                    and len(rb_text) >= _RUTA_C_STRIP_RETRY_MIN_LEN)):
            return True
    return False


def _ruta_c_prepare_crops(
    img_bgr: _Img,
    regions: list[dict[str, Any]],
    upscale: float,
) -> list[dict[str, Any] | None]:
    """Prepara los crops de la Ruta C (pad + upscale + cls de rotación).

    Resultado alineado con ``regions``: una entrada por región, ``None`` si el
    crop no es procesable (tamaño mínimo o límite de resolución). Compartido por
    el batch estructural (strip) y el loop de mapeo/fallback de
    _recover_regions_with_easyocr para que AMBOS usen EXACTAMENTE los mismos
    crops (mismo pad 3%, mismo upscale INTER_CUBIC, mismo cls de rotación).
    """
    h_page, w_page = img_bgr.shape[:2]
    prepared: list[dict[str, Any] | None] = []
    for r in regions:
        region_type = _semantic_type_for_region(r)
        pad = max(_RUTA_C_PAD_MIN, int(min(r["w"], r["h"]) * _RUTA_C_PAD_FACTOR))
        x0, y0 = max(0, r["x"] - pad), max(0, r["y"] - pad)
        x1, y1 = min(w_page, r["x"] + r["w"] + pad), min(h_page, r["y"] + r["h"] + pad)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            prepared.append(None)
            continue
        up_w, up_h = int(crop.shape[1] * upscale), int(crop.shape[0] * upscale)
        if up_w > 8000 or up_h > 8000:
            prepared.append(None)
            continue
        up_img = cv2.resize(crop, (up_w, up_h), interpolation=_RUTA_C_INTERP)
        # Clasificar antes del primer OCR: RapidOCR también debe recibir el
        # crop corregido cuando el texto está girado 180°.
        up_img_ocr, se_roto, cls_score = _classify_rotate_crop(up_img)
        if se_roto:
            print(f"[OCR] Ruta C: globo rotado 180° "
                  f"(score {cls_score:.2f}) — corregido")
        prepared.append({
            "img": up_img_ocr,
            "x0": x0, "y0": y0,
            # Dimensiones del upscale PRE-cls: idénticas a las del rotado
            # (ROTATE_180 conserva forma) y son las que usa el mapeo de
            # des-rotación de coordenadas.
            "up_w": up_img.shape[1], "up_h": up_img.shape[0],
            "se_roto": se_roto, "cls_score": cls_score,
            "type": region_type,
        })
    return prepared


def _rapidocr_blocks_from_lines(
    crop_img: _Img,
    items: list[tuple[Any, Any, Any]],
) -> list[dict[str, Any]]:
    """Construye bloques internos desde líneas (bbox, text, conf) de un crop.

    Réplica exacta del bloque de construcción de _run_rapidocr (conf mínima
    0.08, textColor por brillo del ROI) sin volver a ejecutar det/rec. NO
    agrupa: cada caller aplica _group_and_merge_blocks con su propio
    page_height (None para crops — los bordes del globo son texto legítimo —
    y la altura de página para páginas completas).
    """
    blocks: list[dict[str, Any]] = []
    for bbox, text, conf in items:
        try:
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
            roi = crop_img[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
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
    return blocks


def _rapidocr_strip_batch(
    crops: list[dict[str, Any] | None],
    upscale: float = 3.5,
) -> dict[int, list[dict[str, Any]]]:
    """Re-OCR estructural de los crops de la Ruta C en UNA pasada det+rec.

    En vez de det DBNet + rec por crop (~1.2 s/crop de det), los crops se
    apilan en un strip vertical (gap _RUTA_C_STRIP_GAP, chunks de alto
    <= _RUTA_C_STRIP_MAX_CHUNK_H) y se ejecuta:
      1. det DBNet UNA vez por chunk (box_thresh/unclip: defaults de la Ruta C
         0.5/1.6, con los overrides de A/B si están parcheados).
      2. UNA sola llamada text_rec con TODAS las líneas de todos los chunks
         (batch nativo del recognizer).
      3. Mapeo de cada línea a su crop por centro-y (líneas cuyo centro cae en
         el gap se descartan — no pertenecen a ningún crop) y construcción de
         bloques en coordenadas del crop upscaleado.

    Returns:
        {índice en ``crops``: bloques en formato interno (coords del crop
        upscaleado, ya agrupados por _group_and_merge_blocks)}. Los índices
        None o sin líneas recuperadas no aparecen: el caller los trata como
        "rapid no recuperó" y cae al fallback EasyOCR por crop.

    Degradación segura: sin engine o con semáforo agotado → {} (el fallback
    por crop queda intacto, igual que si RapidOCR no recuperara nada).
    """
    engine = _get_rapid_engine()
    if engine is None:
        return {}
    acquired = _rapid_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        print("[OCR] Timeout adquiriendo semaforo RapidOCR (strip, 120s)")
        return {}
    try:
        valid: list[tuple[int, dict[str, Any]]] = [
            (i, c) for i, c in enumerate(crops) if c is not None
        ]
        if not valid:
            return {}
        # Stitch vertical con gap blanco, en chunks de <= MAX_CHUNK_H.
        chunks: list[dict[str, Any]] = []
        current: list[tuple[int, dict[str, Any]]] = []
        cur_h, cur_w = 0, 0

        def flush() -> None:
            nonlocal current, cur_h, cur_w
            if not current:
                return
            img = np.full((cur_h, cur_w, 3), 255, np.uint8)
            bands: list[dict[str, Any]] = []
            top = 0
            for idx, c in current:
                img[top:top + c["up_h"], 0:c["up_w"]] = c["img"]
                bands.append({"idx": idx, "top": top, "h": c["up_h"]})
                top += c["up_h"] + _RUTA_C_STRIP_GAP
            chunks.append({"img": img, "bands": bands})
            current, cur_h, cur_w = [], 0, 0

        for item in valid:
            _cidx, cinfo = item
            h, w = cinfo["up_h"], cinfo["up_w"]
            if current and cur_h + _RUTA_C_STRIP_GAP + h > _RUTA_C_STRIP_MAX_CHUNK_H:
                flush()
            if not current:
                current = [item]
                cur_h, cur_w = h, w
            else:
                current.append(item)
                cur_h += _RUTA_C_STRIP_GAP + h
                cur_w = max(cur_w, w)
        flush()

        # Parámetros de detección: mismos defaults que _run_rapidocr (0.5/1.6)
        # con los overrides de A/B si están parcheados en runtime.
        engine.text_det.postprocess_op.box_thresh = (
            _RUTA_C_RAPID_BOX_THRESH if _RUTA_C_RAPID_BOX_THRESH is not None else 0.5)
        engine.text_det.postprocess_op.unclip_ratio = (
            _RUTA_C_RAPID_UNCLIP_RATIO if _RUTA_C_RAPID_UNCLIP_RATIO is not None else 1.6)
        engine.text_score = 0.5

        all_lines: list[tuple[Any, Any, int]] = []  # (line, bbox, crop_idx)
        for chunk in chunks:
            dt_boxes, _ = engine.text_det(chunk["img"])
            if dt_boxes is None:
                continue
            boxes = list(dt_boxes)
            if not boxes:
                continue
            lines = engine.get_crop_img_list(chunk["img"], boxes)
            for box, line in zip(boxes, lines):
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                cy = int(min(ys) + (max(ys) - min(ys)) / 2)
                band = next(
                    (b for b in chunk["bands"]
                     if b["top"] <= cy < b["top"] + b["h"]), None)
                if band is None:
                    # Centro en el gap: la caja no pertenece a ningún crop.
                    continue
                all_lines.append((line, box, band["idx"]))
        if not all_lines:
            return {}

        rec_res, _ = engine.text_rec([l for l, _b, _i in all_lines])

        per_crop: dict[int, list[Any]] = {}
        for (line, box, crop_idx), (text, conf) in zip(all_lines, rec_res):
            per_crop.setdefault(crop_idx, []).append((box, text, conf))

        tops: dict[int, int] = {}
        for chunk in chunks:
            for band in chunk["bands"]:
                tops[band["idx"]] = band["top"]

        result: dict[int, list[dict[str, Any]]] = {}
        for idx, items in per_crop.items():
            c = crops[idx]
            if c is None:
                continue
            top = tops[idx]
            local: list[Any] = []
            for box, text, conf in items:
                bbox_local = [(float(p[0]), float(p[1]) - top) for p in box]
                local.append((bbox_local, text, conf))
            result[idx] = _group_and_merge_blocks(
                _rapidocr_blocks_from_lines(c["img"], local), None)
        return result
    except Exception as e:
        print(f"[OCR] Error en el batch estructural RapidOCR: {e}")
        return {}
    finally:
        if acquired:
            _rapid_semaphore.release()


def _recover_regions_with_easyocr(
    img_bgr: _Img,
    regions: list[dict[str, Any]],
    lang_hint: str = "es",
    upscale: float = 3.5,
    hybrid_blocks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Re-OCR de regiones de texto/globos a nivel individual (Ruta C).

    Para cada región (coordenadas de página):
      1. Recortar con padding (A/B 2026-08-15: 3% en vez de 6% — 107 = 107
         bloques y textos idénticos a −23.6 %, ver _RUTA_C_PAD_FACTOR).
      2. Upscale 3.5× (INTER_CUBIC) — A/B corregido 2026-08-15
         (benchmark_rutac_upscale/recovery, daemon detenido, --reps 3): 3.5×
         recupera 2 bloques más que 2× (pág 11) a tiempo neutro. El 2×
         aplicado 2026-08-14 se revierte: su A/B estaba roto (comparaba
         3.5× vs 3.5×) y el "−24%" era deriva de GPU.
      3. OCR sobre el recorte upscaleado.
      4. Mapear bloques de vuelta a coordenadas de página (÷ upscale).

    Motor: RapidOCR CPU por defecto en el crop; EasyOCR GPU queda como
    fallback si RapidOCR no devuelve texto fiable. Si el daemon U-OCR está
    infiriendo (_uocr_inferring activo, v4.2), EasyOCR no se carga nunca y
    la Ruta C permanece en RapidOCR para no competir por la GTX. El chequeo
    ocurre ANTES de _get_ocr_reader(), igual que en _detect_and_ocr.

    Estructural (2026-08-15): con el motor rapid activo, el paso 3 se ejecuta
    en batch — los crops se apilan en un strip vertical y corren det DBNet por
    chunk + UNA llamada text_rec para todos (−76.6 % del núcleo en el A/B,
    benchmark_rutac_batch.py); el fallback EasyOCR por crop se conserva para
    los crops que el strip no recupera.

    Fallback híbrido (2026-08-16): ``hybrid_blocks`` es el texto YA leído
    (híbrido/YOLO previos) en la página. Si el strip (con su retry individual)
    queda DÉBIL (< _RUTA_C_STRIP_HYBRID_MIN_CONF) en un crop que solapa ese
    texto previo, se corre igualmente EasyOCR y se FUSIONAN ambos resultados —
    protege el texto real previo de ser descartado por un bloque débil de la
    Ruta C en la fusión final.

    Returns:
        Bloques en formato interno ({x, y, w, h, text, confidence, fontSize}).
    """
    # §8.4.4: si el daemon U-OCR está infiriendo, degradar a RapidOCR CPU.
    use_rapid = _uocr_inferring.is_set()
    prefer_rapid = bool(RUTA_C_RAPID_PRIMARY) and not use_rapid
    # Con RapidOCR primario no cargamos EasyOCR GPU por adelantado. Solo se
    # inicializa si el recorte no devuelve texto fiable y hace falta fallback.
    reader = None
    if not use_rapid and not prefer_rapid:
        reader = _get_ocr_reader(lang_hint)
        if reader is None:
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
    # Batch estructural: preparar los crops UNA vez (pad + upscale + cls de
    # rotación) y, si el motor rapid está activo, correr el strip (det por
    # chunk + UNA text_rec para todos). El fallback EasyOCR por crop se
    # conserva: los crops que el strip no recupera caen al reader lazy.
    # Con _RUTA_C_STRIP_BATCH=False se vuelve al re-OCR por crop (_run_rapidocr
    # por región) — A/B del benchmark y válvula de rollback.
    prepared = _ruta_c_prepare_crops(img_bgr, regions, upscale)
    strip_by_crop: dict[int, list[dict[str, Any]]] = {}
    if (use_rapid or prefer_rapid) and _RUTA_C_STRIP_BATCH:
        strip_by_crop = _rapidocr_strip_batch(prepared, upscale)

    recovered: list[dict[str, Any]] = []
    for i, r in enumerate(regions):
        crop_info = prepared[i]
        if crop_info is None:
            continue
        region_type = crop_info["type"]
        x0, y0 = crop_info["x0"], crop_info["y0"]
        up_img_ocr = crop_info["img"]
        se_roto = crop_info["se_roto"]
        # Dimensiones del upscale PRE-cls (idénticas tras ROTATE_180): son
        # las que usa la des-rotación de coordenadas del bloque.
        up_w_original = crop_info["up_w"]
        up_h_original = crop_info["up_h"]
        if use_rapid or prefer_rapid:
            # RapidOCR es el motor primario de Ruta C porque el benchmark de
            # crops artísticos mostró mejor lectura que EasyOCR. En el camino
            # de daemon, además, es obligatorio para no competir por VRAM.
            # Los bloques vienen del batch estructural (strip) por crop; con
            # _RUTA_C_STRIP_BATCH=False se re-OCRea el crop individual.
            if _RUTA_C_STRIP_BATCH:
                rapid_blocks = strip_by_crop.get(i, [])
                # Fix 2026-08-16: si el strip no devolvió nada >= 0.7 para
                # ESTE crop (ruido 'OE O' 0.5 / 'S E' 0.6 / 'P!' 0.46 o nada),
                # reintentar el crop individual con _run_rapidocr — el det
                # DBNet sobre el crop solo lee bien las líneas que el strip
                # apilado detecta mal (A/B: recupera los 4 diálogos perdidos
                # sin coste en los crops que el strip sí resuelve).
                if (_RUTA_C_STRIP_RETRY_INDIVIDUAL
                        and not _rapid_blocks_usable(rapid_blocks)):
                    retry_blocks = _run_rapidocr(
                        up_img_ocr, filter_page_margins=False,
                        box_thresh=_RUTA_C_RAPID_BOX_THRESH,
                        unclip_ratio=_RUTA_C_RAPID_UNCLIP_RATIO)
                    # Solo reemplazar si el retry produjo algo: si devuelve
                    # vacío, se conservan los bloques originales del strip
                    # (aunque débiles) porque su TEXTO alimenta el script
                    # hint del fallback (_script_language_hints) para elegir
                    # el lector CJK en lugar del latino de "auto".
                    if retry_blocks:
                        rapid_blocks = retry_blocks
            else:
                rapid_blocks = _run_rapidocr(
                    up_img_ocr, filter_page_margins=False,
                    box_thresh=_RUTA_C_RAPID_BOX_THRESH,
                    unclip_ratio=_RUTA_C_RAPID_UNCLIP_RATIO)
            usable_rapid: list[dict[str, Any]] = []
            for rb in rapid_blocks:
                if not isinstance(rb, dict):
                    continue
                try:
                    rb_text = str(rb.get("text", "")).strip()
                    rb_conf = float(rb.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                if rb_text and np.isfinite(rb_conf) and rb_conf >= RUTA_C_RAPID_MIN_CONF:
                    usable_rapid.append(rb)
            if usable_rapid or use_rapid:
                for rb in usable_rapid:
                    text = str(rb.get("text", "")).strip()
                    if not text:
                        continue
                    bx = x0 + int(rb.get("x", 0) / upscale)
                    by = y0 + int(rb.get("y", 0) / upscale)
                    bw = int(rb.get("w", 0) / upscale)
                    bh = int(rb.get("h", 0) / upscale)
                    if se_roto:
                        bx_u = int(rb.get("x", 0))
                        by_u = int(rb.get("y", 0))
                        bx = x0 + int((up_w_original - bx_u - int(rb.get("w", 0))) / upscale)
                        by = y0 + int((up_h_original - by_u - int(rb.get("h", 0))) / upscale)
                    if bw < 3 or bh < 3:
                        continue
                    recovered.append({
                        "x": bx, "y": by, "w": bw, "h": bh,
                        "text": text,
                        "confidence": float(rb.get("confidence", 0.5)),
                        "fontSize": max(8, int(bh * 0.75)),
                        "textColor": rb.get("textColor", "#000000"),
                        "engine": "rapidocr-region",
                        "type": region_type,
                    })
                # Fallback híbrido (2026-08-16): el strip ya con el retry
                # individual sigue DÉBIL (< _RUTA_C_STRIP_HYBRID_MIN_CONF) y
                # el crop ya tenía texto del híbrido en la zona → NO descartar:
                # correr igualmente EasyOCR y FUSIONAR con lo recuperado
                # arriba (el merge final deduplica por overlap). Protege el
                # texto real del híbrido de ser descartado por un bloque débil
                # de la Ruta C (caso pág 4 del A/B). Con daemon infiriendo
                # (use_rapid) EasyOCR está prohibido → continue.
                if not (_RUTA_C_STRIP_HYBRID_FALLBACK
                        and not use_rapid
                        and hybrid_blocks is not None
                        and _rapid_blocks_max_conf(rapid_blocks)
                        < _RUTA_C_STRIP_HYBRID_MIN_CONF
                        and _crop_has_prior_text(r, hybrid_blocks)):
                    # El daemon infiere o RapidOCR leyó bien el crop (o no
                    # aplica el fallback híbrido): no ejecutar EasyOCR extra.
                    continue
                # Cae al EasyOCR de abajo para fusionar con lo ya recuperado.
            # RapidOCR no recuperó el crop (o el fallback híbrido pide
            # fusionar): aquí se permite EasyOCR GPU, lazy una única vez.
            if reader is None:
                fallback_lang = lang_hint
                prefer_gpu = True
                if lang_hint == "auto":
                    # Aunque la confianza sea insuficiente para aceptar el
                    # texto de RapidOCR, sus caracteres siguen siendo una
                    # señal útil del alfabeto. Usar el lector CJK en CPU
                    # evita caer silenciosamente al lector latino de auto y
                    # evita duplicar otro modelo en la GTX de 4 GB.
                    script_hints = _script_language_hints(
                        rapid_blocks, min_conf=0.15)[:1]
                    if script_hints:
                        fallback_lang = script_hints[0]
                        prefer_gpu = False
                reader = _get_ocr_reader(
                    fallback_lang, prefer_gpu=prefer_gpu)
            if reader is None:
                continue
        if use_rapid:
            # El daemon está infiriendo y no se debe abrir EasyOCR GPU.
            continue
        if reader is None:
            continue
        # Fase 3 punto 3: el crop ya fue corregido por el TextClassifier
        # antes del primer intento, tanto para RapidOCR como para EasyOCR.
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
                bx_u = up_w_original - bx_u - bw_u
                by_u = up_h_original - by_u - bh_u
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
                "type": region_type,
            })
    if not recovered:
        return []
    # Cada bloque proviene de una región ya validada por YOLO/CTD. Sus bordes
    # son los del globo, no los de la página; reaplicar filtros de margen aquí
    # perdería diálogo legítimo pegado al borde superior/inferior.
    return _group_and_merge_blocks(recovered, None)


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
    # El layout por sí solo colisiona entre páginas con la misma composición.
    # Añadir un digest corto del thumbnail conserva la velocidad (~32x32 bytes)
    # y evita reutilizar decisiones OCR para imágenes con contenido distinto.
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    content_digest = hashlib.blake2b(
        f"{sh}x{sw}".encode("ascii") + thumb.tobytes(),
        digest_size=8,
    ).hexdigest()
    return f"{dark_ratio:.1f}:{bits:0{grid * grid}x}:{content_digest}"


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
def _canonical_block_type(value: Any) -> str:
    """Normaliza etiquetas semanticas de OCR a un conjunto estable."""
    raw = str(value or "text").strip().lower().replace("-", "_")
    if any(token in raw for token in ("title", "chapter", "heading")):
        return "title"
    if any(token in raw for token in ("header", "caption", "narrat", "cartel")):
        return "header"
    if any(token in raw for token in ("sfx", "sound", "onomat")):
        return "sfx"
    if any(token in raw for token in ("thought", "thinking")):
        return "thought"
    if any(token in raw for token in ("dialog", "speech", "bubble", "balloon")):
        return "dialogue"
    return "text"


def _merged_block_type(group: list[dict[str, Any]]) -> str | None:
    """Devuelve el tipo dominante, sin inventarlo para OCR sin tipado."""
    explicit = [item.get("type") for item in group if item.get("type")]
    if not explicit:
        return None
    types = {_canonical_block_type(item) for item in explicit}
    priority = ("title", "header", "sfx", "thought", "dialogue", "text")
    return next(kind for kind in priority if kind in types)


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

        # Preservar marcadores de pensamiento/onomatopeya antes de limpiar
        # sÃ­mbolos: ``*sigh*`` pierde su semÃ¡ntica si se convierte en ``sigh``.
        from translator import _es_sfx
        if _es_sfx(text_raw):
            b["text"] = text_raw
            pre_filtered.append(b)
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

            group_types = {
                _canonical_block_type(item.get("type"))
                for item in group
                if item.get("type")
            }
            candidate_type = _canonical_block_type(b2.get("type")) if b2.get("type") else "text"
            concrete_types = group_types - {"text"}
            incompatible_types = bool(
                concrete_types
                and candidate_type != "text"
                and candidate_type not in concrete_types
            )
            if (not incompatible_types
                    and abs(cy1 - cy2) < max_h * 0.45
                    and -b2["w"] < gap_x < max(35, group[-1]["w"] * 2.5)):
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
            merged_block = {
                "x": mx, "y": my, "w": mw, "h": mh,
                "text": " ".join(g["text"] for g in group),
                "confidence": float(np.mean([g["confidence"] for g in group])),
                "fontSize": max(g["fontSize"] for g in group),
                "textColor": group[0]["textColor"],
            }
            merged_type = _merged_block_type(group)
            if merged_type is not None:
                merged_block["type"] = merged_type
            merged.append(merged_block)

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
                group_types = {
                    _canonical_block_type(item.get("type"))
                    for item in v_group
                    if item.get("type")
                }
                candidate_type = _canonical_block_type(b2.get("type")) if b2.get("type") else "text"
                concrete_types = group_types - {"text"}
                incompatible_types = bool(
                    concrete_types
                    and candidate_type != "text"
                    and candidate_type not in concrete_types
                )
                if not incompatible_types and gap_y < max(b["h"], b2["h"]) * 1.5:
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
                merged_block = {
                    "x": mx, "y": my, "w": mw, "h": mh,
                    "text": " ".join(g["text"] for g in v_group),
                    "confidence": float(np.mean([g["confidence"] for g in v_group])),
                    "fontSize": max(g["fontSize"] for g in v_group),
                    "textColor": v_group[0]["textColor"],
                }
                merged_type = _merged_block_type(v_group)
                if merged_type is not None:
                    merged_block["type"] = merged_type
                v_merged.append(merged_block)
            else:
                v_merged.append(b)
        merged = v_merged

    final_blocks: list[dict[str, Any]] = []
    # Solo se aplica a resultados débiles y no-CJK. La heurística de ruido de
    # traducción es útil como gate OCR en este punto, pero sus reglas de
    # caracteres extendidos no deben borrar japonés/coreano/chino de baja
    # confianza.
    from translator import _es_ocr_noise
    for b in merged:
        w, h = b["w"], b["h"]
        text = str(b["text"]).strip()
        text_len = len(text)
        conf = float(b.get("confidence", 0))

        if text_len == 0:
            continue

        has_cjk = any(
            0x3040 <= ord(char) <= 0x30FF
            or 0x3130 <= ord(char) <= 0x318F
            or 0xAC00 <= ord(char) <= 0xD7A3
            or 0x4E00 <= ord(char) <= 0x9FFF
            for char in text
        )
        if conf < 0.35 and not has_cjk and _es_ocr_noise(text):
            print(
                f"[OCR] Filtrando basura OCR de baja confianza: "
                f"'{text[:50]}' conf={conf:.2f}"
            )
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
    diff_from_median = cv2.absdiff(
        roi_gray, np.full(roi_gray.shape, local_median, dtype=roi_gray.dtype))
    
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
        if _is_inside_speech_bubble(img_bgr, block):
            # En globos uniformes preservamos el borde y el relleno; solo
            # marcamos los trazos para no destruir la forma de la burbuja.
            glyph_mask = _build_glyph_mask_for_bubble(img_bgr, block)
            if int(np.sum(glyph_mask > 0)) > 0:
                mask = cv2.bitwise_or(mask, glyph_mask)
            continue

        # Texto flotante/SFX sobre arte: el fondo no es uniforme y una mÃ¡scara
        # de glifos deja halos y restos. En este caso se borra la caja completa
        # para que TELEA pueda reconstruir la textura circundante.
        x0 = max(0, int(block.get("x", 0)))
        y0 = max(0, int(block.get("y", 0)))
        x1 = min(w, x0 + max(0, int(block.get("w", 0))))
        y1 = min(h, y0 + max(0, int(block.get("h", 0))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255

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
        blend_mask = cv2.dilate(blend_mask, k, iterations=2).astype(np.float32)
        blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), sigmaX=2).astype(np.float32)

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
