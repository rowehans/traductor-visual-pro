"""
config.py — Constantes, patrones de ruido y configuración global.

Extraído de server.py para mantener el archivo principal más ligero.
Todas las constantes se importan desde aquí a los módulos que las necesitan.
"""

import os
import re
import sys
from pathlib import Path
from typing import Final


# ─── Paths (compatible with PyInstaller frozen mode) ──────────────
# En modo frozen (.exe), ROOT debe apuntar al DIRECTORIO DEL PROYECTO
# (donde esta env/) para que EasyOCR, CT2 y otros modulos encuentren
# sus modelos descargados. _MEIPASS es un dir temporal de solo lectura
# dentro del .exe donde NO se pueden descargar modelos.
# Buscamos el proyecto real subiendo desde el .exe hacia donde esta env/.
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _exe_dir = os.path.dirname(sys.executable)
    _found = False
    _current = _exe_dir
    for _ in range(10):
        if os.path.isdir(os.path.join(_current, 'env')):
            ROOT = Path(_current)
            _found = True
            break
        _parent = os.path.dirname(_current)
        if _parent == _current:
            break
        _current = _parent
    if not _found:
        # Fallback: ubicaciones conocidas
        for _loc in [r'D:\crear traductor']:
            if os.path.isdir(os.path.join(_loc, 'env')):
                ROOT = Path(_loc)
                _found = True
                break
    if not _found:
        ROOT = Path(_exe_dir)
else:
    ROOT = Path(__file__).resolve().parent
DIST: Final[Path] = ROOT / "dist"
IS_PRODUCTION: bool = DIST.exists() and (DIST / "index.html").exists()

# ─── Server constants ────────────────────────────────────────────
MAX_WORKERS: Final[int] = min(8, (os.cpu_count() or 4))
REQUEST_TIMEOUT: Final[int] = 20  # seconds for external API calls
MAX_IMAGE_DIMENSION: Final[int] = 4096  # max width/height for OCR processing
APP_VERSION: Final[str] = "20260715"

# ─── Validation limits (single source of truth — shared by server.py and routes/) ─
MAX_TEXT_LENGTH: Final[int] = 20_000           # ~20k chars por texto a traducir
MAX_BATCH_SIZE: Final[int] = 500               # max textos por batch
MAX_IMAGE_BYTES: Final[int] = 50 * 1024 * 1024 # 50MB raw base64 (evitar OOM)


# ─── Fusión de motores OCR (modo "fusion") ────────────────────────
# Trigger de refuerzo: se consulta Unlimited-OCR vía daemon solo si la página
# es difícil para los motores rápidos (EasyOCR+RapidOCR): confianza media baja,
# pocos bloques, o un bloque image >15% de la página (diálogo en arte).
# Validado empíricamente (2026-08-03): el re-OCR del panel image completo NO
# recupera el diálogo; la fusión de bloques sí aporta (pág. 11: U-OCR lee lo que
# EasyOCR garbea).
UOCR_TRIGGER_CONF: Final[float] = 0.20          # confianza media mínima del híbrido (v4.2: <0.2 para disparar refuerzo)
UOCR_TRIGGER_MIN_BLOCKS: Final[int] = 3          # mínimo de bloques para no reforzar
UOCR_IMAGE_BLOCK_RATIO: Final[float] = 0.15      # bloque image ≥15% de la página
OCR_ENGINE_WEIGHTS: Final[dict[str, float]] = {
    "easyocr": 1.0,     # motor base (GPU, fiable)
    "rapid": 0.9,        # complemento CPU (menos preciso en general)
    "unlimited": 1.1,    # VLM 3B 4-bit (el más preciso donde detecta)
    "yolo": 0.9,         # Fase 6: bloques re-OCR de regiones YOLO (EasyOCR sobre
                          # crops 3.5× — misma fiabilidad que el híbrido, peso rapid)
}

# ─── Reintento agresivo de RapidOCR (Fase 2) ──────────────────────
# ANTES de disparar el VLM (daemon U-OCR, ~2-8 min/pág), se reintenta
# RapidOCR con parámetros de detección más agresivos (CPU, ~1.5s/pág):
#   box_thresh 0.50→0.30: el detector admite cajas más débiles (texto
#       artístico/decorativo que el umbral default descarta).
#   unclip_ratio 1.60→2.20: las cajas crecen más tras la máscara binaria,
#       recuperando texto partido en glifos sueltos.
#   text_score 0.50→0.40: el recognizer acepta caracteres menos confiables.
# Solo se intenta si la confianza media del híbrido es baja (<0.35) y NO
# hay panel image grande (el diálogo en arte solo lo recupera el VLM). Si
# el merge resuelve la página según el trigger v4.2 (>=3 bloques Y
# conf >=0.20), se evita la inferencia VLM completa.
RAPID_AGGRESSIVE_PARAMS: Final[dict[str, float]] = {
    "box_thresh": 0.30,
    "unclip_ratio": 2.2,
    "text_score": 0.40,
}
# Guarda defensiva: con el trigger v4.2, el reintento solo se alcanza con
# 0 bloques (conf=0.0) o conf<0.2 — ambos < 0.35. Este límite protege ante
# cambios futuros del trigger (p.ej. si algún día dispara con conf > 0.35,
# el reintento no añadiría nada: el texto ya se detectó bien).
RAPID_RETRY_MAX_CONF: Final[float] = 0.35

# ─── Ponderación por tipo semántico en la fusión (Fase 3) ─────────
# Solo los bloques de Unlimited-OCR llevan "type" (text/title/header/...):
# el VLM emite el tipo semántico en el stream (<|det|>title [x,y,w,h]<|/det|>)
# y _ocr_with_unlimited lo propaga a los bloques. En la votación de
# _fusionar_blocks_multi, el acuerdo entre motores sobre texto con tipo
# distintivo (title/header) es evidencia más fuerte que sobre diálogo común:
#   - FUSION_TYPE_REINFORCE: refuerzo de confianza cuando 2+ motores
#     coinciden en texto/región (base 0.15 para diálogo normal; title/header
#     refuerzan más).
#   - FUSION_TYPE_WEIGHTS: peso extra del tipo en _block_score (dedup/NMS:
#     un bloque tipado del VLM gana empatando contra un bloque sin tipo).
FUSION_TYPE_REINFORCE: Final[dict[str, float]] = {
    "title": 0.20,   # títulos (capítulo, portadas): acuerdo = muy fiable
    "header": 0.18,  # cabeceras/running headers
    "text": 0.15,    # diálogo normal (comportamiento base actual)
}
FUSION_TYPE_WEIGHTS: Final[dict[str, float]] = {
    "title": 1.15,
    "header": 1.05,
}

# El reintento solo se considera "salvado" si el merge supera el trigger
# v4.2 con margen (conf >= 0.30, no solo >= 0.20): promediar TODOS los
# bloques (incluidos los híbridos débiles) con un solo bloque fuerte puede
# cruzar 0.2 con facilidad y saltarse el VLM en páginas que aún lo
# necesitan. 0.30 exige que la página quedó CLARAMENTE mejor.
RAPID_RETRY_SALVADO_CONF: Final[float] = 0.30

# ─── TextClassifier de RapidOCR en la Ruta C (Fase 3 punto 3) ─────
# El Cls de RapidOCR (PP-OCRv4, ONNX CPU) clasifica tiras de texto como
# 0° o 180°. EasyOCR NO detecta texto girado; el globo rotado se pierde.
# Con esto, cada crop de globo de la Ruta C pasa por el clasificador y,
# si sale 180° con confianza suficiente, se rota ANTES del re-OCR.
RUTA_C_CLS_ENABLED: Final[bool] = True
# Umbral de confianza del clasificador para aceptar la rotación (el default
# del modelo es 0.9). Solo rota si score > umbral: rotar con confianza baja
# pondría el texto cabeza abajo.
RUTA_C_CLS_THRESH: Final[float] = 0.9

# ─── Detector YOLO de regiones de texto (Fase 6) ──────────────────
# Tier 3.5 de detección: un YOLO fine-tuned (nano/small, ~3M params) detecta
# regiones de diálogo — globos (speech bubbles), cartelas narrativas y títulos
# tipográficos grandes — como OBJETOS (no como texto). Cada región detectada
# alimenta la Ruta C existente (_recover_regions_with_easyocr, upscale 3.5× +
# cls de rotación + degradación CPU) — la brecha central: los OCR solo ven
# "texto" si detectan glifos, pero un globo/título artístico puede no tenerlos
# detectables. Recupera parte del ~12.2% de bloques perdidos SIN la inferencia
# VLM (2-8 min/pág).
#
# Integración: ultralytics se importa EN RUNTIME (dentro de _get_yolo_engine),
# no en el import del módulo — el .exe no se infla y el tier degrada a [] si la
# librería o el modelo no están disponibles.
#
# Modelo: colocar un .pt/.onnx fine-tuned en YOLO_MODEL_PATH. Opciones públicas:
#   - ogkalu/comic-speech-bubble-detector-yolov8m → comic-speech-bubble-detector.pt
#     (YOLOv8m, 8K+ imágenes manga/webtoon/manhua/comic: globos de diálogo).
#   - huyvux3005/manga109-segmentation-bubble → weights/best.pt (YOLOv11n-seg).
#   - kitsumed/yolov8m_seg-speech-bubble → model.pt (YOLOv8m-seg con máscaras).
# El descargado por defecto es ogkalu (box-based, suficiente para la Ruta C;
# CPU ~200-400ms/pág, GPU 8-25ms).
#
# El Trigger Selectivo v4.2 NO se altera: YOLO corre SIEMPRE en fusion como
# recuperador de regiones de alto impacto (antes del trigger) — si los bloques
# recuperados llevan la página por encima del umbral, el VLM no dispara.
YOLO_ENABLED: Final[bool] = True
YOLO_MODEL_PATH: Final[str] = str(ROOT / "models" / "comic-speech-bubble-detector.pt")
# Descarga automática si el archivo no existe (requiere internet a GitHub
# para yolov8n.pt). False por defecto: sin red, el tier degrada a [] en vez
# de bloquear el pipeline con un timeout de descarga.
YOLO_AUTODOWNLOAD: Final[bool] = False
YOLO_CONF_THRESH: Final[float] = 0.25        # confianza mínima de detección
YOLO_IOU_THRESH: Final[float] = 0.45         # NMS
YOLO_IMGSZ: Final[int] = 1280                # tamaño de inferencia (balance velocidad/precisión)
# device: "cpu" por defecto (200-400ms, compatible con cualquier máquina);
# "0" (GPU) solo si CUDA está disponible. Se decide en runtime.
#
# POLÍTICA DETERMINISTA (sesión 116): "auto" resuelve el device UNA sola vez
# por proceso (no por llamada) — el resultado del trigger v4.2 no puede
# depender del estado dinámico de _gpu_lock/_uocr_inferring en cada página
# (causa raíz del no-determinismo: la misma página disparaba U-OCR en single
# pero no en batch según si otro worker tenía el lock al correr YOLO, y GPU
# vs CPU dan detecciones marginalmente distintas que cruzan el umbral 0.25).
YOLO_DEVICE: Final[str] = "auto"  # 'auto': resuelto UNA vez (GPU si CUDA, si no CPU)
YOLO_MAX_REGIONS: Final[int] = 40            # límite de regiones → Ruta C (evita saturar el re-OCR)
YOLO_MIN_AREA_RATIO: Final[float] = 0.0015   # región mínima (0.15% de la página): filtra ruido del detector
# Gate heurístico (code review Fase 6): YOLO solo corre en páginas que el
# híbrido detectó DÉBILMENTE (menos bloques que el mínimo del trigger v4.2, o
# confianza media baja). En páginas normales (bien detectadas) el detector no
# aporta nada y el re-OCR de hasta 40 crops costaría ~2-6s/pág sin beneficio.
# NO es el trigger v4.2 (que decide el VLM): es un filtro previo barato que
# limita el coste del recuperador YOLO a donde tiene impacto (el 12.2% perdido).
YOLO_GATE_MIN_BLOCKS: Final[int] = UOCR_TRIGGER_MIN_BLOCKS  # <3 bloques → correr
YOLO_GATE_MAX_CONF: Final[float] = 0.35      # o conf media < 0.35 → correr
# Substrings de nombres de clase aceptados (independiente del modelo concreto):
# globos, cartelas narrativas, títulos y cajas de texto. Las clases que no
# matcheen (p.ej. "person", "face") se ignoran — el tier solo recupera texto.
YOLO_CLASS_KEYWORDS: Final[tuple[str, ...]] = (
    "bubble", "balloon", "speech", "caption", "narration",
    "title", "text", "letter", "sfx", "sound",
)

# ─── Detector de texto de cómic (comic-text-detector ONNX, CPU) ─
# Tier 3.6 de detección (complementa al YOLO de globos de la Fase 6): dmMaze
# comic-text-detector (port ONNX de mayocream, GPL-3.0) detecta REGIONES de
# texto — globos sin borde, texto flotante sobre el dibujo, pensamientos,
# títulos y tipografías de arte — que los OCR y el detector de globos pierden.
#
# Modelo: models/comic-text-detector.onnx (94.7 MB). Firma verificada:
#   IN  images[1,3,1024,1024] float  (letterbox 1024², BGR CHW, /255)
#   OUT blk[1,64512,7]  YOLOv5 decodificado: cx,cy,w,h,obj,cls_eng,cls_ja
#       seg[1,1,1024,1024]  máscara de texto (UNet, sigmoide)
#       det[1,2,1024,1024]  líneas (DBNet: shrink + threshold maps)
# Corre 100% en CPU (onnxruntime) → 0 VRAM extra; batch=1 estricto.
#
# Post-proceso replicado de dmMaze (inference.py / db_utils.py /
# yolov5_utils.py): blk → conf=obj*cls ≥ CONF_THRESH + NMS por clase;
# det → binarizar > MASK_THRESH + contornos + unclip + score > LINE_SCORE;
# seg → binarizar > MASK_THRESH + contornos NO cubiertos por blk/det.
# Los umbrales por defecto son los de dmMaze (conf 0.4, NMS 0.35, mask 0.3,
# score línea 0.6, unclip 1.5).
COMIC_DETECTOR_ENABLED: Final[bool] = True
COMIC_DETECTOR_MODEL_PATH: Final[str] = str(ROOT / "models" / "comic-text-detector.onnx")
COMIC_DETECTOR_CONF_THRESH: Final[float] = 0.4      # YOLO blk: obj*cls (dmMaze)
COMIC_DETECTOR_NMS_THRESH: Final[float] = 0.35      # NMS por clase (dmMaze)
COMIC_DETECTOR_MASK_THRESH: Final[float] = 0.3      # binarización seg/det (dmMaze)
COMIC_DETECTOR_LINE_SCORE_THRESH: Final[float] = 0.6  # score DBNet (inference.py)
COMIC_DETECTOR_UNCLIP_RATIO: Final[float] = 1.5     # expansión DBNet (dmMaze)
COMIC_DETECTOR_MAX_REGIONS: Final[int] = 60         # límite regiones → Ruta C
COMIC_DETECTOR_MIN_AREA_RATIO: Final[float] = 0.0005  # región mínima (0.05% página)
# Gate heurístico del tier CTD en la fusión (Paso 4, PLAN_MANGA_OCR — lección
# del benchmark del Paso 5: en páginas bien detectadas el re-OCR de crops
# cuesta 2-4s y la mayoría de regiones CTD duplica a YOLO). Mismo patrón que
# el gate YOLO: el tier solo corre en páginas que el pipeline aún detecta
# DÉBILMENTE (menos bloques que el mínimo del trigger, o conf media baja).
# Se evalúa con los bloques POST-YOLO (cascada: si YOLO ya resolvió la
# página, CTD no corre).
COMIC_DETECTOR_GATE_MIN_BLOCKS: Final[int] = UOCR_TRIGGER_MIN_BLOCKS  # <3 bloques → correr
COMIC_DETECTOR_GATE_MAX_CONF: Final[float] = 0.35   # o conf media < 0.35 → correr
# Dedup de regiones CTD vs YOLO por overlap ANTES de la Ruta C: una región
# CTD que solape (inter/min_area, _overlap_ratio) más que el umbral con una
# región YOLO ya detectada se descarta — YOLO ya cubre esa zona y ya va a
# re-OCRear el crop (no pagar el re-OCR dos veces: 85 regiones → 21
# recuperados → solo 5 nuevos en el benchmark de 5 páginas).
COMIC_DETECTOR_DEDUP_IOU: Final[float] = 0.40

# ─── Rotación de texto en la Ruta C (Fase 6) ─────────────────────
# rotation_info es un kwarg de EasyOCR.readtext(): con él, el reader rota el
# crop y elige el ángulo con mejor confianza — recupera títulos verticales
# (tategaki japonés), cartelas rotadas 90°/270° y tipografía estilizada.
# NO se activa en el tier 1 de página completa (multiplicaría ~4x el tiempo
# del camino caliente): solo se pasa en los CROPS de la Ruta C, donde vive el
# texto artístico/vertical que YOLO detecta como región. El cls de 180°
# (_classify_rotate_crop) sigue complementando para el caso horizontal-invertido.
#
# IMPORTANTE: valores ENTEROS (no strings). EasyOCR pasa cada ángulo a
# scipy.ndimage.rotate y, con numpy 2.5 + scipy 1.17, un ángulo string ('90')
# rompe el casting de la ufunc 'cosdg' ("not supported for the input types").
# Con enteros, scipy rota correctamente (verificado empíricamente 2026-08-04).
EASYOCR_ROTATION_INFO: Final[tuple[int, ...]] = (0, 90, 180, 270)


# ─── Cache de decisiones U-OCR (§8.4.1) ───────────────────────────
# Gate global del refuerzo VLM (daemon U-OCR, puerto 5177) dentro de la
# fusión. True = comportamiento histórico: el trigger v4.2 puede disparar la
# inferencia VLM si el daemon responde. False = el refuerzo se anula por
# completo (solo el VLM; YOLO/Ruta C/cls de rotación SIGUEN activos — a
# diferencia de disable_uocr que apaga todo el pipeline de recuperación).
# El CLI manga_ocr.py (extracción pura) lo desactiva por defecto para no
# disparar inferencias de 2-8 min/pág sin pedirlo (--vlm). Se lee en runtime
# dentro de _reforzar_con_unlimited (mutar config desde el CLI surte efecto).
UOCR_ENABLED: Final[bool] = True

# Si una página con una firma de layout concreta ya disparó el refuerzo U-OCR
# y NO recuperó nada (0 bloques nuevos), las páginas repetitivas del capítulo
# con la MISMA firma no deben volver a disparar la inferencia VLM (~2-8 min
# por página). La firma es la distribución espacial de oscuridad (ver
# _page_signature en ocr_utils.py).
UOCR_CACHE_TTL_S: Final[float] = 1800.0          # 30 min: ventana en la que se respeta la decisión negativa
UOCR_CACHE_MAX_ENTRIES: Final[int] = 256         # eviction LRU: capítulos ~5-10 firmas, 256 sobra
# ── Persistencia de las negativas §8.4.1 (sesión 127) ─────────────────
# Default True desde la sesión 129: la salvaguarda mucho_mas_debil en la
# consulta de negativas (misma que _trigger_con_cache) hace SEGURA su
# persistencia — una página gemela que se detecta MUCHO más débil que la
# que registró la negativa ignora la supresión y re-dispara el VLM (el
# diálogo artístico que el híbrido pierde es justo el que el VLM podría
# recuperar).
#
# TRADE-OFF:
#   Pro: determinismo de EJECUCIÓN completo entre servidores — no solo las
#        decisiones de trigger (sesión 125) sino también los saltos §8.4.1
#        de páginas repetitivas se congelan en disco: 2 corridas en procesos
#        separados hacen EXACTAMENTE las mismas llamadas VLM.
#   Contra: la salvaguarda solo cubre el caso "detección actual mucho más
#        débil" — una página gemela con detección COMPARABLE o mejor sigue
#        honrando la negativa (determinismo), y el scope por doc_id (sesión
#        126) impide que capítulos de la misma serie (94% colisión de firma,
#        sesión 124) hereden decisiones entre sí. El coste residual: dentro
#        del MISMO documento, una página con layout repetido y detección
#        comparable no re-dispara — el trade-off aceptado del determinismo.
UOCR_NEG_CACHE_PERSIST: Final[bool] = True

# ─── Salvaguarda de detección débil en las negativas §8.4.1 (sesión 134) ──
# El caso p5 (sesiones 128-129): el híbrido detecta una página artística con
# MUY pocos bloques y confianza baja (p5 registró la negativa con 2-3 bloques
# conf ~0.42; p17 con 1 bloque conf 0.53), el VLM corre y no recupera nada →
# negativa CONGELADA. La variación cuDNN del híbrido puede hacer que la página
# GEMELA se detecte igual de pobre (detección COMPARABLE → el much_mas_debil
# de la sesión 129 no la libera) y la negativa la mata a pesar de tener
# diálogo artístico que el VLM sí leería.
#
# Estas constantes definen cuándo una negativa viene de una detección
# DEMASIADO POBRE como para congelarla: si el híbrido que la registró detectó
# < UOCR_NEG_WEAK_MAX_BLOCKS bloques O conf < UOCR_NEG_WEAK_MIN_CONF, la
# página gemela puede RE-DISPARAR el VLM hasta UOCR_NEG_MAX_REINTENTOS veces
# (contador por firma) antes de congelarla — si la variación cuDNN era el
# problema, el diálogo artístico se recupera en el reintento. El contador
# acota el coste: 1 corrida VLM extra (~2-8 min) por firma débil por TTL,
# NO infinitas.
#
# Sesión 136: las MISMAS constantes aplican al cache de decisión del TRIGGER
# (_trigger_dec_cache en ocr_engine.py). Ahí el caso análogo es: una decisión
# NEGATIVA de trigger ("no disparar el VLM") cacheada por firma cuando el
# híbrido detectó la página con pocos bloques/conf baja se congelaba para
# gemelas con detección COMPARABLE — si la gemela es artística, la negativa
# la mataba. Con estas constantes, la gemela puede RECOMPUTAR el trigger una
# vez por firma (contador re_computes, mismo límite UOCR_NEG_MAX_REINTENTOS)
# en vez de honrar a ciegas la decisión negativa débil.
#
# Efecto amplio acotado: TODA negativa registrada con <3 bloques califica
# como débil — incluidas las de texto débil que dispararon el trigger v4.2
# (<3 bloques Y conf <0.2). El coste máximo es ~1 inferencia VLM extra por
# firma débil por TTL (el contador), no infinitas: páginas repetitivas siguen
# ahorrando tras el re-disparo fallido.
#
# Nota de calibración: el ejemplo del usuario era "<2 bloques o conf <0.3",
# pero p5 registró la negativa con 2-3 bloques conf 0.42 — con esos umbrales
# NO se cubriría el caso objetivo. Se usan <3 bloques (cubre 0/1/2) O conf
# <0.45 (cubre 0.42): p5 y p17 caen dentro; una página con 3 bloques conf 0.9
# (detección real) NO se considera débil y se congela como antes.
UOCR_NEG_WEAK_MAX_BLOCKS: Final[int] = 3      # negativa registrada con < 3 bloques → débil
UOCR_NEG_WEAK_MIN_CONF: Final[float] = 0.45   # o conf < 0.45 → débil
UOCR_NEG_MAX_REINTENTOS: Final[int] = 1       # re-disparos/recomputes permitidos por firma (por TTL)

# ─── Cache de decisión del TRIGGER por firma (sesión 116) ───────────
# El trigger v4.2 depende de blocks/avg_conf del híbrido (cuDNN puede dar
# resultados ligeramente distintos entre corridas) y del device YOLO. Para
# garantizar que 2 corridas idénticas tomen SIEMPRE la misma decisión por
# página, la decisión del trigger (disparar/no disparar U-OCR) se cachea por
# firma de layout (_page_signature) con TTL/LRU — misma imagen → misma firma
# → misma decisión. No aplica con force_uocr/disable_uocr (modos benchmark).
TRIGGER_CACHE_TTL_S: Final[float] = 1800.0       # 30 min (mismo que §8.4.1)
TRIGGER_CACHE_MAX_ENTRIES: Final[int] = 256      # eviction LRU

# YOLO GPU comparte la GTX con EasyOCR GPU (mismo proceso) y el daemon U-OCR:
# con device resuelto a "0", YOLO adquiere _gpu_lock de forma BLOQUEANTE
# (espera a EasyOCR de otro worker) en vez de degradar a CPU — el device
# SIEMPRE es el mismo → detección determinista. La VRAM cabe (benchmark
# sesión 103: daemon 2.25GB + YOLO ~1GB + EasyOCR 0.13GB < 4GB). Timeout de
# la espera: EasyOCR por página toma ~0.9-2s, así que 30s es generoso (solo
# se agota si la GPU está realmente saturada y entonces degrada a CPU, que es
# el fallback de emergencia, no la política).
YOLO_GPU_LOCK_BLOQUEANTE: Final[bool] = True
YOLO_GPU_LOCK_TIMEOUT_S: Final[float] = 30.0


# ─── Timeouts (single source of truth — shared via /api/config) ────
TIMEOUT_OPENCV_INIT_MS: Final[int] = 15000       # app.js: OpenCV init + poll timeout
TIMEOUT_PDFJS_CDN_MS: Final[int] = 10000          # app.js: PDF.js CDN load (UMD)
TIMEOUT_PDFJS_ES_MODULE_MS: Final[int] = 10000    # app.js: PDF.js ES module import
TIMEOUT_PDF_RENDER_MS: Final[int] = 60000          # app.js: PDF page render (60s para PDFs escaneados pesados)
TIMEOUT_TRANSLATE_MS: Final[int] = 30000           # app.js: translate single request
TIMEOUT_TRANSLATE_BATCH_MS: Final[int] = 60000     # app.js: translate batch request
TIMEOUT_PROCESS_PAGE_MS: Final[int] = 120000       # app.js: server process-page
TIMEOUT_INPAINTED_IMAGE_MS: Final[int] = 15000     # app.js: inpainted image decode
TIMEOUT_EXPORT_REVOKE_MS: Final[int] = 10000       # app.js: export URL.revokeObjectURL
TIMEOUT_CDN_LOAD_MS: Final[int] = 8000             # index.html: __loadCdn default

LANGUAGES: Final[dict[str, str]] = {
    "es": "spanish", "en": "english", "pt": "portuguese",
    "fr": "french",  "de": "german",  "it": "italian",
    "ja": "japanese","ko": "korean",  "zh": "chinese (simplified)",
    "zh-cn": "chinese (simplified)", "zh-tw": "chinese (traditional)",
    "auto": "auto",
}


# ─── Diccionario de corrección PRE-OCR (Errores comunes de OCR) ──
# Versión raw (strings de regex) para compatibilidad hacia atrás.
GLOSARIO_PRE: Final[dict[str, str]] = {
    r"\bCoirectamente\b": "Correctamente",
    r"\bVilianos\b": "Villanos",
    r"\bYilianos\b": "Villanos",  # OCR: 'V' misread as 'Y' en tipografia artistica
    r"\bIFMPORAOA1\b": "TEMPORADA 1",
    r"\bIFMPORAOA\b": "TEMPORADA",
    r"\bTEMPORADA1\b": "TEMPORADA 1",       # OCR: fusionó palabra+número
    r"\bTEMPORADA(\d+)\b": "TEMPORADA \1", # OCR: "TEMPORADA7" sin espacio
    r"\btuis\b": "tus",
    r"\bvisitos\b": "visitas",
    r"\bporu\b": "para",
    r"\bAdhniaistrur\b": "Administrar",
    r"\belscon\b": "el scan",
    r"\bIo web\b": "la web",
    r"\bJPuede\b": "¿Puede",
    r"\bSIEMPPE\b": "SIEMPRE",
    r"\bEMPPEDECIBLE\b": "IMPREDECIBLE",
    r"\bPELACIONADO\b": "RELACIONADO",
    r"\bTRAJCÓN\b": "TRAICIÓN",
    r"\bTRAIQÓN\b": "TRAICIÓN",  # OCR: 'C' misread as 'Q'
    r"\b@NCO\b": "CINCO",
    r"\bNCO\b": "CINCO",        # OCR: 'CI' misread as '@' then stripped
    r"\bNC0\b": "CINCO",
    r"\bLaaYUDa\b": "La ayuda",
    r"\bCavBrE\b": "Cabre",
    r"\bHANTENÍA\b": "MANTENÍA",
    r"\bNEŒSITABA\b": "NECESITABA",
    r"\bNEŒSTO\b": "NECESITO",
    r"\bMucha\s+DOdaS\b": "Muchas dudas",
    r"\bUn\s+Scomumano\b": "Un ser humano",
    r"\bcnar\b": "criar",  # OCR confusion: 'ri' misread as 'n'
    r"\bccrrettimerte\b": "correctamente",  # OCR: stylized font compression
    r"\baiar\b": "criar",  # OCR: 'ri' misread as 'ai'
    r"\bMUESIRA\b": "MUESTRA",  # OCR: 'T' misread as 'I'
    r"\bPaDRiNO\b": "PADRINO",  # OCR: mixed case
    r"\bScomumano\b": "ser humano",  # OCR: fused words
    r"\bConFguroción\b": "Configuración",  # OCR: 'nfig' misread as 'Fgur'
    r"\bconfguroción\b": "configuración",  # OCR: lowercase variant
    r"\b0\b": "a",  # OCR: standalone 'a' misread as digit '0'
    r"\bmuesira\b": "muestra",  # OCR: 'T' misread as 'I' (lowercase)
    r"\bScomunano\b": "ser humano",  # OCR: fusion n-variant
    r"\bshInel\b": "Shinel",  # OCR: username handle mixed case
    r"\b@\b": "",              # OCR: @ solitario (ruido de escaneo)
    r"@": "",                  # OCR: @ en cualquier posición (ruido decorativo)
    r"\bC0M0\b": "COMO",          # OCR: 'O' misread as '0'
    r"\bC0RRECTAMENTE\b": "CORRECTAMENTE",
    r"\bC0RREC\b": "CORREC",
    r"\bSer1e\b": "Serie",          # OCR: 'i' misread as '1'
    r"\b5in\b": "sin",             # OCR: 's' misread as '5'
    r"\bR\b": "Y",                 # OCR: 'Y' misread as 'R'
}

# Versión pre-compilada (evita re.compile() implícito en cada llamada)
GLOSARIO_REGEX: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in GLOSARIO_PRE.items()
]


# ─── Patrones de ruido en márgenes (fecha/hora/numeración) ───────
MARGIN_NOISE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}'),
    re.compile(r'\d{1,2}[/.\-]\d{1,2}1?\d{2}\b'),
    re.compile(r'\d{1,2}[:.]\d{2}\s*([ap]\.?\s?m\.?)?', re.IGNORECASE),
    re.compile(r'^\d{1,4}\s*/\s*\d{1,4}$'),
    re.compile(r'\b\d{1,4}\s+de\s+\d{1,4}\b', re.IGNORECASE),
    re.compile(r'\bp[aá]g(?:ina)?\.?\s?\d{1,4}\b', re.IGNORECASE),
    re.compile(r'\b(?:p[aá]g(?:ina)?|page)\s+\d+\s+(?:de|of)\s+\d+\b', re.IGNORECASE),
    # Metadatos de exportación y timestamps en márgenes (ej. 20260713-11032519C, 13726, 458pm)
    re.compile(r'\b\d{8,14}[A-Za-z0-9_\-]*\b'),
    re.compile(r'\b\d{1,6}\s*[,.]?\s*\d{1,4}\s*p\.?m\.?\b', re.IGNORECASE),
    re.compile(r'\b\d{1,4}\s*p\.?m\.?\b', re.IGNORECASE),
]

# Patrones de marcas de agua globales (sellos de grupos de escaneo)
# NOTA: Nombres de grupos de scanlation ("olympus", "scanlation") NO se incluyen
# porque aparecen legítimamente en títulos de capítulo. Solo se filtran patrones
# específicos de sellos/watermarks.
WATERMARK_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r'zonaolympus[\s-]?com', re.IGNORECASE),
    re.compile(r'\b1\s*[\s-]?c\s*[\s-]?2\s*[\s-]?e\b', re.IGNORECASE),
    # Broken "http://" — OCR mangles "https://" into "htps fo", "htp ://", "htpsjj" etc.
    re.compile(r'\bhtps?\s*[:\s/\'"\\\\]', re.IGNORECASE),
    # Domain with underscore/apostrophe instead of dot before TLD: "xyz_com", "site'com"
    re.compile(r'[a-z]+[_\'"\s]\s*(?:com|net|org|xyz|io)\b', re.IGNORECASE),
]


# ─── Glosario post-traducción para corregir salidas literales de CT2 ──
# CT2 (OPUS-MT) tiende a traducciones literales. Este glosario aplica
# correcciones específicas para términos comunes de manga.
# Formato: (patron_regex, reemplazo) — aplicado con re.IGNORECASE
GLOSARIO_POST: Final[list[tuple[str, str]]] = [
    # Términos de capítulos/episodios
    (r"\bTEMPORARY\s+(\d+)\b", r"SEASON \1"),          # TEMPORADA 7 → SEASON 7
    (r"\bTEMPORARILY\s+(\d+)\b", r"SEASON \1"),         # variante con espacio
    (r"\bTEMPORARY(\d+)\b", r"SEASON \1"),                # TEMPORADA1 → SEASON 1 (sin espacio)
    (r"\bTEMPORARILY(\d+)\b", r"SEASON \1"),              # variante sin espacio
    # Términos de scanlation
    (r"\bSCAN\b", r"scan"),                                # normalizar mayúsculas
    (r"\bSCANLATION\b", r"scanlation"),
    # Términos de configuración/página
    (r"\bCONFIGURATION\b", r"Settings"),
    (r"\bPAGE\b", r"page"),
    # Ordinales — CT2 tiende a dejar las abreviaturas literales
    # 1er, 1ro, 1ero → First; 1a → First (femenino)
    (r"\b1[erro]{1,3}\b", r"First"),   # 1er, 1ro, 1ero, 1o
    (r"\b1[da]\b", r"First"),           # 1a, 1d
    (r"\b2[do]{1,3}\b", r"Second"),    # 2do, 2o (antes {2,3} omitía "2o")
    (r"\b2[da]\b", r"Second"),          # 2a, 2d
    (r"\b3[erro]{1,3}\b", r"Third"),   # 3er, 3ro, 3ero, 3o
    (r"\b3[da]\b", r"Third"),           # 3a, 3d
    # Ordinales completos (español literal) que CT2 a veces preserva
    (r"\bPRIMERO\b", r"FIRST"),
    (r"\bPRIMERA\b", r"FIRST"),         # femenino
    (r"\bSEGUNDO\b", r"SECOND"),
    (r"\bSEGUNDA\b", r"SECOND"),        # femenino
    (r"\bTERCERO\b", r"THIRD"),
    (r"\bTERCERA\b", r"THIRD"),         # femenino
]


# ─── Security headers (CSP, Brave Leo opt-out) ───────────────────
CSP_POLICY: Final[str] = (
    "default-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com "
    "https://docs.opencv.org "
    "https://fonts.googleapis.com https://fonts.gstatic.com "
    "data: blob:; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com "
    "https://docs.opencv.org; "
    "worker-src 'self' blob: https://cdnjs.cloudflare.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' http://127.0.0.1:5174 https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://unpkg.com data:;"
    "frame-ancestors 'none'; "
    "form-action 'self';"
)
