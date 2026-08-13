"""etiquetar_con_vlm.py — Teacher de la destilación: pseudo-etiquetas YOLO.

PLAN_MANGA_OCR Paso 7 (destilación): el daemon Unlimited-OCR (127.0.0.1:5177)
es el TEACHER — detecta regiones de texto incluyendo el diálogo artístico que
los detectores baratos pierden (2-8 min/pág, GPU). Este script corre el daemon
sobre páginas reales de un PDF y escribe, para cada página:

  train_data/vlm/{train,val}/images/pNNN.jpg      (render zoom 2.0)
  train_data/vlm/{train,val}/labels/pNNN.txt      (cls cx cy w h normalizadas)
  train_data/vlm/manifest.json                    (trazabilidad: caja, texto, clase)

Clases (mismo orden que ogkalu comic-speech-bubble-detector):
  text_bubble=0, text_free=1. La clase de cada bloque del teacher se decide con
  un oráculo: si el bloque solapa una detección del modelo ogkalu ACTUAL
  (IoU>0.3), hereda su clase (el student aprende la semántica del pretrained);
  si no, se usa el type semántico del VLM (title/header → texto libre, text →
  globo). Así el student aprende del VLM la COBERTURA (regiones que ogkalu no
  ve) sin perder la semántica de clases ya aprendida.

El resultado lo consume tools/entrenar_detector.py (student, fine-tune).

Uso:
  env/Scripts/python.exe tools/etiquetar_con_vlm.py \
      --pdf "Capítulo 43 de Cómo criar villanos correctamente.pdf" --pages 1-16
  env/Scripts/python.exe tools/etiquetar_con_vlm.py \
      --carpeta "input_manga/BookDownloads/…/capitulo"   # capítulo en .webp

La fuente puede ser un PDF (--pdf) o una CARPETA de imágenes (--carpeta,
páginas en orden natural 0.webp, 1.webp, …, 10.webp — p.ej. los mangas de
input_manga/). Cada documento usa un PREFIJO propio en los nombres de página
({{prefijo}}_pNNN) para que capítulos distintos no colisionen entre sí en
el manifest ni en train/val — el append puede acumular varias series.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Clases YOLO del modelo ogkalu (orden del archivo .pt):
CLASES = ["text_bubble", "text_free"]
IDX_BUBBLE, IDX_FREE = 0, 1

# Filtros de calidad del bloque del teacher (px a zoom 2.0):
MIN_BLOQUE_W = 15
MIN_BLOQUE_H = 12
MAX_AREA_RATIO = 0.60   # bloques sospechosamente gigantes (parse error del daemon)
ORACULO_IOU = 0.30      # IoU con ogkalu para heredar clase
DEDUP_IOU = 0.50        # dos etiquetas con más overlap → conservar la mayor


def _render_pagina(pdf, idx0: int, zoom: float) -> np.ndarray:
    """Render de una página del PDF a BGR (patrón manga_ocr.py/process_all_pages)."""
    import cv2
    import fitz

    pix = pdf.load_page(idx0).get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n >= 3:
        img = img[:, :, :3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


_EXT_IMAGENES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def _orden_natural(nombre: str) -> list:
    """Clave de orden natural: '2.webp' < '10.webp' (mismo esquema que
    manga_ocr.py — las páginas de un capítulo en carpetas vienen así)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", nombre)]


def _paginas_de_carpeta(carpeta: Path) -> list[Path]:
    """Imágenes soportadas dentro de `carpeta` (solo nivel directo), orden
    natural — cada carpeta de capítulo es un documento de páginas."""
    return sorted(
        (p for p in carpeta.iterdir() if p.is_file()
         and p.suffix.lower() in _EXT_IMAGENES),
        key=lambda p: _orden_natural(p.name))


def _cargar_imagen(path: Path) -> np.ndarray:
    """Imagen (png/jpg/webp) → BGR."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"no se pudo leer la imagen: {path}")
    return img


def _prefijo_documento(fuente: str) -> str:
    """Prefijo de página único por documento: stem sanitizado (los nombres de
    carpeta de capítulo son IDs numéricos → únicos entre series)."""
    stem = Path(fuente).stem
    limpio = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    return (limpio or "doc")[:24]


def _clasificar_bloque(b: dict, regiones_yolo: list[dict]) -> int:
    """Clase YOLO (0 bubble / 1 free) para un bloque del teacher.

    Oráculo ogkalu primero (IoU > ORACULO_IOU): hereda la clase del pretrained —
    el estudiante aprende la semántica de clases ya entrenada. Sin overlap:
    type semántico del VLM (title/header = texto libre; text = globo).
    """
    import ocr_utils

    mejor_iou = 0.0
    mejor_label = ""
    for r in regiones_yolo:
        iou = ocr_utils._overlap_ratio(b, r)
        if iou > mejor_iou:
            mejor_iou = iou
            mejor_label = str(r.get("label", "")).lower()
    if mejor_iou > ORACULO_IOU:
        return IDX_FREE if "free" in mejor_label else IDX_BUBBLE
    btype = str(b.get("type", "text")).lower()
    return IDX_FREE if btype in ("title", "header") else IDX_BUBBLE


def _bloques_a_labels(
    bloques: list[dict],
    regiones_yolo: list[dict],
    w_pag: int,
    h_pag: int,
) -> list[dict]:
    """Filtra y convierte bloques VLM → etiquetas con coords YOLO normalizadas."""
    area_pag = float(w_pag * h_pag)
    out: list[dict] = []
    for b in bloques:
        bw, bh = int(b["w"]), int(b["h"])
        bx, by = int(b["x"]), int(b["y"])
        if bw < MIN_BLOQUE_W or bh < MIN_BLOQUE_H:
            continue
        if (bw * bh) > area_pag * MAX_AREA_RATIO:
            continue
        if bx + bw > w_pag * 1.5 or by + bh > h_pag * 1.5:  # fuera de página (raro)
            continue
        cls = _clasificar_bloque(b, regiones_yolo)
        cx = min(max((bx + bw / 2) / w_pag, 0.0), 1.0)
        cy = min(max((by + bh / 2) / h_pag, 0.0), 1.0)
        wn = min(bw / w_pag, 1.0)
        hn = min(bh / h_pag, 1.0)
        out.append({
            "cls": cls, "cx": cx, "cy": cy, "w": wn, "h": hn,
            "x": bx, "y": by, "w_px": bw, "h_px": bh,
            "text": str(b.get("text", ""))[:40],
            "type": str(b.get("type", "text")),
        })
    return _dedup_etiquetas(out)


def _dedup_etiquetas(labels: list[dict], iou_thresh: float = DEDUP_IOU) -> list[dict]:
    """Dedup por solape: de dos etiquetas con IoU > umbral, conserva la mayor
    (las detecciones duplicadas del mismo texto las absorbe la caja más grande)."""
    import ocr_utils

    keep: list[dict] = []
    for lab in sorted(labels, key=lambda x: x["w_px"] * x["h_px"], reverse=True):
        b1 = {"x": lab["x"], "y": lab["y"], "w": lab["w_px"], "h": lab["h_px"]}
        if any(ocr_utils._overlap_ratio(
                b1, {"x": k["x"], "y": k["y"], "w": k["w_px"], "h": k["h_px"]})
                > iou_thresh for k in keep):
            continue
        keep.append(lab)
    return keep


def _escribir_pagina(split_dir: Path, nombre: str, img: np.ndarray,
                     labels: list[dict], manifest_row: dict) -> None:
    """Escribe imagen JPG + labels YOLO de una página (si hay etiquetas)."""
    import cv2

    img_dir = split_dir / "images"
    lab_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{nombre}.jpg"
    lab_path = lab_dir / f"{nombre}.txt"
    cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    with open(lab_path, "w", encoding="utf-8") as f:
        for l in labels:
            f.write(f"{l['cls']} {l['cx']:.6f} {l['cy']:.6f} "
                    f"{l['w']:.6f} {l['h']:.6f}\n")
    manifest_row.update({"imagen": img_path.name, "etiquetas": lab_path.name})


def _daemon_ocr(img: np.ndarray):
    """Llama al daemon VLM (teacher). Acceso en runtime a routes.api para
    respetar los mocks de pytest (mismo patrón que ocr_engine)."""
    import routes.api as ra
    return ra._ocr_with_unlimited(img)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Teacher VLM → pseudo-etiquetas YOLO (text_bubble/text_free).")
    ap.add_argument("--pdf", default=None, help="PDF del manga (o --carpeta)")
    ap.add_argument("--carpeta", default=None,
                    help="carpeta de imágenes del capítulo (.webp/png/jpg, "
                         "páginas en orden natural)")
    ap.add_argument("--pages", default=None, help="Rango 1-indexado (p.ej. 1-16)")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "train_data" / "vlm"))
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fracción de páginas para validación (split determinista)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true",
                    help=("ANADE al dataset existente: las paginas nuevas van a "
                          "train/val y las ya presentes conservan su split. "
                          "Sin el flag, se re-crea el split completo con el rango."))
    args = ap.parse_args()

    if bool(args.pdf) == bool(args.carpeta):
        ap.error("debes pasar exactamente uno de --pdf o --carpeta")
    fuente = args.pdf or args.carpeta
    prefijo = _prefijo_documento(fuente)

    out = Path(args.out)
    rango = None
    if args.pages:
        a, _, b = args.pages.partition("-")
        rango = (int(a), int(b))
    print(f"[teacher] daemon VLM → {out} ({fuente}, páginas {rango or 'todas'}, "
          f"prefijo '{prefijo}')")

    if args.carpeta:
        paginas_src = [(n, _cargar_imagen(p))
                       for n, p in enumerate(_paginas_de_carpeta(
                           Path(args.carpeta)), 1)]
        if rango:
            paginas_src = [(n, img) for (n, img) in paginas_src
                           if rango[0] <= n <= rango[1]]
        total = len(paginas_src)
    else:
        import fitz

        doc = fitz.open(str(args.pdf))
        total = doc.page_count
        paginas_src = [(n, _render_pagina(doc, idx0, args.zoom))
                       for idx0 in range(doc.page_count)
                       for n in [idx0 + 1]
                       if not rango or rango[0] <= n <= rango[1]]
    paginas_procesadas: list[tuple[int, np.ndarray, list[dict], dict]] = []
    for n, img in paginas_src:
        print(f"[teacher] página {n}/{total}... ", end="", flush=True)
        try:
            ublocks, _panels, t_s = _daemon_ocr(img)
        except Exception as e:
            print(f"ERROR daemon: {e}; salto")
            continue
        # Oráculo de clases: detecciones del modelo ogkalu actual.
        import ocr_utils
        regiones_yolo = ocr_utils._detect_text_regions_in_page(img)
        h_pag, w_pag = img.shape[:2]
        labels = _bloques_a_labels(ublocks, regiones_yolo, w_pag, h_pag)
        if not labels:
            print(f"0 etiquetas ({len(ublocks)} bloques crudos); salto")
            continue
        nombre = f"{prefijo}_p{n:03d}"
        row = {
            "pagina": n, "daemon_s": round(t_s, 1),
            "bloques_crudos": len(ublocks), "yolo_oraculo": len(regiones_yolo),
            "bloques": [{"x": l["x"], "y": l["y"], "w": l["w_px"], "h": l["h_px"],
                         "text": l["text"], "type": l["type"], "cls": l["cls"]}
                        for l in labels],
        }
        paginas_procesadas.append((n, img, labels, row))
        print(f"{len(labels)} etiquetas ({t_s:.0f}s daemon)")

    if not paginas_procesadas:
        print("[teacher] Sin páginas procesables — revisa el rango y el daemon.")
        sys.exit(1)

    # Split determinista (semilla fija) por número de página. Con --append, las
    # páginas ya presentes en manifest.json conservan su split (el dataset crece
    # por capítulo sin reorganizar lo ya etiquetado — el val se mantiene estable
    # para A/B honesto); las NUEVAS se reparten train/val por --val-frac.
    rng = random.Random(args.seed)
    orden = sorted(paginas_procesadas, key=lambda x: x[0])
    manifest_prev: dict = {}
    if args.append and (out / "manifest.json").exists():
        try:
            manifest_prev = json.loads(
                (out / "manifest.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest_prev = {}
    prev_split = {row.get("imagen", "").split(".")[0]: row.get("split")
                  for row in manifest_prev.get("paginas", [])}
    splits = {"train": [], "val": []}
    nuevas = [(n, img, labels, row) for (n, img, labels, row) in orden
              if f"{prefijo}_p{n:03d}" not in prev_split]
    n_val = max(1, round(len(nuevas) * args.val_frac))
    indices_val = set(range(len(nuevas) - n_val, len(nuevas)))
    for i, (n, img, labels, row) in enumerate(orden):
        if f"{prefijo}_p{n:03d}" in prev_split:
            split = prev_split[f"{prefijo}_p{n:03d}"]
        else:
            split = "val" if i in indices_val else "train"
        splits[split].append((n, img, labels, row))

    manifest: dict = {"fuente": fuente, "clases": CLASES,
                      "paginas": list(manifest_prev.get("paginas", []))}
    for split, items in splits.items():
        split_dir = out / split
        for n, img, labels, row in items:
            _escribir_pagina(split_dir, f"{prefijo}_p{n:03d}", img, labels, row)
            row["split"] = split
            if f"{prefijo}_p{n:03d}" not in prev_split:
                manifest["paginas"].append(row)
            print(f"[teacher] {split}/{prefijo}_p{n:03d}: {len(labels)} etiquetas")
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_lab = sum(len(t[2]) for items in splits.values() for t in items)
    print(f"[teacher] OK: {n_train} train + {n_val} val → {out} "
          f"({n_lab} etiquetas en total)")


if __name__ == "__main__":
    main()
