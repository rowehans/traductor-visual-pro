"""calificar_detector.py — Califica cuánto texto real encuentra el detector.

El bucle de mejora del usuario:

  1. El usuario corrige/añade cajas a mano en X-AnyLabeling (workspace de
     tools/exportar_anotaciones.py → train_data/corregir/). Eso es el "marcado":
     la respuesta correcta de qué texto hay que traducir en cada página.
  2. Este script corre el modelo ACTUAL sobre ese marcado y da una NOTA (0-100)
     de cuánto se acercó a encontrar TODO el texto: recall (cajas GT cubiertas),
     precision (cajas extra), y la lista de páginas donde el modelo no encontró
     traducción que sí existía ("páginas con diálogo perdido").
  3. Cada ronda se guarda en el historial (train_data/calificaciones.json), así
     que se ve la nota subir ronda a ronda hasta llegar a la meta.
  4. tools/fusionar_correcciones.py --train re-entrena con ese oro y el ciclo
     se repite: corregir → calificar → reentrenar → calificar...

Uso:
  env/Scripts/python.exe tools/calificar_detector.py
  env/Scripts/python.exe tools/calificar_detector.py --model models/finetune_synth/weights/best.pt --etiqueta "ronda 2: synth_solo"
  env/Scripts/python.exe tools/calificar_detector.py --nota-minima 90
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

CLASES = ["text_bubble", "text_free"]


def _leer_gt(img: Path, labels_dir: Path) -> list[dict]:
    """Etiquetas GT YOLO del marcado manual: {labels_dir}/{img.stem}.txt con
    formato 'cls cx cy w h' normalizado → cajas {x,y,w,h} absolutas en px."""
    import cv2

    lab = labels_dir / f"{img.stem}.txt"
    if not lab.exists():
        return []
    h_pag, w_pag = cv2.imread(str(img)).shape[:2]
    gt: list[dict] = []
    for line in lab.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0])) if p[0] else 0
        _, cx, cy, bw, bh = (float(v) for v in p[:5])
        gt.append({"cls": cls,
                   "x": (cx - bw / 2) * w_pag, "y": (cy - bh / 2) * h_pag,
                   "w": bw * w_pag, "h": bh * h_pag})
    return gt


def _overlap_ratio(a: dict, b: dict) -> float:
    """Ratio de solapamiento espacial (inter/min_area) — la MISMA métrica que
    usa el pipeline real (ocr_utils._overlap_ratio) y el A/B del trainer
    (_eval_rapida con umbral 0.3). Para cajas anidadas (un globo dentro de la
    región que lo contiene) da ~1.0 aunque la región sea enorme; el IoU plano
    (inter/unión) castigaría las GT grandes y reportaría "el modelo no
    encuentra nada" cuando sí lo encuentra (lección sesión 161)."""
    x0 = max(a["x"], b["x"])
    y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    min_area = min(a["w"] * a["h"], b["w"] * b["h"])
    return inter / float(min_area) if min_area > 0 else 0.0


def _evaluar_pagina(boxes: list[dict], gt: list[dict], overlap_min: float) -> dict:
    """Compara las detecciones del modelo contra el marcado real de una página
    con la métrica canónica del pipeline (overlap_ratio inter/min_area).

    - hits:  GT cubierto por al menos una detección (ratio > overlap_min) →
             texto que el modelo SÍ encontró.
    - fns:   GT sin cubrir → texto que faltaba traducir y el modelo NO encontró.
    - fps:   detecciones que no cubren ninguna GT → cajas extra (falsos positivos).
    """
    hit_gt: set[int] = set()
    matched_det: set[int] = set()
    for gi, g in enumerate(gt):
        for di, d in enumerate(boxes):
            if di in matched_det:
                continue
            if _overlap_ratio(g, d) > overlap_min:
                hit_gt.add(gi)
                matched_det.add(di)
                break
    res = {
        "gt": len(gt),
        "detecciones": len(boxes),
        "hits": len(hit_gt),
        "fns": len(gt) - len(hit_gt),
        "fps": len(boxes) - len(matched_det),
        "recall": len(hit_gt) / len(gt) if gt else 1.0,
        "precision": (len(matched_det) / len(boxes)) if boxes else 1.0,
    }
    # desglose por clase (0=text_bubble, 1=text_free) — para responder si el
    # modelo detecta mejor el texto libre que los globos
    idx_cls = {0: [], 1: []}
    for i, g in enumerate(gt):
        idx_cls[g.get("cls", 0)].append(i)
    for cls, nombre in ((0, "bubble"), (1, "free")):
        res[f"gt_{nombre}"] = len(idx_cls[cls])
        res[f"hit_{nombre}"] = sum(1 for i in idx_cls[cls] if i in hit_gt)
    return res


def _evaluar_workspace(model: Path, images: list[Path], labels_dir: Path,
                       conf: float, imgsz: int, overlap_min: float,
                       device_arg: str = "auto") -> dict:
    """Corre el modelo sobre todas las páginas con marcado y agrega el resumen."""
    from ultralytics import YOLO

    import ocr_utils

    m = YOLO(str(model))
    # mismo dispositivo que el pipeline real (fase de detección) salvo que
    # --device lo fuerce (útil cuando el daemon ocupa la VRAM)
    if device_arg == "auto":
        try:
            device = ocr_utils._resolver_device_yolo()  # type: ignore[attr-defined]
        except AttributeError:
            device = "cpu"
    else:
        device = device_arg

    paginas: list[dict] = []
    agg = {"gt": 0, "detecciones": 0, "hits": 0, "fns": 0, "fps": 0,
           "gt_bubble": 0, "hit_bubble": 0, "gt_free": 0, "hit_free": 0}
    gt_gigantes = 0   # cajas del marcado >25% del área de la página
    for img in sorted(images):
        gt = _leer_gt(img, labels_dir)
        import cv2
        h_pag, w_pag = cv2.imread(str(img)).shape[:2]
        for g in gt:
            if g["w"] * g["h"] > 0.25 * (w_pag * h_pag):
                gt_gigantes += 1
        agg["gt"] += len(gt)
        boxes = _predecir(m, img, conf, imgsz, device)
        agg["detecciones"] += len(boxes)
        res = _evaluar_pagina(boxes, gt, overlap_min)
        agg["hits"] += res["hits"]
        agg["fns"] += res["fns"]
        agg["fps"] += res["fps"]
        for cls in ("bubble", "free"):
            agg[f"gt_{cls}"] += res[f"gt_{cls}"]
            agg[f"hit_{cls}"] += res[f"hit_{cls}"]
        paginas.append({"pagina": img.stem, **res})

    total_gt = agg["gt"]
    recall = agg["hits"] / total_gt if total_gt else 1.0
    precision = (agg["hits"] / agg["detecciones"]) if agg["detecciones"] else 1.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)
    # páginas donde HABÍA traducción pero el modelo no encontró NADA
    perdidas = [p for p in paginas if p["gt"] > 0 and p["hits"] == 0]
    paginas.sort(key=lambda p: (p["recall"], -p["gt"]))
    return {
        "modelo": str(model),
        "conf": conf,
        "overlap": overlap_min,
        "paginas_evaluadas": len(paginas),
        "paginas": paginas,
        "gt_total": total_gt,
        "gt_gigantes": gt_gigantes,
        "detecciones": agg["detecciones"],
        "hits": agg["hits"],
        "fns": agg["fns"],
        "fps": agg["fps"],
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "recall_bubble": (agg["hit_bubble"] / agg["gt_bubble"]
                           if agg["gt_bubble"] else 1.0),
        "recall_free": (agg["hit_free"] / agg["gt_free"]
                         if agg["gt_free"] else 1.0),
        "gt_bubble": agg["gt_bubble"],
        "hit_bubble": agg["hit_bubble"],
        "gt_free": agg["gt_free"],
        "hit_free": agg["hit_free"],
        "nota": round(100 * recall),
        "paginas_perdidas_dialogo": [
            {"pagina": p["pagina"], "gt": p["gt"]} for p in perdidas],
        "peores_paginas": [
            {"pagina": p["pagina"], "recall": p["recall"], "fns": p["fns"],
             "gt": p["gt"]} for p in paginas[:5]],
    }


def _predecir(m, img: Path, conf: float, imgsz: int, device: str) -> list[dict]:
    """Corre el modelo sobre una imagen y devuelve las cajas {x,y,w,h}."""
    r = m.predict(str(img), conf=conf, imgsz=imgsz, device=device,
                  verbose=False)[0]
    boxes: list[dict] = []
    if r.boxes is not None:
        for box in r.boxes.xyxy:
            x0, y0, x1, y1 = box.tolist()
            boxes.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return boxes


def _letterbox(img: Path, tile_w: int) -> tuple | None:
    """Resize + letterbox de una página sobre un lienzo tile_w x tile_h
    (ratio manga aproximado). Devuelve (lienzo, sx, sy, ancho_pag, alto_pag)
    donde ancho/alto_pag son las dimensiones ESCALADAS de la página dentro
    del lienzo (la página se centra; fuera de ella hay banda gris) — o None
    si no lee."""
    import cv2

    img_bgr = cv2.imread(str(img))
    if img_bgr is None:
        return None
    h0, w0 = img_bgr.shape[:2]
    tile_h = max(1, round(tile_w / 0.7))        # ratio manga aproximado
    sx = min(tile_w / w0, tile_h / h0)
    nuevo = (round(w0 * sx), round(h0 * sx))
    img_bgr = cv2.resize(img_bgr, nuevo)
    lienzo = cv2.copyMakeBorder(img_bgr, 0, tile_h - nuevo[1], 0,
                                tile_w - nuevo[0], cv2.BORDER_CONSTANT,
                                value=(60, 60, 60))
    return lienzo, sx, sx, nuevo[0], nuevo[1]


def _dibujar_cajas(lienzo, cajas: list[dict], sx: float, sy: float,
                   color, grosor: int = 2, relleno: float | None = None,
                   ancho_pag: int | None = None,
                   alto_pag: int | None = None) -> None:
    """Dibuja cajas {x,y,w,h} (y cls) escaladas sobre el lienzo.

    - `relleno` (0-1): pinta un relleno TRANSLÚCIDO del color para que las
      regiones grandes del oro (p. ej. w=1.00 del teacher) NO tapen el arte
      de la página — se ve la caja y lo que hay debajo.
    - `ancho_pag`/`alto_pag`: si se pasan, las cajas se RECORTAN a la página
      (nunca pintan sobre la banda letterbox gris).
    """
    import cv2

    xmax = ancho_pag if ancho_pag is not None else 1 << 30
    ymax = alto_pag if alto_pag is not None else 1 << 30
    for c in cajas:
        x0 = max(0, round(c["x"] * sx))
        y0 = max(0, round(c["y"] * sy))
        x1 = min(xmax, round((c["x"] + c["w"]) * sx))
        y1 = min(ymax, round((c["y"] + c["h"]) * sy))
        if x1 <= x0 or y1 <= y0:
            continue
        if relleno:
            overlay = lienzo.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            cv2.addWeighted(overlay, relleno, lienzo, 1 - relleno, 0,
                            lienzo)
        cv2.rectangle(lienzo, (x0, y0), (x1, y1), color, grosor)


def _cabecera(lienzo, titulo: str) -> "object":
    """Franja oscura con el título encima de la imagen (ASCII)."""
    import cv2

    con_cab = cv2.copyMakeBorder(lienzo, 26, 0, 0, 0, cv2.BORDER_CONSTANT,
                                 value=(30, 30, 30))
    cv2.putText(con_cab, titulo, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return con_cab


def _celda_grid(img: Path, gt: list[dict], dets: list[dict], titulo: str,
                tile_w: int) -> "object":
    """Renderiza una página con las cajas del oro (verde/magenta por clase) y
    las detecciones del modelo (rojo) superpuestas, más cabecera. Devuelve la
    imagen de la celda (letterboxed)."""
    lb = _letterbox(img, tile_w)
    if lb is None:
        return None
    lienzo, sx, sy, w_pag, h_pag = lb
    grosor = max(2, round(tile_w / 120))
    for g in gt:
        # oro por clase: globos verde, texto libre magenta — con relleno
        # translúcido para que las regiones grandes no tapen el arte
        color = (255, 0, 255) if g.get("cls", 0) == 1 else (0, 255, 0)
        _dibujar_cajas(lienzo, [g], sx, sy, color, grosor=grosor,
                       relleno=0.18, ancho_pag=w_pag, alto_pag=h_pag)
    _dibujar_cajas(lienzo, dets, sx, sy, (0, 0, 255), grosor=grosor,
                   relleno=0.12, ancho_pag=w_pag, alto_pag=h_pag)
    return _cabecera(lienzo, titulo)


def _par_comparativa(img: Path, gt: list[dict], dets: list[dict],
                     titulo: str, tile_w: int) -> "object":
    """Par lado a lado de UNA página: [ORO | MODELO] — el oro con sus cajas
    (verde globos, magenta texto libre) a la izquierda, y las detecciones del
    deep learning (rojo) a la derecha, para ver de un vistazo qué cajas del
    oro no tiene el modelo."""
    import cv2

    izq = _letterbox(img, tile_w)
    der = _letterbox(img, tile_w)
    if izq is None or der is None:
        return None
    l, sx, sy, w_pag, h_pag = izq
    grosor = max(2, round(tile_w / 120))
    for g in gt:
        color = (255, 0, 255) if g.get("cls", 0) == 1 else (0, 255, 0)
        _dibujar_cajas(l, [g], sx, sy, color, grosor=grosor, relleno=0.18,
                       ancho_pag=w_pag, alto_pag=h_pag)
    r, sx2, sy2, w_pag2, h_pag2 = der
    _dibujar_cajas(r, dets, sx2, sy2, (0, 0, 255), grosor=grosor,
                   relleno=0.12, ancho_pag=w_pag2, alto_pag=h_pag2)
    par = cv2.hconcat([l, r])
    return _cabecera(par, f"{titulo}   [ORO | MODELO]")


def _generar_grid(model: Path, paginas: list[dict], labels_dir: Path,
                  conf: float, imgsz: int, device_arg: str,
                  out_path: Path, cols: int = 4, tile_w: int = 320,
                  max_paginas: int = 16) -> Path | None:
    """Montaje visual: grid de miniaturas de las páginas con texto perdido,
    con el marcado del usuario (verde) y las detecciones del modelo (rojo)
    superpuestas — para ver de un vistazo qué cajas no encontró."""
    import cv2

    from ultralytics import YOLO

    import ocr_utils

    # peores primero: páginas que perdieron texto (fns>0), con GT>0
    seleccion = [p for p in paginas if p["gt"] > 0 and p["fns"] > 0]
    seleccion.sort(key=lambda p: (p["recall"], -p["gt"]))
    seleccion = seleccion[:max_paginas]
    if not seleccion:
        return None

    m = YOLO(str(model))
    if device_arg == "auto":
        try:
            device = ocr_utils._resolver_device_yolo()  # type: ignore[attr-defined]
        except AttributeError:
            device = "cpu"
    else:
        device = device_arg

    images_dir = labels_dir.parent / "images"
    celdas: list = []
    for p in seleccion:
        imgs = sorted(images_dir.glob(f"{p['pagina']}.*"))
        if not imgs:
            continue
        gt = _leer_gt(imgs[0], labels_dir)
        dets = _predecir(m, imgs[0], conf, imgsz, device)
        titulo = _titulo_cabecera(p)
        celda = _celda_grid(imgs[0], gt, dets, titulo, tile_w)
        if celda is not None:
            celdas.append(celda)
    if not celdas:
        return None

    ancho_grid = cols * tile_w
    filas = [_leyenda(ancho_grid)]
    for i in range(0, len(celdas), cols):
        fila = celdas[i:i + cols]
        while len(fila) < cols:
            fila.append(_celda_vacia(tile_w, celdas[0].shape[0]))
        filas.append(cv2.hconcat(fila))
    grid = cv2.vconcat(filas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path


def _generar_comparativa(model: Path, paginas: list[dict], labels_dir: Path,
                         conf: float, imgsz: int, device_arg: str,
                         out_path: Path, tile_w: int = 480,
                         max_paginas: int = 8) -> Path | None:
    """Montaje lado a lado [ORO | MODELO] de las páginas con texto perdido:
    cada fila es una página con el oro (verde/magenta) a la izquierda y las
    detecciones del modelo (rojo) a la derecha — el preview que compara el
    marcado manual contra el deep learning."""
    import cv2

    from ultralytics import YOLO

    import ocr_utils

    seleccion = [p for p in paginas if p["gt"] > 0 and p["fns"] > 0]
    seleccion.sort(key=lambda p: (p["recall"], -p["gt"]))
    seleccion = seleccion[:max_paginas]
    if not seleccion:
        return None

    m = YOLO(str(model))
    if device_arg == "auto":
        try:
            device = ocr_utils._resolver_device_yolo()  # type: ignore[attr-defined]
        except AttributeError:
            device = "cpu"
    else:
        device = device_arg

    images_dir = labels_dir.parent / "images"
    filas: list = []
    for p in seleccion:
        imgs = sorted(images_dir.glob(f"{p['pagina']}.*"))
        if not imgs:
            continue
        gt = _leer_gt(imgs[0], labels_dir)
        dets = _predecir(m, imgs[0], conf, imgsz, device)
        titulo = _titulo_cabecera(p)
        par = _par_comparativa(imgs[0], gt, dets, titulo, tile_w)
        if par is not None:
            filas.append(par)
    if not filas:
        return None

    ancho = 2 * tile_w
    comp = cv2.vconcat([_leyenda(ancho)] + filas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), comp)
    return out_path


def _leyenda(ancho: int) -> "object":
    """Franja de leyenda con la clave de colores (ASCII: putText no soporta
    acentos). El relleno translúcido marca la REGIÓN que cubre cada caja —
    las cajas grandes del oro se ven sin tapar el arte de la página."""
    import cv2
    import numpy as np
    img = np.full((30, ancho, 3), 240, dtype=np.uint8)
    cv2.putText(img, "VERDE = globos (oro)   MAGENTA = texto libre (oro)   "
                     "ROJO = detecciones del modelo   (relleno = region cubierta)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                cv2.LINE_AA)
    return img


def _titulo_cabecera(p: dict) -> str:
    """Cabecera de página en el montaje: nombre, hits/gt, recall y desglose
    por clase (G = globos, L = texto libre) para ver de un vistazo qué
    clase pierde el modelo. ASCII porque putText no soporta acentos."""
    hb = p.get("hit_bubble", 0)
    gb = p.get("gt_bubble", 0)
    hf = p.get("hit_free", 0)
    gf = p.get("gt_free", 0)
    return (f"{p['pagina']}  {p['hits']}/{p['gt']}  "
            f"({round(p['recall']*100)}%)  "
            f"G {hb}/{gb}  L {hf}/{gf}")


def _celda_vacia(tile_w: int, alto: int) -> "object":
    import cv2
    import numpy as np
    return np.full((alto, tile_w, 3), 240, dtype=np.uint8)


def _escribir_html_preview(png_path: Path, html_path: Path, titulo: str,
                           subtitulo: str) -> Path:
    """Empaqueta la comparativa en un HTML autocontenido (base64) con la
    leyenda HONESTA de colores — exactamente los que dibuja el PNG. Sin
    colores fantasma (el antiguo preview prometía un amarillo de
    'Coincidencia oro∩modelo' que nunca se dibujó)."""
    import base64

    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>{titulo}</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; background:#1e1e1e; color:#eee; margin:0; padding:24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#aaa; font-size:13px; margin-bottom:16px; }}
  .key {{ background:#2b2b2b; border:1px solid #444; border-radius:6px; padding:10px 14px; font-size:13px; margin-bottom:16px; }}
  .sw {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin:0 6px 0 14px; vertical-align:-1px; }}
  img {{ max-width:100%; height:auto; border:1px solid #444; border-radius:8px; }}
</style></head><body>
<h1>{titulo}</h1>
<div class="sub">{subtitulo}</div>
<div class="key">
  <span class="sw" style="background:#2ecc71"></span>Globos del oro
  <span class="sw" style="background:#e84393"></span>Texto libre del oro
  <span class="sw" style="background:#e74c3c"></span>Detecciones del modelo
  <span style="margin-left:14px">Relleno translucido = region cubierta (las cajas grandes del oro se ven sin tapar el arte)</span>
</div>
<img src="data:image/png;base64,{b64}" alt="Comparativa oro vs modelo">
</body></html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _append_historia(historia: Path, ronda: dict) -> list[dict]:
    """Añade la ronda al historial acumulado (crea el archivo si no existe)."""
    if historia.exists():
        hist = json.loads(historia.read_text(encoding="utf-8"))
    else:
        hist = []
    hist.append(ronda)
    historia.parent.mkdir(parents=True, exist_ok=True)
    historia.write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    return hist


def _imprimir(ronda: dict, historia: list[dict],
              tabla_paginas: bool = False) -> None:
    print("=" * 60)
    print(f"CALIFICACIÓN del detector — nota: {ronda['nota']}/100")
    print("=" * 60)
    print(f"  Modelo      : {ronda['modelo']}")
    print(f"  Umbral conf : {ronda['conf']}  | overlap mínimo: "
          f"{ronda['overlap']}")
    print(f"  Páginas     : {ronda['paginas_evaluadas']} "
          f"(GT total: {ronda['gt_total']} cajas)")
    gt_gig = ronda.get("gt_gigantes", 0)
    if gt_gig:
        pct = 100 * gt_gig / ronda["gt_total"]
        print(f"  Marcado     : {gt_gig} cajas ({pct:.0f}%) son regiones grandes "
              ">25% de la página — el score mide cobertura de regiones, no de "
              "globos individuales (para globos apretados, dibújalos así en "
              "X-AnyLabeling)")
    print(f"  Recuperó    : {ronda['hits']}/{ronda['gt_total']} cajas "
          f"({ronda['recall']*100:.1f}% recall)")
    print(f"  Perdió      : {ronda['fns']} cajas con texto que no encontró "
          f"({ronda['fps']} extra)")
    print(f"  Precisión   : {ronda['precision']*100:.1f}%  |  "
          f"F1: {ronda['f1']*100:.1f}%")
    print(f"  Por clase   : globos {ronda['hit_bubble']}/{ronda['gt_bubble']} "
          f"({ronda['recall_bubble']*100:.0f}%) | texto libre "
          f"{ronda['hit_free']}/{ronda['gt_free']} "
          f"({ronda['recall_free']*100:.0f}%)")
    if ronda["paginas_perdidas_dialogo"]:
        print("\n  ⚠ PÁGINAS CON DIÁLOGO QUE NO ENCONTRÓ (corrígelas primero):")
        for p in ronda["paginas_perdidas_dialogo"]:
            print(f"    - {p['pagina']}  ({p['gt']} cajas de texto perdidas)")
    if ronda.get("grid"):
        print(f"\n  🖼 Montaje visual (cajas del oro en verde, detecciones del "
              f"modelo en rojo):")
        print(f"    {ronda['grid']}")
    if tabla_paginas and ronda.get("paginas"):
        print("\n  RECALL POR PÁGINA (peor primero):")
        print("    página                  hits/gt  perdidas  recall  precisión")
        for p in ronda["paginas"]:
            print(f"    {p['pagina']:<24} {p['hits']}/{p['gt']:<5} "
                  f"{p['fns']:<8} {p['recall']*100:>5.0f}%  "
                  f"{p['precision']*100:>5.0f}%")
    elif ronda["peores_paginas"]:
        print("\n  Peores páginas (menor recall):")
        for p in ronda["peores_paginas"]:
            print(f"    - {p['pagina']}: recall {p['recall']*100:.0f}% "
                  f"({p['fns']}/{p['gt']} perdidas)")
    if len(historia) > 1:
        print("\n  Historial de notas (ronda → nota):")
        for i, h in enumerate(historia, 1):
            tag = h.get("etiqueta", "")
            print(f"    {i}. {h['fecha']}  nota {h['nota']}/100  "
                  f"recall {h['recall']*100:.1f}%  {tag}")
        print(f"\n  Progreso: {historia[0]['nota']} → {historia[-1]['nota']} "
              f"({historia[-1]['nota'] - historia[0]['nota']:+d} pts)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Califica cuánto texto real (marcado manual) encuentra "
                    "el detector — el bucle corregir → calificar → reentrenar.")
    ap.add_argument("--workspace", default=str(ROOT / "train_data" / "corregir"),
                    help="Workspace con images/ y labels/ (el marcado del usuario).")
    ap.add_argument("--model", default=config.YOLO_MODEL_PATH,
                    help="Modelo a calificar (default: el de producción).")
    ap.add_argument("--conf", type=float, default=config.YOLO_CONF_THRESH,
                    help="Umbral de confianza (default: el del pipeline).")
    ap.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|0 — auto usa el mismo dispositivo que el "
                         "pipeline (default); cpu evita pelear por VRAM.")
    ap.add_argument("--overlap", type=float, default=0.3,
                    help="overlap_ratio mínimo (inter/min_area, la métrica del "
                         "pipeline) para considerar una caja encontrada.")
    ap.add_argument("--historia", default=str(ROOT / "train_data" / "calificaciones.json"),
                    help="Historial acumulado de rondas.")
    ap.add_argument("--etiqueta", default="", help="Nombre de esta ronda.")
    ap.add_argument("--grid", default="",
                    help="PNG del montaje visual de las páginas con texto "
                         "perdido (default: train_data/calificaciones_grid_<ts>.png; "
                         "--no-grid para desactivar).")
    ap.add_argument("--no-grid", action="store_true",
                    help="No generar el montaje visual.")
    ap.add_argument("--grid-cols", type=int, default=4)
    ap.add_argument("--grid-width", type=int, default=320,
                    help="Ancho de cada miniatura en píxeles.")
    ap.add_argument("--grid-max-paginas", type=int, default=16,
                    help="Máximo de páginas en el montaje (peores primero).")
    ap.add_argument("--comparativa", default="",
                    help="PNG del preview lado a lado [ORO | MODELO] de las "
                         "páginas con texto perdido (default: vacío = no "
                         "generar; p. ej. train_data/comparativa.png).")
    ap.add_argument("--comparativa-paginas", type=int, default=8,
                    help="Máximo de páginas en el preview comparativo.")
    ap.add_argument("--preview-html", default="",
                    help="HTML autocontenido del preview (con la leyenda "
                         "correcta de colores) cuando se genera una "
                         "comparativa. Ej: train_data/comparativa_preview.html")
    ap.add_argument("--paginas", default="",
                    help="Solo estas páginas del marcado (stems separados por "
                         "coma, ej. 'p002,1490498_p001'). Con el flag, el "
                         "reporte muestra el recall por página de cada una.")
    args = ap.parse_args()

    ws = Path(args.workspace)
    labels_dir = ws / "labels"
    images = sorted(
        p for p in (ws / "images").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and
        (labels_dir / f"{p.stem}.txt").exists())
    pedidas = [s.strip() for s in args.paginas.split(",") if s.strip()] \
        if args.paginas else []
    if pedidas:
        conjunto = set(pedidas)
        images = [p for p in images if p.stem in conjunto]
        faltantes = sorted(conjunto - {p.stem for p in images})
        if faltantes:
            print(f"  ⚠ Páginas pedidas sin marcado: {', '.join(faltantes)}")
        if not images:
            print(f"Ninguna de las páginas pedidas tiene marcado en {ws}.")
            return
    if not images:
        print(f"Sin páginas con marcado en {ws} — corrige primero en "
              f"X-AnyLabeling (tools/exportar_anotaciones.py).")
        return

    ronda = _evaluar_workspace(Path(args.model), images, labels_dir,
                               args.conf, args.imgsz, args.overlap,
                               args.device)
    ronda["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    ronda["etiqueta"] = args.etiqueta
    if not args.no_grid:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_path = Path(args.grid) if args.grid else \
            ROOT / "train_data" / f"calificaciones_grid_{ts}.png"
        grid = _generar_grid(Path(args.model), ronda["paginas"], labels_dir,
                             args.conf, args.imgsz, args.device, grid_path,
                             args.grid_cols, args.grid_width,
                             args.grid_max_paginas)
        if grid:
            ronda["grid"] = str(grid)
    if args.comparativa:
        comp = _generar_comparativa(Path(args.model), ronda["paginas"],
                                    labels_dir, args.conf, args.imgsz,
                                    args.device, Path(args.comparativa),
                                    tile_w=480,
                                    max_paginas=args.comparativa_paginas)
        if comp:
            ronda["comparativa"] = str(comp)
            print(f"\n  🖼 Preview [ORO | MODELO]: {comp}")
            if args.preview_html:
                html = _escribir_html_preview(
                    comp, Path(args.preview_html),
                    "Comparativa: tu marcado (oro) vs el deep learning",
                    "Cada fila = una página con diálogo perdido · izquierda = "
                    "oro · derecha = lo que detectó el modelo (ogkalu, "
                    "producción)")
                print(f"  🖼 Preview HTML: {html}")
    historia = _append_historia(Path(args.historia), ronda)
    _imprimir(ronda, historia, tabla_paginas=bool(pedidas))


if __name__ == "__main__":
    main()
