"""
analizar_dialogo_artistico.py v2 — Verificación visual programática de páginas 3 y 12.

Pregunta: ¿el diálogo artístico está dentro de globos de diálogo o pintado
directamente sobre la ilustración? Esto determina el fix (recorte de regiones
y re-OCR vs ajuste del detector).

Método (OpenCV + numpy, sin ML salvo EasyOCR para localizar bloques):
  1. Detectar globos: blobs de luminancia >200, morfología de cierre,
     roundness real (4π·area/perimeter²), con borde definido alrededor.
  2. Detectar trazos de texto con gradiente morfológico (dilate-erode),
     robusto a la tinta del manga (a diferencia del umbral adaptativo que
     satura en todas partes).
  3. Clasificar cada componente de texto: fondo local claro (globo) vs
     fondo de tono medio (arte).
  4. Ejecutar EasyOCR para localizar los bloques de diálogo que sí detectó
     y medir la luminancia del fondo en esas coordenadas exactas.
  5. Guardar crops y visualizaciones para revisión humana.

Salida: dialogo_analysis_out/ (anotados + crops + resumen.json).
"""

import json
import os

import cv2
import numpy as np

OUT_DIR = "dialogo_analysis_out"
os.makedirs(OUT_DIR, exist_ok=True)

PAGES = [
    ("benchmark_page3.png", "page3"),
    ("benchmark_page12.png", "page12"),
]


def detect_bubbles(gray: np.ndarray) -> list[dict]:
    """Globos de diálogo: regiones claras grandes, elípticas y con borde."""
    h, w = gray.shape
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    num, labels, stats, cents = cv2.connectedComponentsWithStats(bright)
    bubbles = []
    min_area = (h * w) * 0.004
    max_area = (h * w) * 0.30
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area or bw < 40 or bh < 25:
            continue
        # Roundness real: 4π·area/perímetro² (0= línea, 1= círculo)
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perim = cv2.arcLength(contours[0], True)
        roundness = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        if roundness < 0.30:  # blob demasiado irregular para ser globo
            continue
        # Borde alrededor del blob (contorno oscuro rodeándolo)
        mask_region = bright[y:y + bh, x:x + bw]
        border_mask = cv2.dilate(mask_region, np.ones((5, 5), np.uint8)) ^ mask_region
        border_hits = cv2.countNonZero(cv2.bitwise_and(edges[y:y + bh, x:x + bw], border_mask))
        border_ratio = border_hits / max(cv2.countNonZero(border_mask), 1)
        if border_ratio < 0.08:
            continue
        interior = gray[y:y + bh, x:x + bw]
        dark_ratio = float(np.mean(interior[mask_region.astype(bool)] < 100))
        bubbles.append({
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "area": int(area), "roundness": round(float(roundness), 3),
            "border_ratio": round(float(border_ratio), 2),
            "dark_text_ratio": round(dark_ratio, 3),
        })
    return bubbles


def detect_text_strokes(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trazos de texto vía gradiente morfológico (robusto al manga).

    Retorna (stroke_mask, labels, stats).
    """
    # Gradiente morfológico: resalta estructuras delgadas de alto contraste
    # (letras, tanto oscuras como claras) sin saturar fondos planos.
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # Texto = gradiente alto; el 97% de la página tiene gradiente bajo
    thr = max(40, float(np.percentile(grad, 97)))
    _, strokes = cv2.threshold(grad, thr, 255, cv2.THRESH_BINARY)
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(strokes)
    return strokes, labels, stats


def sample_bg_luminance(gray: np.ndarray, x: int, y: int, w: int, h: int,
                        stroke_mask: np.ndarray) -> float:
    """Luminancia mediana del fondo alrededor de un trazo (sin contar el trazo)."""
    x0, y0 = max(0, x - 4), max(0, y - 4)
    x1, y1 = min(gray.shape[1], x + w + 4), min(gray.shape[0], y + h + 4)
    region = gray[y0:y1, x0:x1]
    m = stroke_mask[y0:y1, x0:x1]
    bg = region[m == 0]
    if bg.size == 0:
        return float(np.median(region))
    return float(np.median(bg))


def easyocr_blocks(img_bgr: np.ndarray):
    """Ejecuta EasyOCR (GPU) y devuelve bloques con bbox y texto."""
    from ocr_utils import _get_ocr_reader
    reader = _get_ocr_reader()
    res = reader.readtext(img_bgr, detail=1, paragraph=False)
    blocks = []
    for bbox, text, conf in res:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        blocks.append({
            "x": int(min(xs)), "y": int(min(ys)),
            "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys)),
            "text": str(text), "conf": round(float(conf), 2),
        })
    return blocks


def analyze_page(path: str, name: str, use_easyocr: bool) -> dict:
    img = cv2.imread(path)
    assert img is not None, f"No se pudo leer {path}"
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bubbles = detect_bubbles(gray)
    stroke_mask, labels, stats = detect_text_strokes(gray)

    # Mascara de "dentro de globo" (interior claro del blob)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    inside_mask = np.zeros((h, w), np.uint8)
    for b in bubbles:
        inside_mask[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = 255
    inside_mask = cv2.bitwise_and(inside_mask, bright)

    # Clasificar trazos de texto: fondo local claro (globo) vs arte
    comps_in_bubble = 0
    comps_on_art = 0
    px_in_bubble = 0
    total_px = 0
    art_strokes = []  # (x,y,w,h,area,bg_lum) — candidatos a "texto pintado"
    for i in range(1, stats.shape[0]):
        x, y, bw, bh, area = stats[i]
        if area < 25 or area > (h * w) * 0.10:
            continue
        if bw > w * 0.6 or bh > h * 0.4:
            continue
        cx, cy = int(stats[i, 0] + bw / 2), int(stats[i, 1] + bh / 2)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        total_px += area
        bg_lum = sample_bg_luminance(gray, x, y, bw, bh, stroke_mask)
        in_bubble = inside_mask[cy, cx] > 0 and bg_lum > 170
        if in_bubble:
            comps_in_bubble += 1
            px_in_bubble += area
        else:
            comps_on_art += 1
            art_strokes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                                "area": int(area), "bg_lum": round(bg_lum, 0)})

    # ── Visualización: globos (verde) + trazos en globo (amarillo) + arte (magenta)
    vis = img.copy()
    for b in bubbles:
        cv2.rectangle(vis, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 0), 3)
    for i in range(1, stats.shape[0]):
        x, y, bw, bh, area = stats[i]
        if area < 25 or area > (h * w) * 0.10 or bw > w * 0.6 or bh > h * 0.4:
            continue
        cx, cy = int(stats[i, 0] + bw / 2), int(stats[i, 1] + bh / 2)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        bg_lum = sample_bg_luminance(gray, x, y, bw, bh, stroke_mask)
        in_bubble = inside_mask[cy, cx] > 0 and bg_lum > 170
        col = (0, 200, 255) if in_bubble else (255, 0, 255)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 1)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_anotado.png"), vis)

    # Crops de globos detectados
    crops = []
    for bi, b in enumerate(bubbles):
        pad = 10
        x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
        x1, y1 = min(w, b["x"] + b["w"] + pad), min(h, b["y"] + b["h"] + pad)
        cpath = os.path.join(OUT_DIR, f"{name}_globo{bi}.png")
        cv2.imwrite(cpath, img[y0:y1, x0:x1])
        crops.append(cpath)

    # Crops de los mayores trazos "sobre arte" (candidatos a texto pintado)
    art_crops = []
    for si, s in enumerate(sorted(art_strokes, key=lambda s: -s["area"])[:6]):
        pad = 12
        x0, y0 = max(0, s["x"] - pad), max(0, s["y"] - pad)
        x1, y1 = min(w, s["x"] + s["w"] + pad), min(h, s["y"] + s["h"] + pad)
        cpath = os.path.join(OUT_DIR, f"{name}_arttext{si}.png")
        cv2.imwrite(cpath, img[y0:y1, x0:x1])
        art_crops.append({"path": cpath, **s})

    result = {
        "page": name, "size": f"{w}x{h}",
        "globos_detectados": len(bubbles),
        "globos": bubbles,
        "trazos_arte": comps_on_art,
        "trazos_en_globo": comps_in_bubble,
        "px_texto_en_globo_ratio": round(px_in_bubble / max(total_px, 1), 3),
        "art_text_crops": art_crops,
        "anotado": os.path.join(OUT_DIR, f"{name}_anotado.png"),
        "crops_globos": crops,
    }

    if use_easyocr:
        print(f"  [easyocr] OCR en {name}... (GPU, ~8s)")
        blocks = easyocr_blocks(img)
        classified = []
        for b in blocks:
            # Fondo local alrededor del bloque de EasyOCR
            bg_lum = sample_bg_luminance(gray, b["x"], b["y"], b["w"], b["h"], stroke_mask)
            bubble_like = bg_lum > 170
            classified.append({**b, "bg_lum": round(bg_lum, 0),
                               "en_globo": bubble_like})
        result["easyocr_blocks"] = classified
        # Crop del bloque de diálogo más grande detectado por EasyOCR
        dialog = [b for b in classified if b["w"] * b["h"] > 5000]
        if dialog:
            d = max(dialog, key=lambda b: b["w"] * b["h"])
            pad = 15
            x0, y0 = max(0, d["x"] - pad), max(0, d["y"] - pad)
            x1, y1 = min(w, d["x"] + d["w"] + pad), min(h, d["y"] + d["h"] + pad)
            dpath = os.path.join(OUT_DIR, f"{name}_easyocr_dialogo.png")
            cv2.imwrite(dpath, img[y0:y1, x0:x1])
            result["easyocr_dialogo_crop"] = dpath
            result["easyocr_dialogo"] = d
    return result


def main() -> None:
    results = []
    for path, name in PAGES:
        print(f"=== {name} ({path}) ===")
        r = analyze_page(path, name, use_easyocr=True)
        results.append(r)
        print(f"  Globos detectados: {r['globos_detectados']}")
        for g in r["globos"]:
            print(f"    - rect=({g['x']},{g['y']},{g['w']}x{g['h']}) "
                  f"roundness={g['roundness']} borde={g['border_ratio']}")
        print(f"  Trazos de texto: globo={r['trazos_en_globo']} arte={r['trazos_arte']} "
              f"(px en globo: {r['px_texto_en_globo_ratio']*100:.1f}%)")
        if r.get("easyocr_blocks"):
            print(f"  EasyOCR detectó {len(r['easyocr_blocks'])} bloques:")
            for b in r["easyocr_blocks"]:
                print(f"    - ({b['x']},{b['y']},{b['w']}x{b['h']}) lum_fondo={b['bg_lum']} "
                      f"globo={b['en_globo']} conf={b['conf']} '{b['text'][:42]}'")
        print(f"  Visualización: {r['anotado']}")
        print()

    with open(os.path.join(OUT_DIR, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("Guardado en dialogo_analysis_out/resumen.json")


if __name__ == "__main__":
    main()
