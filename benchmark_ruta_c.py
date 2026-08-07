"""
benchmark_ruta_c.py — Benchmark de la Ruta C: re-OCR por región con EasyOCR GPU.

Pregunta: ¿cuánto diálogo artístico recupera EasyOCR GPU al recortar los paneles
que U-OCR clasificó como type="image" y upscale 2-3x, vs. el re-OCR con el
propio Unlimited-OCR (from_art_recrop, ya implementado en el daemon)?

Método por página (3, 11, 12):
  1. Llamar al daemon /ocr (uocr_client.process_page) → bloques completos:
     - type="image" (los paneles que U-OCR no supo leer → regiones a recortar)
     - from_art_recrop=True (lo que el re-OCR del daemon recuperó)
  2. Para cada bloque image grande (>15% página):
     - Recortar de la página original
     - Upscale 2x y 3x (INTER_CUBIC)
     - EasyOCR GPU sobre el recorte upscalado
     - Mapear los bloques recuperados de vuelta al espacio de la página
  3. Comparar contra la referencia (diálogo artístico conocido de págs. 3/12).

Uso (venv env/ con easyocr):
  env\\Scripts\\python.exe benchmark_ruta_c.py
"""
import json
import os
import re
import sys
import time
import unicodedata

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

PAGES = [3, 11, 12]
IMG = {p: f"benchmark_page{p}.png" for p in PAGES}
# Referencia: diálogo artístico que se sabe perdido en EasyOCR full-page
# (del análisis dialogo_analysis_out/resumen.json + re-OCR validado).
REF = {
    3: "INCREÍBLE REALMENTE",          # SFX pintado sobre arte
    12: "ERA UNA PROPUESTA QUE SOLO PODÍA BENEFICIARME PERO",  # globo en panel
    11: "",                             # U-OCR lee bien la página completa
}
_IMAGE_MIN_RATIO = 0.15  # bloque image con área >= 15% de la página


def _norm(t: str) -> str:
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _overlap_ratio(b1: dict, b2: dict) -> float:
    x1 = max(b1["x"], b2["x"])
    y1 = max(b1["y"], b2["y"])
    x2 = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
    y2 = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    min_area = min(b1["w"] * b1["h"], b2["w"] * b2["h"])
    return inter / float(min_area) if min_area > 0 else 0.0


def run_easyocr_crop(img_bgr: np.ndarray, upscale: float):
    """EasyOCR GPU sobre un recorte (ya upscalado). Retorna bloques en espacio del recorte."""
    import easyocr
    reader = easyocr.Reader(["es", "en"], gpu=True, verbose=False)
    t0 = time.time()
    results = reader.readtext(img_bgr, detail=1, paragraph=False, min_size=6,
                              text_threshold=0.15, low_text=0.10, canvas_size=2500)
    elapsed = time.time() - t0
    blocks = []
    for (bbox, text, conf) in results:
        text = str(text).strip()
        if not text or conf < 0.08:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        blocks.append({"x": int(min(xs)), "y": int(min(ys)),
                       "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys)),
                       "text": text, "conf": float(conf)})
    return blocks, elapsed


def process_page_uocr(page_no: int):
    """Llama al daemon y separa: bloques normales, image-blocks, from_art_recrop."""
    import uocr_client
    res = uocr_client.process_page(IMG[page_no], max_length=4096, wait_timeout_s=1500)
    if res.get("error"):
        raise RuntimeError(res["error"])
    image_blocks = [b for b in res.get("blocks", []) if b.get("type") == "image"]
    recrop = [b for b in res.get("blocks", [])
              if b.get("from_art_recrop") and b.get("text")]
    normal = [b for b in res.get("blocks", [])
              if b.get("type") != "image" and b.get("text")]
    return {"normal": normal, "image": image_blocks, "recrop": recrop,
            "infer_s": res.get("infer_s", 0.0), "raw": res}


def main() -> int:
    print("=" * 78)
    print("BENCHMARK RUTA C: re-OCR por región (EasyOCR GPU) vs re-OCR U-OCR")
    print("=" * 78)

    results = {"pages": []}

    for page_no in PAGES:
        print(f"\n--- Página {page_no} ---")
        img = cv2.imread(IMG[page_no])
        if img is None:
            print(f"[!] No se pudo leer {IMG[page_no]}")
            continue
        ph, pw = img.shape[:2]
        page_area = pw * ph
        ref_norm = _norm(REF.get(page_no, ""))

        # ── 1. Daemon: bloques image + from_art_recrop ──────────
        t0 = time.time()
        d = process_page_uocr(page_no)
        dt_daemon = time.time() - t0
        big_images = [b for b in d["image"]
                      if (b.get("w", 0) * b.get("h", 0)) >= _IMAGE_MIN_RATIO * page_area]
        print(f"[daemon] {dt_daemon:.1f}s | {len(d['normal'])} normales, "
              f"{len(d['image'])} image, {len(big_images)} image>={_IMAGE_MIN_RATIO:.0%}, "
              f"{len(d['recrop'])} from_art_recrop")

        # ── 2. Ruta C: EasyOCR sobre cada panel image ───────────
        easy_recrop: list[dict] = []
        crop_times = []
        for bi, ib in enumerate(big_images):
            pad = 8
            x0, y0 = max(0, ib["x"] - pad), max(0, ib["y"] - pad)
            x1, y1 = min(pw, ib["x"] + ib["w"] + pad), min(ph, ib["y"] + ib["h"] + pad)
            crop = img[y0:y1, x0:x1]
            cw, ch = crop.shape[1], crop.shape[0]
            if cw < 32 or ch < 32:
                continue
            for scale in (2.0, 3.0):
                nw, nh = int(round(cw * scale)), int(round(ch * scale))
                up = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
                blocks, t_e = run_easyocr_crop(up, scale)
                crop_times.append(t_e)
                mapped = []
                for b in blocks:
                    mapped.append({
                        "x": int(round(x0 + b["x"] / scale)),
                        "y": int(round(y0 + b["y"] / scale)),
                        "w": int(round(b["w"] / scale)),
                        "h": int(round(b["h"] / scale)),
                        "text": b["text"], "conf": b["conf"], "scale": scale,
                    })
                easy_recrop.extend(mapped)
                if mapped:
                    print(f"  [RutaC x{scale:.0f}] panel {bi} ({cw}x{ch}): "
                          f"{len(mapped)} bloques | {[m['text'][:40] for m in mapped[:4]]} | {t_e:.2f}s")

        # ── 3. Comparar cobertura de diálogo ────────────────────
        easy_texts = " ".join(_norm(b["text"]) for b in easy_recrop)
        uocr_recrop_texts = " ".join(_norm(b["text"]) for b in d["recrop"])
        uocr_normal_texts = " ".join(_norm(b["text"]) for b in d["normal"])

        row = {
            "page": page_no,
            "reference": REF.get(page_no, ""),
            "daemon_s": round(dt_daemon, 1),
            "n_image_blocks": len(d["image"]),
            "n_big_image_blocks": len(big_images),
            "n_uocr_recrop": len(d["recrop"]),
            "n_easy_recrop": len(easy_recrop),
            "easy_recrop_texts": [b["text"] for b in easy_recrop],
            "uocr_recrop_texts": [b["text"] for b in d["recrop"]],
        }
        # Métricas de recuperación del diálogo de referencia
        if ref_norm:
            ref_words = set(ref_norm.split())
            row["ref_words"] = len(ref_words)
            row["ref_found_by_easy_recrop"] = len(
                ref_words & set(easy_texts.split()))
            row["ref_found_by_uocr_recrop"] = len(
                ref_words & set(uocr_recrop_texts.split()))
            row["ref_found_by_uocr_full"] = len(
                ref_words & set(uocr_normal_texts.split()))
        results["pages"].append(row)

        print(f"  referencia: {REF.get(page_no, '—')!r}")
        if ref_norm:
            print(f"  palabras ref: {row['ref_words']} | "
                  f"EasyOCR-crop: {row['ref_found_by_easy_recrop']} | "
                  f"U-OCR-recrop: {row['ref_found_by_uocr_recrop']} | "
                  f"U-OCR-full: {row['ref_found_by_uocr_full']}")

    with open("benchmark_ruta_c_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[OK] benchmark_ruta_c_results.json guardado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
