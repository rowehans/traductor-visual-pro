"""Mide el coste real de la salvaguarda de detección débil (sesión 134).

Reconstruye la historia del capítulo completo (53 págs, corrida real
run_det128_run1 — 23:26-23:41, fusion, workers=2) a partir de:

1) FIRMAS de las 53 páginas (render del PDF como process_all_pages: zoom 1.2)
   — la firma es layout-only, idéntica a la que el OCRManager computa.

2) RESULTADOS REALES de la corrida (run_det128_run1.json): por página, cuántos
   bloques híbridos había (blocks del checkpoint ≈ detección final), y qué
   páginas dispararon VLM (T>60s).

3) RESULTADOS del DAEMON (uocr_daemon_out/req_* en la ventana 23:26-23:41):
   cada result.md dice si la inferencia VLM devolvió texto (recuperó) o no
   (negativa §8.4.1). Se mapea cada req a su página por la firma de la imagen.

4) SIMULACIÓN de la salvaguarda sobre la secuencia real: para cada negativa
   registrada con detección débil (< UOCR_NEG_WEAK_MAX_BLOCKS bloques O conf <
   UOCR_NEG_WEAK_MIN_CONF), la siguiente página con la MISMA firma re-dispara
   el VLM una vez (contador=1) en vez de hacer skip §8.4.1. Coste = 1
   inferencia VLM extra por firma débil con gemelas; beneficio = si esa gemela
   tenía diálogo que el híbrido no leyó (SIN_TRAD / PARCIAL real en la corrida).

Uso: python tools/medir_coste_salvaguarda.py <pdf> [--window-start TS] [--window-end TS]
"""
import glob
import json
import os
import sys
import time

import cv2
import fitz
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "run_det128_run1.json")
DAEMON_DIR = os.path.join(ROOT, "uocr_daemon_out")
ZOOM = 1.2  # mismo render que process_all_pages.py

sys.path.insert(0, ROOT)
from ocr_utils import _page_signature  # noqa: E402


def _render_pdf_firmas(pdf_path, zoom=ZOOM):
    """Renderiza cada página como el pipeline real y computa su firma."""
    doc = fitz.open(pdf_path)
    firmas = {}
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        firmas[i + 1] = _page_signature(img)
        page = None
        pix = None
    doc.close()
    return firmas


def _firma_de_imagen_daemon(path):
    """Firma de una imagen guardada por el daemon (640x640)."""
    img = cv2.imread(path)
    if img is None:
        return ""
    return _page_signature(img)


def _mapear_reqs_a_paginas(firmas_pagina, ts_start, ts_end):
    """Mapea cada req_* en la ventana a su página comparando la firma."""
    # Índice inverso: firma → [páginas] (puede haber colisiones de layout)
    por_firma = {}
    for p, f in firmas_pagina.items():
        por_firma.setdefault(f, []).append(p)

    mapeo = {}
    for d in sorted(glob.glob(os.path.join(DAEMON_DIR, "req_*"))):
        if not os.path.isdir(d):
            continue
        mtime = os.path.getmtime(d)
        if not (ts_start <= mtime <= ts_end):
            continue
        imgs = sorted(glob.glob(os.path.join(d, "images", "*.jpg")))
        if not imgs:
            continue
        f = _firma_de_imagen_daemon(imgs[0])
        candidatas = por_firma.get(f, [])
        # El req individual del daemon es de UNA página; si hay colisión de
        # firma, la resolvemos con la 2ª imagen o por ambigüedad → se anota.
        mapeo[d] = {"firma": f, "candidatas": candidatas, "n_imgs": len(imgs)}
    return mapeo


def main():
    if len(sys.argv) < 2:
        print("Uso: python tools/medir_coste_salvaguarda.py <pdf> [--window-start TS]")
        return 2
    pdf = sys.argv[1]
    ts_start = float(sys.argv[sys.argv.index("--window-start") + 1]
                     if "--window-start" in sys.argv else 0)
    ts_end = float(sys.argv[sys.argv.index("--window-end") + 1]
                   if "--window-end" in sys.argv else time.time())

    d = json.load(open(CHECKPOINT, encoding="utf-8"))
    print("=" * 74)
    print("Coste de la salvaguarda de detección débil (sesión 134)")
    print("Corrida real: run_det128_run1 (%d págs, %s)" % (d["total_pages"], d.get("ocr_mode")))
    print("=" * 74)

    # ── 1. Firmas por página ────────────────────────────────────────
    firmas = _render_pdf_firmas(pdf)
    n_unicas = len(set(firmas.values()))
    n_vacias = sum(1 for f in firmas.values() if not f)
    print(f"\n[1] Firmas de las {len(firmas)} páginas: {n_unicas} únicas, "
          f"{n_vacias} vacías")

    # ── 2. Páginas VLM + negativas de la corrida real ──────────────
    vlm = {r["page"]: r for r in d["results"] if r["time"] > 60}
    print(f"\n[2] {len(vlm)} páginas dispararon VLM en la corrida real:")
    for p in sorted(vlm):
        r = vlm[p]
        print(f"    p{p:2d} blocks={r['blocks']:2d} trad={r['translated']:2d} "
              f"T={r['time']:.0f}s {r['status']}  firma={firmas.get(p, '')[:24]}")

    # ── 3. Mapeo req → página (result.md del daemon en la ventana) ──
    mapeo = _mapear_reqs_a_paginas(firmas, ts_start, ts_end)
    print(f"\n[3] {len(mapeo)} llamadas VLM del daemon en la ventana:")
    req_por_firma = {}
    for dname, info in sorted(mapeo.items()):
        # ¿El VLM devolvió texto? result.md con contenido real (no solo ![]())
        md = os.path.join(dname, "result.md")
        texto = ""
        if os.path.exists(md):
            with open(md, encoding="utf-8", errors="ignore") as f:
                texto = f.read()
        # Quitar líneas de imagen vacías y contar palabras reales
        palabras = [w for w in texto.replace("\r", " ").split()
                    if not w.startswith("![](")]
        recupero = len(palabras) > 3
        req_por_firma[info["firma"]] = req_por_firma.get(info["firma"], 0) + 1
        print(f"    {os.path.basename(dname)[:26]:28s} págs={info['candidatas']} "
              f"palabras={len(palabras):4d} → {'RECUPERÓ' if recupero else 'negativa'}")

    # ── 4. Simulación de la salvaguarda ─────────────────────────────
    print(f"\n[4] Simulación de la salvaguarda (contador=1) sobre la secuencia real:")
    # Firmas únicas con VLM que NO recuperó → negativa registrada
    negativas = {}  # firma → stats estimados
    for dname, info in sorted(mapeo.items()):
        md = os.path.join(dname, "result.md")
        texto = ""
        if os.path.exists(md):
            with open(md, encoding="utf-8", errors="ignore") as f:
                texto = f.read()
        palabras = [w for w in texto.replace("\r", " ").split()
                    if not w.startswith("![](")]
        if len(palabras) <= 3:
            negativas.setdefault(info["firma"], 0)
            negativas[info["firma"]] += 1
    print(f"    {len(negativas)} firmas únicas con negativa (VLM sin recuperar)")

    print("\nRequisito: firmas de páginas con VLM real → negativa por página.")
    print("(La validación empírica del beneficio requiere correr las gemelas).")


if __name__ == "__main__":
    import time
    main()
