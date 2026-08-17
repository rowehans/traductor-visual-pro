"""Exporta recortes visuales de los 4 textos sospechosos del strip (2026-08-16).

Para cada página sospechosa (1, 4, 40, 52), corre el pipeline con
_RUTA_C_STRIP_BATCH=False (per-crop, el camino que SÍ recupera los textos),
localiza los bloques cuyos textos contienen los fragmentos sospechosos, y
guarda un recorte de la página (con margen) como PNG. Genera un HTML que
muestra cada recorte junto a su texto OCR para verificación visual.

Resultado: benchmark_results/strip_visual/ con PNGs + index.html.
"""
import json
import re
import shutil
from pathlib import Path
from typing import Any

import fitz
import numpy as np

import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2
MARGIN = 0.30  # margen del recorte (30% del tamaño del bloque)

SOSPECHOSOS = {
    1: ["enverdades"],
    4: ["ysihubiese"],
    40: ["ipadrino"],
    52: ["comerlashoy"],
}


def _norm(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


def main() -> None:
    out_dir = Path("benchmark_results/strip_visual")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    doc = fitz.open(PDF)
    ocr = OCRManager()
    cards: list[str] = []

    for pno in sorted(SOSPECHOSOS):
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, ::-1].copy()
        H, W = img.shape[:2]

        ocr_utils._RUTA_C_STRIP_BATCH = False
        blocks, _, _ = ocr.run_ocr(img, OCR_LANG, "fusion", prefilter=True)

        for b in blocks:
            text = _norm(b.get("text", ""))
            if not any(f in text for f in SOSPECHOSOS[pno]):
                continue
            x, y, w, h = (int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]))
            mx, my = int(w * MARGIN), int(h * MARGIN)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(W, x + w + mx), min(H, y + h + my)
            crop = img[y0:y1, x0:x1]
            fname = f"p{pno}_{x}_{y}.png"
            from PIL import Image
            Image.fromarray(crop[:, :, ::-1]).save(out_dir / fname, "PNG")
            cards.append(
                f'<div class="card"><h3>pág {pno} — "{b.get("text", "")[:60]}"</h3>'
                f'<p class="meta">bloque x={x} y={y} w={w} h={h} '
                f'conf={b.get("confidence", 0):.2f}</p>'
                f'<img src="{fname}" alt="recorte p{pno}"/></div>'
            )
            print(f"pág {pno}: guardado {fname}  texto='{b.get('text', '')[:70]}' "
                  f"conf={b.get('confidence', 0):.2f}")

    doc.close()
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 24px; }}
  h1 {{ font-size: 18px; }}
  h3 {{ margin: 0 0 6px; font-size: 15px; }}
  .meta {{ color: #999; font-size: 12px; margin: 0 0 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .card {{ background: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 12px; }}
  img {{ max-width: 100%; border: 1px solid #444; border-radius: 4px; }}
  .note {{ color: #ffb454; font-size: 13px; margin-top: 10px; }}
</style></head><body>
<h1>Recortes de los 4 textos sospechosos del strip (verificación visual)</h1>
<p class="note">Si el recorte muestra texto real (diálogo), el strip lo pierde = pérdida real.
Si es ruido/artefacto/URL, no es contenido.</p>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\nHTML: {out_dir / 'index.html'}  ({len(cards)} recortes)")


if __name__ == "__main__":
    main()
