"""Verificación de los 6 bloques perdidos del A/B del strip (2026-08-16).

El A/B del capítulo completo (production_full53 vs production_strip_full53)
mostró 6 páginas que perdieron 1 bloque cada una: 1, 4, 30, 34, 40, 52.
Este script corre el pipeline COMPLETO de producción (run_ocr fusion) sobre
esas 6 páginas con _RUTA_C_STRIP_BATCH en AMBOS estados en el MISMO proceso
(mismo estado GPU, pares consecutivos) y compara los textos finales:

  - contenido del baseline presente en el strip (fusionado/segmentado) = OK
  - texto del strip ausente del baseline o viceversa = posible pérdida real

Las 6 páginas NO disparan el trigger VLM (verificado en el capítulo completo),
así que la corrida es segura con el daemon arriba.
"""
import json
import re
import time
from pathlib import Path
from typing import Any

import fitz

import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2
PAGES = [1, 4, 30, 34, 40, 52]


def _norm(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


def _run(ocr: OCRManager, img: Any, strip: bool) -> dict[str, Any]:
    ocr_utils._RUTA_C_STRIP_BATCH = strip
    t0 = time.time()
    blocks, engine, _ = ocr.run_ocr(img, OCR_LANG, "fusion", prefilter=True)
    dt = time.time() - t0
    texts = sorted(_norm(b.get("text", "")) for b in blocks if _norm(b.get("text", "")))
    return {"dt": dt, "n": len(blocks), "texts": texts, "engine": engine}


def main() -> None:
    doc = fitz.open(PDF)
    ocr = OCRManager()
    results: dict[str, Any] = {}
    print(f"Verificación de bloques perdidos del strip — páginas {PAGES}")
    print("(cada página se corre 2× en el mismo proceso: per-crop y strip)\n")
    for pno in PAGES:
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
        import numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, ::-1].copy()

        # Pares consecutivos para minimizar deriva de GPU.
        base = _run(ocr, img, strip=False)
        strip = _run(ocr, img, strip=True)

        # ── Comparación de contenido ──
        tb, ts = set(base["texts"]), set(strip["texts"])
        solo_base = sorted(tb - ts)
        solo_strip = sorted(ts - tb)
        # Detectar si un texto 'perdido' es subcadena de uno fusionado en el otro.
        fusionados: list[str] = []
        for t in solo_base:
            # ¿t está contenido en algún bloque del strip (fusionado)?
            if any(t in s for s in strip["texts"]):
                fusionados.append(t)
        # Normalizar el conteo: quitar de solo_base los que están fusionados
        perdidos_reales = [t for t in solo_base if not any(t in s for s in strip["texts"])]

        print(f"pág {pno}: per-crop {base['n']} bloques ({base['dt']:.1f}s) | "
              f"strip {strip['n']} bloques ({strip['dt']:.1f}s)")
        if solo_base:
            print(f"  solo en per-crop: {solo_base}")
        if solo_strip:
            print(f"  solo en strip:    {solo_strip}")
        if fusionados:
            print(f"  de los solo-per-crop, fusionados como subcadena en strip: {fusionados}")
        if perdidos_reales:
            print(f"  ⚠ TEXTO AUSENTE EN STRIP (posible pérdida real): {perdidos_reales}")
        else:
            print(f"  ✓ contenido del per-crop íntegro en strip "
                  f"({len(solo_base)} diferencias = segmentación/lectura)")
        print()
        results[str(pno)] = {
            "per_crop": {"n": base["n"], "dt": round(base["dt"], 2), "texts": base["texts"]},
            "strip": {"n": strip["n"], "dt": round(strip["dt"], 2), "texts": strip["texts"]},
            "solo_per_crop": solo_base,
            "solo_strip": solo_strip,
            "fusionados": fusionados,
            "ausentes_reales": perdidos_reales,
        }

    doc.close()
    Path("benchmark_results/strip_blocks_check.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Resultado: benchmark_results/strip_blocks_check.json")


if __name__ == "__main__":
    main()
