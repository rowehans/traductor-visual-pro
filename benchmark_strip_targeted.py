"""Verificación dirigida de los textos sospechosos del strip (2026-08-16).

Del A/B mismo-proceso, 4 textos del per-crop no tienen correspondencia clara
en el strip. Este script corre cada página afectada N veces en cada modo y
comprueba si el texto sospechoso aparece en ALGUNA corrida del strip:

  - aparece en ≥1 corrida del strip → varianza de lectura, NO pérdida
  - nunca aparece en strip pero sí en per-crop → posible pérdida real
    (requiere revisión visual de la página para confirmar contenido real)
"""
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz
import numpy as np

import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2
REPS = 3

# (página, fragmento sospechoso) del análisis mismo-proceso
SOSPECHOSOS = {
    1: ["enverdades"],
    4: ["ysihubiese"],
    40: ["ipadrino"],
    52: ["comerlashoy"],
}


def _norm(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


def _run(ocr: OCRManager, img: Any, strip: bool) -> list[str]:
    ocr_utils._RUTA_C_STRIP_BATCH = strip
    blocks, _, _ = ocr.run_ocr(img, OCR_LANG, "fusion", prefilter=True)
    return [_norm(b.get("text", "")) for b in blocks if _norm(b.get("text", ""))]


def main() -> None:
    doc = fitz.open(PDF)
    ocr = OCRManager()
    out: dict[str, Any] = {}
    for pno in sorted(SOSPECHOSOS):
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, ::-1].copy()
        print(f"pág {pno} — fragmentos: {SOSPECHOSOS[pno]} ({REPS} corridas por modo)")
        all_strip: list[str] = []
        all_pc: list[str] = []
        for i in range(REPS):
            # alternar para diluir deriva de GPU
            a = _run(ocr, img, strip=(i % 2 == 0))
            b = _run(ocr, img, strip=(i % 2 != 0))
            if i % 2 == 0:
                all_strip.extend(a); all_pc.extend(b)
            else:
                all_pc.extend(a); all_strip.extend(b)
        strip_blob = " ".join(all_strip)
        pc_blob = " ".join(all_pc)
        verdict: dict[str, Any] = {"reps": REPS, "fragmentos": {}}
        for frag in SOSPECHOSOS[pno]:
            en_strip = frag in strip_blob
            en_pc = frag in pc_blob
            # contexto: qué fragmento del strip es el más similar
            import difflib
            cand = max(all_strip, key=lambda s: difflib.SequenceMatcher(None, frag, s).ratio())
            ratio = difflib.SequenceMatcher(None, frag, cand).ratio()
            status = ("✓ varianza de lectura (aparece en strip)"
                      if en_strip else
                      "⚠ posible pérdida (nunca en strip, sí en per-crop)"
                      if en_pc else
                      "✗ ni siquiera estable en per-crop (ruido)")
            print(f"  '{frag}': {status}")
            if not en_strip:
                print(f"      mejor match en strip: '{cand[:70]}' ratio={ratio:.2f}")
            verdict["fragmentos"][frag] = {
                "en_strip": en_strip, "en_per_crop": en_pc,
                "mejor_match": cand, "ratio": round(ratio, 3),
            }
        out[str(pno)] = verdict
        print()
    doc.close()
    Path("benchmark_results/strip_targeted.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Resultado: benchmark_results/strip_targeted.json")


if __name__ == "__main__":
    main()
