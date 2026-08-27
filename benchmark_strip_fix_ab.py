"""benchmark_strip_fix_ab.py — A/B del fix de la pérdida del strip (2026-08-16).

Antecedente: el batch estructural (`_rapidocr_strip_batch`) leyó ruido con
conf >= RUTA_C_RAPID_MIN_CONF (0.45) en 4 crops ('OE O' 0.5, 'S E' 0.6,
'P!' 0.46) → `usable_rapid` no vacío → el fallback EasyOCR por crop NO corre
→ 4 diálogos reales perdidos (págs 1, 4, 40, 52; pág 4 por mecanismo de
merge). Fix candidato: subir el umbral de aceptación de usable_rapid a 0.7
para que el ruido 0.45-0.7 caiga al fallback.

Este script corre el pipeline COMPLETO (run_ocr fusion) en el MISMO proceso
sobre las 6 páginas del análisis (1, 4, 30, 34, 40, 52) + 2 controles de
tiempo con carga de Ruta C, en modo alternado anti-deriva:

  base  : strip + umbral 0.45 (producción actual)
  fix   : strip + umbral 0.70 (candidato)
  ref   : per-crop sin strip + 0.45 (referencia de recuperación, 1 corrida)

Métricas por página y modo: tiempo, nº de bloques, textos (para detectar
regresiones base-vs-fix) y presencia de los 4 fragmentos de diálogo perdidos.

Uso:
  python benchmark_strip_fix_ab.py --json benchmark_results/strip_fix_ab.json

Las páginas NO disparan el trigger VLM; el daemon puede estar arriba (ready).
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np

import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2

# Las 4 pérdidas reales + las 2 de segmentación (controles que NO deben
# regresar) + 2 páginas con carga de Ruta C para el tiempo.
PAGES = [1, 4, 30, 34, 40, 43, 52]
FRAGMENTOS = {
    1: ["enverdades"],
    4: ["ysihubiese"],
    40: ["ipadrino"],
    52: ["comerlashoy"],
}


def _norm(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


def _run(ocr: OCRManager, img: Any, strip: bool, retry: bool) -> dict[str, Any]:
    ocr_utils._RUTA_C_STRIP_BATCH = strip
    ocr_utils._RUTA_C_STRIP_RETRY_INDIVIDUAL = retry
    t0 = time.time()
    blocks, engine, _ = ocr.run_ocr(img, OCR_LANG, "fusion", prefilter=True)
    dt = time.time() - t0
    texts = sorted(_norm(b.get("text", "")) for b in blocks if _norm(b.get("text", "")))
    return {"dt": dt, "n": len(blocks), "texts": texts, "engine": engine}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="benchmark_results/strip_fix_ab.json")
    args = ap.parse_args()

    doc = fitz.open(PDF)
    ocr = OCRManager()
    out: dict[str, Any] = {}
    print("A/B fix del strip — base(0.45) vs fix(0.7) vs per-crop(ref)\n")
    for pno in PAGES:
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)[:, :, ::-1].copy()

        # Pares alternados anti-deriva: base, fix, base, fix, luego ref.
        base_runs: list[dict[str, Any]] = []
        fix_runs: list[dict[str, Any]] = []
        for i in range(2):
            if i % 2 == 0:
                base_runs.append(_run(ocr, img, strip=True, retry=False))
                fix_runs.append(_run(ocr, img, strip=True, retry=True))
            else:
                fix_runs.append(_run(ocr, img, strip=True, retry=True))
                base_runs.append(_run(ocr, img, strip=True, retry=False))
        ref = _run(ocr, img, strip=False, retry=True)

        base_texts = sorted(set(t for r in base_runs for t in r["texts"]))
        fix_texts = sorted(set(t for r in fix_runs for t in r["texts"]))
        base_dt = sum(r["dt"] for r in base_runs) / len(base_runs)
        fix_dt = sum(r["dt"] for r in fix_runs) / len(fix_runs)

        sb, sf = set(base_texts), set(fix_texts)
        regresiones = sorted(sb - sf)
        nuevas = sorted(sf - sb)

        print(f"pág {pno}: base {base_dt:.2f}s ({base_runs[0]['n']}bl) | "
              f"fix {fix_dt:.2f}s ({fix_runs[0]['n']}bl) | "
              f"ref {ref['dt']:.2f}s ({ref['n']}bl)")

        frag_status: dict[str, Any] = {}
        for frag in FRAGMENTOS.get(pno, []):
            en_base = frag in " ".join(base_texts)
            en_fix = frag in " ".join(fix_texts)
            en_ref = frag in " ".join(ref["texts"])
            print(f"  '{frag}': base={en_base} fix={en_fix} ref={en_ref}")
            frag_status[frag] = {"base": en_base, "fix": en_fix, "ref": en_ref}

        if regresiones:
            print(f"  ⚠ en base pero NO en fix (regresión): {regresiones[:6]}")
        if nuevas:
            print(f"  + en fix pero no en base: {nuevas[:6]}")

        out[str(pno)] = {
            "base_dt": round(base_dt, 3), "fix_dt": round(fix_dt, 3),
            "ref_dt": round(ref["dt"], 3),
            "base_n": base_runs[0]["n"], "fix_n": fix_runs[0]["n"],
            "ref_n": ref["n"],
            "fragmentos": frag_status,
            "regresiones": regresiones,
            "nuevas": nuevas,
            "base_texts": base_texts,
            "fix_texts": fix_texts,
            "ref_texts": ref["texts"],
        }
        print()
    doc.close()
    Path(args.json).write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {args.json}")


if __name__ == "__main__":
    main()
