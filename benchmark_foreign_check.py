"""benchmark_foreign_check.py — Costo por fase de _contains_foreign_latin_tokens.

Mide cuánto cuesta la detección de tokens latinos extranjeros (en/pt) sobre
los bloques reales del capítulo: cuántas veces se llama, cuánto tiempo total,
y dónde se va el tiempo (tokenización, known() es, known() en/pt).

Uso:
  python benchmark_foreign_check.py --pages 1-53 --json benchmark_results/foreign_check.json

El daemon VLM debe estar DETENIDO (mismo estado que los benchmarks del
capítulo) para que la corrida no tarde minutos en las páginas trigger.
"""

import argparse
import json
import time
from typing import Any

import fitz
import numpy as np

import ocr_utils
from ocr_engine import OCRManager
from ocr_utils import _get_spellchecker, _get_foreign_spellchecker

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
DEFAULT_SCALE = 1.2

# ── Instrumentación ──────────────────────────────────────────────
STATS: dict[str, Any] = {
    "calls": 0,
    "foreign_true": 0,
    "early_es_all_known": 0,   # todos los tokens son espanoles
    "early_no_candidates": 0,  # longitud filtra todo
    "total_s": 0.0,
    "known_calls": {"es": 0, "en": 0, "pt": 0},
    "known_s": {"es": 0.0, "en": 0.0, "pt": 0.0},
    "known_words": {"es": 0, "en": 0, "pt": 0},
    "tokens_seen": 0,
}

_orig_func = ocr_utils._contains_foreign_latin_tokens


def _install() -> Any:
    """Parchea la funcion y los known() de los tres checkers con timing."""
    es_sp = _get_spellchecker()
    en_sp = _get_foreign_spellchecker("en")
    pt_sp = _get_foreign_spellchecker("pt")
    langs = {id(es_sp): "es", id(en_sp): "en", id(pt_sp): "pt"}

    import spellchecker.spellchecker as sc

    orig_known = sc.SpellChecker.known

    def known_wrapper(self: Any, words: Any) -> Any:
        lang = langs.get(id(self), "?")
        t0 = time.perf_counter()
        res = orig_known(self, words)
        dt = time.perf_counter() - t0
        if lang != "?":
            STATS["known_calls"][lang] += 1
            STATS["known_s"][lang] += dt
            n = len(words) if hasattr(words, "__len__") else 0
            STATS["known_words"][lang] += n
        return res

    sc.SpellChecker.known = known_wrapper  # type: ignore[method-assign]

    def wrapper(text: str, spanish_checker: Any) -> bool:
        t0 = time.perf_counter()
        res = _orig_func(text, spanish_checker)
        STATS["total_s"] += time.perf_counter() - t0
        STATS["calls"] += 1
        if res:
            STATS["foreign_true"] += 1
        else:
            # clasificar el early-exit aproximado re-tokenizando (barato)
            tokens = [t.strip("'\".,;:!?¡¿()[]{}").lower()
                      for t in text.split()]
            tokens = [t for t in tokens
                      if len(t) > 1 and not any(c.isdigit() for c in t)]
            STATS["tokens_seen"] += len(tokens)
            if tokens:
                try:
                    if set(tokens) <= set(spanish_checker.known(tokens)):
                        STATS["early_es_all_known"] += 1
                    else:
                        STATS["early_no_candidates"] += 1
                except Exception:
                    pass
        return res

    ocr_utils._contains_foreign_latin_tokens = wrapper
    return es_sp


def _parse_pages(spec: str) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="1-53")
    ap.add_argument("--json", default="benchmark_results/foreign_check.json")
    args = ap.parse_args()

    _install()
    doc = fitz.open(PDF)
    ocr = OCRManager()
    per_page: dict[str, float] = {}
    t_start = time.perf_counter()
    for pno in _parse_pages(args.pages):
        page = doc.load_page(pno - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(DEFAULT_SCALE, DEFAULT_SCALE))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        img_bgr = img[:, :, :3][:, :, ::-1].copy()
        t0 = time.perf_counter()
        blocks, engine_used, _ = ocr.run_ocr(
            img_bgr, OCR_LANG, "fusion", prefilter=True
        )
        per_page[str(pno)] = time.perf_counter() - t0
        print(f"pág {pno:>3}: {per_page[str(pno)]:.2f}s bloques={len(blocks)}", flush=True)
    total = time.perf_counter() - t_start
    payload = {
        "pages": [str(p) for p in _parse_pages(args.pages)],
        "total_s": round(total, 2),
        "stats": STATS,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n== foreign check: {STATS['calls']} llamadas, "
          f"{STATS['total_s']:.3f}s total, "
          f"{STATS['total_s']/max(STATS['calls'],1)*1000:.3f} ms/llamada")
    print(f"   foreign=True: {STATS['foreign_true']} | es todo conocido: "
          f"{STATS['early_es_all_known']} | sin candidatos por longitud: "
          f"{STATS['early_no_candidates']}")
    print(f"   known(): calls {STATS['known_calls']} | s {STATS['known_s']} | "
          f"words {STATS['known_words']}")


if __name__ == "__main__":
    main()
