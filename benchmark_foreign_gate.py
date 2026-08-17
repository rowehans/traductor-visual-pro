"""benchmark_foreign_gate.py — ¿Los bloques mixtos (foreign=True) del capítulo
justifican afinar cuándo se llama _contains_foreign_latin_tokens?

Para cada bloque del capítulo que llega al chequeo extranjero (langdetect es +
>= 2 palabras), registra:
  - texto, página, nº de palabras/caracteres
  - resultado foreign (True/False)
  - para los foreign=True:
      * evidencia extranjera: tokens conocidos en en/pt y no en es
      * longitud mínima de la evidencia: 2 = inofensiva (el loop del corrector
        NUNCA corrige palabras <= 2 chars), >= 3 = corregible
      * CORRECCIONES FORZADAS: las que aplicaría _ocr_spellcheck si el gate
        no existiera (se llama _ocr_spellcheck con el gate desactivado y se
        hace diff palabra a palabra), y cuáles tocan tokens extranjeros
        (corrupción evitada por el gate) vs españoles (correcciones perdidas
        por el gate)
      * ambigüedad del langdetect: _detect_language_simple (no robusto) — si
        dice distinto de "es" para un bloque que pasó el gate robusto == "es",
        el bloque es ambiguo

Uso:
  python benchmark_foreign_gate.py --json benchmark_results/foreign_gate.json

El daemon VLM debe estar DETENIDO (protocolo estándar de los benchmarks del
capítulo) para que las páginas trigger no tarden minutos.
"""

import argparse
import json
import time
from typing import Any

import fitz
import numpy as np

import ocr_utils
from ocr_engine import OCRManager
from ocr_utils import _get_spellchecker, _get_foreign_spellchecker, _ocr_spellcheck

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
DEFAULT_SCALE = 1.2

_CUR_PAGE = "0"
RESULTS: list[dict[str, Any]] = []


def _false_gate(text: str, spanish_checker: Any,
                tokens: list[str] | None = None) -> bool:
    return False


def _clean_tokens(text: str) -> list[str]:
    tokens = [t.strip("'\".,;:!?¡¿()[]{}").lower() for t in text.split()]
    return [t for t in tokens
            if len(t) > 1 and not any(c.isdigit() for c in t)]


def _analyze_foreign(text: str) -> dict[str, Any]:
    """Para un bloque foreign=True: evidencia, correcciones forzadas, ambigüedad."""
    es_sp = _get_spellchecker()
    en_sp = _get_foreign_spellchecker("en")
    pt_sp = _get_foreign_spellchecker("pt")

    tokens = _clean_tokens(text)
    try:
        es_known = set(es_sp.known(tokens))
    except Exception:
        es_known = set()
    unknown = [t for t in tokens if t not in es_known]

    foreign_tokens: set[str] = set()
    for checker in (en_sp, pt_sp):
        if checker is None:
            continue
        try:
            fk = set(checker.known(unknown))
        except Exception:
            fk = set()
        foreign_tokens |= fk - es_known
    foreign_sorted = sorted(foreign_tokens)
    min_foreign_len = min((len(t) for t in foreign_sorted), default=0)

    # Correcciones forzadas: re-correr el spellcheck con el gate desactivado.
    orig_gate = ocr_utils._contains_foreign_latin_tokens
    ocr_utils._contains_foreign_latin_tokens = _false_gate
    try:
        corrected = _ocr_spellcheck(text)
    finally:
        ocr_utils._contains_foreign_latin_tokens = orig_gate

    changes: list[dict[str, Any]] = []
    for a, b in zip(text.split(), corrected.split()):
        if a != b:
            changes.append({
                "word": a,
                "correction": b,
                "foreign": a.strip("'\".,;:!?¡¿()[]{}").lower()
                in foreign_tokens,
            })

    simple = "?"
    try:
        from translator import _detect_language_simple
        simple = _detect_language_simple(text)
    except Exception:
        pass

    return {
        "foreign_tokens": foreign_sorted,
        "min_foreign_len": min_foreign_len,
        "n_forced": len(changes),
        "n_forced_foreign": sum(1 for c in changes if c["foreign"]),
        "changes": changes,
        "lang_simple": simple,
    }


_orig_func = ocr_utils._contains_foreign_latin_tokens


def _recording_wrapper(text: str, spanish_checker: Any,
                       tokens: list[str] | None = None) -> bool:
    res = _orig_func(text, spanish_checker, tokens)
    rec: dict[str, Any] = {
        "page": _CUR_PAGE,
        "text": text,
        "n_words": len(text.split()),
        "n_chars": len(text),
        "foreign": res,
    }
    if res:
        rec.update(_analyze_foreign(text))
    RESULTS.append(rec)
    return res


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
    global _CUR_PAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="1-53")
    ap.add_argument("--json", default="benchmark_results/foreign_gate.json")
    args = ap.parse_args()

    ocr_utils._contains_foreign_latin_tokens = _recording_wrapper
    doc = fitz.open(PDF)
    ocr = OCRManager()
    per_page: dict[str, float] = {}
    t_start = time.perf_counter()
    for pno in _parse_pages(args.pages):
        _CUR_PAGE = str(pno)
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
        print(f"pág {pno:>3}: {per_page[str(pno)]:.2f}s bloques={len(blocks)}",
              flush=True)
    total = time.perf_counter() - t_start

    calls = len(RESULTS)
    trues = [r for r in RESULTS if r["foreign"]]
    payload = {
        "pages": [str(p) for p in _parse_pages(args.pages)],
        "total_s": round(total, 2),
        "calls": calls,
        "foreign_true": len(trues),
        "trues": trues,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n== {calls} bloques llegaron al chequeo, {len(trues)} mixtos")
    for i, r in enumerate(trues, 1):
        print(f"  [{i}] pág {r['page']:>3} | {r['n_words']} palabras | "
              f"min_foreign_len={r['min_foreign_len']} | "
              f"forzadas={r['n_forced']} (extranjeras={r['n_forced_foreign']}) | "
              f"simple={r['lang_simple']} | tokens={r['foreign_tokens'][:6]}")
        print(f"      texto: {r['text'][:90]!r}")
        for c in r["changes"][:4]:
            tag = "★EXT" if c["foreign"] else "   "
            print(f"      {tag} '{c['word']}' -> '{c['correction']}'")


if __name__ == "__main__":
    main()
