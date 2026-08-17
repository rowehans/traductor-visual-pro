"""benchmark_spell_touchpoints.py — Costo real por página de CADA punto del
pipeline que toca pyspellchecker (correction, known, candidates, índice,
réplica, detección extranjera), dentro y fuera de _ocr_spellcheck.

Mapa de touchpoints (todos confluyen en _group_and_merge_blocks -> _ocr_spellcheck,
línea 3119 de ocr_utils.py; el único otro uso es el preload de server.py, one-time):

  - sc.SpellChecker.correction   : camino corto (<13 chars) dentro de _ocr_spellcheck
  - known() interno de correction : la expansión known(edit_1)/known(edit_2) masiva
  - sc.SpellChecker.known (foreign): el chequeo extranjero (es/en/pt)
  - _spellcheck_correction        : la réplica (camino largo >=13)
  - _spell_candidates             : candidatos por longitud + Damerau acotado
  - _spell_words_by_len           : índice lazy por longitud (se construye en el
                                    PRIMER uso real, sin warm-up aquí)
  - _contains_foreign_latin_tokens: detección extranjera (wrapper de sitio)
  - _ocr_spellcheck               : total por bloque (referencia)

Atribución por pila de sitios (el pipeline es síncrono): cada wrapper entra a
su sitio al empezar y sale al terminar; los known()/contadores internos se
atribuyen al sitio activo de la pila. Esto separa el known() del camino corto
del known() del foreign check, y el índice de los candidatos de la réplica.

Uso:
  python benchmark_spell_touchpoints.py --json benchmark_results/spell_touchpoints.json

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

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
DEFAULT_SCALE = 1.2

# ── Contadores por sitio (totales del capítulo) ──────────────────
SITES: dict[str, dict[str, float]] = {}
PER_PAGE: dict[str, dict[str, dict[str, float]]] = {}
_CUR_PAGE = "0"
_STACK: list[str] = []


def _site(s: str) -> dict[str, float]:
    return SITES.setdefault(s, {"calls": 0.0, "s": 0.0, "words": 0.0})


def _enter(site: str) -> None:
    _STACK.append(site)


def _exit(site: str) -> None:
    assert _STACK and _STACK[-1] == site, f"pila corrupta: {_STACK} vs {site}"
    _STACK.pop()


def _cur() -> str:
    return _STACK[-1] if _STACK else "none"


def _record(site: str, dt: float, words: int = 0) -> None:
    s = _site(site)
    s["calls"] += 1
    s["s"] += dt
    s["words"] += words
    pp = PER_PAGE.setdefault(_CUR_PAGE, {})
    p = pp.setdefault(site, {"calls": 0.0, "s": 0.0, "words": 0.0})
    p["calls"] += 1
    p["s"] += dt
    p["words"] += words


# ── Wrappers ─────────────────────────────────────────────────────
def _install() -> None:
    import spellchecker.spellchecker as sc

    _orig_correction = sc.SpellChecker.correction

    def correction_wrapper(self: Any, word: str) -> Any:
        _enter("correction")
        t0 = time.perf_counter()
        try:
            return _orig_correction(self, word)
        finally:
            _record("correction", time.perf_counter() - t0)
            _exit("correction")

    sc.SpellChecker.correction = correction_wrapper  # type: ignore[assignment,method-assign]

    _orig_known = sc.SpellChecker.known

    def known_wrapper(self: Any, words: Any) -> Any:
        cur = _cur()
        if cur == "correction":
            site = "known_internal_correction"
        elif cur == "foreign_check":
            site = "known_foreign_check"
        else:
            site = "known_other"
        _enter(site)
        t0 = time.perf_counter()
        try:
            return _orig_known(self, words)
        finally:
            n = len(words) if hasattr(words, "__len__") else 0
            _record(site, time.perf_counter() - t0, words=n)
            _exit(site)

    sc.SpellChecker.known = known_wrapper  # type: ignore[method-assign]

    _orig_replica = ocr_utils._spellcheck_correction

    def replica_wrapper(sp: Any, word: str) -> str | None:
        _enter("replica")
        t0 = time.perf_counter()
        try:
            return _orig_replica(sp, word)
        finally:
            _record("replica", time.perf_counter() - t0)
            _exit("replica")

    ocr_utils._spellcheck_correction = replica_wrapper

    _orig_candidates = ocr_utils._spell_candidates

    def candidates_wrapper(sp: Any, word: str, max_dist: int) -> list[str]:
        _enter("candidates")
        t0 = time.perf_counter()
        try:
            return _orig_candidates(sp, word, max_dist)
        finally:
            _record("candidates", time.perf_counter() - t0)
            _exit("candidates")

    ocr_utils._spell_candidates = candidates_wrapper

    _orig_index = ocr_utils._spell_words_by_len

    def index_wrapper(sp: Any) -> dict[int, list[tuple[str, bytes]]]:
        _enter("index")
        t0 = time.perf_counter()
        try:
            return _orig_index(sp)
        finally:
            _record("index", time.perf_counter() - t0)
            _exit("index")

    ocr_utils._spell_words_by_len = index_wrapper

    _orig_foreign = ocr_utils._contains_foreign_latin_tokens

    def foreign_wrapper(text: str, spanish_checker: Any,
                        tokens: list[str] | None = None) -> bool:
        _enter("foreign_check")
        t0 = time.perf_counter()
        try:
            return _orig_foreign(text, spanish_checker, tokens)
        finally:
            _record("foreign_check", time.perf_counter() - t0)
            _exit("foreign_check")

    ocr_utils._contains_foreign_latin_tokens = foreign_wrapper

    _orig_spellcheck = ocr_utils._ocr_spellcheck

    def spellcheck_wrapper(text: str) -> str:
        t0 = time.perf_counter()
        try:
            return _orig_spellcheck(text)
        finally:
            _record("spellcheck_total", time.perf_counter() - t0)

    ocr_utils._ocr_spellcheck = spellcheck_wrapper

    # ── Contexto: langdetect (el mayor costo de la maquinaria spellcheck)
    # ── y cargas one-time de diccionarios (producción las precarga) ──
    try:
        import translator

        _orig_detect = translator._detect_language_robust

        def detect_wrapper(text: str) -> str:
            t0 = time.perf_counter()
            try:
                return _orig_detect(text)
            finally:
                _record("langdetect", time.perf_counter() - t0)

        translator._detect_language_robust = detect_wrapper  # type: ignore[assignment]
    except Exception:
        pass

    _orig_get_sp = ocr_utils._get_spellchecker

    def get_sp_wrapper(lang: str = "es") -> Any:
        t0 = time.perf_counter()
        try:
            return _orig_get_sp(lang)
        finally:
            _record("checker_load_" + lang, time.perf_counter() - t0)

    ocr_utils._get_spellchecker = get_sp_wrapper

    _orig_get_fsp = ocr_utils._get_foreign_spellchecker

    def get_fsp_wrapper(lang: str) -> Any:
        t0 = time.perf_counter()
        try:
            return _orig_get_fsp(lang)
        finally:
            _record("checker_load_" + lang, time.perf_counter() - t0)

    ocr_utils._get_foreign_spellchecker = get_fsp_wrapper


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
    ap.add_argument("--json", default="benchmark_results/spell_touchpoints.json")
    args = ap.parse_args()

    _install()
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

    payload = {
        "pages": [str(p) for p in _parse_pages(args.pages)],
        "total_s": round(total, 2),
        "sites": SITES,
        "per_page": PER_PAGE,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    total_sc = _site("spellcheck_total")["s"]
    print(f"\n== Touchpoints pyspellchecker (capítulo, {len(_parse_pages(args.pages))} págs)")
    print(f"{'sitio':>28} {'calls':>8} {'s':>9} {'%spell':>7} {'words':>10}")
    for name in sorted(SITES, key=lambda k: -SITES[k]["s"]):
        s = SITES[name]
        pct = f"{s['s'] / max(total_sc, 1e-9) * 100:6.1f}%" if total_sc else "-"
        print(f"{name:>28} {s['calls']:>8.0f} {s['s']:>9.3f} {pct:>7} "
              f"{s['words']:>10.0f}")
    print(f"  total spellcheck: {total_sc:.3f}s | pipeline: {total:.2f}s "
          f"({total/len(_parse_pages(args.pages)):.2f}s/pág)")


if __name__ == "__main__":
    main()
