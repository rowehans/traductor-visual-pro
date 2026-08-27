"""benchmark_foreign_check.py — Costo por fase de _contains_foreign_latin_tokens
+ camino corto de sp.correction() (2026-08-16).

Dos mediciones en una:

1. La detección de tokens latinos extranjeros (en/pt) sobre los bloques
   reales del capítulo: cuántas veces se llama, cuánto tiempo total, y dónde
   se va el tiempo (tokenización, known() es, known() en/pt).

2. El camino CORTO del spellcheck (< _SPELL_CORRECTION_MIN_LEN=13, que delega
   a sp.correction() — la expansión known(edit_1)/known(edit_2) con listas
   masivas que la réplica con índice evita para las largas). Se mide por
   RANGO de longitud de la palabra:
     - cuántas palabras de cada rango pasan por sp.correction()
     - cuánto cuesta sp.correction() y sus known() internos
     - cuántas CORRECCIONES REALES produce por rango (sp.correction() != word)
     - para las de 8-12 chars (el candidato a mover a la réplica): cuánto
       tardaría la réplica _spellcheck_correction() y si produce el MISMO
       resultado — el insumo para decidir si bajar _SPELL_CORRECTION_MIN_LEN

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

# ── Camino corto de sp.correction() (2026-08-16) ────────────────
# Por rango de longitud de la palabra: llamadas, tiempo, known() interno,
# correcciones reales, y (para 8-12) el costo/resultado de la réplica.
SHORT_STATS: dict[str, Any] = {
    # rango -> {calls, total_s, known_calls, known_words, known_s,
    #           corregidas (sp.correction() != word), replica_s, replica_diff}
}
_CUR_LEN: int = 0  # longitud de la palabra actual (contexto para known_wrapper)

_orig_func = ocr_utils._contains_foreign_latin_tokens
_orig_sp_correction = None  # se llena en _install()


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
        n = len(words) if hasattr(words, "__len__") else 0
        if lang != "?":
            STATS["known_calls"][lang] += 1
            STATS["known_s"][lang] += dt
            STATS["known_words"][lang] += n
        # Atribuir los known() internos del camino corto al rango de la
        # palabra actual (solo si estamos dentro de sp.correction()).
        if _CUR_LEN and lang == "es":
            rk = range_key(_CUR_LEN)
            bucket = SHORT_STATS.setdefault(
                rk, {"calls": 0, "total_s": 0.0, "known_calls": 0,
                     "known_words": 0, "known_s": 0.0, "corregidas": 0,
                     "replica_calls": 0, "replica_s": 0.0,
                     "replica_same": 0, "replica_diff": 0})
            bucket["known_calls"] += 1
            bucket["known_words"] += n
            bucket["known_s"] += dt
        return res

    sc.SpellChecker.known = known_wrapper  # type: ignore[method-assign]

    # ── Instrumentar sp.correction() (camino corto <13) ──
    orig_correction = sc.SpellChecker.correction

    def range_key(n: int) -> str:
        if n <= 5:
            return "3-5"
        if n <= 7:
            return "6-7"
        if n <= 10:
            return "8-10"
        if n <= 12:
            return "11-12"
        if n <= 14:
            return "13-14"
        return "15+"

    def correction_wrapper(self: Any, word: str) -> Any:
        global _CUR_LEN
        _CUR_LEN = len(word)
        t0 = time.perf_counter()
        res = orig_correction(self, word)
        dt = time.perf_counter() - t0
        _CUR_LEN = 0
        rk = range_key(len(word))
        bucket = SHORT_STATS.setdefault(
            rk, {"calls": 0, "total_s": 0.0, "known_calls": 0,
                 "known_words": 0, "known_s": 0.0, "corregidas": 0,
                 "replica_calls": 0, "replica_s": 0.0,
                 "replica_same": 0, "replica_diff": 0})
        bucket["calls"] += 1
        bucket["total_s"] += dt
        if isinstance(res, str) and res.lower() != word.lower():
            bucket["corregidas"] += 1
        # Para las de 8-12 (candidato a réplica): medir el costo de la réplica
        # y comparar el resultado con el de sp.correction()
        if 8 <= len(word) <= 12:
            try:
                r0 = time.perf_counter()
                rep = ocr_utils._spellcheck_correction(es_sp, word)
                r1 = time.perf_counter() - r0
                bucket["replica_calls"] += 1
                bucket["replica_s"] += r1
                same = (rep == res) if isinstance(res, str) else (rep == word)
                if same:
                    bucket["replica_same"] += 1
                else:
                    bucket["replica_diff"] += 1
                    bucket.setdefault("diffs", []).append(
                        {"word": word, "sp": res, "replica": rep})
            except Exception:
                pass
        # Guardar también las 6-7 chars con costo alto (candidato a réplica si
        # se baja el umbral a 6): registrar resultado de la réplica.
        if 6 <= len(word) <= 7:
            try:
                r0 = time.perf_counter()
                rep = ocr_utils._spellcheck_correction(es_sp, word)
                r1 = time.perf_counter() - r0
                bucket.setdefault("replica_calls", 0)
                bucket["replica_calls"] += 1
                bucket.setdefault("replica_s", 0.0)
                bucket["replica_s"] += r1
                same = (rep == res) if isinstance(res, str) else (rep == word)
                if same:
                    bucket.setdefault("replica_same", 0)
                    bucket["replica_same"] += 1
                else:
                    bucket.setdefault("replica_diff", 0)
                    bucket["replica_diff"] += 1
                    bucket.setdefault("diffs", []).append(
                        {"word": word, "sp": res, "replica": rep})
            except Exception:
                pass
        return res

    sc.SpellChecker.correction = correction_wrapper  # type: ignore[assignment,method-assign]

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

    es_sp = _install()
    # Warm-up del índice de la réplica y del checker: la construcción lazy del
    # índice (por id(sp)) distorsionaría la primera llamada de la réplica — se
    # amortiza una sola vez por proceso en el pipeline real.
    try:
        ocr_utils._spell_words_by_len(es_sp)
        ocr_utils._spellcheck_correction(es_sp, "incorrectament")
        ocr_utils._spellcheck_correction(es_sp, "inconstitucinal")
    except Exception:
        pass
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
        "short_stats": SHORT_STATS,
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
    print("\n== Camino corto sp.correction() por rango de longitud ==")
    for rk, b in sorted(SHORT_STATS.items(),
                        key=lambda kv: (int(kv[0].split("-")[0]),)):
        ms = b["total_s"] / max(b["calls"], 1) * 1000
        print(f"   {rk:>5} chars: {b['calls']:>4} llamadas  "
              f"{b['total_s']:6.3f}s total ({ms:6.3f} ms/call)  "
              f"known {b['known_words']:>8} words ({b['known_s']:.3f}s)  "
              f"corregidas={b['corregidas']}")
        if b.get("replica_calls"):
            rms = b["replica_s"] / b["replica_calls"] * 1000
            print(f"        └─ réplica (8-12): {b['replica_calls']} calls  "
                  f"{b['replica_s']:.3f}s ({rms:.3f} ms/call)  "
                  f"mismo resultado {b['replica_same']} / difiere {b['replica_diff']}")


if __name__ == "__main__":
    main()
