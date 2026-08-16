"""benchmark_spellcheck_ab.py — Calibración del límite de edición por longitud.

Mide las correcciones REALES que `_ocr_spellcheck` aplica sobre el capítulo
completo (53 págs, pipeline de producción) junto con la distancia Damerau
REQUERIDA para cada una, y evalúa si un límite de edición dependiente de la
longitud (p. ej. edición 1 para 3-5 chars, 2 para 6-14, 0 para >14) las
preserva sin perder ninguna.

Uso:
  python benchmark_spellcheck_ab.py --collect --pages 1-53 --json records.json
      # Corre el pipeline de producción con instrumentación y guarda
      # (palabra -> corrección, camino replica/corto) por página.
  python benchmark_spellcheck_ab.py --analyze --json records.json
      # Carga los registros, calcula la distancia requerida por corrección,
      # evalúa el schedule propuesto (+ variantes) y hace el replay de tiempo.

El daemon VLM debe estar DETENIDO para --collect (mismo estado que los
benchmarks del capítulo) — el trigger degrada en las 11 páginas, pero el
spellcheck corre sobre los bloques OCR independientemente del VLM.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import fitz

import ocr_utils
from ocr_engine import OCRManager
from ocr_utils import _get_spellchecker, _spell_candidates, _remove_diacritics

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
DEFAULT_SCALE = 1.2  # mismo default que app.js (optimización 2.5)

# ── Instrumentación ──────────────────────────────────────────────
RECORDS: list[dict[str, Any]] = []
SPELL_TIME = {"replica": 0.0, "short": 0.0, "n_replica": 0, "n_short": 0}


def _install_recording() -> Any:
    """Parchea _spellcheck_correction y SpellChecker.correction para grabar."""
    es_sp = _get_spellchecker()

    orig_replica = ocr_utils._spellcheck_correction

    def replica_wrapper(sp: Any, word: str) -> str | None:
        t0 = time.perf_counter()
        res = orig_replica(sp, word)
        SPELL_TIME["replica"] += time.perf_counter() - t0
        SPELL_TIME["n_replica"] += 1
        RECORDS.append({"word": word, "result": res, "path": "replica"})
        return res

    ocr_utils._spellcheck_correction = replica_wrapper

    # SpellChecker tiene __slots__ sin __dict__ → parchear la CLASE y filtrar
    # por identidad con el checker es (el único que usa correction()).
    import spellchecker.spellchecker as sc

    orig_corr: Callable[[Any, Any], Any] = sc.SpellChecker.correction

    def corr_wrapper(self: Any, word: Any) -> Any:
        t0 = time.perf_counter()
        res = orig_corr(self, word)
        if self is es_sp:
            SPELL_TIME["short"] += time.perf_counter() - t0
            SPELL_TIME["n_short"] += 1
            RECORDS.append(
                {"word": str(word), "result": res, "path": "short"}
            )
        return res

    sc.SpellChecker.correction = corr_wrapper  # type: ignore[method-assign]
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


def collect(pages: list[int], out: str) -> None:
    _install_recording()
    doc = fitz.open(PDF)
    ocr = OCRManager()
    per_page: dict[str, float] = {}
    t_start = time.perf_counter()
    for pno in pages:
        page = doc.load_page(pno - 1)
        mat = fitz.Matrix(DEFAULT_SCALE, DEFAULT_SCALE)
        pix = page.get_pixmap(matrix=mat)
        import numpy as np

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
        "pages": [str(p) for p in pages],
        "per_page_s": per_page,
        "total_s": round(total, 2),
        "spellcheck_s": {k: round(v, 4) for k, v in SPELL_TIME.items()},
        "records": RECORDS,
        "n_records": len(RECORDS),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    n_corr = sum(
        1 for r in RECORDS
        if r["result"] is not None and r["result"] != r["word"]
    )
    print(
        f"\n{len(pages)} págs en {total:.1f}s | spellcheck replica "
        f"{SPELL_TIME['replica']:.3f}s ({SPELL_TIME['n_replica']} calls) + "
        f"short {SPELL_TIME['short']:.3f}s ({SPELL_TIME['n_short']} calls) | "
        f"{n_corr} correcciones | -> {out}"
    )


# ── Análisis ─────────────────────────────────────────────────────
def _required_distance(sp: Any, word: str, corr: str) -> int:
    """1 si la corrección está a distancia 1, 2 si solo a distancia 2."""
    if corr in _spell_candidates(sp, word, 1):
        return 1
    return 2


def _replay(sp: Any, word: str, max_dist: int) -> str | None:
    """Selección idéntica a _spellcheck_correction con límite max_dist."""
    dictionary = sp.word_frequency.dictionary
    if word in dictionary:
        return word
    if max_dist < 1:
        return None
    pool = _spell_candidates(sp, word, 1)
    if not pool and max_dist >= 2:
        pool = _spell_candidates(sp, word, 2)
    if not pool:
        return None
    wn = _remove_diacritics(word)
    diac = [c for c in pool if _remove_diacritics(c) == wn]
    if diac:
        pool = diac
    freq_get = getattr(dictionary, "get", None)
    if freq_get is None:
        return max(pool)
    return max(pool, key=lambda c: freq_get(c, 0) or 0)


SCHEDULES: dict[str, dict[str, int]] = {
    "actual (1->2, <13 a pyspell)": {3: 2, 6: 2, 13: 2, 15: 2},
    "propuesto (1/2/0)": {3: 1, 6: 2, 13: 2, 15: 0},
    "variante A (1/2/1)": {3: 1, 6: 2, 13: 2, 15: 1},
    "todo distancia 1": {3: 1, 6: 1, 13: 1, 15: 1},
}


def _max_dist_for(sched: dict[str, int], length: int) -> int:
    if length <= 5:
        return sched[3]
    if length <= 14:
        return sched[6]
    return sched[15]


def analyze(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sp = _get_spellchecker()
    records = payload["records"]
    corrections = [
        r for r in records
        if r["result"] is not None and r["result"] != r["word"]
    ]
    print(
        f"{len(records)} llamadas ({payload['spellcheck_s']}), "
        f"{len(corrections)} correcciones reales"
    )

    # Distancia requerida por corrección, agrupada por longitud.
    analyzed: list[dict[str, Any]] = []
    for r in corrections:
        w, c = r["word"], r["result"]
        req = _required_distance(sp, w, c)
        analyzed.append(
            {"word": w, "corr": c, "len": len(w), "req": req, "path": r["path"]}
        )
    buckets = [("3-5", 3, 5), ("6-12", 6, 12), ("13-14", 13, 14), ("15-21", 15, 21)]
    print("\nDistribución por longitud × distancia requerida:")
    print(f"{'longitud':<8}{'dist 1':>8}{'dist 2':>8}{'total':>8}")
    for name, lo, hi in buckets:
        grp = [a for a in analyzed if lo <= a["len"] <= hi]
        d1 = sum(1 for a in grp if a["req"] == 1)
        d2 = sum(1 for a in grp if a["req"] == 2)
        print(f"{name:<8}{d1:>8}{d2:>8}{len(grp):>8}")
        for a in grp:
            print(f"   len={a['len']:>2} req={a['req']} '{a['word']}' -> '{a['corr']}'")

    # Evaluación de schedules: qué correcciones se pierden + tiempo de replay.
    print("\nEvaluación de schedules:")
    print(f"{'schedule':<32}{'perdidas':>9}{'preservadas':>12}{'replay s':>10}")
    for name, sched in SCHEDULES.items():
        lost = []
        for a in analyzed:
            md = _max_dist_for(sched, a["len"])
            if md == 0:
                lost.append(a)
                continue
            replay_res = _replay(sp, a["word"], md)
            if replay_res != a["corr"]:
                lost.append(a)
        t0 = time.perf_counter()
        n_all = 0
        for r in records:
            md = _max_dist_for(sched, len(r["word"]))
            _replay(sp, r["word"], md)
            n_all += 1
        dt = time.perf_counter() - t0
        print(
            f"{name:<32}{len(lost):>9}{len(analyzed) - len(lost):>12}"
            f"{dt:>10.3f}"
        )
        for a in lost[:8]:
            print(f"   PIERDE len={a['len']:>2} req={a['req']} "
                  f"'{a['word']}' -> '{a['corr']}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--pages", default="1-53")
    ap.add_argument("--json", default="benchmark_results/spellcheck_ab_records.json")
    args = ap.parse_args()
    if args.collect:
        collect(_parse_pages(args.pages), args.json)
    if args.analyze:
        analyze(args.json)


if __name__ == "__main__":
    main()
