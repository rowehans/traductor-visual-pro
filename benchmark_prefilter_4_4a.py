"""
benchmark_prefilter_4_4a.py — A/B de diseño para 4.4A (prefilter condicional).

Pregunta: ¿cuántas páginas del capítulo completo están por debajo del umbral
que JUSTIFICARÍA correr el prefilter de 0.53 s/pág, y qué pasaría con el
trigger v4.2 si el prefilter solo corriera en esas páginas débiles?

Método (mismo-proceso, daemon detenido para evitar VLM):
  Por cada página del capítulo, corre el pipeline real (OCRManager.run_ocr,
  fusion) DOS veces:
    - raw:    prefilter=False (tier-1 sobre imagen cruda — el caso "si 4.4A
              estuviera activo y la página no es débil")
    - pref:   prefilter=True  (producción actual)
  Captura por página: bloques/conf del resultado final, trigger v4.2 y razón,
  y tiempo por variante. Con eso se computa:
    1. Distribución de calidad sin prefilter (n_blocks/conf raw) — cuántas
       páginas caerían bajo el umbral que justifica el prefilter.
    2. Delta del trigger: cuántas páginas CAMBIARÍAN su decisión v4.2 si el
       prefilter no corriera (el riesgo de 4.4A).
    3. Ahorro bruto: 0.53 s × páginas no-débiles (prefilter solo en débiles)
       contra el riesgo medido.

Uso (daemon VLM DETENIDO):
  PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_prefilter_4_4a.py \
      --pages 1-53 --json benchmark_results/prefilter_4_4a.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import fitz

import benchmark_production as bp
import ocr_engine
import ocr_utils

# Umbral de "página débil" que justificaría el prefilter (v4.2): pocos
# bloques y/o confianza baja — el prefilter recupera texto artístico que el
# tier-1 crudo pierde.
WEAK_MAX_BLOCKS = 6
WEAK_MIN_CONF = 0.60


def _stats(blocks: list[dict[str, Any]]) -> tuple[int, float]:
    n = len(blocks)
    conf = (float(sum(b.get("confidence", 0) for b in blocks)) / n
            if blocks else 0.0)
    return n, conf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default="1-53")
    ap.add_argument("--json", default="benchmark_results/prefilter_4_4a.json")
    args = ap.parse_args()

    pages: list[int] = []
    for part in args.pages.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))

    doc = fitz.open(bp.PDF)
    # Cache de decisiones fresco: las variantes raw/pref deben computar su
    # trigger sin heredar decisiones de corridas anteriores.
    ocr_engine.OCRManager.clear_decision_cache()
    per_page: dict[str, dict[str, Any]] = {}
    n_trigger_raw = n_trigger_pref = 0
    sum_raw = sum_pref = 0.0
    n_weak = 0
    n_trigger_change = 0
    n_pref_helps = 0  # páginas donde prefilter gana bloques o conf claramente

    for pno in pages:
        img = bp.render_page(doc, pno, bp.DEFAULT_SCALE)

        # ── Variante raw (sin prefilter) ──────────────────────────
        ocr_raw = ocr_engine.OCRManager()
        trigger_raw: dict[str, Any] = {"triggered": False, "reason": None}
        bp._install_trigger_capture(ocr_raw, trigger_raw)
        t0 = time.time()
        blocks_raw, _, _ = ocr_raw.run_ocr(img, bp.OCR_LANG, "fusion", prefilter=False)
        t_raw = time.time() - t0
        n_raw, conf_raw = _stats(blocks_raw)

        # ── Variante pref (producción actual) ─────────────────────
        # Firma/trigger frescos: la variante pref no debe heredar la decisión
        # cacheada por la raw (misma página, mismo layout → misma firma).
        ocr_engine.OCRManager.clear_decision_cache()
        ocr_pref = ocr_engine.OCRManager()
        trigger_pref: dict[str, Any] = {"triggered": False, "reason": None}
        bp._install_trigger_capture(ocr_pref, trigger_pref)
        t0 = time.time()
        blocks_pref, _, _ = ocr_pref.run_ocr(img, bp.OCR_LANG, "fusion", prefilter=True)
        t_pref = time.time() - t0
        n_pref, conf_pref = _stats(blocks_pref)

        weak_raw = n_raw < WEAK_MAX_BLOCKS or conf_raw < WEAK_MIN_CONF
        if weak_raw:
            n_weak += 1
        if trigger_raw["triggered"]:
            n_trigger_raw += 1
        if trigger_pref["triggered"]:
            n_trigger_pref += 1
        if trigger_raw["triggered"] != trigger_pref["triggered"]:
            n_trigger_change += 1
        # el prefilter "ayuda" si gana >= 2 bloques o sube conf >= 0.10
        if (n_pref - n_raw >= 2) or (conf_pref - conf_raw >= 0.10):
            n_pref_helps += 1
        sum_raw += t_raw
        sum_pref += t_pref

        per_page[str(pno)] = {
            "raw": {"n_blocks": n_raw, "avg_conf": round(conf_raw, 3),
                    "time_s": round(t_raw, 3),
                    "trigger": trigger_raw["triggered"],
                    "trigger_reason": trigger_raw["reason"]},
            "pref": {"n_blocks": n_pref, "avg_conf": round(conf_pref, 3),
                     "time_s": round(t_pref, 3),
                     "trigger": trigger_pref["triggered"],
                     "trigger_reason": trigger_pref["reason"]},
            "weak_raw": weak_raw,
            "delta_blocks": n_pref - n_raw,
            "delta_conf": round(conf_pref - conf_raw, 3),
        }
        print(f"pág {pno:>2}: raw {n_raw:>2}/{conf_raw:.2f} {'WEAK' if weak_raw else '    '} "
              f"pref {n_pref:>2}/{conf_pref:.2f}  Δblk={n_pref - n_raw:+d} "
              f"trigger {trigger_raw['triggered']}→{trigger_pref['triggered']} "
              f"({t_raw:.2f}s vs {t_pref:.2f}s)")

    N = len(pages)
    res = {
        "benchmark": "benchmark_prefilter_4_4a.py",
        "pages": pages, "n_pages": N,
        "weak_threshold": {"max_blocks": WEAK_MAX_BLOCKS, "min_conf": WEAK_MIN_CONF},
        "n_weak_raw": n_weak,
        "pct_weak_raw": round(100.0 * n_weak / N, 1),
        "n_trigger_raw": n_trigger_raw,
        "n_trigger_pref": n_trigger_pref,
        "n_trigger_change": n_trigger_change,
        "n_pref_helps": n_pref_helps,
        "avg_raw_s": round(sum_raw / N, 3),
        "avg_pref_s": round(sum_pref / N, 3),
        "prefilter_delta_s_per_page": round((sum_pref - sum_raw) / N, 3),
        "saved_if_conditional_s": round(0.53 * (N - n_weak), 1),
        "per_page": per_page,
    }
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n==== RESUMEN 4.4A ====")
    print(f"Páginas débiles sin prefilter (n<{WEAK_MAX_BLOCKS} o conf<{WEAK_MIN_CONF}): "
          f"{n_weak}/{N} ({res['pct_weak_raw']}%)")
    print(f"Triggers v4.2: raw {n_trigger_raw} vs pref {n_trigger_pref} — "
          f"{n_trigger_change} páginas CAMBIARÍAN de decisión")
    print(f"Páginas donde el prefilter gana claramente (≥2 bloques o +0.10 conf): {n_pref_helps}")
    print(f"Tiempo medio/pág: raw {res['avg_raw_s']}s vs pref {res['avg_pref_s']}s "
          f"(Δ prefilter {res['prefilter_delta_s_per_page']}s)")
    print(f"Ahorro si el prefilter solo corriera en débiles: ~{res['saved_if_conditional_s']}s/capítulo")
    print(f"Resultado: {args.json}")


if __name__ == "__main__":
    main()
