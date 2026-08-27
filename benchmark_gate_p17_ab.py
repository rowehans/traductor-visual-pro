"""
benchmark_gate_p17_ab.py — A/B del gate de la pág 17 sobre el capítulo completo.

La pág 17 es la ÚNICA llamada VLM del cap. 43 que recupera 0 bloques (21-40 s
de inferencia sin nada útil). Este benchmark mide cuánto ahorra saltarse el VLM
en ESA página concreta, sin tocar el resto del trigger v4.2:

  - base: pipeline normal (pág 17 dispara VLM como producción).
  - gate_off: se anula el trigger SOLO de la pág 17 (wrapper de
    `_trigger_con_cache` que devuelve False cuando la página actual es 17;
    el resto de páginas pasan por el trigger y el cache reales, y la
    fase rapid-agresiva corre igual).

Uso (daemon VLM READY, 2 corridas ~15-20 min c/u):
  PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_gate_p17_ab.py \\
      --pages 1-53 --json benchmark_results/gate_p17_base.json          # base
  PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_gate_p17_ab.py \\
      --pages 1-53 --gate-off-p17 --json benchmark_results/gate_p17_off.json

Output: benchmark_results/gate_p17_{base,off}.json con per_page idéntico a
benchmark_production.py (total_s, stages, vlm_called, vlm_recovered...).
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default="1-53")
    ap.add_argument("--json", default="benchmark_results/gate_p17_base.json")
    ap.add_argument("--gate-off-p17", action="store_true",
                    help="Anula el trigger VLM SOLO de la pág 17")
    args = ap.parse_args()

    pages: list[int] = []
    for part in args.pages.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))

    ocr = ocr_engine.OCRManager()
    bp._wrap_dark_ratio()
    # Cache de decisiones fresco por corrida: el trigger v4.2 y el §8.4.1
    # persisten por firma en disco — la corrida base no debe heredar
    # decisiones de la gate-off ni viceversa.
    ocr_engine.OCRManager.clear_decision_cache()

    # ── Anulación selectiva del trigger (solo pág 17) ────────────
    current_page: dict[str, int] = {"n": 0}
    orig_trigger = ocr_engine.OCRManager._trigger_con_cache

    def trigger_wrapper(
        self: Any, firma: str | None, blocks: list[dict[str, Any]],
        avg_conf: float, has_big_panel: bool,
        force_uocr: bool, disable_uocr: bool,
    ) -> bool:
        out = orig_trigger(self, firma, blocks, avg_conf, has_big_panel,
                           force_uocr=force_uocr, disable_uocr=disable_uocr)
        if args.gate_off_p17 and current_page["n"] == 17 and out:
            reason = self._trigger_reason(
                blocks, avg_conf, has_big_panel, force_uocr=force_uocr)
            print(f"[gate-off] pág 17: trigger '{reason}' anulado — sin VLM")
            return False
        return bool(out)

    ocr_engine.OCRManager._trigger_con_cache = trigger_wrapper  # type: ignore[method-assign]

    per_page: dict[str, dict[str, Any]] = {}
    n_trigger = n_vlm = n_rapid_salv = n_yolo = n_ctd = 0
    total_t = 0.0
    total_vlm_recovered = 0
    doc = fitz.open(bp.PDF)

    for pno in pages:
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        results: dict[str, int] = {}
        bp.DARK_RATIOS.clear()
        trigger_holder: dict[str, Any] = {"triggered": False, "reason": None}
        bp._wrap_timing("_pre_filter_image", "prefilter", timings, counts)
        bp._wrap_timing("_run_rapidocr", "rapid", timings, counts)
        bp._wrap_timing("_recover_regions_with_easyocr", "ruta_c", timings, counts)
        bp._wrap_method(ocr, "_reforzar_con_unlimited", "vlm", timings, counts, results)
        bp._wrap_method(ocr, "_reforzar_con_rapid_agresivo", "rapid_aggr", timings, counts, results)
        bp._wrap_method(ocr, "_ruta_c_yolo", "yolo", timings, counts, results)
        bp._wrap_method(ocr, "_ruta_c_ctd", "ctd", timings, counts, results)
        bp._install_trigger_capture(ocr, trigger_holder)

        current_page["n"] = pno
        img_bgr = bp.render_page(doc, pno, bp.DEFAULT_SCALE)
        t0 = time.time()
        blocks, engine_used, _ = ocr.run_ocr(img_bgr, bp.OCR_LANG, "fusion", prefilter=True)
        dt = time.time() - t0
        total_t += dt
        avg_conf = (float(sum(b.get("confidence", 0) for b in blocks)) / len(blocks)
                    if blocks else 0.0)

        trigger = bool(trigger_holder["triggered"])
        vlm_called = counts.get("vlm", 0) > 0
        rapid_salvaged = results.get("rapid_aggr_salvaged", 0) > 0
        negative_skip = trigger and counts.get("rapid_aggr", 0) == 0 and not vlm_called
        if trigger:
            n_trigger += 1
        if vlm_called:
            n_vlm += 1
        if rapid_salvaged:
            n_rapid_salv += 1
        if counts.get("yolo", 0) > 0:
            n_yolo += 1
        if counts.get("ctd", 0) > 0:
            n_ctd += 1
        vlm_recovered = results.get("vlm_recovered", 0)
        total_vlm_recovered += vlm_recovered

        per_page[str(pno)] = {
            "total_s": round(dt, 3),
            "n_blocks": len(blocks),
            "avg_conf": round(avg_conf, 3),
            "trigger_v42": trigger,
            "trigger_reason": trigger_holder["reason"],
            "vlm_called": vlm_called,
            "vlm_recovered": vlm_recovered,
            "rapid_aggr_salvaged": rapid_salvaged,
            "negative_skip_841": negative_skip,
            "stages": {k: round(v, 3) for k, v in timings.items()},
        }
        print(f"pág {pno:>2}: {dt:6.2f}s  bloques={len(blocks):>2}  "
              f"conf={avg_conf:.2f}  trigger={trigger}({trigger_holder['reason']})  "
              f"vlm={vlm_called}  rec={vlm_recovered}")

    result = {
        "benchmark": "benchmark_gate_p17_ab.py",
        "gate_off_p17": args.gate_off_p17,
        "pages": pages,
        "total_s": round(total_t, 3),
        "avg_s_per_page": round(total_t / len(pages), 3),
        "trigger_v42_count": n_trigger,
        "vlm_called_count": n_vlm,
        "vlm_recovered_total": total_vlm_recovered,
        "rapid_aggr_salvaged_count": n_rapid_salv,
        "yolo_pages": n_yolo,
        "ctd_pages": n_ctd,
        "per_page": per_page,
    }
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}  (total {total_t:.1f}s, {n_trigger} triggers, "
          f"{n_vlm} VLM, {total_vlm_recovered} recuperados)")


if __name__ == "__main__":
    main()
