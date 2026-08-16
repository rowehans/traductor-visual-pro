"""Compara la corrida con max_length=1280 (chunks) contra el baseline 2048.

Uso: PYTHONIOENCODING=utf-8 env/Scripts/python.exe tools/analizar_vlm_1280.py
     [chunks.json...] --baseline benchmark_results/production_full53.json
"""
import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", nargs="+",
                    help="JSONs de los chunks (benchmark_production.py)")
    ap.add_argument("--baseline",
                    default="benchmark_results/production_full53.json")
    args = ap.parse_args()

    merged: dict[str, dict[str, Any]] = {}
    totals = {
        "total_s": 0.0, "n": 0, "trigger": 0, "vlm_called": 0,
        "vlm_recovered": 0, "rapid_salv": 0, "yolo": 0, "ctd": 0,
        "neg_skip": 0,
    }
    for path in args.chunks:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for pno, page in d["per_page"].items():
            merged[pno] = page
            totals["total_s"] += page["total_s"]
            totals["n"] += 1
            totals["trigger"] += int(page["trigger_v42"])
            totals["vlm_called"] += int(page["vlm_called"])
            totals["vlm_recovered"] += int(page.get("vlm_recovered", 0))
            totals["rapid_salv"] += int(page.get("rapid_aggr_salvaged", False))
            totals["yolo"] += int(page.get("calls", {}).get("yolo", 0) > 0)
            totals["ctd"] += int(page.get("calls", {}).get("ctd", 0) > 0)
            totals["neg_skip"] += int(page.get("negative_skip_841", False))

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_pages = base["per_page"]

    n = totals["n"]
    print(f"== max_length=1280: {n} páginas ==  "
          f"total {totals['total_s']:.1f}s, promedio {totals['total_s']/n:.2f}s/pág")
    print(f"trigger v4.2: {totals['trigger']} | VLM llamado: {totals['vlm_called']} | "
          f"VLM recuperado (bloques): {totals['vlm_recovered']} | "
          f"rapid-agr salvó: {totals['rapid_salv']} | neg_skip: {totals['neg_skip']} | "
          f"YOLO: {totals['yolo']} CTD: {totals['ctd']}")

    print(f"\nBaseline 2048: trigger {base['trigger_v42_count']} | "
          f"VLM llamado {base['vlm_called_count']} | YOLO {base['yolo_pages']} | "
          f"CTD {base['ctd_pages']}")

    # Comparación por página del trigger + recuperación
    print("\n== Páginas del trigger: 1280 vs 2048 ==")
    print(f"{'pág':>4} | {'trigger':>7} | {'1280 rec':>8} | {'2048 rec':>8} | "
          f"{'1280 s':>8} | {'2048 s':>8} | {'nblk 1280':>9} | {'nblk 2048':>9}")
    print("-" * 78)
    pages = sorted(set(merged) | set(base_pages), key=int)
    for pno in pages:
        a = merged.get(pno)
        b = base_pages.get(pno)
        if a is None or b is None:
            print(f"{pno:>4} | {'—' if a is None else 'solo-1280':>7} | ... | solo en "
                  f"{'1280' if b is None else '2048'}")
            continue
        a_trig = int(a["trigger_v42"])
        a_rec = int(a.get("vlm_recovered", 0))
        b_rec = int(b.get("vlm_recovered", 0)) if "vlm_recovered" in b else None
        a_t = a["total_s"]
        b_t = b["total_s"]
        print(f"{pno:>4} | {a_trig!s:>7} | {a_rec!s:>8} | "
              f"{str(b_rec):>8} | {a_t:>7.1f}s | {b_t:>7.1f}s | "
              f"{a['n_blocks']:>9} | {b['n_blocks']:>9}")

    # Deltas de bloques finales entre 1280 y 2048 (recuperación del pipeline)
    block_deltas = {}
    for pno in pages:
        a, b = merged.get(pno), base_pages.get(pno)
        if a is not None and b is not None:
            db = a["n_blocks"] - b["n_blocks"]
            if db != 0:
                block_deltas[pno] = db
    if block_deltas:
        print(f"\nΔ bloques finales (1280 − 2048) distinto de 0: {block_deltas}")
    else:
        print("\nΔ bloques finales (1280 − 2048): 0 en todas las páginas compartidas")


if __name__ == "__main__":
    main()
