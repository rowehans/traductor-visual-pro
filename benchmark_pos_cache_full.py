"""
benchmark_pos_cache_full.py — Mide el ahorro del cache de recuperación
positiva VLM (plan §11 P1) en el capítulo COMPLETO (53 páginas).

Protocolo (daemon VLM READY):
  1. clear_decision_cache → benchmark_production sobre las 53 págs
     (pasada 1 = baseline — el VLM corre 11 llamadas, ~573 s stage VLM)
  2. benchmark_production sobre las MISMAS 53 págs SIN clear
     (pasada 2 = cache caliente — las 9 págs con recuperación reinyectan;
     las 2 gated (13/17) saltan por ledger de ceros)

Salida: benchmark_results/pos_cache_full.json con tiempos, VLM calls,
recuperación, y ahorro por página.
"""
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import fitz

import benchmark_production as bp
import ocr_engine
from ocr_engine import OCRManager

DOC_ID = "pos_cache_full_v1"
OUT = Path("benchmark_results/pos_cache_full.json")


def _run_chapter(ocr, doc, tag, vlm_counter):
    """Procesa el capítulo completo y retorna métricas por página."""
    results = []
    t0 = time.time()
    for page_no in range(1, doc.page_count + 1):
        img = bp.render_page(doc, page_no, bp.DEFAULT_SCALE)
        antes = vlm_counter["count"]
        t_start = time.time()
        blocks, engine, engines = ocr.run_ocr(
            img, bp.OCR_LANG, "fusion", prefilter=True,
            doc_id=DOC_ID,
        )
        dt = time.time() - t_start
        llamado_vlm = vlm_counter["count"] > antes
        with OCRManager._uocr_cache_lock:
            n_pos = len(OCRManager._uocr_pos_cache)
            n_neg = len(OCRManager._uocr_neg_cache)
            n_cero = len(OCRManager._uocr_neg_ceros)
        results.append({
            "page": page_no,
            "time_s": round(dt, 2),
            "vlm_called": llamado_vlm,
            "blocks": len(blocks),
            "engines": engines,
        })
        print(f"  [{tag} p{page_no:2d}] {dt:5.1f}s | VLM={'SÍ' if llamado_vlm else 'no'} "
              f"| blk={len(blocks):3d} | {','.join(engines)}")
    wall = round(time.time() - t0, 1)
    n_vlm = sum(1 for r in results if r["vlm_called"])
    vlm_time = sum(r["time_s"] for r in results if r["vlm_called"])
    return {
        "tag": tag,
        "wall_s": wall,
        "vlm_calls": n_vlm,
        "vlm_time_s": round(vlm_time, 1),
        "pages": results,
        "pos_cache_size": n_pos,
        "neg_cache_size": n_neg,
        "cero_cache_size": n_cero,
    }


def main():
    OCRManager.clear_decision_cache()
    ocr = OCRManager()
    vlm_counter = {"count": 0}
    orig = ocr_engine.OCRManager._unlimited_ocr

    def spy(self, img):
        vlm_counter["count"] += 1
        return orig(self, img)

    ocr_engine.OCRManager._unlimited_ocr = spy

    doc = fitz.open(bp.PDF)
    print(f"Capítulo: {doc.page_count} páginas, daemon VLM activo\n")

    # Pasada 1: baseline (VLM corre todas las que necesita)
    print("=" * 60)
    print("  PASADA 1 — Baseline (VLM real)")
    print("=" * 60)
    p1 = _run_chapter(ocr, doc, "P1", vlm_counter)

    # No limpiar el cache — la pasada 2 debe reusar las recuperaciones
    print(f"\n  Cache post-P1: pos={p1['pos_cache_size']}, "
          f"neg={p1['neg_cache_size']}, cero={p1['cero_cache_size']}")

    # Pasada 2: cache caliente (misma decisión del trigger, reinyección del cache)
    print("\n" + "=" * 60)
    print("  PASADA 2 — Cache caliente (VLM reinyectado)")
    print("=" * 60)
    p2 = _run_chapter(ocr, doc, "P2", vlm_counter)

    # Calcular ahorro
    p1_vlm_pages = [r for r in p1["pages"] if r["vlm_called"]]
    p2_reinjected = [r for r in p2["pages"] if not r["vlm_called"]
                     and any(pr["page"] == r["page"] and pr["vlm_called"]
                             for pr in p1["pages"])]
    saved_time = sum(pr["time_s"] for pr in p1_vlm_pages) - sum(
        r["time_s"] for r in p2["pages"]
        if any(pr["page"] == r["page"] and pr["vlm_called"] for pr in p1["pages"])
    )

    summary = {
        "benchmark": "pos_cache_full — Cache de recuperación positiva VLM",
        "paginas": doc.page_count,
        "daemon": "ready",
        "p1": {
            "wall_s": p1["wall_s"],
            "vlm_calls": p1["vlm_calls"],
            "vlm_time_s": p1["vlm_time_s"],
        },
        "p2": {
            "wall_s": p2["wall_s"],
            "vlm_calls": p2["vlm_calls"],
            "vlm_time_s": p2["vlm_time_s"],
        },
        "ahorro": {
            "wall_s": round(p1["wall_s"] - p2["wall_s"], 1),
            "wall_pct": round((1 - p2["wall_s"] / p1["wall_s"]) * 100, 1)
            if p1["wall_s"] else 0,
            "vlm_calls_saved": p1["vlm_calls"] - p2["vlm_calls"],
            "vlm_time_saved_s": round(p1["vlm_time_s"] - p2["vlm_time_s"], 1),
        },
        "veredicto": (
            f"P1 OK: {p2['vlm_calls']} llamadas VLM en P2 vs {p1['vlm_calls']} "
            f"en P1; ahorro {round(p1['wall_s'] - p2['wall_s'], 1)}s "
            f"({round((1 - p2['wall_s'] / max(p1['wall_s'], 0.1)) * 100, 1)}%)"
        ),
        "pages_detail_p1": p1["pages"],
        "pages_detail_p2": p2["pages"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  P1 (baseline)  : {p1['wall_s']}s | VLM={p1['vlm_calls']} calls | "
          f"VLM stage={p1['vlm_time_s']}s")
    print(f"  P2 (cache hot) : {p2['wall_s']}s | VLM={p2['vlm_calls']} calls | "
          f"VLM stage={p2['vlm_time_s']}s")
    print(f"  AHORRO          : {summary['ahorro']['wall_s']}s "
          f"({summary['ahorro']['wall_pct']}%)")
    print(f"  VLM calls saved : {summary['ahorro']['vlm_calls_saved']}")
    print(f"  VLM time saved  : {summary['ahorro']['vlm_time_saved_s']}s")
    print("=" * 60)
    print(f"  JSON: {OUT}")


if __name__ == "__main__":
    main()
