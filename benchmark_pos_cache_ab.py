"""
benchmark_pos_cache_ab.py — Valida el cache de RECUPERACIÓN POSITIVA (P1).

Escenario real: la pág 31 del cap. 43 dispara el VLM y recupera 4 bloques en
5/5 corridas deterministas (plan §4.6 tabla ROI, 9.9 s/bloque = MEJOR ROI).
Con el cache positivo (_uocr_pos_cache, plan §11 P1), la PRIMERA pasada paga
la inferencia y guarda la recuperación por firma dHash; la SEGUNDA pasada
reinyecta los bloques SIN llamar al daemon (573 s/capítulo → ~0 en
re-procesados del mismo documento).

Protocolo (daemon VLM READY):
  1. clear_decision_cache → run_ocr(pág 31) → VLM real (~40-200 s), recupera
     bloques → _uocr_pos_cache guarda la recuperación por firma
  2. run_ocr(pág 31) otra vez (mismo proceso) → el cache positivo reinyecta
     → el daemon NO se vuelve a llamar; los bloques recuperados se mantienen

Salida: benchmark_results/pos_cache_ab.json con tiempos, llamadas VLM,
recuperación y estado del cache por pasada.
"""
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import fitz

import benchmark_production as bp
import ocr_engine
from ocr_engine import OCRManager

PAGINA = 31
DOC_ID = "pos_cache_p31_val"
OUT = Path("benchmark_results/pos_cache_ab.json")


def main() -> None:
    OCRManager.clear_decision_cache()
    ocr = OCRManager()

    vlm_calls = 0
    orig = ocr_engine.OCRManager._unlimited_ocr

    def spy(self, img):
        nonlocal vlm_calls
        vlm_calls += 1
        return orig(self, img)

    ocr_engine.OCRManager._unlimited_ocr = spy  # type: ignore[method-assign]

    doc = fitz.open(bp.PDF)
    img = bp.render_page(doc, PAGINA, bp.DEFAULT_SCALE)

    pasadas = []
    for i in range(2):
        antes_calls = vlm_calls
        t0 = time.time()
        blocks, engine, _ = ocr.run_ocr(
            img, bp.OCR_LANG, "fusion", prefilter=True,
            doc_id=DOC_ID,
        )
        dt = time.time() - t0
        llamo_vlm = vlm_calls > antes_calls
        with OCRManager._uocr_cache_lock:
            pos_doc = {k: (v[0], v[1], v[2], len(v[3]), len(v[4]))
                       for k, v in OCRManager._uocr_pos_cache.items()
                       if k.startswith(f"{DOC_ID}:")}
        pasadas.append({
            "pasada": i + 1,
            "tiempo_s": round(dt, 2),
            "llamo_vlm": llamo_vlm,
            "bloques_finales": len(blocks),
            "motores": engine,
            "pos_cache_doc": pos_doc,
        })
        print(f"[pasada {i + 1}] {dt:.1f}s | VLM={'SÍ' if llamo_vlm else 'NO'} "
              f"| bloques={len(blocks)} | motores={engine}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "benchmark": "pos_cache_ab (plan §11 P1)",
        "pagina": PAGINA,
        "pasadas": pasadas,
        "veredicto": (
            "P1 OK: 2ª pasada reinyecta del cache positivo sin llamar al daemon"
            if (len(pasadas) == 2 and pasadas[0]["llamo_vlm"]
                and not pasadas[1]["llamo_vlm"])
            else "P1 NO validado — revisar pasadas"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVeredicto: {pasadas[-1]}")


if __name__ == "__main__":
    main()
