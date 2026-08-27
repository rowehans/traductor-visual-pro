"""
benchmark_gate_cero_confirmado.py — Valida el gate de cero confirmado (Item 1).

Escenario real: la pág 13 del cap. 43 dispara el VLM (large_image_panel) y
recupera 0 bloques en TODAS las corridas (31-68 s/llamada). El ledger de
ceros confirmados (_uocr_neg_ceros, plan §10.2 item 1) debe suprimir la
inferencia desde el TERCER encuentro (2 fallos previos en ventanas TTL
distintas) sin tocar la recuperación de las páginas que SÍ recuperan.

Protocolo (daemon VLM READY):
  1. run_ocr(pág 13) → VLM real (31-68 s), 0 recuperación → negativa + ledger 1
  2. simular expiración del TTL corto (limpiar SOLO _uocr_neg_cache) → run_ocr
     → VLM real otra vez, 0 recuperación → ledger 2
  3. simular expiración otra vez → run_ocr → el ledger (>= 2 fallos) salta el VLM

Salida: benchmark_results/gate_cero_confirmado.json con tiempos, llamadas VLM
y decisión por pasada.
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

PAGINA = 13
DOC_ID = "gate_cero_p13_val"


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
    for i in range(3):
        antes_calls = vlm_calls
        t0 = time.time()
        blocks, engine, _ = ocr.run_ocr(
            img, bp.OCR_LANG, "fusion", prefilter=True,
            doc_id=DOC_ID,
        )
        dt = time.time() - t0
        llamo_vlm = vlm_calls > antes_calls
        with OCRManager._uocr_cache_lock:
            ceros_doc = {k: v for k, v in OCRManager._uocr_neg_ceros.items()
                         if k.startswith(f"{DOC_ID}:")}
        pasadas.append({
            "pasada": i + 1,
            "tiempo_s": round(dt, 2),
            "llamo_vlm": llamo_vlm,
            "bloques_finales": len(blocks),
            "motores": engine,
            "ceros_doc": ceros_doc,
        })
        print(f"[pasada {i + 1}] {dt:.1f}s | VLM={'SÍ' if llamo_vlm else 'NO'} "
              f"| bloques={len(blocks)} | motores={engine}")
        # Simular expiración del TTL corto entre corridas (el ledger persiste):
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()

    ok = not pasadas[0]["llamo_vlm"] is False or True  # no-op
    # Veredicto: las 2 primeras pasadas corren el VLM; la 3ª (cero confirmado) NO.
    veredicto = (
        pasadas[0]["llamo_vlm"] and pasadas[1]["llamo_vlm"]
        and not pasadas[2]["llamo_vlm"]
    )
    print(f"\nVeredicto: {veredicto}")
    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    (out / "gate_cero_confirmado.json").write_text(
        json.dumps({"pagina": PAGINA, "pasadas": pasadas,
                    "veredicto": veredicto},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    ocr_engine.OCRManager._unlimited_ocr = orig  # type: ignore[method-assign]


if __name__ == "__main__":
    main()
