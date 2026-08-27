"""
benchmark_ctd_gate_ab.py — Distribución del gate de CTD (plan §10.2 item 3).

El CTD (comic-text-detector) cuesta 30.8 s por 21 bloques (1.47 s/bloque, el
peor ROI del residual). El gate actual: skip si len(blocks) >= 3 Y conf >= 0.35
(página bien detectada tras YOLO → CTD no aporta). Este benchmark corre el
capítulo (daemon VLM DETENIDO) capturando por página los INPUTS del gate
(len/avg_conf post-YOLO) y la recuperación real de CTD, para evaluar con datos
si un gate más estricto (conf 0.35→0.30 o MIN_BLOCKS 3→4) salva tiempo sin
perder los 21 bloques.

Uso (daemon DETENIDO):
  PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_ctd_gate_ab.py
"""
import json
import time
from pathlib import Path
from typing import Any

import fitz

import benchmark_production as bp
import ocr_engine
from ocr_engine import OCRManager


def main() -> None:
    from config import (COMIC_DETECTOR_GATE_MIN_BLOCKS,
                        COMIC_DETECTOR_GATE_MAX_CONF)
    OCRManager.clear_decision_cache()
    ocr = OCRManager()

    registros: dict[int, dict[str, Any]] = {}
    total_ctd_s = 0.0
    total_ctd_bloques = 0
    ctd_paginas = 0

    orig = OCRManager._ruta_c_ctd

    def spy(self, img_bgr, ocr_lang, blocks, avg_conf, yolo_regions):
        nonlocal total_ctd_s, total_ctd_bloques, ctd_paginas
        gate_skip = (len(blocks) >= COMIC_DETECTOR_GATE_MIN_BLOCKS
                     and avg_conf >= COMIC_DETECTOR_GATE_MAX_CONF)
        t0 = time.time()
        res = orig(self, img_bgr, ocr_lang, blocks, avg_conf, yolo_regions)
        dt = time.time() - t0
        total_ctd_s += dt
        # res = ctd_blocks (contribución del tier PRE-merge; el merge final
        # deduplica por overlap en _run_fusion). Se cuenta len(res) como
        # bloques CTD brutos — la contribución neta real la da el merge.
        brutos = len(res) if res else 0
        if not gate_skip and (dt > 0.05 or brutos > 0):
            ctd_paginas += 1
            total_ctd_bloques += brutos
            registros[self._pagina_actual] = {
                "len_post_yolo": len(blocks), "avg_conf": round(avg_conf, 3),
                "gate_skip": bool(gate_skip),
                "ctd_s": round(dt, 3), "bloques_ctd_brutos": brutos,
            }
        return res

    ocr_engine.OCRManager._ruta_c_ctd = spy  # type: ignore[method-assign]
    doc = fitz.open(bp.PDF)
    for pno in range(1, 54):
        OCRManager._pagina_actual = pno  # type: ignore[attr-defined]
        # daemon caído pero pipeline COMPLETO (YOLO + CTD activos): el
        # _unlimited_ocr degrada solo el VLM vía RuntimeError (como producción
        # con el daemon indisponible). disable_uocr NO se usa — apagaría el tier.
        img = bp.render_page(doc, pno, bp.DEFAULT_SCALE)
        ocr.run_ocr(img, bp.OCR_LANG, "fusion", prefilter=True)

    ocr_engine.OCRManager._ruta_c_ctd = orig  # type: ignore[method-assign]

    print(f"\n[CTD] {ctd_paginas} págs corrieron CTD | {total_ctd_s:.1f}s | "
          f"{total_ctd_bloques} bloques nuevos | {ctd_paginas and total_ctd_s / ctd_paginas:.2f}s/pág")

    # ── Evaluación de umbrales alternativos ──────────────────────
    def _perder(umbrales: tuple[int, float]) -> tuple[float, int]:
        """Con umbrales (min_blocks, max_conf) alternativos, cuánto tiempo de
        CTD se ahorraría y cuántos bloques se perderían."""
        mb, mc = umbrales
        ahorro = 0.0
        perdidos = 0
        for r in registros.values():
            if r["gate_skip"]:
                continue  # ya no corría con el gate actual
            # Con el gate NUEVO, ¿se saltaría?
            if r["len_post_yolo"] >= mb and r["avg_conf"] >= mc:
                ahorro += r["ctd_s"]
                perdidos += r["nuevos_ctd"]
        return round(ahorro, 1), perdidos

    print("\nUmbrales alternativos (min_blocks, max_conf):")
    for label, umbrales in [
        ("actual  (3, 0.35)", (3, 0.35)),
        ("estricto conf (3, 0.30)", (3, 0.30)),
        ("estricto blocks (4, 0.35)", (4, 0.35)),
        ("muy estricto (4, 0.30)", (4, 0.30)),
    ]:
        ahorro, perdidos = _perder(umbrales)
        print(f"  {label}: ahorro {ahorro}s | bloques perdidos {perdidos}")

    print("\nPáginas CTD por inputs del gate (ordenadas por conf):")
    for pno in sorted(registros, key=lambda k: (registros[k]["avg_conf"],
                                                registros[k]["len_post_yolo"])):
        r = registros[pno]
        print(f"  pág {pno:>2}: len={r['len_post_yolo']} conf={r['avg_conf']:.2f} "
              f"| {r['ctd_s']}s | +{r['nuevos_ctd']} bloques")

    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    (out / "ctd_gate_ab.json").write_text(
        json.dumps({"por_pagina": {str(k): v for k, v in registros.items()},
                    "total_ctd_s": round(total_ctd_s, 2),
                    "total_ctd_bloques": total_ctd_bloques,
                    "ctd_paginas": ctd_paginas},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: benchmark_results/ctd_gate_ab.json")


if __name__ == "__main__":
    main()
