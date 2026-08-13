#!/usr/bin/env python3
"""Benchmark de extracción con el tier comic-text-detector (Paso 5, PLAN_MANGA_OCR).

Compara sobre las MISMAS páginas reales, en el MISMO proceso:

  A) SIN el tier:  OCRManager.run_ocr() modo fusion (EasyOCR GPU + RapidOCR +
                   YOLO de globos Fase 6, refuerzo VLM apagado).
  B) CON el tier:  A + regiones comic-text-detector (CPU, ONNX) → Ruta C
                   (_recover_regions_with_easyocr, upscale 3.5×) → merge con
                   _fusionar_blocks_multi. Es EXACTAMENTE el pase aditivo que
                   el Paso 4 integrará en OCRManager.

Mide, por página:
  - tiempo del pipeline base (t_base), detección CTD (t_ctd_det), re-OCR Ruta C
    (t_ctd_rec) y merge (t_ctd_merge)
  - tasa de detección: bloques únicos sin el tier vs con el tier (nuevos = n_b - n_a)
  - VRAM (nvidia-smi, muestreo en hilo): pico total, pico por página y el delta
    que aporta el tier CTD (debe ser 0 — el tier corre 100% en CPU).

Uso:
    env/Scripts/python.exe tools/benchmark_extraccion_ctd.py [--pdf X] [--pages 2-6]
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from ocr_engine import OCRManager  # noqa: E402
from ocr_utils import (  # noqa: E402
    _detect_text_regions_comic_detector,
    _recover_regions_with_easyocr,
    _fusionar_blocks_multi,
)


class _VramSampler(threading.Thread):
    """Muestrea nvidia-smi memory.used cada 0.4s en un hilo daemon; guarda el
    pico acumulado (monótono) y la serie temporal."""

    def __init__(self, intervalo: float = 0.4):
        super().__init__(daemon=True)
        self.intervalo = intervalo
        self.pico = 0
        self.muestras: list[tuple[float, int]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                mb = int(out.splitlines()[0])
                self.muestras.append((time.time(), mb))
                self.pico = max(self.pico, mb)
            except Exception:
                pass
            self._stop.wait(self.intervalo)

    def stop(self) -> None:
        self._stop.set()


def _pdf_por_defecto() -> Path:
    pdfs = sorted(p for p in ROOT.glob("*.pdf") if not p.name.startswith("_"))
    return pdfs[0] if pdfs else Path("")

def _render(pdf: Path, n: int, zoom: float) -> np.ndarray:
    import fitz
    with fitz.open(str(pdf)) as doc:
        pix = doc.load_page(n - 1).get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n >= 3:
        img = img[:, :, :3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, ::-1].copy()  # RGB → BGR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pdf", default=str(_pdf_por_defecto()))
    parser.add_argument("--pages", default="2-6", help="páginas a medir (1-indexado)")
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--out", default="benchmark_ctd_results.json")
    args = parser.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[bench-ctd] ERROR: no existe el PDF: {pdf}")
        return 1
    a, _, b = args.pages.partition("-")
    paginas = list(range(int(a), int(b) + 1))
    print(f"[bench-ctd] {pdf.name} | páginas {paginas} | zoom {args.zoom}")

    # Extracción pura: sin refuerzo VLM (el daemon 5177 sigue vivo y no debe
    # participar — el benchmark mide el tier CTD, no el VLM).
    config.UOCR_ENABLED = False

    sampler = _VramSampler()
    sampler.start()
    try:
        time.sleep(1.0)  # primera muestra estable (solo daemon)
        vram_idle = sampler.pico

        mgr = OCRManager()
        # Precalentamiento: carga EasyOCR GPU + RapidOCR + YOLO + CTD en la
        # primera página (se descarta) — las mediciones por página son steady-state.
        warm = _render(pdf, paginas[0], args.zoom)
        t0 = time.time()
        mgr.run_ocr(warm, "es", "fusion", doc_id="bench-warm")
        _detect_text_regions_comic_detector(warm)
        t_warm = time.time() - t0
        vram_warm = sampler.pico
        print(f"[bench-ctd] precalentamiento (carga de motores): "
              f"{t_warm:.1f}s | VRAM pico {vram_warm} MiB")

        peso = config.OCR_ENGINE_WEIGHTS["easyocr"]
        por_pagina: list[dict] = []
        for n in paginas:
            img = _render(pdf, n, args.zoom)
            # ── A) pipeline base (fusion sin CTD) ──
            t0 = time.time()
            blocks_a, engine, engines = mgr.run_ocr(
                img, "es", "fusion", doc_id=f"bench-{n}")
            t_base = time.time() - t0
            vram_base = sampler.pico
            # ── B) pase aditivo del tier CTD ──
            t0 = time.time()
            regiones = _detect_text_regions_comic_detector(img)
            t_ctd_det = time.time() - t0
            t0 = time.time()
            ctd_blocks = _recover_regions_with_easyocr(
                img, regiones, lang_hint="es", upscale=3.5)
            t_ctd_rec = time.time() - t0
            t0 = time.time()
            merged = _fusionar_blocks_multi(
                [blocks_a, ctd_blocks],
                weights=[peso, config.OCR_ENGINE_WEIGHTS["yolo"]],
            )
            t_ctd_merge = time.time() - t0
            vram_ctd = sampler.pico
            n_a = len(blocks_a)
            n_b = len(merged)
            por_pagina.append({
                "pagina": n,
                "n_a": n_a, "n_b": n_b, "nuevos": n_b - n_a,
                "n_regiones_ctd": len(regiones),
                "n_recuperados_ctd": len(ctd_blocks),
                "t_base_s": round(t_base, 2),
                "t_ctd_det_s": round(t_ctd_det, 2),
                "t_ctd_rec_s": round(t_ctd_rec, 2),
                "t_ctd_merge_s": round(t_ctd_merge, 2),
                "vram_base_mib": vram_base,
                "vram_ctd_mib": vram_ctd,
                "vram_ctd_delta_mib": vram_ctd - vram_base,
                "engines": engines,
            })
            fila = por_pagina[-1]
            print(
                f"  p{n}: A={n_a} bloques | B={n_b} (+{fila['nuevos']}) | "
                f"CTD {fila['n_regiones_ctd']} regiones → "
                f"{fila['n_recuperados_ctd']} bloques | "
                f"t={t_base:.1f}+{t_ctd_det:.2f}+{t_ctd_rec:.1f}s | "
                f"VRAM pico {vram_ctd} MiB (ΔCTD {fila['vram_ctd_delta_mib']})")

        tot = {
            "pdf": pdf.name, "paginas": paginas, "zoom": args.zoom,
            "precalentamiento_s": round(t_warm, 1),
            "vram_idle_mib": vram_idle,
            "vram_warm_mib": vram_warm,
            "vram_pico_total_mib": sampler.pico,
            "n_a_total": sum(p["n_a"] for p in por_pagina),
            "n_b_total": sum(p["n_b"] for p in por_pagina),
            "nuevos_total": sum(p["nuevos"] for p in por_pagina),
            "n_regiones_ctd_total": sum(p["n_regiones_ctd"] for p in por_pagina),
            "n_recuperados_ctd_total": sum(p["n_recuperados_ctd"] for p in por_pagina),
            "t_base_total_s": round(sum(p["t_base_s"] for p in por_pagina), 2),
            "t_ctd_det_total_s": round(sum(p["t_ctd_det_s"] for p in por_pagina), 2),
            "t_ctd_rec_total_s": round(sum(p["t_ctd_rec_s"] for p in por_pagina), 2),
            "t_ctd_merge_total_s": round(sum(p["t_ctd_merge_s"] for p in por_pagina), 2),
            "por_pagina": por_pagina,
        }
        Path(args.out).write_text(
            json.dumps(tot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[bench-ctd] TOTAL: {tot['n_a_total']} → {tot['n_b_total']} bloques "
              f"(+{tot['nuevos_total']}) | {tot['n_regiones_ctd_total']} regiones CTD → "
              f"{tot['n_recuperados_ctd_total']} recuperados | "
              f"t_base {tot['t_base_total_s']}s + CTD {tot['t_ctd_det_total_s'] + tot['t_ctd_rec_total_s']}s | "
              f"VRAM pico {tot['vram_pico_total_mib']} MiB (ΔCTD 0 esperado)")
        print(f"[bench-ctd] resultados → {args.out}")
        return 0
    finally:
        sampler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
