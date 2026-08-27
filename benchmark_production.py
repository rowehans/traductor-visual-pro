"""Benchmark del pipeline REAL a la resolución de producción.

Los benchmarks previos del repo renderizaban a 300 dpi (2480x3509 = 8.7 MP),
pero el frontend manda el canvas a pdf.js scale 1.2 (714x1011 = 0.7 MP, 8% del
área). Este benchmark mide el flujo real de producción: tiempo por página,
desglose por etapa (prefilter/rapid/ruta_c), cuántas páginas disparan el
trigger v4.2 (VLM de 2-8 min) y cuántas corren YOLO/CTD.

Corrección (2026-08-15): el trigger NO se puede leer de `_active_diagnostics`
después de `run_ocr` (lo restaura al valor previo al salir). Se instrumenta
`_trigger_con_cache` (única llamada por página en modo fusion, línea 931 de
ocr_engine.py) para capturar la decisión real + su razón. Además se
instrumentan `_reforzar_con_unlimited` (VLM real), `_reforzar_con_rapid_agresivo`
(salvataje pre-VLM) y `_ruta_c_yolo`/`_ruta_c_ctd`.

Excluido del CI por prefijo benchmark_ (regla _PROD_PY_FILES).
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import fitz
import numpy as np

import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
DEFAULT_SCALE = 1.2  # mismo default que app.js (optimización 2.5)


def render_page(doc: Any, page_no: int, scale: float) -> Any:
    page = doc.load_page(page_no - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[:, :, ::-1].copy()


# Las funciones/métodos originales se cachean UNA vez. Cada página reinstala
# los wrappers; si envolvieran al wrapper de la página anterior (cadena), las
# llamadas posteriores escribirían en los dicts de las páginas ya serializadas
# y los contadores se inflarían (bug medido: pág 1 con 53 calls de prefilter
# en una corrida de 53). Envolver SIEMPRE la original rompe la cadena.
_ORIG_FN: dict[str, Any] = {}


def _get_orig(scope: Any, name: str) -> Any:
    key = f"{id(scope)}:{name}"
    if key not in _ORIG_FN:
        _ORIG_FN[key] = getattr(scope, name)
    return _ORIG_FN[key]


def _wrap_timing(name: str, key: str, timings: dict[str, float], counts: dict[str, int]) -> None:
    fn = _get_orig(ocr_utils, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            timings[key] = timings.get(key, 0.0) + (time.time() - t0)
            counts[key] = counts.get(key, 0) + 1

    wrapper.__name__ = f"_wrapped_{name}"
    setattr(ocr_utils, name, wrapper)


def _wrap_method(
    obj: OCRManager,
    name: str,
    key: str,
    timings: dict[str, float],
    counts: dict[str, int],
    results: dict[str, int],
) -> None:
    """Envuelve un método del OCRManager para medir tiempo y resultados."""
    fn = _get_orig(obj, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        if name == "_reforzar_con_unlimited":
            # La función muta blocks in-place (blocks[:] = merged): el delta
            # antes/después es exactamente la recuperación del VLM.
            before = len(args[2])
        out = fn(*args, **kwargs)
        dt = time.time() - t0
        timings[key] = timings.get(key, 0.0) + dt
        counts[key] = counts.get(key, 0) + 1
        if name == "_reforzar_con_rapid_agresivo" and out is True:
            results["rapid_aggr_salvaged"] = results.get("rapid_aggr_salvaged", 0) + 1
        if name == "_reforzar_con_unlimited":
            recovered = max(0, len(args[2]) - before)
            results["vlm_recovered"] = results.get("vlm_recovered", 0) + recovered
        return out

    wrapper.__name__ = f"_wrapped_{name}"
    setattr(obj, name, wrapper)


def _install_trigger_capture(ocr: OCRManager, holder: dict[str, Any]) -> None:
    """Captura la decisión real del trigger (run_ocr restaura los diagnostics).

    En modo fusion `_trigger_con_cache` se llama UNA vez por página (línea 931).
    """
    fn = _get_orig(ocr, "_trigger_con_cache")

    def wrapper(
        firma: str,
        blocks: list[dict[str, Any]],
        avg_conf: float,
        has_big_panel: bool,
        force_uocr: bool = False,
        disable_uocr: bool = False,
    ) -> bool:
        out = fn(firma, blocks, avg_conf, has_big_panel,
                 force_uocr=force_uocr, disable_uocr=disable_uocr)
        reason = ocr._trigger_reason(
            blocks, avg_conf, has_big_panel,
            force_uocr=force_uocr,
        )
        holder["triggered"] = bool(out)
        holder["reason"] = reason
        holder["n_blocks_at_trigger"] = len(blocks)
        holder["avg_conf_at_trigger"] = float(avg_conf)
        holder["has_big_panel"] = bool(has_big_panel)
        return bool(out)

    setattr(ocr, "_trigger_con_cache", wrapper)


DARK_RATIOS: list[float] = []


def _wrap_dark_ratio() -> None:
    """Captura el dark_ratio crudo de _page_dark_features (1 llamada/pág)."""
    fn = getattr(ocr_utils, "_page_dark_features", None)
    if fn is None:
        return

    def wrapper(img_bgr: Any) -> Any:
        out = fn(img_bgr)
        if out is not None:
            DARK_RATIOS.append(float(out[0]))
        return out

    wrapper.__name__ = "_wrapped_page_dark_features"
    setattr(ocr_utils, "_page_dark_features", wrapper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="1-12", help="rango de páginas, ej. 3,11 o 1-12")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--json", default="benchmark_results/production.json")
    args = ap.parse_args()

    pages: list[int] = []
    for part in args.pages.split(","):
        if "-" in part:
            a, b = part.split("-")
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))

    doc = fitz.open(PDF)
    ocr = OCRManager()
    _wrap_dark_ratio()

    per_page: dict[str, dict[str, Any]] = {}
    n_trigger = 0
    n_vlm_called = 0
    n_rapid_aggr_salvaged = 0
    n_yolo = 0
    n_ctd = 0
    total_t = 0.0
    total_vlm_recovered = 0

    for pno in pages:
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        results: dict[str, int] = {}
        DARK_RATIOS.clear()
        trigger_holder: dict[str, Any] = {"triggered": False, "reason": None}
        _wrap_timing("_pre_filter_image", "prefilter", timings, counts)
        _wrap_timing("_run_rapidocr", "rapid", timings, counts)
        _wrap_timing("_recover_regions_with_easyocr", "ruta_c", timings, counts)
        _wrap_method(ocr, "_reforzar_con_unlimited", "vlm", timings, counts, results)
        _wrap_method(ocr, "_reforzar_con_rapid_agresivo", "rapid_aggr", timings, counts, results)
        _wrap_method(ocr, "_ruta_c_yolo", "yolo", timings, counts, results)
        _wrap_method(ocr, "_ruta_c_ctd", "ctd", timings, counts, results)
        _install_trigger_capture(ocr, trigger_holder)

        img_bgr = render_page(doc, pno, args.scale)
        t0 = time.time()
        blocks, engine_used, _ = ocr.run_ocr(img_bgr, OCR_LANG, "fusion", prefilter=True)
        dt = time.time() - t0
        total_t += dt
        avg_conf = (float(np.mean([b.get("confidence", 0) for b in blocks]))
                    if blocks else 0.0)

        trigger = bool(trigger_holder["triggered"])
        vlm_called = counts.get("vlm", 0) > 0
        rapid_calls = counts.get("rapid_aggr", 0)
        rapid_salvaged = results.get("rapid_aggr_salvaged", 0) > 0
        # Decomposición del flujo del trigger v4.2:
        #  - trigger=True y ni rapid ni vlm → skip por cache de negativas §8.4.1
        #  - trigger=True, rapid salvó → sin VLM
        #  - trigger=True, rapid no salvó → VLM llamado
        negative_skip = trigger and rapid_calls == 0 and not vlm_called
        if trigger:
            n_trigger += 1
        if vlm_called:
            n_vlm_called += 1
        if rapid_salvaged:
            n_rapid_aggr_salvaged += 1
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
            "engine": engine_used,
            "trigger_v42": trigger,
            "trigger_reason": trigger_holder["reason"],
            "n_blocks_at_trigger": trigger_holder.get("n_blocks_at_trigger"),
            "avg_conf_at_trigger": trigger_holder.get("avg_conf_at_trigger"),
            "has_big_panel": trigger_holder.get("has_big_panel"),
            "dark_ratio": round(DARK_RATIOS[-1], 3) if DARK_RATIOS else None,
            "vlm_called": vlm_called,
            "vlm_recovered": vlm_recovered,
            "rapid_aggr_calls": rapid_calls,
            "rapid_aggr_salvaged": rapid_salvaged,
            "negative_skip_841": negative_skip,
            "stages": {k: round(v, 3) for k, v in timings.items()},
            "calls": dict(counts),
        }
        print(f"pág {pno:>2}: {dt:5.2f}s  bloques={len(blocks):>2}  conf={avg_conf:.2f}  "
              f"trigger={trigger}({trigger_holder['reason']})  vlm={vlm_called}  "
              f"rapid_calls={rapid_calls} rapid_salv={rapid_salvaged}  "
              f"neg_skip={negative_skip}  yolo={counts.get('yolo', 0)} "
              f"ctd={counts.get('ctd', 0)}  "
              f"etapas={ {k: round(v,1) for k,v in timings.items()} }")

    doc.close()
    n = len(pages)
    result = {
        "benchmark": "benchmark_production.py",
        "pdf": PDF,
        "scale_pdfjs": args.scale,
        "pages": pages,
        "avg_s_per_page": round(total_t / n, 3),
        "total_s": round(total_t, 3),
        "trigger_v42_count": n_trigger,
        "vlm_called_count": n_vlm_called,
        "vlm_recovered_total": total_vlm_recovered,
        "rapid_aggr_salvaged_count": n_rapid_aggr_salvaged,
        "yolo_pages": n_yolo,
        "ctd_pages": n_ctd,
        "per_page": per_page,
    }
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{pages} págs: promedio {total_t/n:.2f}s/pág | "
          f"trigger v4.2 en {n_trigger}/{n} | VLM llamado en {n_vlm_called} | "
          f"rapid-agresivo salvó {n_rapid_aggr_salvaged} | YOLO en {n_yolo} | CTD en {n_ctd}")
    print(f"Resultado: {args.json}")


if __name__ == "__main__":
    main()
