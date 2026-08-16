"""Distribución de tokens generados por el VLM en las páginas del trigger v4.2.

Corre con el venv del daemon (env_uocr_gpu) — es el proceso que tiene el modelo
4-bit cargado. Carga uocr_daemon (misma carga que el daemon), envuelve
`_model.generate` para contar tokens NUEVOS por inferencia, y corre
`_run_ocr` (el flujo exacto del daemon) por página. Cada `_run_ocr` puede
disparar varias llamadas a `generate` (infer principal + re-OCR de paneles
artísticos) — se suman por página.

Uso (venv env_uocr_gpu):
  PYTHONIOENCODING=utf-8 env_uocr_gpu/Scripts/python.exe benchmark_vlm_tokens.py \
      --pages 21,25 --max_length 2048
"""
import argparse
import json
import pathlib
import time
from typing import Any

import uocr_daemon as ud

PAGE_DIR = pathlib.Path("_tmp_vlm_pages")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="21,25")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--json", default="benchmark_results/vlm_tokens.json")
    args = ap.parse_args()
    pages = [int(x) for x in args.pages.split(",")]

    print(f"== cargando modelo (igual que el daemon) ==")
    ud._load_model()
    model = ud._model

    per_call: list[dict[str, Any]] = []
    _orig_generate = model.generate

    def gen_wrapper(*gen_args: Any, **gen_kwargs: Any) -> Any:
        t0 = time.time()
        out = _orig_generate(*gen_args, **gen_kwargs)
        dt = time.time() - t0
        input_ids = gen_kwargs.get("input_ids")
        new_tokens = int(out.shape[1] - input_ids.shape[1]) if input_ids is not None else -1
        per_call.append({"new_tokens": new_tokens, "infer_call_s": round(dt, 3)})
        return out

    model.generate = gen_wrapper

    results: dict[str, Any] = {}
    for pno in pages:
        img = PAGE_DIR / f"p{pno}.png"
        per_call.clear()
        t0 = time.time()
        res = ud._run_ocr(str(img), args.max_length)
        wall = time.time() - t0
        calls = list(per_call)
        results[str(pno)] = {
            "wall_s": round(wall, 3),
            "n_generate_calls": len(calls),
            "total_new_tokens": sum(c["new_tokens"] for c in calls),
            "per_call": calls,
            "texto_chars": len(str(res.get("text", ""))),
            "n_blocks_raw": len(res.get("blocks", [])),
        }
        print(f"pág {pno:>2}: {wall:6.1f}s | {len(calls)} llamadas generate | "
              f"{results[str(pno)]['total_new_tokens']} tokens nuevos | "
              f"bloques crudos {results[str(pno)]['n_blocks_raw']}")

    pathlib.Path(args.json).parent.mkdir(exist_ok=True)
    pathlib.Path(args.json).write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")


if __name__ == "__main__":
    main()
