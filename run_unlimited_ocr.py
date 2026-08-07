"""
run_unlimited_ocr.py — Ejecuta baidu/Unlimited-OCR sobre una imagen en CPU
y escribe un JSON con {load_s, infer_s, text}.

Uso (desde env_uocr):
    python run_unlimited_ocr.py <imagen> <json_salida> [max_length]
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch

# ── CPU fallback: el código del modelo llama .cuda() de forma hardcodeada.
# En CPU neutralizamos esas llamadas (el tensor ya está en CPU).
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[method-assign]

from transformers import AutoModel, AutoTokenizer

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hf_cache", "hub", "models--baidu--Unlimited-OCR",
    "snapshots", "07dea832e22aefee32ad281d4b80551282e1c168",
)


def main():
    image_file = sys.argv[1]
    out_json = sys.argv[2]
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 32768

    t_load0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval()
    t_load = time.time() - t_load0
    print(f"[uocr] modelo cargado en {t_load:.1f}s", flush=True)

    # Directorio de salida único por imagen (permite benchmarks multi-página)
    image_stem = os.path.splitext(os.path.basename(image_file))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(out_json)), f"uocr_out_{image_stem}")
    os.makedirs(out_dir, exist_ok=True)

    t_inf0 = time.time()
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=image_file,
        output_path=out_dir,
        base_size=1024, image_size=640, crop_mode=True,
        max_length=max_length,
        no_repeat_ngram_size=35, ngram_window=128,
        save_results=True,
    )
    t_inf = time.time() - t_inf0
    print(f"[uocr] inferencia en {t_inf:.1f}s", flush=True)

    text = ""
    md_path = os.path.join(out_dir, "result.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            text = f.read()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"load_s": round(t_load, 2), "infer_s": round(t_inf, 2), "text": text},
                  f, ensure_ascii=False)
    print(f"[uocr] OK chars={len(text)} -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
