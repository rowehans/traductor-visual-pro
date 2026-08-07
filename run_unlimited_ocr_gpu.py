"""
run_unlimited_ocr_gpu.py — Ejecuta baidu/Unlimited-OCR con cuantización
bitsandbytes (4-bit/8-bit) en GPU y escribe un JSON con {load_s, infer_s, text, vram}.

Uso (desde env_uocr_gpu):
    python run_unlimited_ocr_gpu.py <imagen> <json_salida> [bits] [max_length]
        bits: 4 (default) | 8
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models_unlimited_patched",  # copia parcheada bf16->fp16 (Pascal no soporta bf16 nativo)
)


def main():
    image_file = sys.argv[1]
    out_json = sys.argv[2]
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    max_length = int(sys.argv[4]) if len(sys.argv) > 4 else 32768

    if not torch.cuda.is_available():
        print("ERROR: CUDA no disponible en este entorno")
        sys.exit(1)

    # Pascal (sm_61) NO soporta bf16 nativo -> compute dtype fp16
    if bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = torch.float16
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.float16

    t_load0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map={"": 0},  # forzar todo a la única GPU (4GB VRAM)
    )
    model = model.eval()
    t_load = time.time() - t_load0
    torch.cuda.reset_peak_memory_stats()
    print(f"[uocr-gpu] modelo {bits}-bit cargado en {t_load:.1f}s", flush=True)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(out_json)),
        f"uocr_out_{bits}bit_{os.path.splitext(os.path.basename(image_file))[0]}",
    )
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
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[uocr-gpu] inferencia {t_inf:.1f}s | pico VRAM {peak_vram:.2f} GB", flush=True)

    text = ""
    md_path = os.path.join(out_dir, "result.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            text = f.read()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "load_s": round(t_load, 2),
            "infer_s": round(t_inf, 2),
            "peak_vram_gb": round(peak_vram, 2),
            "bits": bits,
            "text": text,
        }, f, ensure_ascii=False)
    print(f"[uocr-gpu] OK chars={len(text)} -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
