"""
test_uocr_stream_conf.py — Prueba empírica del formato de salida de Unlimited-OCR.

Preguntas a responder:
  Q1. ¿El modelo emite tokens 'confidence' / 'fontSize' en el stream de salida?
      (Los tokens existen en el vocabulario del tokenizer, pero ¿aparecen en la
      salida real de parsing?)
  Q2. ¿Es viable la confianza por logits (output_scores=True + media geométrica)?
      ¿Es estable y diferenciadora entre bloques bien/mal leídos?

Método:
  1. Cargar el modelo 4-bit (mismo setup que uocr_daemon.py).
  2. Pase A: model.infer() con redirect_stdout → stream RAW completo.
     - Imprimir el stream para inspección.
     - Buscar 'confidence' / 'fontSize' / tokens similares en el stream.
  3. Pase B: replicar el preprocesado de infer() y llamar generate() con
     output_scores=True + return_dict_in_generate=True.
     - Por cada token generado: p = softmax(scores[i])[token_id].
     - Mapear tokens → bloques <|det|>...<|/det|> por offsets de caracteres.
     - Confianza de bloque = exp(mean(log p)) (media geométrica).
  4. Reportar: bloques con texto + confianza, y veredicto Q1/Q2.

Uso (venv GPU): env_uocr_gpu\\Scripts\\python.exe test_uocr_stream_conf.py [imagen]
"""

import contextlib
import importlib.util
import io
import math
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(ROOT, "hf_cache", "hub"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT, "hf_cache", "hub"))

MODEL_DIR = os.path.join(ROOT, "models_unlimited_patched")
# Flags van primero; el primer argumento no-flag es la ruta de imagen
_POS_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
IMAGE = _POS_ARGS[0] if _POS_ARGS else "benchmark_page11.png"
MAX_LEN_A = 2048  # pase A (stream completo)
# Pase B: misma max_length que el Pase A para que la generación sea idéntica.
MAX_LEN_B = 2048

_DET_RE = re.compile(
    r"<\|det\|>(\w+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*<\|/det\|>[ \t]*([^\n]*)",
    re.MULTILINE,
)


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float16,
        quantization_config=quant,
        device_map={"": 0},
    ).eval()
    print(f"[test] modelo cargado en {time.time() - t0:.1f}s", flush=True)
    return model, tok


# ─── Pase A: stream RAW vía infer() ──────────────────────────────
def pase_a_stream(model, tok):
    print("=" * 60)
    print("PASE A: stream RAW del modelo (¿confidence/fontSize en la salida?)")
    print("=" * 60)
    out_dir = os.path.join(ROOT, "uocr_daemon_out", f"conf_{int(time.time() * 1000)}")
    os.makedirs(out_dir, exist_ok=True)
    stream = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(stream):
        model.infer(
            tok,
            prompt="<image>document parsing.",
            image_file=IMAGE,
            output_path=out_dir,
            base_size=1024, image_size=640, crop_mode=True,
            max_length=MAX_LEN_A,
            no_repeat_ngram_size=35, ngram_window=128,
            save_results=True,
        )
    raw = stream.getvalue()
    dt = time.time() - t0
    print(f"[test] infer() en {dt:.1f}s | stream {len(raw)} chars", flush=True)
    with open(os.path.join(ROOT, "uocr_stream_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)
    print("── stream RAW ──")
    print(raw[:4000])
    print("── fin stream ──")

    # Buscar tokens de confianza/fontSize en el stream
    hits = {
        "confidence": len(re.findall(r"confidence", raw, re.IGNORECASE)),
        "fontSize": len(re.findall(r"font[Ss]ize", raw)),
        "font_size": len(re.findall(r"font_size", raw, re.IGNORECASE)),
        "score": len(re.findall(r"\bscore\b", raw, re.IGNORECASE)),
        "prob": len(re.findall(r"\bprob\b", raw, re.IGNORECASE)),
        "<|det|>": len(re.findall(r"<\|det\|>", raw)),
    }
    print("── conteo de tokens clave en el stream ──")
    for k, v in hits.items():
        print(f"  {k}: {v}")
    return raw


# ─── Pase B: confianza por logits ────────────────────────────────
def pase_b_logits(model, tok):
    """Confianza por logits usando MONKEYPATCH de model.generate.

    En vez de replicar el preprocesado a mano (que divergió: la copia manual
    generaba solo footer/page_number mientras infer() real produce 9 bloques),
    se parchea model.generate para añadir output_scores=True y
    return_dict_in_generate=True a la llamada REAL de infer(). Así el
    preprocesado es 100% idéntico al Pase A.
    """
    import torch

    print("=" * 60)
    print("PASE B: confianza por logits (monkeypatch de generate())")
    print("=" * 60)

    captured = {}
    orig_generate = model.generate

    def patched_generate(*args, **kwargs):
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        out = orig_generate(*args, **kwargs)
        captured["out"] = out
        return out

    model.generate = patched_generate

    out_dir = os.path.join(ROOT, "uocr_daemon_out", f"conf_{int(time.time() * 1000)}")
    os.makedirs(out_dir, exist_ok=True)
    stream = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(stream):
            model.infer(
                tok,
                prompt="<image>document parsing.",
                image_file=IMAGE,
                output_path=out_dir,
                base_size=1024, image_size=640, crop_mode=True,
                max_length=MAX_LEN_B,
                no_repeat_ngram_size=35, ngram_window=128,
                save_results=True,
            )
    except Exception as e:
        # Esperado: con return_dict_in_generate=True, infer() crashea al indexar
        # output_ids[0, ...] porque ya no es un tensor. La generación YA terminó
        # y quedó capturada en captured["out"]. Tolerar y continuar.
        print(f"[test] infer() lanzó {type(e).__name__} tras la generación "
              f"(esperado con return_dict): {e}", flush=True)
    raw = stream.getvalue()
    dt = time.time() - t0
    print(f"[test] infer() (parcheado) en {dt:.1f}s | stream {len(raw)} chars",
          flush=True)

    out = captured.get("out")
    if out is None:
        raise RuntimeError("generate() parcheado no capturó output")

    scores = out.scores  # tuple[step] de [batch, vocab]
    sequences = out.sequences  # [1, total_len]

    # Calcular los tokens generados (post-input): buscar en la secuencia
    # decodificada el primer marcador <|det|> y retroceder hasta el token
    # que empieza ese marcador. Si no aparece, asumir todos los tokens tras
    # el último token de imagen (128815).
    ids = sequences[0].tolist()
    full_text = tok.decode(ids, skip_special_tokens=False)
    marker = "<|det|>"
    gen_ids = []
    if marker in full_text:
        acc = ""
        gen_start = len(ids)
        for i, tid in enumerate(ids):
            acc += tok.decode([tid], skip_special_tokens=False)
            if marker in acc:
                gen_start = i
                break
        gen_ids = ids[gen_start:]
    else:
        last_img = max(i for i, t in enumerate(ids) if t == 128815)
        gen_ids = ids[last_img + 1:]

    print(f"[test] {len(gen_ids)} tokens generados, {len(scores)} steps", flush=True)
    print(f"[test] stream crudo (primeros 1200 chars):", flush=True)
    print("  " + raw[:1200].replace("\n", "\\n"), flush=True)

    # Per-token log-prob (los scores aplican logits_processor, igual que infer)
    logprobs = []
    for i, tid in enumerate(gen_ids):
        if i >= len(scores):
            break
        sc = scores[i][0]
        lp = torch.log_softmax(sc, dim=-1)[tid].item()
        logprobs.append(lp)
    mean_lp_all = sum(logprobs) / len(logprobs) if logprobs else 0.0
    print(f"  logprob medio global: {mean_lp_all:.4f} "
          f"(perplexidad {math.exp(-mean_lp_all):.3f})", flush=True)

    # Mapear bloques → rango de tokens usando el texto completo decodificado
    full_gen_text = tok.decode(gen_ids, skip_special_tokens=False)
    offsets = []
    acc = ""
    for tid in gen_ids:
        offsets.append(len(acc))
        acc += tok.decode([tid], skip_special_tokens=False)
    offsets.append(len(acc))

    blocks = []
    for m in _DET_RE.finditer(full_gen_text):
        typ = m.group(1)
        x, y, w, h = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        content = m.group(6).strip()
        blocks.append({"type": typ, "x": x, "y": y, "w": w, "h": h,
                       "text": content, "start": m.start(), "end": m.end()})
    blocks.sort(key=lambda b: b["start"])
    for i, b in enumerate(blocks):
        b["content_end"] = blocks[i + 1]["start"] if i + 1 < len(blocks) else len(full_gen_text)

    print("── bloques con confianza por logits (media geométrica) ──", flush=True)
    import bisect
    results = []
    for b in blocks:
        if not b["text"]:
            continue
        c_start = b["end"]
        c_end = b["content_end"]
        i0 = bisect.bisect_right(offsets, c_start)
        i1 = bisect.bisect_left(offsets, c_end)
        i1 = max(i1, i0)
        lps = logprobs[i0:i1]
        if not lps:
            continue
        conf = math.exp(sum(lps) / len(lps))
        results.append({**b, "conf": conf, "n_tokens": len(lps)})
        print(f"  [{b['type']} ({b['x']},{b['y']},{b['w']}x{b['h']})] "
              f"conf={conf:.4f} ({len(lps)} tok) {b['text'][:50]!r}", flush=True)

    confs = [r["conf"] for r in results if r["text"]]
    if confs:
        print(f"\n  distribución: min={min(confs):.4f} max={max(confs):.4f} "
              f"media={sum(confs)/len(confs):.4f}", flush=True)
    return results


def main():
    model, tok = load_model()
    only_b = "--only-b" in sys.argv
    if only_b:
        with open(os.path.join(ROOT, "uocr_stream_raw.txt"), encoding="utf-8") as f:
            raw = f.read()
        print("[test] --only-b: stream RAW leído de uocr_stream_raw.txt "
              f"({len(raw)} chars)")
    else:
        raw = pase_a_stream(model, tok)
    results = pase_b_logits(model, tok)

    print("=" * 60)
    print("VEREDICTO")
    print("=" * 60)
    conf_hits = len(re.findall(r"confidence", raw, re.IGNORECASE))
    fs_hits = len(re.findall(r"font[Ss]ize", raw))
    if conf_hits == 0 and fs_hits == 0:
        print("Q1: El stream NO emite tokens confidence/fontSize → "
              "la confianza debe extraerse por logits (o heurística).")
    else:
        print(f"Q1: El stream SÍ contiene confidence x{conf_hits}, fontSize x{fs_hits}")
    if results:
        confs = [r["conf"] for r in results]
        spread = max(confs) - min(confs)
        print(f"Q2: confianza por logits calculable ({len(results)} bloques, "
              f"rango {min(confs):.4f}-{max(confs):.4f}, spread {spread:.4f})")
        print("    → Diferenciadora si el spread es apreciable (>>0.01).")
    else:
        print("Q2: no se obtuvieron bloques con texto en el pase B")


if __name__ == "__main__":
    main()
