"""
Descarga los tokenizers OPUS-MT que faltan en la caché de HuggingFace.

El servidor principal (translator.py) NO define HF_HOME, así que
AutoTokenizer.from_pretrained(..., local_files_only=True) lee la caché
por defecto: ~/.cache/huggingface/hub (HF_HUB_CACHE por defecto).

Aquí descargamos SOLO los archivos del tokenizer (pequeños, ~5-20MB) de
los 16 repos Helsinki-NLP OPUS-MT usados por _CT2_MODELS. Los pesos grandes
(pytorch_model.bin) NO se descargan: las conversiones CT2 ya existen en
models/ct2/.
"""
import os
import sys

from huggingface_hub import snapshot_download

# Los mismos repos de translator._CT2_MODELS, con los nombres corregidos:
#  - "en|ja"  -> Helsinki-NLP/opus-mt-en-jap  (opus-mt-en-ja NO existe)
#  - "en|ko"  -> Helsinki-NLP/opus-mt-tc-big-en-ko (opus-mt-en-ko NO existe)
MODELS = {
    "es|en": "Helsinki-NLP/opus-mt-es-en",
    "en|es": "Helsinki-NLP/opus-mt-en-es",
    "en|fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr|en": "Helsinki-NLP/opus-mt-fr-en",
    "en|de": "Helsinki-NLP/opus-mt-en-de",
    "de|en": "Helsinki-NLP/opus-mt-de-en",
    "en|pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "pt|en": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "en|it": "Helsinki-NLP/opus-mt-en-it",
    "it|en": "Helsinki-NLP/opus-mt-it-en",
    "ja|en": "Helsinki-NLP/opus-mt-ja-en",
    "en|ja": "Helsinki-NLP/opus-mt-en-jap",
    "ko|en": "Helsinki-NLP/opus-mt-ko-en",
    "en|ko": "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "zh|en": "Helsinki-NLP/opus-mt-zh-en",
    "en|zh": "Helsinki-NLP/opus-mt-en-zh",
}

# Solo archivos del tokenizer (config, vocabulario, sentencepiece).
# Excluye pytorch_model.bin / tf_model.h5 (cientos de MB) — no se necesitan.
ALLOW_PATTERNS = [
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "*.spm",
    "special_tokens_map.json",
    "generation_config.json",
    "metadata.json",
]


def main() -> int:
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Descargando tokenizers en caché HF: {cache_dir}")
    print(f"(sin HF_HOME definido en el server → esta es la caché que lee)\n")

    ok, fail = 0, 0
    for pair, repo in MODELS.items():
        try:
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=["*.bin", "*.h5", "*.safetensors", "*.onnx", "*.pt"],
                cache_dir=cache_dir,
            )
            # Listar qué se descargó
            files = []
            for root, _dirs, fs in os.walk(path):
                for f in fs:
                    if not f.endswith((".incomplete", ".lock")):
                        files.append(f)
            print(f"[OK] {pair:>6} <- {repo}: {len(files)} archivos ({', '.join(sorted(files))})")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {pair:>6} <- {repo}: {e}")
            fail += 1

    print(f"\n=== Resumen: {ok} OK, {fail} fallos ===")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
