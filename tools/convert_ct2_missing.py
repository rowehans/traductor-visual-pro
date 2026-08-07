"""
Convierte los 3 pares reverso CJK que nunca tuvieron conversion CT2:
  en|ja -> Helsinki-NLP/opus-mt-en-jap
  en|ko -> Helsinki-NLP/opus-mt-tc-big-en-ko
  en|zh -> Helsinki-NLP/opus-mt-en-zh

Usa la misma logica que translator._get_ct2_translator (TransformersConverter
int8 + centinela .ct2_conversion_ok + checksums SHA256), de forma que al
cargarse el primer par la primera vez no haya descarga/conversion en linea.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT  # noqa: E402
from translator import (  # noqa: E402
    _CT2_BASE_DIR,
    _CT2_MODELS,
    _save_ct2_checksums,
)

PAIRS = ["en|ja", "en|ko", "en|zh"]


def convert(pair_key: str) -> bool:
    model_name = _CT2_MODELS.get(pair_key)
    if model_name is None:
        print(f"[{pair_key}] sin repo en _CT2_MODELS")
        return False
    model_dir = os.path.join(_CT2_BASE_DIR, pair_key.replace("|", "-"))
    sentinel = os.path.join(model_dir, ".ct2_conversion_ok")
    if os.path.exists(sentinel):
        print(f"[{pair_key}] ya convertido, salto")
        return True
    try:
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        print(f"[{pair_key}] Convirtiendo {model_name} a CT2 int8 (descarga ~300MB)...")
        # NOTA: NO usar TransformersConverter.convert(): ctranslate2 4.8.1 le pasa
        # el kwarg dtype= a MarianMTModel.from_pretrained, incompatible con
        # transformers 4.48.3 ("unexpected keyword argument 'dtype'").
        # Se replica su logica manualmente: loader -> validate -> optimize -> save.
        from transformers import AutoConfig, AutoTokenizer, MarianMTModel
        from ctranslate2.converters.transformers import _MODEL_LOADERS

        config = AutoConfig.from_pretrained(model_name)
        loader = _MODEL_LOADERS[config.__class__.__name__]
        model_obj = MarianMTModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        spec = loader(model_obj, tokenizer)
        spec.validate()
        spec.optimize(quantization="int8")
        spec.save(model_dir)
        with open(sentinel, "w") as f:
            f.write("ok")
        _save_ct2_checksums(pair_key, model_dir)
        print(f"[{pair_key}] Conversion completada -> {model_dir}")
        return True
    except Exception as e:
        print(f"[{pair_key}] ERROR: {e}")
        return False


def main() -> int:
    print(f"ROOT={ROOT}")
    print(f"CT2_BASE_DIR={_CT2_BASE_DIR}")
    results = {p: convert(p) for p in PAIRS}
    print("\n=== Resumen ===")
    for p, ok in results.items():
        print(f"  {p}: {'OK' if ok else 'FALLO'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
