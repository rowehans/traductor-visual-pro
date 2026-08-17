"""
benchmark_langdetect_cache.py — Mide la palanca del cache de langdetect.

Contexto (plan §4.6, "hallazgo langdetect"): _detect_language_robust cuesta
2.95 s/capítulo (523 llamadas × 5.65 ms, 70 % de la maquinaria spellcheck).
Ya tiene lru_cache(maxsize=4096), pero cachea por TEXTO EXACTO — cada bloque
OCR distinto es un miss. Este benchmark mide en el pipeline REAL (capítulo
completo, daemon detenido):

  1. Cuántas llamadas reales llegan al cuerpo de _detect_language_robust
     (misses del lru_cache) y cuánto tiempo consumen.
  2. Cuántos textos únicos colapsarían a la misma FIRMA DE CARACTERES
     (perfil de n-gramas: las primeras k letras más frecuentes + categorías
     de scripts) — el ahorro potencial de cachear por firma en vez de por
     texto exacto.
  3. Verificación de equivalencia: para los textos que compartirían firma,
     ¿el resultado de langdetect es el mismo?

Uso (daemon VLM DETENIDO):
  PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_langdetect_cache.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import fitz

import benchmark_production as bp
import ocr_engine
import translator


def _firma_caracteres(text: str, top_k: int = 15) -> tuple[Any, ...]:
    """Firma de perfil de caracteres: las top_k letras más frecuentes con su
    frecuencia relativa + flags de script. Mismo espíritu que el perfil de
    n-gramas de langdetect: dos textos con la misma distribución colapsan."""
    t = text.lower()
    total = max(1, len(t))
    top = tuple(sorted(Counter(t).items(), key=lambda x: (-x[1], x[0]))[:top_k])
    flags = (
        any(0x3040 <= ord(c) <= 0x30FF for c in t),   # kana
        any(0x4E00 <= ord(c) <= 0x9FAF for c in t),   # hanzi
        any(0xAC00 <= ord(c) <= 0xD7A3 for c in t),   # hangul
        any(c in "áéíóúñüÁÉÍÓÚÑÜ¿¡" for c in text),   # acentos es
        len(t),
    )
    return top, flags


def main() -> None:
    misses: dict[str, int] = {}
    miss_s: dict[str, float] = {}
    total_s = 0.0

    orig_fn = translator._detect_language_robust.__wrapped__ \
        if hasattr(translator._detect_language_robust, "__wrapped__") \
        else translator._detect_language_robust
    # Limpiar el lru_cache: queremos medir los misses REALES del capítulo.
    translator._detect_language_robust.cache_clear()  # type: ignore[attr-defined]

    def spy(text: str) -> str:
        nonlocal total_s
        t0 = time.perf_counter()
        out = orig_fn(text)
        dt = time.perf_counter() - t0
        total_s += dt
        misses[text] = misses.get(text, 0) + 1
        miss_s[text] = miss_s.get(text, 0.0) + dt
        return out

    # El spy reemplaza el lru_cache decorado: mismo comportamiento (cachea por
    # texto exacto) pero contando los misses que entrarían al cuerpo real.
    translator._detect_language_robust = spy

    ocr = ocr_engine.OCRManager()
    doc = fitz.open(bp.PDF)
    t_start = time.time()
    for pno in range(1, 54):
        img = bp.render_page(doc, pno, bp.DEFAULT_SCALE)
        ocr.run_ocr(img, bp.OCR_LANG, "fusion", prefilter=True,
                    disable_uocr=True)  # daemon fuera: solo pipeline no-VLM
    wall = time.time() - t_start

    n_calls = sum(misses.values())
    n_unique = len(misses)
    print(f"\n[langdetect] llamadas al cuerpo (misses lru): {n_calls} en {n_unique} textos "
          f"únicos | tiempo real {total_s:.2f}s | wall capítulo {wall:.1f}s "
          f"({100 * total_s / wall:.1f}%)")

    # ── Ahorro potencial por firma de caracteres ────────────────
    firmas: dict[Any, list[str]] = {}
    for t in misses:
        firmas.setdefault(_firma_caracteres(t), []).append(t)
    n_firmas = len(firmas)
    ahorro = 1.0 - n_firmas / max(1, n_unique)

    # Equivalencia: mismo resultado langdetect dentro de cada firma
    resultados: dict[str, str] = {}
    for t in misses:
        resultados[t] = orig_fn(t)
    inconsistentes = 0
    for sig, texts in firmas.items():
        if len(texts) > 1:
            rs = {resultados[t] for t in texts}
            if len(rs) > 1:
                inconsistentes += 1
    print(f"[langdetect] firmas únicas: {n_firmas} de {n_unique} textos "
          f"(ahorro potencial {100 * ahorro:.0f}% de misses)")
    print(f"[langdetect] firmas con resultado INCONSISTENTE dentro: {inconsistentes}")

    # Ejemplos de colapso
    ejemplos = 0
    for sig, texts in firmas.items():
        if len(texts) >= 3 and ejemplos < 3:
            print(f"  ejemplo firma: {[t[:40] for t in texts[:3]]} -> "
                  f"{[resultados[t] for t in texts[:3]]}")
            ejemplos += 1

    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    summary = {
        "n_calls_body": n_calls, "n_unique_texts": n_unique,
        "langdetect_total_s": round(total_s, 3),
        "chapter_wall_s": round(wall, 2),
        "pct_of_chapter": round(100 * total_s / wall, 1),
        "n_firmas": n_firmas, "ahorro_potencial_pct": round(100 * ahorro, 1),
        "firmas_inconsistentes": inconsistentes,
        "top_misses_s": sorted(miss_s.items(), key=lambda x: -x[1])[:10],
    }
    (out / "langdetect_cache.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: benchmark_results/langdetect_cache.json")


if __name__ == "__main__":
    main()
