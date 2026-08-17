"""baseline_calidad.py — Baseline de calidad ESTABLE del corpus actual (plan §11 P2).

Problema que resuelve: `analisis_calidad.py` reporta una tasa global que
incluye en el denominador pares que NO son fallos de traducción — SFX
preservados, nombres propios, OCR basura y bloques vacíos. El 32.2 % del
cap. 53 arrastra 96 UNTRANSLATED y 103 REVIEW_LANGUAGE (ruido, fragmentos,
URLs) que hunden la cifra y la hacen NO COMPARABLE entre corridas (el 75.8 %
de Julio era de otro corpus con otro pipeline).

Este script calcula la tasa EFECTIVA sobre los pares TRADUCIBLES (los que el
traductor debía producir en el idioma destino), separa los no-traducibles
(con su desglose) y guarda el baseline en un JSON versionado para que
cualquier corrida futura pueda comparar el delta con `--compare`.

Uso:
    python baseline_calidad.py --input resultados_progreso_calidad_fix.json
    python baseline_calidad.py --input ... --compare      # delta vs baseline
    python baseline_calidad.py --input ... --save path    # JSON de salida
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from quality_analysis import analyze_checkpoint

# Categorías que NO son fallos de traducción (se excluyen del denominador de
# la tasa efectiva). SFX/names se preservan por diseño; OCR_GARBAGE y EMPTY
# son ruido de la fuente, no del traductor.
_NO_TRADUCIBLES = {
    "EMPTY",            # ambos textos vacíos
    "OCR_GARBAGE",      # fuente con señales fuertes de OCR basura
    "SFX_PRESERVED",    # efecto de sonido preservado (comportamiento correcto)
    "SFX_TRANSLATED",   # efecto de sonido traducido (comportamiento correcto)
    "NAME_PRESERVED",   # nombre propio preservado (comportamiento correcto)
}

# Categorías de ÉXITO real de traducción (suman al numerador de la tasa
# efectiva). ALREADY_TARGET = el texto ya estaba en el idioma destino.
_EXITO = {
    "GOOD_TRANSLATION",
    "LITERAL_TRANSLATION",
    "ALREADY_TARGET",
    "OCR_NOISY_RECOVERED",  # OCR ruidoso recuperado (el traductor lo resolvió)
}

# Categorías de FALLO (numeran los problemas reales de traducción).
_FALLO = {
    "UNTRANSLATED",      # texto idéntico sin contexto / destino vacío
    "REVIEW_LANGUAGE",   # destino no coincide con el idioma declarado
    "BAD_TRANSLATION",   # destino con señales de OCR basura
}


def compute_baseline(checkpoint: dict[str, Any], *,
                     source_lang: str, target_lang: str) -> dict[str, Any]:
    """Calcula las métricas de calidad ESTABLES del corpus.

    Reutiliza `classify_pair` de quality_analysis (la misma clasificación que
    analisis_calidad.py) pero separa la tasa global de la EFECTIVA y guarda el
    desglose de no-traducibles para que la cifra sea comparable entre corridas.
    """
    report = analyze_checkpoint(checkpoint, source_lang=source_lang, target_lang=target_lang)

    total = report.total
    categorias = dict(report.categories)
    no_tradu = sum(categorias.get(c, 0) for c in _NO_TRADUCIBLES)
    exitos = sum(categorias.get(c, 0) for c in _EXITO)
    fallos = sum(categorias.get(c, 0) for c in _FALLO)
    traducibles = total - no_tradu
    tasa_efectiva = (exitos / traducibles * 100.0) if traducibles else 0.0

    return {
        "corpus": {
            "total_pares": total,
            "traducibles": traducibles,
            "no_traducibles": no_tradu,
            "fuente": checkpoint.get("source_lang", source_lang),
            "destino": checkpoint.get("target_lang", target_lang),
            "paginas": checkpoint.get("total_pages", len(checkpoint.get("results", []))),
        },
        "tasa_global": round(report.acceptance_rate, 2),
        "tasa_efectiva": round(tasa_efectiva, 2),
        "exitos": exitos,
        "fallos": fallos,
        "categorias": categorias,
        "no_traducibles_detalle": {
            c: categorias.get(c, 0) for c in sorted(_NO_TRADUCIBLES)
        },
        "fallos_detalle": {c: categorias.get(c, 0) for c in sorted(_FALLO)},
    }


def _guardar(baseline: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Baseline guardado: {path}")


def _comparar(actual: dict[str, Any], baseline_path: Path) -> int:
    if not baseline_path.is_file():
        print(f"  [WARN] No hay baseline previo en {baseline_path} — guarda uno "
              f"primero (sin --compare)")
        return 2
    prev = json.loads(baseline_path.read_text(encoding="utf-8"))
    print("  Delta vs baseline:")
    print(f"    Tasa efectiva : {prev.get('tasa_efectiva')}% → "
          f"{actual.get('tasa_efectiva')}%  "
          f"({actual.get('tasa_efectiva', 0) - prev.get('tasa_efectiva', 0):+.2f} pp)")
    print(f"    Tasa global   : {prev.get('tasa_global')}% → "
          f"{actual.get('tasa_global')}%  "
          f"({actual.get('tasa_global', 0) - prev.get('tasa_global', 0):+.2f} pp)")
    for c in sorted(_FALLO):
        p = prev.get("categorias", {}).get(c, 0)
        a = actual.get("categorias", {}).get(c, 0)
        print(f"    {c:20s}: {p} → {a}  ({a - p:+d})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline de calidad estable")
    parser.add_argument("--input", required=True, help="Checkpoint JSON del corpus")
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Comparar contra el baseline guardado")
    parser.add_argument("--save", default="calidad_baseline.json",
                        help="Ruta del JSON de baseline (default calidad_baseline.json)")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.is_file():
        print(f"[ERROR] Corpus no encontrado: {path}", file=sys.stderr)
        return 2
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] No se pudo leer {path}: {exc}", file=sys.stderr)
        return 2

    source = args.source if args.source is not None else checkpoint.get("source_lang", "auto")
    target = args.target if args.target is not None else checkpoint.get("target_lang", "en")

    baseline = compute_baseline(checkpoint, source_lang=source, target_lang=target)
    print("=" * 70)
    print(f"  BASELINE DE CALIDAD ESTABLE — {path.name}")
    print("=" * 70)
    c = baseline["corpus"]
    print(f"  Pares totales      : {c['total_pares']}")
    print(f"  Traducibles        : {c['traducibles']} "
          f"({c['no_traducibles']} no-traducibles excluidos)")
    print(f"  Tasa EFECTIVA      : {baseline['tasa_efectiva']}% "
          f"(exitos {baseline['exitos']} / traducibles {c['traducibles']})")
    print(f"  Tasa global        : {baseline['tasa_global']}% "
          f"(métrica legada de analisis_calidad)")
    print(f"  Fallos reales      : {baseline['fallos']}")
    for cat, n in baseline["fallos_detalle"].items():
        print(f"    {cat:20s}: {n}")
    print(f"  No-traducibles     : {c['no_traducibles']}")
    for cat, n in baseline["no_traducibles_detalle"].items():
        print(f"    {cat:20s}: {n}")
    print("=" * 70)

    if args.compare:
        return _comparar(baseline, Path(args.save))
    _guardar(baseline, Path(args.save))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
