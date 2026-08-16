"""CLI de auditoría de calidad multilingüe.

Uso:
    python analisis_calidad.py
    python analisis_calidad.py --input resultados_progreso_ja_es.json --source ja --target es
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from quality_analysis import analyze_checkpoint, render_report

try:
    sys.stdout.reconfigure(  # type: ignore[union-attr]
        encoding="utf-8", errors="replace")
except AttributeError:
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audita traducciones OCR por par de idiomas")
    parser.add_argument(
        "--input", default="resultados_progreso.json",
        help="Checkpoint JSON a analizar",
    )
    parser.add_argument("--source", default=None, help="Idioma origen (si falta en el checkpoint)")
    parser.add_argument("--target", default=None, help="Idioma destino (si falta en el checkpoint)")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.is_file():
        print(f"[SKIP] Corpus no encontrado: {path}", file=sys.stderr)
        return 2
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] No se pudo leer {path}: {exc}", file=sys.stderr)
        return 2

    # Los checkpoints antiguos del proyecto eran español→inglés, aunque no
    # guardaban idiomas en su raíz. Conservamos ese default solo para ellos;
    # los runs nuevos llevan source_lang/target_lang explícitos.
    source = args.source if args.source is not None else checkpoint.get("source_lang", "auto")
    target = args.target if args.target is not None else checkpoint.get("target_lang", "en")
    report = analyze_checkpoint(checkpoint, source_lang=source, target_lang=target)
    sys.stdout.write(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
