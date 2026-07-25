#!/usr/bin/env python3
"""
extraer_sin_traducir.py — Analiza logs del servidor y extrae todas las líneas
relacionadas con páginas sin traducir o traducciones fallidas, agrupadas por
patrón de error.

Uso:
    python extraer_sin_traducir.py [archivos.log ...]
    python extraer_sin_traducir.py               # Analiza todos los *.log en cwd
    python extraer_sin_traducir.py server_run.log
    python extraer_sin_traducir.py server_run.log ci_server.log
"""

import os
import re
import sys
from collections import defaultdict
from typing import DefaultDict, Optional


# ─── Categorías de error ─────────────────────────────────────────
# Cada categoría tiene: nombre, regex, y extracción de contexto
Category = tuple[str, re.Pattern[str], Optional[re.Pattern[str]]]

CATEGORIES: list[Category] = [
    # ── OCR / Detección de texto ─────────────────────────────────
    (
        "OCR_VACIO",
        re.compile(r'\[process-page\] OCR \([^)]+\): 0 bloques'),
        None,
    ),
    (
        "OCR_FILTRO_PRE",
        re.compile(r'\[OCR\] Filtrando .+? pre-merge:'),
        None,
    ),
    (
        "OCR_FILTRO_POST",
        re.compile(r'\[OCR\] Filtrando .+? post-merge:'),
        None,
    ),
    (
        "OCR_NOISE_DETECTADO",
        re.compile(r'\[translate\] OCR noise detectado'),
        re.compile(r"\[translate\].+?text='([^']+)'"),
    ),
    # ── Errores de motor de traducción ──────────────────────────
    (
        "SAME_TEXT_DESCARTADO",
        re.compile(r'\[translate\] .+? mismo texto .+?— descartado'),
        None,
    ),
    (
        "TRADUCCION_INVALIDA",
        re.compile(r'\[translate\] .+? inválido:'),
        None,
    ),
    (
        "FALLBACK_USADO",
        re.compile(r'\[translate\] Usando fallback'),
        None,
    ),
    (
        "TODOS_FALLARON",
        re.compile(r'\[translate\] Todos los métodos fallaron'),
        None,
    ),
    (
        "ENGINE_ERROR",
        re.compile(r'\[(?:google|argos|CT2|offline)\] Error'),
        None,
    ),
    (
        "BLOQUE_ERROR",
        re.compile(r'Error traduciendo bloque idx'),
        None,
    ),
    (
        "CACHE_HIT",
        re.compile(r'\[translate\] Cache HIT:'),
        None,
    ),
    # ── Rate limiting ────────────────────────────────────────────
    (
        "RATE_LIMIT_BACKOFF",
        re.compile(r'\[google\] Rate limit backoff'),
        None,
    ),
    (
        "RATE_LIMIT_DETECTADO",
        re.compile(r'\[google\] ¡Rate limit detectado'),
        None,
    ),
    (
        "GOOGLE_UNCHANGED_COUNT",
        re.compile(r'\[google\] Texto sin cambios \(\d+/\d+\)'),
        None,
    ),
    # ── Langdetect ──────────────────────────────────────────────
    (
        "LANGDETECT_SOBRESCRITO",
        re.compile(r'\[langdetect\]'),
        None,
    ),
    # ── Timeout / Conexión ──────────────────────────────────────
    (
        "TIMEOUT_PAGINA",
        re.compile(r'TIMEOUT'),
        None,
    ),
    (
        "TIMEOUT_DEFINITIVO",
        re.compile(r'TIMEOUT definitivo'),
        None,
    ),
    (
        "ERROR_CONEXION",
        re.compile(r'ERROR conexión'),
        None,
    ),
    (
        "RENDER_ERROR",
        re.compile(r'ERROR render'),
        None,
    ),
    # ── Páginas del reporte final ───────────────────────────────
    (
        "SIN_TRAD",
        re.compile(r'(?:SIN_TRAD|PARCIAL)(?:_R\d)?\b'),
        None,
    ),
]


def gather_log_files(args: list[str]) -> list[str]:
    """Recolecta archivos .log. Si no se pasan argumentos, busca *.log en cwd."""
    if args:
        return [a for a in args if os.path.isfile(a)]
    import glob
    return sorted(glob.glob("*.log"))


def classify_line(line: str, lineno: int, filename: str
                  ) -> Optional[tuple[str, str, str]]:
    """
    Clasifica una línea del log.
    Retorna (categoria, line_text, context) o None si no coincide.
    """
    for cat_name, pattern, context_pat in CATEGORIES:
        if pattern.search(line):
            line_stripped = line.rstrip("\n\r")
            ctx = ""
            if context_pat:
                m = context_pat.search(line)
                if m:
                    ctx = m.group(1)
            return (cat_name, line_stripped, ctx)
    return None


def extract_text_before_translate(lines: list[str], idx: int, max_lookback: int = 30
                                  ) -> str:
    """
    Busca hacia atrás el texto que se estaba traduciendo.
    Busca 'text=...' en líneas previas.
    """
    for i in range(max(0, idx - max_lookback), idx):
        m = re.search(r"text='([^']+)'", lines[i])
        if m:
            return m.group(1)
    return ""


def build_report(log_files: list[str]) -> str:
    """Procesa archivos de log y genera un reporte estructurado."""
    if not log_files:
        return "No se encontraron archivos .log."

    # { categoria: [(filename, line_text, text_context, lineno)] }
    results: DefaultDict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    total_lines = 0
    matched_lines = 0

    for fpath in log_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            results["ERROR LECTURA"].append((fpath, str(e), "", 0))
            continue

        fname = os.path.basename(fpath)
        for lineno, line in enumerate(lines, 1):
            total_lines += 1
            classified = classify_line(line, lineno, fname)
            if classified:
                cat_name, line_text, ctx = classified
                # Mejorar contexto: extraer texto traducido si no vino con la línea
                text_ctx = ctx or extract_text_before_translate(lines, lineno - 1)
                results[cat_name].append((fname, line_text, text_ctx, lineno))
                matched_lines += 1

    # ── Construir reporte ────────────────────────────────────────
    lines_out: list[str] = []
    lines_out.append("=" * 80)
    lines_out.append("EXTRACTOR DE ERRORES DE TRADUCCIÓN — REPORTE")
    lines_out.append("=" * 80)
    lines_out.append(f"Archivos analizados: {len(log_files)}")
    lines_out.append(f"Líneas totales:      {total_lines}")
    lines_out.append(f"Líneas con error:    {matched_lines}")
    lines_out.append(f"Tasa de error:       {matched_lines/max(total_lines,1)*100:.1f}%")
    lines_out.append("")

    # Ordenar categorías por frecuencia (descendente)
    sorted_cats = sorted(results.items(), key=lambda x: -len(x[1]))

    for cat_name, entries in sorted_cats:
        unique_files = sorted(set(e[0] for e in entries))
        lines_out.append(f"{'─' * 60}")
        lines_out.append(f"  📁 {cat_name}  ({len(entries)} ocurrencias)")
        lines_out.append(f"     Archivos: {', '.join(unique_files)}")
        lines_out.append("")

        # Agrupar textos de contexto similares
        text_groups: DefaultDict[str, int] = defaultdict(int)
        for _, line_text, ctx, lineno in entries:
            if ctx:
                text_groups[ctx] += 1

        # Mostrar líneas de ejemplo (hasta 5)
        n_shown = 0
        for _, line_text, ctx, lineno in entries:
            if n_shown >= 5:
                remaining = len(entries) - n_shown
                lines_out.append(f"     ... y {remaining} más")
                break
            lines_out.append(f"     Ln {lineno:>6d} | {line_text[:100]}")
            if ctx:
                lines_out.append(f"              └─ texto: '{ctx[:80]}'")
            n_shown += 1

        # Mostrar resumen de textos distintos
        if text_groups:
            lines_out.append("")
            lines_out.append(f"     Textos distintos ({len(text_groups)}):")
            for text, count in sorted(text_groups.items(), key=lambda x: -x[1])[:8]:
                lines_out.append(f"       · '{text[:60]}' ({count}x)")
        lines_out.append("")

    # ── Resumen de impacto ──────────────────────────────────────
    lines_out.append("=" * 80)
    lines_out.append("RESUMEN DE IMPACTO")
    lines_out.append("=" * 80)

    # Calcular páginas potencialmente perdidas
    ocr_vacio = len(results.get("OCR_VACIO", []))
    same_text = len(results.get("SAME_TEXT_DESCARTADO", []))
    invalidas = len(results.get("TRADUCCION_INVALIDA", []))
    fallbacks = len(results.get("FALLBACK_USADO", []))
    todos_fallaron = len(results.get("TODOS_FALLARON", []))
    engine_errors = len(results.get("ENGINE_ERROR", []))
    timeouts = len(results.get("TIMEOUT_PAGINA", []))

    lines_out.append(f"  Páginas con OCR vacío:           {ocr_vacio}")
    lines_out.append(f"  Textos same-text descartados:    {same_text}")
    lines_out.append(f"  Traducciones inválidas:          {invalidas}")
    lines_out.append(f"  Fallback usado (todo rechazado): {fallbacks}")
    lines_out.append(f"  Todos los motores fallaron:      {todos_fallaron}")
    lines_out.append(f"  Errores de motor:                {engine_errors}")
    lines_out.append(f"  Timeouts de página:              {timeouts}")
    lines_out.append("")

    # Estimación de bloques sin traducir
    # same_text e invalidas son conteos engine-level (varios motores
    # por bloque). Todos_fallaron ya es un evento block-level.
    bloques_perdidos = same_text + invalidas + todos_fallaron
    lines_out.append(f"  Estimación de bloques sin traducir: ~{bloques_perdidos}")
    lines_out.append("")
    lines_out.append(f"  Reporte generado: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines_out.append("=" * 80)

    return "\n".join(lines_out)


def main() -> None:
    log_files = gather_log_files(sys.argv[1:])
    report = build_report(log_files)

    # ── Imprimir con encoding seguro ────────────────────────────
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)

    # ── Guardar reporte a archivo ───────────────────────────────
    report_path = "reporte_sin_traducir.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
