"""Harness A/B anti-deriva compartido por los benchmarks de la Ruta C.

Patrón (estandarizado 2026-08-15 desde benchmark_rutac_params.py):
1. **Intercalado por página** — base y alt se corren contiguos en la misma
   página (ventana de deriva de GPU mínima), en vez de dos pasadas separadas
   sobre todas las páginas (el sesgo de orden que inflaba los A/B previos).
2. **Orden alternado** — la página par corre base→alt y la impar alt→base: el
   sesgo residual "la segunda corrida del par es más rápida" se cancela.
3. **Páginas de control automáticas** (sin etapa Ruta C: el parámetro no
   aplica) → noise-floor (máx |Δ| de control) contra el que se juzga el
   efecto; veredicto explícito (atribuible / cautela / NO CONCLUYENTE).
4. **--reps N con mediana** — robusto al ruido térmico puntual.

Con el daemon VLM detenido el noise-floor de esta máquina baja de ±0.3-0.7 s
a ~0.02 s (25×), lo que convierte los veredictos en fiables.

Los benchmarks que lo usan deben:
- Aplicar el valor del parámetro vía ``apply_patch("base"/"alt")`` parcheando
  UNA sola vez el original (capturado tras los wraps de timing, para no
  anidar wrappers — el bug que hacía que las pasadas "alternativas" de los
  benchmarks de upscale anteriores corrieran el valor base).
- ``run_one(pno)`` debe devolver un dict con ``total_s`` y ``stages`` (mapa
  con la clave ``ruta_c`` > 0 para marcar páginas afectadas).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class Timing:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def reset(self) -> None:
        self.timings.clear()
        self.counts.clear()


def pick_median(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Corrida con tiempo mediano (robusto al ruido térmico)."""
    ordered = sorted(runs, key=lambda r: r["total_s"])
    return ordered[len(ordered) // 2]


def run_page_pair(apply_patch: Callable[[str], None],
                  run_one: Callable[[], dict[str, Any]],
                  order: str, reps: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Base y alt de UNA página, intercalados (order "ba" o "ab"), reps c/u.

    ``apply_patch("base"/"alt")`` aplica el valor del parámetro; ``run_one()``
    corre la página actual (closure) y devuelve el resultado. Al final se
    restaura el valor base.
    """
    runs: dict[str, list[dict[str, Any]]] = {"b": [], "a": []}
    for _ in range(reps):
        for w in order:
            apply_patch("base" if w == "b" else "alt")
            runs[w].append(run_one())
    apply_patch("base")
    return pick_median(runs["b"]), pick_median(runs["a"])


def run_ab(pages: list[int], apply_patch: Callable[[str], None],
           run_one: Callable[[int], dict[str, Any]],
           reps: int = 1) -> dict[str, Any]:
    """A/B por página con orden alternado (par b→a, impar a→b)."""
    out: dict[str, Any] = {}
    for i, pno in enumerate(pages):
        order = "ba" if i % 2 == 0 else "ab"
        b, a = run_page_pair(apply_patch, lambda: run_one(pno), order, reps)
        out[str(pno)] = {"base": b, "alt": a}
    return out


def control_stats(pages: list[int], ab: dict[str, Any],
                  stage_key: str = "ruta_c") -> tuple[list[int], float, float]:
    """Estadísticas de deriva en páginas SIN etapa Ruta C (el parámetro no
    aplica). Returns: (controles, Δ medio, máx |Δ|) — el máx |Δ| es el
    noise-floor del estado de GPU.
    """
    controls = [p for p in pages
                if ab[str(p)]["base"]["stages"].get(stage_key, 0) == 0]
    if not controls:
        return [], 0.0, 0.0
    deltas = [ab[str(p)]["alt"]["total_s"] - ab[str(p)]["base"]["total_s"]
              for p in controls]
    return controls, float(np.mean(deltas)), float(max(abs(d) for d in deltas))


def effect_stats(pages: list[int], ab: dict[str, Any],
                 stage_key: str = "ruta_c") -> tuple[list[int], float]:
    """Páginas donde el parámetro SÍ aplica y su Δ medio (alt - base)."""
    affected = [p for p in pages
                if ab[str(p)]["base"]["stages"].get(stage_key, 0) > 0]
    effect = [ab[str(p)]["alt"]["total_s"] - ab[str(p)]["base"]["total_s"]
              for p in affected]
    return affected, (float(np.mean(effect)) if effect else 0.0)


def verdict(controls: list[int], drift_mean: float, drift_max: float,
            effect_avg: float, avg_base: float) -> str:
    """Veredicto separando deriva de efecto (patrón benchmark_rutac_params)."""
    thr = max(0.10, 0.04 * avg_base)
    if not controls:
        return ("sin páginas de control — Δ orientativo (añade páginas sin "
                "Ruta C para separar deriva de efecto)")
    if abs(drift_mean) <= thr and drift_max <= thr:
        return (f"estable (control Δ medio {drift_mean:+.3f}s, máx "
                f"{drift_max:.3f}s) — Δ atribuible al parámetro")
    if abs(effect_avg) > drift_max:
        return (f"⚠ deriva de control alta (máx {drift_max:.3f}s) pero "
                f"|Δ afectadas| {abs(effect_avg):.3f}s la supera — concluir "
                "con cautela")
    return (f"✗ NO CONCLUYENTE — el noise-floor de control "
            f"({drift_max:.3f}s) es del orden o mayor que el Δ de las "
            f"afectadas ({effect_avg:+.3f}s); re-corre con --reps N")


def summary_block(controls: list[int], drift_mean: float, drift_max: float,
                  affected: list[int], effect_avg: float,
                  verdict_str: str) -> list[str]:
    """Líneas del resumen final (control + efecto + veredicto)."""
    return [
        f"Páginas de control (sin Ruta C): {controls} — Δ medio "
        f"{drift_mean:+.3f}s, noise-floor (máx |Δ|) {drift_max:.3f}s",
        f"Δ afectadas (con Ruta C): {effect_avg:+.3f}s/pág  "
        f"({len(affected)}/{len(controls) + len(affected)} páginas)",
        f"Veredicto: {verdict_str}",
    ]
