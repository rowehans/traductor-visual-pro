# -*- coding: utf-8 -*-
"""Mide el coste/beneficio de la salvaguarda débil del TRIGGER (sesión 136/138)
en una corrida del capítulo: cuántas decisiones negativas débiles había,
cuántos recomputes consumió el contador=1, cuántos flipearon a positivo (y
por tanto dispararon el VLM), y lo correlaciona con el log del servidor
(línea "[trigger] sesión 136: salvaguarda débil — recompute 1/1").

Uso:  python tools/medir_salvaguarda_trigger.py [--cache cache/ocr_decision_cache.json] [--log run_139_server.log]

El cache v5 guarda por firma: trigger = [ts, n_blocks, avg_conf, decision, re_computes].
Una "negativa débil" = decision False y (n_blocks < UOCR_NEG_WEAK_MAX_BLOCKS o
conf < UOCR_NEG_WEAK_MIN_CONF). re_computes >= 1 = la salvaguarda consumió su
recompute (una vez). Un "flip a positivo" = decision True con re_computes >= 1
(el recompute recomputó y la página disparó el VLM).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    UOCR_NEG_WEAK_MAX_BLOCKS,
    UOCR_NEG_WEAK_MIN_CONF,
    UOCR_NEG_MAX_REINTENTOS,
)

WEAK_MAX_B = UOCR_NEG_WEAK_MAX_BLOCKS
WEAK_MIN_C = UOCR_NEG_WEAK_MIN_CONF
MAX_RE = UOCR_NEG_MAX_REINTENTOS


def es_debil(n_blocks: int, avg_conf: float) -> bool:
    return n_blocks < WEAK_MAX_B or avg_conf < WEAK_MIN_C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/ocr_decision_cache.json")
    ap.add_argument("--log", default="run_139_server.log")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint de process_all_pages para la tasa de traducción")
    args = ap.parse_args()

    datos = json.load(open(args.cache, encoding="utf-8"))
    print(f"Versión del cache: {datos.get('version')}")
    trig = datos.get("trigger", {})
    print(f"Entradas de trigger (firmas únicas): {len(trig)}")

    neg_debiles = 0
    consumidos = 0
    flips_positivo = 0
    positivos = 0
    neg_fuertes = 0
    re_por_firma = {}
    for k, v in trig.items():
        # v = [ts, n, conf, decision, re_computes]
        n, conf, decision, re = v[1], v[2], v[3], v[4]
        if decision:
            positivos += 1
        elif es_debil(n, conf):
            neg_debiles += 1
        else:
            neg_fuertes += 1
        if re >= 1:
            consumidos += 1
            re_por_firma[k] = (n, conf, decision, re)
        if decision and re >= 1:
            flips_positivo += 1

    print(f"\n── Decisiones de trigger (por firma, estado FINAL) ──")
    print(f"  Positivas (VLM):        {positivos}")
    print(f"  Negativas DÉBILES:      {neg_debiles}   (<{WEAK_MAX_B} bloq o conf <{WEAK_MIN_C})")
    print(f"  Negativas fuertes:      {neg_fuertes}")
    print(f"\n── Salvaguarda débil (sesión 136) ──")
    print(f"  Recomputes consumidos (contador={MAX_RE}): {consumidos}")
    print(f"  → flips a positivo (dispararon VLM):        {flips_positivo}")
    for k, (n, conf, decision, re) in sorted(re_por_firma.items()):
        print(f"    {k[:24]}… n={n} conf={conf:.2f} decision={'VLM' if decision else 'no VLM'} re={re}")

    # Log del servidor: fuente de verdad de los recomputes EN VIVO (sesión 138)
    log_lines = 0
    if args.log and os.path.isfile(args.log):
        with open(args.log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "salvaguarda débil — recompute" in line:
                    log_lines += 1
        print(f"\n  Líneas '[trigger] sesión 136: salvaguarda débil — recompute' en el log: {log_lines}")
    else:
        print("\n  (log no encontrado)")

    # Tasa de traducción desde el checkpoint (campo stats, autoritativo)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        cp = json.load(open(args.checkpoint, encoding="utf-8"))
        st = cp.get("stats", {})
        tot = st.get("total_blocks_found", 0)
        trad = st.get("total_blocks_translated", 0)
        if tot:
            print(f"\n  Tasa de traducción (checkpoint stats): {trad}/{tot} = "
                  f"{100.0 * trad / tot:.1f}%")
        print(f"  stats: {st}")


if __name__ == "__main__":
    main()
