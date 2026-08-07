"""Mide el overhead de escritura de UOCR_NEG_CACHE_PERSIST=True.

Combina DOS fuentes:

1) EVENTOS REALES de la corrida run_det128_run1 (sesión 128, capítulo 53 págs,
   fusion, workers=2) extraídos de server_output.log y del checkpoint:
   - puts de trigger (una por página con firma, la primera vez que se ve)
   - hits de trigger (refresh del timestamp, sesión 128)
   - registros de negativa §8.4.1 (VLM que NO recuperó nada)
   - hits de negativa §8.4.1 (saltos de refuerzo)

2) MICROBENCHMARK del costo real de _persistir_cache() con el cache poblado
   (json.dumps + write + os.replace atómico), medido en el tamaño real del
   archivo de la corrida.

Salida: escrituras por página VLM / por página normal, escrituras totales del
capítulo con flag True vs False, tiempo total añadido y % sobre el tiempo de
pared del capítulo. Decisión: si el overhead es < 1% del capítulo, el batching
de la persistencia NO merece la pena.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "server_output.log")
CHECKPOINT = os.path.join(ROOT, "run_det128_run1.json")
CACHE_FILE = os.path.join(ROOT, "cache", "ocr_decision_cache.json")


def _count_writes_from_log():
    """Cuenta los eventos de escritura del log real del servidor."""
    with open(LOG, encoding="utf-8", errors="ignore") as f:
        log = f.read()

    # Hits de trigger (refresh del timestamp → 1 escritura cada uno, sesión 128):
    trigger_hits = len(re.findall(r"decisión cacheada por firma", log))
    # Saltos de negativa §8.4.1 (hit de negativa → 1 escritura con flag True):
    neg_hits = len(re.findall(r"§8\.4\.1: firma.*repetitiva", log))
    # Fusiones donde U-OCR SÍ recuperó (cada una = 1 llamada VLM que NO registra
    # negativa): '[process-page] Fusión:' (single) y '[fusion-batch] Página'
    vlm_recovered = len(re.findall(r"\[process-page\] Fusión:", log))
    vlm_recovered += len(re.findall(r"\[fusion-batch\] Página \d+: \d+ híbrido \+ \d+ U-OCR \(batch\) →", log))
    # Registros de negativa (VLM sin recuperación) — se infieren por diferencia:
    # llamadas VLM totales = req_* del daemon creados durante la corrida. Como
    # el log no los lista con timestamp exacto, los inferimos: toda página cuyo
    # tiempo de procesado > ~120s disparó VLM (ver checkpoint) y si no aparece
    # una línea 'Fusión:' para esa página, no recuperó → registró negativa.
    return {
        "trigger_hits": trigger_hits,
        "neg_hits": neg_hits,
        "vlm_recovered": vlm_recovered,
    }


def _count_vlm_calls_daemon():
    """Cuenta las llamadas REALES al daemon U-OCR (dirs req_*) creadas durante
    la ventana de la corrida: desde el mtime del checkpoint hasta +15 min.
    (Verificado en la sesión 128: 11 req_* en la ventana 23:26-23:41.)"""
    daemon_dir = os.path.join(ROOT, "uocr_daemon_out")
    if not os.path.isdir(daemon_dir) or not os.path.exists(CHECKPOINT):
        return 0
    cp_mtime = os.path.getmtime(CHECKPOINT)  # fin de la corrida
    start = cp_mtime - 20 * 60               # ventana generosa hacia atrás
    n = 0
    try:
        for name in os.listdir(daemon_dir):
            if name.startswith("req_"):
                full = os.path.join(daemon_dir, name)
                if os.path.isdir(full) and start <= os.path.getmtime(full) <= cp_mtime:
                    n += 1
    except OSError:
        pass
    return n


def _benchmark_write(cache_dict, neg_dict, n_iter=200):
    """Mide el costo real de una escritura _persistir_cache() (mismo patrón:
    snapshot de dicts, json.dumps, write a .tmp, os.replace). Usa un path
    temporal para no tocar el archivo real del proyecto."""
    import tempfile

    import ocr_engine  # noqa: F401  (importa config/constantes)

    tmpdir = tempfile.mkdtemp(prefix="persist_bench_")
    path = os.path.join(tmpdir, "ocr_decision_cache.json")

    data = {"version": 3, "trigger": cache_dict, "neg": neg_dict}

    # Warm-up
    for _ in range(10):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data))
        os.replace(tmp, path)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data))
        os.replace(tmp, path)
    elapsed = (time.perf_counter() - t0) / n_iter

    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)
    return elapsed, len(json.dumps(data).encode("utf-8"))


def main():
    print("=" * 72)
    print("Overhead de escritura de UOCR_NEG_CACHE_PERSIST=True (sesión 127/129)")
    print("=" * 72)

    # ── 1. Eventos reales del log ──────────────────────────────────────
    ev = _count_writes_from_log()
    print(f"\n[1] Eventos reales de la corrida (53 págs, fusion, workers=2):")
    print(f"    hits de trigger (refresh, 1 escritura c/u):   {ev['trigger_hits']}")
    print(f"    hits de negativa §8.4.1 (1 escritura c/u):     {ev['neg_hits']}")
    print(f"    VLM que SÍ recuperó (0 escrituras de negativa):{ev['vlm_recovered']}")

    # Puts de trigger: una por página con firma NO vacía la primera vez.
    # De las 53 páginas, las que comparten firma (hits) no hacen put.
    # Estimación conservadora: 1 put por página (53) es el tope real porque
    # la primera página de cada firma hace put; el resto hace hit. Con ~10
    # firmas únicas y 10 hits → ~43 puts. Medimos mejor: firmas únicas del
    # archivo persistido de la corrida + hits.
    cache_dict = {}
    neg_dict = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        cache_dict = data.get("trigger", {})
        neg_dict = data.get("neg", {})
    n_unique = len(cache_dict)
    trigger_puts = n_unique  # primera vez de cada firma → put
    print(f"    firmas únicas (puts de trigger, 1 c/u):        {n_unique}")

    # Neg registradas = llamadas VLM reales al daemon - llamadas que recuperaron:
    n_vlm = _count_vlm_calls_daemon()
    neg_regs = max(0, n_vlm - ev["vlm_recovered"])
    print(f"    llamadas VLM reales (dirs req_* del daemon):   {n_vlm}")
    print(f"    registros de negativa (VLM sin recuperar):     {neg_regs}")

    # ── 2. Microbenchmark del costo de escritura ──────────────────────
    w_time, w_bytes = _benchmark_write(cache_dict, neg_dict)
    print(f"\n[2] Microbenchmark de _persistir_cache() (tamaño real {w_bytes} bytes):")
    print(f"    {w_time * 1000:.3f} ms por escritura (json.dumps + write + os.replace)")

    # ── 3. Extrapolación ──────────────────────────────────────────────
    writes_flag_off = trigger_puts + ev["trigger_hits"]
    writes_flag_on = (writes_flag_off + neg_regs + ev["neg_hits"])
    t_off = writes_flag_off * w_time
    t_on = writes_flag_on * w_time
    extra = t_on - t_off

    # Tiempo de pared del capítulo: del checkpoint stats o por diferencia
    # de timestamps. Estimación conocida (benchmarks previos): ~20-45 min.
    wall = 40 * 60  # 40 min representativo (sesión 128: 15 min por ~35 págs)

    print(f"\n[3] Extrapolación al capítulo (53 págs):")
    print(f"    escrituras con flag OFF: {writes_flag_off}  →  {t_off * 1000:.0f} ms")
    print(f"    escrituras con flag ON:  {writes_flag_on}  →  {t_on * 1000:.0f} ms")
    print(f"    overhead añadido:        {writes_flag_on - writes_flag_off} escrituras  →  {extra * 1000:.0f} ms")
    print(f"    % del tiempo de pared ({wall / 60:.0f} min): {extra / wall * 100:.4f}%")
    print(f"    escrituras por página VLM (≈{writes_flag_on / max(1, n_vlm):.1f}): "
          f"la regla '2 por decisión' = put trigger + registro negativa")

    print()
    if extra / wall < 0.01:
        print("DECISIÓN: overhead < 1% del capítulo → batching de la persistencia NO")
        print("merece la pena. Mantener escritura por mutación (simple y robusta).")
    else:
        print("DECISIÓN: overhead ≥ 1% → considerar batching (acumular N mutaciones)")


if __name__ == "__main__":
    main()
