# -*- coding: utf-8 -*-
"""Analiza server_output.log segmentando por ejecucion (benchmark F6).

El log contiene varias sesiones. Identificamos la ejecucion del capitulo
completo (53 paginas, fusion + batch-window 4) y extraemos por lote/pagina:
regiones YOLO, bloques recuperados y tiempos. Salida ASCII pura.
"""
import re
from collections import OrderedDict

LOG = "server_output.log"

events = []  # (lineno, tipo, detalle)

pbatch_re = re.compile(
    r"\[process-page-batch\] (\d+) páginas, (\d+) bloques, ([\d.]+)s \(OCR:([\d.]+)s\) \| engines: (\[.*\])"
)
req_re = re.compile(r'"(POST|GET) (/api/[^ ]+) HTTP/1.1" (\d+)')
yolo_detect_re = re.compile(r"\[YOLO\] (\d+) regiones de diálogo detectadas")
fase6_re = re.compile(r"Fase 6 \(YOLO\): (\d+) regiones .*?(\d+) bloques recuperados")
fbatch_re = re.compile(r"Página (\d+): Fase 6 (\d+) híbrido \+ (\d+) YOLO → (\d+)")
ppage_re = re.compile(r"\[process-page\] ([\d.]+)s \| (\d+) bloques \| (\[.*\])")

with open(LOG, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    s = line.strip()
    if not s:
        continue
    m = pbatch_re.search(s)
    if m:
        events.append((i, "pbatch", (int(m.group(1)), int(m.group(2)),
                                     float(m.group(3)), float(m.group(4)), m.group(5))))
        continue
    m = req_re.search(s)
    if m:
        events.append((i, "req", (m.group(1), m.group(2), m.group(3))))
        continue
    m = yolo_detect_re.search(s)
    if m:
        events.append((i, "yolo", int(m.group(1))))
        continue
    m = fase6_re.search(s)
    if m:
        events.append((i, "fase6", (int(m.group(1)), int(m.group(2)))))
        continue
    m = fbatch_re.search(s)
    if m:
        events.append((i, "fbatch", tuple(int(x) for x in m.groups())))
        continue
    m = ppage_re.search(s)
    if m:
        events.append((i, "ppage", (float(m.group(1)), int(m.group(2)), m.group(3))))

print("=== Secuencia de eventos (lineno | tipo | detalle) ===")
for i, t, d in events:
    print(f"{i:5d} | {t:7s} | {d}")
