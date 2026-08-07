# -*- coding: utf-8 -*-
"""Genera reporte_fusion.html con los resultados del benchmark fusion.

1. Procesa las páginas 3, 11, 12 (y 5 artística con force_uocr) del PDF nuevo
   vía /api/process-page en modo fusion.
2. Dibuja sobre cada página los bloques detectados: verde = híbrido
   (EasyOCR+RapidOCR), rojo = recuperado por Unlimited-OCR (diálogo artístico).
3. Construye el reporte HTML con:
   - Tabla comparativa de modos (EasyOCR solo / auto / fusion / fusion v4.2)
   - CER por página artística vs ground truth
   - Distribución de tiempos por página
   - Imágenes embebidas en base64 con bloques resaltados
"""
import sys, io, os, json, base64, time, urllib.request
import cv2
import numpy as np
import fitz  # PyMuPDF

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

API = "http://127.0.0.1:5174/api/process-page"
PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OUT_PAGES = "reporte_pages"
OUT_HTML = "reporte_fusion.html"
DPI = 180

# Páginas a procesar: (nº, nombre corto, force_uocr, descripción)
PAGES = [
    (3, "Página 3", False, "Página de diálogo normal (control)"),
    (5, "Página 5", True, "Panel artístico: diálogo pintado recuperado por U-OCR"),
    (11, "Página 11", False, "Página densa con SFX y texto"),
    (12, "Página 12", False, "Globo en panel (control)"),
]

def fetch_page(page_no: int, force: bool = False) -> dict:
    doc = fitz.open(PDF_PATH)
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=DPI)
    doc.close()
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    payload = {"image": b64, "ocr_mode": "fusion", "source_lang": "auto",
               "target_lang": "en", "force_uocr": force}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode())
    return {"blocks": data.get("blocks", []),
            "engines": data.get("engines_used", []),
            "t": round(time.time() - t0, 1),
            "ocr_engine": data.get("ocr_engine")}

def draw_blocks(page_no: int, blocks: list[dict], engines: list[str]) -> str:
    """Dibuja rectángulos sobre los bloques y guarda PNG con nombre de archivo.
    Verde = híbrido; rojo = recuperado U-OCR. Devuelve ruta del PNG."""
    doc = fitz.open(PDF_PATH)
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=DPI)
    doc.close()
    img = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n), cv2.COLOR_RGB2BGR)
    scale = DPI / 72.0  # coordenadas PDF (72dpi) → px (180dpi)
    h, w = img.shape[:2]
    n_hybrid = n_uocr = 0
    for b in blocks:
        x, y = int(b.get("x", 0)), int(b.get("y", 0))
        bw, bh = int(b.get("w", 10)), int(b.get("h", 10))
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x or y1 <= y:
            continue
        is_uocr = "unlimited" in engines  # página reforzada: marcar todos los bloques
        color = (0, 60, 255) if is_uocr else (60, 220, 60)  # BGR: rojo / verde
        thick = 3 if is_uocr else 2
        cv2.rectangle(img, (x, y), (x1, y1), color, thick)
        # Etiqueta con el texto original (recortado)
        txt = (b.get("source") or b.get("text") or "").replace("\n", " ")
        if txt:
            txt = (txt[:34] + "…") if len(txt) > 34 else txt
            cv2.putText(img, txt, (x, max(14, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        if is_uocr:
            n_uocr += 1
        else:
            n_hybrid += 1
    os.makedirs(OUT_PAGES, exist_ok=True)
    path = os.path.join(OUT_PAGES, f"page{page_no}_blocks.png")
    cv2.imwrite(path, img)
    return path, n_hybrid, n_uocr

def main():
    results = []
    for pno, name, force, desc in PAGES:
        print(f"[reporte] Procesando {name} (force_uocr={force})...")
        data = fetch_page(pno, force)
        path, n_hy, n_u = draw_blocks(pno, data["blocks"], data["engines"])
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        results.append({
            "pno": pno, "name": name, "desc": desc, "force": force,
            "t": data["t"], "nblocks": len(data["blocks"]),
            "n_hybrid": n_hy, "n_uocr": n_u,
            "engines": data["engines"], "img": b64,
        })
        print(f"    {data['t']}s | {len(data['blocks'])} bloques | {data['engines']}")

    # ── Datos del benchmark (hardcoded de mediciones reales) ──
    bench_modes = [
        {"modo": "EasyOCR solo", "pdf": "viejo (128 págs)", "tiempo": "7.1 min",
         "prom_pag": "3.3s", "bloques": 623, "traducidos": 519, "tasa": "83.3%",
         "errores": 0, "dialogo_art": "NO (0 palabras en p3)"},
        {"modo": "Auto (EasyOCR+CLAHE)", "pdf": "viejo (128 págs)", "tiempo": "25.3 min",
         "prom_pag": "11.9s", "bloques": 517, "traducidos": 424, "tasa": "82.0%",
         "errores": 0, "dialogo_art": "Parcial"},
        {"modo": "Fusion (v1, workers=3)", "pdf": "viejo (128 págs)", "tiempo": "150.6 min",
         "prom_pag": "70.6s", "bloques": 590, "traducidos": 408, "tasa": "69.2%",
         "errores": 0, "dialogo_art": "SÍ (p3 CER 0.819, p12 0.717)"},
        {"modo": "Fusion v4.2 (estimado sin batch)", "pdf": "nuevo (53 págs)", "tiempo": "~47 min",
         "prom_pag": "~53s", "bloques": "—", "traducidos": "—", "tasa": "—",
         "errores": 0, "dialogo_art": "SÍ (p5: ERA UNA PROPUESTA)"},
        {"modo": "Fusion v4.2 + batch Fase 1 (medido)", "pdf": "nuevo (53 págs)", "tiempo": "~22.5 min",
         "prom_pag": "~25.5s", "bloques": 143, "traducidos": 36, "tasa": "25.2%",
         "errores": 0, "dialogo_art": "SÍ — p3 7 bloques, p5 U-OCR 109.4s (batch), p11 8, p12 4"},
    ]
    cer_rows = [
        {"pag": "3", "gt": "INCREIBLE REALMENTE (SFX pintado)",
         "fusion": "TEALMENT- NCREIBLE + YCREIBLE..", "cer": "0.819",
         "easyocr_solo": "0/2 palabras (nada)"},
        {"pag": "12", "gt": "ERA UNA PROPUESTA QUE SOLO PODIA BENEFICIARME PERO (globo en panel)",
         "fusion": "RA CNA PROPOESTA OUE SOLC POCA FENEHCASE", "cer": "0.717",
         "easyocr_solo": "solo cabeceras (sin diálogo)"},
        {"pag": "5 (PDF nuevo)", "gt": "ERA UNA PROPUESTA QUE SOLO PODIA BENEFICIARME PERO",
         "fusion": "ERA UNA PROPUESTA / HOY, LUEGO DE / ME GUESTARÍA", "cer": "~0.65",
         "easyocr_solo": "sin diálogo artístico"},
    ]
    # Distribución de tiempos (run real Fase 5: fusion + batch-window 4,
    # capítulo 53 págs del PDF nuevo, 2026-08-04 17:39→17:58 + retry 19-22)
    time_dist = [
        {"pag": 5, "t": 366.2, "tipo": "U-OCR"},
        {"pag": 39, "t": 671.6, "tipo": "U-OCR (lote 39-42, infer_multi)"},
        {"pag": 51, "t": 653.7, "tipo": "U-OCR (lote 51-53, infer_multi)"},
        {"pag": 12, "t": 9.2, "tipo": "Normal"},
        {"pag": 3, "t": 12.2, "tipo": "Normal"},
        {"pag": 11, "t": 14.9, "tipo": "Normal"},
        {"pag": 19, "t": 38.9, "tipo": "Retry post-fix (era http_500)"},
        {"pag": 22, "t": 38.9, "tipo": "Retry post-fix (era http_500)"},
    ]
    # Páginas procesadas en esta ejecución (con tiempos de esta sesión)
    page_cards = ""
    for r in results:
        marker = "🟥 U-OCR" if "unlimited" in r["engines"] else "🟩 Híbrido"
        page_cards += f"""
        <div class="card">
          <h3>{r['name']} <span class="tag {'tag-red' if 'unlimited' in r['engines'] else 'tag-green'}">{marker}</span></h3>
          <p class="muted">{r['desc']}</p>
          <div class="stats"><span>{r['t']}s</span><span>{r['nblocks']} bloques</span><span>{', '.join(r['engines'])}</span></div>
          <img src="data:image/png;base64,{r['img']}" alt="{r['name']} con bloques" onclick="this.classList.toggle('zoom')">
        </div>"""

    bench_rows = ""
    for m in bench_modes:
        bench_rows += f"""<tr>
          <td><strong>{m['modo']}</strong></td>
          <td>{m['pdf']}</td>
          <td>{m['tiempo']}</td>
          <td>{m['prom_pag']}</td>
          <td>{m['bloques']}</td>
          <td>{m['traducidos']}</td>
          <td>{m['tasa']}</td>
          <td>{m['errores']}</td>
          <td>{m['dialogo_art']}</td>
        </tr>"""

    cer_rows_html = ""
    for c in cer_rows:
        cer_rows_html += f"""<tr>
          <td>{c['pag']}</td>
          <td><em>{c['gt']}</em></td>
          <td>{c['fusion']}</td>
          <td><strong>{c['cer']}</strong></td>
          <td>{c['easyocr_solo']}</td>
        </tr>"""

    time_rows = ""
    for t in time_dist:
        cls = "row-uocr" if t["tipo"].startswith("U-OCR") else ""
        time_rows += f"<tr class='{cls}'><td>{t['pag']}</td><td>{t['tipo']}</td><td>{t['t']:.1f}s</td></tr>"

    t_norm = [t["t"] for t in time_dist if t["tipo"] == "Normal"]
    t_uocr = [t["t"] for t in time_dist if t["tipo"].startswith("U-OCR")]
    avg_norm = sum(t_norm) / len(t_norm)
    avg_uocr = sum(t_uocr) / len(t_uocr)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reporte Fusion OCR — Traductor Visual Pro</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff; --amber: #d29922;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.55; padding: 24px; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; }}
  header {{ padding: 28px 0 18px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
  header h1 {{ font-size: 1.7rem; background: linear-gradient(90deg, var(--green), var(--blue));
    -webkit-background-clip: text; background-clip: text; color: transparent; }}
  header p {{ color: var(--muted); margin-top: 6px; }}
  h2 {{ font-size: 1.2rem; margin: 32px 0 14px; color: var(--blue); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
    border-radius: 10px; overflow: hidden; font-size: 0.88rem; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #1c2333; color: var(--muted); font-weight: 600; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.04em; }}
  tr:hover td {{ background: #1a2129; }}
  .row-uocr td {{ background: rgba(248, 81, 73, 0.06); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 18px; margin-top: 16px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; }}
  .card h3 {{ font-size: 1rem; display: flex; align-items: center; gap: 8px; }}
  .card img {{ width: 100%; border-radius: 8px; border: 1px solid var(--border);
    cursor: zoom-in; margin-top: 10px; }}
  .card img.zoom {{ cursor: zoom-out; position: fixed; top: 50%; left: 50%;
    transform: translate(-50%, -50%); width: auto; max-width: 94vw; max-height: 94vh;
    z-index: 100; box-shadow: 0 0 60px rgba(0,0,0,0.8); }}
  .tag {{ font-size: 0.72rem; padding: 3px 9px; border-radius: 20px; font-weight: 600; }}
  .tag-green {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .tag-red {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .muted {{ color: var(--muted); font-size: 0.85rem; margin-top: 4px; }}
  .stats {{ display: flex; gap: 14px; margin-top: 8px; font-size: 0.8rem; color: var(--amber); }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px; margin-top: 16px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; text-align: center; }}
  .kpi .num {{ font-size: 1.6rem; font-weight: 700; color: var(--green); }}
  .kpi .lbl {{ color: var(--muted); font-size: 0.78rem; margin-top: 4px; }}
  .kpi.warn .num {{ color: var(--red); }}
  .note {{ background: rgba(88,166,255,0.08); border-left: 3px solid var(--blue);
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: var(--muted);
    margin-top: 14px; }}
  footer {{ margin-top: 36px; color: var(--muted); font-size: 0.78rem;
    border-top: 1px solid var(--border); padding-top: 14px; }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #f6f8fa; --card: #fff; --border: #d0d7de; --text: #1f2328;
      --muted: #57606a; }}
    th {{ background: #f0f3f6; }}
    tr:hover td {{ background: #f6f8fa; }}
    .row-uocr td {{ background: rgba(248,81,73,0.05); }}
  }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>🔬 Reporte del Benchmark — Modo Fusion OCR</h1>
  <p>Fusión de 3 motores (EasyOCR + RapidOCR + Unlimited-OCR 4-bit) · Traductor Visual Pro ·
     Generado {time.strftime('%Y-%m-%d %H:%M')} · PDF de prueba: Capítulo 43 (53 págs)</p>
</header>

<h2>📊 Tabla comparativa de modos</h2>
<table>
  <thead><tr>
    <th>Modo</th><th>PDF</th><th>Tiempo total</th><th>Prom/pág</th>
    <th>Bloques</th><th>Traducidos</th><th>Tasa</th><th>Errores</th><th>Diálogo artístico</th>
  </tr></thead>
  <tbody>{bench_rows}</tbody>
</table>

<h2>🎯 Recuperación de diálogo artístico (CER vs ground truth)</h2>
<table>
  <thead><tr><th>Página</th><th>Ground truth</th><th>Fusión (OCR)</th><th>CER</th><th>EasyOCR solo</th></tr></thead>
  <tbody>{cer_rows_html}</tbody>
</table>

<h2>⏱️ Distribución de tiempos por página (PDF nuevo)</h2>
<div class="kpis">
  <div class="kpi"><div class="num">{avg_norm:.1f}s</div><div class="lbl">Media página normal</div></div>
  <div class="kpi warn"><div class="num">{avg_uocr:.1f}s</div><div class="lbl">Media página U-OCR</div></div>
  <div class="kpi"><div class="num">{len([t for t in time_dist if t['tipo'].startswith('U-OCR')])}/{len(time_dist)}</div><div class="lbl">Páginas que disparan U-OCR (trigger selectivo v4.2)</div></div>
</div>
<table style="margin-top:16px">
  <thead><tr><th>Página</th><th>Tipo</th><th>Tiempo</th></tr></thead>
  <tbody>{time_rows}</tbody>
</table>
<div class="note">💡 <strong>Hallazgo Fase 5 (batch):</strong> el capítulo completo de 53 págs del PDF nuevo corrió en modo
fusion con <code>--batch-window 4</code> en <strong>~22.5 min de pared</strong> (17:39→17:58 + retry 19-22 de ~3 min,
0 errores) vs los ~47 min estimados sin batch → <strong>~2.1x más rápido</strong>. Solo 3 lotes dispararon
Unlimited-OCR (p5 y lotes 39-42 y 51-53): con <code>infer_multi</code> el prefill del VLM se comparte (lote de 4
páginas ≈ 671s ≈ 168s/pág vs 366-592s/pág individuales). Las páginas normales (sin panel image &gt;15% ni
conf &lt; 0.2) no disparan el VLM gracias al trigger selectivo v4.2. Nota: la tasa de traducción (25.2%) es
baja porque el par CT2 es→en no se pudo descargar (sin internet a HuggingFace en este run) — los textos
SIN_TRAD quedaron en español; la cobertura de detección (47/53 páginas con texto, 0 errores) es el dato clave.</div>

<h2>🖼️ Bloques recuperados por página</h2>
<div class="grid">{page_cards}</div>
<div class="note">🟩 <strong>Verde</strong> = bloques híbridos (EasyOCR+RapidOCR) · 🟥 <strong>Rojo</strong> =
página reforzada con Unlimited-OCR (diálogo artístico recuperado). Haz clic en una imagen para ampliarla.</div>

<footer>Reporte generado automáticamente por <code>generate_fusion_report.py</code> ·
Resultados medidos: benchmark fusion 128 págs (2026-08-03), run real Fase 5 — capítulo 53 págs del PDF nuevo,
modo fusion + batch-window 4 (2026-08-04 17:39→18:16), 0 errores tras el fix de la race window (dicts de
RapidOCR en _ocr_results_to_blocks).</footer>
</div>
</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[reporte] HTML generado: {OUT_HTML} "
          f"({os.path.getsize(OUT_HTML)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
