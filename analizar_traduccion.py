"""
analizar_traduccion.py — Escanea todas las páginas de un PDF y detecta
traducciones deficientes: idénticas al original, corruptas, o con
longitud anormal. Genera reporte en consola y archivo HTML.
"""
import base64, io, json, os, sys, time, traceback
from pathlib import Path

import cv2, fitz, numpy as np, requests
from PIL import Image

PDF_PATH = Path(__file__).parent / "test_input.pdf"
SERVER_URL = "http://127.0.0.1:5174"
MAX_PAGES = 0  # 0 = todas
TIMEOUT = 120
TARGET_LANG = "es"

# ─── Heurísticas de mala traducción ──────────────────────────
FLAGS = {
    "IDENTICO": lambda s, t: s == t,
    "VACIO":   lambda s, t: not t or len(t.strip()) < 2,
    "CORTO":   lambda s, t: len(t) < len(s) * 0.2,
    "LARGO":   lambda s, t: len(t) > len(s) * 5,
    "ENCODING_ROTO": lambda s, t: "\ufffd" in t,
    "GARBAGE": lambda s, t: sum(c in "[]`~^<>{}|\\" for c in t) > len(t) * 0.3,
}


def verificar_servidor():
    try:
        r = requests.get(f"{SERVER_URL}/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def esperar_servidor(timeout=45):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if verificar_servidor():
            print("[OK] Servidor detectado.")
            return True
        time.sleep(2)
    return False


def analizar_bloque(src: str, tgt: str) -> list[str]:
    """Retorna lista de flags de anomalía para un bloque traducido."""
    alertas = []
    for nombre, fn in FLAGS.items():
        if fn(src, tgt):
            alertas.append(nombre)
    return alertas


def procesar_pagina(pix, page_num: int) -> dict:
    """Envía una página al servidor y retorna resultados."""
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    b64 = base64.b64encode(buf.tobytes()).decode()

    t0 = time.time()
    r = requests.post(
        f"{SERVER_URL}/api/process-page",
        json={"image": f"data:image/png;base64,{b64}", "target": TARGET_LANG, "source": "auto"},
        timeout=TIMEOUT,
    )
    dt = time.time() - t0
    data = r.json()
    bloques = data.get("blocks", [])

    resultado = {
        "pagina": page_num,
        "tiempo": dt,
        "total_bloques": len(bloques),
        "status": r.status_code,
        "bloques": [],
        "errores": [],
        "salud": "OK",
    }

    if r.status_code != 200:
        resultado["salud"] = "ERROR_HTTP"
        resultado["errores"].append(f"HTTP {r.status_code}")
        return resultado

    for i, b in enumerate(bloques):
        src = b.get("source", "")
        tgt = b.get("translated", "")
        alertas = analizar_bloque(src, tgt)
        info_bloque = {
            "idx": i,
            "source": src,
            "translated": tgt,
            "alertas": alertas,
        }
        resultado["bloques"].append(info_bloque)
        if alertas:
            resultado["errores"].extend(alertas)

    # Salud general de la página
    if not bloques:
        resultado["salud"] = "SIN_TEXTO"
    elif resultado["errores"]:
        resultado["salud"] = "ALERTAS"
    return resultado


def generar_reporte(resultados: list[dict], pdf_name: str):
    """Imprime reporte y genera HTML."""
    # ── Resumen consola ──
    total_ok = sum(1 for r in resultados if r["salud"] == "OK")
    total_alertas = sum(1 for r in resultados if r["salud"] == "ALERTAS")
    total_sin_texto = sum(1 for r in resultados if r["salud"] == "SIN_TEXTO")
    total_error = sum(1 for r in resultados if r["salud"] == "ERROR_HTTP")

    print("=" * 70)
    print(f"RESUMEN — {pdf_name}")
    print("=" * 70)
    print(f"  Páginas analizadas: {len(resultados)}")
    print(f"  ✅ OK:               {total_ok}")
    print(f"  ⚠️  Alertas:          {total_alertas}")
    print(f"  🔇 Sin texto:        {total_sin_texto}")
    print(f"  ❌ Error HTTP:       {total_error}")
    print()

    # Páginas con problemas
    problemas = [r for r in resultados if r["salud"] != "OK" and r["salud"] != "SIN_TEXTO"]
    if problemas:
        print("PÁGINAS CON PROBLEMAS DE TRADUCCIÓN:")
        print("-" * 70)
        for r in sorted(problemas, key=lambda x: -len(x["errores"])):
            print(f"  Pág {r['pagina']:3d} ({r['tiempo']:5.1f}s) — {r['salud']}")
            for b in r["bloques"]:
                if b["alertas"]:
                    print(f"    Bloque #{b['idx']}: {', '.join(b['alertas'])}")
                    print(f"      src: {b['source'][:80]}")
                    print(f"      tgt: {b['translated'][:80]}")
            print()

    # ── Reporte HTML ──
    html_path = Path(__file__).parent / "reporte_traduccion.html"
    rows = ""
    for r in resultados:
        color = {"OK": "green", "ALERTAS": "orange", "SIN_TEXTO": "gray", "ERROR_HTTP": "red"}.get(r["salud"], "black")
        bloques_html = ""
        for b in r["bloques"]:
            alerts = ", ".join(b["alertas"]) if b["alertas"] else "—"
            bloques_html += f"""
            <tr>
                <td>{b['idx']}</td>
                <td>{b['source'][:80]}</td>
                <td>{b['translated'][:80]}</td>
                <td style="color:{'red' if b['alertas'] else 'green'}">{alerts}</td>
            </tr>"""
        rows += f"""
        <tr style="background:{color}20">
            <td><b>{r['pagina']}</b></td>
            <td>{r['tiempo']:.1f}s</td>
            <td>{r['total_bloques']}</td>
            <td style="color:{color};font-weight:bold">{r['salud']}</td>
            <td>{', '.join(set(r['errores'])) or '—'}</td>
        </tr>
        {bloques_html}"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Reporte Traducción</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; font-size: 13px; }}
th {{ background: #333; color: #fff; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
</style></head><body>
<h1>Reporte de Traducción — {pdf_name}</h1>
<p>Total páginas: {len(resultados)} | ✅ {total_ok} | ⚠️ {total_alertas} | 🔇 {total_sin_texto} | ❌ {total_error}</p>
<table><thead><tr>
<th>Pág</th><th>Tiempo</th><th>Bloques</th><th>Salud</th><th>Alertas</th>
</tr></thead><tbody>{rows}</tbody></table>
<p><i>Generado por analizar_traduccion.py</i></p>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    print(f"[📄] Reporte HTML guardado: {html_path}")
    return html_path


def main():
    pdf = PDF_PATH
    if not pdf.exists():
        print(f"[ERROR] No se encuentra el PDF: {pdf}")
        print("Copia tu PDF aquí o edita PDF_PATH en el script.")
        return

    print(f"[📂] PDF: {pdf} ({pdf.stat().st_size // 1024 // 1024} MB)")
    doc = fitz.open(str(pdf))
    total = doc.page_count
    a_analizar = min(total, MAX_PAGES) if MAX_PAGES else total
    print(f"[📄] Páginas totales: {total}, a analizar: {a_analizar}")

    if not esperar_servidor():
        print("[ERROR] No se pudo conectar al servidor. ¿Está corriendo Flask?")
        print("  Ejecuta primero: $env:SKIP_MIT_INIT='1'; & '.\\env\\Scripts\\python.exe' server.py")
        return

    resultados = []
    t_inicio = time.time()

    for idx in range(a_analizar):
        page = doc[idx]
        pix = page.get_pixmap(dpi=150)
        t_pag = time.time()
        print(f"[▶] Página {idx+1}/{a_analizar}... ", end="", flush=True)

        try:
            res = procesar_pagina(pix, idx + 1)
            resultados.append(res)
            icono = {"OK": "✅", "ALERTAS": "⚠️", "SIN_TEXTO": "🔇", "ERROR_HTTP": "❌"}.get(res["salud"], "?")
            print(f"{icono} {res['salud']} ({res['total_bloques']} bloques, {res['tiempo']:.1f}s)")
        except Exception as e:
            print(f"❌ Error: {e}")
            resultados.append({"pagina": idx + 1, "salud": "ERROR_HTTP", "errores": [str(e)], "tiempo": time.time() - t_pag, "total_bloques": 0, "bloques": []})

        # Progreso estimado
        if (idx + 1) % 5 == 0:
            transcurrido = time.time() - t_inicio
            ritmo = transcurrido / (idx + 1)
            restante = ritmo * (a_analizar - idx - 1)
            print(f"     [⏱ {transcurrido:.0f}s transcurridos, ~{restante:.0f}s restantes]")

    doc.close()

    # ── Reporte final ──
    print()
    reporte = generar_reporte(resultados, pdf.name)
    total_t = time.time() - t_inicio
    print(f"[⏱] Tiempo total: {total_t:.0f}s ({total_t/60:.1f} min)")
    print(f"[📊] Reporte: {reporte}")
    print()
    print("PÁGINAS CON TRADUCCIÓN DEFICIENTE (prioritarias):")
    malas = sorted(
        [r for r in resultados if r["salud"] in ("ALERTAS", "ERROR_HTTP")],
        key=lambda x: -len(x.get("errores", [])),
    )
    for r in malas[:10]:
        print(f"  🔴 Pág {r['pagina']}: {', '.join(set(r.get('errores', [])))}")
    if not malas:
        print("  ✅ Ninguna — todas las páginas traducidas correctamente.")


if __name__ == "__main__":
    main()