"""
reprocess_failed.py — Reintenta lo que falló en el batch original.

Cubre DOS tipos de fallo distintos:

    1. FALLOS DE PÁGINA (visibles): status timeout / render_error /
        conn_error / http_*. Se reprocesa la página completa contra
        /api/process-page (igual que antes).

    2. FALLOS DE BLOQUE (silenciosos): bloques donde source == translated
        dentro de una página que SÍ se marcó como procesada con éxito.
        Causa raíz: en translator.py, is_lenient permite que frases de
        <=3 palabras pasen la validación aunque la "traducción" sea
        idéntica al original — esos bloques nunca se marcan como error
        y por eso nunca se reintentaban antes de este parche.
        Se reparan SIN reabrir el PDF: se llama directo a /api/translate
        con el texto ya extraído, después de purgar la entrada de caché
        correspondiente (si no se purga, /api/translate devolvería el
        mismo resultado fallido cacheado).

Actualiza resultados_progreso.json al final con ambos tipos de fix.
"""
import os, sys, time, json, base64, gc, hashlib, re, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import requests

CHECKPOINT_FILE = "resultados_progreso.json"
PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
API_URL = "http://127.0.0.1:5174"
# DOC_ID (sesión 126): scope por documento de los caches de decisión del
# servidor — mismo hash que process_all_pages.py para que el reproceso del
# capítulo 43 reutilice las decisiones del run original (y el capítulo 47 no
# herede las del 43).
DOC_ID = hashlib.md5(os.path.basename(PDF_PATH).encode("utf-8")).hexdigest()[:12]
ZOOM = 1.5
TARGET = "en"
SOURCE = "es"
TIMEOUT = 90
MAX_RETRIES = 3
CACHE_DIR = os.path.join("cache", "translations")

# ── heurística de onomatopeya (misma lógica que analisis_calidad.py) ──
SHORT_SPANISH = {
    'EL','LA','LOS','LAS','UN','UNA','UNOS','UNAS','DEL','CON','POR','QUE','QUÉ',
    'SER','ESTA','ESTE','ES','NO','SI','YA','PERO','MAS','MÁS','SUS','ERA','HAN',
    'LES','LE','TUS','NOS','SON','AL','MI','TU','SE','TE','ME','LO','DE','EN','A',
    'Y','O','NI','VA','VE','FUE','IR','HAY','HE','HA','HAS','SOY','ERES','SEA',
    'TAN','TAL','AH','OH','EH','AY','OK','BIEN','MAL','TODO','SOLO','SÓLO','MUY',
    'VALE','LISTO','CLARO','CIERTO','BUENO','MALO','COMO','CÓMO','AHORA','HOY',
}
KNOWN_SFX = {
    'BOOM','PUM','ZAS','CRASH','CLICK','PLOP','TOC','RING','FLASH','BOING','POW',
    'BANG','SMASH','SPLASH','BUMP','THUD','WHAM','GRRR','GRR','CLANG','SNIFF',
    'GROAN','SLAM','BEEP','WOOSH','KABOOM','RUMBLE','SQUEAK','WHIR','ZOOM',
    'VROOM','SCREECH','GROWL','HOWL','SNAP','CRACKLE','POP','FIZZ','HISS','BUZZ',
    'DING','DONG','SPLAT','SQUISH','WHOOSH','HUH','HEH','HAH','HMPH','PSST','SHH',
    'GASP','PANT','PHEW','WHEW','SIGH','UFF','OW','OUCH','AY','OY','BAH','MEH',
    'ACK','EEK','TAP','KNOCK','RAP','PIT','PAT',
}

def es_onomatopeya(t: str) -> bool:
    s = t.strip(' \'"\u00a1\u00bf!?.,;:~-_()').upper()
    if not s or not s.isalpha():
        return False
    if len(s) < 3 or len(s) > 8:
        return False
    if s in SHORT_SPANISH:
        return False
    if s in KNOWN_SFX:
        return True
    if re.search(r'(.)\1{2,}', s):
        return True
    return False

def cache_key(text: str, src: str, tgt: str) -> str:
    raw = f"{text}||{src}||{tgt}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def borrar_cache(text: str, src: str, tgt: str) -> bool:
    path = os.path.join(CACHE_DIR, f"{cache_key(text, src, tgt)}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def pedir_traduccion(text: str) -> str | None:
    try:
        r = requests.post(f"{API_URL}/api/translate",
                            json={"text": text, "source": SOURCE, "target": TARGET},
                            timeout=30)
        if r.status_code == 200:
            return r.json().get("translatedText")
    except Exception as e:
        print(f"    [ERROR] {e}")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-bloques", action="store_true",
                        help="Omite el reprocesamiento de páginas completas (no requiere el PDF ni fitz/cv2)")
    args = ap.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"ERROR: No se encuentra {CHECKPOINT_FILE}")
        sys.exit(1)

    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        cp = json.load(f)

    total_pages = cp.get("total_pages", 128)
    results = cp.get("results", [])
    pages_done = set(cp.get("pages_done", []))

    # ── Identify page-level failures ────────────────────────────
    failed_pages = []
    for r in results:
        status = r.get("status", "")
        if status in ("timeout", "render_error", "conn_error") or str(status).startswith("http"):
            failed_pages.append(r["page"])
    failed_pages = sorted(set(failed_pages))

    # ── Identify block-level silent failures (src == tgt) ──────
    # Se excluyen las páginas que ya van a reprocesarse completas.
    bloques_fallidos = []  # (result_idx, texto_idx, page, src)
    for ri, r in enumerate(results):
        if r.get("page") in failed_pages:
            continue
        for ti, t in enumerate(r.get("texts", [])):
            src = (t.get("src") or "").strip()
            tgt = (t.get("tgt") or "").strip()
            if src and src == tgt and not es_onomatopeya(src):
                bloques_fallidos.append((ri, ti, r.get("page"), src))

    print(f"Páginas fallidas (error de página): {len(failed_pages)}")
    print(f"  {failed_pages}")
    print(f"Bloques fallidos (traducción silenciosamente idéntica): {len(bloques_fallidos)}")
    print()

    if not failed_pages and not bloques_fallidos:
        print("No hay nada que reprocesar. ")
        sys.exit(0)

    # ── Verify server ────────────────────────────────────────────
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        health = r.json()
        print(f"Servidor OK - memoria: {health.get('memory','?')}")
    except Exception as e:
        print(f"ERROR: No se puede conectar al servidor: {e}")
        sys.exit(1)

    new_results = []
    recovered = 0

    # ═══════════════════ FASE 1: páginas completas ═══════════════
    if failed_pages and not args.solo_bloques:
        import fitz, cv2, numpy as np
        from PIL import Image
        from io import BytesIO

        doc = fitz.open(PDF_PATH)

        for retry_count in range(1, MAX_RETRIES + 1):
            recovered_set = {r["page"] for r in new_results}
            remaining = [p for p in failed_pages if p not in recovered_set]
            if not remaining:
                break

            print(f"\n{'='*60}")
            print(f"  INTENTO {retry_count}/{MAX_RETRIES} — {len(remaining)} páginas pendientes")
            print(f"{'='*60}")

            for i, page_num in enumerate(remaining):
                pg_idx = page_num - 1
                print(f"  [{i+1}/{len(remaining)}] Pág {page_num} (intento {retry_count})...", end=" ", flush=True)

                try:
                    page = doc[pg_idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
                    img = Image.open(BytesIO(pix.tobytes('png')))
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode('.png', img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    b64 = 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()
                    del img, img_cv, buf, pix, page

                    t0 = time.time()
                    resp = requests.post(f"{API_URL}/api/process-page",
                        json={'image': b64, 'target': TARGET, 'source': SOURCE,
                              'doc_id': DOC_ID},
                        timeout=TIMEOUT)
                    elapsed = time.time() - t0
                    del b64

                    if resp.status_code == 200:
                        data = resp.json()
                        blocks = data.get("blocks", [])
                        n_blocks = len(blocks)
                        n_translated = sum(1 for b in blocks if b.get("source","") != b.get("translated",""))
                        status = ("OK" if n_translated == n_blocks > 0 else
                                    "PARCIAL" if n_translated > 0 else
                                    "SIN_TRAD" if n_blocks > 0 else "VACIO")

                        print(f" {status} ({elapsed:.1f}s, {n_blocks} bloques, {n_translated} trad)")

                        new_entry = {
                            "page": page_num,
                            "status": status,
                            "blocks": n_blocks,
                            "translated": n_translated,
                            "time": elapsed,
                            "texts": [{"src": b.get("source",""), "tgt": b.get("translated","")} for b in blocks]
                        }
                        new_results.append(new_entry)
                        pages_done.add(page_num)
                        recovered += 1
                    else:
                        err_msg = ""
                        try:
                            err_data = resp.json()
                            err_msg = err_data.get("error", str(resp.status_code))
                        except Exception:
                            err_msg = f"HTTP {resp.status_code}"
                        print(f" {err_msg} ({elapsed:.1f}s)")

                except requests.Timeout:
                    print(f"⏰ Timeout (>{TIMEOUT}s)")
                except Exception as e:
                    print(f" Error: {str(e)[:60]}")

                gc.collect()
                if i < len(remaining) - 1:
                    time.sleep(0.5)

            if retry_count < MAX_RETRIES:
                recovered_set = {r["page"] for r in new_results}
                still_failed_count = len([p for p in failed_pages if p not in recovered_set])
                if still_failed_count == 0:
                    break
                print(f"\n  {still_failed_count} páginas aún fallan. Reintentando...")

        doc.close()

    recovered_pages = {r["page"] for r in new_results}
    still_failed_pages = sorted([p for p in failed_pages if p not in recovered_pages])
    total_recovered = len(recovered_pages)

    for entry in new_results:
        pg = entry["page"]
        found = False
        for i, r in enumerate(results):
            if r["page"] == pg:
                results[i] = entry
                found = True
                break
        if not found:
            results.append(entry)

    # ═══════════════════ FASE 2: bloques silenciosos ══════════════
    bloques_mejorados = 0
    bloques_reporte = []  # (page, src, nueva, estado)
    if bloques_fallidos:
        print(f"\n{'='*60}")
        print(f"  REPARANDO {len(bloques_fallidos)} BLOQUES SILENCIOSOS")
        print(f"{'='*60}")
        for ri, ti, page, src in bloques_fallidos:
            borrado = borrar_cache(src, SOURCE, TARGET)
            nueva = pedir_traduccion(src)
            estado = "SIN CAMBIO"
            if nueva and nueva.strip() and nueva.strip() != src:
                results[ri]["texts"][ti]["tgt"] = nueva
                # Actualizar contador "translated" de esa página si existe
                if 0 <= ri < len(results) and "translated" in results[ri]:
                    results[ri]["translated"] = sum(
                        1 for tt in results[ri]["texts"]
                        if (tt.get("src") or "") != (tt.get("tgt") or "")
                    )
                estado = "MEJORADO"
                bloques_mejorados += 1
            print(f"  Pág {page:>3} [{'cache borrada' if borrado else 'sin cache':>13}] "
                    f"{estado:>10}: {src[:35]!r} -> {(nueva or src)[:35]!r}")
            bloques_reporte.append((page, src, nueva or src, estado))

    # ── Re-count stats from scratch for accuracy ────────────────
    all_pages_with_text = 0
    all_pages_translated = 0
    all_pages_empty = 0
    all_pages_error = 0
    all_blocks_found = 0
    all_blocks_translated = 0

    for r in results:
        status = r.get("status", "")
        if status in ("timeout", "render_error", "conn_error") or str(status).startswith("http"):
            all_pages_error += 1
            continue
        if status == "VACIO":
            all_pages_empty += 1
            continue
        n_blocks = r.get("blocks", 0)
        n_translated = sum(
            1 for tt in r.get("texts", [])
            if (tt.get("src") or "") != (tt.get("tgt") or "")
        ) if r.get("texts") else r.get("translated", 0)
        all_blocks_found += n_blocks
        all_blocks_translated += n_translated
        if n_blocks > 0:
            all_pages_with_text += 1
            if n_translated > 0:
                all_pages_translated += 1

    updated_cp = {
        "total_pages": total_pages,
        "pages_done": sorted(pages_done),
        "results": results,
        "stats": {
            "total_blocks_found": all_blocks_found,
            "total_blocks_translated": all_blocks_translated,
            "pages_with_text": all_pages_with_text,
            "pages_translated": all_pages_translated,
            "pages_empty": all_pages_empty,
            "pages_error": all_pages_error,
        },
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    tmp_file = CHECKPOINT_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(updated_cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, CHECKPOINT_FILE)

    # ── Final report ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REPORTE DE RECUPERACION")
    print("=" * 60)
    if failed_pages:
        print(f"\n  Páginas fallidas originales: {len(failed_pages)}")
        print(f"   Recuperadas: {total_recovered}")
        print(f"   Siguen fallando: {len(still_failed_pages)}")
        print(f"     Páginas: {still_failed_pages if still_failed_pages else 'ninguna'}")
    if bloques_fallidos:
        print(f"\n  Bloques silenciosos detectados: {len(bloques_fallidos)}")
        print(f"   Mejorados: {bloques_mejorados}")
        print(f"    Sin cambio: {len(bloques_fallidos) - bloques_mejorados}")
    print()
    print(f"  Nuevas estadísticas globales:")
    s = updated_cp["stats"]
    print(f"   Traducidas correctamente: {s['pages_translated']}")
    print(f"    Con texto sin traducir:  {s['pages_with_text'] - s['pages_translated']}")
    print(f"  ℹ  Vacías (arte):            {s['pages_empty']}")
    print(f"   Con error:                 {s['pages_error']}")
    print(f"   Total bloques:             {s['total_blocks_found']}")
    print(f"   Traducidos:                {s['total_blocks_translated']}")
    if s['total_blocks_found'] > 0:
        print(f"   Tasa traducción:           {s['total_blocks_translated']/s['total_blocks_found']*100:.1f}%")
    print(f"\n  Checkpoint actualizado: {CHECKPOINT_FILE}")

    # ── Reporte en texto plano para pegar a una IA de solo texto ──
    pendientes = [x for x in bloques_reporte if x[3] == "SIN CAMBIO"]
    if pendientes:
        print()
        print("=" * 60)
        print("  BLOQUES QUE SIGUEN SIN TRADUCIR (para IA de texto)")
        print("=" * 60)
        for page, src, _, _ in pendientes:
            print(f"  Página {page} | texto original (es): {src}")

if __name__ == "__main__":
    main()
