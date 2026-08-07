"""Mide la tasa real de colisión de _page_signature entre PDFs distintos.

Riesgo a cuantificar: los caches de decisión por firma (trigger sesión 116 y
§8.4.1 de negativas) son POR PROCESO y viven 30 min (TTL). Si el usuario
procesa dos PDFs distintos en la misma ventana, una página del PDF B con la
MISMA firma exacta que una página del PDF A heredaría la decisión de A —
interferencia cross-PDF.

La firma es layout-only (grid 8x8 de oscuridad + dark_ratio cuantizado a 1
decimal), así que el riesgo es real si manga distintos comparten layout.

Sesión 126: el servidor ya escopea la firma por documento (doc_id) — la clave
real del cache es "doc_id:firma". Con --scoped se mide la colisión EFECTIVA
entre documentos (con prefijo) que, al ser doc_ids distintos, debe ser 0.

Uso: python tools/medir_colision_firmas.py <pdf_a> <pdf_b> [zoom] [--scoped]
"""
import sys
from collections import Counter

import cv2
import fitz
import numpy as np

sys.path.insert(0, ".")
from ocr_utils import _page_signature  # noqa: E402

ZOOM = 1.2  # mismo render que process_all_pages.py


def _scoped_key(doc_id: str, firma: str) -> str:
    """Mismo prefijo que OCRManager._firma_documento (sesión 126): la clave
    real del cache de decisiones es "doc_id:firma" cuando el caller envía
    doc_id. doc_id vacío o firma vacía → sin prefijo (scope legacy)."""
    if not doc_id or not firma:
        return firma
    return f"{doc_id}:{firma}"


def firmas_de_pdf(path, zoom=ZOOM):
    """Renderiza cada página como el pipeline real y computa su firma."""
    doc = fitz.open(path)
    firmas = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA → BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        firmas.append((i + 1, _page_signature(img)))
        page = None
        pix = None
    doc.close()
    return firmas


def main():
    if len(sys.argv) < 3:
        print("Uso: python tools/medir_colision_firmas.py <pdf_a> <pdf_b> [zoom]")
        return 2
    path_a, path_b = sys.argv[1], sys.argv[2]
    # El zoom es el 3er argumento posicional; los flags (--scoped) se ignoran
    zoom = ZOOM
    for arg in sys.argv[3:]:
        if arg.replace(".", "", 1).isdigit():
            zoom = float(arg)
            break

    fa = firmas_de_pdf(path_a, zoom)
    fb = firmas_de_pdf(path_b, zoom)

    def _stats(nombre, firmas):
        sigs = [s for _, s in firmas]
        unicas = set(sigs)
        vacias = sum(1 for s in sigs if not s)
        rep = Counter(sigs)
        print(f"--- {nombre}: {len(firmas)} páginas ---")
        print(f"  firmas únicas: {len(unicas)} ({len(unicas)/max(len(firmas),1)*100:.1f}%)")
        print(f"  firmas vacías (no procesable): {vacias}")
        # páginas que comparten firma dentro del MISMO pdf
        dup = {s: c for s, c in rep.items() if c > 1 and s}
        if dup:
            print(f"  firmas repetidas intra-pdf: {len(dup)}")
            for s, c in sorted(dup.items(), key=lambda kv: -kv[1])[:5]:
                paginas = [p for p, ss in firmas if ss == s]
                print(f"    {s[:20]}… ×{c} en págs {paginas[:8]}")
        return unicas

    ua = _stats(path_a, fa)
    ub = _stats(path_b, fb)

    # ── Colisión EXACTA cross-PDF (firma BRUTA, sin scope) ──────
    sig_a = {s for _, s in fa if s}
    sig_b = {s for _, s in fb if s}
    comunes = sig_a & sig_b
    print()
    print(f"=== Colisión EXACTA de firmas entre PDFs (firma BRUTA) ===")
    print(f"firmas A: {len(sig_a)} | firmas B: {len(sig_b)} | comunes: {len(comunes)}")
    if comunes:
        for s in sorted(comunes)[:15]:
            pa = [p for p, ss in fa if ss == s]
            pb = [p for p, ss in fb if ss == s]
            print(f"  {s[:30]}… → A págs {pa} | B págs {pb}")
        # páginas afectadas
        pag_a = sum(1 for _, s in fa if s in comunes)
        pag_b = sum(1 for _, s in fb if s in comunes)
        print(f"páginas de A que colisionan: {pag_a}/{len(fa)}")
        print(f"páginas de B que colisionan: {pag_b}/{len(fb)}")
    else:
        print("  → NINGUNA colisión exacta (riesgo cero de interferencia cross-PDF)")

    # ── Colisión EFECTIVA con scope por documento (sesión 126) ──
    # La clave real del cache es "doc_id:firma" (OCRManager._firma_documento).
    # Con doc_ids distintos (uno por PDF), la colisión efectiva es SIEMPRE 0
    # aunque las firmas brutas colisionen — el capítulo 47 nunca hereda las
    # decisiones del 43. Solo se muestra con --scoped (requiere 2 PDFs).
    if "--scoped" in sys.argv:
        import hashlib
        doc_a = "doc" + hashlib.md5(path_a.encode("utf-8")).hexdigest()[:8]
        doc_b = "doc" + hashlib.md5(path_b.encode("utf-8")).hexdigest()[:8]
        esc_a = {_scoped_key(doc_a, s) for _, s in fa if s}
        esc_b = {_scoped_key(doc_b, s) for _, s in fb if s}
        comunes_esc = esc_a & esc_b
        print()
        print("=== Colisión EFECTIVA con scope por documento (sesión 126) ===")
        print(f"claves A ({doc_a}): {len(esc_a)} | claves B ({doc_b}): {len(esc_b)}"
              f" | comunes: {len(comunes_esc)}")
        print(f"  → {'NINGUNA colisión: el scope por doc_id elimina la interferencia cross-PDF ✅' if not comunes_esc else '¡COLISIÓN! (revisar _firma_documento)'}")

    # ── Proximidad: distancia de Hamming entre firmas cercanas ──
    # Si no hay colisión exacta pero los layouts son similares, el dark_ratio
    # cuantizado o 1 bit de la cuadrícula pueden variar → casi-colisiones.
    print()
    print("=== Proximidad (Hamming entre bits de la cuadrícula) ===")
    cercanas = []
    for s_a in sig_a:
        bits_a = int(s_a.split(":")[1], 16) if ":" in s_a else 0
        for s_b in sig_b:
            bits_b = int(s_b.split(":")[1], 16) if ":" in s_b else 0
            h = bin(bits_a ^ bits_b).count("1")
            if h <= 4:  # hasta 4 celdas de 64 difieren → layout casi idéntico
                cercanas.append((s_a, s_b, h))
    print(f"pares firma-A × firma-B con ≤4 celdas de diferencia: {len(cercanas)}")
    if cercanas:
        for s_a, s_b, h in cercanas[:15]:
            print(f"  Hamming={h}: A {s_a[:30]}… | B {s_b[:30]}…")
    # peor caso: ¿algún par comparte el MISMO dark_ratio y difiere en ≤1 bit?
    mismo_ratio = [
        (s_a, s_b, h) for s_a, s_b, h in cercanas
        if s_a.split(":")[0] == s_b.split(":")[0] and h <= 1
    ]
    print(f"  con mismo dark_ratio y ≤1 bit: {len(mismo_ratio)}")

    print()
    print("Interpretación: una colisión EXACTA dentro del TTL de 30 min hace que")
    print("el cache de decisión (trigger/§8.4.1) de A se aplique a la página de B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
