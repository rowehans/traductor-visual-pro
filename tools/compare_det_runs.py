"""Compara 2 checkpoints de process_all_pages (runs A y B) para verificar el
determinismo de la política (sesión 116): mismas decisiones de trigger por
página (mismo n_blocks/n_translated/status), mismos textos fuente (OCR) y
mismas traducciones. Encoding seguro para CJK (evita cp1252).

Uso: python tools/compare_det_runs.py <cp_a.json> <cp_b.json>
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])

    ra = {r["page"]: r for r in a.get("results", [])}
    rb = {r["page"]: r for r in b.get("results", [])}
    pages = sorted(set(ra) | set(rb))

    sa = a.get("stats", {})
    sb = b.get("stats", {})
    print(f"{'métrica':<28} {'run A':>10} {'run B':>10} {'Δ':>8}")
    print("-" * 60)
    for k in ("total_blocks_found", "total_blocks_translated",
              "pages_with_text", "pages_translated", "pages_empty", "pages_error"):
        va, vb = sa.get(k, 0), sb.get(k, 0)
        print(f"{k:<28} {va:>10} {vb:>10} {vb-va:>+8}")
    ta = sa.get("total_blocks_found", 0) or 1
    tb = sb.get("total_blocks_found", 0) or 1
    tasa_a = sa.get("total_blocks_translated", 0) / ta * 100
    tasa_b = sb.get("total_blocks_translated", 0) / tb * 100
    print(f"tasa traducción           {tasa_a:>9.1f}% {tasa_b:>9.1f}% {tasa_b-tasa_a:>+7.1f}pt")

    print()
    print("=== Diferencias por página (si hay) ===")
    diffs = 0
    for p in pages:
        ra_p = ra.get(p)
        rb_p = rb.get(p)
        if ra_p is None or rb_p is None:
            print(f"p{p}: presente solo en {'A' if rb_p is None else 'B'}")
            diffs += 1
            continue
        fa = (ra_p.get("blocks"), ra_p.get("translated"), ra_p.get("status"))
        fb = (rb_p.get("blocks"), rb_p.get("translated"), rb_p.get("status"))
        texts_a = [(t.get("src", ""), t.get("tgt", "")) for t in ra_p.get("texts", [])]
        texts_b = [(t.get("src", ""), t.get("tgt", "")) for t in rb_p.get("texts", [])]
        same = fa == fb and texts_a == texts_b
        if not same:
            diffs += 1
            print(f"p{p}: A={fa} B={fb}")
            if texts_a != texts_b:
                print(f"    textos A: {[s[:20] for s, _ in texts_a]}")
                print(f"    textos B: {[s[:20] for s, _ in texts_b]}")
    if diffs == 0:
        print("  (ninguna — ambas corridas idénticas)")
    print()
    print(f"Páginas con diferencias: {diffs}/{len(pages)}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
