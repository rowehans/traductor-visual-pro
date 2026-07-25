import requests, base64, sys, fitz, io, numpy as np, cv2, json, time
from PIL import Image

PDF = r"D:\crear traductor\test_input.pdf"
doc = fitz.open(PDF)

INDICES = [
    (0, "Portada"),
    (3, "Pagina 4 (carteleria)"),
    (4, "Pagina 5 (INCREIBLE)"),
    (9, "Pagina 10"),
    (16, "Pagina 17"),
    (124, "Pagina 125 (comentarios web)"),
    (125, "Pagina 126 (comentarios web)"),
    (127, "Pagina 128"),
]

for page_num, label in INDICES:
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    _, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    b64 = base64.b64encode(buf.tobytes()).decode()

    t0 = time.time()
    r = requests.post("http://127.0.0.1:5174/api/process-page",
                      json={"image": f"data:image/png;base64,{b64}", "target": "es", "source": "auto"},
                      timeout=120)
    dt = time.time() - t0
    try:
        data = r.json()
    except Exception:
        print(f"  Pagina {page_num+1} {label}: ERROR {r.status_code} {r.text[:80]}")
        continue
    blocks = data.get("blocks", [])
    print(f"\n----- Pagina {page_num+1} {label} ({dt:.1f}s, {len(blocks)} bloques) -----")
    for i, b in enumerate(blocks[:8]):
        src = b.get("source", "")
        txt = b.get("translated", "")
        x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0)
        print(f"  #{i+1} [{x},{y} {w}x{h}]")
        print(f"    src: {src[:90]!r}")
        if src != txt:
            print(f"    tra: {txt[:90]!r}")
