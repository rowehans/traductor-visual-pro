import sys, fitz, io, numpy as np, cv2
sys.path.insert(0, '.')
import manga_pipeline
from manga_pipeline import run_pipeline, ensure_ready, _looks_like_hallucination, _count_japanese_chars, _has_japanese_text

PDF = r"D:\crear traductor\test_input.pdf"

def diag_page(page_num):
    doc = fitz.open(PDF)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    from PIL import Image
    img = Image.open(io.BytesIO(pix.tobytes('png')))
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    print(f'\n===== PAGINA {page_num+1} {img_bgr.shape} =====')
    result = run_pipeline(img_bgr, 'auto')
    blocks = result.get('blocks', [])
    print(f'Total bloques despues filtro: {len(blocks)}')
    for i, b in enumerate(blocks):
        text = b.get('text', '')
        conf = b.get('confidence', 0)
        h = b.get('h', 0)
        x, y, w = b.get('x', 0), b.get('y', 0), b.get('w', 0)
        is_jp = _has_japanese_text(text)
        n_jp = _count_japanese_chars(text)
        n_latin = sum(1 for c in text if c.isascii() and c.isalpha())
        n_total = len(text)
        halluc = _looks_like_hallucination(text)
        print(f'  #{i+1} h={h} [{x},{y} {w}x{h}] conf={conf:.2f} jp={n_jp} latin={n_latin}/{n_total} halluc={halluc}')
        print(f'     {text[:90]!r}')

if __name__ == "__main__":
    print("Inicializando modelos...")
    ensure_ready()
    diag_page(0)  # pagina 1
    diag_page(3)  # pagina 4
    diag_page(4)  # pagina 5
