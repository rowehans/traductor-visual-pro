"""
diag_pdf.py — Diagnóstico de páginas PDF (DEPRECATED: manga_pipeline eliminado).

Este script quedó obsoleto tras la eliminación del módulo manga_pipeline.
El pipeline OCR actual usa EasyOCR + CTD (opcional).
"""

import sys, fitz, io, numpy as np, cv2
from PIL import Image
sys.path.insert(0, '.')

# manga_pipeline fue eliminado del proyecto (commit de9ba71).
# Usar ocr_utils._detect_and_ocr() en su lugar.
from ocr_utils import _detect_and_ocr, _group_and_merge_blocks

PDF = r"D:\crear traductor\test_input.pdf"


def diag_page(page_num):
    doc = fitz.open(PDF)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes('png')))
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    print(f'\n===== PAGINA {page_num+1} {img_bgr.shape} =====')
    # OCR con EasyOCR + pipeline de 2 niveles
    results = _detect_and_ocr(img_bgr, "es")
    blocks = _group_and_merge_blocks(results, img_bgr.shape[0])
    print(f'Total bloques despues filtro: {len(blocks)}')
    for i, b in enumerate(blocks):
        text = b.get('text', '')
        conf = b.get('confidence', 0)
        x, y, w, h = b.get('x', 0), b.get('y', 0), b.get('w', 0), b.get('h', 0)
        n_jp = sum(1 for c in text if 0x3040 <= ord(c) <= 0x30ff or 0x4e00 <= ord(c) <= 0x9faf)
        n_latin = sum(1 for c in text if c.isascii() and c.isalpha())
        n_total = len(text)
        print(f'  #{i+1} h={h} [{x},{y} {w}x{h}] conf={conf:.2f} jp={n_jp} latin={n_latin}/{n_total}')
        print(f'     {text[:90]!r}')


if __name__ == "__main__":
    print("Usando pipeline OCR actual (EasyOCR)...")
    diag_page(0)  # pagina 1
    diag_page(3)  # pagina 4
    diag_page(4)  # pagina 5
