"""
test_uocr_art_recover.py — Test unitario (sin GPU) del post-procesado de
recuperación de diálogo en arte de uocr_daemon.py.

Valida la matemática de _recover_art_dialogue con un modelo simulado:
  1. Recorte de un bloque image grande.
  2. Escala + letterbox blanco a 640x640.
  3. Mapeo de vuelta al espacio de píxeles de la página original.

Uso:  env/Scripts/python.exe test_uocr_art_recover.py
"""

import os
import sys

import cv2
import numpy as np

# El daemon solo importa stdlib a nivel de módulo (torch/cv2 son lazy) → seguro
import uocr_daemon as ud


def _make_page(w: int, h: int) -> str:
    """Página sintética: fondo blanco con un 'panel' gris y un trazo negro."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # panel gris en (200,300,600x800)
    img[300:1100, 200:800] = (200, 200, 200)
    # 'texto': rectángulo negro dentro del panel (diálogo incrustado en arte)
    cv2.rectangle(img, (260, 420), (740, 460), (0, 0, 0), -1)
    path = "_test_page.png"
    cv2.imwrite(path, img)
    return path

# Posición del 'texto' en la página sintética (round-trip esperado)
_PAGE_TEXT = {"x": 260, "y": 420, "w": 480, "h": 40}
_PAGE_PANEL = {"x": 200, "y": 300, "w": 600, "h": 800}


def _fake_infer_once(image_path, out_dir, max_length, crop_mode=True, image_size=640):
    """Simula el modelo leyendo el canvas 640x640 REAL que generó el daemon.

    Detecta el rectángulo oscuro en el canvas (misma geometría que el daemon:
    recorte con pad=8 + escala + letterbox) y devuelve su bbox en coordenadas
    del canvas. Esto valida el round-trip completo con geometría real.
    """
    canvas = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert canvas is not None, f"El canvas no existe: {image_path}"
    _, dark = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(dark)
    # componente oscuro más grande (el 'texto')
    best = max((stats[i] for i in range(1, num)), key=lambda s: s[4])
    x, y, bw, bh = best[0], best[1], best[2], best[3]
    stream = (
        f"<|det|>text [{x}, {y}, {bw}, {bh}]<|/det|>DIÁLOGO RECUPERADO\n"
    )
    return "texto limpio", ud._parse_blocks(stream), 1.23


def test_coord_mapping():
    page_path = _make_page(1000, 1400)
    try:
        big_image_block = {"type": "image", "x": _PAGE_PANEL["x"], "y": _PAGE_PANEL["y"],
                           "w": _PAGE_PANEL["w"], "h": _PAGE_PANEL["h"], "text": ""}
        blocks = [big_image_block]

        orig_infer = ud._infer_once
        ud._infer_once = _fake_infer_once  # mock
        try:
            final_blocks, n_rec = ud._recover_art_dialogue(page_path, blocks, 4096)
        finally:
            ud._infer_once = orig_infer

        assert n_rec == 1, f"Esperaba 1 bloque recuperado, obtuve {n_rec}"
        rec = final_blocks[-1]
        assert rec["from_art_recrop"] is True
        assert rec["text"] == "DIÁLOGO RECUPERADO", rec

        # Round-trip: el texto recuperado debe volver a la posición original en
        # la página (con tolerancia de ±4px por los redondeos de la escala).
        tol = 5
        for key in ("x", "y", "w", "h"):
            assert abs(rec[key] - _PAGE_TEXT[key]) <= tol, (
                f"{key}: esperado {_PAGE_TEXT[key]}, obtuve {rec[key]}"
            )
        print(f"  ✓ mapeo de coordenadas correcto: "
              f"({rec['x']},{rec['y']},{rec['w']}x{rec['h']}) ≈ ({_PAGE_TEXT['x']},"
              f"{_PAGE_TEXT['y']},{_PAGE_TEXT['w']}x{_PAGE_TEXT['h']})")
    finally:
        os.remove(page_path)


def test_no_recover_when_small_image_block():
    """Un bloque image pequeño (<30%) NO debe disparar re-OCR."""
    page_path = _make_page(1000, 1400)
    try:
        small_block = {"type": "image", "x": 200, "y": 300, "w": 200, "h": 200, "text": ""}
        # area = 40K < 420K (30% de 1.4M) → sin re-OCR
        orig_infer = ud._infer_once
        calls = []
        def spy(*a, **k):
            calls.append(1)
            return _fake_infer_once(*a, **k)
        ud._infer_once = spy
        try:
            final_blocks, n_rec = ud._recover_art_dialogue(page_path, [small_block], 4096)
        finally:
            ud._infer_once = orig_infer
        assert n_rec == 0, f"Esperaba 0 recuperados, obtuve {n_rec}"
        assert not calls, "No debería haberse llamado a _infer_once"
        assert len(final_blocks) == 1
        print("  ✓ bloque image pequeño ignorado (sin re-OCR)")
    finally:
        os.remove(page_path)


def test_missing_image_file_graceful():
    """Archivo de página inexistente → no debe explotar."""
    final_blocks, n_rec = ud._recover_art_dialogue("no_existe.png",
                                                   [{"type": "image", "x": 0, "y": 0,
                                                     "w": 500, "h": 500, "text": ""}], 4096)
    assert n_rec == 0
    assert len(final_blocks) == 1
    print("  ✓ archivo faltante manejado con gracia")


def test_sub_block_image_skipped():
    """El re-OCR no debe añadir sub-bloques tipo image ni vacíos."""
    page_path = _make_page(1000, 1400)
    try:
        big = {"type": "image", "x": 200, "y": 300, "w": 600, "h": 800, "text": ""}
        orig_infer = ud._infer_once
        def fake(image_path, out_dir, max_length, crop_mode=True, image_size=640):
            # El modelo real emite CADA bloque en su propia línea
            stream = ("<|det|>image [0, 0, 100, 100]<|/det|>\n"
                      "<|det|>text [128, 96, 384, 32]<|/det|>DIÁLOGO\n")
            return "x", ud._parse_blocks(stream), 0.5
        ud._infer_once = fake
        try:
            final_blocks, n_rec = ud._recover_art_dialogue(page_path, [big], 4096)
        finally:
            ud._infer_once = orig_infer
        assert n_rec == 1, f"Esperaba 1, obtuve {n_rec}"
        # El sub-bloque image no se añade; solo el text
        assert len(final_blocks) == 2, f"Esperaba 2 bloques (1 image + 1 text), obtuve {len(final_blocks)}"
        print("  ✓ sub-bloques image/vacíos descartados correctamente")
    finally:
        os.remove(page_path)


def main():
    print("=== Test unitario _recover_art_dialogue (sin GPU) ===")
    test_coord_mapping()
    test_no_recover_when_small_image_block()
    test_missing_image_file_graceful()
    test_sub_block_image_skipped()
    print("=== TODOS LOS TESTS PASARON ===")


if __name__ == "__main__":
    main()
