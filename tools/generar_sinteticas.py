"""generar_sinteticas.py — Páginas de manga sintéticas con etiquetas EXACTAS.

Segunda fuente de clases (después del teacher VLM): genera páginas tipo manga
con PIL — paneles, globos con cola, texto libre, títulos, onomatopeyas y arte
simple — y escribe etiquetas YOLO PERFECTAS (el cuadro del texto dibujado, sin
oráculo ni heurística). El detector aprende el patrón "región de texto" con
volumen masivo; los datos reales del teacher aportan el estilo del capítulo.

Mismo enfoque que dmMaze (1/3 de su set fue sintético). El VAL siempre es real
(train_data/vlm/val) para que el A/B siga siendo honesto.

Uso:
  env/Scripts/python.exe tools/generar_sinteticas.py --n 150
  env/Scripts/python.exe tools/entrenar_detector.py --extra-data train_data/synth
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASES = ["text_bubble", "text_free"]

# Vocabulario corto de diálogo (el OCR no importa: el detector aprende cajas).
VOCAB = [
    "HOLA", "NO PUEDO", "ESPERA", "QUE", "SI", "NO", "VEN", "YA SE",
    "PERO", "POR QUE", "MIRA", "ALTO", "CREO", "QUE ES", "MIO", "TU",
    "EL", "ELLA", "NOS", "VAMOS", "AHORA", "CUIDADO", "BASTA", "ESCUCHA",
    "DIME", "QUIEN", "DONDE", "CUANDO", "ASI", "CLARO", "PUEDO", "DEJAME",
    "NO SE", "UNO", "DOS", "TRES", "MAS", "MENOS", "AQUI", "ALLA",
]

_FONT_DIR = Path("C:/Windows/Fonts")
_FONTS = [
    ("comic.ttf", "comicbd.ttf"),     # Comic Sans MS — el más manga
    ("arial.ttf", "arialbd.ttf"),
    ("cour.ttf", "courbd.ttf"),
]
_IMPACT = _FONT_DIR / "impact.ttf"

W, H = 1200, 1680


def _ri(rng: random.Random, a: int, b: int) -> int:
    """randint con rango clampeado (paneles pequeños no rompen el layout)."""
    return a if b <= a else rng.randint(a, b)


def _cargar_fuentes() -> list[tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]]:
    """Pares (regular, bold) de las fuentes disponibles."""
    pares = []
    for reg, bold in _FONTS:
        try:
            pares.append((ImageFont.truetype(str(_FONT_DIR / reg), 32),
                          ImageFont.truetype(str(_FONT_DIR / bold), 32)))
        except OSError:
            continue
    if not pares:
        pares = [(ImageFont.load_default(), ImageFont.load_default())]
    return pares


def _layout_paneles(rng: random.Random):
    """Divide la página en paneles tipo manga: filas × columnas con márgenes."""
    margen = 22
    n_rows = rng.randint(1, 3)
    filas = []
    y = margen
    for r in range(n_rows):
        # Altura de fila aleatoria (las últimas filas se reparten el resto)
        if r == n_rows - 1:
            h_fila = H - 2 * margen - y
        else:
            h_fila = rng.randint(320, 560)
        n_cols = rng.randint(1, 3)
        x = margen
        panel_row = []
        for c in range(n_cols):
            if c == n_cols - 1:
                w_col = W - 2 * margen - x
            else:
                w_col = rng.randint(280, 520)
            panel_row.append((x, y, x + w_col, y + h_fila))
            x += w_col + 14
        filas.append(panel_row)
        y += h_fila + 14
    return filas


def _dibujar_arte(draw: ImageDraw.ImageDraw, rng: random.Random, panel: tuple):
    """Arte simple (manchas oscuras, líneas de velocidad) en un panel."""
    x0, y0, x1, y1 = panel
    estilo = rng.random()
    if estilo < 0.35:
        # mancha oscura grande (panel image)
        cx = rng.randint(x0 + 30, x1 - 30)
        cy = rng.randint(y0 + 30, y1 - 30)
        r = rng.randint(40, 120)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(28, 28, 28))
        for _ in range(rng.randint(2, 5)):
            sx = rng.randint(x0 + 10, x1 - 10)
            sy = rng.randint(y0 + 10, y1 - 10)
            sr = rng.randint(8, 30)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(60, 60, 60))
    elif estilo < 0.55:
        # líneas de velocidad diagonales
        for _ in range(rng.randint(6, 14)):
            sx = rng.randint(x0 + 20, x1 - 20)
            sy = rng.randint(y0 + 20, y1 - 20)
            ln = rng.randint(60, 160)
            draw.line([(sx, sy), (sx + ln, sy + ln)], fill=(180, 180, 180), width=3)
    elif estilo < 0.7:
        # puntos de medio tono (retícula de puntos)
        for px in range(x0 + 20, x1 - 20, 12):
            for py in range(y0 + 20, y1 - 20, 12):
                if rng.random() < 0.4:
                    draw.ellipse([px, py, px + 4, py + 4], fill=(150, 150, 150))


def _texto_multilinea(draw: ImageDraw.ImageDraw, rng: random.Random,
                      font: ImageFont.FreeTypeFont, frase: str, cx: int, cy: int,
                      max_w: int):
    """Dibuja la frase centrada en (cx, cy) con ancho máximo; devuelve bbox."""
    palabras = frase.split()
    lineas: list[str] = []
    actual = ""
    for p in palabras:
        prueba = f"{actual} {p}".strip()
        if draw.textbbox((0, 0), prueba, font=font)[2] <= max_w or not actual:
            actual = prueba
        else:
            lineas.append(actual)
            actual = p
    lineas.append(actual)
    altos = [draw.textbbox((0, 0), l, font=font)[3] for l in lineas]
    alto_total = sum(altos) + 8 * (len(lineas) - 1)
    y = cy - alto_total // 2
    x0, y0, x1, y1 = 10**9, 10**9, -1, -1
    for linea, alto in zip(lineas, altos):
        bb = draw.textbbox((0, 0), linea, font=font)
        x = cx - (bb[2] - bb[0]) // 2
        draw.text((x, y), linea, font=font, fill=(20, 20, 20))
        x0 = min(x0, x + bb[0]); y0 = min(y0, y + bb[1])
        x1 = max(x1, x + bb[2]); y1 = max(y1, y + bb[3])
        y += alto + 8
    return x0, y0, x1, y1


def _generar_pagina(rng: random.Random, fuentes):
    """Una página sintética → (img BGR, labels [(cls, x0, y0, x1, y1), ...])."""
    img = Image.new("RGB", (W, H), (252, 250, 246))
    draw = ImageDraw.Draw(img)
    # ruido de papel muy sutil
    for _ in range(3000):
        draw.point((rng.randint(0, W - 1), rng.randint(0, H - 1)),
                   fill=(rng.randint(240, 250),) * 3)
    labels: list[tuple[int, int, int, int, int]] = []

    filas = _layout_paneles(rng)
    for row in filas:
        for panel in row:
            x0, y0, x1, y1 = panel
            relleno = (255, 255, 255) if rng.random() < 0.8 else (232, 232, 232)
            draw.rectangle([x0, y0, x1, y1], fill=relleno, outline=(20, 20, 20), width=4)
            _dibujar_arte(draw, rng, panel)
            reg, bold = rng.choice(fuentes)
            usar_bold = rng.random() < 0.4
            font = bold if usar_bold else reg
            tam = rng.randint(26, 42)
            font = font.font_variant(size=tam)

            # 1-2 globos con texto (clase 0)
            n_globos = rng.randint(0, 2)
            for _ in range(n_globos):
                if x1 - x0 < 330 or y1 - y0 < 240:
                    break
                gx = _ri(rng, x0 + 60, x1 - 260)
                gy = _ri(rng, y0 + 60, y1 - 220)
                gw = rng.randint(200, max(200, min(420, x1 - gx - 20)))
                gh = rng.randint(120, max(120, min(200, y1 - gy - 20)))
                # globo: elipse blanca con borde + cola
                draw.ellipse([gx, gy, gx + gw, gy + gh], fill=(255, 255, 255),
                             outline=(20, 20, 20), width=4)
                tx = rng.randint(gx - 30, gx + 20)
                ty = gy + gh + 4
                draw.polygon([(gx + gw // 2 - 12, gy + gh - 2),
                              (tx, ty), (gx + gw // 2 + 12, gy + gh - 2)],
                             fill=(255, 255, 255), outline=(20, 20, 20))
                # texto dentro del globo
                frase = " ".join(rng.sample(VOCAB, rng.randint(1, 4)))
                cx = gx + gw // 2
                cy = gy + gh // 2
                bb = _texto_multilinea(draw, rng, font, frase, cx, cy, gw - 60)
                labels.append((0, bb[0], bb[1], bb[2], bb[3]))

            # texto libre (clase 1): cartela pequeña / onomatopeya
            if rng.random() < 0.45 and x1 - x0 > 240:
                frase = " ".join(rng.sample(VOCAB, rng.randint(1, 3)))
                f2 = font.font_variant(size=rng.randint(30, 56))
                cx = _ri(rng, x0 + 90, x1 - 90)
                cy = _ri(rng, y0 + 90, y1 - 90)
                bb = _texto_multilinea(draw, rng, f2, frase, cx, cy, x1 - x0 - 120)
                labels.append((1, bb[0], bb[1], bb[2], bb[3]))

    # título grande (clase 1) arriba en ~40% de páginas
    if rng.random() < 0.4:
        try:
            titulo_font = ImageFont.truetype(str(_IMPACT), 64)
        except OSError:
            titulo_font = fuentes[0][1].font_variant(size=64)
        frase = " ".join(rng.sample(VOCAB, rng.randint(2, 4)))
        bb = _texto_multilinea(draw, rng, titulo_font, frase, W // 2, 90, W - 400)
        labels.append((1, bb[0], bb[1], bb[2], bb[3]))

    # clip a la página
    labels = [(c, max(0, a), max(0, b), min(W, d), min(H, e))
              for (c, a, b, d, e) in labels]
    labels = [(c, a, b, d, e) for (c, a, b, d, e) in labels
              if d - a >= 20 and e - b >= 10]
    img_bgr = np.array(img)[:, :, ::-1]  # RGB → BGR
    return img_bgr, labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Páginas de manga sintéticas con etiquetas YOLO exactas.")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "train_data" / "synth"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    lab_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    fuentes = _cargar_fuentes()

    import cv2

    n_bubble = n_free = 0
    for i in range(args.n):
        img, labels = _generar_pagina(rng, fuentes)
        nombre = f"syn_{i:04d}"
        cv2.imwrite(str(img_dir / f"{nombre}.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        h, w = img.shape[:2]
        with open(lab_dir / f"{nombre}.txt", "w", encoding="utf-8") as f:
            for cls, x0, y0, x1, y1 in labels:
                cx = (x0 + x1) / 2 / w
                cy = (y0 + y1) / 2 / h
                bw = (x1 - x0) / w
                bh = (y1 - y0) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n_bubble += sum(1 for l in labels if l[0] == 0)
        n_free += sum(1 for l in labels if l[0] == 1)
        if (i + 1) % 25 == 0:
            print(f"[synth] {i + 1}/{args.n} páginas...")

    print(f"[synth] ✅ {args.n} páginas → {out} "
          f"({n_bubble} text_bubble + {n_free} text_free etiquetas)")
    print("[synth]   entrena con: "
          "env/Scripts/python.exe tools/entrenar_detector.py --extra-data "
          f"{out}")


if __name__ == "__main__":
    main()
