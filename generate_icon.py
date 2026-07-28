"""
generate_icon.py — Genera el icono .ico profesional para Traductor Visual Pro.

Construye el .ico manualmente para evitar limitaciones de Pillow con ICO multi-resolucion.
Cada frame se renderiza como PNG y se empaqueta en el formato ICO.

6 resoluciones: 16, 32, 48, 64, 128, 256
Diseno: fondo degradado indigo->violeta + burbuja de dialogo blanca + texto "T->"
"""
import io
import os
import struct
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFont  # type: ignore


# Colores del tema
BG_START = (79, 70, 229)
BG_END = (124, 58, 237)
BUBBLE_FILL = (255, 255, 255, 230)
SHADOW_COLOR = (0, 0, 0, 50)
TEXT_COLOR = (79, 70, 229)

SIZES = [16, 32, 48, 64, 128, 256]
OUTPUT = os.path.join(os.path.dirname(__file__), "icon.ico")


def _get_font(size: int):
    """Obtiene una fuente TrueType; fallback a None (default bitmap)."""
    for fp in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return None


def _render_frame(size: int) -> bytes:
    """Renderiza un frame y devuelve los bytes PNG."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cr = int(size * 0.2)

    # 1. Fondo degradado
    for y in range(size):
        t = y / size
        r = int(BG_START[0] + (BG_END[0] - BG_START[0]) * t)
        g = int(BG_START[1] + (BG_END[1] - BG_START[1]) * t)
        b = int(BG_START[2] + (BG_END[2] - BG_START[2]) * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # 2. Mascara rounded rect
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([(0, 0), (size, size)], radius=cr, fill=255)
    img.putalpha(mask)

    # 3. Burbuja de dialogo blanca
    margin = int(size * 0.12)
    bx0, by0 = margin, margin + int(size * 0.02)
    bx1, by1 = size - margin, size - margin - int(size * 0.08)
    br = int(size * 0.12)
    soff = max(1, int(size * 0.015))

    # Sombra
    draw.rounded_rectangle(
        [(bx0 + soff, by0 + soff), (bx1 + soff, by1 + soff)],
        radius=br, fill=SHADOW_COLOR,
    )
    # Cuerpo burbuja
    draw.rounded_rectangle(
        [(bx0, by0), (bx1, by1)], radius=br, fill=BUBBLE_FILL,
    )
    # Cola
    ts = int(size * 0.08)
    tx = bx1 - int(size * 0.2)
    ty = by1
    draw.polygon([(tx, ty), (tx + ts, ty + ts), (tx + ts * 2, ty)], fill=BUBBLE_FILL)

    # 4. Texto "T->" centrado
    fs = int(size * 0.42)
    text = "T->"
    font = _get_font(fs)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = fs, fs

    cx, cy = size // 2, size // 2 - int(size * 0.02)
    tx_pos, ty_pos = cx - tw // 2, cy - th // 2

    if size >= 32:
        draw.text((tx_pos + 1, ty_pos + 1), text, fill=(0, 0, 0, 40), font=font)
    draw.text((tx_pos, ty_pos), text, fill=TEXT_COLOR, font=font)

    # 5. Puntitos decorativos (para resoluciones >= 48)
    if size >= 48:
        dr = max(1, int(size * 0.015))
        dy = cy + int(size * 0.25)
        ds = int(size * 0.06)
        for dx in (-ds, 0, ds):
            draw.ellipse(
                [(cx + dx - dr, dy - dr), (cx + dx + dr, dy + dr)],
                fill=(79, 70, 229, 150),
            )

    # Exportar como PNG en memoria
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_ico(png_data: dict[int, bytes]) -> bytes:
    """
    Construye manualmente un archivo .ico.
    Formato ICO:
      - Header: reserved(2) + type(2) + count(2)
      - Directory entries (16 bytes c/u)
      - Image data (PNG para cada frame)
    """
    frames = sorted(png_data.keys())
    count = len(frames)

    # Calcular offsets
    header_size = 6 + count * 16
    offsets = {}
    current_offset = header_size
    for size in frames:
        offsets[size] = current_offset
        current_offset += len(png_data[size])

    # Construir header
    buf = io.BytesIO()
    buf.write(struct.pack("<HHH", 0, 1, count))  # reserved, type=1 (ICO), count

    # Construir directory entries
    for size in frames:
        data = png_data[size]
        w = size if size < 256 else 0
        h = size if size < 256 else 0
        # ICO directory entry:
        #   w(1), h(1), colors(1), reserved(1), planes(2), bpp(2), size(4), offset(4)
        buf.write(struct.pack(
            "<BBBBHHII",
            w, h, 0, 0,  # width, height, colors, reserved
            1, 32,       # planes=1, bpp=32
            len(data),   # image data size
            offsets[size],  # offset
        ))

    # Escribir datos de imagen
    for size in frames:
        buf.write(png_data[size])

    return buf.getvalue()


def generate_icon() -> None:
    """Genera el archivo .ico con todas las resoluciones."""
    print("=== Generando icono para Traductor Visual Pro ===\n")

    png_data: dict[int, bytes] = {}
    for size in SIZES:
        png_bytes = _render_frame(size)
        png_data[size] = png_bytes
        print(f"  [{size}x{size}] PNG = {len(png_bytes):,} bytes")

    ico_bytes = _build_ico(png_data)

    with open(OUTPUT, "wb") as f:
        f.write(ico_bytes)

    print(f"\n[OK] Icono guardado: {OUTPUT}")
    print(f"     Tamanio: {len(ico_bytes) / 1024:.1f} KB")
    print(f"     Resoluciones: {len(png_data)} ({', '.join(str(s) for s in SIZES)})")

    # Verificacion
    print(f"\n--- Verificacion ---")
    try:
        with Image.open(OUTPUT) as img:
            print(f"Formato: {img.format}")
            print(f"Tamanio: {img.size}")
    except Exception as e:
        print(f"Error verificando: {e}")


if __name__ == "__main__":
    generate_icon()
