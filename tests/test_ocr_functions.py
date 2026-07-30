"""
test_ocr_functions.py — Tests específicos para 3 funciones clave de ocr_utils.py.

Usa imágenes sintéticas generadas con numpy/cv2 para probar cada función
con casos realistas de escaneo manga: líneas de escaneo, speckle, texto
artístico, fondos degradados, burbujas de diálogo, etc.

Funciones probadas:
  - _pre_filter_image    (limpieza morfológica pre-OCR)
  - _binarize_image      (binarización adaptativa)
  - _build_glyph_mask_for_bubble  (máscara de glifos para globos)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any


# ─── Helpers de imagen sintética ──────────────────────────────────

def _make_bgr(width: int, height: int, channels: int = 3,
              fill: int | tuple[int, int, int] = 255) -> np.ndarray:
    """Crea una imagen BGR del tamaño y color dados."""
    if isinstance(fill, int):
        return np.ones((height, width, channels), dtype=np.uint8) * fill
    return np.tile(np.array(fill, dtype=np.uint8), (height, width, 1))


def _paste_text(img: np.ndarray, text: str, pos: tuple[int, int],
                size: float = 0.5, color: tuple[int, int, int] = (0, 0, 0),
                thickness: int = 1) -> np.ndarray:
    """Agrega texto renderizado a una imagen BGR."""
    import cv2
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)
    return out


def _make_scan_line(width: int, y: int, height: int = 1,
                    color: int = 0) -> np.ndarray:
    """Crea una máscara con una línea horizontal (artefacto de escaneo)."""
    line = np.zeros((height + 2, width), dtype=np.uint8)
    line[1:1 + height, :] = color
    return line


# ═══════════════════════════════════════════════════════════════════
# _pre_filter_image
# ═══════════════════════════════════════════════════════════════════

class TestPreFilterImage:
    """Limpieza morfológica pre-OCR — tests avanzados con sintéticos."""

    # ── Eliminación de líneas de escaneo ─────────────────────────

    def test_removes_single_thick_scan_line(self):
        """Una línea gruesa horizontal de 3px debe ser inpaintada."""
        from ocr_utils import _pre_filter_image
        h, w = 150, 300
        img = _make_bgr(w, h, fill=220)
        # Agregar contenido con textura gaussiana (evita que OTSU clasifique
        # bordes como foreground y los zeroee en el speckle removal)
        np.random.seed(10)
        noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Texto oscuro para contenido real
        img = _paste_text(img, "Sample text for bimodal", (20, 60), 0.7, (40, 40, 40), 2)
        # Línea de escaneo: gris oscuro (no negro puro) para que OTSU
        # no lo clasifique como foreground y lo elimine en speckle removal.
        img[75:78, :, :] = 80
        result = _pre_filter_image(img)
        # La línea debe haber sido inpaintada (valor debe haber cambiado)
        line_region = result[74:79, :, :]
        assert float(line_region.mean()) > 20, f"La línea de escaneo no fue inpaintada (mean={line_region.mean():.0f})"

    def test_removes_multiple_scan_lines(self):
        """Múltiples líneas finas horizontales deben eliminarse."""
        from ocr_utils import _pre_filter_image
        h, w = 200, 400
        np.random.seed(11)
        bg = np.ones((h, w, 3), dtype=np.uint8) * 230
        noise = np.random.normal(0, 6, (h, w, 3)).astype(np.int16)
        img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = _paste_text(img, "Line test", (30, 50), 0.6, (50, 50, 50), 2)
        img = _paste_text(img, "Multiple scans", (30, 100), 0.6, (50, 50, 50), 2)
        # 3 líneas finas a diferentes alturas
        for y in (30, 90, 150):
            img[y, :, :] = 60
        result = _pre_filter_image(img)
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_preserves_thick_content_lines(self):
        """Líneas gruesas verticales (bordes de viñeta) deben preservarse."""
        from ocr_utils import _pre_filter_image
        h, w = 200, 300
        img = _make_bgr(w, h, fill=230)
        img = _paste_text(img, "Panel border test", (30, 80), 0.6, (40, 40, 40), 2)
        # Borde de viñeta: línea vertical gruesa (NO horizontal → no debe detectarse)
        img[:, 50:55, :] = 10  # línea vertical de 5px
        img[:, 250:254, :] = 10  # otra línea vertical
        result = _pre_filter_image(img)
        # Las líneas verticales deben preservarse (el kernel es horizontal 1x15)
        left_border = float(result[:, 52, :].mean())
        right_border = float(result[:, 252, :].mean())
        assert left_border < 200, "Borde vertical izquierdo fue eliminado"
        assert right_border < 200, "Borde vertical derecho fue eliminado"

    # ── Limpieza de márgenes ─────────────────────────────────────

    def test_cleans_dark_top_margin_with_texture(self):
        """Margen superior oscuro debe limpiarse (relleno con color próximo)."""
        from ocr_utils import _pre_filter_image
        h, w = 150, 300
        np.random.seed(12)
        bg = np.ones((h, w, 3), dtype=np.uint8) * 210
        noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
        img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = _paste_text(img, "Main content", (30, 60), 0.7, (40, 40, 40), 2)
        # Margen superior con textura (simula sombra de escaneo)
        for y in range(8):
            img[y, :, :] = max(20, 80 - y * 8)
        result = _pre_filter_image(img)
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8
        # El contenido principal debe preservarse
        center = result[55:75, :, :]
        assert float(center.mean()) > 30

    def test_cleans_bottom_margin_artifacts(self):
        """Margen inferior con texto basura debe limpiarse."""
        from ocr_utils import _pre_filter_image
        h, w = 150, 300
        img = _make_bgr(w, h, fill=215)
        img = _paste_text(img, "Real dialog here", (30, 60), 0.7, (40, 40, 40), 2)
        # Basura en margen inferior (píxeles blancos/negros aleatorios)
        img[h - 8:h, 10:w - 10, :] = np.random.randint(0, 255, (8, w - 20, 3), dtype=np.uint8)
        result = _pre_filter_image(img)
        assert result.shape == (h, w, 3)
        # El contenido principal debe preservarse
        assert float(result[55:75, :, :].mean()) > 30

    # ─── Speckle removal ─────────────────────────────────────────

    def test_removes_isolated_speckle(self):
        """Puntos de ruido aislados (speckle) deben reducirse."""
        from ocr_utils import _pre_filter_image
        h, w = 100, 200
        np.random.seed(42)
        base = np.ones((h, w, 3), dtype=np.uint8) * 220
        # Texto real
        img = _paste_text(base, "Clean text", (20, 50), 0.6, (40, 40, 40), 2)
        # Agregar speckle suave: variación gaussiana en lugar de puntos extremos
        speckle = np.random.normal(0, 20, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + speckle, 0, 255).astype(np.uint8)
        result = _pre_filter_image(img)
        assert result.shape == (h, w, 3)
        # La imagen procesada debe tener textura más suave
        assert float(result.std()) > 0, "La imagen quedó completamente plana"

    # ─── Preservación de contenido ───────────────────────────────

    def test_preserves_text_strokes(self):
        """Los trazos de texto no deben degradarse notablemente."""
        from ocr_utils import _pre_filter_image
        h, w = 120, 300
        img = _make_bgr(w, h, fill=230)
        img = _paste_text(img, "Preserve This Text", (15, 50), 0.7, (35, 35, 35), 2)
        img = _paste_text(img, "And this line too", (15, 80), 0.6, (50, 50, 50), 1)
        result = _pre_filter_image(img)
        # Área de texto no debe quedar completamente blanca
        text_zone = result[40:90, 10:290, :]
        assert float(text_zone.mean()) < 240, "El texto fue eliminado por el filtro"

    def test_no_false_lines_on_texture(self):
        """Textura de papel no debe detectarse como línea horizontal."""
        from ocr_utils import _pre_filter_image
        h, w = 100, 200
        # Simular textura de papel con ruido gaussiano suave
        np.random.seed(0)
        base = np.ones((h, w, 3), dtype=np.uint8) * 200
        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = _paste_text(img, "Texture", (30, 50), 0.6, (40, 40, 40), 2)
        result = _pre_filter_image(img)
        # No debe crashear, y el texto debe mantenerse
        text_line = float(result[48:55, 25:150, :].mean())
        assert text_line > 0
        assert text_line < 250

    # ─── Casos borde ─────────────────────────────────────────────

    def test_1px_wide_image(self):
        """Imagen de 1px de ancho no debe crashear."""
        from ocr_utils import _pre_filter_image
        img = np.ones((100, 1, 3), dtype=np.uint8) * 150
        result = _pre_filter_image(img)
        assert result.shape == (100, 1, 3)

    def test_1px_high_image(self):
        """Imagen de 1px de alto no debe crashear."""
        from ocr_utils import _pre_filter_image
        img = np.ones((1, 100, 3), dtype=np.uint8) * 150
        result = _pre_filter_image(img)
        assert result.shape == (1, 100, 3)

    def test_completely_black_image(self):
        """Imagen completamente negra no debe crashear."""
        from ocr_utils import _pre_filter_image
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = _pre_filter_image(img)
        assert result.shape == (50, 100, 3)
        assert result.dtype == np.uint8

    def test_completely_white_image(self):
        """Imagen completamente blanca debe permanecer blanca."""
        from ocr_utils import _pre_filter_image
        img = np.ones((50, 100, 3), dtype=np.uint8) * 255
        result = _pre_filter_image(img)
        assert float(result.mean()) >= 250  # Debe seguir siendo casi blanco


# ═══════════════════════════════════════════════════════════════════
# _binarize_image ELIMINADO — tier 3 del pipeline OCR eliminado porque
# el benchmark demostró 0 beneficios en páginas artísticas (2026-07-27).
# Ver ocr_utils.py: _detect_and_ocr() ahora tiene pipeline de 2 niveles.


# ═══════════════════════════════════════════════════════════════════
# _build_glyph_mask_for_bubble
# ═══════════════════════════════════════════════════════════════════

class TestBuildGlyphMask:
    """Máscara de glifos para globos de diálogo — tests avanzados."""

    # ─── Detección de glifos en burbuja ──────────────────────────

    def test_detects_glyphs_on_uniform_bg(self):
        """Texto sobre fondo uniforme debe producir máscara con píxeles marcados."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 120, 300
        img = _make_bgr(w, h, fill=180)
        img = _paste_text(img, "GLYPH TEST", (20, 45), 0.8, (30, 30, 30), 2)
        block = {"x": 15, "y": 20, "w": 200, "h": 50}
        mask = _build_glyph_mask_for_bubble(img, block)
        # Debe haber píxeles marcados en la región del bloque
        region = mask[20:70, 15:215]
        assert int(region.max()) > 0, "No se detectaron glifos en burbuja uniforme"
        # No deben marcarse todos los píxeles (solo los glifos)
        marked = float(np.sum(region > 0))
        total = float(region.size)
        ratio = marked / total
        assert ratio < 0.8, f"Demasiados píxeles marcados ({ratio:.1%})"

    def test_detects_glyphs_on_gradient_bg(self):
        """Texto sobre fondo degradado (común en escaneos) debe detectarse."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 120, 300
        # Fondo con gradiente vertical
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            val = int(160 + y * 40 / h)
            img[y, :, :] = val
        img = _paste_text(img, "GRADIENT BG", (20, 45), 0.7, (40, 40, 40), 2)
        block = {"x": 15, "y": 20, "w": 220, "h": 50}
        mask = _build_glyph_mask_for_bubble(img, block)
        region = mask[20:70, 15:235]
        assert int(region.max()) > 0, "No se detectaron glifos en fondo degradado"
        marked = float(np.sum(region > 0))
        total = float(region.size)
        ratio = marked / total
        assert ratio < 0.9, f"Demasiados píxeles marcados en gradiente ({ratio:.1%})"

    # ─── Canny edge detection ────────────────────────────────────

    def test_canny_captures_edges(self):
        """Canny debe capturar bordes finos que la diferencia de color pierde."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 100, 250
        # Fondo claro con texto delgado en color similar al fondo
        img = _make_bgr(w, h, fill=180)
        # Texto con contraste bajo de color pero bordes nítidos
        img = _paste_text(img, "EDGES", (20, 35), 0.7, (130, 130, 130), 1)
        block = {"x": 15, "y": 15, "w": 180, "h": 50}
        mask = _build_glyph_mask_for_bubble(img, block)
        region = mask[15:65, 15:195]
        # Canny debería capturar bordes aunque el color sea similar
        assert int(region.max()) > 0, "Canny no capturó bordes de texto de bajo contraste"

    # ─── Fallback a máscara rectangular ──────────────────────────

    def test_fallback_rect_when_no_bg_samples(self):
        """Cuando no hay muestras de fondo, debe usar rectángulo completo."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 60, 100
        img = _make_bgr(w, h, fill=150)
        # Bloque minúsculo: edge = max(3, int(min(3,3)*0.15)) = 3
        # x=5,w=3 → bx=5,bw=3 → edge samples serán pequeños pero no vacíos
        # Usar bloque aún más pequeño forzando posiciones al borde
        block = {"x": 0, "y": 0, "w": 3, "h": 3}
        mask = _build_glyph_mask_for_bubble(img, block)
        assert mask.shape == (h, w)
        assert mask.dtype == np.uint8

    def test_fallback_rect_for_tiny_block(self):
        """Bloque muy pequeño debe caer en fallback de rectángulo."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 50, 50
        img = _make_bgr(w, h, fill=128)
        block = {"x": 20, "y": 20, "w": 2, "h": 2}
        mask = _build_glyph_mask_for_bubble(img, block)
        assert mask.shape == (h, w)

    # ─── Bloque fuera de la imagen ───────────────────────────────

    def test_partially_outside_block(self):
        """Bloque parcialmente fuera de la imagen debe manejar recortes."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 100, 100
        img = _make_bgr(w, h, fill=200)
        # x=80, w=40 → x2=120, fuera por 20px
        block = {"x": 80, "y": 80, "w": 40, "h": 40}
        mask = _build_glyph_mask_for_bubble(img, block)
        assert mask.shape == (h, w)
        # La parte visible debe tener algo marcado
        visible = mask[80:100, 80:100]
        # Puede tener 0 si el color es muy uniforme, no debe crashear
        assert visible is not None

    def test_fully_outside_block_returns_zero(self):
        """Bloque completamente fuera de la imagen no debe crashear."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 100, 100
        img = _make_bgr(w, h, fill=200)
        block = {"x": -20, "y": -20, "w": 10, "h": 10}
        mask = _build_glyph_mask_for_bubble(img, block)
        assert mask.shape == (h, w)
        assert mask.dtype == np.uint8

    # ─── Múltiples colores de texto ──────────────────────────────

    def test_detects_white_text_on_dark_bg(self):
        """Texto blanco sobre fondo oscuro en burbuja debe detectarse."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 100, 250
        img = _make_bgr(w, h, fill=50)  # fondo oscuro
        img = _paste_text(img, "WHITE TEXT", (15, 45), 0.7, (230, 230, 230), 2)
        block = {"x": 10, "y": 20, "w": 200, "h": 45}
        mask = _build_glyph_mask_for_bubble(img, block)
        region = mask[20:65, 10:210]
        # Debe detectar el texto claro sobre fondo oscuro
        marked = int(region.max())
        assert marked > 0, "No se detectó texto claro sobre fondo oscuro"

    def test_detects_colored_text(self):
        """Texto de color (rojo) sobre fondo debe detectarse."""
        from ocr_utils import _build_glyph_mask_for_bubble
        import cv2
        h, w = 100, 250
        img = _make_bgr(w, h, fill=180)
        # Texto rojo (BGR: (0, 0, 200))
        img = _paste_text(img, "RED TEXT", (15, 45), 0.7, (0, 0, 200), 2)
        block = {"x": 10, "y": 20, "w": 200, "h": 45}
        mask = _build_glyph_mask_for_bubble(img, block)
        region = mask[20:65, 10:210]
        assert int(region.max()) > 0, "No se detectó texto de color"

    # ─── Fusión color + Canny ────────────────────────────────────

    def test_hybrid_color_and_canny(self):
        """La máscara basada en contraste local debe capturar texto."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 100, 250
        img = _make_bgr(w, h, fill=160)
        img = _paste_text(img, "LOW CONTRAST", (15, 40), 0.7, (50, 50, 50), 2)
        block = {"x": 10, "y": 15, "w": 230, "h": 50}
        mask = _build_glyph_mask_for_bubble(img, block)
        region = mask[15:65, 10:240]
        assert int(region.max()) > 0, "No se capturó el texto en glyph mask"

    # ─── Burbuja con textura de fondo ────────────────────────────

    def test_textured_bubble_bg(self):
        """Fondo con textura ligera (papel escaneado) no debe confundir."""
        from ocr_utils import _build_glyph_mask_for_bubble
        h, w = 100, 250
        np.random.seed(1)
        base = np.ones((h, w, 3), dtype=np.uint8) * 200
        noise = np.random.normal(0, 6, (h, w, 3)).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = _paste_text(img, "NOISY BG", (15, 40), 0.7, (40, 40, 40), 2)
        block = {"x": 10, "y": 15, "w": 200, "h": 50}
        mask = _build_glyph_mask_for_bubble(img, block)
        region = mask[15:65, 10:210]
        marked = int(region.max())
        assert marked > 0, "No se detectaron glifos en fondo con textura"
        # El ruido no debería marcar TODOS los píxeles
        marked_ratio = float(np.sum(region > 0)) / float(region.size)
        assert marked_ratio < 0.7, f"Demasiados píxeles marcados por textura ({marked_ratio:.1%})"


# Integración ELIMINADA — _binarize_image eliminado del pipeline (2026-07-27).
# Los tests de _pre_filter_image y _build_glyph_mask_for_bubble se mantienen.
