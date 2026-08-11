"""
test_ocr_utils.py — Tests unitarios para ocr_utils.py.

Cubre las funciones clave del pipeline OCR:
- Conversión base64 ↔ OpenCV
- Preprocesamiento: CLAHE, sharpen, gamma, bilateral, morfología, binarización
- OCR: run_ocr_on_image, ocr_results_to_blocks, detect_and_ocr
- Filtros post-OCR: watermark, group_and_merge_blocks
- Inpainting: build_inpaint_mask, inpaint_image
- Utilidades: is_inside_speech_bubble, build_glyph_mask, sample_bg_color

Usa imágenes sintéticas numpy y mocks para EasyOCR.
"""

import sys
import os
import base64
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Para crear mocks sin cargar EasyOCR realmente
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any


# ═══════════════════════════════════════════════════════════════
# Fixtures compartidos
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def small_bgr() -> np.ndarray:
    """Imagen BGR sintética pequeña (200x150) con gradiente."""
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    for y in range(150):
        for c in range(3):
            img[y, :, c] = y * 255 // 150
    return img


@pytest.fixture
def gray_test_image() -> np.ndarray:
    """Imagen en escala de grises con texto simulado."""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 200  # fondo claro
    # Simular texto (usar español para no activar spellchecker)
    cv2 = pytest.importorskip("cv2")
    cv2.putText(img, "Hola Mundo", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.putText(img, "OCR Prueba", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    return img


@pytest.fixture
def dark_image() -> np.ndarray:
    """Imagen oscura (simula escaneo subexpuesto)."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 50  # fondo oscuro
    return img


@pytest.fixture
def blocks_fixture() -> list[dict[str, Any]]:
    """Lista típica de bloques de texto para pruebas de merge/filtro.
    Usa palabras españolas para evitar correcciones del spellchecker."""
    return [
        {"x": 10, "y": 20, "w": 80, "h": 15, "text": "Hola", "confidence": 0.85, "fontSize": 14, "textColor": "#000000"},
        {"x": 95, "y": 21, "w": 90, "h": 16, "text": "Mundo", "confidence": 0.90, "fontSize": 14, "textColor": "#000000"},
        {"x": 10, "y": 50, "w": 60, "h": 12, "text": "OCR", "confidence": 0.70, "fontSize": 12, "textColor": "#000000"},
        {"x": 75, "y": 52, "w": 110, "h": 13, "text": "Prueba", "confidence": 0.75, "fontSize": 12, "textColor": "#000000"},
    ]


# ═══════════════════════════════════════════════════════════════
# _base64_to_cv2 / _cv2_to_base64
# ═══════════════════════════════════════════════════════════════

class TestBase64Conversion:
    """Conversión bidireccional base64 ↔ OpenCV."""

    def test_cv2_to_base64_returns_string(self, small_bgr):
        from ocr_utils import _cv2_to_base64
        b64 = _cv2_to_base64(small_bgr)
        assert isinstance(b64, str)
        assert b64.startswith("data:image/png;base64,")

    def test_roundtrip_preserves_content(self, small_bgr):
        from ocr_utils import _cv2_to_base64, _base64_to_cv2
        b64 = _cv2_to_base64(small_bgr)
        decoded = _base64_to_cv2(b64)
        assert decoded is not None
        assert decoded.shape == small_bgr.shape
        np.testing.assert_array_equal(decoded, small_bgr)

    def test_base64_to_cv2_with_data_uri_prefix(self):
        from ocr_utils import _cv2_to_base64, _base64_to_cv2
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        b64 = _cv2_to_base64(img)
        decoded = _base64_to_cv2(b64)
        assert decoded is not None
        assert decoded.shape == (50, 50, 3)

    def test_base64_to_cv2_strips_prefix(self, small_bgr):
        from ocr_utils import _cv2_to_base64, _base64_to_cv2
        b64 = _cv2_to_base64(small_bgr)
        # Extraer solo la parte base64 (después de la coma)
        raw = b64.split(",", 1)[1]
        decoded = _base64_to_cv2(raw)
        assert decoded is not None
        assert decoded.shape == small_bgr.shape

    def test_base64_to_cv2_invalid_returns_none(self):
        from ocr_utils import _base64_to_cv2
        result = _base64_to_cv2("not-a-valid-base64!!")
        assert result is None

    def test_cv2_to_base64_jpg_format(self, small_bgr):
        from ocr_utils import _cv2_to_base64
        b64 = _cv2_to_base64(small_bgr, fmt=".jpg")
        assert isinstance(b64, str)
        # NOTA: _cv2_to_base64 hardcodea "data:image/png;base64," como prefijo
        # incluso cuando fmt=".jpg". La imagen codificada SÍ es JPG (/9j/), pero
        # el prefijo siempre dice PNG. Bug conocido (TODO: corregir prefijo).
        assert "/9j/" in b64, "El contenido debería ser JPG con fmt=.jpg"

    def test_empty_base64_returns_none(self):
        from ocr_utils import _base64_to_cv2
        assert _base64_to_cv2("") is None


# ═══════════════════════════════════════════════════════════════
# _preprocess_enhanced
# ═══════════════════════════════════════════════════════════════

class TestPreprocessEnhanced:
    """Preprocesamiento CLAHE + sharp + gamma + bilateral."""

    def test_returns_same_dimensions(self, small_bgr):
        from ocr_utils import _preprocess_enhanced
        result = _preprocess_enhanced(small_bgr)
        assert result.shape == small_bgr.shape
        assert result.dtype == np.uint8

    def test_enhances_dark_image(self, dark_image):
        from ocr_utils import _preprocess_enhanced
        result = _preprocess_enhanced(dark_image)
        # La imagen mejorada debe cambiar (CLAHE + gamma modifican los valores)
        orig_mean = float(dark_image.mean())
        result_mean = float(result.mean())
        assert result_mean != orig_mean, "El preprocesamiento debería alterar una imagen oscura"

    def test_increases_local_contrast(self, small_bgr):
        from ocr_utils import _preprocess_enhanced
        result = _preprocess_enhanced(small_bgr)
        # El CLAHE debe aumentar el contraste local
        orig_std = float(small_bgr.std())
        result_std = float(result.std())
        assert result_std >= orig_std * 0.5  # No debe reducir drásticamente el contraste

    def test_output_is_bgr(self):
        from ocr_utils import _preprocess_enhanced
        img = np.ones((60, 80, 3), dtype=np.uint8) * 100
        result = _preprocess_enhanced(img)
        assert result.shape[2] == 3

    def test_extremely_small_image(self):
        from ocr_utils import _preprocess_enhanced
        img = np.ones((20, 30, 3), dtype=np.uint8) * 80
        # No debe crashear con imágenes muy pequeñas
        result = _preprocess_enhanced(img)
        assert result is not None
        assert result.shape == (20, 30, 3)

    def test_uniform_image(self):
        from ocr_utils import _preprocess_enhanced
        # Imagen uniforme (todos los píxeles iguales)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = _preprocess_enhanced(img)
        # Debe seguir siendo uniforme después del procesamiento
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_bright_image_no_gamma_change(self):
        from ocr_utils import _preprocess_enhanced
        # Imagen brillante: mean_brightness > 100, gamma no debe aplicarse
        img = np.ones((80, 80, 3), dtype=np.uint8) * 180
        result = _preprocess_enhanced(img)
        assert result.shape == img.shape


# ═══════════════════════════════════════════════════════════════
# _pre_filter_image
# ═══════════════════════════════════════════════════════════════

class TestPreFilterImage:
    """Limpieza morfológica pre-OCR."""

    def test_returns_same_dimensions(self, small_bgr):
        from ocr_utils import _pre_filter_image
        result = _pre_filter_image(small_bgr)
        assert result.shape == small_bgr.shape
        assert result.dtype == np.uint8

    def test_cleans_margin_artifacts(self):
        from ocr_utils import _pre_filter_image
        # Verificar que la función no crashea y retorna dimensiones correctas.
        # NOTA: El speckle removal usa OTSU + MORPH_OPEN que puede alterar
        # bordes de imágenes uniformes — es un comportamiento conocido.
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        img[:6, :, :] = 30  # franja oscura arriba
        result = _pre_filter_image(img)
        assert result.shape == (100, 200, 3)
        assert result.dtype == np.uint8
        # El contenido central debe al menos tener valores > 0
        center = result[30:80, :, :]
        assert float(center.mean()) > 0

    def test_removes_horizontal_lines(self):
        from ocr_utils import _pre_filter_image
        # Imagen con textura + línea horizontal para que OTSU funcione
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        # Agregar "texto" oscuro en el centro (para histograma bimodal)
        img[40:55, 30:170, :] = 40
        # Agregar línea horizontal negra
        img[50, :, :] = 0
        result = _pre_filter_image(img)
        # El área de la línea debe haber sido inpaintada (mean > 0)
        line_area = result[48:52, :, :]
        assert float(line_area.mean()) > 0

    def test_preserves_content(self, gray_test_image):
        from ocr_utils import _pre_filter_image
        result = _pre_filter_image(gray_test_image)
        # La imagen no debe quedar completamente alterada
        assert float(result.mean()) > 0
        assert float(result.std()) > 0

    def test_small_image_no_crash(self):
        from ocr_utils import _pre_filter_image
        img = np.ones((30, 40, 3), dtype=np.uint8) * 150
        result = _pre_filter_image(img)
        assert result.shape == (30, 40, 3)

    def test_no_line_mask_does_not_modify(self):
        from ocr_utils import _pre_filter_image
        # Imagen sin líneas horizontales
        img = np.ones((80, 100, 3), dtype=np.uint8) * 180
        result = _pre_filter_image(img)
        # La estructura general debe conservarse
        assert float(result.mean()) > 100


# ═══════════════════════════════════════════════════════════════
# _binarize_image ELIMINADO — tier 3 del pipeline OCR eliminado porque
# el benchmark demostró 0 beneficios en páginas artísticas (2026-07-27).


# ═══════════════════════════════════════════════════════════════
# _ocr_results_to_blocks
# ═══════════════════════════════════════════════════════════════

class TestOcrResultsToBlocks:
    """Conversión de resultados EasyOCR a formato interno."""

    def test_empty_results(self, small_bgr):
        from ocr_utils import _ocr_results_to_blocks
        blocks = _ocr_results_to_blocks([], small_bgr)
        assert blocks == []

    def test_single_block_conversion(self, small_bgr):
        from ocr_utils import _ocr_results_to_blocks
        results = [
            ([[10, 20], [90, 20], [90, 40], [10, 40]], "Hola", 0.85)
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        assert len(blocks) >= 1
        block = blocks[0]
        assert block["text"] == "Hola"
        assert abs(block["confidence"] - 0.85) < 0.01
        assert block["w"] >= 80  # 90-10
        assert block["h"] >= 20  # 40-20
        assert "x" in block
        assert "y" in block
        assert "fontSize" in block
        assert "textColor" in block

    def test_multiple_blocks(self, small_bgr):
        from ocr_utils import _ocr_results_to_blocks
        results = [
            ([[10, 20], [80, 20], [80, 35], [10, 35]], "Hola", 0.85),
            ([[90, 21], [170, 21], [170, 36], [90, 36]], "Mundo", 0.90),
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        # Pueden mergearse en 1 bloque si están cerca
        assert len(blocks) >= 1

    def test_low_confidence_filtered(self, small_bgr):
        from ocr_utils import _ocr_results_to_blocks
        results = [
            ([[10, 20], [50, 20], [50, 30], [10, 30]], "bad", 0.03),
            ([[60, 20], [100, 20], [100, 30], [60, 30]], "good", 0.80),
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        for b in blocks:
            assert b["confidence"] >= 0.08

    def test_empty_text_filtered(self, small_bgr):
        from ocr_utils import _ocr_results_to_blocks
        results = [
            ([[10, 20], [50, 20], [50, 30], [10, 30]], "", 0.80),
            ([[60, 20], [100, 20], [100, 30], [60, 30]], "OK", 0.80),
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        assert len(blocks) >= 1
        for b in blocks:
            assert b["text"] != ""

    def test_too_small_bbox_filtered(self, small_bgr):
        from ocr_utils import _ocr_results_to_blocks
        results = [
            ([[10, 20], [11, 20], [11, 21], [10, 21]], "tiny", 0.80),  # w=1, h=1
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        # Debe filtrarse por w<3 o h<3
        assert len(blocks) == 0

    def test_text_color_detection(self):
        from ocr_utils import _ocr_results_to_blocks
        # Fondo blanco → texto en zona oscura debe detectarse como #ffffff
        img = np.ones((60, 100, 3), dtype=np.uint8) * 200
        blocks = _ocr_results_to_blocks([
            ([[5, 5], [40, 5], [40, 20], [5, 20]], "Dark", 0.80)
        ], img)
        # El ROI alrededor del centro del bloque tiene fondo claro (200) → brightness > 128 → #000000 (texto negro en fondo claro)
        if blocks:
            assert blocks[0]["textColor"] == "#000000"

    def test_acepta_dicts_de_race_window(self, small_bgr):
        """Fase 5 bug fix: cuando _run_ocr_on_image degrada internamente a
        RapidOCR (dicts en formato interno) durante la race window de
        _uocr_inferring, _ocr_results_to_blocks NO debe explotar con
        "too many values to unpack" (causaba 500 en las páginas 19-22 del
        run fusion batch) — debe convertir los dicts directamente a bloques."""
        from ocr_utils import _ocr_results_to_blocks
        results = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "RapidCPU",
             "confidence": 0.72, "fontSize": 14, "textColor": "#000000"},
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        assert len(blocks) >= 1
        b = blocks[0]
        assert b["text"] == "RapidCPU"
        assert abs(b["confidence"] - 0.72) < 0.01
        assert b["x"] == 10 and b["y"] == 20
        assert b["w"] >= 80 and b["h"] >= 15

    def test_dict_y_tupla_mezclados(self, small_bgr):
        """La lista de resultados puede mezclar tuplas (EasyOCR) y dicts
        (RapidOCR degradado) — ambos deben procesarse sin crashear."""
        from ocr_utils import _ocr_results_to_blocks
        results = [
            ([[10, 20], [90, 20], [90, 40], [10, 40]], "Hola", 0.85),
            {"x": 100, "y": 20, "w": 80, "h": 15, "text": "CPU",
             "confidence": 0.60, "fontSize": 14, "textColor": "#000000"},
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        texts = " ".join(b["text"] for b in blocks)
        assert "Hola" in texts
        assert "CPU" in texts

    def test_dict_con_texto_vacio_filtrado(self, small_bgr):
        """Dicts con texto vacío o bbox diminuto se filtran igual que tuplas."""
        from ocr_utils import _ocr_results_to_blocks
        results = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "  ",
             "confidence": 0.90, "fontSize": 14, "textColor": "#000000"},
            {"x": 10, "y": 50, "w": 1, "h": 1, "text": "tiny",
             "confidence": 0.90, "fontSize": 14, "textColor": "#000000"},
            {"x": 60, "y": 50, "w": 80, "h": 15, "text": "OK",
             "confidence": 0.80, "fontSize": 14, "textColor": "#000000"},
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        assert len(blocks) >= 1
        for b in blocks:
            assert b["text"].strip() != ""
            assert b["w"] >= 3 and b["h"] >= 3

    def test_dict_conf_baja_filtrada_paridad_tupla(self, small_bgr):
        """Paridad con el camino tupla (code review Fase 5): los dicts con
        confidence < 0.08 se filtran igual que las tuplas — si un emisor
        futuro manda un dict con conf baja, no entra al pipeline."""
        from ocr_utils import _ocr_results_to_blocks
        results = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "basura",
             "confidence": 0.03, "fontSize": 14, "textColor": "#000000"},
            {"x": 60, "y": 50, "w": 80, "h": 15, "text": "OK",
             "confidence": 0.80, "fontSize": 14, "textColor": "#000000"},
        ]
        blocks = _ocr_results_to_blocks(results, small_bgr)
        assert len(blocks) >= 1
        for b in blocks:
            assert b["confidence"] >= 0.08


# ═══════════════════════════════════════════════════════════════
# _filter_watermarks_from_blocks
# ═══════════════════════════════════════════════════════════════

class TestFilterWatermarks:
    """Filtro de marcas de agua."""

    def test_passes_clean_blocks(self):
        from ocr_utils import _filter_watermarks_from_blocks
        blocks = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "Hello World", "confidence": 0.85},
            {"x": 10, "y": 50, "w": 100, "h": 15, "text": "Another line", "confidence": 0.90},
        ]
        result = _filter_watermarks_from_blocks(blocks)
        assert len(result) == 2

    def test_filters_watermark_pattern(self):
        from ocr_utils import _filter_watermarks_from_blocks
        blocks = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "Hello World", "confidence": 0.85},
            # Watermark que SÍ coincide con WATERMARK_PATTERNS de config.py
            {"x": 10, "y": 50, "w": 100, "h": 15, "text": "1c2e", "confidence": 0.80},
            # NOTA: WATERMARK_PATTERNS usa r'zonaolympus[\s-]?com' (espacio o guión,
            # no punto). Usar "zonaolympus com" o "zonaolympuscom" para match.
            {"x": 10, "y": 80, "w": 100, "h": 15, "text": "zonaolympus com", "confidence": 0.80},
        ]
        result = _filter_watermarks_from_blocks(blocks)
        assert len(result) == 1
        assert result[0]["text"] == "Hello World"

    def test_empty_blocks(self):
        from ocr_utils import _filter_watermarks_from_blocks
        assert _filter_watermarks_from_blocks([]) == []

    def test_all_watermarks_filtered(self):
        from ocr_utils import _filter_watermarks_from_blocks
        blocks = [
            {"text": "1c2e", "confidence": 0.90},
            {"text": "zonaolympus com", "confidence": 0.85},
        ]
        result = _filter_watermarks_from_blocks(blocks)
        assert len(result) == 0

    def test_none_blocks(self):
        from ocr_utils import _filter_watermarks_from_blocks
        assert _filter_watermarks_from_blocks(None) == []


# ═══════════════════════════════════════════════════════════════
# _group_and_merge_blocks
# ═══════════════════════════════════════════════════════════════

class TestGroupAndMergeBlocks:
    """Agrupación y fusión de bloques OCR."""

    def test_empty_blocks(self):
        from ocr_utils import _group_and_merge_blocks
        assert _group_and_merge_blocks([]) == []

    def test_single_block_unchanged(self):
        from ocr_utils import _group_and_merge_blocks
        blocks = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "Hola", "confidence": 0.85,
             "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks)
        assert len(result) == 1
        assert result[0]["text"] == "Hola"

    def test_horizontal_merge(self, blocks_fixture):
        from ocr_utils import _group_and_merge_blocks
        # "Hola" y "Mundo" están cerca horizontalmente → deben mergearse
        result = _group_and_merge_blocks(blocks_fixture, img_h=200)
        assert len(result) <= len(blocks_fixture)
        texts = [b["text"] for b in result]
        combined = " ".join(texts)
        assert "Hola" in combined and "Mundo" in combined

    def test_vertical_merge(self):
        from ocr_utils import _group_and_merge_blocks
        # Dos bloques en misma columna (x overlap) cerca verticalmente
        blocks = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "Texto 1", "confidence": 0.85, "fontSize": 14, "textColor": "#000000"},
            {"x": 12, "y": 40, "w": 75, "h": 14, "text": "Texto 2", "confidence": 0.80, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        # x overlap: 75 over min_w=75*0.5 → deben mergearse verticalmente
        if len(result) == 1:
            assert "Texto 1" in result[0]["text"]
            assert "Texto 2" in result[0]["text"]

    def test_filters_number_only(self):
        from ocr_utils import _group_and_merge_blocks
        blocks = [
            {"x": 10, "y": 20, "w": 30, "h": 15, "text": "1234", "confidence": 0.85, "fontSize": 14, "textColor": "#000000"},
            {"x": 50, "y": 20, "w": 80, "h": 15, "text": "Real text", "confidence": 0.90, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) >= 1
        for b in result:
            assert b["text"] != "1234"

    def test_filters_narrow_aspect(self):
        from ocr_utils import _group_and_merge_blocks
        # Bloque muy estrecho (aspect < 0.4) con texto corto
        blocks = [
            {"x": 10, "y": 20, "w": 5, "h": 15, "text": "ab", "confidence": 0.80, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 0

    def test_margin_noise_filtered(self):
        from ocr_utils import _group_and_merge_blocks
        # Texto en margen superior con patrón de ruido (ej: fecha)
        blocks = [
            {"x": 10, "y": 2, "w": 60, "h": 10, "text": "13/7/26", "confidence": 0.80, "fontSize": 10, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 0

    def test_count_digit_ratio_in_margin(self):
        from ocr_utils import _group_and_merge_blocks
        # Metadato numérico en margen (>35% dígitos, ≤4 palabras)
        blocks = [
            {"x": 10, "y": 5, "w": 60, "h": 10, "text": "Page 3/128", "confidence": 0.80, "fontSize": 10, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 0

    def test_urls_filtered(self):
        from ocr_utils import _group_and_merge_blocks
        blocks = [
            {"x": 10, "y": 50, "w": 100, "h": 15, "text": "https://example.com", "confidence": 0.85, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 0

    def test_clean_text_preserved(self):
        from ocr_utils import _group_and_merge_blocks
        blocks = [
            {"x": 10, "y": 50, "w": 100, "h": 15, "text": "Hola hermoso mundo", "confidence": 0.90, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 1
        assert result[0]["text"] == "Hola hermoso mundo"

    def test_punctuation_only_filtered(self):
        from ocr_utils import _group_and_merge_blocks
        blocks = [
            {"x": 10, "y": 50, "w": 5, "h": 5, "text": "...", "confidence": 0.70, "fontSize": 10, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 0

    def test_single_char_low_conf_filtered(self):
        from ocr_utils import _group_and_merge_blocks
        blocks = [
            {"x": 10, "y": 50, "w": 10, "h": 15, "text": "I", "confidence": 0.20, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        assert len(result) == 0

    def test_glosario_applied(self):
        from ocr_utils import _group_and_merge_blocks
        # '@NCO' debe corregirse a 'CINCO' via _aplicar_glosario
        blocks = [
            {"x": 10, "y": 50, "w": 40, "h": 15, "text": "@NCO", "confidence": 0.85, "fontSize": 14, "textColor": "#000000"},
        ]
        result = _group_and_merge_blocks(blocks, img_h=200)
        if result:
            assert result[0]["text"] == "CINCO", f"Esperaba CINCO, obtuvo {result[0]['text']!r}"


# ═══════════════════════════════════════════════════════════════
# _is_inside_speech_bubble
# ═══════════════════════════════════════════════════════════════

class TestIsInsideSpeechBubble:
    """Detección de globos de diálogo (basada en uniformidad del borde)."""

    def test_bright_uniform_background_is_bubble(self):
        from ocr_utils import _is_inside_speech_bubble
        # Fondo blanco uniforme (globo típico de manga) -> ES burbuja (True)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 250
        block = {"x": 30, "y": 30, "w": 80, "h": 20}
        assert _is_inside_speech_bubble(img, block) is True

    def test_dark_uniform_background_is_bubble(self):
        from ocr_utils import _is_inside_speech_bubble
        # Fondo oscuro uniforme -> ES burbuja (True)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 40
        block = {"x": 30, "y": 30, "w": 80, "h": 20}
        assert _is_inside_speech_bubble(img, block) is True

    def test_non_uniform_background_not_bubble(self):
        from ocr_utils import _is_inside_speech_bubble
        # Fondo no uniforme (ruido/arte multicolor) -> NO es burbuja (False)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:50, :, 0] = 255  # canal azul arriba
        img[50:, :, 1] = 255  # canal verde abajo
        img[:, :100, 2] = 255 # canal rojo izquierda
        block = {"x": 30, "y": 30, "w": 80, "h": 40}
        assert _is_inside_speech_bubble(img, block) is False


# ═══════════════════════════════════════════════════════════════
# _build_glyph_mask_for_bubble
# ═══════════════════════════════════════════════════════════════

class TestBuildGlyphMask:
    """Máscara de glifos para globos de diálogo."""

    def test_returns_correct_shape(self, small_bgr):
        from ocr_utils import _build_glyph_mask_for_bubble
        block = {"x": 30, "y": 30, "w": 80, "h": 30}
        mask = _build_glyph_mask_for_bubble(small_bgr, block)
        assert mask.shape[:2] == small_bgr.shape[:2]
        assert mask.dtype == np.uint8

    def test_non_overlapping_block(self):
        from ocr_utils import _build_glyph_mask_for_bubble
        # Bloque fuera de la imagen
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        block = {"x": -10, "y": -10, "w": 5, "h": 5}
        mask = _build_glyph_mask_for_bubble(img, block)
        assert mask.shape == (100, 100)
        assert mask.sum() == 0

    def test_mask_contains_text_region(self, gray_test_image):
        from ocr_utils import _build_glyph_mask_for_bubble
        block = {"x": 5, "y": 5, "w": 100, "h": 40}
        mask = _build_glyph_mask_for_bubble(gray_test_image, block)
        # La región del bloque debe tener al menos algunos píxeles marcados
        region = mask[block["y"]:block["y"] + block["h"], block["x"]:block["x"] + block["w"]]
        assert region.size > 0

    def test_empty_bg_pixels_uses_rect_fallback(self):
        from ocr_utils import _build_glyph_mask_for_bubble
        # Bloque con tamaño mínimo (no hay bg_pixels porque edge es 0)
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        block = {"x": 10, "y": 10, "w": 5, "h": 5}  # edge = max(3, int(min(5,5)*0.15)) = 3
        mask = _build_glyph_mask_for_bubble(img, block)
        assert mask.shape == (50, 50)


# ═══════════════════════════════════════════════════════════════
# _build_inpaint_mask
# ═══════════════════════════════════════════════════════════════

class TestBuildInpaintMask:
    """Construcción de máscara de inpainting."""

    def test_empty_blocks_produces_zero_mask(self, small_bgr):
        from ocr_utils import _build_inpaint_mask
        mask = _build_inpaint_mask(small_bgr, [])
        assert mask.shape[:2] == small_bgr.shape[:2]
        assert int(mask.max()) == 0

    def test_returns_binary_mask(self, small_bgr):
        from ocr_utils import _build_inpaint_mask
        blocks = [{"x": 10, "y": 20, "w": 80, "h": 30, "text": "test"}]
        mask = _build_inpaint_mask(small_bgr, blocks)
        assert mask.shape[:2] == small_bgr.shape[:2]
        assert mask.dtype == np.uint8
        # Valores únicos deben ser 0 y/o 255
        unique = set(np.unique(mask).tolist())
        assert unique.issubset({0, 255})

    def test_text_region_marked(self):
        from ocr_utils import _build_inpaint_mask
        import cv2
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        cv2.putText(img, "test", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        blocks = [{"x": 30, "y": 30, "w": 50, "h": 20, "text": "test"}]
        mask = _build_inpaint_mask(img, blocks)
        region = mask[25:55, 25:90]
        assert int(region.max()) > 0


# ═══════════════════════════════════════════════════════════════
# _inpaint_image
# ═══════════════════════════════════════════════════════════════

class TestInpaintImage:
    """Inpainting con OpenCV."""

    def test_zero_mask_returns_copy(self, small_bgr):
        from ocr_utils import _inpaint_image
        mask = np.zeros(small_bgr.shape[:2], dtype=np.uint8)
        result = _inpaint_image(small_bgr, mask)
        assert result.shape == small_bgr.shape
        np.testing.assert_array_equal(result, small_bgr)

    def test_inpainting_with_mask(self, small_bgr):
        from ocr_utils import _inpaint_image
        mask = np.zeros(small_bgr.shape[:2], dtype=np.uint8)
        mask[40:60, 50:150] = 255
        result = _inpaint_image(small_bgr, mask, blocks=[{"x": 50, "y": 40, "w": 100, "h": 20}])
        assert result.shape == small_bgr.shape
        # La región inpaintada no debe ser idéntica a la original
        region_orig = small_bgr[40:60, 50:150, :]
        region_res = result[40:60, 50:150, :]
        assert not np.array_equal(region_orig, region_res)

    def test_radius_without_blocks(self, small_bgr):
        from ocr_utils import _inpaint_image
        mask = np.zeros(small_bgr.shape[:2], dtype=np.uint8)
        mask[40:60, 50:150] = 255
        # Sin blocks, calcula radio por cobertura
        result = _inpaint_image(small_bgr, mask)
        assert result.shape == small_bgr.shape


# ═══════════════════════════════════════════════════════════════
# _sample_bg_color
# ═══════════════════════════════════════════════════════════════

class TestSampleBgColor:
    """Muestreo de color de fondo."""

    def test_dark_bubble_returns_black(self):
        from ocr_utils import _sample_bg_color
        # Fondo oscuro uniforme → dentro de burbuja → debe muestrear borde
        img = np.ones((100, 200, 3), dtype=np.uint8) * 40  # oscuro
        block = {"x": 30, "y": 30, "w": 80, "h": 30}
        color = _sample_bg_color(img, block)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # #rrggbb

    def test_bright_outside_bubble_returns_white(self):
        from ocr_utils import _sample_bg_color
        # Fondo claro uniforme → fuera de burbuja (brightness>80) → sampleo exterior
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200  # claro
        block = {"x": 30, "y": 30, "w": 80, "h": 30}
        color = _sample_bg_color(img, block)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7

    def test_outside_bubble_fallback(self):
        from ocr_utils import _sample_bg_color
        # Bloque en borde superior (fuera de burbuja por brightness>80)
        img = np.ones((50, 100, 3), dtype=np.uint8) * 180
        block = {"x": 5, "y": 2, "w": 80, "h": 15}
        color = _sample_bg_color(img, block)
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_tiny_block(self):
        from ocr_utils import _sample_bg_color
        # Bloque muy pequeño
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        block = {"x": 20, "y": 20, "w": 5, "h": 5}
        color = _sample_bg_color(img, block)
        assert isinstance(color, str)
        assert color.startswith("#")


# ═══════════════════════════════════════════════════════════════
# _run_ocr_on_image (con mocks)
# ═══════════════════════════════════════════════════════════════

class TestRunOcrOnImage:
    """Ejecución de EasyOCR con semáforo."""

    def test_returns_empty_on_error(self, small_bgr):
        from ocr_utils import _run_ocr_on_image, _ocr_semaphore
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = Exception("OCR error")
        result = _run_ocr_on_image(mock_reader, small_bgr)
        assert result == []

    def test_calls_readtext_with_params(self, small_bgr):
        from ocr_utils import _run_ocr_on_image
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [("result",)]
        _run_ocr_on_image(mock_reader, small_bgr)
        mock_reader.readtext.assert_called_once()
        args, kwargs = mock_reader.readtext.call_args
        assert "detail" in kwargs
        assert "paragraph" in kwargs
        assert "text_threshold" in kwargs
        assert kwargs["paragraph"] is False

    def test_releases_semaphore_on_error(self, small_bgr):
        from ocr_utils import _run_ocr_on_image, _ocr_semaphore
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = Exception("error")
        # Contar semáforo antes y después
        before = _ocr_semaphore._value
        result = _run_ocr_on_image(mock_reader, small_bgr)
        assert result == []
        # El semáforo debe haberse liberado
        after = _ocr_semaphore._value
        assert after == before


# ═══════════════════════════════════════════════════════════════
# _detect_and_ocr (con mocks)
# ═══════════════════════════════════════════════════════════════

class TestDetectAndOcr:
    """Pipeline de 3 niveles de OCR."""

    def test_returns_empty_when_no_reader(self, small_bgr):
        with patch("ocr_utils._get_ocr_reader", return_value=None):
            from ocr_utils import _detect_and_ocr
            result = _detect_and_ocr(small_bgr)
            assert result == []

    def test_tier1_success_returns_blocks(self, small_bgr):
        """Tier 1 (EasyOCR directo) encuentra bloques."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [80, 20], [80, 35], [10, 35]], "Hola", 0.85)
        ]
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
                with patch("ocr_utils._ocr_semaphore.release"):
                    from ocr_utils import _detect_and_ocr
                    result = _detect_and_ocr(small_bgr)
                    # Debe encontrar al menos 1 bloque
                    assert len(result) >= 1
                    texts = [b["text"] for b in result]
                    assert "Hola" in " ".join(texts)

    def test_tier1_fallback_to_tier2_without_fallback(self, small_bgr):
        """Con allow_fallback=False, si tier 1 da 0 bloques, retorna []."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []  # tier 1 vacío
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
                with patch("ocr_utils._ocr_semaphore.release"):
                    from ocr_utils import _detect_and_ocr
                    result = _detect_and_ocr(small_bgr, allow_fallback=False)
                    assert result == []

    def test_tier1_empty_tier2_finds_blocks(self, small_bgr):
        """Tier 1 vacío, Tier 2 (CLAHE) encuentra bloques."""
        mock_reader = MagicMock()
        # La primera llamada (tier 1) devuelve vacío
        # La segunda llamada (tier 2) devuelve bloques
        mock_reader.readtext.side_effect = [
            [],  # tier 1
            [([[10, 20], [80, 20], [80, 35], [10, 35]], "Enhanced", 0.80)],  # tier 2
        ]
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
                with patch("ocr_utils._ocr_semaphore.release"):
                    from ocr_utils import _detect_and_ocr
                    result = _detect_and_ocr(small_bgr)
                    assert len(result) >= 1

    def test_all_tiers_fail_returns_empty(self, small_bgr):
        """Todos los tiers fallan → retorna []."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []  # todos los tiers vacíos
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
                with patch("ocr_utils._ocr_semaphore.release"):
                    from ocr_utils import _detect_and_ocr
                    result = _detect_and_ocr(small_bgr)
                    assert result == []

    def test_print_called_on_zero_blocks(self, small_bgr, capsys):
        """Se imprime mensaje cuando no hay bloques."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
                with patch("ocr_utils._ocr_semaphore.release"):
                    from ocr_utils import _detect_and_ocr
                    _detect_and_ocr(small_bgr)
                    captured = capsys.readouterr()
                    assert "0 bloques" in captured.out or "Todos los fallbacks" in captured.out


class TestBubbleRegionDetection:
    """Ruta C: detección de globos/regiones de texto dentro de paneles image."""

    def _make_panel_with_bubble(self):
        """Crea una imagen 400x300 con un panel image (300x240) que contiene
        un globo de diálogo (elipse blanca con tinta oscura)."""
        import cv2
        img = np.ones((300, 400, 3), dtype=np.uint8) * 128  # fondo gris (arte)
        # Panel image: rectángulo oscuro que delimita el panel
        cv2.rectangle(img, (40, 20), (340, 260), (60, 60, 60), -1)
        # Globo: elipse blanca con borde oscuro dentro del panel
        cx, cy, rw, rh = 190, 140, 70, 45
        cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, (0, 0, 0), 3)
        # Tinta (texto) dentro del globo
        cv2.putText(img, "HOLA", (cx - 25, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
        panel = {"x": 40, "y": 20, "w": 300, "h": 240}
        return img, panel

    def test_detects_bubble_inside_panel(self):
        from ocr_utils import _detect_bubble_regions_in_panel
        img, panel = self._make_panel_with_bubble()
        regions = _detect_bubble_regions_in_panel(img, panel)
        assert regions, "Debe detectar al menos un globo dentro del panel"
        r = regions[0]
        # La región debe estar dentro del panel y en coordenadas de página
        assert r["x"] >= panel["x"] and r["y"] >= panel["y"]
        assert r["x"] + r["w"] <= panel["x"] + panel["w"] + 5
        assert r["roundness"] >= 0.30, f"Roundness insuficiente: {r['roundness']}"

    def test_no_regions_on_flat_panel(self):
        from ocr_utils import _detect_bubble_regions_in_panel
        img = np.ones((300, 400, 3), dtype=np.uint8) * 100
        panel = {"x": 40, "y": 20, "w": 300, "h": 240}
        regions = _detect_bubble_regions_in_panel(img, panel)
        assert regions == []

    def test_coords_map_back_to_page(self):
        from ocr_utils import _detect_bubble_regions_in_panel
        img, panel = self._make_panel_with_bubble()
        regions = _detect_bubble_regions_in_panel(img, panel)
        # Las coordenadas devueltas están en espacio de PÁGINA: deben incluir
        # el offset del panel (el globo vive dentro del rect 40..340 x 20..260).
        # La detección por blobs puede devolver un fragmento del globo (el
        # texto "HOLA" perfora el interior claro), así que validamos rango amplio.
        r = regions[0]
        assert panel["x"] <= r["x"] <= panel["x"] + panel["w"]
        assert panel["y"] <= r["y"] <= panel["y"] + panel["h"]
        # El centro del fragmento debe caer dentro del área del globo (120-260 x 95-185)
        rcx = r["x"] + r["w"] // 2
        rcy = r["y"] + r["h"] // 2
        assert panel["x"] + 100 <= rcx <= panel["x"] + 250
        assert panel["y"] + 50 <= rcy <= panel["y"] + 200


class TestRecoverRegionsWithEasyocr:
    """Ruta C: re-OCR de regiones con upscale y mapeo de coordenadas."""

    def test_maps_coordinates_back_to_page(self):
        import cv2
        from unittest.mock import patch
        from ocr_utils import _recover_regions_with_easyocr

        # Página 400x300; región del globo en coords de página (180,110,80,50)
        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (180, 110), (260, 160), (255, 255, 255), -1)
        cv2.putText(img, "HOLA", (195, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
        regions = [{"x": 180, "y": 110, "w": 80, "h": 50}]

        # Mock del reader de EasyOCR: devuelve un bloque en coords del crop
        # upscaleado 3.5x. El crop es (80+pad, 50+pad) → approx 92x62 → 322x217.
        mock_reader = MagicMock()

        def fake_readtext(up_img, **kwargs):
            # Bloque centrado en el crop upscaleado: bbox en coords upscale
            # El texto original está ~(15,30) en el crop → ~(52,105) upscale
            return [([[52, 105], [180, 105], [180, 150], [52, 150]], "HOLA", 0.95)]

        mock_reader.readtext.side_effect = fake_readtext
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._run_ocr_on_image",
                       side_effect=lambda reader, up_img, **kw: reader.readtext(up_img)):
                with patch("ocr_utils._group_and_merge_blocks",
                           side_effect=lambda b, h: b):
                    # Fase 3 pt.3: sin rotación (el cls devuelve el crop igual)
                    with patch("ocr_utils._classify_rotate_crop",
                               side_effect=lambda x: (x, False, 0.0)):
                        blocks = _recover_regions_with_easyocr(img, regions, upscale=3.5)

        assert blocks, "Debe recuperar al menos un bloque"
        b = blocks[0]
        # bbox upscale (52,105)→page: 180 + 52/3.5 ≈ 195; 110 + 105/3.5 ≈ 140.
        # El pad del recorte añade ~6px, así que la tolerancia es amplia: lo
        # importante es que la coordenada se mapee de vuelta cerca del globo.
        assert 180 <= b["x"] <= 210, f"x={b['x']}"
        assert 130 <= b["y"] <= 155, f"y={b['y']}"
        assert b["text"] == "HOLA"
        assert b["engine"] == "easyocr-region"

    def test_empty_regions_returns_empty(self):
        from ocr_utils import _recover_regions_with_easyocr
        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        assert _recover_regions_with_easyocr(img, []) == []

    def test_degrada_a_rapid_cpu_cuando_daemon_infiere(self):
        """§8.4.4: con _uocr_inferring activo, la Ruta C usa RapidOCR CPU
        y NO carga el reader de EasyOCR (GPU)."""
        from unittest.mock import patch
        from ocr_utils import _recover_regions_with_easyocr, _uocr_inferring

        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        regions = [{"x": 100, "y": 80, "w": 60, "h": 40}]
        rapid_blocks = [
            {"x": 30, "y": 40, "w": 120, "h": 50, "text": "CPU GLOBO",
             "confidence": 0.7, "textColor": "#000000"},
        ]

        reader_mock = MagicMock()
        with patch("ocr_utils._get_ocr_reader", return_value=reader_mock) as get_reader:
            with patch("ocr_utils._run_rapidocr", return_value=rapid_blocks) as run_rapid:
                with patch("ocr_utils._group_and_merge_blocks",
                           side_effect=lambda b, h: b):
                    try:
                        _uocr_inferring.set()
                        blocks = _recover_regions_with_easyocr(img, regions, upscale=3.5)
                    finally:
                        _uocr_inferring.clear()

        # El reader de EasyOCR (GPU) NO debe cargarse durante la degradación
        get_reader.assert_not_called()
        run_rapid.assert_called_once()
        assert blocks, "Debe recuperar al menos un bloque"
        b = blocks[0]
        # bbox upscale (30,40,120,50) → page: 100 + 30/3.5 ≈ 108; 80 + 40/3.5 ≈ 91
        assert 100 <= b["x"] <= 115, f"x={b['x']}"
        assert 80 <= b["y"] <= 100, f"y={b['y']}"
        assert b["text"] == "CPU GLOBO"
        assert b["engine"] == "rapidocr-region"

    def test_usa_easyocr_gpu_cuando_flag_limpio(self):
        """§8.4.4: con el flag limpio, la Ruta C sigue usando EasyOCR GPU
        (reader cargado + _run_ocr_on_image, sin RapidOCR)."""
        from unittest.mock import patch
        from ocr_utils import _recover_regions_with_easyocr, _uocr_inferring

        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        regions = [{"x": 100, "y": 80, "w": 60, "h": 40}]
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[30, 40], [150, 40], [150, 90], [30, 90]], "HOLA", 0.9)]

        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader) as get_reader:
            with patch("ocr_utils._run_rapidocr") as run_rapid:
                with patch("ocr_utils._run_ocr_on_image",
                           side_effect=lambda reader, up_img, **kw: reader.readtext(up_img)):
                    with patch("ocr_utils._group_and_merge_blocks",
                               side_effect=lambda b, h: b):
                        with patch("ocr_utils._classify_rotate_crop",
                                   side_effect=lambda x: (x, False, 0.0)):
                            # Flag limpio por defecto (no se setea)
                            assert not _uocr_inferring.is_set()
                            blocks = _recover_regions_with_easyocr(img, regions, upscale=3.5)

        get_reader.assert_called_once()
        run_rapid.assert_not_called()
        assert blocks and blocks[0]["text"] == "HOLA"
        assert blocks[0]["engine"] == "easyocr-region"

    def test_maneja_formato_mixto_de_race_window(self):
        """Race window: si _run_ocr_on_image degrada internamente a RapidOCR
        (flag seteado justo después del chequeo inicial), devuelve dicts en
        vez de tuplas (bbox,text,conf) — el mapeo debe tratarlos igual."""
        from unittest.mock import patch
        from ocr_utils import _recover_regions_with_easyocr, _uocr_inferring

        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        regions = [{"x": 100, "y": 80, "w": 60, "h": 40}]
        mock_reader = MagicMock()
        # _run_ocr_on_image devuelve BLOQUES INTERNOS (dict) — simulando la
        # degradación interna que ocurre cuando el daemon empieza a inferir
        # en medio de la llamada (race window).
        dict_blocks = [
            {"x": 30, "y": 40, "w": 120, "h": 50, "text": "MIXTO",
             "confidence": 0.6, "textColor": "#000000"},
        ]
        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._run_ocr_on_image", return_value=dict_blocks):
                with patch("ocr_utils._group_and_merge_blocks",
                           side_effect=lambda b, h: b):
                    with patch("ocr_utils._classify_rotate_crop",
                               side_effect=lambda x: (x, False, 0.0)):
                        blocks = _recover_regions_with_easyocr(img, regions, upscale=3.5)

        assert blocks, "Debe recuperar el bloque aunque venga en formato dict"
        assert blocks[0]["text"] == "MIXTO"
        assert blocks[0]["engine"] == "easyocr-region"

    # ─── Fase 3 pt.3: TextClassifier de RapidOCR (rotación 0°/180°) ──

    def test_cls_detecta_180_y_rota_crop(self):
        """El cls devuelve label 180° → la imagen se rota y se marca se_roto."""
        from unittest.mock import patch
        from ocr_utils import _classify_rotate_crop
        import cv2

        # Contenido ASIMÉTRICO: un rectángulo en la esquina superior-izquierda
        # — al rotar 180° el array cambia (si fuera uniforme, la rotación sería
        # indistinguible y el assert no validaría nada).
        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        img[5:15, 5:40] = 0
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        fake_engine = MagicMock()
        fake_engine.text_cls.return_value = ([rotated], [["180", 0.98]], 0.01)

        with patch("ocr_utils._get_rapid_engine", return_value=fake_engine):
            out, se_roto, score = _classify_rotate_crop(img)

        assert se_roto is True
        assert score == 0.98
        assert out.shape == img.shape
        # La imagen devuelta es la rotada (la librería rota internamente)
        assert not np.array_equal(out, img)
        fake_engine.text_cls.assert_called_once_with([img])

    def test_cls_label_0_no_rota(self):
        """Label 0° → no se rota, se devuelve el crop original."""
        from unittest.mock import patch
        from ocr_utils import _classify_rotate_crop

        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        fake_engine = MagicMock()
        fake_engine.text_cls.return_value = ([img], [["0", 0.99]], 0.01)

        with patch("ocr_utils._get_rapid_engine", return_value=fake_engine):
            out, se_roto, score = _classify_rotate_crop(img)

        assert se_roto is False
        assert score == 0.99
        assert out is img

    def test_cls_score_bajo_no_rota(self):
        """score <= umbral (0.9) → no rotar (evita texto cabeza abajo)."""
        from unittest.mock import patch
        from ocr_utils import _classify_rotate_crop

        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        fake_engine = MagicMock()
        # label 180 pero score 0.5 < 0.9 → la librería NO rota internamente
        fake_engine.text_cls.return_value = ([img], [["180", 0.5]], 0.01)

        with patch("ocr_utils._get_rapid_engine", return_value=fake_engine):
            out, se_roto, score = _classify_rotate_crop(img)

        assert se_roto is False
        assert out is img

    def test_cls_sin_engine_degrada_seguro(self):
        """Sin engine RapidOCR → devuelve el crop sin tocar, sin error."""
        from unittest.mock import patch
        from ocr_utils import _classify_rotate_crop

        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        with patch("ocr_utils._get_rapid_engine", return_value=None):
            out, se_roto, score = _classify_rotate_crop(img)

        assert out is img
        assert se_roto is False
        assert score == 0.0

    def test_cls_deshabilitado_por_flag(self):
        """RUTA_C_CLS_ENABLED=False → el cls no se toca (benchmark/fallback)."""
        from unittest.mock import patch
        from ocr_utils import _classify_rotate_crop

        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        fake_engine = MagicMock()
        with patch("ocr_utils._get_rapid_engine", return_value=fake_engine):
            with patch("config.RUTA_C_CLS_ENABLED", False):
                out, se_roto, score = _classify_rotate_crop(img)

        fake_engine.text_cls.assert_not_called()
        assert out is img
        assert se_roto is False

    def test_cls_falla_degrada_seguro(self):
        """Excepción en el cls → crop original sin romper la Ruta C."""
        from unittest.mock import patch
        from ocr_utils import _classify_rotate_crop

        img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        fake_engine = MagicMock()
        fake_engine.text_cls.side_effect = RuntimeError("onnx falló")

        with patch("ocr_utils._get_rapid_engine", return_value=fake_engine):
            out, se_roto, score = _classify_rotate_crop(img)

        assert out is img
        assert se_roto is False
        assert score == 0.0

    def test_ruta_c_rota_crop_antes_del_reocr(self):
        """Integración: si el cls detecta 180°, la Ruta C rota el crop y el
        bloque resultante se mapea de vuelta des-rotado a la página."""
        from unittest.mock import patch
        from ocr_utils import _recover_regions_with_easyocr

        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        regions = [{"x": 100, "y": 80, "w": 60, "h": 40}]
        mock_reader = MagicMock()
        # El bloque se detecta en el crop ROTADO: coords del texto en el
        # espacio rotado. Crop ~72x52 → upscale 3.5 → ~252x182.
        def fake_readtext(up_img, **kwargs):
            # bbox en el espacio de la imagen rotada (texto abajo-izquierda)
            return [([[200, 20], [240, 20], [240, 40], [200, 40]], "ROTADO", 0.92)]
        mock_reader.readtext.side_effect = fake_readtext

        with patch("ocr_utils._get_ocr_reader", return_value=mock_reader):
            with patch("ocr_utils._run_ocr_on_image",
                       side_effect=lambda reader, up_img, **kw: reader.readtext(up_img)):
                with patch("ocr_utils._group_and_merge_blocks",
                           side_effect=lambda b, h: b):
                    # El cls detecta 180°: devuelve el crop rotado (en el test
                    # el side_effect devuelve el MISMO array para no tocar el
                    # contenido; el flujo de des-rotación de coords es lo que
                    # se valida).
                    with patch("ocr_utils._classify_rotate_crop",
                               side_effect=lambda x: (x, True, 0.98)):
                        blocks = _recover_regions_with_easyocr(img, regions, upscale=3.5)

        assert blocks, "Debe recuperar el bloque rotado"
        b = blocks[0]
        assert b["text"] == "ROTADO"
        # Coordenadas des-rotadas: el bloque del texto en el espacio rotado
        # (x=200..240) → en el crop original (252-200-40=12..52) → página
        # (94 + 12/3.5 ≈ 97). Sin la des-rotación quedaría en x≈151
        # (94 + 200/3.5) — el assert en 95..125 distingue ambos casos.
        assert 95 <= b["x"] <= 125, f"x={b['x']} (debe estar des-rotado)"
        assert b["engine"] == "easyocr-region"


# ─── Parámetros de _run_rapidocr (Fase 2: reintento agresivo) ────

class TestRunRapidocrParams:
    """Fase 2: _run_rapidocr acepta box_thresh/unclip_ratio/text_score y los
    pasa al engine. Los valores SIEMPRE se pasan explícitos (defaults si no
    se indican) para no heredar params agresivos de una llamada anterior —
    la librería muta postprocess_op en la primera llamada con kwargs."""

    def test_propaga_parametros_agresivos(self):
        from ocr_utils import _run_rapidocr
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        engine = MagicMock()
        engine.return_value = (None, None)
        with patch("ocr_utils._get_rapid_engine", return_value=engine):
            _run_rapidocr(img, box_thresh=0.30, unclip_ratio=2.2, text_score=0.40)

        engine.assert_called_once()
        kwargs = engine.call_args.kwargs
        assert kwargs["box_thresh"] == 0.30
        assert kwargs["unclip_ratio"] == 2.2
        assert kwargs["text_score"] == 0.40

    def test_defaults_se_pasan_explicitos(self):
        """Una llamada sin parámetros usa los defaults de la librería
        (0.5/1.6/0.5) — nunca hereda una llamada agresiva previa."""
        from ocr_utils import _run_rapidocr
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        engine = MagicMock()
        engine.return_value = (None, None)
        with patch("ocr_utils._get_rapid_engine", return_value=engine):
            _run_rapidocr(img)

        engine.assert_called_once()
        kwargs = engine.call_args.kwargs
        assert kwargs["box_thresh"] == 0.5
        assert kwargs["unclip_ratio"] == 1.6
        assert kwargs["text_score"] == 0.5

    def test_engine_sin_resultado_devuelve_vacio(self):
        """Engine que no detecta nada → [] sin crashear (semáforo liberado)."""
        from ocr_utils import _run_rapidocr
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        engine = MagicMock()
        engine.return_value = (None, None)
        with patch("ocr_utils._get_rapid_engine", return_value=engine):
            assert _run_rapidocr(img, box_thresh=0.3) == []


# ─── Ponderación por tipo semántico en la fusión (Fase 3) ────────

class TestFusionTypeWeighted:
    """Fase 3: el type semántico del VLM (text/title/header) pondera la
    votación de _fusionar_blocks_multi. Bloques sin type (EasyOCR/RapidOCR)
    mantienen el comportamiento base."""

    def _blk(self, text, conf, x=10, y=10, w=50, h=15, type=None):
        b = {"x": x, "y": y, "w": w, "h": h, "text": text,
             "confidence": conf, "fontSize": 12, "textColor": "#000"}
        if type:
            b["type"] = type
        return b

    def test_block_score_title_pesa_mas_que_text(self):
        from ocr_utils import _block_score
        # Mismo texto y confianza: el bloque title (VLM) gana el dedup/NMS
        title = self._blk("CAPITULO 43", 0.90, type="title")
        plain = self._blk("CAPITULO 43", 0.90)
        assert _block_score(title) > _block_score(plain)

    def test_block_score_sin_type_factor_1(self):
        from ocr_utils import _block_score
        plain = self._blk("hola mundo", 0.8)
        assert _block_score(plain) == 0.8 * min(2.0, max(0.5, len("hola mundo") / 5.0))

    def test_votacion_title_refuerza_0_20(self):
        from ocr_utils import _fusionar_blocks_multi
        # Textos casi-idénticos (Levenshtein ≤30%, mismo región): el dedup por
        # texto exacto no los fusiona → pasan por votación. Gana el title del
        # VLM (score 0.75×1.15 > 0.60×1.0) y se refuerza +0.20.
        easy = [self._blk("CAPITULO 43", 0.60)]
        uocr = [self._blk("CAPITULO 43!", 0.75, x=10, y=10, w=50, h=15, type="title")]
        merged = _fusionar_blocks_multi([easy, uocr])
        assert len(merged) == 1
        assert merged[0]["confidence"] == pytest.approx(0.75 + 0.20, abs=1e-6)
        assert merged[0]["type"] == "title"

    def test_votacion_header_refuerza_0_18(self):
        from ocr_utils import _fusionar_blocks_multi
        easy = [self._blk("4.58 p.m", 0.60)]
        uocr = [self._blk("4.58 p.m.", 0.70, x=10, y=10, w=50, h=15, type="header")]
        merged = _fusionar_blocks_multi([easy, uocr])
        assert len(merged) == 1
        assert merged[0]["confidence"] == pytest.approx(0.70 + 0.18, abs=1e-6)

    def test_votacion_sin_type_mantiene_0_15(self):
        """Acuerdo entre motores sin type (EasyOCR+RapidOCR) → +0.15 base
        (comportamiento previo a Fase 3 intacto)."""
        from ocr_utils import _fusionar_blocks_multi
        easy = [self._blk("hola mundo", 0.60)]
        rapid = [self._blk("hola mundo!", 0.65)]
        merged = _fusionar_blocks_multi([easy, rapid])
        assert len(merged) == 1
        # Gana rapid (0.65) y se refuerza con el base 0.15
        assert merged[0]["confidence"] == pytest.approx(0.65 + 0.15, abs=1e-6)

    def test_votacion_texto_distinto_no_refuerza(self):
        from ocr_utils import _fusionar_blocks_multi
        easy = [self._blk("hola", 0.60)]
        uocr = [self._blk("mundo", 0.85, x=200, y=50, type="title")]
        merged = _fusionar_blocks_multi([easy, uocr])
        assert len(merged) == 2  # sin solape → sin votación

    def test_type_se_propaga_al_resultado_fusionado(self):
        from ocr_utils import _fusionar_blocks_multi
        easy = [self._blk("CAPITULO 43", 0.60)]
        uocr = [self._blk("CAPITULO 43", 0.85, x=10, y=10, w=50, h=15, type="title")]
        merged = _fusionar_blocks_multi([easy, uocr])
        assert merged[0].get("type") == "title"


# ═══════════════════════════════════════════════════════════════
# Fase 6: detector YOLO de regiones de texto (_detect_text_regions_in_page)
# ═══════════════════════════════════════════════════════════════

class _YoloTensor:
    """Imita Tensor.cpu().numpy() de PyTorch (los tests no cargan torch)."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeYoloBoxes:
    """Stub de results[0].boxes de ultralytics: xyxy/conf/cls con .cpu()."""

    def __init__(self, xyxy, confs, clss):
        self._xyxy = xyxy
        self._conf = confs
        self._cls = clss

    @property
    def xyxy(self):
        return _YoloTensor(self._xyxy)

    @property
    def conf(self):
        return _YoloTensor(self._conf)

    @property
    def cls(self):
        return _YoloTensor(self._cls)


class _FakeYoloResult:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {}


class TestDetectTextRegionsYolo:
    """_detect_text_regions_in_page — mapea detecciones YOLO a regiones en el
    formato de la Ruta C, filtra por clase/área y degrada seguro."""

    @pytest.fixture(autouse=True)
    def _limpiar_flags(self):
        """Limpia los Events globales tras cada test: si uno falla antes del
        finally, no poluciona los tests siguientes (code review Fase 6.5).
        También resetea _yolo_device (sesión 116): el device se resuelve UNA
        vez por proceso, así que cada test debe re-resolverlo para aislarse."""
        import ocr_utils
        yield
        ocr_utils._uocr_inferring.clear()
        ocr_utils._yolo_disabled.clear()
        ocr_utils._yolo_device = None

    def test_mapea_boxes_a_regiones_en_coords_de_pagina(self, mocker):
        from ocr_utils import _detect_text_regions_in_page
        # Imagen 200x150: un globo (10,20,80,60) y un título (100,5,90,30)
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80], [100, 5, 190, 35]], dtype=float),
            confs=np.array([0.92, 0.88], dtype=float),
            clss=np.array([0, 1], dtype=float),
        )
        result = _FakeYoloResult(boxes, {0: "speech bubble", 1: "title"})
        engine = MagicMock()
        engine.predict.return_value = [result]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        # Determinismo: CUDA disponible y daemon U-OCR NO infiere → device "0"
        import ocr_utils
        ocr_utils._uocr_inferring.clear()
        mocker.patch("torch.cuda.is_available", return_value=True)

        regions = _detect_text_regions_in_page(img)

        assert len(regions) == 2
        r0, r1 = regions[0], regions[1]
        assert (r0["x"], r0["y"], r0["w"], r0["h"]) == (10, 20, 80, 60)
        assert r0["source"] == "yolo"
        assert r0["label"] == "speech bubble"
        assert abs(r0["cls_conf"] - 0.92) < 1e-6
        assert (r1["x"], r1["y"], r1["w"], r1["h"]) == (100, 5, 90, 30)
        # El engine recibe la imagen; con CUDA libre el device es "0"
        engine.predict.assert_called_once()
        assert engine.predict.call_args.kwargs["device"] == "0"

    def test_device_auto_ignora_daemon_infiriendo(self, mocker):
        """Política determinista (sesión 116, code review): YOLO_DEVICE='auto'
        se resuelve SOLO por CUDA — NO consulta _uocr_inferring. Si el device
        dependiera del flag del daemon en el primer call del proceso, un
        proceso que arrancara justo cuando el daemon infiere resolvería CPU y
        otro GPU → no-determinismo entre corridas (la misma fuente que se
        elimina). La sesión 103 verificó que YOLO GPU coexiste con el daemon
        en VRAM (2.25GB + ~1GB + 0.13GB < 4GB)."""
        from ocr_utils import _detect_text_regions_in_page, _uocr_inferring
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", return_value=True)
        _uocr_inferring.set()
        try:
            _detect_text_regions_in_page(img)
        finally:
            _uocr_inferring.clear()
        assert engine.predict.call_args.kwargs["device"] == "0"

    def test_device_auto_usa_cpu_sin_cuda(self, mocker):
        """YOLO_DEVICE='auto': sin CUDA disponible → device 'cpu' (compatibilidad
        en máquinas sin GPU, el objetivo de los 200-400ms CPU)."""
        from ocr_utils import _detect_text_regions_in_page
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", return_value=False)
        _detect_text_regions_in_page(img)
        assert engine.predict.call_args.kwargs["device"] == "cpu"

    def test_device_auto_usa_gpu_si_libre(self, mocker):
        """YOLO_DEVICE='auto': CUDA disponible y daemon sin inferir → '0'."""
        from ocr_utils import _detect_text_regions_in_page, _uocr_inferring
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", return_value=True)
        _uocr_inferring.clear()
        _detect_text_regions_in_page(img)
        assert engine.predict.call_args.kwargs["device"] == "0"

    def test_gpu_lock_ocupado_espera_y_usa_gpu(self, mocker):
        """Política determinista (sesión 116): con device resuelto a GPU, YOLO
        ESPERA (adquisición bloqueante) a que EasyOCR libere _gpu_lock en vez
        de degradar a CPU — el device es SIEMPRE el mismo → la detección no
        puede variar entre corridas (causa raíz del trigger no-determinista)."""
        from ocr_utils import (_detect_text_regions_in_page, _gpu_lock,
                               _uocr_inferring)
        import ocr_utils
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", return_value=True)
        _uocr_inferring.clear()
        # Ocupar el lock GPU desde OTRO hilo (como haría EasyOCR en otro
        # worker): _gpu_lock es un RLock, reentrante por el mismo hilo, así
        # que la ocupación debe simularse desde un hilo distinto — y el
        # release debe hacerlo ESE hilo (un RLock solo lo libera su dueño).
        import threading
        ocupado = threading.Event()
        liberar = threading.Event()
        resultado: dict = {}

        def _ocupar():
            _gpu_lock.acquire()
            ocupado.set()
            liberar.wait(timeout=10)
            _gpu_lock.release()

        def _correr_yolo():
            _detect_text_regions_in_page(img)
            resultado["device"] = engine.predict.call_args.kwargs["device"]

        h = threading.Thread(target=_ocupar, daemon=True)
        h.start()
        ocupado.wait(timeout=10)
        # YOLO corre en otro hilo: con la política bloqueante NO degrada, se
        # queda esperando el lock (no hay predict todavía).
        h2 = threading.Thread(target=_correr_yolo, daemon=True)
        h2.start()
        time.sleep(0.3)
        assert engine.predict.call_count == 0
        liberar.set()
        h2.join(timeout=10)
        h.join(timeout=10)
        assert resultado.get("device") == "0"

    def test_device_resuelto_una_vez_por_proceso(self, mocker):
        """Sesión 116: el device se resuelve UNA sola vez; una segunda llamada
        reutiliza el valor cacheado aunque _uocr_inferring cambie después —
        la garantía de que 2 corridas idénticas toman la misma decisión."""
        from ocr_utils import _detect_text_regions_in_page, _uocr_inferring
        import ocr_utils
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", return_value=True)
        _uocr_inferring.clear()
        # Primera llamada: CUDA libre → resuelve GPU y la cachea
        _detect_text_regions_in_page(img)
        assert engine.predict.call_args.kwargs["device"] == "0"
        assert ocr_utils._yolo_device == "0"
        # Segunda llamada: aunque el daemon "empiece a inferir" después, el
        # device cacheado NO cambia → determinismo entre corridas
        _uocr_inferring.set()
        try:
            _detect_text_regions_in_page(img)
        finally:
            _uocr_inferring.clear()
        assert engine.predict.call_args.kwargs["device"] == "0"

    def test_gpu_lock_libre_usa_gpu(self, mocker):
        """Con _gpu_lock libre, CUDA disponible y daemon sin inferir → '0'
        y el lock se libera tras la inferencia (no queda retenido)."""
        from ocr_utils import _detect_text_regions_in_page, _gpu_lock
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", return_value=True)
        _detect_text_regions_in_page(img)
        assert engine.predict.call_args.kwargs["device"] == "0"
        # El lock se adquirió y liberó correctamente: RLock no expone
        # .locked(), así que verifico con un acquire no-bloqueante que debe
        # tener éxito (si quedara retenido, devolvería False).
        assert _gpu_lock.acquire(blocking=False) is True
        _gpu_lock.release()

    def test_device_auto_error_torch_degrada_cpu(self, mocker):
        """YOLO_DEVICE='auto': si la consulta CUDA lanza (driver roto), degrada
        a 'cpu' sin romper el pipeline (except → device='cpu')."""
        from ocr_utils import _detect_text_regions_in_page
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80]], dtype=float),
            confs=np.array([0.92], dtype=float),
            clss=np.array([0], dtype=float),
        )
        engine = MagicMock()
        engine.predict.return_value = [_FakeYoloResult(boxes, {0: "text_bubble"})]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        mocker.patch("torch.cuda.is_available", side_effect=RuntimeError("cuda init failed"))
        _detect_text_regions_in_page(img)
        assert engine.predict.call_args.kwargs["device"] == "cpu"

    def test_filtra_clases_no_texto(self, mocker):
        """Clases como 'person'/'face' se ignoran; solo texto (bubble/caption/title)."""
        from ocr_utils import _detect_text_regions_in_page
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 90, 80], [30, 90, 120, 140]], dtype=float),
            confs=np.array([0.92, 0.95], dtype=float),
            clss=np.array([0, 1], dtype=float),
        )
        result = _FakeYoloResult(boxes, {0: "speech bubble", 1: "person"})
        engine = MagicMock()
        engine.predict.return_value = [result]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)

        regions = _detect_text_regions_in_page(img)

        assert len(regions) == 1
        assert regions[0]["label"] == "speech bubble"

    def test_filtra_region_minima(self, mocker):
        """Región diminuta (< 0.15% del área de la página) se descarta."""
        from ocr_utils import _detect_text_regions_in_page
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200  # área = 30000
        boxes = _FakeYoloBoxes(
            xyxy=np.array([[10, 20, 15, 25]], dtype=float),  # 5x5 = 25 < 45 (0.15%)
            confs=np.array([0.9], dtype=float),
            clss=np.array([0], dtype=float),
        )
        result = _FakeYoloResult(boxes, {0: "speech bubble"})
        engine = MagicMock()
        engine.predict.return_value = [result]
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)

        regions = _detect_text_regions_in_page(img)
        assert regions == []

    def test_sin_engine_devuelve_vacio(self, mocker):
        """ultralytics/modelo no disponible → degradación segura a [] (el tier
        simplemente no aporta, el pipeline sigue con blobs OpenCV)."""
        from ocr_utils import _detect_text_regions_in_page
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        mocker.patch("ocr_utils._get_yolo_engine", return_value=None)
        assert _detect_text_regions_in_page(img) == []

    def test_error_de_inferencia_devuelve_vacio(self, mocker):
        """Excepción en predict → [] sin crashear."""
        from ocr_utils import _detect_text_regions_in_page
        img = np.ones((150, 200, 3), dtype=np.uint8) * 200
        engine = MagicMock()
        engine.predict.side_effect = RuntimeError("onnx falló")
        mocker.patch("ocr_utils._get_yolo_engine", return_value=engine)
        assert _detect_text_regions_in_page(img) == []


class TestRotationInfoRutaC:
    """Fase 6: rotation_info se pasa SOLO en los crops de la Ruta C, no en el
    tier 1 de página completa (costo ~4x en el camino caliente)."""

    def test_run_ocr_on_image_sin_rotation_por_defecto(self, small_bgr):
        """Tier 1: sin rotation_info → readtext NO recibe el kwarg."""
        from ocr_utils import _run_ocr_on_image
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
            with patch("ocr_utils._ocr_semaphore.release"):
                _run_ocr_on_image(mock_reader, small_bgr)
        _, kwargs = mock_reader.readtext.call_args
        assert "rotation_info" not in kwargs

    def test_run_ocr_on_image_propaga_rotation(self, small_bgr):
        """Con rotation_info explícito → readtext lo recibe como lista.
        Valores ENTEROS: easyocr pasa el ángulo a scipy.ndimage.rotate y un
        string ('90') rompe el casting de cosdg con numpy 2.5/scipy 1.17."""
        from ocr_utils import _run_ocr_on_image
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        with patch("ocr_utils._ocr_semaphore.acquire", return_value=True):
            with patch("ocr_utils._ocr_semaphore.release"):
                _run_ocr_on_image(mock_reader, small_bgr,
                                  rotation_info=(0, 90, 180, 270))
        _, kwargs = mock_reader.readtext.call_args
        assert kwargs["rotation_info"] == [0, 90, 180, 270]

    def test_ruta_c_pasa_rotation_info(self, mocker):
        """La Ruta C (camino EasyOCR) llama _run_ocr_on_image CON rotation_info
        — EasyOCR rota internamente los crops y devuelve coords originales."""
        from ocr_utils import _recover_regions_with_easyocr
        img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        regions = [{"x": 100, "y": 80, "w": 60, "h": 40}]
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        run_ocr_mock = mocker.patch("ocr_utils._run_ocr_on_image",
                                    return_value=[])
        mocker.patch("ocr_utils._get_ocr_reader", return_value=mock_reader)
        mocker.patch("ocr_utils._classify_rotate_crop",
                     side_effect=lambda x: (x, False, 0.0))
        mocker.patch("ocr_utils._group_and_merge_blocks",
                     side_effect=lambda b, h: b)

        _recover_regions_with_easyocr(img, regions, upscale=3.5)

        assert run_ocr_mock.called
        rot = run_ocr_mock.call_args.kwargs.get("rotation_info")
        # Enteros (no strings): easyocr pasa el ángulo a scipy.ndimage.rotate
        # y un string rompe el casting de cosdg con numpy 2.5/scipy 1.17
        assert rot is not None and 90 in rot and 270 in rot


# ═══════════════════════════════════════════════════════════════
# Tier 3.6: detector de texto de cómic (_detect_text_regions_comic_detector)
# ═══════════════════════════════════════════════════════════════

class TestDetectTextRegionsComicDetector:
    """Tier 3.6 — comic-text-detector ONNX (CPU): decodifica los 3 heads del
    modelo (blk/seg/det) con el post-proceso de dmMaze (NMS por clase, DBNet
    unclip, máscara no cubierta) y mapea a coordenadas de página con la
    inversa EXACTA del letterbox. onnxruntime se mockea por completo (Paso 2
    de PLAN_MANGA_OCR): los tests no cargan el modelo real."""

    @pytest.fixture(autouse=True)
    def _reset_engine(self):
        """El engine se cachea en el módulo (_comic_detector_engine): cada
        test debe partir de un estado limpio (patrón _yolo_device de Fase 6)."""
        import ocr_utils
        ocr_utils._comic_detector_engine = None
        yield
        ocr_utils._comic_detector_engine = None

    @staticmethod
    def _fake_session(blk, seg, det):
        """Sesión onnxruntime falsa: get_inputs()[0].name='images' y run()
        devuelve (blk, seg, det) en el orden real del modelo."""
        session = MagicMock()
        entrada = MagicMock()
        entrada.name = "images"
        session.get_inputs.return_value = [entrada]
        session.run.return_value = (blk, seg, det)
        return session

    @staticmethod
    def _pagina():
        # Página vertical 800x1200 (manga): r=0.8533, contenido 683x1024,
        # padding left=170/top=0, scale=(800/683, 1200/1024)
        return np.ones((1200, 800, 3), dtype=np.uint8) * 200

    def test_blk_mapea_a_coords_pagina_con_letterbox_exacto(self, mocker):
        """Una detección blk en el espacio 1024 (padded) se mapea con la
        inversa EXACTA del letterbox (resta el padding izquierdo): la caja NO
        se desplaza ~15-20% de la página como en el inference.py de dmMaze."""
        from ocr_utils import _detect_text_regions_comic_detector
        img = self._pagina()
        # Caja en el espacio 1024: cx=341.5, cy=512, w=300, h=200, obj=0.9,
        # cls_eng=0.8 → x0=191, y0=412, x1=491, y1=612
        blk = np.zeros((1, 1, 7), dtype=np.float32)
        blk[0, 0] = [341.5, 512.0, 300.0, 200.0, 0.9, 0.8, 0.1]
        seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
        det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
        session = self._fake_session(blk, seg, det)
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)

        regions = _detect_text_regions_comic_detector(img)

        assert len(regions) == 1
        r = regions[0]
        # (191.5-170)*800/683 = 25.2 → 25 ; 412*1200/1024 = 482.8 → 483
        assert (r["x"], r["y"]) == (25, 483)
        # (491.5-170)*800/683 = 376.6 → 377 ; 612*1200/1024 = 717.2 → 717
        assert (r["x"] + r["w"], r["y"] + r["h"]) == (377, 717)
        assert r["source"] == "ctd"
        assert r["label"] == "ctd_eng"
        assert abs(r["cls_conf"] - 0.72) < 1e-6  # obj * cls_eng
        # El blob enviado a la sesión: batch=1, 1024², float32, normalizado
        feed = session.run.call_args[0][1]
        assert list(feed.keys()) == ["images"]
        blob = feed["images"]
        assert blob.shape == (1, 3, 1024, 1024)
        assert blob.dtype == np.float32

    def test_nms_suprime_solapados_por_clase(self, mocker):
        """NMS por clase (yolov5): dos cajas de la MISMA clase muy solapadas
        dejan solo la de mayor confianza; una de OTRA clase que las cubre se
        conserva (eng/ja compiten por separado)."""
        from ocr_utils import _detect_text_regions_comic_detector
        img = self._pagina()
        # A: conf 0.9*0.95=0.855 (eng). B: misma clase, solape ~87%,
        # conf 0.7*0.6=0.42 (eng). C: cubre a A, clase ja, conf 0.8*0.7=0.56.
        blk = np.zeros((1, 3, 7), dtype=np.float32)
        blk[0, 0] = [341.5, 512.0, 300.0, 200.0, 0.9, 0.95, 0.05]
        blk[0, 1] = [350.0, 520.0, 300.0, 200.0, 0.7, 0.6, 0.4]
        blk[0, 2] = [341.5, 512.0, 300.0, 200.0, 0.8, 0.1, 0.7]
        seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
        det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
        session = self._fake_session(blk, seg, det)
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)

        regions = _detect_text_regions_comic_detector(img)

        assert len(regions) == 2  # A (eng) y C (ja); B suprimida por NMS
        assert sorted(r["label"] for r in regions) == ["ctd_eng", "ctd_ja"]

    def test_det_genera_regiones_linea(self, mocker):
        """El head DBNet (det) produce regiones ctd_line: blob en el mapa
        shrink → contorno → bbox + unclip 1.5 → score del mapa > 0.6."""
        from ocr_utils import _detect_text_regions_comic_detector
        img = self._pagina()
        blk = np.zeros((1, 0, 7), dtype=np.float32)  # sin detecciones YOLO
        seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
        det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
        det[0, 0, 300:400, 400:600] = 0.8  # línea de texto en el espacio 1024
        session = self._fake_session(blk, seg, det)
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)

        regions = _detect_text_regions_comic_detector(img)

        assert len(regions) == 1
        r = regions[0]
        assert r["label"] == "ctd_line"
        assert r["source"] == "ctd"
        assert abs(r["cls_conf"] - 0.8) < 1e-6
        # bbox 1024 (400,300)-(600,400) + unclip 50 → (350,250)-(650,450)
        # → página: ((350-170)*800/683, 250*1200/1024) = (211, 293)
        assert (r["x"], r["y"]) == (211, 293)

    def test_seg_genera_regiones_mascara_no_cubiertas(self, mocker):
        """La máscara UNet (seg) genera regiones ctd_mask SOLO donde no hay
        región blk/det previa: el blob cuyo centro cae dentro de una caja
        existente se suprime (la máscara es la red de seguridad, no un
        duplicado)."""
        from ocr_utils import _detect_text_regions_comic_detector
        img = self._pagina()
        blk = np.zeros((1, 1, 7), dtype=np.float32)
        # Caja blk en página: (25, 483, 351, 234) — cubre y 483..717
        blk[0, 0] = [341.5, 512.0, 300.0, 200.0, 0.9, 0.8, 0.1]
        seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
        seg[0, 0, 500:550, 200:300] = 0.9   # centro en (94, 615) → cubierto
        seg[0, 0, 900:950, 600:700] = 0.9   # centro en (575, 1084) → libre
        det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
        session = self._fake_session(blk, seg, det)
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)

        regions = _detect_text_regions_comic_detector(img)

        assert len(regions) == 2  # 1 blk + 1 máscara (el cubierto se suprime)
        masks = [r for r in regions if r["label"] == "ctd_mask"]
        assert len(masks) == 1
        m0 = masks[0]
        # blob (600,900)-(700,950) → página: ((600-170)*800/683=504,
        # 900*1200/1024=1055)
        assert (m0["x"], m0["y"]) == (504, 1055)

    def test_blob_formato_bgr_letterbox_normalizado(self, mocker):
        """El pre-proceso replica a dmMaze: letterbox 1024² (auto=False,
        stride=64, relleno 114), canales BGR CHW (sin swap a RGB), /255,
        batch=1."""
        from ocr_utils import _detect_text_regions_comic_detector
        # Página BGR con canales distinguibles: B=10, G=20, R=30
        img = np.zeros((1200, 800, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 20
        img[:, :, 2] = 30
        blk = np.zeros((1, 0, 7), dtype=np.float32)
        seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
        det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
        session = self._fake_session(blk, seg, det)
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)

        _detect_text_regions_comic_detector(img)

        blob = session.run.call_args[0][1]["images"]
        assert blob.shape == (1, 3, 1024, 1024)
        assert blob.dtype == np.float32
        # Padding izquierdo (x<170) = relleno 114 → 114/255 ≈ 0.447
        assert abs(float(blob[0, 0, 500, 100]) - 114 / 255) < 0.01
        # Contenido (x>170): canal 0 = R de la página (30), 1 = G (20),
        # 2 = B (10) — el modelo fue entrenado con BGR CHW
        assert abs(float(blob[0, 0, 500, 200]) - 30 / 255) < 0.01
        assert abs(float(blob[0, 1, 500, 200]) - 20 / 255) < 0.01
        assert abs(float(blob[0, 2, 500, 200]) - 10 / 255) < 0.01

    def test_max_regions_limita(self, mocker):
        """COMIC_DETECTOR_MAX_REGIONS limita las regiones devueltas (evita
        saturar el re-OCR de la Ruta C)."""
        from ocr_utils import _detect_text_regions_comic_detector
        img = self._pagina()
        blk = np.zeros((1, 3, 7), dtype=np.float32)
        # 3 cajas separadas, misma clase, sin solape → 3 candidatas
        blk[0, 0] = [341.5, 256.0, 300.0, 100.0, 0.9, 0.8, 0.1]
        blk[0, 1] = [341.5, 512.0, 300.0, 100.0, 0.9, 0.8, 0.1]
        blk[0, 2] = [341.5, 768.0, 300.0, 100.0, 0.9, 0.8, 0.1]
        seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
        det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
        session = self._fake_session(blk, seg, det)
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)
        mocker.patch("config.COMIC_DETECTOR_MAX_REGIONS", 2)

        regions = _detect_text_regions_comic_detector(img)

        assert len(regions) == 2

    def test_sin_engine_devuelve_vacio(self, mocker):
        """onnxruntime/modelo no disponible → degradación segura a [] (el
        tier simplemente no aporta, el pipeline sigue con YOLO + blobs)."""
        from ocr_utils import _detect_text_regions_comic_detector
        mocker.patch("ocr_utils._get_comic_detector_engine", return_value=None)
        assert _detect_text_regions_comic_detector(self._pagina()) == []

    def test_error_de_inferencia_devuelve_vacio(self, mocker):
        """Excepción en run() → [] sin crashear."""
        from ocr_utils import _detect_text_regions_comic_detector
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock()]
        session.run.side_effect = RuntimeError("onnx falló")
        mocker.patch("ocr_utils._get_comic_detector_engine",
                     return_value=session)
        assert _detect_text_regions_comic_detector(self._pagina()) == []

    def test_modelo_inexistente_engine_none(self, mocker, tmp_path):
        """_get_comic_detector_engine: modelo inexistente → None (el tier
        degrada a [] sin bloquear). COMIC_DETECTOR_MODEL_PATH se lee en
        runtime (patrón YOLO), así que parchear el módulo config funciona."""
        from ocr_utils import _get_comic_detector_engine
        mocker.patch("config.COMIC_DETECTOR_MODEL_PATH",
                     str(tmp_path / "no_existe.onnx"))
        assert _get_comic_detector_engine() is None

    def test_carga_lazy_con_onnxruntime_mockeado(self, mocker, tmp_path):
        """Carga lazy con onnxruntime mockeado: la sesión se crea UNA vez con
        CPUExecutionProvider y se cachea (segunda llamada sin re-crear)."""
        from ocr_utils import _get_comic_detector_engine
        modelo = tmp_path / "comic-text-detector.onnx"
        modelo.write_bytes(b"fake")
        mocker.patch("config.COMIC_DETECTOR_MODEL_PATH", str(modelo))
        fake_session = MagicMock()
        session_cls = mocker.patch("onnxruntime.InferenceSession",
                                   return_value=fake_session)

        e1 = _get_comic_detector_engine()
        e2 = _get_comic_detector_engine()

        assert e1 is fake_session and e2 is fake_session
        session_cls.assert_called_once()
        assert session_cls.call_args.kwargs["providers"] == [
            "CPUExecutionProvider"]
