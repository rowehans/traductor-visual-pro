"""
test_ocr_engine.py — Tests del OCRManager (ocr_engine.py).

Prueba la orquestación de los 3 motores OCR en aislamiento:
- Trigger selectivo v4.2 (fusion): conf alta NO dispara daemon, conf baja SÍ.
- disable_uocr anula el refuerzo.
- Modo unlimited: daemon OK / fallback a EasyOCR si RuntimeError.
- Modo hybrid: pure_easyocr desactiva el tier RapidOCR.
- Ruta C: re-OCR de globos se invoca y fusiona con los bloques híbridos.

El OCRManager accede a ocr_utils.<fn> y routes.api._ocr_with_unlimited en
RUNTIME, así que los mocks parchean los módulos (no las referencias importadas).
"""
import sys
import json
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ocr_engine
from ocr_engine import OCRManager


# ─── Helpers ─────────────────────────────────────────────────────

def _make_img(h: int = 100, w: int = 100):
    """Imagen BGR simulada (objeto con shape; los mocks evitan cv2 real)."""
    return type("Img", (), {"shape": (h, w, 3)})()


def _block(text: str, conf: float, x: int = 10, y: int = 10,
           w: int = 50, h: int = 15) -> dict:
    return {"x": x, "y": y, "w": w, "h": h, "text": text,
            "confidence": conf, "fontSize": 12, "textColor": "#000"}


# ─── Trigger selectivo v4.2 ──────────────────────────────────────

class TestTrigger:
    def test_no_dispara_con_conf_alta_y_3_bloques(self, mocker):
        """v4.2: 3 bloques con conf>=0.2 → el trigger NO dispara (frontera)."""
        mgr = OCRManager()
        blocks = [_block(f"t{i}", 0.5) for i in range(3)]
        # avg_conf=0.5 >= 0.20, len=3 >= 3, sin panel grande, sin force
        assert mgr._compute_trigger(blocks, 0.5, has_big_panel=False) is False

    def test_dispara_con_menos_de_3_bloques_y_conf_baja(self, mocker):
        """v4.2: <3 bloques Y conf<0.2 → dispara refuerzo."""
        mgr = OCRManager()
        blocks = [_block("hola", 0.15), _block("mundo", 0.15)]
        assert mgr._compute_trigger(blocks, 0.15, has_big_panel=False) is True

    def test_no_dispara_con_3_bloques_aunque_conf_baja(self, mocker):
        """v4.2: con >=3 bloques, la conf baja sola NO dispara (evita 41/128)."""
        mgr = OCRManager()
        blocks = [_block(f"t{i}", 0.15) for i in range(3)]
        # len=3 (no < 3) y conf 0.15 < 0.2 pero la 1ª condición falla → no dispara
        assert mgr._compute_trigger(blocks, 0.15, has_big_panel=False) is False

    def test_dispara_con_0_bloques(self, mocker):
        mgr = OCRManager()
        assert mgr._compute_trigger([], 0.0, has_big_panel=False) is True

    def test_dispara_con_panel_image_grande(self, mocker):
        mgr = OCRManager()
        blocks = [_block(f"t{i}", 0.7) for i in range(5)]
        assert mgr._compute_trigger(blocks, 0.7, has_big_panel=True) is True

    def test_panel_grande_con_ocr_fuerte_no_dispara_vlm(self, mocker):
        """Un panel oscuro ya resuelto no debe pagar una inferencia VLM."""
        mgr = OCRManager()
        blocks = [_block(f"t{i}", 0.9) for i in range(5)]

        assert mgr._compute_trigger(blocks, 0.9, has_big_panel=True) is False

    def test_force_uocr_fuerza_el_disparo(self, mocker):
        mgr = OCRManager()
        blocks = [_block(f"t{i}", 0.9) for i in range(5)]
        assert mgr._compute_trigger(blocks, 0.9, has_big_panel=False,
                                    force_uocr=True) is True

    def test_disable_uocr_anula_todo(self, mocker):
        """disable_uocr (benchmark) anula incluso force_uocr."""
        mgr = OCRManager()
        blocks = []
        assert mgr._compute_trigger(blocks, 0.0, has_big_panel=True,
                                    force_uocr=True, disable_uocr=True) is False


# ─── Modo fusion (integración del trigger + daemon) ──────────────

class TestFusion:
    def test_run_ocr_exposes_structured_diagnostics_without_changing_contract(self, mocker):
        mgr = OCRManager()
        img = _make_img()
        blocks = [_block("hola", 0.8), _block("mundo", 0.9)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=blocks)

        result = mgr.run_ocr(img, "es", ocr_mode="easyocr")

        assert result == (blocks, "easyocr", ["easyocr"])
        diagnostics = mgr.last_diagnostics
        assert diagnostics is not None
        assert diagnostics["blocks"]["initial"] == 2
        assert diagnostics["blocks"]["final"] == 2
        assert diagnostics["engine_used"] == "easyocr"
        assert diagnostics["engines"] == ["easyocr"]
        assert diagnostics["finished"] is True
        assert "total" in diagnostics["timings"]

    def test_fusion_no_llama_daemon_con_conf_alta(self, mocker):
        """Conf alta + >=3 bloques → solo híbrido, sin U-OCR."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block(f"t{i}", 0.5) for i in range(3)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        uocr_mock.assert_not_called()
        assert blocks == hybrid
        assert engine == "fusion"
        assert engines == ["easyocr+rapid"]

    def test_fusion_llama_daemon_y_fusiona(self, mocker):
        """<3 bloques con conf baja → U-OCR + Ruta C + fusión."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        ublocks = [_block("título dorado", 0.93, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 5.0))
        # Ruta C: sin paneles ni regiones → sin re-OCR
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        # Fase 2: el reintento agresivo no aporta (daemon sí) → VLM
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        # Fusión real: 2 híbridos + 1 U-OCR sin solape espacial → 3 bloques
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert engine == "fusion"
        assert engines == ["easyocr+rapid", "unlimited"]
        texts = [b["text"] for b in blocks]
        assert "título dorado" in texts
        assert len(blocks) == 3

    def test_fusion_daemon_caido_degrada_a_hibrido(self, mocker):
        """Daemon en error (RuntimeError) → solo híbrido, sin crash."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]  # dispararía, pero daemon cae
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     side_effect=RuntimeError("daemon no listo"))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon (que cae)
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert engine == "fusion"
        assert engines == ["easyocr+rapid"]
        assert blocks == hybrid

    def test_fusion_disable_uocr_no_llama_daemon(self, mocker):
        """disable_uocr: página difícil pero sin refuerzo (benchmark)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")

        blocks, engine, engines = mgr.run_ocr(
            img, "es", ocr_mode="fusion", disable_uocr=True)

        uocr_mock.assert_not_called()
        assert engines == ["easyocr+rapid"]

    def test_fusion_reintento_rapid_agresivo_salva_sin_vlm(self, mocker):
        """Fase 2: conf baja + <3 bloques → el reintento agresivo recupera
        bloques y resuelve la página → NO se llama al daemon VLM."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        extra = [_block("título dorado", 0.85, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=extra)
        # merge = 2 híbridos + 1 extra → 3 bloques, conf avg sube a 0.38
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        uocr_mock.assert_not_called()
        assert engine == "fusion"
        assert "rapid-aggressive" in engines
        assert "unlimited" not in engines
        assert len(blocks) == 3  # bloques fusionados in-place
        assert any(b["text"] == "título dorado" for b in blocks)

    def test_fusion_reintento_rapid_no_salva_llama_vlm(self, mocker):
        """Fase 2: el reintento agresivo no resuelve la página (solo
        duplicados, conf sigue baja) → se dispara el VLM."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        ublocks = [_block("título dorado", 0.93, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        # Reintento devuelve SOLO duplicados del híbrido → no aporta
        mocker.patch("ocr_utils._run_rapidocr", return_value=list(hybrid))
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert "unlimited" in engines
        assert "rapid-aggressive" not in engines
        texts = [b["text"] for b in blocks]
        assert "título dorado" in texts

    def test_fusion_panel_grande_va_directo_a_vlm(self, mocker):
        """Fase 2: has_big_panel → el reintento agresivo NO aplica (el
        diálogo en arte solo lo lee el VLM) → daemon llamado directo."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        ublocks = [_block("título dorado", 0.93, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 5.0))
        rapid_mock = mocker.patch("ocr_utils._run_rapidocr")
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        rapid_mock.assert_not_called()
        assert "unlimited" in engines

    def test_fusion_reintento_rapid_3_bloques_conf_marginal_no_salva(self, mocker):
        """Fase 2 (frontera): el merge llega a 3 bloques pero con conf
        promedio solo marginal (0.2-0.3) → NO se considera salvado y el VLM
        se dispara (no saltarse el daemon con mejora dudosa)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        extra = [_block("título dorado", 0.55, x=200, y=50, w=80, h=20)]
        # avg = (0.15+0.15+0.55)/3 = 0.283 → entre trigger (0.2) y salvado (0.3)
        ublocks = [_block("título dorado", 0.93, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=extra)
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert "unlimited" in engines
        assert "rapid-aggressive" not in engines

    def test_fusion_force_uocr_salta_reintento_rapid(self, mocker):
        """Fase 2: force_uocr (orden explícito de VLM) salta el reintento
        agresivo y va directo al daemon."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]
        ublocks = [_block("título dorado", 0.93, x=200, y=50)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 5.0))
        rapid_mock = mocker.patch("ocr_utils._run_rapidocr")
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(
            img, "es", ocr_mode="fusion", force_uocr=True)

        rapid_mock.assert_not_called()
        assert "unlimited" in engines

    def test_reforzar_rapid_agresivo_skip_con_conf_alta(self, mocker):
        """Fase 2 (unidad): avg_conf >= RAPID_RETRY_MAX_CONF → no reintenta."""
        mgr = OCRManager()
        img = _make_img()
        blocks = [_block("a", 0.8), _block("b", 0.9)]
        rapid_mock = mocker.patch("ocr_utils._run_rapidocr")

        salvado = mgr._reforzar_con_rapid_agresivo(
            img, blocks, avg_conf=0.8, has_big_panel=False)

        assert not salvado
        rapid_mock.assert_not_called()

    def test_reforzar_rapid_agresivo_fallo_no_tumba_pagina(self, mocker):
        """Fase 2 (unidad): excepción en el reintento → degrada a VLM."""
        mgr = OCRManager()
        img = _make_img()
        blocks = [_block("a", 0.15)]
        mocker.patch("ocr_utils._preprocess_rapid",
                     side_effect=RuntimeError("cv2 falló"))

        salvado = mgr._reforzar_con_rapid_agresivo(
            img, blocks, avg_conf=0.1, has_big_panel=False)

        assert not salvado


# ─── Cache de decisiones §8.4.1 ──────────────────────────────────

class TestCacheDecisionesUOCR:
    """Cache de decisiones: páginas repetitivas con la misma firma no
    re-disparan el refuerzo U-OCR si una página anterior no recuperó nada."""

    def setup_method(self):
        OCRManager.clear_decision_cache()

    def test_firma_repetitiva_no_redispare_tras_decision_negativa(self, mocker):
        """Página 1: trigger + U-OCR sin recuperar nada → cache negativo.
        Página 2 (misma firma): NO llama al daemon.

        La detección es FUERTE (5 bloques conf 0.7 + panel image grande que
        dispara el trigger): con stats fuertes la negativa es fiable y la
        salvaguarda de detección débil (sesión 134) no aplica — comportamiento
        base del §8.4.1 intacto."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block(f"t{i}", 0.7) for i in range(5)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        # Panel image grande → el trigger dispara pese a los 5 bloques
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:abcdef12")
        # U-OCR responde PERO sin bloques útiles (0 recuperados)
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        # Fase 2: panel grande → el reintento agresivo NO aplica → daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        # Página 1: dispara (panel image grande) y no recupera
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        assert "unlimited" not in engines

        # Página 2: misma firma → el cache salta el refuerzo
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1, "No debe re-disparar el daemon"
        assert engines == ["easyocr+rapid"]

    def test_firma_con_recuperacion_se_cachea_positivo(self, mocker):
        """Plan §11 P1: si U-OCR SÍ recupera bloques, la recuperación se
        guarda en el cache POSITIVO por firma (no en el de negativas). Una
        página gemela con la misma firma y detección comparable REINYECTA la
        recuperación sin volver a llamar al daemon — el determinismo 5/5 de
        la recuperación hace seguro cachearla."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]
        ublocks = [_block("título dorado", 0.93, x=200, y=50)]
        # _detect_and_ocr devuelve una lista NUEVA por llamada (como en
        # producción) — _reforzar_con_unlimited muta blocks in-place, así que
        # devolver la misma lista en 2 llamadas contaminaría la 2ª página.
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in hybrid])
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.500:aaaa")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([dict(b) for b in ublocks], [], 5.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        assert "unlimited" in engines
        # La recuperación quedó en el cache positivo (no en el de negativas):
        with OCRManager._uocr_cache_lock:
            assert "0.500:aaaa" in OCRManager._uocr_pos_cache
            assert "0.500:aaaa" not in OCRManager._uocr_neg_cache

        # Página gemela: misma firma, detección comparable → reinyecta desde
        # el cache positivo, el daemon NO se vuelve a llamar.
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1, "El cache positivo evita re-inferir"
        # La recuperación sigue presente (se fusionó de nuevo):
        assert any("título dorado" in (b.get("text") or "") for b in blocks)

    def test_firma_distinta_si_redispare(self, mocker):
        """Firma diferente → el cache no aplica y se vuelve a llamar al daemon."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", side_effect=["0.400:aaa", "0.700:bbb"])
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        mgr.run_ocr(img, "es", ocr_mode="fusion")
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 2

    def test_cache_expira_tras_ttl(self, mocker):
        """Pasado el TTL, una firma repetitiva vuelve a disparar."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:abcdef12")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        mgr.run_ocr(img, "es", ocr_mode="fusion")  # decisión negativa registrada
        assert uocr_mock.call_count == 1

        # Simular expiración: envejecer la entrada directamente (sesión 129:
        # formato (ts, n_blocks, avg_conf); sesión 134: + contador re_disparos)
        import time
        from ocr_engine import UOCR_CACHE_TTL_S
        with OCRManager._uocr_cache_lock:
            for firma in OCRManager._uocr_neg_cache:
                OCRManager._uocr_neg_cache[firma] = (
                    time.time() - UOCR_CACHE_TTL_S - 1, 0, 0.0, 0)

        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 2, "TTL expirado → debe re-disparar"

    def test_eviction_lru_del_cache(self, mocker):
        """Con más firmas que el máximo, se evictan las más antiguas (LRU)."""
        from ocr_engine import UOCR_CACHE_MAX_ENTRIES
        mgr = OCRManager()
        # Llenar el cache por encima del límite
        for i in range(UOCR_CACHE_MAX_ENTRIES + 5):
            mgr._registrar_decision_negativa(f"firma_{i}")
        with OCRManager._uocr_cache_lock:
            n = len(OCRManager._uocr_neg_cache)
        assert n <= UOCR_CACHE_MAX_ENTRIES

    def test_force_uocr_ignora_cache(self, mocker):
        """force_uocr (debug explícito) ignora el cache negativo."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:abcdef12")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        # Primera llamada: decisión negativa registrada
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        # force_uocr=true ignora el cache
        mgr.run_ocr(img, "es", ocr_mode="fusion", force_uocr=True)
        assert uocr_mock.call_count == 2

    # ── Sesión 129: salvaguarda mucho_mas_debil en la negativa ──────

    def test_negativa_mucho_mas_debil_ignora_y_redispare(self, mocker):
        """Sesión 129: la negativa se registró cuando el híbrido detectó la
        página FUERTE (3 bloques conf 0.5). Si la página gemela ahora se
        detecta MUCHO más débil (1 bloque conf 0.1), la negativa NO aplica:
        se ignora y se re-dispara el VLM (el diálogo artístico que el híbrido
        pierde es justo el que el VLM podría recuperar)."""
        mgr = OCRManager()
        img = _make_img()
        # Registro la negativa directamente con los stats de la página fuerte:
        mgr._registrar_decision_negativa("0.400:salv", 3, 0.5)

        hybrid_debil = [_block("a", 0.1)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid_debil)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:salv")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([_block("ARTE", 0.9)], [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1, "mucho_mas_debil → debe re-disparar el VLM"

    def test_negativa_comparable_se_honra(self, mocker):
        """Sesión 129: con detección COMPARABLE (3 bloques conf 0.5 vs 0.45
        cacheados) la negativa se honra — determinismo: la página gemela no
        re-dispara sin motivo."""
        mgr = OCRManager()
        img = _make_img()
        mgr._registrar_decision_negativa("0.400:comp", 3, 0.5)

        hybrid = [_block(f"t{i}", 0.45) for i in range(3)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:comp")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 0, "comparable → honra la negativa (sin VLM)"

    def test_salvaguarda_negativa_se_persiste_con_stats(self):
        """Sesión 129: los stats guardados viajan por la persistencia — un
        proceso nuevo puede aplicar la salvaguarda mucho_mas_debil con los
        stats originales de la negativa."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("0.400:pers", 4, 0.6)
        mgr2 = TestPersistenciaDisco()._recargar_desde_disco()

        # Detección comparable (4 bloq conf 0.55) → honra:
        assert mgr2._is_decision_negativa_vigente("0.400:pers", 4, 0.55)
        # Detección mucho más débil (1 bloque conf 0.1) → ignora (re-dispara):
        assert not mgr2._is_decision_negativa_vigente("0.400:pers", 1, 0.1)

    # ── Sesión 134: salvaguarda de detección débil (caso p5) ────────

    def test_negativa_debil_redispare_una_vez_y_luego_se_congela(self):
        """Caso p5 (sesión 129): la negativa se registró con 2 bloques conf
        0.42 — detección DEMASIADO débil para congelar. La página gemela con
        detección COMPARABLE re-dispara el VLM UNA vez (contador 0→1) y, al
        agotar el contador, la negativa se congela (el VLM ya tuvo su
        oportunidad)."""
        mgr = OCRManager()
        firma = "0.400:p5"
        mgr._registrar_decision_negativa(firma, 2, 0.42)

        # 1er hit (gemela comparable): no vigente → re-disparo permitido
        assert mgr._is_decision_negativa_vigente(firma, 2, 0.42) is False
        with OCRManager._uocr_cache_lock:
            assert OCRManager._uocr_neg_cache[firma][3] == 1  # contador

        # 2do hit: contador agotado → la negativa se congela (sin VLM)
        assert mgr._is_decision_negativa_vigente(firma, 2, 0.42) is True

    def test_negativa_fuerte_se_congela_sin_redispare(self):
        """Negativa registrada con detección FUERTE (6 bloques conf 0.7): la
        supresión es fiable → la gemela comparable se congela directamente,
        sin re-disparo (comportamiento previo a la sesión 134)."""
        mgr = OCRManager()
        firma = "0.400:fuerte"
        mgr._registrar_decision_negativa(firma, 6, 0.7)
        assert mgr._is_decision_negativa_vigente(firma, 6, 0.7) is True
        with OCRManager._uocr_cache_lock:
            assert OCRManager._uocr_neg_cache[firma][3] == 0  # sin re-disparos

    def test_negativa_debil_por_conf_redispare(self):
        """La debilidad también se detecta por CONFIANZA: 6 bloques pero conf
        0.25 (< 0.45) → la negativa no es fiable y la gemela re-dispara una
        vez."""
        mgr = OCRManager()
        firma = "0.400:conf"
        mgr._registrar_decision_negativa(firma, 6, 0.25)
        assert mgr._is_decision_negativa_vigente(firma, 6, 0.25) is False
        assert mgr._is_decision_negativa_vigente(firma, 6, 0.25) is True

    def test_negativa_debil_no_afecta_mucho_mas_debil(self):
        """Las dos salvaguardas coexisten: la negativa débil (2 bloq conf 0.42)
        re-dispara una vez por contador, pero una gemela MUCHO más débil
        (1 bloque conf 0.1) re-dispara sin consumir el contador (much_mas_debil
        va primero, como en la sesión 129)."""
        mgr = OCRManager()
        firma = "0.400:dual"
        mgr._registrar_decision_negativa(firma, 2, 0.42)
        # much_mas_debil: 1 < 2 AND 0.1 < 0.42*0.8 → re-dispara sin contador
        assert mgr._is_decision_negativa_vigente(firma, 1, 0.1) is False
        with OCRManager._uocr_cache_lock:
            assert OCRManager._uocr_neg_cache[firma][3] == 0

    def test_redispare_debil_integracion_gemela(self, mocker):
        """El escenario p5 completo: página 1 dispara VLM y no recupera nada
        (negativa débil registrada). Página 2 (gemela, detección comparable)
        RE-DISPARA una vez por la salvaguarda. Página 3: contador agotado →
        congelada (sin VLM)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.42), _block("mundo", 0.42)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        # Panel image grande → el trigger dispara aunque conf 0.42 > 0.2
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:p5_int")
        # VLM corre pero NUNCA recupera nada:
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        # Página 1: VLM corre y no recupera → negativa débil (2 bloq conf 0.42)
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        # Página 2 (gemela comparable): salvaguarda sesión 134 → re-dispara
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 2
        # Página 3: contador agotado → congelada (el VLM ya tuvo 2 oportunidades)
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 2

    def test_redispare_exitoso_limpia_la_negativa(self, mocker):
        """Si el re-disparo (sesión 134) SÍ recupera el diálogo artístico, la
        negativa queda REFUTADA y se elimina — las gemelas posteriores vuelven
        a intentar el VLM (no honran una negativa obsoleta)."""
        mgr = OCRManager()
        img = _make_img()
        mgr._registrar_decision_negativa("0.400:limp", 2, 0.42)
        hybrid = [_block("hola", 0.42), _block("mundo", 0.42)]
        ublocks = [_block("ERA UNA PROPUESTA", 0.95, x=200, y=50)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:limp")
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        # La recuperación limpió la negativa:
        with OCRManager._uocr_cache_lock:
            assert "0.400:limp" not in OCRManager._uocr_neg_cache
        assert "ERA UNA PROPUESTA" in [b["text"] for b in blocks]
        assert "unlimited" in engines

    def test_redispare_exitoso_limpia_la_negativa_batch(self, mocker):
        """El camino BATCH (Fase B) también limpia la negativa cuando el batch
        recupera algo — misma semántica que el single: la recuperación refuta
        la negativa débil y las gemelas posteriores vuelven a intentar el VLM."""
        mgr = OCRManager()
        img = _make_img()
        firma = "capA:0.400:limp_b"  # escopeada por doc_id (sesión 126)
        mgr._registrar_decision_negativa(firma, 2, 0.42)
        hybrid = [_block("hola", 0.42), _block("mundo", 0.42)]
        ublocks = [_block("RECUPERADO", 0.95, x=200, y=50)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:limp_b")
        # YOLO sin regiones (degradación limpia, sin ruido en el log):
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        # El batch del daemon SÍ recupera bloques en ambas páginas:
        mocker.patch("routes.api._ocr_with_unlimited_batch",
                     return_value=([(ublocks, []), (ublocks, [])], 5.0))
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        mgr.run_ocr_batch([img, img], "es", ocr_mode="fusion", doc_id="capA")

        # La recuperación del batch limpió la negativa de la firma escopeada:
        with OCRManager._uocr_cache_lock:
            assert firma not in OCRManager._uocr_neg_cache

    def test_redispare_debil_se_persiste_con_contador(self):
        """El contador viaja por la persistencia: tras un re-disparo (0→1), un
        proceso NUEVO carga el contador agotado y congela — el determinismo
        entre servidores se mantiene (no se regala el re-disparo de nuevo)."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        firma = "0.400:cont"
        mgr._registrar_decision_negativa(firma, 2, 0.42)
        assert mgr._is_decision_negativa_vigente(firma, 2, 0.42) is False

        # El disco tiene el contador=1 (formato v4: [ts, n, c, re]):
        datos = json.loads(ocr_engine._DECISION_CACHE_PATH.read_text(
            encoding="utf-8"))
        assert datos["neg"][firma][1:] == [2, 0.42, 1]

        # Proceso nuevo: contador agotado → congela (sin regalar el re-disparo)
        mgr2 = TestPersistenciaDisco()._recargar_desde_disco()
        assert mgr2._is_decision_negativa_vigente(firma, 2, 0.42) is True


# ─── Cache de decisión del TRIGGER por firma (sesión 116) ────────

class TestTriggerDecisionCache:
    """Política determinista del trigger v4.2: la decisión (disparar o no el
    VLM) se cachea por firma de layout — misma imagen → misma firma → misma
    decisión entre corridas idénticas, aunque cuDNN varíe len(blocks)/avg_conf
    del híbrido. Es la garantía de que la p4 (que disparaba U-OCR en single
    pero no en batch) decida SIEMPRE igual."""

    def setup_method(self):
        OCRManager.clear_decision_cache()

    def test_primera_llamada_cachea_y_segunda_reutiliza(self, mocker):
        """Firma repetida: la 2ª llamada reutiliza la decisión cacheada AUNQUE
        los inputs del híbrido cambien (cuDNN) — el determinismo manda."""
        mgr = OCRManager()
        firma = "0.400:same"
        # Llamada 1: 2 bloques conf 0.15 → trigger True
        d1 = mgr._trigger_con_cache(
            firma, [_block("a", 0.15), _block("b", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d1 is True
        # Llamada 2: la MISMA firma, pero el híbrido ahora daría 5 bloques conf
        # 0.9 (sin cache → False). Con cache → True (decisión bloqueada).
        d2 = mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.9) for i in range(5)], 0.9,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d2 is True
        # La entrada guarda los inputs de la PRIMERA decisión (diagnóstico)
        with OCRManager._trigger_dec_lock:
            entrada = OCRManager._trigger_dec_cache[firma]
        assert entrada[1] == 2 and entrada[3] is True

    def test_decision_negativa_tambien_se_cachea(self, mocker):
        """La decisión NEGATIVA (no disparar) también queda cacheada: una
        página gemela con detección COMPARABLE (3 bloques conf 0.45 vs 0.5
        cacheados) no se desvía — misma firma → misma decisión."""
        mgr = OCRManager()
        firma = "0.400:neg"
        d1 = mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.5) for i in range(3)], 0.5,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d1 is False
        d2 = mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.45) for i in range(3)], 0.45,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d2 is False

    def test_force_uocr_no_consulta_ni_guarda_cache(self, mocker):
        """force_uocr (benchmark explícito): SIEMPRE compute fresco, ni
        consulta ni guarda — los modos forzados no contaminan el cache."""
        mgr = OCRManager()
        firma = "0.400:force"
        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=True, disable_uocr=False)
        assert d is True
        with OCRManager._trigger_dec_lock:
            assert firma not in OCRManager._trigger_dec_cache

    def test_disable_uocr_no_guarda_cache(self, mocker):
        """disable_uocr (benchmark): idem, nunca toca el cache."""
        mgr = OCRManager()
        firma = "0.400:dis"
        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=True)
        assert d is False
        with OCRManager._trigger_dec_lock:
            assert firma not in OCRManager._trigger_dec_cache

    def test_firma_distinta_recalcula(self, mocker):
        """Firmas distintas → decisiones independientes (una página nueva no
        hereda la decisión de otra con layout distinto)."""
        mgr = OCRManager()
        d1 = mgr._trigger_con_cache(
            "0.400:aaa", [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        d2 = mgr._trigger_con_cache(
            "0.700:bbb", [_block(f"t{i}", 0.5) for i in range(3)], 0.5,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d1 is True and d2 is False

    def test_cache_expira_tras_ttl(self, mocker):
        """TTL expirado → recomputa (la decisión cacheada no es eterna)."""
        import time
        from ocr_engine import TRIGGER_CACHE_TTL_S
        mgr = OCRManager()
        firma = "0.400:ttl"
        assert mgr._trigger_con_cache(
            firma, [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False) is True
        with OCRManager._trigger_dec_lock:
            OCRManager._trigger_dec_cache[firma] = (
                time.time() - TRIGGER_CACHE_TTL_S - 1, 2, 0.15, True, 0)
        assert mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.5) for i in range(3)], 0.5,
            has_big_panel=False, force_uocr=False, disable_uocr=False) is False

    def test_hit_refresca_timestamp_no_expira_a_mitad_corrida(self, mocker):
        """Sesión 128: un HIT refresca el timestamp (touch LRU) — la ventana
        TTL se cuenta desde la ÚLTIMA consulta, no desde el guardado. Una
        corrida larga (>30 min) donde una firma reaparece cada pocas páginas
        ya no expira la decisión a mitad de capítulo: el 3er hit (con ts
        artificialmente envejecido a justo antes de expirar) sigue devolviendo
        la decisión cacheada y refresca de nuevo."""
        import time
        from ocr_engine import TRIGGER_CACHE_TTL_S
        mgr = OCRManager()
        firma = "0.400:touch"
        # 1er hit: guarda la decisión
        assert mgr._trigger_con_cache(
            firma, [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False) is True
        # Envejecer la entrada a 5s antes de expirar (simula 29m55s después):
        with OCRManager._trigger_dec_lock:
            OCRManager._trigger_dec_cache[firma] = (
                time.time() - TRIGGER_CACHE_TTL_S + 5, 1, 0.15, True, 0)
        # 2do hit: debería refrescar (touch) y devolver la decisión. Con el
        # comportamiento viejo devolvía la decisión pero NO extendía la ventana.
        assert mgr._trigger_con_cache(
            firma, [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False) is True
        # El ts se refrescó a ~ahora (la ventana se rearmó):
        with OCRManager._trigger_dec_lock:
            ts_nuevo = OCRManager._trigger_dec_cache[firma][0]
        assert time.time() - ts_nuevo < 5  # recién refrescado

    def test_refresh_del_hit_se_persiste_en_disco(self):
        """Sesión 128: el refresh del hit también se persiste — un servidor
        NUEVO (proceso distinto) carga la ventana EXTENDIDA, no la original.
        Sin esto, un restart a mitad de corrida recargaría el ts viejo y la
        decisión expiraría igual."""
        import time
        from ocr_engine import TRIGGER_CACHE_TTL_S
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        firma = "0.400:touch_disk"
        mgr._trigger_cache_put(firma, 1, 0.15, True)
        # Envejecer a 5s de expirar y dar un hit → refresh + persistir:
        with OCRManager._trigger_dec_lock:
            OCRManager._trigger_dec_cache[firma] = (
                time.time() - TRIGGER_CACHE_TTL_S + 5, 1, 0.15, True, 0)
        assert mgr._trigger_cache_get(firma) == (1, 0.15, True)
        # El disco tiene el ts refrescado (dentro de la ventana, no el viejo):
        datos = json.loads(ocr_engine._DECISION_CACHE_PATH.read_text(
            encoding="utf-8"))
        ts_disco = datos["trigger"][firma][0]
        assert time.time() - ts_disco < 5  # recién refrescado
        # Y un proceso nuevo la carga como vigente (mismo helper que
        # TestPersistenciaDisco — simula un servidor recién arrancado):
        mgr2 = TestPersistenciaDisco()._recargar_desde_disco()
        assert mgr2._trigger_cache_get(firma) == (1, 0.15, True)

    def test_hit_negativa_refresca_timestamp(self, monkeypatch):
        """Sesión 128: mismo touch LRU en la consulta de negativas §8.4.1 —
        una corrida larga no expira la negativa a mitad de capítulo mientras
        la firma siga consultándose."""
        import time
        from ocr_engine import UOCR_CACHE_TTL_S
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        firma = "0.400:neg_touch"
        mgr._registrar_decision_negativa(firma)
        # Envejecer la entrada a 5s de expirar (formato v4: + contador):
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache[firma] = (
                time.time() - UOCR_CACHE_TTL_S + 5, 6, 0.7, 0)
        # Stats fuertes (6 bloq conf 0.7): no es detección débil → se honra
        assert mgr._is_decision_negativa_vigente(firma, 6, 0.7) is True
        with OCRManager._uocr_cache_lock:
            ts_nuevo = OCRManager._uocr_neg_cache[firma][0]
        assert time.time() - ts_nuevo < 5  # recién refrescado

    def test_integracion_2_corridas_misma_firma_misma_decision(self, mocker):
        """El escenario real de la p4: la misma página produce SIEMPRE la
        misma decisión (disparar U-OCR) aunque el híbrido varíe entre corridas.
        Corrida 1: 2 bloques conf 0.15 → VLM. Corrida 2: misma firma con 5
        bloques conf 0.9 (fresco = no disparar) → la caché mantiene la decisión
        → VLM igualmente."""
        mgr = OCRManager()
        img = _make_img()
        ublocks = [_block("título dorado", 0.93, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:same")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([dict(b) for b in ublocks], [], 5.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        # Corrida 1: página difícil (2 bloques, conf 0.15)
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("hola", 0.15), _block("mundo", 0.15)]])
        _, _, engines1 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" in engines1
        # Corrida 2: mismo layout, híbrido "mejor" (5 bloques conf 0.9) —
        # sin la caché no dispararía; con ella SÍ (decisión bloqueada). La
        # decisión del trigger es lo que se verifica (unlimited sigue
        # entrando); con el cache positivo (plan §11 P1) la RECUPERACIÓN se
        # reinyecta de la corrida 1 (la firma es la misma y el híbrido es
        # MÁS fuerte → la salvaguarda no aplica) → el daemon no se re-llama.
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block(f"t{i}", 0.9) for i in range(5)]])
        _, _, engines2 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" in engines2
        assert uocr_mock.call_count == 1, "Recuperación cacheada → sin re-inferencia"

    def test_integracion_decision_negativa_determinista(self, mocker):
        """El caso base: una página bien detectada (3 bloques conf 0.5 → sin
        VLM) guarda la decisión NEGATIVA; una gemela con detección comparable
        (3 bloques conf 0.45) no se desvía — no dispara (misma firma → misma
        decisión)."""
        mgr = OCRManager()
        img = _make_img()
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:neg")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")
        # Corrida 1: página bien detectada → sin VLM (decisión negativa cacheada)
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block(f"t{i}", 0.5) for i in range(3)]])
        _, _, engines1 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" not in engines1
        # Corrida 2: misma firma, híbrido débil pero comparable → la caché
        # mantiene NO-disparar (determinismo)
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("a", 0.45), _block("b", 0.45), _block("c", 0.45)]])
        _, _, engines2 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" not in engines2
        uocr_mock.assert_not_called()

    def test_negativa_cacheada_no_suprime_gemela_mucho_mas_debil(self, mocker):
        """Code review sesión 116: la firma es de LAYOUT, no de contenido. Si
        la decisión negativa se cacheó con una página FUERTE (3 bloques conf
        0.5) pero llega una página GEMELA con el mismo layout cuyo diálogo
        artístico el híbrido detecta MUCHO peor (1 bloque conf 0.1 → en fresco
        dispararía), la caché NO debe suprimir el VLM: se detecta la diferencia
        (menos bloques Y conf < 80% de la cacheada) → recomputa → dispara."""
        mgr = OCRManager()
        img = _make_img()
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:gemela")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([dict(b) for b in
                                                [_block("recuperado", 0.9)]], [], 3.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        # Página fuerte → decisión NEGATIVA cacheada (sin VLM)
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block(f"t{i}", 0.5) for i in range(3)]])
        _, _, engines1 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" not in engines1
        # Gemela mucho más débil (1 bloque conf 0.1): recomputa → dispara VLM
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("hola", 0.1)]])
        _, _, engines2 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" in engines2
        assert uocr_mock.call_count == 1

    def test_gemela_comparable_honra_negativa_cacheada(self, mocker):
        """Frontera: la gemela es más débil pero DENTRO de la tolerancia (2
        bloques conf 0.45 vs 3 bloques conf 0.5 cacheados: conf 0.45 >= 0.5*0.8)
        → honra la decisión negativa (determinismo, sin VLM innecesario)."""
        mgr = OCRManager()
        img = _make_img()
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:comp")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block(f"t{i}", 0.5) for i in range(3)]])
        _, _, engines1 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" not in engines1
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("a", 0.45), _block("b", 0.45)]])
        _, _, engines2 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" not in engines2
        uocr_mock.assert_not_called()

    # ── Sesión 136: salvaguarda de detección débil en el TRIGGER ──────
    # Espejo de la sesión 134 (negativas §8.4.1): una decisión NEGATIVA de
    # trigger cacheada con detección híbrida DEMASIADO POBRE (< 3 bloques o
    # conf < 0.45) no es fiable — una página gemela con detección COMPARABLE
    # puede RECOMPUTAR el trigger hasta UOCR_NEG_MAX_REINTENTOS veces por
    # firma (contador persistido) en vez de honrar a ciegas el "no VLM".

    def test_trigger_negativo_debil_recomputa_una_vez_y_luego_congela(self, mocker):
        """Decisión negativa cacheada con detección débil (2 bloq conf 0.42):
        la 1ª gemela comparable RECOMPUTA el trigger (consume el contador
        0→1 y vuelve a decidir), la 2ª (contador agotado) honra la negativa
        sin recomputar — el VLM ya tuvo su oportunidad."""
        mgr = OCRManager()
        firma = "0.400:p5_trig"
        mgr._trigger_cache_put(firma, 2, 0.42, False)  # débil → negativa

        compute_calls = []
        orig = mgr._compute_trigger

        def spy(*a, **k):
            compute_calls.append(a)
            return orig(*a, **k)

        mocker.patch.object(mgr, "_compute_trigger", side_effect=spy)

        # 1ª gemela comparable: recomputa (salvaguarda) → decide de nuevo
        d1 = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d1 is False
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 1  # contador

        # 2ª gemela: contador agotado → honra sin recomputar
        d2 = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d2 is False
        assert len(compute_calls) == 1  # solo la 1ª gemela recomputó

    def test_trigger_negativo_debil_recompute_dispara_vlm(self, mocker):
        """El caso de VALOR de la sesión 136: la negativa se cacheó con 2
        bloq conf 0.42 (no dispara: conf > 0.2), pero la gemela artística se
        detecta con 2 bloq conf 0.15 (ahora < 0.2 → en fresco dispararía).
        Sin la salvaguarda, la negativa la congelaba; con ella, recomputa y
        dispara el VLM."""
        mgr = OCRManager()
        firma = "0.400:valiosa"
        mgr._trigger_cache_put(firma, 2, 0.42, False)  # negativa débil

        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.15), _block("b", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is True  # recompute → ahora SÍ dispara
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][3] is True
            assert OCRManager._trigger_dec_cache[firma][4] == 1

    def test_trigger_negativo_fuerte_no_recomputa(self):
        """Decisión negativa con detección FUERTE (6 bloq conf 0.7): la
        supresión es fiable → la gemela comparable honra sin recomputar
        (comportamiento previo a la sesión 136 intacto)."""
        mgr = OCRManager()
        firma = "0.400:fuerte_trig"
        mgr._trigger_cache_put(firma, 6, 0.7, False)

        d = mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.7) for i in range(6)], 0.7,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is False  # honra
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 0  # sin recomputes

    def test_trigger_negativo_debil_por_conf_recomputa(self):
        """La debilidad también se detecta por CONFIANZA: 6 bloques pero conf
        0.25 (< 0.45) → la decisión negativa no es fiable y la gemela
        recomputa una vez."""
        mgr = OCRManager()
        firma = "0.400:conf_trig"
        mgr._trigger_cache_put(firma, 6, 0.25, False)

        d = mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.25) for i in range(6)], 0.25,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is False  # recompute con los mismos stats → sigue negativo
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 1
        # Y al agotar el contador, la siguiente gemela congela:
        d2 = mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.25) for i in range(6)], 0.25,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d2 is False
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 1  # sin más consumo

    def test_trigger_positivo_siempre_honra_sin_contador(self):
        """Las decisiones POSITIVAS (VLM) no tienen nada que suprimir → se
        honran siempre, incluso con stats débiles, SIN consumir el contador
        (la salvaguarda solo libera negativas)."""
        mgr = OCRManager()
        firma = "0.400:pos"
        mgr._trigger_cache_put(firma, 2, 0.42, True)  # positiva débil

        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is True
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 0  # sin consumo

    def test_trigger_debil_no_afecta_mucho_mas_debil(self):
        """Las dos salvaguardas coexisten: la negativa débil (2 bloq conf 0.42)
        recomputa una vez por contador, pero una gemela MUCHO más débil
        (1 bloque conf 0.1) recomputa por much_mas_debil SIN consumir el
        contador (va primero, como en la sesión 116)."""
        mgr = OCRManager()
        firma = "0.400:dual_trig"
        mgr._trigger_cache_put(firma, 2, 0.42, False)

        # much_mas_debil: 1 < 2 AND 0.1 < 0.42*0.8 → recomputa sin contador
        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.1)], 0.1,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is True  # recompute: 1 bloq conf 0.1 → dispara
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 0  # sin consumo

    def test_trigger_debil_integracion_gemela_dispara(self, mocker):
        """Escenario p5 del trigger end-to-end: página 1 bien detectada a
        medias (2 bloq conf 0.42 → decisión NEGATIVA débil cacheada, sin
        VLM). Página 2 (gemela, el híbrido la detecta con 2 bloq conf 0.15 —
        cuDNN/cache fresco): la salvaguarda recomputa → SÍ dispara el VLM.
        Página 3 (contador agotado): honra la decisión (ahora positiva) sin
        recomputa adicional."""
        mgr = OCRManager()
        img = _make_img()
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:p5_int_trig")
        ublocks = [_block("ERA UNA PROPUESTA", 0.95, x=200, y=50)]
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([dict(b) for b in ublocks], [], 5.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        # YOLO sin regiones (degradación limpia):
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])

        # Página 1: 2 bloq conf 0.42 → negativa débil cacheada, sin VLM
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("hola", 0.42), _block("mundo", 0.42)]])
        _, _, engines1 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" not in engines1
        assert uocr_mock.call_count == 0

        # Página 2 (gemela, detección más débil → cruza el umbral v4.2):
        # la salvaguarda recomputa → dispara el VLM (recupera el arte)
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("hola", 0.15), _block("mundo", 0.15)]])
        _, _, engines2 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" in engines2
        assert uocr_mock.call_count == 1

        # Página 3: la decisión ahora es positiva (VLM) → se honra, VLM corre.
        # Con el cache positivo (plan §11 P1) la recuperación de la página 2
        # se reinyecta (misma firma, misma detección débil → la salvaguarda
        # no aplica) → el daemon no se vuelve a llamar.
        _, _, engines3 = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert "unlimited" in engines3
        assert uocr_mock.call_count == 1, "Recuperación cacheada → sin re-inferencia"

    def test_trigger_contador_recompute_se_persiste_con_reload(self):
        """El contador de recomputes viaja por la persistencia: tras consumir
        el recompute (0→1), un proceso NUEVO carga el contador agotado y
        congela — el determinismo entre servidores se mantiene (no se regala
        el recompute de nuevo)."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        firma = "0.400:cont_trig"
        mgr._trigger_cache_put(firma, 2, 0.42, False)
        # Consumir el recompute (1ª gemela):
        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is False

        # El disco tiene el contador=1 (formato v5: [ts, n, c, decision, re]):
        datos = json.loads(ocr_engine._DECISION_CACHE_PATH.read_text(
            encoding="utf-8"))
        assert datos["trigger"][firma][1:] == [2, 0.42, False, 1]

        # Proceso nuevo: contador agotado → congela (sin regalar el recompute)
        mgr2 = TestPersistenciaDisco()._recargar_desde_disco()
        assert mgr2._trigger_cache_get(firma) == (2, 0.42, False)
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma][4] == 1

    def test_trigger_debil_recompute_se_loguea(self, capsys):
        """Sesión 137: el recompute de la salvaguarda débil imprime una línea
        EXPLÍCITA en el log (`[trigger] sesión 136: salvaguarda débil —
        recompute 1/1 de firma …`) sin depender del cache persistido. Al
        congelarse (contador agotado), NO imprime la línea de recompute y sí
        la de decisión cacheada de la sesión 116."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        firma = "0.400:log_debil"
        mgr._trigger_cache_put(firma, 2, 0.42, False)  # débil → negativa

        # 1ª gemela: consume el contador → línea de salvaguarda en el log
        d1 = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d1 is False
        out1 = capsys.readouterr().out
        assert ("[trigger] sesión 136: salvaguarda débil — recompute 1/"
                f"{ocr_engine.UOCR_NEG_MAX_REINTENTOS} de firma "
                f"{firma[:16]}…") in out1

        # 2ª gemela: contador agotado → congela (sin línea de recompute)
        d2 = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d2 is False
        out2 = capsys.readouterr().out
        assert "salvaguarda débil — recompute" not in out2
        assert "sesión 116: decisión cacheada" in out2
        assert "no VLM" in out2

    def test_trigger_debil_entrada_evictada_no_imprime_recompute(
            self, capsys, mocker):
        """Sesión 137: si la entrada desapareció entre get y consumir (el
        helper devuelve 0 = recomputar SIN consumo), NO imprime la línea de
        salvaguarda — solo se loguea el consumo real del contador. Se simula
        la evicción mockeando el get (devuelve la decisión débil) pero
        dejando el dict vacío (la entrada real ya no está)."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        firma = "0.400:evictada"
        # El get devuelve la decisión negativa débil cacheada…
        mocker.patch.object(
            mgr, "_trigger_cache_get", return_value=(2, 0.42, False))
        # …pero la entrada REAL ya no está en el dict (evicción) → el helper
        # devuelve 0 (permitido sin consumo) → recomputa sin loguear.
        d = mgr._trigger_con_cache(
            firma, [_block("a", 0.42), _block("b", 0.42)], 0.42,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d is False  # recomputó con los mismos stats → sigue negativo
        out = capsys.readouterr().out
        assert "salvaguarda débil" not in out  # sin consumo → sin print


# ─── Scope por documento del cache de decisiones (sesión 126) ────

class TestDocIdScope:
    """El cache de decisiones (trigger sesión 116 + §8.4.1 negativas) se
    escopea por DOCUMENTO (doc_id): la sesión 124 midió 94% de colisión de
    firma de layout entre capítulos de la MISMA serie — sin scope, el capítulo
    47 heredaría las decisiones del 43 (VLM suprimido en diálogo artístico).
    La clave pasa de "firma" a "doc_id:firma" cuando el caller envía doc_id."""

    def setup_method(self):
        OCRManager.clear_decision_cache()

    def test_firma_documento_prefija_la_firma(self):
        """_firma_documento: doc_id no vacío → "doc:firma"; vacío → firma
        (scope legacy compartido, sin cambio de comportamiento)."""
        mgr = OCRManager()
        assert mgr._firma_documento("abc123", "0.400:ff") == "abc123:0.400:ff"
        assert mgr._firma_documento("", "0.400:ff") == "0.400:ff"
        assert mgr._firma_documento("abc123", "") == ""

    def test_misma_firma_distintos_docs_no_comparten_decision(self):
        """La MISMA firma de layout en dos documentos distintos produce claves
        distintas → decisiones independientes (el 47 no hereda del 43)."""
        mgr = OCRManager()
        firma = "0.400:same-layout"
        # Doc A: página débil → VLM (decisión True cacheada bajo "A:firma")
        d_a = mgr._trigger_con_cache(
            mgr._firma_documento("docA", firma),
            [_block("a", 0.15), _block("b", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d_a is True
        # Doc B: misma firma de layout pero página FUERTE → en fresco no
        # dispara; con scope no hereda la decisión de A → False
        d_b = mgr._trigger_con_cache(
            mgr._firma_documento("docB", firma),
            [_block(f"t{i}", 0.9) for i in range(5)], 0.9,
            has_big_panel=False, force_uocr=False, disable_uocr=False)
        assert d_b is False
        # Ambas claves conviven en el cache con su prefijo
        with OCRManager._trigger_dec_lock:
            assert "docA:0.400:same-layout" in OCRManager._trigger_dec_cache
            assert "docB:0.400:same-layout" in OCRManager._trigger_dec_cache

    def test_sin_doc_id_comparte_scope_legacy(self):
        """Sin doc_id (callers antiguos/benchmarks) → clave sin prefijo: los
        callers legacy siguen compartiendo el scope de siempre."""
        mgr = OCRManager()
        firma = "0.400:legacy"
        assert mgr._trigger_con_cache(
            firma, [_block("a", 0.15)], 0.15,
            has_big_panel=False, force_uocr=False, disable_uocr=False) is True
        assert mgr._trigger_con_cache(
            firma, [_block(f"t{i}", 0.9) for i in range(5)], 0.9,
            has_big_panel=False, force_uocr=False, disable_uocr=False) is True

    def test_run_ocr_escopea_la_firma_con_doc_id(self, mocker):
        """run_ocr(doc_id=...) → la firma que llega al cache del trigger lleva
        el prefijo del documento."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block(f"t{i}", 0.5) for i in range(3)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:x")
        mocker.patch("routes.api._ocr_with_unlimited")

        mgr.run_ocr(img, "es", ocr_mode="fusion", doc_id="cap47")

        with OCRManager._trigger_dec_lock:
            # La decisión NEGATIVA quedó cacheada bajo la clave escopeada
            assert "cap47:0.400:x" in OCRManager._trigger_dec_cache
            assert "0.400:x" not in OCRManager._trigger_dec_cache

    def test_run_ocr_batch_escopea_firmas_por_documento(self, mocker):
        """run_ocr_batch(doc_id=...) → las firmas de Fase A (trigger + §8.4.1)
        llevan el prefijo del documento."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.400:batch")
        # Fase 2 no resuelve → va al batch; el batch no recupera nada → §8.4.1
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("routes.api._ocr_with_unlimited_batch",
                     return_value=([([], []), ([], [])], 5.0))
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])

        mgr.run_ocr_batch([img, img], "es", ocr_mode="fusion", doc_id="cap47")

        # Trigger cache escopeado
        with OCRManager._trigger_dec_lock:
            assert "cap47:0.400:batch" in OCRManager._trigger_dec_cache
        # §8.4.1 negativas escopeado (el batch no recuperó nada)
        with OCRManager._uocr_cache_lock:
            assert "cap47:0.400:batch" in OCRManager._uocr_neg_cache


# ─── Modo unlimited ──────────────────────────────────────────────

class TestUnlimited:
    def test_unlimited_ok(self, mocker):
        mgr = OCRManager()
        img = _make_img()
        ublocks = [_block("texto daemon", 0.9)]
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [], 4.0))
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="unlimited")
        assert engine == "unlimited"
        assert engines == ["unlimited"]
        assert blocks == ublocks

    def test_unlimited_fallback_easyocr(self, mocker):
        """Daemon no listo → fallback automático a EasyOCR (híbrido)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.8)]
        mocker.patch("routes.api._ocr_with_unlimited",
                     side_effect=RuntimeError("modelo no listo"))
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="unlimited")
        assert engine == "easyocr"
        assert engines == ["easyocr"]
        assert blocks == hybrid

    def test_uocr_enabled_false_anula_refuerzo_sin_daemon(self, mocker):
        """Gate UOCR_ENABLED (sesión 143, PLAN_MANGA_OCR Paso 3): con False el
        refuerzo se anula POR COMPLETO — ni se llama al daemon ni se registra
        la negativa §8.4.1. Es el mecanismo que usa manga_ocr.py (extracción
        pura sin VLM) y anula SOLO el VLM: YOLO/Ruta C/cls siguen activos (a
        diferencia de disable_uocr). Con True (default) el flujo histórico
        sigue intacto."""
        mgr = OCRManager()
        OCRManager.clear_decision_cache()  # aísla el cache positivo (plan §11 P1)
        img = _make_img()
        ublocks = [_block("texto daemon", 0.9)]
        mocker.patch.object(mgr, "_unlimited_ocr",
                            return_value=(ublocks, [], 4.0))
        mocker.patch.object(mgr, "_ruta_c_globos", return_value=[])
        mocker.patch.object(mgr, "_registrar_decision_negativa")

        # Gate apagado → ni daemon ni negativa
        mocker.patch("config.UOCR_ENABLED", False)
        engines = mgr._reforzar_con_unlimited(
            img, "es", [], 0.0, firma="firma-test")
        assert engines == []
        mgr._unlimited_ocr.assert_not_called()
        mgr._registrar_decision_negativa.assert_not_called()

        # Gate encendido (default True) → flujo histórico (daemon + fusión)
        mocker.patch("config.UOCR_ENABLED", True)
        engines = mgr._reforzar_con_unlimited(
            img, "es", [], 0.0, firma="firma-test")
        assert engines == ["unlimited"]
        mgr._unlimited_ocr.assert_called_once()

    def test_modo_cpu_anula_refuerzo_aun_con_uocr_enabled(self, mocker):
        """Preset modo_cpu (soporte sin GPU dedicada): MODO_CPU=True apaga el
        refuerzo VLM AUNQUE UOCR_ENABLED=True — el VLM es el único componente
        del pipeline que exige GPU. Igual que el gate UOCR_ENABLED, anula SOLO
        el VLM: YOLO/Ruta C/cls de rotación siguen activos y no se registra la
        negativa §8.4.1 (no hay decisión que cachear)."""
        mgr = OCRManager()
        OCRManager.clear_decision_cache()  # aísla el cache positivo (plan §11 P1)
        img = _make_img()
        ublocks = [_block("texto daemon", 0.9)]
        mocker.patch.object(mgr, "_unlimited_ocr",
                            return_value=(ublocks, [], 4.0))
        mocker.patch.object(mgr, "_ruta_c_globos", return_value=[])
        mocker.patch.object(mgr, "_registrar_decision_negativa")

        # El gate global sigue ON pero MODO_CPU (preset sin GPU) lo anula
        mocker.patch("config.UOCR_ENABLED", True)
        mocker.patch("config.MODO_CPU", True)
        engines = mgr._reforzar_con_unlimited(
            img, "es", [], 0.0, firma="firma-test")
        assert engines == []
        mgr._unlimited_ocr.assert_not_called()
        mgr._registrar_decision_negativa.assert_not_called()

        # MODO_CPU=False (default) → flujo histórico intacto
        mocker.patch("config.MODO_CPU", False)
        engines = mgr._reforzar_con_unlimited(
            img, "es", [], 0.0, firma="firma-test")
        assert engines == ["unlimited"]
        mgr._unlimited_ocr.assert_called_once()


# ─── Modo hybrid (easyocr/auto) ──────────────────────────────────

class TestHybrid:
    def test_easyocr_usa_hibrido(self, mocker):
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.8)]
        mock_ocr = mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="easyocr")
        assert engine == "easyocr"
        # El default ya corre el híbrido (use_hybrid=True)
        assert mock_ocr.call_args.kwargs["use_hybrid"] is True
        assert blocks == hybrid

    def test_pure_easyocr_desactiva_rapid(self, mocker):
        mgr = OCRManager()
        img = _make_img()
        pure = [_block("hola", 0.8)]
        mock_ocr = mocker.patch("ocr_utils._detect_and_ocr", return_value=pure)
        blocks, engine, engines = mgr.run_ocr(
            img, "es", ocr_mode="easyocr", pure_easyocr=True)
        assert mock_ocr.call_args.kwargs["use_hybrid"] is False
        assert blocks == pure

    def test_auto_tiene_fallback(self, mocker):
        mgr = OCRManager()
        img = _make_img()
        blocks_out = [_block("hola", 0.8)]
        mock_ocr = mocker.patch("ocr_utils._detect_and_ocr",
                                return_value=blocks_out)
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="auto")
        assert engine == "auto"
        assert mock_ocr.call_args.kwargs["allow_fallback"] is True


# ─── Ruta C ──────────────────────────────────────────────────────

class TestRutaC:
    def test_reocr_globos_se_invoca_y_fusiona(self, mocker):
        """Con paneles image del daemon, se detectan globos y se re-OCRean."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]
        ublocks = [_block("título", 0.9, x=200, y=50)]
        panel = {"x": 0, "y": 0, "w": 100, "h": 100}
        bubble = _block("DIÁLOGO ARTÍSTICO", 0.95, x=12, y=12, w=38, h=28)

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [panel], 5.0))
        # Ruta C: detecta 1 globo (en el panel y en full-page; región distante
        # del bloque híbrido para que _overlap_ratio mockeado a 0.0 no la descarte)
        region = {"x": 300, "y": 300, "w": 40, "h": 30}
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel",
                     return_value=[region])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.0)
        mocker.patch("ocr_utils._recover_regions_with_easyocr",
                     return_value=[bubble])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        texts = [b["text"] for b in blocks]
        assert "DIÁLOGO ARTÍSTICO" in texts
        assert "título" in texts
        assert "unlimited" in engines

    def test_ruta_c_errores_no_tumban_la_pagina(self, mocker):
        """Si la detección de globos lanza excepción, se degrada silenciosamente."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]
        ublocks = [_block("título", 0.9, x=200, y=50)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(ublocks, [{"x": 0, "y": 0, "w": 100, "h": 100}], 5.0))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel",
                     side_effect=RuntimeError("cv2 falló"))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        # U-OCR igual se fusiona; la Ruta C falló pero no mató la página
        texts = [b["text"] for b in blocks]
        assert "título" in texts
        assert "unlimited" in engines


# ─── Fase 6: YOLO → Ruta C (recuperador de regiones) ────────────

class TestRutaCYolo:
    """_ruta_c_yolo — YOLO detecta regiones (globos/cartelas/títulos) y las
    re-OCRea con la Ruta C; no altera el trigger v4.2."""

    def test_yolo_recupera_y_fusiona_con_hibrido(self, mocker):
        """Página débil (1 bloque, conf 0.6 < gate 0.35? no — conf 0.6 > 0.35
        pero 1 bloque < YOLO_GATE_MIN_BLOCKS 3) → YOLO corre, recupera 1 región,
        se fusiona y el engine registra 'yolo+rutac'."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]
        region = {"x": 300, "y": 300, "w": 60, "h": 40, "source": "yolo",
                  "label": "speech bubble", "cls_conf": 0.9}
        yolo = [_block("TÍTULO ARTÍSTICO", 0.8, x=300, y=300)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page",
                     return_value=[region])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.0)
        mocker.patch("ocr_utils._recover_regions_with_easyocr",
                     return_value=yolo)
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        texts = [b["text"] for b in blocks]
        assert "TÍTULO ARTÍSTICO" in texts
        assert "hola" in texts
        assert "yolo+rutac" in engines
        # El trigger v4.2 NO se alteró: con conf media alta y 2 bloques no
        # dispara el daemon (len<3 Y conf<0.2: conf=0.7 no cumple).
        assert "unlimited" not in engines

    def test_yolo_gate_no_corre_en_pagina_bien_detectada(self, mocker):
        """Página con >= YOLO_GATE_MIN_BLOCKS bloques y conf >= gate → YOLO NO
        corre (el texto ya está; el re-OCR de crops no aporta)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block(f"hola{i}", 0.7) for i in range(4)]  # 4 bloques conf 0.7

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        detect_mock = mocker.patch("ocr_utils._detect_text_regions_in_page")

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        detect_mock.assert_not_called()
        assert blocks == hybrid
        assert "yolo+rutac" not in engines

    def test_yolo_disable_uocr_apaga_detector(self, mocker):
        """disable_uocr (benchmark) apaga YOLO igual que el cls: el detector no
        se consulta y no se recupera nada."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]  # débil → gate pasaría

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        detect_mock = mocker.patch("ocr_utils._detect_text_regions_in_page")

        blocks, engine, engines = mgr.run_ocr(
            img, "es", ocr_mode="fusion", disable_uocr=True)

        detect_mock.assert_not_called()
        assert blocks == hybrid

    def test_yolo_sin_modelo_degrada_a_pipeline_normal(self, mocker):
        """Sin ultralytics/modelo → _detect_text_regions_in_page devuelve [] →
        la página sigue igual (trigger v4.2 intacto)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert blocks == hybrid
        assert "yolo+rutac" not in engines

    def test_yolo_descarta_regiones_cubiertas_por_bloques(self, mocker):
        """Regiones con overlap > 0.5 con un bloque híbrido NO se re-OCRean
        (solo diálogo perdido) — mismo patrón que _ruta_c_globos."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]
        region = {"x": 10, "y": 10, "w": 50, "h": 15, "source": "yolo"}

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page",
                     return_value=[region])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.9)

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert blocks == hybrid
        assert "yolo+rutac" not in engines

    def test_yolo_batch_recupera_en_fase_a(self, mocker):
        """En run_ocr_batch, la Fase 6 corre por página antes del trigger
        (página débil: 1 bloque → gate permite)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]
        region = {"x": 300, "y": 300, "w": 60, "h": 40, "source": "yolo"}
        yolo = [_block("CARTELA", 0.8, x=300, y=300)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page",
                     return_value=[region])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.0)
        mocker.patch("ocr_utils._recover_regions_with_easyocr",
                     return_value=yolo)
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        out = mgr.run_ocr_batch([img], "es", ocr_mode="fusion")

        blocks, engine, engines = out[0]
        texts = [b["text"] for b in blocks]
        assert "CARTELA" in texts
        assert "yolo+rutac" in engines


# ─── Fase 6.5: comic-text-detector → Ruta C (texto SIN globo) ───

class TestRutaCCTD:
    """_ruta_c_ctd — el tier comic-text-detector detecta texto flotante /
    pensamientos / tipografías de arte que híbrido y YOLO pierden, y lo
    re-OCRea por la Ruta C. Gate en cascada (solo si YOLO no resolvió la
    página) + dedup de regiones vs YOLO (lección del benchmark Paso 5)."""

    def test_ctd_recupera_y_fusiona_con_hibrido(self, mocker):
        """Página débil (1 bloque) → YOLO no aporta ([]), CTD corre, recupera
        1 región de texto sin globo, se fusiona y el engine registra
        'ctd+rutac' — sin disparar el VLM (conf media alta)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]
        region = {"x": 300, "y": 300, "w": 60, "h": 40, "source": "ctd",
                  "label": "ctd_eng", "cls_conf": 0.9}
        ctd = [_block("TEXTO FLOTANTE", 0.85, x=300, y=300)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                     return_value=[region])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.0)
        mocker.patch("ocr_utils._recover_regions_with_easyocr",
                     return_value=ctd)
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        texts = [b["text"] for b in blocks]
        assert "TEXTO FLOTANTE" in texts
        assert "hola" in texts
        assert "ctd+rutac" in engines
        # Trigger v4.2 intacto: 1 bloque conf 0.6 no cumple (len<3 Y conf<0.2)
        assert "unlimited" not in engines

    def test_ctd_cascada_tras_yolo(self, mocker):
        """YOLO recupera una cartela y CTD añade su propio bloque (región NO
        duplicada): ambos tiers conviven en cascada pre-trigger."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]
        r_yolo = {"x": 300, "y": 300, "w": 60, "h": 40, "source": "yolo"}
        r_ctd = {"x": 200, "y": 500, "w": 50, "h": 30, "source": "ctd"}
        yolo_b = [_block("CARTELA", 0.8, x=300, y=300)]
        ctd_b = [_block("PENSAMIENTO", 0.9, x=200, y=500)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page",
                     return_value=[r_yolo])
        mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                     return_value=[r_ctd])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.0)
        mocker.patch("ocr_utils._recover_regions_with_easyocr",
                     side_effect=[[yolo_b[0]], [ctd_b[0]]])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        texts = [b["text"] for b in blocks]
        assert {"hola", "CARTELA", "PENSAMIENTO"} <= set(texts)
        assert "yolo+rutac" in engines
        assert "ctd+rutac" in engines

    def test_ctd_gate_no_corre_en_pagina_bien_detectada(self, mocker):
        """Página con >= GATE_MIN_BLOCKS y conf >= gate → CTD NO corre (el
        texto ya está; el re-OCR de crops no aporta). El gate se evalúa con
        los bloques POST-YOLO (cascada)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block(f"hola{i}", 0.7) for i in range(4)]  # 4 bloques conf 0.7

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        detect_yolo = mocker.patch("ocr_utils._detect_text_regions_in_page")
        detect_ctd = mocker.patch("ocr_utils._detect_text_regions_comic_detector")

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        detect_yolo.assert_not_called()
        detect_ctd.assert_not_called()
        assert blocks == hybrid
        assert "ctd+rutac" not in engines

    def test_ctd_disable_uocr_apaga_detector(self, mocker):
        """disable_uocr (benchmark) apaga CTD igual que YOLO/cls: el detector
        no se consulta y no se recupera nada."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]  # débil → gate pasaría

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        detect_yolo = mocker.patch("ocr_utils._detect_text_regions_in_page")
        detect_ctd = mocker.patch("ocr_utils._detect_text_regions_comic_detector")

        blocks, engine, engines = mgr.run_ocr(
            img, "es", ocr_mode="fusion", disable_uocr=True)

        detect_yolo.assert_not_called()
        detect_ctd.assert_not_called()
        assert blocks == hybrid

    def test_ctd_sin_modelo_degrada_a_pipeline_normal(self, mocker):
        """Sin onnxruntime/modelo → _detect_text_regions_comic_detector
        devuelve [] → la página sigue igual (trigger v4.2 intacto)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                     return_value=[])

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert blocks == hybrid
        assert "ctd+rutac" not in engines

    def test_ctd_descarta_regiones_duplicadas_de_yolo(self, mocker):
        """Región CTD con overlap > DEDUP_IOU con una región YOLO NO se
        re-OCRea (lección del benchmark: no pagar 2 veces la misma zona).
        El detector CTD SÍ corre (gate pasado) pero no recupera nada."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]  # x=10: fuera de la zona YOLO/CTD
        r_yolo = {"x": 300, "y": 300, "w": 60, "h": 40, "source": "yolo"}
        r_ctd = {"x": 310, "y": 310, "w": 50, "h": 30, "source": "ctd"}  # duplica r_yolo
        yolo_b = [_block("CARTELA", 0.8, x=300, y=300)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page",
                     return_value=[r_yolo])
        detect_ctd = mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                                  return_value=[r_ctd])

        # overlap 0.0 contra bloques híbridos (x<100), 0.9 entre regiones
        # YOLO/CTD (misma zona del dibujo) → el dedup 1 descarta r_ctd.
        def ov(a, b):
            return 0.0 if (a["x"] < 100 or b["x"] < 100) else 0.9
        mocker.patch("ocr_utils._overlap_ratio", side_effect=ov)
        recover = mocker.patch("ocr_utils._recover_regions_with_easyocr",
                               return_value=yolo_b)
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        detect_ctd.assert_called_once()
        # Solo la Ruta C de YOLO corrió: la región CTD duplicada no se pagó.
        assert recover.call_count == 1
        assert "CARTELA" in [b["text"] for b in blocks]
        assert "ctd+rutac" not in engines

    def test_ctd_descarta_regiones_cubiertas_por_bloques(self, mocker):
        """Región CTD con overlap > 0.5 con un bloque ya detectado NO se
        re-OCRea (solo diálogo perdido) — mismo patrón que YOLO."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]  # ocupa (10,10,50,15)
        r_ctd = {"x": 10, "y": 10, "w": 50, "h": 15, "source": "ctd"}

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                     return_value=[r_ctd])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.9)
        recover = mocker.patch("ocr_utils._recover_regions_with_easyocr")

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        recover.assert_not_called()
        assert blocks == hybrid
        assert "ctd+rutac" not in engines

    def test_ctd_error_no_tumba_la_pagina(self, mocker):
        """Si el detector CTD lanza excepción, se degrada silenciosamente:
        la página sigue con el pipeline estándar."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                     side_effect=RuntimeError("onnxruntime falló"))

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")

        assert blocks == hybrid
        assert "ctd+rutac" not in engines

    def test_ctd_batch_recupera_en_fase_a(self, mocker):
        """En run_ocr_batch, la Fase 6.5 corre por página antes del trigger
        (página débil: 1 bloque → gate permite; YOLO no aportó → CTD sí)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.6)]
        region = {"x": 300, "y": 300, "w": 60, "h": 40, "source": "ctd"}
        ctd = [_block("TEXTO SIN GLOBO", 0.85, x=300, y=300)]

        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._detect_text_regions_comic_detector",
                     return_value=[region])
        mocker.patch("ocr_utils._overlap_ratio", return_value=0.0)
        mocker.patch("ocr_utils._recover_regions_with_easyocr",
                     return_value=ctd)
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))

        out = mgr.run_ocr_batch([img], "es", ocr_mode="fusion")

        blocks, engine, engines = out[0]
        assert "TEXTO SIN GLOBO" in [b["text"] for b in blocks]
        assert "ctd+rutac" in engines


# ─── Batch multi-página (Fase 1: infer_multi) ────────────────────

class TestRunOcrBatch:
    """run_ocr_batch — agrupa las páginas que disparan U-OCR en UN daemon call."""

    def test_batch_agrupa_triggers_en_una_llamada(self, mocker):
        """2 de 3 páginas disparan → _ocr_with_unlimited_batch se llama UNA
        vez con las 2 imágenes; la 3ª queda solo híbrida."""
        mgr = OCRManager()
        img_ok = _make_img()
        img_bad1 = _make_img()
        img_bad2 = _make_img()

        # Página 0: buena (3 bloques conf alta) → sin trigger
        ok_blocks = [_block(f"t{i}", 0.5) for i in range(3)]
        # Páginas 1-2: difíciles (<3 bloques conf baja) → trigger
        bad_blocks = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch(
            "ocr_utils._detect_and_ocr",
            side_effect=[ok_blocks, bad_blocks, bad_blocks],
        )
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="")
        # Fase 2: el reintento agresivo no aporta en las páginas malas
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        ublocks = [_block("título dorado", 0.93, x=200, y=50, w=80, h=20)]
        batch_mock = mocker.patch(
            "routes.api._ocr_with_unlimited_batch",
            return_value=([(ublocks, []), (ublocks, [])], 5.0),
        )
        # Ruta C: sin paneles → sin re-OCR
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        results = mgr.run_ocr_batch([img_ok, img_bad1, img_bad2], "es",
                                    ocr_mode="fusion")

        assert batch_mock.call_count == 1
        # El batch recibe exactamente las 2 imágenes trigger
        batch_imgs = batch_mock.call_args.args[0]
        assert len(batch_imgs) == 2
        # Resultados por página, mismo orden
        assert len(results) == 3
        assert results[0][1] == "fusion"
        assert results[0][2] == ["easyocr+rapid"]          # sin VLM
        assert results[1][2] == ["easyocr+rapid", "unlimited-batch"]
        assert results[2][2] == ["easyocr+rapid", "unlimited-batch"]
        assert any(b["text"] == "título dorado" for b in results[1][0])

    def test_batch_directo_mayor_de_cuatro_procesa_todas_las_pendientes(self, mocker):
        """La API limita a 4, pero el manager directo no debe perder páginas."""
        mgr = OCRManager()
        images = [_make_img() for _ in range(5)]
        mocker.patch(
            "ocr_utils._detect_and_ocr",
            side_effect=lambda *args, **kwargs: [
                _block("dificil", 0.15), _block("pagina", 0.15)
            ],
        )
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="")
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch(
            "ocr_utils._fusionar_blocks_multi",
            side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]),
        )

        def fake_batch(batch_images):
            return [([_block("recuperada", 0.95)], []) for _ in batch_images], 1.0

        batch_mock = mocker.patch(
            "routes.api._ocr_with_unlimited_batch", side_effect=fake_batch)

        results = mgr.run_ocr_batch(images, "es", ocr_mode="fusion")

        assert [len(call.args[0]) for call in batch_mock.call_args_list] == [4, 1]
        assert len(results) == 5
        assert all("unlimited-batch" in result[2] for result in results)

    def test_batch_sin_triggers_no_llama_daemon(self, mocker):
        """Todas las páginas resueltas por el híbrido → sin daemon batch."""
        mgr = OCRManager()
        imgs = [_make_img(), _make_img()]
        ok_blocks = [_block(f"t{i}", 0.5) for i in range(3)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=ok_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        batch_mock = mocker.patch("routes.api._ocr_with_unlimited_batch")

        results = mgr.run_ocr_batch(imgs, "es", ocr_mode="fusion")

        batch_mock.assert_not_called()
        assert [r[2] for r in results] == [["easyocr+rapid"], ["easyocr+rapid"]]

    def test_disable_uocr_apaga_el_cls_de_rotacion(self, mocker):
        """disable_uocr → el Event _ruta_c_cls_disabled queda seteado durante
        el run (el cls de rotación de la Ruta C no se ejecuta en benchmark)."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=([], [], 0.0))
        # Fase 2: el reintento agresivo no aporta → sigue al daemon
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        import ocr_utils as ou
        mgr.run_ocr(img, "es", ocr_mode="fusion", disable_uocr=True)

        assert ou._ruta_c_cls_disabled.is_set()
        # Y con disable_uocr=False vuelve a estar limpio
        mgr.run_ocr(img, "es", ocr_mode="fusion", disable_uocr=False)
        assert not ou._ruta_c_cls_disabled.is_set()

    def test_batch_daemon_caido_degrada_por_pagina(self, mocker):
        """RuntimeError del daemon batch → todas las páginas quedan híbridas."""
        mgr = OCRManager()
        imgs = [_make_img(), _make_img()]
        bad_blocks = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=bad_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="")
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("routes.api._ocr_with_unlimited_batch",
                     side_effect=RuntimeError("daemon no listo"))

        results = mgr.run_ocr_batch(imgs, "es", ocr_mode="fusion")

        assert all(r[2] == ["easyocr+rapid"] for r in results)
        assert all(r[1] == "fusion" for r in results)

    def test_batch_respeta_disable_uocr(self, mocker):
        """disable_uocr (benchmark) → nada de VLM, ni en batch."""
        mgr = OCRManager()
        imgs = [_make_img(), _make_img()]
        bad_blocks = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=bad_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        batch_mock = mocker.patch("routes.api._ocr_with_unlimited_batch")

        results = mgr.run_ocr_batch(imgs, "es", ocr_mode="fusion",
                                    disable_uocr=True)

        batch_mock.assert_not_called()
        assert all(r[2] == ["easyocr+rapid"] for r in results)

    def test_batch_fase2_resuelve_y_excluye_del_daemon(self, mocker):
        """Fase 2 (reintento agresivo) resuelve una página → NO va al batch VLM."""
        mgr = OCRManager()
        img1 = _make_img()
        img2 = _make_img()
        # Listas INDEPENDIENTES por página: _fusionar_blocks_multi muta la
        # lista in-place (blocks[:] = merged) — una referencia compartida
        # haría que la Fase 2 de la pág.1 contaminara la pág.2.
        bad1 = [_block("hola", 0.15), _block("mundo", 0.15)]
        bad2 = [_block("hola", 0.15), _block("mundo", 0.15)]
        extra = [_block("título", 0.85, x=200, y=50, w=80, h=20)]
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=[bad1, bad2])
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="")
        # Fase 2 resuelve la 1ª (devuelve bloques extra), la 2ª no
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr",
                     side_effect=[extra, []])  # pág1: recupera, pág2: no
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))
        batch_mock = mocker.patch(
            "routes.api._ocr_with_unlimited_batch",
            return_value=([([_block("vlm", 0.9, x=300, y=10)], [])], 5.0),
        )

        results = mgr.run_ocr_batch([img1, img2], "es", ocr_mode="fusion")

        # El batch solo recibe la 2ª página (la 1ª la resolvió Fase 2)
        batch_imgs = batch_mock.call_args.args[0]
        assert len(batch_imgs) == 1
        assert "rapid-aggressive" in results[0][2]
        assert "unlimited-batch" not in results[0][2]
        assert "unlimited-batch" in results[1][2]

    def test_batch_modo_no_fusion_delega_individual(self, mocker):
        """easyocr/unlimited → run_ocr individual por página (sin batch)."""
        mgr = OCRManager()
        imgs = [_make_img(), _make_img()]
        ok_blocks = [_block(f"t{i}", 0.5) for i in range(3)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=ok_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        batch_mock = mocker.patch("routes.api._ocr_with_unlimited_batch")

        results = mgr.run_ocr_batch(imgs, "es", ocr_mode="easyocr")

        batch_mock.assert_not_called()
        assert len(results) == 2
        assert results[0][1] == "easyocr"

    def test_batch_salvaguarda_debil_trigger_recomputa_y_congela(
            self, mocker, capsys):
        """run_ocr_batch (Fase A) aplica la salvaguarda débil del trigger
        IGUAL que el single (mismo _trigger_con_cache): una página con
        negativa débil cacheada y contador disponible RECOMPUTA y dispara el
        VLM en el batch (sesión 136 + print sesión 138); otra con el contador
        agotado se CONGELA (fuera del batch, 0 llamadas VLM para su firma)."""
        # TestRunOcrBatch no tiene setup_method: limpiar el cache de clase
        # (el put preserva el contador previo — una entrada heredada con
        # re_computes=1 congelaría la página 1 y rompería el test).
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        img1, img2 = _make_img(), _make_img()
        firma1 = "0.400:b1"   # débil, contador disponible → recomputa
        firma2 = "0.400:b2"   # débil, contador agotado → congela
        # 1 imagen → 1 firma en Fase A (exactamente 2 llamadas: i=0, i=1).
        mocker.patch("ocr_utils._page_signature",
                     side_effect=[firma1, firma2])

        # Negativas de trigger DÉBILES cacheadas (2 bloq conf 0.42):
        mgr._trigger_cache_put(firma1, 2, 0.42, False)
        mgr._trigger_cache_put(firma2, 2, 0.42, False)
        # firma2: agotar el contador (re_computes=1 = UOCR_NEG_MAX_REINTENTOS)
        with OCRManager._trigger_dec_lock:
            ts, n, c, dec, _re = OCRManager._trigger_dec_cache[firma2]
            OCRManager._trigger_dec_cache[firma2] = (ts, n, c, dec, 1)

        # Detección actual de AMBAS páginas: 2 bloq conf 0.15 (<0.2 → el
        # trigger v4.2 dispararía si recomputa). La 1ª recomputa → VLM; la
        # 2ª congelada → no VLM.
        weak_now = [_block("a", 0.15), _block("b", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=weak_now)
        mocker.patch("ocr_utils._page_has_large_image_panel",
                     return_value=False)
        # YOLO no aporta (determinista) y Fase 2 no resuelve:
        mocker.patch("ocr_utils._detect_text_regions_in_page", return_value=[])
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel",
                     return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))

        ublocks = [_block("arte recuperado", 0.93, x=200, y=50, w=80, h=20)]
        batch_mock = mocker.patch(
            "routes.api._ocr_with_unlimited_batch",
            return_value=([(ublocks, [])], 5.0),
        )

        results = mgr.run_ocr_batch([img1, img2], "es", ocr_mode="fusion")

        # La página 1 recomputó (contador 0→1) y disparó el VLM en el batch:
        assert batch_mock.call_count == 1
        batch_imgs = batch_mock.call_args.args[0]
        assert len(batch_imgs) == 1          # solo la página 1 en el daemon
        assert results[0][2] == ["easyocr+rapid", "unlimited-batch"]
        assert any(b["text"] == "arte recuperado" for b in results[0][0])
        # La página 2 se congeló: fuera del batch, solo híbrida
        assert results[1][2] == ["easyocr+rapid"]
        # Contador consumido solo en la 1ª; la 2ª quedó intacta (agotada) y su
        # decisión siguió NEGATIVA (congelada); la 1ª ahora es positiva:
        with OCRManager._trigger_dec_lock:
            assert OCRManager._trigger_dec_cache[firma1][3] is True
            assert OCRManager._trigger_dec_cache[firma1][4] == 1
            assert OCRManager._trigger_dec_cache[firma2][3] is False
            assert OCRManager._trigger_dec_cache[firma2][4] == 1
        # El log muestra el recompute (sesión 138) y la congelación:
        out = capsys.readouterr().out
        assert (f"[trigger] sesión 136: salvaguarda débil — recompute 1/"
                f"{ocr_engine.UOCR_NEG_MAX_REINTENTOS} de firma "
                f"{firma1[:16]}…") in out
        assert "sesión 116: decisión cacheada" in out


# ─── Persistencia en disco (sesión 125) ─────────────────────────
# El path del archivo lo aísla el fixture autouse de conftest.py a tmp_path.

class TestPersistenciaDisco:
    def _recargar_desde_disco(self):
        """Simula un proceso/servidor NUEVO: memoria vacía + recarga del
        archivo persistido (el fixture autouse ya lo aisló a tmp_path)."""
        with OCRManager._trigger_dec_lock:
            OCRManager._trigger_dec_cache.clear()
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        OCRManager._cargar_cache_disco(force=True)
        return OCRManager()

    def test_decision_trigger_se_persiste_y_un_proceso_nuevo_la_reutiliza(self):
        """Una decisión de trigger guardada se escribe en disco y un OCRManager
        nuevo (memoria vacía, como un servidor recién arrancado) la recarga y
        la honra — determinismo entre procesos separados."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._trigger_cache_put("firma_persistida", 2, 0.31, True)

        # El archivo se escribió tras el put:
        assert ocr_engine._DECISION_CACHE_PATH.exists()
        datos = json.loads(ocr_engine._DECISION_CACHE_PATH.read_text(encoding="utf-8"))
        assert datos["trigger"]["firma_persistida"][1:] == [2, 0.31, True, 0]

        # Proceso nuevo: memoria limpia → recarga del disco:
        mgr2 = self._recargar_desde_disco()
        assert mgr2._trigger_cache_get("firma_persistida") == (2, 0.31, True)

    def test_negativa_se_persiste_por_defecto_desde_sesion_129(self):
        """Sesión 129: con la salvaguarda mucho_mas_debil en la consulta, la
        persistencia de las negativas §8.4.1 ya NO es ciega → el flag
        UOCR_NEG_CACHE_PERSIST es True por defecto. Una negativa registrada
        se escribe en disco (clave "neg", formato [ts, n_blocks, avg_conf])
        y un proceso nuevo la carga y la honra."""
        assert ocr_engine.UOCR_NEG_CACHE_PERSIST is True
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("firma_neg", 3, 0.5)

        # Se escribió con stats + contador de re-disparos (sesión 134):
        path = ocr_engine._DECISION_CACHE_PATH
        assert path.exists()
        datos = json.loads(path.read_text(encoding="utf-8"))
        entry = datos.get("neg", {}).get("firma_neg")
        assert entry is not None and entry[1:] == [3, 0.5, 0]

        mgr2 = self._recargar_desde_disco()
        assert mgr2._is_decision_negativa_vigente("firma_neg", 3, 0.5)

    def test_carga_descarta_entradas_expiradas(self):
        """Entradas fuera de TTL en el archivo se podan en la carga — el TTL
        se respeta también a través de reinicios del servidor."""
        OCRManager.clear_decision_cache()
        path = ocr_engine._DECISION_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": ocr_engine._DECISION_CACHE_VERSION,
            "trigger": {"vieja": [time.time() - 99999, 2, 0.31, True, 0]},
        }), encoding="utf-8")

        mgr = self._recargar_desde_disco()
        assert mgr._trigger_cache_get("vieja") is None

    def test_archivo_version_v1_sin_scope_se_descarta(self):
        """Sesión 126: el archivo v1 tenía claves sin scope por documento
        (firma bruta). Un archivo v1 (o de versión desconocida) se descarta
        en la carga y se elimina — nunca matchearía los lookups escopeados
        "doc_id:firma" de la v2."""
        OCRManager.clear_decision_cache()
        path = ocr_engine._DECISION_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 1,
            "trigger": {"firma_bruta_v1": [time.time(), 3, 0.6, True, 0]},
        }), encoding="utf-8")

        mgr = self._recargar_desde_disco()
        # No se carga NADA de la v1 (ni las entradas frescas):
        assert mgr._trigger_cache_get("firma_bruta_v1") is None
        assert not path.exists()  # clean start: el archivo se eliminó

    def test_archivo_corrupto_degrada_sin_crash_y_se_elimina(self):
        """Un archivo corrupto (crash a mitad de escritura, edición manual)
        nunca rompe el OCR: se descarta y se parte de cero."""
        OCRManager.clear_decision_cache()
        path = ocr_engine._DECISION_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{esto no es json válido", encoding="utf-8")

        mgr = self._recargar_desde_disco()  # no debe lanzar
        assert mgr._trigger_cache_get("cualquiera") is None
        assert not path.exists()  # el corrupto se eliminó

    def test_clear_decision_cache_elimina_tambien_el_archivo(self):
        """clear_decision_cache limpia memoria Y elimina el archivo persistido
        — una sesión nueva arranca sin decisiones heredadas."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._trigger_cache_put("firma_y", 3, 0.5, False)
        assert ocr_engine._DECISION_CACHE_PATH.exists()

        OCRManager.clear_decision_cache()
        assert not ocr_engine._DECISION_CACHE_PATH.exists()
        assert mgr._trigger_cache_get("firma_y") is None

    # ── Sesión 127: flag UOCR_NEG_CACHE_PERSIST (ahora default True) ──

    def test_negativa_se_persiste_con_flag_activado(self):
        """Con UOCR_NEG_CACHE_PERSIST=True (default desde la sesión 129),
        registrar una negativa §8.4.1 la escribe en disco (clave "neg") y un
        proceso nuevo la carga y la honra — determinismo de ejecución
        completo entre servidores."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        # Stats FUERTES (5 bloq conf 0.6): la negativa no es de detección
        # débil → el honor check de abajo no se ve afectado por la salvaguarda
        # de la sesión 134 (que re-dispararía una negativa débil).
        mgr._registrar_decision_negativa("firma_neg_flag", 5, 0.6)

        # La negativa se escribió junto al trigger (v4, scope por documento):
        path = ocr_engine._DECISION_CACHE_PATH
        assert path.exists()
        datos = json.loads(path.read_text(encoding="utf-8"))
        assert "firma_neg_flag" in datos.get("neg", {})

        # Proceso nuevo → la negativa es vigente (no se re-dispara el VLM):
        mgr2 = self._recargar_desde_disco()
        assert mgr2._is_decision_negativa_vigente("firma_neg_flag", 5, 0.6)

    def test_negativa_persistida_expirada_se_poda_en_carga(self):
        """Con el flag activo (default True), las negativas fuera de TTL en
        el archivo se podan en la carga (mismo TTL que en memoria, respetado
        a través de reinicios). Formato [ts, n_blocks, avg_conf]."""
        OCRManager.clear_decision_cache()
        path = ocr_engine._DECISION_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": ocr_engine._DECISION_CACHE_VERSION,
            "trigger": {},
            "neg": {
                "fresca": [time.time(), 5, 0.6, 0],
                "caducada": [time.time() - 99999, 5, 0.6, 0],
            },
        }), encoding="utf-8")

        mgr = self._recargar_desde_disco()
        assert mgr._is_decision_negativa_vigente("fresca", 5, 0.6)
        assert not mgr._is_decision_negativa_vigente("caducada", 5, 0.6)

    def test_negativas_no_se_cargan_si_flag_desactivado(self, monkeypatch):
        """Aunque el archivo contenga negativas, un proceso con el flag en
        False NO las carga — la consulta §8.4.1 queda en memoria y el nuevo
        proceso re-corre el VLM (dirección segura: solo recupera)."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("firma_neg_apagada", 2, 0.4)
        path = ocr_engine._DECISION_CACHE_PATH
        assert path.exists()

        # Ahora el flag se apaga → la recarga NO lee la sección "neg":
        monkeypatch.setattr("ocr_engine.UOCR_NEG_CACHE_PERSIST", False)
        mgr2 = self._recargar_desde_disco()
        assert not mgr2._is_decision_negativa_vigente("firma_neg_apagada", 2, 0.4)

    def test_carga_descarta_entradas_malformadas_y_aplica_cap_lru(self):
        """Cobertura de las ramas defensivas de la carga: entradas con forma
        inválida (TypeError/ValueError) se ignoran una a una sin tumbar la
        carga; y cuando el archivo trae más entradas que el máximo, el cap
        LRU por timestamp poda las más viejas (trigger, neg y ceros)."""
        OCRManager.clear_decision_cache()
        path = ocr_engine._DECISION_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        ahora = time.time()
        # 300 entradas de trigger (cap 256): las 44 más viejas se podan.
        trigger = {f"t{i}": [ahora - i, 2, 0.5, True, 0] for i in range(300)}
        # Entrada malformada (tupla corta) → se descarta sin crashear.
        trigger["t_mal"] = [ahora, 2]  # unpack a 5 falla → TypeError
        trigger["t_mal2"] = ["no-es-ts", 2, 0.5, True, 0]  # ValueError
        # Igual para negativas (cap 256):
        neg = {f"n{i}": [ahora - i, 5, 0.6, 0] for i in range(300)}
        neg["n_mal"] = [ahora, 5]  # unpack a 4 falla
        # Y para el ledger de ceros (misma cap):
        ceros = {f"c{i}": [ahora - i, 2, 3, 0.1] for i in range(300)}
        ceros["c_mal"] = [ahora, 2, 3]  # unpack a 4 falla
        path.write_text(json.dumps({
            "version": ocr_engine._DECISION_CACHE_VERSION,
            "trigger": trigger,
            "neg": neg,
            "ceros": ceros,
        }), encoding="utf-8")

        mgr = self._recargar_desde_disco()
        with ocr_engine.OCRManager._trigger_dec_lock:
            assert len(ocr_engine.OCRManager._trigger_dec_cache) <= 256
        with ocr_engine.OCRManager._uocr_cache_lock:
            assert len(ocr_engine.OCRManager._uocr_neg_cache) <= 256
            assert len(ocr_engine.OCRManager._uocr_neg_ceros) <= 256
        # Las más frescas sobreviven al cap:
        assert mgr._trigger_cache_get("t0") is not None
        assert mgr._is_decision_negativa_vigente("n0", 5, 0.6)
        with ocr_engine.OCRManager._uocr_cache_lock:
            assert "c0" in ocr_engine.OCRManager._uocr_neg_ceros
        # Las malformadas no entraron:
        assert mgr._trigger_cache_get("t_mal") is None
        assert mgr._trigger_cache_get("t_mal2") is None
        assert not mgr._is_decision_negativa_vigente("n_mal", 5, 0.6)

    def test_persistencia_y_clear_toleran_errores_de_io(self, monkeypatch):
        """Las ramas OSError de _persistir_cache y clear_decision_cache
        degradan con aviso sin crashear (disco lleno, permisos, antivirus)."""
        import ocr_engine as oe

        def _boom_os_replace(src, dst):
            raise OSError("disco lleno (simulado)")

        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        monkeypatch.setattr(oe.os, "replace", _boom_os_replace)
        # El put intenta persistir → el OSError se captura y avisa:
        mgr._trigger_cache_put("firma_io", 2, 0.31, True)
        assert mgr._trigger_cache_get("firma_io") == (2, 0.31, True)  # en memoria sigue

        # clear con unlink que falla → no lanza:
        from pathlib import Path

        def _boom_unlink(self, *args, **kw):
            raise OSError("permiso denegado (simulado)")

        monkeypatch.setattr(Path, "unlink", _boom_unlink)
        OCRManager.clear_decision_cache()  # no debe lanzar

    def test_carga_corrupta_con_version_fresca_se_descarta(self):
        """Versión desconocida con JSON válido: se descarta todo el archivo y
        se elimina (clean start), cubriendo la rama de version mismatch."""
        OCRManager.clear_decision_cache()
        path = ocr_engine._DECISION_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 999,
            "trigger": {"fresca": [time.time(), 3, 0.6, True, 0]},
            "ceros": {"p13": [time.time(), 2, 3, 0.1]},
        }), encoding="utf-8")

        mgr = self._recargar_desde_disco()
        assert mgr._trigger_cache_get("fresca") is None
        with ocr_engine.OCRManager._uocr_cache_lock:
            assert ocr_engine.OCRManager._uocr_neg_ceros == {}
        assert not path.exists()


# ═══════════════════════════════════════════════════════════════
# Cero confirmado del VLM (plan §10.2 item 1, 2026-08-16)
# ═══════════════════════════════════════════════════════════════

class TestCeroConfirmado:
    """Gate persistente para páginas con recuperación VLM SIEMPRE 0 (pág
    13/17 del cap. 43). El cache §8.4.1 suprime dentro del TTL corto (30
    min); el ledger de ceros confirmados (_uocr_neg_ceros) extiende la
    supresión a TTL largo (7 días) cuando la firma falló >= UOCR_NEG_CERO_MIN
    veces en ventanas TTL distintas — la firma estable (dHash) hace que la
    MISMA página matchee entre corridas."""

    def test_un_fallo_no_confirma_cero(self):
        """Un solo fallo no congela: sin entrada vigente (TTL corto expirado),
        la página vuelve a disparar el VLM hasta confirmarse."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p13", 3, 0.0)
        # Sin entrada activa (TTL corto expirado/evictado) → no vigente:
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        assert not mgr._is_decision_negativa_vigente("p13", 3, 0.0)

    def test_dos_fallos_confirman_cero(self):
        """Dos fallos en ventanas TTL distintas (la entrada corta se limpió
        entre medias, como entre corridas) → cero confirmado: sin entrada
        activa, la consulta suprime el VLM."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p13", 6, 0.59)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()  # expira el TTL corto
        mgr._registrar_decision_negativa("p13", 3, 0.0)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        assert mgr._is_decision_negativa_vigente("p13", 3, 0.0)
        # El ledger acumuló las dos confirmaciones:
        with OCRManager._uocr_cache_lock:
            assert OCRManager._uocr_neg_ceros["p13"][1] == 2

    def test_cero_confirmado_salvaguarda_mucho_mas_debil(self):
        """Un cero confirmado NO suprime si la página actual se detecta MUCHO
        más débil que la última confirmación — el diálogo artístico que el
        híbrido ahora pierde es justo el que el VLM podría leer."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p13", 3, 0.0)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        # La ÚLTIMA confirmación (6/0.59) es la que fija los stats del ledger:
        mgr._registrar_decision_negativa("p13", 6, 0.59)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        # Detección actual MUCHO más débil que la última (6/0.59):
        # 2 < 6 Y 0.30 < 0.59*0.8=0.472 → la salvaguarda libera → VLM corre.
        assert not mgr._is_decision_negativa_vigente("p13", 2, 0.30)
        # Detección comparable → suprime:
        assert mgr._is_decision_negativa_vigente("p13", 6, 0.59)

    def test_cero_confirmado_no_aplica_con_ttl_largo_expirado(self):
        """Pasado el TTL largo (7 días), el cero confirmado deja de suprimir
        — el VLM vuelve a tener su oportunidad."""
        from ocr_engine import UOCR_NEG_CERO_TTL_S
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p13", 6, 0.59)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        mgr._registrar_decision_negativa("p13", 3, 0.0)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
            # Envejecer el ledger más allá del TTL largo:
            for firma in OCRManager._uocr_neg_ceros:
                ts, c, n, cf = OCRManager._uocr_neg_ceros[firma]
                OCRManager._uocr_neg_ceros[firma] = (
                    time.time() - UOCR_NEG_CERO_TTL_S - 1, c, n, cf)
        assert not mgr._is_decision_negativa_vigente("p13", 3, 0.0)

    def test_recovery_limpia_el_cero_confirmado(self):
        """Si el VLM SÍ recupera algo (o el daemon cae), la recuperación
        refuta el cero confirmado: _limpiar_decision_negativa borra la
        entrada del ledger y las gemelas vuelven a intentar el VLM."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p13", 6, 0.59)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        mgr._registrar_decision_negativa("p13", 3, 0.0)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        assert mgr._is_decision_negativa_vigente("p13", 3, 0.0)
        mgr._limpiar_decision_negativa("p13")
        assert not mgr._is_decision_negativa_vigente("p13", 3, 0.0)

    def test_cero_confirmado_se_persiste_y_se_recarga(self):
        """El ledger viaja con la persistencia: 2 corridas en procesos
        separados acumulan los fallos de la misma página y un proceso nuevo
        honra el cero confirmado (sección "ceros" del archivo)."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p17", 1, 0.53)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        mgr._registrar_decision_negativa("p17", 2, 0.1)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()

        # La sección "ceros" está en el archivo:
        path = ocr_engine._DECISION_CACHE_PATH
        assert path.exists()
        datos = json.loads(path.read_text(encoding="utf-8"))
        assert datos.get("ceros", {}).get("p17") is not None

        # Proceso nuevo (memoria limpia) → recarga y honra:
        with OCRManager._trigger_dec_lock:
            OCRManager._trigger_dec_cache.clear()
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        OCRManager._cargar_cache_disco(force=True)
        mgr2 = OCRManager()
        assert mgr2._is_decision_negativa_vigente("p17", 2, 0.1)

    def test_clear_decision_cache_limpia_el_ledger(self):
        """clear_decision_cache también borra el ledger de ceros — una sesión
        nueva arranca sin supresiones heredadas de capítulos anteriores."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        mgr._registrar_decision_negativa("p13", 6, 0.59)
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
        mgr._registrar_decision_negativa("p13", 3, 0.0)
        with OCRManager._uocr_cache_lock:
            assert len(OCRManager._uocr_neg_ceros) == 1
        OCRManager.clear_decision_cache()
        with OCRManager._uocr_cache_lock:
            assert len(OCRManager._uocr_neg_ceros) == 0
        assert not mgr._is_decision_negativa_vigente("p13", 3, 0.0)


# ═══════════════════════════════════════════════════════════════
# Cache de RECUPERACIÓN POSITIVA del VLM (plan §11 P1, 2026-08-17)
# ═══════════════════════════════════════════════════════════════

class TestCacheRecuperacionPositiva:
    """Complemento simétrico del ledger de ceros: cuando el VLM SÍ recupera
    bloques para una firma, la recuperación se cachea (TTL 7 días) y se
    reinyecta en re-corridas del mismo documento sin re-pagar la inferencia
    (573 s/capítulo → ~0). El determinismo 5/5 de la recuperación por página
    (plan §4.6 tabla ROI) hace seguro cachear."""

    def setup_method(self):
        OCRManager.clear_decision_cache()

    def test_recuperacion_se_guarda_y_se_reinyecta(self, mocker):
        """Una recuperación exitosa se guarda en _uocr_pos_cache; la misma
        firma con detección comparable reinyecta sin llamar al daemon."""
        mgr = OCRManager()
        img = _make_img()
        hybrid = [_block("hola", 0.15)]
        ublocks = [_block("título dorado", 0.93, x=200, y=50)]
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in hybrid])
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.500:pos1")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([dict(b) for b in ublocks], [], 5.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        assert "unlimited" in engines
        with OCRManager._uocr_cache_lock:
            assert "0.500:pos1" in OCRManager._uocr_pos_cache

        # Re-corrida: misma firma, detección comparable → reinyecta sin daemon
        blocks, engine, engines = mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        assert "unlimited" in engines
        assert any("título dorado" in (b.get("text") or "") for b in blocks)

    def test_recuperacion_con_deteccion_mucho_mas_debil_redispare(self, mocker):
        """La salvaguarda mucho_mas_debil aplica también al cache positivo:
        si la página actual se detecta MUCHO más débil que cuando se cacheó,
        la entrada NO aplica y el VLM vuelve a correr (el diálogo artístico
        que el híbrido pierde ahora es lo que el VLM leería)."""
        mgr = OCRManager()
        img = _make_img()
        ublocks = [_block("título dorado", 0.93, x=200, y=50)]
        # Panel grande: el trigger v4.2 dispara pese a la detección fuerte
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("ocr_utils._page_signature", return_value="0.500:pos2")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([dict(b) for b in ublocks], [], 5.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        # 1ª corrida: 8 bloques conf 0.6 + panel grande (0.6 < 0.75 del skip
        # → el panel necesita refuerzo) → VLM recupera → cache con stats
        # (8 bloques, conf 0.6).
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block(f"t{i}", 0.6) for i in range(8)]])
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1
        with OCRManager._uocr_cache_lock:
            assert "0.500:pos2" in OCRManager._uocr_pos_cache

        # 2ª corrida: misma firma pero híbrido MUCHO más débil (1 bloque conf
        # 0.1 < 8 bloques y 0.1 < 0.6*0.8=0.48) → la salvaguarda libera → el
        # daemon se vuelve a llamar (no se reinyecta la recuperación de otra
        # detección).
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in
                                                  [_block("hola", 0.1)]])
        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 2, "Detección mucho más débil → re-inferir"

    def test_recuperacion_expira_tras_ttl(self, mocker):
        """Pasado el TTL largo (7 días), la recuperación deja de reinyectarse
        y el VLM vuelve a correr (el modelo puede haber mejorado)."""
        from ocr_engine import UOCR_POS_CACHE_TTL_S
        mgr = OCRManager()
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_pos_cache["0.500:pos3"] = (
                time.time() - UOCR_POS_CACHE_TTL_S - 1, 8, 0.8,
                [_block("viejo", 0.9)], [])
        img = _make_img()
        # Detección débil (2 bloques conf 0.15) → dispara el trigger v4.2
        hybrid = [_block("hola", 0.15), _block("mundo", 0.15)]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="0.500:pos3")
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited",
                                 return_value=([], [], 5.0))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])

        mgr.run_ocr(img, "es", ocr_mode="fusion")
        assert uocr_mock.call_count == 1, "TTL expirado → el VLM vuelve a correr"
        with OCRManager._uocr_cache_lock:
            assert "0.500:pos3" not in OCRManager._uocr_pos_cache

    def test_recuperacion_se_persiste_y_se_recarga(self, mocker):
        """La sección 'pos' viaja con la persistencia: un proceso nuevo
        (servidor reiniciado) reinyecta las recuperaciones sin re-pagar la
        inferencia VLM."""
        OCRManager.clear_decision_cache()
        mgr = OCRManager()
        ublocks = [_block("título dorado", 0.93, x=200, y=50)]
        # Panel grande: el trigger dispara pese a la detección fuerte
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=True)
        mocker.patch("ocr_utils._page_signature", return_value="0.500:pos4")
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=([dict(b) for b in ublocks], [], 5.0))
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights: list(sources[0]) + list(sources[1]))
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("ocr_utils._detect_and_ocr",
                     side_effect=lambda *a, **k: [dict(b) for b in [_block("hola", 0.15)]])
        mgr.run_ocr(_make_img(), "es", ocr_mode="fusion")

        path = ocr_engine._DECISION_CACHE_PATH
        assert path.exists()
        datos = json.loads(path.read_text(encoding="utf-8"))
        assert "0.500:pos4" in datos.get("pos", {})

        # Proceso nuevo (memoria limpia) → recarga y reinyecta sin daemon:
        with OCRManager._trigger_dec_lock:
            OCRManager._trigger_dec_cache.clear()
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_neg_cache.clear()
            OCRManager._uocr_neg_ceros.clear()
            OCRManager._uocr_pos_cache.clear()
        OCRManager._cargar_cache_disco(force=True)
        mgr2 = OCRManager()
        with OCRManager._uocr_cache_lock:
            assert "0.500:pos4" in OCRManager._uocr_pos_cache
        # La salvaguarda usa los stats guardados (1 bloque, conf 0.15 del
        # híbrido pre-fusión): detección comparable → hit.
        hit = mgr2._get_pos_cache("0.500:pos4", 1, 0.15)
        assert hit is not None
        assert any("título dorado" in (b.get("text") or "") for b in hit[0])

    def test_clear_decision_cache_limpia_la_recuperacion(self):
        """clear_decision_cache también borra el cache positivo — una sesión
        nueva re-procesa el VLM desde cero."""
        with OCRManager._uocr_cache_lock:
            OCRManager._uocr_pos_cache["f"] = (
                time.time(), 2, 0.5, [_block("x", 0.9)], [])
        OCRManager.clear_decision_cache()
        with OCRManager._uocr_cache_lock:
            assert OCRManager._uocr_pos_cache == {}

