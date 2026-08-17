"""
ocr_engine.py — OCRManager: orquesta los 3 motores OCR en una única clase.

Centraliza la lógica de decisión (trigger) y los tiers de OCR que antes vivían
en el bloque de /api/process-page de routes/api.py:

  Tier 1+2: EasyOCR GPU + RapidOCR CPU (híbrido, _detect_and_ocr) — SIEMPRE en
            modo fusion; solo en fallback en modo unlimited.
  Tier 3:   Unlimited-OCR (daemon VLM 3B 4-bit) — SOLO si el trigger decide que
            la página es difícil (v4.2): 0 bloques, o <UOCR_TRIGGER_MIN_BLOCKS
            con conf <UOCR_TRIGGER_CONF, o panel image grande, o force_uocr.
  Tier 3.5: Ruta C — re-OCR a nivel de GLOBO (OpenCV blobs + EasyOCR 3.5×) sobre
            los paneles image del daemon + página completa.

La respuesta de run_ocr() es (blocks, ocr_engine_used, engines_used) — el mismo
contrato que el endpoint usaba. El acceso a las funciones de ocr_utils se hace
en RUNTIME vía módulo (self.ou.<fn>) para que los mocks de pytest que parchean
"ocr_utils.<fn>" o "routes.api._ocr_with_unlimited" sigan surtiendo efecto sin
tocar los tests existentes.

Dependencias: ocr_utils, config (sin ciclos — routes/api.py importa OCRManager
dentro del endpoint; OCRManager accede a routes.api solo en runtime).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

import ocr_utils
from runtime_diagnostics import PageDiagnostics

from config import (
    ROOT,
    UOCR_TRIGGER_CONF,
    UOCR_TRIGGER_MIN_BLOCKS,
    UOCR_IMAGE_BLOCK_RATIO,
    UOCR_PANEL_SKIP_MIN_CONF,
    OCR_ENGINE_WEIGHTS,
    UOCR_CACHE_TTL_S,
    UOCR_CACHE_MAX_ENTRIES,
    TRIGGER_CACHE_TTL_S,
    TRIGGER_CACHE_MAX_ENTRIES,
    RAPID_AGGRESSIVE_PARAMS,
    RAPID_RETRY_MAX_CONF,
    RAPID_RETRY_SALVADO_CONF,
    UOCR_NEG_CACHE_PERSIST,
    UOCR_NEG_WEAK_MAX_BLOCKS,
    UOCR_NEG_WEAK_MIN_CONF,
    UOCR_NEG_MAX_REINTENTOS,
    UOCR_NEG_CERO_MIN,
    UOCR_NEG_CERO_TTL_S,
    UOCR_POS_CACHE_TTL_S,
    UOCR_POS_CACHE_SALVAGUARDA,
)


# ── Persistencia en disco del cache de decisión del trigger (sesión 125) ──
# Para que 2 corridas en servidores SEPARADOS tomen las MISMAS decisiones de
# trigger (determinismo inter-proceso, no solo intra-proceso como la sesión
# 116), el cache de decisión del trigger por firma (_trigger_dec_cache) se
# persiste en disco tras cada mutación y se carga en el primer uso del
# OCRManager. Las entradas expiradas (TTL) se podan en la carga y se evictan
# perezosamente en la consulta (mismo comportamiento que en memoria). La
# escritura es atómica (temp + os.replace) y tolerante a fallos de disco.
#
# NOTA (code review sesión 125): el cache §8.4.1 de negativas NO se persiste
# por defecto. (a) El usuario pedía _trigger_dec_cache; (b) el §8.4.1 no tenía
# persistencia antes y su consulta era ciega (sin salvaguarda como la de
# _trigger_con_cache → una página gemela con diálogo artístico podría quedar
# suprimida); (c) la sesión 124 midió 94% de colisión de firma entre capítulos
# de la MISMA serie → persistirlo amplificaría el riesgo cross-PDF. El trigger
# cache por sí solo garantiza decisiones de trigger idénticas entre servidores.
# Desde la sesión 129 (salvaguarda mucho_mas_debil en la consulta de
# negativas, igual que el trigger), la supresión ya NO es ciega: una página
# gemela que se detecta MUCHO más débil que la que registró la negativa
# ignora la supresión y re-dispara el VLM. Por eso UOCR_NEG_CACHE_PERSIST
# (config.py) es ahora True por defecto — el determinismo de ejecución
# completo entre servidores ya no sacrifica recuperación artística.
_DECISION_CACHE_PATH = ROOT / "cache" / "ocr_decision_cache.json"
# Sesión 126: v2 = claves escopeadas por documento ("doc_id:firma"). Las
# claves v1 (firma bruta) de la sesión 125 ya no se cargan (ver
# _cargar_cache_disco) — una corrida nueva arranca con scope limpio.
# Sesión 129: v3 = formato de negativas con stats [ts, n_blocks, avg_conf]
# (necesario para la salvaguarda mucho_mas_debil tras reinicios). Los
# archivos v2 (negativas como ts plano) se descartan en la carga.
# Sesión 134: v4 = las negativas llevan además el contador de re-disparos
# [ts, n_blocks, avg_conf, re_disparos] (salvaguarda de detección débil —
# una gemela puede re-disparar el VLM hasta N veces por firma; el contador
# viaja por la persistencia para mantener el determinismo entre servidores).
# Los archivos v3 (stats sin contador) se descartan en la carga.
# Sesión 136: v5 = las entradas del TRIGGER llevan además el contador de
# recomputes [ts, n_blocks, avg_conf, decision, re_computes] (misma
# salvaguarda de detección débil que las negativas: una decisión NEGATIVA
# de trigger cacheada con detección pobre puede recomputarse hasta
# UOCR_NEG_MAX_REINTENTOS veces por firma). Los archivos v4 (trigger con
# 4 campos) se descartan en la carga.
# v6: _page_signature incorpora un digest del contenido del thumbnail; las
# entradas v5 basadas solo en layout deben invalidarse para evitar colisiones.
# v7: _page_signature cambia el digest exacto por un dHash (robusto a ruido
# leve — plan §10.2 item 1) y el archivo añade la sección "ceros" (ledger de
# ceros confirmados del VLM con TTL largo). Los archivos v6 (digest exacto +
# sin ledger) se descartan en la carga.
_DECISION_CACHE_VERSION = 8
_DISK_LOCK = threading.Lock()


class OCRManager:
    """Orquesta EasyOCR + RapidOCR + Unlimited-OCR con trigger selectivo v4.2.

    Cache de decisiones (§8.4.1): si una página con una firma de layout concreta
    ya disparó el refuerzo U-OCR y no recuperó nada (0 bloques nuevos), las
    páginas repetitivas del capítulo con la MISMA firma no vuelven a disparar
    la inferencia VLM (~2-8 min c/u) hasta que expira el TTL. Las variables de
    clase (compartidas entre instancias — OCRManager se instancia por request)
    con lock garantizan la persistencia entre páginas del mismo capítulo.
    Desde la sesión 125 el cache se persiste en disco
    (cache/ocr_decision_cache.json) para que también 2 servidores SEPARADOS
    tomen decisiones idénticas (determinismo inter-proceso).
    """

    # ── Cache de decisiones negativas (§8.4.1): firma → (ts, n_blocks, avg_conf, re_disparos) ──
    # Solo se cachean resultados NEGATIVOS (el refuerzo no recuperó nada):
    # son la señal de "páginas repetitivas no aportan". Los resultados con
    # recuperación NO se cachean — una página gemela con diálogo distinto
    # debe poder re-disparar. Sesión 129: la entrada guarda además los stats
    # de detección (n_blocks, avg_conf) del híbrido cuando se registró la
    # negativa — la salvaguarda mucho_mas_debil compara contra ellos.
    # Sesión 134: el 4º campo es el contador de re-disparos permitidos por la
    # salvaguarda de detección débil (caso p5) — ver config.py.
    _uocr_neg_cache: dict[str, tuple[float, int, float, int]] = {}
    # ── Ledger de CEROS CONFIRMADOS (2026-08-16, plan §10.2 item 1) ──
    # firma → (ts, count, n_blocks, avg_conf): cuántas veces una firma ha
    # sido un cero del VLM en ventanas TTL distintas. Cuando count >=
    # UOCR_NEG_CERO_MIN y la última confirmación está dentro de
    # UOCR_NEG_CERO_TTL_S (7 días), _is_decision_negativa_vigente suprime
    # el VLM incluso sin entrada vigente en _uocr_neg_cache (el TTL corto
    # de 30 min expiró entre corridas). El 3º/4º campo guardan los stats de
    # la ÚLTIMA confirmación para la salvaguarda mucho_mas_debil (si una
    # gemela se detecta MUCHO más débil, el VLM vuelve a correr). Se
    # persiste en la sección "ceros" del archivo de decisiones; un recovery
    # (_limpiar_decision_negativa) borra la entrada — la señal se refuta.
    _uocr_neg_ceros: dict[str, tuple[float, int, int, float]] = {}
    # ── Cache de RECUPERACIÓN POSITIVA (2026-08-17, plan §11 P1) ──
    # firma → (ts, n_blocks, avg_conf, ublocks, uimage_panels): cuando el VLM
    # SÍ recuperó bloques, se guardan para reinyectar en re-corridas del mismo
    # documento sin re-pagar la inferencia (573 s/capítulo → ~0). Los stats
    # (n_blocks, avg_conf) son del híbrido en el momento de la recuperación —
    # la salvaguarda mucho_mas_debil los usa para que una gemela detectada
    # MUCHO más débil no herede la recuperación de otra página (el diálogo
    # que el híbrido pierde ahora es justo el que el VLM leería). Persiste en
    # la sección "pos" del archivo de decisiones; TTL largo (7 días) como el
    # ledger de ceros; clear_decision_cache lo limpia; un re-run del VLM
    # sobreescribe la entrada.
    _uocr_pos_cache: dict[str, tuple[float, int, float, list[Any], list[Any]]] = {}
    _uocr_cache_lock: threading.Lock = threading.Lock()

    # ── Cache de decisión del TRIGGER por firma (sesión 116) ──
    # El trigger v4.2 depende de len(blocks)/avg_conf del híbrido, que cuDNN
    # puede hacer variar ligeramente entre corridas (y antes dependía del
    # device YOLO GPU/CPU por llamada — fijado ahora por proceso en
    # ocr_utils._resolver_device_yolo). Para garantizar que 2 corridas
    # idénticas tomen SIEMPRE la misma decisión por página, la DECISIÓN del
    # trigger se cachea por firma de layout (_page_signature) con TTL/LRU:
    # misma imagen → misma firma → misma decisión (positiva O negativa, a
    # diferencia del §8.4.1 que solo cachea negativas). No aplica con
    # force_uocr/disable_uocr (modos benchmark explícitos). La tupla guarda
    # (timestamp, n_blocks, avg_conf, decision, re_computes) — los inputs
    # para diagnóstico + el contador de recomputes de la salvaguarda de
    # detección débil (sesión 136, misma semántica que el re_disparos de las
    # negativas): si la decisión NEGATIVA se cacheó con detección pobre, la
    # gemela puede recomputar el trigger hasta UOCR_NEG_MAX_REINTENTOS veces.
    _trigger_dec_cache: dict[str, tuple[float, int, float, bool, int]] = {}
    _trigger_dec_lock: threading.Lock = threading.Lock()

    # ── Persistencia en disco (sesión 125) ────────────────────────
    # Flag + lock para cargar cache/ocr_decision_cache.json una sola vez por
    # proceso. clear_decision_cache lo resetea (una sesión nueva arranca sin
    # decisiones heredadas); los tests lo redirigen a tmp_path (conftest).
    # Persiste _trigger_dec_cache siempre; las negativas §8.4.1 SOLO si
    # UOCR_NEG_CACHE_PERSIST=True (sesión 127 — ver trade-off en config.py).
    _cache_cargado: bool = False
    _cache_load_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        # Acceso en runtime a las funciones de OCR: los tests parchean
        # ocr_utils.<fn>, y leer el atributo del módulo aquí (en lugar de
        # importar la referencia en el import) respeta esos mocks.
        self.ou = ocr_utils
        self.last_diagnostics: dict[str, Any] | None = None
        self.last_batch_diagnostics: list[dict[str, Any]] = []
        self._active_diagnostics: PageDiagnostics | None = None
        # Carga las decisiones persistidas de una corrida anterior (una vez
        # por proceso, ver _cache_cargado).
        self._cargar_cache_disco()

    @contextmanager
    def _diagnostic_stage(self, name: str) -> Iterator[None]:
        """Measure a stage when run_ocr owns an active diagnostic record."""
        if self._active_diagnostics is None:
            yield
            return
        with self._active_diagnostics.stage(name):
            yield

    # ─── Cache de decisión del trigger (sesión 116) ───────────────
    def _trigger_cache_get(self, firma: str) -> tuple[int, float, bool] | None:
        """Devuelve (n_blocks, avg_conf, decision) cacheados para la firma, o
        None si no hay entrada vigente (no existe o TTL expirado).

        Sesión 128: un HIT refresca el timestamp (touch LRU) — la decisión
        se mantiene viva mientras la firma siga apareciendo con una
        separación < TTL, de modo que una corrida larga (>30 min) no expira
        decisiones a mitad de capítulo. El refresh también se persiste: la
        ventana extendida debe sobrevivir a un reinicio del servidor.

        Sesión 136: el touch conserva el contador de recomputes (5º campo)
        de la salvaguarda de detección débil — un hit no resetea los
        recomputes ya consumidos.
        """
        with self._trigger_dec_lock:
            entrada = self._trigger_dec_cache.get(firma)
            if not entrada:
                return None
            ts, n_blocks, avg_conf, decision, re_computes = entrada
            if time.time() - ts > TRIGGER_CACHE_TTL_S:
                del self._trigger_dec_cache[firma]
                return None
            # Touch LRU (sesión 128): la ventana se cuenta desde AHORA, no
            # desde el guardado original — misma firma seguida de cerca no
            # caduca a mitad de corrida. (Solo un hit válido llega aquí.)
            self._trigger_dec_cache[firma] = (
                time.time(), n_blocks, avg_conf, decision, re_computes)
        # El refresh modifica el estado persistido: escribirlo (mismo patrón
        # que el put) para que un servidor nuevo honre la ventana extendida.
        self._persistir_cache()
        return n_blocks, avg_conf, decision

    def _trigger_cache_put(
        self, firma: str, n_blocks: int, avg_conf: float, decision: bool
    ) -> None:
        """Guarda la decisión del trigger por firma (LRU + TTL).

        Sesión 136: si la firma ya tenía una entrada, su contador de
        recomputes se PRESERVA (no se resetea) — misma semántica que
        _registrar_decision_negativa con el re_disparos: si un recompute de
        la salvaguarda de detección débil vuelve a decidir negativo, la
        firma queda congelada al agotar el contador (evita el bucle
        recomputa→negativo→recomputa infinito).
        """
        with self._trigger_dec_lock:
            prev = self._trigger_dec_cache.get(firma)
            re_computes = prev[4] if prev else 0
            self._trigger_dec_cache[firma] = (
                time.time(), n_blocks, avg_conf, decision, re_computes)
            if len(self._trigger_dec_cache) > TRIGGER_CACHE_MAX_ENTRIES:
                oldest = min(
                    self._trigger_dec_cache,
                    key=lambda k: self._trigger_dec_cache[k][0],
                )
                del self._trigger_dec_cache[oldest]
        # Sesión 125: persistir tras cada mutación para que un servidor NUEVO
        # (proceso distinto) tome la misma decisión por firma.
        self._persistir_cache()

    def _trigger_con_cache(
        self,
        firma: str | None,
        blocks: list[dict[str, Any]],
        avg_conf: float,
        has_big_panel: bool,
        force_uocr: bool,
        disable_uocr: bool,
    ) -> bool:
        """Trigger v4.2 con determinismo garantizado (sesión 116).

        Si hay una decisión cacheada para la firma (y no es modo benchmark),
        se reutiliza — misma imagen → misma firma → misma decisión entre
        corridas, aunque cuDNN varíe los inputs del híbrido. Si no hay caché,
        computa el trigger y LO GUARDA (también las decisiones NEGATIVAS:
        así la no-detección también es determinista entre corridas).
        """
        if firma and not disable_uocr and not force_uocr:
            cached = self._trigger_cache_get(firma)
            if cached is not None:
                n_c, conf_c, decision_c = cached
                # Salvaguarda de calidad (code review sesión 116): la firma
                # es de LAYOUT (grid de oscuridad), no de contenido. Una
                # decisión NEGATIVA cacheada (página fuerte → no VLM) no debe
                # suprimir el VLM en una página GEMELA con el mismo layout
                # pero diálogo artístico que el híbrido ahora detecta mucho
                # peor. Solo se honra la decisión cacheada si los stats
                # actuales son comparables; si la detección es claramente más
                # débil (menos bloques Y conf sustancialmente menor), se
                # recomputa — la página débil puede volver a disparar. El
                # determinismo se conserva: 2 corridas idénticas producen
                # stats idénticos → misma comparación → misma decisión.
                mucho_mas_debil = (
                    len(blocks) < n_c
                    and avg_conf < conf_c * 0.8
                )
                # Sesión 136 (salvaguarda de detección débil, caso p5 del
                # trigger): si la decisión NEGATIVA cacheada vino de una
                # detección híbrida DEMASIADO POBRE (< UOCR_NEG_WEAK_MAX_BLOCKS
                # bloques O conf < UOCR_NEG_WEAK_MIN_CONF), no es fiable — una
                # página GEMELA con detección COMPARABLE (que el much_mas_debil
                # no libera) puede recomputar el trigger hasta
                # UOCR_NEG_MAX_REINTENTOS veces por firma (contador persistido)
                # en vez de honrar a ciegas el "no VLM". Decisiones POSITIVAS
                # (VLM) se honran siempre (nada que suprimir). Al agotar el
                # contador, la negativa se congela (el trigger ya tuvo su
                # oportunidad de recomputar).
                # Positivas → honrar siempre (nada que suprimir): nunca
                # consumen el contador. Negativas con much_mas_debil →
                # recomputar sin contador (sesión 116, va primero). Solo una
                # negativa con detección COMPARABLE consulta la salvaguarda
                # débil (sesión 136) que sí consume el contador.
                recomputar = False
                recompute_n = 0
                if not decision_c:
                    if mucho_mas_debil:
                        recomputar = True  # sesión 116: sin consumir contador
                    else:
                        recompute_n = self._consumir_recompute_salvaguarda(firma)
                        recomputar = recompute_n >= 0
                        if recompute_n >= 1:
                            # Sesión 137: print explícito del recompute de la
                            # salvaguarda débil — el log muestra el consumo
                            # del contador sin depender del cache persistido.
                            print(f"[trigger] sesión 136: salvaguarda débil — "
                                  f"recompute {recompute_n}/"
                                  f"{UOCR_NEG_MAX_REINTENTOS} de firma "
                                  f"{firma[:16]}…")
                if not recomputar:
                    print(f"[trigger] sesión 116: decisión cacheada por firma "
                          f"{firma[:16]}… ({n_c} bloq, conf {conf_c:.2f} → "
                          f"{'VLM' if decision_c else 'no VLM'})")
                    return decision_c
            trigger = self._compute_trigger(
                blocks, avg_conf, has_big_panel,
                force_uocr=force_uocr, disable_uocr=disable_uocr,
            )
            if firma and not disable_uocr and not force_uocr:
                self._trigger_cache_put(firma, len(blocks), avg_conf, trigger)
            return trigger
        return self._compute_trigger(
            blocks, avg_conf, has_big_panel,
            force_uocr=force_uocr, disable_uocr=disable_uocr,
        )

    def _consumir_recompute_salvaguarda(self, firma: str) -> int:
        """Retorna el nº de recompute consumido (>= 1) si la salvaguarda de
        detección débil permite RECOMPUTAR el trigger y consume el contador
        de la firma; 0 si permite recomputar SIN consumo (la entrada
        desapareció o expiró — no hay negativa que honrar); -1 si congela
        (contador agotado: el trigger ya tuvo su oportunidad). El caller usa
        `>= 0` como "recomputar" y `>= 1` para loguear el consumo (sesión
        137: la línea `recompute 1/1` del log).

        Sesión 136 (caso p5 del trigger, espejo de la sesión 134 en las
        negativas §8.4.1): la decisión NEGATIVA cacheada se registró cuando
        el híbrido detectó la página con n_c/conf_c. Si esos stats son
        DEMASIADO POBRES (< UOCR_NEG_WEAK_MAX_BLOCKS bloques O conf <
        UOCR_NEG_WEAK_MIN_CONF) y el contador de recomputes no está agotado
        (< UOCR_NEG_MAX_REINTENTOS), se consume un recompute (contador + 1)
        → el caller recomputa el trigger (en vez de honrar el "no VLM").
        Al agotar el contador, congela (el trigger ya tuvo su oportunidad).

        NO se usa para decisiones POSITIVAS (el caller solo la llama con
        decision_c == False) ni para el much_mas_debil (que no consume
        contador y va por el camino de la sesión 116). Los stats se re-leen
        de la entrada ALMACENADA bajo el lock (fuente de verdad, evita race
        con otro worker) — por eso no recibe n_c/conf_c por parámetro. La
        mutación se persiste FUERA del lock (mismo patrón que la 128/134).

        NOTA (duplicación aceptada): este patrón (check débil + contador +
        touch + persist fuera del lock) es espejo de
        _is_decision_negativa_vigente (sesión 134) — los dos caches
        (trigger y §8.4.1) comparten la salvaguarda pero con dict/lock/TTL y
        gating de persistencia distintos, así que extraer un helper
        paramétrico añadiría indirección sin quitar duplicación real.
        """
        mutado = False
        resultado = -1
        with self._trigger_dec_lock:
            entrada = self._trigger_dec_cache.get(firma)
            if not entrada:
                return 0  # desapareció (evicción): sin negativa que honrar
            ts, n_c2, conf_c2, decision2, re_computes = entrada
            if (time.time() - ts) >= TRIGGER_CACHE_TTL_S:
                # Expiró (branch defensivo: el get acaba de hacer touch, así
                # que es casi código muerto). La eliminación NO se persiste
                # aquí a propósito: el caller recomputa y hace put, que sí
                # persiste el estado fresco; y un proceso nuevo poda las
                # expiradas en la carga (sesión 125) — inofensivo.
                del self._trigger_dec_cache[firma]
                return 0  # expiró: sin negativa que honrar → recomputar
            if (n_c2 < UOCR_NEG_WEAK_MAX_BLOCKS
                    or conf_c2 < UOCR_NEG_WEAK_MIN_CONF) \
                    and re_computes < UOCR_NEG_MAX_REINTENTOS:
                # Salvaguarda de detección débil: la decisión negativa viene
                # de una detección pobre → permitir el recompute (una vez por
                # firma, contador) en vez de congelar a ciegas.
                self._trigger_dec_cache[firma] = (
                    time.time(), n_c2, conf_c2, decision2, re_computes + 1)
                mutado = True
                resultado = re_computes + 1
            else:
                # Touch LRU (sesión 128): la ventana se cuenta desde AHORA.
                self._trigger_dec_cache[firma] = (
                    time.time(), n_c2, conf_c2, decision2, re_computes)
                mutado = True
                resultado = -1
        if mutado:
            self._persistir_cache()
        return resultado

    # ─── API pública ─────────────────────────────────────────────
    def run_ocr(
        self,
        img_bgr: Any,
        ocr_lang: str,
        ocr_mode: str,
        prefilter: bool = True,
        force_uocr: bool = False,
        disable_uocr: bool = False,
        pure_easyocr: bool = False,
        doc_id: str = "",
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        """Ejecuta el OCR según el modo. Retorna (blocks, engine_used, engines).

        doc_id (sesión 126): identificador de documento/sesión que escopea la
        clave de la firma en los caches de decisión (trigger + §8.4.1). La
        sesión 124 midió 94% de colisión de firma entre capítulos de la MISMA
        serie — sin scope, el capítulo 47 heredaría las decisiones del 43
        (supresión cruzada de VLM). El caller deriva el doc_id del PDF/archivo.
        """
        diagnostics = PageDiagnostics(ocr_mode=ocr_mode, ocr_lang=ocr_lang, doc_id=doc_id)
        previous_diagnostics = self._active_diagnostics
        self._active_diagnostics = diagnostics
        try:
            with diagnostics.stage("total"):
                if ocr_mode == "unlimited":
                    result = self._run_unlimited(img_bgr, ocr_lang, prefilter)
                elif ocr_mode == "fusion":
                    result = self._run_fusion(
                        img_bgr, ocr_lang, prefilter,
                        force_uocr=force_uocr, disable_uocr=disable_uocr,
                        doc_id=doc_id,
                    )
                else:
                    # easyocr / auto: híbrido EasyOCR+RapidOCR (auto con fallback CLAHE)
                    result = self._run_hybrid(
                        img_bgr, ocr_lang, prefilter,
                        ocr_mode=ocr_mode, pure_easyocr=pure_easyocr,
                    )
            blocks, engine_used, engines_used = result
            if not diagnostics.has_initial_counts():
                diagnostics.set_counts(initial_blocks=len(blocks))
            diagnostics.set_counts(final_blocks=len(blocks))
            diagnostics.set_engines(engine_used, engines_used)
            return result
        except Exception as exc:
            diagnostics.add_error(f"run_ocr: {type(exc).__name__}: {exc}")
            raise
        finally:
            diagnostics.finish()
            self.last_diagnostics = diagnostics.to_dict()
            self._active_diagnostics = previous_diagnostics

    # ─── API batch (Fase 1): N páginas, UN solo VLM si varias disparan ──
    def run_ocr_batch(
        self,
        images: list[Any],
        ocr_lang: str,
        ocr_mode: str = "fusion",
        prefilter: bool = True,
        force_uocr: bool = False,
        disable_uocr: bool = False,
        pure_easyocr: bool = False,
        doc_id: str = "",
    ) -> list[tuple[list[dict[str, Any]], str, list[str]]]:
        """OCR de VARIAS páginas agrupando los triggers U-OCR en UN batch.

        Procesa cada página con el pipeline de fusión individual (híbrido +
        trigger v4.2 + Fase 2), acumula las páginas que necesitan el VLM y
        las envía juntas al daemon con /ocr-batch (infer_multi) — las N
        imágenes comparten el prefill del modelo, amortizando ~60-110s/pág.

        Retorna una lista (una por imagen, mismo orden) de
        (blocks, engine_used, engines_used).

        No aplica a modo unlimited (ya es VLM forzado por página) ni a
        easyocr/auto (sin tier VLM): en esos casos delega en run_ocr
        individual por imagen.
        """
        n = len(images)
        if ocr_mode != "fusion":
            results: list[tuple[list[dict[str, Any]], str, list[str]]] = []
            self.last_batch_diagnostics = []
            for img in images:
                results.append(self.run_ocr(
                    img, ocr_lang, ocr_mode,
                    prefilter=prefilter,
                    force_uocr=force_uocr,
                    disable_uocr=disable_uocr,
                    pure_easyocr=pure_easyocr,
                    doc_id=doc_id,
                ))
                if self.last_diagnostics is not None:
                    self.last_batch_diagnostics.append(self.last_diagnostics)
            return results

        batch_diagnostics = [
            PageDiagnostics(ocr_mode=ocr_mode, ocr_lang=ocr_lang, doc_id=doc_id)
            for _ in images
        ]
        self.last_batch_diagnostics = []

        # Benchmark: disable_uocr también apaga el cls de rotación, el
        # detector YOLO y el detector comic-text-detector de regiones (mismo
        # patrón que _run_fusion).
        self.ou._ruta_c_cls_disabled.set() if disable_uocr else self.ou._ruta_c_cls_disabled.clear()
        self.ou._yolo_disabled.set() if disable_uocr else self.ou._yolo_disabled.clear()
        self.ou._ctd_disabled.set() if disable_uocr else self.ou._ctd_disabled.clear()

        # ── Fase A: híbrido + trigger + Fase 2 por página ──
        per_page_blocks: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        per_page_engines: list[list[str]] = [[] for _ in range(n)]
        per_page_firmas: list[str | None] = [None] * n  # para §8.4.1 en Fase B
        per_page_avg_conf: list[float] = [0.0] * n  # sesión 129: stats para la salvaguarda
        vlm_pending: list[int] = []  # índices que requieren VLM
        for i, img in enumerate(images):
            diagnostics = batch_diagnostics[i]
            with diagnostics.stage("hybrid"):
                blocks = self.ou._detect_and_ocr(
                    img, ocr_lang,
                    allow_fallback=True,
                    prefilter=prefilter,
                )
            engines: list[str] = ["easyocr+rapid"]
            avg_conf = (float(np.mean([b.get("confidence", 0) for b in blocks]))
                        if blocks else 0.0)
            # avg_conf refleja el HÍBRIDO (no se recomputa tras el merge YOLO
            # de abajo): el par (len(blocks), avg_conf) mezcla conteo post-YOLO
            # con conf pre-YOLO — consistente entre la consulta de la salvaguarda
            # (Fase A) y el registro de la negativa (Fase B), misma convención
            # que _run_fusion single.
            per_page_avg_conf[i] = avg_conf
            # ── Fase 6 (batch): YOLO → Ruta C por página, antes del trigger ──
            # Mismo recuperador de regiones que _run_fusion (con gate
            # heurístico): si YOLO recupera globos/cartelas/títulos, se
            # fusionan y el trigger v4.2 puede no disparar (menos VLM).
            with diagnostics.stage("yolo"):
                yolo_blocks, yolo_regiones = self._ruta_c_yolo(
                    img, ocr_lang, blocks, avg_conf)
            if yolo_blocks:
                merged = self.ou._fusionar_blocks_multi(
                    [blocks, yolo_blocks],
                    weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                             OCR_ENGINE_WEIGHTS["yolo"]],
                )
                print(f"[fusion-batch] Página {i}: Fase 6 {len(blocks)} "
                      f"híbrido + {len(yolo_blocks)} YOLO → {len(merged)}")
                blocks[:] = merged
                engines.append("yolo+rutac")
            # ── Fase 6.5 (batch): comic-text-detector → Ruta C ──
            # Mismo pase aditivo que _run_fusion (gate en cascada + dedup de
            # regiones vs YOLO): los bloques recuperados pueden evitar el VLM.
            with diagnostics.stage("ctd"):
                ctd_blocks = self._ruta_c_ctd(
                    img, ocr_lang, blocks, avg_conf, yolo_regiones)
            if ctd_blocks:
                merged = self.ou._fusionar_blocks_multi(
                    [blocks, ctd_blocks],
                    weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                             OCR_ENGINE_WEIGHTS["yolo"]],
                )
                print(f"[fusion-batch] Página {i}: Fase 6.5 {len(blocks)} "
                      f"híbrido+YOLO + {len(ctd_blocks)} CTD → {len(merged)}")
                blocks[:] = merged
                engines.append("ctd+rutac")
            # Sesión 116: firma computada SIEMPRE — la decisión del trigger
            # (positiva O negativa) se cachea por firma para que 2 corridas
            # idénticas tomen la misma decisión por página (el device YOLO ya
            # es fijo por proceso; la caché cierra el determinismo aunque
            # cuDNN varíe los inputs del híbrido entre corridas).
            # Sesión 126: la firma se escopea por DOCUMENTO (doc_id) — prefija
            # la clave para que el capítulo 47 (94% de firmas idénticas al 43
            # dentro de la misma serie, sesión 124) NUNCA herede decisiones
            # del 43. Aplica a ambos caches (trigger + §8.4.1 negativas).
            with diagnostics.stage("panel_check"):
                has_big_panel = self._has_big_panel(img)
            with diagnostics.stage("signature"):
                firma = self._firma_documento(doc_id, self._page_signature(img))
            per_page_firmas[i] = firma
            trigger = self._trigger_con_cache(
                firma, blocks, avg_conf, has_big_panel,
                force_uocr=force_uocr, disable_uocr=disable_uocr,
            )
            diagnostics.set_counts(initial_blocks=len(blocks))
            diagnostics.set_trigger(
                trigger,
                self._trigger_reason(
                    blocks, avg_conf, has_big_panel,
                    force_uocr=force_uocr,
                ),
            )
            needs_vlm = False
            if trigger:
                # Sesión 129: la salvaguarda mucho_mas_debil de la negativa
                # recibe los stats ACTUALES del híbrido (len(blocks)/avg_conf)
                # — si esta página se detecta mucho más débil que la que
                # registró la negativa, se ignora y se re-dispara el VLM.
                if (firma and not disable_uocr and not force_uocr
                        and self._is_decision_negativa_vigente(
                            firma, len(blocks), avg_conf)):
                    print(f"[fusion-batch] §8.4.1: firma {firma[:20]} repetitiva "
                          f"(U-OCR no recuperó antes) — salto de refuerzo")
                elif (not force_uocr and not disable_uocr
                        and self._reforzar_con_rapid_agresivo(
                            img, blocks, avg_conf, has_big_panel)):
                    print("[fusion-batch] Fase 2: reintento RapidOCR "
                          "agresivo resolvió la página — sin VLM")
                    engines.append("rapid-aggressive")
                else:
                    needs_vlm = True
            per_page_blocks[i] = blocks
            per_page_engines[i] = engines
            if needs_vlm:
                vlm_pending.append(i)

        # ── Fase B: UN solo batch daemon con TODAS las pendientes ──
        # Límite 4 por llamada (VRAM del daemon): la API batch ya lo impone
        # (1-4 imágenes), pero run_ocr_batch se protege por si se llama directo.
        if vlm_pending:
            # La API HTTP limita a cuatro imágenes por request por VRAM, pero
            # este manager también se usa directamente desde herramientas y
            # tests. Procesar en ventanas evita perder silenciosamente las
            # páginas 5+ cuando el caller entrega un lote mayor.
            for batch_start in range(0, len(vlm_pending), 4):
                batch_indices = vlm_pending[batch_start:batch_start + 4]
                # Cache de recuperación positiva (plan §11 P1): separar las
                # páginas con recuperación cacheada (reinyectar sin daemon)
                # de las que necesitan inferencia nueva. La firma dHash es
                # estable; la salvaguarda mucho_mas_debil (misma que los
                # ceros) evita reinyectar a gemelas detectadas MUCHO más
                # débiles (el diálogo que el híbrido pierde es lo que el VLM
                # leería). Un re-run con el VLM activo sobreescribe.
                cached: dict[int, tuple[list[Any], list[Any]]] = {}
                need_infer: list[int] = []
                for idx in batch_indices:
                    firma_idx = per_page_firmas[idx]
                    if firma_idx:
                        hit = self._get_pos_cache(
                            firma_idx,
                            len(per_page_blocks[idx]),
                            per_page_avg_conf[idx],
                        )
                        if hit is not None:
                            cached[idx] = hit
                            continue
                    need_infer.append(idx)
                u_pages: dict[int, tuple[list[Any], list[Any]]] = {}
                _infer_s = 0.0
                if need_infer:
                    batch_imgs = [images[i] for i in need_infer]
                    try:
                        with batch_diagnostics[need_infer[0]].stage("uocr_batch"):
                            u_pages_res, _infer_s = self._unlimited_ocr_batch(
                                batch_imgs)
                        for k, idx in enumerate(need_infer):
                            u_pages[idx] = u_pages_res[k]
                    except RuntimeError as uerr:
                        print(f"[fusion-batch] U-OCR batch no disponible "
                              f"({uerr}); páginas de la ventana con solo "
                              f"híbrido")
                for idx in batch_indices:
                    if idx in cached:
                        ublocks, uimage_panels = cached[idx]
                        came_from_cache = True
                    elif idx in u_pages:
                        ublocks, uimage_panels = u_pages[idx]
                        came_from_cache = False
                    else:
                        continue  # el batch falló para esta página
                    blocks = per_page_blocks[idx]
                    # Stats del híbrido ANTES de la fusión: la salvaguarda del
                    # cache positivo compara la detección híbrida, no el
                    # resultado ya fusionado (blocks se muta abajo con
                    # blocks[:] = merged — capturar antes).
                    n_hybrid = len(blocks)
                    # Ruta C: re-OCR a nivel de globo por página
                    bubble_blocks = self._ruta_c_globos(
                        images[idx], ocr_lang, blocks, uimage_panels)
                    combined_u = ublocks + bubble_blocks
                    if combined_u:
                        merged = self.ou._fusionar_blocks_multi(
                            [blocks, combined_u],
                            weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                                     OCR_ENGINE_WEIGHTS["unlimited"]],
                        )
                        print(f"[fusion-batch] Página {idx}: {len(blocks)} híbrido + "
                              f"{len(combined_u)} U-OCR "
                              f"({'cache' if came_from_cache else 'batch'}) "
                              f"→ {len(merged)}")
                        blocks[:] = merged
                        per_page_engines[idx].append(
                            "unlimited-cache" if came_from_cache
                            else "unlimited-batch")
                        # Sesión 134: recuperación exitosa → la negativa
                        # anterior de esta firma (si existía) queda refutada.
                        firma_idx = per_page_firmas[idx]
                        if firma_idx:
                            self._limpiar_decision_negativa(firma_idx)
                            # Plan §11 P1: cachear la recuperación nueva solo
                            # si vino del daemon (un hit no necesita re-guardar).
                            if not came_from_cache:
                                self._put_pos_cache(
                                    firma_idx,
                                    n_hybrid,
                                    per_page_avg_conf[idx],
                                    ublocks, uimage_panels,
                                )
                    else:
                        # §8.4.1: el batch no recuperó nada para esta página
                        # (la firma ya se calculó en Fase A — no recomputar).
                        # Sesión 129: registrar con los stats del híbrido de
                        # Fase A para la salvaguarda mucho_mas_debil.
                        firma_idx = per_page_firmas[idx]
                        if firma_idx:
                            self._registrar_decision_negativa(
                                firma_idx,
                                len(per_page_blocks[idx]),
                                per_page_avg_conf[idx],
                            )

        results = []
        for i in range(n):
            diagnostics = batch_diagnostics[i]
            blocks = per_page_blocks[i]
            engines = per_page_engines[i]
            diagnostics.set_counts(final_blocks=len(blocks))
            diagnostics.set_engines("fusion", engines)
            diagnostics.finish()
            self.last_batch_diagnostics.append(diagnostics.to_dict())
            results.append((blocks, "fusion", engines))
        return results

    # ─── Tier 3.5 (Fase 6): YOLO → Ruta C ────────────────────────
    def _ruta_c_yolo(
        self,
        img_bgr: Any,
        ocr_lang: str,
        blocks: list[dict[str, Any]],
        avg_conf: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Recupera diálogo/títulos artísticos con el detector YOLO (Fase 6).

        YOLO detecta regiones de texto como OBJETOS (globos, cartelas,
        títulos) — independiente de si el OCR ve glifos. Cada región se envía
        a la Ruta C existente (_recover_regions_with_easyocr con upscale 3.5× —
        A/B corregido 2026-08-15 (benchmark_rutac_upscale/recovery): 3.5×
        recupera 2 bloques más que 2× (pág 11) a tiempo neutro; el "−24%"
        previo era deriva de un benchmark roto — cls de rotación 180° y
        rotation_info 0/90/180/270 para títulos verticales/rotados).

        No altera el trigger v4.2: corre en fusion como recuperador de alto
        impacto (CPU 200-400ms + re-OCR de crops) — si los bloques recuperados
        elevan la página por encima del umbral, el VLM no dispara.

        Gate heurístico (code review): solo corre cuando el híbrido detectó la
        página DÉBILMENTE (menos bloques que YOLO_GATE_MIN_BLOCKS o conf media
        < YOLO_GATE_MAX_CONF). En páginas bien detectadas el re-OCR de hasta
        40 crops costaría ~2-6s/pág sin beneficio. NO es el trigger v4.2
        (que decide el VLM): es un filtro previo barato que limita el coste
        del recuperador a donde tiene impacto (el 12.2% perdido).

        Retorna (bloques_recuperados, regiones_utilizadas) — las regiones se
        pasan al tier CTD (Fase 6.5) para que deduplique por overlap ANTES de
        su propia Ruta C (Paso 4, PLAN_MANGA_OCR): no re-OCRear la misma zona
        dos veces.

        Degradación segura: sin ultralytics/modelo (→ []) o error → la página
        sigue con el pipeline estándar (blobs OpenCV de la Ruta C existente).
        """
        from config import YOLO_GATE_MIN_BLOCKS, YOLO_GATE_MAX_CONF
        # Benchmark: disable_uocr apaga YOLO (mismo patrón que el cls). Se
        # chequea aquí (además de dentro de _detect_text_regions_in_page)
        # para que los tests que mockean la función del detector lo respeten.
        if self.ou._yolo_disabled.is_set():
            return [], []
        # Gate: página bien detectada → YOLO no aporta (el texto ya está).
        if (len(blocks) >= YOLO_GATE_MIN_BLOCKS
                and avg_conf >= YOLO_GATE_MAX_CONF):
            return [], []
        yolo_blocks: list[dict[str, Any]] = []
        regiones_utilizadas: list[dict[str, Any]] = []
        try:
            regions = self.ou._detect_text_regions_in_page(img_bgr)
            if not regions:
                return [], []
            # Descartar regiones ya cubiertas por bloques híbridos: solo
            # re-OCR las que representan diálogo perdido (mismo patrón que
            # _ruta_c_globos).
            if blocks:
                regions = [
                    r for r in regions
                    if not any(self.ou._overlap_ratio(r, b) > 0.5 for b in blocks)
                ]
            regiones_utilizadas = list(regions)
            if regions:
                yolo_blocks = self.ou._recover_regions_with_easyocr(
                    img_bgr, regions, ocr_lang, upscale=3.5,
                    hybrid_blocks=blocks)
                print(f"[process-page] Fase 6 (YOLO): {len(regions)} regiones "
                      f"(globos/cartelas/títulos) → {len(yolo_blocks)} bloques "
                      f"recuperados")
        except Exception as yerr:
            print(f"[process-page] Fase 6 (YOLO → Ruta C) falló: {yerr}")
        return yolo_blocks, regiones_utilizadas

    # ─── Tier 3.6 (Fase 6.5): comic-text-detector → Ruta C ──────
    def _ruta_c_ctd(
        self,
        img_bgr: Any,
        ocr_lang: str,
        blocks: list[dict[str, Any]],
        avg_conf: float,
        yolo_regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Recupera texto SIN globo con comic-text-detector (Fase 6.5, Tier 3.6).

        Complementa a YOLO (Fase 6): detecta texto flotante sobre el dibujo,
        pensamientos y tipografías de arte que ni el híbrido ni el detector de
        globos ven. Cada región se envía a la Ruta C existente (mismo camino
        que YOLO: _recover_regions_with_easyocr, upscale 3.5× + cls/rotación).

        Lección del benchmark (Paso 5, PLAN_MANGA_OCR): la detección cuesta
        ~0.8s CPU pero el re-OCR de crops 2-4s, y la mayoría de regiones CTD
        DUPLICA a YOLO (85 regiones → 21 recuperados → solo 5 nuevos en 5
        págs). Por eso: (1) gate tipo YOLO evaluado con los bloques POST-YOLO
        (si YOLO ya resolvió la página, CTD no corre — cascada); (2) dedup de
        regiones CTD vs yolo_regions por overlap ANTES de la Ruta C (una zona
        que YOLO ya va a re-OCRear no se paga dos veces); (3) además, el
        filtro de regiones cubiertas por bloques existentes (híbrido + YOLO).

        Degradación segura: sin modelo/onnxruntime (→ []) o error → la página
        sigue igual (el tier simplemente no aporta).
        """
        from config import (COMIC_DETECTOR_GATE_MIN_BLOCKS,
                            COMIC_DETECTOR_GATE_MAX_CONF,
                            COMIC_DETECTOR_DEDUP_IOU)
        # Benchmark: disable_uocr apaga el tier (mismo patrón que YOLO). Se
        # chequea aquí (además de dentro de _detect_text_regions_comic_detector)
        # para que los tests que mockean el detector lo respeten.
        if self.ou._ctd_disabled.is_set():
            return []
        # Gate (cascada): página bien detectada TRAS YOLO → CTD no aporta.
        if (len(blocks) >= COMIC_DETECTOR_GATE_MIN_BLOCKS
                and avg_conf >= COMIC_DETECTOR_GATE_MAX_CONF):
            return []
        ctd_blocks: list[dict[str, Any]] = []
        try:
            regions = self.ou._detect_text_regions_comic_detector(img_bgr)
            if not regions:
                return []
            # Dedup 1: regiones CTD que duplican a YOLO (lección del
            # benchmark Paso 5) — YOLO ya cubre esa zona.
            if yolo_regions:
                regions = [
                    r for r in regions
                    if not any(self.ou._overlap_ratio(r, yr)
                               > COMIC_DETECTOR_DEDUP_IOU for yr in yolo_regions)
                ]
            # Dedup 2: regiones cubiertas por bloques ya detectados (híbrido
            # + YOLO recuperado) — solo diálogo perdido (patrón de YOLO).
            if blocks:
                regions = [
                    r for r in regions
                    if not any(self.ou._overlap_ratio(r, b) > 0.5 for b in blocks)
                ]
            if regions:
                ctd_blocks = self.ou._recover_regions_with_easyocr(
                    img_bgr, regions, ocr_lang, upscale=3.5,
                    hybrid_blocks=blocks)
                print(f"[process-page] Fase 6.5 (CTD): {len(regions)} regiones "
                      f"(texto sin globo) → {len(ctd_blocks)} bloques recuperados")
        except Exception as cerr:
            print(f"[process-page] Fase 6.5 (CTD → Ruta C) falló: {cerr}")
        return ctd_blocks

    # ─── Tier 1+2: híbrido EasyOCR + RapidOCR ───────────────────
    def _run_hybrid(
        self,
        img_bgr: Any,
        ocr_lang: str,
        prefilter: bool,
        ocr_mode: str,
        pure_easyocr: bool,
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        allow_fallback = ocr_mode != "easyocr"  # auto tiene fallback CLAHE
        with self._diagnostic_stage("hybrid"):
            blocks = self.ou._detect_and_ocr(
                img_bgr, ocr_lang,
                allow_fallback=allow_fallback,
                prefilter=prefilter,
                # pure_easyocr (benchmark): desactiva el tier híbrido RapidOCR →
                # solo EasyOCR GPU puro. El default ya corre el híbrido.
                use_hybrid=not pure_easyocr,
            )
        return blocks, ocr_mode, [ocr_mode]

    # ─── Tier 3: Unlimited-OCR (daemon) con fallback a híbrido ──
    def _run_unlimited(
        self,
        img_bgr: Any,
        ocr_lang: str,
        prefilter: bool,
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        try:
            with self._diagnostic_stage("uocr"):
                blocks, _image_panels, _ = self._unlimited_ocr(img_bgr)
            return blocks, "unlimited", ["unlimited"]
        except RuntimeError as uerr:
            print(f"[process-page] Unlimited-OCR no disponible ({uerr}); "
                  f"fallback a EasyOCR")
            blocks = self.ou._detect_and_ocr(
                img_bgr, ocr_lang,
                allow_fallback=True,
                prefilter=prefilter,
            )
            return blocks, "easyocr", ["easyocr"]

    # ─── Modo fusion: híbrido SIEMPRE + U-OCR solo con trigger ──
    def _run_fusion(
        self,
        img_bgr: Any,
        ocr_lang: str,
        prefilter: bool,
        force_uocr: bool,
        disable_uocr: bool,
        doc_id: str = "",
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        # Benchmark: disable_uocr también apaga el cls de rotación de la Ruta C
        # (mismo patrón que _uocr_inferring — Event global, set/clear por request).
        self.ou._ruta_c_cls_disabled.set() if disable_uocr else self.ou._ruta_c_cls_disabled.clear()
        with self._diagnostic_stage("hybrid"):
            blocks = self.ou._detect_and_ocr(
                img_bgr, ocr_lang,
                allow_fallback=True,
                prefilter=prefilter,
            )
        engines_used: list[str] = ["easyocr+rapid"]
        # ── Fase 6: YOLO → Ruta C (recuperador de regiones de alto impacto) ──
        # Corre SIEMPRE en fusion, ANTES del trigger v4.2: detecta globos/
        # cartelas/títulos como objetos y re-OCRea sus crops (upscale 2× +
        # rotación). Si los bloques recuperados elevan la página por encima
        # del umbral del trigger, el VLM no dispara — rescata parte del 12.2%
        # sin los 2-8 min de inferencia. El trigger v4.2 NO se modifica.
        avg_conf = (float(np.mean([b.get("confidence", 0) for b in blocks]))
                    if blocks else 0.0)
        # Fase 6: YOLO → Ruta C con gate heurístico (páginas débilmente
        # detectadas) + disable_uocr apaga el detector (benchmark de overhead).
        self.ou._yolo_disabled.set() if disable_uocr else self.ou._yolo_disabled.clear()
        with self._diagnostic_stage("yolo"):
            yolo_blocks, yolo_regiones = self._ruta_c_yolo(
                img_bgr, ocr_lang, blocks, avg_conf)
        if yolo_blocks:
            merged = self.ou._fusionar_blocks_multi(
                [blocks, yolo_blocks],
                weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                         OCR_ENGINE_WEIGHTS["yolo"]],
            )
            print(f"[process-page] Fase 6: {len(blocks)} híbrido + "
                  f"{len(yolo_blocks)} YOLO → {len(merged)}")
            blocks[:] = merged
            engines_used.append("yolo+rutac")
        # ── Fase 6.5: comic-text-detector → Ruta C (texto SIN globo) ──
        # Tier 3.6 (PLAN_MANGA_OCR Paso 4): detecta texto flotante sobre el
        # dibujo, pensamientos y tipografías de arte que híbrido y YOLO
        # pierden. Misma posición pre-trigger que YOLO (si los bloques
        # recuperados elevan la página, el VLM no dispara), con gate en
        # cascada (solo si YOLO no resolvió la página) y dedup de regiones vs
        # YOLO por overlap (lección del benchmark Paso 5: no re-OCRear la
        # misma zona dos veces). disable_uocr lo apaga (mismo patrón).
        self.ou._ctd_disabled.set() if disable_uocr else self.ou._ctd_disabled.clear()
        with self._diagnostic_stage("ctd"):
            ctd_blocks = self._ruta_c_ctd(
                img_bgr, ocr_lang, blocks, avg_conf, yolo_regiones)
        if ctd_blocks:
            merged = self.ou._fusionar_blocks_multi(
                [blocks, ctd_blocks],
                weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                         OCR_ENGINE_WEIGHTS["yolo"]],
            )
            print(f"[process-page] Fase 6.5: {len(blocks)} híbrido+YOLO + "
                  f"{len(ctd_blocks)} CTD → {len(merged)}")
            blocks[:] = merged
            engines_used.append("ctd+rutac")
        with self._diagnostic_stage("panel_check"):
            has_big_panel = self._has_big_panel(img_bgr)
        # Sesión 116: la firma se computa SIEMPRE (no solo con trigger) para
        # que la decisión NEGATIVA también quede cacheada y sea determinista
        # entre corridas idénticas (misma imagen → misma firma → misma
        # decisión, aunque cuDNN varíe len(blocks)/avg_conf del híbrido).
        # Sesión 126: la firma se escopea por DOCUMENTO (doc_id) — prefija la
        # clave para que capítulos de la MISMA serie (94% colisión de layout,
        # sesión 124) no hereden decisiones entre sí. El §8.4.1 de negativas
        # recibe esta misma firma escopeada (vía _reforzar_con_unlimited).
        with self._diagnostic_stage("signature"):
            firma = self._firma_documento(doc_id, self._page_signature(img_bgr))
        trigger = self._trigger_con_cache(
            firma, blocks, avg_conf, has_big_panel,
            force_uocr=force_uocr, disable_uocr=disable_uocr,
        )
        if self._active_diagnostics is not None:
            self._active_diagnostics.set_counts(initial_blocks=len(blocks))
            self._active_diagnostics.set_trigger(
                trigger,
                self._trigger_reason(
                    blocks, avg_conf, has_big_panel,
                    force_uocr=force_uocr,
                ),
            )
        if has_big_panel and not disable_uocr:
            print(f"[process-page] Fusion: panel image grande detectado "
                  f"(dark_ratio>{(UOCR_IMAGE_BLOCK_RATIO * 100):.0f}%)")
        if trigger:
            # §8.4.1: cache de decisiones — si una página con esta firma de
            # layout ya disparó el refuerzo y NO recuperó nada, saltarse el VLM
            # (páginas repetitivas del capítulo). Solo aplica a triggers NO
            # forzados y cuando el cache no esté deshabilitado. (firma ya se
            # computó arriba para la caché del trigger de la sesión 116).
            # Sesión 129: la salvaguarda mucho_mas_debil de la negativa
            # recibe los stats ACTUALES del híbrido — si esta página se
            # detecta mucho más débil que la que registró la negativa, se
            # ignora y se re-dispara el VLM.
            if (firma and not disable_uocr and not force_uocr
                    and self._is_decision_negativa_vigente(
                        firma, len(blocks), avg_conf)):
                print(f"[fusion] §8.4.1: firma {firma[:20]} repetitiva "
                      f"(U-OCR no recuperó antes) — salto de refuerzo")
            else:
                # Fase 2: reintento agresivo de RapidOCR (CPU, ~1.5s) ANTES
                # del VLM (~2-8 min/pág). Si el merge resuelve la página
                # según el trigger v4.2, no se dispara el daemon. Skipeado
                # con force_uocr (orden explícito de VLM) y disable_uocr
                # (el trigger ya está anulado).
                with self._diagnostic_stage("rapid_aggressive"):
                    rapid_salvado = (
                        not force_uocr and not disable_uocr
                        and self._reforzar_con_rapid_agresivo(
                            img_bgr, blocks, avg_conf, has_big_panel)
                    )
                if rapid_salvado:
                    print("[process-page] Fase 2: reintento RapidOCR "
                          "agresivo resolvió la página — sin VLM")
                    engines_used.append("rapid-aggressive")
                    return blocks, "fusion", engines_used
                with self._diagnostic_stage("uocr"):
                    engines_used.extend(self._reforzar_con_unlimited(
                        img_bgr, ocr_lang, blocks, avg_conf, firma=firma))
        return blocks, "fusion", engines_used

    # ─── Trigger v4.2 (testeable en aislamiento) ─────────────────
    def _compute_trigger(
        self,
        blocks: list[dict[str, Any]],
        avg_conf: float,
        has_big_panel: bool,
        force_uocr: bool = False,
        disable_uocr: bool = False,
    ) -> bool:
        """Decide si la página necesita el refuerzo de Unlimited-OCR.

        v4.2: se dispara por detección débil o panel grande sin OCR suficiente.
        Un panel oscuro con al menos tres bloques y confianza media alta se
        considera ya resuelto y no activa el VLM costoso.
        disable_uocr (benchmark) anula cualquier trigger.
        """
        panel_needs_refinement = (
            has_big_panel
            and not (
                len(blocks) >= UOCR_TRIGGER_MIN_BLOCKS
                and avg_conf >= UOCR_PANEL_SKIP_MIN_CONF
            )
        )
        trigger = (not blocks
                   or (len(blocks) < UOCR_TRIGGER_MIN_BLOCKS
                       and avg_conf < UOCR_TRIGGER_CONF)
                   or panel_needs_refinement
                   or force_uocr)
        if disable_uocr:
            trigger = False
        return trigger

    @staticmethod
    def _trigger_reason(
        blocks: list[dict[str, Any]],
        avg_conf: float,
        has_big_panel: bool,
        force_uocr: bool,
    ) -> str:
        if force_uocr:
            return "forced"
        if not blocks:
            return "no_blocks"
        if (has_big_panel
                and not (len(blocks) >= UOCR_TRIGGER_MIN_BLOCKS
                         and avg_conf >= UOCR_PANEL_SKIP_MIN_CONF)):
            return "large_image_panel"
        if len(blocks) < UOCR_TRIGGER_MIN_BLOCKS and avg_conf < UOCR_TRIGGER_CONF:
            return "low_block_count_and_confidence"
        return "threshold_not_met"

    # ─── Helpers ─────────────────────────────────────────────────
    def _firma_documento(self, doc_id: str, firma: str) -> str:
        """Escopea la firma de layout por documento (sesión 126).

        La sesión 124 midió 94% de colisión de firma entre capítulos de la
        MISMA serie (el layout 8×8 de oscuridad no discrimina paneles
        similares). Prefijar la clave con un identificador de PDF/sesión hace
        que el capítulo 47 NUNCA herede las decisiones de cache del 43
        (trigger sesión 116 + §8.4.1 negativas). doc_id vacío → firma sin
        prefijo (comportamiento legacy: un solo scope compartido).
        """
        if not doc_id or not firma:
            return firma
        return f"{doc_id}:{firma}"

    def _has_big_panel(self, img_bgr: Any) -> bool:
        """Heurística barata: panel image grande (arte oscuro domina)."""
        try:
            return self.ou._page_has_large_image_panel(
                img_bgr, UOCR_IMAGE_BLOCK_RATIO)
        except Exception:
            return False

    def _unlimited_ocr(
        self, img_bgr: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """Llama al daemon U-OCR. Acceso en runtime a routes.api para que
        los mocks que parchean routes.api._ocr_with_unlimited sigan
        funcionando (evita dependencia circular en el import)."""
        import routes.api as ra
        return ra._ocr_with_unlimited(img_bgr)

    def _unlimited_ocr_batch(
        self, img_bgrs: list[Any],
    ) -> tuple[list[tuple[list[dict[str, Any]], list[dict[str, Any]]]], float]:
        """Llama al daemon U-OCR en MODO BATCH (infer_multi, Fase 1). Acceso
        en runtime a routes.api para respetar los mocks de pytest."""
        import routes.api as ra
        return ra._ocr_with_unlimited_batch(img_bgrs)

    def _reforzar_con_unlimited(
        self,
        img_bgr: Any,
        ocr_lang: str,
        blocks: list[dict[str, Any]],
        avg_conf: float,
        firma: str | None = None,
    ) -> list[str]:
        """Refuerza con U-OCR + Ruta C (re-OCR de globos) y fusiona.

        Retorna los motores adicionales usados (["unlimited"] si el daemon
        respondió y aportó bloques). Si el daemon no está disponible, degrada
        al híbrido. Si NO aportó nada (0 bloques nuevos), registra la decisión
        negativa en el cache §8.4.1 para que páginas repetitivas con la misma
        firma no re-disparen la inferencia VLM.
        """
        # Gate global UOCR_ENABLED (sesión 143): anula SOLO el refuerzo VLM
        # (el CLI manga_ocr.py lo desactiva por defecto — extracción pura sin
        # inferencias de 2-8 min/pág). A diferencia de disable_uocr, YOLO /
        # Ruta C / cls de rotación siguen activos. MODO_CPU (preset sin GPU
        # dedicada) apaga el VLM de la misma forma aunque UOCR_ENABLED=True.
        # Se leen en runtime para que mutar config desde el caller surta efecto.
        from config import MODO_CPU, UOCR_ENABLED
        if not UOCR_ENABLED or MODO_CPU:
            return []
        # Cache de recuperación positiva (plan §11 P1): si esta firma ya
        # recuperó bloques en una corrida anterior (TTL 7 días) y la página
        # actual se detecta de forma comparable, reinyectar la recuperación
        # SIN llamar al daemon — la inferencia VLM (30-190 s/pág) no se paga
        # de nuevo en re-corridas del mismo documento. El daemon sigue siendo
        # la fuente de verdad en el primer run; este cache solo evita
        # re-pagar lo ya resuelto. Un re-run con el VLM activo sobreescribe.
        cached: tuple[list[Any], list[Any]] | None = None
        # Stats del híbrido ANTES de la fusión: la salvaguarda mucho_mas_debil
        # del cache positivo compara la DETECCIÓN híbrida (lo que el VLM
        # podría leer de nuevo), no el resultado ya fusionado.
        n_hybrid = len(blocks)
        if firma:
            cached = self._get_pos_cache(firma, n_hybrid, avg_conf)
        if cached is not None:
            ublocks, uimage_panels = cached
        else:
            try:
                ublocks, uimage_panels, _ = self._unlimited_ocr(img_bgr)
            except RuntimeError as uerr:
                print(f"[process-page] Fusión: U-OCR no disponible ({uerr}); "
                      f"solo híbrido")
                return []
        try:
            # ── Ruta C: re-OCR a nivel de GLOBO (OpenCV blobs + EasyOCR 3.5x) ──
            bubble_blocks = self._ruta_c_globos(img_bgr, ocr_lang, blocks, uimage_panels)
            combined_u = ublocks + bubble_blocks
            if combined_u:
                merged = self.ou._fusionar_blocks_multi(
                    [blocks, combined_u],
                    weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                             OCR_ENGINE_WEIGHTS["unlimited"]],
                )
                print(f"[process-page] Fusión: {len(blocks)} híbrido + "
                      f"{len(combined_u)} U-OCR (incl. {len(bubble_blocks)} "
                      f"bubble-recovery) → {len(merged)} (conf_avg="
                      f"{avg_conf:.2f} → disparó refuerzo)")
                # Asignación POR REBANADA (no blocks = merged): este método
                # corre en un scope anidado; un rebind no sería visible para
                # _run_fusion, que retorna la lista original. La mutación
                # in-place garantiza que el endpoint reciba los bloques fusionados.
                blocks[:] = merged
                # Sesión 134: la recuperación refuta la negativa anterior (si
                # la había) — las gemelas deben volver a intentar el VLM.
                if firma:
                    self._limpiar_decision_negativa(firma)
                    # Plan §11 P1: solo guardar la recuperación cacheable si
                    # vino del daemon (un hit del cache no necesita re-guardar).
                    if cached is None:
                        self._put_pos_cache(
                            firma, n_hybrid, avg_conf,
                            ublocks, uimage_panels,
                        )
                return ["unlimited"]
            # §8.4.1: el refuerzo NO recuperó nada → cachear decisión negativa
            # (sesión 129: con los stats del híbrido para la salvaguarda).
            if firma:
                self._registrar_decision_negativa(
                    firma, len(blocks), avg_conf)
            return []
        except RuntimeError as uerr:
            print(f"[process-page] Fusión: U-OCR no disponible ({uerr}); "
                  f"solo híbrido")
            return []

    def _reforzar_con_rapid_agresivo(
        self,
        img_bgr: Any,
        blocks: list[dict[str, Any]],
        avg_conf: float,
        has_big_panel: bool,
    ) -> bool:
        """Fase 2: reintento de RapidOCR con parámetros agresivos (pre-VLM).

        Cuando la confianza media del híbrido es baja, el texto se detectó
        débilmente y el reintento con box_thresh/unclip_ratio más agresivos
        (CPU, ~1.5s) puede recuperar bloques que el umbral default descartó
        — evitando la inferencia VLM completa (~2-8 min/pág).

        NO aplica cuando:
        - has_big_panel: el diálogo incrustado en arte solo lo lee el VLM
          (el reintento de detección no recorta paneles).
        - avg_conf >= RAPID_RETRY_MAX_CONF: la página ya se detectó bien; el
          trigger entonces es por panel grande o fuerza explícita, no por
          texto débil.

        Retorna True SOLO si el merge resuelve la página con margen
        (>= UOCR_TRIGGER_MIN_BLOCKS bloques Y conf >= RAPID_RETRY_SALVADO_CONF,
        por encima del trigger v4.2 para no saltarse el VLM en páginas
        marginales) y muta blocks in-place con los bloques fusionados (mismo
        patrón que _reforzar_con_unlimited: la mutación por rebanada es
        visible para _run_fusion, que retorna la lista original).
        """
        if has_big_panel or avg_conf >= RAPID_RETRY_MAX_CONF:
            return False
        try:
            img_rapid = self.ou._preprocess_rapid(img_bgr)
            rapid = self.ou._run_rapidocr(
                img_rapid,
                box_thresh=RAPID_AGGRESSIVE_PARAMS["box_thresh"],
                unclip_ratio=RAPID_AGGRESSIVE_PARAMS["unclip_ratio"],
                text_score=RAPID_AGGRESSIVE_PARAMS["text_score"],
            )
            if not rapid:
                return False
            merged = self.ou._fusionar_blocks_multi(
                [blocks, rapid],
                weights=[OCR_ENGINE_WEIGHTS["easyocr"],
                         OCR_ENGINE_WEIGHTS["rapid"]],
            )
            if not merged:
                return False
            merged_avg_conf = float(np.mean(
                [b.get("confidence", 0) for b in merged]))
            # Conf > trigger v4.2 con MARGEN (0.30 no 0.20): promediar todos
            # los bloques (incluidos los híbridos débiles) con un bloque
            # fuerte cruza 0.2 con facilidad — 0.30 exige mejora clara.
            salvado = (len(merged) >= UOCR_TRIGGER_MIN_BLOCKS
                       and merged_avg_conf >= RAPID_RETRY_SALVADO_CONF)
            print(f"[process-page] Fase 2: reintento RapidOCR agresivo "
                  f"({len(blocks)} → {len(merged)} bloques, conf "
                  f"{avg_conf:.2f}→{merged_avg_conf:.2f}, salvado={salvado})")
            if salvado:
                blocks[:] = merged
            return salvado
        except Exception as e:
            print(f"[process-page] Fase 2: reintento RapidOCR agresivo "
                  f"falló: {e}")
            return False

    # ─── Cache de decisiones §8.4.1 ──────────────────────────────
    def _page_signature(self, img_bgr: Any) -> str:
        """Firma de layout de la página (distribución espacial de oscuridad)."""
        try:
            return self.ou._page_signature(img_bgr)
        except Exception:
            return ""

    def _is_decision_negativa_vigente(
        self, firma: str, n_blocks: int = 0, avg_conf: float = 0.0,
    ) -> bool:
        """True si una página con esta firma ya disparó U-OCR sin recuperar
        nada y la decisión sigue dentro del TTL.

        Sesión 128: un hit dentro del TTL refresca el timestamp (mismo touch
        LRU que el trigger) — una corrida larga no expira la negativa a
        mitad de capítulo. El refresh se persiste solo si
        UOCR_NEG_CACHE_PERSIST está activo.

        Sesión 129 (salvaguarda mucho_mas_debil, misma que el trigger): la
        negativa se registró cuando el híbrido detectó la página con
        n_c/conf_c. Si la página ACTUAL se detecta MUCHO más débil (menos
        bloques Y confianza sustancialmente menor), la negativa NO aplica:
        el diálogo artístico que el híbrido ahora pierde es justo el que el
        VLM podría recuperar → se ignora la negativa y se re-dispara. La
        firma es de LAYOUT, no de contenido; sin esta salvaguarda una
        página gemela con el mismo layout pero contenido distinto heredaría
        la supresión.

        Sesión 134 (salvaguarda de detección débil, caso p5): si la
        negativa se registró con una detección híbrida DEMASIADO POBRE
        (< UOCR_NEG_WEAK_MAX_BLOCKS bloques O conf < UOCR_NEG_WEAK_MIN_CONF),
        la negativa no es fiable — una página gemela con detección
        COMPARABLE (que el much_mas_debil no libera) puede re-disparar el VLM
        hasta UOCR_NEG_MAX_REINTENTOS veces por firma (contador persistido),
        cubriendo el diálogo artístico que la variación cuDNN del híbrido
        pierde. Al agotar el contador, la negativa se congela (el VLM ya
        tuvo su oportunidad). Detección fuerte (>= N bloques y conf alta) →
        se congela directamente, como antes de la sesión 134.
        """
        vigente = False
        mutado = False
        with self._uocr_cache_lock:
            # 0) Cero confirmado (plan §10.2 item 1): la firma falló >= MIN
            # veces en ventanas TTL distintas dentro del TTL largo (7 días) →
            # suprimir el VLM AUNQUE el TTL corto de 30 min ya expiró (el
            # caso de las págs 13/17 del cap. 43: 0 recuperaciones en todas
            # las corridas, 31-68 s/llamada). Salvaguarda mucho_mas_debil
            # contra los stats de la ÚLTIMA confirmación: si la página actual
            # se detecta MUCHO más débil, el VLM vuelve a correr (el diálogo
            # artístico que el híbrido ahora pierde es justo el que el VLM
            # podría leer). El gate es FIRME frente a la salvaguarda débil
            # (sesión 134): un cero confirmado no re-dispara más.
            cero = self._uocr_neg_ceros.get(firma)
            if cero is not None:
                ts_c, count_c, n_c_c, conf_c_c = cero
                if (count_c >= UOCR_NEG_CERO_MIN
                        and (time.time() - ts_c) < UOCR_NEG_CERO_TTL_S
                        and not (n_blocks < n_c_c
                                 and avg_conf < conf_c_c * 0.8)):
                    print(f"[process-page] VLM: cero confirmado por firma "
                          f"{firma[:16]}… ({count_c} fallos) — saltando "
                          f"inferencia")
                    return True
            entrada = self._uocr_neg_cache.get(firma)
            if entrada is None:
                return False
            ts, n_c, conf_c, re_disparos = entrada
            if (time.time() - ts) >= UOCR_CACHE_TTL_S:
                del self._uocr_neg_cache[firma]
                return False
            mucho_mas_debil = (n_blocks < n_c and avg_conf < conf_c * 0.8)
            if mucho_mas_debil:
                # NO consume el contador (sesión 134): la re-registración tras
                # el fallo baja los stats (n_c/conf_c) y la cadena termina —
                # en el peor caso llega a 0 bloques, donde el much_mas_debil
                # deja de aplicarse y la salvaguarda débil consume el contador
                # hasta congelar. Acotado, no infinito.
                return False
            if (n_c < UOCR_NEG_WEAK_MAX_BLOCKS
                    or conf_c < UOCR_NEG_WEAK_MIN_CONF) \
                    and re_disparos < UOCR_NEG_MAX_REINTENTOS:
                # Salvaguarda de detección débil (sesión 134): la negativa
                # viene de una detección pobre → permitir el re-disparo (una
                # vez por firma, contador) en vez de congelar a ciegas.
                self._uocr_neg_cache[firma] = (
                    time.time(), n_c, conf_c, re_disparos + 1)
                mutado = True
                vigente = False  # no vigente → el caller re-dispara el VLM
            else:
                # Touch LRU (sesión 128): la ventana se cuenta desde AHORA.
                self._uocr_neg_cache[firma] = (
                    time.time(), n_c, conf_c, re_disparos)
                mutado = True
                vigente = True
        if mutado and UOCR_NEG_CACHE_PERSIST:
            self._persistir_cache()
        return vigente

    def _registrar_decision_negativa(
        self, firma: str, n_blocks: int = 0, avg_conf: float = 0.0,
    ) -> None:
        """Registra que el refuerzo U-OCR no recuperó nada para esta firma.
        Sesión 129: guarda también los stats de detección (n_blocks,
        avg_conf) para que la salvaguarda mucho_mas_debil pueda comparar.
        ⚠️ SIEMPRE pasar los stats (n_blocks, avg_conf) del híbrido actual:
        con los defaults (0, 0.0) la salvaguarda queda DESACTIVADA para esa
        entrada (n_blocks < 0 nunca se cumple → supresión ciega como antes
        de la sesión 129). Eviction LRU cuando el cache supera el máximo.
        Sesión 134: si la firma ya tenía una negativa, su contador de
        re-disparos se PRESERVA (no se resetea) — si un re-disparo de la
        salvaguarda de detección débil vuelve a fallar, la firma queda
        congelada al agotar el contador (evita el bucle re-dispara→falla→
        re-registra→re-dispara infinito)."""
        now = time.time()
        with self._uocr_cache_lock:
            # Ledger de ceros confirmados (plan §10.2 item 1): cada registro
            # de negativa (el VLM corrió y no recuperó nada) incrementa el
            # contador de la firma — la señal de "este layout no aporta"
            # que, al alcanzar UOCR_NEG_CERO_MIN en ventanas TTL distintas,
            # extiende la supresión con TTL largo (7 días) vía
            # _is_decision_negativa_vigente. Los stats se actualizan a los de
            # ESTA confirmación (el par más reciente para la salvaguarda
            # mucho_mas_debil). Podado por TTL largo + LRU (mismo patrón que
            # el cache de negativas) para no crecer sin límite.
            prev_cero = self._uocr_neg_ceros.get(firma)
            count_cero = (prev_cero[1] + 1) if prev_cero else 1
            self._uocr_neg_ceros[firma] = (now, count_cero, n_blocks, avg_conf)
            if len(self._uocr_neg_ceros) > UOCR_CACHE_MAX_ENTRIES:
                while len(self._uocr_neg_ceros) > UOCR_CACHE_MAX_ENTRIES:
                    oldest = min(
                        self._uocr_neg_ceros,
                        key=lambda k: self._uocr_neg_ceros[k][0],
                    )
                    del self._uocr_neg_ceros[oldest]
            prev = self._uocr_neg_cache.get(firma)
            re_disparos = prev[3] if prev else 0
            self._uocr_neg_cache[firma] = (now, n_blocks, avg_conf, re_disparos)
            # LRU simple: si hay más entradas que el máximo, eliminar la más
            # antigua (min por timestamp) hasta caber.
            if len(self._uocr_neg_cache) > UOCR_CACHE_MAX_ENTRIES:
                while len(self._uocr_neg_cache) > UOCR_CACHE_MAX_ENTRIES:
                    oldest = min(
                        self._uocr_neg_cache,
                        key=lambda k: self._uocr_neg_cache[k][0],
                    )
                    del self._uocr_neg_cache[oldest]
        # Las negativas solo se persisten si UOCR_NEG_CACHE_PERSIST=True
        # (sesión 127): por defecto viven en memoria — optimización
        # intra-corrida, y su persistencia amplificaría la supresión de VLM
        # sin salvaguarda (trade-off completo en config.py).
        if UOCR_NEG_CACHE_PERSIST:
            self._persistir_cache()

    def _limpiar_decision_negativa(self, firma: str) -> None:
        """Borra la negativa §8.4.1 de la firma (sesión 134).

        Se llama cuando el refuerzo U-OCR RECUPERA algo (o el daemon cae) en
        una página cuya firma tenía una negativa: la recuperación refuta la
        decisión anterior, así que las gemelas deben volver a intentar el VLM
        en vez de honrar una negativa obsoleta. Sin esto, la secuencia
        "negativa débil → re-disparo (sesión 134) → éxito" dejaría la negativa
        (ahora estale) congelando a las gemelas posteriores."""
        with self._uocr_cache_lock:
            existe = firma in self._uocr_neg_cache
            if existe:
                del self._uocr_neg_cache[firma]
            # Sesión 2026-08-16 (plan §10.2 item 1): la recuperación también
            # refuta el CERO CONFIRMADO — si el VLM SÍ recuperó algo en esta
            # página, el ledger no debe seguir suprimiendo a las gemelas.
            en_ceros = firma in self._uocr_neg_ceros
            if en_ceros:
                del self._uocr_neg_ceros[firma]
        if (existe or en_ceros) and UOCR_NEG_CACHE_PERSIST:
            self._persistir_cache()

    # ── Cache de RECUPERACIÓN POSITIVA (plan §11 P1, 2026-08-17) ──

    def _get_pos_cache(
        self, firma: str, n_blocks: int = 0, avg_conf: float = 0.0,
    ) -> tuple[list[Any], list[Any]] | None:
        """Devuelve la recuperación VLM cacheada (ublocks, uimage_panels)
        para esta firma, o None si no hay entrada vigente.

        Complemento simétrico de _is_decision_negativa_vigente: el ledger de
        ceros suprime páginas que SIEMPRE fallan; este cache reinyecta
        páginas que SIEMPRE recuperan. El determinismo 5/5 de la
        recuperación por página (plan §4.6 tabla ROI) hace que cachear sea
        seguro: misma firma dHash → mismo resultado VLM. Aplica la MISMA
        salvaguarda mucho_mas_debil que los ceros: si la página actual se
        detecta MUCHO más débil (menos bloques Y confianza <<) que cuando se
        cacheó, la entrada NO aplica — el diálogo artístico que el híbrido
        ahora pierde es justo el que el VLM podría leer, así que se re-corre.
        No aplica con force_uocr/disable_uocr (lo decide el caller pasando
        n_blocks/avg_conf reales; los modos benchmark pasan sin consultar)."""
        with self._uocr_cache_lock:
            entrada = self._uocr_pos_cache.get(firma)
            if entrada is None:
                return None
            ts, n_c, conf_c, ublocks, uimage_panels = entrada
            if (time.time() - ts) >= UOCR_POS_CACHE_TTL_S:
                del self._uocr_pos_cache[firma]
                return None
            mucho_mas_debil = (
                n_blocks < n_c
                and avg_conf < conf_c * UOCR_POS_CACHE_SALVAGUARDA
            )
            if mucho_mas_debil:
                return None
            print(f"[process-page] VLM: recuperación cacheada por firma "
                  f"{firma[:16]}… — reinyectando {len(ublocks)} bloques")
            return ublocks, uimage_panels

    def _put_pos_cache(
        self,
        firma: str,
        n_blocks: int,
        avg_conf: float,
        ublocks: list[Any],
        uimage_panels: list[Any],
    ) -> None:
        """Guarda la recuperación VLM positiva de esta firma (ublocks +
        uimage_panels) con TTL largo (7 días) y eviction LRU.

        Solo se llama cuando el VLM SÍ recuperó bloques (combined_u no vacío).
        Un re-run con el daemon activo sobreescribe la entrada (los bloques
        nuevos de esta corrida son la verdad más reciente). El cache vive en
        la sección "pos" del archivo de decisiones y viaja con la
        persistencia: un proceso nuevo (servidor reiniciado) reinyecta las
        recuperaciones sin volver a pagar la inferencia."""
        now = time.time()
        with self._uocr_cache_lock:
            self._uocr_pos_cache[firma] = (
                now, n_blocks, avg_conf, ublocks, uimage_panels)
            if len(self._uocr_pos_cache) > UOCR_CACHE_MAX_ENTRIES:
                while len(self._uocr_pos_cache) > UOCR_CACHE_MAX_ENTRIES:
                    oldest = min(
                        self._uocr_pos_cache,
                        key=lambda k: self._uocr_pos_cache[k][0],
                    )
                    del self._uocr_pos_cache[oldest]
        self._persistir_cache()

    @classmethod
    def _cargar_cache_disco(cls, force: bool = False) -> None:
        """Carga los caches de decisiones persistidos (una vez por proceso).

        force=True re-lee el archivo (tests que simulan un proceso nuevo).
        Las entradas fuera de TTL se descartan en la carga; un archivo
        corrupto se elimina y se parte de cero (nunca rompe el OCR).
        """
        with cls._cache_load_lock:
            if cls._cache_cargado and not force:
                return
            cls._cache_cargado = True
            path = _DECISION_CACHE_PATH
            if not path.exists():
                return
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Sesión 126: versión 2 = claves escopeadas por documento
                # ("doc_id:firma"). Las claves v1 (firma bruta, sin prefijo)
                # nunca matchearían los lookups escopeados — cargarlas solo
                # ensucia el archivo hasta el TTL. Con version mismatch se
                # descarta TODO el archivo y se parte de cero (clean start).
                if data.get("version") != _DECISION_CACHE_VERSION:
                    print(f"[ocr_engine] cache de decisiones v{data.get('version')} "
                          f"(esperaba v{_DECISION_CACHE_VERSION}) — formato "
                          f"desactualizado, recreando desde cero")
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return
                ahora = time.time()
                with cls._trigger_dec_lock:
                    trigger = data.get("trigger", {}) or {}
                    for firma, entry in trigger.items():
                        try:
                            # Sesión 136: las entradas del trigger llevan
                            # además el contador de recomputes [ts, n, conf,
                            # decision, re_computes] — se conserva al cargar
                            # para que el determinismo entre servidores no
                            # regale el recompute de la salvaguarda débil.
                            ts, n_bloques, conf, decision, re_computes = entry
                            if ahora - ts < TRIGGER_CACHE_TTL_S:
                                cls._trigger_dec_cache[firma] = (
                                    ts, int(n_bloques), float(conf),
                                    bool(decision), int(re_computes))
                        except (TypeError, ValueError):
                            continue
                    # Cap defensivo (LRU por timestamp) si el archivo viniera
                    # más grande que el máximo:
                    if len(cls._trigger_dec_cache) > TRIGGER_CACHE_MAX_ENTRIES:
                        for firma in sorted(
                            cls._trigger_dec_cache,
                            key=lambda k: cls._trigger_dec_cache[k][0],
                        )[:len(cls._trigger_dec_cache) - TRIGGER_CACHE_MAX_ENTRIES]:
                            del cls._trigger_dec_cache[firma]
                # Sesión 127: las negativas §8.4.1 solo se cargan del disco si
                # el flag UOCR_NEG_CACHE_PERSIST está activo (default True
                # desde la sesión 129 — la salvaguarda mucho_mas_debil hace
                # segura su persistencia). Sesión 129: formato por entrada
                # [ts, n_blocks, avg_conf] (la salvaguarda compara contra los
                # stats guardados, no solo el timestamp). Sesión 134: formato
                # [ts, n_blocks, avg_conf, re_disparos] — el contador de la
                # salvaguarda de detección débil viaja por la persistencia
                # (un servidor nuevo respeta los re-disparos ya consumidos).
                if UOCR_NEG_CACHE_PERSIST:
                    with cls._uocr_cache_lock:
                        neg = data.get("neg", {}) or {}
                        for firma, entry in neg.items():
                            try:
                                ts, n_blocks, conf, re_disparos = entry
                                if ahora - float(ts) < UOCR_CACHE_TTL_S:
                                    cls._uocr_neg_cache[firma] = (
                                        float(ts), int(n_blocks), float(conf),
                                        int(re_disparos))
                            except (TypeError, ValueError):
                                continue
                        if len(cls._uocr_neg_cache) > UOCR_CACHE_MAX_ENTRIES:
                            for firma in sorted(
                                cls._uocr_neg_cache,
                                key=lambda k: cls._uocr_neg_cache[k][0],
                            )[:len(cls._uocr_neg_cache) - UOCR_CACHE_MAX_ENTRIES]:
                                del cls._uocr_neg_cache[firma]
                        # Ledger de ceros confirmados (plan §10.2 item 1):
                        # [ts, count, n_blocks, avg_conf], podado por el TTL
                        # LARGO (7 días) + cap LRU. Se carga solo con el flag
                        # de persistencia activo (misma política que "neg").
                        ceros = data.get("ceros", {}) or {}
                        for firma, entry in ceros.items():
                            try:
                                ts, count, n_blocks, conf = entry
                                if ahora - float(ts) < UOCR_NEG_CERO_TTL_S:
                                    cls._uocr_neg_ceros[firma] = (
                                        float(ts), int(count),
                                        int(n_blocks), float(conf))
                            except (TypeError, ValueError):
                                continue
                        if len(cls._uocr_neg_ceros) > UOCR_CACHE_MAX_ENTRIES:
                            for firma in sorted(
                                cls._uocr_neg_ceros,
                                key=lambda k: cls._uocr_neg_ceros[k][0],
                            )[:len(cls._uocr_neg_ceros) - UOCR_CACHE_MAX_ENTRIES]:
                                del cls._uocr_neg_ceros[firma]
                        # Cache de recuperación positiva (plan §11 P1):
                        # [ts, n_blocks, avg_conf, ublocks, uimage_panels]
                        # (listas de dicts serializables), podado por el TTL
                        # largo (7 días) + cap LRU. Misma política de
                        # persistencia que "neg"/"ceros".
                        pos = data.get("pos", {}) or {}
                        for firma, entry in pos.items():
                            try:
                                ts, n_blocks, conf, ublocks, panels = entry
                                if ahora - float(ts) < UOCR_POS_CACHE_TTL_S:
                                    cls._uocr_pos_cache[firma] = (
                                        float(ts), int(n_blocks),
                                        float(conf),
                                        list(ublocks or []),
                                        list(panels or []))
                            except (TypeError, ValueError):
                                continue
                        if len(cls._uocr_pos_cache) > UOCR_CACHE_MAX_ENTRIES:
                            for firma in sorted(
                                cls._uocr_pos_cache,
                                key=lambda k: cls._uocr_pos_cache[k][0],
                            )[:len(cls._uocr_pos_cache) - UOCR_CACHE_MAX_ENTRIES]:
                                del cls._uocr_pos_cache[firma]
            except Exception as e:
                print(f"[ocr_engine] cache de decisiones corrupto, "
                      f"recreando desde cero: {e}")
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @classmethod
    def _persistir_cache(cls) -> None:
        """Persiste los caches de decisiones en disco.

        El trigger (sesión 125) siempre; las negativas §8.4.1 SOLO si
        UOCR_NEG_CACHE_PERSIST=True (sesión 127). Escritura atómica bajo
        _DISK_LOCK; snapshot de los dicts bajo sus locks de memoria. El
        archivo es pequeño (≤256 entradas c/u) y el dump es ~ms, así que
        escribir tras cada mutación es barato.
        """
        with cls._trigger_dec_lock:
            trigger_snapshot = dict(cls._trigger_dec_cache)
        data = {"version": _DECISION_CACHE_VERSION, "trigger": trigger_snapshot}
        if UOCR_NEG_CACHE_PERSIST:
            with cls._uocr_cache_lock:
                data["neg"] = dict(cls._uocr_neg_cache)
                # Plan §10.2 item 1: el ledger de ceros confirmados viaja con
                # la persistencia — 2 corridas en procesos separados acumulan
                # los fallos de la misma página (firma dHash estable).
                data["ceros"] = dict(cls._uocr_neg_ceros)
                # Plan §11 P1: la recuperación positiva viaja también — un
                # proceso nuevo (servidor reiniciado) reinyecta las páginas
                # resueltas sin re-pagar la inferencia VLM.
                data["pos"] = dict(cls._uocr_pos_cache)
        path = _DECISION_CACHE_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            print(f"[ocr_engine] aviso: no se pudo persistir el cache de "
                  f"decisiones: {e}")

    @classmethod
    def clear_decision_cache(cls) -> None:
        """Limpia los caches de decisiones (tests, nueva sesión): el §8.4.1
        de negativas y el de decisión del trigger por firma (sesión 116).
        También elimina el archivo persistido (sesión 125) — una sesión nueva
        arranca sin decisiones heredadas de capítulos anteriores."""
        with cls._uocr_cache_lock:
            cls._uocr_neg_cache.clear()
            # Plan §10.2 item 1: el ledger de ceros confirmados también se
            # limpia — una sesión nueva arranca sin supresiones heredadas.
            cls._uocr_neg_ceros.clear()
            # Plan §11 P1: la recuperación positiva también — una sesión
            # nueva re-procesa el VLM desde cero (sin reinyecciones viejas).
            cls._uocr_pos_cache.clear()
        with cls._trigger_dec_lock:
            cls._trigger_dec_cache.clear()
        cls._cache_cargado = False
        try:
            _DECISION_CACHE_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    def _ruta_c_globos(
        self,
        img_bgr: Any,
        ocr_lang: str,
        blocks: list[dict[str, Any]],
        uimage_panels: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-OCR a nivel de globo dentro de paneles image (Ruta C).

        El benchmark (PLAN_FUSION_OCR.md §3.5) demostró que recortar el panel
        image COMPLETO no recupera el diálogo artístico — la única granularidad
        que funciona es el GLOBO individual (upscale 3-4×).
        """
        bubble_blocks: list[dict[str, Any]] = []
        try:
            regions: list[dict[str, Any]] = []
            # 1) Globos dentro de los paneles image del daemon
            for _panel in uimage_panels:
                regions.extend(
                    self.ou._detect_bubble_regions_in_panel(img_bgr, _panel))
            # 2) Globos a página completa (cubre el caso pág. 12 donde el
            #    diálogo artístico está FUERA de los rects del daemon)
            _ph, _pw = img_bgr.shape[:2]
            regions.extend(self.ou._detect_bubble_regions_in_panel(
                img_bgr, {"x": 0, "y": 0, "w": _pw, "h": _ph}))
            # 3) Descartar globos ya cubiertos por bloques híbridos:
            #    solo re-OCR los globos que representan diálogo perdido.
            if regions and blocks:
                regions = [
                    r for r in regions
                    if not any(self.ou._overlap_ratio(r, b) > 0.5 for b in blocks)
                ]
            if regions:
                bubble_blocks = self.ou._recover_regions_with_easyocr(
                    img_bgr, regions, ocr_lang, upscale=3.5,
                    hybrid_blocks=blocks)
                print(f"[process-page] Ruta C: {len(uimage_panels)} paneles + "
                      f"full-page → {len(regions)} globos → "
                      f"{len(bubble_blocks)} bloques recuperados")
        except Exception as cerr:
            print(f"[process-page] Ruta C (bubble re-OCR) falló: {cerr}")
        return bubble_blocks
