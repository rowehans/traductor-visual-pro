"""
conftest.py — Fixtures compartidos para tests del Traductor Visual Pro.
"""

import sys
import os

import pytest

# Asegurar que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _cache_decisiones_tmp(tmp_path, monkeypatch):
    """Aísla el cache persistido de decisiones (ocr_engine, sesión 125) a un
    directorio temporal por test — evita escribir/borrar el archivo real
    cache/ocr_decision_cache.json del proyecto durante la suite."""
    monkeypatch.setattr(
        "ocr_engine._DECISION_CACHE_PATH",
        tmp_path / "ocr_decision_cache.json",
    )
