"""
conftest.py — Fixtures compartidos para tests del Traductor Visual Pro.
"""

import sys
import os
from types import SimpleNamespace

import pytest

# Asegurar que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stub_module_if_missing(name: str, attrs: dict) -> None:
    """Registra un stub de ``name`` en sys.modules si el módulo real no está
    instalado. El CI de GitHub no instala las deps de OCR pesadas (torch,
    onnxruntime — decisión documentada en pyproject.toml), pero los tests
    de la Ruta C mockean ``torch.cuda.is_available`` y
    ``onnxruntime.InferenceSession``: necesitan que el módulo exista para
    parchearlo. Con el stub esos tests corren igual y la cobertura de los
    módulos de producción (ocr_utils, runtime_diagnostics) se mantiene.
    Localmente el módulo real existe y el stub nunca se registra."""
    try:
        __import__(name)
    except ImportError:
        mod = SimpleNamespace(**attrs)
        sys.modules[name] = mod


# torch: los tests de YOLO y runtime_diagnostics acceden a torch.cuda.* (con
# mock). El stub solo necesita existir; is_available() devuelve False por
# defecto (los tests lo sobrescriben con mocker.patch).
_stub_module_if_missing(
    "torch",
    {
        "cuda": SimpleNamespace(
            is_available=lambda: False,
            current_device=lambda: 0,
            memory_allocated=lambda *a, **k: 0,
            memory_reserved=lambda *a, **k: 0,
            max_memory_allocated=lambda *a, **k: 0,
            mem_get_info=lambda *a, **k: (0, 0),
        ),
        "backends": SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True),
        ),
    },
)

# onnxruntime: el tier comic-text-detector lo mockea por completo (los tests
# no cargan el modelo real); el stub hace que mocker.patch funcione en CI.
_stub_module_if_missing(
    "onnxruntime",
    {"InferenceSession": lambda *a, **k: None},
)


@pytest.fixture(autouse=True)
def _cache_decisiones_tmp(tmp_path, monkeypatch):
    """Aísla el cache persistido de decisiones (ocr_engine, sesión 125) a un
    directorio temporal por test — evita escribir/borrar el archivo real
    cache/ocr_decision_cache.json del proyecto durante la suite."""
    # Ultralytics inicializa settings.json al importarse. En Windows/CI la
    # ruta de perfil del usuario puede ser de solo lectura; aislarla evita que
    # los tests del detector fallen antes de aplicar sus mocks.
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    monkeypatch.setattr(
        "ocr_engine._DECISION_CACHE_PATH",
        tmp_path / "ocr_decision_cache.json",
    )
