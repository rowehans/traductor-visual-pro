from cache import _make_key
from pathlib import Path
from unittest.mock import patch


def test_cache_key_normaliza_espacios_y_codigos():
    assert _make_key("Hola mundo", "es", "en") == _make_key(
        "  Hola   mundo ", "ES", "EN")


def test_cache_escritura_atomica_y_lectura(tmp_path, monkeypatch):
    """Una escritura no debe dejar un JSON parcialmente escrito."""
    import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    with patch.object(cache.os, "replace", wraps=cache.os.replace) as replace:
        cache.set("Hola", "es", "en", "Hello")

    replace.assert_called_once()
    assert cache.get("Hola", "es", "en") == "Hello"
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_reutiliza_lectura_en_memoria(tmp_path, monkeypatch):
    """Un hit repetido no debe volver a abrir el JSON persistido."""
    import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.set("Memoria", "es", "en", "Memory")
    cache._memory_cache.clear()
    cache._memory_chars = 0
    path = tmp_path / f"{cache._make_key('Memoria', 'es', 'en')}.json"

    with patch.object(
        Path, "read_text", autospec=True, side_effect=Path.read_text
    ) as read_text:
        assert cache.get("Memoria", "es", "en") == "Memory"
        assert cache.get("Memoria", "es", "en") == "Memory"

    assert read_text.call_count == 1


def test_cache_miss_no_hace_exists_separado(tmp_path, monkeypatch):
    """Un miss debe resolverse con una sola lectura fallida."""
    import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache._memory_cache.clear()
    cache._memory_chars = 0

    with patch.object(Path, "exists", side_effect=AssertionError("exists extra")):
        assert cache.get("Ausente", "es", "en") is None
