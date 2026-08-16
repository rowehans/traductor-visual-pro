from pathlib import Path

from routes.main import _is_within


def test_path_security_rechaza_prefijo_de_directorio_homonimo(tmp_path):
    root = tmp_path / "app"
    sibling = tmp_path / "app_evil" / "secret.txt"
    root.mkdir()
    sibling.parent.mkdir()
    sibling.write_text("secret", encoding="utf-8")

    assert _is_within(root, root / "index.html") is True
    assert _is_within(root, sibling) is False
