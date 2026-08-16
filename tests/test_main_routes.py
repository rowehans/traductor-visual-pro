"""
test_main_routes.py — Cobertura del blueprint de rutas estáticas (routes/main.py).

Prueba _is_within (path traversal), index (dev vs producción) y static_files
(servir archivo, 404 por inexistente o por salirse del root).
"""
from pathlib import Path

import pytest
from flask import Flask

from routes.main import main_bp, _is_within


def _make_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.register_blueprint(main_bp)
    return app


class TestIsWithin:
    def test_dentro_devuelve_true(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        target = root / "sub" / "archivo.txt"
        root.mkdir()
        (root / "sub").mkdir()
        (root / "sub" / "archivo.txt").touch()
        assert _is_within(root, target) is True

    def test_fuera_devuelve_false(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        root.mkdir()
        fuera = tmp_path / "fuera.txt"
        fuera.touch()
        assert _is_within(root, fuera) is False

    def test_prefijo_similar_no_es_dentro(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        root.mkdir()
        # "proj2" comparte prefijo con "proj" pero está fuera
        hermano = tmp_path / "proj2" / "x.txt"
        hermano.parent.mkdir()
        hermano.touch()
        assert _is_within(root, hermano) is False


class TestIndex:
    def test_dev_sirve_index_de_root(self, monkeypatch: pytest.MonkeyPatch,
                                    tmp_path: Path) -> None:
        from routes import main as main_mod
        (tmp_path / "index.html").write_text("<html>dev</html>", encoding="utf-8")
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", False)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        rv = app.test_client().get("/")
        assert rv.status_code == 200
        assert b"dev" in rv.data

    def test_produccion_sirve_index_de_dist(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from routes import main as main_mod
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>prod</html>", encoding="utf-8")
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", True)
        monkeypatch.setattr(main_mod, "DIST", dist)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        rv = app.test_client().get("/")
        assert rv.status_code == 200
        assert b"prod" in rv.data


class TestStaticFiles:
    def test_dev_sirve_archivo(self, monkeypatch: pytest.MonkeyPatch,
                               tmp_path: Path) -> None:
        from routes import main as main_mod
        (tmp_path / "app.js").write_text("console.log(1)", encoding="utf-8")
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", False)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        rv = app.test_client().get("/app.js")
        assert rv.status_code == 200
        assert b"console.log" in rv.data

    def test_dev_archivo_inexistente_404(self, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:
        from routes import main as main_mod
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", False)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        assert app.test_client().get("/no_existe_xyz.js").status_code == 404

    def test_dev_bloquea_path_traversal(self, monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
        from routes import main as main_mod
        (tmp_path / "secreto.txt").write_text("secreto", encoding="utf-8")
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", False)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        rv = app.test_client().get("/../secreto.txt")
        assert rv.status_code == 404

    def test_produccion_sirve_archivo_de_dist(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from routes import main as main_mod
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "app.min.js").write_text("min", encoding="utf-8")
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", True)
        monkeypatch.setattr(main_mod, "DIST", dist)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        rv = app.test_client().get("/app.min.js")
        assert rv.status_code == 200
        assert b"min" in rv.data

    def test_produccion_archivo_fuera_de_dist_404(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from routes import main as main_mod
        dist = tmp_path / "dist"
        dist.mkdir()
        # Archivo que existe en el root pero NO en dist: en producción no se sirve
        (tmp_path / "index.html").write_text("x", encoding="utf-8")
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", True)
        monkeypatch.setattr(main_mod, "DIST", dist)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        rv = app.test_client().get("/index.html")
        # No está en dist -> cae al fallback dev (root); está en root -> 200
        assert rv.status_code == 200

    def test_produccion_inexistente_404(self, monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
        from routes import main as main_mod
        dist = tmp_path / "dist"
        dist.mkdir()
        monkeypatch.setattr(main_mod, "IS_PRODUCTION", True)
        monkeypatch.setattr(main_mod, "DIST", dist)
        monkeypatch.setattr(main_mod, "ROOT", tmp_path)

        app = _make_app()
        assert app.test_client().get("/zzz.js").status_code == 404
