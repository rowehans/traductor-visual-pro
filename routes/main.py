"""
routes/main.py — Blueprint para rutas estaticas.
"""
from pathlib import Path

from flask import Blueprint, Response, send_from_directory

from config import ROOT, DIST, IS_PRODUCTION

main_bp = Blueprint("main", __name__)


def _is_within(root: Path, target: Path) -> bool:
    """True si target estÃ¡ dentro de root, sin falsos positivos de prefijo."""
    try:
        root.resolve()
        target.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


@main_bp.get("/")
def index() -> Response:
    if IS_PRODUCTION:
        return send_from_directory(DIST, "index.html")
    return send_from_directory(ROOT, "index.html")


@main_bp.get("/<path:path>")
def static_files(path: str) -> Response:
    if IS_PRODUCTION:
        prod_target = (DIST / path).resolve()
        if _is_within(DIST, prod_target) and prod_target.exists():
            return send_from_directory(DIST, path)
    target = (ROOT / path).resolve()
    if not _is_within(ROOT, target) or not target.exists():
        return Response("Not found", 404)
    return send_from_directory(ROOT, path)
