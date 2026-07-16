"""
routes/main.py — Blueprint para rutas estaticas.
"""
from flask import Blueprint, send_from_directory

main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    from server import ROOT, DIST, IS_PRODUCTION
    if IS_PRODUCTION:
        return send_from_directory(DIST, "index.html")
    return send_from_directory(ROOT, "index.html")


@main_bp.get("/<path:path>")
def static_files(path: str):
    from server import ROOT, DIST, IS_PRODUCTION
    if IS_PRODUCTION:
        prod_target = (DIST / path).resolve()
        if str(prod_target).startswith(str(DIST)) and prod_target.exists():
            return send_from_directory(DIST, path)
    target = (ROOT / path).resolve()
    if not str(target).startswith(str(ROOT)) or not target.exists():
        return "Not found", 404
    return send_from_directory(ROOT, path)