"""
ratelimit.py — Módulo compartido de rate limiting.
Evita imports circulares entre server.py y routes/.
"""
from typing import Any

from flask import Flask, request
from flask_limiter import Limiter

limiter: Limiter | None = None
RATE_LIMIT_AVAILABLE: bool = False


# ─── Custom key function: desactiva rate limit para localhost ──────────────
def _rate_limit_key() -> str | None:
    """
    Retorna None únicamente para loopback real (127.0.0.1, ::1,
    localhost), lo que desactiva el rate limiting para la propia UI.
    Las redes privadas siguen siendo remotas y conservan el límite.
    """
    ip: str = request.remote_addr or ""
    if ip in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"):
        return None  # Sin límite
    return ip


def init_limiter(app: Flask) -> bool:
    global limiter, RATE_LIMIT_AVAILABLE  # noqa: PLW0603
    try:
        limiter = Limiter(
            app=app,
            # El stub de flask-limiter tipa key_func como Callable[[], str],
            # pero retornar None es el patrón documentado para eximir a
            # loopback del límite (ver _rate_limit_key).
            key_func=_rate_limit_key,  # type: ignore[arg-type]
            default_limits=["2000 per day", "500 per hour"],
            storage_uri="memory://",
        )
        RATE_LIMIT_AVAILABLE = True
        print("[rate] Rate limiting activo (sin límite para localhost)")
        return True
    except Exception as e:
        RATE_LIMIT_AVAILABLE = False
        print(f"[rate] Rate limiting no disponible: {e}")
        return False
