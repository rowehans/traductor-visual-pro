"""
ratelimit.py — Módulo compartido de rate limiting.
Evita imports circulares entre server.py y routes/.
"""
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = None
RATE_LIMIT_AVAILABLE = False


def init_limiter(app):
    global limiter, RATE_LIMIT_AVAILABLE
    try:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"],
            storage_uri="memory://",
        )
        RATE_LIMIT_AVAILABLE = True
        print("[rate] Rate limiting activo")
        return True
    except Exception as e:
        RATE_LIMIT_AVAILABLE = False
        print(f"[rate] Rate limiting no disponible: {e}")
        return False