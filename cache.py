"""
cache.py — Cache de traducciones en filesystem.
Evita retraducir textos repetidos, mejora velocidad en mangas con frases recurrentes.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache" / "translations"
CACHE_TTL_SECS = 7 * 24 * 3600  # 7 dias
MAX_CACHE_ENTRIES = 5000

_lock = threading.Lock()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _make_key(text: str, src: str, tgt: str) -> str:
    raw = f"{text}||{src}||{tgt}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get(text: str, src: str, tgt: str) -> str | None:
    """Retorna traduccion del cache o None."""
    key = _make_key(text, src, tgt)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data["ts"] > CACHE_TTL_SECS:
            path.unlink(missing_ok=True)
            return None
        # Actualizar timestamp de acceso
        data["ts"] = time.time()
        with _lock:
            path.write_text(json.dumps(data), encoding="utf-8")
        return data["result"]
    except Exception:
        return None


def set(text: str, src: str, tgt: str, result: str) -> None:
    """Guarda traduccion en cache."""
    key = _make_key(text, src, tgt)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"text": text, "src": src, "tgt": tgt, "result": result, "ts": time.time()}
    path = _cache_path(key)
    with _lock:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _evict_if_needed()


def _evict_if_needed() -> None:
    """Elimina entradas vencidas o sobrantes (LRU simple)."""
    if not CACHE_DIR.exists():
        return
    entries = []
    for f in CACHE_DIR.iterdir():
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                entries.append((data.get("ts", 0), f))
            except Exception:
                f.unlink(missing_ok=True)
    # Ordenar por timestamp (mas viejo primero)
    entries.sort(key=lambda x: x[0])
    # Eliminar vencidos
    now = time.time()
    keep = []
    for ts, f in entries:
        if now - ts > CACHE_TTL_SECS:
            f.unlink(missing_ok=True)
        else:
            keep.append((ts, f))
    # Si aun supera el maximo, eliminar los mas viejos
    if len(keep) > MAX_CACHE_ENTRIES:
        for _, f in keep[:len(keep) - MAX_CACHE_ENTRIES]:
            f.unlink(missing_ok=True)


def clear() -> None:
    """Limpia toda la cache."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            f.unlink(missing_ok=True)