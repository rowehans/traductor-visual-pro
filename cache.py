"""
cache.py — Cache de traducciones en filesystem.
Evita retraducir textos repetidos, mejora velocidad en mangas con frases recurrentes.
"""
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent / "cache" / "translations"
CACHE_TTL_SECS = 7 * 24 * 3600  # 7 dias
MAX_CACHE_ENTRIES = 5000
_MEMORY_CACHE_MAX_CHARS = 2_000_000

_lock = threading.Lock()
_memory_lock = threading.Lock()
_memory_cache: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()
_memory_chars = 0


def _memory_key(key: str) -> tuple[str, str]:
    """Incluye el directorio para aislar tests y cambios de configuracion."""
    return str(CACHE_DIR), key


def _memory_get(key: str) -> str | None:
    now = time.time()
    memory_key = _memory_key(key)
    with _memory_lock:
        entry = _memory_cache.get(memory_key)
        if entry is None:
            return None
        ts, result = entry
        if now - ts > CACHE_TTL_SECS:
            _memory_cache.pop(memory_key, None)
            return None
        _memory_cache.move_to_end(memory_key)
        return result


def _memory_set(key: str, ts: float, result: str) -> None:
    global _memory_chars
    memory_key = _memory_key(key)
    with _memory_lock:
        previous = _memory_cache.pop(memory_key, None)
        if previous is not None:
            _memory_chars -= len(previous[1])
        if len(result) > _MEMORY_CACHE_MAX_CHARS:
            return
        _memory_cache[memory_key] = (ts, result)
        _memory_cache.move_to_end(memory_key)
        _memory_chars += len(result)
        while (
            len(_memory_cache) > MAX_CACHE_ENTRIES
            or _memory_chars > _MEMORY_CACHE_MAX_CHARS
        ):
            _, evicted = _memory_cache.popitem(last=False)
            _memory_chars -= len(evicted[1])


def _memory_discard(key: str) -> None:
    global _memory_chars
    with _memory_lock:
        previous = _memory_cache.pop(_memory_key(key), None)
        if previous is not None:
            _memory_chars -= len(previous[1])


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _make_key(text: str, src: str, tgt: str) -> str:
    # La misma frase puede llegar con padding o espacios internos distintos
    # desde OCR de cajas adyacentes. Canonicalizar evita perder consistencia
    # entre pÃ¡ginas por diferencias de formato irrelevantes.
    canonical_text = " ".join(str(text).split())
    raw = f"{canonical_text}||{src.strip().lower()}||{tgt.strip().lower()}"
    # MD5 usado SOLO para generar keys de cache, NO para seguridad.
    # usedforsecurity=False evita bloqueos en entornos FIPS.
    return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def get(text: str, src: str, tgt: str) -> str | None:
    """Retorna traduccion del cache o None."""
    key = _make_key(text, src, tgt)
    memory_result = _memory_get(key)
    if memory_result is not None:
        return memory_result
    path = _cache_path(key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data["ts"] > CACHE_TTL_SECS:
            path.unlink(missing_ok=True)
            _memory_discard(key)
            return None
        # No actualizamos el timestamp interno aquí (evita escritura I/O en cada get).
        # La evicción LRU usa mtime del filesystem, que el OS actualiza automáticamente.
        result = str(data["result"])
        _memory_set(key, float(data["ts"]), result)
        return result
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def set(text: str, src: str, tgt: str, result: str) -> None:
    """Guarda traduccion en cache."""
    key = _make_key(text, src, tgt)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "text": text, "src": src, "tgt": tgt, "result": result, "ts": time.time(),
    }
    path = _cache_path(key)
    with _lock:
        # Escribir en el mismo directorio y reemplazar al final evita que un
        # cierre del proceso deje el JSON destino a medio escribir. El nombre
        # temporal es unico para no colisionar con otra instancia del server.
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=str(CACHE_DIR))
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                json.dump(data, temp_file, ensure_ascii=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        _evict_if_needed()
        _memory_set(key, float(data["ts"]), str(result))


def _evict_if_needed() -> None:
    """Elimina entradas vencidas o sobrantes (LRU simple).
    Usa mtime del filesystem en vez de parsear JSON para el escaneo,
    ~10x más rápido que la versión anterior.
    Optimizado: solo escanea si el contador de archivos supera el umbral
    (MAX_CACHE_ENTRIES * 1.2), evitando O(n log n) en cada escritura."""
    if not CACHE_DIR.exists():
        return
    # Umbral: solo evict cuando hay 20% más archivos que el máximo
    try:
        file_count = sum(1 for _ in CACHE_DIR.iterdir())
        if file_count <= MAX_CACHE_ENTRIES * 1.2:
            return
    except OSError:
        return
    now = time.time()
    entries = []
    for f in CACHE_DIR.iterdir():
        if f.suffix == ".json":
            try:
                entries.append((f.stat().st_mtime, f))
            except OSError:
                f.unlink(missing_ok=True)
    # Ordenar por timestamp (mas viejo primero)
    entries.sort(key=lambda x: x[0])
    # Eliminar vencidos (usando mtime como proxy de ts interno)
    keep = []
    for mtime, f in entries:
        if now - mtime > CACHE_TTL_SECS:
            f.unlink(missing_ok=True)
            _memory_discard(f.stem)
        else:
            keep.append((mtime, f))
    # Si aun supera el maximo, eliminar los mas viejos
    if len(keep) > MAX_CACHE_ENTRIES:
        for _, f in keep[:len(keep) - MAX_CACHE_ENTRIES]:
            f.unlink(missing_ok=True)
            _memory_discard(f.stem)


def clear() -> None:
    """Limpia toda la cache."""
    global _memory_chars
    with _memory_lock:
        _memory_cache.clear()
        _memory_chars = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            f.unlink(missing_ok=True)
