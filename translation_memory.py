"""Memoria automÃ¡tica y acotada de traducciones por documento.

La memoria aprende de traducciones que ya pasaron por el pipeline existente;
no requiere que el usuario mantenga un glosario. Su scope es ``doc_id`` para
evitar que un nombre o tÃ©rmino de un manga contamine otro documento.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import ROOT


MEMORY_TTL_SECONDS = 30 * 24 * 3600
MEMORY_MAX_ENTRIES = 2048
MEMORY_MIN_QUALITY = 0.65
MEMORY_REPLACE_MARGIN = 0.12
DEFAULT_STORAGE_DIR = ROOT / "cache" / "translation_memory"


def _canonical_text(text: str) -> str:
    """Normaliza espacios y Unicode sin destruir la forma mostrada."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def _entry_key(text: str, source_lang: str, target_lang: str) -> str:
    source = _canonical_text(text).casefold()
    return "\x1f".join((source, source_lang.strip().lower(), target_lang.strip().lower()))


def _is_cjk_term(text: str) -> bool:
    """True para términos cortos sin espacios que contienen escritura CJK."""
    value = _canonical_text(text)
    if not value or len(value) > 16 or any(ch.isspace() for ch in value):
        return False
    return any(
        "\u3040" <= ch <= "\u30ff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\u4e00" <= ch <= "\u9fff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in value
    )


def _edit_distance_at_most_one(left: str, right: str) -> int | None:
    """Distancia de edición acotada: devuelve None si supera una edición."""
    if abs(len(left) - len(right)) > 1:
        return None
    if left == right:
        return 0
    if len(left) == len(right):
        return 1 if sum(a != b for a, b in zip(left, right)) <= 1 else None
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = j = differences = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            differences += 1
            j += 1
            if differences > 1:
                return None
        else:
            i += 1
            j += 1
    return 1


@dataclass
class _MemoryEntry:
    source: str
    translation: str
    source_lang: str
    target_lang: str
    quality: float
    occurrences: int
    first_seen: float
    last_seen: float


class DocumentTranslationMemory:
    """Memoria thread-safe de traducciones aprendidas para un documento.

    Solo se almacenan resultados por encima de ``MEMORY_MIN_QUALITY``. Si dos
    traducciones compiten, una nueva debe mejorar claramente la calidad antes
    de reemplazar la entrada existente; así se evita propagar una variación
    aislada del OCR o de un motor externo.
    """

    def __init__(
        self,
        doc_id: str,
        *,
        persist: bool = True,
        storage_dir: Path | str | None = None,
        max_entries: int = MEMORY_MAX_ENTRIES,
    ) -> None:
        self.doc_id = str(doc_id or "")
        self.persist = bool(persist and self.doc_id)
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, _MemoryEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._dirty = False
        if self.persist:
            self._load()

    @property
    def path(self) -> Path:
        digest = hashlib.sha256(self.doc_id.encode("utf-8")).hexdigest()[:32]
        return self.storage_dir / f"{digest}.json"

    def lookup(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str | None:
        key = _entry_key(text, source_lang, target_lang)
        if not _canonical_text(text):
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if time.time() - entry.last_seen > MEMORY_TTL_SECONDS:
                del self._entries[key]
                self._dirty = True
                return None
            entry.last_seen = time.time()
            self._entries.move_to_end(key)
            return entry.translation

    def lookup_variant(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str | None:
        """Busca una variante CJK con como máximo un error de OCR.

        Es deliberadamente conservador: solo acepta términos cortos, vistos
        al menos dos veces y con calidad >= 0.90. Si hay dos candidatos a la
        misma distancia, devuelve ``None`` para no confundir personajes.
        """
        exact = self.lookup(text, source_lang, target_lang)
        if exact is not None:
            return exact
        canonical = _canonical_text(text).casefold()
        if not _is_cjk_term(canonical):
            return None
        source_key = source_lang.strip().lower()
        target_key = target_lang.strip().lower()
        now = time.time()
        candidates: list[tuple[int, str, _MemoryEntry]] = []
        with self._lock:
            expired: list[str] = []
            for key, entry in self._entries.items():
                if now - entry.last_seen > MEMORY_TTL_SECONDS:
                    expired.append(key)
                    continue
                if (
                    entry.source_lang != source_key
                    or entry.target_lang != target_key
                    or entry.occurrences < 2
                    or entry.quality < 0.90
                    or not _is_cjk_term(entry.source)
                ):
                    continue
                distance = _edit_distance_at_most_one(
                    canonical, entry.source.casefold())
                if distance is not None and distance > 0:
                    candidates.append((distance, key, entry))
            for key in expired:
                self._entries.pop(key, None)
                self._dirty = True
            if not candidates:
                return None
            best_distance = min(item[0] for item in candidates)
            best = [item for item in candidates if item[0] == best_distance]
            if len(best) != 1:
                return None
            _, key, entry = best[0]
            entry.last_seen = now
            self._entries.move_to_end(key)
            return entry.translation

    def discard(self, text: str, source_lang: str, target_lang: str) -> bool:
        """Elimina una entrada exacta que ya no supera los gates de calidad."""
        key = _entry_key(text, source_lang, target_lang)
        with self._lock:
            if key not in self._entries:
                return False
            del self._entries[key]
            self._dirty = True
            return True

    def learn(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        *,
        quality: float,
    ) -> bool:
        source_canonical = _canonical_text(source)
        translation_canonical = _canonical_text(translation)
        if not source_canonical or not translation_canonical:
            return False
        if source_canonical.casefold() == translation_canonical.casefold():
            return False
        score = max(0.0, min(1.0, float(quality)))
        if score < MEMORY_MIN_QUALITY:
            return False

        key = _entry_key(source_canonical, source_lang, target_lang)
        now = time.time()
        with self._lock:
            existing = self._entries.get(key)
            if existing is None:
                self._entries[key] = _MemoryEntry(
                    source=source_canonical,
                    translation=translation_canonical,
                    source_lang=source_lang.strip().lower(),
                    target_lang=target_lang.strip().lower(),
                    quality=score,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                )
                self._dirty = True
                self._trim_locked()
                return True

            self._entries.move_to_end(key)
            existing.last_seen = now
            if existing.translation.casefold() == translation_canonical.casefold():
                existing.quality = max(existing.quality, score)
                existing.occurrences += 1
                self._dirty = True
                return True

            if score >= existing.quality + MEMORY_REPLACE_MARGIN:
                existing.translation = translation_canonical
                existing.quality = score
                existing.occurrences = 1
                self._dirty = True
                return True

            # Mantener la traducciÃ³n estable ante un conflicto marginal.
            existing.occurrences += 1
            self._dirty = True
            return False

    def save(self) -> bool:
        """Persiste cambios de forma atÃ³mica; devuelve si se escribiÃ³."""
        if not self.persist:
            return False
        with self._lock:
            if not self._dirty:
                return False
            payload = {
                "version": 1,
                "doc_id": self.doc_id,
                "saved_at": time.time(),
                "entries": [entry.__dict__ for entry in self._entries.values()],
            }
            try:
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    prefix=".translation_memory_", suffix=".tmp",
                    dir=str(self.storage_dir),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(tmp_name, self.path)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
                self._dirty = False
                return True
            except (OSError, TypeError, ValueError):
                return False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._dirty = True
            if self.persist:
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _trim_locked(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != 1:
                return
            now = time.time()
            for raw in payload.get("entries", []):
                entry = _MemoryEntry(
                    source=str(raw["source"]),
                    translation=str(raw["translation"]),
                    source_lang=str(raw["source_lang"]),
                    target_lang=str(raw["target_lang"]),
                    quality=float(raw["quality"]),
                    occurrences=max(1, int(raw.get("occurrences", 1))),
                    first_seen=float(raw.get("first_seen", now)),
                    last_seen=float(raw.get("last_seen", now)),
                )
                if now - entry.last_seen <= MEMORY_TTL_SECONDS:
                    self._entries[_entry_key(entry.source, entry.source_lang, entry.target_lang)] = entry
            self._trim_locked()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return


_registry: OrderedDict[str, DocumentTranslationMemory] = OrderedDict()
_registry_lock = threading.RLock()
_REGISTRY_MAX_DOCUMENTS = 16


def get_document_memory(doc_id: str) -> DocumentTranslationMemory | None:
    """Obtiene la memoria automÃ¡tica del documento, o ``None`` sin scope."""
    normalized = str(doc_id or "").strip()
    if not normalized:
        return None
    with _registry_lock:
        memory = _registry.get(normalized)
        if memory is None:
            memory = DocumentTranslationMemory(normalized)
            _registry[normalized] = memory
        _registry.move_to_end(normalized)
        while len(_registry) > _REGISTRY_MAX_DOCUMENTS:
            _, evicted = _registry.popitem(last=False)
            evicted.save()
        return memory


def clear_document_memories() -> None:
    with _registry_lock:
        for memory in _registry.values():
            memory.save()
        _registry.clear()
