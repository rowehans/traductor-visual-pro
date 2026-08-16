"""Lightweight runtime diagnostics shared by the OCR pipeline and API.

The module deliberately keeps optional dependencies lazy. Importing it must not
load PyTorch, EasyOCR, or any model; this makes it safe to use from tests and
from the PyInstaller entry point.
"""
from __future__ import annotations

import importlib
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator


def _round(value: float | int | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _process_rss_mb() -> float | None:
    try:
        psutil = importlib.import_module("psutil")
        return _round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    except Exception:
        return None


def gpu_memory_snapshot() -> dict[str, Any]:
    """Return a stable, JSON-safe snapshot of the current CUDA device.

    CUDA and PyTorch are optional at import time. An unavailable or broken
    CUDA installation is represented explicitly instead of raising, so this
    helper is safe on CPU-only machines and during partial model startup.
    """
    snapshot: dict[str, Any] = {
        "available": False,
        "device": None,
        "allocated_mb": None,
        "reserved_mb": None,
        "max_allocated_mb": None,
        "free_mb": None,
        "total_mb": None,
    }
    try:
        torch = importlib.import_module("torch")
        if torch is None or not bool(torch.cuda.is_available()):
            return snapshot
        device_index = int(torch.cuda.current_device())
        snapshot["available"] = True
        snapshot["device"] = f"cuda:{device_index}"
        snapshot["allocated_mb"] = _round(
            torch.cuda.memory_allocated(device_index) / 1024 / 1024
        )
        snapshot["reserved_mb"] = _round(
            torch.cuda.memory_reserved(device_index) / 1024 / 1024
        )
        snapshot["max_allocated_mb"] = _round(
            torch.cuda.max_memory_allocated(device_index) / 1024 / 1024
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        snapshot["free_mb"] = _round(free_bytes / 1024 / 1024)
        snapshot["total_mb"] = _round(total_bytes / 1024 / 1024)
    except Exception:
        # Diagnostics must never make OCR fail. Keep the stable unavailable
        # shape so callers can still serialize and display the result.
        return {
            **snapshot,
            "error": "cuda_probe_failed",
        }
    return snapshot


def gpu_budget_allows(
    snapshot: dict[str, Any],
    required_free_mb: float,
    budget_mb: float | None = None,
) -> bool:
    """Return whether an operation has enough observable GPU headroom.

    Unknown GPU state is treated as allowed because the caller may be running
    on CPU or in a process whose CUDA runtime is not installed. The policy is
    therefore a guard against known pressure, not a new hard dependency.
    """
    if not snapshot.get("available"):
        return True
    free_mb = snapshot.get("free_mb")
    if free_mb is not None and float(free_mb) < float(required_free_mb):
        return False
    if budget_mb is not None:
        total_mb = snapshot.get("total_mb")
        if total_mb is not None and free_mb is not None:
            used_mb = float(total_mb) - float(free_mb)
            if used_mb > float(budget_mb):
                return False
    return True


def configure_torch_determinism(torch_module: Any | None = None) -> bool:
    """Configure cuDNN for repeatable convolution selection when available.

    This intentionally avoids ``torch.use_deterministic_algorithms(True)``:
    EasyOCR may use operations without deterministic CUDA kernels, and turning
    that global switch on would convert a quality safeguard into a runtime
    failure. Returning ``False`` simply means that the optional backend was
    unavailable.
    """
    try:
        if torch_module is None:
            torch_module = importlib.import_module("torch")
        cudnn = torch_module.backends.cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
        return True
    except Exception:
        return False


class PageDiagnostics:
    """Mutable per-page diagnostic record with a backwards-compatible API."""

    schema_version = 1

    def __init__(self, ocr_mode: str, ocr_lang: str, doc_id: str = "") -> None:
        self.ocr_mode = ocr_mode
        self.ocr_lang = ocr_lang
        self.doc_id = doc_id
        self._started = time.perf_counter()
        self._finished = False
        self._elapsed_s = 0.0
        self._timings: dict[str, float] = {}
        self._blocks: dict[str, int | None] = {
            "initial": None, "final": None, "discarded": None,
        }
        self._engines: list[str] = []
        self._engine_used: str | None = None
        self._trigger: dict[str, bool | str | None] = {
            "triggered": False, "reason": None,
        }
        self._resources = {
            "start": {"process_rss_mb": _process_rss_mb(), "gpu": gpu_memory_snapshot()},
            "end": None,
        }
        self._errors: list[str] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self._errors.append(f"{name}: {type(exc).__name__}: {exc}")
            raise
        finally:
            self._timings[name] = round(
                self._timings.get(name, 0.0) + time.perf_counter() - started,
                6,
            )

    def set_counts(
        self,
        initial_blocks: int | None = None,
        final_blocks: int | None = None,
        discarded_blocks: int | None = None,
    ) -> None:
        if initial_blocks is not None:
            self._blocks["initial"] = int(initial_blocks)
        if final_blocks is not None:
            self._blocks["final"] = int(final_blocks)
        if discarded_blocks is not None:
            self._blocks["discarded"] = int(discarded_blocks)

    def has_initial_counts(self) -> bool:
        return self._blocks["initial"] is not None

    def set_engines(self, engine_used: str, engines: list[str]) -> None:
        self._engine_used = engine_used
        self._engines = list(engines)

    def set_trigger(self, triggered: bool, reason: str | None = None) -> None:
        self._trigger = {"triggered": bool(triggered), "reason": reason}

    def add_error(self, message: str) -> None:
        self._errors.append(str(message))

    def finish(self) -> None:
        if self._finished:
            return
        self._elapsed_s = time.perf_counter() - self._started
        self._finished = True
        self._resources["end"] = {
            "process_rss_mb": _process_rss_mb(),
            "gpu": gpu_memory_snapshot(),
        }

    def to_dict(self) -> dict[str, Any]:
        if not self._finished:
            self.finish()
        return {
            "schema_version": self.schema_version,
            "ocr_mode": self.ocr_mode,
            "ocr_lang": self.ocr_lang,
            "doc_id": self.doc_id,
            "finished": self._finished,
            "elapsed_s": _round(self._elapsed_s),
            "timings": dict(self._timings),
            "blocks": dict(self._blocks),
            "engine_used": self._engine_used,
            "engines": list(self._engines),
            "trigger": dict(self._trigger),
            "resources": self._resources,
            "errors": list(self._errors),
        }
