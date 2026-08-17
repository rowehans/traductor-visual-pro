import json
from types import SimpleNamespace


def test_page_diagnostics_records_stages_and_serializes_json():
    from runtime_diagnostics import PageDiagnostics

    diagnostics = PageDiagnostics(ocr_mode="fusion", ocr_lang="ja", doc_id="cap-01")
    diagnostics.set_counts(initial_blocks=2, final_blocks=3)
    diagnostics.set_engines("fusion", ["easyocr+rapid", "unlimited"])
    diagnostics.set_trigger(triggered=True, reason="low_confidence")

    with diagnostics.stage("hybrid"):
        pass

    diagnostics.finish()
    payload = diagnostics.to_dict()

    assert payload["ocr_mode"] == "fusion"
    assert payload["ocr_lang"] == "ja"
    assert payload["doc_id"] == "cap-01"
    assert payload["blocks"]["initial"] == 2
    assert payload["blocks"]["final"] == 3
    assert payload["engines"] == ["easyocr+rapid", "unlimited"]
    assert payload["trigger"] == {"triggered": True, "reason": "low_confidence"}
    assert payload["timings"]["hybrid"] >= 0
    assert payload["finished"] is True
    json.dumps(payload)


def test_gpu_snapshot_has_stable_schema_when_cuda_is_unavailable(monkeypatch):
    from runtime_diagnostics import gpu_memory_snapshot

    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    snapshot = gpu_memory_snapshot()

    assert snapshot["available"] is False
    assert snapshot["device"] is None
    assert set(snapshot) >= {"available", "device", "allocated_mb", "reserved_mb", "free_mb", "total_mb"}


def test_gpu_snapshot_llena_datos_con_cuda_fake_disponible(monkeypatch):
    """El camino CUDA de gpu_memory_snapshot se cubre con un torch falso
    (is_available=True + funciones de memoria). Sin GPU real ni torch
    instalado (CI), así la cobertura del módulo no depende del hardware:
    el runner no cubre estas líneas porque el stub del conftest devuelve
    is_available=False."""
    from runtime_diagnostics import gpu_memory_snapshot

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        memory_allocated=lambda *a, **k: 1024 * 1024 * 300,
        memory_reserved=lambda *a, **k: 1024 * 1024 * 320,
        max_memory_allocated=lambda *a, **k: 1024 * 1024 * 350,
        mem_get_info=lambda *a, **k: (1024 * 1024 * 800, 1024 * 1024 * 4096),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    snapshot = gpu_memory_snapshot()

    assert snapshot["available"] is True
    assert snapshot["device"] == "cuda:0"
    assert snapshot["allocated_mb"] == 300.0
    assert snapshot["reserved_mb"] == 320.0
    assert snapshot["max_allocated_mb"] == 350.0
    assert snapshot["free_mb"] == 800.0
    assert snapshot["total_mb"] == 4096.0


def test_round_maneja_none():
    from runtime_diagnostics import _round
    assert _round(None) is None
    assert _round(1234.56789) == 1234.568


def test_gpu_budget_usa_budget_mb():
    """El camino de budget_mb (no solo free_mb) de gpu_budget_allows."""
    from runtime_diagnostics import gpu_budget_allows

    snapshot = {
        "available": True,
        "free_mb": 500.0,
        "total_mb": 4000.0,
    }
    # used = 3500 > budget 2000 → False
    assert gpu_budget_allows(snapshot, required_free_mb=100.0,
                             budget_mb=2000.0) is False
    # used = 3500 < budget 4000 → True
    assert gpu_budget_allows(snapshot, required_free_mb=100.0,
                             budget_mb=4000.0) is True


def test_configure_torch_determinism_degrada_sin_backends():
    """Sin backends.cudnn (torch roto o stub sin el atributo) → False."""
    from runtime_diagnostics import configure_torch_determinism

    broken = SimpleNamespace()  # sin .backends
    assert configure_torch_determinism(broken) is False


def test_has_initial_counts_vacio():
    """has_initial_counts() antes de set_counts → False (y to_dict serializa)."""
    from runtime_diagnostics import PageDiagnostics

    diagnostics = PageDiagnostics(ocr_mode="fusion", ocr_lang="ja")
    assert diagnostics.has_initial_counts() is False
    diagnostics.set_counts(initial_blocks=4, final_blocks=4)
    assert diagnostics.has_initial_counts() is True
    json.dumps(diagnostics.to_dict())


def test_gpu_budget_reports_insufficient_headroom():
    from runtime_diagnostics import gpu_budget_allows

    snapshot = {
        "available": True,
        "device": "cuda:0",
        "allocated_mb": 3000.0,
        "reserved_mb": 3200.0,
        "free_mb": 500.0,
        "total_mb": 4000.0,
    }

    assert gpu_budget_allows(snapshot, required_free_mb=1000.0) is False
    assert gpu_budget_allows(snapshot, required_free_mb=400.0) is True


def test_configure_torch_determinism_disables_cudnn_autotuning():
    from runtime_diagnostics import configure_torch_determinism

    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    torch_module = SimpleNamespace(
        backends=SimpleNamespace(cudnn=cudnn),
    )

    assert configure_torch_determinism(torch_module) is True
    assert cudnn.deterministic is True
    assert cudnn.benchmark is False
