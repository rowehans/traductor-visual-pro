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
