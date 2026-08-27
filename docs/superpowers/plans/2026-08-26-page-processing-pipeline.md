# Page Processing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert single-page OCR, inpainting, translation, diagnostics, and batch integration into one explicit, testable page-processing result without changing the public HTTP routes.

**Architecture:** `OCRManager` remains the owner of OCR orchestration, while a small page-processing coordinator owns the page lifecycle and returns one structured result. EasyOCR, RapidOCR, Unlimited-OCR, translation, and rendering remain existing seams; this plan removes tuple interpretation from callers rather than adding a second generic abstraction. Single-page API and batch API will consume the same result-building path, while the frontend and CLI continue using their existing payloads and response compatibility fields.

**Tech Stack:** Python 3.12, Flask, pytest/pytest-mock, OpenCV, NumPy, existing `OCRManager`, existing `routes/api.py`, existing `process_all_pages.py`; no new runtime dependency.

## Global Constraints

- Preserve public endpoints `/api/process-page` and `/api/process-page-batch`.
- Preserve existing response fields `blocks`, `inpainted_image`, `ocr_engine`, `engines_used`, and `detected_lang`.
- Preserve `ocr_mode` values `fusion`, `unlimited`, `easyocr`, and `auto`.
- Do not add a new OCR engine or change thresholds during this refactor.
- Do not change the frontend rendering contract or the PDF export behavior.
- Keep image-size and decoded-pixel validation before expensive OpenCV/OCR work.
- Keep Unlimited-OCR optional: fusion must continue degrading to available engines.
- Keep `doc_id` propagation and document-scoped decision caches unchanged.
- Every task must leave focused tests green before its commit.
- Do not persist raw user images in new files.

## File map

- Create: `page_processing.py` — page lifecycle coordinator and immutable result data structures. It owns orchestration only; engine internals stay in existing modules.
- Modify: `routes/api.py` — call the page coordinator and translate its result into the existing JSON response. Remove only duplicated tuple unpacking once the new path is proven.
- Modify: `ocr_engine.py` — expose a direct structured OCR result while preserving `run_ocr()` and `run_ocr_batch()` compatibility for existing callers during migration.
- Modify: `process_all_pages.py` — consume stable response fields and keep checkpoint semantics unchanged.
- Test: `tests/test_page_processing.py` — coordinator contract, empty input, engine fallback, inpainting result, diagnostics, and failure behavior.
- Modify: `tests/test_api.py` — endpoint compatibility and propagation assertions.
- Modify: `tests/test_ocr_engine.py` — structured result compatibility and no-regression checks.
- Modify: `tests/test_process_all_pages.py` — verify single and batch checkpoint behavior remains unchanged.
- Inspect only: `app.js`, `uocr_client.py`, `uocr_daemon.py`, `translator.py`, `config.py`, `main.spec`, `.github/workflows/ci.yml`.

### Task 1: Establish the page result contract

**Files:**
- Create: `page_processing.py`
- Create: `tests/test_page_processing.py`

**Interfaces:**
- Produces `PageProcessingResult` with fields:
  - `blocks: list[dict[str, Any]]`
  - `inpainted_image: str | None`
  - `detected_lang: str`
  - `ocr_engine: str`
  - `engines_used: list[str]`
  - `diagnostics: dict[str, Any]`
  - `error: str | None = None`
- Produces `PageProcessor.process(image: Any, *, source: str, target: str, ocr_mode: str, doc_id: str = "", prefilter: bool = False, force_uocr: bool = False) -> PageProcessingResult`.
- Consumes no new external dependency.

- [ ] **Step 1: Write the failing contract test**

```python
from page_processing import PageProcessingResult


def test_result_has_stable_page_contract():
    result = PageProcessingResult(
        blocks=[],
        inpainted_image=None,
        detected_lang="es",
        ocr_engine="easyocr",
        engines_used=["easyocr"],
        diagnostics={"finished": True},
    )

    assert result.blocks == []
    assert result.inpainted_image is None
    assert result.detected_lang == "es"
    assert result.ocr_engine == "easyocr"
    assert result.engines_used == ["easyocr"]
    assert result.error is None
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_page_processing.py::test_result_has_stable_page_contract -q
```

Expected: FAIL because `page_processing.py` and `PageProcessingResult` do not yet exist.

- [ ] **Step 3: Implement the smallest result structure**

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageProcessingResult:
    blocks: list[dict[str, Any]]
    inpainted_image: str | None
    detected_lang: str
    ocr_engine: str
    engines_used: list[str]
    diagnostics: dict[str, Any]
    error: str | None = None
```

Add only the imports and dataclass needed by the test. Do not add a processor implementation in this step.

- [ ] **Step 4: Run the focused test and verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add page_processing.py tests/test_page_processing.py
git commit -m "refactor: define structured page processing result"
```

### Task 2: Add the OCR structured-result seam

**Files:**
- Modify: `ocr_engine.py` near `run_ocr()` and `run_ocr_batch()`
- Modify: `tests/test_ocr_engine.py`

**Interfaces:**
- Add `OCRResult` in `ocr_engine.py` with fields:
  - `blocks: list[dict[str, Any]]`
  - `engine: str`
  - `engines: list[str]`
  - `diagnostics: dict[str, Any]`
- Add `OCRManager.run_ocr_result(image: Any, source_lang: str, *, ocr_mode: str = "fusion", force_uocr: bool = False, disable_uocr: bool = False, doc_id: str = "") -> OCRResult`.
- Keep `run_ocr(...) -> tuple[list[dict[str, Any]], str, list[str]]` unchanged by delegating to `run_ocr_result()`.

- [ ] **Step 1: Write the failing compatibility test**

```python
from ocr_engine import OCRManager, OCRResult


def test_run_ocr_result_exposes_the_same_easyocr_data(mocker):
    blocks = [{"text": "hola", "confidence": 0.9}]
    mocker.patch("ocr_utils._detect_and_ocr", return_value=blocks)

    result = OCRManager().run_ocr_result(object(), "es", ocr_mode="easyocr")

    assert isinstance(result, OCRResult)
    assert result.blocks == blocks
    assert result.engine == "easyocr"
    assert result.engines == ["easyocr"]
    assert result.diagnostics["finished"] is True
```

- [ ] **Step 2: Run it and verify it fails**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_ocr_engine.py::test_run_ocr_result_exposes_the_same_easyocr_data -q
```

Expected: FAIL because `OCRResult` and `run_ocr_result()` do not exist.

- [ ] **Step 3: Implement the result dataclass and delegate existing API**

Use a dataclass adjacent to the existing OCRManager definitions:

```python
@dataclass(frozen=True)
class OCRResult:
    blocks: list[dict[str, Any]]
    engine: str
    engines: list[str]
    diagnostics: dict[str, Any]
```

Move the current body of `run_ocr()` into `run_ocr_result()` without changing conditions, cache calls, engine order, or fallback behavior. Implement the old method as:

```python
def run_ocr(self, image, source_lang, *, ocr_mode="fusion", force_uocr=False,
            disable_uocr=False, doc_id=""):
    result = self.run_ocr_result(
        image,
        source_lang,
        ocr_mode=ocr_mode,
        force_uocr=force_uocr,
        disable_uocr=disable_uocr,
        doc_id=doc_id,
    )
    return result.blocks, result.engine, result.engines
```

Preserve `self.last_diagnostics` assignment so existing callers and tests continue to work.

- [ ] **Step 4: Run focused OCR tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_ocr_engine.py -q
```

Expected: all existing OCRManager tests plus the new test pass.

- [ ] **Step 5: Commit**

```bash
git add ocr_engine.py tests/test_ocr_engine.py
git commit -m "refactor(ocr): expose structured engine result"
```

### Task 3: Implement page orchestration without changing endpoint behavior

**Files:**
- Modify: `page_processing.py`
- Modify: `tests/test_page_processing.py`

**Interfaces:**
- `PageProcessor` consumes `OCRManager.run_ocr_result()` and the existing helpers currently used by `routes/api.py` for inpainting and translation.
- `PageProcessor.process(...)` returns `PageProcessingResult`.

- [ ] **Step 1: Write failing orchestration tests**

```python
from page_processing import PageProcessor


def test_processor_preserves_engine_and_diagnostics(mocker):
    ocr_result = type("OCR", (), {
        "blocks": [{"source": "hola", "translated": "hello"}],
        "engine": "fusion",
        "engines": ["easyocr+rapid"],
        "diagnostics": {"finished": True, "blocks": {"final": 1}},
    })()
    manager = mocker.Mock()
    manager.run_ocr_result.return_value = ocr_result

    processor = PageProcessor(ocr_manager=manager)
    result = processor.process(object(), source="es", target="en", ocr_mode="fusion")

    assert result.blocks == ocr_result.blocks
    assert result.ocr_engine == "fusion"
    assert result.engines_used == ["easyocr+rapid"]
    assert result.diagnostics["finished"] is True


def test_processor_returns_empty_result_for_empty_image(mocker):
    manager = mocker.Mock()
    result = PageProcessor(ocr_manager=manager).process(
        None, source="es", target="en", ocr_mode="fusion"
    )

    assert result.blocks == []
    assert result.error is not None
    manager.run_ocr_result.assert_not_called()
```

- [ ] **Step 2: Run focused tests and verify failure**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_page_processing.py -q
```

Expected: FAIL because `PageProcessor` does not exist.

- [ ] **Step 3: Implement the coordinator using existing seams**

Implement the coordinator in this order:

1. Reject `None` before touching OCR.
2. Call `OCRManager.run_ocr_result()` with the exact options received.
3. Read the existing dropped-noise state only through the current documented OCRManager seam.
4. Call the existing inpainting/translation finalization path currently used by `routes/api.py`; do not duplicate its algorithms.
5. Construct `PageProcessingResult` with the existing engine names and diagnostics.
6. On a known processing exception, return `PageProcessingResult(error=str(exc), blocks=[], ...)` only where the current endpoint converts that exception into a controlled response. Do not swallow unexpected exceptions silently.

The coordinator must not write files, start threads, modify caches directly, or alter `doc_id`.

- [ ] **Step 4: Add boundary tests for ordering and fallback**

```python

def test_processor_passes_doc_id_and_force_flags_to_ocr(mocker):
    manager = mocker.Mock()
    manager.run_ocr_result.return_value = type("OCR", (), {
        "blocks": [], "engine": "easyocr", "engines": ["easyocr"],
        "diagnostics": {"finished": True},
    })()

    PageProcessor(ocr_manager=manager).process(
        object(), source="auto", target="es", ocr_mode="fusion",
        doc_id="doc123", prefilter=True, force_uocr=True,
    )

    manager.run_ocr_result.assert_called_once()
    kwargs = manager.run_ocr_result.call_args.kwargs
    assert kwargs["doc_id"] == "doc123"
    assert kwargs["force_uocr"] is True
```

- [ ] **Step 5: Run the focused coordinator tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_page_processing.py -q
```

Expected: all page-processing tests pass.

- [ ] **Step 6: Commit**

```bash
git add page_processing.py tests/test_page_processing.py
git commit -m "refactor: centralize page processing orchestration"
```

### Task 4: Migrate the single-page API while preserving JSON compatibility

**Files:**
- Modify: `routes/api.py` in `process_page()`
- Modify: `tests/test_api.py`

**Interfaces:**
- The endpoint continues accepting the current JSON and binary image inputs.
- The endpoint continues returning all existing response keys.
- The endpoint consumes `PageProcessor.process()`.

- [ ] **Step 1: Write compatibility tests before migration**

```python

def test_process_page_keeps_legacy_response_fields(client, mocker):
    mocker.patch("routes.api._decode_image_from_request", return_value=object())
    mocker.patch("routes.api.PageProcessor.process", return_value=type("Result", (), {
        "blocks": [{"source": "hola", "translated": "hello"}],
        "inpainted_image": "b64",
        "detected_lang": "es",
        "ocr_engine": "fusion",
        "engines_used": ["easyocr+rapid"],
        "diagnostics": {"finished": True},
        "error": None,
    })())

    response = client.post(
        "/api/process-page?target=en&source=es",
        json={"image": "data:image/png;base64,AAAA"},
    )

    assert response.status_code in (200, 400)
    if response.status_code == 200:
        body = response.get_json()
        assert body["blocks"]
        assert body["inpainted_image"] == "b64"
        assert body["ocr_engine"] == "fusion"
        assert body["engines_used"] == ["easyocr+rapid"]
```

Adapt the decode helper name to the actual existing route seam rather than creating a new helper solely for the test.

- [ ] **Step 2: Run the endpoint test and verify the migration test fails or exercises the old path**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_api.py -k legacy_response_fields -q
```

Expected: the new processor mock is not yet used, proving the test targets the migration.

- [ ] **Step 3: Replace only tuple interpretation in `process_page()`**

Keep validation, rate limiting, image decoding, request fields, response status codes, and error messages unchanged. Replace the inline OCR/finalization orchestration with:

```python
page_result = PageProcessor().process(
    img_bgr,
    source=source,
    target=target,
    ocr_mode=ocr_mode,
    doc_id=doc_id,
    prefilter=prefilter,
    force_uocr=force_uocr,
)
```

Map the result back to existing keys. Keep diagnostics additive and JSON-serializable. Do not remove compatibility fields.

- [ ] **Step 4: Run API tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_api.py -q
```

Expected: existing API tests and the new compatibility test pass.

- [ ] **Step 5: Commit**

```bash
git add routes/api.py tests/test_api.py
 git commit -m "refactor(api): consume structured page result"
```

### Task 5: Migrate batch through the same page result semantics

**Files:**
- Modify: `routes/api.py` in `process_page_batch()`
- Modify: `tests/test_api.py`
- Modify: `tests/test_process_all_pages.py`

**Interfaces:**
- Batch response remains `{"results": [...]}` in input order.
- Each item retains `blocks`, `inpainted_image`, `ocr_engine`, `engines_used`, and diagnostics where already exposed.
- Missing or malformed internal results continue to be represented by existing caller error handling.

- [ ] **Step 1: Write ordering and empty-batch tests**

```python

def test_batch_result_order_matches_input_pages(client, mocker):
    results = [
        type("Result", (), {"blocks": [{"source": "p1"}], "inpainted_image": None,
             "detected_lang": "es", "ocr_engine": "easyocr",
             "engines_used": ["easyocr"], "diagnostics": {}, "error": None})(),
        type("Result", (), {"blocks": [{"source": "p2"}], "inpainted_image": None,
             "detected_lang": "es", "ocr_engine": "fusion",
             "engines_used": ["easyocr+rapid"], "diagnostics": {}, "error": None})(),
    ]
    mocker.patch("routes.api._process_batch_with_page_processor", return_value=results)

    response = client.post("/api/process-page-batch", json={"images": ["a", "b"]})

    assert response.status_code in (200, 400)
    if response.status_code == 200:
        body = response.get_json()
        assert body["results"][0]["blocks"][0]["source"] == "p1"
        assert body["results"][1]["blocks"][0]["source"] == "p2"
```

- [ ] **Step 2: Run the focused batch tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_api.py -k 'batch' tests/test_process_all_pages.py -q
```

Expected: the new test fails or does not find the intended seam before migration.

- [ ] **Step 3: Route batch pages through the shared coordinator semantics**

Preserve the optimized `run_ocr_batch()` path where it is materially different and already tested. The batch adapter may convert each internal result into `PageProcessingResult`, but it must not silently reorder results. Use `enumerate()` over the input list and construct output in that same order.

Do not duplicate single-page translation or inpainting algorithms. If the current batch code has a specialized implementation, keep it behind the batch coordinator and normalize its output only.

- [ ] **Step 4: Test malformed, short, and empty results**

```python

def test_batch_missing_result_does_not_shift_later_pages():
    # Keep the existing process_all_pages regression setup and assert that
    # page 3 remains page 3 when page 2 is represented as missing_result.
    ...
```

Replace the ellipsis with the existing helper and assertions already present in `tests/test_process_all_pages.py`; do not invent a second batch harness.

- [ ] **Step 5: Run batch and checkpoint tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_api.py tests/test_process_all_pages.py -q
```

Expected: all tests pass with input ordering and checkpoint semantics unchanged.

- [ ] **Step 6: Commit**

```bash
git add routes/api.py tests/test_api.py tests/test_process_all_pages.py
git commit -m "refactor(api): normalize batch page results"
```

### Task 6: Add structured diagnostics without leaking raw document content

**Files:**
- Modify: `page_processing.py`
- Modify: `tests/test_page_processing.py`
- Modify: `routes/api.py` only if diagnostics are not already returned

**Interfaces:**
- Diagnostics use numeric counts and timings only:
  - `finished: bool`
  - `blocks.initial: int`
  - `blocks.final: int`
  - `timings.total: float`
  - engine names and fallback reason where known
- No raw OCR text or base64 image is added to logs or diagnostics.

- [ ] **Step 1: Write diagnostics tests**

```python

def test_diagnostics_are_numeric_and_do_not_contain_source_text(mocker):
    # Use the existing fake OCR result and assert only stable aggregate fields.
    result = ...
    assert result.diagnostics["finished"] is True
    assert isinstance(result.diagnostics["timings"]["total"], (int, float))
    assert "hola secreto" not in repr(result.diagnostics)
```

Replace the setup with the already-created `PageProcessor` test fixture; do not add a new fake engine.

- [ ] **Step 2: Run and verify the diagnostic test fails**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_page_processing.py -k diagnostics -q
```

Expected: FAIL if the required aggregate fields are absent.

- [ ] **Step 3: Add only aggregate diagnostics**

Record stage timings with `time.perf_counter()`. Preserve existing timing values where available. Do not add new profiling branches, logging frameworks, persistence, or text samples.

- [ ] **Step 4: Run focused tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_page_processing.py tests/test_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add page_processing.py routes/api.py tests/test_page_processing.py tests/test_api.py
git commit -m "feat(diagnostics): expose aggregate page metrics"
```

### Task 7: Remove obsolete indirection only after callers migrate

**Files:**
- Modify: `ocr_engine.py`
- Modify: `routes/api.py`
- Modify: `tests/test_ocr_engine.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Existing `OCRManager.run_ocr()` remains only if external callers still use it. If repository search finds no production caller after migration, remove it and its compatibility test in this task.
- Do not remove `run_ocr_batch()` while batch still uses it.

- [ ] **Step 1: Search for remaining tuple-based callers**

```bash
rg -n "\.run_ocr\(|run_ocr\(" --glob '*.py'
```

Expected: only compatibility tests or intentional legacy callers remain.

- [ ] **Step 2: Delete unused compatibility code**

Remove only methods and tests with zero production consumers. Keep public behavior if any script or documented entrypoint still invokes it.

- [ ] **Step 3: Run focused tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_ocr_engine.py tests/test_api.py tests/test_process_all_pages.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ocr_engine.py routes/api.py tests/test_ocr_engine.py tests/test_api.py
git commit -m "refactor: remove obsolete OCR result indirection"
```

### Task 8: Update packaging, CI discovery, and documentation contracts

**Files:**
- Modify: `main.spec`
- Modify: `run_ci.py`
- Modify: `tests/test_packaging.py`
- Modify: `README.md`
- Modify: `AGENTS.md` only if the architecture table is stale

**Interfaces:**
- `page_processing.py` must be included in the PyInstaller data/module strategy used by the existing build.
- JavaScript syntax checks remain automatic for `app.js` and all `js/**/*.js`.
- Documentation must describe the new page-processing ownership without claiming unsupported daemon availability.

- [ ] **Step 1: Write packaging regression test**

```python

def test_page_processing_module_is_packaged():
    spec = Path("main.spec").read_text(encoding="utf-8")
    assert "page_processing.py" in spec
```

Adapt to the existing packaging-test style if it inspects `DATAS` programmatically.

- [ ] **Step 2: Run the test and verify it fails**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_packaging.py -k page_processing -q
```

Expected: FAIL because the new module is not listed in the packaging contract.

- [ ] **Step 3: Add the module to the existing packaging seam**

Add `page_processing.py` alongside the existing lightweight Python data modules. Do not add heavy dependencies to `hiddenimports`.

- [ ] **Step 4: Make JavaScript discovery cover all modules**

Inspect `_js_syntax_files()` in `run_ci.py`. If it remains hardcoded, replace only the list construction with a stable glob of `app.js` plus `js/**/*.js`, preserving output order and existing missing-file behavior.

- [ ] **Step 5: Update documentation**

Document:

- page as the processing unit;
- OCRManager as the OCR orchestration owner;
- unchanged endpoint names and response compatibility;
- structured aggregate diagnostics;
- no guarantee that Unlimited-OCR is available on every installation.

- [ ] **Step 6: Run packaging and CI-contract tests**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_packaging.py tests/test_run_ci.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add main.spec run_ci.py tests/test_packaging.py README.md AGENTS.md
 git commit -m "build: include page processing in runtime contracts"
```

### Task 9: Full proportionate validation and visual smoke test

**Files:**
- No production changes expected.
- Inspect: all files changed by Tasks 1–8.

- [ ] **Step 1: Run focused suite**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m pytest tests/test_page_processing.py tests/test_ocr_engine.py tests/test_api.py tests/test_process_all_pages.py tests/test_packaging.py tests/test_run_ci.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run syntax checks**

```bash
'/d/crear traductor/env/Scripts/python.exe' -m py_compile page_processing.py ocr_engine.py routes/api.py process_all_pages.py
node --check app.js
for f in js/*.js; do node --check "$f"; done
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the canonical CI test list**

```bash
'/d/crear traductor/env/Scripts/python.exe' run_ci.py --skip-server --skip-mypy --skip-bandit --skip-cov
```

Expected: canonical tests pass and no production module is unclassified.

- [ ] **Step 4: Build the frontend and validate the bundle**

```bash
'/d/crear traductor/env/Scripts/python.exe' build.py
find dist -maxdepth 2 -type f -print
```

Expected: `dist/index.html`, `dist/app.min.js`, `dist/styles.min.css`, and every `dist/js/*.js` exist.

- [ ] **Step 5: Run the real endpoint smoke test**

Start the existing server entrypoint using the project’s normal environment, then verify:

```bash
curl -sS http://127.0.0.1:5174/api/health
curl -sS http://127.0.0.1:5174/api/config
```

Submit one fixed test image through `/api/process-page` and verify:

- HTTP 200 for a valid image;
- response contains `blocks`, `inpainted_image`, `ocr_engine`, and `engines_used`;
- diagnostics contain aggregate counts and timings;
- fusion still returns a controlled fallback when Unlimited-OCR is unavailable.

Do not use a production document or upload user data during this smoke test.

- [ ] **Step 6: Capture visual evidence**

Using the existing browser automation workflow, capture one before/after pair for:

- a normal dialogue page;
- a watermark page;
- an artistic page if the Unlimited-OCR daemon is ready.

Review that the original background is not duplicated, filtered noise is removed, boxes remain aligned, and exported text is visible. Save captures outside the repository.

- [ ] **Step 7: Review diff and stop**

```bash
git diff --check HEAD~8..HEAD
git status --short
git log --oneline -8
```

Expected: no whitespace errors, clean working tree, and one focused commit per task.

## Self-review

- **Spec coverage:** The plan covers the requested architectural direction: explicit page unit, OCRManager ownership, engine/fallback locality, structured diagnostics, single/batch ordering, packaging, CI, and visual validation. It deliberately does not include unrelated frontend modularization or a new OCR engine.
- **Placeholder scan:** No implementation step is intentionally left as TBD. The two test snippets marked with `...` must be replaced by existing fixture setup during execution; if strict no-placeholder enforcement is required, copy the exact existing setup before beginning Task 5 or split that test into a separate task.
- **Type consistency:** `OCRResult` is produced by `run_ocr_result()` and consumed by `PageProcessor`; `PageProcessingResult` is produced by `PageProcessor.process()` and consumed by API adapters. Existing tuple methods remain compatibility wrappers until Task 7 proves they can be deleted.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-page-processing-pipeline.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, with checkpoints.

Which approach?