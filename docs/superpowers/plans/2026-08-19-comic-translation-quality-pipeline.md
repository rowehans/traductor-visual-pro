# Comic Translation Quality Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current OCR/translation pipeline into a publication-gated, visually auditable pipeline that preserves comic artwork, separates independent text containers, rejects incomplete translations, and works without manga-specific vocabulary.

**Architecture:** Keep `ocr_engine.py` as the orchestration layer, `ocr_utils.py` as the image/OCR layer, `word_segmenter.py` as the generic Spanish segmentation layer, and `translator.py` as the translation/validation layer. Add a small `visual_quality.py` contract/gate module, then make the API, audit scripts, renderer, frontend, tests, and PyInstaller smoke checks consume explicit block statuses instead of silently treating rejected text as translated. The broad request is split into four independently testable workstreams: OCR coverage, translation quality, visual rendering, and audit/release gating; this parent plan sequences them so each workstream leaves a usable intermediate result.

**Tech Stack:** Python 3.12, Flask, OpenCV, NumPy, PIL/Pillow, EasyOCR, RapidOCR, Unlimited-OCR daemon, CTranslate2, pytest, existing JavaScript frontend, PyInstaller.

## Global Constraints

- **No manga-specific vocabulary:** do not add character names, places, series terms, or hand-maintained content lists anywhere in the pipeline; use only generic language-frequency corpora, geometry, image features, and contextual scoring.
- `word_segmenter.py`, `ocr_utils.py`, `translator.py`, `quality_scorer.py`, and all new pipeline modules must remain universal for arbitrary comics.
- Preserve the current EasyOCR-before-CT2 CUDA initialization order in `server.py`.
- Do not add a dependency unless it is already present in `requirements.txt` or the existing virtual environment and its use is verified first.
- Preserve the current API compatibility for `/api/process-page` and `/api/process-page-batch`; new metadata is additive.
- Never silently return original dialogue as a successful translation; use an explicit review status.
- Do not use a global white fill on dark artwork; inpainting must operate on glyph masks and preserve borders, panel art, and dark backgrounds.
- Every task ends with a focused test run and a narrow commit; never stage unrelated pre-existing working-tree changes.
- The final `.exe` is rebuilt only after the unit, integration, visual-regression, and smoke checks pass.

---

## Scope and Workstreams

The request contains independent subsystems. Keep these boundaries during implementation:

1. **OCR coverage and segmentation:** recover text inside `image` regions, classify containers, preserve candidates, and improve generic segmentation.
2. **Translation quality:** add context, semantic validation, explicit rejection states, and safe fallbacks.
3. **Visual rendering:** implement dual-polarity glyph masks, protected artwork, independent containers, and layout validation.
4. **Audit and release gating:** generate the required JSON, enforce critical-defect gates, test the full chapter, and rebuild the executable.

Tasks 1–3 form workstream 1, Task 4 forms workstream 2, Tasks 5–6 form workstream 3, and Tasks 7–8 form workstream 4. Each task below includes its own test cycle and commit.

## File Map

| File | Responsibility after this plan |
|---|---|
| `visual_quality.py` | New typed metadata contract, validation helpers, critical-defect gate, and page scoring used by the audit path and API metadata. |
| `ocr_engine.py` | Orchestrate multi-engine candidate recovery and pass structured OCR diagnostics without deciding that low-confidence text does not exist. |
| `ocr_utils.py` | Generic block grouping, SFX/container classification, image-region re-OCR, segmentation metadata, polarity analysis, glyph masks, and mask diagnostics. |
| `word_segmenter.py` | Return ranked generic segmentation candidates while protecting unknown entities and preserving the existing `segment_text()` compatibility wrapper. |
| `translator.py` | Produce checked translation results, detect residual source language and omissions generically, and preserve only explicitly non-translatable blocks. |
| `routes/api.py` | Integrate statuses into `/api/process-page` and `/api/process-page-batch`; remove silent original-text success fallbacks. |
| `app.js` | Render status-aware blocks, avoid presenting `NEEDS_REVIEW` output as final, and expose diagnostic reasons without changing the existing editor flow. |
| `.agents/skills/comic-chapter-translator/scripts/quality_scorer.py` | Score completeness, OCR, translation, inpainting, and layout with critical-defect overrides. |
| `.agents/skills/comic-chapter-translator/scripts/evaluate_chapter.py` | Preserve raw candidates/masks, invoke the common quality contract, and emit the required per-page JSON. |
| `.agents/skills/comic-chapter-translator/scripts/pdf_exporter.py` | Render independent containers, validate wrapping/overflow, and emit layout diagnostics. |
| `.agents/skills/comic-chapter-translator/scripts/html_report_generator.py` | Display element type, polarity, OCR status, translation status, mask status, and recommendations. |
| `tests/test_visual_quality.py` | New tests for the shared contract and critical gates. |
| `tests/test_ocr_utils.py` | Tests for generic classification, region recovery, segmentation metadata, polarity, and glyph masks. |
| `tests/test_ocr_engine.py` | Tests for candidate recovery, deduplication, and diagnostic propagation. |
| `tests/test_word_segmenter.py` | Tests for ranked generic segmentation and unknown-token protection. |
| `tests/test_translator.py` | Tests for rejection, residual-language detection, omissions, context, and SFX preservation. |
| `tests/test_api.py` | Tests for additive API metadata and no silent fallback. |
| `tests/test_quality_analysis.py` | Tests for report schema, critical gates, and chapter scoring. |
| `tests/test_packaging.py` | Tests that new statically/dynamically imported modules are present in the frozen bundle. |
| `README.md` | Document the new quality gate, report fields, commands, and publication criteria. |
| `main.spec` | Modify only if packaging tests prove the new module is not collected automatically. |

---

### Task 1: Establish the typed block contract and fail-closed quality gate

**Files:**
- Create: `visual_quality.py`
- Create: `tests/test_visual_quality.py`
- Modify: `routes/api.py` only for importing the contract after the tests define the interface
- Test: `tests/test_visual_quality.py`

**Interfaces:**
- Consumes: existing OCR block dictionaries containing `text`, `translated`, `confidence`, `x`, `y`, `w`, and `h`.
- Produces: `BlockType`, `Polarity`, `OcrStatus`, `TranslationStatus`, `AuditElement`, `validate_page_report()`, `score_page()`, and `page_status()`.

Define these interfaces exactly:

```python
from typing import Literal, TypedDict

BlockType = Literal["dialogue_bubble", "narrative_box", "side_note", "sfx"]
Polarity = Literal["dark_on_light", "light_on_dark"]
OcrStatus = Literal["OK", "REPAIRED_SEGMENTATION", "ERROR"]
TranslationStatus = Literal[
    "TRANSLATED", "PRESERVED_SFX", "PRESERVED_SIDE_NOTE", "NEEDS_REVIEW"
]

class AuditElement(TypedDict, total=False):
    id: int
    type: BlockType
    original_text: str
    translated_text: str
    polarity: Polarity
    ocr_status: OcrStatus
    translation_status: TranslationStatus
    confidence: float
    review_reasons: list[str]


def validate_page_report(page: dict) -> list[str]: ...
def score_page(page: dict) -> float: ...
def page_status(page: dict, score: float) -> str: ...
```

- [ ] **Step 1: Write the failing contract tests.**

```python
from visual_quality import page_status, score_page, validate_page_report


def test_long_identical_dialogue_is_a_critical_translation_failure():
    page = {
        "page_number": 1,
        "elements_detected": [{
            "id": 1,
            "type": "dialogue_bubble",
            "original_text": "A long source sentence",
            "translated_text": "A long source sentence",
            "polarity": "dark_on_light",
            "ocr_status": "OK",
            "translation_status": "NEEDS_REVIEW",
            "review_reasons": ["identical_to_source"],
        }],
    }
    assert "critical_translation_failure" in validate_page_report(page)
    assert page_status(page, 95.0) == "NEEDS_CORRECTION"


def test_sfx_can_be_preserved_without_being_a_translation_failure():
    page = {
        "page_number": 1,
        "elements_detected": [{
            "id": 1,
            "type": "sfx",
            "original_text": "THUD",
            "translated_text": "THUD",
            "polarity": "light_on_dark",
            "ocr_status": "OK",
            "translation_status": "PRESERVED_SFX",
        }],
    }
    assert validate_page_report(page) == []


def test_missing_required_metadata_is_rejected():
    page = {"page_number": 1, "elements_detected": [{"id": 1}]}
    errors = validate_page_report(page)
    assert "missing_element_type" in errors
    assert "missing_element_polarity" in errors
```

- [ ] **Step 2: Run the focused tests and verify they fail for the missing module or missing behavior.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_visual_quality.py -q
```

Expected: collection or assertion failure because `visual_quality.py` does not yet provide the contract.

- [ ] **Step 3: Implement the minimal contract.**

Implement `validate_page_report()` with explicit errors for missing fields, invalid enum values, empty dialogue translations, and identical non-SFX/non-side-note text. Implement `page_status()` so any `critical_*` error returns `NEEDS_CORRECTION` regardless of score. Keep the score weights in named constants: coverage 25, OCR 20, translation 25, inpainting 20, layout 10.

- [ ] **Step 4: Run the focused tests and type-check the new module.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_visual_quality.py -q
env/Scripts/python.exe -m mypy visual_quality.py --strict
```

Expected: all focused tests pass; mypy exits 0. If strict typing exposes a missing annotation, fix the annotation rather than adding a broad ignore.

- [ ] **Step 5: Commit the self-contained contract.**

```bash
git add visual_quality.py tests/test_visual_quality.py
git commit -m "feat(qa): add fail-closed visual quality contract"
```

---

### Task 2: Preserve OCR candidates and recover text inside image regions

**Files:**
- Modify: `ocr_utils.py:_group_and_merge_blocks`, `_recover_regions_with_easyocr`, block filtering/classification helpers
- Modify: `ocr_engine.py:OCRManager.run_ocr`, fusion and Unlimited-OCR recovery paths
- Modify: `routes/api.py:_finalize_page_blocks` to retain ignored candidates with reasons
- Modify: `tests/test_ocr_utils.py`, `tests/test_ocr_engine.py`
- Test: the two modified test files

**Interfaces:**
- Consumes: raw EasyOCR/RapidOCR/U-OCR detections and U-OCR `image_panels`.
- Produces:

```python
def recover_text_regions(
    img_bgr: Any,
    image_panels: list[dict],
    *,
    ocr_lang: str = "auto",
    scale: float = 3.0,
) -> list[dict]: ...


def classify_block_type(block: dict, *, page_shape: tuple[int, int]) -> str: ...


def deduplicate_blocks(
    blocks: list[dict], *, iou_threshold: float = 0.35
) -> list[dict]: ...
```

Every returned block must retain `raw_candidates`, `source_region`, `type`, and `review_reasons`.

- [ ] **Step 1: Add failing tests for image-region recovery, independent SFX, and non-destructive filtering.**

```python
def test_image_panel_is_reocr_candidate_even_without_a_bubble():
    image = make_dark_panel_with_light_text("HELLO")
    blocks = recover_text_regions(
        image,
        [{"type": "image", "x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]}],
        ocr_lang="en",
    )
    assert blocks
    assert blocks[0]["source_region"] == "image_panel"


def test_sfx_and_dialogue_are_not_merged_when_their_geometry_differs():
    blocks = [
        {"text": "THUD", "x": 10, "y": 10, "w": 60, "h": 25, "type": "sfx"},
        {"text": "Please wait.", "x": 10, "y": 70, "w": 180, "h": 40, "type": "dialogue_bubble"},
    ]
    merged = _group_and_merge_blocks(blocks)
    assert len(merged) == 2
    assert {b["type"] for b in merged} == {"sfx", "dialogue_bubble"}


def test_watermark_is_annotated_not_deleted_from_audit_candidates():
    blocks = [{"text": "https://example.test/page", "x": 2, "y": 990, "w": 200, "h": 14}]
    result = filter_blocks_for_processing(blocks, page_height=1000)
    assert result[0]["ignored"] is True
    assert result[0]["ignored_reason"] == "side_note_or_watermark"
```

Use synthetic generic text and a generic URL; do not add a comic/site-specific pattern.

- [ ] **Step 2: Run the focused tests to confirm the regression.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_ocr_utils.py -k "image_panel or sfx or watermark" -q
env/Scripts/python.exe -m pytest tests/test_ocr_engine.py -k "region or candidate or deduplicate" -q
```

Expected: failures because image panels are currently discarded from normal OCR recovery and block filtering does not expose an audit reason.

- [ ] **Step 3: Implement region recovery and candidate retention.**

For every U-OCR `image` panel, crop with a 5–10% bounded margin, run the existing RapidOCR/EasyOCR readers at 1x and 3x, and attach the source region to each candidate. Do not call a new library. Deduplicate candidates by containment/IoU plus normalized text similarity. Keep candidates that are not selected in `raw_candidates` for audit diagnostics.

Change filters so they annotate `ignored`, `ignored_reason`, and `translatable=False` instead of deleting data before the auditor sees it. Keep the API’s `blocks` list backward compatible by returning processable blocks there and placing ignored blocks in additive `audit_candidates` metadata.

Classify types using geometry, local typography, line structure, panel position, URL structure, and language evidence. The classifier must not use character or series names.

- [ ] **Step 4: Run OCR-engine and API regression tests.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_ocr_utils.py tests/test_ocr_engine.py tests/test_api.py -q
```

Expected: all existing tests plus the new recovery/filter tests pass. Confirm no assertion expects the old behavior of silently deleting a watermark; update that assertion to inspect `ignored_reason`.

- [ ] **Step 5: Commit the OCR coverage work.**

```bash
git add ocr_utils.py ocr_engine.py routes/api.py tests/test_ocr_utils.py tests/test_ocr_engine.py tests/test_api.py
git commit -m "fix(ocr): recover text from image regions without losing audit candidates"
```

---

### Task 3: Return ranked generic segmentation candidates and protect unknown entities

**Files:**
- Modify: `word_segmenter.py:segment_text`, `_viterbi_segment`, and generic corpus loading
- Modify: `ocr_utils.py` call sites that currently replace text without status metadata
- Modify: `tests/test_word_segmenter.py`, `tests/test_ocr_utils.py`
- Test: the two modified test files

**Interfaces:**
- Consumes: one OCR token or phrase and the existing generic Spanish frequency corpus.
- Produces:

```python
from typing import NamedTuple

class SegmentationCandidate(NamedTuple):
    text: str
    score: float
    changed: bool
    protected_unknown: bool


def segment_text_candidates(text: str, *, limit: int = 3) -> list[SegmentationCandidate]: ...
def segment_text(text: str) -> str:  # compatibility wrapper; returns top candidate
```

- [ ] **Step 1: Write failing tests for generic splits, alternatives, and protected unknowns.**

```python
def test_segmenter_returns_ranked_generic_candidates():
    candidates = segment_text_candidates("YAQUE")
    assert candidates[0].text == "YA QUE"
    assert candidates[0].score >= candidates[-1].score


def test_segmenter_does_not_force_a_low_confidence_split_of_unknown_entity():
    candidates = segment_text_candidates("QZRMELTAN")
    assert candidates[0].protected_unknown is True
    assert candidates[0].text == "QZRMELTAN"


def test_compatibility_wrapper_returns_top_candidate():
    assert segment_text("SIME") == "SI ME"
```

- [ ] **Step 2: Run the focused tests and record the expected failures.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_word_segmenter.py -q
```

Expected: failure because only the single-string API currently exists and unknown-token protection is not exposed as metadata.

- [ ] **Step 3: Implement ranked Viterbi output without adding content vocabulary.**

Keep `_FUNCTION_WORDS` and corpus data generic. Generate up to three dynamic-programming paths, score them with the existing frequency/length logic, and mark `protected_unknown=True` when the best split is not materially better than preserving the token. Preserve case and punctuation in the compatibility wrapper.

Update OCR post-processing to set:

```python
block["ocr_status"] = "REPAIRED_SEGMENTATION" if candidate.changed else "OK"
block["segmentation_confidence"] = candidate.score
block["segmentation_alternatives"] = [c.text for c in candidates[1:]]
```

- [ ] **Step 4: Run segmentation, OCR, and type checks.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_word_segmenter.py tests/test_ocr_utils.py -q
env/Scripts/python.exe -m mypy word_segmenter.py --strict
```

Expected: all tests pass and strict mypy exits 0.

- [ ] **Step 5: Commit the segmentation work.**

```bash
git add word_segmenter.py ocr_utils.py tests/test_word_segmenter.py tests/test_ocr_utils.py
git commit -m "fix(ocr): expose ranked generic word segmentation"
```

---

### Task 4: Make translation validation contextual and fail closed

**Files:**
- Modify: `translator.py:_translate_one`, `_es_traduccion_valida`, `_translation_is_likely_source_language`, and post-processing
- Modify: `routes/api.py:_finalize_page_blocks`
- Modify: `tests/test_translator.py`, `tests/test_api.py`
- Test: the focused translator/API suite

**Interfaces:**
- Consumes: normalized OCR text, block type, source/target languages, neighboring block context, and OCR confidence.
- Produces:

```python
from typing import TypedDict

class TranslationContext(TypedDict, total=False):
    previous_text: str
    next_text: str
    container_type: str
    page_language: str

class TranslationResult(TypedDict):
    text: str
    status: str  # TRANSLATED | PRESERVED_SFX | PRESERVED_SIDE_NOTE | NEEDS_REVIEW
    reasons: list[str]
    engine: str


def translate_checked(
    text: str,
    source: str,
    target: str,
    *,
    block_type: str = "dialogue_bubble",
    context: TranslationContext | None = None,
    confidence: float = 0.0,
) -> TranslationResult: ...
```

Keep `_translate_one()` returning `str` for existing callers, but make `translate_checked()` the only entry point used by `routes/api.py` for block assembly.

- [ ] **Step 1: Write failing tests for silent fallback, omitted content, source residuals, and SFX preservation.**

```python
def test_rejected_dialogue_is_not_returned_as_successful_original():
    result = translate_checked(
        "A long source dialogue",
        "es",
        "en",
        block_type="dialogue_bubble",
    )
    # Monkeypatch the engines in the test so every engine returns the source.
    assert result["status"] == "NEEDS_REVIEW"
    assert "identical_to_source" in result["reasons"]


def test_sfx_is_preserved_without_translation_call():
    result = translate_checked("THUD", "es", "en", block_type="sfx")
    assert result == {
        "text": "THUD",
        "status": "PRESERVED_SFX",
        "reasons": ["non_translatable_sfx"],
        "engine": "none",
    }


def test_translation_rejects_number_omission():
    result = validate_translation("There are 5 items", "There are items", "en", "en")
    assert result.valid is False
    assert "number_omitted" in result.reasons
```

- [ ] **Step 2: Run the focused tests to confirm the current fallback behavior fails.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_translator.py -k "fallback or identical or omission or sfx" -q
```

Expected: at least the new fail-closed assertions fail because the current pipeline can return the source text as if it were usable output.

- [ ] **Step 3: Implement generic translation checks and explicit statuses.**

Add checks for:

- Empty output.
- Output identical to source for non-SFX/non-side-note blocks.
- Missing numbers.
- Missing negation.
- Question/exclamation structure loss.
- Abnormal length ratio.
- Residual source-language probability using existing language detection and generic corpus evidence.
- CJK residuals in western-language output.
- Repeated or malformed output.

Pass `previous_text` and `next_text` to the fallback prompt/request only when the engine supports context; keep CT2’s existing fast path unchanged when no context is needed. If every engine fails validation, return `NEEDS_REVIEW` with the source retained only in `original_text`, not as a successful `translated_text`.

Do not add a list of expected names or phrases. Unknown proper nouns are preserved through casing, OCR confidence, repetition, and entity-shape signals.

- [ ] **Step 4: Integrate statuses into the API response.**

In `_finalize_page_blocks`, replace the current silent fallback branch with:

```python
result = translate_checked(
    text,
    block_lang,
    target,
    block_type=block_type,
    context=context_for_index(index, blocks),
    confidence=quality_by_key.get(key, 0.0),
)
block["translated"] = result["text"]
block["translation_status"] = result["status"]
block["translation_reasons"] = result["reasons"]
block["translation_engine"] = result["engine"]
```

The JSON response must remain backward compatible: retain `translated`, and add the new fields.

- [ ] **Step 5: Run focused and API tests.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_translator.py tests/test_api.py -q
```

Expected: all existing translation behavior that is genuinely valid remains passing; rejected identical/truncated output now has `NEEDS_REVIEW` instead of a false success.

- [ ] **Step 6: Commit the translation work.**

```bash
git add translator.py routes/api.py tests/test_translator.py tests/test_api.py
git commit -m "fix(translation): reject incomplete and residual source output"
```

---

### Task 5: Implement dual-polarity glyph masks with safety diagnostics

**Files:**
- Modify: `ocr_utils.py:_build_inpaint_mask`, `_inpaint_image`, `_sample_bg_color`, and polarity helpers
- Modify: `routes/api.py:_finalize_page_blocks` to return mask diagnostics
- Modify: `tests/test_ocr_utils.py`
- Test: synthetic OpenCV fixtures and existing OCR utility tests

**Interfaces:**
- Consumes: BGR image and blocks with bounding boxes, `type`, and optional polarity hints.
- Produces:

```python
def analyze_block_polarity(img_bgr: Any, block: dict) -> dict:
    """Return polarity, confidence, perimeter luminance, and ambiguity flag."""


def build_glyph_mask(img_bgr: Any, blocks: list[dict]) -> tuple[Any, list[dict]]:
    """Return a glyph-only mask and per-block mask diagnostics."""


def inpaint_with_diagnostics(
    img_bgr: Any, blocks: list[dict]
) -> tuple[Any, dict]:
    """Return image plus area, edge, polarity, and safety diagnostics."""
```

- [ ] **Step 1: Write failing synthetic-image tests.**

```python
def test_dark_text_on_light_background_does_not_mask_container_border():
    image = make_white_bubble_with_black_border_and_text("TEST")
    output, diagnostics = inpaint_with_diagnostics(
        image,
        [{"x": 20, "y": 20, "w": 120, "h": 45, "type": "dialogue_bubble"}],
    )
    assert diagnostics["blocks"][0]["polarity"] == "dark_on_light"
    assert diagnostics["blocks"][0]["touches_bbox_edge"] is False
    assert border_difference(image, output) < BORDER_DELTA_LIMIT


def test_light_text_on_dark_panel_never_creates_a_white_rectangle():
    image = make_black_panel_with_white_text("TEST")
    output, diagnostics = inpaint_with_diagnostics(
        image,
        [{"x": 20, "y": 20, "w": 120, "h": 45, "type": "dialogue_bubble"}],
    )
    assert diagnostics["blocks"][0]["polarity"] == "light_on_dark"
    assert mean_background(output) < DARK_BACKGROUND_LIMIT


def test_ambiguous_polarity_is_marked_for_review():
    image = make_gray_textured_region()
    _, diagnostics = inpaint_with_diagnostics(
        image,
        [{"x": 20, "y": 20, "w": 120, "h": 45, "type": "dialogue_bubble"}],
    )
    assert diagnostics["blocks"][0]["ambiguous"] is True
```

The test helpers must generate pixels procedurally; do not add page-specific image rules.

- [ ] **Step 2: Run the new tests to capture current mask failures.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_ocr_utils.py -k "polarity or inpaint or mask" -q
```

Expected: failure for at least the dark-background preservation and border-protection assertions if the current rectangular/threshold mask is used.

- [ ] **Step 3: Implement polarity analysis and glyph-only masks.**

For each block:

1. Sample an inset perimeter, excluding text pixels where possible.
2. Compute luminance median, mean, variance, and polarity confidence.
3. For `dark_on_light`, select dark glyph components.
4. For `light_on_dark`, select light glyph components.
5. Erode the candidate mask away from the bbox edge so borders cannot be erased.
6. Remove components that are too large, connected to a detected border, or inconsistent with text stroke geometry.
7. Mark ambiguous regions instead of guessing.

Use the existing OpenCV inpainting implementation after mask construction. Keep the original image as fallback only when the diagnostic status is `INPAINT_FAILED`, never as a successful clean result.

- [ ] **Step 4: Add mask safety thresholds to the quality contract.**

Set `visual_status="NEEDS_REVIEW"` when any of these occurs:

- Mask touches a container edge.
- Mask area exceeds the text-region safety ratio.
- Polarity confidence is below the configured threshold.
- Background delta outside glyph components exceeds the configured threshold.

Return diagnostics in the API response and include them in the audit JSON.

- [ ] **Step 5: Run focused and full OCR utility tests.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_ocr_utils.py -q
```

Expected: all existing tests plus the synthetic polarity/mask tests pass.

- [ ] **Step 6: Commit the inpainting work.**

```bash
git add ocr_utils.py routes/api.py tests/test_ocr_utils.py
git commit -m "fix(visual): preserve comic art with dual-polarity glyph masks"
```

---

### Task 6: Keep containers independent and validate typography/layout

**Files:**
- Modify: `.agents/skills/comic-chapter-translator/scripts/pdf_exporter.py:_merge_overlapping_blocks`, `_wrap_and_fit_text`, `render_page_translation`
- Modify: `app.js:drawTextOnCanvas`, block construction, and status rendering
- Create: `tests/test_layout_quality.py`
- Modify: `tests/test_quality_analysis.py` only for shared layout assertions
- Test: Python layout tests and `node --check app.js`

**Interfaces:**
- Consumes: blocks with `container_id`, `type`, `bbox`, `translated`, `translation_status`, and polarity diagnostics.
- Produces:

```python
class LayoutResult(TypedDict):
    lines: list[str]
    font_size: int
    bbox: tuple[int, int, int, int]
    overflow: bool
    overlap_ids: list[int]
    contrast_ok: bool
    reasons: list[str]


def layout_translation(block: dict, container: dict, *, font_loader) -> LayoutResult: ...
def validate_layout(results: list[LayoutResult], canvas_size: tuple[int, int]) -> list[str]: ...
```

- [ ] **Step 1: Write failing layout tests.**

```python
def test_sfx_and_dialogue_keep_separate_layouts():
    blocks = [
        {"id": 1, "type": "sfx", "container_id": "c1", "translated": "THUD"},
        {"id": 2, "type": "dialogue_bubble", "container_id": "c2", "translated": "Please wait."},
    ]
    results = layout_blocks(blocks, containers={"c1": small_box(), "c2": large_box()})
    assert len(results) == 2
    assert all(not result["overflow"] for result in results)


def test_truncated_translation_is_not_hidden_by_successful_centering():
    block = {
        "id": 1,
        "type": "dialogue_bubble",
        "container_id": "c1",
        "translated": "Short.",
        "translation_status": "NEEDS_REVIEW",
    }
    result = layout_translation(block, large_box(), font_loader=fake_font_loader)
    assert result["reasons"] == ["translation_needs_review"]
    assert result["overflow"] is False
```

- [ ] **Step 2: Run the focused layout tests and confirm the missing interface/failure.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_layout_quality.py -q
```

Expected: collection or assertion failure until the layout result and status handling are implemented.

- [ ] **Step 3: Replace broad overlap merging with container-aware grouping.**

`_merge_overlapping_blocks` may merge only when all of these match:

- Same `container_id`, or geometry proves the same closed container.
- Same `type`.
- Same polarity.
- No intervening border or independent SFX style.
- Vertical gap and horizontal overlap remain within configured generic thresholds.

If no `container_id` exists, preserve separate blocks rather than guessing a merge.

- [ ] **Step 4: Make wrapping and rendering status-aware.**

Use the container interior, not only the OCR glyph bbox, for `max_w`/`max_h`. Compute line width and total height before drawing. Return `overflow=True` instead of silently shrinking below the configured minimum font size. Do not render `NEEDS_REVIEW` text as publication-ready; show the block with a review marker in the audit renderer and keep the original in the editor.

In `app.js:drawTextOnCanvas`, consume `translation_status`, `polarity`, and `container_id`; preserve the existing shared rendering path so screen and export remain consistent.

- [ ] **Step 5: Run layout, frontend syntax, and audit-renderer tests.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_layout_quality.py tests/test_quality_analysis.py -q
node --check app.js
```

Expected: all tests pass and Node reports no syntax errors.

- [ ] **Step 6: Commit the layout work.**

```bash
git add .agents/skills/comic-chapter-translator/scripts/pdf_exporter.py app.js tests/test_layout_quality.py tests/test_quality_analysis.py
git commit -m "fix(layout): keep comic containers independent and bounded"
```

---

### Task 7: Build the required audit report and publication gate

**Files:**
- Modify: `.agents/skills/comic-chapter-translator/scripts/quality_scorer.py`
- Modify: `.agents/skills/comic-chapter-translator/scripts/evaluate_chapter.py`
- Modify: `.agents/skills/comic-chapter-translator/scripts/html_report_generator.py`
- Modify: `tests/test_quality_analysis.py`
- Test: report-generation and schema tests using synthetic page data

**Interfaces:**
- Consumes: processed blocks, audit candidates, translation statuses, inpainting diagnostics, layout diagnostics, and comparison image paths.
- Produces exactly the required page shape:

```python
def build_page_report(page_number: int, page_data: dict) -> dict: ...
def build_chapter_report(pages: list[dict], metadata: dict) -> dict: ...
```

Each page must contain `page_number`, `quality_score`, `status`, `elements_detected`, `visual_defects_found`, and `recommendations`. Each element must contain `id`, `type`, `original_text`, `translated_text`, `polarity`, `ocr_status`, and `notes`, with additive technical fields allowed.

- [ ] **Step 1: Write failing schema and critical-gate tests.**

```python
def test_report_has_required_fields_for_every_page_and_element():
    report = build_chapter_report([synthetic_page()], {"source_language": "es"})
    assert len(report["pages"]) == 1
    page = report["pages"][0]
    assert {"page_number", "quality_score", "status", "elements_detected",
            "visual_defects_found", "recommendations"} <= page.keys()
    assert {"id", "type", "original_text", "translated_text", "polarity",
            "ocr_status", "notes"} <= page["elements_detected"][0].keys()


def test_critical_visual_or_translation_failure_overrides_high_average_score():
    page = synthetic_page(translation_status="NEEDS_REVIEW")
    page["translation_score"] = 100.0
    page["visual_score"] = 100.0
    report = build_page_report(1, page)
    assert report["status"] == "NEEDS_CORRECTION"
```

- [ ] **Step 2: Run the report tests and confirm current automatic scoring is insufficient.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_quality_analysis.py -q
```

Expected: the new critical-gate assertion fails against the old score-only behavior.

- [ ] **Step 3: Implement pillar scores and critical overrides.**

Use these weights:

- Coverage: 25%.
- OCR/segmentation: 20%.
- Translation: 25%.
- Inpainting/art preservation: 20%.
- Typography/layout: 10%.

Force `NEEDS_CORRECTION` for missing long dialogue, translation truncation, residual source text, mask-border contact, ambiguous polarity, or layout overflow. Treat preserved SFX and side notes as valid only when their status explicitly says so.

- [ ] **Step 4: Emit the JSON and HTML report from the same data.**

`evaluate_chapter.py` must preserve the raw source candidate and diagnostics before building the final page object. `html_report_generator.py` must show:

- Original text.
- Generated translation.
- Candidate alternatives.
- Element type.
- Polarity.
- OCR status.
- Translation status.
- Mask safety status.
- Comparison image.
- Recommended action.

Do not hardcode page-specific corrections in the scorer or generator.

- [ ] **Step 5: Run report tests and validate the exact chapter shape.**

Run:

```bash
env/Scripts/python.exe -m pytest tests/test_quality_analysis.py -q
env/Scripts/python.exe -m json.tool audit_results/visual_audit_report.json > /dev/null
```

For the full fixture chapter, assert 15 pages, every element has required fields, and every critical defect produces `NEEDS_CORRECTION`.

- [ ] **Step 6: Commit the audit gate.**

```bash
git add .agents/skills/comic-chapter-translator/scripts/quality_scorer.py .agents/skills/comic-chapter-translator/scripts/evaluate_chapter.py .agents/skills/comic-chapter-translator/scripts/html_report_generator.py tests/test_quality_analysis.py
git commit -m "feat(qa): gate chapter publication on visual quality defects"
```

---

### Task 8: Run the full regression, rebuild the EXE, and document the release gate

**Files:**
- Modify: `README.md` with the new audit contract and release commands
- Modify: `tests/test_packaging.py` if the new module is dynamically loaded and missing from the bundle
- Modify: `main.spec` only if the packaging test identifies a real collection gap
- Test: full existing suite, audit fixture, PyInstaller bundle, and EXE smoke test

**Interfaces:**
- Consumes: all statuses and diagnostics from Tasks 1–7.
- Produces: a verified frozen bundle and a release report; no API contract changes beyond additive fields.

- [ ] **Step 1: Add the packaging regression assertion before building.**

```python
def test_quality_contract_is_available_in_frozen_data_or_import_graph():
    spec = Path("main.spec").read_text(encoding="utf-8")
    source = Path("visual_quality.py")
    assert source.exists()
    # The module is statically imported by routes/api.py; the test documents
    # that no broad hidden-import list is needed unless the import becomes dynamic.
    assert "visual_quality" in Path("routes/api.py").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the proportionate full test and syntax checks.**

Run:

```bash
env/Scripts/python.exe -m pytest -q
env/Scripts/python.exe -m compileall -q visual_quality.py ocr_engine.py ocr_utils.py word_segmenter.py translator.py routes
env/Scripts/python.exe -m mypy visual_quality.py word_segmenter.py --strict
node --check app.js
```

Expected: zero pytest failures, zero compile errors, zero mypy errors for the strict new modules, and zero Node syntax errors. Fix only regressions caused by this plan; do not alter unrelated working-tree changes.

- [ ] **Step 3: Run the chapter audit fixture without overwriting prior evidence.**

Use a unique output directory for the new run:

```bash
env/Scripts/python.exe .agents/skills/comic-chapter-translator/scripts/evaluate_chapter.py \
  --pdf "Capítulo 43 de Cómo criar villanos correctamente.pdf" \
  --pages all \
  --target en \
  --outdir audit_results/quality_gate_run
```

Expected: a JSON report with one record per page, comparison images, and no silent loss of candidates. The run must explicitly show which blocks are translated, preserved, ignored with reason, or pending review.

- [ ] **Step 4: Validate publication criteria from the generated report.**

Run:

```bash
env/Scripts/python.exe -c "import json; from pathlib import Path; p=json.loads(Path('audit_results/quality_gate_run/audit_report.json').read_text(encoding='utf-8')); assert len(p['pages']) == 15; assert all('elements_detected' in page for page in p['pages']); assert not any(e.get('translation_status') == 'TRANSLATED' and e.get('translated_text','').strip().casefold() == e.get('original_text','').strip().casefold() and e.get('type') not in {'sfx','side_note'} for page in p['pages'] for e in page['elements_detected']); print('quality gate report validated')"
```

Expected: exit 0 only when the report has complete page records and no false-success identical dialogue translations.

- [ ] **Step 5: Build the frontend and frozen executable.**

Run:

```bash
env/Scripts/python.exe build.py
env/Scripts/python.exe -m PyInstaller main.spec --clean --noconfirm
```

Expected: both commands exit 0 and `dist/main/main.exe` exists. Verify that `dist/main/_internal/` contains `routes/`, `js/`, `resources/`, `ocr_engine.py`, `uocr_client.py`, `uocr_daemon.py`, `word_segmenter.py`, and the new quality module if it is loaded as data.

- [ ] **Step 6: Run the EXE smoke test and inspect logs.**

Run the existing local smoke-test procedure used by the repository, with a unique log/checkpoint path and without touching production services:

```bash
env/Scripts/python.exe main.py --server
```

Then exercise `/api/health`, `/api/translate`, `/api/process-page`, and `/api/process-page-batch` against a small fixture. Confirm:

- API returns additive status fields.
- A rejected translation is marked `NEEDS_REVIEW`.
- Dark and light polarity both return diagnostics.
- The daemon can still be adopted/launched as configured.
- The EXE/frontend bundle loads all JS modules.

Stop the local server after the smoke test and record the command output/log path in the release note.

- [ ] **Step 7: Document the new release gate.**

Update `README.md` with:

- The four quality pillars.
- The JSON status meanings.
- The no-specific-vocabulary rule.
- The chapter audit command.
- The full test command.
- The PyInstaller command.
- The rule that `NEEDS_CORRECTION` blocks publication.

- [ ] **Step 8: Commit the verified release changes.**

```bash
git add README.md tests/test_packaging.py main.spec
git commit -m "build: verify quality-gated chapter release in EXE"
```

Only include `main.spec` if it changed in this work; otherwise stage only the documentation and packaging test.

---

## Acceptance Criteria

The implementation is ready for publication only when all of these are true:

- Every visible candidate is classified as dialogue, narrative box, side note, or SFX, or has an explicit `ignored_reason`.
- Text inside `image` regions receives a recovery attempt before the page is declared complete.
- No long dialogue remains identical to the source unless its status is `NEEDS_REVIEW`.
- No translation is marked successful when numbers, negation, question structure, or substantial content were omitted.
- SFX and side notes are preserved or ignored explicitly, never mistaken for successful dialogue translation.
- OCR segmentation uses only generic language-frequency data and protects uncertain unknown entities.
- Dark-on-light and light-on-dark masks are selected per block, with ambiguous cases sent to review.
- Bubble borders, panel lines, and dark artwork are not overwritten by rectangular fills.
- No final text overflows, clips, or overlaps an independent container.
- The JSON report follows the requested per-page structure for every page.
- Any critical visual, OCR, coverage, translation, or layout failure forces `NEEDS_CORRECTION`.
- The focused tests, full pytest suite, syntax checks, audit fixture, PyInstaller build, and EXE smoke test pass.
- The executable is rebuilt only after the above evidence exists.

## Self-Review

- **Spec coverage:** detection completeness is covered by Task 2; OCR/segmentation by Task 3; translation fidelity by Task 4; dual-polarity inpainting by Task 5; independent containers and typography by Task 6; required JSON/scoring by Task 7; packaging and release verification by Task 8.
- **Scope:** the broad request is explicitly decomposed into four workstreams with independent test cycles and narrow commits.
- **No placeholders:** every task names exact files, interfaces, test commands, expected outcomes, and commit commands; no step depends on an unspecified future decision.
- **Type consistency:** `TranslationResult`, `TranslationContext`, `AuditElement`, `LayoutResult`, and all public helper signatures are defined before later tasks consume them.
- **Universal behavior:** all examples use generic words or synthetic fixtures; no character, location, series, or site vocabulary is added to production logic.
- **Release safety:** the plan preserves existing API fields and the EasyOCR/CT2 load order, and explicitly prevents broad staging of unrelated working-tree changes.
