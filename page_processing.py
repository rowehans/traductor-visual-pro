"""Structured page-processing result and coordinator."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PageProcessingResult:
    blocks: list[dict[str, Any]]
    inpainted_image: str | None
    detected_lang: str
    ocr_engine: str
    engines_used: list[str]
    diagnostics: dict[str, Any]
    noise_removed: int = 0
    error: str | None = None


class PageProcessor:
    """Coordinates OCR and delegates finalization to the existing API seam."""

    def __init__(self, ocr_manager: Any = None) -> None:
        if ocr_manager is None:
            from ocr_engine import OCRManager
            ocr_manager = OCRManager()
        self.ocr_manager = ocr_manager

    def process(
        self,
        image: Any,
        *,
        source: str,
        target: str,
        ocr_mode: str,
        doc_id: str = "",
        prefilter: bool = False,
        force_uocr: bool = False,
        disable_uocr: bool = False,
        pure_easyocr: bool = False,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        detected_lang: str | None = None,
        response_format: str = "png",
    ) -> PageProcessingResult:
        if image is None:
            return PageProcessingResult(
                [], None, detected_lang or source, "", [], {}, 0, "image is required"
            )

        ocr = self.ocr_manager.run_ocr_result(
            image, source,
            ocr_mode=ocr_mode,
            prefilter=prefilter,
            force_uocr=force_uocr,
            disable_uocr=disable_uocr,
            pure_easyocr=pure_easyocr,
            doc_id=doc_id,
        )
        from routes.api import _finalize_page_blocks

        removed: list[dict[str, Any]] = []
        blocks, inpainted, language, _ = _finalize_page_blocks(
            image, ocr.blocks, source, target, scale_x, scale_y,
            detected_lang or source, doc_id, response_format,
            dropped_noise=getattr(self.ocr_manager, "last_dropped_noise", []),
            noise_removed_out=removed,
        )
        return PageProcessingResult(
            blocks, inpainted, language, ocr.engine, ocr.engines,
            ocr.diagnostics,
            len(removed),
        )
