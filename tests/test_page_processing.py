from page_processing import PageProcessingResult, PageProcessor


class FakeOCR:
    def __init__(self, *, blocks=None, engine="easyocr", engines=None,
                 diagnostics=None):
        self.blocks = blocks or []
        self.engine = engine
        self.engines = engines or [engine]
        self.diagnostics = diagnostics or {"finished": True}


def test_result_has_stable_page_contract():
    result = PageProcessingResult([], None, "es", "easyocr", ["easyocr"],
                                  {"finished": True})

    assert result == PageProcessingResult(
        [], None, "es", "easyocr", ["easyocr"], {"finished": True}
    )
    assert result.error is None


def test_processor_returns_finalized_result(mocker):
    manager = mocker.Mock()
    manager.last_dropped_noise = []
    manager.run_ocr_result.return_value = FakeOCR(
        engine="fusion", engines=["easyocr+rapid"]
    )
    mocker.patch("routes.api._finalize_page_blocks", return_value=([], "b64", "es", 0.0))

    result = PageProcessor(manager).process(
        object(), source="es", target="en", ocr_mode="fusion"
    )

    assert (result.inpainted_image, result.ocr_engine,
            result.engines_used, result.diagnostics["finished"]) == (
                "b64", "fusion", ["easyocr+rapid"], True
            )


def test_processor_returns_error_for_empty_image(mocker):
    manager = mocker.Mock()
    result = PageProcessor(manager).process(
        None, source="es", target="en", ocr_mode="fusion"
    )

    assert result.blocks == []
    assert result.error is not None
    manager.run_ocr_result.assert_not_called()


def test_processor_passes_ocr_options(mocker):
    manager = mocker.Mock()
    manager.last_dropped_noise = []
    manager.run_ocr_result.return_value = FakeOCR()
    mocker.patch("routes.api._finalize_page_blocks", return_value=([], None, "es", 0.0))

    PageProcessor(manager).process(
        object(), source="auto", target="es", ocr_mode="fusion",
        doc_id="doc123", prefilter=True, force_uocr=True,
    )

    assert manager.run_ocr_result.call_args.kwargs == {
        "ocr_mode": "fusion", "prefilter": True, "force_uocr": True,
        "disable_uocr": False, "pure_easyocr": False, "doc_id": "doc123",
    }
