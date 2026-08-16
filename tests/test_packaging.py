from pathlib import Path


def test_pyinstaller_incluye_runtime_diagnostics():
    """El mÃ³dulo importado por el OCR debe viajar junto al .exe."""
    spec = Path(__file__).parents[1] / "main.spec"
    text = spec.read_text(encoding="utf-8")
    assert "runtime_diagnostics.py" in text


def test_pyinstaller_incluye_memoria_automatica():
    """La memoria por documento debe estar disponible en el ejecutable."""
    spec = Path(__file__).parents[1] / "main.spec"
    text = spec.read_text(encoding="utf-8")
    assert "translation_memory.py" in text


def test_timeout_de_procesamiento_cubre_paginas_con_uocr():
    """El cliente no debe cancelar una página VLM antes del servidor."""
    import re

    root = Path(__file__).parents[1]
    py_config = (root / "config.py").read_text(encoding="utf-8")
    js_config = (root / "js" / "config.js").read_text(encoding="utf-8")
    py_timeout = int(re.search(
        r"TIMEOUT_PROCESS_PAGE_MS:\s*Final\[int\]\s*=\s*(\d+)",
        py_config,
    ).group(1))
    js_timeout = int(re.search(
        r"TIMEOUT_PROCESS_PAGE_MS:\s*(\d+)",
        js_config,
    ).group(1))

    assert py_timeout == js_timeout
    assert py_timeout >= 300_000


def test_lanzador_no_mata_procesos_ajenos_por_puerto():
    """El launcher debe reutilizar o informar, nunca hacer taskkill global."""
    text = (Path(__file__).parents[1] / "start-app.ps1").read_text(
        encoding="utf-8"
    ).lower()

    assert "taskkill" not in text
    assert "netstat" not in text


def test_launchers_reutilizan_servidor_existente():
    root = Path(__file__).parents[1]
    main = (root / "main.py").read_text(encoding="utf-8")
    alternate = (root / "launcher.py").read_text(encoding="utf-8")

    assert "if wait_for_port(timeout=1):" in main
    assert "if port_open(HOST, PORT):" in alternate


def test_toast_no_inserta_mensajes_externos_como_html():
    """Los errores de API/OCR deben renderizarse como texto, no como markup."""
    text = (Path(__file__).parents[1] / "js" / "toast.js").read_text(
        encoding="utf-8"
    )

    assert "messageEl.textContent = String(message ?? \"\")" in text
    assert "${message}" not in text


def test_idiomas_pausados_no_se_exponen_en_la_ui():
    """La UI no debe volver a ofrecer idiomas desactivados por accidente."""
    root = Path(__file__).parents[1]
    html = (root / "index.html").read_text(encoding="utf-8")

    assert 'value="fr"' not in html
    assert 'value="de"' not in html
    assert 'value="it"' not in html
    assert "eng+spa+fra+deu" not in html


def test_servidor_precarga_rapidocr_antes_de_la_primera_pagina():
    """La primera página no debe pagar la inicialización del motor CPU."""
    server = (Path(__file__).parents[1] / "server.py").read_text(encoding="utf-8")

    assert "from ocr_utils import _get_rapid_engine" in server
    assert "engine_rapid = _get_rapid_engine()" in server


def test_traduccion_total_no_renderiza_cada_pagina_dos_veces():
    """El flujo masivo delega el render limpio en serverProcessPage()."""
    app = (Path(__file__).parents[1] / "app.js").read_text(encoding="utf-8")
    start = app.index("async function autoTranslateAllPages()")
    end = app.index("async function ", start + len("async function "))
    body = app[start:end]

    assert "const result = await renderPage(p);" not in body
    assert "async function serverProcessPage(pageNo" in app


def test_server_process_page_no_copia_canvas_completo_antes_de_codificar():
    """Optimización 2.4: el canvas limpio se envía como JPEG binario
    (toBlob) sin copiarlo a un canvas intermedio."""
    app = (Path(__file__).parents[1] / "app.js").read_text(encoding="utf-8")
    start = app.index("async function serverProcessPage(pageNo")
    end = app.index("async function autoTranslateCurrentPage", start)
    body = app[start:end]

    assert "document.createElement(\"canvas\")" not in body
    assert "cleanBgCanvas.toBlob(resolve, \"image/jpeg\", 0.92)" in body
    assert "Content-Type\": \"image/jpeg\"" in body


def test_servidor_precarga_corrector_ocr_en_background():
    """La primera pÃ¡gina no debe cargar el diccionario ortogrÃ¡fico."""
    server = (Path(__file__).parents[1] / "server.py").read_text(encoding="utf-8")

    assert "from ocr_utils import _get_spellchecker" in server
    assert "_get_spellchecker()" in server
