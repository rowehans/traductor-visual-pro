"""
test_correccion_detector.py — Flujo de "darle clases" al detector.

Prueba tools/exportar_anotaciones.py (pseudo-etiquetas VLM → workspace YOLO
editable) y tools/fusionar_correcciones.py (correcciones manuales → dataset
re-entrenable). Los módulos se cargan en fresco por test (importlib con nombre
único, mismo patrón que test_process_all_pages.py); no tocan el daemon ni
ultralytics — solo copian archivos y parsean .txt YOLO.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_TOOLS = {
    "exportar": _ROOT / "tools" / "exportar_anotaciones.py",
    "fusionar": _ROOT / "tools" / "fusionar_correcciones.py",
    "etiquetar": _ROOT / "tools" / "etiquetar_con_vlm.py",
    "entrenar": _ROOT / "tools" / "entrenar_detector.py",
    "calificar": _ROOT / "tools" / "calificar_detector.py",
}

_load_counter = [0]


def _load_tool(nombre: str):
    """Carga una tool en fresco (módulos con nombre único por test)."""
    _load_counter[0] += 1
    spec = importlib.util.spec_from_file_location(
        f"{nombre}_mod_{_load_counter[0]}", _TOOLS[nombre])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv):
    old = sys.argv
    sys.argv = [str(_TOOLS["exportar"])] + argv
    try:
        return mod.main()
    finally:
        sys.argv = old


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def dataset_teacher(tmp_path):
    """Dataset pseudo-etiquetas: 2 páginas train + 1 val + manifest."""
    vlm = tmp_path / "vlm"
    for split, paginas in (("train", ["p001", "p002"]), ("val", ["p003"])):
        (vlm / split / "images").mkdir(parents=True)
        (vlm / split / "labels").mkdir(parents=True)
        for p in paginas:
            (vlm / split / "images" / f"{p}.jpg").write_bytes(b"IMG")
            n = 2 if p == "p001" else 1
            with open(vlm / split / "labels" / f"{p}.txt", "w") as f:
                for i in range(n):
                    f.write(f"{i % 2} 0.5 0.5 0.2 0.1\n")
    manifest = {
        "fuente": "test.pdf", "clases": ["text_bubble", "text_free"],
        "paginas": [
            {"imagen": "p001.jpg", "split": "train", "pagina": 1},
            {"imagen": "p002.jpg", "split": "train", "pagina": 2},
            {"imagen": "p003.jpg", "split": "val", "pagina": 3},
        ],
    }
    (vlm / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return vlm


@pytest.fixture
def workspace(tmp_path, dataset_teacher):
    """Workspace YOLO plano (como lo deja exportar_anotaciones)."""
    w = tmp_path / "corregir"
    (w / "images").mkdir(parents=True)
    (w / "labels").mkdir(parents=True)
    for split in ("train", "val"):
        for img in (dataset_teacher / split / "images").glob("*.jpg"):
            (w / "images" / img.name).write_bytes(img.read_bytes())
            lab = dataset_teacher / split / "labels" / f"{img.stem}.txt"
            (w / "labels" / f"{img.stem}.txt").write_bytes(lab.read_bytes())
    return w


# ─── Exportar ────────────────────────────────────────────────────

class TestExportarAnotaciones:

    def test_exporta_workspace_yolo_completo(self, tmp_path, dataset_teacher):
        mod = _load_tool("exportar")
        out = tmp_path / "corregir"
        _run(mod, ["--data", str(dataset_teacher), "--out", str(out)])

        imgs = sorted(p.name for p in (out / "images").glob("*.jpg"))
        assert imgs == ["p001.jpg", "p002.jpg", "p003.jpg"]  # train+val fusionados
        for p in ("p001", "p002", "p003"):
            assert (out / "labels" / f"{p}.txt").exists()
        classes = (out / "classes.txt").read_text().split()
        assert classes == ["text_bubble", "text_free"]
        yaml = (out / "data.yaml").read_text()
        assert "names: ['text_bubble', 'text_free']" in yaml
        assert (out / "LEEME.txt").exists()
        assert "X-AnyLabeling" in (out / "LEEME.txt").read_text(encoding="utf-8")

    def test_exporta_sin_dataset_aborta(self, tmp_path):
        mod = _load_tool("exportar")
        with pytest.raises(SystemExit):
            _run(mod, ["--data", str(tmp_path / "nada"),
                       "--out", str(tmp_path / "out")])


# ─── Fusionar ────────────────────────────────────────────────────

class TestFusionarCorrecciones:

    def test_merge_reemplaza_solo_paginas_corregidas(self, tmp_path,
                                                     dataset_teacher, workspace):
        mod = _load_tool("fusionar")
        # p001 corregida: 2 → 3 etiquetas (una nueva bubble)
        with open(workspace / "labels" / "p001.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.1\n1 0.3 0.7 0.15 0.08\n0 0.8 0.2 0.25 0.12\n")
        _run(mod, ["--corregidas", str(workspace), "--data", str(dataset_teacher)])

        train_lab = (dataset_teacher / "train" / "labels" / "p001.txt").read_text()
        assert len(train_lab.splitlines()) == 3  # corregida
        # p002 (train) no tocada → conserva las 1 del teacher
        p002 = (dataset_teacher / "train" / "labels" / "p002.txt").read_text()
        assert len(p002.splitlines()) == 1
        # p003 (val) corregida igual que teacher → conserva 1
        p003 = (dataset_teacher / "val" / "labels" / "p003.txt").read_text()
        assert len(p003.splitlines()) == 1
        # el split se preserva: p003 sigue en val
        assert (dataset_teacher / "val" / "images" / "p003.jpg").exists()

    def test_merge_pagina_nueva_va_a_train(self, tmp_path, dataset_teacher,
                                           workspace):
        mod = _load_tool("fusionar")
        # Usuario añade p099 (nueva, sin manifest)
        (workspace / "images" / "p099.jpg").write_bytes(b"IMG")
        (workspace / "labels" / "p099.txt").write_text("0 0.5 0.5 0.3 0.15\n")
        _run(mod, ["--corregidas", str(workspace), "--data", str(dataset_teacher)])

        assert (dataset_teacher / "train" / "images" / "p099.jpg").exists()
        assert (dataset_teacher / "train" / "labels" / "p099.txt").exists()

    def test_merge_pagina_vaciada_escribe_negativo(self, tmp_path,
                                                   dataset_teacher, workspace):
        mod = _load_tool("fusionar")
        (workspace / "labels" / "p001.txt").write_text("")  # sin texto
        _run(mod, ["--corregidas", str(workspace), "--data", str(dataset_teacher)])
        # .txt vacío = página sin texto (ejemplo negativo válido)
        assert (dataset_teacher / "train" / "labels" / "p001.txt").read_text() == ""

    def test_leer_labels_filtra_invalidas(self, tmp_path):
        mod = _load_tool("fusionar")
        lab = tmp_path / "p001.txt"
        lab.write_text(
            "0 0.5 0.5 0.2 0.1\n"          # válida
            "5 0.5 0.5 0.2 0.1\n"          # clase inexistente → filtro
            "1 0.5 1.7 0.2 0.1\n"          # cy fuera de rango → filtro
            "0 0.5 0.5 0.2\n"              # 4 campos → filtro
            "1 0.5 0.5 0.2 0.1\n"          # válida
        )
        labels = mod._leer_labels(lab, "p001")
        assert len(labels) == 2
        assert labels[0][0] == 0.0 and labels[1][0] == 1.0

    def test_buscar_etiquetas_en_varios_layouts(self, tmp_path):
        mod = _load_tool("fusionar")
        # layout labels/ (estándar)
        w1 = tmp_path / "w1"
        (w1 / "labels").mkdir(parents=True)
        (w1 / "labels" / "p001.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        assert mod._buscar_etiquetas(w1, "p001") == w1 / "labels" / "p001.txt"
        # layout plano (junto a la imagen)
        w2 = tmp_path / "w2"
        w2.mkdir()
        (w2 / "p001.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        assert mod._buscar_etiquetas(w2, "p001") == w2 / "p001.txt"
        # sin etiquetas → None (página sin corregir)
        assert mod._buscar_etiquetas(tmp_path / "nada", "p001") is None

    def test_merge_sin_correcciones_avisa(self, tmp_path, dataset_teacher,
                                          workspace, capsys):
        mod = _load_tool("fusionar")
        # workspace sin labels (páginas no corregidas)
        for lab in (workspace / "labels").glob("*.txt"):
            lab.unlink()
        _run(mod, ["--corregidas", str(workspace), "--data", str(dataset_teacher)])
        assert "Sin correcciones" in capsys.readouterr().out

    def test_merge_train_lanza_entrenamiento(self, tmp_path, dataset_teacher,
                                             workspace, mocker):
        mod = _load_tool("fusionar")
        run_mock = mocker.patch("subprocess.run")
        _run(mod, ["--corregidas", str(workspace), "--data", str(dataset_teacher),
                   "--train"])
        run_mock.assert_called_once()
        args = run_mock.call_args.args[0]
        assert str(args[1]).endswith("entrenar_detector.py")


# ─── Teacher (lógica pura: clasificación + normalización + dedup) ─

class TestEtiquetarConVlm:

    def test_clasificar_title_es_texto_libre(self, tmp_path):
        mod = _load_tool("etiquetar")
        bloque = {"x": 0, "y": 0, "w": 50, "h": 20, "type": "title"}
        assert mod._clasificar_bloque(bloque, []) == mod.IDX_FREE

    def test_clasificar_text_sin_overlap_es_globo(self, tmp_path):
        mod = _load_tool("etiquetar")
        bloque = {"x": 0, "y": 0, "w": 50, "h": 20, "type": "text"}
        assert mod._clasificar_bloque(bloque, []) == mod.IDX_BUBBLE

    def test_clasificar_hereda_clase_del_oraculo_yolo(self, tmp_path):
        mod = _load_tool("etiquetar")
        bloque = {"x": 0, "y": 0, "w": 100, "h": 50, "type": "text"}
        # Región ogkalu text_free que SOLAPA al bloque (misma zona)
        region_free = {"x": 0, "y": 0, "w": 100, "h": 50, "label": "text_free"}
        assert mod._clasificar_bloque(bloque, [region_free]) == mod.IDX_FREE
        region_bubble = {"x": 0, "y": 0, "w": 100, "h": 50,
                         "label": "text_bubble"}
        assert mod._clasificar_bloque(bloque, [region_bubble]) == mod.IDX_BUBBLE

    def test_clasificar_overlap_bajo_no_hereda(self, tmp_path):
        mod = _load_tool("etiquetar")
        # Región diminuta muy alejada → IoU ~0 → no hereda (queda por type)
        bloque = {"x": 0, "y": 0, "w": 100, "h": 50, "type": "text"}
        lejos = {"x": 500, "y": 500, "w": 10, "h": 10, "label": "text_free"}
        assert mod._clasificar_bloque(bloque, [lejos]) == mod.IDX_BUBBLE

    def test_bloques_a_labels_normaliza_y_filtra(self, tmp_path):
        mod = _load_tool("etiquetar")
        bloques = [
            {"x": 100, "y": 200, "w": 200, "h": 100, "text": "hola",
             "type": "text"},            # válido → bubble
            {"x": 5, "y": 5, "w": 3, "h": 3, "text": "x",
             "type": "text"},            # < MIN_BLOQUE_W/H → filtrado
            {"x": 0, "y": 0, "w": 5000, "h": 5000, "text": "big",
             "type": "text"},            # > MAX_AREA_RATIO → filtrado
            {"x": 300, "y": 50, "w": 100, "h": 40, "text": "TÍTULO",
             "type": "title"},           # válido → free
        ]
        labels = mod._bloques_a_labels(bloques, [], w_pag=1000, h_pag=800)
        assert len(labels) == 2
        assert labels[0]["cls"] == mod.IDX_BUBBLE
        assert labels[0]["cx"] == 200 / 1000 and labels[0]["cy"] == 250 / 800
        assert labels[0]["w"] == 0.2 and labels[0]["h"] == 0.125
        assert labels[1]["cls"] == mod.IDX_FREE

    def test_dedup_conserva_la_mayor(self, tmp_path):
        mod = _load_tool("etiquetar")
        grande = {"x": 0, "y": 0, "w_px": 200, "h_px": 100, "cx": .1, "cy": .1,
                  "w": .2, "h": .1, "cls": 0, "text": "a", "type": "text"}
        pequena = {"x": 50, "y": 25, "w_px": 100, "h_px": 50, "cx": .1,
                   "cy": .1, "w": .1, "h": .05, "cls": 0, "text": "b",
                   "type": "text"}   # dentro de la grande (IoU alto)
        separada = {"x": 500, "y": 500, "w_px": 100, "h_px": 50, "cx": .5,
                    "cy": .5, "w": .1, "h": .05, "cls": 1, "text": "c",
                    "type": "title"}  # sin solape
        out = mod._dedup_etiquetas([pequena, grande, separada])
        assert out == [grande, separada]  # la mayor absorbe a la pequeña

    def test_paginas_de_carpeta_orden_natural_y_filtro(self, tmp_path):
        mod = _load_tool("etiquetar")
        cap = tmp_path / "cap1"
        cap.mkdir()
        for nombre in ("1.webp", "10.webp", "2.webp", "0.webp", "nota.txt"):
            (cap / nombre).write_bytes(b"x")
        paginas = mod._paginas_de_carpeta(cap)
        assert [p.name for p in paginas] == ["0.webp", "1.webp", "2.webp",
                                             "10.webp"]

    def test_prefijo_documento_sanea_stem(self, tmp_path):
        mod = _load_tool("etiquetar")
        assert mod._prefijo_documento(
            "input_manga/BookDownloads/…/1490498") == "1490498"
        assert mod._prefijo_documento(
            "capítulo 43 de cómo criar villanos correctamente.pdf") \
            == "cap_tulo_43_de_c_mo_cria"  # truncado a 24 chars
        assert mod._prefijo_documento("!!!") == "doc"  # vacío tras sanear

    def test_main_carpeta_etiqueta_con_prefijo(self, tmp_path, monkeypatch):
        """--carpeta: cada página hereda el prefijo del documento en el nombre
        (colisión cross-capítulo imposible) y el manifest guarda la fuente."""
        mod = _load_tool("etiquetar")
        import numpy as np
        import cv2

        cap = tmp_path / "1490498"
        cap.mkdir()
        for i in range(3):
            img = np.full((120, 160, 3), 255, dtype=np.uint8)
            cv2.rectangle(img, (20, 20), (100, 60), (0, 0, 0), -1)
            cv2.imwrite(str(cap / f"{i}.webp"), img)
        bloques_fake = [{"x": 20, "y": 20, "w": 80, "h": 40,
                         "text": "HOLA", "type": "text"}]
        monkeypatch.setattr(mod, "_daemon_ocr",
                            lambda img: (bloques_fake, [], 1.5))
        import ocr_utils
        monkeypatch.setattr(ocr_utils, "_detect_text_regions_in_page",
                            lambda img: [])
        out = tmp_path / "out"

        _run(mod, ["--carpeta", str(cap), "--out", str(out)])

        # 3 páginas → split determinista (val_frac 0.15 → 1 val)
        imgs_train = sorted(p.name for p in
                            (out / "train" / "images").iterdir())
        imgs_val = sorted(p.name for p in (out / "val" / "images").iterdir())
        assert imgs_train == ["1490498_p001.jpg", "1490498_p002.jpg"]
        assert imgs_val == ["1490498_p003.jpg"]
        # labels presentes y con clase oráculo (sin yolo → bubble)
        lab = (out / "train" / "labels" / "1490498_p001.txt").read_text()
        assert lab.splitlines()[0].startswith("0 ")  # text_bubble
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert m["fuente"] == str(cap)
        assert len(m["paginas"]) == 3

    def test_main_append_prefijos_distintos_no_colisionan(
            self, tmp_path, monkeypatch):
        """Append de dos capítulos distintos: nombres con prefijos separados
        y el split previo se conserva (sin reorganizar lo etiquetado)."""
        mod = _load_tool("etiquetar")
        import numpy as np
        import cv2

        def _cap(nombre, n_pags):
            d = tmp_path / nombre
            d.mkdir()
            for i in range(n_pags):
                img = np.full((120, 160, 3), 255, dtype=np.uint8)
                cv2.imwrite(str(d / f"{i}.webp"), img)
            return d

        cap_a = _cap("serieA_cap1", 2)
        cap_b = _cap("serieB_cap2", 2)
        bloques_fake = [{"x": 20, "y": 20, "w": 80, "h": 40,
                         "text": "HOLA", "type": "text"}]
        import ocr_utils
        monkeypatch.setattr(ocr_utils, "_detect_text_regions_in_page",
                            lambda img: [])
        out = tmp_path / "out"

        monkeypatch.setattr(mod, "_daemon_ocr",
                            lambda img: (bloques_fake, [], 1.0))
        _run(mod, ["--carpeta", str(cap_a), "--out", str(out)])
        _run(mod, ["--carpeta", str(cap_b), "--out", str(out),
                   "--append"])

        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        nombres = [r["imagen"] for r in m["paginas"]]
        assert nombres == ["serieA_cap1_p001.jpg", "serieA_cap1_p002.jpg",
                           "serieB_cap2_p001.jpg", "serieB_cap2_p002.jpg"]
        # el primer capítulo conserva su split (p001 train, p002 val) y las
        # páginas del segundo no reorganizan lo ya etiquetado
        splits = [r["split"] for r in m["paginas"]]
        assert splits == ["train", "val", "train", "val"]
        assert (out / "train" / "images" /
                "serieA_cap1_p001.jpg").exists()
        assert (out / "train" / "images" /
                "serieB_cap2_p001.jpg").exists()


# ─── Entrenador (data.yaml + swap reversible) ────────────────────

class TestEntrenarDetector:

    def test_build_data_yaml_valida_y_escribe(self, tmp_path):
        mod = _load_tool("entrenar")
        for split in ("train", "val"):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "labels").mkdir(parents=True)
            (tmp_path / split / "images" / "a.jpg").write_bytes(b"x")
        yaml_path = tmp_path / "data.yaml"
        info = mod._build_data_yaml(tmp_path, yaml_path)
        assert info["n_train"] == 1 and info["n_val"] == 1
        yaml = yaml_path.read_text()
        assert "train: train/images" in yaml and "val: val/images" in yaml
        assert "names: ['text_bubble', 'text_free']" in yaml

    def test_build_data_yaml_sin_val_falla(self, tmp_path):
        mod = _load_tool("entrenar")
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "labels").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            mod._build_data_yaml(tmp_path, tmp_path / "data.yaml")

    def test_swap_respalda_y_activa(self, tmp_path):
        mod = _load_tool("entrenar")
        original = tmp_path / "modelo.pt"
        finetuned = tmp_path / "fine.pt"
        original.write_bytes(b"ORIGINAL")
        finetuned.write_bytes(b"FINETUNED")
        backup = mod._swap_model(original, finetuned)
        assert backup == tmp_path / "modelo.pt.bak"
        assert original.read_bytes() == b"FINETUNED"
        assert backup.read_bytes() == b"ORIGINAL"
        # segunda llamada no machaca el backup
        mod._swap_model(original, finetuned)
        assert backup.read_bytes() == b"ORIGINAL"

    def test_entrenar_resuelve_project_a_absoluto(self, tmp_path, monkeypatch):
        """_entrenar pasa project ABSOLUTO a ultralytics (quirk de la sesión
        153): con project relativo, get_save_dir lo redirige bajo
        SETTINGS['runs_dir'] (runs/detect/<project>/...) y el trainer no
        encuentra best.pt donde lo busca. Resolver con .resolve() garantiza
        que train() reciba una ruta absoluta y que best.pt aparezca en
        project/name/weights/."""
        import os

        mod = _load_tool("entrenar")
        weights = tmp_path / "base.pt"
        weights.write_bytes(b"W")
        yaml = tmp_path / "data.yaml"
        yaml.write_text("path: x\n")
        # CWD = tmp_path para que el project RELATIVO se resuelva dentro del
        # tmp (el test no debe escribir en el repo) — el bug de la sesión 153
        # es precisamente que project relativo se escapa a runs/detect/.
        monkeypatch.chdir(tmp_path)
        project_rel = Path("mi_proyecto_relativo")   # NO absoluto (el bug)
        name = "run_test"

        visto = {}

        class FakeYOLO:
            def __init__(self, path):
                pass
            def train(self, **kwargs):
                visto["project"] = kwargs["project"]
                # simula ultralytics: escribe best.pt donde crea save_dir
                save_dir = Path(kwargs["project"]) / kwargs["name"] / "weights"
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / "best.pt").write_bytes(b"BEST")
                (save_dir / "last.pt").write_bytes(b"LAST")

        import ultralytics
        monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

        best = mod._entrenar(weights, yaml, 1, 1, 512, "cpu", 10, 1e-5,
                             project_rel, name)

        # el project que recibió ultralytics es ABSOLUTO (no relativo) y
        # resuelto contra el CWD — el fix del quirk runs/detect
        assert os.path.isabs(visto["project"]), \
            f"project quedó relativo: {visto['project']}"
        assert visto["project"] == str((tmp_path / "mi_proyecto_relativo").resolve())
        # y best.pt está donde el trainer lo espera (project/name/weights/,
        # sin la redirección a runs/detect/<project>/)
        assert best == Path(visto["project"]) / name / "weights" / "best.pt"
        assert best.exists()

    def test_eval_rapida_usa_conf_y_mide_recall(self, tmp_path, monkeypatch):
        """_eval_rapida pasa el umbral --conf a predict y calcula recall por IoU
        contra las GT del val (la métrica decisiva de la sesión 151)."""
        import numpy as np
        import cv2
        import ocr_utils

        mod = _load_tool("entrenar")
        # imagen val 100x100 con GT: un globo en el centro (cx .5 cy .5 w .4 h .4)
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        val = tmp_path / "val"
        (val / "images").mkdir(parents=True)
        (val / "labels").mkdir(parents=True)
        img_p = val / "images" / "p001.jpg"
        cv2.imwrite(str(img_p), img)
        (val / "labels" / "p001.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        a = tmp_path / "a.pt"; b = tmp_path / "b.pt"
        a.write_bytes(b"A"); b.write_bytes(b"B")

        confs_vistos = []
        detecciones_a = [[10, 30, 90, 70, 0, 0.9]]   # cubre la GT (IoU alto)
        detecciones_b = [[10, 30, 90, 70, 0, 0.9]]   # idem

        class FakeBoxes:
            def __init__(self, dets):
                self.xyxy = [np.array(d[:4], dtype=float) for d in dets]
                self.conf = np.array([d[5] for d in dets])
                self.cls = np.array([d[4] for d in dets])
            def __len__(self):
                return len(self.xyxy)

        class FakeResult:
            def __init__(self, dets):
                self.boxes = FakeBoxes(dets) if dets else None
                self.names = {0: "text_bubble", 1: "text_free"}

        class FakeYOLO:
            def __init__(self, path):
                self._p = str(path)
            def predict(self, img, conf=None, imgsz=None, device=None, verbose=None):
                confs_vistos.append(conf)
                dets = detecciones_a if self._p.endswith("a.pt") else detecciones_b
                return [FakeResult(dets)]

        import ultralytics
        monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
        monkeypatch.setattr(mod, "_resolver_device", lambda: "cpu")
        monkeypatch.setattr(ocr_utils, "_overlap_ratio",
                            ocr_utils._overlap_ratio)

        res = mod._eval_rapida(a, b, [img_p], imgsz=512, conf=0.25,
                               labels_dir=val / "labels")
        assert confs_vistos == [0.25, 0.25]  # el umbral del pipeline se usa
        assert res["a"]["recall"] == 1.0     # 1/1 GT cubierta
        assert res["b"]["recall"] == 1.0
        assert res["a"]["gt"] == 1

    def test_eval_rapida_conf_baja_no_cubre_gt(self, tmp_path, monkeypatch):
        """Un modelo con detecciones de conf < --conf no aporta al recall
        (el A/B es el veredicto con el umbral REAL, no con conf baja)."""
        import numpy as np
        import cv2

        mod = _load_tool("entrenar")
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        val = tmp_path / "val"
        (val / "images").mkdir(parents=True)
        (val / "labels").mkdir(parents=True)
        img_p = val / "images" / "p001.jpg"
        cv2.imwrite(str(img_p), img)
        (val / "labels" / "p001.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        a = tmp_path / "a.pt"; b = tmp_path / "b.pt"
        a.write_bytes(b"A"); b.write_bytes(b"B")

        class FakeBoxes:
            def __init__(self, dets):
                self.xyxy = [np.array(d[:4], dtype=float) for d in dets]
                self.conf = np.array([d[5] for d in dets])
                self.cls = np.array([d[4] for d in dets])
            def __len__(self):
                return len(self.xyxy)

        class FakeResult:
            def __init__(self, dets):
                self.boxes = FakeBoxes(dets) if dets else None
                self.names = {0: "text_bubble"}

        class FakeYOLO:
            def __init__(self, path):
                self._p = str(path)
            def predict(self, img, conf=None, imgsz=None, device=None, verbose=None):
                # simula el filtrado real de ultralytics: descarta detecciones < conf
                dets = ([[10, 30, 90, 70, 0, 0.15]]
                        if self._p.endswith("b.pt") else [])
                if conf is not None:
                    dets = [d for d in dets if d[5] >= conf]
                return [FakeResult(dets)]

        import ultralytics
        monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
        monkeypatch.setattr(mod, "_resolver_device", lambda: "cpu")
        res = mod._eval_rapida(a, b, [img_p], imgsz=512, conf=0.25,
                               labels_dir=val / "labels")
        # la GT no se cubre con conf 0.25 (la única detección tiene 0.15)
        assert res["b"]["recall"] == 0.0
        assert res["b"]["detecciones"] == 0

    # ─── Gate del swap: --conf decide, --swap solo ejecuta si gana ─

    def _dataset_minimo(self, tmp_path):
        """dataset vlm mínimo (train+val con 1 imagen jpg) para main()."""
        import numpy as np
        import cv2
        data = tmp_path / "data"
        for split in ("train", "val"):
            (data / split / "images").mkdir(parents=True)
            (data / split / "labels").mkdir(parents=True)
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        for split in ("train", "val"):
            cv2.imwrite(str(data / split / "images" / "p001.jpg"), img)
        return data

    def _mocks_main(self, mocker, mod, data, recall_a=0.5, recall_b=0.4,
                    conf=0.25):
        """Mockea el entrenamiento y el A/B; devuelve el mock de _swap_model.
        recall_b < recall_a por defecto → el fine-tuned PIERDE.

        _swap_model se MOCKEA (no spy) a propósito: un spy EJECUTA la función
        real, y main() la llama con la ruta real de --weights (default de
        argparse) — un spy copiaría el best.pt falso del test sobre el modelo
        de producción (lección: sesión 152, corrompió
        models/comic-speech-bubble-detector.pt)."""
        out = data.parent / "out"
        import numpy as np

        def _entrenar_fake(weights, data_yaml, *a, **k):
            best = out / "run" / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"BEST")
            return best

        def _eval_fake(model_a, model_b, images, imgsz, conf=None,
                       labels_dir=None):
            return {"a": {"detecciones": 5, "conf_media": 0.8,
                           "bubble": 4, "free": 1,
                           "recall": recall_a, "gt": 10},
                    "b": {"detecciones": 6, "conf_media": 0.3,
                           "bubble": 4, "free": 2,
                           "recall": recall_b, "gt": 10}}

        mocker.patch.object(mod, "_entrenar", side_effect=_entrenar_fake)
        mocker.patch.object(mod, "_build_data_yaml",
                            return_value={"n_train": 1, "n_val": 1,
                                          "yaml": str(data / "data.yaml")})
        mocker.patch.object(mod, "_eval_rapida", side_effect=_eval_fake)
        mocker.patch.object(mod, "_resolver_device", return_value="cpu")
        mock_swap = mocker.patch.object(mod, "_swap_model",
                                        return_value=data / "backup.pt")
        return out, mock_swap

    def test_main_swap_perdedor_no_activa(self, tmp_path, mocker):
        """--swap con A/B perdedor a --conf: NO se activa (lección sesión 151:
        un modelo des-calibrado nunca debe reemplazar a ogkalu)."""
        mod = _load_tool("entrenar")
        data = self._dataset_minimo(tmp_path)
        out, spy_swap = self._mocks_main(mocker, mod, data,
                                         recall_a=0.55, recall_b=0.07)

        _run(mod, ["--data", str(data), "--out", str(out),
                   "--extra-data", "", "--epochs", "1", "--device", "cpu",
                   "--swap"])

        spy_swap.assert_not_called()
        # el original NO se sobrescribió
        assert (out / "comic-speech-bubble-detector-finetuned.pt").exists()
        # --weights por defecto apunta al modelo real de producción — el mock
        # garantiza que NUNCA se toque el disco en un test

    def test_main_swap_ganador_activa(self, tmp_path, mocker):
        """--swap con A/B ganador a --conf: sí se activa con backup."""
        mod = _load_tool("entrenar")
        data = self._dataset_minimo(tmp_path)
        out, spy_swap = self._mocks_main(mocker, mod, data,
                                         recall_a=0.55, recall_b=0.70)

        _run(mod, ["--data", str(data), "--out", str(out),
                   "--extra-data", "", "--epochs", "1", "--device", "cpu",
                   "--swap"])

        spy_swap.assert_called_once()
        # verifica que el mock recibió el modelo REAL de producción como origen
        # (args.weights default) pero no tocó el disco: solo se llamó al mock
        assert str(spy_swap.call_args.args[0]).endswith(
            "comic-speech-bubble-detector.pt")

    def test_main_swap_no_toca_disco_de_produccion(self, tmp_path, mocker):
        """REGRESIÓN de la corrupción de la sesión 152: _swap_model es un MOCK
        PURO (nunca se ejecuta el real) y main() le pasa la ruta real de
        producción como origen — pero el archivo de producción en disco queda
        INTACTO (mismos bytes antes y después de main() con --swap ganador).

        El bug de la sesión 152: el test usaba mocker.spy (que EJECUTA la
        función real) y main() lo llamaba con --weights default = el modelo de
        producción real, así que el best.pt falso del test (4 bytes b"BEST")
        sobrescribía models/comic-speech-bubble-detector.pt (corrupción
        detectada como 'pickle data was truncated')."""
        mod = _load_tool("entrenar")
        prod_model = _ROOT / "models" / "comic-speech-bubble-detector.pt"
        assert prod_model.exists(), \
            "modelo de producción ausente — el test necesita el archivo real"
        bak_model = prod_model.with_suffix(".pt.bak")
        # snapshot del estado del dir de producción (modelo + backup): si
        # _swap_model real se ejecutara, tanto el .pt como el .bak cambiarían
        estado_prod = {p.name: (p.exists(),
                                p.read_bytes() if p.exists() else None)
                       for p in (prod_model, bak_model)}

        data = self._dataset_minimo(tmp_path)
        out, mock_swap = self._mocks_main(mocker, mod, data,
                                          recall_a=0.55, recall_b=0.70)

        _run(mod, ["--data", str(data), "--out", str(out),
                   "--extra-data", "", "--epochs", "1", "--device", "cpu",
                   "--swap"])

        # 1) el mock recibió el modelo REAL de producción como origen
        mock_swap.assert_called_once()
        assert str(mock_swap.call_args.args[0]) == str(prod_model)
        # 2) ...y el disco NO se tocó: los bytes del modelo de producción son
        # idénticos (el mock puro nunca ejecutó el copy2 real)
        assert prod_model.read_bytes() == estado_prod[prod_model.name][1]
        # 3) el backup de producción tampoco cambió ni se creó uno nuevo
        assert {p.name: (p.exists(),
                         p.read_bytes() if p.exists() else None)
                for p in (prod_model, bak_model)} == estado_prod

    def test_main_sin_val_no_activa(self, tmp_path, mocker):
        """Sin imágenes val, --swap no puede verificar el modelo → no activa
        (defensa ante un dataset sin split de validación)."""
        mod = _load_tool("entrenar")
        data = tmp_path / "data"
        (data / "train" / "images").mkdir(parents=True)
        (data / "train" / "labels").mkdir(parents=True)
        out = tmp_path / "out"

        def _entrenar_fake(weights, data_yaml, *a, **k):
            best = out / "run" / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"BEST")
            return best

        mocker.patch.object(mod, "_entrenar", side_effect=_entrenar_fake)
        mocker.patch.object(mod, "_build_data_yaml",
                            return_value={"n_train": 1, "n_val": 0,
                                          "yaml": str(data / "data.yaml")})
        mocker.patch.object(mod, "_resolver_device", return_value="cpu")
        # mock PURO (no spy): aunque main() llegara a llamarlo, nunca toca disco
        mock_swap = mocker.patch.object(mod, "_swap_model",
                                        return_value=data / "backup.pt")

        _run(mod, ["--data", str(data), "--out", str(out),
                   "--extra-data", "", "--epochs", "1", "--device", "cpu",
                   "--swap"])

        mock_swap.assert_not_called()


# ─── Calificar detector: la NOTA del bucle corregir→calificar→reentrenar ─

class TestCalificarDetector:

    def _workspace_con_marcado(self, tmp_path):
        """Workspace corregir/ con 2 páginas y 2 GT c/u (marcado manual).
        img1: GT en (0.2,0.2,0.3,0.15) y (0.5,0.6,0.2,0.1) → el modelo las
        encuentra. img2: 2 GT → el modelo no detecta NADA (página perdida)."""
        import numpy as np
        import cv2
        ws = tmp_path / "corregir"
        (ws / "images").mkdir(parents=True)
        (ws / "labels").mkdir(parents=True)
        for p in ("img1", "img2"):
            img = np.full((100, 100, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(ws / "images" / f"{p}.jpg"), img)
            with open(ws / "labels" / f"{p}.txt", "w") as f:
                f.write("0 0.2 0.2 0.3 0.15\n")
                f.write("1 0.5 0.6 0.2 0.1\n")
        return ws

    def test_evaluar_pagina_conteo_hits_fns_fps(self, tmp_path):
        """_evaluar_pagina: GT cubierta = hit; GT sin cubrir = fn; detección
        extra sin GT = fp — la NOTA nace de estos tres contadores."""
        mod = _load_tool("calificar")
        gt = [{"x": 0, "y": 0, "w": 10, "h": 10},
              {"x": 50, "y": 50, "w": 10, "h": 10}]
        # 2 detecciones: una cubre la GT0 (IoU>0.3), otra no toca ninguna GT
        dets = [{"x": 1, "y": 1, "w": 8, "h": 8},
                {"x": 90, "y": 90, "w": 5, "h": 5}]
        res = mod._evaluar_pagina(dets, gt, overlap_min=0.3)
        assert res["hits"] == 1
        assert res["fns"] == 1      # la GT1 quedó sin cubrir → texto perdido
        assert res["fps"] == 1      # la detección de (90,90) no es real
        assert res["recall"] == 0.5

    def test_evaluar_pagina_desglosa_por_clase(self, tmp_path):
        """El recall se desglosa por clase: un globo encontrado y un texto
        libre perdido → recall_bubble 100%, recall_free 0%."""
        mod = _load_tool("calificar")
        gt = [{"cls": 0, "x": 0, "y": 0, "w": 10, "h": 10},
              {"cls": 1, "x": 50, "y": 50, "w": 10, "h": 10}]
        dets = [{"x": 1, "y": 1, "w": 8, "h": 8}]   # solo el globo
        res = mod._evaluar_pagina(dets, gt, overlap_min=0.3)
        assert res["gt_bubble"] == 1 and res["hit_bubble"] == 1
        assert res["gt_free"] == 1 and res["hit_free"] == 0
        assert res["recall"] == 0.5

    def test_calificar_nota_paginas_perdidas_e_historia(self, tmp_path, monkeypatch):
        """_evaluar_workspace: el modelo encuentra img1 y falla img2 →
        recall 0.5, nota 50, img2 listada como 'página con diálogo perdido',
        y la ronda se acumula en el historial."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)

        class _T:
            """simula un tensor: .tolist() devuelve la lista."""
            def __init__(self, v):
                self._v = v

            def tolist(self):
                return self._v

        class FakeBoxes:
            def __init__(self, xyxy):
                self.xyxy = [_T(b) for b in xyxy]

        class FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class FakeYOLO:
            def __init__(self, *a, **k):
                pass

            def predict(self, img, **k):
                if Path(img).stem == "img1":
                    # GT real de img1 en píxeles (labels normalizados sobre
                    # 100x100): (5,12.5,35,27.5) y (40,55,60,65) — el modelo
                    # las encuentra EXACTAS (IoU=1)
                    return [FakeResult(FakeBoxes([[5, 12.5, 35, 27.5],
                                                  [40, 55, 60, 65]]))]
                return [FakeResult(None)]   # img2: no detecta NADA

        import ultralytics
        import ocr_utils
        monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")

        ronda = mod._evaluar_workspace(
            tmp_path / "modelo.pt",
            sorted((ws / "images").glob("*.jpg")),
            ws / "labels", conf=0.25, imgsz=512, overlap_min=0.3)
        assert ronda["gt_total"] == 4
        assert ronda["hits"] == 2
        assert ronda["fns"] == 2
        assert ronda["recall"] == 0.5
        assert ronda["nota"] == 50
        assert ronda["paginas_perdidas_dialogo"] == [
            {"pagina": "img2", "gt": 2}]
        assert ronda["peores_paginas"][0]["pagina"] == "img2"

        hist = mod._append_historia(
            tmp_path / "calificaciones.json",
            {**ronda, "fecha": "2026-08-11 12:00", "nota": 50})
        assert len(hist) == 1
        hist2 = mod._append_historia(
            tmp_path / "calificaciones.json",
            {**ronda, "fecha": "2026-08-12 12:00", "nota": 80})
        assert len(hist2) == 2
        assert hist2[-1]["nota"] == 80   # la nota sube ronda a ronda

    def _FakeYOLO_imgs_detectadas(self, detecta: set):
        """FakeYOLO: detecta 2 cajas por página salvo las de `detecta`.
        Los labels del workspace generan GT: (5,12.5,35,27.5) y (40,55,60,65)
        sobre 100x100; las detecciones coinciden exactamente con esas GT."""
        class _T:
            def __init__(self, v):
                self._v = v

            def tolist(self):
                return self._v

        class FakeBoxes:
            def __init__(self, xyxy):
                self.xyxy = [_T(b) for b in xyxy]

        class FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class FakeYOLO:
            def __init__(self, *a, **k):
                pass

            def predict(self, img, **k):
                if Path(img).stem in detecta:
                    return [FakeResult(None)]
                return [FakeResult(FakeBoxes([[5, 12.5, 35, 27.5],
                                              [40, 55, 60, 65]]))]

        return FakeYOLO

    def test_generar_grid_crea_png_con_la_perdida(self, tmp_path,
                                                  monkeypatch):
        """El montaje visual se genera con las páginas que perdieron texto:
        img2 (0/2) entra al grid; img1 (perfecta) no. PNG creado y no vacío."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        import ultralytics
        import ocr_utils
        monkeypatch.setattr(ultralytics, "YOLO",
                            self._FakeYOLO_imgs_detectadas({"img2"}))
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")
        paginas = [
            {"pagina": "img1", "gt": 2, "hits": 2, "fns": 0,
             "detecciones": 2, "fps": 0, "recall": 1.0, "precision": 1.0},
            {"pagina": "img2", "gt": 2, "hits": 0, "fns": 2,
             "detecciones": 0, "fps": 0, "recall": 0.0, "precision": 1.0},
        ]
        out = tmp_path / "grid.png"
        res = mod._generar_grid(tmp_path / "modelo.pt", paginas,
                                ws / "labels", conf=0.25, imgsz=512,
                                device_arg="cpu", out_path=out,
                                cols=2, tile_w=120, max_paginas=16)
        assert res == out
        assert out.exists() and out.stat().st_size > 1000

    def test_generar_comparativa_crea_png(self, tmp_path, monkeypatch):
        """El preview [ORO | MODELO] se genera con la página perdida y es
        más ancho que alto (par lado a lado)."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        import ultralytics
        import ocr_utils
        monkeypatch.setattr(ultralytics, "YOLO",
                            self._FakeYOLO_imgs_detectadas({"img2"}))
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")
        paginas = [
            {"pagina": "img1", "gt": 2, "hits": 2, "fns": 0,
             "detecciones": 2, "fps": 0, "recall": 1.0, "precision": 1.0},
            {"pagina": "img2", "gt": 2, "hits": 0, "fns": 2,
             "detecciones": 0, "fps": 0, "recall": 0.0, "precision": 1.0},
        ]
        out = tmp_path / "comparativa.png"
        res = mod._generar_comparativa(tmp_path / "modelo.pt", paginas,
                                       ws / "labels", conf=0.25, imgsz=512,
                                       device_arg="cpu", out_path=out,
                                       tile_w=120, max_paginas=8)
        assert res == out
        assert out.exists() and out.stat().st_size > 1000
        import cv2
        im = cv2.imread(str(out))
        assert im.shape[1] > im.shape[0]   # par [ORO | MODELO] es ancho

    def test_comparativa_rellenos_translucidos_visibles(self, tmp_path,
                                                        monkeypatch):
        """Las cajas del oro se dibujan con relleno TRANSLÚCIDO: además del
        borde (verde puro) aparecen px de relleno verde claro — así las
        regiones grandes del oro no tapan el arte de la página."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        import ultralytics
        import ocr_utils
        import numpy as np
        monkeypatch.setattr(ultralytics, "YOLO",
                            self._FakeYOLO_imgs_detectadas({"img2"}))
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")
        paginas = [
            {"pagina": "img2", "gt": 2, "hits": 0, "fns": 2,
             "detecciones": 0, "fps": 0, "recall": 0.0, "precision": 1.0,
             "gt_bubble": 1, "hit_bubble": 0,
             "gt_free": 1, "hit_free": 0},
        ]
        out = tmp_path / "comparativa.png"
        mod._generar_comparativa(tmp_path / "modelo.pt", paginas,
                                 ws / "labels", conf=0.25, imgsz=512,
                                 device_arg="cpu", out_path=out,
                                 tile_w=120, max_paginas=8)
        import cv2
        im = cv2.imread(str(out))
        # borde verde puro (el rectángulo) — sigue ahí
        verde_puro = ((im[:, :, 1] == 255) & (im[:, :, 0] == 0)
                      & (im[:, :, 2] == 0)).sum()
        # relleno translúcido: verde claro (verde mezclado con el gris 200 de
        # la página) — presente y en mayor cantidad que el borde
        relleno = ((im[:, :, 1] > 180) & (im[:, :, 1] < 250)
                   & (im[:, :, 0] < 190) & (im[:, :, 0] > 100)).sum()
        assert verde_puro > 0, "el borde verde del globo debería dibujarse"
        assert relleno > verde_puro, \
            "el relleno translúcido debería cubrir más área que el borde"

    def test_escribir_html_preview_leyenda_honesta(self, tmp_path):
        """El HTML del preview usa SOLO los colores que el PNG dibuja (verde,
        magenta, rojo) — sin el amarillo fantasma 'Coincidencia oro∩modelo'
        que el antiguo preview prometía y nunca dibujaba."""
        mod = _load_tool("calificar")
        png = tmp_path / "comp.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        html = mod._escribir_html_preview(
            png, tmp_path / "preview.html",
            "Comparativa", "izq oro | der modelo")
        assert html.exists()
        txt = html.read_text(encoding="utf-8")
        assert "data:image/png;base64," in txt
        for sw in ("#2ecc71", "#e84393", "#e74c3c"):
            assert sw in txt
        assert "#f1c40f" not in txt          # sin el amarillo fantasma
        assert "Coincidencia" not in txt
        for leg in ("Globos del oro", "Texto libre del oro",
                    "Detecciones del modelo"):
            assert leg in txt

    def test_generar_grid_sin_perdidas_devuelve_none(self, tmp_path):
        """Si ninguna página perdió texto, no se genera el montaje."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        paginas = [
            {"pagina": "img1", "gt": 2, "hits": 2, "fns": 0,
             "detecciones": 2, "fps": 0, "recall": 1.0, "precision": 1.0},
        ]
        out = tmp_path / "grid.png"
        res = mod._generar_grid(tmp_path / "modelo.pt", paginas,
                                ws / "labels", conf=0.25, imgsz=512,
                                device_arg="cpu", out_path=out)
        assert res is None
        assert not out.exists()

    def test_main_no_grid_no_crea_montaje(self, tmp_path, monkeypatch,
                                          capsys):
        """main() con --no-grid no crea ningún PNG del montaje (aunque haya
        páginas con texto perdido) y sigue guardando la ronda."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        import ultralytics
        import ocr_utils
        monkeypatch.setattr(ultralytics, "YOLO",
                            self._FakeYOLO_imgs_detectadas({"img1", "img2"}))
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")
        hist = tmp_path / "h.json"
        _run(mod, ["--workspace", str(ws), "--historia", str(hist),
                   "--no-grid", "--imgsz", "512", "--device", "cpu"])
        assert not list(tmp_path.glob("*.png"))     # sin montaje
        assert hist.exists()                          # pero la ronda se guardó
        ronda = json.loads(hist.read_text(encoding="utf-8"))[0]
        assert "grid" not in ronda

    def test_main_paginas_filtra_y_muestra_tabla(self, tmp_path,
                                                  monkeypatch, capsys):
        """--paginas filtra la calificación a esas páginas y el reporte
        muestra el recall por página en la misma pasada (tabla peor primero)."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        import ultralytics
        import ocr_utils
        # img1 detectada (2/2), img2 NO detectada (0/2)
        monkeypatch.setattr(ultralytics, "YOLO",
                            self._FakeYOLO_imgs_detectadas({"img2"}))
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")
        hist = tmp_path / "h.json"
        _run(mod, ["--workspace", str(ws), "--historia", str(hist),
                   "--no-grid", "--imgsz", "512", "--device", "cpu",
                   "--paginas", "img1"])
        out = capsys.readouterr().out
        # solo img1 se evaluó
        assert "RECALL POR PÁGINA" in out
        assert "img1" in out
        assert "2/2" in out          # img1 recuperó sus 2 cajas
        assert "img2" not in out.split("CALIFICACIÓN")[-1]
        ronda = json.loads(hist.read_text(encoding="utf-8"))[0]
        assert ronda["paginas_evaluadas"] == 1
        assert ronda["gt_total"] == 2
        assert ronda["nota"] == 100

    def test_main_paginas_sin_marcado_avisa(self, tmp_path, monkeypatch,
                                             capsys):
        """--paginas con una página que no existe avisa y no guarda ronda."""
        mod = _load_tool("calificar")
        ws = self._workspace_con_marcado(tmp_path)
        import ultralytics
        import ocr_utils
        monkeypatch.setattr(ultralytics, "YOLO",
                            self._FakeYOLO_imgs_detectadas({"img1"}))
        monkeypatch.setattr(ocr_utils, "_resolver_device_yolo",
                            lambda: "cpu")
        hist = tmp_path / "h.json"
        _run(mod, ["--workspace", str(ws), "--historia", str(hist),
                   "--no-grid", "--imgsz", "512", "--device", "cpu",
                   "--paginas", "p999"])
        out = capsys.readouterr().out
        assert "Páginas pedidas sin marcado" in out
        assert not hist.exists()      # sin ronda guardada (nada que calificar)

    def test_calificar_etiqueta_y_sin_paginas_marcadas(self, tmp_path, capsys):
        """main() con --etiqueta guarda el tag en el historial; y si el
        workspace no tiene páginas con marcado, avisa sin crashear."""
        mod = _load_tool("calificar")
        ws = tmp_path / "corregir"
        (ws / "images").mkdir(parents=True)
        (ws / "labels").mkdir(parents=True)
        _run(mod, ["--workspace", str(ws),
                   "--historia", str(tmp_path / "h.json"),
                   "--etiqueta", "ronda de prueba"])
        out = capsys.readouterr().out
        assert "corrige primero" in out
