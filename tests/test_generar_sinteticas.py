"""
test_generar_sinteticas.py — Datos sintéticos y dataset aumentado.

Prueba tools/generar_sinteticas.py (páginas de manga con etiquetas YOLO
exactas) y el --extra-data de tools/entrenar_detector.py (train real +
sintético, val siempre real). Carga los módulos en fresco por test
(importlib, mismo patrón que el resto de tests de tools).
"""
import importlib.util
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_TOOLS = {
    "generar": _ROOT / "tools" / "generar_sinteticas.py",
    "entrenar": _ROOT / "tools" / "entrenar_detector.py",
}

_load_counter = [0]


def _load_tool(nombre: str):
    _load_counter[0] += 1
    spec = importlib.util.spec_from_file_location(
        f"{nombre}_synth_{_load_counter[0]}", _TOOLS[nombre])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def dataset_real(tmp_path):
    """Dataset del teacher mínimo: 2 train + 1 val."""
    vlm = tmp_path / "vlm"
    for split, paginas in (("train", ["p001", "p002"]), ("val", ["p003"])):
        (vlm / split / "images").mkdir(parents=True)
        (vlm / split / "labels").mkdir(parents=True)
        for p in paginas:
            (vlm / split / "images" / f"{p}.jpg").write_bytes(b"IMG")
            (vlm / split / "labels" / f"{p}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    return vlm


class TestGenerarSinteticas:

    def test_genera_paginas_con_etiquetas_validas(self, tmp_path):
        mod = _load_tool("generar")
        out = tmp_path / "synth"
        old = sys.argv
        sys.argv = ["generar_sinteticas.py", "--n", "5", "--out", str(out),
                    "--seed", "7"]
        try:
            mod.main()
        finally:
            sys.argv = old
        imgs = sorted(p.name for p in (out / "images").glob("*.jpg"))
        assert len(imgs) == 5
        todas: list[tuple] = []
        for i in range(5):
            lab = (out / "labels" / f"syn_{i:04d}.txt").read_text().splitlines()
            assert lab, f"syn_{i:04d} sin etiquetas"
            for linea in lab:
                p = linea.split()
                assert len(p) == 5
                cls, cx, cy, w, h = map(float, p)
                assert cls in (0, 1)
                assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0
                todas.append((cls, w, h))
        # con 5 páginas, ambas clases aparecen (títulos + onomatopeyas free)
        clases = {c for (c, _, _) in todas}
        assert clases == {0, 1}

    def test_pagina_generada_determinista_por_seed(self, tmp_path):
        mod = _load_tool("generar")
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        fuentes = mod._cargar_fuentes()
        img1, lab1 = mod._generar_pagina(rng1, fuentes)
        img2, lab2 = mod._generar_pagina(rng2, fuentes)
        assert len(lab1) == len(lab2)
        assert (img1 == img2).all()


class TestExtraData:

    def test_extra_data_aumenta_train_conserva_val(self, tmp_path,
                                                   dataset_real):
        mod = _load_tool("entrenar")
        extra = tmp_path / "synth"
        (extra / "images").mkdir(parents=True)
        (extra / "labels").mkdir(parents=True)
        for i in range(3):
            (extra / "images" / f"syn_{i}.jpg").write_bytes(b"S")
            (extra / "labels" / f"syn_{i}.txt").write_text("1 0.5 0.5 0.2 0.1\n")

        out = mod._preparar_dataset(dataset_real, extra, tmp_path / "aug")
        n_train = len(list((out / "train" / "images").glob("*")))
        n_val = len(list((out / "val" / "images").glob("*")))
        assert n_train == 2 + 3   # reales + sintéticas
        assert n_val == 1         # val siempre real

    def test_extra_data_vacio_no_rompe(self, tmp_path, dataset_real):
        mod = _load_tool("entrenar")
        out = mod._preparar_dataset(dataset_real, tmp_path / "nada",
                                    tmp_path / "aug")
        assert len(list((out / "train" / "images").glob("*"))) == 2
        assert len(list((out / "val" / "images").glob("*"))) == 1
