"""exportar_anotaciones.py — Workspace de corrección (X-AnyLabeling / YOLO).

Toma el dataset de pseudo-etiquetas generado por tools/etiquetar_con_vlm.py
(train_data/vlm/{train,val}) y lo exporta a UN workspace plano listo para
CORREGIR a mano sin instalar nada del proyecto:

  train_data/corregir/
    images/   pNNN.jpg                 (todas las páginas, train+val)
    labels/   pNNN.txt                 (cls cx cy w h normalizadas — YOLO)
    classes.txt                        (text_bubble, text_free)
    data.yaml                          (para el re-entrenamiento post-merge)
    LEEME.txt                          (instrucciones de corrección)

Formato universal YOLO: lo abren X-AnyLabeling (Ctrl+U sobre la carpeta con
formato YOLO), Label Studio, CVAT, Roboflow y el propio ultralytics. El flujo
completo de "darle clases":

  1. etiquetar_con_vlm.py     (teacher → pseudo-etiquetas)         [auto]
  2. exportar_anotaciones.py  (este script — workspace listo)      [auto]
  3. corregir en X-AnyLabeling (añadir globos que falten, borrar malos,
     corregir clase bubble/free, ajustar cajas)                    [manual]
  4. fusionar_correcciones.py --train  (merge + re-entrenar)       [auto]

Uso:
  env/Scripts/python.exe tools/exportar_anotaciones.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASES = ["text_bubble", "text_free"]

LEEME = """\
CORRECCIÓN DE ETIQUETAS — flujo de "darle clases" al detector
================================================================

Estas son las pseudo-etiquetas que generó el daemon VLM (teacher) sobre
páginas reales del capítulo. Tu trabajo: CORREGIR (no etiquetar de cero).

Herramienta recomendada: X-AnyLabeling (gratis, Windows).
  1. Descarga el .exe desde https://github.com/CVHub520/X-AnyLabeling/releases
  2. Ábrelo. Configura el FORMATO DE ETIQUETAS en YOLO (antes de abrir las
     imágenes) — usa el archivo de clases classes.txt de esta carpeta.
  3. Archivo > Directorio de imágenes (Ctrl+U) → selecciona esta carpeta.
  4. Corrige página por página (D = siguiente, A = anterior):
     - cajas que sobren o estén mal → selecciónalas y pulsa Supr
     - globos/ textos que falten → dibuja un rectángulo (R) y elige clase
     - clase equivocada (bubble/free) → cambia la etiqueta en el panel
     - ajusta los bordes de la caja arrastrando los tiradores
     - marca la página como revisada (Ctrl+Alt+K) para no perderte
  5. Guarda (auto-save está activo por defecto). El merge lee los .txt de
     labels/ (o los exporta: Tools > Export Annotations > YOLO).

Clases: text_bubble=0 (texto dentro de globo), text_free=1 (texto libre:
cartelas, títulos, texto flotante, onomatopeyas).

Al terminar:
  env/Scripts/python.exe tools/fusionar_correcciones.py --train

Esto fusiona tus correcciones con las pseudo-etiquetas, re-entrena el YOLO
y te muestra el A/B contra el modelo original.
"""


def _exportar(data_dir: Path, out: Path) -> dict:
    """Copia imágenes+etiquetas de {train,val} a un workspace plano."""
    img_out = out / "images"
    lab_out = out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    n_img = 0
    for split in ("train", "val"):
        src_img = data_dir / split / "images"
        src_lab = data_dir / split / "labels"
        if not src_img.is_dir():
            continue
        for img in sorted(src_img.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            shutil.copy2(img, img_out / img.name)
            lab = src_lab / f"{img.stem}.txt"
            if lab.exists():
                shutil.copy2(lab, lab_out / lab.name)
            n_img += 1

    (out / "classes.txt").write_text(
        "\n".join(CLASES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images\nval: images\n"
        f"nc: {len(CLASES)}\nnames: {CLASES}\n",
        encoding="utf-8")
    (out / "LEEME.txt").write_text(LEEME, encoding="utf-8")
    return {"imagenes": n_img, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exporta pseudo-etiquetas VLM a workspace YOLO editable.")
    ap.add_argument("--data", default=str(ROOT / "train_data" / "vlm"))
    ap.add_argument("--out", default=str(ROOT / "train_data" / "corregir"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    out = Path(args.out)
    if not (data_dir / "train" / "images").is_dir():
        sys.exit(f"[export] No hay dataset en {data_dir} — ejecuta primero "
                 f"tools/etiquetar_con_vlm.py")
    info = _exportar(data_dir, out)
    print(f"[export] ✅ {info['imagenes']} imágenes → {info['out']}")
    print(f"[export]   clases: {CLASES}")
    print(f"[export]   siguiente paso: corregir en X-AnyLabeling "
          f"(abre {info['out']}) y luego:")
    print(f"[export]   env/Scripts/python.exe tools/fusionar_correcciones.py --train")


if __name__ == "__main__":
    main()
