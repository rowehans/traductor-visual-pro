"""fusionar_correcciones.py — Fusiona las correcciones manuales del usuario.

Después de corregir en X-AnyLabeling (o cualquier tool YOLO) el workspace de
tools/exportar_anotaciones.py, este script:

  1. Lee las etiquetas CORREGIDAS del workspace (labels/*.txt).
  2. Reemplaza las pseudo-etiquetas del teacher solo en las páginas corregidas
     (el resto del dataset queda igual — las correcciones son el dato dorado).
  3. Respeta el split original de cada página (de manifest.json del teacher):
     una página val corregida sigue en val (A/B honesto a lo largo del tiempo);
     páginas NUEVAS que añadas van a train.
  4. Reporta el diff (etiquetas añadidas/eliminadas por página corregida).
  5. Con --train, lanza tools/entrenar_detector.py al terminar (flujo de un
     comando: corregir → merge → re-entrenar → A/B).

Uso:
  env/Scripts/python.exe tools/fusionar_correcciones.py            # solo merge
  env/Scripts/python.exe tools/fusionar_correcciones.py --train    # merge + entrenar
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASES = ["text_bubble", "text_free"]
CLASES_VALIDAS = set(range(len(CLASES)))

IMG_EXT = (".jpg", ".jpeg", ".png")


def _buscar_etiquetas(workspace: Path, nombre: str) -> Path | None:
    """Localiza el .txt de una imagen en los layout típicos de YOLO."""
    candidatos = [
        workspace / "labels" / f"{nombre}.txt",
        workspace / f"{nombre}.txt",
        workspace / "images" / f"{nombre}.txt",
    ]
    for c in candidatos:
        if c.is_file():
            return c
    return None


def _leer_labels(path: Path, nombre: str) -> list[list[float]]:
    """Parsea un .txt YOLO (cls cx cy w h normalizadas). Filtra clases inválidas
    y coordenadas fuera de rango; una línea mala se salta con aviso."""
    out: list[list[float]] = []
    if path is None or not path.exists():
        return out
    for i, linea in enumerate(path.read_text(encoding="utf-8").splitlines()):
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        partes = linea.split()
        if len(partes) != 5:
            print(f"[merge] {nombre}.txt línea {i + 1}: formato inválido, salto: "
                  f"{linea[:40]}")
            continue
        try:
            vals = [float(p) for p in partes]
        except ValueError:
            print(f"[merge] {nombre}.txt línea {i + 1}: no numérica, salto: "
                  f"{linea[:40]}")
            continue
        cls = int(vals[0])
        if cls not in CLASES_VALIDAS:
            print(f"[merge] {nombre}.txt línea {i + 1}: clase {cls} no existe "
                  f"({CLASES}), salto")
            continue
        if not (0.0 <= vals[1] <= 1.0 and 0.0 <= vals[2] <= 1.0
                and 0.0 <= vals[3] <= 1.0 and 0.0 <= vals[4] <= 1.0):
            print(f"[merge] {nombre}.txt línea {i + 1}: coords fuera de rango, "
                  "salto")
            continue
        out.append(vals)
    return out


def _cargar_manifest(data_dir: Path) -> dict[str, dict]:
    """map página (pNNN) → {split, n_etiquetas} desde manifest.json del teacher."""
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        datos = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    mapa = {}
    for row in datos.get("paginas", []):
        nombre = Path(row.get("imagen", "")).stem
        if nombre:
            mapa[nombre] = {
                "split": row.get("split", "train"),
                "n": int(row.get("pagina", 0)),
            }
    return mapa


def _fusionar(workspace: Path, data_dir: Path) -> dict:
    """Aplica las correcciones sobre train_data/vlm. Devuelve resumen."""
    img_dir = workspace / "images"
    if not img_dir.is_dir():
        sys.exit(f"[merge] No hay workspace en {workspace} — ejecuta primero "
                 f"tools/exportar_anotaciones.py")
    manifest = _cargar_manifest(data_dir)
    corregidas = 0
    nuevas = 0
    vaciadas = 0
    total_lab_corr = 0
    total_lab_prev = 0
    diffs: list[str] = []

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        nombre = img.stem
        lab_path = _buscar_etiquetas(workspace, nombre)
        if lab_path is None:
            # Sin archivo .txt: página NO corregida (X-AnyLabeling no la tocó
            # o el workspace no trae labels) → conservar las pseudo-etiquetas.
            continue
        labels = _leer_labels(lab_path, nombre)
        info = manifest.get(nombre)
        if info is None:
            split = "train"   # página nueva añadida por el usuario
            nuevas += 1
        else:
            split = info["split"]
            corregidas += 1
            total_lab_prev += int(info.get("n", 0))
            if not labels:
                vaciadas += 1
            diffs.append(f"  {nombre} [{split}]: "
                         f"{info.get('n', 0)} → {len(labels)} etiquetas")
        total_lab_corr += len(labels)

        split_dir = data_dir / split
        img_out = split_dir / "images" / img.name
        lab_out = split_dir / "labels" / f"{nombre}.txt"
        shutil.copy2(img, img_out)
        if labels:
            lab_out.write_text(
                "\n".join(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} "
                          f"{l[3]:.6f} {l[4]:.6f}" for l in labels) + "\n",
                encoding="utf-8")
        else:
            lab_out.write_text("", encoding="utf-8")  # página sin texto (negativo)

    resumen = {
        "corregidas": corregidas, "nuevas": nuevas, "vaciadas": vaciadas,
        "etiquetas_previas": total_lab_prev, "etiquetas_corregidas": total_lab_corr,
        "diffs": diffs,
    }
    return resumen


def _entrenar() -> None:
    """Lanza entrenar_detector.py en el mismo intérprete (flujo de 1 comando)."""
    script = ROOT / "tools" / "entrenar_detector.py"
    print("[merge] Re-entrenando el detector...")
    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[merge] El entrenamiento falló (exit {e.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fusiona correcciones manuales con pseudo-etiquetas VLM.")
    ap.add_argument("--corregidas", default=str(ROOT / "train_data" / "corregir"))
    ap.add_argument("--data", default=str(ROOT / "train_data" / "vlm"))
    ap.add_argument("--train", action="store_true",
                    help="Tras fusionar, lanza tools/entrenar_detector.py")
    args = ap.parse_args()

    workspace = Path(args.corregidas)
    data_dir = Path(args.data)
    if not (data_dir / "train" / "images").is_dir():
        sys.exit(f"[merge] No hay dataset en {data_dir} — ejecuta primero "
                 f"tools/etiquetar_con_vlm.py")

    r = _fusionar(workspace, data_dir)
    print(f"[merge] ✅ páginas corregidas: {r['corregidas']} "
          f"(nuevas: {r['nuevas']}, vaciadas: {r['vaciadas']})")
    print(f"[merge]   etiquetas: {r['etiquetas_previas']} → "
          f"{r['etiquetas_corregidas']} (solo páginas tocadas)")
    for d in r["diffs"]:
        print(d)
    if r["corregidas"] == 0 and r["nuevas"] == 0:
        print("[merge] Sin correcciones detectadas — revisa el workspace "
              f"{workspace} (¿guardaste en formato YOLO?)")
    else:
        print(f"[merge] Dataset actualizado en {data_dir} "
              "(páginas no corregidas conservan las pseudo-etiquetas)")
        if args.train:
            _entrenar()


if __name__ == "__main__":
    main()
