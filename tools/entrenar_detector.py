"""entrenar_detector.py — Student de la destilación: fine-tune del YOLO de globos.

PLAN_MANGA_OCR Paso 7 (destilación): entrena el detector de regiones de texto
(student, barato: 0.02-0.4s/pág) con las pseudo-etiquetas generadas por el
daemon VLM (teacher, tools/etiquetar_con_vlm.py). El resultado reemplaza/aumenta
la Fase 6 con la cobertura que el VLM ve y ogkalu no (diálogo artístico, texto
sin globo).

Anti-olvido (el student no debe olvidar lo que ogkalu ya sabía):
  - arranca de los pesos ogkalu (no de cero),
  - congela el backbone (freeze=10),
  - LR inicial bajo (lr0=1e-4) y epochs modestos,
  - NUNCA sobrescribe models/comic-speech-bubble-detector.pt: escribe
    models/comic-speech-bubble-detector-finetuned.pt y solo con --swap lo
    activa (con backup .bak del original, reversible).

Uso:
  env/Scripts/python.exe tools/entrenar_detector.py            # dataset por defecto
  env/Scripts/python.exe tools/entrenar_detector.py --epochs 60 --batch 4
  env/Scripts/python.exe tools/entrenar_detector.py --swap     # activa el fine-tuned
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


def _build_data_yaml(data_dir: Path, out_yaml: Path) -> dict:
    """Construye data.yaml de ultralytics desde el layout train/val del teacher.

    Valida que existan imágenes y etiquetas en ambos splits; sin val → error
    claro (el teacher siempre genera val con --val-frac>0).
    """
    train_img = data_dir / "train" / "images"
    val_img = data_dir / "val" / "images"
    train_lab = data_dir / "train" / "labels"
    val_lab = data_dir / "val" / "labels"
    for d in (train_img, val_img, train_lab, val_lab):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Falta {d} — ejecuta primero tools/etiquetar_con_vlm.py")
    n_train = len(list(train_img.glob("*.jpg"))) + len(list(train_img.glob("*.png")))
    n_val = len(list(val_img.glob("*.jpg"))) + len(list(val_img.glob("*.png")))
    if n_train == 0 or n_val == 0:
        raise ValueError(f"Dataset vacío: train={n_train} val={n_val} "
                         f"(mínimo 1 imagen por split)")
    # data.yaml necesita rutas ABSOLUTAS (ultralytics no resuelve relativas
    # a la ubicación del yaml de forma confiable).
    yaml_text = (
        f"path: {data_dir.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\n"
        f"nc: {len(CLASES)}\nnames: {CLASES}\n"
    )
    out_yaml.write_text(yaml_text, encoding="utf-8")
    return {"n_train": n_train, "n_val": n_val, "yaml": str(out_yaml)}


def _preparar_dataset(data_dir: Path, extra_dir: Path | None, out_dir: Path) -> Path:
    """Arma el dataset de entrenamiento en un dir de trabajo limpio.

    train = imágenes del teacher (train) + opcional extra (sintéticas); val =
    siempre las del teacher (val real — A/B honesto). Copiar en vez de
    referenciar mantiene el dataset original intacto para el loop de corrección.
    """
    import shutil

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "train" / "images").mkdir(parents=True)
    (out_dir / "train" / "labels").mkdir(parents=True)
    (out_dir / "val" / "images").mkdir(parents=True)
    (out_dir / "val" / "labels").mkdir(parents=True)
    for split in ("train", "val"):
        for img in (data_dir / split / "images").iterdir():
            shutil.copy2(img, out_dir / split / "images" / img.name)
            lab = data_dir / split / "labels" / f"{img.stem}.txt"
            if lab.exists():
                shutil.copy2(lab, out_dir / split / "labels" / lab.name)
    if extra_dir is not None and extra_dir.is_dir():
        n_extra = 0
        for img in (extra_dir / "images").iterdir():
            shutil.copy2(img, out_dir / "train" / "images" / img.name)
            lab = extra_dir / "labels" / f"{img.stem}.txt"
            if lab.exists():
                shutil.copy2(lab, out_dir / "train" / "labels" / lab.name)
            n_extra += 1
        print(f"[train] +{n_extra} imágenes sintéticas/extra en train")
    return out_dir


def _entrenar(weights: Path, data_yaml: Path, epochs: int, batch: int,
              imgsz: int, device: str, freeze: int, lr0: float,
              project: Path, name: str) -> Path:
    """Fine-tune del YOLO con ultralytics. Retorna la ruta de best.pt."""
    from ultralytics import YOLO

    # ultralytics redirige project RELATIVO bajo SETTINGS['runs_dir']
    # (runs/detect/<project>/...) y entonces best.pt no aparece donde el
    # trainer lo busca. Resolver a ABSOLUTO elimina el quirk (lección:
    # sesión 153, el retrain synth_solo escribió en runs/detect/models/).
    project = Path(project).resolve()

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        freeze=freeze,          # congela el backbone (anti-olvido)
        lr0=lr0,
        lrf=0.05,
        workers=2,
        patience=15,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
        plots=False,
        val=True,
    )
    best = project / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"No se generó best.pt en {best}")
    return best


def _resolver_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _swap_model(original: Path, finetuned: Path) -> Path:
    """Activa el modelo fine-tuned como YOLO_MODEL_PATH con backup reversible.

    Respaldos el original en <nombre>.pt.bak (solo la primera vez) y copia el
    fine-tuned sobre la ruta que usa config.YOLO_MODEL_PATH. Retorna el backup
    (para restaurar). Nunca borra el original.
    """
    backup = original.with_suffix(".pt.bak")
    if not backup.exists():
        shutil.copy2(original, backup)
    shutil.copy2(finetuned, original)
    return backup


def _eval_rapida(model_a: Path, model_b: Path, images: list[Path],
                 imgsz: int = 640, conf: float = 0.25,
                 labels_dir: Path | None = None) -> dict:
    """A/B con el umbral REAL del pipeline (conf): detecciones por imagen,
    conf media, distribución de clases Y **recall por IoU contra las etiquetas
    GT del val** (si labels_dir se da).

    El recall a `conf` es la métrica decisiva (lección de la sesión 151): un
    fine-tune des-calibrado puede detectar MÁS cajas a conf baja (0.10) pero
    colapsar a conf>=0.25 (el umbral que usa el pipeline real en ocr_utils) —
    el A/B lo detecta porque mide cobertura con el MISMO umbral de producción.
    """
    from ultralytics import YOLO

    import ocr_utils

    a, b = YOLO(str(model_a)), YOLO(str(model_b))
    dev = _resolver_device()
    acc = {"a": {"n": 0, "conf": 0.0, "bubble": 0, "free": 0,
                  "hit": 0, "gt": 0},
           "b": {"n": 0, "conf": 0.0, "bubble": 0, "free": 0,
                  "hit": 0, "gt": 0}}
    for img in images:
        gt = _leer_gt(img, labels_dir)
        for key, m in (("a", a), ("b", b)):
            r = m.predict(str(img), conf=conf, imgsz=imgsz, device=dev,
                          verbose=False)[0]
            boxes = []
            if r.boxes is not None:
                acc[key]["n"] += len(r.boxes)
                acc[key]["conf"] += float(r.boxes.conf.sum())
                for box, c in zip(r.boxes.xyxy, r.boxes.cls.tolist()):
                    x0, y0, x1, y1 = box.tolist()
                    boxes.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
                    name = str((r.names or {}).get(int(c), "")).lower()
                    if "free" in name:
                        acc[key]["free"] += 1
                    else:
                        acc[key]["bubble"] += 1
            # recall: GT cubierta por al menos una detección (IoU>0.3). Se
            # cuenta SIEMPRE — si el modelo no detecta nada en la página,
            # boxes queda vacío y la página aporta 0 hits sobre su GT (un
            # modelo que falla una página debe pagar su recall, no saltársela).
            hit = sum(any(
                ocr_utils._overlap_ratio(g, b) > 0.3 for b in boxes)
                for g in gt)
            acc[key]["hit"] += hit
            acc[key]["gt"] += len(gt)
    res = {}
    for key in ("a", "b"):
        n = acc[key]["n"]
        gt = acc[key]["gt"]
        res[key] = {
            "detecciones": n,
            "conf_media": round(acc[key]["conf"] / n, 3) if n else 0.0,
            "bubble": acc[key]["bubble"],
            "free": acc[key]["free"],
            "recall": round(acc[key]["hit"] / gt, 4) if gt else None,
            "gt": gt,
        }
    return res


def _leer_gt(img: Path, labels_dir: Path | None) -> list[dict]:
    """Etiquetas GT YOLO del val en píxeles: {labels_dir}/{img.stem}.txt con
    formato 'cls cx cy w h' normalizado → cajas {x,y,w,h} absolutas."""
    if labels_dir is None:
        return []
    import cv2

    lab = labels_dir / f"{img.stem}.txt"
    if not lab.exists():
        return []
    h_pag, w_pag = cv2.imread(str(img)).shape[:2]
    gt: list[dict] = []
    for line in lab.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in p[:5])
        x0 = (cx - bw / 2) * w_pag
        y0 = (cy - bh / 2) * h_pag
        gt.append({"x": x0, "y": y0, "w": bw * w_pag, "h": bh * h_pag})
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune del YOLO de globos con pseudo-etiquetas del VLM.")
    ap.add_argument("--data", default=str(ROOT / "train_data" / "vlm"))
    ap.add_argument("--extra-data", default=str(ROOT / "train_data" / "synth"),
                    help="Dir extra de entrenamiento (p. ej. sintéticas): sus "
                         "imágenes se añaden a TRAIN; el val siempre es real. "
                         "Pasa '' para desactivar.")
    ap.add_argument("--weights", default=str(ROOT / "models" /
                                             "comic-speech-bubble-detector.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4,
                    help="4GB de VRAM: batch 4 con AMP; bajar a 2 si OOM")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--device", default=None,
                    help="GPU '0' o 'cpu' (default: auto — CUDA si disponible)")
    ap.add_argument("--name", default="finetune_vlm")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--conf", type=float, default=None,
                    help="Umbral de confianza del A/B post-entrenamiento. "
                         "Default = YOLO_CONF_THRESH de config.py (0.25, el "
                         "que usa el pipeline real). El --swap SOLO se ejecuta "
                         "si el fine-tuned gana el recall por IoU a ESTE umbral "
                         "(un modelo des-calibrado que detecta más a conf baja "
                         "pero colapsa a 0.25 nunca se activa — sesión 151).")
    ap.add_argument("--swap", action="store_true",
                    help="Activa el modelo fine-tuned como YOLO_MODEL_PATH "
                         "(backup .bak del original, reversible). Solo surte "
                         "efecto si el A/B a --conf lo declara ganador.")
    args = ap.parse_args()

    data_dir = Path(args.data)
    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[train] No existe el modelo base: {weights}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_dir = Path(args.extra_data) if args.extra_data else None
    if extra_dir is not None and extra_dir.is_dir():
        data_dir = _preparar_dataset(data_dir, extra_dir,
                                     data_dir.parent / "vlm_aug")
    data_yaml = data_dir / "data.yaml"
    info = _build_data_yaml(data_dir, data_yaml)
    print(f"[train] dataset: {info['n_train']} train + {info['n_val']} val "
          f"(yaml={info['yaml']})")

    device = args.device or _resolver_device()
    print(f"[train] device={device} (GTX 1050 Ti 4GB: imgsz={args.imgsz}, "
          f"batch={args.batch}, freeze={args.freeze}, lr0={args.lr0})")
    best = _entrenar(weights, data_yaml, args.epochs, args.batch, args.imgsz,
                     device, args.freeze, args.lr0, out_dir, args.name)
    finetuned = out_dir / "comic-speech-bubble-detector-finetuned.pt"
    shutil.copy2(best, finetuned)
    print(f"[train] ✅ modelo fine-tuned: {finetuned}")

    # A/B sobre las imágenes de validación con el umbral REAL del pipeline.
    # conf por defecto = YOLO_CONF_THRESH (config.py), el que usa ocr_utils en
    # producción — así el veredicto refleja cómo se comportaría en la fusión.
    if args.conf is None:
        from config import YOLO_CONF_THRESH
        args.conf = YOLO_CONF_THRESH
    val_imgs = sorted((data_dir / "val" / "images").glob("*.jp*"))[:8]
    ab = None
    if val_imgs:
        ab = _eval_rapida(weights, finetuned, val_imgs, args.imgsz,
                          conf=args.conf, labels_dir=data_dir / "val" / "labels")
        print(f"[train] A/B (ogkalu vs finetuned) sobre {len(val_imgs)} págs "
              f"val a conf>={args.conf}:")
        for key, label in (("a", "ogkalu   "), ("b", "finetuned")):
            r = ab[key]
            rec = f"{r['recall']:.1%}" if r["recall"] is not None else "n/a"
            print(f"  {label}: {r['detecciones']} detecciones | conf "
                  f"{r['conf_media']} | bubble {r['bubble']} free {r['free']} "
                  f"| recall IoU {rec} ({r['gt']} GT)")
        gana = (ab["b"]["recall"] is not None and ab["a"]["recall"] is not None
                and ab["b"]["recall"] > ab["a"]["recall"])
    else:
        gana = False
        print("[train] A/B: sin imágenes val — no se puede verificar el "
              "modelo; no se activa nada.")

    if args.swap and gana:
        backup = _swap_model(Path(weights), finetuned)
        print(f"[train] ✅ swap: el fine-tuned GANÓ el A/B a conf>={args.conf} "
              f"(recall {ab['b']['recall']:.1%} vs {ab['a']['recall']:.1%}) — "
              f"YOLO_MODEL_PATH ahora usa el fine-tuned (original respaldado "
              f"en {backup}; restaurar copiando el .bak)")
    elif args.swap and not gana:
        r_a = ab["a"]["recall"] if ab else None
        r_b = ab["b"]["recall"] if ab else None
        f_a = f"{r_a:.1%}" if r_a is not None else "n/a"
        f_b = f"{r_b:.1%}" if r_b is not None else "n/a"
        print(f"[train] ⛔ --swap IGNORADO: el fine-tuned NO supera a ogkalu a "
              f"conf>={args.conf} (recall {f_b} vs {f_a}) — des-calibrado o "
              f"sin ganancia. No se activa un modelo peor que el actual.")
    else:
        if ab is not None:
            r_a = ab["a"]["recall"]
            r_b = ab["b"]["recall"]
            f_a = f"{r_a:.1%}" if r_a is not None else "n/a"
            f_b = f"{r_b:.1%}" if r_b is not None else "n/a"
            ganador = "fine-tuned" if gana else "ogkalu"
        else:
            f_a = f_b = "n/a"
            ganador = "n/a (sin val)"
        print(f"[train] No activado: el A/B a conf>={args.conf} es el criterio "
              f"(recall {f_b} vs {f_a} — ganador: {ganador}). Para activar, "
              f"corre con --swap (solo surte efecto si gana el A/B).")


if __name__ == "__main__":
    main()
