#!/usr/bin/env python3
"""manga_ocr.py — Extracción de texto de manga (sin traducción ni inpainting).

    input_manga/  →  output_texto/<archivo>.json  +  <archivo>.txt

Escaneo recursivo (sesión 148): además de PDFs e imágenes sueltas, cada
carpeta que contiene imágenes directamente (p.ej. input_manga/…/serie/cap/
0.webp …) es UN documento con N páginas en orden natural (0,1,2,…,10,11).
--solo '<subcadena>' filtra por nombre de documento para procesar un
capítulo concreto sin recorrer todo; --pages sigue aplicando al documento.

Paso 3 de PLAN_MANGA_OCR.md. Reutiliza la maquinaria del proyecto:
  - OCRManager (ocr_engine.py) en modo fusion: EasyOCR GPU + RapidOCR +
    fusión multi-motor + YOLO de globos (Fase 6).
  - El refuerzo VLM (daemon U-OCR, puerto 5177) está DESACTIVADO por defecto
    (--vlm para activarlo si el daemon corre): la extracción pura no debe
    disparar inferencias de 2-8 min/pág.
  - batch=1 estricto: una página a la vez (VRAM de la GTX 1050 Ti, 4 GB).
  - doc_id = md5 del nombre del archivo [:12] (mismo esquema que
    process_all_pages) → los caches de decisión quedan escopeados por
    documento (sesión 126, sin interferencia cross-PDF).

El tier comic-text-detector (Paso 2, PLAN_MANGA_OCR) se integra en el
pipeline en el Paso 4 (OCRManager → Ruta C); hasta entonces las regiones CTD
no participan en run_ocr.

Uso:
    env/Scripts/python.exe manga_ocr.py                          # input_manga/ → output_texto/
    env/Scripts/python.exe manga_ocr.py --input cap/ --output out/ --zoom 2.0
    env/Scripts/python.exe manga_ocr.py --pages 1-5 --vlm        # solo págs 1-5, con daemon VLM
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from config import YOLO_MODEL_PATH
from ocr_engine import OCRManager

_EXT_IMAGENES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
_STEM_EN_USO: set[str] = set()  # detecta colisiones de nombre (cap.pdf vs cap.png)


def _doc_id(nombre_archivo: str) -> str:
    """Identificador de documento: scope de los caches de decisión del
    trigger/negativas (sesión 126). Mismo esquema que process_all_pages:
    md5 del nombre del archivo, 12 hex. (El md5 es identificador, no hash de
    seguridad — bandit lo marca como falso positivo.)"""
    return hashlib.md5(nombre_archivo.encode("utf-8", "ignore")).hexdigest()[:12]  # nosec B324


def _orden_natural(nombre: str) -> list[Any]:
    """Clave de orden natural: '2.webp' < '10.webp' (numérico, no léxico).
    Divide el nombre en trozos de dígitos y no-dígitos y compara cada trozo
    como int cuando es numérico."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", nombre)]


@dataclass
class _Documento:
    """Un manga procesable: PDF o una carpeta de imágenes (páginas).

    - PDF: path apunta al .pdf; las páginas se renderizan con fitz.
    - Carpeta: path apunta a la carpeta; las páginas son sus imágenes
      (recursivas) ordenadas naturalmente (0.webp, 1.webp, …, 10.webp).
    - Imagen suelta: path apunta al archivo; es un documento de 1 página.
    """
    path: Path
    nombre: str          # nombre de salida (stem del archivo o carpeta)
    tipo: str            # "pdf" | "carpeta" | "imagen"
    paginas: list[Path] = field(default_factory=list)


def _escaneo_documentos(input_dir: Path) -> list[_Documento]:
    """Documentos de manga en input_dir, escaneando recursivamente.

    - Archivos (PDF/imagen) sueltos en el nivel superior → un documento cada
      uno (compatibilidad con el comportamiento original).
    - Carpetas que CONTIENEN imágenes directamente (p.ej. cada carpeta de
      capítulo en una estructura anidada tipo
      input_manga/…/serie/capitulo/0.webp) → un documento por carpeta: sus
      imágenes, ordenadas naturalmente, son las páginas.
    - Las carpetas que solo contienen subcarpetas (el contenedor de la serie)
      no generan documento: se recorren para encontrar los capítulos.
    - Colisiones de nombre entre capítulos de distintas series: si dos
      carpetas comparten stem, el nombre de salida se prefija con la serie
      ('serie_capitulo') para no sobrescribirse.
    """
    docs: list[_Documento] = []
    usados: dict[str, Path] = {}

    def _add(doc: _Documento) -> None:
        prev = usados.get(doc.nombre)
        if prev is not None and prev != doc.path:
            # Colisión: prefijar con el nombre de la carpeta padre (la serie)
            doc.nombre = f"{doc.path.parent.name}_{doc.nombre}"
        usados[doc.nombre] = doc.path
        docs.append(doc)

    for p in sorted(input_dir.iterdir(), key=lambda x: _orden_natural(x.name)):
        if p.is_file():
            if p.suffix.lower() == ".pdf" or p.suffix.lower() in _EXT_IMAGENES:
                tipo = "pdf" if p.suffix.lower() == ".pdf" else "imagen"
                _add(_Documento(p, p.stem, tipo, [p]))
        elif p.is_dir():
            _agregar_carpetas_con_imagenes(p, _add)
    return docs


def _agregar_carpetas_con_imagenes(carpeta: Path,
                                   add) -> None:
    """Recorre `carpeta` recursivamente y añade como documento cada
    subcarpeta que contenga imágenes directamente (patrón walk: no mezcla
    niveles — un capítulo con páginas en subcarpetas más profundas se
    agrupa en su propio documento)."""
    # 1) Esta carpeta, si tiene imágenes directas → documento
    paginas = _imagenes_directas(carpeta)
    if paginas:
        add(_Documento(carpeta, carpeta.stem, "carpeta", paginas))
    # 2) Recursión en subcarpetas (un contenedor con imágenes Y subcarpetas
    #    aporta ambas: las suyas + las de sus hijos)
    for sub in sorted(carpeta.iterdir(), key=lambda x: _orden_natural(x.name)):
        if sub.is_dir():
            _agregar_carpetas_con_imagenes(sub, add)


def _imagenes_directas(carpeta: Path) -> list[Path]:
    """Imágenes soportadas directamente en `carpeta` (sin recursión),
    orden natural."""
    return sorted(
        (p for p in carpeta.iterdir() if p.is_file()
         and p.suffix.lower() in _EXT_IMAGENES),
        key=lambda p: _orden_natural(p.name))


def _render_paginas_pdf(path: Path, zoom: float,
                        rango: tuple[int, int] | None) -> Iterator[tuple[int, np.ndarray]]:
    """Renderiza cada página del PDF a BGR con fitz (patrón de
    process_all_pages.py: get_pixmap directo, sin PIL/numpy intermedios).
    Solo rinde el rango pedido (--pages) para no trabajar de balde."""
    import fitz
    doc = fitz.open(str(path))
    try:
        for i in range(doc.page_count):
            n = i + 1
            if rango and not (rango[0] <= n <= rango[1]):
                continue
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n >= 3:
                img = img[:, :, :3]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            yield n, img[:, :, ::-1].copy()  # RGB → BGR
    finally:
        doc.close()


def _cargar_imagen(path: Path) -> np.ndarray:
    """Imagen suelta (una sola "página")."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"no se pudo leer la imagen: {path}")
    return img


def _parse_rango(spec: str) -> tuple[int, int] | None:
    """'3-5' → (3, 5); '7' → (7, 7); vacío → None (todas las páginas)."""
    if not spec:
        return None
    try:
        a, _, b = spec.partition("-")
        ini = int(a)
        fin = int(b) if b else ini
        return max(ini, 1), max(fin, ini)
    except ValueError:
        raise SystemExit(f"--pages inválido: '{spec}' (esperado 'A-B', 1-indexado)")


def _bloque_a_schema(b: dict[str, Any]) -> dict[str, Any]:
    """Bloque interno (x/y/w/h/text/confidence[/source]) → entrada del schema
    de PLAN_MANGA_OCR.md: {texto, bbox [x0,y0,x1,y1], conf, motores, detector}."""
    x = int(b.get("x", 0))
    y = int(b.get("y", 0))
    w = int(b.get("w", 0))
    h = int(b.get("h", 0))
    src = str(b.get("source", "") or "").lower()
    if src == "yolo":
        motores, detector = ["easyocr", "rapid", "yolo"], "yolo"
    elif src in ("ctd", "ctd_line", "ctd_mask"):
        motores, detector = ["easyocr", "rapid", "comic_text_detector"], "ctd"
    elif src in ("unlimited", "uocr", "vlm"):
        motores, detector = ["easyocr", "rapid", "vlm"], "vlm"
    else:
        motores, detector = ["easyocr", "rapid"], "hibrido"
    return {
        "texto": str(b.get("text", "")),
        "bbox": [x, y, x + w, y + h],
        "conf": round(float(b.get("confidence", 0.0)), 3),
        "motores": motores,
        "detector": detector,
    }


def _texto_plano(bloques: list[dict[str, Any]]) -> str:
    """Texto plano de la página, ordenado de arriba a abajo y luego de
    izquierda a derecha (lectura natural del manga)."""
    orden = sorted(bloques, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return "\n".join(b["texto"] for b in orden if b["texto"].strip())


def _detectores_disponibles() -> list[str]:
    """Tiers de detección que participan en el pipeline (metadata del JSON).
    El tier comic-text-detector se añade en el Paso 4 (integración en
    OCRManager); hasta entonces las regiones CTD no se invocan."""
    detectores = ["easyocr", "rapid"]
    if os.path.exists(YOLO_MODEL_PATH):
        detectores.append("yolo_globos")
    return detectores


def _escribir_salida(output_dir: Path, stem: str, meta: dict[str, Any],
                     paginas: list[dict[str, Any]]) -> None:
    """Escribe <stem>.json + <stem>.txt. Se llama tras CADA página (el JSON
    es incremental): un crash no pierde las páginas ya procesadas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = {**meta, "paginas": paginas}
    with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    lineas: list[str] = []
    for p in paginas:
        lineas.append(f"=== Página {p['n']} ===")
        tp = p.get("texto_plano", "")
        lineas.append(tp if tp else "(sin texto detectado)")
    with open(output_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extracción de texto de manga: input_manga/ → output_texto/ "
                    "(JSON + TXT por archivo, una página a la vez)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default="input_manga",
                        help="carpeta de entrada (PDF o imágenes de manga)")
    parser.add_argument("--output", default="output_texto",
                        help="carpeta de salida (un .json + .txt por archivo)")
    parser.add_argument("--zoom", type=float, default=2.0,
                        help="escala de render de PDFs (1.0 ≈ 72-96 dpi)")
    parser.add_argument("--ocr-mode", default="fusion",
                        choices=["fusion", "easyocr", "auto"],
                        help="modo OCR de OCRManager")
    parser.add_argument("--lang", default="es",
                        help="idioma del texto de origen para el OCR")
    parser.add_argument("--vlm", action="store_true",
                        help="permitir el refuerzo VLM (requiere daemon U-OCR en 5177)")
    parser.add_argument("--pages", default="",
                        help="rango de páginas por documento, ej. '3-5' "
                             "(1-indexado); vacío = todas")
    parser.add_argument("--solo", default="",
                        help="procesar solo documentos cuyo nombre contenga "
                             "esta subcadena (p.ej. un ID de capítulo)")
    parser.add_argument("--force", action="store_true",
                        help="reprocesar aunque el .json de salida ya exista")
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        print(f"[manga_ocr] ERROR: la carpeta de entrada no existe: {input_dir}")
        return 1
    documentos = _escaneo_documentos(input_dir)
    if args.solo:
        documentos = [d for d in documentos if args.solo in d.nombre]
    if not documentos:
        print(f"[manga_ocr] No hay PDF/imágenes en {input_dir}"
              + (f" (ni con --solo '{args.solo}')" if args.solo else ""))
        return 0

    import config
    # Extracción pura: sin VLM salvo --vlm explícito. El gate UOCR_ENABLED
    # (sesión 143) anula SOLO el refuerzo — YOLO/Ruta C/cls siguen activos.
    if not args.vlm:
        config.UOCR_ENABLED = False
        print("[manga_ocr] Refuerzo VLM desactivado (--vlm para activarlo)",
              flush=True)

    rango = _parse_rango(args.pages)
    manager = OCRManager()
    total = len(documentos)
    for idx, doc in enumerate(documentos, 1):
        stem = doc.nombre
        if stem in _STEM_EN_USO:
            print(f"[manga_ocr] AVISO: '{stem}' ya se usó (colisión de nombre "
                  f"{doc.path.name}); la salida se sobrescribirá")
        _STEM_EN_USO.add(stem)
        json_path = output_dir / f"{stem}.json"
        if json_path.exists() and not args.force:
            print(f"[manga_ocr] {stem}: ya existe {json_path.name} "
                  f"(--force para reprocesar)")
            continue
        print(f"[manga_ocr] [{idx}/{total}] {stem}"
              f" ({len(doc.paginas)} páginas)", flush=True)
        meta = {
            "archivo": doc.path.name,
            "tipo": doc.tipo,
            "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ocr_mode": args.ocr_mode,
            "detectores": _detectores_disponibles(),
            "zoom": args.zoom if doc.tipo == "pdf" else None,
        }
        paginas: list[dict[str, Any]] = []
        t_archivo = time.time()
        n_procesadas = 0
        try:
            if doc.tipo == "pdf":
                iterador = _render_paginas_pdf(doc.path, args.zoom, rango)
            else:
                # Carpeta/imagen: cada archivo es una página, en orden natural.
                # --pages filtra por índice dentro del documento.
                lista = doc.paginas
                if rango:
                    lista = [p for i, p in enumerate(lista, 1)
                             if rango[0] <= i <= rango[1]]
                iterador = ((i, _cargar_imagen(p))
                            for i, p in enumerate(lista, 1))
            for n, img_bgr in iterador:
                t0 = time.time()
                blocks, engine_used, engines = manager.run_ocr(
                    img_bgr, args.lang, args.ocr_mode,
                    doc_id=_doc_id(doc.nombre))
                bloques = [_bloque_a_schema(b) for b in blocks
                           if str(b.get("text", "")).strip()]
                bloques.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
                t_pag = round(time.time() - t0, 2)
                paginas.append({
                    "n": n,
                    "bloques": bloques,
                    "texto_plano": _texto_plano(bloques),
                    "t_s": t_pag,
                    "engines": engines,
                    "n_bloques": len(bloques),
                })
                n_procesadas += 1
                print(f"  p{n}: {len(bloques)} bloques "
                      f"({engine_used}, {t_pag:.1f}s)", flush=True)
                _escribir_salida(output_dir, stem, meta, paginas)
        except Exception as e:
            print(f"[manga_ocr] ERROR en {stem}: {e}")
            return 1
        print(f"[manga_ocr] {stem}: {n_procesadas} páginas en "
              f"{time.time() - t_archivo:.1f}s → "
              f"{output_dir / (stem + '.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
