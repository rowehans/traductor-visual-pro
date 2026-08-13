"""importar_paginas.py — Importa páginas de manga NUEVAS al corrector de oro.

Copia un lote de páginas reales desde input_manga/BookDownloads al workspace
del corrector (train_data/corregir/images) para que el humano las corrija a
mano. Detalles:

  - Convierte webp → jpg (el corrector y el pipeline del dataset solo
    aceptan jpg/png).
  - Salta las primeras y últimas 2 páginas de cada capítulo (portadas /
    contraportadas — no llevan diálogo útil).
  - NO vuelve a importar las series que ya forman parte del oro corregido
    (evita darle al humano páginas que ya marcó).
  - Salta los capítulos que YA tienen páginas importadas en el destino:
    cada ejecución avanza a capítulos nuevos en vez de re-picar los mismos
    (variedad de capítulos > repetir los primeros).
  - Nombres únicos y estables: s{serie}_{capitulo}_{pagina}.jpg — si el
    destino ya tiene ese archivo, se salta (idempotente: re-ejecutar no
    duplica).
  - Reparte el lote entre muchas series distintas (variedad > profundidad
    para un detector de texto).

Uso:
  env/Scripts/python.exe tools/importar_paginas.py                 # ~300 páginas
  env/Scripts/python.exe tools/importar_paginas.py --max 150 --por-serie 6
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGEN = ROOT / "input_manga" / "BookDownloads" / "BookDownloads"
DESTINO = ROOT / "train_data" / "corregir" / "images"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
# Series cuyas páginas ya corrigió el humano (no re-importar).
EXCLUIDAS = {"1103524", "1457338", "1490498"}


def _natural(key: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", key)]


def _serie_unica(dirs: list[Path]) -> list[Path]:
    """Series únicas: descarta las copias 'Nombre (1)' y las excluidas."""
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        if re.search(r"\(\d+\)$", d.name):
            continue
        if d.name in EXCLUIDAS:
            continue
        out.append(d)
    return sorted(out, key=lambda p: p.name)


def _caps_ya_usados(serie: Path, destino: Path) -> set[str]:
    """Capítulos de la serie que ya tienen páginas en el destino O en
    terminadas/ (los lotes ya corregidos también cuentan: un capítulo
    representado en cualquier lote no se re-pica)."""
    prefix = f"s{_sanitizar(serie.name)}_"
    usados: set[str] = set()
    carpetas = [destino]
    term = destino.parent / "terminadas" / "images"
    if term.is_dir():
        carpetas.append(term)
    for carpeta in carpetas:
        for f in carpeta.glob(f"{prefix}*.jpg"):
            resto = f.name[len(prefix):]
            idx = resto.rfind("_p")    # parte por el último _p (la página)
            if idx > 0:
                usados.add(resto[:idx])
    return usados


def _paginas_de_serie(serie: Path, por_serie: int,
                      destino: Path) -> list[tuple[str, Path]]:
    """Devuelve hasta `por_serie` páginas de una serie, repartidas entre sus
    capítulos NUEVOS (sin páginas ya importadas) y saltando
    portadas/contraportadas de cada uno."""
    caps_usados = _caps_ya_usados(serie, destino)
    cap: list[tuple[str, Path]] = []
    for ch in sorted(serie.iterdir(), key=lambda p: _natural(p.name)):
        if not ch.is_dir():
            continue
        if _sanitizar(ch.name) in caps_usados:
            continue                    # capítulo ya representado en el oro
        paginas = sorted(
            (p for p in ch.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXT),
            key=lambda p: _natural(p.stem))
        if len(paginas) <= 4:            # poco margen para saltar portadas
            cuerpo = paginas
        else:
            cuerpo = paginas[2:-2]
        for p in cuerpo:
            cap.append((ch.name, p))
        if len(cap) >= por_serie * 2:    # ya hay de sobra en esta serie
            break
    if not cap:
        return []
    # reparto uniforme sobre el total de la serie
    n = min(por_serie, len(cap))
    idx = [round(i * (len(cap) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    sel = {cap[i] for i in idx}
    return sorted(sel, key=lambda x: x[1].name)


def _sanitizar(nombre: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", nombre).strip("_") or "x"


def importar(origen: Path, destino: Path, max_paginas: int,
             por_serie: int, calidad: int) -> dict:
    destino.mkdir(parents=True, exist_ok=True)
    series = _serie_unica(list(origen.iterdir()))
    importados = 0
    ya_existian = 0
    rotos = 0
    por_serie_res: dict[str, int] = {}
    for serie in series:
        if importados >= max_paginas:
            break
        for cap_nombre, pagina in _paginas_de_serie(serie, por_serie, destino):
            if importados >= max_paginas:
                break
            nombre = f"s{_sanitizar(serie.name)}_" \
                     f"{_sanitizar(cap_nombre)}_p{_sanitizar(pagina.stem)}.jpg"
            salida = destino / nombre
            if salida.exists():
                ya_existian += 1
                continue
            img = cv2.imread(str(pagina))
            if img is None:
                rotos += 1
                print(f"  ! no se pudo leer: {pagina}")
                continue
            if not cv2.imwrite(str(salida), img,
                               [cv2.IMWRITE_JPEG_QUALITY, calidad]):
                rotos += 1
                print(f"  ! no se pudo escribir: {salida}")
                continue
            importados += 1
            por_serie_res[serie.name] = por_serie_res.get(serie.name, 0) + 1
    return {"series_disponibles": len(series), "importados": importados,
            "ya_existian": ya_existian, "rotos": rotos,
            "por_serie": por_serie_res}


def main() -> None:
    ap = argparse.ArgumentParser(description="Importa páginas nuevas al corrector de oro")
    ap.add_argument("--origen", default=str(ORIGEN))
    ap.add_argument("--destino", default=str(DESTINO))
    ap.add_argument("--max", type=int, default=300, help="máx páginas a importar")
    ap.add_argument("--por-serie", type=int, default=8, help="máx páginas por serie")
    ap.add_argument("--calidad", type=int, default=90)
    args = ap.parse_args()
    res = importar(Path(args.origen), Path(args.destino),
                   args.max, args.por_serie, args.calidad)
    print(f"[importar] series disponibles: {res['series_disponibles']}")
    print(f"[importar] importadas: {res['importados']} (ya existían: "
          f"{res['ya_existian']}, rotas: {res['rotos']})")
    if res["por_serie"]:
        print(f"[importar] reparto: {len(res['por_serie'])} series tocadas, "
              f"{min(res['por_serie'].values())}-"
              f"{max(res['por_serie'].values())} páginas cada una")
    print(f"[importar] total en {args.destino}: "
          f"{len(list(Path(args.destino).glob('*.jpg')))} páginas")


if __name__ == "__main__":
    main()
