"""corrector_oro.py — Corrige las anotaciones ORO con una app en el navegador.

El oro de este proyecto son los .txt YOLO de train_data/corregir/labels/:
tools/calificar_detector.py los usa para evaluar al modelo y
tools/fusionar_correcciones.py los aplica al dataset. El teacher VLM los
generó con cajas gigantes (w=1.00) que no sirven; aquí las corriges tú.

Este script arranca un mini-servidor local (solo stdlib) que sirve la app
tools/corrector_oro.html sobre las páginas del workspace. Cada "Guardar" de
la app escribe el .txt de esa página (vacío si no quedan cajas) y lo marca
en revisadas.json — sin tocar nada más del pipeline.

Uso:
  env/Scripts/python.exe tools/corrector_oro.py              # puerto libre
  env/Scripts/python.exe tools/corrector_oro.py --port 8789
  env/Scripts/python.exe tools/corrector_oro.py --no-browser
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASES = ["text_bubble", "text_free"]
CLASES_VALIDAS = set(range(len(CLASES)))
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
HTML = ROOT / "tools" / "corrector_oro.html"
REVISADAS = "revisadas.json"


# ─── lógica pura (testeable) ───────────────────────────────────────

def _orden_natural(nombre: str):
    """Clave de orden natural: p002 < p010 (no léxico)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nombre)]


def _leer_yolo(path: Path) -> list[list[float]]:
    """Parsea un .txt YOLO (cls cx cy w h normalizadas). Líneas malas se saltan."""
    out: list[list[float]] = []
    if not path.exists():
        return out
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        partes = linea.split()
        if len(partes) != 5:
            continue
        try:
            vals = [float(p) for p in partes]
        except ValueError:
            continue
        cls = int(vals[0])
        if cls not in CLASES_VALIDAS:
            continue
        cx, cy, w, h = vals[1], vals[2], vals[3], vals[4]
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            continue
        if w < 1e-4 or h < 1e-4:
            continue
        out.append([cls, cx, cy, w, h])
    return out


def _escribir_yolo(path: Path, cajas: list[list[float]]) -> None:
    """Escribe cajas [cls, cx, cy, w, h] normalizadas en formato YOLO."""
    path.write_text(
        "".join(f"{int(c[0])} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f} {c[4]:.6f}\n"
                for c in cajas),
        encoding="utf-8",
    )


def _box_gigante(caja: list[float]) -> bool:
    """Caja que cubre ~un cuarto o más de la página (p. ej. w=1.00 del teacher)."""
    _, _, _, w, h = caja
    return w > 0.5 or h > 0.5 or w * h >= 0.25


def _normalizar(cajas: list[list[float]]) -> list[list[float]]:
    """Recorta a [0,1], descarta degeneradas y redondea a 6 decimales."""
    out: list[list[float]] = []
    for c in cajas:
        cls = int(round(c[0]))
        if cls not in CLASES_VALIDAS:
            continue
        cx = max(0.0, min(1.0, c[1]))
        cy = max(0.0, min(1.0, c[2]))
        w = max(0.0, min(1.0, c[3]))
        h = max(0.0, min(1.0, c[4]))
        if w < 0.002 or h < 0.002:      # más pequeño que ~4 px en 1920
            continue
        out.append([cls, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)])
    return out


def _backup_originales(ws: Path) -> None:
    """Copia UNA sola vez el oro original a labels/_original/ (red de seguridad)."""
    labels = ws / "labels"
    if not labels.is_dir():
        return
    dest = labels / "_original"
    if dest.exists():
        return
    dest.mkdir(parents=True)
    for txt in labels.glob("*.txt"):
        shutil.copy2(txt, dest / txt.name)


def _cargar_revisadas(ws: Path) -> dict[str, bool]:
    p = ws / REVISADAS
    if not p.exists():
        return {}
    try:
        datos = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(datos, dict):
        return {}
    return {k: bool(v) for k, v in datos.items()}


def _guardar_pagina(ws: Path, nombre: str, cajas: list[list[float]],
                    revisada: bool) -> dict:
    """Escribe el oro de una página en labels/{nombre}.txt.

    Una página sin cajas escribe un .txt VACÍO (no borra el archivo): así
    fusionar_correcciones.py la trata como "sin texto" y NO conserva las
    pseudo-etiquetas malas del teacher.
    """
    _backup_originales(ws)
    norm = _normalizar(cajas)
    labels = ws / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    _escribir_yolo(labels / f"{nombre}.txt", norm)
    revisadas = _cargar_revisadas(ws)
    revisadas[nombre] = bool(revisada)
    (ws / REVISADAS).write_text(
        json.dumps(revisadas, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "nombre": nombre, "cajas": len(norm),
            "gigantes": sum(1 for c in norm if _box_gigante(c))}


def _listar_paginas(ws: Path) -> list[dict]:
    images = ws / "images"
    revisadas = _cargar_revisadas(ws)
    paginas: list[dict] = []
    for img in sorted(images.iterdir(), key=lambda p: _orden_natural(p.stem)):
        if img.suffix.lower() not in IMG_EXT:
            continue
        cajas = _leer_yolo(ws / "labels" / f"{img.stem}.txt")
        paginas.append({
            "nombre": img.stem,
            "archivo": img.name,
            "img": f"/api/img/{img.name}",
            "cajas": cajas,
            "revisada": bool(revisadas.get(img.stem, False)),
            "gigantes": [i for i, c in enumerate(cajas) if _box_gigante(c)],
        })
    return paginas


# ─── servidor HTTP ─────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    ws: Path

    # -- rutas -----------------------------------------------------
    def do_GET(self) -> None:
        ruta = self.path.split("?", 1)[0]
        if ruta in ("/", "/index.html"):
            return self._enviar_archivo(HTML, "text/html; charset=utf-8")
        if ruta == "/api/paginas":
            return self._json({"paginas": _listar_paginas(self.ws)})
        if ruta == "/api/estado":
            return self._json(self._estado())
        if ruta.startswith("/api/img/"):
            nombre = Path(ruta[len("/api/img/"):]).name
            img = self.ws / "images" / nombre
            if img.is_file() and img.suffix.lower() in IMG_EXT:
                ctype = mimetypes.guess_type(img.name)[0] or "application/octet-stream"
                return self._enviar_archivo(img, ctype)
            return self._error(404, "imagen no encontrada")
        return self._error(404, "ruta no existe")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/guardar":
            return self._error(404, "ruta no existe")
        largo = int(self.headers.get("Content-Length", 0) or 0)
        try:
            datos = json.loads(self.rfile.read(largo).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(400, "JSON inválido")
        nombre = str(datos.get("nombre", ""))
        if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", nombre):
            return self._error(400, "página desconocida")
        if not any((self.ws / "images" / f"{nombre}{ext}").is_file()
                   for ext in IMG_EXT):
            return self._error(400, "página desconocida")
        cajas = datos.get("cajas", [])
        if not isinstance(cajas, list):
            return self._error(400, "cajas inválidas")
        try:
            cajas = [[float(v) for v in c] for c in cajas]
        except (TypeError, ValueError):
            return self._error(400, "cajas inválidas")
        revisada = bool(datos.get("revisada", False))
        return self._json(_guardar_pagina(self.ws, nombre, cajas, revisada))

    # -- helpers ---------------------------------------------------
    def _estado(self) -> dict:
        paginas = _listar_paginas(self.ws)
        return {
            "total": len(paginas),
            "revisadas": sum(1 for p in paginas if p["revisada"]),
            "cajas": sum(len(p["cajas"]) for p in paginas),
            "paginas_con_gigantes": [p["nombre"] for p in paginas if p["gigantes"]],
        }

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _enviar_archivo(self, path: Path, ctype: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            return self._error(404, "no existe")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str) -> None:
        self._json({"ok": False, "error": msg}, code)

    def log_message(self, fmt: str, *args) -> None:
        if args and args[0].startswith("POST"):
            return
        super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Corrector interactivo del oro (YOLO). Guardas = el oro de verdad.")
    ap.add_argument("--workspace", default=str(ROOT / "train_data" / "corregir"))
    ap.add_argument("--port", type=int, default=0, help="0 = puerto libre")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    ws = Path(args.workspace)
    if not (ws / "images").is_dir():
        sys.exit(f"[oro] No hay workspace en {ws} (falta images/)")
    _backup_originales(ws)

    Handler = type("Handler", (_Handler,), {"ws": ws})
    servidor = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{servidor.server_address[1]}/"
    print(f"[oro] LISTO — abre {url}")
    print(f"[oro] workspace: {ws}  (guardas = labels/*.txt, el ORO de verdad)")
    print("[oro] Ctrl+C para parar")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[oro] adiós")


if __name__ == "__main__":
    main()
