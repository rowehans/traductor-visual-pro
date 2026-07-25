# -*- coding: utf-8 -*-
"""
gestor.py — Traductor automático de manga español → inglés
con detección y corrección de páginas mal traducidas.
Reintenta páginas fallidas con mayor resolución automáticamente.
"""
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

RUTA_RAIZ = Path(__file__).parent
RUTA_PDF = RUTA_RAIZ / "Capítulo 43 de Cómo criar villanos correctamente _ Olympus Scanlation_compressed.pdf"
RUTA_SERVIDOR = RUTA_RAIZ / "server.py"
RUTA_PYTHON = RUTA_RAIZ / "env" / "Scripts" / "python.exe"
RUTA_REPORTE = RUTA_RAIZ / "reporte_final.html"
RUTA_PROGRESO = RUTA_RAIZ / "progreso.json"
RUTA_FALLIDAS = RUTA_RAIZ / "paginas_fallidas"

URL_SERVIDOR = "http://127.0.0.1:5174"
IDIOMA_ORIGEN = "es"
IDIOMA_DESTINO = "en"
MAX_PAGINAS = 0
REINTENTOS_POR_FALLO = 2
DPI_POR_DEFECTO = 150
DPI_REINTENTO = [200, 300, 450]
WORKERS_PARALELOS = 2  # paginas procesadas concurrentemente (servidor aguanta 4-5)

_lock_progreso = threading.Lock()
_lock_print = threading.Lock()
_documento = None  # fitz document compartido (PyMuPDF es thread-safe en get_pixmap)


def mostrar(texto):
    marca = datetime.now().strftime("%H:%M:%S")
    print(f"[{marca}] {texto}", flush=True)


def esperar_servidor(segundo_tope=60):
    import urllib.request
    inicio = time.time()
    while time.time() - inicio < segundo_tope:
        try:
            with urllib.request.urlopen(f"{URL_SERVIDOR}/api/health", timeout=3) as r:
                if r.status == 200:
                    mostrar("[OK] Servidor detectado.")
                    return True
        except Exception:
            pass
        time.sleep(2)
    mostrar("[X] No se pudo conectar al servidor.")
    return False


def arrancar_servidor():
    mostrar("[..] Arrancando servidor Flask...")
    entorno = os.environ.copy()
    entorno["SKIP_MIT_INIT"] = "1"
    entorno["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [str(RUTA_PYTHON), str(RUTA_SERVIDOR)],
        cwd=str(RUTA_RAIZ),
        env=entorno,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


PALABRAS_INGLES = {
    "the","and","for","are","but","not","you","all","can","had","her","was","one","our","out",
    "has","have","been","some","them","than","what","when","your","said","each","will","other",
    "which","their","would","about","there","could","should","after","every","because","between",
    "through","without","another","however","therefore","although","everything","something",
    "together","meanwhile","nevertheless","notwithstanding","straightforward",
    "chapter","season","episode","volume","page","thank","thanks","yes","no","hello","hey",
    "well","just","like","right","really","look","know","think","want","need","come","go","see",
    "get","make","take","give","find","keep","let","tell","ask","show","try","leave","call",
    "move","turn","put","set","start","stop","help","work","play","believe","feel","remember",
    "understand","wait","sorry","please","sorry","enough","already","still","even","though",
    "maybe","perhaps","always","never","ever","today","tomorrow","yesterday","now","then",
    "later","before","after","first","last","next","again","also","too","very","quite","much",
    "many","some","any","every","each","both","few","several","such","own","same","other",
    "another","different","important","possible","necessary","general","special","common",
    "simple","clear","sure","true","real","full","great","good","bad","new","old","long",
    "short","high","low","large","small","big","little","strong","weak","fast","slow","hard",
    "soft","easy","difficult","early","late","light","dark","deep","shallow","wide","narrow",
    "open","closed","free","busy","quiet","loud","happy","sad","angry","calm","warm","cool",
    "hot","cold","young","old","beautiful","ugly","rich","poor","clean","dirty","sharp","dull",
    "sweet","sour","bitter","fresh","stale","alive","dead","awake","asleep","sick","healthy",
    "near","far","left","right","inside","outside","above","below","front","back","top","bottom",
    "middle","center","side","edge","corner","end","beginning","middle","start","finish","rest",
    "peace","war","life","death","love","hate","joy","pain","hope","fear","dream","night","day",
    "sun","moon","star","sky","earth","land","sea","river","mountain","forest","tree","flower",
    "grass","stone","fire","water","air","wind","rain","snow","ice","cloud","smoke","dust","mud",
    "road","path","bridge","house","home","door","window","wall","floor","roof","room","table",
    "chair","bed","lamp","clock","bell","key","lock","chain","rope","stick","stone","knife",
    "sword","shield","armor","helmet","crown","ring","flag","sign","mark","seal","letter","word",
    "name","title","lord","king","queen","prince","princess","duke","count","earl","baron",
    "knight","soldier","warrior","guard","captain","general","master","servant","slave","friend",
    "enemy","ally","stranger","guest","host","owner","ruler","leader","teacher","student",
    "father","mother","brother","sister","son","daughter","husband","wife","uncle","aunt",
    "cousin","nephew","niece","grandfather","grandmother","child","baby","adult","elder",
    "youth","boy","girl","man","woman","person","people","family","clan","tribe","nation",
    "kingdom","empire","republic","state","city","town","village","street","market","shop",
    "church","temple","school","library","garden","park","field","farm","factory","mine",
    "castle","tower","gate","wall","dungeon","prison","cell","cage","arena","colosseum",
    "vista","glimpse","shadow","light","bright","glimmer","gleam","shimmer","sparkle",
    "glow","flash","beam","ray","wave","surge","flow","stream","current","tide","flood",
    "drought","storm","thunder","lightning","rainbow","fog","mist","haze","frost","dew",
    "harvest","feast","famine","plague","blessing","curse","omen","miracle","wonder",
    "monster","beast","creature","dragon","demon","spirit","ghost","phantom","angel","devil",
    "god","goddess","myth","legend","tale","story","history","fate","destiny","fortune",
    "luck","chance","risk","danger","safety","trust","faith","belief","doubt","suspicion",
    "proof","evidence","truth","lie","secret","mystery","riddle","puzzle","clue","answer",
    "question","problem","solution","reason","cause","effect","result","purpose","goal",
    "aim","target","mission","quest","journey","path","way","method","plan","scheme","plot",
    "trick","trap","ambush","attack","defense","battle","fight","war","conflict","struggle",
    "victory","defeat","triumph","loss","gain","profit","benefit","advantage","disadvantage",
    "price","cost","value","worth","wealth","treasure","gold","silver","bronze","coin","gem",
    "jewel","crystal","diamond","ruby","sapphire","emerald","pearl","amber","ivory","silk",
    "cotton","wool","leather","fur","feather","bone","horn","claw","fang","tail","wing","fin",
    "scale","shell","web","nest","den","lair","cave","tunnel","mine","shaft","pit","trench",
}


def analizar_bloque(origen, destino, confianza=1.0):
    """Revisa si un bloque traducido tiene problemas.
    confianza: valor 0-1 del OCR (si esta disponible).
    Devuelve lista de alertas (vacia si esta bien).
    Umbrales calibrados para ES->EN (ingles suele expandir longitud y usar contracciones).
    """
    alertas = []
    if not destino or len(destino.strip()) < 2:
        alertas.append("VACIO")
    if origen and destino and origen.strip() == destino.strip():
        # Solo marcar IDENTICO si ademas no parece ingles (si el original era ingles, esta bien)
        if not _es_ingles(destino):
            alertas.append("IDENTICO")
        elif confianza < 0.30 and origen and len(origen) > 3:
            alertas.append("CONFIANZA_BAJA")
        # Longitud anormal (umbrales relajados: ingles expansivo es legitimo)
    if origen and destino and len(origen) > 5:
        if len(destino) < len(origen) * 0.2:
            alertas.append("DEMASIADO_CORTO")
        if len(destino) > len(origen) * 8:
            alertas.append("DEMASIADO_LARGO")
        # Si casi la misma longitud pero mucho mas corto -> mal traducido
        if abs(len(destino) - len(origen)) < 3 and destino and origen and destino.strip() != origen.strip():
            if not _es_ingles(destino):
                alertas.append("POSIBLE_NO_TRADUCIDO")

    # Caracteres basura
    caracteres_raros = sum(1 for c in destino if c in "[]`~^<>{}|\\")
    if destino and caracteres_raros > len(destino) * 0.25:
        alertas.append("BASURA")

    # OCR basura: el texto original no parece lenguaje real
    if _es_ocr_basura(origen):
        alertas.append("OCR_BASURA")

    # Detectar bloques donde traduccione no es ingles (umbral mas alto y contracciones normalizadas)
    if destino and len(destino) > 3 and "[!]" not in destino and "[!]" not in destino:
        if not _es_ingles(destino):
            # Solo marcar NO_INGLES si hay suficientes palabras para evaluar
            palabras = _normalizar_palabras(destino)
            if len(palabras) >= 3:
                alertas.append("NO_INGLES")

    # Repeticion de caracteres (OCR alucinado)
    if destino and len(destino) > 4:
        chars_unicos = len(set(destino.lower()))
        if chars_unicos <= 3:
            alertas.append("TEXTO_REPETITIVO")

    return alertas


def _normalizar_palabras(texto):
    """Normaliza contracciones inglesas y signos para comparacion con diccionario."""
    t = texto.lower()
    # Contracciones comunes -> palabras base
    reemplazos = {
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am", "i'm": "i am", "don't": "do not",
        "can't": "can not", "won't": "will not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
        "didn't": "did not", "doesn't": "does not", "hasn't": "has not",
        "haven't": "have not", "hadn't": "had not", "that's": "that is",
        "there's": "there is", "it's": "it is", "he's": "he is", "she's": "she is",
        "what's": "what is", "let's": "let us",
    }
    for k, v in reemplazos.items():
        t = t.replace(k, v)
    return [p.strip(".,;:!?\"'()[]{}*") for p in t.split() if p.strip(".,;:!?\"'()[]{}*")]


def _es_ingles(texto):
    """Verifica si un texto parece ingles real (umbral ajustado + contracciones)."""
    if not texto or len(texto) < 4:
        return True
    palabras = _normalizar_palabras(texto)
    if not palabras:
        return True
    encontradas = sum(1 for p in palabras if p in PALABRAS_INGLES)
    # Umbral 25% (era 15%) + al menos 1 palabra conocida si hay 2-3
    if len(palabras) <= 3:
        return encontradas >= 1
    return encontradas >= len(palabras) * 0.25


def _es_ocr_basura(texto):
    """Detecta si el texto original es basura generada por OCR."""
    if not texto or len(texto.strip()) < 3:
        return False
    texto = texto.strip()
    letras = sum(1 for c in texto if c.isalpha())
    if letras / max(len(texto), 1) < 0.35:
        return True
    # Detectar secuencias anormales de mayúsculas/minúsculas
    if letras >= 4:
        mayus = sum(1 for c in texto if c.isupper())
        minus = sum(1 for c in texto if c.islower())
        if mayus > 0 and minus > 0:
            cambios = sum(1 for i in range(1, len(texto))
                         if texto[i].isalpha() and texto[i-1].isalpha()
                         and texto[i].isupper() != texto[i-1].isupper())
            if cambios > len(texto) * 0.35:
                return True
    return False


def al_menos_3(texto):
    return texto and len(texto.strip()) >= 3


def contar_letras(texto):
    return sum(1 for c in texto if c.isalpha())


def pagina_tiene_error(resultado):
    """Determina si una pagina necesita reintento. Solo errores criticos."""
    if resultado["salud"] in ("ERROR", "ERROR_HTTP"):
        return True
    if resultado["salud"] == "ALERTAS":
        for alerta in resultado["alertas"]:
            if alerta in ("OCR_BASURA", "BASURA", "IDENTICO", "VACIO", "TEXTO_REPETITIVO", "FALLO_TRADUCCION", "FALLÓ_TRADUCCIÓN"):
                return True
        if resultado["cantidad_bloques"] == 0:
            return True
    return False


def procesar_pagina(numero_pagina, documento, dpi=DPI_POR_DEFECTO, intento=1):
    """Envía una página al servidor. Si falla, reintenta con mayor dpi."""
    from PIL import Image
    import cv2
    import numpy as np
    import requests as peticiones

    pagina = documento[numero_pagina - 1]
    pix = pagina.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Si es reintento, aplicar mejora de contraste adicional
    if intento > 1:
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gris = clahe.apply(gris)
        img_bgr = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    _, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    codigo_b64 = base64.b64encode(buf.tobytes()).decode()

    insistir = 0
    while insistir <= REINTENTOS_POR_FALLO:
        try:
            t0 = time.time()
            respuesta = peticiones.post(
                f"{URL_SERVIDOR}/api/process-page",
                json={
                    "image": f"data:image/png;base64,{codigo_b64}",
                    "target": IDIOMA_DESTINO,
                    "source": IDIOMA_ORIGEN,
                },
                timeout=180,
            )
            datos = respuesta.json()
            bloques = datos.get("blocks", [])
            duracion = time.time() - t0

            resultado = {
                "pagina": numero_pagina,
                "tiempo_segundos": round(duracion, 1),
                "cantidad_bloques": len(bloques),
                "codigo_http": respuesta.status_code,
                "dpi_usado": dpi,
                "intento": intento,
                "bloques": [],
                "alertas": [],
                "salud": "OK",
            }

            if respuesta.status_code != 200:
                resultado["salud"] = "ERROR_HTTP"
                resultado["alertas"].append(f"HTTP_{respuesta.status_code}")
                return resultado

            for i, b in enumerate(bloques):
                origen = b.get("source", "")
                destino = b.get("translated", "")
                alertas_bloque = analizar_bloque(origen, destino)
                resultado["bloques"].append({
                    "indice": i,
                    "origen": origen,
                    "destino": destino,
                    "alertas": alertas_bloque,
                })
                resultado["alertas"].extend(alertas_bloque)

            if resultado["alertas"]:
                resultado["salud"] = "ALERTAS"
            elif not bloques:
                resultado["salud"] = "SIN_TEXTO"

            return resultado

        except Exception as error:
            insistir += 1
            if insistir > REINTENTOS_POR_FALLO:
                return {
                    "pagina": numero_pagina,
                    "tiempo_segundos": 0,
                    "cantidad_bloques": 0,
                    "codigo_http": 0,
                    "dpi_usado": dpi,
                    "intento": intento,
                    "bloques": [],
                    "alertas": [str(error)],
                    "salud": "ERROR",
                }
            mostrar(f"  -> Reintento {insistir}/{REINTENTOS_POR_FALLO} pag {numero_pagina}")


def guardar_progreso(resultados):
    completados = [r for r in resultados if r is not None]
    datos = {
        "ultima_actualizacion": datetime.now().isoformat(),
        "total_procesadas": len(completados),
        "paginas": [{
            "pagina": r["pagina"],
            "salud": r["salud"],
            "alertas": list(set(r["alertas"]))[:10],
            "tiempo": r["tiempo_segundos"],
            "bloques": r["cantidad_bloques"],
        } for r in completados],
    }
    with open(RUTA_PROGRESO, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def generar_reporte(resultados, nombre_pdf):
    completados = [r for r in resultados if r is not None]
    total_ok = sum(1 for r in completados if r["salud"] == "OK")
    total_alertas = sum(1 for r in completados if r["salud"] == "ALERTAS")
    total_sin_texto = sum(1 for r in completados if r["salud"] == "SIN_TEXTO")
    total_errores = sum(1 for r in completados if r["salud"] in ("ERROR", "ERROR_HTTP"))

    mostrar("=" * 70)
    mostrar("INFORME FINAL DE TRADUCCION")
    mostrar("=" * 70)
    mostrar(f"  Archivo: {nombre_pdf}")
    mostrar(f"  Paginas analizadas: {len(completados)}")
    mostrar(f"  [OK] Correctas:        {total_ok}")
    mostrar(f"  [!]  Con alertas:      {total_alertas}")
    mostrar(f"  [--] Sin texto:        {total_sin_texto}")
    mostrar(f"  [X]  Con errores:      {total_errores}")

    resumen_retry = [r for r in completados if r.get("intento", 1) > 1]
    if resumen_retry:
        mostrar(f"  [->] Reintentadas: {len(resumen_retry)} (mejoradas con mayor resolucion)")
        for r in resumen_retry:
            mostrar(f"     Pag {r['pagina']}: intento {r['intento']} @ {r.get('dpi_usado', 150)}dpi")

    paginas_problema = [r for r in completados if r["salud"] in ("ALERTAS", "ERROR", "ERROR_HTTP")]
    if paginas_problema:
        mostrar("\n[!] PAGINAS CON PROBLEMAS (ordenadas por gravedad):")
        mostrar("-" * 70)
        for r in sorted(paginas_problema, key=lambda x: -len(x["alertas"])):
            mostrar(f"  Pag {r['pagina']:3d} ({r['tiempo_segundos']}s) - {r['salud']} [intento {r.get('intento', 1)} @ {r.get('dpi_usado', 150)}dpi]")
            for b in r["bloques"]:
                if b["alertas"]:
                    mostrar(f"    #{b['indice']}: {', '.join(b['alertas'])}")
                    mostrar(f"      Original:  {b['origen'][:80]}")
                    mostrar(f"      Traducido: {b['destino'][:80]}")
            mostrar("")

    # Guardar HTML
    paginas_html = ""
    for r in completados:
        color = {"OK": "#28a745", "ALERTAS": "#ffc107", "SIN_TEXTO": "#6c757d", "ERROR": "#dc3545", "ERROR_HTTP": "#dc3545"}.get(r["salud"], "#000")
        alertas = ", ".join(set(r["alertas"]))[:80] or "—"
        reintento = f"#{r.get('intento', 1)} @ {r.get('dpi_usado', 150)}dpi" if r.get("intento", 1) > 1 else ""
        paginas_html += f"""
        <tr style="background:{color}15">
            <td><b>{r['pagina']}</b></td>
            <td>{r['tiempo_segundos']}s</td>
            <td>{r['cantidad_bloques']}</td>
            <td style="color:{color};font-weight:bold">{r['salud']}</td>
            <td>{alertas}</td>
            <td>{reintento}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"><title>Reporte gestor.py</title>
<style>
body {{ font-family:'Segoe UI',sans-serif; margin:20px; background:#1a1a2e; color:#eee; }}
h1 {{ color:#e94560; }}
table {{ border-collapse:collapse; width:100%; margin:10px 0; }}
th,td {{ border:1px solid #333; padding:8px; text-align:left; font-size:13px; }}
th {{ background:#16213e; color:#eee; }}
tr:nth-child(even) {{ background:#0f3460; }}
.resumen {{ display:flex; gap:20px; margin:20px 0; }}
.tarjeta {{ background:#16213e; padding:15px; border-radius:8px; text-align:center; flex:1; }}
.tarjeta h2 {{ margin:0; font-size:36px; }}
.verde {{ color:#28a745; }} .amarillo {{ color:#ffc107; }} .gris {{ color:#6c757d; }} .rojo {{ color:#dc3545; }}
</style></head><body>
<h1>📊 Reporte de Traducción</h1>
<p>{nombre_pdf} | {len(resultados)} páginas | {IDIOMA_ORIGEN.upper()} → {IDIOMA_DESTINO.upper()}</p>
<div class="resumen">
    <div class="tarjeta"><h2 class="verde">{total_ok}</h2><p>Correctas</p></div>
    <div class="tarjeta"><h2 class="amarillo">{total_alertas}</h2><p>Alertas</p></div>
    <div class="tarjeta"><h2 class="gris">{total_sin_texto}</h2><p>Sin texto</p></div>
    <div class="tarjeta"><h2 class="rojo">{total_errores}</h2><p>Errores</p></div>
</div>
<table><thead><tr><th>Pág</th><th>Tiempo</th><th>Bloques</th><th>Salud</th><th>Alertas</th><th>Reintento</th></tr></thead>
<tbody>{paginas_html}</tbody></table>
<p><i>Generado {datetime.now().strftime('%d/%m/%Y %H:%M')}</i></p></body></html>"""
    with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write(html)
    mostrar(f"[HTML] Reporte: {RUTA_REPORTE}")


def ejecutar():
    mostrar("=" * 70)
    mostrar("GESTOR DE TRADUCCION - Espanol -> Ingles (con reinteligente)")
    mostrar("=" * 70)

    if not RUTA_PDF.exists():
        mostrar(f"[X] No se encuentra: {RUTA_PDF}")
        return

    tama_mb = RUTA_PDF.stat().st_size // (1024 * 1024)
    mostrar(f"[PDF] {RUTA_PDF.name} ({tama_mb} MB)")

    import fitz
    global _documento
    _documento = fitz.open(str(RUTA_PDF))
    total_paginas = _documento.page_count
    a_procesar = min(total_paginas, MAX_PAGINAS) if MAX_PAGINAS else total_paginas
    mostrar(f"[PDF] {a_procesar}/{total_paginas} paginas | {IDIOMA_ORIGEN.upper()} -> {IDIOMA_DESTINO.upper()}")

    proc_servidor = arrancar_servidor()
    time.sleep(10)
    if not esperar_servidor(60):
        return

    resultados = [None] * a_procesar
    tiempo_inicio = time.time()
    completadas = 0

    def procesar_con_reintentos(numero_pagina):
        resultado = procesar_pagina(numero_pagina, _documento, dpi=DPI_POR_DEFECTO, intento=1)
        reintentado = False
        if pagina_tiene_error(resultado):
            for idx_re, dpi_alto in enumerate(DPI_REINTENTO):
                if resultado["salud"] in ("ERROR", "ERROR_HTTP") or \
                    (resultado["salud"] == "SIN_TEXTO" and resultado["cantidad_bloques"] == 0) or \
                    any(a in resultado.get("alertas", []) for a in ["OCR_BASURA", "BASURA", "FALLO_TRADUCCION", "FALLÓ_TRADUCCIÓN"]):
                    with _lock_print:
                        mostrar(f"  -> Reintento {idx_re+1}/3 con {dpi_alto}dpi...")
                    if idx_re >= 1:
                        time.sleep(5)
                    resultado = procesar_pagina(numero_pagina, _documento, dpi=dpi_alto, intento=idx_re+2)
                    reintentado = True
                    if not pagina_tiene_error(resultado):
                        break
        return (numero_pagina - 1, resultado, reintentado)

    with ThreadPoolExecutor(max_workers=WORKERS_PARALELOS) as executor:
        futuros = {executor.submit(procesar_con_reintentos, i+1): i for i in range(a_procesar)}
        for future in as_completed(futuros):
            idx, resultado, reintentado = future.result()
            resultados[idx] = resultado
            marca = {"OK": "[OK]", "ALERTAS": "[!]", "SIN_TEXTO": "[--]", "ERROR": "[X]", "ERROR_HTTP": "[X]"}.get(resultado["salud"], "?")
            reintento_str = f" (reintento {resultado.get('intento', 1)} @ {resultado.get('dpi_usado', 150)}dpi)" if reintentado else ""
            with _lock_print:
                mostrar(f"  {marca} {resultado['salud']} ({resultado['cantidad_bloques']} bloques, {resultado['tiempo_segundos']}s){reintento_str}")
            completadas += 1
            if completadas % 5 == 0:
                guardar_progreso(resultados)
                transcurrido = time.time() - tiempo_inicio
                ritmo = transcurrido / completadas
                restante = ritmo * (a_procesar - completadas)
                with _lock_print:
                    mostrar(f"  [T] {completadas}/{a_procesar} | {int(transcurrido//60)}m {int(transcurrido%60)}s | ~{int(restante//60)}m {int(restante%60)}s restantes")

    _documento.close()
    guardar_progreso(resultados)
    generar_reporte(resultados, RUTA_PDF.name)

    tiempo_total = time.time() - tiempo_inicio
    mostrar(f"\n[T] Total: {int(tiempo_total//60)}m {int(tiempo_total%60)}s")
    proc_servidor.terminate()
    mostrar("[OK] Servidor detenido.")

    mostrar("\n[!] PAGINAS QUE REQUIEREN REVISION MANUAL:")
    malas = sorted(
        [r for r in resultados if r and r["salud"] in ("ALERTAS", "ERROR", "ERROR_HTTP")],
        key=lambda x: -len(x.get("alertas", [])),
    )
    if malas:
        for r in malas:
            resumen = ', '.join(set(r.get('alertas', [])))[:80]
            mostrar(f"  [X] Pag {r['pagina']:3d} (intento {r.get('intento', 1)}/{len(DPI_REINTENTO)+1}): {resumen}")
    else:
        mostrar("  [OK] Todas las paginas traducidas correctamente.")


if __name__ == "__main__":
    ejecutar()
