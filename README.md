# Traductor Visual Pro

<p align="center">
  <a href="https://github.com/rowehans/traductor-visual-pro/releases/latest">
    <img src="https://img.shields.io/badge/versi%C3%B3n%20%E2%86%92%20Releases-brightgreen?logo=github" alt="Latest release">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/releases/tag/v0.1.44">
    <img src="https://img.shields.io/badge/descargar%20.exe%20(360MB)-blue?logo=windows" alt="Download">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/releases">
    <img src="https://img.shields.io/badge/descargas%20%E2%86%92%20Releases-success?logo=download" alt="Downloads">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro/actions/workflows/ci.yml">
    <img src="https://img.shields.io/badge/CI%20%E2%86%92%20Actions-blueviolet?logo=github" alt="CI status">
  </a>
  <a href="https://github.com/rowehans/traductor-visual-pro">
    <img src="https://img.shields.io/badge/repo%20privado-lightgrey?logo=github" alt="Private repo">
  </a>
</p>

App local para traducir manga, cómics y documentos en PDF e imagen. Backend Python con EasyOCR + OpenCV + ArgosTranslate, frontend JavaScript con canvas interactivo y editor de burbujas.

---

## Descargar

**Opción recomendada — Instalador profesional:**

[![Download .exe](https://img.shields.io/badge/Descargar_main.exe-360MB-blue?style=for-the-badge&logo=windows)](https://github.com/rowehans/traductor-visual-pro/releases/download/v0.1.44/main.exe)

El ejecutable incluye:
- Ejecutable `main.exe` (360 MB) con EasyOCR GPU + RapidOCR (ONNX)
- Pipeline híbrido EasyOCR+RapidOCR: 128 páginas en 11.7 min
- Clasificación OCR_GARBAGE mejorada con 8 filtros (16.8% de detección)
- Script `setup.ps1` para configuración automatizada del entorno
- Descarga bajo demanda de modelos OCR y CT2 durante la instalación

**Requisitos:** Windows 10/11 (64-bit), Python 3.8+, 8 GB RAM, GPU NVIDIA con CUDA (opcional, acelera EasyOCR ~5.7x)

---

## Iniciar la app

**Opción 1 — Script PowerShell:**
```powershell
.\start-app.ps1
```

**Opción 2 — Directo:**
```powershell
.\env\Scripts\python.exe server.py
```

La app abre en: http://127.0.0.1:5174/

## Arquitectura

```
server.py         → Entry point Flask (puerto 5174)
config.py         → Constantes, patrones de ruido, CSP
translator.py     → Detección de idioma + traducción (Argos → Google → CT2)
ocr_utils.py      → OCR EasyOCR + inpainting OpenCV + filtros
routes/api.py     → Endpoints REST
routes/main.py    → Rutas estáticas
app.js            → Frontend canvas + UI (~2514 líneas)
index.html        → Estructura HTML
styles.css        → Estilos premium dark/light
cache.py          → Caché de traducciones filesystem
models.py         → Modelos SQLAlchemy
ratelimit.py      → Rate limiting
```

## Funciones principales

- **Carga PDF, PNG, JPG, WebP** — renderizado vía pdf.js (ESM v4.10 con fallback UMD v3.11)
- **OCR automático** — EasyOCR en servidor, GPU→CPU fallback
- **Traducción** — ArgosTranslate (offline) → Google Translate (online) → CT2 OPUS-MT es→en
- **Detección de idioma** — langdetect + heurística española (acentos, verbos, diccionario)
- **Filtros de ruido** — 9 filtros post-OCR + 2 pre-OCR (morfología OpenCV)
- **Inpainting** — OpenCV INPAINT_NS con máscara adaptativa (glifos vs rectángulo)
- **Editor de burbujas** — dibujar, mover, redimensionar, estilizar texto
- **Exportación** — PNG individual, PDF página actual, PDF completo
- **Temas** — oscuro/claro con toggle, persistencia en localStorage
- **Atajos de teclado** — D/V modos, Ctrl+T traducir, Ctrl+E/P exportar, flechas navegar

## Requisitos

- Python 3.10+
- Entorno virtual en `env/` con todas las dependencias
- OpenCV.js carga desde CDN jsDelivr (con callback nativo `onRuntimeInitialized`)
- Conexión a internet para primera descarga de modelos EasyOCR y ArgosTranslate

## Modo CPU (sin GPU dedicada)

El pipeline corre sin GPU: EasyOCR, YOLO, RapidOCR y CT2 degradan solos. Para
máquinas sin CUDA (o con GPU muy débil) existe el preset `MODO_CPU` en
`config.py`:

```python
MODO_CPU = True            # desactiva el VLM, fuerza YOLO a CPU
MODO_CPU_OCR_SCALE = 0.8   # escala de render sugerida al frontend
```

Con el preset activo:
- **VLM (U-OCR) apagado** aunque `UOCR_ENABLED=True` — es el único componente
  que exige GPU; sin él no hay inferencias de ~2-5 min por página.
- **YOLO forzado a CPU** (`_resolver_device_yolo` ignora `YOLO_DEVICE="auto"`).
- **Escala de render menor** servida al frontend vía `/api/config`
  (`ocr_scale: 0.8` ≈ 0.31 MP/pág vs 0.72 MP a 1.2 — ~2.3× menos píxeles).

Se puede activar desde el launcher/CLI mutando `config` en runtime (mismo
patrón que `UOCR_ENABLED`). Sin GPU, el cuello es el detector CRAFT de
EasyOCR: una página normal pasa de ~1.5 s en GPU a decenas de segundos, y las
páginas de arte oscuro pierden la recuperación del VLM (solo el daemon lee ese
texto). El resto del pipeline (RapidOCR CPU, CTD 100 % CPU) no cambia.

## Notas técnicas

- **OpenCV.js**: No más polling. Usa `cv['onRuntimeInitialized']` con 3 casos de carga y timeout 15s.
- **Caché**: Traducciones repetidas se cachean en `cache/translations/` por 7 días (5000 entradas máx).
- **Rate limiting**: 200 req/día, 50 req/hora global. Endpoints sensibles: 30/min (translate), 20/min (batch), 5/min (process-page).
- **CSP**: Política estricta inyectada en todas las respuestas. Permite CDNs específicos.
- **Seguridad**: Protección contra path traversal en rutas estáticas.

## Tests

Los tests unitarios (290+) se ejecutan automáticamente en cada commit vía pre-commit hook.

```powershell
# Tests rápidos (recomendado)
.\env\Scripts\python.exe run_ci.py

# Tests completos con reporte HTML
.\env\Scripts\python.exe run_ci.py --full --report

# Solo tests unitarios (pytest)
.\env\Scripts\python.exe -m pytest tests/ -q --tb=short
```

## CI

- **Pre-commit hook**: Syntax check + bandit (seguridad) + pytest en archivos modificados
- **GitHub Actions**: Syntax check + mypy + tests en push/PR con cambios en server.py, routes/ o core
- **CI local completa**: `python run_ci.py` (~30s) o `python run_ci.py --full` (~10 min, incluye stress test)

### Qué archivos audita el CI automáticamente

`run_ci.py` genera la lista de módulos de producción (`_PROD_PY_FILES`)
**recorriendo el repo** en vez de mantener una lista a mano. Los tres pasos
(syntax check, bandit y mypy) usan exactamente esa misma lista, así que un
módulo nuevo entra solo, sin editar nada.

**Se incluye todo `*.py` de la raíz y de `routes/`** (el único paquete de la
app), excepto los scripts de desarrollo. La regla es por convención de
nombres:

| Tipo | Regla | Ejemplos |
|------|-------|----------|
| **Producción** (pasa por el CI) | `.py` en raíz o `routes/` que no sea script de dev | `server.py`, `translator.py`, `ocr_engine.py`, `routes/api.py` |
| **Dev — prefijo** (se excluye) | Empieza con `benchmark_`, `test_`, `check_`, `generate_`, `analizar_`, `extraer_`, `rebuild_`, `reprocess_`, `run_`, `codegraph` o `diag_` | `benchmark_ocr.py`, `generate_report.py`, `run_unlimited_ocr.py` |
| **Dev — nombre exacto** (se excluye) | Lista fija en `_DEV_SCRIPT_NAMES` | `build.py`, `manga_ocr.py`, `stress_test_memory.py`, `translator_offline.py` |

**Regla para contribuidores:**

- ¿Es un módulo que la app importa o una herramienta de calidad que debe
  auditarse? **Ponlo en la raíz o en `routes/`** y pasa el CI automáticamente.
- ¿Es un script de desarrollo (benchmark, diagnóstico, generador, runner)?
  **Nómbralo con uno de los prefijos de dev** (`benchmark_x.py`, `test_x.py`,
  `generate_x.py`, …) y se excluye solo.
- ¿Un archivo `.py` nuevo apareció en la raíz sin clasificar? El CI lo
  **audita igualmente** (default seguro) pero lo reporta: *si es producción*,
  agrégalo a `_KNOWN_PROD_PY_FILES` (el baseline explícito) para silenciar
  el aviso; *si es dev*, renómbralo con un prefijo o agrégalo a
  `_DEV_SCRIPT_NAMES`. El test
  `test_prod_py_files_discovery_contract` valida que el walk y el baseline
  estén sincronizados — ningún módulo entra al CI silenciosamente.

  El reporte depende del entorno: **en GitHub Actions (`CI=true`) un archivo
  sin clasificar FALLA el job** (`[FAIL] clasificación de módulos`, exit 1) —
  ningún PR puede introducir un módulo sin clasificar; **en local solo avisa**
  con `[WARN]` y no afecta el resultado. Si corrés el CI local y querés
  simular el comportamiento estricto de GitHub Actions: `CI=true python run_ci.py`,
  o pasá **`--strict-classification`** para exigencia total en cualquier
  entorno (archivo sin clasificar = error, exit 1) — pensado para equipos
  que quieren el gate estricto también en local, sin depender del env CI.

### Cobertura por módulo

El paso pytest mide cobertura **por ruta** (`--cov=.`, vía `.coveragerc` que
omite site-packages/venv/dist y tolera módulos internos sin source como
`cv2/config-3.py` — un `--cov=cache` por nombre resolvía al *directorio*
`cache/` y no medía `cache.py`). Tras la suite, `_check_module_coverage`
verifica que **ningún módulo de producción toque por el diff baje de su
umbral** (`_COVERAGE_THRESHOLDS`, baseline medido con la suite completa).

**El gate se acota al diff del PR/rama**, no al estado global del repo:

- `_git_base_commit()` resuelve la base (`GITHUB_BASE_REF` en los PRs de
  GitHub Actions; localmente `origin/main`/`master`/`develop` o ramas
  locales equivalentes).
- `_touched_prod_modules()` combina `git diff <base>...HEAD` (los commits
  que la rama introduce) con `git diff HEAD` (working tree + staged — el
  caso típico de correr el runner local con trabajo sin commitear). Solo
  los módulos de producción listados fallan si bajan de umbral.
- Un módulo que el PR **no toca** y está bajo su umbral es deuda acumulada
  del repo, no responsabilidad del PR: se reporta como `[WARN]` y **no**
  hace fallar el paso. Si no hay diff disponible (sin git o sin base), el
  gate verifica todos los módulos (modo completo, el comportamiento
  original).

**Regla para contribuidores:** al tocar un módulo, no bajes su cobertura
(agrega tests o sube el umbral deliberadamente). Los módulos de pipeline
principal están exigidos por encima de la barrera de 70%: `translator.py`
(85.5%, checksums SHA256 CT2, carga de modelos y rate limiting de Google
cubiertos), `routes/main.py` (100%, rutas estáticas dev/prod + path
traversal), además de `server.py`, `uocr_client.py` y `uocr_daemon.py`
(60%+). Los entry points ya no están en 0%: `launcher.py` (97.96%) y
`main.py` (88.0%) se cubren con tests de arranque que mockean `argv`,
`subprocess.Popen` y `sys.frozen` (sin lanzar procesos ni navegadores
reales). Si agregas un módulo de
producción nuevo, dale umbral en `_COVERAGE_THRESHOLDS` — el test
`test_coverage_thresholds_cubren_todos_los_modulos_prod` lo exige. Con
`--skip-cov` se omite toda la medición y el gate. Los jobs de GitHub
Actions usan `actions/checkout` con `fetch-depth: 0` para que la base
esté disponible y el diff del PR se calcule bien.

**Reporte HTML por módulo (artifact del PR):** tras la suite, `run_ci.py`
escribe `coverage_html/index.html` (autocontenido, gitignored) con la
cobertura de los 20 módulos de producción: % cubierto, umbral, badge de
estado (OK / FAIL / bajo umbral no tocado), líneas sin cubrir compactadas
en rangos y marcador ▲ en los módulos que toca el diff del PR. Si existe
`coverage_base.json` (snapshot del % por módulo medido en main, refrescado
automáticamente por el workflow en cada push a main), agrega la columna Δ
con el cambio contra la base del PR, coloreada por signo (verde si subió,
rojo si bajó). Se genera desde el JSON de coverage (no con el HTML de
pytest-cov, que arrastraría los scripts de dev al reporte). El job `lint-test` lo sube como artifact
`coverage-report` con `if: always()`, así que se puede inspeccionar
incluso cuando el job falla.

**El arranque real del servidor también cuenta (job `server-test`):** el
job `server-test` corre el mismo runner (`run_ci.py --skip-mypy`) y, además
de la suite de pytest, `step_server` lanza `server.py` bajo
`coverage run --append` (misma base `.coverage` que pytest-cov). Al
término de los checks de health + endpoints, el servidor se detiene
limpiamente y se regenera el JSON de cobertura combinado para re-correr el
gate por módulo — así los imports reales del servidor (blueprints, rutas,
config, rate limiting) suman a la medición. Detalles de la implementación:

- Solo aplica cuando pytest ya midió (el JSON de baseline existe); sin
  baseline el servidor corre normal. `--skip-cov`, `--server` (modo smoke)
  y `--full` (el stress test necesita el servidor vivo, y bajo tracing el
  stress sería lentísimo) desactivan la medición del servidor.
- La detención limpia usa SIGINT en POSIX (GitHub Actions) — Python corre
  atexit y coverage flushea la data del servidor. En Windows local el
  equivalente requiere process group, así que se degrada al kill duro
  (avisa con `[WARN]` y re-verifica la data de pytest, idéntica a la que
  ya validó el paso anterior).
- El resultado se registra como `cobertura por módulo (con servidor)` en
  el resumen, y el reporte HTML se regenera con la data combinada. El job
  `server-test` lo sube como artifact `coverage-report-server` (nombre
  distinto para no colisionar con el de `lint-test` en el mismo run).

### Benchmark del split detector/recognizer

`benchmark_ocr_stages.py` mide cuánto paga **cada etapa** del OCR de
EasyOCR (detección CRAFT vs reconocimiento) llamando `detect()` y
`recognize()` por separado, igual que hace `readtext` internamente. Es la
herramienta de decisión para futuros cambios de modelo: antes de tocar el
pipeline hay que saber qué etapa domina el coste. No es parte del CI
(excluida por el prefijo `benchmark_`).

**Uso:**

```powershell
.\env\Scripts\python.exe benchmark_ocr_stages.py --pages 3,11,12 [--reps 2]
```

**Cómo leer los resultados:**

| Columna | Significado |
|---------|-------------|
| `detect` | Segundos del detector CRAFT (buscar cajas de texto) |
| `recognize` | Segundos del recognizer (leer el texto de cada caja) |
| `boxes` | Número de cajas detectadas en la página |
| `%rec` | Porcentaje del tiempo total que consume el recognize |

La regla de decisión (la misma que imprime el script al final):

- **recognize domina (>60%)** → probar recortar las cajas detectadas y
  reconocer con RapidOCR (PP-OCRv4) — recomendación 4.1 del plan.
- **detector domina (<40%)** → dejar CRAFT como está; los detectores
  alternativos rinden similar en manga y cambiar el recognizer no mueve
  la aguja.
- **Split equilibrado** → ganancia parcial; evaluar con
  `analisis_calidad.py` antes de adoptar el cambio.

Los resultados se persisten en `benchmark_results/ocr_stages.json` para
comparar entre sesiones (mismo patrón que `stress_semaphore.json`).

**Medición de referencia (cap. 43):** el detector domina con 72.9% del
coste — el recognizer NO es el cuello de botella, así que un cambio de
modelo de reconocimiento no acelera el pipeline actual. Si en el futuro
cambia el corpus (p.ej. páginas CJK) o la GPU, se re-mide con este mismo
script y la decisión se toma con datos.

### Mediciones de la Ruta C (A/B de parámetros)

La Ruta C (`_recover_regions_with_easyocr`) re-OCRea los crops de
YOLO/CTD y es el segundo coste más grande del pipeline. Todos sus A/B
usan un **protocolo estándar** para que los veredictos sean comparables
entre sesiones y no se repita la deriva de orden que invalidó los
primeros benchmarks (medían 3.5× vs 3.5× por un wrapper anidado).

**Protocolo estándar (obligatorio para cualquier A/B de la Ruta C):**

1. **Daemon VLM DETENIDO** — el daemon comparte la GPU y su ruido es
   ~25× el noise-floor del harness (medido 2026-08-15); con el daemon
   arriba los Δ quedan ocultos en la varianza.
2. **`--reps 3`** — mediana de 3 pares base/alt por página (suaviza el
   ruido térmico/GPU). Con `--reps 1` los veredictos salen NO
   CONCLUYENTES por diseño.
3. **Páginas de control automáticas** — el harness elige las páginas
   sin etapa Ruta C como noise-floor y el veredicto compara el Δ de las
   afectadas contra ese ruido.
4. **Escala 1.2** (pdf.js default de producción), nunca 300 dpi.
5. Todos los benchmarks comparten `benchmark_ab_utils.py` (intercalado
   por página, orden alternado par b→a / impar a→b, veredicto explícito:
   atribuible / cautela / NO CONCLUYENTE).

**Uso:**

```powershell
# 1) detener el daemon VLM (puerto 5177) para medir limpio
# 2) correr los A/B (cada uno escribe su JSON):
.\env\Scripts\python.exe benchmark_rutac_params.py --reps 3 --param pad      # pad
.\env\Scripts\python.exe benchmark_rutac_params.py --reps 3 --param box_thresh
.\env\Scripts\python.exe benchmark_rutac_params.py --reps 3 --param unclip
.\env\Scripts\python.exe benchmark_rutac_upscale.py --reps 3                 # 3.5× vs 2×
.\env\Scripts\python.exe benchmark_rutac_recovery.py --reps 3                # recuperación texto-a-texto
.\env\Scripts\python.exe benchmark_rutac_batch.py --reps 3                   # batch estructural (strip)
```

**Veredictos vigentes (2026-08-15, cap. 43, daemon detenido, `--reps 3`):**

| A/B | Resultado | Veredicto | JSON |
|-----|-----------|-----------|------|
| **upscale** 3.5× vs 2× | 2× NO es más rápido (+0.115 s/pág, dentro del noise 0.116 s) y **pierde bloques**: 49→47 en 7 págs (105 vs 100 en 14); recovery 34 vs 32, con la pérdida concentrada en pág 11 | **Mantener 3.5×** | `rutac_upscale_ab.json`, `rutac_recovery_ab.json` |
| **pad** 0.03 (prod.) vs 0.06 | −0.08 s/pág (−3.0 %), 49 = 49 bloques, crops 7 = 7, control estable 0.083 s | **Sin cambio** (efecto pequeño atribuible) | `rutac_params_pad.json` |
| **box_thresh** 0.5 (prod.) vs 0.35 | 0.35 pierde 1 bloque | **Default confirmado (0.5)** | `rutac_params_reps3.json` |
| **unclip_ratio** 1.6 (prod.) vs 2.2 | 2.2 pierde 4 bloques | **Default confirmado (1.6)** | `rutac_params_reps3.json` |
| **interp** CUBIC vs LINEAR | −0.12 s/pág pero dentro del noise de control (0.456 s) | **NO CONCLUYENTE — sin cambio** | `rutac_params_interp.json` |
| **rotation** (0,180) | −0.06 s/pág dentro del noise (0.212 s) | **NO CONCLUYENTE — sin cambio** | `rutac_params_rotation.json` |
| **batch estructural (strip)** | **−2.38 s/pág** en las 8 páginas con Ruta C | **Aplicado en producción** | `rutac_batch_ab.json` |

Los resultados se persisten en `benchmark_results/*.json` y el plan de
optimización (`PLAN_OPTIMIZACION_RENDIMIENTO.md`, §4.6) mantiene el
cuadro consolidado con el noise-floor de cada corrida. Regla de lectura:
un veredicto NO CONCLUYENTE significa que el Δ no supera el ruido de las
páginas de control a ese `--reps` — re-correr con `--reps 5-7` o más
páginas de control antes de decidir, nunca cambiar producción con datos
ambiguos.
