# PLAN_MEJORA_GLOBAL — Optimización, mejora y corrección integral

> **Fecha**: 2026-08-26 · **Alcance**: escaneo completo del repo (raíz, `routes/`, `js/`, `tests/`, `tools/`, `.github/`, artefactos).
> **Método**: inventario por carpeta, mapa de funciones de los módulos core, búsqueda de patrones de deuda
> (TODO/FIXME, `except` amplios, `shell=True`, llamadas sin timeout), verificación de sincronización
> frontend↔backend y del pipeline de build/CI.

---

## 0. Resumen ejecutivo

El proyecto está **sano en su núcleo**: CI maduro con gate de cobertura por módulo acotado al diff,
985 funciones de test, mypy strict, bandit, benchmarks A/B con protocolo estadístico y documentación
viva (`AGENTS.md`, `CODEGRAPH.md`). Los problemas concentrados son:

| # | Hallazgo | Severidad | Esfuerzo |
|---|----------|-----------|----------|
| BUG-1 | `build.py` NO copia `js/` a `dist/` → en modo producción (`IS_PRODUCTION`) los imports ES fallan y la app no arranca | 🔴 P0 | 15 min |
| BUG-2 | Sin test que garantice la paridad de patrones de ruido entre `js/filters.js` y `config.py`. **Verificado 2026-08-26: hay divergencias reales hoy** — (a) 2 de los 10 patrones MARGIN difieren sutilmente (JS `p\.?\s*m\.?` permite espacio; Py `p\.?m\.?` no); (b) el patrón URL existe solo en JS (el backend lo aplica aparte en `ocr_utils.py:3268`); (c) `zonaolympus` tiene anclas `\b…\b` en JS y ninguna en Py | 🟠 P1 | 1-2 h |
| DEUDA-1 | Raíz saturada: 36 `benchmark_*.py` (6.529 líneas) + ~15 scripts dev one-off conviven con los módulos de producción | 🟠 P1 | 2-3 h |
| DEUDA-2 | `app.js`: 3.086 líneas monolíticas; solo 5 de sus responsabilidades están modularizadas | 🟠 P1 | 3-5 días |
| DEUDA-3 | `_js_syntax_files()` en `run_ci.py` hardcodea la lista de JS — un módulo nuevo no entra al CI solo | 🟡 P2 | 30 min |
| DEUDA-4 | 81 `except Exception` en módulos core — varios tragan errores sin log | 🟡 P2 | progresivo |
| DEUDA-5 | `ocr_utils.py` (3.815 lín., 67 defs) mezcla 5 responsabilidades; split posible pero con costo PyInstaller | 🟢 P3 | opcional |
| MEJORA-1 | `build.js` no minifica JS ("copia preservando ES6") ni hay bundle real; `replace("app.js", "app.min.js")` es naive | 🟡 P2 | 2-4 h |
| MEJORA-2 | Listeners `input` de estilos re-renderizan todo el overlay sin debounce (glow preview hace `renderBoxes()` completo por hover) | 🟢 P3 | 2 h |

**Orden recomendado**: W1 (fixes P0/P1) → W2 (modularización app.js) → W3 (saneamiento raíz)
→ W4 (robustez backend) → W5 (perf) → W6 (opcional: split ocr_utils).

---

## 1. Hallazgos por carpeta (escaneo)

### 1.1 Raíz — núcleo de producción ✅ mayormente sano

| Archivo | Lín. | Estado | Observaciones |
|---------|-----:|--------|---------------|
| `server.py` | 388 | ✅ | Bien: preload ordenado EasyOCR→CT2, waitress, límites de contenido. 9 `except Exception` — revisar cuáles silencian sin log. |
| `config.py` | 658 | ✅ | Fuente única de constantes y timeouts. `IS_PRODUCTION` se resuelve por existencia de `dist/index.html` — **trampa**: correr `build.py` una vez activa producción para siempre hasta borrar `dist/`. Documentar o hacer opt-in por env var. |
| `translator.py` | 1.498 | ✅ | Pipeline secuencial CT2→Google→SIN_TRAD correcto; checksums SHA256 CT2. 15 `except Exception`. |
| `ocr_utils.py` | 3.815 | ⚠️ | Monolito de 67 defs: EasyOCR + RapidOCR + fusión + inpainting + filtros + YOLO + spellcheck. Ver W6. |
| `ocr_engine.py` | 1.792 | ✅ | `OCRManager` bien aislado y testeado (94,6% cov). 1 clase gigante; los triggers ya son métodos testeables. |
| `routes/api.py` | 1.888 | ⚠️ | 40 defs en un blueprint. Candidato a dividir en `routes/ocr.py` + `routes/translate.py` (bajo riesgo: mismo patrón que `routes/main.py`). |
| `uocr_client.py` / `uocr_daemon.py` | 283/655 | ✅ | Cliente stdlib puro, health/retry correctos. |
| `cache.py` / `models.py` / `ratelimit.py` | 188/363/45 | ✅ | Cobertura 68%/76%/80%. `cache.py` bajo umbral relativo — subir a ≥80% es barato. |
| `main.py` / `launcher.py` / `main.spec` | 244/—/— | ✅ | Spec correcto: `js/` viaja como carpeta completa en DATAS (**la división de app.js NO requiere tocar el spec**). 88% cov. |
| `run_ci.py` | 1.958 | ⚠️ | Excelente gate de cobertura, pero `_js_syntax_files()` hardcodeada (DEUDA-3) y clasificación prod/dev por prefijos depende de que los scripts dev vivan en raíz (ver W3: moverlos cambia el walk — actualizar contrato). |

### 1.2 Raíz — frontend ⚠️ el foco principal

- **`app.js` (3.086 lín.)**: concentra DOM refs, estado global, carga de CDNs, render, editor,
  layout de texto, inpainting cliente, exportación, panel de bloques, efectos, atajos y boot.
  Mapa de extracción propuesto en §W2.
- **`index.html`** (454+): CSP correcta, cargadores CDN con fallback. El `<script type="module" src="app.js?v=...">`
  lleva versión hardcodeada — automatizar bump.
- **`styles.css`** (~1.811): zona segura según AGENTS.md. OK.
- **`js/` (5 módulos)**: patrón limpio (exports nombrados, sin side effects globales salvo `CLIENT_CONFIG`).
  - `filters.js` vs `config.py`: comparación programática (AST Python + scanner JS) confirmó:
    `MARGIN_NOISE_PATTERNS` 10 vs 10 con **8 idénticos y 2 divergentes** (`\d{1,4}\s*p\.?\s*m\.?` en JS
    vs `\d{1,4}\s*p\.?m\.?` en Py — el JS acepta "4.58 p. m.", el Py solo "4.58 p.m");
    `GLOBAL_NOISE_PATTERNS` (JS, 5) ≠ `WATERMARK_PATTERNS` (Py, 4): el patrón URL vive solo en JS porque
    el backend lo filtra por su cuenta (`ocr_utils.py:3268`), y `zonaolympus` lleva `\b…\b` en JS pero no en Py.

### 1.3 Raíz — scripts de desarrollo 🟠

- **36 × `benchmark_*.py` (6.529 lín.)** + `test_*.py`/`check_*.py`/`run_*.py`/`analizar_*.py`/`diag_*.py`/
  `reprocess_*.py`/`rebuild_*.py`/`extraer_*.py` (~15 más). Están en raíz porque `run_ci._PROD_PY_FILES`
  camina raíz+`routes/` y excluye por prefijo. Consecuencias: ruido extremo, riesgo de clasificar mal un
  módulo nuevo, diffs imposibles de navegar. → W3 propone moverlos a `benchmarks/` y `tools/dev/`.
- `rebuild_and_fix.py` usa `shell=True` (×3) — script dev local, riesgo bajo, pero migrar a listas de args.

### 1.4 `tests/` ✅

27 archivos, 985 `def test_`. Cobertura base por módulo en `coverage_base.json` (mínimos: `server.py`,
`uocr_*` 60%+; core ≥70%). Mejoras puntuales: tests que greppean texto de `app.js`
(`test_packaging.py::test_traduccion_total_no_renderiza…`, `test_server_process_page_no_copia…`)
**se romperán con W2** — deben apuntar al archivo nuevo del módulo extraído.

### 1.5 `tools/` ✅

20 scripts (entrenamiento detector, etiquetado VLM, medición de firmas). Bien aislados; solo les falta
un README índice.

### 1.6 `.github/workflows/ci.yml` ✅ con mejoras menores

Jobs lint-test/server-test/build/release-validate bien encadenados; bot de coverage baseline idempotente.
Mejoras: (a) el job `build` sube `dist/` como artifact pero nadie valida que `dist/js/config.js` exista
(hubiera atrapado BUG-1); (b) pip-install duplicado entre jobs → matrix o composite action;
(c) considerar Dependabot para `requirements.txt`.

### 1.7 Artefactos (`benchmark_results/`, `audit_results/`, `docs/superpowers/`)

JSON de resultados versionados — correcto para comparar sesiones. Sugerencia: moverlos bajo
`benchmark_results/` ya existente está bien; agregar `.gitattributes` para diffs limpios de JSON.

---

## 2. Bugs confirmados

### 🔴 BUG-1 — `dist/` roto: `build.py` no copia `js/`

`build.py::copy_assets()` solo copia `assets/` (que no existe) y `copy_index()` reemplaza
`app.js`→`app.min.js`. Resultado: `dist/app.min.js` importa `./js/config.js` … que **no existe en dist/**.
Como `config.py` decide `IS_PRODUCTION = dist/index.html existe`, basta haber corrido `build.py` una vez
para que el server sirva desde `dist/` y la app muera con el badge "Cargando OpenCV…" (el modo de fallo
exacto que documenta `main.spec`). El job `build` de GitHub Actions genera este artifact roto en cada push a main.

**Fix (W1-A)**: en `build.py` añadir `shutil.copytree(js/, dist/js/, dirs_exist_ok=True)` +
copiar `icon.ico`/favicon si aplica. **Guardia (W1-B)**: test pytest que corre `build.py` en `tmp_path`
(monkeypatching ROOT/DIST) y assertion `dist/js/*.js` existen + `node --check` sobre el resultado.

### 🟠 BUG-2 — Paridad de patrones de ruido sin red (con divergencias reales verificadas)

AGENTS.md lo marca como zona crítica ("divergencia = textos basura o diálogos eliminados").
Comparación programática del 2026-08-26 (AST sobre `config.py`, scanner de literales sobre `js/filters.js`):

1. **MARGIN_NOISE_PATTERNS**: 10 vs 10 entradas; 8 idénticas; **2 divergen**: JS `\b\d{1,4}\s*p\.?\s*m\.?\b`
   y `\b\d{1,6}…p\.?\s*m\.?\b` aceptan espacios entre "p." y "m."; las variantes Py `p\.?m\.?` no.
2. **GLOBAL_NOISE_PATTERNS (JS, 5) vs WATERMARK_PATTERNS (Py, 4)**: el patrón URL `https?://|www\.|…`
   existe SOLO en JS; el equivalente backend está hardcodeado aparte en `ocr_utils.py:3268`
   (dentro del filtro post-merge), fuera del array sincronizado. Además JS ancla `\bzonaolympus…com\b`
   mientras Py usa `zonaolympus[\s-]?com` sin anclas.
3. No existe ningún test que detecte estas derivas automáticamente.

**Fix (W1-C)**: `tests/test_filters_sync.py` que parsee ambos archivos (vía `ast` en Python y scanner de
literales en JS) y compare cantidad + semántica, con allowlist explícita para las diferencias intencionales
(patrón URL backend-side). Primero alinear las 2 divergencias reales de MARGIN (decidir si `\s*` es correcto
y replicarlo en ambos lados).

### 🟡 Bugs menores

- `module.exports = {}` vacío al final de `app.js` — código muerto (eliminar en W2).
- `html.replace("app.js", "app.min.js")` reemplaza cualquier ocurrencia (comentarios incluidos) — usar
  replace acotado al atributo `src`.
- `minify_js()` no minifica (rjsmin rompería ES modules) — documentarlo o introducir esbuild (W5-C).

---

## 3. Planes de trabajo

### W1 — Fixes inmediatos (P0/P1) · ~1 día

1. **W1-A** `build.py`: copiar `js/` a `dist/` (y assets opcionales). Criterio: job `build` verde con
   `dist/js/config.js` presente.
2. **W1-B** Test de humo de build en pytest (corre `build.py` contra tmp y valida estructura de `dist/`).
3. **W1-C** `tests/test_filters_sync.py` (paridad de patrones JS↔Py con allowlist documentada).
4. **W1-D** `run_ci.py::_js_syntax_files()` → derivarla con `glob` de `app.js + js/**/*.js`
   (nuevos módulos entran solos; elimina DEUDA-3).

### W2 — Modularización de `app.js` (P1) · 3-5 días

Estrategia *strangler*: un módulo por commit, `node --check` + smoke manual después de cada uno.
**El bundle PyInstaller no cambia** (`main.spec` ya empaqueta `js/` completo).

Capas (de abajo hacia arriba, sin imports circulares):

```
js/dom.js          ← $ + todas las refs de elementos (~60 lín, app.js:48-127)
js/state.js        ← state + getPageBoxes() (~40 lín, :130-155, :573)
js/api-client.js   ← translateOnline/Batch, detectPage*, checkServerHealth, getDocId,
                     getSelectedOcrMode, updateOcrEngineStatus(+_stopUocrPolling),
                     buildProcessPageQuery, finalizeServerResult, serverProcessPage (~400 lín)
js/pdf-loader.js   ← loadPdfJs + _loadPdfJsUmd (~100 lín, :471-567)
js/opencv-loader.js← initOpenCv (~90 lín, :156-246)
js/progress.js     ← showProgress (~90 lín, :1161-1245)
js/text-layout.js  ← drawTextOnCanvas, renderBoxes, fitTextLayout, containsCJK,
                     wrap* — ZONA SENSIBLE bug histórico CJK (~340 lín, :1802-2135)
js/renderer.js     ← renderPage, renderImage, updateErasedBg, refreshScreenCanvas,
                     resizeStage, renderRawPdfPage, fitPageToStage (~250 lín)
js/inpaint-client.js ← eraseWithInpainting, markGlyphMask, fallbackEraseBox,
                     sampleEraseColor (~180 lín, :2340-2510)
js/editor.js       ← selectBox, updateSelectedBox, startOverlayAction, pointer*,
                     getRelativePoint, toHexColor, makeAutoTextBox,
                     sampleBgColorAround/sampleTextColor, isLikelyBubble (~450 lín)
js/blocks-panel.js ← renderBlockList, escHtml, truncate (~140 lín)
js/effects-panel.js← updateTextEffectsPanelState, listeners glow/fillOpacity/estilos (~250 lín)
js/exporters.js    ← _getExportCanvas, renderEditedCanvas, exportCurrentPng/Pdf,
                     exportFullPdf, drawProfessionalText (~220 lín)
js/shortcuts.js    ← initKeyboardShortcuts, checkLanguageWarning (~220 lín)
app.js (queda)     ← bootApp, wiring de botones, fetchClientConfig, interceptores
                     Tesseract/error, autoTranslateAllPages/orquestación (~350 lín)
```

Pasos por commit:
1. Extraer módulo → importar en `app.js` → `window.__appJsLoaded = true` sigue siendo la primera línea
   ejecutable (cualquier import roto lo impide y el safety timeout de index.html lo reporta).
2. Actualizar tests que greppean `app.js` (`test_packaging.py`) para apuntar al archivo nuevo.
3. `node --check` en todos los JS (tras W1-D es automático) + suite pytest.
4. Smoke manual: cargar PDF → Traducir Página → editar burbuja → glow → export PNG y PDF completo.
5. Commit `refactor(frontend): extraer js/<modulo> desde app.js`.

Reglas de oro: `state` y los canvas buffer (`cleanBgCanvas`, `erasedBgCanvas`) viven SOLO en `state.js`;
ningún módulo muta DOM fuera de su responsabilidad; `drawTextOnCanvas` sigue siendo el único punto de dibujo.

### W3 — Saneamiento de la raíz (P1) · medio día

1. Crear `benchmarks/` y mover los 36 `benchmark_*.py` + sus JSON de salida siguen en `benchmark_results/`.
2. Mover scripts one-off (`analizar_*`, `check_*`, `diag_*`, `rebuild_and_fix`, `restart_and_process.ps1`, etc.)
   a `tools/dev/`.
3. Actualizar `run_ci._PROD_PY_FILES`: el walk de raíz queda limpio (solo prod) — simplifica el contrato
   de descubrimiento; ajustar `test_prod_py_files_discovery_contract` y referencias en README/AGENTS/CODEGRAPH.
4. Migrar `shell=True` de `rebuild_and_fix.py` a lista de args.

### W4 — Robustez backend (P2) · progresivo

1. Auditoría de los 81 `except Exception`: clasificar en (a) legítimo con log, (b) estrechable a excepción
   concreta, (c) silenciador peligroso. Regla: ningún `except Exception: pass` sin comentario `# nosec` + log.
2. Dividir `routes/api.py` en blueprints `translate` / `ocr` / `health` (mismo patrón que `routes/main.py`;
   actualizar `_KNOWN_PROD_PY_FILES` y umbrales de cobertura).
3. Subir cobertura de `cache.py` (68%) y consolidar TTL/LRU en tests de propiedad.

### W5 — Rendimiento (P3, medido con los benchmarks existentes)

1. **Frontend**: debounce (100ms) en listeners `input` de estilos; en glow preview re-renderizar solo la
   caja afectada en vez de `renderBoxes()` completo. Verificar con el profile de DevTools que el hover deja
   de costar O(cajas).
2. **Build real (opcional)**: esbuild para bundle+minificar `app.js + js/*` → un único `app.min.js`
   (elimina cascada de imports y reduce requests); mantener fallback de archivos sueltos.
3. **Backend**: ya está medido que el detector CRAFT domina (72,9%) — no tocar sin re-medir con
   `benchmark_ocr_stages.py`; candidata barata: persistir el cache positivo VLM dHash ya implementado
   (verificar hit-rate con `benchmark_vlm_tokens.py`).

### W6 — (Opcional) Split de `ocr_utils.py` · solo si W2-W4 están asentados

Dividir en paquete `ocr/` (`easyocr.py`, `rapid.py`, `fusion.py`, `inpaint.py`, `filters.py`, `yolo.py`).
**Costo real**: `main.spec` lista cada .py en DATAS (habría que añadir el paquete, como `routes/`);
umbrales de cobertura y contract test de `run_ci` cambian; los mocks de pytest acceden por nombre de
módulo. Recomendación: hacerlo detrás de un commit único con A/B de calidad (`analisis_calidad.py`)
antes/después.

---

## 4. Roadmap sugerido

| Sprint | Contenido | Entregable verificable |
|--------|-----------|------------------------|
| S1 | W1 completo | CI verde + `dist/js/` validado + test de sync de filtros |
| S2-S3 | W2 (14 módulos, ~1 commit c/u) | `app.js` ≤400 lín; suite y smoke en verde |
| S4 | W3 + W4.1 | Raíz ≤25 .py (solo producción); auditoría except documentada |
| S5 | W4.2 + W5.1 | `routes/` dividido; hover sin re-render completo (medición antes/después) |
| S6+ | W5.2/W6 según necesidades | esbuild o paquete `ocr/` con A/B de calidad |

## 5. Métricas de éxito

- `node --check` automático sobre todo `js/**/*.js` en cada CI (sin lista manual).
- Test de sync de patrones en verde y con allowlist documentada.
- Build artifact de GitHub Actions ejecutable de punta a punta (smoke de `dist/`).
- `app.js` < 500 líneas; ningún archivo frontend > 600.
- Raíz del repo sin scripts `benchmark_*/dev`.
- Ningún `except Exception: pass` sin justificación en módulos de producción.
