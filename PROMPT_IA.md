# PROMPT_IA.md — Cómo trabajar en **Traductor Visual Pro** sin gastar tokens

> **Ubicación del proyecto:** `D:\crear traductor\` (raíz del repo)

---

## 1. Lee SOLO estos 3 archivos (en orden)

| Archivo | Ruta absoluta | Qué aporta | Tokens aprox |
|---|---|---|---|
| **`AGENTS.md`** | `D:\crear traductor\AGENTS.md` | Guía completa: arquitectura, zonas sensibles, zonas seguras, bugs históricos, flujo de trabajo, sincronizaciones críticas | ~8k |
| **`CODEGRAPH.md`** | `D:\crear traductor\CODEGRAPH.md` | Grafo de funciones/imports de `server.py`, `app.js`, `index.html` — quién llama a quién | ~3k |
| **`AGENTS.md` §4 (Estado Actual)** | `D:\crear traductor\AGENTS.md` (sección 4) | Últimos cambios, bugs pendientes, zonas a vigilar | ~2k |

**Total: ~13k tokens** — entiendes el proyecto completo.

---

## 2. NO leas (ahorra tokens)

- `server.py` completo (léelo solo si tocas zona sensible citada en AGENTS.md)
- `app.js` completo (idem)
- `manga_pipeline.py` (solo si tocas MIT pipeline)
- `codegraph.py` / `codegraph_html.py` (son generadores, no documentación)

---

## 3. Protocolo de sesión

1. **Inicio**: Lee `AGENTS.md` + `CODEGRAPH.md` + §4 de AGENTS.md.
2. **Durante**: Si tocas zona sensible → documenta en §4 (Estado Actual) qué cambias y por qué.
3. **Antes de agotar contexto**: Actualiza §4 con:
   - Qué cambiaste (archivo, línea aprox, qué)
   - Queda pendiente / a medio terminar
4. **No pospongas** la actualización de §4.

---

## 4. Zonas sensibles (resumen rápido — véase AGENTS.md §2)

| Archivo | Función crítica | Por qué |
|---|---|---|
| `app.js` | `wrapTextLines`, `containsCJK`, `filterPageBlocks`, `makeAutoTextBox`, `serverProcessPage`, `autoTranslateCurrentPage`, `fitTextLayout`, `drawProfessionalText` | Cambios rompen layout de texto, filtrado de ruido, fusión OCR, renderizado, exportación |
| `server.py` | `_group_and_merge_blocks`, `_detect_and_ocr`, `_translate_one`, `_ensure_argo_package`, `process_page`, `_build_inpaint_mask`, `_inpaint_image`, `_get_ocr_reader` | Cambios rompen fusión OCR, detección, traducción, inpainting, carga de modelos |
| Sincronizadas | `filterPageBlocks` ↔ `_group_and_merge_blocks`, `MARGIN_NOISE_PATTERNS` ↔ `_MARGIN_NOISE_PATTERNS`, `GLOBAL_NOISE_PATTERNS` ↔ `_WATERMARK_PATTERNS`, `state.inpaintedBgByPage` ↔ `inpainted_image`, `makeAutoTextBox.eraseMode` ↔ inpainting servidor | Deben mantenerse idénticas; divergencia = comportamiento impredecible |

---

## 5. Comandos útiles (copia/pega)

```powershell
# Levantar servidor (desde raíz del proyecto)
$env:PYTHONIOENCODING="utf-8"; & "D:\crear traductor\env\Scripts\python.exe" "D:\crear traductor\server.py"

# Rebuild .exe (si tocaste main.py/server.py)
cd D:\crear traductor
.\env\Scripts\pyinstaller.exe --onedir --add-data "dist;dist" --exclude-module torch --exclude-module transformers --exclude-module timm --exclude-module easyocr --exclude-module manga_ocr --noconfirm main.py

# Copiar a escritorio
$src = "D:\crear traductor\dist\main"; $dst = "$env:USERPROFILE\Desktop\TraductorVisual"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
Copy-Item $src $dst -Recurse -Force

# Git
git add -A && git commit -m "msg" && git push origin main
```

---

## 5. Estado actual (resumen ultra-corto)

- **Último deploy**: `main.exe` en escritorio (`C:\Users\roweh\Desktop\TraductorVisual\main.exe`) — consola se auto-oculta.
- **OCR**: umbrales bajados (0.15/0.10/0.08), fusión horizontal tolerante (`max(35, w*2.5)`), char suelto tolerado si `conf≥0.25`.
- **MIT pipeline** disponible como fallback opcional (`manga_pipeline.py`).
- **Pendiente vigilar**: burbujas en margen 5%, sello `1 C 2 E`, PDF.js fallback frequency.

---

## 6. Cómo pasarle esto a la IA

> **Copia esto al inicio de tu chat con la IA:**
> ```
> Lee AGENTS.md, CODEGRAPH.md y la sección 4 de AGENTS.md. No leas server.py/app.js completos salvo que toque zona sensible (ver tabla en PROMPT_IA.md). Sigue el protocolo de §3.
> ```

---

**Ubicación de este archivo:** `D:\crear traductor\PROMPT_IA.md`