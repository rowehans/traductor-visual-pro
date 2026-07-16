# CODEGRAPH — Traductor Visual Pro (token-minimal)

## server.py (Flask, port 5174)
**Core**: `_get_executor` `shutdown_executor` `add_security_headers`  
**OCR**: `_get_ocr_reader(lang)` → lazy EasyOCR (latin/ja/ko/zh) GPU→CPU fallback — **OR** `manga_pipeline.run_pipeline` → MIT (CTD + OCR 48px)  
**Translate**: `_translate_one(txt,src,tgt)` → Argos→Google fallback; `_translate_batch` ThreadPool  
**LangDetect**: `_detect_language_robust` (thread-local langdetect + heuristics ES/JA/KO/ZH)  
**Pipeline** (`process_page`): b64→cv2 → MIT (CTD detection + OCR 48px + LaMa inpainting) **OR** legacy (EasyOCR + OpenCV INPAINT_NS) → translate batch → b64 response  
**Routes**: `GET /` `GET /<path>` `POST /api/translate` `POST /api/translate-batch` `POST /api/process-page` `GET /api/health`

## manga_pipeline.py (new)
**Import**: solo submódulos de `manga-image-translator` (detection, ocr, textline_merge, inpainting) — NO carga translators/renderers  
**Init**: `ensure_ready()` → descarga CTD (76MB), OCR 48px (195MB), LaMa (195MB)  
**Pipeline**: `run_pipeline(img_bgr)` → CTD detect → OCR 48px → textline_merge → LaMa inpaint → `{inpainted_image, blocks}`  
**Fallback**: si error → server.py usa EasyOCR + OpenCV legacy automáticamente

## app.js (~2500 LOC)
**State**: `kind/pdf|image` `pdf` `page/pageCount` `scale=1.8` `boxesByPage:Map` `selectedId` `cvLoaded` `inpaintedBgByPage:Map`  
**Boot**: `loadPdfJs` (ESM v4.10→UMD v3.11 fallback) `checkOpenCv` (12s timeout) `initTheme` `initKeyboardShortcuts`  
**PDF**: `renderPage` → pdf.js @scale → `cleanBgCanvas` → `updateErasedBg` (server inpainted || local OpenCV) → `renderBoxes`  
**Editor**: `renderBoxes` (canvas text + overlay divs) `fitTextLayout` (wrap CJK char / latin word) `selectBox` → sync UI  
**Events**: `pointerdown/move/up` (draw/move/resize) `autoTranslateCurrentPage` (server OCR+inpaint+translate) `autoTranslateAllPages`  
**Export**: `renderEditedCanvas` (inpaint + drawProfessionalText) → PNG / jsPDF page / multi-page PDF  
**Shortcuts**: D/V modes, Ctrl+T/E/P/S, arrows, Del, Ctrl+N/I/B

## index.html
**IDs (44)**: `fileInput` `prevPage` `pageNumber` `pageTotal` `nextPage` `sourceLang` `targetLang` `drawMode` `moveMode` `autoTranslateOnLoad` `autoDetectPage` `autoTranslateAll` `eraseMode` `coverOriginal` `bubbleColor` `textColor` `strokeColor` `strokeWidth` `fontFamily` `fontSize` `btnItalic` `btnBold` `sourceText` `translateBtn` `translatedText` `placeManualBtn` `deleteBox` `clearPageBoxes` `exportName` `exportPng` `exportPdf` `exportAllPdf` `docName` `status` `mobileMenuBtn` `opencvBadge` `fitPage` `printPage` `stageWrap` `stage` `pdfCanvas` `overlay` `emptyState` `opencvScript` `dismiss-leo-warning`  
**Scripts**: jspdf (cdn.jsdelivr), OpenCV.js (docs.opencv.org), app.js (defer)  
**CSP**: `connect-src 'self' http://127.0.0.1:5174 https://cdnjs.cloudflare.com https://cdn.jsdelivr.net data:`

## styles.css
**Tokens**: `--bg-app #040406` `--accent #10b981` `--radius-md 12px` `--transition 200ms`  
**Layout**: `.app` grid `minmax(240px,25%) 1fr` → `.sidebar` fixed 250px → `.workspace` flex-col `flex:1` → `.stage-wrap` overflow-auto center  
**Responsive**: ≤1024px sidebar drawer (hamburger) ≤640px stacked controls

## Launch
`start-app.bat` / `start-app.ps1` → `env\Scripts\python.exe server.py` → open `http://127.0.0.1:5174` (Chrome app mode)

## Env
`env/` (venv completo) `ocr_models/` (EasyOCR cache) `requirements.txt` pinned