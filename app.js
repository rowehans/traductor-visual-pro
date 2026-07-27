// ─── App entry point ─────────────────────────────────────────
// Imports from modularized JS modules
import { formatDuration, canvasToBase64, loadBase64IntoCanvas, getBlockText, mergeLinesIntoBlocks, isLightColor } from "./js/utils.js";
import { CLIENT_CONFIG, fetchClientConfig } from "./js/config.js";
import { showToast } from "./js/toast.js";
import { initTheme, toggleTheme } from "./js/theme.js";
import { filterPageBlocks, MARGIN_NOISE_PATTERNS, GLOBAL_NOISE_PATTERNS } from "./js/filters.js";

// Make CLIENT_CONFIG available globally for inline scripts
window.__CLIENT_CONFIG = CLIENT_CONFIG;

// SELECTOR DE ELEMENTOS DOM
// APP VERSION: 20260714-NO-TESSERACT
// Intercept any Tesseract loading attempts
if (window.Tesseract) {
    console.error('Tesseract.js found in window at startup! Source:', (document.currentScript?.src || 'inline'));
} else {
    // Watch for Tesseract being loaded
    let _tesseractVal;
    Object.defineProperty(window, 'Tesseract', {
        set: function(val) {
            console.error('Tesseract.js being SET on window! Stack:', new Error().stack);
            _tesseractVal = val;
        },
        get: function() {
            return _tesseractVal;
        },
        configurable: true
    });
}

// ─── Fetch server config to override defaults ────────────────
fetchClientConfig(); // async, non-blocking

const $ = (selector) => document.querySelector(selector);

console.log("[BOOT] app.js cargado, document.readyState:", document.readyState);

// GLOBAL ERROR INTERCEPTOR - captures ALL errors with full stack
// GLOBAL ERROR INTERCEPTOR - captures ALL errors with full stack
window.addEventListener('error', function(e) {
    console.error('=== GLOBAL ERROR CAUGHT ===');
    console.error('Message:', e.message);
    console.error('Filename:', e.filename);
    console.error('Line:', e.lineno, 'Col:', e.colno);
    console.error('Stack:', e.error?.stack);
    // Catch the specific "Cannot read image.png" error
    if (e.message && e.message.includes('Cannot read') && e.message.includes('image.png')) {
        console.error('>>> SPECIFIC IMAGE.PNG ERROR DETECTED <<<');
        console.error('This error is NOT from our code - likely from browser extension or external script');
        console.error('Full error object:', e);
    }
    console.error('==============================');
}, true);

window.addEventListener('unhandledrejection', function(e) {
    console.error('=== UNHANDLED REJECTION ===');
    console.error('Reason:', e.reason);
    console.error('Stack:', e.reason?.stack);
    console.error('==============================');
}, true);

// Referencias de Elementos
const fileInput = $("#fileInput");
const pageNumber = $("#pageNumber");
const pageTotal = $("#pageTotal");
const prevPage = $("#prevPage");
const nextPage = $("#nextPage");
const docName = $("#docName");
const statusText = $("#status");
const pdfCanvas = $("#pdfCanvas");
const cleanBgCanvas = document.createElement("canvas"); // Buffer de fondo limpio
const erasedBgCanvas = document.createElement("canvas"); // Buffer de fondo borrado (inpainted)
const overlay = $("#overlay");
const stage = $("#stage");
const stageWrap = $("#stageWrap");
const emptyState = $("#emptyState");
const drawModeBtn = $("#drawMode");
const moveModeBtn = $("#moveMode");
const coverOriginal = $("#coverOriginal");
const bubbleColor = $("#bubbleColor");
const textColor = $("#textColor");
const strokeColor = $("#strokeColor");
const strokeWidth = $("#strokeWidth");
const fontSize = $("#fontSize");
const fontFamily = $("#fontFamily");
const sourceText = $("#sourceText");
const translatedText = $("#translatedText");
const targetLang = $("#targetLang");
const sourceLang = $("#sourceLang");
const translateBtn = $("#translateBtn");
const autoDetectPage = $("#autoDetectPage");
const autoTranslateAll = $("#autoTranslateAll");
const deleteBox = $("#deleteBox");
const clearPageBoxes = $("#clearPageBoxes");
const exportPng = $("#exportPng");
const exportPdf = $("#exportPdf");
const exportAllPdf = $("#exportAllPdf");
const exportName = $("#exportName");
const printPage = $("#printPage");
const fitPage = $("#fitPage");
const btnItalic = $("#btnItalic");
const btnBold = $("#btnBold");
const opencvBadge = $("#opencvBadge");
const eraseMode = $("#eraseMode");
const placeManualBtn = $("#placeManualBtn");

// Controles de efectos de texto (glow y fillOpacity)
const glowToggle = $("#glowToggle");
const glowColor = $("#glowColor");
const glowBlur = $("#glowBlur");
const glowBlurValue = $("#glowBlurValue");
const fillOpacity = $("#fillOpacity");
const fillOpacityValue = $("#fillOpacityValue");

// ESTADO GLOBAL DE LA APLICACIÓN
const state = {
  kind: null,            // 'pdf' o 'image'
  pdf: null,             // Instancia de pdf.js
  image: null,           // Elemento de imagen
  page: 1,               // Página actual
  pageCount: 0,          // Total de páginas
  scale: 1.8,            // Escala de renderizado PDF para calidad OCR
  boxesByPage: new Map(),// Número de página -> array de cajas de texto
  selectedId: null,      // ID de la caja seleccionada
  mode: "draw",          // 'draw' o 'move'
  draft: null,           // Parámetros del trazo o arrastre activo
  cvLoaded: false,       // Estado de carga de OpenCV.js
  italic: true,          // Estilo de fuente itálica por defecto
  bold: true,            // Estilo de fuente negrita por defecto
  inpaintedBgByPage: new Map(), // Caché de imágenes de fondo limpias por página
  abortTranslation: false, // Bandera para cancelar traducción automática
  theme: "dark",         // Tema actual: 'dark' o 'light'
};

// 1. CONTROL DE OPENCV.JS (CARGA ASÍNCRONA CON CALLBACK)
function initOpenCv() {
  function onOpenCvReady() {
    if (state.cvLoaded) return; // evitar duplicados
    state.cvLoaded = true;
    opencvBadge.textContent = "OpenCV Activo";
    opencvBadge.className = "badge active";
    setStatus("OpenCV cargado. Borrado inteligente disponible.");
    console.log("[OpenCV] Cargado exitosamente");
  }

  function onOpenCvTimeout() {
    if (!state.cvLoaded) {
      opencvBadge.textContent = "OpenCV Inactivo";
      opencvBadge.className = "badge failed";
      setStatus("OpenCV no cargó en 15s. Se usará borrado básico por color promedio.");
      console.warn("[OpenCV] Timeout de carga (15s)");
    }
  }

  // Caso 1: OpenCV ya está completamente cargado
  if (window.cv && window.cv.Mat) {
    onOpenCvReady();
    return; // ← No crear timeouts innecesarios
  }

  // Caso 2: cv existe pero aún no inicializó runtime
  if (window.cv) {
    window.cv['onRuntimeInitialized'] = onOpenCvReady;
    setTimeout(onOpenCvTimeout, window.__CLIENT_CONFIG.TIMEOUT_OPENCV_INIT_MS);
    return;
  }

  // Caso 3: cv no existe aún (script async cargando)
  // Polling breve hasta que aparezca window.cv
  const checkInterval = setInterval(() => {
    if (window.cv) {
      clearInterval(checkInterval);
      if (window.cv.Mat) {
        onOpenCvReady();
      } else {
        window.cv['onRuntimeInitialized'] = onOpenCvReady;
      }
    }
  }, 200);

  // Timeout de seguridad solo para este caso
  const safetyTimer = setTimeout(() => {
    clearInterval(checkInterval);
    onOpenCvTimeout();
  }, window.__CLIENT_CONFIG.TIMEOUT_OPENCV_INIT_MS);

  // Limpiar timer si cv carga antes del timeout
  const origReady = onOpenCvReady;
  onOpenCvReady = function() {
    clearTimeout(safetyTimer);
    origReady();
  };
}
initOpenCv();

// =============================================================================
// TEMA OSCURO/CLARO - Toggle
// =============================================================================
initTheme(state, (theme) => {
  showToast(theme === "dark" ? "Tema oscuro activado" : "Tema claro activado", "info");
});

// =============================================================================
// TOAST NOTIFICATIONS SYSTEM (imported from ./js/toast.js)
// =============================================================================
// showToast() is now imported from "./js/toast.js" at the top of this file

// =============================================================================
// KEYBOARD SHORTCUTS
// =============================================================================
function initKeyboardShortcuts() {
  if (window.__kbInit) return;
  window.__kbInit = true;
  document.addEventListener("keydown", (e) => {
    // Ignorar si estamos en un input/textarea
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) {
      // Pero permitir Escape para cerrar modales/limpiar selección
      if (e.key === "Escape") {
        e.preventDefault();
        if (state.selectedId) {
          state.selectedId = null;
          sourceText.value = "";
          translatedText.value = "";
          renderBoxes();
        }
      }
      return;
    }
    
    const isCtrl = e.ctrlKey || e.metaKey;
    const isShift = e.shiftKey;
    
    switch (e.key.toLowerCase()) {
      case "d":
        if (!isCtrl) {
          e.preventDefault();
          state.mode = "draw";
          drawModeBtn.classList.add("active");
          moveModeBtn.classList.remove("active");
          overlay.className = "overlay drawing";
          showToast("Modo: Dibujar burbuja", "info", 1500);
        }
        break;
        
      case "v":
        if (!isCtrl) {
          e.preventDefault();
          state.mode = "move";
          moveModeBtn.classList.add("active");
          drawModeBtn.classList.remove("active");
          overlay.className = "overlay";
          showToast("Modo: Mover/Editar", "info", 1500);
        }
        break;
        
      case "t":
        if (isCtrl) {
          e.preventDefault();
          if (isShift) {
            autoTranslateAllPages().catch(err => setStatus(`Error: ${err.message}`));
          } else if (state.selectedId) {
            translateBtn.click();
          } else {
            autoTranslateCurrentPage().catch(err => setStatus(`Error: ${err.message}`));
          }
        }
        break;

      case "e":
        if (isCtrl) {
          e.preventDefault();
          exportCurrentPng().catch(err => setStatus(`Error: ${err.message}`));
        }
        break;
        
      case "p":
        if (isCtrl && isShift) {
          e.preventDefault();
          exportCurrentPdf().catch(err => setStatus(`Error: ${err.message}`));
        }
        break;
        
      case "s":
        if (isCtrl && isShift) {
          e.preventDefault();
          exportFullPdf().catch(err => setStatus(`Error: ${err.message}`));
        }
        break;
        
      case "f":
        if (isCtrl) {
          e.preventDefault();
          fitPageToStage();
        }
        break;
        
      case "arrowleft":
        if (!isCtrl) {
          e.preventDefault();
          if (state.kind === "pdf" && state.page > 1) {
            renderPage(state.page - 1).catch(err => setStatus(err.message));
          }
        }
        break;
        
      case "arrowright":
        if (!isCtrl) {
          e.preventDefault();
          if (state.kind === "pdf" && state.page < state.pageCount) {
            renderPage(state.page + 1).catch(err => setStatus(err.message));
          }
        }
        break;
        
      case "delete":
      case "backspace":
        if (state.selectedId) {
          e.preventDefault();
          const boxes = getPageBoxes();
          const index = boxes.findIndex(b => b.id === state.selectedId);
          if (index >= 0) {
            boxes.splice(index, 1);
            state.selectedId = null;
            sourceText.value = "";
            translatedText.value = "";
            updateErasedBg().then(() => refreshScreenCanvas());
            showToast("Burbuja eliminada", "success", 2000);
          }
        }
        break;
        
      case "n":
        if (isCtrl) {
          e.preventDefault();
          placeManualBtn.click();
        }
        break;

      case "i":
        if (isCtrl) {
          e.preventDefault();
          btnItalic.click();
        }
        break;
        
      case "b":
        if (isCtrl) {
          e.preventDefault();
          btnBold.click();
        }
        break;

      case "g":
        if (!isCtrl && state.selectedId) {
          e.preventDefault();
          glowToggle.checked = !glowToggle.checked;
          glowToggle.dispatchEvent(new Event("change"));
          showToast(
            glowToggle.checked ? "Glow activado" : "Glow desactivado",
            "info", 1500
          );
        }
        break;
        
      case "escape":
        if (state.selectedId) {
          state.selectedId = null;
          sourceText.value = "";
          translatedText.value = "";
          renderBoxes();
          showToast("Selección cancelada", "info", 1500);
        }
        break;
    }
  });
}

// Inicializar atajos de teclado
initKeyboardShortcuts();
// Configuración de pdf.js worker
let pdfjsPromise = null;

// Carga un script UMD y resuelve con window.pdfjsLib, o rechaza en ~10s
function _loadPdfJsUmd(name, scriptUrl, workerUrl) {
  return new Promise((resolve, reject) => {
    console.log(`[PDF.js] Intentando UMD ${name}: ${scriptUrl}`);
    const timer = setTimeout(
      () => reject(new Error(`Timeout UMD ${name} (10s)`)),
      window.__CLIENT_CONFIG.TIMEOUT_PDFJS_CDN_MS
    );
    const script = document.createElement("script");
    script.src = scriptUrl;
    script.onload = () => {
      clearTimeout(timer);
      if (window.pdfjsLib && typeof window.pdfjsLib.getDocument === "function") {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
        console.log(`[PDF.js] UMD ${name} OK`);
        resolve(window.pdfjsLib);
      } else {
        reject(new Error(`UMD ${name}: pdfjsLib no está en window`));
      }
    };
    script.onerror = () => {
      clearTimeout(timer);
      reject(new Error(`No se pudo descargar UMD ${name}`));
    };
    script.crossOrigin = "anonymous";
    document.head.appendChild(script);
  });
}

// Carga PDF.js desde 4 CDNs en PARALELO — gana el más rápido via Promise.any()
async function loadPdfJs() {
  if (pdfjsPromise) return pdfjsPromise;

  pdfjsPromise = (async () => {
    setStatus("Cargando PDF.js (4 CDNs en paralelo)...");

    const attempts = [
      // Estrategia 1: ES Module .mjs (más rápido, nativo — v4)
      (async () => {
        console.log("[PDF.js] Intentando ES module v4.10.38...");
        const importPromise = import("https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs");
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error("Timeout ES module (10s)")), window.__CLIENT_CONFIG.TIMEOUT_PDFJS_ES_MODULE_MS)
        );
        const pdfjsLib = await Promise.race([importPromise, timeoutPromise]);
        pdfjsLib.GlobalWorkerOptions.workerSrc =
          "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";
        if (typeof pdfjsLib.getDocument !== "function") {
          throw new Error("ES module: getDocument no es función");
        }
        console.log("[PDF.js] ES module v4.10.38 OK");
        return pdfjsLib;
      })(),

      // Estrategia 2: UMD clásico desde cdnjs (v3)
      _loadPdfJsUmd(
        "cdnjs",
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js"
      ),

      // Estrategia 3: UMD desde jsDelivr (v3)
      _loadPdfJsUmd(
        "jsDelivr",
        "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js",
        "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js"
      ),

      // Estrategia 4: UMD desde unpkg (v3)
      _loadPdfJsUmd(
        "unpkg",
        "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.min.js",
        "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js"
      ),
    ];

    try {
      const pdfjsLib = await Promise.any(attempts);
      setStatus("PDF.js cargado");
      return pdfjsLib;
    } catch (aggErr) {
      // Promise.any() rechaza con AggregateError si TODOS fallan
      pdfjsPromise = null;  // permitir reintento
      const reasons = (aggErr.errors || []).map(e => e.message).join("; ");
      const msg = `No se pudo cargar PDF.js desde ningún CDN. Errores: ${reasons}`;
      console.error("[PDF.js] Todos los CDNs fallaron:", aggErr);
      setStatus(`Error: ${msg}`);
      throw new Error(msg);
    }
  })();

  return pdfjsPromise;
}

// Actualizar textos de estado
function setStatus(text) {
  statusText.textContent = text;
}

// Retorna las cajas de texto de la página dada
function getPageBoxes(page = state.page) {
  if (!state.boxesByPage.has(page)) {
    state.boxesByPage.set(page, []);
  }
  return state.boxesByPage.get(page);
}

// 2. APERTURA Y CARGA DE ARCHIVOS
async function openFile(file) {
  //console.log("[openFile] LLAMADO con archivo:", file?.name, file?.type, file?.size);
  document.title = "🔄 openFile: " + (file?.name || "none");
  if (!file) return;
  state.pdf = null;
  state.image = null;
  state.kind = null;
  state.pageCount = 0;
  state.boxesByPage.clear();
  state.inpaintedBgByPage.clear();
  state.selectedId = null;
  state.page = 1;
  docName.textContent = file.name;
  emptyState.style.display = "none";
  overlay.innerHTML = "";
  
  // Resetear textarea
  sourceText.value = "";
  translatedText.value = "";

  try {
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      //console.log("[openFile] Detectado como PDF:", file.name, "type:", file.type, "size:", file.size);
      setStatus("Cargando PDF...");
      showToast("Cargando PDF.js...", "info", 10000);
      const pdfjs = await loadPdfJs();
      showToast("PDF.js listo, parseando archivo...", "info", 10000);
      const buffer = await file.arrayBuffer();
      state.pdf = await pdfjs.getDocument({ data: buffer }).promise;
      state.kind = "pdf";
      state.pageCount = state.pdf.numPages;
      pageTotal.textContent = `/ ${state.pageCount}`;
      pageNumber.max = String(state.pageCount);
      pageNumber.value = "1";
      await renderPage(1);
      showToast(`PDF cargado: ${state.pageCount} páginas`, "success", 3000);
      document.title = "✅ PDF: " + file.name;
      
      // Auto traducción opcional al cargar - esperar a que renderPage termine
      if ($("#autoTranslateOnLoad")?.checked) {
        await autoTranslateAllPages();
      }
    } else if (file.type.startsWith("image/")) {
      setStatus("Cargando Imagen...");
      const img = new Image();
      img.src = URL.createObjectURL(file);
      await img.decode();
      URL.revokeObjectURL(img.src);
      state.image = img;
      state.kind = "image";
      state.pageCount = 1;
      pageTotal.textContent = "/ 1";
      pageNumber.max = "1";
      pageNumber.value = "1";
      await renderImage();
      showToast("Imagen cargada", "success", 3000);
      document.title = "✅ Imagen: " + file.name;
      
      // Auto traducción opcional al cargar - esperar a que renderImage termine
      if ($("#autoTranslateOnLoad")?.checked) {
        await autoTranslateCurrentPage();
      }
    } else {
      setStatus("Formato no compatible. Carga un PDF o imagen.");
    }
  } catch (error) {
    console.error("[openFile] Error:", error);
    console.error("[openFile] Error stack:", error?.stack);
    showToast(`Error: ${error?.message || error || "Desconocido"}`, "error", 8000);
    setStatus(`Error al abrir archivo: ${error?.message || error || "Desconocido"}`);
    document.title = "❌ Error: " + (error?.message || error);
  }
}

// Renderizar una página de PDF
let _renderToken = null;
let _renderTempCanvas = null;
async function renderPage(page = state.page) {
  if (state.kind === "image") return renderImage();
  if (!state.pdf) return {aborted: true};
  const myToken = {};
  _renderToken = myToken;
  state.page = page;
  setStatus(`Renderizando página ${page}...`);
  
  try {
    //console.log("[renderPage] Obteniendo página", page, "pdf:", !!state.pdf);
    const pdfPage = await state.pdf.getPage(page);
    if (_renderToken !== myToken) return {aborted: true};
    //console.log("[renderPage] Página obtenida, viewport...");
    const viewport = pdfPage.getViewport({ scale: state.scale });
    
    resizeStage(viewport.width, viewport.height);
    
    // Usar un canvas temporal para evitar el error "multiple render() operations"
    if (!_renderTempCanvas) _renderTempCanvas = document.createElement("canvas");
    const tempCanvas = _renderTempCanvas;
    tempCanvas.width = viewport.width;
    tempCanvas.height = viewport.height;
    const tempCtx = tempCanvas.getContext("2d");
    
    const renderTask = pdfPage.render({ canvasContext: tempCtx, viewport });
    const renderPromise = renderTask.promise;
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error(`Timeout renderizando página (${window.__CLIENT_CONFIG.TIMEOUT_PDF_RENDER_MS/1000}s)`)), window.__CLIENT_CONFIG.TIMEOUT_PDF_RENDER_MS)
    );
    try {
      await Promise.race([renderPromise, timeoutPromise]);
    } catch (e) {
      renderTask.cancel();
      throw e;
    }
    if (_renderToken !== myToken) return {aborted: true};
    
    // Copiar del temporal al cleanBgCanvas
    const cleanCtx = cleanBgCanvas.getContext("2d");
    cleanCtx.clearRect(0, 0, cleanBgCanvas.width, cleanBgCanvas.height);
    cleanCtx.drawImage(tempCanvas, 0, 0);
    
    pageNumber.value = String(page);
    await updateErasedBg();
    if (_renderToken !== myToken) return {aborted: true};
    await refreshScreenCanvas();
    renderBlockList();
    setStatus(`Página ${page} de ${state.pageCount} lista.`);
    return {aborted: false};
  } catch (error) {
    console.error("Error rendering page:", error);
    setStatus(`Error al renderizar página ${page}: ${error.message}`);
    throw error;
  }
}

// Renderizar una imagen
async function renderImage() {
  if (!state.image) return;
  const maxWidth = 1200;
  const scale = Math.min(1, maxWidth / state.image.naturalWidth);
  const width = Math.round(state.image.naturalWidth * scale);
  const height = Math.round(state.image.naturalHeight * scale);
  
  resizeStage(width, height);
  
  // Dibujar la imagen limpia en el canvas oculto
  const cleanCtx = cleanBgCanvas.getContext("2d");
  cleanCtx.drawImage(state.image, 0, 0, width, height);
  
  await updateErasedBg();
  await refreshScreenCanvas();
  setStatus("Imagen lista.");
}

// Actualizar el lienzo con el fondo borrado por OpenCV o color muestreado (Caché estable)
async function updateErasedBg() {
  try {
    if (!state.kind) return;
    const ctx = erasedBgCanvas.getContext("2d");

    // Si tenemos una imagen de fondo limpia cacheada del servidor para esta página, usarla de base
    const inpaintedImg = state.inpaintedBgByPage?.get(state.page);
    const hasServerInpainted = inpaintedImg?.complete && inpaintedImg.naturalWidth > 0;
    const useServerInpainted = hasServerInpainted && coverOriginal.checked;

    if (useServerInpainted) {
      ctx.drawImage(state.inpaintedBgByPage.get(state.page), 0, 0);
    } else {
      ctx.drawImage(cleanBgCanvas, 0, 0);
    }

    const boxes = getPageBoxes();
    // Solo hacer borrado local si NO tenemos imagen inpaintada del servidor
    // (el servidor ya hizo el inpainting, no hacer doble borrado)
    if (coverOriginal.checked && boxes.length > 0 && !hasServerInpainted) {
      // Intentar borrado avanzado OpenCV
      const ok = await eraseWithInpainting(erasedBgCanvas, boxes);
      if (!ok) {
        // Fallback a borrado básico de color muestreado
        for (const box of boxes) {
          fallbackEraseBox(ctx, box);
        }
      }
    }
  } catch (error) {
    console.error("Error in updateErasedBg:", error);
    setStatus(`Error al borrar fondo: ${error.message}`);
  }
}

// Actualizar el lienzo de pantalla (copia el fondo borrado sin texto y dibuja el nuevo texto encima)
async function refreshScreenCanvas() {
  try {
    if (!state.kind) return;
    const ctx = pdfCanvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(erasedBgCanvas, 0, 0);
    renderBoxes();
  } catch (error) {
    console.error("Error in refreshScreenCanvas:", error);
    setStatus(`Error en pantalla: ${error.message}`);
  }
}

// Cambiar tamaño del lienzo y el overlay
function resizeStage(width, height) {
  pdfCanvas.width = width;
  pdfCanvas.height = height;
  cleanBgCanvas.width = width;
  cleanBgCanvas.height = height;
  erasedBgCanvas.width = width;
  erasedBgCanvas.height = height;
  stage.style.width = `${width}px`;
  stage.style.height = `${height}px`;
  stage.style.minWidth = `${width}px`;
  stage.style.minHeight = `${height}px`;
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
}

// 3. LOGICA DEL MOTOR DE TRADUCCIÓN (API PYTHON CON FALLBACK)
async function translateOnline(text, langDest) {
  const trimmed = text.trim();
  if (!trimmed) return "";
  
  try {
    const controller = new AbortController();    const timeoutId = setTimeout(() => controller.abort(), window.__CLIENT_CONFIG.TIMEOUT_TRANSLATE_MS);

    const response = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: trimmed, target: langDest, source: "auto" }),
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      try {
        const data = await response.json();
        if (data && data.translatedText) return data.translatedText;
      } catch (e) { /* invalid JSON response */ }
    }
  } catch (error) {
    console.warn("Error en traducción online", error);
  }
  
  // Sin diccionario de respaldo: devolver el texto original
  return trimmed;
}

async function translateBatch(texts, langDest) {
  const trimmed = texts.map(t => (t || "").trim());
  let srcLang = sourceLang.value;
  // Si contiene el signo + (Occidentales), le pasamos "auto" para que el servidor detecte el idioma exacto
  if (srcLang.includes("+")) {
    srcLang = "auto";
  }
  try {
    const controller = new AbortController();    const timeoutId = setTimeout(() => controller.abort(), window.__CLIENT_CONFIG.TIMEOUT_TRANSLATE_BATCH_MS);

    const response = await fetch("/api/translate-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: trimmed, target: langDest, source: srcLang }),
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      try {
        const data = await response.json();
        if (Array.isArray(data.results)) return data.results;
      } catch (e) { /* invalid JSON response */ }
    }
  } catch (error) {
    console.warn("Error en traducción batch, usando traducciones individuales", error);
  }
  // Fallback: traducir uno a uno si el batch falla
  return Promise.all(trimmed.map(t => translateOnline(t, langDest)));
}

// 4. LÓGICA DE OCR (SERVIDOR EASYOCR - Tesseract.js eliminado)

// Detección nativa en PDFs de texto no escaneados (Mucho más rápido y preciso)
async function detectPdfTextLines(pageNo) {
  if (state.kind !== "pdf" || !state.pdf) return [];
  const page = await state.pdf.getPage(pageNo);
  const viewport = page.getViewport({ scale: state.scale });
  const content = await page.getTextContent();
  
  const lines = [];
  for (const item of content.items || []) {
    const text = item.str.trim();
    if (!text || text.length < 2) continue;
    
    const tr = item.transform || [1, 0, 0, 1, 0, 0];
    const size = Math.max(8, Math.abs(tr[3]) * state.scale);
    const x = tr[4] * state.scale;
    const y = viewport.height - tr[5] * state.scale - size;
    const w = Math.max(20, (item.width || text.length * size * 0.55) * state.scale);
    
    lines.push({
      text,
      x,
      y,
      w,
      h: size * 1.25,
      size: Math.round(size)
    });
  }

  // Agrupar líneas en bloques de burbuja
  return mergeLinesIntoBlocks(lines);
}

// mergeLinesIntoBlocks, filterPageBlocks, MARGIN_NOISE_PATTERNS, GLOBAL_NOISE_PATTERNS, getBlockText
// are imported from ./js/utils.js and ./js/filters.js at the top of this file

// Analizar la página actual: solo usa bloques nativos del PDF (sin OCR local)
// El OCR real lo hace el servidor (EasyOCR) vía botón "Traducir Página Actual"
async function detectPageBlocks(pageNo = state.page) {
  // Solo bloques de texto nativo del PDF
  const pdfBlocks = await detectPdfTextLines(pageNo);
  return filterPageBlocks(pdfBlocks, pdfCanvas.height);
}

function isLikelyBubble(rect) {
  try {
    const ctx = cleanBgCanvas.getContext("2d", { willReadFrequently: true });
    const x = Math.max(0, Math.floor(rect.x)), y = Math.max(0, Math.floor(rect.y));
    const w = Math.max(1, Math.min(cleanBgCanvas.width - x, Math.ceil(rect.w)));
    const h = Math.max(1, Math.min(cleanBgCanvas.height - y, Math.ceil(rect.h)));
    const data = ctx.getImageData(x, y, w, h).data;
    
    // Muestrear únicamente el perímetro exterior (15% del borde)
    // Esto evita que letras gruesas blancas/doradas en el centro falseen el brillo de fondo
    let sum = 0, count = 0;
    const edgeW = Math.max(1, Math.round(w * 0.15));
    const edgeH = Math.max(1, Math.round(h * 0.15));
    
    for (let py = 0; py < h; py += 2) {
      for (let px = 0; px < w; px += 2) {
        const isBorder = px < edgeW || py < edgeH || px > w - edgeW || py > h - edgeH;
        if (!isBorder) continue;
        
        const i = (py * w + px) * 4;
        sum += data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114;
        count++;
      }
    }
    return count ? (sum / count) > 180 : false;
  } catch (e) {
    return false;
  }
}

// Muestrear el color de fondo FUERA del bloque (para borrar el texto original correctamente)
function sampleBgColorAround(block) {
  try {
    const ctx = cleanBgCanvas.getContext("2d", { willReadFrequently: true });
    const margin = Math.max(8, Math.round(Math.min(block.w, block.h) * 0.25));
    const sx = Math.max(0, Math.floor(block.x) - margin);
    const sy = Math.max(0, Math.floor(block.y) - margin);
    const sx2 = Math.min(cleanBgCanvas.width,  Math.ceil(block.x + block.w) + margin);
    const sy2 = Math.min(cleanBgCanvas.height, Math.ceil(block.y + block.h) + margin);
    const sw = sx2 - sx, sh = sy2 - sy;
    if (sw <= 0 || sh <= 0) return "#ffffff";
    const data = ctx.getImageData(sx, sy, sw, sh).data;

    // Solo pixels FUERA del rectángulo interior (el propio bloque)
    const ix = Math.floor(block.x) - sx;
    const iy = Math.floor(block.y) - sy;
    const iw = Math.ceil(block.w);
    const ih = Math.ceil(block.h);

    const buckets = {};
    for (let py = 0; py < sh; py += 2) {
      for (let px = 0; px < sw; px += 2) {
        if (px >= ix && px < ix + iw && py >= iy && py < iy + ih) continue; // dentro del bloque
        const i = (py * sw + px) * 4;
        const r = Math.round(data[i]   / 32) * 32;
        const g = Math.round(data[i+1] / 32) * 32;
        const b = Math.round(data[i+2] / 32) * 32;
        const key = `${r},${g},${b}`;
        buckets[key] = (buckets[key] || 0) + 1;
      }
    }
    let bestKey = null, bestVal = 0;
    for (const [k, v] of Object.entries(buckets)) {
      if (v > bestVal) { bestVal = v; bestKey = k; }
    }
    if (bestKey) {
      const [r, g, b] = bestKey.split(",").map(Number);
      return `rgb(${r},${g},${b})`;
    }
    return "#ffffff";
  } catch (e) { return "#ffffff"; }
}
function sampleTextColor(block) {
  try {
    const ctx = cleanBgCanvas.getContext("2d", { willReadFrequently: true });
    const x = Math.max(0, Math.floor(block.x));
    const y = Math.max(0, Math.floor(block.y));
    const w = Math.max(1, Math.min(cleanBgCanvas.width - x, Math.ceil(block.w)));
    const h = Math.max(1, Math.min(cleanBgCanvas.height - y, Math.ceil(block.h)));
    const data = ctx.getImageData(x, y, w, h).data;

    // Calcular brillo promedio del fondo (borde exterior 10%)
    let bgBrightness = 0, bgCount = 0;
    const edge = Math.max(2, Math.round(Math.min(w, h) * 0.1));
    for (let py = 0; py < h; py += 2) {
      for (let px = 0; px < w; px += 2) {
        const isBorder = px < edge || py < edge || px > w - edge || py > h - edge;
        if (!isBorder) continue;
        const i = (py * w + px) * 4;
        bgBrightness += data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114;
        bgCount++;
      }
    }
    bgBrightness = bgCount > 0 ? bgBrightness / bgCount : 128;

    // Acumular colores del centro (zona de texto)
    const colorBuckets = {};
    for (let py = edge; py < h - edge; py += 1) {
      for (let px = edge; px < w - edge; px += 1) {
        const i = (py * w + px) * 4;
        const r = data[i], g = data[i+1], b = data[i+2];
        const bright = r * 0.299 + g * 0.587 + b * 0.114;
        // Si el pixel contrasta con el fondo, es probable texto
        const contraste = Math.abs(bright - bgBrightness);
        if (contraste < 40) continue;
        // Cuantizar en 32 niveles para agrupar colores similares
        const key = `${Math.round(r/32)*32},${Math.round(g/32)*32},${Math.round(b/32)*32}`;
        colorBuckets[key] = (colorBuckets[key] || 0) + contraste;
      }
    }
    // Encontrar el color de texto más dominante
    let bestKey = null, bestVal = 0;
    for (const [k, v] of Object.entries(colorBuckets)) {
      if (v > bestVal) { bestVal = v; bestKey = k; }
    }
    if (bestKey) {
      const [r, g, b] = bestKey.split(",").map(Number);
      return `rgb(${r},${g},${b})`;
    }
    // Fallback según brillo de fondo
    return bgBrightness > 128 ? "#000000" : "#ffffff";
  } catch (e) {
    return "#000000";
  }
}

// isLightColor is imported from ./js/utils.js at the top of this file

function makeAutoTextBox(block, translated = "", serverData = null) {
  // Si tenemos datos del servidor (inpainting + OCR profesional), usarlos directamente
  const isBubble = isLikelyBubble(block);
  const padX = isBubble ? Math.max(4, block.h * 0.12) : 3;
  const padY = isBubble ? Math.max(3, block.h * 0.08) : 3;

  // Tamaño de fuente: del servidor si existe, sino estimado
  const estFontSize = serverData?.fontSize
    || Math.max(10, Math.min(48, Math.round(block.size || block.h * 0.8)));

  // CON DATOS DEL SERVIDOR: elegir fuente según el estilo del texto
  // Si el texto original está en MAYÚSCULAS o tiene tono agresivo (!¡?¿),
  // usar fuente bold/impact para preservar la expresividad.
  // Para texto normal, usar fuente manga limpia.
  const useServerFont = !!serverData;
  const serverFontSize = estFontSize;
  let finalFontFamily, finalFontStyle, finalFontWeight;

  // Detectar estilo expresivo: mayúsculas, tono agresivo (signos !¡?¿)
  const rawText = block.source || block.text || "";
  const hasUpper = rawText === rawText.toUpperCase() && rawText.length > 2;
  const hasAggressive = /[!¡?¿]/.test(rawText) || /[A-Z]{4,}/.test(rawText);
  const isExpressive = hasUpper || hasAggressive;

  if (useServerFont) {
    if (isExpressive) {
      finalFontFamily = "'Impact', 'Arial Black', 'Franklin Gothic Heavy', sans-serif";
      finalFontStyle = "normal";
      finalFontWeight = "900";
    } else {
      finalFontFamily = "'Comic Sans MS', 'Trebuchet MS', sans-serif";
      finalFontStyle = "normal";
      finalFontWeight = "400";
    }
  } else {
    finalFontFamily = fontFamily.value;
    finalFontStyle = state.italic ? "italic" : "normal";
    finalFontWeight = state.bold ? "700" : "400";
  }

  // Colores: preferir datos del servidor si existen
  const textCol = serverData?.textColor || sampleTextColor(block);
  let bgCol     = serverData?.bgColor   || sampleBgColorAround(block);

  // Si el servidor detectó un globo de diálogo (fondo muy oscuro, brillo < 60),
  // usar bg transparente para que el canvas inpainted (que preserva el globo)
  // se vea a través, en vez de pintar un rectángulo opaco que destruye el arte.
  if (serverData && bgCol) {
    const m = bgCol.match(/\d+/g);
    if (m) {
      const [r, g, b] = m.map(Number);
      const brightness = r * 0.299 + g * 0.587 + b * 0.114;
      if (brightness < 60) {
        bgCol = "transparent";
      }
    }
  }

  // Para texto flotante sobre arte, agregar contorno de contraste para legibilidad
  // Si serverData existe y tiene bgColor oscuro (globo de diálogo), no usar contorno
  const isServerBubble = serverData?.bgColor && (() => {
    const m = serverData.bgColor.match(/\d+/g);
    if (m) { const [r,g,b] = m.map(Number); return (r*0.299 + g*0.587 + b*0.114) < 80; }
    return false;
  })();
  // Glow exterior (brillo tipo neón) — controlado por UI
  // Lee el estado actual de los controles de efectos de texto
  const useGlow = !isBubble || (isExpressive && hasAggressive);
  glowToggle.checked = useGlow;
  let glowColorResult = "transparent";
  let glowBlurResult = 0;
  let fillOpacityResult = 0;
  
  if (glowToggle.checked) {
    glowColorResult = glowColor.value;
    glowBlurResult = Math.max(1, Number(glowBlur.value) || 12);
  }
  fillOpacityResult = (!isBubble && bgCol !== "transparent") ? (Number(fillOpacity.value) / 100) || 0.35 : 0;

  const strokeC  = (isBubble || isServerBubble) ? "transparent" : (isLightColor(textCol) ? "#000000" : "#ffffff");
  const strokeW  = isBubble ? 0 : 2;

  return {
    id: crypto.randomUUID(),
    x: Math.max(0, block.x - padX),
    y: Math.max(0, block.y - padY),
    w: Math.min(pdfCanvas.width - (block.x - padX), block.w + padX * 2),
    h: Math.min(pdfCanvas.height - (block.y - padY), block.h + padY * 2),
    source: block.text,
    text: translated || block.text,
    bg: bgCol,
    color: textCol,
    strokeColor: strokeC,
    strokeWidth: strokeW,
    glowColor: glowColorResult,
    glowBlur: glowBlurResult,
    fillOpacity: fillOpacityResult,
    fontSize: serverFontSize,
    fontFamily: finalFontFamily,
    fontStyle: finalFontStyle,
    fontWeight: finalFontWeight,
    eraseMode: serverData ? "none" : "area",
    _serverInpainted: !!serverData,
    confidence: block.confidence || 0,
    shadow: true
  };
}

// Barra de progreso y estimación de ETA
let progressContainer = null;
function showProgress(label, done, total, startedAt) {
  if (!progressContainer) {
    progressContainer = document.createElement("div");
    progressContainer.className = "progress-info";
    progressContainer.innerHTML = `
      <div style="font-size:11px; margin-bottom:4px; display:flex; justify-content:space-between; align-items:center;">
        <span class="progress-lbl"></span>
        <span class="progress-pct">0%</span>
        <button class="progress-cancel" style="font-size:10px; padding:2px 6px; background:var(--danger); color:white; border:none; border-radius:3px; cursor:pointer;">Cancelar</button>
      </div>
      <div class="progress-bar-container"><div class="progress-bar-fill"></div></div>
      <div class="progress-eta" style="font-size:10px; color:var(--text-secondary); margin-top:4px;"></div>
    `;
    $(".auto-controls").appendChild(progressContainer);
    progressContainer.querySelector(".progress-cancel").addEventListener("click", () => {
      state.abortTranslation = true;
      if (progressContainer) {
        progressContainer.remove();
        progressContainer = null;
      }
      setStatus("Cancelando traducción...");
    });
  }
  
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  progressContainer.querySelector(".progress-lbl").textContent = label;
  progressContainer.querySelector(".progress-pct").textContent = `${percent}% (${done}/${total})`;
  progressContainer.querySelector(".progress-bar-fill").style.width = `${percent}%`;
  
  const elapsed = Date.now() - startedAt;
  const eta = done > 0 ? Math.max(0, Math.round((elapsed / done) * (total - done))) : 0;
  const etaText = done > 0 ? `ETA: aprox. ${formatDuration(eta)}` : "Calculando tiempo...";
  progressContainer.querySelector(".progress-eta").textContent = done === total ? "¡Completado!" : etaText;
  
  if (done === total) {
    setTimeout(() => {
      if (progressContainer) {
        progressContainer.remove();
        progressContainer = null;
      }
    }, 4000);
  }
}

// formatDuration, canvasToBase64, loadBase64IntoCanvas are imported from ./js/utils.js

// Procesar página completa en el servidor (OCR + inpainting + traducción)
async function serverProcessPage(pageNo = state.page) {
  const pageCanvas = document.createElement("canvas");
  pageCanvas.width = cleanBgCanvas.width;
  pageCanvas.height = cleanBgCanvas.height;
  pageCanvas.getContext("2d").drawImage(cleanBgCanvas, 0, 0);
  const imageB64 = canvasToBase64(pageCanvas);

  const payload = {
    image: imageB64,
    target: targetLang.value,
    source: sourceLang.value || "auto",
  };

  //console.log("[serverProcessPage] Enviando a servidor:", { 
  //  target: payload.target, 
  //  source: payload.source, 
  //  imgSize: imageB64.length 
  //});

  // Fetch con timeout de 120 segundos
  const controller = new AbortController();  const timeoutId = setTimeout(() => controller.abort(), window.__CLIENT_CONFIG.TIMEOUT_PROCESS_PAGE_MS);

  const resp = await fetch("/api/process-page", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: controller.signal
  });
  
  clearTimeout(timeoutId);

  //console.log("[serverProcessPage] Respuesta status:", resp.status);

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    console.error("[serverProcessPage] Error:", err);
    throw new Error(err.error || `Error del servidor: ${resp.status}`);
  }

  let result;
  try { result = await resp.json(); } catch (e) { throw new Error("Respuesta inválida del servidor"); }
  //console.log("[serverProcessPage] Bloques recibidos:", result.blocks?.length || 0);
  return result; // { inpainted_image, blocks }
}

async function autoTranslateCurrentPage(pageNo = state.page, startedAt = Date.now(), progressIndex = 1, totalProgress = 1) {
  // Solo resetear la bandera si se llama individualmente (no en bucle desde autoTranslateAllPages)
  if (totalProgress === 1 && progressIndex === 1) {
    state.abortTranslation = false;
  }
  setStatus(`Procesando página ${pageNo} en el servidor...`);
  showProgress(`Traduciendo Pág. ${pageNo}`, progressIndex - 1, totalProgress, startedAt);

  try {
    // ── Camino ÚNICO: Usar el servidor para OCR + inpainting + traducción ──────
    let serverResult = null;
    try {
      serverResult = await serverProcessPage(pageNo);
    } catch (serverErr) {
      console.error("[server-process] Error:", serverErr.message);
      throw new Error("Servidor no disponible: " + serverErr.message);
    }

    if (state.abortTranslation) return 0;
    
    if (!serverResult || !serverResult.blocks || serverResult.blocks.length === 0) {
      showProgress(`Traduciendo Pág. ${pageNo}`, progressIndex, totalProgress, startedAt);
      setStatus(`Página ${pageNo}: sin texto detectado.`);
      return 0;
    }

    // El servidor devuelve imagen inpainted + bloques ya traducidos
    showProgress(`Traduciendo Pág. ${pageNo}`, progressIndex - 0.3, totalProgress, startedAt);

    // Guardar imagen inpainted en caché
    const inpaintedImg = new Image();
    inpaintedImg.src = serverResult.inpainted_image;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Timeout cargando imagen inpainted")), window.__CLIENT_CONFIG.TIMEOUT_INPAINTED_IMAGE_MS);
      inpaintedImg.onload = () => { clearTimeout(timeout); resolve(); };
      inpaintedImg.onerror = () => { clearTimeout(timeout); reject(new Error("Error decodificando imagen inpainted")); };
    });
    state.inpaintedBgByPage.set(pageNo, inpaintedImg);

    // Cargar imagen inpainted en erasedBgCanvas
    await loadBase64IntoCanvas(serverResult.inpainted_image, erasedBgCanvas);
    serverResult.inpainted_image = null; // liberar memoria base64

    // Filtrar bloques del servidor (márgenes, marcas de agua)
    const filteredServerBlocks = filterPageBlocks(serverResult.blocks, pdfCanvas.height);

    // Construir cajas de burbuja CON DATOS DEL SERVIDOR (posición, tamaño, colores, fuente)
    const boxes = filteredServerBlocks.map(b => {
      const block = { x: b.x, y: b.y, w: b.w, h: b.h, text: b.source, size: b.fontSize };
      return makeAutoTextBox({ ...block, confidence: b.confidence }, b.translated, {
        textColor: b.textColor,
        bgColor: b.bgColor,
        fontSize: b.fontSize,
      });
    });

    state.boxesByPage.set(pageNo, boxes);
    if (pageNo === state.page) {
      const ctx = pdfCanvas.getContext("2d");
      ctx.drawImage(erasedBgCanvas, 0, 0);
      renderBoxes();
      if (boxes.length > 0) renderBlockList();
    }

    showProgress(`Traduciendo Pág. ${pageNo}`, progressIndex, totalProgress, startedAt);
    setStatus(`Página ${pageNo} traducida: ${boxes.length} bloques.`);
    return boxes.length;

  } catch (error) {
    console.error("[autoTranslate] Error:", error);
    setStatus(`Error traduciendo página ${pageNo}: ${error.message}`);
    return -1;  // -1 = error, 0 = no text, >0 = success
  }
}

// Traducir todo el PDF / archivo completo

// Verificar que el servidor Flask responde antes de iniciar un batch
async function checkServerHealth() {
  try {
    const resp = await fetch('/api/health', { method: 'GET', signal: AbortSignal.timeout(5000) });
    if (resp.ok) {
      const data = await resp.json();
      if (data && data.ok) {
        //console.log('[health] Servidor OK:', data);
        return true;
      }
    }
    console.warn('[health] Servidor respondio con status:', resp.status);
    return false;
  } catch (e) {
    console.error('[health] Error de conexion con el servidor:', e.message);
    return false;
  }
}

async function autoTranslateAllPages() {
  if (!state.kind) return setStatus("Carga un archivo primero.");
  
  // Verificar que el servidor está corriendo antes de empezar
  setStatus("Verificando conexión con el servidor...");
  const serverOk = await checkServerHealth();
  if (!serverOk) {
    const msg = "El servidor Flask no responde. Asegúrate de iniciar server.py primero (http://127.0.0.1:5174).";
    setStatus("Error: " + msg);
    showToast(msg, "error", 10000);
    return;
  }
  
  state.abortTranslation = false;
  const startedAt = Date.now();
  const originalPage = state.page;
  const total = state.kind === "pdf" ? state.pageCount : 1;
  
  let totalBlocks = 0;
  let errorPages = 0;
  let emptyPages = 0;
  let successPages = 0;
  
  for (let p = 1; p <= total; p++) {
    if (state.abortTranslation) {
      setStatus("Traducción cancelada por el usuario.");
      break;
    }
    if (state.kind === "pdf") {
      const result = await renderPage(p);
      if (result?.aborted) continue;
    }
    
    const blocksThisPage = await autoTranslateCurrentPage(p, startedAt, p, total);
    
    if (blocksThisPage > 0) {
      totalBlocks += blocksThisPage;
      successPages++;
    } else if (blocksThisPage === -1) {
      errorPages++;  // -1 = error de servidor
    } else {
      emptyPages++;  // 0 = sin texto detectado en la página
    }
    
    showProgress(`Pág. ${p}`, p, total, startedAt);
  }
  
  if (state.kind === "pdf") {
    await renderPage(originalPage);
  }
  
  if (!state.abortTranslation) {
    let report = `Traducción: ${total} páginas, ${totalBlocks} bloques.`;
    if (successPages > 0) report += ` ${successPages} éxito.`;
    if (emptyPages > 0) report += ` ${emptyPages} sin texto.`;
    if (errorPages > 0) report += ` ${errorPages} con error.`;
    setStatus(report);
    showToast(`Traducción completada: ${totalBlocks} bloques en ${successPages} páginas`, 
              errorPages > 0 ? "warning" : "success", 6000);
  }
}

// ─── BLOQUES DETECTADOS: lista en sidebar + re-traducción individual ───
function renderBlockList() {
  const panel = $("#blockListPanel");
  const container = $("#blockListContainer");
  const badge = $("#blockCountBadge");
  if (!panel || !container) return;

  const boxes = getPageBoxes();
  if (!boxes.length) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "flex";
  panel.setAttribute("data-visible", "true");
  if (badge) badge.textContent = String(boxes.length);

  container.innerHTML = "";
  boxes.forEach((box, idx) => {
    const entry = document.createElement("div");
    entry.className = `block-entry${box.id === state.selectedId ? " selected" : ""}`;
    entry.dataset.boxId = box.id;

    const conf = box.confidence || 0;
    const confClass = conf >= 0.5 ? "high" : conf >= 0.25 ? "medium" : "low";

    entry.innerHTML = `
      <div class="block-header">
        <span class="block-index">#${idx + 1}</span>
        <span class="block-confidence ${confClass}">${(conf * 100).toFixed(0)}%</span>
      </div>
      <div class="block-ocr" title="${escHtml(box.source || '')}">${escHtml(truncate(box.source || '(sin OCR)', 60))}</div>
      <div class="block-translated" title="${escHtml(box.text || '')}">${escHtml(truncate(box.text || '(sin traducción)', 60))}</div>
      <div class="block-actions">
        <button class="block-edit-btn" data-action="edit">✎ Editar OCR</button>
        <button class="block-retranslate-btn" data-action="retranslate">⟳ Retraducir</button>
      </div>
      <div class="block-editor">
        <div class="editor-label">Corregir OCR:</div>
        <textarea data-action="ocr-input" rows="2">${escHtml(box.source || '')}</textarea>
        <div class="editor-label">Traducción:</div>
        <textarea data-action="trans-input" rows="2">${escHtml(box.text || '')}</textarea>
      </div>
    `;

    // Click en la entrada → seleccionar burbuja
    entry.addEventListener("click", (e) => {
      if (e.target.closest("button") || e.target.closest("textarea")) return;
      selectBox(box.id);
      // Sincronizar: resaltar esta entrada en la lista
      container.querySelectorAll(".block-entry").forEach(el => el.classList.remove("selected"));
      entry.classList.add("selected");
    });

    // Botón Editar OCR: toggle editor inline
    entry.querySelector("[data-action='edit']").addEventListener("click", (e) => {
      e.stopPropagation();
      const editor = entry.querySelector(".block-editor");
      editor.classList.toggle("open");
    });

    // Botón Retraducir: envía OCR corregido al servidor
    entry.querySelector("[data-action='retranslate']").addEventListener("click", async (e) => {
      e.stopPropagation();
      const ocrInput = entry.querySelector("[data-action='ocr-input']");
      const correctedOcr = ocrInput.value.trim();
      if (!correctedOcr) return;

      const btn = e.currentTarget;
      btn.disabled = true;
      btn.textContent = "...";

      try {
        setStatus(`Retraduciendo bloque #${idx + 1}...`);
        const translated = await translateOnline(correctedOcr, targetLang.value);

        // Actualizar la burbuja en el state
        const targetBox = getPageBoxes().find(b => b.id === box.id);
        if (targetBox) {
          targetBox.source = correctedOcr;
          targetBox.text = translated || correctedOcr;
        }

        // Actualizar editor inline
        const transInput = entry.querySelector("[data-action='trans-input']");
        if (transInput) transInput.value = translated || correctedOcr;

        // Actualizar texto en canvas
        entry.querySelector(".block-ocr").textContent = truncate(correctedOcr, 60);
        entry.querySelector(".block-translated").textContent = truncate(translated || correctedOcr, 60);
        refreshScreenCanvas();

        // Si esta burbuja está seleccionada, actualizar también el panel de Editor
        if (box.id === state.selectedId) {
          sourceText.value = correctedOcr;
          translatedText.value = translated || correctedOcr;
        }

        showToast(`Bloque #${idx + 1} retraducido`, "success", 2000);
      } catch (err) {
        showToast(`Error: ${err.message}`, "error", 3000);
      } finally {
        btn.disabled = false;
        btn.textContent = "⟳ Retraducir";
      }
    });

    // Edición inline de OCR: actualizar source preview y canvas
    entry.querySelector("[data-action='ocr-input']").addEventListener("input", (e) => {
      entry.querySelector(".block-ocr").textContent = truncate(e.target.value, 60);
    });

    // Edición inline de traducción: actualizar canvas
    entry.querySelector("[data-action='trans-input']").addEventListener("input", (e) => {
      const targetBox = getPageBoxes().find(b => b.id === box.id);
      if (targetBox) {
        targetBox.text = e.target.value;
      }
      entry.querySelector(".block-translated").textContent = truncate(e.target.value, 60);
      refreshScreenCanvas();
    });

    container.appendChild(entry);
  });
}

// Helpers para renderBlockList
function escHtml(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function truncate(str, max) {
  return String(str).length > max ? String(str).slice(0, max) + "…" : String(str);
}

// ─── Función compartida de dibujo de texto ────────────────────────
// Unifica sombra, contorno, centrado y medidas entre renderBoxes()
// (pantalla) y drawProfessionalText() (exportación PNG/PDF).
// Cualquier cambio visual debe hacerse AQUÍ, no en las dos funciones.
function drawTextOnCanvas(ctx, text, box, layout) {
  ctx.save();

  // Configurar fuente
  ctx.font = `${box.fontStyle || "normal"} ${box.fontWeight || "400"} ${layout.fontSize}px ${box.fontFamily}`;
  ctx.textBaseline = "top";
  ctx.fillStyle = box.color;

  // ── Sombra de legibilidad ─────────────────────────────────────
  const col = box.color || "#000000";
  const isLightText = (() => {
    const m = col.match(/\d+/g);
    if (!m) return false;
    const [r, g, b] = m.map(Number);
    return (r * 0.299 + g * 0.587 + b * 0.114) > 128;
  })();

  // ── Centrado vertical ─────────────────────────────────────────
  const totalTextHeight = layout.lines.length * layout.lineHeight;
  const startY = box.y + Math.max(0, (box.h - totalTextHeight) / 2);

  // ── Relleno semitransparente de fondo (detrás de cada línea) ──
  // Usa startY para alinearse con el centrado vertical del texto.
  const hasBgFill = box.fillOpacity > 0 && box.bg && box.bg !== "transparent";
  const hasGlow = box.glowColor && box.glowColor !== "transparent" && box.glowBlur > 0;

  // ── Dibujar cada línea centrada horizontalmente ───────────────
  for (let i = 0; i < layout.lines.length; i++) {
    const line = layout.lines[i];
    const textWidth = ctx.measureText(line).width;
    const lineX = box.x + (box.w - textWidth) / 2;
    const lineY = startY + i * layout.lineHeight;

    // 1. Relleno semitransparente por línea (pill-shaped background)
    if (hasBgFill) {
      const pad = Math.max(2, layout.fontSize * 0.12);
      ctx.globalAlpha = box.fillOpacity;
      ctx.fillStyle = box.bg;
      ctx.fillRect(lineX - pad, lineY - pad, textWidth + pad * 2, layout.lineHeight + pad * 2);
      ctx.globalAlpha = 1.0;
    }

    // 2. Glow exterior: dibujar texto transparente con shadow para crear el halo
    if (hasGlow) {
      ctx.shadowColor = box.glowColor;
      ctx.shadowBlur = box.glowBlur;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      ctx.fillStyle = "transparent";
      ctx.fillText(line, lineX, lineY);  // solo el shadow (glow) es visible
      ctx.shadowColor = "transparent";  // reset para el texto principal
      ctx.shadowBlur = 0;
      ctx.fillStyle = box.color;        // restaurar color original
    }

    // 3. Sombra de legibilidad (sombra normal del texto)
    if (box.shadow !== false && !hasGlow) {
      ctx.shadowColor = isLightText ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.6)";
      ctx.shadowBlur = Math.max(3, layout.fontSize * 0.18);
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    } else {
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
    }

    // 4. Contorno de texto
    if (box.strokeColor && box.strokeWidth > 0 && box.strokeColor !== "transparent") {
      ctx.strokeStyle = box.strokeColor;
      ctx.lineWidth = box.strokeWidth * 2;
      ctx.lineJoin = "round";
      ctx.strokeText(line, lineX, lineY);
    }

    // 5. Texto principal
    ctx.fillText(line, lineX, lineY);
  }

  ctx.restore();
}

// 5. EDITOR DE BURBUJAS DOM E INTERACCIONES EN EL OVERLAY
function renderBoxes() {
  try {
    // 1. Limpiar el overlay de cajas interactivas anteriores
    overlay.innerHTML = "";
    const boxes = getPageBoxes();
    const ctx = pdfCanvas.getContext("2d", { willReadFrequently: true });

    for (const box of boxes) {
      const layout = fitTextLayout(box.text || box.source || "", box);

      // --- Dibujar texto en el canvas directamente ---
      ctx.save();

      // Recortar al área de la caja
      ctx.beginPath();
      ctx.rect(box.x, box.y, box.w, box.h);
      ctx.clip();

      // Rellenar el fondo SOLO si la burbuja tiene color sólido (no transparent)
      // Esto preserva el dibujo para texto que está sobre fondos complejos
      if (box.bg && box.bg !== "transparent") {
        ctx.fillStyle = box.bg;
        ctx.fillRect(box.x, box.y, box.w, box.h);
      }

      // Usar función compartida para sombra, contorno, centrado y dibujo
      drawTextOnCanvas(ctx, box.text || box.source || "", box, layout);

      // --- Crear manejador interactivo invisible en el overlay ---
      const div = document.createElement("div");
      div.className = `box${box.id === state.selectedId ? " selected" : ""}`;
      div.dataset.id = box.id;
      div.style.left = `${box.x}px`;
      div.style.top = `${box.y}px`;
      div.style.width = `${box.w}px`;
      div.style.height = `${box.h}px`;
      div.style.backgroundColor = "transparent";
      div.style.color = "transparent";

      // Indicador visual de selección (borde punteado al seleccionar)
      if (box.id === state.selectedId) {
        div.style.outline = "2px dashed #00e5a0";
        div.style.outlineOffset = "2px";
      }

      // Manejador de redimensionamiento
      const resizeHandle = document.createElement("span");
      resizeHandle.className = "resize";
      div.appendChild(resizeHandle);

      overlay.appendChild(div);
    }
  } catch (error) {
    console.error("Error in renderBoxes:", error);
    setStatus(`Error en renderizado: ${error.message}`);
  }
}

// Ajuste dinámico de texto para que quepa en el cuadro (Edición Profesional)
function fitTextLayout(text, box) {
  const canvas = fitTextLayout.canvas || (fitTextLayout.canvas = document.createElement("canvas"));
  const ctx = canvas.getContext("2d");
  // NO convertir a mayúsculas - preservar el casing de la traducción
  const value = String(text || "").trim();
  // Padding generoso: 75% del ancho/alto para que respire como burbujas de manga
  const maxWidth = Math.max(24, box.w * 0.75);
  const maxHeight = Math.max(18, box.h * 0.78);
  const minSize = 11;  // Mínimo legible
  const maxSize = Math.min(box.fontSize || 18, Math.max(12, box.h * 0.82));
  
  for (let size = maxSize; size >= minSize; size--) {
    const lineHeight = Math.ceil(size * 1.35);  // Interlineado 1.35x (más aire)
    ctx.font = `${box.fontStyle || "normal"} ${box.fontWeight || "400"} ${size}px ${box.fontFamily}`;
    const lines = wrapTextLines(ctx, value, maxWidth);
    if (lines.length * lineHeight <= maxHeight) {
      return { lines, fontSize: size, lineHeight };
    }
  }
  
  // Si no cabe ni al mínimo, escalamos proporcionalmente
  ctx.font = `${box.fontStyle || "normal"} ${box.fontWeight || "400"} ${minSize}px ${box.fontFamily}`;
  const lines = wrapTextLines(ctx, value, maxWidth);
  const lineHeight = Math.ceil(minSize * 1.35);
  
  // Si aún así no cabe verticalmente, reducimos un poco más la fuente
  if (lines.length * lineHeight > maxHeight) {
    const scale = maxHeight / (lines.length * lineHeight);
    const adjustedSize = Math.max(9, Math.floor(minSize * scale));
    ctx.font = `${box.fontStyle || "normal"} ${box.fontWeight || "400"} ${adjustedSize}px ${box.fontFamily}`;
    const adjustedLines = wrapTextLines(ctx, value, maxWidth);
    return {
      lines: adjustedLines,
      fontSize: adjustedSize,
      lineHeight: Math.ceil(adjustedSize * 1.35)
    };
  }
  
  return { lines, fontSize: minSize, lineHeight };
}

// Detecta si el texto contiene escritura CJK (chino, japonés, coreano),
// que se segmenta carácter por carácter porque no usa espacios entre palabras.
function containsCJK(text) {
  return /[\u3040-\u309f\u30a0-\u30ff\u3130-\u318f\u3200-\u32ff\u3300-\u33ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7a3\uff00-\uffef]/.test(text);
}

// Segmentación carácter por carácter (solo para CJK, o como último recurso
// en textos latinos cuando una sola palabra no cabe ni sola en el ancho disponible).
function wrapByCharacters(ctx, text, maxWidth) {
  const chars = Array.from(text);
  const lines = [];
  let currentLine = "";
  for (const ch of chars) {
    const testLine = currentLine + ch;
    const testWidth = ctx.measureText(testLine).width;
    if (testWidth > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = ch;
    } else {
      currentLine = testLine;
    }
  }
  if (currentLine) lines.push(currentLine);
  return lines.length ? lines : [""];
}

function wrapTextLines(ctx, text, maxWidth) {
  // CJK: sin espacios entre palabras, se segmenta por carácter (comportamiento correcto).
  if (containsCJK(text)) {
    return wrapByCharacters(ctx, text, maxWidth);
  }

  // Latino/occidental: SIEMPRE por palabras completas, nunca letra por letra,
  // aunque el texto entero venga sin ningún espacio (ej. "CORRECTLY", "INCREDIBLE").
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let currentLine = "";

  for (const word of words) {
    const testLine = currentLine ? `${currentLine} ${word}` : word;
    const testWidth = ctx.measureText(testLine).width;
    if (testWidth > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = testLine;
    }
  }
  if (currentLine) lines.push(currentLine);

  // Último recurso: si una sola palabra es más ancha que la caja incluso sola
  // (nombre largo sin espacios, ej. una URL), se rompe por caracteres SOLO esa
  // palabra, para no desbordar el lienzo. Esto no afecta al caso normal.
  const finalLines = [];
  for (const line of lines) {
    if (!line.includes(" ") && ctx.measureText(line).width > maxWidth) {
      finalLines.push(...wrapByCharacters(ctx, line, maxWidth));
    } else {
      finalLines.push(line);
    }
  }
  return finalLines.length ? finalLines : [""];
}

// Obtener punto relativo al overlay (considera transform CSS del contenedor)
function getRelativePoint(event) {
  const rect = overlay.getBoundingClientRect();
  // Obtener la transformación CSS del stage para corregir coordenadas
  const stageRect = stage.getBoundingClientRect();
  const scaleX = rect.width / stageRect.width;
  const scaleY = rect.height / stageRect.height;
  
  return {
    x: (event.clientX - rect.left) / scaleX,
    y: (event.clientY - rect.top) / scaleY
  };
}

// Convierte cualquier color CSS a hex válido para inputs type=color
function toHexColor(col, fallback = "#ffffff") {
  if (!col || col === "transparent" || col === "none") return fallback;
  if (col.startsWith("#")) return col;
  // rgb(r,g,b) o rgba(r,g,b,a)
  const m = col.match(/(\d+),\s*(\d+),\s*(\d+)/);
  if (m) {
    const hex = n => Number(n).toString(16).padStart(2, "0");
    return `#${hex(m[1])}${hex(m[2])}${hex(m[3])}`;
  }
  // Intentar resolver colores con nombre (red, blue, etc) via canvas
  try {
    const ctx = document.createElement("canvas").getContext("2d");
    ctx.fillStyle = col;
    const resolved = ctx.fillStyle; // Devuelve rgb(r,g,b)
    const rm = resolved.match(/(\d+),\s*(\d+),\s*(\d+)/);
    if (rm) {
      const hex = n => Number(n).toString(16).padStart(2, "0");
      return `#${hex(rm[1])}${hex(rm[2])}${hex(rm[3])}`;
    }
  } catch (e) { /* ignore */ }
  return fallback;
}

// Selección y edición de cajas
function selectBox(id) {
  state.selectedId = id;
  const box = getPageBoxes().find(b => b.id === id);
  if (box) {
    sourceText.value = box.source || "";
    translatedText.value = box.text || "";
    bubbleColor.value = toHexColor(box.bg, "#ffffff");
    textColor.value = toHexColor(box.color, "#000000");
    strokeColor.value = toHexColor(box.strokeColor || "transparent", "#000000");
    strokeWidth.value = String(box.strokeWidth || 0);
    fontSize.value = String(box.fontSize || 18);
    fontFamily.value = box.fontFamily || "Comic Sans MS";
    eraseMode.value = box.eraseMode || "area";

    // Sincronizar controles de glow y fillOpacity
    const hasGlow = box.glowColor && box.glowColor !== "transparent" && box.glowBlur > 0;
    glowToggle.checked = hasGlow;
    if (hasGlow && box.glowColor !== "transparent") {
      glowColor.value = toHexColor(box.glowColor, "#ffd700");
    }
    glowBlur.value = String(box.glowBlur || 0);
    glowBlurValue.textContent = box.glowBlur || "0";
    fillOpacity.value = String(Math.round((box.fillOpacity || 0) * 100));
    fillOpacityValue.textContent = Math.round((box.fillOpacity || 0) * 100) + "%";

    // Actualizar botones Bold/Italic
    if (box.fontStyle === "italic") btnItalic.classList.add("active-style");
    else btnItalic.classList.remove("active-style");

    if (box.fontWeight === "800" || box.fontWeight === "700" || box.fontWeight === "bold") btnBold.classList.add("active-style");
    else btnBold.classList.remove("active-style");

    state.italic = box.fontStyle === "italic";
    state.bold = box.fontWeight === "800" || box.fontWeight === "700" || box.fontWeight === "bold";

    // Sincronizar selección en la block list
    const container = $("#blockListContainer");
    if (container) {
      container.querySelectorAll(".block-entry").forEach(el => {
        el.classList.toggle("selected", el.dataset.boxId === id);
      });
    }
  } else {
    sourceText.value = "";
    translatedText.value = "";
  }
  renderBoxes();
}

async function updateSelectedBox(patch, requireBgUpdate = false) {
  if (!state.selectedId) return;
  const box = getPageBoxes().find(b => b.id === state.selectedId);
  if (box) {
    Object.assign(box, patch);
    if (requireBgUpdate) {
      await updateErasedBg();
    }
    refreshScreenCanvas();
  }
}

// Eventos de ratón/cursor en el overlay
function startOverlayAction(event) {
  if (!state.kind) return;
  const point = getRelativePoint(event);
  const targetBox = event.target.closest(".box");
  
  if (state.mode === "draw" && !targetBox) {
    // Modo dibujar caja
    const newBox = {
      id: crypto.randomUUID(),
      x: point.x,
      y: point.y,
      w: 16,
      h: 12,
      source: "",
      text: "",
      bg: "transparent",     // Se actualiza al soltar con el color de fondo real
      color: textColor.value,
      strokeColor: "transparent",
      strokeWidth: 0,
      fontSize: Number(fontSize.value),
      fontFamily: fontFamily.value,
      fontStyle: state.italic ? "italic" : "normal",
      fontWeight: state.bold ? "800" : "700",
      eraseMode: "area"      // Siempre borra el original con el color de fondo muestreado
    };
    getPageBoxes().push(newBox);
    state.draft = { id: newBox.id, startX: point.x, startY: point.y, kind: "draw", w: pdfCanvas.width, h: pdfCanvas.height };
    selectBox(newBox.id);
  } else if (targetBox) {
    const id = targetBox.dataset.id;
    const box = getPageBoxes().find(b => b.id === id);
    selectBox(id);
    
    if (event.target.classList.contains("resize")) {
      // Redimensionamiento
      state.draft = { id, kind: "resize", w: pdfCanvas.width, h: pdfCanvas.height };
    } else if (state.mode === "move") {
      // Movimiento
      state.draft = { id, kind: "move", offsetX: point.x - box.x, offsetY: point.y - box.y };
    }
  }
}

function pointerMove(event) {
  if (!state.draft) return;
  const point = getRelativePoint(event);
  const box = getPageBoxes().find(b => b.id === state.draft.id);
  if (!box) return;

  if (state.draft.kind === "move") {
    box.x = Math.max(0, Math.min(state.draft.w - box.w, point.x - state.draft.offsetX));
    box.y = Math.max(0, Math.min(state.draft.h - box.h, point.y - state.draft.offsetY));
  } else if (state.draft.kind === "resize") {
    box.w = Math.max(20, Math.min(state.draft.w - box.x, point.x - box.x));
    box.h = Math.max(14, Math.min(pdfCanvas.height - box.y, point.y - box.y));
  } else if (state.draft.kind === "draw") {
    box.x = Math.min(state.draft.startX, point.x);
    box.y = Math.min(state.draft.startY, point.y);
    box.w = Math.max(20, Math.min(pdfCanvas.width - box.x, Math.abs(point.x - state.draft.startX)));
    box.h = Math.max(14, Math.min(pdfCanvas.height - box.y, Math.abs(point.y - state.draft.startY)));
  }
  renderBoxes();
}

function pointerUp() {
  if (state.draft) {
    const wasDrawing = state.draft.kind === "draw";
    const drawnId = state.draft.id;
    const movedId = state.draft.kind !== "draw" ? state.draft.id : null;
    state.draft = null;

    if (movedId) {
      // Re-samplear colores al mover/redimensionar ya que el fondo debajo cambió
      const box = getPageBoxes().find(b => b.id === movedId);
      if (box) {
        box.bg = sampleBgColorAround(box);
        box.color = sampleTextColor(box);
      }
      updateErasedBg().then(() => refreshScreenCanvas());
    } else {
      refreshScreenCanvas();
    }

    // Si se acaba de DIBUJAR una burbuja nueva: NO hacer auto-OCR local (usa Tesseract.js que falla)
    // El flujo correcto es: dibuja burbujas -> "Traducir Página Actual" (servidor)
    // if (wasDrawing && window.Tesseract) {
    //   const box = getPageBoxes().find(b => b.id === drawnId);
    //   if (box && box.w > 20 && box.h > 14) {
    //     setTimeout(() => autoOcrAndTranslateBox(box), 50);
    //   }
    // }
    window.removeEventListener("pointermove", pointerMove);
    window.removeEventListener("pointerup", pointerUp);
  }
}

// 6. BORRADO AVANZADO POR OPENCV (INPAINTING)
async function eraseWithInpainting(canvas, boxes) {
  if (!boxes.length || !coverOriginal.checked) return false;
  if (!state.cvLoaded) return false;
  
  const cv = window.cv;
  let src = null, mask = null, dst = null;
  
  try {
    src = cv.imread(canvas);
    mask = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);

    let hasInpaintArea = false;
    for (const box of boxes) {
      if (box.eraseMode === "none") continue;
      hasInpaintArea = true;
      
      if (box.eraseMode === "glyph") {
        // Borrar solo las letras detectadas (glifos)
        markGlyphMask(cv, canvas, mask, box);
      } else {
        // Borrado del área completa (con un pequeño margen interior)
        const inset = Math.max(1, Math.round(Math.min(box.w, box.h) * 0.03));
        const x = Math.max(0, Math.round(box.x + inset));
        const y = Math.max(0, Math.round(box.y + inset));
        const w = Math.max(1, Math.min(src.cols - x, Math.round(box.w - inset * 2)));
        const h = Math.max(1, Math.min(src.rows - y, Math.round(box.h - inset * 2)));
        
        const p1 = new cv.Point(x, y), p2 = new cv.Point(x + w, y + h), sc = new cv.Scalar(255);
        cv.rectangle(mask, p1, p2, sc, -1);
        [p1, p2, sc].forEach(o => o.delete());
      }
    }

    if (!hasInpaintArea) {
      return true;
    }

    // Verificar que la máscara no esté vacía
    if (cv.countNonZero(mask) === 0) {
      return true;
    }

    dst = new cv.Mat();
    // Ejecutar Inpainting Telea en la máscara
    cv.inpaint(src, mask, dst, 15, cv.INPAINT_TELEA);
    cv.imshow(canvas, dst);
    
    return true;

  } catch (error) {
    console.warn("Fallo en inpainting de OpenCV.js, cayendo a borrado básico", error);
    return false;
  } finally {
    // Liberar memoria WASM de OpenCV siempre
    if (src) src.delete();
    if (mask) mask.delete();
    if (dst) dst.delete();
  }
}

// Máscara de glifos: detecta pixels de texto de CUALQUIER COLOR comparando con el fondo del borde
function markGlyphMask(cv, sourceCanvas, mask, box) {
  const ctx = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const w = Math.max(1, Math.min(sourceCanvas.width - x, Math.ceil(box.w)));
  const h = Math.max(1, Math.min(sourceCanvas.height - y, Math.ceil(box.h)));
  const pixels = ctx.getImageData(x, y, w, h).data;

  // Calcular el color de fondo promedio del borde exterior (15%)
  const edgeW = Math.max(1, Math.round(w * 0.15));
  const edgeH = Math.max(1, Math.round(h * 0.15));
  let bgR = 0, bgG = 0, bgB = 0, bgCount = 0;
  for (let py = 0; py < h; py += 2) {
    for (let px = 0; px < w; px += 2) {
      const isBorder = px < edgeW || py < edgeH || px > w - edgeW || py > h - edgeH;
      if (!isBorder) continue;
      const i = (py * w + px) * 4;
      bgR += pixels[i]; bgG += pixels[i+1]; bgB += pixels[i+2];
      bgCount++;
    }
  }
  if (bgCount > 0) { bgR /= bgCount; bgG /= bgCount; bgB /= bgCount; }

  let local = null, kernel = null, roi = null;
  try {
    local = cv.Mat.zeros(h, w, cv.CV_8UC1);
    const ptr = local.data;

    // Marcar como glifo cualquier pixel que difiera del fondo por más de 40 unidades (contraste de color)
    const threshold = 40;
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4;
        const dr = pixels[i] - bgR;
        const dg = pixels[i+1] - bgG;
        const db = pixels[i+2] - bgB;
        const colorDist = Math.sqrt(dr*dr + dg*dg + db*db);
        if (colorDist > threshold) {
          ptr[py * w + px] = 255;
        }
      }
    }

    // Dilatar la máscara para abarcar contornos difusos
    kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(local, local, kernel);
    
    roi = mask.roi(new cv.Rect(x, y, w, h));
    local.copyTo(roi);
  } finally {
    if (roi) roi.delete();
    if (kernel) kernel.delete();
    if (local) local.delete();
  }
}

// Borrado básico por color muestreado (Fallback cuando OpenCV no está disponible)
function fallbackEraseBox(ctx, box) {
  if (box.eraseMode === "none") return;
  const pad = Math.max(1, Math.round(Math.min(box.w, box.h) * 0.03));
  // Muestrear siempre del cleanBgCanvas (limpio) excepto si estamos exportando en un canvas externo
  const sampleCanvas = ctx.canvas === pdfCanvas ? cleanBgCanvas : ctx.canvas;
  const color = sampleEraseColor(box, sampleCanvas);
  
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = box.eraseMode === "glyph" ? 0.9 : 1.0;
  
  // Dibujar rectángulo de borrado con esquinas ligeramente redondeadas
  const rx = box.x + pad;
  const ry = box.y + pad;
  const rw = Math.max(1, box.w - pad * 2);
  const rh = Math.max(1, box.h - pad * 2);
  const radius = box.eraseMode === "glyph" ? 4 : 8;
  
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(rx, ry, rw, rh, radius) : ctx.rect(rx, ry, rw, rh);
  ctx.fill();
  ctx.restore();
}

function sampleEraseColor(box, canvas = cleanBgCanvas) {
  try {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const x = Math.max(0, Math.floor(box.x));
    const y = Math.max(0, Math.floor(box.y));
    const w = Math.max(1, Math.min(canvas.width - x, Math.ceil(box.w)));
    const h = Math.max(1, Math.min(canvas.height - y, Math.ceil(box.h)));
    const data = ctx.getImageData(x, y, w, h).data;
    const colors = [];
    const edge = Math.max(2, Math.round(Math.min(w, h) * 0.1));
    
    for (let py = 0; py < h; py += 2) {
      for (let px = 0; px < w; px += 2) {
        const isBorder = px < edge || py < edge || px > w - edge || py > h - edge;
        if (!isBorder) continue;
        const i = (py * w + px) * 4;
        colors.push([data[i], data[i+1], data[i+2]]);
      }
    }
    if (!colors.length) return box.bg || "#ffffff";
    colors.sort((a, b) => (a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2]));
    const c = colors[Math.floor(colors.length * 0.5)];
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  } catch (e) {
    return box.bg || "#ffffff";
  }
}

// 7. RENDERIZADO Y DIBUJADO PROFESIONAL DE TEXTO EN EL CANVAS
function drawProfessionalText(ctx, text, box) {
  const layout = fitTextLayout(text, box);
  drawTextOnCanvas(ctx, text, box, layout);
}

// 8. RENDERIZADO DEL LIENZO FINAL EDITADO (COMBINA BORRADO Y TEXTOS NUEVOS)
let _exportCanvas = null;
function _getExportCanvas(w, h) {
  if (!_exportCanvas) _exportCanvas = document.createElement("canvas");
  _exportCanvas.width = w;
  _exportCanvas.height = h;
  return _exportCanvas;
}
async function renderEditedCanvas(pageNo = state.page, rawCanvas = pdfCanvas) {
  const output = _getExportCanvas(rawCanvas.width, rawCanvas.height);
  
  const ctx = output.getContext("2d");
  ctx.drawImage(rawCanvas, 0, 0);
  
  const boxes = getPageBoxes(pageNo);
  if (coverOriginal.checked && boxes.length > 0) {
    // Intentar borrado avanzado OpenCV
    const ok = await eraseWithInpainting(output, boxes);
    if (!ok) {
      // Fallback a borrado básico de color muestreado
      for (const box of boxes) {
        fallbackEraseBox(ctx, box);
      }
    }
  }

  // Dibujar el texto traducido sobre el canvas final
  const outCtx = output.getContext("2d");
  for (const box of boxes) {
    drawProfessionalText(outCtx, box.text || box.source || "", box);
  }
  
  return output;
}

// Renderizar una página PDF cruda (sin interactividad de la UI) a un canvas temporal
async function renderRawPdfPage(pageNo, targetCanvas) {
  const page = await state.pdf.getPage(pageNo);
  const viewport = page.getViewport({ scale: state.scale });
  targetCanvas.width = Math.round(viewport.width);
  targetCanvas.height = Math.round(viewport.height);
  await page.render({ canvasContext: targetCanvas.getContext("2d"), viewport }).promise;
}

// 9. EXPORTACIÓN DE ARCHIVOS (PNG Y PDF)
async function exportCurrentPng() {
  if (!state.kind) return setStatus("No hay ningún archivo cargado.");
  setStatus("Preparando exportación a imagen PNG...");
  const startedAt = Date.now();
  showProgress("Exportando PNG", 0, 1, startedAt);
  
  const editedCanvas = await renderEditedCanvas(state.page, cleanBgCanvas);
  editedCanvas.toBlob(blob => {
    const link = document.createElement("a");
    const customName = exportName.value.trim();
    link.download = customName ? `${customName}.png` : `pagina-${state.page}-traducida.png`;
    link.href = URL.createObjectURL(blob);
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), window.__CLIENT_CONFIG.TIMEOUT_EXPORT_REVOKE_MS);
  });
  
  showProgress("Exportando PNG", 1, 1, startedAt);
  setStatus("PNG exportado correctamente.");
}

async function exportCurrentPdf() {
  if (!state.kind || !window.jspdf) return setStatus("jsPDF no se ha cargado.");
  setStatus("Preparando PDF de página actual...");
  const startedAt = Date.now();
  showProgress("Exportando PDF", 0, 1, startedAt);
  
  const editedCanvas = await renderEditedCanvas(state.page, cleanBgCanvas);
  const { jsPDF } = window.jspdf;
  const orientation = editedCanvas.width > editedCanvas.height ? "landscape" : "portrait";
  
  const pdf = new jsPDF({
    orientation,
    unit: "px",
    format: [editedCanvas.width, editedCanvas.height]
  });
  
  pdf.addImage(editedCanvas.toDataURL("image/png"), "PNG", 0, 0, editedCanvas.width, editedCanvas.height);
  const customName = exportName.value.trim();
  pdf.save(customName ? `${customName}.pdf` : `pagina-${state.page}-traducida.pdf`);
  
  showProgress("Exportando PDF", 1, 1, startedAt);
  setStatus("PDF de página actual exportado.");
}

async function exportFullPdf() {
  if (!state.kind) return setStatus("No hay ningún archivo cargado.");
  if (state.kind !== "pdf") return exportCurrentPdf();
  if (!window.jspdf) return setStatus("jsPDF no se ha cargado.");
  
  setStatus("Preparando exportación de PDF completo (esto puede tardar)...");
  const startedAt = Date.now();
  const { jsPDF } = window.jspdf;
  const originalPage = state.page;
  let pdf = null;
  
  for (let p = 1; p <= state.pageCount; p++) {
    showProgress("Generando PDF Completo", p - 1, state.pageCount, startedAt);
    
    // Crear canvas temporal para la página
    const tempCanvas = document.createElement("canvas");
    await renderRawPdfPage(p, tempCanvas);
    
    // Aplicar inpainting y textos en el canvas temporal
    const editedCanvas = await renderEditedCanvas(p, tempCanvas);
    const orientation = editedCanvas.width > editedCanvas.height ? "landscape" : "portrait";
    
    if (!pdf) {
      pdf = new jsPDF({
        orientation,
        unit: "px",
        format: [editedCanvas.width, editedCanvas.height]
      });
    } else {
      pdf.addPage([editedCanvas.width, editedCanvas.height], orientation);
    }
    
    pdf.addImage(editedCanvas.toDataURL("image/png"), "PNG", 0, 0, editedCanvas.width, editedCanvas.height);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  
  const customName = exportName.value.trim();
  pdf.save(customName ? `${customName}.pdf` : "documento-completo-traducido.pdf");
  showProgress("Generando PDF Completo", state.pageCount, state.pageCount, startedAt);
  
  // Regresar a la página original en la visualización
  await renderPage(originalPage);
  setStatus("PDF completo descargado con éxito.");
}

// 10. BINDEO DE EVENTOS DEL DOM
//console.log("[init] fileInput element:", fileInput);
fileInput.addEventListener("change", (e) => {
  //console.log("[fileInput.change] event fired, files:", e.target.files);
  const files = e.target.files;
  if (!files || !files.length) return;
  const file = files[0];
  if (file) openFile(file).catch(err => setStatus(`Error: ${err.message}`));
});

// El label ya envuelve nativamente al <input>, no necesita JS

// Arrastrar y soltar archivos en la pantalla
stageWrap.addEventListener("dragover", (e) => e.preventDefault());
stageWrap.addEventListener("drop", (e) => {
  e.preventDefault();
  const files = e.dataTransfer.files;
  if (!files || !files.length) return;
  const file = files[0];
  if (file) openFile(file).catch(err => setStatus(`Error: ${err.message}`));
});

// Navegación de páginas
prevPage.addEventListener("click", () => {
  if (state.kind === "pdf" && state.page > 1) {
    renderPage(state.page - 1).catch(err => setStatus(err.message));
  }
});

nextPage.addEventListener("click", () => {
  if (state.kind === "pdf" && state.page < state.pageCount) {
    renderPage(state.page + 1).catch(err => setStatus(err.message));
  }
});

pageNumber.addEventListener("change", () => {
  if (state.kind === "pdf") {
    const val = Math.min(state.pageCount, Math.max(1, Number(pageNumber.value) || 1));
    renderPage(val).catch(err => setStatus(err.message));
  }
});

// Modos de cursor
drawModeBtn.addEventListener("click", () => {
  state.mode = "draw";
  drawModeBtn.classList.add("active");
  moveModeBtn.classList.remove("active");
  overlay.className = "overlay drawing";
});

moveModeBtn.addEventListener("click", () => {
  state.mode = "move";
  moveModeBtn.classList.add("active");
  drawModeBtn.classList.remove("active");
  overlay.className = "overlay";
});

// Bindeo de clic e interacciones en overlay
overlay.addEventListener("pointerdown", (e) => {
  window.addEventListener("pointermove", pointerMove);
  window.addEventListener("pointerup", pointerUp);
  startOverlayAction(e);
});

// Edición en tiempo real desde los inputs del panel
sourceText.addEventListener("input", () => {
  updateSelectedBox({ source: sourceText.value });
});

translatedText.addEventListener("input", () => {
  updateSelectedBox({ text: translatedText.value });
});

coverOriginal.addEventListener("change", async () => {
  await updateErasedBg();
  refreshScreenCanvas();
});

// ── Helper: actualiza clase active en panel de efectos ──────
function updateTextEffectsPanelState() {
  const panel = document.getElementById('textEffectsPanel');
  if (!panel) return;
  panel.classList.toggle('has-active-glow', glowToggle.checked);
  panel.classList.toggle('has-active-fill', Number(fillOpacity.value) > 0);
}

// ── Preview hover: muestra glow temporal al pasar mouse ─────
let _hoverGlowPreview = null;

glowToggle.addEventListener("mouseenter", () => {
  if (!state.selectedId || glowToggle.checked) return;
  const box = getPageBoxes().find(b => b.id === state.selectedId);
  if (!box) return;
  // Guardar estado original
  _hoverGlowPreview = {
    glowColor: box.glowColor,
    glowBlur: box.glowBlur
  };
  // Mostrar preview con valores actuales de UI
  Object.assign(box, {
    glowColor: glowColor.value,
    glowBlur: Math.max(1, Number(glowBlur.value) || 12)
  });
  renderBoxes();
});

glowToggle.addEventListener("mouseleave", () => {
  if (!_hoverGlowPreview) return;
  const box = getPageBoxes().find(b => b.id === state.selectedId);
  if (box) {
    Object.assign(box, _hoverGlowPreview);
  }
  _hoverGlowPreview = null;
  renderBoxes();
});

// ── Controles de efectos de texto (glow y fillOpacity) ──────
glowToggle.addEventListener("change", () => {
  _hoverGlowPreview = null;  // Descartar preview pendiente, el usuario ya decidió
  const enabled = glowToggle.checked;
  const patch = {
    glowColor: enabled ? glowColor.value : "transparent",
    glowBlur: enabled ? Number(glowBlur.value) : 0
  };
  updateSelectedBox(patch);
  updateTextEffectsPanelState();
});

glowColor.addEventListener("input", () => {
  if (glowToggle.checked) {
    updateSelectedBox({ glowColor: glowColor.value });
  }
});

glowBlur.addEventListener("input", () => {
  const val = Number(glowBlur.value);
  glowBlurValue.textContent = val;
  if (glowToggle.checked) {
    updateSelectedBox({ glowBlur: val });
  }
});

fillOpacity.addEventListener("input", () => {
  const val = Number(fillOpacity.value) / 100;
  fillOpacityValue.textContent = Math.round(val * 100) + "%";
  updateSelectedBox({ fillOpacity: val });
  updateTextEffectsPanelState();
});

// Cambios de estilos rápidos
bubbleColor.addEventListener("input", () => {
  updateSelectedBox({ bg: bubbleColor.value }, true);
});

textColor.addEventListener("input", () => {
  updateSelectedBox({ color: textColor.value });
});

strokeColor.addEventListener("input", () => {
  updateSelectedBox({ strokeColor: strokeColor.value });
});

strokeWidth.addEventListener("input", () => {
  updateSelectedBox({ strokeWidth: Number(strokeWidth.value) });
});

fontSize.addEventListener("input", () => {
  updateSelectedBox({ fontSize: Number(fontSize.value) });
});

fontFamily.addEventListener("change", () => {
  updateSelectedBox({ fontFamily: fontFamily.value });
});

eraseMode.addEventListener("change", () => {
  updateSelectedBox({ eraseMode: eraseMode.value }, true);
});

// Botones de estilo (Itálica y Negrita)
btnItalic.addEventListener("click", () => {
  state.italic = !state.italic;
  if (state.italic) btnItalic.classList.add("active-style");
  else btnItalic.classList.remove("active-style");
  updateSelectedBox({ fontStyle: state.italic ? "italic" : "normal" });
});

btnBold.addEventListener("click", () => {
  state.bold = !state.bold;
  if (state.bold) btnBold.classList.add("active-style");
  else btnBold.classList.remove("active-style");
  updateSelectedBox({ fontWeight: state.bold ? "800" : "400" });
});

// Botón de traducción manual de la burbuja seleccionada
translateBtn.addEventListener("click", async () => {
  try {
    if (!state.selectedId) return setStatus("Selecciona una burbuja primero.");
    const text = sourceText.value.trim();
    if (!text) return setStatus("El texto original está vacío.");

    setStatus("Traduciendo texto...");
    const translated = await translateOnline(text, targetLang.value);
    translatedText.value = translated;
    updateSelectedBox({ text: translated });
    setStatus("Traducido.");
  } catch (err) {
    setStatus(`Error traduciendo: ${err.message}`);
    console.error("[translateBtn] Error:", err);
  }
});

// Botón para hacer OCR solo en la burbuja seleccionada
autoDetectPage.addEventListener("click", async () => {
  if (!state.kind) return setStatus("Carga un archivo primero.");
  autoTranslateCurrentPage().catch(err => setStatus(`Error: ${err.message}`));
});

// Botón para eliminar caja seleccionada
deleteBox.addEventListener("click", async () => {
  if (!state.selectedId) return;
  const boxes = getPageBoxes();
  const index = boxes.findIndex(b => b.id === state.selectedId);
  if (index >= 0) {
    boxes.splice(index, 1);
  }
  state.selectedId = null;
  sourceText.value = "";
  translatedText.value = "";
  if (boxes.length === 0 && state.inpaintedBgByPage) {
    state.inpaintedBgByPage.delete(state.page);
  }
  await updateErasedBg();
  refreshScreenCanvas();
  renderBlockList();
  setStatus("Burbuja eliminada.");
});

// Botón para limpiar toda la página (eliminar todas las burbujas de la página actual)
clearPageBoxes.addEventListener("click", async () => {
  if (!state.kind) return;
  const boxes = getPageBoxes();
  if (!boxes.length) return setStatus("No hay burbujas en esta página.");
  
  if (confirm("¿Estás seguro de que quieres eliminar TODAS las traducciones de la página actual?")) {
    state.boxesByPage.set(state.page, []);
    if (state.inpaintedBgByPage) {
      state.inpaintedBgByPage.delete(state.page);
    }
    state.selectedId = null;
    sourceText.value = "";
    translatedText.value = "";
    await updateErasedBg();
    refreshScreenCanvas();
    renderBlockList();
    setStatus("Se eliminaron todas las burbujas de la página.");
  }
});

// Crear burbuja manual en el centro de la pantalla
placeManualBtn.addEventListener("click", async () => {
  if (!state.kind) return setStatus("Carga un archivo primero.");
  const w = Math.round(pdfCanvas.width * 0.3);
  const h = 56;
  const newBox = {
    id: crypto.randomUUID(),
    x: Math.round((pdfCanvas.width - w) / 2),
    y: Math.round((pdfCanvas.height - h) / 2),
    w,
    h,
    source: sourceText.value || "Texto Original",
    text: translatedText.value || "Texto Traducido",
    bg: "transparent",
    color: textColor.value,
    strokeColor: "transparent",
    strokeWidth: 0,
    fontSize: Number(fontSize.value),
    fontFamily: fontFamily.value,
    fontStyle: state.italic ? "italic" : "normal",
    fontWeight: state.bold ? "800" : "700",
    eraseMode: "area"
  };
  newBox.bg = sampleBgColorAround(newBox);
  newBox.color = sampleTextColor(newBox);
  getPageBoxes().push(newBox);
  selectBox(newBox.id);
  await updateErasedBg();
  refreshScreenCanvas();
  setStatus("Burbuja manual creada.");
});

// Traducción automática rápida
autoTranslateAll.addEventListener("click", () => {
  autoTranslateAllPages().catch(err => setStatus(`Error: ${err.message}`));
});

// Botones de exportación
exportPng.addEventListener("click", () => {
  exportCurrentPng().catch(err => setStatus(`Error: ${err.message}`));
});

exportPdf.addEventListener("click", () => {
  exportCurrentPdf().catch(err => setStatus(`Error: ${err.message}`));
});

exportAllPdf.addEventListener("click", () => {
  exportFullPdf().catch(err => setStatus(`Error: ${err.message}`));
});

printPage.addEventListener("click", () => window.print());

// Mobile menu toggle
const mobileMenuBtn = $("#mobileMenuBtn");
const sidebar = $(".sidebar");if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
        mobileMenuBtn.setAttribute("aria-expanded", sidebar.classList.contains("open"));
    });
    // Close sidebar when clicking outside on mobile
    document.addEventListener("click", (e) => {
        if (window.innerWidth <= 1024 && sidebar.classList.contains("open") &&
            !sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
            sidebar.classList.remove("open");
            mobileMenuBtn.setAttribute("aria-expanded", "false");
        }
    });
}

// Fit page to viewport - calculates correct PDF.js render scale and re-renders
function fitPageToStage() {
  if (!state.kind || state.kind !== "pdf") return;
  const stageWrapEl = $("#stageWrap");
  const canvas = $("#pdfCanvas");
  if (!stageWrapEl || !canvas) return;
  
  state.pdf.getPage(state.page).then(pdfPage => {
    const viewport1 = pdfPage.getViewport({ scale: 1.0 });
    const pageWidth = viewport1.width;
    const pageHeight = viewport1.height;
    
    const wrapRect = stageWrapEl.getBoundingClientRect();
    const padding = 40;
    const availableW = wrapRect.width - padding;
    const availableH = wrapRect.height - padding;
    
    const scaleX = availableW / pageWidth;
    const scaleY = availableH / pageHeight;
    const newScale = Math.min(scaleX, scaleY, 3);
    
    if (newScale > 0 && Math.abs(newScale - state.scale) > 0.05) {
      state.scale = newScale;
      showToast(`Zoom ajustado a ${Math.round(newScale * 100)}%`, "info", 2000);
      renderPage(state.page);
    }
    stage.scrollIntoView({ block: "start", inline: "center", behavior: "smooth" });
  }).catch(err => setStatus(`Error ajustando zoom: ${err.message}`));
}
fitPage.addEventListener("click", fitPageToStage);// Reset zoom on double-click canvas (back to default 1.8)
  $("#pdfCanvas")?.addEventListener("dblclick", (e) => {
  if (!state.kind || state.kind !== "pdf") return;
  if (Math.abs(state.scale - 1.8) > 0.05) {
    state.scale = 1.8;
    showToast("Zoom restablecido al 180% (default)", "info", 1500);
    renderPage(state.page).catch(err => console.error("[dblclick] Error:", err));
  }
});

// Configuración inicial del cursor de overlay
function bootApp() {
  overlay.className = "overlay drawing";
  setStatus("Listo para comenzar. Carga un archivo PDF o Imagen.");
  //console.log("[BOOT] Event listeners attached, fileInput:", !!fileInput);
  if (fileInput) {
    //console.log("[BOOT] fileInput.accept:", fileInput.accept);
  }
}
if (document.readyState !== "loading") {
  bootApp();
} else {
  document.addEventListener("DOMContentLoaded", bootApp);
}

// Compatibilidad con Node.js para tests: exportar funciones si se requiere como módulo
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        // Funciones exportadas para tests automatizados
    };
}