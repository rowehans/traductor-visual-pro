/**
 * config.js — Client configuration.
 * Timeout defaults overridden by /api/config at runtime.
 */

/** @type {CLIENT_CONFIG} */
export const CLIENT_CONFIG = {
  OCR_SCALE: 1.2, // Escala de render PDF para OCR; el server la baja en modo_cpu
  TIMEOUT_OPENCV_INIT_MS: 15000,
  TIMEOUT_PDFJS_CDN_MS: 10000,
  TIMEOUT_PDFJS_ES_MODULE_MS: 10000,
  TIMEOUT_PDF_RENDER_MS: 60000,
  TIMEOUT_TRANSLATE_MS: 30000,
  TIMEOUT_TRANSLATE_BATCH_MS: 60000,
  TIMEOUT_PROCESS_PAGE_MS: 300000,
  TIMEOUT_INPAINTED_IMAGE_MS: 15000,
  TIMEOUT_EXPORT_REVOKE_MS: 10000,
  TIMEOUT_CDN_LOAD_MS: 8000,
};

/**
 * Fetches server config to override client defaults.
 * Non-blocking — silently falls back to defaults on error.
 */
export async function fetchClientConfig() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (typeof data.ocr_scale === "number" && data.ocr_scale > 0) {
      CLIENT_CONFIG.OCR_SCALE = data.ocr_scale;
    }
    if (data.timeouts_ms) {
      Object.assign(CLIENT_CONFIG, {
        TIMEOUT_OPENCV_INIT_MS: data.timeouts_ms.opencv_init,
        TIMEOUT_PDFJS_CDN_MS: data.timeouts_ms.pdfjs_cdn,
        TIMEOUT_PDFJS_ES_MODULE_MS: data.timeouts_ms.pdfjs_es_module,
        TIMEOUT_PDF_RENDER_MS: data.timeouts_ms.pdf_render,
        TIMEOUT_TRANSLATE_MS: data.timeouts_ms.translate,
        TIMEOUT_TRANSLATE_BATCH_MS: data.timeouts_ms.translate_batch,
        TIMEOUT_PROCESS_PAGE_MS: data.timeouts_ms.process_page,
        TIMEOUT_INPAINTED_IMAGE_MS: data.timeouts_ms.inpainted_image,
        TIMEOUT_EXPORT_REVOKE_MS: data.timeouts_ms.export_revoke,
        TIMEOUT_CDN_LOAD_MS: data.timeouts_ms.cdn_load,
      });
      console.log("[config] Timeouts actualizados desde el servidor");
    }
  } catch (e) {
    console.warn("[config] Usando timeouts por defecto:", e.message);
  }
}
