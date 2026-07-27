/**
 * utils.js — Utility functions for the Traductor Visual Pro app.
 * Pure functions with no side effects or DOM dependencies.
 */

/**
 * Formatea milisegundos a "Xm Ys" o "Xs".
 * @param {number} ms
 * @returns {string}
 */
export function formatDuration(ms) {
  const sec = Math.ceil(ms / 1000);
  const min = Math.floor(sec / 60);
  const restSec = sec % 60;
  return min > 0 ? `${min}m ${restSec}s` : `${restSec}s`;
}

/**
 * Convierte un canvas a base64 (PNG).
 * @param {HTMLCanvasElement} canvas
 * @returns {string}
 */
export function canvasToBase64(canvas) {
  return canvas.toDataURL("image/png");
}

/**
 * Carga una imagen base64 en un canvas.
 * @param {string} b64 - Base64 data URL
 * @param {HTMLCanvasElement} targetCanvas
 * @returns {Promise<void>}
 */
export function loadBase64IntoCanvas(b64, targetCanvas) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const ctx = targetCanvas.getContext("2d");
      ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
      ctx.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = reject;
    img.src = b64;
  });
}

/**
 * Obtiene texto limpio de un bloque (source o text).
 * @param {object} b
 * @returns {string}
 */
export function getBlockText(b) {
  return String((b && (b.text || b.source)) || "").trim();
}

/**
 * Agrupa líneas de PDF en bloques de burbuja.
 * @param {Array} lines - Array de objetos con x, y, w, h, text, size
 * @returns {Array}
 */
export function mergeLinesIntoBlocks(lines) {
  const blocks = [];
  for (const line of lines.sort((a, b) => a.y - b.y || a.x - b.x)) {
    let block = blocks.find(x =>
      line.y - x.y1 < Math.max(20, line.h * 1.3) &&
      line.y - x.y1 > -line.h &&
      Math.abs((line.x + line.w / 2) - x.cx) < Math.max(line.w, x.w) * 0.6
    );
    if (!block) {
      blocks.push({
        lines: [line],
        x: line.x, y: line.y,
        x1: line.x + line.w, y1: line.y + line.h,
        cx: line.x + line.w / 2, w: line.w,
      });
    } else {
      block.lines.push(line);
      block.x = Math.min(block.x, line.x);
      block.y = Math.min(block.y, line.y);
      block.x1 = Math.max(block.x1, line.x + line.w);
      block.y1 = Math.max(block.y1, line.y + line.h);
      block.cx = (block.x + block.x1) / 2;
      block.w = block.x1 - block.x;
    }
  }
  return blocks.map(b => ({
    x: b.x, y: b.y, w: b.w, h: b.y1 - b.y,
    text: b.lines.map(l => l.text).join(" ").replace(/\s+/g, " ").trim(),
    size: Math.round(b.lines.reduce((s, l) => s + l.size, 0) / b.lines.length),
  }));
}

/**
 * Determina si un color RGB es claro (brillo > 128).
 * @param {string} col - Color en formato rgb(r,g,b) o cualquier string con dígitos
 * @returns {boolean}
 */
export function isLightColor(col) {
  try {
    const m = col.match(/\d+/g);
    if (!m) return true;
    const [r, g, b] = m.map(Number);
    return (r * 0.299 + g * 0.587 + b * 0.114) > 128;
  } catch (e) {
    return true;
  }
}
