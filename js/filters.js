/**
 * filters.js — Noise patterns and block filtering for manga OCR.
 * Synchronized with config.py (MARGIN_NOISE_PATTERNS, WATERMARK_PATTERNS).
 */

import { getBlockText } from "./utils.js";

/**
 * Patrones de ruido en márgenes (fecha/hora/numeración).
 * Solo se filtran si el bloque está en el margen superior o inferior de la página.
 */
export const MARGIN_NOISE_PATTERNS = [
  // Fechas: 13/7/26, 13.07.2026, 13-7-26
  /\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}/,
  // Fechas con error de OCR: 13/7126, 13.726
  /\d{1,2}[/.\-]\d{1,2}1?\d{2}\b/,
  // Horas: 4:58 P.M., 4.58 p.m., 4:58
  /\d{1,2}[:.]\d{2}\s*([ap]\.?\s?m\.?)?/i,
  // Numeración de página: 3/128, "3 de 128", "Pág. 3"
  /^\d{1,4}\s*\/\s*\d{1,4}$/,
  /\b\d{1,4}\s+de\s+\d{1,4}\b/i,
  /\bp[aá]g(?:ina)?\.?\s?\d{1,4}\b/i,
  /\b(?:p[aá]g(?:ina)?|page)\s+\d+\s+(?:de|of)\s+\d+\b/i,
  // Timestamps y metadatos de exportación: 20260713-11032519C, 458pm.
  /\b\d{8,14}[A-Za-z0-9_\-]*\b/,
  /\b\d{1,6}\s*[,.]?\s*\d{1,4}\s*p\.?\s*m\.?\b/i,
  /\b\d{1,4}\s*p\.?\s*m\.?\b/i,
];

/**
 * Patrones de marcas de agua globales (URLs, sellos de escaneo).
 * Se filtran en CUALQUIER parte de la página.
 */
export const GLOBAL_NOISE_PATTERNS = [
  /https?:\/\/|www\.|\.(com|net|org|xyz|io)\b/i,
  /\bzonaolympus[\s-]?com\b/i,
  /\b1\s*[\s-]?c\s*[\s-]?2\s*[\s-]?e\b/i,         // Sello "1 C 2 E"
  // Broken "http://" — OCR mangles "https://" into "htps fo", "htp ://" etc.
  /\bhtps?\s*[:\s\/'"\\]/i,
  // Domain with underscore: "xyz_com", "site'com"
  /[a-z]+[_'"\s]\s*(?:com|net|org|xyz|io)\b/i,
];

/**
 * Filtra bloques de metadatos impresos y marcas de agua de grupos de escaneo.
 * @param {Array} blocks - Bloques con x, y, w, h, text/source
 * @param {number} pageHeight - Altura de la página en px
 * @param {Function} [getTextFn] - Función para extraer texto del bloque
 * @returns {Array}
 */
export function filterPageBlocks(blocks, pageHeight = 0, getTextFn = getBlockText) {
  // pageHeight = 0 desactiva el filtro de márgenes (no se puede calcular sin el canvas)
  // El caller (detectPageBlocks en app.js) pasa pdfCanvas.height explícitamente
  if (!blocks || !blocks.length) return blocks || [];
  const marginTop = pageHeight * 0.07;
  const marginBottom = pageHeight * 0.96;
  const getText = getTextFn;

  return blocks.filter(b => {
    const text = getText(b);
    if (!text) return true;

    // Filtrar marcas de agua globales
    if (GLOBAL_NOISE_PATTERNS.some(re => re.test(text))) {
      console.log(`[filtro] Marca de agua/URL descartada: "${text}"`);
      return false;
    }

    // Filtrar metadatos solo en márgenes
    const cy = (b.y || 0) + (b.h || 0) / 2;
    const inMargin = cy < marginTop || cy > marginBottom;
    if (inMargin && MARGIN_NOISE_PATTERNS.some(re => re.test(text))) {
      console.log(`[filtro] Metadato de margen descartado: "${text}" (y=${Math.round(cy)})`);
      return false;
    }

    return true;
  });
}
