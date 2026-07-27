/**
 * theme.js — Theme management (dark/light).
 * Depends on: state object injected via callback.
 */

/** @type {'dark'|'light'} */
let currentTheme = "dark";
let onToggle = null;

/**
 * Inicializa el tema desde localStorage y crea el botón toggle.
 * @param {object} stateRef - Referencia al state de la app (se actualiza state.theme)
 * @param {Function} onToggleCallback - Se llama tras cada toggle
 */
export function initTheme(stateRef, onToggleCallback) {
  onToggle = onToggleCallback || null;
  const saved = localStorage.getItem("theme") || "dark";
  currentTheme = saved;
  stateRef.theme = saved;
  document.documentElement.setAttribute("data-theme", saved);

  if (!document.getElementById("themeToggle")) {
    const btn = document.createElement("button");
    btn.id = "themeToggle";
    btn.className = "theme-toggle";
    btn.setAttribute("aria-label", "Cambiar tema");
    btn.title = "Cambiar tema (T)";
    btn.innerHTML = `
      <svg class="icon-sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none;">
        <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/>
        <line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/>
        <line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
      </svg>
      <svg class="icon-moon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>
    `;
    btn.addEventListener("click", () => toggleTheme(stateRef));

    const topbarActions = document.querySelector(".topbar-actions");
    if (topbarActions) {
      topbarActions.insertBefore(btn, topbarActions.firstChild);
    }
  }

  updateIcons();
}

/**
 * Cambia entre tema oscuro y claro.
 * @param {object} stateRef
 */
export function toggleTheme(stateRef) {
  currentTheme = currentTheme === "dark" ? "light" : "dark";
  stateRef.theme = currentTheme;
  localStorage.setItem("theme", currentTheme);
  document.documentElement.setAttribute("data-theme", currentTheme);
  updateIcons();
  if (onToggle) onToggle(currentTheme);
}

function updateIcons() {
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  const sun = btn.querySelector(".icon-sun");
  const moon = btn.querySelector(".icon-moon");
  if (currentTheme === "dark") {
    if (sun) sun.style.display = "none";
    if (moon) moon.style.display = "block";
  } else {
    if (sun) sun.style.display = "block";
    if (moon) moon.style.display = "none";
  }
}

/**
 * @returns {'dark'|'light'}
 */
export function getCurrentTheme() {
  return currentTheme;
}
