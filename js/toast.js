/**
 * toast.js — Toast notification system.
 * Pure DOM manipulation, no app state dependency.
 */

let container = null;

function initToastContainer() {
  if (document.getElementById("toastContainer")) return;
  const c = document.createElement("div");
  c.id = "toastContainer";
  c.className = "toast-container";
  c.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 600;
    display: flex;
    flex-direction: column;
    gap: 10px;
    pointer-events: none;
  `;
  document.body.appendChild(c);
  container = c;
}

const ICONS = {
  success:
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
  error:
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
  warning:
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--warning)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L21.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  info:
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--info)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
};

/**
 * Muestra un toast de notificación.
 * @param {string} message
 * @param {'success'|'error'|'warning'|'info'} [type='info']
 * @param {number} [duration=4000] - 0 para persistente
 * @returns {HTMLElement} El elemento toast
 */
export function showToast(message, type = "info", duration = 4000) {
  initToastContainer();
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.style.cssText = `
    pointer-events: auto;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 280px;
    max-width: 400px;
    box-shadow: var(--shadow-xl);
    animation: slideInRight 0.4s var(--transition-bounce);
  `;

  toast.innerHTML = `
    <div class="toast-icon" style="flex-shrink:0; width:20px; height:20px;">
      ${ICONS[type] || ICONS.info}
    </div>
    <div class="toast-message" style="flex:1; font-size:13px; color:var(--text-primary); line-height:1.4;"></div>
    <button class="toast-close" style="color:var(--text-muted); cursor:pointer; padding:4px; transition:var(--transition-fast); flex-shrink:0;" aria-label="Cerrar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  `;

  const c = document.getElementById("toastContainer");
  if (c) c.appendChild(toast);

  // Los mensajes pueden incluir texto devuelto por la API o por una
  // excepción del navegador. Asignarlos como texto evita que ese contenido
  // se interprete como HTML dentro de la interfaz local.
  const messageEl = toast.querySelector(".toast-message");
  if (messageEl) messageEl.textContent = String(message ?? "");

  const closeBtn = toast.querySelector(".toast-close");
  closeBtn.addEventListener("click", () => removeToast(toast));

  if (duration > 0) {
    setTimeout(() => {
      if (toast.parentNode) removeToast(toast);
    }, duration);
  }

  return toast;
}

function removeToast(toast) {
  toast.style.animation = "slideOutRight 0.3s ease-in forwards";
  setTimeout(() => toast.remove(), 300);
}
