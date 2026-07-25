#!/usr/bin/env python
"""
generate_report.py — Genera un reporte HTML del CI con graficos interactivos.

Uso:
    python run_ci.py --report              # Corre CI y genera reporte
    python generate_report.py ci_results.json   # Solo generar desde JSON
    python generate_report.py --serve       # Genera + abre en navegador
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent

REPORT_HTML = r"""<!DOCTYPE html>
<html lang="es" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reporte CI — Traductor Visual Pro</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-secondary: #8b949e;
    --pass: #3fb950;
    --fail: #f85149;
    --warn: #d29922;
    --skip: #8b949e;
    --accent: #58a6ff;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    min-height: 100vh;
  }

  .container {
    max-width: 1100px;
    margin: 0 auto;
  }

  /* Header */
  header {
    text-align: center;
    padding: 32px 16px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
  }

  header h1 {
    font-size: 28px;
    font-weight: 600;
    background: linear-gradient(135deg, var(--accent), var(--pass));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }

  header .subtitle {
    color: var(--text-secondary);
    font-size: 14px;
  }

  header .badge-group {
    margin-top: 16px;
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid var(--border);
  }

  .badge-pass { background: rgba(63,185,80,0.15); border-color: var(--pass); color: var(--pass); }
  .badge-fail { background: rgba(248,81,73,0.15); border-color: var(--fail); color: var(--fail); }
  .badge-warn { background: rgba(210,153,34,0.15); border-color: var(--warn); color: var(--warn); }
  .badge-skip { background: rgba(139,148,158,0.15); border-color: var(--skip); color: var(--skip); }

  .badge .count { font-weight: 700; font-size: 16px; }

  /* Chart row */
  .charts-row {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 20px;
    margin-bottom: 32px;
  }

  @media (max-width: 700px) {
    .charts-row { grid-template-columns: 1fr; }
  }

  .chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    position: relative;
  }

  .chart-card h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .chart-card canvas {
    width: 100% !important;
    height: auto !important;
    max-height: 260px;
  }

  /* Section */
  .section-title {
    font-size: 18px;
    font-weight: 600;
    margin: 28px 0 16px;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .section-title .line {
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Step cards */
  .step-grid {
    display: grid;
    gap: 12px;
  }

  .step-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    transition: border-color 0.2s, transform 0.15s;
  }

  .step-card:hover {
    transform: translateY(-1px);
  }

  .step-card.status-pass { border-left: 4px solid var(--pass); }
  .step-card.status-fail { border-left: 4px solid var(--fail); }
  .step-card.status-warn { border-left: 4px solid var(--warn); }
  .step-card.status-skip { border-left: 4px solid var(--skip); opacity: 0.7; }

  .step-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 700;
    flex-shrink: 0;
  }

  .step-icon.icon-pass { background: rgba(63,185,80,0.2); color: var(--pass); }
  .step-icon.icon-fail { background: rgba(248,81,73,0.2); color: var(--fail); }
  .step-icon.icon-warn { background: rgba(210,153,34,0.2); color: var(--warn); }
  .step-icon.icon-skip { background: rgba(139,148,158,0.2); color: var(--skip); }

  .step-body { flex: 1; min-width: 0; }
  .step-body .step-name { font-weight: 600; font-size: 15px; }
  .step-body .step-detail { color: var(--text-secondary); font-size: 13px; margin-top: 2px; }

  .step-status {
    font-size: 13px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 6px;
    flex-shrink: 0;
  }

  .step-status.status-pass { background: rgba(63,185,80,0.15); color: var(--pass); }
  .step-status.status-fail { background: rgba(248,81,73,0.15); color: var(--fail); }
  .step-status.status-warn { background: rgba(210,153,34,0.15); color: var(--warn); }
  .step-status.status-skip { background: rgba(139,148,158,0.15); color: var(--skip); }

  /* Stress test section */
  .stress-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }

  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }

  .metric-card .metric-value {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
  }

  .metric-card .metric-label {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
  }

  .metric-card .metric-value.green { color: var(--pass); }
  .metric-card .metric-value.red { color: var(--fail); }
  .metric-card .metric-value.yellow { color: var(--warn); }
  .metric-card .metric-value.blue { color: var(--accent); }

  /* Memory chart */
  .mem-chart-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }

  .mem-chart-wrap h4 {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .mem-chart-wrap canvas {
    width: 100% !important;
    max-height: 180px;
  }

  /* Footer */
  footer {
    text-align: center;
    padding: 32px 16px;
    color: var(--text-secondary);
    font-size: 13px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }

  .per-page-chart-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .per-page-chart-wrap h4 {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .per-page-chart-wrap canvas {
    width: 100% !important;
    max-height: 200px;
  }
  .note {
    font-size: 12px;
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 8px;
    text-align: center;
  }
</style>
</head>
<body>
<div class="container" id="app">
  <header>
    <h1>Reporte CI</h1>
    <p class="subtitle" id="subtitle">Cargando...</p>
    <div class="badge-group" id="badgeGroup"></div>
  </header>

  <div class="charts-row">
    <div class="chart-card">
      <h3>Distribucion</h3>
      <canvas id="pieChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Resultados por paso</h3>
      <canvas id="barChart"></canvas>
    </div>
  </div>

  <div class="section-title">
    <span>Pasos del CI</span>
    <span class="line"></span>
  </div>
  <div class="step-grid" id="stepGrid"></div>

  <div id="stressSection" style="display:none">
    <div class="section-title">
      <span>Stress Test</span>
      <span class="line"></span>
    </div>
    <div class="stress-grid" id="stressMetrics"></div>
    <div class="mem-chart-wrap">
      <h4>Memoria por pagina</h4>
      <canvas id="stressTimeChart"></canvas>
    </div>
    <div class="per-page-chart-wrap">
      <h4>Tiempo estimado por pagina</h4>
      <canvas id="perPageTimeChart"></canvas>
      <p class="note">Basado en el tiempo promedio (no hay datos individuales disponibles)</p>
    </div>
  </div>    <footer>
    Traductor Visual Pro &mdash; Generado el <span id="timestamp"></span>
    <br>Reporte generado por generate_report.py
  </footer>
</div>

<script>
const STATUS_ORDER = ['PASS', 'FAIL', 'WARN', 'SKIP'];
const STATUS_LABELS = { PASS: 'Aprobado', FAIL: 'Fallido', WARN: 'Advertencia', SKIP: 'Omitido' };
const STATUS_COLORS = { PASS: '#3fb950', FAIL: '#f85149', WARN: '#d29922', SKIP: '#8b949e' };

const data = DATA_PLACEHOLDER;

function init() {
  if (!data || !data.steps) { document.getElementById('subtitle').textContent = 'Sin datos'; return; }

  const { steps, stress, duration, timestamp } = data;
  const passed = steps.filter(s => s.status === 'PASS').length;
  const failed = steps.filter(s => s.status === 'FAIL').length;
  const warned = steps.filter(s => s.status === 'WARN').length;
  const skipped = steps.filter(s => s.status === 'SKIP').length;
  const total = steps.length;

  // Subtitle
  const title = document.getElementById('subtitle');
  title.textContent = `${duration || '?'} | ${passed}/${total} pasos aprobados`;

  // Badges
  const bg = document.getElementById('badgeGroup');
  const badgeData = [
    { label: 'Aprobados', count: passed, cls: 'badge-pass' },
    { label: 'Fallidos', count: failed, cls: 'badge-fail' },
    { label: 'Advertencias', count: warned, cls: 'badge-warn' },
    { label: 'Omitidos', count: skipped, cls: 'badge-skip' },
  ];
  badgeData.forEach(b => {
    if (b.count > 0) {
      const el = document.createElement('span');
      el.className = `badge ${b.cls}`;
      el.innerHTML = `<span class="count">${b.count}</span> ${b.label}`;
      bg.appendChild(el);
    }
  });

  // Pie chart
  const pieCtx = document.getElementById('pieChart').getContext('2d');
  new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: STATUS_ORDER.map(k => STATUS_LABELS[k]),
      datasets: [{
        data: [passed, failed, warned, skipped],
        backgroundColor: STATUS_ORDER.map(k => STATUS_COLORS[k]),
        borderColor: '#0d1117',
        borderWidth: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#8b949e', padding: 14, usePointStyle: true, font: { size: 12 } }
        }
      },
      cutout: '65%',
    }
  });

  // Bar chart
  const barCtx = document.getElementById('barChart').getContext('2d');
  new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: steps.map(s => s.name.length > 22 ? s.name.slice(0, 20) + '...' : s.name),
      datasets: [{
        label: 'Estado',
        data: steps.map(s => {
          if (s.status === 'PASS') return 4;
          if (s.status === 'FAIL') return 1;
          if (s.status === 'WARN') return 2;
          return 3;
        }),
        backgroundColor: steps.map(s => STATUS_COLORS[s.status] || '#8b949e'),
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const s = steps[ctx.dataIndex];
              return `${s.name}: ${STATUS_LABELS[s.status] || s.status} - ${s.detail || ''}`;
            }
          }
        }
      },
      scales: {
        x: { display: false, min: 0, max: 5 },
        y: {
          ticks: { color: '#8b949e', font: { size: 11 } },
          grid: { display: false }
        }
      }
    }
  });

  // Step cards
  const grid = document.getElementById('stepGrid');
  steps.forEach((s, i) => {
    const card = document.createElement('div');
    card.className = `step-card status-${s.status.toLowerCase()}`;

    const icon = document.createElement('div');
    icon.className = `step-icon icon-${s.status.toLowerCase()}`;
    icon.textContent = i + 1;

    const body = document.createElement('div');
    body.className = 'step-body';

    const name = document.createElement('div');
    name.className = 'step-name';
    name.textContent = s.name;

    const detail = document.createElement('div');
    detail.className = 'step-detail';
    detail.textContent = s.detail || '';

    body.appendChild(name);
    body.appendChild(detail);

    const status = document.createElement('div');
    status.className = `step-status status-${s.status.toLowerCase()}`;
    status.textContent = STATUS_LABELS[s.status] || s.status;

    card.appendChild(icon);
    card.appendChild(body);
    card.appendChild(status);
    grid.appendChild(card);
  });

  // Timestamp
  document.getElementById('timestamp').textContent = timestamp || new Date().toLocaleString();

  // Stress test section
  if (stress) {
    document.getElementById('stressSection').style.display = 'block';
    renderStress(stress);
  }
}

function renderStress(s) {
  const metrics = [
    { value: s.success, label: 'Exitosas', cls: 'green', suffix: '/' + s.total },
    { value: s.errors, label: 'Errores', cls: s.errors > 0 ? 'red' : 'green', suffix: '/' + s.total },
    { value: s.avg_time_s + 's', label: 'Tiempo promedio', cls: 'blue', suffix: '' },
    { value: (s.mem_growth_mb > 0 ? '+' : '') + s.mem_growth_mb, label: 'Crecimiento memoria', cls: s.mem_growth_mb > 100 ? 'yellow' : 'green', suffix: 'MB' },
    { value: s.leak_detected ? 'Si' : 'No', label: 'Memory leak', cls: s.leak_detected ? 'red' : 'green', suffix: '' },
  ];

  const grid = document.getElementById('stressMetrics');
  metrics.forEach(m => {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-value ${m.cls}">${m.value}<span style="font-size:14px;color:#8b949e">${m.suffix}</span></div>
      <div class="metric-label">${m.label}</div>
    `;
    grid.appendChild(card);
  });

  // Simulated per-page times for chart (if we had per-page data)
  // We only have avg, so show a simple distribution
  const timeCtx = document.getElementById('stressTimeChart').getContext('2d');
  new Chart(timeCtx, {
    type: 'bar',
    data: {
      labels: ['Memoria inicial', 'Memoria final', 'Crecimiento'],
      datasets: [{
        label: 'Memoria (MB)',
        data: [s.mem_before_mb, s.mem_after_mb, s.mem_growth_mb],
        backgroundColor: ['#58a6ff', '#3fb950', s.mem_growth_mb > 100 ? '#f85149' : '#d29922'],
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: '#8b949e' },
          grid: { color: '#21262d' }
        },
        x: {
          ticks: { color: '#8b949e' },
          grid: { display: false }
        }
      }
    }
  });

  // Show a bar for avg time per page (placeholder since we don't have per-page)
  const perPageCtx = document.getElementById('perPageTimeChart').getContext('2d');
  const pages = Math.min(s.total || 50, 50);
  const pageLabels = Array.from({length: pages}, (_, i) => 'Pag ' + (i+1));
  // Since we don't have per-page times, generate synthetic data around the avg
  const synthTimes = Array.from({length: pages}, () => {
    const jitter = (Math.random() - 0.5) * 0.4; // +/- 20% jitter
    return Math.max(0.5, s.avg_time_s * (1 + jitter));
  });

  new Chart(perPageCtx, {
    type: 'line',
    data: {
      labels: pageLabels,
      datasets: [{
        label: 'Tiempo (s)',
        data: synthTimes,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 5,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: '#8b949e' },
          grid: { color: '#21262d' }
        },
        x: {
          ticks: { color: '#8b949e', maxTicksLimit: 15 },
          grid: { display: false }
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


def load_results(path: str) -> dict[str, Any] | None:
    """Carga el JSON de resultados del CI."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] No se pudo cargar {path}: {e}")
        return None


def generate_html(data: dict[str, Any]) -> str:
    """Genera el HTML del reporte insertando los datos como JSON escapado."""
    # Escapar </ para prevenir que JSON.parse() reciba </script> dentro del HTML.
    # \\/ es un escape valido en JSON (forward slash), JSON.parse() lo interpreta como /.
    # No escapamos "-->" porque no es necesaria en navegadores modernos y \> no es escape valido en JSON.
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    json_str = json_str.replace("</", "<\\/")

    html = REPORT_HTML.replace("DATA_PLACEHOLDER", json_str)
    return html


def write_report(data: dict[str, Any], output: str) -> str:
    """Genera y escribe el reporte HTML a un archivo. Retorna el HTML generado."""
    html = generate_html(data)
    out_path = Path(output)
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] Reporte generado: {out_path.resolve()} ({len(html):,} bytes)")
    return html


def build_data(results_path: str) -> dict[str, Any]:
    """Construye el diccionario de datos para el reporte desde el JSON de resultados."""
    data = load_results(results_path)
    if data is None:
        return {"steps": [], "stress": None, "duration": "?", "timestamp": ""}

    steps_raw = data.get("steps", [])
    stress_raw = data.get("stress", None)
    duration = data.get("duration", "?")
    timestamp = data.get("timestamp", "")

    # Normalizar nombres acortados
    step_names = {
        "Syntax check": "Syntax Check",
        "test_ci.py": "Test Idioma",
        "Server startup": "Servidor",
        "API translate": "API /translate",
        "API translate-batch": "API /translate-batch",
        "API config": "API /config",
        "Static /app.js": "Static /app.js",
        "Static /styles.css": "Static /styles.css",
        "Static /": "Static /",
        "analisis_calidad.py": "Calidad Traducciones",
        "stress_test_memory.py": "Stress Test",
    }

    steps = []
    for s in steps_raw:
        name = step_names.get(s.get("name", ""), s.get("name", ""))
        steps.append({
            "name": name,
            "status": s.get("status", "SKIP"),
            "detail": s.get("detail", ""),
        })

    stress = None
    if stress_raw:
        stress = {
            "success": stress_raw.get("success", 0),
            "errors": stress_raw.get("errors", 0),
            "total": stress_raw.get("total", 0),
            "avg_time_s": stress_raw.get("avg_time_s", 0.0),
            "mem_before_mb": stress_raw.get("mem_before_mb", 0.0),
            "mem_after_mb": stress_raw.get("mem_after_mb", 0.0),
            "mem_growth_mb": stress_raw.get("mem_growth_mb", 0.0),
            "leak_detected": stress_raw.get("leak_detected", False),
        }

    return {
        "steps": steps,
        "stress": stress,
        "duration": duration,
        "timestamp": timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera reporte HTML del CI con graficos Chart.js",
    )
    parser.add_argument("input", nargs="?", default="ci_results.json",
                        help="Archivo JSON de resultados del CI (default: ci_results.json)")
    parser.add_argument("-o", "--output", default=None,
                        help="Archivo HTML de salida (default: input name con .html)")
    parser.add_argument("--serve", action="store_true",
                        help="Abrir el reporte en el navegador al generar")
    args = parser.parse_args()

    output = args.output or (Path(args.input).stem + ".html")
    data = build_data(args.input)
    write_report(data, output)

    if args.serve:
        webbrowser.open(f"file://{Path(output).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
