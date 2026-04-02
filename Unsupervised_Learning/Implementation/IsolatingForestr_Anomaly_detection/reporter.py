"""
reporter.py - Generates a human-readable HTML + JSON report summarising
              the entire pipeline run.
"""

import os
import json
import logging
from datetime import datetime
from config import REPORT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def generate_report(report: dict, best_params: dict,
                    feature_names: list[str]) -> str:
    """
    Writes reports/summary.json  and  reports/report.html.
    Returns path to the HTML report.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_obj = {
        "timestamp"    : timestamp,
        "best_params"  : best_params,
        "n_features"   : len(feature_names),
        "feature_names": feature_names,
        "metrics"      : report,
    }

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = os.path.join(REPORT_DIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(report_obj, f, indent=2)
    logger.info("JSON report → %s", json_path)

    # ── HTML ─────────────────────────────────────────────────────────────────
    html_path = os.path.join(REPORT_DIR, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_build_html(report_obj))
    logger.info("HTML report → %s", html_path)

    return html_path


# ──────────────────────────────────────────────────────────────────────────────

def _build_html(obj: dict) -> str:
    metrics = obj["metrics"]
    params  = obj["best_params"]

    def row(k, v):
        return f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"

    metric_rows = "\n".join(row(k, v) for k, v in metrics.items())
    param_rows  = "\n".join(row(k, v) for k, v in params.items())

    # Relative paths to plots
    plot_files  = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".png")
    ])
    plot_html   = "\n".join(
        f'<div class="plot">'
        f'<p><em>{f}</em></p>'
        f'<img src="../outputs/{f}" alt="{f}"/>'
        f'</div>'
        for f in plot_files
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Isolation Forest — Anomaly Detection Report</title>
<style>
  body  {{ font-family: "Segoe UI", Arial, sans-serif; margin: 40px;
           background: #f8f9fa; color: #212529; }}
  h1   {{ color: #343a40; border-bottom: 3px solid #DD4444; padding-bottom: 8px;}}
  h2   {{ color: #495057; margin-top: 32px; }}
  table{{ border-collapse: collapse; width: 100%; max-width: 600px; }}
  th,td{{ padding: 8px 14px; border: 1px solid #dee2e6; text-align: left; }}
  th   {{ background: #343a40; color: #fff; }}
  tr:nth-child(even){{ background:#e9ecef; }}
  .plot{{ margin: 20px 0; background:#fff; padding:12px;
          border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.12); }}
  .plot img{{ max-width:100%; border-radius:4px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:4px;
            background:#DD4444; color:#fff; font-weight:bold; font-size:.85em;}}
</style>
</head>
<body>
<h1>🔍 Isolation Forest — Anomaly Detection Report</h1>
<p>Generated: <em>{obj["timestamp"]}</em></p>

<h2>⚙️ Best Hyper-parameters</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>{param_rows}</tbody>
</table>

<h2>📊 Evaluation Metrics</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{metric_rows}</tbody>
</table>

<h2>📈 Visualisations</h2>
{plot_html}

<hr/>
<p style="color:#999;font-size:.8em">
  Isolation Forest Anomaly Detection · Powered by scikit-learn
</p>
</body>
</html>
"""