"""
Module 6: Report Generator
===========================
Assembles a rich HTML report embedding all figures and metrics.
"""

import os
import base64
from datetime import datetime


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _img_tag(path: str, caption: str = "", width: str = "100%") -> str:
    if path is None or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"""
    <figure>
      <img src="data:image/png;base64,{b64}" style="width:{width};border-radius:8px;
           box-shadow:0 4px 12px rgba(0,0,0,.12);margin:10px 0" alt="{caption}">
      <figcaption style="text-align:center;color:#6c757d;font-size:.9rem;margin-top:6px">
        {caption}
      </figcaption>
    </figure>"""


def generate_html_report(
    metrics: dict,        # {model_name: {accuracy, f1, roc_auc}}
    tree_info: dict,      # {depth, leaves}
    plot_paths: dict,     # {key: filepath}
    feature_names: list,
    dataset_info: dict,   # {rows, cols, missing, pos_pct}
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Metric cards
    m = metrics.get("Tuned", {})
    card_html = ""
    for label, key, color in [
        ("Accuracy",  "accuracy", "#3498db"),
        ("F1 Score",  "f1",       "#e74c3c"),
        ("ROC AUC",   "roc_auc",  "#2ecc71"),
    ]:
        val = m.get(key, 0)
        card_html += f"""
        <div style="background:{color};color:white;border-radius:12px;
                    padding:20px 30px;text-align:center;flex:1;min-width:160px;
                    box-shadow:0 4px 12px rgba(0,0,0,.15)">
          <div style="font-size:2.4rem;font-weight:800">{val:.3f}</div>
          <div style="font-size:1rem;margin-top:4px;opacity:.9">{label}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Heart Disease – Decision Tree Report</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0 }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#f0f2f5; color:#2d3436 }}
    .page {{ max-width:1100px; margin:0 auto; padding:30px 20px }}
    h1 {{ font-size:2.2rem; color:#2c3e50; margin-bottom:6px }}
    h2 {{ font-size:1.4rem; color:#34495e; margin:32px 0 14px;
          border-left:4px solid #3498db; padding-left:10px }}
    .subtitle {{ color:#636e72; margin-bottom:28px }}
    .card-row {{ display:flex; gap:18px; flex-wrap:wrap; margin-bottom:30px }}
    .info-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:30px }}
    .info-box {{ background:white; border-radius:10px; padding:16px; text-align:center;
                 box-shadow:0 2px 8px rgba(0,0,0,.08) }}
    .info-box .val {{ font-size:1.6rem; font-weight:700; color:#2c3e50 }}
    .info-box .lbl {{ font-size:.85rem; color:#636e72; margin-top:2px }}
    .section {{ background:white; border-radius:12px; padding:24px;
                margin-bottom:24px; box-shadow:0 2px 8px rgba(0,0,0,.07) }}
    table {{ width:100%; border-collapse:collapse; font-size:.95rem }}
    th {{ background:#2c3e50; color:white; padding:10px 14px; text-align:left }}
    td {{ padding:9px 14px; border-bottom:1px solid #f0f2f5 }}
    tr:hover td {{ background:#f8f9fa }}
    .badge {{ display:inline-block; background:#3498db; color:white;
              border-radius:4px; padding:2px 8px; font-size:.8rem }}
    footer {{ text-align:center; color:#b2bec3; margin-top:30px; font-size:.85rem }}
  </style>
</head>
<body>
<div class="page">

  <h1>🌳 Heart Disease Classification</h1>
  <p class="subtitle">Decision Tree · UCI Heart Disease Dataset · Generated {now}</p>

  <!-- KPI cards -->
  <div class="card-row">
    {card_html}
  </div>

  <!-- Dataset info -->
  <div class="info-grid">
    <div class="info-box"><div class="val">{dataset_info.get('rows','-')}</div><div class="lbl">Total Samples</div></div>
    <div class="info-box"><div class="val">{dataset_info.get('cols','-')}</div><div class="lbl">Features</div></div>
    <div class="info-box"><div class="val">{dataset_info.get('missing','-')}</div><div class="lbl">Missing Values</div></div>
    <div class="info-box"><div class="val">{dataset_info.get('pos_pct',0):.1f}%</div><div class="lbl">Disease Rate</div></div>
  </div>

  <!-- Tree info -->
  <div class="section">
    <h2>Best Model Parameters</h2>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Tree Depth</td><td><span class="badge">{tree_info.get('depth','—')}</span></td></tr>
      <tr><td>Leaf Nodes</td><td><span class="badge">{tree_info.get('leaves','—')}</span></td></tr>
      <tr><td>Features Used</td><td>{', '.join(feature_names)}</td></tr>
    </table>
  </div>

  <!-- Model comparison -->
  <div class="section">
    <h2>Model Comparison</h2>
    <table>
      <tr><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>ROC AUC</th></tr>
      {"".join(
        f'<tr><td>{name}</td><td>{v.get("accuracy",0):.4f}</td>'
        f'<td>{v.get("f1",0):.4f}</td><td>{v.get("roc_auc",0):.4f}</td></tr>'
        for name, v in metrics.items()
      )}
    </table>
    {_img_tag(plot_paths.get("comparison"), "Model Comparison")}
  </div>

  <!-- EDA section -->
  <div class="section">
    <h2>Exploratory Data Analysis</h2>
    {_img_tag(plot_paths.get("class_dist"),    "Class Distribution")}
    {_img_tag(plot_paths.get("distributions"), "Feature Distributions by Class")}
    {_img_tag(plot_paths.get("heatmap"),       "Correlation Heatmap")}
    {_img_tag(plot_paths.get("boxplots"),      "Boxplots of Key Features")}
  </div>

  <!-- Evaluation section -->
  <div class="section">
    <h2>Model Evaluation</h2>
    {_img_tag(plot_paths.get("confusion"),   "Confusion Matrix",         "60%")}
    {_img_tag(plot_paths.get("roc_pr"),      "ROC and PR Curves")}
    {_img_tag(plot_paths.get("cv_scores"),   "Cross-Validation Scores")}
    {_img_tag(plot_paths.get("depth_acc"),   "Tree Depth vs Accuracy")}
  </div>

  <!-- Feature importance -->
  <div class="section">
    <h2>Feature Importance</h2>
    {_img_tag(plot_paths.get("importance"), "Feature Importance (Gini)")}
  </div>

  <!-- Tree visualisation -->
  <div class="section">
    <h2>Decision Tree Visualisation (first 3 levels)</h2>
    {_img_tag(plot_paths.get("tree_viz"), "Decision Tree Structure")}
  </div>

  <footer>
    Heart Disease Decision Tree Project · Built with scikit-learn · {now}
  </footer>
</div>
</body>
</html>"""

    report_path = os.path.join(OUT_DIR, "report.html")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] HTML report saved → {report_path}")
    return report_path
