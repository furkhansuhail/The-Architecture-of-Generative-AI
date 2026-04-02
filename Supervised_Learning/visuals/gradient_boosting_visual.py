"""
Self-contained HTML visual for Gradient Boosting.
7 interactive tabs: Overview, Algorithm, Walkthrough (stepper),
Learning Rate & Early Stopping, Loss Functions, Libraries (XGBoost/LightGBM/CatBoost), GB vs RF.
Requires Google Fonts CDN for typography (JetBrains Mono, DM Serif Display, DM Sans).
Embed via: st.components.v1.html(GB_VISUAL_HTML, height=GB_VISUAL_HEIGHT, scrolling=True)
"""

GB_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Gradient Boosting — Sequential Error Correction</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --accent: #f59e0b;
    --accent2: #06b6d4;
    --accent3: #10b981;
    --accent4: #f43f5e;
    --accent5: #818cf8;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --text-muted: #374151;
    --mono: 'JetBrains Mono', monospace;
    --serif: 'DM Serif Display', serif;
    --sans: 'DM Sans', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── HEADER ── */
  .header {
    padding: 36px 40px 0;
    display: flex;
    align-items: flex-start;
    gap: 20px;
  }
  .header-icon {
    font-size: 2.4rem;
    line-height: 1;
    margin-top: 4px;
  }
  .header-text h1 {
    font-family: var(--serif);
    font-size: 2rem;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.1;
  }
  .header-text h1 span { color: var(--accent); }
  .header-text p {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* ── TABS ── */
  .tab-bar {
    display: flex;
    gap: 0;
    padding: 28px 40px 0;
    border-bottom: 1px solid var(--border);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tab-bar::-webkit-scrollbar { display: none; }
  .tab-btn {
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 20px 12px;
    cursor: pointer;
    white-space: nowrap;
    transition: color 0.2s, border-color 0.2s;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .tab-btn .tab-num {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--text-muted);
    color: var(--text-dim);
    font-size: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s, color 0.2s;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .tab-btn.active .tab-num {
    background: var(--accent);
    color: #000;
  }

  /* ── PANELS ── */
  .panels { padding: 36px 40px 60px; }
  .panel { display: none; animation: fadeIn 0.3s ease; }
  .panel.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  /* ── SHARED COMPONENTS ── */
  .section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
  }
  .section-title {
    font-family: var(--serif);
    font-size: 1.7rem;
    color: var(--text);
    line-height: 1.2;
    margin-bottom: 14px;
  }
  .section-body {
    font-size: 0.88rem;
    line-height: 1.7;
    color: #94a3b8;
    max-width: 720px;
    margin-bottom: 32px;
  }
  .section-body strong { color: var(--text); font-weight: 600; }
  .section-body em { color: var(--accent); font-style: normal; font-family: var(--mono); font-size: 0.82rem; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 22px;
  }
  .card-title {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 10px;
  }
  .card p {
    font-size: 0.85rem;
    line-height: 1.65;
    color: #94a3b8;
  }
  .card p strong { color: var(--text); }

  .formula-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 16px 20px;
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--accent);
    line-height: 1.8;
    margin: 16px 0;
  }
  .formula-box .comment { color: var(--text-dim); font-style: italic; }

  /* ── STEP STEPPER (Panel 3) ── */
  .step-nav {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
  }
  .step-nav-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.8rem;
    padding: 8px 18px;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
  }
  .step-nav-btn:hover:not(:disabled) { background: var(--surface2); border-color: var(--accent); }
  .step-nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .step-indicator {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.05em;
  }
  .step-indicator span { color: var(--accent); font-weight: 700; }

  .step-panel { display: none; }
  .step-panel.active { display: block; animation: fadeIn 0.3s ease; }

  .step-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
  }
  .step-badge {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: var(--accent);
    color: #000;
    font-family: var(--mono);
    font-size: 0.9rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .step-badge.blue { background: var(--accent2); }
  .step-badge.green { background: var(--accent3); }
  .step-badge.pink { background: var(--accent4); }
  .step-badge.purple { background: var(--accent5); }

  .step-title {
    font-family: var(--serif);
    font-size: 1.35rem;
    color: var(--text);
  }

  /* ── TABLE ── */
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 0.8rem;
    margin: 16px 0;
  }
  .data-table th {
    background: var(--surface2);
    color: var(--text-dim);
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.68rem;
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid var(--border);
  }
  .data-table th:first-child { text-align: left; }
  .data-table td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    color: #94a3b8;
  }
  .data-table td:first-child { text-align: left; color: var(--text); font-weight: 600; }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table .highlight { color: var(--accent); font-weight: 700; }
  .data-table .pos { color: var(--accent3); }
  .data-table .neg { color: var(--accent4); }
  .data-table .neutral { color: var(--text-dim); }

  /* ── CANVAS/CHART ── */
  .chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin: 16px 0;
  }
  canvas { display: block; }

  /* ── PROGRESS BAR ── */
  .mse-bar-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0;
  }
  .mse-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    width: 80px;
    text-align: right;
    flex-shrink: 0;
  }
  .mse-bar-outer {
    flex: 1;
    height: 8px;
    background: var(--surface2);
    border-radius: 4px;
    overflow: hidden;
  }
  .mse-bar-inner {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
  }
  .mse-val {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent);
    width: 55px;
    flex-shrink: 0;
  }

  /* ── TREE VISUAL ── */
  .tree-svg-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    overflow: hidden;
  }

  /* ── PILLS / BADGES ── */
  .pill {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 100px;
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    color: var(--text-dim);
  }
  .pill.amber { border-color: #78350f; background: #1c1007; color: var(--accent); }
  .pill.cyan  { border-color: #164e63; background: #061b22; color: var(--accent2); }
  .pill.green { border-color: #064e3b; background: #051a13; color: var(--accent3); }
  .pill.pink  { border-color: #881337; background: #1a0613; color: var(--accent4); }
  .pill.purple { border-color: #3730a3; background: #0d0b2a; color: var(--accent5); }

  /* ── LOSS TABLE (Panel 5) ── */
  .loss-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 20px 0;
  }
  .loss-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
    transition: border-color 0.2s;
  }
  .loss-card:hover { border-color: var(--accent2); }
  .loss-card .loss-name {
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 6px;
  }
  .loss-card .loss-formula {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--accent);
    margin-bottom: 8px;
  }
  .loss-card .loss-residual {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent3);
    margin-bottom: 6px;
  }
  .loss-card .loss-use {
    font-size: 0.78rem;
    color: var(--text-dim);
    line-height: 1.5;
  }

  /* ── LIBRARY COMPARE (Panel 6) ── */
  .lib-compare {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    margin: 16px 0;
  }
  .lib-compare th {
    background: var(--surface2);
    padding: 12px 16px;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }
  .lib-compare th:first-child { text-align: left; color: var(--text-dim); }
  .lib-compare th.sklearn { color: var(--text-dim); }
  .lib-compare th.xgb { color: var(--accent); }
  .lib-compare th.lgbm { color: var(--accent2); }
  .lib-compare th.catb { color: var(--accent3); }
  .lib-compare td {
    padding: 10px 16px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    color: #94a3b8;
    font-family: var(--mono);
    font-size: 0.76rem;
  }
  .lib-compare td:first-child { text-align: left; color: var(--text); font-family: var(--sans); font-size: 0.82rem; }
  .lib-compare tr:last-child td { border-bottom: none; }
  .lib-compare .yes { color: var(--accent3); }
  .lib-compare .no  { color: #374151; }
  .lib-compare .part { color: var(--accent); }
  .lib-compare tr:hover td { background: rgba(255,255,255,0.02); }

  /* ── GB vs RF ── */
  .vs-grid { display: grid; grid-template-columns: 1fr 40px 1fr; gap: 0; align-items: start; }
  .vs-col {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
  }
  .vs-col.rf { border-top: 3px solid var(--accent2); }
  .vs-col.gb { border-top: 3px solid var(--accent); }
  .vs-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 14px;
  }
  .vs-label.rf { color: var(--accent2); }
  .vs-label.gb { color: var(--accent); }
  .vs-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.5;
  }
  .vs-item::before {
    content: '→';
    font-family: var(--mono);
    color: var(--text-dim);
    flex-shrink: 0;
    margin-top: 1px;
  }
  .vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--serif);
    font-size: 1.1rem;
    color: var(--text-dim);
    padding-top: 50px;
  }

  /* Annotation / callout */
  .callout {
    display: flex;
    gap: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent3);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.85rem;
    line-height: 1.65;
    color: #94a3b8;
  }
  .callout .callout-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
  .callout strong { color: var(--text); }

  /* ── ALGO STEP CONNECTOR ── */
  .algo-steps { position: relative; }
  .algo-step {
    display: flex;
    gap: 16px;
    margin-bottom: 0;
  }
  .algo-step-left {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
  }
  .algo-dot {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--surface2);
    border: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-dim);
    flex-shrink: 0;
    transition: background 0.3s, border-color 0.3s, color 0.3s;
  }
  .algo-dot.lit { background: var(--accent); border-color: var(--accent); color: #000; }
  .algo-line {
    width: 2px;
    flex: 1;
    min-height: 24px;
    background: var(--border);
    margin: 4px 0;
  }
  .algo-step-right {
    padding-bottom: 24px;
    flex: 1;
  }
  .algo-step-title {
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
    letter-spacing: 0.03em;
  }
  .algo-step-body {
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.6;
  }
  .algo-step-body .formula { font-family: var(--mono); color: var(--accent); font-size: 0.78rem; }

  /* tag */
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 600;
    margin-left: 6px;
    vertical-align: middle;
  }
  .tag.new { background: #1c1007; color: var(--accent); }

  /* Responsive */
  @media (max-width: 768px) {
    .header { padding: 20px 20px 0; }
    .tab-bar { padding: 16px 20px 0; }
    .panels { padding: 24px 20px 40px; }
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    .vs-grid { grid-template-columns: 1fr; }
    .vs-divider { display: none; }
    .loss-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">🚀</div>
  <div class="header-text">
    <h1>Gradient <span>Boosting</span></h1>
    <p>Sequential weak learners correcting each other's mistakes · XGBoost · LightGBM · CatBoost</p>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab(0)"><span class="tab-num">1</span> Overview</button>
  <button class="tab-btn" onclick="switchTab(1)"><span class="tab-num">2</span> Algorithm</button>
  <button class="tab-btn" onclick="switchTab(2)"><span class="tab-num">3</span> Walkthrough</button>
  <button class="tab-btn" onclick="switchTab(3)"><span class="tab-num">4</span> Learning Rate</button>
  <button class="tab-btn" onclick="switchTab(4)"><span class="tab-num">5</span> Loss Functions</button>
  <button class="tab-btn" onclick="switchTab(5)"><span class="tab-num">6</span> Libraries</button>
  <button class="tab-btn" onclick="switchTab(6)"><span class="tab-num">7</span> GB vs RF</button>
</div>

<div class="panels">

  <!-- ══════════════════════════════════════════════════
       PANEL 1 — Overview
  ══════════════════════════════════════════════════ -->
  <div class="panel active" id="panel-0">
    <div class="section-label">Concept</div>
    <div class="section-title">What is Gradient Boosting?</div>
    <div class="section-body">
      Gradient Boosting is an ensemble method that builds an <strong>additive model sequentially</strong>:
      at each step, it fits a new weak learner to the <em>negative gradient</em> of the loss function
      with respect to the current ensemble prediction.<br><br>
      In plain English: <strong>each new tree tries to fix what the current ensemble gets wrong.</strong>
      Unlike Random Forests — where each tree is independent and parallel — Gradient Boosting trees
      are <strong>dependent</strong>. Tree <em>b</em> cannot be trained until trees 1 through <em>b−1</em>
      have been trained, because it needs to know their combined prediction error.
    </div>

    <div class="grid-2" style="margin-bottom:20px">
      <div class="card">
        <div class="card-title">🌲 Random Forest (Parallel)</div>
        <p>Trains <strong>B independent trees</strong> on bootstrap samples simultaneously. Every tree votes equally. Reduces <strong>variance</strong> by averaging decorrelated trees.</p>
        <br>
        <div style="font-family:var(--mono);font-size:0.75rem;color:#475569;line-height:2">
          Tree 1 ─── Bootstrap 1<br>
          Tree 2 ─── Bootstrap 2 → avg → ŷ<br>
          Tree B ─── Bootstrap B
        </div>
      </div>
      <div class="card">
        <div class="card-title" style="color:var(--accent)">⛓ Gradient Boosting (Sequential)</div>
        <p>Trains trees <strong>one after another</strong>, each correcting the previous ensemble's errors. Reduces <strong>bias then variance</strong> step by step.</p>
        <br>
        <div style="font-family:var(--mono);font-size:0.75rem;color:#475569;line-height:2">
          F₀ → residuals<br>
          F₀ + α·h₁ → new residuals → h₂<br>
          F₀ + α·h₁ + α·h₂ + … → F_B
        </div>
      </div>
    </div>

    <div class="section-label" style="margin-top:28px">Inductive Bias</div>
    <div class="grid-3">
      <div class="card">
        <div class="card-title" style="color:var(--accent3)">① Additive Structure</div>
        <p>The output is a <strong>sum of simple functions</strong> (shallow trees). Complex patterns are decomposed into a sequence of simple corrections.</p>
      </div>
      <div class="card">
        <div class="card-title" style="color:var(--accent3)">② Sequential Error Reduction</div>
        <p>Each step specifically targets the <strong>current errors</strong> of the ensemble. No effort is wasted on already-correct examples.</p>
      </div>
      <div class="card">
        <div class="card-title" style="color:var(--accent3)">③ Axis-Aligned Interactions</div>
        <p>Inherits from decision trees: features interact through <strong>conditional splits</strong>, not linear combinations.</p>
      </div>
    </div>

    <div class="section-label" style="margin-top:28px">Hyperparameters You Control</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-top:4px">
      <div class="formula-box" style="margin:0;padding:14px 18px">
        <div style="color:var(--text-dim);font-size:0.65rem;margin-bottom:4px">N_ESTIMATORS (B)</div>
        <div>Number of boosting rounds</div>
        <div class="comment">More trees → better fit, but can overfit</div>
      </div>
      <div class="formula-box" style="margin:0;padding:14px 18px">
        <div style="color:var(--text-dim);font-size:0.65rem;margin-bottom:4px">LEARNING_RATE (α)</div>
        <div>Shrinkage per tree</div>
        <div class="comment">Small α → regularisation, needs more B</div>
      </div>
      <div class="formula-box" style="margin:0;padding:14px 18px">
        <div style="color:var(--text-dim);font-size:0.65rem;margin-bottom:4px">MAX_DEPTH</div>
        <div>Depth of each weak learner</div>
        <div class="comment">Usually 1–5; deeper = more capacity</div>
      </div>
      <div class="formula-box" style="margin:0;padding:14px 18px">
        <div style="color:var(--text-dim);font-size:0.65rem;margin-bottom:4px">SUBSAMPLE</div>
        <div>Row fraction per tree</div>
        <div class="comment">Stochastic GB — adds variance reduction</div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════
       PANEL 2 — Algorithm
  ══════════════════════════════════════════════════ -->
  <div class="panel" id="panel-1">
    <div class="section-label">Friedman (2001)</div>
    <div class="section-title">The Full Algorithm</div>
    <div class="section-body">
      Gradient Boosting is <strong>gradient descent in function space</strong>. Instead of updating
      model weights, we update the <em>prediction function</em> F(x) directly, one tree at a time.
    </div>

    <div class="grid-2">
      <div>
        <div class="section-label">Function-Space Gradient Descent</div>
        <div class="formula-box">
          <span class="comment">// Ordinary gradient descent (weight space)</span><br>
          w_t = w_{t-1} − α · ∇_w L(w)<br><br>
          <span class="comment">// Gradient Boosting (function space)</span><br>
          F_t(x) = F_{t-1}(x) + α · h_t(x)<br><br>
          <span class="comment">// Where h_t fits the negative gradient:</span><br>
          r_i = −∂L(y_i, F(x_i)) / ∂F(x_i)
        </div>
        <div class="callout">
          <div class="callout-icon">💡</div>
          <div>The "parameters" being updated are the <strong>predicted values {F(xᵢ)} themselves</strong>,
          not the weights of a neural network or coefficients of a linear model.</div>
        </div>
      </div>

      <div>
        <div class="section-label">The Algorithm (MSE Regression)</div>
        <div class="algo-steps">
          <div class="algo-step">
            <div class="algo-step-left">
              <div class="algo-dot lit">0</div>
              <div class="algo-line"></div>
            </div>
            <div class="algo-step-right">
              <div class="algo-step-title">Initialise — Constant Prediction</div>
              <div class="algo-step-body">
                Predict the mean of all targets:<br>
                <span class="formula">F₀(x) = mean(y)</span>
              </div>
            </div>
          </div>
          <div class="algo-step">
            <div class="algo-step-left">
              <div class="algo-dot lit" style="background:var(--accent2);border-color:var(--accent2)">1</div>
              <div class="algo-line"></div>
            </div>
            <div class="algo-step-right">
              <div class="algo-step-title">Compute Pseudo-Residuals</div>
              <div class="algo-step-body">
                Calculate the negative gradient at each training point:<br>
                <span class="formula">r_i = y_i − F_{t-1}(x_i)  <span style="color:var(--text-dim)">← for MSE</span></span>
              </div>
            </div>
          </div>
          <div class="algo-step">
            <div class="algo-step-left">
              <div class="algo-dot lit" style="background:var(--accent3);border-color:var(--accent3)">2</div>
              <div class="algo-line"></div>
            </div>
            <div class="algo-step-right">
              <div class="algo-step-title">Fit a New Tree to the Residuals</div>
              <div class="algo-step-body">
                Train a shallow decision tree on <em>(X, r)</em>:<br>
                <span class="formula">h_t = DecisionTree().fit(X, r)</span>
              </div>
            </div>
          </div>
          <div class="algo-step">
            <div class="algo-step-left">
              <div class="algo-dot lit" style="background:var(--accent4);border-color:var(--accent4)">3</div>
              <div class="algo-line"></div>
            </div>
            <div class="algo-step-right">
              <div class="algo-step-title">Update the Ensemble</div>
              <div class="algo-step-body">
                Add the new tree (scaled by learning rate):<br>
                <span class="formula">F_t(x) = F_{t-1}(x) + α · h_t(x)</span>
              </div>
            </div>
          </div>
          <div class="algo-step">
            <div class="algo-step-left">
              <div class="algo-dot lit" style="background:var(--accent5);border-color:var(--accent5)">B</div>
            </div>
            <div class="algo-step-right" style="padding-bottom:0">
              <div class="algo-step-title">Final Model</div>
              <div class="algo-step-body">
                After B iterations:<br>
                <span class="formula">F_B(x) = F₀ + α · Σ_{t=1}^{B} h_t(x)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="section-label" style="margin-top:32px">Pseudo-Residuals by Loss Function</div>
    <div class="grid-2" style="margin-top:8px">
      <div class="formula-box">
        <div style="color:var(--text-dim);font-size:0.68rem;margin-bottom:6px">MSE LOSS: L(y,F) = ½(y−F)²</div>
        r_i = −∂L/∂F = y_i − F(x_i)<br>
        <span class="comment">← Exact residuals: "how far off are we?"</span>
      </div>
      <div class="formula-box" style="border-left-color:var(--accent2)">
        <div style="color:var(--text-dim);font-size:0.68rem;margin-bottom:6px">BCE LOSS: L(y,F) = −[y·log(σ(F)) + (1−y)·log(1−σ(F))]</div>
        r_i = y_i − σ(F(x_i)) = y_i − ŷ_i<br>
        <span class="comment">← True label minus predicted probability</span>
      </div>
    </div>
    <div class="callout" style="border-left-color:var(--accent)">
      <div class="callout-icon">🔑</div>
      <div><strong>Key insight:</strong> For both MSE and BCE, the pseudo-residual is
      <strong>"true value minus current prediction"</strong> — the next tree always fits
      "how much the current ensemble is wrong."</div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════
       PANEL 3 — Walkthrough (Interactive Stepper)
  ══════════════════════════════════════════════════ -->
  <div class="panel" id="panel-2">
    <div class="section-label">Concrete Example</div>
    <div class="section-title">5-Point Regression Walkthrough</div>
    <div class="section-body">
      Trace gradient boosting on a tiny dataset with 5 points.
      <strong>α = 0.5, max_depth = 1 (decision stump).</strong>
      Watch how predictions improve and residuals shrink at each iteration.
    </div>

    <div class="step-nav">
      <button class="step-nav-btn" id="wb-prev" onclick="changeWalkStep(-1)" disabled>← Prev</button>
      <button class="step-nav-btn" id="wb-next" onclick="changeWalkStep(1)">Next →</button>
      <div class="step-indicator">Step <span id="ws-cur">1</span> of <span id="ws-tot">5</span></div>
    </div>

    <!-- Step 0 — Init -->
    <div class="step-panel active" id="ws-0">
      <div class="step-header">
        <div class="step-badge">0</div>
        <div class="step-title">Initialise — F₀(x) = mean(y)</div>
      </div>
      <div class="grid-2">
        <div>
          <div class="section-label">Dataset</div>
          <table class="data-table">
            <thead><tr><th>x</th><th>y_true</th><th>F₀(x)</th><th>Residual r</th></tr></thead>
            <tbody>
              <tr><td>1.0</td><td>1.5</td><td class="neutral">2.90</td><td class="neg">−1.40</td></tr>
              <tr><td>2.0</td><td>3.0</td><td class="neutral">2.90</td><td class="pos">+0.10</td></tr>
              <tr><td>3.0</td><td>2.5</td><td class="neutral">2.90</td><td class="neg">−0.40</td></tr>
              <tr><td>4.0</td><td>4.0</td><td class="neutral">2.90</td><td class="pos">+1.10</td></tr>
              <tr><td>5.0</td><td>3.5</td><td class="neutral">2.90</td><td class="pos">+0.60</td></tr>
            </tbody>
          </table>
          <div class="formula-box" style="margin-top:12px">
            F₀ = mean([1.5, 3.0, 2.5, 4.0, 3.5]) = <span style="color:var(--accent)">2.9</span><br>
            MSE(F₀) = 0.710
          </div>
        </div>
        <div>
          <div class="section-label">Chart — Predictions vs True</div>
          <div class="chart-wrap">
            <canvas id="chart-ws0" width="340" height="220"></canvas>
          </div>
          <div class="mse-bar-wrap" style="margin-top:12px">
            <div class="mse-label">MSE</div>
            <div class="mse-bar-outer"><div class="mse-bar-inner" style="width:100%;background:var(--accent4)"></div></div>
            <div class="mse-val">0.710</div>
          </div>
        </div>
      </div>
      <div class="callout">
        <div class="callout-icon">📌</div>
        <div>Before any trees are trained, we predict the <strong>mean of all targets</strong>
        for every data point. This is our starting point F₀. The residuals (y − F₀) tell us
        how far off this flat baseline is for each training example.</div>
      </div>
    </div>

    <!-- Step 1 -->
    <div class="step-panel" id="ws-1">
      <div class="step-header">
        <div class="step-badge blue">1</div>
        <div class="step-title">Iteration 1 — Fit Tree h₁ to Residuals</div>
      </div>
      <div class="grid-2">
        <div>
          <div class="section-label">Tree 1 Decision Stump</div>
          <div class="formula-box" style="margin-bottom:12px">
            Fit tree to residuals: [−1.4, +0.1, −0.4, +1.1, +0.6]<br><br>
            Best split: x ≤ 2.5<br>
            Left (x≤2.5): mean(−1.4, +0.1) = <span style="color:var(--accent4)">−0.65</span><br>
            Right (x>2.5): mean(−0.4,+1.1,+0.6) = <span style="color:var(--accent3)">+0.43</span>
          </div>
          <table class="data-table">
            <thead><tr><th>x</th><th>F₀</th><th>h₁(x)</th><th>F₁ = F₀+0.5·h₁</th><th>New r</th></tr></thead>
            <tbody>
              <tr><td>1.0</td><td>2.90</td><td class="neg">−0.65</td><td class="neutral">2.575</td><td class="neg">−1.075</td></tr>
              <tr><td>2.0</td><td>2.90</td><td class="neg">−0.65</td><td class="neutral">2.575</td><td class="pos">+0.425</td></tr>
              <tr><td>3.0</td><td>2.90</td><td class="pos">+0.43</td><td class="neutral">3.117</td><td class="neg">−0.617</td></tr>
              <tr><td>4.0</td><td>2.90</td><td class="pos">+0.43</td><td class="neutral">3.117</td><td class="pos">+0.883</td></tr>
              <tr><td>5.0</td><td>2.90</td><td class="pos">+0.43</td><td class="neutral">3.117</td><td class="pos">+0.383</td></tr>
            </tbody>
          </table>
        </div>
        <div>
          <div class="section-label">Chart — F₁ vs True</div>
          <div class="chart-wrap">
            <canvas id="chart-ws1" width="340" height="220"></canvas>
          </div>
          <div class="mse-bar-wrap" style="margin-top:12px">
            <div class="mse-label">MSE F₀</div>
            <div class="mse-bar-outer"><div class="mse-bar-inner" style="width:100%;background:#374151"></div></div>
            <div class="mse-val" style="color:#475569">0.710</div>
          </div>
          <div class="mse-bar-wrap">
            <div class="mse-label">MSE F₁</div>
            <div class="mse-bar-outer"><div class="mse-bar-inner" style="width:60%;background:var(--accent)"></div></div>
            <div class="mse-val">0.430</div>
          </div>
        </div>
      </div>
      <div class="callout">
        <div class="callout-icon">🎯</div>
        <div>Tree 1 learns a <strong>single split</strong>: all points with x ≤ 2.5 get pushed down (they were predicting too high) and points with x > 2.5 get pushed up. MSE drops from <strong>0.710 → 0.430</strong>, a 39% improvement from one shallow tree!</div>
      </div>
    </div>

    <!-- Step 2 -->
    <div class="step-panel" id="ws-2">
      <div class="step-header">
        <div class="step-badge green">2</div>
        <div class="step-title">Iteration 2 — Fit h₂ to New Residuals</div>
      </div>
      <div class="grid-2">
        <div>
          <div class="section-label">Tree 2 focuses on remaining errors</div>
          <div class="formula-box" style="margin-bottom:12px">
            New residuals: [−1.075, +0.425, −0.617, +0.883, +0.383]<br><br>
            Best split: x ≤ 1.5<br>
            Left (x≤1.5): −1.075 → <span style="color:var(--accent4)">−1.075</span><br>
            Right (x>1.5): mean = <span style="color:var(--accent3)">+0.27</span><br><br>
            F₂ = F₁ + 0.5 · h₂(x)
          </div>
          <div class="callout" style="margin-top:4px">
            <div class="callout-icon">📊</div>
            <div>The x=1 point still has the <strong>largest residual</strong> (−1.075), so Tree 2 makes a cut to fix it. Each subsequent tree targets whatever the ensemble is <em>currently</em> getting most wrong.</div>
          </div>
        </div>
        <div>
          <div class="section-label">MSE Progression</div>
          <div class="chart-wrap">
            <canvas id="chart-ws2" width="340" height="220"></canvas>
          </div>
          <div class="mse-bar-wrap" style="margin-top:8px">
            <div class="mse-label">F₀</div>
            <div class="mse-bar-outer"><div class="mse-bar-inner" style="width:100%;background:#374151"></div></div>
            <div class="mse-val" style="color:#475569">0.710</div>
          </div>
          <div class="mse-bar-wrap">
            <div class="mse-label">F₁</div>
            <div class="mse-bar-outer"><div class="mse-bar-inner" style="width:60%;background:#f59e0b88"></div></div>
            <div class="mse-val" style="color:#f59e0b88">0.430</div>
          </div>
          <div class="mse-bar-wrap">
            <div class="mse-label">F₂</div>
            <div class="mse-bar-outer"><div class="mse-bar-inner" style="width:36%;background:var(--accent3)"></div></div>
            <div class="mse-val">≈0.256</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Step 3 — More iterations -->
    <div class="step-panel" id="ws-3">
      <div class="step-header">
        <div class="step-badge pink">…</div>
        <div class="step-title">Iterations 3–B — Convergence</div>
      </div>
      <div class="grid-2">
        <div>
          <div class="section-label">MSE converges toward 0</div>
          <div class="chart-wrap">
            <canvas id="chart-ws3" width="340" height="240"></canvas>
          </div>
          <div class="callout" style="margin-top:8px">
            <div class="callout-icon">⚠️</div>
            <div><strong>Key difference from RF:</strong> with enough trees and small enough learning rate, GB can reach near-zero training error — but this means it <em>can overfit</em>. Use early stopping to find the sweet spot.</div>
          </div>
        </div>
        <div>
          <div class="section-label">What each tree is doing</div>
          <table class="data-table">
            <thead><tr><th>Iter</th><th>Max |residual|</th><th>Tree action</th></tr></thead>
            <tbody>
              <tr><td>0 (F₀)</td><td class="neg">1.40</td><td style="text-align:left;font-size:0.75rem;color:#64748b">Flat baseline = mean(y)</td></tr>
              <tr><td>1</td><td class="neg">1.075</td><td style="text-align:left;font-size:0.75rem;color:#64748b">Split: x≤2.5 vs x>2.5</td></tr>
              <tr><td>2</td><td class="neg">0.806</td><td style="text-align:left;font-size:0.75rem;color:#64748b">Split: x≤1.5 (target x=1)</td></tr>
              <tr><td>5</td><td class="pos">0.41</td><td style="text-align:left;font-size:0.75rem;color:#64748b">Finer corrections</td></tr>
              <tr><td>20</td><td class="pos">0.08</td><td style="text-align:left;font-size:0.75rem;color:#64748b">Micro-corrections</td></tr>
              <tr><td>100</td><td class="highlight">≈0.00</td><td style="text-align:left;font-size:0.75rem;color:#64748b">Training error ≈ 0</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Step 4 — Key takeaway -->
    <div class="step-panel" id="ws-4">
      <div class="step-header">
        <div class="step-badge purple">✓</div>
        <div class="step-title">The Complete Picture</div>
      </div>
      <div class="section-body">
        After B iterations, the final prediction is the <strong>sum of all tree contributions</strong>
        scaled by the learning rate, plus the initial baseline.
      </div>
      <div class="formula-box" style="font-size:0.88rem">
        F_B(x) = F₀ + α · h₁(x) + α · h₂(x) + … + α · h_B(x)<br><br>
        <span class="comment">= mean(y) + Σ_{t=1}^{B} α · h_t(x)</span>
      </div>
      <div class="grid-3" style="margin-top:20px">
        <div class="card">
          <div class="card-title">🔁 Sequential</div>
          <p>Each tree depends on all previous trees. Training cannot be parallelised across trees (unlike RF).</p>
        </div>
        <div class="card">
          <div class="card-title">📉 Bias Reduction</div>
          <p>Boosting primarily reduces <strong>bias</strong>. Each tree corrects systematic errors the previous ensemble made.</p>
        </div>
        <div class="card">
          <div class="card-title">⚖️ Regularise!</div>
          <p>More trees <em>can overfit</em>. Use <strong>small α</strong> + <strong>early stopping</strong> to control the bias-variance tradeoff.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════
       PANEL 4 — Learning Rate
  ══════════════════════════════════════════════════ -->
  <div class="panel" id="panel-3">
    <div class="section-label">Regularisation</div>
    <div class="section-title">Learning Rate & The Shrinkage Tradeoff</div>
    <div class="section-body">
      The learning rate α (shrinkage) scales each tree's contribution:
      <em>F_t(x) = F_{t-1}(x) + α · h_t(x)</em><br><br>
      Small α means the model takes many <strong>small, careful steps</strong>. This slows convergence
      but dramatically improves generalisation. The dominant rule: <strong>always set α small
      (0.01–0.1), then find the optimal B via early stopping.</strong>
    </div>

    <div class="grid-2" style="margin-bottom:24px">
      <div>
        <div class="section-label">Learning Rate Comparison</div>
        <div class="chart-wrap">
          <canvas id="chart-lr" width="340" height="260"></canvas>
        </div>
      </div>
      <div>
        <div class="section-label">Shrinkage–Trees Tradeoff</div>
        <table class="data-table">
          <thead><tr><th>α</th><th>B (trees)</th><th>Convergence</th><th>Generalisation</th></tr></thead>
          <tbody>
            <tr><td>0.9</td><td class="neg">~10</td><td>Fast</td><td class="neg">Poor</td></tr>
            <tr><td>0.3</td><td>~30</td><td>Moderate</td><td class="neutral">OK</td></tr>
            <tr><td>0.1</td><td>~100</td><td>Good</td><td class="pos">Good</td></tr>
            <tr><td>0.05</td><td>~200</td><td>Slow</td><td class="highlight">Excellent</td></tr>
            <tr><td>0.01</td><td>~1000</td><td>Very slow</td><td class="highlight">Best</td></tr>
          </tbody>
        </table>
        <div class="callout" style="margin-top:12px">
          <div class="callout-icon">📏</div>
          <div>The product <strong>α × B</strong> controls the total "learning". A smaller α with a larger B reaches the same training error but generalises better because each individual tree has less influence.</div>
        </div>
      </div>
    </div>

    <div class="section-label">Early Stopping — Finding Optimal B</div>
    <div class="grid-2" style="margin-top:8px">
      <div>
        <div class="chart-wrap">
          <canvas id="chart-es" width="340" height="220"></canvas>
        </div>
      </div>
      <div>
        <div class="formula-box">
          <span class="comment">// Early stopping procedure</span><br>
          1. Reserve a validation set<br>
          2. Add trees one by one<br>
          3. Monitor val_loss each round<br>
          4. Stop when no improvement<br>
           &nbsp; &nbsp; for k rounds (patience)<br>
          5. Use best B — not the last
        </div>
        <div class="callout" style="border-left-color:var(--accent4)">
          <div class="callout-icon">⚠️</div>
          <div><strong>Unlike Random Forests</strong>, where more trees always help (variance keeps decreasing), Gradient Boosting has a U-shaped validation curve. Too many trees = overfitting to training noise.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════
       PANEL 5 — Loss Functions
  ══════════════════════════════════════════════════ -->
  <div class="panel" id="panel-4">
    <div class="section-label">Flexibility</div>
    <div class="section-title">Loss Functions — GB is Loss-Agnostic</div>
    <div class="section-body">
      One of Gradient Boosting's great strengths: <strong>any differentiable loss function can be plugged in.</strong>
      The algorithm only needs the gradient (pseudo-residuals), not a closed-form solution.
      This makes GB uniquely versatile across tasks.
    </div>

    <div class="loss-grid">
      <div class="loss-card">
        <div class="loss-name">MSE — Regression</div>
        <div class="loss-formula">L = ½(y−F)²</div>
        <div class="loss-residual">r_i = y_i − F(x_i)</div>
        <div class="loss-use">Default for regression. Residuals = exact errors. Sensitive to outliers due to squaring.</div>
      </div>
      <div class="loss-card">
        <div class="loss-name">MAE — Robust Regression</div>
        <div class="loss-formula">L = |y−F|</div>
        <div class="loss-residual">r_i = sign(y_i − F(x_i))</div>
        <div class="loss-use">Robust to outliers. Residuals are ±1 (sign only). Converges more slowly than MSE.</div>
      </div>
      <div class="loss-card">
        <div class="loss-name">Huber — Best of Both</div>
        <div class="loss-formula">L = MSE if |r|≤δ, else MAE</div>
        <div class="loss-residual">r_i = clipped gradient</div>
        <div class="loss-use">Quadratic near zero (fast convergence), linear in tails (robust to outliers). Best for noisy regression.</div>
      </div>
      <div class="loss-card">
        <div class="loss-name">Binary Cross-Entropy</div>
        <div class="loss-formula">L = −[y·log(σ(F)) + (1−y)·log(1−σ(F))]</div>
        <div class="loss-residual">r_i = y_i − σ(F(x_i))</div>
        <div class="loss-use">Binary classification. F is log-odds, σ converts to probability. Same residual form as MSE!</div>
      </div>
      <div class="loss-card">
        <div class="loss-name">Deviance (Multi-class)</div>
        <div class="loss-formula">L = −Σ_k y_k · log(p_k)</div>
        <div class="loss-residual">r_ik = y_ik − p_k(x_i)</div>
        <div class="loss-use">One tree per class per round (K·B trees total). F_k is log-probability for class k.</div>
      </div>
      <div class="loss-card">
        <div class="loss-name">Quantile Loss</div>
        <div class="loss-formula">L = α·max(r,0) + (1−α)·max(−r,0)</div>
        <div class="loss-residual">r_i = quantile-weighted residual</div>
        <div class="loss-use">Predicts specific percentiles (e.g., 90th percentile). Useful for prediction intervals and risk modelling.</div>
      </div>
    </div>

    <div class="callout" style="margin-top:4px">
      <div class="callout-icon">🧠</div>
      <div><strong>The gradient descent analogy:</strong> Just as neural networks can use any differentiable loss with backprop, Gradient Boosting can use any differentiable loss via pseudo-residuals.
      The tree is simply learning to predict the negative gradient — it doesn't care what loss produced it.</div>
    </div>

    <div class="section-label" style="margin-top:28px">ERM Across the ML Progression</div>
    <div style="overflow-x:auto;margin-top:8px">
      <table class="lib-compare">
        <thead>
          <tr>
            <th style="text-align:left">Algorithm</th>
            <th class="sklearn">Loss</th>
            <th class="xgb">Optimiser</th>
            <th class="lgbm">Space</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Linear Regression</td><td>MSE</td><td>Gradient descent</td><td>Weight</td></tr>
          <tr><td>Logistic Regression</td><td>BCE</td><td>Gradient descent</td><td>Weight</td></tr>
          <tr><td>SVM</td><td>Hinge</td><td>Quadratic programming</td><td>Dual</td></tr>
          <tr><td>Decision Tree</td><td>Gini/entropy</td><td>Greedy split search</td><td>Local</td></tr>
          <tr><td>Random Forest</td><td>0-1 loss</td><td>Average B trees</td><td>Ensemble</td></tr>
          <tr><td style="color:var(--accent);font-weight:700">Gradient Boosting</td><td style="color:var(--accent)">Any L</td><td style="color:var(--accent)">Gradient descent</td><td style="color:var(--accent)">Function space</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════
       PANEL 6 — Libraries
  ══════════════════════════════════════════════════ -->
  <div class="panel" id="panel-5">
    <div class="section-label">Modern Implementations</div>
    <div class="section-title">XGBoost · LightGBM · CatBoost</div>
    <div class="section-body">
      Sklearn's <em>GradientBoostingClassifier</em> is correct but slow — it processes splits
      one by one with no parallelism inside each tree. The three major modern implementations
      each solved this differently, unlocking state-of-the-art performance on large datasets.
    </div>

    <div class="grid-3" style="margin-bottom:24px">
      <div class="card" style="border-top:3px solid var(--accent)">
        <div class="card-title">XGBoost (2014)</div>
        <p><strong>Second-order gradients</strong> — uses both the gradient (g_i) and Hessian (h_i) of the loss to compute optimal leaf values analytically:<br><br>
        <span style="font-family:var(--mono);font-size:0.75rem;color:var(--accent)">w* = −Σg_i / (Σh_i + λ)</span><br><br>
        Also introduced: sparse-aware splits, column subsampling, cache-efficient tree building, built-in cross-validation.</p>
      </div>
      <div class="card" style="border-top:3px solid var(--accent2)">
        <div class="card-title" style="color:var(--accent2)">LightGBM (2017)</div>
        <p><strong>GOSS</strong>: keep hard examples (large gradients), sample easy ones — focuses compute where it matters.<br><br>
        <strong>EFB</strong>: bundle sparse features to reduce feature count.<br><br>
        <strong>Histogram splits</strong>: bin continuous features into 256 buckets — dramatically faster split search for large n.</p>
      </div>
      <div class="card" style="border-top:3px solid var(--accent3)">
        <div class="card-title" style="color:var(--accent3)">CatBoost (2017)</div>
        <p><strong>Ordered Boosting</strong>: prevents target leakage when encoding categoricals by computing statistics only on examples seen earlier in a random permutation.<br><br>
        Native categorical handling without preprocessing. Uses symmetric (oblivious) trees — faster inference, less overfit.</p>
      </div>
    </div>

    <div class="section-label">Feature Comparison</div>
    <div style="overflow-x:auto;margin-top:8px">
      <table class="lib-compare">
        <thead>
          <tr>
            <th style="text-align:left">Feature</th>
            <th class="sklearn">sklearn GB</th>
            <th class="xgb">XGBoost</th>
            <th class="lgbm">LightGBM</th>
            <th class="catb">CatBoost</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Year</td><td>2007</td><td>2014</td><td>2017</td><td>2017</td></tr>
          <tr><td>Second-order gradient</td><td class="no">—</td><td class="yes">✓</td><td class="yes">✓</td><td class="yes">✓</td></tr>
          <tr><td>Histogram splits</td><td class="no">—</td><td class="part">Approx.</td><td class="yes">✓</td><td class="yes">✓</td></tr>
          <tr><td>Sparse data support</td><td class="no">—</td><td class="yes">✓</td><td class="yes">✓</td><td class="yes">✓</td></tr>
          <tr><td>Native categoricals</td><td class="no">—</td><td class="no">—</td><td class="part">Partial</td><td class="yes">✓ (key!)</td></tr>
          <tr><td>Column subsampling</td><td class="no">—</td><td class="yes">✓</td><td class="yes">✓</td><td class="yes">✓</td></tr>
          <tr><td>Row subsampling</td><td class="yes">Random</td><td class="yes">Random</td><td class="yes">GOSS</td><td class="yes">✓</td></tr>
          <tr><td>GPU support</td><td class="no">—</td><td class="yes">✓</td><td class="yes">✓</td><td class="yes">✓</td></tr>
          <tr><td>Training speed (large n)</td><td class="no">Slow</td><td class="part">Fast</td><td class="yes">Fastest</td><td class="part">Fast</td></tr>
          <tr><td>Typical accuracy</td><td>Good</td><td class="yes">Excellent</td><td class="yes">Excellent</td><td class="yes">Excellent</td></tr>
          <tr><td>sklearn-compatible API</td><td class="yes">✓</td><td class="part">~</td><td class="part">~</td><td class="part">~</td></tr>
        </tbody>
      </table>
    </div>

    <div class="callout" style="margin-top:16px;border-left-color:var(--accent5)">
      <div class="callout-icon">🏆</div>
      <div><strong>Kaggle rule of thumb:</strong> Start with XGBoost or LightGBM. Use CatBoost if you have many high-cardinality categorical features. Use sklearn GB only for quick prototyping on small datasets (n &lt; 10k).</div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════
       PANEL 7 — GB vs RF
  ══════════════════════════════════════════════════ -->
  <div class="panel" id="panel-6">
    <div class="section-label">Ensemble Methods</div>
    <div class="section-title">Gradient Boosting vs Random Forest</div>
    <div class="section-body">
      Both are ensembles of decision trees. The fundamental difference is in <strong>how</strong>
      they combine trees — and this difference has deep consequences for bias, variance, and usage.
    </div>

    <div class="vs-grid" style="margin-bottom:24px">
      <div class="vs-col rf">
        <div class="vs-label rf">🌲 Random Forest</div>
        <div class="vs-item">Train B trees <strong>independently</strong> on bootstrap samples</div>
        <div class="vs-item">Each tree sees <strong>random data + random features</strong></div>
        <div class="vs-item">Prediction = <strong>average</strong> (regression) / majority vote (classification)</div>
        <div class="vs-item">Bias = single tree bias (unchanged)</div>
        <div class="vs-item">Variance → 0 as B → ∞ (always improves!)</div>
        <div class="vs-item"><strong>Embarrassingly parallel</strong> — train all trees simultaneously</div>
      </div>
      <div class="vs-divider">vs</div>
      <div class="vs-col gb">
        <div class="vs-label gb">🚀 Gradient Boosting</div>
        <div class="vs-item">Train trees <strong>sequentially</strong> on pseudo-residuals</div>
        <div class="vs-item">Each tree sees <strong>all data</strong> (or subsampled rows)</div>
        <div class="vs-item">Prediction = <strong>sum</strong> of α·h_t(x)</div>
        <div class="vs-item">Bias <strong>decreases</strong> with B (the whole point!)</div>
        <div class="vs-item">Variance <strong>can increase</strong> with B → need regularisation</div>
        <div class="vs-item"><strong>Sequential</strong> — tree t cannot start until tree t-1 is done</div>
      </div>
    </div>

    <div class="grid-2">
      <div>
        <div class="section-label">When to Use Random Forest</div>
        <div class="card" style="border-left:3px solid var(--accent2)">
          <div class="vs-item">Fast training and prediction required</div>
          <div class="vs-item">Minimal hyperparameter tuning budget</div>
          <div class="vs-item">Need OOB error estimate (no separate val set)</div>
          <div class="vs-item">Dataset is noisy (RF more robust by default)</div>
          <div class="vs-item">Need parallel training across many cores</div>
        </div>
      </div>
      <div>
        <div class="section-label">When to Use Gradient Boosting</div>
        <div class="card" style="border-left:3px solid var(--accent)">
          <div class="vs-item"><strong>Maximum accuracy is the priority</strong> (Kaggle)</div>
          <div class="vs-item">Medium-to-large dataset (LightGBM scales beautifully)</div>
          <div class="vs-item">You can afford hyperparameter tuning time</div>
          <div class="vs-item">Custom loss functions needed (ranking, quantile, etc.)</div>
          <div class="vs-item"><strong>Tabular data benchmark: GBT almost always wins</strong></div>
        </div>
      </div>
    </div>

    <div class="section-label" style="margin-top:28px">Progression in the ML Family</div>
    <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;margin-top:12px">
      <div style="text-align:center;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:0.78rem;color:#64748b">Perceptron</div>
      <div style="color:var(--text-muted);padding:0 6px;font-size:1.2rem">→</div>
      <div style="text-align:center;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:0.78rem;color:#64748b">Logistic<br>Regression</div>
      <div style="color:var(--text-muted);padding:0 6px;font-size:1.2rem">→</div>
      <div style="text-align:center;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:0.78rem;color:#64748b">SVM</div>
      <div style="color:var(--text-muted);padding:0 6px;font-size:1.2rem">→</div>
      <div style="text-align:center;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:0.78rem;color:#64748b">Decision<br>Tree</div>
      <div style="color:var(--text-muted);padding:0 6px;font-size:1.2rem">→</div>
      <div style="text-align:center;padding:12px 16px;background:var(--surface);border:1px solid rgba(6,182,212,0.4);border-radius:8px;font-size:0.78rem;color:var(--accent2)">Random<br>Forest</div>
      <div style="color:var(--text-muted);padding:0 6px;font-size:1.2rem">→</div>
      <div style="text-align:center;padding:14px 18px;background:#1c1007;border:2px solid var(--accent);border-radius:8px;font-size:0.82rem;color:var(--accent);font-weight:600">Gradient<br>Boosting 🚀</div>
      <div style="color:var(--text-muted);padding:0 6px;font-size:1.2rem">→</div>
      <div style="text-align:center;padding:12px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:0.78rem;color:#64748b">Neural<br>Networks</div>
    </div>
    <div style="font-family:var(--mono);font-size:0.68rem;color:#374151;margin-top:8px">
      ← linear, hard boundaries &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; non-linear, flexible, ensemble &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; learns representations →
    </div>
  </div>

</div><!-- /panels -->

<script>
/* ── TAB LOGIC ── */
function switchTab(idx) {
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', i===idx));
  document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i===idx));
  if (idx === 3) { setTimeout(drawLRChart, 50); setTimeout(drawESChart, 50); }
}

/* ── WALKTHROUGH STEPPER ── */
let walkStep = 0;
const walkMax = 4;

function changeWalkStep(dir) {
  document.getElementById('ws-'+walkStep).classList.remove('active');
  walkStep = Math.max(0, Math.min(walkMax, walkStep + dir));
  document.getElementById('ws-'+walkStep).classList.add('active');
  document.getElementById('wb-prev').disabled = walkStep === 0;
  document.getElementById('wb-next').disabled = walkStep === walkMax;
  document.getElementById('ws-cur').textContent = walkStep + 1;
  const drawFns = [drawWS0, drawWS1, drawWS2, drawWS3, null];
  if (drawFns[walkStep]) setTimeout(drawFns[walkStep], 30);
}

/* ── CHART HELPERS ── */
function getCtx(id) {
  const c = document.getElementById(id);
  if (!c) return null;
  return c.getContext('2d');
}

function clearCanvas(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, w, h);
}

const C = { amber:'#f59e0b', cyan:'#06b6d4', green:'#10b981', pink:'#f43f5e',
            purple:'#818cf8', dim:'#374151', line:'#1e2d45', text:'#94a3b8', white:'#e2e8f0' };

function drawAxes(ctx, ox, oy, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel) {
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(ox+w,oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox,oy); ctx.lineTo(ox,oy-h); ctx.stroke();
  ctx.fillStyle = C.text;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  const xsteps = 4;
  for (let i = 0; i <= xsteps; i++) {
    const xv = xmin + (xmax - xmin) * i / xsteps;
    const px = ox + (xv - xmin) / (xmax - xmin) * w;
    ctx.fillText(xv.toFixed(1), px, oy + 14);
    ctx.strokeStyle = '#1a2235';
    ctx.beginPath(); ctx.moveTo(px, oy); ctx.lineTo(px, oy - h); ctx.stroke();
  }
  const ysteps = 4;
  for (let i = 0; i <= ysteps; i++) {
    const yv = ymin + (ymax - ymin) * i / ysteps;
    const py = oy - (yv - ymin) / (ymax - ymin) * h;
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(1), ox - 6, py + 4);
  }
  ctx.textAlign = 'center';
  if (xlabel) { ctx.fillStyle = C.dim; ctx.fillText(xlabel, ox + w/2, oy + 28); }
  if (ylabel) {
    ctx.save(); ctx.translate(ox - 30, oy - h/2); ctx.rotate(-Math.PI/2);
    ctx.fillStyle = C.dim; ctx.fillText(ylabel, 0, 0); ctx.restore();
  }
}

function toPixel(v, min, max, size) { return (v - min) / (max - min) * size; }

/* Chart WS0 — Flat baseline */
function drawWS0() {
  const ctx = getCtx('chart-ws0'); if (!ctx) return;
  const W=340, H=220, ox=50, oy=190, w=W-70, h=160;
  clearCanvas(ctx, W, H);
  drawAxes(ctx, ox, oy, w, h, 0.5, 5.5, 0, 5, 'x', 'y');
  const xs=[1,2,3,4,5], yt=[1.5,3,2.5,4,3.5], yp=[2.9,2.9,2.9,2.9,2.9];
  ctx.strokeStyle = C.amber; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
  ctx.beginPath();
  xs.forEach((x,i)=>{ const px=ox+toPixel(x,0.5,5.5,w), py=oy-toPixel(yp[i],0,5,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); });
  ctx.stroke(); ctx.setLineDash([]);
  xs.forEach((x,i)=>{
    const px=ox+toPixel(x,0.5,5.5,w), py=oy-toPixel(yt[i],0,5,h);
    ctx.fillStyle=C.cyan; ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
    const pyp=oy-toPixel(yp[i],0,5,h);
    ctx.strokeStyle='#374151'; ctx.lineWidth=1; ctx.setLineDash([2,2]);
    ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px,pyp); ctx.stroke(); ctx.setLineDash([]);
  });
  ctx.fillStyle=C.cyan; ctx.fillRect(ox,12,10,10); ctx.fillStyle=C.text; ctx.font='10px JetBrains Mono,monospace'; ctx.textAlign='left'; ctx.fillText('y_true',ox+14,21);
  ctx.strokeStyle=C.amber; ctx.lineWidth=1.5; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(ox+80,17); ctx.lineTo(ox+95,17); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.text; ctx.fillText('F\u2080 = 2.9',ox+98,21);
}

/* Chart WS1 — After tree 1 */
function drawWS1() {
  const ctx = getCtx('chart-ws1'); if (!ctx) return;
  const W=340, H=220, ox=50, oy=190, w=W-70, h=160;
  clearCanvas(ctx, W, H);
  drawAxes(ctx, ox, oy, w, h, 0.5, 5.5, 0, 5, 'x', 'y');
  const xs=[1,2,3,4,5], yt=[1.5,3,2.5,4,3.5], yp0=[2.9,2.9,2.9,2.9,2.9], yp1=[2.575,2.575,3.117,3.117,3.117];
  ctx.strokeStyle = '#374151'; ctx.lineWidth=1; ctx.setLineDash([2,2]);
  ctx.beginPath(); xs.forEach((x,i)=>{ const px=ox+toPixel(x,0.5,5.5,w), py=oy-toPixel(yp0[i],0,5,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = C.amber; ctx.lineWidth=2; ctx.setLineDash([]);
  ctx.beginPath(); xs.forEach((x,i)=>{ const px=ox+toPixel(x,0.5,5.5,w), py=oy-toPixel(yp1[i],0,5,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }); ctx.stroke();
  xs.forEach((x,i)=>{
    const px=ox+toPixel(x,0.5,5.5,w), py=oy-toPixel(yt[i],0,5,h);
    ctx.fillStyle=C.cyan; ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
    const py1=oy-toPixel(yp1[i],0,5,h);
    ctx.strokeStyle='#f43f5e66'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px,py1); ctx.stroke();
  });
  ctx.fillStyle=C.text; ctx.font='10px JetBrains Mono,monospace'; ctx.textAlign='left';
  ctx.fillStyle=C.cyan; ctx.fillRect(ox,12,10,10); ctx.fillStyle=C.text; ctx.fillText('y_true',ox+14,21);
  ctx.strokeStyle=C.amber; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(ox+80,17); ctx.lineTo(ox+95,17); ctx.stroke();
  ctx.fillStyle=C.text; ctx.fillText('F\u2081(x)',ox+98,21);
}

/* Chart WS2 — MSE bar progress */
function drawWS2() {
  const ctx = getCtx('chart-ws2'); if (!ctx) return;
  const W=340, H=220, ox=50, oy=190, w=W-70, h=160;
  clearCanvas(ctx, W, H);
  const iters=[0,1,2,3,4,5,8,12,20,50,100];
  const mse=[0.710,0.430,0.256,0.153,0.092,0.055,0.015,0.004,0.001,0.0001,0.00001];
  drawAxes(ctx, ox, oy, w, h, 0, 100, 0, 0.8, 'Iteration', 'MSE');
  ctx.strokeStyle=C.amber; ctx.lineWidth=2;
  ctx.beginPath();
  iters.forEach((it,i)=>{
    const px=ox+toPixel(it,0,100,w), py=oy-toPixel(mse[i],0,0.8,h);
    i?ctx.lineTo(px,py):ctx.moveTo(px,py);
  });
  ctx.stroke();
  [0,1,2].forEach(i=>{
    const px=ox+toPixel(iters[i],0,100,w), py=oy-toPixel(mse[i],0,0.8,h);
    ctx.fillStyle=i===0?C.amber:i===1?C.cyan:C.green;
    ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=C.text; ctx.font='9px JetBrains Mono,monospace'; ctx.textAlign='left';
    ctx.fillText(['0.710','0.430','\u22480.256'][i], px+7, py);
  });
}

/* Chart WS3 — Convergence */
function drawWS3() {
  const ctx = getCtx('chart-ws3'); if (!ctx) return;
  const W=340, H=240, ox=60, oy=205, w=W-80, h=175;
  clearCanvas(ctx, W, H);
  drawAxes(ctx, ox, oy, w, h, 0, 100, 0, 0.75, 'Iterations (B)', 'MSE');
  const iters=[0,1,2,3,4,5,6,7,8,10,12,15,20,30,50,75,100];
  const trainMSE=[0.710,0.430,0.256,0.153,0.092,0.055,0.033,0.020,0.012,0.005,0.002,0.0008,0.0002,0.00003,0.000001,0,0];
  const valMSE=[0.720,0.460,0.290,0.195,0.142,0.108,0.085,0.070,0.060,0.048,0.043,0.042,0.044,0.050,0.062,0.078,0.095];
  ctx.strokeStyle=C.cyan; ctx.lineWidth=2;
  ctx.beginPath();
  iters.forEach((it,i)=>{ const px=ox+toPixel(it,0,100,w), py=oy-toPixel(valMSE[i],0,0.75,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); });
  ctx.stroke();
  ctx.strokeStyle=C.amber; ctx.lineWidth=2; ctx.setLineDash([4,3]);
  ctx.beginPath();
  iters.forEach((it,i)=>{ const px=ox+toPixel(it,0,100,w), py=oy-toPixel(trainMSE[i],0,0.75,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); });
  ctx.stroke(); ctx.setLineDash([]);
  const bestIdx=12;
  const bpx=ox+toPixel(iters[bestIdx],0,100,w), bpy=oy-toPixel(valMSE[bestIdx],0,0.75,h);
  ctx.strokeStyle=C.green; ctx.lineWidth=1; ctx.setLineDash([2,2]);
  ctx.beginPath(); ctx.moveTo(bpx,0); ctx.lineTo(bpx,oy); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.green; ctx.beginPath(); ctx.arc(bpx,bpy,6,0,Math.PI*2); ctx.fill();
  ctx.fillStyle=C.green; ctx.font='10px JetBrains Mono,monospace'; ctx.textAlign='left'; ctx.fillText('best B\u224820',bpx+8,bpy);
  ctx.strokeStyle=C.cyan; ctx.lineWidth=2; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(ox,15); ctx.lineTo(ox+20,15); ctx.stroke();
  ctx.fillStyle=C.text; ctx.font='10px JetBrains Mono,monospace'; ctx.textAlign='left'; ctx.fillText('val loss',ox+24,19);
  ctx.strokeStyle=C.amber; ctx.setLineDash([4,3]); ctx.beginPath(); ctx.moveTo(ox+80,15); ctx.lineTo(ox+100,15); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.text; ctx.fillText('train loss',ox+104,19);
}

/* Learning Rate Chart */
function drawLRChart() {
  const ctx = getCtx('chart-lr'); if (!ctx) return;
  const W=340, H=260, ox=50, oy=230, w=W-70, h=190;
  clearCanvas(ctx, W, H);
  drawAxes(ctx, ox, oy, w, h, 0, 50, 0, 1.1, 'Trees (B)', 'Val Loss');
  const B = Array.from({length:51},(_,i)=>i);
  function convergeCurve(alpha, noise=0.02) {
    return B.map(b => {
      const base = 0.7 * Math.exp(-alpha * b * 2) + 0.15 + noise * Math.sin(b * 0.7 + alpha*10) * Math.exp(-b*0.05);
      return Math.max(0.12, base);
    });
  }
  const configs = [
    {alpha:0.9, color:'#f43f5e', label:'\u03b1=0.9 (overfit fast)', dash:[]},
    {alpha:0.3, color:'#f59e0b', label:'\u03b1=0.3', dash:[]},
    {alpha:0.1, color:'#10b981', label:'\u03b1=0.1 (recommended)', dash:[]},
    {alpha:0.02, color:'#06b6d4', label:'\u03b1=0.02 (slow & stable)', dash:[3,3]},
  ];
  configs.forEach(({alpha,color,label,dash}) => {
    const vals = convergeCurve(alpha, 0.015);
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash(dash);
    ctx.beginPath();
    B.forEach((b,i)=>{ const px=ox+toPixel(b,0,50,w), py=oy-toPixel(vals[i],0,1.1,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); });
    ctx.stroke();
  });
  ctx.setLineDash([]);
  configs.forEach(({color,label},i) => {
    const ly = 14 + i*16;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(ox,ly); ctx.lineTo(ox+20,ly); ctx.stroke();
    ctx.fillStyle=C.text; ctx.font='9px JetBrains Mono,monospace'; ctx.textAlign='left'; ctx.fillText(label,ox+24,ly+4);
  });
}

/* Early Stopping Chart */
function drawESChart() {
  const ctx = getCtx('chart-es'); if (!ctx) return;
  const W=340, H=220, ox=50, oy=190, w=W-70, h=160;
  clearCanvas(ctx, W, H);
  drawAxes(ctx, ox, oy, w, h, 0, 200, 0, 0.6, 'Trees (B)', 'Loss');
  const B = Array.from({length:201},(_,i)=>i);
  const train = B.map(b => 0.55 * Math.exp(-b/30) + 0.02);
  const val = B.map(b => {
    if (b < 80) return 0.55 * Math.exp(-b/25) + 0.12;
    return 0.12 + (b-80)*0.0015 + 0.01 * Math.sin(b * 0.3);
  });
  ctx.strokeStyle=C.amber; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); B.forEach((b,i)=>{ const px=ox+toPixel(b,0,200,w), py=oy-toPixel(train[i],0,0.6,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle=C.cyan; ctx.lineWidth=2;
  ctx.beginPath(); B.forEach((b,i)=>{ const px=ox+toPixel(b,0,200,w), py=oy-toPixel(val[i],0,0.6,h); i?ctx.lineTo(px,py):ctx.moveTo(px,py); }); ctx.stroke();
  const bestB=80;
  const bpx=ox+toPixel(bestB,0,200,w);
  ctx.strokeStyle=C.green; ctx.lineWidth=1; ctx.setLineDash([2,2]);
  ctx.beginPath(); ctx.moveTo(bpx,0); ctx.lineTo(bpx,oy); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.green; ctx.beginPath(); ctx.arc(bpx, oy-toPixel(0.12,0,0.6,h), 6,0,Math.PI*2); ctx.fill();
  ctx.fillStyle=C.green; ctx.font='10px JetBrains Mono,monospace'; ctx.textAlign='left'; ctx.fillText('Stop here',bpx+6, oy-toPixel(0.12,0,0.6,h)-6);
  ctx.fillStyle='#f43f5e44';
  const ovX=ox+toPixel(80,0,200,w);
  ctx.fillRect(ovX,0,w-toPixel(80,0,200,w),oy);
  ctx.fillStyle=C.pink; ctx.font='9px JetBrains Mono,monospace'; ctx.textAlign='center'; ctx.fillText('OVERFITTING ZONE', ovX + (w-toPixel(80,0,200,w))/2, 14);
  ctx.strokeStyle=C.cyan; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(ox,12); ctx.lineTo(ox+16,12); ctx.stroke();
  ctx.fillStyle=C.text; ctx.font='10px JetBrains Mono,monospace'; ctx.textAlign='left'; ctx.fillText('val loss',ox+20,16);
  ctx.strokeStyle=C.amber; ctx.setLineDash([4,3]); ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(ox+76,12); ctx.lineTo(ox+92,12); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.text; ctx.fillText('train loss',ox+96,16);
}

/* Auto-draw initial charts */
window.addEventListener('load', () => {
  setTimeout(drawWS0, 80);
});
</script>
</body>
</html>"""

GB_VISUAL_HEIGHT = 1400