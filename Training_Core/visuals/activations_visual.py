ACTIVATIONS_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: #08080f; color: #e4e4e7; padding: 20px; min-height: 100vh; }
h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.15em; color: #ff6b35; margin-bottom: 3px; }
.subtitle { color: #52525b; font-size: 0.8em; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.full { grid-column: 1 / -1; }
.card { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 18px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.card h3 { font-family: 'JetBrains Mono', monospace; font-size: 0.78em; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #ff6b35; margin-bottom: 12px; }
canvas { display: block; border-radius: 8px; }
.row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.row label { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #71717a; min-width: 80px; }
input[type=range] { flex: 1; height: 4px; border-radius: 2px; -webkit-appearance: none; appearance: none; background: #1e1e2e; outline: none; cursor: pointer; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #ff6b35; cursor: pointer; }
.val { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #ff6b35; min-width: 46px; text-align: right; }
.info-box { background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px; padding: 8px 12px; font-size: 0.78em; color: #94a3b8; line-height: 1.7; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
.hl  { color: #ff6b35; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }
.hl4 { color: #4ade80; font-weight: 700; }
.hl5 { color: #ef4444; font-weight: 700; }
.btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
button { background: #1e1e2e; color: #a1a1aa; border: 1px solid #2d2d40; border-radius: 6px; padding: 5px 12px; cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 0.75em; font-weight: 700; transition: all 0.15s; }
button:hover { background: #2d2d40; color: #e4e4e7; }
button.active { background: #ff6b35; color: #08080f; border-color: #ff6b35; }
.canvas-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68em; color: #3f3f46; text-transform: uppercase; letter-spacing: 0.08em; margin: 6px 0 3px; }
.tbl { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.72em; }
.tbl th { padding: 7px 10px; text-align: left; color: #52525b; border-bottom: 2px solid #1e1e2e; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
.tbl td { padding: 7px 10px; border-bottom: 1px solid #151520; color: #94a3b8; vertical-align: top; line-height: 1.5; }
.tbl tr:hover td { background: #0d0d18; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.85em; font-weight: 700; }
.tag-green  { background: #0a1f0f; color: #4ade80; border: 1px solid #4ade8040; }
.tag-orange { background: #1a120a; color: #fb923c; border: 1px solid #fb923c40; }
.tag-red    { background: #1a0808; color: #ef4444; border: 1px solid #ef444440; }
.tag-blue   { background: #0a1520; color: #38bdf8; border: 1px solid #38bdf840; }
.tag-purple { background: #150a20; color: #c084fc; border: 1px solid #c084fc40; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
</style>
</head>
<body>

<h2>&#9889; Activation &amp; Loss Functions</h2>
<p class="subtitle">Non-linearity that gives networks power &middot; Loss that guides training &middot; Interactive explorer &middot; Vanishing gradient &middot; Selection guide</p>

<div class="grid">

<!-- ======================================================
     PANEL 1: ACTIVATION FUNCTION EXPLORER
     ====================================================== -->
<div class="card">
  <h3>&#9312; Activation Function Explorer</h3>
  <div class="btn-row">
    <button id="aBtn_sigmoid"   class="active" onclick="selectAct('sigmoid')">Sigmoid</button>
    <button id="aBtn_tanh"      onclick="selectAct('tanh')">Tanh</button>
    <button id="aBtn_relu"      onclick="selectAct('relu')">ReLU</button>
    <button id="aBtn_leaky"     onclick="selectAct('leaky')">Leaky ReLU</button>
    <button id="aBtn_gelu"      onclick="selectAct('gelu')">GELU</button>
    <button id="aBtn_swish"     onclick="selectAct('swish')">Swish</button>
  </div>

  <div class="canvas-label">f(z) &mdash; output</div>
  <canvas id="cvFn" width="420" height="175"></canvas>

  <div class="canvas-label" style="margin-top:8px;">f'(z) &mdash; gradient</div>
  <canvas id="cvGrad" width="420" height="110"></canvas>

  <div class="row" style="margin-top:8px;">
    <label>z value</label>
    <input type="range" id="aSlider" min="-4" max="4" step="0.05" value="0"
           oninput="updateActZ(this.value)">
    <span class="val" id="aZVal">0.00</span>
  </div>

  <div class="info-box" id="actInfo">Loading&hellip;</div>
</div>


<!-- ======================================================
     PANEL 2: VANISHING GRADIENT PROBLEM
     ====================================================== -->
<div class="card">
  <h3>&#9313; Vanishing Gradient Problem</h3>
  <p style="font-size:0.75em; color:#52525b; margin-bottom:10px; font-family:'JetBrains Mono',monospace; line-height:1.6;">
    Gradient magnitude after backpropagating through N layers.<br>
    Each layer multiplies the gradient by f'(z) &mdash; saturation compounds.
  </p>
  <div class="btn-row">
    <button id="vgBtn_sigmoid"  class="active" onclick="selectVG('sigmoid')">Sigmoid</button>
    <button id="vgBtn_tanh"     onclick="selectVG('tanh')">Tanh</button>
    <button id="vgBtn_relu"     onclick="selectVG('relu')">ReLU</button>
    <button id="vgBtn_leaky"    onclick="selectVG('leaky')">Leaky ReLU</button>
  </div>

  <canvas id="cvVG" width="420" height="215"></canvas>

  <div class="row" style="margin-top:8px;">
    <label>Layers</label>
    <input type="range" id="layerSlider" min="1" max="20" step="1" value="10"
           oninput="updateVG(this.value)">
    <span class="val" id="layerVal">10</span>
  </div>

  <div class="info-box" id="vgInfo">Loading&hellip;</div>
</div>


<!-- ======================================================
     PANEL 3: LOSS FUNCTION EXPLORER (full width)
     ====================================================== -->
<div class="card full">
  <h3>&#9314; Loss Function Explorer</h3>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start;">

    <div>
      <div class="btn-row">
        <button id="lBtn_mse"    class="active" onclick="selectLoss('mse')">MSE</button>
        <button id="lBtn_mae"    onclick="selectLoss('mae')">MAE</button>
        <button id="lBtn_huber"  onclick="selectLoss('huber')">Huber</button>
        <button id="lBtn_bce"    onclick="selectLoss('bce')">BCE</button>
        <button id="lBtn_hinge"  onclick="selectLoss('hinge')">Hinge</button>
        <button id="lBtn_focal"  onclick="selectLoss('focal')">Focal</button>
      </div>

      <div class="canvas-label">Loss curve</div>
      <canvas id="cvLoss" width="420" height="210"></canvas>

      <div class="canvas-label" style="margin-top:6px;">Gradient magnitude</div>
      <canvas id="cvLossGrad" width="420" height="110"></canvas>

      <div class="row" style="margin-top:6px;">
        <label id="lSliderLabel">prediction</label>
        <input type="range" id="lSlider" min="0.01" max="0.99" step="0.01" value="0.5"
               oninput="updateLoss(this.value)">
        <span class="val" id="lSliderVal">0.50</span>
      </div>
    </div>

    <div>
      <div class="info-box" id="lossInfo" style="height:100%; min-height:400px;">Loading&hellip;</div>
    </div>

  </div>
</div>


<!-- ======================================================
     PANEL 4: SELECTION GUIDE (full width)
     ====================================================== -->
<div class="card full">
  <h3>&#9315; When to Use What &mdash; Selection Guide</h3>
  <div class="two-col">

    <div>
      <div style="font-size:0.72em; font-weight:800; color:#4ecdc4; font-family:'JetBrains Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;">Activation Functions</div>
      <table class="tbl">
        <thead><tr>
          <th>Where</th><th>Best Choice</th><th>Avoid</th>
        </tr></thead>
        <tbody>
          <tr><td><span class="hl2">CNN / MLP hidden</span></td><td><span class="hl4">ReLU</span> (default)</td><td><span class="hl5">Sigmoid, Tanh</span></td></tr>
          <tr><td><span class="hl2">If ReLU dying</span></td><td><span class="hl4">Leaky ReLU</span></td><td><span class="hl5">Sigmoid</span></td></tr>
          <tr><td><span class="hl2">Transformer hidden</span></td><td><span class="hl4">GELU</span> (standard)</td><td><span class="hl5">ReLU</span></td></tr>
          <tr><td><span class="hl2">RNN hidden</span></td><td><span class="hl4">Tanh</span></td><td><span class="hl5">ReLU</span></td></tr>
          <tr><td><span class="hl2">Binary output</span></td><td><span class="hl4">Sigmoid</span></td><td><span class="hl5">ReLU, Tanh</span></td></tr>
          <tr><td><span class="hl2">Multi-class output</span></td><td><span class="hl4">Softmax</span></td><td><span class="hl5">Sigmoid</span></td></tr>
          <tr><td><span class="hl2">Regression output</span></td><td><span class="hl4">Linear (none)</span></td><td><span class="hl5">Sigmoid (bounded)</span></td></tr>
        </tbody>
      </table>

      <div style="margin-top:14px; font-family:'JetBrains Mono',monospace; font-size:0.72em; color:#52525b; line-height:1.8;">
        <span class="hl">Rule:</span> Start with ReLU + He init for hidden layers.<br>
        If you see dead neurons, switch to Leaky ReLU.<br>
        For transformers, GELU is the universal default.<br>
        Activation on output layer must match the loss.
      </div>
    </div>

    <div>
      <div style="font-size:0.72em; font-weight:800; color:#c084fc; font-family:'JetBrains Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;">Loss Functions</div>
      <table class="tbl">
        <thead><tr>
          <th>Task</th><th>Loss</th><th>Output Activation</th>
        </tr></thead>
        <tbody>
          <tr><td><span class="hl2">Regression (clean)</span></td><td><span class="hl4">MSE</span></td><td>Linear</td></tr>
          <tr><td><span class="hl2">Regression (outliers)</span></td><td><span class="hl4">Huber</span></td><td>Linear</td></tr>
          <tr><td><span class="hl2">Regression (robust)</span></td><td><span class="hl4">MAE</span></td><td>Linear</td></tr>
          <tr><td><span class="hl2">Binary classification</span></td><td><span class="hl4">BCE</span></td><td>Sigmoid</td></tr>
          <tr><td><span class="hl2">Multi-class</span></td><td><span class="hl4">CrossEntropy</span></td><td>Softmax</td></tr>
          <tr><td><span class="hl2">Imbalanced classes</span></td><td><span class="hl4">Focal Loss</span></td><td>Sigmoid</td></tr>
          <tr><td><span class="hl2">SVM / max-margin</span></td><td><span class="hl4">Hinge</span></td><td>Raw score</td></tr>
          <tr><td><span class="hl2">VAE / distribution</span></td><td><span class="hl4">KL Divergence</span></td><td>Softmax</td></tr>
          <tr><td><span class="hl2">Object detection</span></td><td><span class="hl4">Focal + Huber</span></td><td>Sig + Linear</td></tr>
        </tbody>
      </table>

      <div style="margin-top:14px; font-family:'JetBrains Mono',monospace; font-size:0.72em; color:#52525b; line-height:1.8;">
        <span class="hl5">Red flag:</span> Loss goes to NaN early in training.<br>
        Check: gradient clipping &bull; log(0) protection &bull;<br>
        wrong output activation &bull; too high learning rate.
      </div>
    </div>

  </div>
</div>

</div><!-- end grid -->

<script>
// ============================================================
// COORDINATE HELPERS
// ============================================================
function cvTX(x, xmin, xmax, W, pad) { return pad + (x - xmin) / (xmax - xmin) * (W - pad - 10); }
function cvTY(y, ymin, ymax, H, pad) { return H - pad + 10 - (y - ymin) / (ymax - ymin) * (H - pad - 10); }

function clearCanvas(id, W, H) {
  var cv = document.getElementById(id);
  var ctx = cv.getContext('2d');
  ctx.fillStyle = '#09090f';
  ctx.fillRect(0, 0, W, H);
  return { cv: cv, ctx: ctx };
}

function drawAxes(ctx, W, H, xmin, xmax, ymin, ymax, PAD, xLabel, yLabel) {
  var x0 = cvTX(0, xmin, xmax, W, PAD);
  var y0 = cvTY(0, ymin, ymax, H, PAD);

  // grid lines
  ctx.strokeStyle = '#141420';
  ctx.lineWidth = 1;
  for (var yi = Math.ceil(ymin); yi <= Math.floor(ymax); yi++) {
    var gy = cvTY(yi, ymin, ymax, H, PAD);
    ctx.beginPath(); ctx.moveTo(PAD, gy); ctx.lineTo(W - 10, gy); ctx.stroke();
  }
  for (var xi = Math.ceil(xmin); xi <= Math.floor(xmax); xi++) {
    var gx = cvTX(xi, xmin, xmax, W, PAD);
    ctx.beginPath(); ctx.moveTo(gx, 10); ctx.lineTo(gx, H - PAD + 10); ctx.stroke();
  }

  // axes
  ctx.strokeStyle = '#2d2d40';
  ctx.lineWidth = 1.5;
  if (y0 >= 10 && y0 <= H - PAD + 10) {
    ctx.beginPath(); ctx.moveTo(PAD, y0); ctx.lineTo(W - 10, y0); ctx.stroke();
  }
  if (x0 >= PAD && x0 <= W - 10) {
    ctx.beginPath(); ctx.moveTo(x0, 10); ctx.lineTo(x0, H - PAD + 10); ctx.stroke();
  }

  // tick labels
  ctx.fillStyle = '#3f3f46';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  for (var xi2 = xmin; xi2 <= xmax; xi2 += (xmax - xmin) / 4) {
    var gx2 = cvTX(xi2, xmin, xmax, W, PAD);
    ctx.fillText(xi2.toFixed(xi2 % 1 === 0 ? 0 : 1), gx2, H - PAD + 22);
  }
  ctx.textAlign = 'right';
  for (var yi2 = Math.ceil(ymin); yi2 <= Math.floor(ymax); yi2++) {
    if (yi2 !== 0) {
      var gy2 = cvTY(yi2, ymin, ymax, H, PAD);
      ctx.fillText(yi2, PAD - 4, gy2 + 4);
    }
  }
  ctx.fillText('0', PAD - 4, y0 + 4);

  // axis labels
  ctx.fillStyle = '#52525b';
  ctx.textAlign = 'center';
  ctx.font = '10px JetBrains Mono, monospace';
  if (xLabel) ctx.fillText(xLabel, (PAD + W - 10) / 2, H - 1);
  if (yLabel) {
    ctx.save();
    ctx.translate(9, (10 + H - PAD + 10) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
}

function drawCurve(ctx, fn, xmin, xmax, ymin, ymax, W, H, PAD, color, lw) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw || 2.5;
  ctx.beginPath();
  var first = true;
  for (var i = 0; i <= 300; i++) {
    var zv = xmin + i / 300 * (xmax - xmin);
    var v = fn(zv);
    if (isNaN(v) || !isFinite(v)) { first = true; continue; }
    v = Math.max(ymin - 0.5, Math.min(ymax + 0.5, v));
    var cx = cvTX(zv, xmin, xmax, W, PAD);
    var cy = cvTY(v, ymin, ymax, H, PAD);
    if (first) { ctx.moveTo(cx, cy); first = false; } else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
}

function drawDot(ctx, zv, fn, xmin, xmax, ymin, ymax, W, H, PAD, color) {
  var v = fn(zv);
  if (isNaN(v) || !isFinite(v)) return;
  v = Math.max(ymin, Math.min(ymax, v));
  var cx = cvTX(zv, xmin, xmax, W, PAD);
  var cy = cvTY(v, ymin, ymax, H, PAD);
  var y0 = cvTY(0, ymin, ymax, H, PAD);

  // dashed vertical drop
  ctx.strokeStyle = '#2d2d40';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(cx, Math.min(cy, y0)); ctx.lineTo(cx, Math.max(cy, y0)); ctx.stroke();
  ctx.setLineDash([]);

  // dot
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(cx, cy, 5.5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#08080f';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================
var ACT = {
  sigmoid: {
    label: 'Sigmoid',
    color: '#ff6b35',
    fn:   function(z) { return 1 / (1 + Math.exp(-z)); },
    grad: function(z) { var s = 1/(1+Math.exp(-z)); return s*(1-s); },
    formula: '\u03C3(z) = 1 / (1 + e\u207B\u1D3A)',
    range: '(0, 1)',
    saturates: 'YES &mdash; gradient \u2248 0 for |z| > 4',
    zeroCentered: 'NO &mdash; output always positive',
    use: 'Output layer: binary classification <em>only</em>',
    problem: 'Vanishing gradient in deep nets. Never use in hidden layers.',
    verdict: 'bad'
  },
  tanh: {
    label: 'Tanh',
    color: '#4ecdc4',
    fn:   function(z) { return Math.tanh(z); },
    grad: function(z) { var t = Math.tanh(z); return 1 - t*t; },
    formula: 'tanh(z) = (e\u1D3A \u2212 e\u207B\u1D3A) / (e\u1D3A + e\u207B\u1D3A)',
    range: '(\u22121, +1)',
    saturates: 'YES &mdash; gradient \u2248 0 for |z| > 2',
    zeroCentered: 'YES &mdash; outputs centred around 0',
    use: 'RNN hidden layers. Slightly better than Sigmoid.',
    problem: 'Still saturates. Slower than ReLU. Deep nets suffer.',
    verdict: 'ok'
  },
  relu: {
    label: 'ReLU',
    color: '#4ade80',
    fn:   function(z) { return Math.max(0, z); },
    grad: function(z) { return z > 0 ? 1 : 0; },
    formula: 'f(z) = max(0, z)',
    range: '[0, \u221E)',
    saturates: 'HALF &mdash; zero for z < 0, linear for z > 0',
    zeroCentered: 'NO &mdash; output always non-negative',
    use: 'Default for ALL CNN and MLP hidden layers.',
    problem: 'Dying ReLU: neurons stuck at 0 if z always negative.',
    verdict: 'good'
  },
  leaky: {
    label: 'Leaky ReLU',
    color: '#fbbf24',
    fn:   function(z) { return z > 0 ? z : 0.01 * z; },
    grad: function(z) { return z > 0 ? 1 : 0.01; },
    formula: 'f(z) = z if z > 0 else 0.01z',
    range: '(\u2212\u221E, +\u221E)',
    saturates: 'NO &mdash; gradient always non-zero',
    zeroCentered: 'NO &mdash; but near-symmetric',
    use: 'CNN/MLP hidden layers when dying ReLU is observed.',
    problem: '\u03B1=0.01 is a hyperparameter. Use PReLU to learn it.',
    verdict: 'good'
  },
  gelu: {
    label: 'GELU',
    color: '#c084fc',
    fn:   function(z) { return 0.5*z*(1+Math.tanh(Math.sqrt(2/Math.PI)*(z+0.044715*z*z*z))); },
    grad: function(z) {
      var h = 0.001;
      var fp = function(w) { return 0.5*w*(1+Math.tanh(Math.sqrt(2/Math.PI)*(w+0.044715*w*w*w))); };
      return (fp(z+h)-fp(z-h))/(2*h);
    },
    formula: 'f(z) = z \u00B7 \u03A6(z)  where \u03A6 is Gaussian CDF',
    range: '(\u22120.17, +\u221E)',
    saturates: 'NO &mdash; non-zero gradient everywhere',
    zeroCentered: 'NO &mdash; but slight negative output near z=\u22120.17',
    use: 'Transformers: BERT, GPT, ViT. Universal standard.',
    problem: 'More compute than ReLU. Numerical approximation used.',
    verdict: 'best'
  },
  swish: {
    label: 'Swish / SiLU',
    color: '#38bdf8',
    fn:   function(z) { return z / (1 + Math.exp(-z)); },
    grad: function(z) { var s = 1/(1+Math.exp(-z)); return s + z*s*(1-s); },
    formula: 'f(z) = z \u00B7 \u03C3(z)',
    range: '(\u22120.28, +\u221E)',
    saturates: 'NO &mdash; non-zero gradient everywhere',
    zeroCentered: 'NO &mdash; slight negative region',
    use: 'EfficientNet, modern vision models. Similar to GELU.',
    problem: 'Similar performance to GELU, slightly different shape.',
    verdict: 'good'
  }
};

var curAct = 'sigmoid';
var curActZ = 0;

function selectAct(key) {
  curAct = key;
  document.querySelectorAll('[id^="aBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('aBtn_' + key).classList.add('active');
  renderAct();
}

function updateActZ(v) {
  curActZ = parseFloat(v);
  document.getElementById('aZVal').textContent = curActZ.toFixed(2);
  renderAct();
}

function renderAct() {
  var a = ACT[curAct];
  var W = 420, H1 = 175, H2 = 110, PAD = 34;

  // --- fn(z) canvas ---
  var r1 = clearCanvas('cvFn', W, H1);
  var ctx1 = r1.ctx;
  drawAxes(ctx1, W, H1, -4, 4, -1.5, 1.5, PAD, 'z', 'f(z)');
  drawCurve(ctx1, a.fn, -4, 4, -1.5, 1.5, W, H1, PAD, a.color, 2.5);
  drawDot(ctx1, curActZ, a.fn, -4, 4, -1.5, 1.5, W, H1, PAD, a.color);

  // --- gradient canvas ---
  var r2 = clearCanvas('cvGrad', W, H2);
  var ctx2 = r2.ctx;
  drawAxes(ctx2, W, H2, -4, 4, -0.1, 1.1, PAD, 'z', "f'(z)");
  drawCurve(ctx2, a.grad, -4, 4, -0.1, 1.1, W, H2, PAD, a.color, 2.5);
  drawDot(ctx2, curActZ, a.grad, -4, 4, -0.1, 1.1, W, H2, PAD, a.color);

  // annotation: current values
  var fv = a.fn(curActZ).toFixed(4);
  var gv = a.grad(curActZ).toFixed(4);

  // info box
  var verdictMap = {good:'tag-green', ok:'tag-orange', bad:'tag-red', best:'tag-blue'};
  var verdictText = {good:'Good choice', ok:'Use with care', bad:'Avoid in hidden layers', best:'Standard choice'};
  var vc = verdictMap[a.verdict] || 'tag-orange';
  var vt = verdictText[a.verdict] || '';

  var gradWarning = '';
  if (curAct === 'sigmoid' && (Math.abs(curActZ) > 2)) {
    gradWarning = '<br><span class="hl5">&#x26A0; Saturation zone: gradient = ' + gv + ' (near zero!)</span>';
  } else if (curAct === 'tanh' && (Math.abs(curActZ) > 2)) {
    gradWarning = '<br><span class="hl5">&#x26A0; Saturation zone: gradient = ' + gv + '</span>';
  } else if ((curAct === 'relu') && curActZ < 0) {
    gradWarning = '<br><span class="hl5">&#x26A0; Dead zone: gradient = 0 (neuron is off)</span>';
  }

  document.getElementById('actInfo').innerHTML =
    '<span class="hl">' + a.label + '</span> &nbsp; <span class="badge ' + vc + '">' + vt + '</span><br>' +
    '<span class="hl2">Formula:</span> ' + a.formula + '<br>' +
    '<span class="hl2">Range:</span> ' + a.range + '<br>' +
    '<span class="hl2">Saturates:</span> ' + a.saturates + '<br>' +
    '<span class="hl2">Zero-centred:</span> ' + a.zeroCentered + '<br>' +
    '<span class="hl3">Use for:</span> ' + a.use + '<br>' +
    '<span class="hl5">Problem:</span> ' + a.problem +
    '<br><br>At z = <span class="hl">' + curActZ.toFixed(2) + '</span>: &nbsp; ' +
    'f(z) = <span class="hl3">' + fv + '</span> &nbsp; f\'(z) = <span class="hl2">' + gv + '</span>' +
    gradWarning;
}

// ============================================================
// VANISHING GRADIENT VISUALIZER
// ============================================================
var curVG = 'sigmoid';
var curLayers = 10;

// Typical gradient magnitude at a given activation (at z=0 or realistic z)
var VG_GRAD = {
  sigmoid: function() { return 0.25; },    // max gradient at z=0
  tanh:    function() { return 0.50; },    // realistic average for tanh
  relu:    function() { return 1.00; },    // active neuron
  leaky:   function() { return 1.00; }     // active neuron (worst case: 0.01)
};

var VG_COLOR = { sigmoid:'#ff6b35', tanh:'#4ecdc4', relu:'#4ade80', leaky:'#fbbf24' };

var VG_INFO = {
  sigmoid: {
    title: 'Sigmoid: Severe Vanishing Gradient',
    body: 'Max gradient = <span class="hl5">0.25</span> at z=0.<br>' +
          'After 10 layers: 0.25\u00B9\u2070 \u2248 <span class="hl5">0.000001</span> (effectively zero).<br>' +
          'Early layers receive almost <span class="hl5">no gradient signal</span> \u2014 they stop learning.'
  },
  tanh: {
    title: 'Tanh: Moderate Vanishing Gradient',
    body: 'Max gradient = <span class="hl3">1.0</span> at z=0, but ~0.5 for typical z.<br>' +
          'After 10 layers with avg 0.5: 0.5\u00B9\u2070 \u2248 <span class="hl3">0.001</span>.<br>' +
          'Better than Sigmoid but <span class="hl3">still problematic</span> in deep networks.'
  },
  relu: {
    title: 'ReLU: No Vanishing Gradient (Active)',
    body: 'Gradient = <span class="hl4">exactly 1</span> for all z > 0 (active neurons).<br>' +
          'After 10 active layers: 1.0\u00B9\u2070 = <span class="hl4">1.0</span> (perfectly preserved).<br>' +
          'This is why ReLU enabled training of very deep networks. Risk: dying ReLU (z < 0 forever).'
  },
  leaky: {
    title: 'Leaky ReLU: No Vanishing Gradient (Any)',
    body: 'Gradient = <span class="hl4">1.0</span> for z > 0, <span class="hl3">0.01</span> for z < 0.<br>' +
          'Even "dead" neurons keep gradient = 0.01 \u2014 they can recover.<br>' +
          'After 10 layers (all active): gradient = <span class="hl4">1.0</span>. No vanishing.'
  }
};

function selectVG(key) {
  curVG = key;
  document.querySelectorAll('[id^="vgBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('vgBtn_' + key).classList.add('active');
  renderVG();
}

function updateVG(v) {
  curLayers = parseInt(v);
  document.getElementById('layerVal').textContent = curLayers;
  renderVG();
}

function renderVG() {
  var W = 420, H = 215, PAD = 40;
  var r = clearCanvas('cvVG', W, H);
  var ctx = r.ctx;
  var color = VG_COLOR[curVG];
  var baseGrad = VG_GRAD[curVG]();

  // compute gradient at each layer
  var grads = [];
  var g = 1.0;
  for (var l = 1; l <= curLayers; l++) {
    g *= baseGrad;
    grads.push(g);
  }

  var maxG = Math.max.apply(null, grads);
  var scaleG = maxG > 0 ? maxG : 1;
  var barW = Math.min(30, (W - PAD - 20) / curLayers - 4);
  var barSpacing = (W - PAD - 20) / curLayers;

  // background
  ctx.fillStyle = '#0d0d18';
  ctx.beginPath(); ctx.roundRect(PAD - 10, 10, W - PAD, H - 45, 6); ctx.fill();

  // baseline
  ctx.strokeStyle = '#2d2d40';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD - 10, H - 44); ctx.lineTo(W - 10, H - 44); ctx.stroke();

  // "perfect" line at top
  var perfY = 20;
  ctx.strokeStyle = '#4ade8030';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(PAD - 10, perfY); ctx.lineTo(W - 10, perfY); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#4ade8060';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('gradient = 1.0 (ideal)', PAD - 5, perfY - 3);

  // bars
  for (var l2 = 0; l2 < curLayers; l2++) {
    var gv = grads[l2];
    var barH = Math.max(1, (gv / 1.0) * (H - 44 - 20));
    var bx = PAD - 10 + l2 * barSpacing + (barSpacing - barW) / 2;
    var by = H - 44 - barH;

    // color fades from green -> orange -> red based on gradient magnitude
    var intensity = gv;
    var r2 = intensity < 0.1 ? 239 : intensity < 0.5 ? 251 : 74;
    var g2 = intensity < 0.1 ? 68  : intensity < 0.5 ? 191 : 222;
    var b2 = intensity < 0.1 ? 68  : intensity < 0.5 ? 36  : 128;
    var barColor = 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';

    ctx.fillStyle = barColor + '50';
    ctx.strokeStyle = barColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(bx, by, barW, barH, 2);
    else { ctx.rect(bx, by, barW, barH); }
    ctx.fill();
    ctx.stroke();

    // value label on top of bar
    if (barH > 14 || l2 < 3) {
      ctx.fillStyle = barColor;
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      var labelV = gv < 0.0001 ? gv.toExponential(0) : gv.toFixed(gv < 0.01 ? 4 : gv < 0.1 ? 3 : 2);
      ctx.fillText(labelV, bx + barW/2, Math.max(20, by - 2));
    }

    // layer label
    ctx.fillStyle = '#3f3f46';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('L' + (l2+1), bx + barW/2, H - 30);
  }

  // final gradient annotation
  var finalG = grads[grads.length - 1];
  var finalText = finalG < 0.0001 ? finalG.toExponential(2) : finalG.toFixed(4);
  var finalColor = finalG < 0.01 ? '#ef4444' : finalG < 0.5 ? '#fbbf24' : '#4ade80';
  ctx.fillStyle = finalColor;
  ctx.textAlign = 'center';
  ctx.font = 'bold 11px JetBrains Mono, monospace';
  ctx.fillText('After ' + curLayers + ' layers: gradient = ' + finalText, W/2, H - 12);

  // info box
  var info = VG_INFO[curVG];
  document.getElementById('vgInfo').innerHTML =
    '<span class="hl">' + info.title + '</span><br>' + info.body;
}

// ============================================================
// LOSS FUNCTIONS
// ============================================================
var LOSS = {
  mse: {
    label: 'MSE (L2)',
    color: '#4ecdc4',
    // x = prediction (yhat), true y = 1, so loss = (1 - yhat)^2
    fn:   function(yh) { return (1 - yh) * (1 - yh); },
    grad: function(yh) { return Math.abs(-2 * (1 - yh)); },
    xmin: -0.5, xmax: 2.0,
    xLabel: 'prediction yhat  (true y = 1)',
    sliderMin: -0.49, sliderMax: 1.99,
    sliderStart: 0.5,
    formula: 'L = (1/N) \u03A3 (y \u2212 \u0177)\u00B2',
    gradient: '-2(y \u2212 \u0177) / N  &mdash; proportional to error',
    prob: 'Assumes y | x ~ Gaussian. Minimising MSE = MLE under Gaussian.',
    when: 'Regression with roughly Gaussian errors, no significant outliers.',
    avoid: 'Outlier-heavy data. A single error of 10 outweighs 100 errors of 0.3.',
    verdict: 'ok'
  },
  mae: {
    label: 'MAE (L1)',
    color: '#fbbf24',
    fn:   function(yh) { return Math.abs(1 - yh); },
    grad: function(yh) { return yh > 1 ? 1 : (yh < 1 ? 1 : 0); },
    xmin: -0.5, xmax: 2.0,
    xLabel: 'prediction yhat  (true y = 1)',
    sliderMin: -0.49, sliderMax: 1.99,
    sliderStart: 0.5,
    formula: 'L = (1/N) \u03A3 |y \u2212 \u0177|',
    gradient: '\u00B11/N  &mdash; constant regardless of error size',
    prob: 'Assumes y | x ~ Laplace. Minimising MAE = median regression.',
    when: 'Regression with outliers. Want robust predictions. Median target.',
    avoid: 'When smooth gradients needed near minimum (grad undefined at 0).',
    verdict: 'ok'
  },
  huber: {
    label: 'Huber (\u03B4=1)',
    color: '#a78bfa',
    fn:   function(yh) {
      var e = Math.abs(1 - yh);
      return e <= 1 ? 0.5 * e * e : 1 * (e - 0.5);
    },
    grad: function(yh) {
      var e = Math.abs(1 - yh);
      return e <= 1 ? e : 1;
    },
    xmin: -0.5, xmax: 2.0,
    xLabel: 'prediction yhat  (true y = 1)',
    sliderMin: -0.49, sliderMax: 1.99,
    sliderStart: 0.5,
    formula: '(1/2)|e|\u00B2 if |e|\u2264\u03B4 else \u03B4(|e|\u22120.5\u03B4)',
    gradient: 'e if |e|\u2264\u03B4 else \u00B1\u03B4 &mdash; clipped like MAE, smooth like MSE',
    prob: 'Best of both worlds: MSE near minimum, MAE for outliers.',
    when: 'Regression with potential outliers. Object detection (bounding boxes). RL.',
    avoid: 'When \u03B4 is hard to tune. For very clean data, MSE is simpler.',
    verdict: 'good'
  },
  bce: {
    label: 'BCE (Log Loss)',
    color: '#ff6b35',
    fn:   function(p) { return -Math.log(Math.max(p, 1e-7)); },
    grad: function(p) { return 1 / Math.max(p, 1e-7); },
    xmin: 0.01, xmax: 0.99,
    xLabel: 'predicted probability p  (true y = 1)',
    sliderMin: 0.01, sliderMax: 0.99,
    sliderStart: 0.5,
    formula: 'L = \u2212(y log(p) + (1\u2212y) log(1\u2212p))',
    gradient: '\u2202L/\u2202z = p\u0302 \u2212 y &mdash; just the prediction error (with sigmoid)',
    prob: 'Assumes y | x ~ Bernoulli. MLE under Bernoulli = BCE.',
    when: 'Binary classification. Multi-label (BCE per label). Always pair with Sigmoid.',
    avoid: 'Multi-class (use Categorical CE + Softmax). Passing non-probabilities.',
    verdict: 'good'
  },
  hinge: {
    label: 'Hinge (SVM)',
    color: '#38bdf8',
    fn:   function(yh) { return Math.max(0, 1 - yh); },
    grad: function(yh) { return yh < 1 ? 1 : 0; },
    xmin: -1, xmax: 2.5,
    xLabel: 'y \u00B7 score  (margin)',
    sliderMin: -0.99, sliderMax: 2.49,
    sliderStart: 0.5,
    formula: 'L = (1/N) \u03A3 max(0, 1 \u2212 y \u00B7 \u0177)',
    gradient: '\u22121 if margin < 1 else 0 &mdash; zero loss once correctly classified',
    prob: 'No probabilistic interpretation. Pure margin maximisation (SVM).',
    when: 'SVMs. Max-margin classification. When you need a hard decision boundary.',
    avoid: 'When you need calibrated probabilities. Neural nets prefer BCE/CE.',
    verdict: 'ok'
  },
  focal: {
    label: 'Focal (\u03B3=2)',
    color: '#f472b6',
    fn:   function(p) {
      var gamma = 2;
      return -Math.pow(1 - p, gamma) * Math.log(Math.max(p, 1e-7));
    },
    grad: function(p) {
      var gamma = 2;
      var pt = Math.max(p, 1e-7);
      return Math.pow(1-pt, gamma) / pt + gamma * Math.pow(1-pt, gamma-1) * Math.log(pt);
    },
    xmin: 0.01, xmax: 0.99,
    xLabel: 'p_t (probability of correct class)',
    sliderMin: 0.01, sliderMax: 0.99,
    sliderStart: 0.3,
    formula: 'L = \u2212(1\u2212p_t)\u02E3 log(p_t)  (\u03B3=2)',
    gradient: 'Down-weights easy examples by (1\u2212p_t)\u00B2',
    prob: 'Extension of BCE with modulating factor. Proposed by Lin et al. (2017).',
    when: 'Severe class imbalance. Object detection (RetinaNet). Easy negatives dominate.',
    avoid: 'Balanced datasets (standard BCE is fine). \u03B3=0 gives plain BCE.',
    verdict: 'good'
  }
};

var curLoss = 'mse';
var curLossX = 0.5;

function selectLoss(key) {
  curLoss = key;
  document.querySelectorAll('[id^="lBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('lBtn_' + key).classList.add('active');
  var lo = LOSS[key];
  document.getElementById('lSlider').min = lo.sliderMin;
  document.getElementById('lSlider').max = lo.sliderMax;
  curLossX = lo.sliderStart;
  document.getElementById('lSlider').value = curLossX;
  document.getElementById('lSliderVal').textContent = curLossX.toFixed(2);
  document.getElementById('lSliderLabel').textContent = key === 'bce' || key === 'focal' ? 'prob. p' : 'prediction';
  renderLoss();
}

function updateLoss(v) {
  curLossX = parseFloat(v);
  document.getElementById('lSliderVal').textContent = curLossX.toFixed(2);
  renderLoss();
}

function renderLoss() {
  var lo = LOSS[curLoss];
  var W = 420, H1 = 210, H2 = 110, PAD = 36;
  var xmin = lo.xmin, xmax = lo.xmax;

  // compute ymax
  var vals = [];
  for (var i = 0; i <= 200; i++) {
    var xv = xmin + i/200*(xmax-xmin);
    var v = lo.fn(xv);
    if (isFinite(v)) vals.push(v);
  }
  var rawMax = Math.min(Math.max.apply(null, vals), 5);
  var ymax = Math.ceil(rawMax * 1.1);

  // loss curve
  var r1 = clearCanvas('cvLoss', W, H1);
  var ctx1 = r1.ctx;
  drawAxes(ctx1, W, H1, xmin, xmax, 0, ymax, PAD, lo.xLabel, 'Loss');
  drawCurve(ctx1, lo.fn, xmin, xmax, 0, ymax, W, H1, PAD, lo.color, 2.5);
  drawDot(ctx1, curLossX, lo.fn, xmin, xmax, 0, ymax, W, H1, PAD, lo.color);

  // gradient magnitude
  var gvals = [];
  for (var i2 = 0; i2 <= 200; i2++) {
    var xv2 = xmin + i2/200*(xmax-xmin);
    var gv2 = lo.grad(xv2);
    if (isFinite(gv2)) gvals.push(gv2);
  }
  var gmaxRaw = Math.min(Math.max.apply(null, gvals), 10);
  var gmax = Math.ceil(gmaxRaw * 1.2);

  var r2 = clearCanvas('cvLossGrad', W, H2);
  var ctx2 = r2.ctx;
  drawAxes(ctx2, W, H2, xmin, xmax, 0, gmax, PAD, '', '|grad|');
  drawCurve(ctx2, lo.grad, xmin, xmax, 0, gmax, W, H2, PAD, lo.color, 2.5);
  drawDot(ctx2, curLossX, lo.grad, xmin, xmax, 0, gmax, W, H2, PAD, lo.color);

  var lv = lo.fn(curLossX);
  var gv3 = lo.grad(curLossX);
  var lvStr = isFinite(lv) ? lv.toFixed(4) : 'Inf';
  var gvStr = isFinite(gv3) ? gv3.toFixed(4) : 'Inf';

  var verdictMap = { good:'tag-green', ok:'tag-orange', bad:'tag-red', best:'tag-blue' };
  var verdictText = { good:'Recommended', ok:'Use with care', bad:'Avoid', best:'Best choice' };
  var vc = verdictMap[lo.verdict] || 'tag-orange';

  document.getElementById('lossInfo').innerHTML =
    '<span class="hl">' + lo.label + '</span> &nbsp; <span class="badge ' + vc + '">' + verdictText[lo.verdict] + '</span>' +
    '<br><br>' +
    '<span class="hl2">Formula:</span><br>' + lo.formula +
    '<br><br>' +
    '<span class="hl2">Gradient:</span><br>' + lo.gradient +
    '<br><br>' +
    '<span class="hl2">Probabilistic basis:</span><br>' + lo.prob +
    '<br><br>' +
    '<span class="hl4">Use when:</span><br>' + lo.when +
    '<br><br>' +
    '<span class="hl5">Avoid when:</span><br>' + lo.avoid +
    '<br><br>' +
    'At x = <span class="hl">' + curLossX.toFixed(3) + '</span>:' +
    '&nbsp; Loss = <span class="hl3">' + lvStr + '</span>' +
    '&nbsp; |Grad| = <span class="hl2">' + gvStr + '</span>';
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load', function() {
  renderAct();
  renderVG();
  renderLoss();
});
</script>

</body>
</html>"""

ACTIVATIONS_VISUAL_HEIGHT = 2100