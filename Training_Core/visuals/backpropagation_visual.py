BACKPROP_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: #08080f; color: #e4e4e7; padding: 20px; min-height: 100vh; }
h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.15em; color: #ff6b35; margin-bottom: 3px; }
.subtitle { color: #52525b; font-size: 0.8em; margin-bottom: 20px; }
.card { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 18px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); margin-bottom: 14px; }
.card h3 { font-family: 'JetBrains Mono', monospace; font-size: 0.78em; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #ff6b35; margin-bottom: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
canvas { display: block; border-radius: 8px; background: #09090f; }
.row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.row label { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #71717a; min-width: 100px; }
input[type=range] { flex: 1; height: 4px; border-radius: 2px; -webkit-appearance: none; appearance: none; background: #1e1e2e; outline: none; cursor: pointer; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #ff6b35; cursor: pointer; }
.val { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #ff6b35; min-width: 50px; text-align: right; }
.info-box { background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px; padding: 10px 14px; font-size: 0.78em; color: #94a3b8; line-height: 1.8; margin-top: 10px; font-family: 'JetBrains Mono', monospace; }
.hl  { color: #ff6b35; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }
.hl4 { color: #4ade80; font-weight: 700; }
.hl5 { color: #ef4444; font-weight: 700; }
.hl6 { color: #c084fc; font-weight: 700; }
.btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
button { background: #1e1e2e; color: #a1a1aa; border: 1px solid #2d2d40; border-radius: 6px; padding: 5px 12px; cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 0.75em; font-weight: 700; transition: all 0.15s; }
button:hover { background: #2d2d40; color: #e4e4e7; }
button.active { background: #ff6b35; color: #08080f; border-color: #ff6b35; }
button.nav { background: #1a1a28; color: #e4e4e7; border: 1px solid #2d2d40; padding: 6px 18px; border-radius: 8px; font-size: 0.8em; }
button.nav:disabled { opacity: 0.3; cursor: default; }
button.nav.go { background: #0a1f0f; color: #4ade80; border-color: #4ade8040; }
.step-dots { display: flex; gap: 5px; align-items: center; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #1e1e2e; border: 1px solid #2d2d40; cursor: pointer; transition: all 0.2s; }
.dot.done { background: #4ade8060; border-color: #4ade80; }
.dot.active { background: #ff6b35; border-color: #ff6b35; width: 22px; border-radius: 4px; }
.tbl { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.72em; }
.tbl th { padding: 7px 10px; text-align: left; color: #52525b; border-bottom: 2px solid #1e1e2e; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; }
.tbl td { padding: 7px 10px; border-bottom: 1px solid #151520; color: #94a3b8; line-height: 1.5; vertical-align: top; }
.tbl tr:hover td { background: #0d0d18; }
</style>
</head>
<body>

<h2>&#x1F501; Backpropagation &mdash; How Neural Networks Learn</h2>
<p class="subtitle">Chain rule &middot; Computation graph &middot; Forward &amp; backward pass &middot; Gate gradients &middot; Gradient flow &middot; Skip connections</p>

<!-- ================================================================
     PANEL 1: STEP-BY-STEP WALKTHROUGH
     ================================================================ -->
<div class="card">
  <h3>&#9312; Step-by-Step: Forward Pass Then Backward Pass</h3>
  <p style="font-size:0.75em;color:#52525b;font-family:'JetBrains Mono',monospace;margin-bottom:10px;line-height:1.6;">
    A single neuron: x=2, w=0.5, b=-1, y_true=1 &nbsp;|&nbsp; L = 0.5*(z-y)&sup2; &nbsp;|&nbsp;
    Step through forward then backward &mdash; watch values and gradients flow.
  </p>
  <canvas id="cvGraph" width="860" height="280"></canvas>
  <div style="display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap;">
    <button class="nav" id="btnPrev" onclick="prevStep()" disabled>&#8592; Back</button>
    <div class="step-dots" id="stepDots"></div>
    <button class="nav go" id="btnNext" onclick="nextStep()">Next &#8594;</button>
    <div style="flex:1"></div>
    <button onclick="resetSteps()" style="font-size:0.7em;padding:4px 10px;">Reset</button>
  </div>
  <div class="info-box" id="stepInfo">Press <span class="hl4">Next</span> to begin the forward pass.</div>
</div>


<!-- ================================================================
     PANEL 2: OPERATION GATE EXPLORER  +  PANEL 3: GRADIENT FLOW
     ================================================================ -->
<div class="grid2">

<!-- PANEL 2 -->
<div class="card">
  <h3>&#9313; Operation Gate &mdash; How Each Op Passes Gradients</h3>
  <p style="font-size:0.75em;color:#52525b;font-family:'JetBrains Mono',monospace;margin-bottom:8px;line-height:1.6;">
    Each node receives an upstream gradient and produces downstream gradients.<br>
    Select an operation to see exactly what happens.
  </p>
  <div class="btn-row">
    <button id="gBtn_add"   class="active" onclick="selectGate('add')">Add (+)</button>
    <button id="gBtn_mul"   onclick="selectGate('mul')">Multiply (x)</button>
    <button id="gBtn_relu"  onclick="selectGate('relu')">ReLU</button>
    <button id="gBtn_sig"   onclick="selectGate('sig')">Sigmoid</button>
  </div>
  <canvas id="cvGate" width="420" height="260"></canvas>
  <div class="row">
    <label>Upstream grad</label>
    <input type="range" id="slUpstream" min="-3" max="3" step="0.1" value="1.0" oninput="updateGate()">
    <span class="val" id="valUpstream">1.00</span>
  </div>
  <div class="row">
    <label>Input z</label>
    <input type="range" id="slGateZ" min="-4" max="4" step="0.1" value="1.0" oninput="updateGate()">
    <span class="val" id="valGateZ">1.00</span>
  </div>
  <div class="row" id="rowSecond">
    <label>Input y</label>
    <input type="range" id="slGateY" min="-4" max="4" step="0.1" value="0.5" oninput="updateGate()">
    <span class="val" id="valGateY">0.50</span>
  </div>
  <div class="info-box" id="gateInfo">Loading...</div>
</div>


<!-- PANEL 3 -->
<div class="card">
  <h3>&#9314; Gradient Flow Through Layers</h3>
  <p style="font-size:0.75em;color:#52525b;font-family:'JetBrains Mono',monospace;margin-bottom:8px;line-height:1.6;">
    Gradient magnitude at each layer. Sigmoid compounds attenuation.<br>ReLU preserves it. Skip connections guarantee a gradient highway.
  </p>
  <div class="btn-row">
    <button id="fBtn_sig"  class="active" onclick="selectFlow('sig')">Sigmoid</button>
    <button id="fBtn_relu" onclick="selectFlow('relu')">ReLU</button>
    <button id="fBtn_skip" onclick="selectFlow('skip')">Skip (ResNet)</button>
  </div>
  <canvas id="cvFlow" width="420" height="260"></canvas>
  <div class="row">
    <label>Num layers</label>
    <input type="range" id="slLayers" min="1" max="20" step="1" value="10" oninput="updateFlow()">
    <span class="val" id="valLayers">10</span>
  </div>
  <div class="info-box" id="flowInfo">Loading...</div>
</div>

</div><!-- end grid2 -->


<!-- ================================================================
     PANEL 4: FORWARD vs BACKWARD SUMMARY TABLE  (full width)
     ================================================================ -->
<div class="card">
  <h3>&#9315; Forward Pass vs Backward Pass &mdash; What Happens at Each Layer</h3>
  <div class="grid2" style="gap:20px;">
    <div>
      <div style="font-size:0.72em;font-weight:800;color:#4ecdc4;font-family:'JetBrains Mono',monospace;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">Forward Pass (left &#8594; right)</div>
      <table class="tbl">
        <thead><tr><th>Step</th><th>Computation</th><th>Purpose</th></tr></thead>
        <tbody>
          <tr><td><span class="hl2">1</span></td><td>z[l] = W[l] &times; a[l-1] + b[l]</td><td>Linear transform</td></tr>
          <tr><td><span class="hl2">2</span></td><td>a[l] = activation(z[l])</td><td>Non-linearity</td></tr>
          <tr><td><span class="hl2">3</span></td><td>Cache z[l] and a[l-1]</td><td><span class="hl5">Must store for backward!</span></td></tr>
          <tr><td><span class="hl2">4</span></td><td>Repeat for all L layers</td><td>Propagate forward</td></tr>
          <tr><td><span class="hl2">5</span></td><td>L = Loss(a[L], y)</td><td>Compute final loss</td></tr>
        </tbody>
      </table>
      <div style="margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:0.72em;color:#52525b;line-height:1.8;">
        <span class="hl">Memory cost:</span> O(L &times; batch &times; width)<br>
        Everything cached because backward needs it.<br>
        This is why deep nets use lots of GPU memory.
      </div>
    </div>
    <div>
      <div style="font-size:0.72em;font-weight:800;color:#c084fc;font-family:'JetBrains Mono',monospace;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">Backward Pass (right &#8592; left)</div>
      <table class="tbl">
        <thead><tr><th>Step</th><th>Computation</th><th>Purpose</th></tr></thead>
        <tbody>
          <tr><td><span class="hl6">1</span></td><td>dL/da[L] = &part;Loss/&part;a[L]</td><td>Start gradient at output</td></tr>
          <tr><td><span class="hl6">2</span></td><td>dL/dz[l] = dL/da[l] &times; f'(z[l])</td><td>Through activation (uses z[l])</td></tr>
          <tr><td><span class="hl6">3</span></td><td>dL/dW[l] = dL/dz[l] &times; a[l-1]&sup2;</td><td>Weight gradient (uses a[l-1])</td></tr>
          <tr><td><span class="hl6">4</span></td><td>dL/db[l] = sum(dL/dz[l])</td><td>Bias gradient</td></tr>
          <tr><td><span class="hl6">5</span></td><td>dL/da[l-1] = W[l]&sup2; &times; dL/dz[l]</td><td>Pass gradient to prev layer</td></tr>
        </tbody>
      </table>
      <div style="margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:0.72em;color:#52525b;line-height:1.8;">
        <span class="hl">Cost:</span> 1 forward + 1 backward pass = ALL gradients.<br>
        Numerical diff needs N forward passes (N = # params).<br>
        <span class="hl4">Backprop is N&times; faster</span> (N can be billions).
      </div>
    </div>
  </div>
</div>

<script>
// ============================================================
// UTILITIES
// ============================================================
function sigmoid(z) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z)))); }
function relu(z) { return Math.max(0, z); }
function rnd(v, d) { return typeof v === 'number' ? v.toFixed(d === undefined ? 4 : d) : v; }

function fillRect(ctx, x, y, w, h, r, fill, stroke, sw) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else { ctx.rect(x, y, w, h); }
  if (fill)   { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw || 1.5; ctx.stroke(); }
}

function text(ctx, str, x, y, col, size, weight, align) {
  ctx.fillStyle = col || '#e4e4e7';
  ctx.font = (weight || '500') + ' ' + (size || 11) + 'px JetBrains Mono, monospace';
  ctx.textAlign = align || 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(str, x, y);
}

function arrow(ctx, x1, y1, x2, y2, col, lw, label, labelColor) {
  ctx.strokeStyle = col || '#4ecdc4';
  ctx.lineWidth = lw || 2;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  // arrowhead
  var angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.fillStyle = col || '#4ecdc4';
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 9*Math.cos(angle-0.4), y2 - 9*Math.sin(angle-0.4));
  ctx.lineTo(x2 - 9*Math.cos(angle+0.4), y2 - 9*Math.sin(angle+0.4));
  ctx.closePath(); ctx.fill();
  if (label) {
    var mx = (x1+x2)/2, my = (y1+y2)/2;
    text(ctx, label, mx, my - 10, labelColor || col, 9, '700');
  }
}

// ============================================================
// PANEL 1: STEP-BY-STEP GRAPH
// ============================================================
// Network: x --[*w]--> u --[+b]--> z --[-y]--> e --[*0.5]--> L
// x=2, w=0.5, b=-1, y=1

var X_VAL=2, W_VAL=0.5, B_VAL=-1, Y_VAL=1;
var U_VAL, Z_VAL, E_VAL, L_VAL;
var DL_DE, DL_DZ, DL_DU, DL_DB, DL_DW, DL_DX;

function computeForward() {
  U_VAL = X_VAL * W_VAL;
  Z_VAL = U_VAL + B_VAL;
  E_VAL = Z_VAL - Y_VAL;
  L_VAL = 0.5 * E_VAL * E_VAL;
  DL_DE = E_VAL;
  DL_DZ = DL_DE * 1;
  DL_DU = DL_DZ * 1;
  DL_DB = DL_DU * 1;
  DL_DW = DL_DU * X_VAL;
  DL_DX = DL_DU * W_VAL;
}
computeForward();

var curStep = -1; // -1 = start screen

var STEPS = [
  // Forward steps
  { phase:'fwd', node:'u',
    title:'Forward Step 1: Multiply  u = x \u00D7 w',
    body:'<span class="hl2">u = x \u00D7 w = 2 \u00D7 0.5 = <span class="hl">1.0</span></span><br>This is the linear part \u2014 the input scaled by the weight.<br>This value is <span class="hl5">CACHED</span> because the backward pass needs it to compute dL/dw.' },
  { phase:'fwd', node:'z',
    title:'Forward Step 2: Add bias  z = u + b',
    body:'<span class="hl2">z = u + b = 1.0 + (\u22121) = <span class="hl">0.0</span></span><br>The bias shifts the output. No activation here in this simple graph.<br>z represents the pre-activation value \u2014 also cached.' },
  { phase:'fwd', node:'e',
    title:'Forward Step 3: Error  e = z \u2212 y',
    body:'<span class="hl2">e = z \u2212 y = 0.0 \u2212 1 = <span class="hl5">\u22121.0</span></span><br>e is the prediction error (how far off we are from the true label y=1).<br>Negative means we predicted too low \u2014 the network under-predicted.' },
  { phase:'fwd', node:'L',
    title:'Forward Step 4: Loss  L = \u00BD e\u00B2',
    body:'<span class="hl2">L = 0.5 \u00D7 e\u00B2 = 0.5 \u00D7 (\u22121)\u00B2 = <span class="hl3">0.5</span></span><br>MSE loss \u2014 squaring ensures loss is always positive and penalises large errors more.<br>Forward pass complete. Now we reverse and compute gradients.' },

  // Backward steps
  { phase:'bwd', node:'e',
    title:'Backward Step 1: dL/de \u2014 gradient into loss',
    body:'<span class="hl6">dL/de = e = \u22121.0</span><br>For L = \u00BDe\u00B2: dL/de = e.<br>This is the <span class="hl">upstream gradient</span> entering node e from the right.<br>Negative: the loss decreases if we increase e (makes sense \u2014 e was -1, so we want it to grow toward 0).' },
  { phase:'bwd', node:'z',
    title:'Backward Step 2: dL/dz \u2014 through the subtraction',
    body:'<span class="hl6">dL/dz = dL/de \u00D7 de/dz = \u22121.0 \u00D7 1 = \u22121.0</span><br>e = z \u2212 y, so de/dz = 1. The subtraction gate passes gradient through unchanged.<br>Chain rule: upstream gradient \u00D7 local gradient = \u22121.0 \u00D7 1 = \u22121.0.' },
  { phase:'bwd', node:'u',
    title:'Backward Step 3: dL/du \u2014 through the addition',
    body:'<span class="hl6">dL/du = dL/dz \u00D7 dz/du = \u22121.0 \u00D7 1 = \u22121.0</span><br>z = u + b, so dz/du = 1. Addition gates pass gradient unchanged.<br>Also: <span class="hl3">dL/db = dL/dz \u00D7 dz/db = \u22121.0 \u00D7 1 = \u22121.0</span> \u2190 bias gradient ready!' },
  { phase:'bwd', node:'w',
    title:'Backward Step 4: dL/dw \u2014 weight gradient!',
    body:'<span class="hl4">dL/dw = dL/du \u00D7 du/dw = \u22121.0 \u00D7 x = \u22121.0 \u00D7 2 = \u22122.0</span><br>u = x \u00D7 w, so du/dw = x (the OTHER operand in a multiply gate).<br>Gradient is scaled by the INPUT value. <span class="hl4">w_new = 0.5 \u2212 0.1 \u00D7 (\u22122.0) = 0.7</span> \u2190 weight increases!' },
  { phase:'bwd', node:'x',
    title:'Backward Step 5: dL/dx \u2014 gradient to previous layer',
    body:'<span class="hl6">dL/dx = dL/du \u00D7 du/dx = \u22121.0 \u00D7 w = \u22121.0 \u00D7 0.5 = \u22120.5</span><br>Multiply gate: gradient to x is scaled by the OTHER operand (w).<br>In a deep network, this dL/dx becomes the <span class="hl">upstream gradient for the previous layer</span>.<br><span class="hl4">All gradients computed in ONE backward pass. Training complete for this step!</span>' },
];

var totalSteps = STEPS.length;

function buildDots() {
  var el = document.getElementById('stepDots');
  el.innerHTML = '';
  for (var i = 0; i < totalSteps; i++) {
    var d = document.createElement('div');
    d.className = 'dot' + (i === curStep ? ' active' : (i < curStep ? ' done' : ''));
    (function(ii) { d.onclick = function() { goStep(ii); }; })(i);
    el.appendChild(d);
  }
}

function drawGraph() {
  var cv = document.getElementById('cvGraph');
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#09090f'; ctx.fillRect(0, 0, W, H);

  // Layout
  // Nodes: x | *w | u | +b | z | -y | e | *0.5 | L
  var NODE_Y = 110;
  var xs = [50, 145, 230, 325, 410, 505, 590, 685, 780];
  var labels = ['x', '\u00D7w', 'u', '+b', 'z', '\u2212y', 'e', '\u00BDe\u00B2', 'L'];
  var vals   = ['2', null, '1.0', null, '0.0', null, '\u22121.0', null, '0.5'];
  var isOp   = [false, true, false, true, false, true, false, true, false];

  var step = curStep < 0 ? null : STEPS[curStep];
  var phase = step ? step.phase : null;

  // Determine which nodes are lit
  var fwdDone = { x:true };
  var bwdActive = {};
  if (step) {
    if (phase === 'fwd') {
      if (step.node === 'u') { fwdDone.u = true; fwdDone['*w'] = true; }
      if (step.node === 'z') { fwdDone.u = true; fwdDone['*w'] = true; fwdDone.z = true; fwdDone['+b'] = true; }
      if (step.node === 'e') { fwdDone.u = true; fwdDone['*w'] = true; fwdDone.z = true; fwdDone['+b'] = true; fwdDone.e = true; fwdDone['-y'] = true; }
      if (step.node === 'L') { fwdDone.u = true; fwdDone['*w'] = true; fwdDone.z = true; fwdDone['+b'] = true; fwdDone.e = true; fwdDone['-y'] = true; fwdDone.L = true; fwdDone['bde2'] = true; }
    } else {
      // all forward done
      fwdDone = { x:true, u:true, '*w':true, z:true, '+b':true, e:true, '-y':true, L:true, 'bde2':true };
      bwdActive[step.node] = true;
    }
  }

  // Node keys for lookup
  var nodeKeys = ['x', '*w', 'u', '+b', 'z', '-y', 'e', 'bde2', 'L'];
  var nodeValMap = { x:'2', u:'1.0', z:'0.0', e:'\u22121.0', L:'0.5' };

  // Gradient values
  var gradMap = {
    'L':   'dL/dL=1',
    'e':   'dL/de='+rnd(DL_DE,1),
    'z':   'dL/dz='+rnd(DL_DZ,1),
    'u':   'dL/du='+rnd(DL_DU,1),
    'w':   'dL/dw='+rnd(DL_DW,1),
    'x':   'dL/dx='+rnd(DL_DX,1),
    'b':   'dL/db='+rnd(DL_DB,1)
  };

  // Which backward grads are visible
  var bwdGradDone = {};
  if (step && phase === 'bwd') {
    var bwdOrder = ['e','z','u','w','x'];
    for (var i2 = 0; i2 <= bwdOrder.indexOf(step.node); i2++) {
      bwdGradDone[bwdOrder[i2]] = true;
    }
    if (step.node === 'u') bwdGradDone['b'] = true;
  }

  // Draw forward arrows
  for (var n = 0; n < xs.length - 1; n++) {
    var active2 = fwdDone[nodeKeys[n]] && fwdDone[nodeKeys[n+1]];
    var col = active2 ? '#4ecdc4' : '#1e1e2e';
    arrow(ctx, xs[n]+26, NODE_Y, xs[n+1]-26, NODE_Y, col, active2 ? 2 : 1);
  }

  // Draw backward arrows (above the main line)
  var bwdY = 60;
  var bwdPairs = [ // [from_x_idx, to_x_idx, grad_key]
    [8, 6, 'e'], [6, 4, 'z'], [4, 2, 'u'], [2, 0, 'x']
  ];
  bwdPairs.forEach(function(bp) {
    var fromX = xs[bp[0]]; var toX = xs[bp[2]]; var gk = bp[3];
    var isVis = bwdGradDone[gk];
    var bCol = isVis ? '#c084fc' : '#1e1e2e';
    var lw2 = isVis ? 2 : 1;
    arrow(ctx, fromX, bwdY, toX+26, bwdY, bCol, lw2);
    if (isVis) {
      var gStr = gradMap[gk];
      text(ctx, gStr, (fromX+toX+26)/2, bwdY - 12, '#c084fc', 9, '700');
    }
  });

  // dL/db branch (down from z)
  if (bwdGradDone['b']) {
    var bx = xs[4];
    ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 1.5;
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(bx, bwdY+2); ctx.lineTo(bx, NODE_Y+60); ctx.stroke();
    ctx.setLineDash([]);
    text(ctx, 'dL/db='+rnd(DL_DB,1), bx, NODE_Y+75, '#c084fc', 9, '700');
    text(ctx, 'b='+B_VAL, bx, NODE_Y+90, '#52525b', 9, '400');
  }

  // dL/dw branch (down from u)
  if (bwdGradDone['w']) {
    var ux = xs[2];
    ctx.strokeStyle = '#ff6b35'; ctx.lineWidth = 1.5;
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(ux, bwdY+2); ctx.lineTo(ux, NODE_Y+60); ctx.stroke();
    ctx.setLineDash([]);
    text(ctx, 'dL/dw='+rnd(DL_DW,1), ux, NODE_Y+75, '#ff6b35', 9, '700');
    text(ctx, 'w='+W_VAL, ux, NODE_Y+90, '#52525b', 9, '400');
  }

  // Draw nodes
  for (var n2 = 0; n2 < xs.length; n2++) {
    var nx = xs[n2], ny = NODE_Y;
    var key = nodeKeys[n2];
    var isDone = !!fwdDone[key];
    var isOpNode = isOp[n2];
    var isBwdAct = step && phase === 'bwd' && bwdActive[labels[n2] === '\u22121.0' ? 'e' : labels[n2]];
    // Active in bwd?
    var bwdNodeActive = false;
    if (step && phase === 'bwd') {
      if (step.node==='e' && (n2===6||n2===7)) bwdNodeActive=true;
      if (step.node==='z' && (n2===4||n2===5)) bwdNodeActive=true;
      if (step.node==='u' && (n2===2||n2===3)) bwdNodeActive=true;
      if (step.node==='w' && n2===2) bwdNodeActive=true;
      if (step.node==='x' && n2===0) bwdNodeActive=true;
    }
    var fwdNodeActive = step && phase==='fwd' && (
      (step.node==='u' && (n2===1||n2===2)) ||
      (step.node==='z' && (n2===3||n2===4)) ||
      (step.node==='e' && (n2===5||n2===6)) ||
      (step.node==='L' && (n2===7||n2===8))
    );

    var fillC = '#09090f';
    var strokeC = '#1e1e2e';
    var sw = 1.5;
    if (isOpNode && isDone) { fillC = '#1a1a28'; strokeC = '#4ecdc440'; }
    if (fwdNodeActive) { fillC = '#0a1f20'; strokeC = '#4ecdc4'; sw = 2.5; }
    if (bwdNodeActive) { fillC = '#150a20'; strokeC = '#c084fc'; sw = 2.5; }
    if (n2 === 8 && fwdDone.L) { fillC = '#1a110a'; strokeC = '#ff6b3580'; }

    var r = isOpNode ? 18 : 22;
    fillRect(ctx, nx-r, ny-r, r*2, r*2, isOpNode ? 6 : 22, fillC, strokeC, sw);

    var lbl = labels[n2];
    var lCol = isDone ? '#e4e4e7' : '#3f3f46';
    if (fwdNodeActive) lCol = '#4ecdc4';
    if (bwdNodeActive) lCol = '#c084fc';
    text(ctx, lbl, nx, ny, lCol, 12, '700');

    // Value below
    var vStr = nodeValMap[key];
    if (vStr && isDone) {
      text(ctx, vStr, nx, ny + r + 16, '#fbbf24', 10, '700');
    }

    // Legend above op nodes
    if (isOpNode) {
      var opDesc = {1:'multiply', 3:'add', 5:'subtract', 7:'0.5\u00D7'};
      if (opDesc[n2]) {
        text(ctx, opDesc[n2], nx, ny - r - 10, '#3f3f46', 8, '400');
      }
    }
  }

  // Legend
  text(ctx, 'FORWARD PASS \u2192', 420, 18, '#4ecdc460', 9, '800');
  text(ctx, '\u2190 BACKWARD PASS', 420, 38, '#c084fc60', 9, '800');
  text(ctx, 'x=2', xs[0], NODE_Y + 44, '#52525b', 9, '400');
  text(ctx, 'w=0.5', xs[0], NODE_Y + 55, '#52525b', 9, '400');
  text(ctx, 'y=1', xs[8], NODE_Y + 44, '#52525b', 9, '400');

  // Phase label
  if (step) {
    var phaseLabel = phase === 'fwd' ? 'FORWARD' : 'BACKWARD';
    var phaseCol = phase === 'fwd' ? '#4ecdc4' : '#c084fc';
    fillRect(ctx, 10, 248, 120, 22, 4, phaseCol+'15', phaseCol+'60', 1);
    text(ctx, phaseLabel + ' PASS', 70, 259, phaseCol, 9, '800');
    text(ctx, 'Step '+(curStep+1)+'/'+totalSteps, W-60, 259, '#52525b', 9, '400');
  }
}

function goStep(i) {
  curStep = i;
  updateStepUI();
}

function nextStep() {
  if (curStep < totalSteps - 1) { curStep++; updateStepUI(); }
}

function prevStep() {
  if (curStep > 0) { curStep--; updateStepUI(); }
  else { curStep = -1; updateStepUI(); }
}

function resetSteps() { curStep = -1; updateStepUI(); }

function updateStepUI() {
  buildDots();
  drawGraph();
  var el = document.getElementById('stepInfo');
  if (curStep < 0) {
    el.innerHTML = 'Press <span class="hl4">Next</span> to begin the forward pass. We will compute u=x\u00D7w, z=u+b, e=z\u2212y, L=\u00BDe\u00B2 then reverse.';
  } else {
    var s = STEPS[curStep];
    el.innerHTML = '<span class="'+(s.phase==='fwd'?'hl2':'hl6')+'">' + s.title + '</span><br>' + s.body;
  }
  document.getElementById('btnPrev').disabled = curStep <= 0 && curStep !== -1 ? true : (curStep === -1);
  document.getElementById('btnNext').disabled = curStep >= totalSteps - 1;
}

// ============================================================
// PANEL 2: GATE EXPLORER
// ============================================================
var curGate = 'add';

var GATES = {
  add: {
    label: 'Addition Gate: z = x + y',
    color: '#4ecdc4',
    inputs: ['x', 'y'],
    showY: true,
    localGradX: function(z, y) { return 1; },
    localGradY: function(z, y) { return 1; },
    forward: function(z, y) { return z + y; },
    gradX: function(up, z, y) { return up * 1; },
    gradY: function(up, z, y) { return up * 1; },
    desc: function(up, z, y) {
      return '<span class="hl2">Gate: out = x + y = '+rnd(z+y,2)+'</span><br>' +
        'Local gradient: \u2202out/\u2202x = 1,  \u2202out/\u2202y = 1<br>' +
        '<span class="hl4">dL/dx = upstream \u00D7 1 = '+rnd(up,2)+'</span><br>' +
        '<span class="hl4">dL/dy = upstream \u00D7 1 = '+rnd(up,2)+'</span><br>' +
        'Addition is a <span class="hl">gradient distributor</span> \u2014 passes the same gradient to BOTH inputs unchanged.<br>' +
        'This is why ResNet skip connections (x + F(x)) always provide a direct gradient path.';
    }
  },
  mul: {
    label: 'Multiply Gate: z = x \u00D7 y',
    color: '#fbbf24',
    inputs: ['x', 'y'],
    showY: true,
    forward: function(z, y) { return z * y; },
    gradX: function(up, z, y) { return up * y; },
    gradY: function(up, z, y) { return up * z; },
    desc: function(up, z, y) {
      return '<span class="hl3">Gate: out = x \u00D7 y = '+rnd(z*y,2)+'</span><br>' +
        'Local gradient: \u2202out/\u2202x = y = '+rnd(y,2)+',  \u2202out/\u2202y = x = '+rnd(z,2)+'<br>' +
        '<span class="hl4">dL/dx = upstream \u00D7 y = '+rnd(up,2)+' \u00D7 '+rnd(y,2)+' = '+rnd(up*y,3)+'</span><br>' +
        '<span class="hl4">dL/dy = upstream \u00D7 x = '+rnd(up,2)+' \u00D7 '+rnd(z,2)+' = '+rnd(up*z,3)+'</span><br>' +
        'Multiply is a <span class="hl">gradient scaler</span> \u2014 each input gets the gradient scaled by the <span class="hl5">OTHER</span> operand.<br>' +
        'If y is near 0, dL/dx is near 0 \u2014 the vanishing gradient in RNNs!';
    }
  },
  relu: {
    label: 'ReLU Gate: out = max(0, z)',
    color: '#4ade80',
    inputs: ['z'],
    showY: false,
    forward: function(z) { return Math.max(0, z); },
    gradX: function(up, z) { return up * (z > 0 ? 1 : 0); },
    desc: function(up, z) {
      var gate = z > 0 ? 'OPEN (z > 0)' : 'CLOSED (z \u2264 0)';
      var gateCol = z > 0 ? 'hl4' : 'hl5';
      return '<span class="hl4">Gate: out = max(0, '+rnd(z,2)+') = '+rnd(Math.max(0,z),2)+'</span><br>' +
        'Local gradient: f\'(z) = '+(z > 0 ? '1' : '0')+'  \u2014 Gate is <span class="'+gateCol+'">'+gate+'</span><br>' +
        '<span class="hl4">dL/dz = upstream \u00D7 '+(z>0?'1':'0')+' = '+rnd(z>0?up:0,3)+'</span><br>' +
        (z > 0 ?
          'Gradient passes through <span class="hl4">unchanged</span>. No vanishing for active neurons.' :
          '<span class="hl5">Gradient is KILLED (=0). If z stays negative for all inputs, this neuron is permanently dead (Dying ReLU).</span>');
    }
  },
  sig: {
    label: 'Sigmoid Gate: out = \u03C3(z)',
    color: '#ff6b35',
    inputs: ['z'],
    showY: false,
    forward: function(z) { return sigmoid(z); },
    gradX: function(up, z) { var s = sigmoid(z); return up * s * (1-s); },
    desc: function(up, z) {
      var s = sigmoid(z); var g = s*(1-s);
      var warning = Math.abs(z) > 2 ? '<br><span class="hl5">\u26A0 Saturation zone: gradient = '+rnd(g,4)+' \u2248 0. Severe vanishing gradient!</span>' : '';
      return '<span class="hl">Gate: out = \u03C3('+rnd(z,2)+') = '+rnd(s,4)+'</span><br>' +
        'Local gradient: \u03C3(z)(1\u2212\u03C3(z)) = '+rnd(s,3)+' \u00D7 '+rnd(1-s,3)+' = <span class="hl3">'+rnd(g,4)+'</span><br>' +
        '<span class="hl4">dL/dz = upstream \u00D7 '+rnd(g,4)+' = '+rnd(up*g,4)+'</span><br>' +
        'Max gradient = 0.25 (at z=0). After 10 layers: 0.25\u00B9\u2070 \u2248 10\u207B\u2076.' + warning;
    }
  }
};

function selectGate(k) {
  curGate = k;
  document.querySelectorAll('[id^="gBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('gBtn_' + k).classList.add('active');
  var g = GATES[k];
  document.getElementById('rowSecond').style.display = g.showY ? 'flex' : 'none';
  updateGate();
}

function updateGate() {
  var upstream = parseFloat(document.getElementById('slUpstream').value);
  var z2 = parseFloat(document.getElementById('slGateZ').value);
  var y2 = parseFloat(document.getElementById('slGateY').value);
  document.getElementById('valUpstream').textContent = rnd(upstream, 2);
  document.getElementById('valGateZ').textContent = rnd(z2, 2);
  document.getElementById('valGateY').textContent = rnd(y2, 2);

  var g = GATES[curGate];
  var out = g.forward(z2, y2);
  var dX = g.gradX(upstream, z2, y2);
  var dY = g.showY ? g.gradY(upstream, z2, y2) : null;

  // Draw gate diagram
  var cv = document.getElementById('cvGate');
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#09090f'; ctx.fillRect(0, 0, W, H);

  var gx = 210, gy = g.showY ? 130 : 130;
  var col = g.color;

  // Gate node
  fillRect(ctx, gx-35, gy-25, 70, 50, 8, col+'15', col, 2.5);
  text(ctx, g.showY ? 'op' : (curGate==='relu'?'ReLU':'\u03C3'), gx, gy, col, 14, '800');

  // Input x (from left)
  var xStart = 55;
  arrow(ctx, xStart + 30, g.showY ? gy-14 : gy, gx-36, g.showY ? gy-14 : gy, '#4ecdc4', 2);
  fillRect(ctx, xStart-25, (g.showY?gy-14:gy)-18, 50, 36, 20, '#0a1520', '#4ecdc4', 1.5);
  text(ctx, g.inputs[0]+'='+rnd(z2,2), xStart, (g.showY?gy-14:gy), '#4ecdc4', 10, '700');

  // Input y (from left, below)
  if (g.showY) {
    arrow(ctx, xStart + 30, gy+14, gx-36, gy+14, '#4ecdc4', 2);
    fillRect(ctx, xStart-25, gy+14-18, 50, 36, 20, '#0a1520', '#4ecdc4', 1.5);
    text(ctx, 'y='+rnd(y2,2), xStart, gy+14, '#4ecdc4', 10, '700');
  }

  // Output (to right)
  var outX = gx + 100;
  arrow(ctx, gx+36, gy, outX - 10, gy, '#fbbf24', 2);
  fillRect(ctx, outX-20, gy-18, 60, 36, 8, '#1a110a', '#fbbf2480', 1.5);
  text(ctx, rnd(out,3), outX+10, gy, '#fbbf24', 11, '700');
  text(ctx, 'output', outX+10, gy+22, '#52525b', 8, '400');

  // Upstream gradient (from right, backward)
  var upX = outX + 80;
  var bwdY2 = gy - 55;
  fillRect(ctx, upX-30, bwdY2-15, 70, 30, 6, '#150a20', '#c084fc80', 1.5);
  text(ctx, 'grad='+rnd(upstream,2), upX+5, bwdY2, '#c084fc', 9, '700');
  arrow(ctx, upX-30, bwdY2, gx+38, bwdY2, '#c084fc', 2);
  ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 1.5;
  ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(gx+38, bwdY2); ctx.lineTo(gx+38, gy-26); ctx.stroke();
  ctx.setLineDash([]);
  text(ctx, 'upstream', upX+5, bwdY2+18, '#52525b', 8, '400');

  // Downstream gradients
  var downY = gy + 60;
  // to x
  ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 1.5;
  ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(gx-36, g.showY?gy-14:gy); ctx.lineTo(xStart+30, g.showY?gy-14:gy); ctx.stroke();
  ctx.setLineDash([]);
  var dxY = g.showY ? gy-14 : gy;
  var dxLabel = (g.showY?'dL/dx':'dL/dz')+'='+rnd(dX,3);
  fillRect(ctx, xStart-38, dxY+5, 80, 22, 4, '#150a20', '#c084fc60', 1);
  text(ctx, dxLabel, xStart+2, dxY+16, '#c084fc', 8, '700');

  // to y (if applicable)
  if (g.showY && dY !== null) {
    ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 1.5;
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(gx-36, gy+14); ctx.lineTo(xStart+30, gy+14); ctx.stroke();
    ctx.setLineDash([]);
    var dyLabel = 'dL/dy='+rnd(dY,3);
    fillRect(ctx, xStart-38, gy+14+5, 80, 22, 4, '#150a20', '#c084fc60', 1);
    text(ctx, dyLabel, xStart+2, gy+14+16, '#c084fc', 8, '700');
  }

  // Title
  text(ctx, g.label, W/2, 18, col, 10, '700');
  text(ctx, 'FORWARD \u2192', W*0.35, H-12, '#4ecdc440', 8, '700');
  text(ctx, '\u2190 BACKWARD', W*0.65, H-12, '#c084fc40', 8, '700');

  document.getElementById('gateInfo').innerHTML = g.desc(upstream, z2, y2);
}

// ============================================================
// PANEL 3: GRADIENT FLOW
// ============================================================
var curFlow = 'sig';
var flowLayers = 10;

var FLOW_META = {
  sig:  { label:'Sigmoid', color:'#ff6b35', maxGrad:0.25,
    gradFn: function(l) { return Math.pow(0.25, l); },
    info: function(L, g) {
      return '<span class="hl5">Sigmoid vanishing gradient</span><br>' +
        'Max local gradient = 0.25 at z=0.<br>' +
        'After '+L+' layers: 0.25^'+L+' = <span class="hl5">'+g.toExponential(2)+'</span><br>' +
        (g < 1e-4 ? 'Gradient has <span class="hl5">effectively vanished</span>. Early layers receive no training signal.' :
         'Gradient is still detectable but weakening rapidly.');
    }
  },
  relu: { label:'ReLU (active)', color:'#4ade80', maxGrad:1.0,
    gradFn: function(l) { return Math.pow(1.0, l); },
    info: function(L, g) {
      return '<span class="hl4">ReLU: no vanishing gradient (active neurons)</span><br>' +
        'Local gradient = 1.0 for all active neurons (z > 0).<br>' +
        'After '+L+' layers: 1.0^'+L+' = <span class="hl4">1.0</span><br>' +
        'Gradient is perfectly preserved. This is why ReLU enabled training of very deep networks.';
    }
  },
  skip: { label:'Skip Connection (ResNet)', color:'#c084fc', maxGrad:1.0,
    gradFn: function(l) { return 1.0; },   // always 1 due to +1 in gradient
    info: function(L, g) {
      return '<span class="hl6">Skip connection: gradient highway</span><br>' +
        'd/dx(x + F(x)) = <span class="hl6">1</span> + dF/dx<br>' +
        'The +1 guarantees gradient \u2265 1 always \u2014 regardless of dF/dx.<br>' +
        'After '+L+' layers with skips: gradient \u2248 <span class="hl6">1.0</span> (preserved perfectly)<br>' +
        'This is why ResNet-152 can train while a plain 20-layer net struggles.';
    }
  }
};

function selectFlow(k) {
  curFlow = k;
  document.querySelectorAll('[id^="fBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('fBtn_' + k).classList.add('active');
  updateFlow();
}

function updateFlow() {
  flowLayers = parseInt(document.getElementById('slLayers').value);
  document.getElementById('valLayers').textContent = flowLayers;

  var fm = FLOW_META[curFlow];
  var cv = document.getElementById('cvFlow');
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#09090f'; ctx.fillRect(0, 0, W, H);

  var PAD = 38, bH = H - PAD - 30;
  var barW = Math.min(28, (W - PAD - 10) / flowLayers - 3);
  var spacing = (W - PAD - 10) / flowLayers;

  // baseline
  ctx.strokeStyle = '#2d2d40'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H-PAD); ctx.lineTo(W-10, H-PAD); ctx.stroke();

  // ideal line
  ctx.strokeStyle = '#4ade8025'; ctx.lineWidth = 1;
  ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.moveTo(PAD, H-PAD-bH); ctx.lineTo(W-10, H-PAD-bH); ctx.stroke();
  ctx.setLineDash([]);
  text(ctx, 'ideal (1.0)', PAD+2, H-PAD-bH-8, '#4ade8050', 8, '700', 'left');

  var finalGrad = fm.gradFn(flowLayers);

  for (var l2 = 1; l2 <= flowLayers; l2++) {
    var g = fm.gradFn(l2);
    var barH2 = Math.max(1, g * bH);
    var bx = PAD + (l2-1) * spacing + (spacing - barW) / 2;
    var by = H - PAD - barH2;

    // color by magnitude
    var intens = Math.min(1, g);
    var rr = intens < 0.1 ? 239 : intens < 0.5 ? 251 : 74;
    var gg = intens < 0.1 ? 68  : intens < 0.5 ? 191 : 222;
    var bb = intens < 0.1 ? 68  : intens < 0.5 ? 36  : 128;
    var barCol = 'rgb('+rr+','+gg+','+bb+')';

    ctx.fillStyle = barCol + '50';
    ctx.strokeStyle = barCol;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(bx, by, barW, barH2, 2);
    else ctx.rect(bx, by, barW, barH2);
    ctx.fill(); ctx.stroke();

    // value on top
    ctx.fillStyle = barCol;
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    var gStr = g < 0.001 ? g.toExponential(0) : g.toFixed(g < 0.01 ? 4 : g < 0.1 ? 3 : 2);
    if (barH2 > 12 || l2 <= 2) ctx.fillText(gStr, bx + barW/2, Math.max(20, by - 1));

    // layer label
    ctx.fillStyle = '#3f3f46';
    ctx.textBaseline = 'top';
    ctx.fillText('L'+l2, bx + barW/2, H-PAD+3);
  }

  // final gradient annotation
  var fCol = finalGrad < 0.01 ? '#ef4444' : finalGrad < 0.5 ? '#fbbf24' : '#4ade80';
  text(ctx, 'After '+flowLayers+' layers: '+finalGrad.toExponential(2), W/2, H-12, fCol, 9, '700');

  // Title
  text(ctx, fm.label, W/2, 16, fm.color, 10, '700');

  document.getElementById('flowInfo').innerHTML = fm.info(flowLayers, finalGrad);
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load', function() {
  buildDots();
  drawGraph();
  updateGate();
  updateFlow();
});
</script>

</body>
</html>"""

BACKPROP_VISUAL_HEIGHT = 2200