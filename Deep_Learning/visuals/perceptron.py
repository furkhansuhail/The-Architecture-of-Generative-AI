PERCEPTRON_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', sans-serif;
  background: #08080f;
  color: #e4e4e7;
  padding: 20px;
  min-height: 100vh;
}

h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.15em; color: #ff6b35; margin-bottom: 3px; }
.subtitle { color: #52525b; font-size: 0.8em; margin-bottom: 20px; }

.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.full { grid-column: 1 / -1; }

.card {
  background: #111118;
  border: 1px solid #1e1e2e;
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.card h3 {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78em;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #ff6b35;
  margin-bottom: 12px;
}

canvas { display: block; border-radius: 8px; }

.row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.row label { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #71717a; min-width: 80px; }
input[type=range] {
  flex: 1; height: 4px; border-radius: 2px;
  -webkit-appearance: none; appearance: none;
  background: #1e1e2e; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px;
  border-radius: 50%; background: #ff6b35; cursor: pointer;
}
.val { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #ff6b35; min-width: 46px; text-align: right; }

.info-box {
  background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px;
  padding: 8px 12px; font-size: 0.78em; color: #94a3b8;
  line-height: 1.7; margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
}
.hl  { color: #ff6b35; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }

.btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
button {
  background: #1e1e2e; color: #a1a1aa;
  border: 1px solid #2d2d40; border-radius: 6px;
  padding: 5px 12px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75em; font-weight: 700;
  transition: all 0.15s;
}
button:hover { background: #2d2d40; color: #e4e4e7; }
button.active { background: #ff6b35; color: #08080f; border-color: #ff6b35; }

/* ── Quiz ── */
.quiz-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 4px;
}
.q-card {
  background: #0d0d18;
  border: 1px solid #1e1e2e;
  border-radius: 10px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.q-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68em;
  color: #3f3f46;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.q-prompt {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78em;
  color: #cbd5e1;
  line-height: 1.55;
  font-weight: 700;
}
.q-opt {
  background: #111118;
  border: 1px solid #2d2d40;
  border-radius: 6px;
  padding: 6px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72em;
  color: #71717a;
  cursor: pointer;
  text-align: left;
  transition: all 0.15s;
  line-height: 1.4;
  width: 100%;
}
.q-opt:hover:not([disabled]) { background: #1e1e2e; color: #e4e4e7; border-color: #3f3f46; }
.q-opt.correct  { border-color: #4ade80; color: #4ade80; background: #0a1f0f !important; }
.q-opt.wrong    { border-color: #ef4444; color: #ef4444; background: #1a0808 !important; }
.q-feedback {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.71em;
  line-height: 1.55;
  padding: 6px 8px;
  border-radius: 6px;
  margin-top: 2px;
  display: none;
}
.q-feedback.show { display: block; }
.q-feedback.ok  { background: #0a1f0f; border: 1px solid #4ade8040; color: #4ade80; }
.q-feedback.bad { background: #1a0808; border: 1px solid #ef444440; color: #ef4444; }

.score-bar {
  display: flex; align-items: center; gap: 14px;
  margin-top: 14px; padding: 10px 16px;
  background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.8em;
}
.pip {
  width: 11px; height: 11px; border-radius: 50%;
  background: #1e1e2e; border: 1px solid #2d2d40;
  transition: all 0.3s; flex-shrink: 0;
}
.pip.ok  { background: #4ade80; border-color: #4ade80; box-shadow: 0 0 6px #4ade8066; }
.pip.bad { background: #ef4444; border-color: #ef4444; }
</style>
</head>
<body>

<h2>&#9889; Perceptron &#8212; The Atom of Deep Learning</h2>
<p class="subtitle">Forward pass &middot; Activation functions &middot; Decision boundaries &middot; Live training &middot; MLP XOR &middot; Quiz</p>

<div class="grid">

  <!-- ═══ PANEL 1: Forward Pass ═══ -->
  <div class="card">
    <h3>&#9312; Forward Pass &#8212; Live Neuron</h3>
    <canvas id="cvNeuron" width="420" height="230"></canvas>
    <div class="row"><label>x&#x2081; (input)</label><input type="range" id="x1" min="-2" max="2" step="0.05" value="1.0"><span class="val" id="x1v">1.00</span></div>
    <div class="row"><label>x&#x2082; (input)</label><input type="range" id="x2" min="-2" max="2" step="0.05" value="0.5"><span class="val" id="x2v">0.50</span></div>
    <div class="row"><label>w&#x2081; (weight)</label><input type="range" id="w1" min="-2" max="2" step="0.05" value="0.7"><span class="val" id="w1v">0.70</span></div>
    <div class="row"><label>w&#x2082; (weight)</label><input type="range" id="w2" min="-2" max="2" step="0.05" value="0.3"><span class="val" id="w2v">0.30</span></div>
    <div class="row"><label>bias b</label><input type="range" id="bias1" min="-2" max="2" step="0.05" value="-0.5"><span class="val" id="bias1v">-0.50</span></div>
    <div class="info-box" id="neuronInfo">&#8212;</div>
  </div>

  <!-- ═══ PANEL 2: Activation Functions ═══ -->
  <div class="card">
    <h3>&#9313; Activation Functions</h3>
    <div class="btn-row">
      <button id="btnStep"    class="active" onclick="setAct('step')">Step</button>
      <button id="btnSigmoid" onclick="setAct('sigmoid')">Sigmoid</button>
      <button id="btnTanh"   onclick="setAct('tanh')">Tanh</button>
      <button id="btnRelu"   onclick="setAct('relu')">ReLU</button>
      <button id="btnGelu"   onclick="setAct('gelu')">GELU</button>
    </div>
    <canvas id="cvAct" width="420" height="230"></canvas>
    <div class="row" style="margin-top:8px;"><label>z (input)</label><input type="range" id="zSlider" min="-4" max="4" step="0.05" value="0.5"><span class="val" id="zVal">0.50</span></div>
    <div class="info-box" id="actInfo">&#8212;</div>
  </div>

  <!-- ═══ PANEL 3: Decision Boundary ═══ -->
  <div class="card">
    <h3>&#9314; Decision Boundary &#8212; Geometry of a Perceptron</h3>
    <div class="btn-row">
      <button id="btnAND"  class="active" onclick="setGate('AND')">AND</button>
      <button id="btnOR"   onclick="setGate('OR')">OR</button>
      <button id="btnNAND" onclick="setGate('NAND')">NAND</button>
      <button id="btnXOR"  onclick="setGate('XOR')">XOR &#9888;</button>
    </div>
    <canvas id="cvBound" width="420" height="240"></canvas>
    <div class="row"><label>w&#x2081;</label><input type="range" id="dw1" min="-3" max="3" step="0.05" value="1.0"><span class="val" id="dw1v">1.00</span></div>
    <div class="row"><label>w&#x2082;</label><input type="range" id="dw2" min="-3" max="3" step="0.05" value="1.0"><span class="val" id="dw2v">1.00</span></div>
    <div class="row"><label>bias b</label><input type="range" id="db" min="-3" max="3" step="0.05" value="-1.5"><span class="val" id="dbv">-1.50</span></div>
    <div class="info-box" id="boundInfo">&#8212;</div>
  </div>

  <!-- ═══ PANEL 4: Live Training ═══ -->
  <div class="card">
    <h3>&#9315; Live Training &#8212; Perceptron Learning Rule</h3>
    <div class="btn-row">
      <button id="btnTrAND"  class="active" onclick="setTrainGate('AND')">AND</button>
      <button id="btnTrOR"   onclick="setTrainGate('OR')">OR</button>
      <button id="btnTrNAND" onclick="setTrainGate('NAND')">NAND</button>
      <button id="btnTrXOR"  onclick="setTrainGate('XOR')">XOR &#9888;</button>
    </div>
    <canvas id="cvTrain" width="420" height="220"></canvas>
    <div class="btn-row" style="margin-top:8px;">
      <button id="btnPlay" onclick="toggleTrain()">&#9654; Train</button>
      <button onclick="resetTrain()">&#8635; Reset</button>
      <button onclick="stepTrain()">Step &#8250;</button>
    </div>
    <div class="row"><label>Learning rate</label><input type="range" id="lr" min="0.01" max="1.0" step="0.01" value="0.1"><span class="val" id="lrv">0.10</span></div>
    <div class="info-box" id="trainInfo">&#8212;</div>
  </div>

  <!-- ═══ PANEL 5: MLP XOR (full-width React) ═══ -->
  <div class="card full" style="padding:0;overflow:hidden;background:#050508;">
    <div style="padding:18px 18px 0 18px;border-bottom:1px solid #1e1e2e;">
      <h3 style="margin-bottom:12px;">&#9316; Why XOR Needs an MLP &#8212; Step-by-Step Walkthrough</h3>
    </div>
    <div id="mlp-root"></div>
  </div>

  <!-- ═══ PANEL 6: Quiz (full-width vanilla JS) ═══ -->
  <div class="card full">
    <h3>&#9317; Knowledge Check &#8212; Perceptron Quiz</h3>
    <div id="quiz-root"></div>
  </div>

</div><!-- /grid -->

<!-- ════════════════════════════════════════════════════
     VANILLA JS — Panels 1-4 + zoom
     ════════════════════════════════════════════════════ -->
<script>
// ── Seeded RNG ────────────────────────────────────────────────────────────────
let sd = 42;
const rng = () => { sd = (sd * 1664525 + 1013904223) >>> 0; return sd / 4294967296; };

// ── Activation functions ──────────────────────────────────────────────────────
const ACT = {
  step:    z => z >= 0 ? 1 : 0,
  sigmoid: z => 1 / (1 + Math.exp(-z)),
  tanh:    z => Math.tanh(z),
  relu:    z => Math.max(0, z),
  gelu:    z => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3))),
};
const ACT_META = {
  step:    { color:'#ff6b35', range:[-5,5],  yrange:[-0.2,1.3],  formula:'f(z) = 1 if z\u22650 else 0',               desc:'Hard 0\u21921 jump at z=0. Not differentiable \u2014 cannot use backprop. Original Rosenblatt (1958).' },
  sigmoid: { color:'#4ecdc4', range:[-6,6],  yrange:[-0.1,1.1],  formula:'f(z) = 1/(1+e\u207b\u1d63)',                desc:'Smooth 0\u21921 curve. Differentiable \u2014 enables backprop. Suffers vanishing gradients for large |z|.' },
  tanh:    { color:'#a78bfa', range:[-5,5],  yrange:[-1.3,1.3],  formula:'f(z) = tanh(z)',                            desc:'Zero-centred \u22121\u21921. Better gradient flow than sigmoid. tanh(z) = 2\u03c3(2z)\u22121.' },
  relu:    { color:'#fbbf24', range:[-5,5],  yrange:[-0.5,5.5],  formula:'f(z) = max(0, z)',                          desc:'Fast and sparse. No vanishing gradient for z>0. \u201cDying ReLU\u201d problem for z<0.' },
  gelu:    { color:'#f472b6', range:[-4,4],  yrange:[-0.5,4.5],  formula:'f(z) \u2248 0.5z(1+tanh(\u221a(2/\u03c0)(z+0.044z\u00b3)))', desc:'Gaussian-gated. Used in GPT, BERT, ViTs. Smooth approximation of ReLU.' },
};
let curAct = 'step';

function gs(id) { return parseFloat(document.getElementById(id).value); }

// ── Panel 1 ───────────────────────────────────────────────────────────────────
const cvN = document.getElementById('cvNeuron'), ctxN = cvN.getContext('2d');

function drawNeuron() {
  const W = 420, H = 230;
  ctxN.clearRect(0, 0, W, H);
  ctxN.fillStyle = '#0a0a12'; ctxN.fillRect(0, 0, W, H);

  const x1 = gs('x1'), x2 = gs('x2'), w1 = gs('w1'), w2 = gs('w2'), b = gs('bias1');
  document.getElementById('x1v').textContent    = x1.toFixed(2);
  document.getElementById('x2v').textContent    = x2.toFixed(2);
  document.getElementById('w1v').textContent    = w1.toFixed(2);
  document.getElementById('w2v').textContent    = w2.toFixed(2);
  document.getElementById('bias1v').textContent = b.toFixed(2);

  const z   = x1 * w1 + x2 * w2 + b;
  const out = ACT[curAct](z);
  const fires = out > 0.5;
  const outCol = fires ? '#4ade80' : '#ef4444';

  const INX = 70, INY1 = 80, INY2 = 160, SX = 210, SY = 120, AX = 310, AY = 120, OX = 390, OY = 120;
  const wCol = w => w > 0
    ? `rgba(78,205,196,${Math.min(Math.abs(w) / 2, 1)})`
    : `rgba(244,114,182,${Math.min(Math.abs(w) / 2, 1)})`;

  // Input→sigma connections
  [[INX, INY1, SX, SY, w1], [INX, INY2, SX, SY, w2]].forEach(([ax, ay, bx, by, w]) => {
    ctxN.beginPath(); ctxN.moveTo(ax + 16, ay); ctxN.lineTo(bx - 20, by);
    ctxN.strokeStyle = wCol(w); ctxN.lineWidth = 2 + Math.abs(w) * 1.5;
    ctxN.globalAlpha = 0.8; ctxN.stroke(); ctxN.globalAlpha = 1;
  });

  // Bias dashed
  ctxN.beginPath(); ctxN.moveTo(SX, SY + 20); ctxN.lineTo(SX, SY + 46);
  ctxN.setLineDash([3, 3]); ctxN.strokeStyle = '#71717a'; ctxN.lineWidth = 1.5; ctxN.stroke();
  ctxN.setLineDash([]);

  // sigma→act
  ctxN.beginPath(); ctxN.moveTo(SX + 20, SY); ctxN.lineTo(AX - 18, AY);
  ctxN.strokeStyle = '#3f3f46'; ctxN.lineWidth = 2; ctxN.stroke();

  // act→output
  ctxN.beginPath(); ctxN.moveTo(AX + 22, AY); ctxN.lineTo(OX - 16, OY);
  ctxN.strokeStyle = outCol; ctxN.lineWidth = 2.5; ctxN.stroke();

  // Input nodes
  [[INX, INY1, 'x\u2081', x1], [INX, INY2, 'x\u2082', x2]].forEach(([cx, cy, lbl, val]) => {
    ctxN.beginPath(); ctxN.arc(cx, cy, 16, 0, 2 * Math.PI);
    ctxN.fillStyle = '#12121f'; ctxN.fill();
    ctxN.strokeStyle = '#2d2d40'; ctxN.lineWidth = 2; ctxN.stroke();
    ctxN.fillStyle = '#94a3b8'; ctxN.font = 'bold 11px JetBrains Mono,monospace'; ctxN.textAlign = 'center';
    ctxN.fillText(lbl, cx, cy - 3);
    ctxN.fillStyle = '#ff6b35'; ctxN.font = 'bold 10px JetBrains Mono,monospace';
    ctxN.fillText(val.toFixed(2), cx, cy + 10);
  });

  // Weight labels
  ctxN.font = 'bold 10px JetBrains Mono,monospace'; ctxN.textAlign = 'center';
  ctxN.fillStyle = '#4ecdc4'; ctxN.fillText(`w\u2081=${w1.toFixed(2)}`, 120, INY1 - 10);
  ctxN.fillStyle = '#4ecdc4'; ctxN.fillText(`w\u2082=${w2.toFixed(2)}`, 120, INY2 + 18);
  ctxN.fillStyle = '#71717a'; ctxN.font = '10px JetBrains Mono,monospace';
  ctxN.fillText(`b=${b.toFixed(2)}`, SX, SY + 60);

  // Sigma node
  ctxN.beginPath(); ctxN.arc(SX, SY, 20, 0, 2 * Math.PI);
  ctxN.fillStyle = '#0f1420'; ctxN.fill();
  ctxN.strokeStyle = '#4ecdc4'; ctxN.lineWidth = 2; ctxN.stroke();
  ctxN.fillStyle = '#4ecdc4'; ctxN.font = 'bold 14px JetBrains Mono,monospace'; ctxN.textAlign = 'center';
  ctxN.fillText('\u03a3', SX, SY + 5);
  ctxN.fillStyle = '#71717a'; ctxN.font = '10px JetBrains Mono,monospace';
  ctxN.fillText(`z=${z.toFixed(3)}`, SX, SY + 36);

  // Act node
  const am = ACT_META[curAct];
  ctxN.beginPath(); ctxN.arc(AX, AY, 18, 0, 2 * Math.PI);
  ctxN.fillStyle = '#0f1420'; ctxN.fill();
  ctxN.strokeStyle = am.color; ctxN.lineWidth = 2; ctxN.stroke();
  ctxN.fillStyle = am.color; ctxN.font = 'bold 10px JetBrains Mono,monospace'; ctxN.textAlign = 'center';
  ctxN.fillText('act', AX, AY + 4);

  // Output node
  ctxN.beginPath(); ctxN.arc(OX, OY, 16, 0, 2 * Math.PI);
  ctxN.fillStyle = fires ? '#0f2018' : '#200f0f'; ctxN.fill();
  ctxN.strokeStyle = outCol; ctxN.lineWidth = 2.5; ctxN.stroke();
  ctxN.fillStyle = outCol; ctxN.font = 'bold 13px JetBrains Mono,monospace'; ctxN.textAlign = 'center';
  ctxN.fillText(out.toFixed(2), OX, OY + 5);
  ctxN.font = 'bold 10px JetBrains Mono,monospace';
  ctxN.fillText(fires ? '\uD83D\uDD25 FIRE' : '\u2744 SILENT', OX, OY + 30);

  // Top formula
  ctxN.fillStyle = '#3f3f46'; ctxN.font = '11px JetBrains Mono,monospace'; ctxN.textAlign = 'left';
  ctxN.fillText(`z = (${x1.toFixed(2)}\u00d7${w1.toFixed(2)}) + (${x2.toFixed(2)}\u00d7${w2.toFixed(2)}) + (${b.toFixed(2)}) = ${z.toFixed(4)}`, 12, 20);

  const dot = x1 * w1 + x2 * w2;
  document.getElementById('neuronInfo').innerHTML =
    `<span class="hl">z</span> = ${z.toFixed(4)} &nbsp;|&nbsp; ` +
    `<span class="hl2">act(z)</span> = ${out.toFixed(4)} &nbsp;|&nbsp; ` +
    `Neuron <span style="color:${outCol};font-weight:bold">${fires ? 'FIRES \uD83D\uDD25' : 'stays SILENT \u2744'}</span><br>` +
    `W\u00b7X = ${dot.toFixed(4)} \u2014 ` +
    (Math.abs(dot) < 0.3 ? 'inputs orthogonal to learned pattern' : dot > 0 ? 'input aligns with pattern' : 'input opposes pattern');
}
['x1','x2','w1','w2','bias1'].forEach(id => document.getElementById(id).addEventListener('input', drawNeuron));
drawNeuron();

// ── Panel 2 ───────────────────────────────────────────────────────────────────
const cvA = document.getElementById('cvAct'), ctxA = cvA.getContext('2d');

function setAct(name) {
  curAct = name;
  ['Step','Sigmoid','Tanh','Relu','Gelu'].forEach(n =>
    document.getElementById('btn' + n).classList.toggle('active', n.toLowerCase() === name));
  drawAct(); drawNeuron();
}

function drawAct() {
  const W = 420, H = 230, PAD = 36;
  ctxA.clearRect(0, 0, W, H); ctxA.fillStyle = '#0a0a12'; ctxA.fillRect(0, 0, W, H);
  const m = ACT_META[curAct];
  const [xmin, xmax] = m.range, [ymin, ymax] = m.yrange;
  const sxa = x => PAD + (x - xmin) / (xmax - xmin) * (W - 2 * PAD);
  const sya = y => H - PAD - (y - ymin) / (ymax - ymin) * (H - 2 * PAD);

  // Grid
  ctxA.strokeStyle = '#1e1e2e'; ctxA.lineWidth = 0.5;
  for (let gx = Math.ceil(xmin); gx <= xmax; gx++) {
    ctxA.beginPath(); ctxA.moveTo(sxa(gx), PAD); ctxA.lineTo(sxa(gx), H - PAD); ctxA.stroke();
  }
  [0, 0.25, 0.5, 0.75, 1].forEach(gy => {
    ctxA.beginPath(); ctxA.moveTo(PAD, sya(gy)); ctxA.lineTo(W - PAD, sya(gy)); ctxA.stroke();
  });

  // Axes
  ctxA.strokeStyle = '#2d2d40'; ctxA.lineWidth = 1.5;
  const yz = sya(0); if (yz > PAD && yz < H - PAD) { ctxA.beginPath(); ctxA.moveTo(PAD, yz); ctxA.lineTo(W - PAD, yz); ctxA.stroke(); }
  const xz = sxa(0); if (xz > PAD && xz < W - PAD) { ctxA.beginPath(); ctxA.moveTo(xz, PAD); ctxA.lineTo(xz, H - PAD); ctxA.stroke(); }

  // Labels
  ctxA.fillStyle = '#3f3f46'; ctxA.font = '9px JetBrains Mono,monospace'; ctxA.textAlign = 'center';
  for (let gx = Math.ceil(xmin); gx <= xmax; gx += 2) ctxA.fillText(gx, sxa(gx), H - PAD + 14);
  ctxA.textAlign = 'right';
  (curAct === 'tanh' ? [-1,-0.5,0,0.5,1] : [0,0.25,0.5,0.75,1]).forEach(v =>
    ctxA.fillText(v.toFixed(curAct === 'tanh' ? 1 : 2), PAD - 3, sya(v) + 4));

  // Curve
  ctxA.beginPath();
  for (let i = 0; i <= 300; i++) {
    const x = xmin + i / 300 * (xmax - xmin);
    i === 0 ? ctxA.moveTo(sxa(x), sya(ACT[curAct](x))) : ctxA.lineTo(sxa(x), sya(ACT[curAct](x)));
  }
  ctxA.strokeStyle = m.color; ctxA.lineWidth = 2.5; ctxA.stroke();

  // Marker
  const z = gs('zSlider'), y = ACT[curAct](z);
  const px = sxa(z), py = sya(y);
  document.getElementById('zVal').textContent = z.toFixed(2);
  ctxA.setLineDash([3, 3]); ctxA.strokeStyle = '#52525b'; ctxA.lineWidth = 1;
  ctxA.beginPath(); ctxA.moveTo(px, py); ctxA.lineTo(px, sya(0)); ctxA.stroke();
  ctxA.beginPath(); ctxA.moveTo(px, py); ctxA.lineTo(PAD, py); ctxA.stroke();
  ctxA.setLineDash([]);
  ctxA.beginPath(); ctxA.arc(px, py, 6, 0, 2 * Math.PI);
  ctxA.fillStyle = m.color; ctxA.fill(); ctxA.strokeStyle = '#fff'; ctxA.lineWidth = 1.5; ctxA.stroke();
  ctxA.fillStyle = m.color; ctxA.font = 'bold 10px JetBrains Mono,monospace';
  ctxA.textAlign = px + 8 > W - PAD - 100 ? 'right' : 'left';
  ctxA.fillText(`f(${z.toFixed(2)})=${y.toFixed(4)}`, px + (px + 8 > W - PAD - 100 ? -8 : 8), py - 6);

  document.getElementById('actInfo').innerHTML =
    `<span class="hl" style="color:${m.color}">${curAct.toUpperCase()}</span>: ` +
    `<span style="color:${m.color}">${m.formula}</span><br>${m.desc}<br>` +
    `f(<span class="hl">z=${z.toFixed(2)}</span>) = <span class="hl">${y.toFixed(5)}</span>`;
}
document.getElementById('zSlider').addEventListener('input', drawAct);
drawAct();

// ── Panel 3 ───────────────────────────────────────────────────────────────────
const cvB = document.getElementById('cvBound'), ctxB = cvB.getContext('2d');
const GATES = {
  AND:  { pts:[[0,0,0],[0,1,0],[1,0,0],[1,1,1]], initW:[1,1],   initB:-1.5, canLearn:true  },
  OR:   { pts:[[0,0,0],[0,1,1],[1,0,1],[1,1,1]], initW:[1,1],   initB:-0.5, canLearn:true  },
  NAND: { pts:[[0,0,1],[0,1,1],[1,0,1],[1,1,0]], initW:[-1,-1], initB:1.5,  canLearn:true  },
  XOR:  { pts:[[0,0,0],[0,1,1],[1,0,1],[1,1,0]], initW:[1,1],   initB:-1.0, canLearn:false },
};
let curGate = 'AND';

function setGate(name) {
  curGate = name;
  ['AND','OR','NAND','XOR'].forEach(g => document.getElementById('btn' + g).classList.toggle('active', g === name));
  const g = GATES[name];
  document.getElementById('dw1').value = g.initW[0];
  document.getElementById('dw2').value = g.initW[1];
  document.getElementById('db').value  = g.initB;
  drawBoundary();
}

function drawBoundary() {
  const W = 420, H = 240, PAD = 44;
  ctxB.clearRect(0, 0, W, H); ctxB.fillStyle = '#0a0a12'; ctxB.fillRect(0, 0, W, H);
  const dw1 = gs('dw1'), dw2 = gs('dw2'), db = gs('db');
  document.getElementById('dw1v').textContent = dw1.toFixed(2);
  document.getElementById('dw2v').textContent = dw2.toFixed(2);
  document.getElementById('dbv').textContent  = db.toFixed(2);
  const IW = W - 2 * PAD, IH = H - 2 * PAD;
  const sxb = x => PAD + x * IW, syb = y => H - PAD - y * IH;

  // Region fill
  const IMG = ctxB.createImageData(W, H);
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const wx = (px - PAD) / IW, wy = (H - PAD - py) / IH, pos = dw1 * wx + dw2 * wy + db >= 0;
    const idx = (py * W + px) * 4;
    IMG.data[idx] = pos?20:30; IMG.data[idx+1] = pos?40:10; IMG.data[idx+2] = pos?30:10; IMG.data[idx+3] = 130;
  }
  ctxB.putImageData(IMG, 0, 0);

  // Grid
  ctxB.strokeStyle = '#1e1e2e'; ctxB.lineWidth = 0.5;
  [0, 0.5, 1].forEach(v => {
    ctxB.beginPath(); ctxB.moveTo(sxb(v), PAD); ctxB.lineTo(sxb(v), H - PAD); ctxB.stroke();
    ctxB.beginPath(); ctxB.moveTo(PAD, syb(v)); ctxB.lineTo(W - PAD, syb(v)); ctxB.stroke();
  });

  // Decision line
  if (Math.abs(dw2) > 0.01) {
    ctxB.beginPath();
    ctxB.moveTo(sxb(-0.4), syb((-dw1 * (-0.4) - db) / dw2));
    ctxB.lineTo(sxb( 1.4), syb((-dw1 *   1.4  - db) / dw2));
    ctxB.strokeStyle = '#ff6b35'; ctxB.lineWidth = 2.5; ctxB.stroke();
  } else if (Math.abs(dw1) > 0.01) {
    const xv = -db / dw1;
    ctxB.beginPath(); ctxB.moveTo(sxb(xv), PAD); ctxB.lineTo(sxb(xv), H - PAD);
    ctxB.strokeStyle = '#ff6b35'; ctxB.lineWidth = 2.5; ctxB.stroke();
  }

  // Points
  let correct = 0;
  GATES[curGate].pts.forEach(([xi1, xi2, label]) => {
    const pred = dw1*xi1 + dw2*xi2 + db >= 0 ? 1 : 0, ok = pred === label;
    if (ok) correct++;
    const col = label === 1 ? '#4ade80' : '#ef4444';
    const cx = sxb(xi1), cy = syb(xi2);
    ctxB.beginPath(); ctxB.arc(cx, cy, 10, 0, 2*Math.PI);
    ctxB.fillStyle = label===1 ? '#0f2018' : '#200f0f'; ctxB.fill();
    ctxB.strokeStyle = ok ? col : '#fbbf24'; ctxB.lineWidth = ok ? 2.5 : 3; ctxB.stroke();
    ctxB.fillStyle = col; ctxB.font = 'bold 11px JetBrains Mono,monospace'; ctxB.textAlign = 'center';
    ctxB.fillText(label, cx, cy + 4);
    if (!ok) { ctxB.fillStyle = '#fbbf24'; ctxB.font = 'bold 10px JetBrains Mono,monospace'; ctxB.fillText('\u2717', cx+13, cy-8); }
  });

  ctxB.fillStyle = '#3f3f46'; ctxB.font = '9px JetBrains Mono,monospace';
  ctxB.textAlign = 'center'; ctxB.fillText('x\u2081=0', sxb(0), H-8); ctxB.fillText('x\u2081=1', sxb(1), H-8);
  ctxB.textAlign = 'right';  ctxB.fillText('x\u2082=0', PAD-4, syb(0)+4); ctxB.fillText('x\u2082=1', PAD-4, syb(1)+4);

  const can = GATES[curGate].canLearn;
  document.getElementById('boundInfo').innerHTML =
    `<span class="${can?'hl2':'hl'}">${curGate}</span> &nbsp;|&nbsp; ` +
    `Correct: <span class="${correct===4?'hl2':'hl'}">${correct}/4</span> &nbsp;|&nbsp; ` +
    `<span class="hl3">${dw1.toFixed(2)}x\u2081 + ${dw2.toFixed(2)}x\u2082 + (${db.toFixed(2)}) = 0</span><br>` +
    (can ? '\u2713 Linearly separable \u2014 a perceptron CAN learn this.'
         : '\u26a0 NOT linearly separable \u2014 no single line can separate XOR. Need an MLP.');
}
['dw1','dw2','db'].forEach(id => document.getElementById(id).addEventListener('input', drawBoundary));
drawBoundary();

// ── Panel 4 ───────────────────────────────────────────────────────────────────
const cvT = document.getElementById('cvTrain'), ctxT = cvT.getContext('2d');
const GATE_DATA = {
  AND:  { X:[[0,0],[0,1],[1,0],[1,1]], Y:[0,0,0,1] },
  OR:   { X:[[0,0],[0,1],[1,0],[1,1]], Y:[0,1,1,1] },
  NAND: { X:[[0,0],[0,1],[1,0],[1,1]], Y:[1,1,1,0] },
  XOR:  { X:[[0,0],[0,1],[1,0],[1,1]], Y:[0,1,1,0] },
};
let trainGate='AND', tw=[0,0], tb=0, tEpoch=0, errorHist=[], trainTimer=null;

function resetTrain() {
  sd=123; tw=[rng()*0.4-0.2, rng()*0.4-0.2]; tb=0; tEpoch=0; errorHist=[];
  if (trainTimer) { clearInterval(trainTimer); trainTimer=null; }
  document.getElementById('btnPlay').textContent = '\u25b6 Train';
  drawTrain();
}
function setTrainGate(name) {
  trainGate = name;
  ['AND','OR','NAND','XOR'].forEach(g => document.getElementById('btnTr'+g).classList.toggle('active', g===name));
  resetTrain();
}
function stepTrain() {
  const lr = gs('lr'); document.getElementById('lrv').textContent = lr.toFixed(2);
  const {X,Y} = GATE_DATA[trainGate]; let errors = 0;
  X.forEach((x,i) => {
    const z = tw[0]*x[0] + tw[1]*x[1] + tb, pred = z>=0?1:0, err = Y[i]-pred;
    if (err !== 0) { errors++; tw[0]+=lr*err*x[0]; tw[1]+=lr*err*x[1]; tb+=lr*err; }
  });
  tEpoch++; errorHist.push(errors);
  if (errorHist.length > 80) errorHist.shift();
  drawTrain();
}
function toggleTrain() {
  if (trainTimer) { clearInterval(trainTimer); trainTimer=null; document.getElementById('btnPlay').textContent='\u25b6 Train'; }
  else { trainTimer=setInterval(stepTrain,120); document.getElementById('btnPlay').textContent='\u23f8 Pause'; }
}
function drawTrain() {
  const W=420,H=220,PAD=36;
  ctxT.clearRect(0,0,W,H); ctxT.fillStyle='#0a0a12'; ctxT.fillRect(0,0,W,H);
  const {X,Y} = GATE_DATA[trainGate];
  const MW=160,MH=H-2*PAD,MX=PAD,MY=PAD;
  const sxt=x=>MX+x*MW, syt=y=>MY+MH-y*MH;

  for (let py=0;py<MH;py++) for (let px=0;px<MW;px++) {
    const z = tw[0]*(px/MW) + tw[1]*((MH-py)/MH) + tb;
    ctxT.fillStyle = z>=0 ? 'rgba(20,60,40,0.7)':'rgba(50,15,15,0.7)';
    ctxT.fillRect(MX+px,MY+py,1,1);
  }
  ctxT.strokeStyle='#1e1e2e'; ctxT.lineWidth=0.5;
  [0,0.5,1].forEach(v=>{
    ctxT.beginPath(); ctxT.moveTo(sxt(v),MY); ctxT.lineTo(sxt(v),MY+MH); ctxT.stroke();
    ctxT.beginPath(); ctxT.moveTo(MX,syt(v)); ctxT.lineTo(MX+MW,syt(v)); ctxT.stroke();
  });
  if (Math.abs(tw[1])>0.01) {
    ctxT.beginPath();
    ctxT.moveTo(sxt(-0.3), syt((-tw[0]*(-0.3)-tb)/tw[1]));
    ctxT.lineTo(sxt( 1.3), syt((-tw[0]*  1.3 -tb)/tw[1]));
    ctxT.strokeStyle='#ff6b35'; ctxT.lineWidth=2; ctxT.stroke();
  }
  X.forEach((x,i)=>{
    const pred = tw[0]*x[0]+tw[1]*x[1]+tb>=0?1:0, ok=pred===Y[i];
    ctxT.beginPath(); ctxT.arc(sxt(x[0]),syt(x[1]),8,0,2*Math.PI);
    ctxT.fillStyle = Y[i]===1?'#0f2018':'#200f0f'; ctxT.fill();
    ctxT.strokeStyle = ok?(Y[i]===1?'#4ade80':'#ef4444'):'#fbbf24';
    ctxT.lineWidth = ok?2:2.5; ctxT.stroke();
    ctxT.fillStyle = Y[i]===1?'#4ade80':'#ef4444';
    ctxT.font='bold 9px JetBrains Mono,monospace'; ctxT.textAlign='center';
    ctxT.fillText(Y[i], sxt(x[0]), syt(x[1])+4);
  });

  const CX=220,CW=W-CX-14,CH=H-2*PAD;
  ctxT.fillStyle='#0d0d16'; ctxT.fillRect(CX,PAD,CW,CH);
  ctxT.strokeStyle='#1e1e2e'; ctxT.lineWidth=0.8; ctxT.strokeRect(CX,PAD,CW,CH);
  [0,1,2,3,4].forEach(v=>{
    const gy=PAD+CH-v/4*CH;
    ctxT.beginPath(); ctxT.strokeStyle='#1e1e2e'; ctxT.lineWidth=0.5;
    ctxT.moveTo(CX,gy); ctxT.lineTo(CX+CW,gy); ctxT.stroke();
    ctxT.fillStyle='#3f3f46'; ctxT.font='8px JetBrains Mono,monospace'; ctxT.textAlign='right';
    ctxT.fillText(v, CX-3, gy+3);
  });
  if (errorHist.length>1) {
    const n=Math.min(errorHist.length,70), sl=errorHist.slice(-n);
    ctxT.beginPath();
    sl.forEach((e,i)=>{
      const px=CX+1+(i/(n-1))*(CW-2), py=PAD+CH-1-e/4*CH;
      i===0?ctxT.moveTo(px,py):ctxT.lineTo(px,py);
    });
    ctxT.strokeStyle='#ff6b35'; ctxT.lineWidth=2; ctxT.stroke();
    ctxT.lineTo(CX+CW-1,PAD+CH-1); ctxT.lineTo(CX+1,PAD+CH-1); ctxT.closePath();
    ctxT.fillStyle='rgba(255,107,53,0.08)'; ctxT.fill();
  }
  ctxT.fillStyle='#3f3f46'; ctxT.font='9px JetBrains Mono,monospace';
  ctxT.textAlign='center'; ctxT.fillText('Epoch \u2192',CX+CW/2,H-6);
  ctxT.save(); ctxT.translate(CX-22,PAD+CH/2); ctxT.rotate(-Math.PI/2);
  ctxT.fillText('Errors',0,0); ctxT.restore();
  ctxT.fillStyle='#52525b'; ctxT.textAlign='left'; ctxT.fillText('Error History',CX+5,PAD+12);

  const lastErr = errorHist.length>0 ? errorHist[errorHist.length-1] : 4;
  const converged = lastErr===0&&tEpoch>0, canLearn = trainGate!=='XOR';
  document.getElementById('lrv').textContent = gs('lr').toFixed(2);
  document.getElementById('trainInfo').innerHTML =
    `Epoch <span class="hl">${tEpoch}</span> &nbsp;|&nbsp; ` +
    `Errors: <span class="${lastErr===0?'hl2':'hl'}">${lastErr}/4</span> &nbsp;|&nbsp; ` +
    `w\u2081=<span class="hl2">${tw[0].toFixed(3)}</span> ` +
    `w\u2082=<span class="hl2">${tw[1].toFixed(3)}</span> ` +
    `b=<span class="hl2">${tb.toFixed(3)}</span><br>` +
    (converged&&canLearn
      ? `<span class="hl2">\u2713 Converged! Perceptron learned ${trainGate} perfectly.</span>`
      : !canLearn&&tEpoch>12
        ? `<span class="hl">\u26a0 XOR is not linearly separable \u2014 oscillates forever. Need MLP!</span>`
        : `Perceptron learning rule: w\u1d62 \u2190 w\u1d62 + lr \u00d7 error \u00d7 x\u1d62`);
}
document.getElementById('lr').addEventListener('input', () => { document.getElementById('lrv').textContent = gs('lr').toFixed(2); });
resetTrain();

// ── Ctrl+scroll zoom ──────────────────────────────────────────────────────────
var ZOOM=1.0, zoomTimer=null;
function applyZoom() {
  document.body.style.zoom = ZOOM; clearTimeout(zoomTimer);
  var t = document.getElementById('zoom-toast');
  if (!t) {
    t = document.createElement('div'); t.id='zoom-toast';
    t.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#111118;' +
      'border:1px solid #ff6b35;color:#ff6b35;font-family:JetBrains Mono,monospace;' +
      'font-size:12px;font-weight:700;padding:8px 14px;border-radius:8px;z-index:9999;' +
      'pointer-events:none;transition:opacity .3s;';
    document.body.appendChild(t);
  }
  t.textContent = 'zoom ' + Math.round(ZOOM*100) + '%'; t.style.opacity='1';
  zoomTimer = setTimeout(() => t.style.opacity='0', 1200);
}
document.addEventListener('wheel', e => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  ZOOM = Math.min(3.0, Math.max(0.4, Math.round((ZOOM + (e.deltaY>0?-0.05:0.05))*100)/100));
  applyZoom();
}, { passive: false });

// ── Panel 6: Standalone Quiz ──────────────────────────────────────────────────
const QUESTIONS = [
  {
    q: "What does the perceptron compute in Step 1 before activation?",
    opts: [
      "The product of all inputs multiplied together",
      "z = \u03a3(w\u1d62 \u00b7 x\u1d62) + b \u2014 a weighted sum plus bias",
      "The maximum input value across all features",
      "The Euclidean distance between inputs and weights"
    ],
    correct: 1,
    ok:  "\u2713 Correct! z = w\u2081x\u2081 + w\u2082x\u2082 + \u2026 + b is the dot product W\u00b7X plus bias \u2014 the core linear operation of every neuron.",
    bad: "The weighted sum z = \u03a3(w\u1d62x\u1d62) + b is the dot product W\u00b7X plus bias. The activation function comes after."
  },
  {
    q: "Why can a single perceptron NOT learn XOR?",
    opts: [
      "XOR requires too many weights to store",
      "The step activation is too slow to converge",
      "XOR is NOT linearly separable \u2014 no single straight line can divide its four points correctly",
      "XOR needs negative inputs, which a perceptron can\u2019t process"
    ],
    correct: 2,
    ok:  "\u2713 Correct! XOR\u2019s 1-outputs are at opposite corners. You cannot draw one line separating them from the 0s. Two lines (a hidden layer) are needed.",
    bad: "The real reason is geometric: XOR=1 at (0,1) and (1,0) are diagonally opposite \u2014 no single hyperplane separates them."
  },
  {
    q: "What is the role of the bias term b?",
    opts: [
      "It scales all weights uniformly to prevent overflow",
      "It shifts the decision boundary away from the origin, controlling how easily the neuron fires",
      "It adds noise to prevent overfitting during training",
      "It stores the class label for the training example"
    ],
    correct: 1,
    ok:  "\u2713 Correct! Without bias, the decision boundary must pass through the origin. b = \u2212\u03b8 (threshold), giving the neuron tuneable firing sensitivity.",
    bad: "Bias b shifts the boundary: z = W\u00b7X + b \u2265 0 is equivalent to W\u00b7X \u2265 \u2212b. It lets the neuron fire more or less easily."
  },
  {
    q: "The perceptron learning rule updates weights ONLY when\u2026",
    opts: [
      "Every epoch, regardless of prediction correctness",
      "The output is greater than 0.5",
      "The prediction is wrong (error = true_label \u2212 prediction \u2260 0)",
      "The learning rate is above 0.5"
    ],
    correct: 2,
    ok:  "\u2713 Correct! error = 0 \u2192 no update. error = \u00b11 \u2192 w\u1d62 += lr \u00d7 error \u00d7 x\u1d62. Only mistakes drive learning.",
    bad: "Weights only change on errors: w\u1d62 \u2190 w\u1d62 + lr \u00d7 error \u00d7 x\u1d62. If prediction is correct, error = 0 and nothing changes."
  },
  {
    q: "Why did deep learning switch from step activation to ReLU/Sigmoid/GELU?",
    opts: [
      "Step functions compute slower on modern GPUs",
      "The step function is not differentiable \u2014 its gradient is zero everywhere, blocking backpropagation",
      "Step functions produce outputs outside the [0,1] range",
      "Step functions cannot be computed in parallel across a batch"
    ],
    correct: 1,
    ok:  "\u2713 Correct! Backprop requires the chain rule, which needs differentiable functions. The step gradient is 0 almost everywhere \u2014 no gradient can flow backward.",
    bad: "The step gradient is 0 everywhere except z=0, so \u2202L/\u2202w = 0 for hidden layers. Backprop cannot update weights \u2014 you need smooth, differentiable activations."
  },
  {
    q: "In the MLP that solves XOR, what does the hidden layer do geometrically?",
    opts: [
      "It memorises each training example individually",
      "Each hidden neuron draws one linear boundary; combined, they carve out a non-linear region the output neuron can linearly separate",
      "It projects inputs into 3D space where linear separation is always possible",
      "It averages features to reduce dimensionality"
    ],
    correct: 1,
    ok:  "\u2713 Correct! Neuron 1 (OR) eliminates (0,0); Neuron 2 (NAND) eliminates (1,1). The output (AND) fires only when both pass \u2014 exactly the XOR=1 region.",
    bad: "Each hidden neuron is still linear (one line). But combining two lines lets the output neuron linearly separate a region that was non-linear in input space."
  }
];

(function buildQuiz() {
  const root  = document.getElementById('quiz-root');
  const state = QUESTIONS.map(() => ({ chosen: null })); // chosen: null | index

  function render() {
    const answered = state.filter(s => s.chosen !== null).length;
    const correct  = state.filter((s, i) => s.chosen === QUESTIONS[i].correct).length;

    const cards = QUESTIONS.map((q, qi) => {
      const s = state[qi];
      const answered = s.chosen !== null;
      const opts = q.opts.map((opt, oi) => {
        let cls = 'q-opt';
        if (answered) {
          if (oi === q.correct)   cls += ' correct';
          else if (oi === s.chosen) cls += ' wrong';
        }
        return `<button class="${cls}" onclick="quizAnswer(${qi},${oi})" ${answered?'disabled':''}>${String.fromCharCode(65+oi)}. ${opt}</button>`;
      }).join('');

      const fbCls  = answered ? ('q-feedback show ' + (s.chosen===q.correct?'ok':'bad')) : 'q-feedback';
      const fbText = answered ? (s.chosen===q.correct ? q.ok : q.bad) : '';

      return `<div class="q-card">
        <div class="q-num">Question ${qi+1} of ${QUESTIONS.length}</div>
        <div class="q-prompt">${q.q}</div>
        ${opts}
        <div class="${fbCls}" id="qfb${qi}">${fbText}</div>
      </div>`;
    }).join('');

    const pips = state.map((s, i) => {
      const cls = s.chosen===null ? 'pip' : (s.chosen===QUESTIONS[i].correct?'pip ok':'pip bad');
      return `<div class="${cls}"></div>`;
    }).join('');

    const allDone = answered === QUESTIONS.length;
    const scoreCol = correct === QUESTIONS.length ? '#4ade80' : correct >= 4 ? '#fbbf24' : '#ff6b35';

    root.innerHTML = `
      <div class="quiz-grid">${cards}</div>
      <div class="score-bar">
        <span style="color:#52525b;font-size:0.8em;white-space:nowrap;">Score</span>
        <div style="display:flex;gap:5px;flex-wrap:wrap;">${pips}</div>
        <span style="color:${scoreCol};font-weight:700;">${correct}/${QUESTIONS.length}${correct===QUESTIONS.length?' \u2014 Perfect! \uD83C\uDF89':''}</span>
        ${allDone && correct < QUESTIONS.length
          ? `<span style="color:#52525b;font-size:0.82em;">(${QUESTIONS.length-correct} to review)</span>` : ''}
      </div>`;
  }

  window.quizAnswer = function(qi, oi) {
    if (state[qi].chosen !== null) return;
    state[qi].chosen = oi;
    render();
  };

  render();
})();
</script>

<!-- ════════════════════════════════════════════════════
     REACT / BABEL — Panel 5: MLP XOR Walkthrough
     ════════════════════════════════════════════════════ -->
<script type="text/babel">
(function () {
  const { useState } = React;

  const PLOT = 200, PAD = 30, S = PLOT - PAD * 2;
  const toX = v => PAD + v * S;
  const toY = v => PAD + (1 - v) * S;

  const points = [
    { x:0, y:0, xor:0 }, { x:0, y:1, xor:1 },
    { x:1, y:0, xor:1 }, { x:1, y:1, xor:0 },
  ];

  const stages = [
    {
      id:"input", title:"Input Space", subtitle:"The XOR problem",
      desc:"XOR outputs sit at opposite corners. No single straight line can separate the 1s from the 0s \u2014 a fundamental limitation of any single perceptron.",
      lines:[], region:null,
      pointColor: p => p.xor===1?"#4ade80":"#334155",
      pointStroke: p => p.xor===1?"#4ade80":"#475569",
    },
    {
      id:"or", title:"Hidden Neuron 1: OR", subtitle:'\u201cIs at least one input = 1?\u201d',
      desc:"This neuron draws a line that separates (0,0) from everything else. Every input except both-zero passes through \u2014 computing OR.",
      lines:[{x1:-0.1,y1:0.6,x2:0.6,y2:-0.1,color:"#38bdf8"}],
      region:{type:"or",color:"#38bdf820"},
      pointColor: p => (p.x===0&&p.y===0)?"#334155":"#38bdf8",
      pointStroke: p => (p.x===0&&p.y===0)?"#475569":"#38bdf8",
      outputLabels:[0,1,1,1],
    },
    {
      id:"nand", title:"Hidden Neuron 2: NAND", subtitle:'\u201cAre BOTH inputs NOT 1?\u201d',
      desc:"This neuron draws a line that separates (1,1) from everything else. It fires for every input except both-on \u2014 computing NAND.",
      lines:[{x1:0.4,y1:1.1,x2:1.1,y2:0.4,color:"#c084fc"}],
      region:{type:"nand",color:"#c084fc20"},
      pointColor: p => (p.x===1&&p.y===1)?"#334155":"#c084fc",
      pointStroke: p => (p.x===1&&p.y===1)?"#475569":"#c084fc",
      outputLabels:[1,1,1,0],
    },
    {
      id:"combined", title:"Both Lines Together", subtitle:"Two boundaries carve the space",
      desc:"Each line eliminates one incorrect corner. The region between both lines contains exactly the XOR=1 inputs \u2014 (0,1) and (1,0) \u2014 which pass both hidden neurons.",
      lines:[
        {x1:-0.1,y1:0.6,x2:0.6,y2:-0.1,color:"#38bdf8"},
        {x1:0.4,y1:1.1,x2:1.1,y2:0.4,color:"#c084fc"},
      ],
      region:{type:"between",color:"#4ade8015"},
      pointColor: p => p.xor===1?"#4ade80":"#334155",
      pointStroke: p => p.xor===1?"#4ade80":"#475569",
    },
    {
      id:"output", title:"Output Neuron: AND", subtitle:'\u201cDo BOTH hidden neurons agree?\u201d',
      desc:"The AND output neuron combines both results. Only (0,1) and (1,0) pass BOTH hidden neurons \u2192 XOR solved! Two simple linear boundaries produce a non-linear decision.",
      lines:[
        {x1:-0.1,y1:0.6,x2:0.6,y2:-0.1,color:"#38bdf860"},
        {x1:0.4,y1:1.1,x2:1.1,y2:0.4,color:"#c084fc60"},
      ],
      region:{type:"between",color:"#4ade8020"},
      pointColor: p => p.xor===1?"#4ade80":"#334155",
      pointStroke: p => p.xor===1?"#4ade80":"#475569",
    },
  ];

  function MiniPlot({ stage }) {
    const fillRegion = () => {
      if (!stage.region) return null;
      const {type,color} = stage.region;
      if (type==="or")      return <polygon points={`${toX(0.6)},${toY(0)} ${toX(0)},${toY(0.6)} ${toX(0)},${toY(1)} ${toX(1)},${toY(1)} ${toX(1)},${toY(0)}`} fill={color}/>;
      if (type==="nand")    return <polygon points={`${toX(0)},${toY(0)} ${toX(1)},${toY(0)} ${toX(1)},${toY(0.4)} ${toX(0.4)},${toY(1)} ${toX(0)},${toY(1)}`} fill={color}/>;
      if (type==="between") return <polygon points={`${toX(0)},${toY(0.6)} ${toX(0.6)},${toY(0)} ${toX(1)},${toY(0.4)} ${toX(0.4)},${toY(1)}`} fill={color} stroke="#4ade8030" strokeWidth={1}/>;
      return null;
    };
    return (
      <svg width={PLOT} height={PLOT} viewBox={`0 0 ${PLOT} ${PLOT}`}
        style={{background:"#08080d",borderRadius:10,border:"2px solid #151520",transition:"all 0.3s"}}>
        {[0,0.5,1].map((v,i) => (
          <g key={i}>
            <line x1={toX(v)} y1={PAD} x2={toX(v)} y2={PAD+S} stroke="#111122" strokeWidth={v===0.5?0.5:1}/>
            <line x1={PAD} y1={toY(v)} x2={PAD+S} y2={toY(v)} stroke="#111122" strokeWidth={v===0.5?0.5:1}/>
          </g>
        ))}
        <text x={toX(0)} y={PLOT-6} fill="#444" fontSize={9} textAnchor="middle" fontFamily="JetBrains Mono,monospace">0</text>
        <text x={toX(1)} y={PLOT-6} fill="#444" fontSize={9} textAnchor="middle" fontFamily="JetBrains Mono,monospace">1</text>
        <text x={10}     y={toY(0)+3} fill="#444" fontSize={9} textAnchor="middle" fontFamily="JetBrains Mono,monospace">0</text>
        <text x={10}     y={toY(1)+3} fill="#444" fontSize={9} textAnchor="middle" fontFamily="JetBrains Mono,monospace">1</text>
        {fillRegion()}
        {stage.lines.map((l,i) => (
          <line key={i} x1={toX(l.x1)} y1={toY(l.y1)} x2={toX(l.x2)} y2={toY(l.y2)}
            stroke={l.color} strokeWidth={2} strokeDasharray="5 3"/>
        ))}
        {points.map((p,i) => (
          <g key={i}>
            <circle cx={toX(p.x)} cy={toY(p.y)} r={12} fill={stage.pointColor(p)+"15"}/>
            <circle cx={toX(p.x)} cy={toY(p.y)} r={9}  fill={stage.pointColor(p)} stroke={stage.pointStroke(p)} strokeWidth={2}/>
            <text x={toX(p.x)} y={toY(p.y)+3.5} fill={stage.pointColor(p)==="#334155"?"#94a3b8":"#0a0a0f"}
              fontSize={10} fontWeight={700} textAnchor="middle" fontFamily="JetBrains Mono,monospace">{p.xor}</text>
          </g>
        ))}
      </svg>
    );
  }

  function NetworkDiagram({ activeStage }) {
    const neurons = {
      x1:   {x:60,  y:60,  label:"x\u2081", color:"#94a3b8"},
      x2:   {x:60,  y:160, label:"x\u2082", color:"#94a3b8"},
      or:   {x:200, y:55,  label:"OR",      color:"#38bdf8"},
      nand: {x:200, y:165, label:"NAND",    color:"#c084fc"},
      and:  {x:340, y:110, label:"AND",     color:"#4ade80"},
    };
    const connections = [
      {from:"x1",to:"or"},{from:"x2",to:"or"},
      {from:"x1",to:"nand"},{from:"x2",to:"nand"},
      {from:"or",to:"and"},{from:"nand",to:"and"},
    ];
    const getHighlight = id => {
      if (activeStage==="input")    return id==="x1"||id==="x2"?1:0.25;
      if (activeStage==="or")       return id==="x1"||id==="x2"||id==="or"?1:0.25;
      if (activeStage==="nand")     return id==="x1"||id==="x2"||id==="nand"?1:0.25;
      if (activeStage==="combined") return id==="or"||id==="nand"?1:0.35;
      if (activeStage==="output")   return 1;
      return 0.5;
    };
    const getConnHL = (from,to) => {
      if (activeStage==="or")       return to==="or"?1:0.15;
      if (activeStage==="nand")     return to==="nand"?1:0.15;
      if (activeStage==="combined") return to==="and"?0.6:0.3;
      if (activeStage==="output")   return 0.8;
      return 0.2;
    };
    const layerLabels = [
      {x:60,  label:"Input",  active:activeStage==="input"},
      {x:200, label:"Hidden", active:["or","nand","combined"].includes(activeStage)},
      {x:340, label:"Output", active:activeStage==="output"},
    ];
    return (
      <svg width={400} height={220} viewBox="0 0 400 220"
        style={{background:"#08080d",borderRadius:12,border:"2px solid #151520"}}>
        {layerLabels.map((l,i) => (
          <text key={i} x={l.x} y={205} fill={l.active?"#e2e8f0":"#333"}
            fontSize={10} fontWeight={600} textAnchor="middle"
            fontFamily="JetBrains Mono,monospace" style={{transition:"fill 0.3s"}}>{l.label}</text>
        ))}
        {connections.map((c,i) => {
          const f=neurons[c.from], t=neurons[c.to], op=getConnHL(c.from,c.to);
          return <line key={i} x1={f.x+18} y1={f.y} x2={t.x-18} y2={t.y}
            stroke={neurons[c.to].color} strokeWidth={op>0.5?2:1} opacity={op}
            style={{transition:"all 0.4s"}}/>;
        })}
        {Object.entries(neurons).map(([id,n]) => {
          const op=getHighlight(id), active=op>0.5;
          return (
            <g key={id} opacity={op} style={{transition:"opacity 0.4s"}}>
              <circle cx={n.x} cy={n.y} r={22} fill={active?n.color+"18":"#08080d"} stroke={n.color} strokeWidth={active?2.5:1.5}/>
              <text x={n.x} y={n.y+4} fill={active?n.color:"#555"}
                fontSize={id==="x1"||id==="x2"?13:11} fontWeight={700}
                textAnchor="middle" fontFamily="JetBrains Mono,monospace">{n.label}</text>
            </g>
          );
        })}
        {activeStage==="output" && (
          <g>
            <text x={370} y={110} fill="#4ade80" fontSize={16} textAnchor="start" fontFamily="monospace">{"\u2192"}</text>
            <text x={385} y={115} fill="#4ade80" fontSize={10} fontWeight={700} fontFamily="JetBrains Mono,monospace" textAnchor="start">XOR</text>
          </g>
        )}
      </svg>
    );
  }

  function TruthTable({ activeStage }) {
    const rows = [
      {x1:0,x2:0,or:0,nand:1,and:0},{x1:0,x2:1,or:1,nand:1,and:1},
      {x1:1,x2:0,or:1,nand:1,and:1},{x1:1,x2:1,or:1,nand:0,and:0},
    ];
    const showOr   = ["or","combined","output"].includes(activeStage);
    const showNand = ["nand","combined","output"].includes(activeStage);
    const showAnd  = activeStage==="output";

    const cellStyle = (hl,color) => ({
      padding:"6px 10px", fontSize:12, fontFamily:"JetBrains Mono,monospace",
      textAlign:"center", color:hl?color:"#334155", fontWeight:hl?700:400,
      background:hl?color+"08":"transparent", transition:"all 0.3s", borderBottom:"1px solid #111118",
    });
    const headStyle = (active,color) => ({
      padding:"8px 10px", fontSize:10, fontFamily:"JetBrains Mono,monospace",
      textAlign:"center", color:active?color:"#333", fontWeight:700,
      borderBottom:"2px solid #1a1a2a", transition:"color 0.3s", letterSpacing:"0.05em",
    });
    return (
      <table style={{borderCollapse:"collapse",background:"#08080d",borderRadius:10,
        overflow:"hidden",border:"2px solid #151520",width:"100%",maxWidth:420}}>
        <thead><tr>
          <th style={headStyle(true,"#94a3b8")}>x&#x2081;</th>
          <th style={headStyle(true,"#94a3b8")}>x&#x2082;</th>
          <th style={headStyle(showOr,"#38bdf8")}>OR</th>
          <th style={headStyle(showNand,"#c084fc")}>NAND</th>
          <th style={headStyle(showAnd,"#4ade80")}>AND&#x2192;XOR</th>
        </tr></thead>
        <tbody>{rows.map((r,i) => (
          <tr key={i}>
            <td style={cellStyle(true,"#e2e8f0")}>{r.x1}</td>
            <td style={cellStyle(true,"#e2e8f0")}>{r.x2}</td>
            <td style={cellStyle(showOr,  r.or===1  ?"#38bdf8":"#f87171")}>{showOr   ? r.or   : "\u2014"}</td>
            <td style={cellStyle(showNand,r.nand===1 ?"#c084fc":"#f87171")}>{showNand ? r.nand : "\u2014"}</td>
            <td style={cellStyle(showAnd, r.and===1  ?"#4ade80":"#64748b")}>{showAnd  ? r.and  : "\u2014"}</td>
          </tr>
        ))}</tbody>
      </table>
    );
  }

  function MLPXORVisual() {
    const [step, setStep] = useState(0);
    const stage = stages[step];

    const stageColor = {
      or:"#38bdf8", nand:"#c084fc", output:"#4ade80", combined:"#fbbf24", input:"#e2e8f0"
    }[stage.id] || "#e2e8f0";

    const navBtn = (label, onClick, disabled, highlight) => ({
      style: {
        padding:"8px 20px", borderRadius:8,
        border: disabled ? "1px solid #1e1e2e" : highlight ? "1px solid #4ade8040" : "1px solid #1e1e2e",
        background: disabled ? "#08080d" : highlight ? "#0a1a10" : "#0f0f18",
        color: disabled ? "#333" : highlight ? "#4ade80" : "#e2e8f0",
        cursor: disabled ? "default" : "pointer",
        fontFamily:"JetBrains Mono,monospace", fontSize:13, fontWeight:600,
      }
    });

    return (
      <div style={{background:"#050508",color:"#e2e8f0",fontFamily:"JetBrains Mono,monospace",
        display:"flex",flexDirection:"column",alignItems:"center",padding:"28px 16px"}}>

        <NetworkDiagram activeStage={stage.id}/>

        <div style={{height:20}}/>

        <div style={{display:"flex",gap:20,flexWrap:"wrap",justifyContent:"center",alignItems:"flex-start",maxWidth:680}}>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:11,fontWeight:700,color:"#52525b",letterSpacing:"0.08em",marginBottom:6}}>DECISION BOUNDARY</div>
            <MiniPlot stage={stage}/>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:11,fontWeight:700,color:"#52525b",letterSpacing:"0.08em",marginBottom:6}}>TRUTH TABLE</div>
            <TruthTable activeStage={stage.id}/>
          </div>
        </div>

        <div style={{background:"#0a0a10",border:"1px solid #1e1e2e",borderRadius:12,
          padding:"14px 22px",maxWidth:500,textAlign:"center",marginTop:20}}>
          <div style={{fontSize:16,fontWeight:700,color:stageColor,marginBottom:2}}>{stage.title}</div>
          <div style={{fontSize:13,color:"#64748b",fontStyle:"italic",marginBottom:6}}>{stage.subtitle}</div>
          <div style={{fontSize:13,color:"#94a3b8",lineHeight:1.6}}>{stage.desc}</div>
        </div>

        <div style={{display:"flex",gap:10,marginTop:22,alignItems:"center"}}>
          <button onClick={() => setStep(Math.max(0,step-1))} disabled={step===0}
            style={navBtn('Back', null, step===0, false).style}>&#8592; Back</button>

          <div style={{display:"flex",gap:6}}>
            {stages.map((_,i) => (
              <div key={i} onClick={() => setStep(i)} style={{
                width:i===step?24:8, height:8, borderRadius:4, cursor:"pointer",
                background:i===step?"#4ade80":i<step?"#4ade8060":"#1e1e2e", transition:"all 0.3s",
              }}/>
            ))}
          </div>

          <button onClick={() => setStep(Math.min(stages.length-1,step+1))} disabled={step===stages.length-1}
            style={navBtn('Next', null, step===stages.length-1, true).style}>Next &#8594;</button>
        </div>

        <div style={{color:"#333",fontSize:11,marginTop:8}}>Step {step+1} of {stages.length}</div>
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById("mlp-root")).render(<MLPXORVisual/>);
})();
</script>

</body>
</html>"""

PERCEPTRON_VISUAL_HEIGHT = 2050
