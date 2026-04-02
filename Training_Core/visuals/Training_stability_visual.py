TRAINING_STABILITY_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',sans-serif;background:#08080f;color:#e4e4e7;padding:20px;min-height:100vh;}
h2{font-family:'JetBrains Mono',monospace;font-size:1.1em;color:#ff6b35;margin-bottom:3px;}
.sub{color:#52525b;font-size:0.78em;margin-bottom:18px;}
.tabs{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:18px;overflow-x:auto;}
.tab{padding:10px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:0.72em;font-weight:700;white-space:nowrap;margin-bottom:-2px;transition:all 0.2s;}
.tab:hover{color:#e4e4e7;}.tab.active{color:#ff6b35;border-bottom-color:#ff6b35;}
.panel{display:none;}.panel.active{display:block;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.card{background:#111118;border:1px solid #1e1e2e;border-radius:14px;padding:18px;margin-bottom:14px;}
.card h3{font-family:'JetBrains Mono',monospace;font-size:0.75em;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#ff6b35;margin-bottom:10px;}
canvas{display:block;border-radius:8px;background:#09090f;}
.row{display:flex;align-items:center;gap:10px;margin:6px 0;}
.row label{font-family:'JetBrains Mono',monospace;font-size:0.73em;color:#71717a;min-width:100px;}
input[type=range]{flex:1;height:4px;border-radius:2px;-webkit-appearance:none;background:#1e1e2e;outline:none;cursor:pointer;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#ff6b35;cursor:pointer;}
.v{font-family:'JetBrains Mono',monospace;font-size:0.73em;color:#ff6b35;min-width:50px;text-align:right;}
.info{background:#0d0d18;border:1px solid #1e1e2e;border-radius:8px;padding:10px 14px;font-size:0.76em;color:#94a3b8;line-height:1.8;margin-top:10px;font-family:'JetBrains Mono',monospace;}
.hl{color:#ff6b35;font-weight:700;}.hl2{color:#4ecdc4;font-weight:700;}.hl3{color:#fbbf24;font-weight:700;}.hl4{color:#4ade80;font-weight:700;}.hl5{color:#ef4444;font-weight:700;}.hl6{color:#c084fc;font-weight:700;}
.brow{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
button.s{background:#1e1e2e;color:#a1a1aa;border:1px solid #2d2d40;border-radius:6px;padding:5px 12px;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:0.73em;font-weight:700;transition:all 0.15s;}
button.s:hover{background:#2d2d40;color:#e4e4e7;}button.s.active{background:#ff6b35;color:#08080f;border-color:#ff6b35;}
.tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.71em;}
.tbl th{padding:6px 9px;text-align:left;color:#52525b;border-bottom:2px solid #1e1e2e;font-weight:800;text-transform:uppercase;letter-spacing:.05em;}
.tbl td{padding:6px 9px;border-bottom:1px solid #0f0f18;color:#94a3b8;vertical-align:middle;line-height:1.5;}
.tbl tr:hover td{background:#0d0d18;}
</style>
</head>
<body>
<h2>&#x1F4C8; Training Stability</h2>
<p class="sub">Loss Curves &middot; Divergence &middot; Precision &middot; Monitoring &middot; Reproducibility &middot; Debug Workflow</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Loss Curves</button>
  <button class="tab" onclick="showTab(1)">Divergence</button>
  <button class="tab" onclick="showTab(2)">Precision</button>
  <button class="tab" onclick="showTab(3)">Monitoring</button>
  <button class="tab" onclick="showTab(4)">Reproducibility</button>
  <button class="tab" onclick="showTab(5)">Debug Protocol</button>
</div>

<!-- TAB 0: LOSS CURVES -->
<div id="tab0" class="panel active">
<div class="g2">
<div class="card">
  <h3>&#9312; Stability Spectrum &mdash; Six Loss Curve Patterns</h3>
  <canvas id="cvSpectrum" width="420" height="290"></canvas>
  <div class="brow">
    <button class="s active" id="spBtn_A" onclick="selSpectrum('A')">A: Stable</button>
    <button class="s" id="spBtn_B" onclick="selSpectrum('B')">B: Spikes</button>
    <button class="s" id="spBtn_C" onclick="selSpectrum('C')">C: Diverge</button>
    <button class="s" id="spBtn_D" onclick="selSpectrum('D')">D: Oscillate</button>
    <button class="s" id="spBtn_E" onclick="selSpectrum('E')">E: Silent</button>
    <button class="s" id="spBtn_F" onclick="selSpectrum('F')">F: NaN</button>
  </div>
  <div class="info" id="iSpectrum">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Raw vs Smoothed Loss (EMA)</h3>
  <canvas id="cvSmooth" width="420" height="290"></canvas>
  <div class="row"><label>EMA beta &beta;</label>
    <input type="range" id="slBeta" min="0" max="99" step="1" value="95" oninput="renderSmooth()">
    <span class="v" id="vBeta">0.95</span></div>
  <div class="info" id="iSmooth">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Full Training Run Anatomy</h3>
  <canvas id="cvAnatomy" width="860" height="155"></canvas>
</div>
</div>

<!-- TAB 1: DIVERGENCE -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; LR Phase Diagram &mdash; Stability Region</h3>
  <canvas id="cvPhase" width="420" height="270"></canvas>
  <div class="row"><label>Your LR</label>
    <input type="range" id="slLR" min="0" max="100" step="1" value="40" oninput="renderPhase()">
    <span class="v" id="vLR">1e-3</span></div>
  <div class="info" id="iPhase">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Loss Curve Shapes: Cause &amp; Fix</h3>
  <canvas id="cvShapes" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="shBtn_osc"   onclick="selShape('osc')">Oscillating</button>
    <button class="s" id="shBtn_spike"  onclick="selShape('spike')">Spike</button>
    <button class="s" id="shBtn_plat"   onclick="selShape('plat')">Plateau</button>
    <button class="s" id="shBtn_over"   onclick="selShape('over')">Overfit</button>
    <button class="s" id="shBtn_drift"  onclick="selShape('drift')">Drift</button>
  </div>
  <div class="info" id="iShapes">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Systematic Divergence Diagnosis: 7-Step Elimination</h3>
  <canvas id="cvDiag" width="860" height="148"></canvas>
</div>
</div>

<!-- TAB 2: PRECISION -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; FP32 / FP16 / BF16 Numeric Ranges</h3>
  <canvas id="cvPrecision" width="420" height="260"></canvas>
  <div class="info" id="iPrecision">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Dynamic Loss Scaling (FP16)</h3>
  <canvas id="cvLossScale" width="420" height="260"></canvas>
  <div class="row"><label>Scale factor S</label>
    <input type="range" id="slScale" min="0" max="100" step="1" value="70" oninput="renderLossScale()">
    <span class="v" id="vScale">65536</span></div>
  <div class="info" id="iLossScale">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Mixed Precision Strategy</h3>
  <table class="tbl">
    <thead><tr><th>Format</th><th>Bits</th><th>Range</th><th>Loss scaling?</th><th>Hardware</th><th>Recommendation</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">BF16</span></td><td>16</td><td>\u00B13.4\u00D710<sup>38</sup> (same as FP32)</td><td><span class="hl4">Not needed</span></td><td>A100, H100, TPU v3+</td><td><span class="hl4">Use if available. Simplest, most stable.</span></td></tr>
      <tr><td><span class="hl3">FP16</span></td><td>16</td><td>\u00B165,504</td><td><span class="hl5">Required (GradScaler)</span></td><td>V100, T4, older GPUs</td><td><span class="hl3">Use with GradScaler when BF16 unavailable.</span></td></tr>
      <tr><td><span class="hl2">FP32</span></td><td>32</td><td>\u00B13.4\u00D710<sup>38</sup></td><td><span class="hl4">Not needed</span></td><td>Any CUDA GPU, CPU</td><td>Baseline. Safest but 2-4\u00D7 slower than BF16.</td></tr>
      <tr><td><span class="hl6">FP8</span></td><td>8</td><td>Tiny range</td><td><span class="hl5">Required + careful tuning</span></td><td>H100 only</td><td>Cutting-edge LLM training only. Complex setup.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 3: MONITORING -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Gradient Norm Over Training</h3>
  <canvas id="cvGradNorm" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="gnBtn_healthy"   onclick="selGN('healthy')">Healthy</button>
    <button class="s" id="gnBtn_explode"  onclick="selGN('explode')">Exploding</button>
    <button class="s" id="gnBtn_vanish"   onclick="selGN('vanish')">Vanishing</button>
    <button class="s" id="gnBtn_clipped"  onclick="selGN('clipped')">Clipped</button>
  </div>
  <div class="info" id="iGradNorm">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Update Ratio (grad/weight) &mdash; Karpathy Diagnostic</h3>
  <canvas id="cvUpdateRatio" width="420" height="260"></canvas>
  <div class="row"><label>LR</label>
    <input type="range" id="slURlr" min="0" max="100" step="1" value="50" oninput="renderUpdateRatio()">
    <span class="v" id="vURlr">1e-3</span></div>
  <div class="info" id="iUpdateRatio">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Health Monitoring Reference Ranges</h3>
  <table class="tbl">
    <thead><tr><th>Statistic</th><th>Healthy</th><th>Warning</th><th>Critical</th><th>Action</th></tr></thead>
    <tbody>
      <tr><td><span class="hl2">Global gradient norm</span></td><td>0.05 &ndash; 5.0</td><td>&gt;10 or &lt;0.001</td><td>&gt;100</td><td>Add clipping (max_norm=1.0) or reduce LR</td></tr>
      <tr><td><span class="hl4">Activation std (post-LN)</span></td><td>~1.0 (0.5&ndash;2.0)</td><td>&lt;0.1 or &gt;5.0</td><td>&lt;0.01 or &gt;50</td><td>Check LayerNorm placement; check init</td></tr>
      <tr><td><span class="hl3">Weight std</span></td><td>0.01 &ndash; 0.5</td><td>Consistently growing</td><td>&gt;10</td><td>Reduce LR; check weight decay</td></tr>
      <tr><td><span class="hl">Update ratio (\u03B7\u00B7grad/weight)</span></td><td>1e-4 &ndash; 1e-2</td><td>&gt;1e-1 or &lt;1e-5</td><td>&gt;1.0</td><td>Too high: reduce LR. Too low: increase LR.</td></tr>
      <tr><td><span class="hl6">Loss scale (FP16)</span></td><td>128 &ndash; 65536</td><td>&lt;64 or declining</td><td>=1.0</td><td>Reduce LR; check for problematic layers</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 4: REPRODUCIBILITY -->
<div id="tab4" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Seed-to-Seed Variance Visualiser</h3>
  <canvas id="cvSeeds" width="420" height="270"></canvas>
  <div class="row"><label>Num seeds</label>
    <input type="range" id="slSeeds" min="1" max="7" step="1" value="5" oninput="renderSeeds()">
    <span class="v" id="vSeeds">5</span></div>
  <div class="info" id="iSeeds">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Sources of Non-Reproducibility</h3>
  <canvas id="cvRepro" width="420" height="270"></canvas>
  <div class="info" id="iRepro">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Reproducibility Checklist</h3>
  <div class="g2">
  <table class="tbl">
    <thead><tr><th>Source</th><th>Fix</th><th>Cost</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">Weight init</span></td><td>torch.manual_seed(seed)</td><td>None</td></tr>
      <tr><td><span class="hl5">Data ordering</span></td><td>Seed DataLoader sampler</td><td>None</td></tr>
      <tr><td><span class="hl5">Dropout / augmentation</span></td><td>random.seed + np.random.seed</td><td>None</td></tr>
      <tr><td><span class="hl5">CUDA non-determinism</span></td><td>cudnn.deterministic=True</td><td><span class="hl5">10&ndash;30% slower</span></td></tr>
      <tr><td><span class="hl5">Worker RNG</span></td><td>worker_init_fn with unique seed</td><td>None</td></tr>
    </tbody>
  </table>
  <table class="tbl">
    <thead><tr><th>Reporting Rule</th><th>Detail</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">Run 3&ndash;5 seeds</span></td><td>Always. Single-seed results are misleading.</td></tr>
      <tr><td><span class="hl4">Report mean \u00B1 std</span></td><td>84.3\u00B10.3% not just 84.7% (best run)</td></tr>
      <tr><td><span class="hl4">Statistical test</span></td><td>t-test / Mann-Whitney if comparing models</td></tr>
      <tr><td><span class="hl3">Same tuning budget</span></td><td>Compare models tuned equally. Else tuning inflates winner.</td></tr>
      <tr><td><span class="hl5">Never tune on test set</span></td><td>Evaluate test set ONCE on final model only.</td></tr>
    </tbody>
  </table>
  </div>
</div>
</div>

<!-- TAB 5: DEBUG PROTOCOL -->
<div id="tab5" class="panel">
<div class="card">
  <h3>&#9312; Make It Work &rarr; Make It Right &rarr; Make It Fast</h3>
  <canvas id="cvProtocol" width="860" height="175"></canvas>
</div>
<div class="g2">
<div class="card">
  <h3>&#9313; Pre-Training Sanity Checks</h3>
  <canvas id="cvSanity" width="420" height="260"></canvas>
  <div class="info" id="iSanity">Loading...</div>
</div>
<div class="card">
  <h3>&#9314; Initial Loss Baseline Calculator</h3>
  <canvas id="cvBaseline" width="420" height="260"></canvas>
  <div class="row"><label>Task</label>
    <input type="range" id="slTask" min="0" max="4" step="1" value="0" oninput="renderBaseline()">
    <span class="v" id="vTask">Binary</span></div>
  <div class="info" id="iBaseline">Loading...</div>
</div>
</div>
</div>

<script>
// ============================================================
// UTILS
// ============================================================
function rnd(v,d){return v.toFixed(d===undefined?3:d);}
function clr(id,W,H){var cv=document.getElementById(id),ctx=cv.getContext('2d');ctx.fillStyle='#09090f';ctx.fillRect(0,0,W,H);return ctx;}
function tx(ctx,s,x,y,col,sz,wt,al){ctx.fillStyle=col||'#e4e4e7';ctx.font=(wt||'500')+' '+(sz||11)+'px JetBrains Mono,monospace';ctx.textAlign=al||'center';ctx.textBaseline='middle';ctx.fillText(s,x,y);}
function ln(ctx,x1,y1,x2,y2,col,lw,dash){ctx.strokeStyle=col;ctx.lineWidth=lw||1.5;if(dash)ctx.setLineDash(dash);else ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);}
function bx(ctx,x,y,w,h,r,fill,stroke,sw){ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,w,h,r||4);else ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw||1.5;ctx.stroke();}}
function crv(pts,ctx,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||2.5;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}
function axesPAD(ctx,W,H,PAD,xl,yl){ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);if(xl)tx(ctx,xl,(PAD+W-10)/2,H-5,'#52525b',8,'400');if(yl)tx(ctx,yl,PAD+20,22,'#52525b',8,'400');}
var seed=42;function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
function resetSeed(s){seed=s||42;}

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderSpectrum();renderSmooth();renderAnatomy();}if(i===1){renderPhase();renderShapes();renderDiag();}if(i===2){renderPrecision();renderLossScale();}if(i===3){renderGradNorm();renderUpdateRatio();}if(i===4){renderSeeds();renderRepro();}if(i===5){renderProtocol();renderSanity();renderBaseline();}}

// ============================================================
// TAB 0: LOSS CURVES
// ============================================================
var curSpectrum='A';
var SPECTRUM={
  A:{col:'#4ade80',label:'A: Stable',info:'<span class="hl4">Stable:</span> loss decreases monotonically (with allowable stochastic noise). Gradient norms bounded. Val loss tracks train. This is the goal.',fix:'No action needed. Monitor for overfit as training continues.'},
  B:{col:'#fbbf24',label:'B: Spikes',info:'<span class="hl3">Loss spikes:</span> sudden large increases followed by partial recovery. Single corrupted batch, gradient explosion on rare example, or brief instability.',fix:'Add gradient clipping (max_norm=1.0). Filter corrupted data. Check for extreme loss values in individual batches.'},
  C:{col:'#ef4444',label:'C: Divergence',info:'<span class="hl5">Divergence:</span> monotone increase that does not reverse. LR too large, gradient explosion, or a fundamental model/loss bug.',fix:'Reduce LR by 10x first. Add gradient clipping. Check loss reduction mode (mean not sum). Check labels.'},
  D:{col:'#ff6b35',label:'D: Oscillation',info:'<span class="hl">Oscillation:</span> loss zigzags without a downward trend. LR is overshooting the minimum on every step.',fix:'Reduce LR by 2-10x. Add LR warmup. Add gradient clipping.'},
  E:{col:'#3f3f46',label:'E: Silent fail',info:'<span class="hl5">Silent failure:</span> loss barely moves from its initial random-chance value. Model appears to train but learns nothing.',fix:'Check initial loss vs log(K) baseline. Check label indexing (0-indexed?). Check data loading (images visualised correctly?). Check model output shape.'},
  F:{col:'#c084fc',label:'F: NaN collapse',info:'<span class="hl6">NaN cascade:</span> loss becomes NaN and training collapses. FP16 overflow, gradient explosion to Inf, or NaN in input data.',fix:'Enable torch.autograd.set_detect_anomaly(True). Check input data for NaN. Add gradient clipping. Use BF16 instead of FP16.'},
};
function selSpectrum(k){curSpectrum=k;Object.keys(SPECTRUM).forEach(function(b){document.getElementById('spBtn_'+b).classList.remove('active');});document.getElementById('spBtn_'+k).classList.add('active');renderSpectrum();}
function renderSpectrum(){
  var cfg=SPECTRUM[curSpectrum];
  var ctx=clr('cvSpectrum',420,290),W=420,H=290,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Loss');
  resetSeed(99);
  var N=80,pts=[];
  for(var i=0;i<=N;i++){
    var t=i/N,cx=PAD+t*pW,loss,cy;
    if(curSpectrum==='A'){
      loss=2.5*Math.exp(-t*4)+0.4+rng()*0.06-0.03;
    } else if(curSpectrum==='B'){
      loss=2.5*Math.exp(-t*3)+0.5+rng()*0.08-0.04;
      if(i===28)loss+=2.8;
      if(i===29)loss+=1.5;
    } else if(curSpectrum==='C'){
      loss=Math.min(8,1.2+Math.pow(t*2.5,1.8)+rng()*0.2);
    } else if(curSpectrum==='D'){
      loss=1.8-t*0.3+Math.sin(t*40)*0.7+rng()*0.15;
    } else if(curSpectrum==='E'){
      loss=2.302+rng()*0.05-0.025;
    } else { // F NaN
      if(t<0.35) loss=2.5*Math.exp(-t*3)+0.8+rng()*0.06;
      else loss=NaN;
    }
    cy=isNaN(loss)?null:H-PAD-Math.min(Math.max(loss,0),8)/8*pH;
    pts.push([cx,cy]);
  }
  // Draw
  ctx.strokeStyle=cfg.col;ctx.lineWidth=2.5;ctx.beginPath();
  var started=false;
  pts.forEach(function(p){
    if(p[1]===null){started=false;return;}
    if(!started){ctx.moveTo(p[0],p[1]);started=true;}
    else ctx.lineTo(p[0],p[1]);
  });
  ctx.stroke();
  // NaN dot
  if(curSpectrum==='F'){
    var nanIdx=Math.round(0.35*N);
    tx(ctx,'NaN',PAD+0.4*pW,H-PAD-pH*0.8,'#c084fc',12,'800');
    ctx.fillStyle='#c084fc';ctx.beginPath();ctx.arc(PAD+0.35*pW,H-PAD-5,5,0,Math.PI*2);ctx.fill();
  }
  tx(ctx,cfg.label,W/2,22,cfg.col,10,'800');
  document.getElementById('iSpectrum').innerHTML=cfg.info+'<br><span class="hl2">Fix:</span> '+cfg.fix;
}

function renderSmooth(){
  var beta=parseInt(document.getElementById('slBeta').value)/100;
  document.getElementById('vBeta').textContent=rnd(beta,2);
  var ctx=clr('cvSmooth',420,290),W=420,H=290,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Loss');
  resetSeed(7);
  var N=120,rawPts=[],smPts=[],ema=0,raw0=2.5;
  for(var i=0;i<=N;i++){
    var t=i/N,cx=PAD+t*pW;
    var trend=raw0*Math.exp(-t*3.5)+0.4;
    var raw=trend+(rng()-0.5)*0.7;
    raw=Math.max(0.15,raw);
    ema=i===0?raw:beta*ema+(1-beta)*raw;
    rawPts.push([cx,H-PAD-Math.min(raw/raw0,1)*pH]);
    smPts.push([cx,H-PAD-Math.min(ema/raw0,1)*pH]);
  }
  // Raw (faint)
  ctx.strokeStyle='#4ecdc440';ctx.lineWidth=1;ctx.beginPath();
  rawPts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();
  // Smoothed
  crv(smPts,ctx,'#ff6b35',2.5);
  ln(ctx,PAD+40,24,PAD+58,24,'#4ecdc440',1);tx(ctx,'Raw (noisy)',PAD+110,24,'#4ecdc4',8,'700');
  ln(ctx,PAD+180,24,PAD+198,24,'#ff6b35',2.5);tx(ctx,'EMA (\u03B2='+rnd(beta,2)+')',PAD+270,24,'#ff6b35',8,'700');
  var lag=beta<0.5?'< 1 step':beta<0.9?'~'+Math.round(1/(1-beta))+' steps':'~'+Math.round(1/(1-beta))+' steps';
  document.getElementById('iSmooth').innerHTML=
    '\u03B2=<span class="hl3">'+rnd(beta,2)+'</span>. Lag: ~<span class="hl3">'+lag+'</span>.<br>'+
    (beta>=0.99?'<span class="hl3">Very smooth:</span> trend clear, but lags sharp changes by ~100 steps. Spikes become invisible.':
     beta>=0.9?'<span class="hl4">Good balance:</span> trend visible, spikes still detectable. Standard for monitoring.':
     beta<0.5?'<span class="hl5">No smoothing:</span> raw per-step loss. Useful for detecting individual spike events.':'<span class="hl2">Light smoothing:</span> lag small, still shows most fluctuation.');
}

function renderAnatomy(){
  var ctx=clr('cvAnatomy',860,155),W=860,H=155,PAD=50,pW=W-PAD-20,pH=H-PAD-25;
  axesPAD(ctx,W,H,PAD,'Epoch','Loss');
  var N=100;
  var trainPts=[],valPts=[];
  for(var i=0;i<=N;i++){
    var t=i/N,cx=PAD+t*pW;
    var warmup=t<0.05?2.5*(1-t/0.05*0.3):null;
    var train=t<0.05?2.5*(1-t/0.05*0.4):2.5*Math.exp(-(t-0.05)*4.5)+0.35;
    var val=t<0.05?2.6*(1-t/0.05*0.35):2.6*Math.exp(-(t-0.05)*4)+0.42+Math.max(0,t-0.7)*0.3;
    trainPts.push([cx,H-PAD-Math.min(train,2.5)/2.5*pH]);
    valPts.push([cx,H-PAD-Math.min(val,2.5)/2.5*pH]);
  }
  // Shade phases
  var p1x=PAD+0.05*pW,p2x=PAD+0.45*pW,p3x=PAD+0.75*pW;
  bx(ctx,PAD,15,p1x-PAD,H-PAD-15,0,'#c084fc10',null);
  bx(ctx,p1x,15,p2x-p1x,H-PAD-15,0,'#4ecdc410',null);
  bx(ctx,p2x,15,p3x-p2x,H-PAD-15,0,'#fbbf2410',null);
  bx(ctx,p3x,15,W-10-p3x,H-PAD-15,0,'#4ade8010',null);
  tx(ctx,'\u2460 Warmup',(PAD+p1x)/2,26,'#c084fc',8,'700');
  tx(ctx,'\u2461 Fast learning',(p1x+p2x)/2,26,'#4ecdc4',8,'700');
  tx(ctx,'\u2462 Gradual decay',(p2x+p3x)/2,26,'#fbbf24',8,'700');
  tx(ctx,'\u2463 Converged',(p3x+W-10)/2,26,'#4ade80',8,'700');
  crv(trainPts,ctx,'#4ecdc4',2.5);
  crv(valPts,ctx,'#ff6b35',2);
  tx(ctx,'Train',W-30,trainPts[N][1],'#4ecdc4',8,'700','right');
  tx(ctx,'Val',W-30,valPts[N][1]+12,'#ff6b35',8,'700','right');
  // Gen gap
  bx(ctx,p3x,valPts[90][1],W-20-p3x,trainPts[90][1]-valPts[90][1],0,'#fbbf2410',null);
  tx(ctx,'Gen gap',(p3x+W-20)/2,(trainPts[90][1]+valPts[90][1])/2,'#fbbf2460',7,'700');
}

// ============================================================
// TAB 1: DIVERGENCE
// ============================================================
var LR_VALUES=[1e-1,3e-2,1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5,1e-5];
function renderPhase(){
  var idx=Math.round(parseInt(document.getElementById('slLR').value)/100*( LR_VALUES.length-1));
  var lr=LR_VALUES[idx];
  document.getElementById('vLR').textContent=lr.toExponential(0);
  var ctx=clr('cvPhase',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Learning Rate (log scale)','Final val loss');
  // U-shaped curve: diverge left, good middle, slow right
  var lrMin=Math.log10(1e-6),lrMax=Math.log10(1.0);
  var optLR=Math.log10(1e-3);
  function perfFn(logLR){
    if(logLR>Math.log10(2e-2)) return 8; // diverge
    var d=logLR-optLR;
    if(d<0) return 0.5+Math.pow(-d*0.8,2.5)*3; // too slow
    return 0.5+Math.pow(d*1.2,2.0)*6; // diverging
  }
  var N=150,pts=[];
  for(var i=0;i<=N;i++){
    var logLR=lrMin+i/N*(lrMax-lrMin);
    var perf=perfFn(logLR);
    pts.push([PAD+(logLR-lrMin)/(lrMax-lrMin)*pW,H-PAD-Math.min(perf,8)/8*pH]);
  }
  // Shade zones
  var divX=PAD+((Math.log10(2e-2)-lrMin)/(lrMax-lrMin))*pW;
  var slowX=PAD+((Math.log10(3e-4)-lrMin)/(lrMax-lrMin))*pW;
  bx(ctx,divX,15,W-10-divX,H-PAD-15,0,'#ef444415',null);
  bx(ctx,slowX,15,divX-slowX,H-PAD-15,0,'#4ade8015',null);
  bx(ctx,PAD,15,slowX-PAD,H-PAD-15,0,'#4ecdc415',null);
  tx(ctx,'Too slow',( PAD+slowX)/2,30,'#4ecdc4',8,'700');
  tx(ctx,'Productive',(slowX+divX)/2,30,'#4ade80',8,'700');
  tx(ctx,'Diverges',(divX+W-10)/2,30,'#ef4444',8,'700');
  crv(pts,ctx,'#ff6b35',2.5);
  // Current LR marker
  var logLR0=Math.log10(lr);
  var curX=PAD+(logLR0-lrMin)/(lrMax-lrMin)*pW;
  var curPerf=perfFn(logLR0);
  var curY=H-PAD-Math.min(curPerf,8)/8*pH;
  ln(ctx,curX,15,curX,H-PAD,'#fbbf2470',1.5,[4,4]);
  ctx.fillStyle='#fbbf24';ctx.beginPath();ctx.arc(curX,curY,7,0,Math.PI*2);ctx.fill();
  tx(ctx,lr.toExponential(0),curX,curY-16,'#fbbf24',9,'800');
  // Tick labels
  [-6,-5,-4,-3,-2,-1,0].forEach(function(v){
    var px=PAD+(v-lrMin)/(lrMax-lrMin)*pW;
    tx(ctx,'1e'+v,px,H-PAD+14,'#3f3f46',6,'400');
  });
  var zone=logLR0>Math.log10(2e-2)?'<span class="hl5">DIVERGENCE ZONE:</span> LR far too large. Loss will explode.':
            logLR0>Math.log10(3e-4)?'<span class="hl4">PRODUCTIVE ZONE:</span> Loss decreases monotonically. This is the target region.':
            '<span class="hl2">TOO SLOW ZONE:</span> LR is so small that training will take forever. Convergence very slow.';
  document.getElementById('iPhase').innerHTML='LR = <span class="hl3">'+lr.toExponential(0)+'</span><br>'+zone+'<br>The stability boundary shifts left (lower max stable LR) for deeper, wider, or poorly conditioned models.';
}

var curShape='osc';
var SHAPES={
  osc:{col:'#ff6b35',fn:function(t){return 1.8-t*0.3+Math.sin(t*45)*0.75;},label:'Oscillating',cause:'LR too large. Steps overshoot minimum.',fix:'Reduce LR 2-10x. Add warmup. Add gradient clipping.'},
  spike:{col:'#fbbf24',fn:function(t,i){var b=1.8*Math.exp(-t*3)+0.5;if(i===28)b+=3.5;if(i===29)b+=1.8;return b;},label:'Single Spike',cause:'Corrupted batch or gradient explosion on rare example.',fix:'Add gradient clipping. Filter corrupted examples. Check individual batch losses.'},
  plat:{col:'#4ecdc4',fn:function(t){return 0.4+1.8*Math.exp(-t*8);},label:'Fast Plateau (Underfitting)',cause:'Model too small, LR too small, or training stopped too early.',fix:'Increase model capacity. Increase LR. Train longer.'},
  over:{col:'#c084fc',fn:function(t,i,isTrain){return isTrain?1.8*Math.exp(-t*4)+0.25:0.5+0.25*Math.exp(-t*3)+Math.max(0,t-0.35)*1.2;},label:'Overfitting (val diverges)',cause:'Model memorising training set (high variance).',fix:'Add regularisation (dropout, L2). More data. Early stopping.'},
  drift:{col:'#38bdf8',fn:function(t){return 0.6+t*0.8+Math.sin(t*12)*0.1;},label:'Gradual Drift Up',cause:'Distribution shift in data stream or catastrophic forgetting.',fix:'Add replay buffer. Check for distributional shift. LoRA for fine-tuning.'},
};
function selShape(k){curShape=k;['osc','spike','plat','over','drift'].forEach(function(b){document.getElementById('shBtn_'+b).classList.remove('active');});document.getElementById('shBtn_'+k).classList.add('active');renderShapes();}
function renderShapes(){
  var cfg=SHAPES[curShape];
  var ctx=clr('cvShapes',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Loss');
  resetSeed(17);
  var N=80,pts=[],valPts=[];
  var hasVal=curShape==='over';
  for(var i=0;i<=N;i++){
    var t=i/N,cx=PAD+t*pW;
    var loss=cfg.fn(t,i,true)+(rng()-0.5)*0.06;
    loss=Math.max(0.1,loss);
    pts.push([cx,H-PAD-Math.min(loss,5)/5*pH]);
    if(hasVal){
      var vl=cfg.fn(t,i,false)+(rng()-0.5)*0.06;
      valPts.push([cx,H-PAD-Math.min(Math.max(vl,0.1),5)/5*pH]);
    }
  }
  crv(pts,ctx,cfg.col,2.5);
  if(hasVal){crv(valPts,ctx,'#ef4444',2);tx(ctx,'Val',W-25,valPts[N][1],'#ef4444',8,'700','right');tx(ctx,'Train',W-25,pts[N][1]+12,cfg.col,8,'700','right');}
  tx(ctx,cfg.label,W/2,22,cfg.col,10,'800');
  document.getElementById('iShapes').innerHTML='<span class="hl">Cause:</span> '+cfg.cause+'<br><span class="hl4">Fix:</span> '+cfg.fix;
}

function renderDiag(){
  var ctx=clr('cvDiag',860,148),W=860,H=148;
  var steps=[
    {n:'1',label:'First step\nloss check',col:'#4ecdc4',sub:'== log(K)?'},
    {n:'2',label:'Check\nNaN inputs',col:'#4ecdc4',sub:'isnan(batch)?'},
    {n:'3',label:'Loss fn\nconfig',col:'#fbbf24',sub:'mean? shapes?'},
    {n:'4',label:'LR \u00D70.1\n(most common)',col:'#ff6b35',sub:'Fixed it?'},
    {n:'5',label:'Overfit\none batch',col:'#ff6b35',sub:'Converges?'},
    {n:'6',label:'Anomaly\ndetection',col:'#c084fc',sub:'set_detect\n_anomaly(True)'},
    {n:'7',label:'Grad norm\nbefore clip',col:'#ef4444',sub:'>>1.0?'},
  ];
  var bw=(W-20)/steps.length-6,sx=10;
  steps.forEach(function(st,i){
    var x=sx+i*(bw+6);
    bx(ctx,x,12,bw,H-24,6,st.col+'15',st.col+'60',1.5);
    tx(ctx,'Step '+st.n,x+bw/2,32,st.col,8,'800');
    st.label.split('\n').forEach(function(l,li){tx(ctx,l,x+bw/2,52+li*14,st.col,9,'800');});
    tx(ctx,st.sub,x+bw/2,H-20,'#52525b',7,'400');
    if(i<steps.length-1){tx(ctx,'\u2192',x+bw+3,H/2,'#3f3f46',12,'700');}
  });
  tx(ctx,'Work through in order. Stop at the step that reveals the problem.',W/2,H-4,'#52525b',8,'400');
}

// ============================================================
// TAB 2: PRECISION
// ============================================================
function renderPrecision(){
  var ctx=clr('cvPrecision',420,260),W=420,H=260;
  var formats=[
    {name:'FP32',bits:32,sign:1,exp:8,mant:23,range:'±3.4×10³⁸',col:'#4ade80',y:40},
    {name:'BF16',bits:16,sign:1,exp:8,mant:7, range:'±3.4×10³⁸',col:'#fbbf24',y:100},
    {name:'FP16',bits:16,sign:1,exp:5,mant:10,range:'±65,504',   col:'#ff6b35',y:160},
    {name:'FP8', bits:8, sign:1,exp:4,mant:3, range:'±448',      col:'#ef4444',y:210},
  ];
  var bh=38,totalBits=32,bw=(W-120)/totalBits;
  formats.forEach(function(f){
    var x=60;
    tx(ctx,f.name,40,f.y+bh/2,f.col,10,'800','right');
    // Sign bit
    bx(ctx,x,f.y,bw-1,bh,2,f.col+'50',f.col,1.5);
    tx(ctx,'S',x+bw/2,f.y+bh/2,f.col,7,'800');x+=bw;
    // Exponent bits
    for(var e=0;e<f.exp;e++){bx(ctx,x,f.y,bw-1,bh,1,'#4ecdc420','#4ecdc4',1);if(e===0)tx(ctx,'exp('+f.exp+')',x+f.exp*bw/2,f.y+bh/2,'#4ecdc4',7,'700');x+=bw;}
    // Mantissa bits
    for(var m=0;m<f.mant;m++){bx(ctx,x,f.y,bw-1,bh,1,'#c084fc20','#c084fc',1);if(m===0)tx(ctx,'mantissa('+f.mant+')',x+f.mant*bw/2,f.y+bh/2,'#c084fc',7,'700');x+=bw;}
    // Unused
    var used=1+f.exp+f.mant;
    for(var u=used;u<totalBits;u++){bx(ctx,x,f.y,bw-1,bh,1,'#141420','#1e1e2e',0.5);x+=bw;}
    tx(ctx,f.range,W-10,f.y+bh/2,f.col,8,'700','right');
  });
  tx(ctx,'Range',W-10,22,'#52525b',7,'400','right');
  document.getElementById('iPrecision').innerHTML=
    '<span class="hl3">Key insight:</span> BF16 has the same 8-bit exponent as FP32 \u2192 same dynamic range \u2192 NO overflow during training.<br>'+
    '<span class="hl5">FP16 danger:</span> only 5 exponent bits \u2192 range limited to \u00B165,504. Gradient values often exceed this \u2192 Inf overflow \u2192 NaN cascade.<br>'+
    '<span class="hl4">Recommendation:</span> Use BF16 on A100/H100/TPU. Use FP16+GradScaler on V100/T4.';
}

function renderLossScale(){
  var scaleIdx=parseInt(document.getElementById('slScale').value);
  var S=Math.round(Math.pow(2,scaleIdx/100*17)); // 1 to 131072
  document.getElementById('vScale').textContent=S>=1024?S.toLocaleString():S;
  var ctx=clr('cvLossScale',420,260),W=420,H=260,PAD=44,pW=W-PAD-20,pH=H-PAD-40;
  axesPAD(ctx,W,H,PAD,'Training step','Loss scale S');
  resetSeed(55);
  // Simulate dynamic loss scaling: S adjusts up/down
  var lsMax=131072,lsMin=1;
  var lsHistory=[65536];
  for(var i=1;i<=80;i++){
    var prev=lsHistory[i-1];
    // Occasionally overflow -> halve; mostly fine -> double every N steps
    var overflow=rng()<(S<1024?0.3:S<4096?0.15:0.05);
    var nxt;
    if(overflow) nxt=Math.max(lsMin,prev/2);
    else nxt=Math.min(lsMax,i%10===0?prev*2:prev);
    lsHistory.push(nxt);
  }
  var pts=lsHistory.map(function(v,i){return[PAD+i/80*pW,H-PAD-Math.log2(Math.max(v,1))/Math.log2(lsMax)*pH];});
  // Reference lines
  [1,64,1024,65536].forEach(function(v){var gy=H-PAD-Math.log2(v)/Math.log2(lsMax)*pH;ln(ctx,PAD,gy,W-10,gy,v===1?'#ef444440':'#2d2d4040',1,[3,3]);tx(ctx,v>=1024?v.toLocaleString():v,PAD-6,gy,'#3f3f46',7,'400','right');});
  // Shade danger zone
  var danger=H-PAD-Math.log2(64)/Math.log2(lsMax)*pH;
  bx(ctx,PAD,danger,pW,H-PAD-danger,0,'#ef444415',null);
  tx(ctx,'Danger zone (S\u226464)',PAD+pW/2,H-PAD-15,'#ef444450',8,'700');
  crv(pts,ctx,'#c084fc',2.5);
  tx(ctx,'Dynamic Loss Scale S (log scale)',W/2,22,'#c084fc',9,'800');
  // Current scale marker line
  var curY=H-PAD-Math.log2(Math.max(S,1))/Math.log2(lsMax)*pH;
  ln(ctx,PAD,curY,W-10,curY,'#fbbf2470',1.5,[4,4]);
  tx(ctx,'Current S='+S,W-12,curY-8,'#fbbf24',8,'700','right');
  var healthStr=S>=4096?'<span class="hl4">Healthy range.</span> Gradients fit in FP16 representable range.':S>=64?'<span class="hl3">Warning: S is falling.</span> Overflow happening frequently. Consider reducing LR.':'<span class="hl5">Critical: S\u226464.</span> Severe overflow. Training barely stable. Reduce LR immediately.';
  document.getElementById('iLossScale').innerHTML='Loss scale S=<span class="hl3">'+S.toLocaleString()+'</span><br>'+healthStr+'<br>Dynamic rule: S halves on any Inf/NaN gradient step; doubles every N clean steps.';
}

// ============================================================
// TAB 3: MONITORING
// ============================================================
var curGN='healthy';
var GN_DATA={
  healthy:{col:'#4ade80',fn:function(i,N){return 0.3+Math.sin(i/N*5)*0.15+(rng()-0.5)*0.1;},label:'Healthy gradient norms',info:'<span class="hl4">Healthy:</span> gradient norms are roughly stable, bounded between 0.1 and 1.0. No explosion risk. Training is stable.'},
  explode:{col:'#ef4444',fn:function(i,N){var t=i/N;return t<0.4?0.3+t*0.5+(rng()-0.5)*0.1:0.3+Math.exp((t-0.4)*8)*(rng()*2+1);},label:'Gradient explosion',info:'<span class="hl5">Exploding:</span> gradient norm grows exponentially. Loss cliff imminent. Add gradient clipping (max_norm=1.0) immediately. Reduce LR.'},
  vanish:{col:'#4ecdc4',fn:function(i,N){var t=i/N;return Math.max(0.0005,0.3*Math.exp(-t*4))+(rng()-0.5)*0.005;},label:'Vanishing gradients',info:'<span class="hl2">Vanishing:</span> gradient norm decays to near-zero. Early layers stop learning. Check residual connections, activation functions (GELU/ReLU), initialisation.'},
  clipped:{col:'#fbbf24',fn:function(i,N){var t=i/N,raw=0.3+Math.sin(i/N*6)*0.5+(rng()-0.5)*0.4;if(i>30&&i<45)raw+=3;return raw;},label:'Clipped (pre-clip norm)',info:'<span class="hl3">Clipping active:</span> pre-clip norm shown. Spikes at steps 30-45 would have caused explosion. Gradient clipping (max_norm=1.0) absorbs them. Normal operation.'},
};
function selGN(k){curGN=k;['healthy','explode','vanish','clipped'].forEach(function(b){document.getElementById('gnBtn_'+b).classList.remove('active');});document.getElementById('gnBtn_'+k).classList.add('active');renderGradNorm();}
function renderGradNorm(){
  var cfg=GN_DATA[curGN];
  resetSeed(33);
  var ctx=clr('cvGradNorm',420,260),W=420,H=260,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Gradient norm');
  var N=80,pts=[];
  var maxV=curGN==='explode'?200:curGN==='vanish'?0.4:5;
  for(var i=0;i<=N;i++){
    var v=cfg.fn(i,N);
    pts.push([PAD+i/N*pW,H-PAD-Math.min(v/maxV,1)*pH]);
  }
  // Safe zone band
  var safeTop=H-PAD-5/maxV*pH, safeBot=H-PAD-0.05/maxV*pH;
  if(safeBot>15&&safeTop<H-PAD){bx(ctx,PAD,Math.max(15,safeTop),pW,Math.min(H-PAD,safeBot)-Math.max(15,safeTop),0,'#4ade8010',null);tx(ctx,'Healthy range (0.05\u20135)',W-14,(Math.max(15,safeTop)+Math.min(H-PAD,safeBot))/2,'#4ade8030',7,'700','right');}
  // Clip line
  if(curGN==='clipped'){var clipY=H-PAD-1/maxV*pH;ln(ctx,PAD,clipY,W-10,clipY,'#fbbf2460',1.5,[4,4]);tx(ctx,'max_norm=1.0',W-14,clipY-8,'#fbbf24',8,'700','right');}
  crv(pts,ctx,cfg.col,2.5);
  tx(ctx,cfg.label,W/2,22,cfg.col,9,'800');
  document.getElementById('iGradNorm').innerHTML=cfg.info;
}

function renderUpdateRatio(){
  var lrIdx=parseInt(document.getElementById('slURlr').value);
  var lr=Math.pow(10,-5+lrIdx/100*4); // 1e-5 to 1e-1
  document.getElementById('vURlr').textContent=lr.toExponential(1);
  var ctx=clr('cvUpdateRatio',420,260),W=420,H=260,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Layer index','Update ratio (\u03B7\u00B7grad/weight)');
  var N=12,wNorm=0.3; // typical weight norm
  resetSeed(21);
  var pts=[];
  for(var i=0;i<N;i++){
    var gradNorm=0.2+rng()*0.3; // typical per-layer grad norm
    var ratio=lr*gradNorm/wNorm;
    var cx=PAD+i/(N-1)*pW;
    pts.push([cx,H-PAD-Math.min(ratio/0.1,1.1)*pH]);
  }
  // Reference bands
  var targetTop=H-PAD-0.01/0.1*pH,targetBot=H-PAD-0.0001/0.1*pH;
  bx(ctx,PAD,Math.max(15,targetTop),pW,Math.min(H-PAD,targetBot)-Math.max(15,targetTop),0,'#4ade8015',null);
  ln(ctx,PAD,H-PAD-0.01/0.1*pH,W-10,H-PAD-0.01/0.1*pH,'#ef444440',1,[3,3]);
  ln(ctx,PAD,H-PAD-0.001/0.1*pH,W-10,H-PAD-0.001/0.1*pH,'#4ade8040',1,[4,4]);
  ln(ctx,PAD,H-PAD-0.0001/0.1*pH,W-10,H-PAD-0.0001/0.1*pH,'#4ecdc440',1,[3,3]);
  tx(ctx,'1e-2 (too high)',W-14,H-PAD-0.01/0.1*pH-8,'#ef444460',7,'700','right');
  tx(ctx,'1e-3 (target)',W-14,H-PAD-0.001/0.1*pH-8,'#4ade8060',7,'700','right');
  tx(ctx,'1e-4 (too low)',W-14,H-PAD-0.0001/0.1*pH-8,'#4ecdc460',7,'700','right');
  crv(pts,ctx,'#c084fc',2.5);
  pts.forEach(function(p){ctx.fillStyle='#c084fc';ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fill();});
  var meanRatio=lr*0.3/wNorm;
  var ratioStr=meanRatio>0.01?'<span class="hl5">TOO HIGH (&gt;1e-2):</span> updates are too large relative to weight magnitude. Instability risk. Reduce LR.':meanRatio<0.0001?'<span class="hl2">TOO LOW (&lt;1e-4):</span> updates tiny. Learning has stalled. Increase LR.':'<span class="hl4">HEALTHY (1e-4\u20131e-2):</span> weights change ~'+Math.round(meanRatio*1000)/10+'% per step. Training is stable.';
  document.getElementById('iUpdateRatio').innerHTML=
    'LR=<span class="hl3">'+lr.toExponential(1)+'</span>. Mean update ratio: <span class="hl3">'+meanRatio.toExponential(1)+'</span><br>'+ratioStr+'<br>Proposed by Andrej Karpathy: log_ratio \u2248 -3 (1e-3) is the target.';
}

// ============================================================
// TAB 4: REPRODUCIBILITY
// ============================================================
function renderSeeds(){
  var ns=parseInt(document.getElementById('slSeeds').value);
  document.getElementById('vSeeds').textContent=ns;
  var ctx=clr('cvSeeds',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Val accuracy (%)');
  var N=60,baseAcc=84.0,colors=['#4ecdc4','#ff6b35','#fbbf24','#c084fc','#4ade80','#38bdf8','#f472b6'];
  var finalAccs=[];
  for(var s=0;s<ns;s++){
    resetSeed(s*137+1);
    var offset=(rng()-0.5)*0.8;
    var pts=[],acc=0;
    for(var i=0;i<=N;i++){
      var t=i/N;
      acc=baseAcc+offset+(84.5-baseAcc)*Math.pow(t,0.4)+(rng()-0.5)*0.4;
      acc=Math.min(acc,baseAcc+offset+0.8);
      pts.push([PAD+t*pW,H-PAD-(acc-80)/8*pH]);
    }
    crv(pts,ctx,colors[s],1.5);
    finalAccs.push(acc);
  }
  // Mean and std
  var mean=finalAccs.reduce(function(a,b){return a+b;},0)/finalAccs.length;
  var std=Math.sqrt(finalAccs.map(function(x){return Math.pow(x-mean,2);}).reduce(function(a,b){return a+b;},0)/finalAccs.length);
  // Mean line
  var meanY=H-PAD-(mean-80)/8*pH;
  ln(ctx,PAD+pW*0.8,meanY,W-10,meanY,'#ffffff60',2);
  tx(ctx,'mean='+rnd(mean,1)+'%',W-12,meanY-8,'#e4e4e7',8,'800','right');
  // Std band
  bx(ctx,PAD+pW*0.8,meanY+std/8*pH,pW*0.2,(std*2)/8*pH,2,'#ffffff10',null);
  tx(ctx,'\u00B1'+rnd(std,2)+'%',W-12,meanY+std/8*pH+4,'#71717a',7,'400','right');
  tx(ctx,ns+' seed runs — variance band',W/2,22,'#fbbf24',9,'800');
  var sigStr=std>0.5?'<span class="hl5">High variance (std='+rnd(std,2)+'%).</span> Differences &lt;'+rnd(std*1.5,1)+'% cannot be claimed with confidence.':'<span class="hl4">Low variance (std='+rnd(std,2)+'%).</span> Results are reproducible. Differences &gt;'+rnd(std*1.5,1)+'% are meaningful.';
  document.getElementById('iSeeds').innerHTML=
    'Mean: <span class="hl4">'+rnd(mean,1)+'%</span> \u00B1 <span class="hl3">'+rnd(std,2)+'%</span> across '+ns+' seeds.<br>'+sigStr+'<br><span class="hl">Always report mean\u00B1std, never cherry-pick the best seed.</span>';
}

function renderRepro(){
  var ctx=clr('cvRepro',420,270),W=420,H=270;
  var sources=[
    {label:'Weight init',fix:'torch.manual_seed()',col:'#4ade80',severity:'Expected variance'},
    {label:'Data order',fix:'Seed sampler',col:'#4ade80',severity:'Fixable'},
    {label:'Dropout/aug',fix:'random + np.random seed',col:'#fbbf24',severity:'Fixable'},
    {label:'CUDA ops',fix:'cudnn.deterministic=True',col:'#ff6b35',severity:'10-30% slow'},
    {label:'Workers RNG',fix:'worker_init_fn',col:'#fbbf24',severity:'Fixable'},
    {label:'Multi-GPU order',fix:'NCCL deterministic',col:'#ef4444',severity:'Complex'},
  ];
  var bh=30,startY=28;
  sources.forEach(function(src,i){
    var y=startY+i*(bh+8);
    bx(ctx,14,y,W-28,bh,6,src.col+'15',src.col+'50',1.5);
    tx(ctx,src.label,80,y+bh/2,src.col,9,'800','left');
    tx(ctx,'\u2192 '+src.fix,250,y+bh/2,'#94a3b8',8,'400');
    bx(ctx,W-130,y+5,115,20,4,src.col+'15',src.col+'40',1);
    tx(ctx,src.severity,W-73,y+15,src.col,7,'700');
  });
  document.getElementById('iRepro').innerHTML=
    '<span class="hl5">Weight init variance</span> is expected and should be reported as mean\u00B1std.<br>'+
    '<span class="hl3">CUDA non-determinism</span> is real: even with the same seed, GPU parallel ops can produce different floating-point sums. Use cudnn.deterministic=True for exact reproducibility at the cost of speed.';
}

// ============================================================
// TAB 5: DEBUG PROTOCOL
// ============================================================
function renderProtocol(){
  var ctx=clr('cvProtocol',860,175),W=860,H=175;
  var phases=[
    {label:'Phase 1\nMAKE IT WORK',col:'#ef4444',x:10,w:260,
     steps:['Overfit 1 batch to near-zero loss','If fails: model/loss/data bug','Fix BEFORE anything else']},
    {label:'Phase 2\nMAKE IT RIGHT',col:'#fbbf24',x:290,w:270,
     steps:['Train on 100-1K examples','Train loss should drop','Val loss should follow']},
    {label:'Phase 3\nMAKE IT FAST',col:'#4ade80',x:580,w:270,
     steps:['Only after phases 1&2 pass','Add mixed precision, grad accum','Profile GPU utilisation']},
  ];
  phases.forEach(function(ph){
    bx(ctx,ph.x,10,ph.w,H-20,8,ph.col+'15',ph.col,2);
    ph.label.split('\n').forEach(function(l,li){tx(ctx,l,ph.x+ph.w/2,32+li*14,ph.col,9,'800');});
    ph.steps.forEach(function(s,si){tx(ctx,'\u2022 '+s,ph.x+14,72+si*24,'#94a3b8',8,'400','left');});
    if(ph.x>10){tx(ctx,'\u2192',ph.x-15,H/2,'#3f3f46',18,'400');}
  });
}

function renderSanity(){
  var ctx=clr('cvSanity',420,260),W=420,H=260;
  var checks=[
    {phase:'Before',col:'#4ecdc4',items:['Initial loss \u2248 log(K)',
     'Print output+target shapes','Visualise first batch','Check model.train()/eval()']},
    {phase:'First steps',col:'#fbbf24',items:['Overfit 1 batch test',
     'Check grad norms backward()','Loss decreasing after 10 steps','set_detect_anomaly(True) if NaN']},
    {phase:'Ongoing',col:'#ff6b35',items:['Log grad norm every step',
     'Log activation stats \u00D7500','Log update ratio \u00D7500','Val loss every epoch']},
  ];
  var bh=H/3-8;
  checks.forEach(function(ch,i){
    var y=4+i*(bh+4);
    bx(ctx,8,y,W-16,bh,6,ch.col+'15',ch.col+'50',1.5);
    tx(ctx,ch.phase,50,y+16,ch.col,8,'800');
    ch.items.forEach(function(item,ii){tx(ctx,'\u2713 '+item,100,y+14+ii*16,'#94a3b8',7,'400','left');});
  });
  document.getElementById('iSanity').innerHTML=
    '<span class="hl5">Most common pre-training bugs</span> are caught by checking the initial loss and visualising the first batch.<br>'+
    '<span class="hl">The overfit test</span> (Phase 1) is the single highest-leverage debugging step. It costs 5 minutes and catches 80% of bugs.';
}

var TASKS=[
  {name:'Binary clf',k:2,loss:Math.log(2),col:'#4ecdc4',note:'log(2) \u2248 0.693'},
  {name:'CIFAR-10',k:10,loss:Math.log(10),col:'#ff6b35',note:'log(10) \u2248 2.303'},
  {name:'ImageNet',k:1000,loss:Math.log(1000),col:'#fbbf24',note:'log(1000) \u2248 6.908'},
  {name:'Regression',k:0,loss:NaN,col:'#c084fc',note:'Expected \u2248 Var(y)'},
  {name:'LM (GPT)',k:50000,loss:Math.log(50000),col:'#4ade80',note:'log(50000) \u2248 10.8'},
];
function renderBaseline(){
  var idx=parseInt(document.getElementById('slTask').value);
  var task=TASKS[idx];
  document.getElementById('vTask').textContent=task.name;
  var ctx=clr('cvBaseline',420,260),W=420,H=260,PAD=54,pW=W-PAD-20,pH=H-PAD-40;
  axesPAD(ctx,W,H,PAD,'Training step','Loss');
  // Show expected baseline as a horizontal line
  var maxL=Math.max(task.k>0?task.loss*1.4:5,3);
  if(task.k>0){
    var baseY=H-PAD-task.loss/maxL*pH;
    ln(ctx,PAD,baseY,W-10,baseY,task.col,2,[4,4]);
    bx(ctx,PAD,baseY-1,pW,2,0,task.col+'40',null);
    tx(ctx,'Expected initial: '+rnd(task.loss,3),W-14,baseY-12,task.col,8,'800','right');
  }
  // Simulate realistic loss curve starting at baseline
  resetSeed(88);
  var N=60,pts=[];
  for(var i=0;i<=N;i++){
    var t=i/N;
    var loss=task.k>0?(task.loss*(1.02+0.1*rng())*Math.exp(-t*3)+0.4*(task.loss/Math.log(10))):3+rng()*0.1;
    pts.push([PAD+t*pW,H-PAD-Math.min(loss/maxL,1.1)*pH]);
  }
  crv(pts,ctx,task.col,2.5);
  // Baseline label
  if(task.k>0){
    var startLoss=task.loss*1.05;
    var startY=H-PAD-startLoss/maxL*pH;
    ctx.fillStyle=task.col;ctx.beginPath();ctx.arc(PAD,startY,6,0,Math.PI*2);ctx.fill();
    tx(ctx,'Step 0: '+rnd(startLoss,2),PAD+50,startY-12,task.col,9,'800');
  }
  // Good/bad zone annotations
  var badLow=H-PAD-task.loss*0.5/maxL*pH;
  var badHigh=H-PAD-task.loss*1.5/maxL*pH;
  if(task.k>0){
    bx(ctx,PAD+10,badHigh,60,badLow-badHigh,3,'#ef444420',null);
    tx(ctx,'Bug zone',PAD+40,(badHigh+badLow)/2,'#ef4444',7,'700');
  }
  tx(ctx,task.name,W/2,22,task.col,10,'800');
  document.getElementById('iBaseline').innerHTML=
    '<span class="hl3">'+task.name+'</span>: expected initial loss = <span class="hl">'+( task.k>0?rnd(task.loss,3)+' = log('+task.k+')':task.note)+'</span><br>'+
    (task.k>0?'<span class="hl5">If initial loss << '+rnd(task.loss,1)+':</span> information leak or non-random init.<br><span class="hl5">If initial loss >> '+rnd(task.loss,1)+':</span> loss function bug or label mismatch.<br><span class="hl5">If initial loss = NaN:</span> init problem or NaN in data.':'<span class="hl5">If initial loss >> Var(y):</span> check loss function and output scale.')+
    '<br><span class="hl4">This check costs one forward pass and catches most setup bugs.</span>';
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){
  renderSpectrum(); renderSmooth(); renderAnatomy();
});
</script>
</body>
</html>"""

TRAINING_STABILITY_VISUAL_HEIGHT = 2200