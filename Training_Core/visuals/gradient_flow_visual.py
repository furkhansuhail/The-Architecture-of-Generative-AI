GRADIENT_FLOW_VISUAL_HTML = r"""<!DOCTYPE html>
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
<h2>&#x26A1; Gradient Flow Issues</h2>
<p class="sub">Vanishing &middot; Exploding &middot; Dead Neurons &middot; Clipping &middot; Residual &amp; Init &middot; Diagnosis</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Gradient Flow</button>
  <button class="tab" onclick="showTab(1)">Vanishing</button>
  <button class="tab" onclick="showTab(2)">Exploding</button>
  <button class="tab" onclick="showTab(3)">Dead Neurons</button>
  <button class="tab" onclick="showTab(4)">Clipping</button>
  <button class="tab" onclick="showTab(5)">Solutions</button>
</div>

<!-- TAB 0: GRADIENT FLOW -->
<div id="tab0" class="panel active">
<div class="g2">
<div class="card">
  <h3>&#9312; Gradient Magnitude Through Layers</h3>
  <canvas id="cvFlow" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="fBtn_vanish"  onclick="selFlow('vanish')">Vanishing</button>
    <button class="s" id="fBtn_stable"  onclick="selFlow('stable')">Stable</button>
    <button class="s" id="fBtn_explode" onclick="selFlow('explode')">Exploding</button>
  </div>
  <div class="row"><label>Num layers</label>
    <input type="range" id="slDepth" min="5" max="50" step="1" value="20" oninput="renderFlow()">
    <span class="v" id="vDepth">20</span></div>
  <div class="info" id="iFlow">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; The Chain Rule Product</h3>
  <canvas id="cvChain" width="420" height="270"></canvas>
  <div class="row"><label>Factor per layer</label>
    <input type="range" id="slFactor" min="0.5" max="1.5" step="0.01" value="0.9" oninput="renderChain()">
    <span class="v" id="vFactor">0.90</span></div>
  <div class="info" id="iChain">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Why Early Layers Are Hit Hardest</h3>
  <canvas id="cvDepthEffect" width="860" height="160"></canvas>
</div>
</div>

<!-- TAB 1: VANISHING -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Sigmoid Saturation &amp; Its Gradient</h3>
  <canvas id="cvSigGrad" width="420" height="270"></canvas>
  <div class="row"><label>Input z</label>
    <input type="range" id="slSigZ" min="-5" max="5" step="0.1" value="0" oninput="renderSigGrad()">
    <span class="v" id="vSigZ">0.00</span></div>
  <div class="info" id="iSigGrad">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Vanishing Through L Sigmoid Layers</h3>
  <canvas id="cvVanish" width="420" height="270"></canvas>
  <div class="row"><label>Num layers L</label>
    <input type="range" id="slVL" min="1" max="20" step="1" value="10" oninput="renderVanish()">
    <span class="v" id="vVL">10</span></div>
  <div class="info" id="iVanish">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Activation Gradient Comparison: Sigmoid vs Tanh vs ReLU vs GELU</h3>
  <canvas id="cvActGrad" width="860" height="180"></canvas>
</div>
</div>

<!-- TAB 2: EXPLODING -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Loss Cliff &mdash; Gradient Explosion Signature</h3>
  <canvas id="cvCliff" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="cBtn_no"  onclick="selCliff('no')">No Clipping</button>
    <button class="s" id="cBtn_yes" onclick="selCliff('yes')">With Clipping</button>
  </div>
  <div class="info" id="iCliff">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Eigenvalue Growth Through Layers</h3>
  <canvas id="cvEigen" width="420" height="270"></canvas>
  <div class="row"><label>Eigenvalue</label>
    <input type="range" id="slEig" min="0.7" max="1.4" step="0.01" value="1.2" oninput="renderEigen()">
    <span class="v" id="vEig">1.20</span></div>
  <div class="info" id="iEigen">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; How NaN Propagates (Gradient Explosion Cascade)</h3>
  <canvas id="cvNaN" width="860" height="140"></canvas>
</div>
</div>

<!-- TAB 3: DEAD NEURONS -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Dying ReLU &mdash; Neuron State Visualiser</h3>
  <canvas id="cvDead" width="420" height="270"></canvas>
  <div class="row"><label>Dead fraction</label>
    <input type="range" id="slDead" min="0" max="0.9" step="0.05" value="0.3" oninput="renderDead()">
    <span class="v" id="vDead">30%</span></div>
  <div class="info" id="iDead">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Activation Comparison: ReLU vs Leaky vs GELU</h3>
  <canvas id="cvActComp" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="acBtn_fn"   onclick="selAC('fn')">f(z)</button>
    <button class="s" id="acBtn_grad" onclick="selAC('grad')">f'(z) gradient</button>
  </div>
  <div class="info" id="iActComp">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Dead Neuron Solutions</h3>
  <table class="tbl">
    <thead><tr><th>Activation</th><th>f(z) for z&lt;0</th><th>Grad for z&lt;0</th><th>Can die?</th><th>Best for</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">ReLU</span></td><td>0 (always)</td><td><span class="hl5">0 (always)</span></td><td><span class="hl5">Yes &mdash; permanently</span></td><td>CNNs, fast training, pre-2018</td></tr>
      <tr><td><span class="hl3">Leaky ReLU (&alpha;=0.01)</span></td><td>0.01z (small)</td><td><span class="hl4">0.01 (never zero)</span></td><td><span class="hl4">No &mdash; can always recover</span></td><td>MLPs, simple ReLU replacement</td></tr>
      <tr><td><span class="hl2">ELU (&alpha;=1)</span></td><td>&alpha;(e^z&minus;1)</td><td><span class="hl4">f(z)+&alpha; (smooth)</span></td><td><span class="hl4">No &mdash; zero-centred</span></td><td>Fast conv nets, mean activation near 0</td></tr>
      <tr><td><span class="hl6">GELU</span></td><td>z&sdot;&Phi;(z) (small neg.)</td><td><span class="hl4">Near-zero but never 0</span></td><td><span class="hl3">Rare &mdash; practical no</span></td><td>All transformers: BERT, GPT, ViT</td></tr>
      <tr><td><span class="hl">Swish/SiLU</span></td><td>z&sdot;&sigma;(z) (small neg.)</td><td><span class="hl4">Non-zero everywhere</span></td><td><span class="hl3">Rare &mdash; practical no</span></td><td>EfficientNet, mobile/edge models</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 4: CLIPPING -->
<div id="tab4" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Clip-by-Norm vs Clip-by-Value</h3>
  <canvas id="cvClip" width="420" height="290"></canvas>
  <div class="row"><label>Max norm</label>
    <input type="range" id="slClipN" min="1" max="10" step="0.5" value="5" oninput="renderClip()">
    <span class="v" id="vClipN">5.0</span></div>
  <div class="row"><label>Clip value</label>
    <input type="range" id="slClipV" min="1" max="10" step="0.5" value="5" oninput="renderClip()">
    <span class="v" id="vClipV">5.0</span></div>
  <div class="info" id="iClip">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Gradient Norm Monitoring Over Training</h3>
  <canvas id="cvNorm" width="420" height="290"></canvas>
  <div class="row"><label>Max norm</label>
    <input type="range" id="slMaxN" min="0.5" max="5" step="0.5" value="1" oninput="renderNorm()">
    <span class="v" id="vMaxN">1.0</span></div>
  <div class="info" id="iNorm">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Clipping Threshold Guide</h3>
  <table class="tbl">
    <thead><tr><th>Architecture</th><th>Recommended max_norm</th><th>Clipping frequency target</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td><span class="hl6">Transformers (NLP/LLM)</span></td><td><span class="hl4">1.0</span></td><td>&lt;1% of steps</td><td>Universal standard for GPT/BERT/LLaMA. Always use.</td></tr>
      <tr><td><span class="hl2">RNNs / LSTMs</span></td><td><span class="hl4">1.0 &ndash; 5.0</span></td><td>&lt;5% of steps</td><td>Essential &mdash; sequence length amplifies explosion.</td></tr>
      <tr><td><span class="hl3">CNNs</span></td><td><span class="hl4">1.0 &ndash; 10.0</span></td><td>Rare</td><td>Less critical with BatchNorm but still good practice.</td></tr>
      <tr><td><span class="hl">RL training</span></td><td><span class="hl4">0.5 &ndash; 1.0</span></td><td>&lt;5% of steps</td><td>Reward signals can be highly non-stationary.</td></tr>
      <tr><td colspan="4"><span class="hl5">Warning:</span> if clipping on &gt;20% of steps, clipping is masking instability &mdash; reduce LR or fix initialisation instead.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 5: SOLUTIONS -->
<div id="tab5" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Residual Connection: Gradient Highway</h3>
  <canvas id="cvRes" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="rBtn_plain"  onclick="selRes('plain')">Plain Network</button>
    <button class="s" id="rBtn_res"    onclick="selRes('res')">Residual Network</button>
  </div>
  <div class="info" id="iRes">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Weight Init: Activation Variance Across Depth</h3>
  <canvas id="cvInit" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="initBtn_he"    onclick="selInit('he')">He (ReLU)</button>
    <button class="s" id="initBtn_xavier" onclick="selInit('xavier')">Xavier</button>
    <button class="s" id="initBtn_small"  onclick="selInit('small')">Too Small</button>
    <button class="s" id="initBtn_large"  onclick="selInit('large')">Too Large</button>
  </div>
  <div class="info" id="iInit">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Diagnosis Table &mdash; Symptom &rarr; Cause &rarr; Fix</h3>
  <table class="tbl">
    <thead><tr><th>Symptom</th><th>Cause</th><th>Fix</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">Loss doesn't decrease despite long training</span></td><td>Vanishing in early layers</td><td>ReLU/GELU activation, residual connections, He init, LayerNorm</td></tr>
      <tr><td><span class="hl5">Early layer grad norms near zero</span></td><td>Vanishing gradients</td><td>Residual connections, proper init, reduce depth, switch to GELU</td></tr>
      <tr><td><span class="hl5">Loss spike / cliff</span></td><td>Exploding gradients</td><td>Gradient clipping (max_norm=1.0), reduce learning rate</td></tr>
      <tr><td><span class="hl5">Loss becomes NaN</span></td><td>Explosion &rarr; Inf weights</td><td>Gradient clipping, reduce LR, check for NaN in input data</td></tr>
      <tr><td><span class="hl5">&gt;40% zero activations in ReLU layers</span></td><td>Dying ReLU</td><td>Switch to Leaky ReLU or GELU, reduce LR, He initialisation</td></tr>
      <tr><td><span class="hl5">RNN fails on long sequences</span></td><td>Vanishing through time</td><td>Use LSTM or GRU, or transformer (attention = O(1) path)</td></tr>
      <tr><td><span class="hl5">Grad/weight ratio &lt;1e-4</span></td><td>LR too low or vanishing</td><td>Increase LR, check per-layer grad norms for vanishing</td></tr>
      <tr><td><span class="hl5">Grad/weight ratio &gt;1e-1</span></td><td>LR too high or explosion starting</td><td>Decrease LR, add gradient clipping</td></tr>
    </tbody>
  </table>
</div>
</div>

<script>
// ============================================================
// UTILS
// ============================================================
function rnd(v,d){return isFinite(v)?v.toFixed(d===undefined?3:d):'Inf';}
function clr(id,W,H){var cv=document.getElementById(id),ctx=cv.getContext('2d');ctx.fillStyle='#09090f';ctx.fillRect(0,0,W,H);return ctx;}
function tx(ctx,s,x,y,col,sz,wt,al){ctx.fillStyle=col||'#e4e4e7';ctx.font=(wt||'500')+' '+(sz||11)+'px JetBrains Mono,monospace';ctx.textAlign=al||'center';ctx.textBaseline='middle';ctx.fillText(s,x,y);}
function ln(ctx,x1,y1,x2,y2,col,lw,dash){ctx.strokeStyle=col;ctx.lineWidth=lw||1.5;if(dash)ctx.setLineDash(dash);else ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);}
function curve(pts,ctx,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||2.5;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}
function bx(ctx,x,y,w,h,r,fill,stroke,sw){ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,w,h,r||4);else ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw||1.5;ctx.stroke();}}
function axesPAD(ctx,W,H,PAD,xl,yl){ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);if(xl)tx(ctx,xl,(PAD+W-10)/2,H-5,'#52525b',8,'400');if(yl)tx(ctx,yl,PAD+20,22,'#52525b',8,'400');}

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderFlow();renderChain();renderDepthEffect();}if(i===1){renderSigGrad();renderVanish();renderActGrad();}if(i===2){renderCliff();renderEigen();renderNaN();}if(i===3){renderDead();renderActComp();}if(i===4){renderClip();renderNorm();}if(i===5){renderRes();renderInit();}}

// ============================================================
// TAB 0: GRADIENT FLOW
// ============================================================
var curFlow='vanish';
function selFlow(k){curFlow=k;['vanish','stable','explode'].forEach(function(b){document.getElementById('fBtn_'+b).classList.remove('active');});document.getElementById('fBtn_'+k).classList.add('active');renderFlow();}
function renderFlow(){
  var L=parseInt(document.getElementById('slDepth').value);
  document.getElementById('vDepth').textContent=L;
  var ctx=clr('cvFlow',420,270),W=420,H=270,PAD=50,pW=W-PAD-20,pH=H-PAD-30;
  // Log scale: y = H-PAD - log10(grad_norm)*pH/logRange
  var logMin=-12, logMax=12, logRange=24;
  axesPAD(ctx,W,H,PAD,'Layer','log|gradient|');
  // Grid and y labels
  for(var lv=-10;lv<=10;lv+=5){var gy=H-PAD-(lv-logMin)/logRange*pH;if(gy>15&&gy<H-PAD){ln(ctx,PAD,gy,W-10,gy,'#141420',1);tx(ctx,lv,PAD-6,gy,'#3f3f46',7,'400','right');}}
  // Zero line
  var zeroY=H-PAD-(-logMin)/logRange*pH;
  ln(ctx,PAD,zeroY,W-10,zeroY,'#2d2d4090',1.5,[4,4]);
  tx(ctx,'0 (ideal)',W-14,zeroY-8,'#4ade8050',8,'700','right');

  var configs={
    vanish:{factor:0.75,col:'#4ecdc4',label:'Vanishing (0.75^L)'},
    stable:{factor:1.0,col:'#4ade80',label:'Stable (1.0^L)'},
    explode:{factor:1.3,col:'#ef4444',label:'Exploding (1.3^L)'},
  };
  var cfg=configs[curFlow];
  var pts=[];
  for(var l=1;l<=L;l++){
    var grad=Math.pow(cfg.factor,l);
    var logGrad=Math.log10(Math.max(1e-12,grad));
    var cx=PAD+(l-1)/(L-1)*pW;
    var cy=H-PAD-(logGrad-logMin)/logRange*pH;
    pts.push([cx,Math.max(16,Math.min(H-PAD,cy))]);
  }
  // Shaded area under/above zero
  if(curFlow==='vanish'){ctx.fillStyle='#4ecdc415';ctx.beginPath();ctx.moveTo(pts[0][0],zeroY);pts.forEach(function(p){ctx.lineTo(p[0],p[1]);});ctx.lineTo(pts[pts.length-1][0],zeroY);ctx.closePath();ctx.fill();}
  if(curFlow==='explode'){ctx.fillStyle='#ef444415';ctx.beginPath();ctx.moveTo(pts[0][0],zeroY);pts.forEach(function(p){ctx.lineTo(p[0],p[1]);});ctx.lineTo(pts[pts.length-1][0],zeroY);ctx.closePath();ctx.fill();}
  curve(pts,ctx,cfg.col,2.5);
  // Dots every 5 layers
  for(var l2=0;l2<pts.length;l2+=5){ctx.fillStyle=cfg.col;ctx.beginPath();ctx.arc(pts[l2][0],pts[l2][1],4,0,Math.PI*2);ctx.fill();}
  tx(ctx,cfg.label,PAD+pW/2,24,cfg.col,9,'800');

  var finalGrad=Math.pow(cfg.factor,L);
  var logFinal=Math.log10(Math.max(1e-12,finalGrad));
  var infoMap={
    vanish:'<span class="hl2">Vanishing:</span> factor=0.75 per layer. After '+L+' layers: 0.75^'+L+' \u2248 <span class="hl5">'+rnd(finalGrad,6)+'</span><br>Early layers receive near-zero gradient. Their weights barely update. Network effectively reduces to only later layers learning.',
    stable:'<span class="hl4">Stable flow:</span> factor=1.0 per layer. Gradient magnitude=<span class="hl4">1.0</span> at every depth.<br>This is the target. He initialisation + ReLU + residual connections all aim to maintain this.',
    explode:'<span class="hl5">Exploding:</span> factor=1.3 per layer. After '+L+' layers: 1.3^'+L+' \u2248 <span class="hl5">'+rnd(Math.min(finalGrad,1e15),0)+'</span><br>Weights jump to completely different region of the loss landscape. Loss spikes. Possible NaN cascade.'
  };
  document.getElementById('iFlow').innerHTML=infoMap[curFlow];
}

function renderChain(){
  var factor=parseFloat(document.getElementById('slFactor').value);
  document.getElementById('vFactor').textContent=rnd(factor,2);
  var ctx=clr('cvChain',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  var layers=[5,10,20,30,50];
  var maxV=Math.max.apply(null,layers.map(function(L){return Math.min(Math.pow(factor,L),1e6);}));
  axesPAD(ctx,W,H,PAD,'Depth (layers)','Gradient magnitude (factor^L)');
  ln(ctx,PAD,H-PAD-pH,W-10,H-PAD-pH,'#4ade8020',1,[4,4]);
  tx(ctx,'magnitude=1 (ideal)',W-14,H-PAD-pH-8,'#4ade8040',7,'700','right');

  var col=factor<1?'#4ecdc4':factor>1?'#ef4444':'#4ade80';
  var pts=[];
  for(var i=0;i<=60;i++){
    var L2=i;
    var v=Math.pow(factor,L2);
    var cx=PAD+L2/60*pW;
    var cy=H-PAD-Math.min(v/Math.max(maxV,1),1)*pH;
    pts.push([cx,cy]);
  }
  curve(pts,ctx,col,2.5);

  // Annotate specific depths
  layers.forEach(function(L3){
    var v=Math.pow(factor,L3);
    var cx=PAD+L3/60*pW;
    var cy=H-PAD-Math.min(v/Math.max(maxV,1),1)*pH;
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(cx,Math.max(20,Math.min(H-PAD,cy)),5,0,Math.PI*2); ctx.fill();
    tx(ctx,'L='+L3,cx,Math.max(32,Math.min(H-PAD-6,cy))-14,col,8,'700');
    tx(ctx,rnd(Math.min(v,1e9),v<0.001?6:v<1?3:0),cx,Math.max(32,Math.min(H-PAD-6,cy))-2,'#fbbf24',7,'700');
  });

  var eff=factor<1?'Vanishing':factor>1?'Exploding':'Stable';
  var col2=factor<1?'#4ecdc4':factor>1?'#ef4444':'#4ade80';
  tx(ctx,eff+' ('+rnd(factor,2)+'^L)',W/2,22,col2,10,'800');
  document.getElementById('iChain').innerHTML=
    'Per-layer factor = <span class="hl3">'+rnd(factor,2)+'</span><br>'+
    'Gradient = factor^L &mdash; a product of L identical terms.<br>'+
    [5,10,20,50].map(function(L4){return 'L='+L4+': '+rnd(factor,2)+'^'+L4+' \u2248 <span style="color:'+(Math.pow(factor,L4)<0.01?'#ef4444':Math.pow(factor,L4)>100?'#ef4444':'#4ade80')+'">'+rnd(Math.min(Math.pow(factor,L4),1e9),Math.pow(factor,L4)<0.001?6:2)+'</span>';}).join('&nbsp; | &nbsp;');
}

function renderDepthEffect(){
  var ctx=clr('cvDepthEffect',860,160),W=860,H=160;
  var L=20, factor=0.85;
  var barH=30,startY=45,PAD=100;
  var bw=(W-PAD-20)/L;
  for(var l=1;l<=L;l++){
    var grad=Math.pow(factor,L-l+1); // layer 1 travels through most, layer L through fewest
    var x=PAD+(l-1)*bw+2;
    var ratio=Math.min(grad,1);
    var r=Math.round(74+ratio*(239-74)),g=Math.round(222+ratio*(68-222)),b2=Math.round(128+ratio*(68-128));
    var col='rgb('+r+','+g+','+b2+')';
    bx(ctx,x,startY,bw-4,barH,3,col+'30',col,1.5);
    if(bw>24)tx(ctx,'L'+l,x+(bw-4)/2,startY+barH+14,'#3f3f46',7,'400');
    if(l===1||l===L||l===5||l===10||l===15){tx(ctx,rnd(grad,4),x+(bw-4)/2,startY+barH/2,col,7,'700');}
  }
  tx(ctx,'\u2190 EARLY LAYERS (gradient travels through most multiplications)',PAD+L*bw*0.25,22,'#ef4444',8,'700');
  tx(ctx,'LATE LAYERS \u2192',PAD+L*bw*0.8,22,'#4ade80',8,'700');
  tx(ctx,'Layer 1 gradient: '+rnd(Math.pow(factor,L),6),PAD+bw/2,100,'#ef4444',8,'700');
  tx(ctx,'Layer '+L+' gradient: '+rnd(factor,2),PAD+(L-1)*bw+bw/2,100,'#4ade80',8,'700');
  tx(ctx,'Both layers use same activation & weights. Layer 1 gradient travels through '+L+' multiplications.',W/2,140,'#52525b',8,'400');
}

// ============================================================
// TAB 1: VANISHING
// ============================================================
function renderSigGrad(){
  var z=parseFloat(document.getElementById('slSigZ').value);
  document.getElementById('vSigZ').textContent=rnd(z,2);
  var ctx=clr('cvSigGrad',420,270),W=420,H=270,PAD=44,pH=H-PAD-30;
  var halfH=pH/2+10;

  // Draw sigma(z) top half
  var sigPts=[],gradPts=[];
  for(var i=0;i<=200;i++){
    var zv=-5+i/200*10;
    var s=1/(1+Math.exp(-zv));
    var sg=s*(1-s);
    var cx=PAD+i/200*(W-PAD-20);
    sigPts.push([cx,18+halfH-s*halfH]);
    gradPts.push([cx,18+halfH+12+(1-sg/0.25)*halfH*0.85]);
  }
  // Divider
  ln(ctx,PAD,18+halfH+8,W-10,18+halfH+8,'#1e1e2e',1);
  tx(ctx,'\u03C3(z) = 1/(1+e^-z)',PAD+50,22,'#ff6b35',9,'700','left');
  tx(ctx,"\u03C3'(z) = \u03C3(z)(1-\u03C3(z))  max=0.25",PAD+50,18+halfH+20,'#fbbf24',9,'700','left');

  curve(sigPts,ctx,'#ff6b35',2.5);
  curve(gradPts,ctx,'#fbbf24',2.5);

  // Horizontal reference lines
  var sig05Y=18+halfH-0.5*halfH;
  var grad025Y=18+halfH+12+(1-1)*halfH*0.85;
  ln(ctx,PAD,sig05Y,W-10,sig05Y,'#ff6b3530',1,[4,4]);
  tx(ctx,'0.5',PAD-6,sig05Y,'#ff6b3560',7,'400','right');
  ln(ctx,PAD,grad025Y,W-10,grad025Y,'#fbbf2430',1,[4,4]);
  tx(ctx,'0.25',PAD-6,grad025Y,'#fbbf2460',7,'400','right');

  // Current z marker
  var sv=1/(1+Math.exp(-z)); var gv=sv*(1-sv);
  var curX=PAD+(z+5)/10*(W-PAD-20);
  var curSY=18+halfH-sv*halfH;
  var curGY=18+halfH+12+(1-gv/0.25)*halfH*0.85;
  ln(ctx,curX,16,curX,H-10,'#3f3f46',1,[3,3]);
  ctx.fillStyle='#ff6b35'; ctx.beginPath(); ctx.arc(curX,curSY,5,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#fbbf24'; ctx.beginPath(); ctx.arc(curX,curGY,5,0,Math.PI*2); ctx.fill();
  tx(ctx,'z='+rnd(z,1),curX,H-8,'#52525b',7,'400');

  var satWarning=Math.abs(z)>3?'<br><span class="hl5">\u26A0 SATURATION ZONE: gradient='+rnd(gv,5)+' \u2248 0. Neuron contributes almost nothing to backprop.</span>':'';
  document.getElementById('iSigGrad').innerHTML=
    'At z=<span class="hl3">'+rnd(z,2)+'</span>: \u03C3(z)=<span class="hl">'+rnd(sv,4)+'</span>  gradient=<span class="hl3">'+rnd(gv,4)+'</span><br>'+
    'Max gradient is <span class="hl5">0.25</span> at z=0. Through 10 sigmoids: 0.25^10 \u2248 1e-6.'+satWarning;
}

function renderVanish(){
  var L=parseInt(document.getElementById('slVL').value);
  document.getElementById('vVL').textContent=L;
  var ctx=clr('cvVanish',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  var maxG=1.0, bW=Math.min(28,(pW/L)-3);

  axesPAD(ctx,W,H,PAD,'Layer','Gradient magnitude');
  ln(ctx,PAD,H-PAD-maxG/maxG*pH,W-10,H-PAD-maxG/maxG*pH,'#4ade8030',1,[4,4]);
  tx(ctx,'starting gradient=1.0',W/2,H-PAD-pH-8,'#4ade8040',7,'700');

  var pairs=[
    {factor:0.25,col:'#ff6b35',label:'Sigmoid (max grad=0.25)'},
    {factor:1.0, col:'#4ade80',label:'ReLU (grad=1.0 active)'},
  ];
  pairs.forEach(function(p){
    var pts=[];
    for(var l=1;l<=L;l++){
      var g=Math.pow(p.factor,l);
      var cx=PAD+(l-1)/(Math.max(L-1,1))*pW;
      var cy=H-PAD-Math.min(g/maxG,1)*pH;
      pts.push([cx,cy]);
    }
    curve(pts,ctx,p.col,2.5);
    tx(ctx,p.label,W-14,pts[pts.length-1][1],p.col,8,'700','right');
  });
  // Final gradient annotation
  var sigFinal=Math.pow(0.25,L);
  document.getElementById('iVanish').innerHTML=
    'After <span class="hl3">'+L+'</span> layers:<br>'+
    '<span class="hl5">Sigmoid (0.25^'+L+'):</span> gradient = <span class="hl5">'+sigFinal.toExponential(2)+'</span> '+(sigFinal<1e-5?'(\u2248 zero \u2014 completely vanished!)':'(getting small...)')+'<br>'+
    '<span class="hl4">ReLU (active, 1.0^'+L+'):</span> gradient = <span class="hl4">1.0</span> \u2014 perfectly preserved at any depth.';
}

function renderActGrad(){
  var ctx=clr('cvActGrad',860,180),W=860,H=180,PAD=30;
  var fns=[
    {name:'Sigmoid',col:'#ff6b35',fn:function(z){var s=1/(1+Math.exp(-z));return[s,s*(1-s)];}},
    {name:'Tanh',   col:'#fbbf24',fn:function(z){var t=Math.tanh(z);return[t,1-t*t];}},
    {name:'ReLU',   col:'#4ade80',fn:function(z){return[Math.max(0,z),z>0?1:0];}},
    {name:'GELU',   col:'#c084fc',fn:function(z){var g=0.5*z*(1+Math.tanh(Math.sqrt(2/Math.PI)*(z+0.044715*z*z*z)));var h=0.001;var gp=0.5*(1+Math.tanh(Math.sqrt(2/Math.PI)*((z+h)+0.044715*Math.pow(z+h,3))))+(z+h)*0.5*(1-Math.pow(Math.tanh(Math.sqrt(2/Math.PI)*((z+h)+0.044715*Math.pow(z+h,3))),2))*Math.sqrt(2/Math.PI)*(1+3*0.044715*(z+h)*(z+h));return[g,gp];}},
  ];
  var halfW=W/fns.length,pH=H-40,pW=halfW-20;
  fns.forEach(function(f,i){
    var x0=i*halfW+10;
    ln(ctx,x0,H-20,x0+pW,H-20,'#2d2d40',1);
    ln(ctx,x0+pW/2,10,x0+pW/2,H-20,'#2d2d40',1);
    var fnPts=[],gPts=[];
    for(var j=0;j<=100;j++){
      var zv=-4+j/100*8;
      var res=f.fn(zv);
      var fx=x0+(j/100)*pW;
      fnPts.push([fx,H-20-Math.max(-2,Math.min(2,res[0]))/2*pH]);
      gPts.push([fx,H-20-Math.max(-0.1,Math.min(1.1,res[1]))*pH*0.85]);
    }
    curve(fnPts,ctx,f.col,2);
    ctx.strokeStyle=f.col+'60'; ctx.lineWidth=1.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); gPts.forEach(function(p,pi){if(pi===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke(); ctx.setLineDash([]);
    tx(ctx,f.name,x0+pW/2,16,f.col,10,'800');
    var maxGrad=f.fn(0)[1];
    tx(ctx,'max grad='+rnd(maxGrad,2)+'@z=0',x0+pW/2,H-7,f.col+'80',7,'700');
  });
  tx(ctx,'\u2014 f(z)',20,H/2,'#e4e4e7',8,'700','left');
  tx(ctx,'-- f\'(z)',20,H/2+14,'#71717a',8,'700','left');
}

// ============================================================
// TAB 2: EXPLODING
// ============================================================
var curCliff='no';
function selCliff(k){curCliff=k;['no','yes'].forEach(function(b){document.getElementById('cBtn_'+b).classList.remove('active');});document.getElementById('cBtn_'+k).classList.add('active');renderCliff();}
function renderCliff(){
  var ctx=clr('cvCliff',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Loss');
  var N=80,cliffAt=40;
  var pts=[];
  // RNG for reproducible bumps
  var seed=7;function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
  for(var i=0;i<=N;i++){
    var t=i/N;
    var loss;
    if(curCliff==='no'){
      if(i<cliffAt) loss=2.0*Math.exp(-t*3)+0.3+rng()*0.04;
      else if(i<cliffAt+3) loss=2.0*Math.exp(-cliffAt/N*3)+0.3+(i-cliffAt)*3.5+rng()*0.1;
      else loss=2.0*Math.exp(-t*3)+0.3+4.5*Math.exp(-(i-cliffAt-3)*0.3)+rng()*0.08;
    } else {
      loss=2.0*Math.exp(-t*3)+0.3+rng()*0.04;
    }
    pts.push([PAD+t*pW,H-PAD-Math.min(loss/7,1)*pH]);
  }
  var maxL=curCliff==='no'?7:2.5;
  pts=[];seed=7;
  for(var i2=0;i2<=N;i2++){
    var t2=i2/N; var loss2;
    if(curCliff==='no'){
      if(i2<cliffAt)loss2=2.0*Math.exp(-t2*3)+0.3+rng()*0.04;
      else if(i2<cliffAt+3)loss2=2.0*Math.exp(-cliffAt/N*3)+0.3+(i2-cliffAt)*3.5+rng()*0.1;
      else loss2=2.0*Math.exp(-t2*3)+0.3+4.5*Math.exp(-(i2-cliffAt-3)*0.3)+rng()*0.08;
    } else {loss2=2.0*Math.exp(-t2*3)+0.3+rng()*0.04;}
    pts.push([PAD+t2*pW,H-PAD-Math.min(loss2/maxL,1)*pH]);
  }
  curve(pts,ctx,curCliff==='no'?'#ff6b35':'#4ade80',2.5);
  if(curCliff==='no'){
    var cx=PAD+cliffAt/N*pW;
    ln(ctx,cx,16,cx,H-PAD,'#fbbf2460',1.5,[4,4]);
    tx(ctx,'Loss Cliff',cx+5,28,'#fbbf24',8,'800','left');
    tx(ctx,'Gradient explosion',cx+5,40,'#fbbf24',7,'400','left');
  } else {
    tx(ctx,'Clipping prevents cliff',W/2,28,'#4ade80',9,'800');
  }
  tx(ctx,curCliff==='no'?'No Clipping \u2014 Loss Cliff at step ~'+cliffAt:'With Clipping \u2014 Smooth Convergence',W/2,H-PAD+18,'#52525b',8,'400');

  document.getElementById('iCliff').innerHTML=curCliff==='no'?
    '<span class="hl5">Loss cliff:</span> training progresses normally until a single batch with large gradient causes a catastrophic update. Loss jumps then partially recovers (if explosion is moderate) or diverges to NaN (if severe).<br><span class="hl">Fix:</span> gradient clipping (max_norm=1.0) and reduce learning rate.':
    '<span class="hl4">With gradient clipping:</span> when the gradient norm exceeds max_norm, ALL gradients are scaled down uniformly. The direction is preserved, only the magnitude is limited. Loss cliff is completely prevented.';
}

function renderEigen(){
  var eig=parseFloat(document.getElementById('slEig').value);
  document.getElementById('vEig').textContent=rnd(eig,2);
  var ctx=clr('cvEigen',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Num layers','Gradient magnitude (eig^L, log scale)');
  var logMax=10,logMin=-5,logRange=15;
  [0,5,10].forEach(function(v){var gy=H-PAD-(v-logMin)/logRange*pH;if(gy>15&&gy<H-PAD){ln(ctx,PAD,gy,W-10,gy,'#141420',1);tx(ctx,'10^'+v,PAD-6,gy,'#3f3f46',7,'400','right');}});
  var zeroY=H-PAD-(-logMin)/logRange*pH;
  ln(ctx,PAD,zeroY,W-10,zeroY,'#4ade8030',1,[4,4]);
  tx(ctx,'magnitude=1',W-14,zeroY-8,'#4ade8040',7,'700','right');
  var col=eig<1?'#4ecdc4':eig>1?'#ef4444':'#4ade80';
  var pts=[];
  for(var i=0;i<=50;i++){
    var L3=i; var g=Math.pow(eig,L3);
    var logG=Math.log10(Math.max(1e-5,Math.min(g,1e10)));
    var cx=PAD+(L3/50)*pW;
    var cy=H-PAD-(logG-logMin)/logRange*pH;
    pts.push([cx,Math.max(16,Math.min(H-PAD,cy))]);
  }
  curve(pts,ctx,col,2.5);
  [10,20,30,50].forEach(function(L4){
    var g2=Math.pow(eig,L4); var logG2=Math.log10(Math.max(1e-5,Math.min(g2,1e10)));
    var cx2=PAD+(L4/50)*pW; var cy2=H-PAD-(logG2-logMin)/logRange*pH;
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(cx2,Math.max(16,Math.min(H-PAD,cy2)),4,0,Math.PI*2); ctx.fill();
    tx(ctx,g2<1?g2.toExponential(1):g2>1e6?g2.toExponential(1):rnd(g2,1),cx2,Math.max(28,Math.min(H-PAD-6,cy2))-12,'#fbbf24',7,'700');
  });
  tx(ctx,'Eigenvalue \u03BB='+rnd(eig,2)+' per layer',W/2,22,col,10,'800');
  document.getElementById('iEigen').innerHTML=
    'Each layer multiplies gradient by W^T. If eigenvalue |\u03BB|=<span class="hl3">'+rnd(eig,2)+'</span>:<br>'+
    (eig>1?'<span class="hl5">Exploding:</span> ':'<span class="hl2">Vanishing:</span> ')+
    [10,20,50].map(function(L5){return'L='+L5+': '+rnd(eig,2)+'^'+L5+'\u2248'+rnd(Math.min(Math.pow(eig,L5),1e9),0);}).join(' | ')+'<br>'+
    (eig>1.2?'After 50 layers: gradient = '+rnd(Math.pow(eig,50),0)+' \u2014 catastrophic explosion':'');
}

function renderNaN(){
  var ctx=clr('cvNaN',860,140),W=860,H=140;
  var stages=[
    {label:'Large gradient',val:'\u2207L = 1e6',col:'#fbbf24',sub:'Explosion event'},
    {label:'Weight update',val:'w \u2190 w - \u03B7\u00D7grad',col:'#ff6b35',sub:'= Inf (overflow)'},
    {label:'Forward pass',val:'0 \u00D7 Inf',col:'#ef4444',sub:'= NaN produced'},
    {label:'Loss',val:'Loss = NaN',col:'#ef4444',sub:'NaN propagates'},
    {label:'All gradients',val:'\u2207 = NaN',col:'#ef4444',sub:'Chain rule \u00D7 NaN = NaN'},
    {label:'Training',val:'FAILED',col:'#ef4444',sub:'Irrecoverable'},
  ];
  var bw=(W-20)/stages.length-6,sx=12;
  stages.forEach(function(st,i){
    var x=sx+i*(bw+6);
    bx(ctx,x,20,bw,80,6,st.col+'18',st.col+'60',1.5);
    tx(ctx,st.label,x+bw/2,40,st.col,8,'800');
    tx(ctx,st.val,x+bw/2,60,'#fbbf24',9,'700');
    tx(ctx,st.sub,x+bw/2,77,'#52525b',7,'400');
    if(i<stages.length-1){tx(ctx,'\u2192',x+bw+3,60,'#3f3f46',12,'700');}
  });
  tx(ctx,'Fix: gradient clipping BEFORE optimizer.step() | assert torch.isfinite(loss) | reduce learning rate',W/2,118,'#52525b',8,'400');
}

// ============================================================
// TAB 3: DEAD NEURONS
// ============================================================
function renderDead(){
  var deadFrac=parseFloat(document.getElementById('slDead').value);
  document.getElementById('vDead').textContent=Math.round(deadFrac*100)+'%';
  var ctx=clr('cvDead',420,270),W=420,H=270;
  var N=48, cols=8, rows=6;
  var cS=36, gap=6;
  var startX=(W-(cols*(cS+gap)-gap))/2, startY=30;
  var deadCount=Math.round(N*deadFrac);
  var deadSet=new Set();
  var rngS=42; function rng2(){rngS=(rngS*1664525+1013904223)&0xffffffff;return(rngS>>>0)/4294967296;}
  var indices=[]; for(var i=0;i<N;i++) indices.push(i);
  for(var i2=0;i2<deadCount;i2++){var swap=i2+Math.floor(rng2()*(N-i2));var tmp=indices[i2];indices[i2]=indices[swap];indices[swap]=tmp;}
  for(var d=0;d<deadCount;d++) deadSet.add(indices[d]);

  for(var n=0;n<N;n++){
    var r2=Math.floor(n/cols),c=n%cols;
    var x=startX+c*(cS+gap), y=startY+r2*(cS+gap);
    var dead=deadSet.has(n);
    bx(ctx,x,y,cS,cS,cS/2,dead?'#09090f':'#4ade8025',dead?'#1e1e28':'#4ade80',dead?0.5:1.5);
    tx(ctx,dead?'0':'f',x+cS/2,y+cS/2,dead?'#1e1e28':'#4ade80',10,'800');
    if(dead){
      ctx.strokeStyle='#ef444440'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(x+4,y+4); ctx.lineTo(x+cS-4,y+cS-4); ctx.stroke();
    }
  }
  var aliveCount=N-deadCount;
  var effectiveCap=Math.round(aliveCount/N*100);
  tx(ctx,Math.round(deadFrac*100)+'% Dead ('+deadCount+' neurons)',W/2,H-40,'#ef4444',10,'800');
  tx(ctx,'Effective capacity: '+effectiveCap+'% of designed size',W/2,H-24,'#52525b',9,'400');
  tx(ctx,'Active: '+aliveCount+' / '+N,W/2,H-8,'#4ade80',9,'700');

  var severity=deadFrac<0.05?'<span class="hl4">Healthy (&lt;5% dead).</span>':deadFrac<0.2?'<span class="hl3">Mild (5\u201320%).</span> Monitor.':deadFrac<0.4?'<span class="hl5">Concerning (20\u201340%).</span> Switch to Leaky ReLU.':'<span class="hl5">Critical (&gt;40%).</span> Network severely degraded. Switch to GELU or Leaky ReLU immediately.';
  document.getElementById('iDead').innerHTML=
    severity+'<br>'+
    'Dead neuron causes: large LR sudden update, negative bias init, heavy L2 regularisation.<br>'+
    '<span class="hl">Fix:</span> Leaky ReLU or GELU, He init, reduce learning rate.';
}

var curAC='fn';
function selAC(k){curAC=k;['fn','grad'].forEach(function(b){document.getElementById('acBtn_'+b).classList.remove('active');});document.getElementById('acBtn_'+k).classList.add('active');renderActComp();}
function renderActComp(){
  var ctx=clr('cvActComp',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'z','f(z) or f\'(z)');
  ln(ctx,PAD,H-PAD-(0+2.5)/5*pH,W-10,H-PAD-(0+2.5)/5*pH,'#2d2d40',1,[3,3]);
  var zeroY=H-PAD-(0+2.5)/5*pH;
  var fns2=[
    {name:'ReLU',col:'#4ade80',fn:function(z){return[Math.max(0,z),z>0?1:0];}},
    {name:'Leaky (0.01)',col:'#fbbf24',fn:function(z){return[z>0?z:0.01*z,z>0?1:0.01];}},
    {name:'GELU',col:'#c084fc',fn:function(z){var g=0.5*z*(1+Math.tanh(Math.sqrt(2/Math.PI)*(z+0.044715*z*z*z)));var h=0.001;var fp=(0.5*(1+Math.tanh(Math.sqrt(2/Math.PI)*((z+h)+0.044715*Math.pow(z+h,3))))+z*0.5*(1-Math.pow(Math.tanh(Math.sqrt(2/Math.PI)*((z+h)+0.044715*Math.pow(z+h,3))),2))*Math.sqrt(2/Math.PI)*(1+3*0.044715*(z+h)*(z+h)));return[g,fp];}},
  ];
  var ymin=-2.5, ymax=2.5, yrange=5;
  fns2.forEach(function(f){
    var pts=[];
    for(var i=0;i<=200;i++){
      var z=-4+i/200*8;
      var res=f.fn(z);
      var val=curAC==='fn'?res[0]:res[1];
      var cx=PAD+(z+4)/8*pW;
      var cy=H-PAD-(Math.max(ymin,Math.min(ymax,val))-ymin)/yrange*pH;
      pts.push([cx,cy]);
    }
    curve(pts,ctx,f.col,2.5);
    tx(ctx,f.name,pts[200][0]-10,pts[200][1]-10,f.col,9,'800','right');
  });
  // Dead zone shading for ReLU
  if(curAC==='grad'){
    ctx.fillStyle='#ef444410';
    ctx.fillRect(PAD,15,pW/2,H-PAD-15);
    tx(ctx,'Dead zone (gradient=0)',PAD+pW/4,H-PAD-40,'#ef444450',8,'700');
  }
  tx(ctx,curAC==='fn'?'Activation f(z)':'Gradient f\'(z)',W/2,22,'#e4e4e7',10,'800');
  document.getElementById('iActComp').innerHTML=
    curAC==='fn'?
    '<span class="hl4">ReLU:</span> hard zero for z&lt;0. Simple and fast.<br><span class="hl3">Leaky:</span> small slope 0.01z for z&lt;0. Never exactly zero.<br><span class="hl6">GELU:</span> smooth, slight negative region near z=-0.17. Transformer default.':
    '<span class="hl5">ReLU gradient=0</span> for ALL z&le;0. Any neuron with negative pre-activation is permanently dead.<br><span class="hl4">Leaky ReLU gradient=0.01</span> for z&lt;0. Always non-zero \u2014 dead neurons can recover.<br><span class="hl6">GELU gradient</span> is near-zero but never exactly zero for z&lt;0. Practically death-free.';
}

// ============================================================
// TAB 4: CLIPPING
// ============================================================
function renderClip(){
  var maxN=parseFloat(document.getElementById('slClipN').value);
  var clipV=parseFloat(document.getElementById('slClipV').value);
  document.getElementById('vClipN').textContent=rnd(maxN,1);
  document.getElementById('vClipV').textContent=rnd(clipV,1);
  var ctx=clr('cvClip',420,290),W=420,H=290;
  var cx0=W/2-40,cy0=H/2;
  var gx=8.0,gy=6.0; // original gradient vector
  var gnorm=Math.sqrt(gx*gx+gy*gy);
  var scale=26;
  var ox=100,oy=200;
  // Coordinate system
  ln(ctx,ox,oy-120,ox,oy+20,'#2d2d40',1);
  ln(ctx,ox-20,oy,ox+240,oy,'#2d2d40',1);
  tx(ctx,'g\u2082',ox-12,oy-110,'#52525b',8,'400','right');
  tx(ctx,'g\u2081',ox+230,oy+4,'#52525b',8,'400');
  // Max norm circle
  ctx.strokeStyle='#4ecdc430'; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.arc(ox,oy,maxN*scale,0,Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
  tx(ctx,'max_norm='+rnd(maxN,1),ox+maxN*scale+5,oy-maxN*scale*0.5,'#4ecdc460',7,'700','left');
  // Original gradient
  var ex=ox+gx*scale, ey=oy-gy*scale;
  ln(ctx,ox,oy,ex,ey,'#fbbf24',2);
  ctx.fillStyle='#fbbf24'; ctx.beginPath(); ctx.moveTo(ex,ey); ctx.lineTo(ex-8*Math.cos(Math.atan2(-(ey-oy),(ex-ox))-0.4),ey-8*Math.sin(Math.atan2(-(ey-oy),(ex-ox))-0.4)); ctx.lineTo(ex-8*Math.cos(Math.atan2(-(ey-oy),(ex-ox))+0.4),ey-8*Math.sin(Math.atan2(-(ey-oy),(ex-ox))+0.4)); ctx.closePath(); ctx.fill();
  tx(ctx,'Original ('+gx+','+gy+') norm='+rnd(gnorm,1),ex+8,ey-4,'#fbbf24',8,'700','left');
  // Clip-by-norm
  if(gnorm>maxN){
    var scaleN=maxN/gnorm;
    var nx=ox+gx*scale*scaleN, ny=oy-gy*scale*scaleN;
    ln(ctx,ox,oy,nx,ny,'#4ecdc4',2.5);
    ctx.fillStyle='#4ecdc4'; ctx.beginPath(); ctx.moveTo(nx,ny); ctx.lineTo(nx-8*Math.cos(Math.atan2(-(ny-oy),(nx-ox))-0.4),ny-8*Math.sin(Math.atan2(-(ny-oy),(nx-ox))-0.4)); ctx.lineTo(nx-8*Math.cos(Math.atan2(-(ny-oy),(nx-ox))+0.4),ny-8*Math.sin(Math.atan2(-(ny-oy),(nx-ox))+0.4)); ctx.closePath(); ctx.fill();
    tx(ctx,'Norm clip ('+rnd(gx*scaleN,1)+','+rnd(gy*scaleN,1)+') \u2714 direction preserved',nx+5,ny-12,'#4ecdc4',8,'700','left');
  } else {
    tx(ctx,'Norm clip: no change (norm\u2264max_norm)',ex+8,ey-18,'#4ecdc4',8,'700','left');
  }
  // Clip-by-value
  var cvx=Math.min(Math.max(gx,-clipV),clipV), cvy=Math.min(Math.max(gy,-clipV),clipV);
  var vex=ox+cvx*scale, vey=oy-cvy*scale;
  ln(ctx,ox,oy,vex,vey,'#ff6b35',2);
  ctx.fillStyle='#ff6b35'; ctx.beginPath(); ctx.moveTo(vex,vey); ctx.lineTo(vex-8*Math.cos(Math.atan2(-(vey-oy),(vex-ox))-0.4),vey-8*Math.sin(Math.atan2(-(vey-oy),(vex-ox))-0.4)); ctx.lineTo(vex-8*Math.cos(Math.atan2(-(vey-oy),(vex-ox))+0.4),vey-8*Math.sin(Math.atan2(-(vey-oy),(vex-ox))+0.4)); ctx.closePath(); ctx.fill();
  tx(ctx,'Value clip ('+rnd(cvx,1)+','+rnd(cvy,1)+') \u2718 direction changed',vex-5,vey+14,'#ff6b35',8,'700','right');
  // Legend
  tx(ctx,'Original',W-20,20,'#fbbf24',8,'700','right');
  tx(ctx,'Clip-by-norm \u2192 preserves direction',W-20,34,'#4ecdc4',8,'700','right');
  tx(ctx,'Clip-by-value \u2192 distorts direction',W-20,48,'#ff6b35',8,'700','right');
  var cvnorm=Math.sqrt(cvx*cvx+cvy*cvy);
  document.getElementById('iClip').innerHTML=
    'Original gradient ('+gx+','+gy+'), norm=<span class="hl3">'+rnd(gnorm,2)+'</span><br>'+
    '<span class="hl2">Clip-by-norm:</span> scale to norm='+rnd(maxN,1)+'. Result=('+rnd(gx*maxN/gnorm,2)+','+rnd(gy*maxN/gnorm,2)+'). <span class="hl4">Direction preserved \u2714</span><br>'+
    '<span class="hl5">Clip-by-value:</span> clamp each component to \u00B1'+rnd(clipV,1)+'. Result=('+rnd(cvx,1)+','+rnd(cvy,1)+'), norm='+rnd(cvnorm,2)+'. <span class="hl5">Direction changed \u2718</span>';
}

function renderNorm(){
  var maxN=parseFloat(document.getElementById('slMaxN').value);
  document.getElementById('vMaxN').textContent=rnd(maxN,1);
  var ctx=clr('cvNorm',420,290),W=420,H=290,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Gradient norm');
  // Simulate gradient norms with occasional spikes
  var N=80,norms=[],seed2=42;
  function rng3(){seed2=(seed2*1664525+1013904223)&0xffffffff;return(seed2>>>0)/4294967296;}
  for(var i=0;i<N;i++){
    var base=0.3+rng3()*0.2;
    var spike=rng3()<0.06?(3+rng3()*8):0;
    norms.push(base+spike);
  }
  var maxNorm=Math.max.apply(null,norms)*1.05;
  // Max norm line
  var clipY=H-PAD-(maxN/maxNorm)*pH;
  ln(ctx,PAD,clipY,W-10,clipY,'#fbbf2480',1.5,[4,4]);
  tx(ctx,'max_norm='+rnd(maxN,1),W-14,clipY-8,'#fbbf24',8,'700','right');
  // Pre-clip bars
  var bw2=pW/N-1;
  norms.forEach(function(g,i){
    var clipped=g>maxN;
    var cx2=PAD+i*(pW/N);
    var pH2=(Math.min(g,maxNorm)/maxNorm)*pH;
    var pH3=(Math.min(Math.min(g,maxN),maxNorm)/maxNorm)*pH;
    bx(ctx,cx2,H-PAD-pH2,bw2,pH2,1,clipped?'#ef444420':'#4ecdc420',clipped?'#ef4444':'#4ecdc4',1);
    if(clipped) bx(ctx,cx2,H-PAD-pH3,bw2,pH3,1,'#4ade8030','#4ade80',1);
  });
  var clippedCount=norms.filter(function(g){return g>maxN;}).length;
  tx(ctx,'Pre-clip norm  Clipped to max_norm  Clip events: '+clippedCount+'/'+N+' ('+Math.round(clippedCount/N*100)+'%)',W/2,H-PAD+18,'#52525b',7,'400');
  var pct=clippedCount/N;
  document.getElementById('iNorm').innerHTML=
    'Clipping at max_norm=<span class="hl3">'+rnd(maxN,1)+'</span>.<br>'+
    'Clip events: <span class="'+(pct<0.01?'hl4':pct<0.2?'hl3':'hl5')+'">'+clippedCount+' / '+N+' steps ('+Math.round(pct*100)+'%)</span><br>'+
    (pct<0.01?'<span class="hl4">Healthy: clipping is a safety net. Normal training.</span>':pct<0.2?'<span class="hl3">Moderate: clipping active but manageable.</span>':'<span class="hl5">Warning: clipping &gt;20% of steps masks underlying instability. Reduce LR or fix initialisation.</span>');
}

// ============================================================
// TAB 5: SOLUTIONS
// ============================================================
var curRes='plain';
function selRes(k){curRes=k;['plain','res'].forEach(function(b){document.getElementById('rBtn_'+b).classList.remove('active');});document.getElementById('rBtn_'+k).classList.add('active');renderRes();}
function renderRes(){
  var ctx=clr('cvRes',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  var L=10,maxG=1.0;
  var factor=curRes==='res'?0.85:0.75;
  axesPAD(ctx,W,H,PAD,'Layer index (1=earliest)','Relative gradient');
  ln(ctx,PAD,H-PAD-pH,W-10,H-PAD-pH,'#4ade8030',1,[4,4]);

  if(curRes==='plain'){
    var pts=[];
    for(var l=1;l<=L;l++){var g=Math.pow(factor,L-l+1);var cx=PAD+(l-1)/(L-1)*pW;pts.push([cx,H-PAD-Math.min(g,1)*pH]);}
    curve(pts,ctx,'#ff6b35',2.5);
    tx(ctx,'Gradient vanishes at early layers',W/2,22,'#ff6b35',9,'800');
    tx(ctx,'Each layer multiplies gradient by '+factor,W/2,H-PAD+18,'#52525b',8,'400');
  } else {
    // Residual: gradient = 1 + dF/dx, always >= 1
    var pts2=[],pts3=[];
    for(var l2=1;l2<=L;l2++){
      var gPlain=Math.pow(factor,L-l2+1);
      var gRes=Math.min(1+gPlain,2); // simplified: 1 + residual branch grad
      var cx2=PAD+(l2-1)/(L-1)*pW;
      pts2.push([cx2,H-PAD-Math.min(gPlain,1)*pH]);
      pts3.push([cx2,H-PAD-Math.min(gRes/2,1)*pH]);
    }
    curve(pts2,ctx,'#ff6b3560',1.5);
    curve(pts3,ctx,'#4ade80',2.5);
    tx(ctx,'Residual gradient \u2265 1.0 (highway always open)',W/2,22,'#4ade80',9,'800');
    tx(ctx,'d(x+F(x))/dx = 1 + dF/dx \u2014 skip connection is the highway',W/2,H-PAD+18,'#52525b',8,'400');
    tx(ctx,'plain (without skip)',W-14,pts2[0][1],'#ff6b3570',8,'700','right');
  }
  document.getElementById('iRes').innerHTML=curRes==='plain'?
    '<span class="hl5">Plain network:</span> gradient \u2202Loss/\u2202w\u2081 = product of L Jacobians. Each multiplication can shrink the signal. Layer 1 gets ('+factor+')^'+L+' \u2248 '+rnd(Math.pow(factor,L),4)+' of the original gradient.':
    '<span class="hl4">Residual network:</span> \u2202Loss/\u2202x = \u2202Loss/\u2202output \u00D7 (1 + \u2202F/\u2202x). The +1 from the skip connection provides a DIRECT gradient path. Even if \u2202F/\u2202x \u2248 0, gradient flows unchanged. This is why ResNet-1000 trains but plain-20 fails.';
}

var curInit='he';
function selInit(k){curInit=k;['he','xavier','small','large'].forEach(function(b){document.getElementById('initBtn_'+b).classList.remove('active');});document.getElementById('initBtn_'+k).classList.add('active');renderInit();}
function renderInit(){
  var ctx=clr('cvInit',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  var configs={
    he:    {label:'He init (ReLU)',col:'#4ade80',fn:function(l){return 1.0;},info:'<span class="hl4">He init:</span> \u03C3=sqrt(2/n_in). Compensates for ReLU\'s 50% kill rate. Variance stays 1.0 at every depth. Activations remain in healthy range for any depth.'},
    xavier:{label:'Xavier/Glorot',col:'#fbbf24',fn:function(l){return Math.pow(0.97,l);},info:'<span class="hl3">Xavier init:</span> \u03C3=sqrt(2/(n_in+n_out)). Designed for tanh/sigmoid. Slight decay with depth. Better than He for non-ReLU activations but still loses signal in very deep nets.'},
    small: {label:'Too small init',col:'#4ecdc4',fn:function(l){return Math.pow(0.5,l);},info:'<span class="hl5">Too small:</span> weights near 0 \u2192 each layer compresses activations. Variance collapses exponentially. Gradient also vanishes. Model starts in a dead zone it cannot escape.'},
    large: {label:'Too large init',col:'#ef4444',fn:function(l){return Math.min(Math.pow(1.5,l),50);},info:'<span class="hl5">Too large:</span> activations explode immediately. Sigmoid/tanh saturate on the first forward pass. Gradients near 0 everywhere (saturation). Model never trains.'},
  };
  var cfg=configs[curInit];
  var L3=30,maxV=50;
  axesPAD(ctx,W,H,PAD,'Layer','Activation variance (relative)');
  ln(ctx,PAD,H-PAD-1/maxV*pH,W-10,H-PAD-1/maxV*pH,'#4ade8030',1,[4,4]);
  tx(ctx,'variance=1.0 (target)',W-14,H-PAD-1/maxV*pH-8,'#4ade8040',7,'700','right');
  var pts=[];
  for(var l3=0;l3<=L3;l3++){
    var v=cfg.fn(l3);
    var cx=PAD+(l3/L3)*pW;
    var cy=H-PAD-Math.min(v/maxV,1)*pH;
    pts.push([cx,Math.max(16,Math.min(H-PAD,cy))]);
  }
  curve(pts,ctx,cfg.col,2.5);
  [5,10,20,30].forEach(function(l4){
    var v2=cfg.fn(l4);
    var cx2=PAD+(l4/L3)*pW; var cy2=H-PAD-Math.min(v2/maxV,1)*pH;
    ctx.fillStyle=cfg.col; ctx.beginPath(); ctx.arc(cx2,Math.max(16,Math.min(H-PAD,cy2)),4,0,Math.PI*2); ctx.fill();
    tx(ctx,rnd(Math.min(v2,999),v2<0.01?4:1),cx2,Math.max(28,Math.min(H-PAD-6,cy2))-12,'#fbbf24',7,'700');
  });
  tx(ctx,cfg.label,W/2,22,cfg.col,10,'800');
  document.getElementById('iInit').innerHTML=cfg.info;
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){
  renderFlow(); renderChain(); renderDepthEffect();
});
</script>
</body>
</html>"""

GRADIENT_FLOW_VISUAL_HEIGHT = 2200