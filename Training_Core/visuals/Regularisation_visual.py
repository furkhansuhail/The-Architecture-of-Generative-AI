REGULARISATION_VISUAL_HTML = r"""<!DOCTYPE html>
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
.tab:hover{color:#e4e4e7;}
.tab.on{color:#ff6b35;border-bottom-color:#ff6b35;}
.panel{display:none;}.panel.on{display:block;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.card{background:#111118;border:1px solid #1e1e2e;border-radius:14px;padding:18px;margin-bottom:14px;}
.card h3{font-family:'JetBrains Mono',monospace;font-size:0.75em;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;color:#ff6b35;margin-bottom:10px;}
canvas{display:block;border-radius:8px;background:#09090f;}
.row{display:flex;align-items:center;gap:10px;margin:6px 0;}
.row label{font-family:'JetBrains Mono',monospace;font-size:0.73em;color:#71717a;min-width:96px;}
input[type=range]{flex:1;height:4px;border-radius:2px;-webkit-appearance:none;background:#1e1e2e;outline:none;cursor:pointer;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#ff6b35;cursor:pointer;}
.val{font-family:'JetBrains Mono',monospace;font-size:0.73em;color:#ff6b35;min-width:44px;text-align:right;}
.info{background:#0d0d18;border:1px solid #1e1e2e;border-radius:8px;padding:10px 14px;font-size:0.76em;color:#94a3b8;line-height:1.8;margin-top:10px;font-family:'JetBrains Mono',monospace;}
.hl{color:#ff6b35;font-weight:700;}.hl2{color:#4ecdc4;font-weight:700;}.hl3{color:#fbbf24;font-weight:700;}
.hl4{color:#4ade80;font-weight:700;}.hl5{color:#ef4444;font-weight:700;}.hl6{color:#c084fc;font-weight:700;}
.brow{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
button.s{background:#1e1e2e;color:#a1a1aa;border:1px solid #2d2d40;border-radius:6px;padding:5px 12px;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:0.73em;font-weight:700;transition:all 0.15s;}
button.s:hover{background:#2d2d40;color:#e4e4e7;}
button.s.on{background:#ff6b35;color:#08080f;border-color:#ff6b35;}
.tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.71em;}
.tbl th{padding:6px 9px;text-align:left;color:#52525b;border-bottom:2px solid #1e1e2e;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;}
.tbl td{padding:6px 9px;border-bottom:1px solid #0f0f18;color:#94a3b8;vertical-align:top;line-height:1.5;}
.tbl tr:hover td{background:#0d0d18;}
</style>
</head>
<body>
<h2>&#x1F6E1; Regularisation</h2>
<p class="sub">Bias-Variance &middot; L1 &amp; L2 Geometry &middot; Dropout &middot; DropPath &middot; Early Stopping &middot; Decision Guide</p>
<div class="tabs">
  <button class="tab on"  onclick="showTab(0)">Bias-Variance</button>
  <button class="tab"     onclick="showTab(1)">L1 vs L2</button>
  <button class="tab"     onclick="showTab(2)">Dropout</button>
  <button class="tab"     onclick="showTab(3)">DropPath</button>
  <button class="tab"     onclick="showTab(4)">Early Stopping</button>
  <button class="tab"     onclick="showTab(5)">Decision Guide</button>
</div>

<!-- ============================================================ TAB 0 ============================================================ -->
<div id="t0" class="panel on">
  <div class="g2">
    <div class="card">
      <h3>&#9312; Overfitting Loss Curves</h3>
      <canvas id="cvLoss" width="420" height="240"></canvas>
      <div class="row"><label>Reg. strength</label>
        <input type="range" id="slReg" min="0" max="1" step="0.02" value="0" oninput="renderLoss()">
        <span class="val" id="vReg">None</span></div>
      <div class="info" id="iLoss">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Bias-Variance U-Curve</h3>
      <canvas id="cvBV" width="420" height="240"></canvas>
      <div class="row"><label>Model complexity</label>
        <input type="range" id="slCmplx" min="0" max="100" step="1" value="50" oninput="renderBV()">
        <span class="val" id="vCmplx">50</span></div>
      <div class="info" id="iBV">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Diagnosis Table</h3>
    <table class="tbl">
      <thead><tr><th>Train loss</th><th>Val loss</th><th>Diagnosis</th><th>Fix</th></tr></thead>
      <tbody>
        <tr><td><span class="hl5">High</span></td><td><span class="hl5">High</span></td><td><span class="hl5">Underfitting (high bias)</span></td><td>More capacity, more training, less regularisation</td></tr>
        <tr><td><span class="hl4">Low</span></td><td><span class="hl5">High</span></td><td><span class="hl3">Overfitting (high variance)</span></td><td>More regularisation, more data, less capacity</td></tr>
        <tr><td><span class="hl4">Low</span></td><td><span class="hl4">Low</span></td><td><span class="hl4">Good fit</span></td><td>No action needed</td></tr>
        <tr><td><span class="hl5">High</span></td><td>Lower than train</td><td><span class="hl6">Data bug / distribution shift</span></td><td>Check data pipeline</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ============================================================ TAB 1 ============================================================ -->
<div id="t1" class="panel">
  <div class="g2">
    <div class="card">
      <h3>&#9312; L1 vs L2 Geometry (Constraint Regions)</h3>
      <canvas id="cvGeo" width="420" height="340"></canvas>
      <div class="brow">
        <button class="s on" id="gL2" onclick="selGeo('l2')">L2 (sphere)</button>
        <button class="s"    id="gL1" onclick="selGeo('l1')">L1 (diamond)</button>
        <button class="s"    id="gEN" onclick="selGeo('en')">Elastic Net</button>
      </div>
      <div class="info" id="iGeo">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Weight Shrinkage: L1 vs L2</h3>
      <canvas id="cvShrink" width="420" height="200"></canvas>
      <div class="row"><label>Lambda &lambda;</label>
        <input type="range" id="slLam" min="0" max="3" step="0.05" value="0.5" oninput="renderShrink()">
        <span class="val" id="vLam">0.50</span></div>
      <div class="card" style="margin-top:10px;padding:12px;">
        <h3>&#9314; Prior Distribution</h3>
        <canvas id="cvPrior" width="380" height="140"></canvas>
        <div class="info" id="iPrior" style="margin-top:6px;">Loading...</div>
      </div>
    </div>
  </div>
</div>

<!-- ============================================================ TAB 2 ============================================================ -->
<div id="t2" class="panel">
  <div class="g2">
    <div class="card">
      <h3>&#9312; Inverted Dropout &mdash; Training vs Inference</h3>
      <canvas id="cvDrop" width="420" height="300"></canvas>
      <div class="row"><label>Drop rate p</label>
        <input type="range" id="slDrop" min="0" max="0.9" step="0.05" value="0.5" oninput="renderDrop()">
        <span class="val" id="vDrop">0.50</span></div>
      <div class="info" id="iDrop">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; MC Dropout &mdash; Uncertainty Estimation</h3>
      <canvas id="cvMC" width="420" height="300"></canvas>
      <div class="row"><label>Drop rate p</label>
        <input type="range" id="slMC" min="0.05" max="0.5" step="0.05" value="0.2" oninput="renderMC()">
        <span class="val" id="vMC">0.20</span></div>
      <div class="row"><label>Passes T</label>
        <input type="range" id="slT" min="5" max="50" step="5" value="20" oninput="renderMC()">
        <span class="val" id="vT">20</span></div>
      <div class="info" id="iMC">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Where to Place Dropout</h3>
    <table class="tbl">
      <thead><tr><th>Location</th><th>Recommendation</th><th>p range</th></tr></thead>
      <tbody>
        <tr><td>After FC / Linear layers (MLP)</td><td><span class="hl4">Standard &mdash; highly effective</span></td><td>0.3&ndash;0.5</td></tr>
        <tr><td>After conv layers</td><td><span class="hl3">Use SpatialDropout2d (drops whole channels)</span></td><td>0.1&ndash;0.2</td></tr>
        <tr><td>After embedding layer (NLP)</td><td><span class="hl4">Common and effective</span></td><td>0.1&ndash;0.3</td></tr>
        <tr><td>Attention dropout (Transformers)</td><td><span class="hl4">Standard in GPT/BERT</span></td><td>0.1</td></tr>
        <tr><td>After BatchNorm / LayerNorm</td><td><span class="hl3">Controversial &mdash; can interfere with BN</span></td><td>Avoid</td></tr>
        <tr><td>After output logits</td><td><span class="hl5">NEVER &mdash; prevents predictions</span></td><td>&mdash;</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ============================================================ TAB 3 ============================================================ -->
<div id="t3" class="panel">
  <div class="g2">
    <div class="card">
      <h3>&#9312; Standard Dropout vs DropPath in a Residual Block</h3>
      <canvas id="cvDP" width="420" height="320"></canvas>
      <div class="brow">
        <button class="s on" id="dpN"  onclick="selDP('none')">Normal</button>
        <button class="s"    id="dpSD" onclick="selDP('std')">Standard Dropout</button>
        <button class="s"    id="dpDP" onclick="selDP('drop')">DropPath</button>
      </div>
      <div class="info" id="iDP">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Linear Drop Rate Schedule</h3>
      <canvas id="cvDPS" width="420" height="280"></canvas>
      <div class="row"><label>Max drop p_max</label>
        <input type="range" id="slPmax" min="0.05" max="0.8" step="0.05" value="0.3" oninput="renderDPS()">
        <span class="val" id="vPmax">0.30</span></div>
      <div class="row"><label>Num layers L</label>
        <input type="range" id="slLayers" min="4" max="24" step="2" value="12" oninput="renderDPS()">
        <span class="val" id="vLayers">12</span></div>
      <div class="info" id="iDPS">Loading...</div>
    </div>
  </div>
</div>

<!-- ============================================================ TAB 4 ============================================================ -->
<div id="t4" class="panel">
  <div class="g2">
    <div class="card">
      <h3>&#9312; Early Stopping &mdash; Loss Curves &amp; Patience</h3>
      <canvas id="cvES" width="420" height="300"></canvas>
      <div class="row"><label>Patience</label>
        <input type="range" id="slPat" min="1" max="30" step="1" value="10" oninput="renderES()">
        <span class="val" id="vPat">10</span></div>
      <div class="row"><label>Overfit speed</label>
        <input type="range" id="slOvf" min="0.1" max="2.0" step="0.1" value="0.8" oninput="renderES()">
        <span class="val" id="vOvf">0.80</span></div>
      <div class="info" id="iES">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Early Stopping = Implicit L2 Regularisation</h3>
      <canvas id="cvESEq" width="420" height="300"></canvas>
      <div class="row"><label>Training steps t</label>
        <input type="range" id="slSteps" min="5" max="200" step="5" value="50" oninput="renderESEq()">
        <span class="val" id="vSteps">50</span></div>
      <div class="info" id="iESEq">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Early Stopping Best Practices</h3>
    <table class="tbl">
      <thead><tr><th>Parameter</th><th>Guideline</th><th>Reasoning</th></tr></thead>
      <tbody>
        <tr><td>patience</td><td>10&ndash;20 for most tasks</td><td>Must outlast natural val loss noise</td></tr>
        <tr><td>min_delta</td><td>1e-4 to 1e-3</td><td>Ignore trivial improvements (&lt;0.001 per epoch)</td></tr>
        <tr><td>Monitor</td><td>Val <span class="hl">loss</span>, not accuracy</td><td>Loss is continuous; accuracy is discrete</td></tr>
        <tr><td>restore_best</td><td><span class="hl4">Always restore best checkpoint</span></td><td>Stop epoch is patience epochs past the best</td></tr>
        <tr><td>patience (small datasets)</td><td>20&ndash;50</td><td>Val loss fluctuates more with small N</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ============================================================ TAB 5 ============================================================ -->
<div id="t5" class="panel">
  <div class="card">
    <h3>&#9312; Standard Regularisation Stacks by Architecture</h3>
    <div class="g2">
      <table class="tbl">
        <thead><tr><th>Architecture</th><th>L2 / Weight decay</th><th>Dropout</th><th>Other</th></tr></thead>
        <tbody>
          <tr><td><span class="hl2">MLP</span></td><td>AdamW &lambda;=0.01</td><td>p=0.3&ndash;0.5 (FC layers)</td><td>Early stopping</td></tr>
          <tr><td><span class="hl3">CNN</span></td><td>AdamW &lambda;=1e-4&ndash;1e-3</td><td>SpatialDropout p=0.1&ndash;0.2</td><td>RandAugment, Mixup, Label smooth</td></tr>
          <tr><td><span class="hl6">ViT / Swin</span></td><td>AdamW &lambda;=0.05&ndash;0.1</td><td>Attention dropout p=0.0&ndash;0.1</td><td>DropPath p_max=0.1&ndash;0.3, Mixup</td></tr>
          <tr><td><span class="hl">GPT / BERT</span></td><td>AdamW &lambda;=0.01 (no norm/bias)</td><td>Residual p=0.1, Attn p=0.1</td><td>Embedding dropout p=0.1</td></tr>
          <tr><td><span class="hl4">GNN</span></td><td>&lambda;=5e-4</td><td>Node features p=0.5</td><td>DropEdge</td></tr>
        </tbody>
      </table>
      <table class="tbl">
        <thead><tr><th>Dataset size</th><th>Regularisation approach</th></tr></thead>
        <tbody>
          <tr><td><span class="hl5">&lt; 1K samples</span></td><td>Strong L2 (&lambda;=0.1), heavy dropout (p=0.5), aggressive augmentation, early stopping, smaller model or transfer learning</td></tr>
          <tr><td><span class="hl3">1K&ndash;100K</span></td><td>Moderate L2 (&lambda;=0.01), dropout (p=0.1&ndash;0.3), augmentation, early stopping patience=10</td></tr>
          <tr><td><span class="hl4">100K&ndash;10M</span></td><td>Light AdamW (&lambda;=0.01), DropPath for transformers, standard augmentation. Early stopping optional.</td></tr>
          <tr><td><span class="hl2">&gt; 10M</span></td><td>Light AdamW only. Large models often train without dropout. Augmentation still beneficial.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <h3>&#9313; Interaction Rules</h3>
    <table class="tbl">
      <thead><tr><th>Technique A</th><th>Technique B</th><th>Interaction</th></tr></thead>
      <tbody>
        <tr><td>L2 / AdamW</td><td>Dropout</td><td><span class="hl4">Complement each other. Use both; tune &lambda; and p independently.</span></td></tr>
        <tr><td>BatchNorm</td><td>Dropout</td><td><span class="hl5">Interact poorly. Prefer one per block. If both: BN before Dropout.</span></td></tr>
        <tr><td>Label smoothing</td><td>Distillation</td><td><span class="hl5">DO NOT combine. Smoothing on teacher degrades soft target dark knowledge.</span></td></tr>
        <tr><td>DropPath</td><td>Dropout</td><td><span class="hl3">Can combine but rarely both needed. DropPath preferred for residual nets.</span></td></tr>
        <tr><td>Data augmentation</td><td>L2 / Dropout</td><td><span class="hl4">Complement. Augmentation = input space; L2/dropout = parameter space.</span></td></tr>
        <tr><td>Early stopping</td><td>Any</td><td><span class="hl4">Compatible with all. Always use as a safety net.</span></td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
// ======================== UTILS ========================
function rnd(v,d){return v.toFixed(d===undefined?3:d);}
function C(id){var cv=document.getElementById(id);var ctx=cv.getContext('2d');ctx.fillStyle='#09090f';ctx.fillRect(0,0,cv.width,cv.height);return{cv:cv,ctx:ctx,W:cv.width,H:cv.height};}
function T(ctx,s,x,y,col,sz,wt,al){ctx.fillStyle=col||'#e4e4e7';ctx.font=(wt||'500')+' '+(sz||11)+'px JetBrains Mono,monospace';ctx.textAlign=al||'center';ctx.textBaseline='middle';ctx.fillText(s,x,y);}
function axis(ctx,x1,y1,x2,y2,col){ctx.strokeStyle=col||'#2d2d40';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();}
function curve(ctx,pts,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||2.5;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}
function dashedLine(ctx,x1,y1,x2,y2,col,da){ctx.strokeStyle=col;ctx.lineWidth=1;ctx.setLineDash(da||[5,4]);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);}
function fillB(ctx,x,y,w,h,r,fill,stroke,sw){ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,w,h,r);else ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw||1.5;ctx.stroke();}}

// ======================== TAB SWITCHING ========================
function showTab(i){
  for(var t=0;t<6;t++){document.getElementById('t'+t).classList.remove('on');document.querySelectorAll('.tab')[t].classList.remove('on');}
  document.getElementById('t'+i).classList.add('on');
  document.querySelectorAll('.tab')[i].classList.add('on');
  var draws=[function(){renderLoss();renderBV();},function(){renderGeo();renderShrink();renderPrior();},function(){renderDrop();renderMC();},function(){renderDP();renderDPS();},function(){renderES();renderESEq();},function(){}];
  draws[i]();
}

// ======================== TAB 0: BIAS-VARIANCE ========================
function renderLoss(){
  var reg=parseFloat(document.getElementById('slReg').value);
  document.getElementById('vReg').textContent=reg===0?'None':rnd(reg,2);
  var r=C('cvLoss');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=38;var plotW=W-PAD-14;var plotH=H-PAD-20;
  axis(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40');axis(ctx,PAD,14,PAD,H-PAD,'#2d2d40');
  T(ctx,'Training Loss',W/2,H-8,'#52525b',9,'400');T(ctx,'Loss',PAD+16,20,'#52525b',9,'400');
  var N=80;
  // Training loss: always drops
  var trainPts=[];for(var i=0;i<N;i++){var t=i/N;var v=1.2*Math.exp(-t*3)+0.05;var cx=PAD+t*plotW;var cy=H-PAD-Math.min(v/1.5,1)*plotH;trainPts.push([cx,cy]);}
  curve(ctx,trainPts,'#4ecdc4',2.5);T(ctx,'train',trainPts[70][0]-10,trainPts[70][1]-10,'#4ecdc4',9,'700');
  // Val loss: diverges based on reg
  var valPts=[];var bestEpoch=0;var bestV=9999;
  for(var i2=0;i2<N;i2++){
    var t2=i2/N;
    var baseDown=0.8*Math.exp(-t2*4)+0.18;
    var overfit=reg<0.05?(0.7*(1-Math.exp(-Math.max(0,t2-0.3)*6))):(0.25*(1-reg)*(1-Math.exp(-Math.max(0,t2-0.5)*3)));
    var v2=baseDown+overfit;
    var cx2=PAD+t2*plotW;var cy2=H-PAD-Math.min(v2/1.5,1)*plotH;
    valPts.push([cx2,cy2]);
    if(v2<bestV){bestV=v2;bestEpoch=i2;}
  }
  curve(ctx,valPts,'#ff6b35',2.5);T(ctx,'val',valPts[70][0]+10,valPts[70][1]-10,'#ff6b35',9,'700');
  // Gap annotation
  var gapX=trainPts[65][0];var gapY1=trainPts[65][1];var gapY2=valPts[65][1];
  if(reg<0.2&&gapY2<gapY1-10){
    ctx.strokeStyle='#fbbf2440';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(gapX,gapY1);ctx.lineTo(gapX,gapY2);ctx.stroke();ctx.setLineDash([]);
    T(ctx,'gap',gapX+10,(gapY1+gapY2)/2,'#fbbf24',8,'700');
  }
  // Tick labels
  T(ctx,'0',PAD-4,H-PAD,'#3f3f46',8,'400','right');T(ctx,'Epoch',W-14,H-PAD,'#3f3f46',8,'400');
  var regLabel=reg===0?'No regularisation':reg<0.3?'Mild regularisation':'Strong regularisation';
  var col=reg===0?'#ef4444':reg<0.3?'#fbbf24':'#4ade80';
  document.getElementById('iLoss').innerHTML='<span style="color:'+col+';font-weight:700">'+regLabel+'</span><br>'+(reg===0?'<span class="hl5">Validation loss diverges</span> while training loss drops. Classic overfitting signature. Gap = generalisation error.':'Regularisation narrows the gap. <span class="hl4">Model generalises better</span>. Training loss is slightly higher (regulariser adds a penalty term).');
}

function renderBV(){
  var cmplx=parseInt(document.getElementById('slCmplx').value);
  document.getElementById('vCmplx').textContent=cmplx;
  var r=C('cvBV');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=38;var plotW=W-PAD-14;var plotH=H-PAD-28;
  axis(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40');axis(ctx,PAD,14,PAD,H-PAD,'#2d2d40');
  T(ctx,'Model Complexity',W/2,H-10,'#52525b',9,'400');T(ctx,'Error',PAD+16,20,'#52525b',9,'400');
  var N2=100;
  var biasArr=[],varArr=[],totalArr=[];
  for(var i=0;i<N2;i++){
    var x=i/N2;
    var bias=0.6*Math.pow(1-x,1.8)+0.04;
    var variance=0.05+0.7*Math.pow(x,2.2);
    var noise=0.12;
    biasArr.push(bias);varArr.push(variance);totalArr.push(bias+variance+noise);
  }
  var maxE=Math.max.apply(null,totalArr)*1.05;
  var pts=function(arr){return arr.map(function(v,i){return[PAD+i/N2*plotW,H-PAD-Math.min(v/maxE,1)*plotH];});};
  curve(ctx,pts(biasArr),'#4ecdc4',2);T(ctx,'Bias\u00B2',pts(biasArr)[80][0]-5,pts(biasArr)[80][1]-10,'#4ecdc4',9,'700');
  curve(ctx,pts(varArr),'#c084fc',2);T(ctx,'Variance',pts(varArr)[80][0]-5,pts(varArr)[80][1]-10,'#c084fc',9,'700');
  // noise floor
  var noiseY=H-PAD-0.12/maxE*plotH;dashedLine(ctx,PAD,noiseY,W-10,noiseY,'#3f3f46',[4,4]);T(ctx,'noise floor',PAD+50,noiseY-8,'#3f3f46',8,'400');
  curve(ctx,pts(totalArr),'#ff6b35',2.5);T(ctx,'Total error',pts(totalArr)[20][0]+35,pts(totalArr)[20][1],'#ff6b35',9,'700');
  // Sweet spot = minimum of total
  var minIdx=totalArr.indexOf(Math.min.apply(null,totalArr));
  var sx=PAD+minIdx/N2*plotW;
  dashedLine(ctx,sx,14,sx,H-PAD,'#fbbf24',[4,4]);
  T(ctx,'sweet spot',sx,20,'#fbbf24',8,'700');
  // Current position
  var cIdx=Math.round(cmplx/100*(N2-1));
  var cx2=PAD+cIdx/N2*plotW;var cy2=H-PAD-Math.min(totalArr[cIdx]/maxE,1)*plotH;
  ctx.fillStyle='#ff6b35';ctx.beginPath();ctx.arc(cx2,cy2,6,0,Math.PI*2);ctx.fill();
  T(ctx,'you',cx2,cy2-14,'#ff6b35',9,'800');
  var state=cmplx<35?'underfitting':cmplx>65?'overfitting':'balanced';
  var sCol=cmplx<35?'#ef4444':cmplx>65?'#fbbf24':'#4ade80';
  document.getElementById('iBV').innerHTML='Complexity=<span class="hl3">'+cmplx+'</span> &rarr; <span style="color:'+sCol+';font-weight:700">'+state+'</span><br>'+(cmplx<35?'<span class="hl5">High bias</span> dominates. Model too simple to capture signal. Increase capacity or reduce regularisation.':cmplx>65?'<span class="hl3">High variance</span> dominates. Model memorises noise. Increase regularisation, get more data.':'<span class="hl4">Near optimal.</span> Bias and variance roughly balanced. Total error near minimum.');
}

// ======================== TAB 1: L1 vs L2 ========================
var curGeo='l2';
function selGeo(k){curGeo=k;document.querySelectorAll('[id^="g"]').forEach(function(b){b.classList.remove('on');});document.getElementById('g'+k.replace('l2','L2').replace('l1','L1').replace('en','EN').toLowerCase()).classList.remove('on');document.getElementById('g'+(k==='l2'?'L2':k==='l1'?'L1':'EN')).classList.add('on');renderGeo();}

function renderGeo(){
  var r=C('cvGeo');var ctx=r.ctx;var W=r.W;var H=r.H;
  var cx=W/2;var cy=H/2+10;var radius=110;
  // OLS solution point (off-axis)
  var olsX=cx+95;var olsY=cy-80;
  // Axes
  axis(ctx,20,cy,W-20,cy,'#2d2d40');axis(ctx,cx,10,cx,H-10,'#2d2d40');
  T(ctx,'w\u2081',W-16,cy+16,'#52525b',10,'700');T(ctx,'w\u2082',cx+12,16,'#52525b',10,'700');
  // OLS solution
  ctx.fillStyle='#fbbf24';ctx.beginPath();ctx.arc(olsX,olsY,7,0,Math.PI*2);ctx.fill();
  T(ctx,'OLS solution',olsX+14,olsY-4,'#fbbf24',9,'700','left');
  // Contour ellipses around OLS
  for(var k=1;k<=3;k++){ctx.strokeStyle='#fbbf24'+(k===1?'60':k===2?'30':'18');ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(olsX,olsY,k*25,k*18,-0.5,0,Math.PI*2);ctx.stroke();}
  if(curGeo==='l2'){
    ctx.strokeStyle='#4ecdc4';ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#4ecdc420';ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.fill();
    // L2 solution (projection onto sphere, NOT on axis)
    var angle=Math.atan2(cy-olsY,olsX-cx);
    var l2x=cx+radius*Math.cos(angle);var l2y=cy-radius*Math.sin(angle);
    ctx.fillStyle='#4ecdc4';ctx.beginPath();ctx.arc(l2x,l2y,6,0,Math.PI*2);ctx.fill();
    dashedLine(ctx,olsX,olsY,l2x,l2y,'#fbbf2480',[4,4]);
    T(ctx,'L2 solution',l2x-14,l2y+18,'#4ecdc4',9,'700');
    T(ctx,'sphere \u2014 no corners',cx,cy+radius+16,'#4ecdc4',9,'700');
    T(ctx,'constraint: ||w||\u00B2 \u2264 r',cx,cy-radius-12,'#4ecdc4',8,'400');
    document.getElementById('iGeo').innerHTML='<span class="hl2">L2 constraint region is a SPHERE.</span> The optimal solution is projected onto the sphere surface.<br>The projection almost <span class="hl5">never lands on an axis</span> \u2014 so weights are <span class="hl5">rarely exactly zero.</span><br>L2 SHRINKS all weights proportionally. Small weights get small, large get smaller. None become exactly 0.';
  } else if(curGeo==='l1'){
    // Diamond
    ctx.strokeStyle='#ff6b35';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(cx,cy-radius);ctx.lineTo(cx+radius,cy);ctx.lineTo(cx,cy+radius);ctx.lineTo(cx-radius,cy);ctx.closePath();ctx.stroke();
    ctx.fillStyle='#ff6b3518';ctx.beginPath();ctx.moveTo(cx,cy-radius);ctx.lineTo(cx+radius,cy);ctx.lineTo(cx,cy+radius);ctx.lineTo(cx-radius,cy);ctx.closePath();ctx.fill();
    // L1 solution hits a corner on an axis
    var l1x=cx+radius;var l1y=cy;
    ctx.fillStyle='#ff6b35';ctx.beginPath();ctx.arc(l1x,l1y,6,0,Math.PI*2);ctx.fill();
    dashedLine(ctx,olsX,olsY,l1x,l1y,'#fbbf2480',[4,4]);
    T(ctx,'L1 solution (w\u2082=0)',l1x+5,l1y+18,'#ff6b35',9,'700','left');
    T(ctx,'diamond \u2014 corners on axes',cx,cy+radius+16,'#ff6b35',9,'700');
    T(ctx,'constraint: |w\u2081|+|w\u2082| \u2264 r',cx,cy-radius-12,'#ff6b35',8,'400');
    document.getElementById('iGeo').innerHTML='<span class="hl">L1 constraint region is a DIAMOND.</span> The corners lie <span class="hl">exactly on the coordinate axes</span>.<br>Projecting the OLS solution tends to hit a corner \u2014 one weight becomes <span class="hl4">exactly 0</span> (sparsity!).<br>As dimensions increase, there are more corners relative to the surface area. L1 becomes <span class="hl4">increasingly sparse</span> in high dimensions.';
  } else {
    // Elastic Net = rounded diamond
    ctx.strokeStyle='#c084fc';ctx.lineWidth=2.5;ctx.beginPath();
    for(var a=0;a<=360;a+=2){var ar=a*Math.PI/180;var p=1.5;var sc=Math.pow(Math.pow(Math.abs(Math.cos(ar)),p)+Math.pow(Math.abs(Math.sin(ar)),p),1/p);var ex=cx+radius/sc*Math.cos(ar);var ey=cy+radius/sc*Math.sin(ar);if(a===0)ctx.moveTo(ex,ey);else ctx.lineTo(ex,ey);}ctx.closePath();ctx.stroke();
    ctx.fillStyle='#c084fc15';ctx.beginPath();for(var a2=0;a2<=360;a2+=2){var ar2=a2*Math.PI/180;var sc2=Math.pow(Math.pow(Math.abs(Math.cos(ar2)),1.5)+Math.pow(Math.abs(Math.sin(ar2)),1.5),1/1.5);var ex2=cx+radius/sc2*Math.cos(ar2);var ey2=cy+radius/sc2*Math.sin(ar2);if(a2===0)ctx.moveTo(ex2,ey2);else ctx.lineTo(ex2,ey2);}ctx.closePath();ctx.fill();
    var enAngle=Math.atan2(cy-olsY,olsX-cx);var ensc=Math.pow(Math.pow(Math.abs(Math.cos(enAngle)),1.5)+Math.pow(Math.abs(Math.sin(enAngle)),1.5),1/1.5);
    var enx=cx+radius/ensc*Math.cos(enAngle);var eny=cy+radius/ensc*Math.sin(enAngle);
    ctx.fillStyle='#c084fc';ctx.beginPath();ctx.arc(enx,eny,6,0,Math.PI*2);ctx.fill();
    dashedLine(ctx,olsX,olsY,enx,eny,'#fbbf2480',[4,4]);
    T(ctx,'Elastic Net solution',enx-16,eny+18,'#c084fc',9,'700');
    T(ctx,'rounded diamond \u2014 some corners',cx,cy+radius+16,'#c084fc',9,'700');
    document.getElementById('iGeo').innerHTML='<span class="hl6">Elastic Net = L1 + L2.</span> Rounded diamond: <span class="hl4">some sparsity</span> (from L1 corners) <span class="hl2">+ stability for correlated features</span> (from L2 rounding).<br>L1 alone: if two features are correlated, it arbitrarily picks one and zeroes the other. Elastic Net picks both or drops both \u2014 more stable selection.';
  }
}

function renderShrink(){
  var lam=parseFloat(document.getElementById('slLam').value);
  document.getElementById('vLam').textContent=rnd(lam,2);
  var r=C('cvShrink');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=38;var plotW=W-PAD-14;var plotH=H-PAD-20;
  axis(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40');axis(ctx,PAD,14,PAD,H-PAD,'#2d2d40');
  T(ctx,'Unregularised weight w',W/2,H-10,'#52525b',9,'400');T(ctx,'Regularised weight',PAD+28,16,'#52525b',9,'400');
  var xmin=-3;var xmax=3;
  var xToC=function(x){return PAD+(x-xmin)/(xmax-xmin)*plotW;};
  var yToC=function(y){return H-PAD-Math.min(Math.max((y-xmin)/(xmax-xmin),0),1)*plotH;};
  // identity line
  dashedLine(ctx,xToC(xmin),yToC(xmin),xToC(xmax),yToC(xmax),'#3f3f46',[4,4]);
  T(ctx,'identity',xToC(2.5),yToC(2.5)-10,'#3f3f46',8,'400');
  // L2: w_reg = w/(1+lam) -- shrinks proportionally
  var l2Pts=[];for(var i=0;i<=80;i++){var w=-3+i*6/80;l2Pts.push([xToC(w),yToC(w/(1+lam))]);}
  curve(ctx,l2Pts,'#4ecdc4',2.5);T(ctx,'L2',xToC(2.8),yToC(2.8/(1+lam))-10,'#4ecdc4',9,'700');
  // L1: soft thresholding -- drives to 0
  var l1Pts=[];for(var i2=0;i2<=80;i2++){var w2=-3+i2*6/80;var v=w2>0?Math.max(0,w2-lam):Math.min(0,w2+lam);l1Pts.push([xToC(w2),yToC(v)]);}
  curve(ctx,l1Pts,'#ff6b35',2.5);T(ctx,'L1 (sparse)',xToC(2.8),yToC(Math.max(0,2.8-lam))-14,'#ff6b35',9,'700');
  // zero region for L1
  var zStart=xToC(-lam);var zEnd=xToC(lam);var zY=yToC(0);
  fillB(ctx,zStart,zY-4,zEnd-zStart,8,2,'#ff6b3530','#ff6b3560',1);
  T(ctx,'dead zone',xToC(0),yToC(0)-14,'#ff6b35',8,'700');
  T(ctx,'0',xToC(0),H-PAD+12,'#3f3f46',8,'400');
  renderPrior();
}

function renderPrior(){
  var r=C('cvPrior');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=30;var plotW=W-PAD-14;var plotH=H-PAD-20;
  axis(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40');axis(ctx,PAD,14,PAD,H-PAD,'#2d2d40');
  T(ctx,'weight w',W/2,H-8,'#52525b',8,'400');T(ctx,'p(w)',PAD+16,16,'#52525b',8,'400');
  var N3=100;
  var gauss=[],lap=[];
  for(var i=0;i<N3;i++){var w=-3+i*6/(N3-1);gauss.push(Math.exp(-w*w/2)/Math.sqrt(2*Math.PI));lap.push(0.5*Math.exp(-Math.abs(w)));}
  var gmax=Math.max.apply(null,gauss);
  var gpts=gauss.map(function(v,i){return[PAD+i/N3*plotW,H-PAD-(v/gmax)*plotH];});
  var lpts=lap.map(function(v,i){return[PAD+i/N3*plotW,H-PAD-(v/gmax)*plotH];});
  curve(ctx,gpts,'#4ecdc4',2);T(ctx,'Gaussian (L2)',gpts[75][0],gpts[75][1]-10,'#4ecdc4',9,'700');
  curve(ctx,lpts,'#ff6b35',2);T(ctx,'Laplace (L1)',lpts[50][0]+30,lpts[50][1]-14,'#ff6b35',9,'700');
  T(ctx,'Sharp peak \u2192 sparsity',gpts[50][0],H-PAD-plotH*0.6,'#ff6b3560',8,'400');
  document.getElementById('iPrior').innerHTML='<span class="hl2">Gaussian prior (L2):</span> smooth peak, tails fall off quickly.<br><span class="hl">Laplace prior (L1):</span> <span class="hl">sharp peak</span> at 0 (strongly promotes sparsity) + heavier tails (allows occasional large weights).';
}

// ======================== TAB 2: DROPOUT ========================
var dropRNG=42;
function pseudoRand(){dropRNG=(dropRNG*1664525+1013904223)&0xffffffff;return(dropRNG>>>0)/4294967296;}

function renderDrop(){
  var p=parseFloat(document.getElementById('slDrop').value);
  document.getElementById('vDrop').textContent=rnd(p,2);
  dropRNG=12345;
  var activations=[0.8,0.5,0.3,0.9,0.6,0.4,0.7,0.2,0.85,0.45];
  var N4=activations.length;
  var r=C('cvDrop');var ctx=r.ctx;var W=r.W;var H=r.H;
  var colW=W/3;var colPad=30;var neuronR=18;var neuronSpacing=(H-80)/(N4+1);
  var trainX=colW/2;var infX=W-colW/2;var midX=W/2;
  T(ctx,'TRAINING (p='+rnd(p,2)+')',trainX,22,'#4ecdc4',9,'800');
  T(ctx,'INFERENCE',infX,22,'#ff6b35',9,'800');
  T(ctx,'activations',midX,22,'#52525b',8,'400');
  var masks=activations.map(function(){return pseudoRand()>p;});
  var scale=1/(1-p);
  var trainOut=activations.map(function(a,i){return masks[i]?a*scale:0;});
  var trainSum=trainOut.reduce(function(s,v){return s+v;},0);
  var infSum=activations.reduce(function(s,v){return s+v;},0);
  for(var i3=0;i3<N4;i3++){
    var ny=44+i3*neuronSpacing+neuronSpacing/2;
    var a=activations[i3];var kept=masks[i3];
    // Training neuron
    var tCol=kept?'#4ecdc4':'#1e1e2e';var tFill=kept?'#4ecdc420':'#09090f';
    fillB(ctx,trainX-neuronR,ny-neuronR,neuronR*2,neuronR*2,neuronR,tFill,tCol,kept?2:1);
    T(ctx,rnd(a,1),trainX,ny,tCol,9,'700');
    if(!kept){T(ctx,'x',trainX+neuronR-6,ny-neuronR+6,'#ef4444',9,'800');}
    // Arrow + scaled output
    var tOut=trainOut[i3];
    ctx.strokeStyle=kept?'#4ecdc460':'#1e1e2e';ctx.lineWidth=kept?1.5:0.5;
    ctx.beginPath();ctx.moveTo(trainX+neuronR,ny);ctx.lineTo(midX-20,ny);ctx.stroke();
    if(kept){T(ctx,rnd(tOut,2),midX,ny,'#fbbf24',8,'700');}else{T(ctx,'0',midX,ny,'#3f3f46',8,'400');}
    ctx.strokeStyle='#ff6b3560';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(midX+20,ny);ctx.lineTo(infX-neuronR,ny);ctx.stroke();
    // Inference neuron (all kept)
    fillB(ctx,infX-neuronR,ny-neuronR,neuronR*2,neuronR*2,neuronR,'#ff6b3420','#ff6b35',1.5);
    T(ctx,rnd(a,1),infX,ny,'#ff6b35',9,'700');
  }
  // Sums
  T(ctx,'sum='+rnd(trainSum,2),trainX,H-20,'#4ecdc4',9,'700');
  T(ctx,'sum='+rnd(infSum,2),infX,H-20,'#ff6b35',9,'700');
  T(ctx,'x'+rnd(scale,2)+'(scale)',midX,H-20,'#fbbf24',8,'700');
  document.getElementById('iDrop').innerHTML='<span class="hl2">Inverted dropout:</span> kept neurons scaled by 1/(1-p)=<span class="hl3">'+rnd(scale,2)+'</span>.<br>Training sum \u2248 inference sum \u2014 no scale mismatch!<br><span class="hl5">Common bug: forget model.eval() at inference</span> \u2014 mask stays active and predictions are wrong.<br>'+rnd(p*100,0)+'% of neurons dropped. '+(p>0.5?'<span class="hl5">Heavy dropout \u2014 use only for large FC layers.</span>':p<0.15?'<span class="hl4">Light dropout \u2014 good for conv/transformer layers.</span>':'<span class="hl3">Moderate dropout \u2014 standard for MLP hidden layers.</span>');
}

function renderMC(){
  var p2=parseFloat(document.getElementById('slMC').value);
  var T2=parseInt(document.getElementById('slT').value);
  document.getElementById('vMC').textContent=rnd(p2,2);
  document.getElementById('vT').textContent=T2;
  dropRNG=99;
  // Simulate T forward passes on a single input
  var truePred=0.72;
  var preds=[];
  for(var i=0;i<T2;i++){var noise=(pseudoRand()-0.5)*p2*1.4;preds.push(Math.max(0.01,Math.min(0.99,truePred+noise)));}
  var mean=preds.reduce(function(s,v){return s+v;},0)/preds.length;
  var variance=preds.reduce(function(s,v){return s+(v-mean)*(v-mean);},0)/preds.length;
  var std=Math.sqrt(variance);
  var r=C('cvMC');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=36;var plotW=W-PAD-20;var plotH=H-PAD-30;
  var xmin=0;var xmax=1;
  // histogram
  var bins=20;var hist=new Array(bins).fill(0);
  preds.forEach(function(v){var b=Math.min(bins-1,Math.floor((v-xmin)/(xmax-xmin)*bins));hist[b]++;});
  var maxH=Math.max.apply(null,hist);
  var bW=plotW/bins;
  hist.forEach(function(c,i){
    var bx=PAD+i*bW;var bh=(c/maxH)*plotH*0.85;var by=H-PAD-bh;
    fillB(ctx,bx+1,by,bW-2,bh,2,'#4ecdc430','#4ecdc480',1);
  });
  axis(ctx,PAD,H-PAD,W-14,H-PAD,'#2d2d40');
  T(ctx,'Predicted Probability',W/2,H-10,'#52525b',9,'400');
  // mean line
  var mx=PAD+(mean-xmin)/(xmax-xmin)*plotW;
  dashedLine(ctx,mx,20,mx,H-PAD,'#ff6b35',[4,4]);T(ctx,'mean='+rnd(mean,3),mx,18,'#ff6b35',9,'700');
  // uncertainty band
  var lo=PAD+Math.max(0,(mean-2*std-xmin)/(xmax-xmin))*plotW;
  var hi=PAD+Math.min(plotW,(mean+2*std-xmin)/(xmax-xmin))*plotW;
  fillB(ctx,lo,H-PAD-plotH*0.5,hi-lo,plotH*0.5,0,'#fbbf2415',null,0);
  ctx.strokeStyle='#fbbf2450';ctx.lineWidth=1;ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(lo,20);ctx.lineTo(lo,H-PAD);ctx.stroke();
  ctx.beginPath();ctx.moveTo(hi,20);ctx.lineTo(hi,H-PAD);ctx.stroke();
  ctx.setLineDash([]);
  T(ctx,'\u00B12\u03C3',W/2,H-PAD-plotH*0.25,'#fbbf2480',8,'700');
  T(ctx,'T='+T2+' forward passes',PAD+70,20,'#52525b',8,'400','left');
  document.getElementById('iMC').innerHTML='<span class="hl">MC Dropout (Gal & Ghahramani 2016):</span> keep dropout ON at inference, run T passes.<br>Mean=<span class="hl3">'+rnd(mean,3)+'</span> Std=<span class="hl5">'+rnd(std,3)+'</span> \u2192 epistemic uncertainty estimate.<br><span class="hl4">Low variance</span> = model confident. <span class="hl5">High variance</span> = uncertain / OOD input.<br>Used in: medical imaging, active learning, autonomous systems.';
}

// ======================== TAB 3: DROPPATH ========================
var curDP='none';
function selDP(k){curDP=k;['dpN','dpSD','dpDP'].forEach(function(id){document.getElementById(id).classList.remove('on');});document.getElementById(k==='none'?'dpN':k==='std'?'dpSD':'dpDP').classList.add('on');renderDP();}

function renderDP(){
  var r=C('cvDP');var ctx=r.ctx;var W=r.W;var H=r.H;
  var x=W/2;var topY=24;var botY=H-24;var midY=(topY+botY)/2;
  var layerH=90;
  // Input
  T(ctx,'x (input)',x,topY,'#e4e4e7',10,'800');
  // Split: skip path and main path
  var mainX=x-70;var skipX=x+70;
  var blockTop=topY+30;var blockBot=blockTop+layerH;
  // skip connection line
  ctx.strokeStyle='#4ade80';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(skipX,topY+18);ctx.lineTo(skipX,blockBot+20);ctx.stroke();
  T(ctx,'skip',skipX+10,midY,'#4ade80',9,'700','left');
  // Main path arrow down
  ctx.strokeStyle='#4ecdc4';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(mainX,topY+18);ctx.lineTo(mainX,blockTop-2);ctx.stroke();
  // Block F(x)
  var bx=mainX-45;var by=blockTop;var bw=90;var bh=layerH;
  var blockBorder='#4ecdc4';var blockAlpha='30';
  if(curDP==='std'){blockBorder='#fbbf24';}
  if(curDP==='drop'){blockBorder='#c084fc';}
  fillB(ctx,bx,by,bw,bh,8,'#'+blockBorder.slice(1)+(blockAlpha)+'00'.slice(0,2-blockAlpha.length+2),blockBorder,2);
  T(ctx,'F(x)',mainX,by+22,'#e4e4e7',11,'800');
  if(curDP==='std'){
    // neurons with some x'd out
    var nx=[mainX-24,mainX-8,mainX+8,mainX+24];
    nx.forEach(function(nxi,i){fillB(ctx,nxi-9,by+40,18,18,9,'#fbbf2420',i%2===0?'#3f3f46':'#fbbf24',1.5);T(ctx,i%2===0?'0':'n',nxi,by+49,i%2===0?'#3f3f46':'#fbbf24',9,'700');});
    T(ctx,'neuron',mainX,by+68,'#fbbf24',8,'400');T(ctx,'dropout',mainX,by+80,'#fbbf24',8,'400');
  } else if(curDP==='drop'){
    // b * F(x) annotation
    T(ctx,'b~Bern(1-p)',mainX,by+48,'#c084fc',9,'700');T(ctx,'b=0: skip ENTIRE block',mainX,by+64,'#c084fc',8,'400');T(ctx,'b=1: compute normally',mainX,by+76,'#c084fc',8,'400');
  } else {
    T(ctx,'Linear + ReLU',mainX,by+48,'#4ecdc4',9,'700');T(ctx,'(always computed)',mainX,by+64,'#4ecdc4',8,'400');
  }
  // Arrow from block to merge
  ctx.strokeStyle=curDP==='drop'?'#c084fc':'#4ecdc4';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(mainX,blockBot+2);ctx.lineTo(mainX,blockBot+20);ctx.stroke();
  if(curDP==='drop'){T(ctx,'x b',mainX-28,blockBot+12,'#c084fc',9,'700');}
  // Merge point
  var mergeY=blockBot+22;
  ctx.strokeStyle='#e4e4e780';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(mainX,mergeY);ctx.lineTo(x,mergeY+14);ctx.stroke();
  ctx.beginPath();ctx.moveTo(skipX,mergeY);ctx.lineTo(x,mergeY+14);ctx.stroke();
  // + node
  fillB(ctx,x-14,mergeY+14,28,22,14,'#111118','#e4e4e780',1.5);T(ctx,'+',x,mergeY+25,'#e4e4e7',14,'700');
  // Output formula
  var formula=curDP==='none'?'output = x + F(x)':curDP==='std'?'output = x + drop(F(x))':'output = x + b\u00B7F(x)';
  var fCol=curDP==='none'?'#4ecdc4':curDP==='std'?'#fbbf24':'#c084fc';
  T(ctx,formula,x,mergeY+50,fCol,10,'800');
  // Output arrow
  ctx.strokeStyle='#e4e4e780';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(x,mergeY+62);ctx.lineTo(x,botY-10);ctx.stroke();
  T(ctx,'output',x,botY,'#e4e4e7',10,'800');
  var infoMap={none:'<span class="hl2">Normal residual:</span> output = x + F(x). The skip connection always carries the full gradient backwards. Both F(x) and x always computed.',std:'<span class="hl3">Standard dropout:</span> individual neurons in F(x) are zeroed. But the <span class="hl5">skip path (x) always passes through</span>. The network always processes the full depth \u2014 never learns to work without specific layers.',drop:'<span class="hl6">DropPath:</span> b~Bernoulli(1\u2212p). When b=0, the <span class="hl6">entire F(x) branch is skipped</span>. Input passes unchanged (skip only). Network trains as a random depth ensemble. Computation saved when block is skipped!'};
  document.getElementById('iDP').innerHTML=infoMap[curDP];
}

function renderDPS(){
  var pmax=parseFloat(document.getElementById('slPmax').value);
  var L=parseInt(document.getElementById('slLayers').value);
  document.getElementById('vPmax').textContent=rnd(pmax,2);
  document.getElementById('vLayers').textContent=L;
  var r=C('cvDPS');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=40;var plotW=W-PAD-20;var plotH=H-PAD-40;
  axis(ctx,PAD,H-PAD,W-14,H-PAD,'#2d2d40');axis(ctx,PAD,20,PAD,H-PAD,'#2d2d40');
  T(ctx,'Layer (1=closest to input)',W/2,H-10,'#52525b',9,'400');T(ctx,'Drop prob',PAD+20,18,'#52525b',9,'400');
  var bW=Math.min(26,plotW/L-3);var spacing=plotW/L;
  var expDepth=0;
  for(var l=1;l<=L;l++){
    var pl=l/L*pmax;
    expDepth+=(1-pl);
    var bh=pl*plotH;var bx=PAD+(l-1)*spacing+(spacing-bW)/2;var by=H-PAD-bh;
    var intensity=l/L;
    var rr=Math.round(74+intensity*165);var gg=Math.round(222-intensity*188);var bb=Math.round(128-intensity*128);
    var col='rgb('+rr+','+gg+','+bb+')';
    fillB(ctx,bx,by,bW,bh,2,col+'40',col,1.5);
    ctx.fillStyle=col;ctx.font='8px JetBrains Mono,monospace';ctx.textAlign='center';ctx.textBaseline='bottom';
    ctx.fillText(rnd(pl,2),bx+bW/2,by-2);
    ctx.fillStyle='#3f3f46';ctx.textBaseline='top';ctx.fillText('L'+l,bx+bW/2,H-PAD+4);
  }
  T(ctx,'0',PAD-4,H-PAD,'#3f3f46',8,'400','right');T(ctx,rnd(pmax,2),PAD-4,20,'#3f3f46',8,'400','right');
  dashedLine(ctx,PAD,H-PAD-pmax*plotH,W-14,H-PAD-pmax*plotH,'#c084fc50',[5,4]);
  T(ctx,'p_max='+rnd(pmax,2),W-15,H-PAD-pmax*plotH-8,'#c084fc',8,'700','right');
  T(ctx,'p_l = (l/L) x p_max',W/2,28,'#e4e4e7',9,'700');
  document.getElementById('iDPS').innerHTML='Early layers (close to input): <span class="hl4">rarely dropped</span> \u2014 they compute basic features every sample needs.<br>Deep layers: <span class="hl6">drop probability = p_max=' + rnd(pmax,2)+'</span>. Task-specific features can be skipped.<br>Expected active depth: <span class="hl3">'+rnd(expDepth,1)+' / '+L+' layers</span> per forward pass.<br>Also <span class="hl4">speeds up training</span> by ~'+rnd((1-pmax/2)*100,0)+'% (skipped layers need no compute).';
}

// ======================== TAB 4: EARLY STOPPING ========================
function renderES(){
  var patience=parseInt(document.getElementById('slPat').value);
  var ovf=parseFloat(document.getElementById('slOvf').value);
  document.getElementById('vPat').textContent=patience;
  document.getElementById('vOvf').textContent=rnd(ovf,2);
  var r=C('cvES');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=38;var plotW=W-PAD-14;var plotH=H-PAD-20;
  axis(ctx,PAD,H-PAD,W-14,H-PAD,'#2d2d40');axis(ctx,PAD,14,PAD,H-PAD,'#2d2d40');
  T(ctx,'Epoch',W/2,H-8,'#52525b',9,'400');T(ctx,'Loss',PAD+16,18,'#52525b',9,'400');
  var N5=80;
  var trainPts2=[],valPts2=[];var bestIdx=0;var bestVal=9999;
  for(var i=0;i<N5;i++){
    var t=i/N5;
    var tL=0.9*Math.exp(-t*3.5)+0.06;
    var vBase=0.7*Math.exp(-t*4)+0.2;
    var vUp=ovf*0.55*(1-Math.exp(-Math.max(0,t-0.25)*5));
    var vL=vBase+vUp;
    trainPts2.push([PAD+t*plotW,H-PAD-Math.min(tL/1.2,1)*plotH]);
    valPts2.push([PAD+t*plotW,H-PAD-Math.min(vL/1.2,1)*plotH]);
    if(vL<bestVal){bestVal=vL;bestIdx=i;}
  }
  curve(ctx,trainPts2,'#4ecdc4',2.5);T(ctx,'train',trainPts2[70][0]-5,trainPts2[70][1]-10,'#4ecdc4',9,'700');
  curve(ctx,valPts2,'#ff6b35',2.5);T(ctx,'val',valPts2[55][0]+10,valPts2[55][1]-10,'#ff6b35',9,'700');
  // Best checkpoint
  var bx=valPts2[bestIdx][0];var by=valPts2[bestIdx][1];
  ctx.fillStyle='#4ade80';ctx.beginPath();ctx.arc(bx,by,7,0,Math.PI*2);ctx.fill();
  T(ctx,'best',bx,by-14,'#4ade80',9,'800');
  // Patience window
  var stopIdx=Math.min(N5-1,bestIdx+patience);
  var sx=valPts2[stopIdx][0];
  fillB(ctx,bx,16,sx-bx,H-PAD-16,0,'#fbbf24',null,0);
  ctx.fillStyle='#fbbf2418';ctx.fillRect(bx,16,sx-bx,H-PAD-16);
  dashedLine(ctx,sx,14,sx,H-PAD,'#fbbf24',[4,4]);
  T(ctx,'stop',sx,20,'#fbbf24',9,'700');
  T(ctx,'patience='+patience+' epochs',bx+(sx-bx)/2,(H-PAD+16)/2,'#fbbf2490',8,'700');
  dashedLine(ctx,bx,14,bx,H-PAD,'#4ade80',[4,4]);
  document.getElementById('iES').innerHTML='<span class="hl4">Best checkpoint at epoch '+(bestIdx+1)+'</span> (lowest val loss).<br>Training continues for <span class="hl3">patience='+patience+'</span> epochs after the best, then stops.<br><span class="hl5">Restore best weights</span> \u2014 the stopping epoch is NOT the best epoch!<br>'+(patience<5?'<span class="hl5">Very short patience. Risk: stopping due to val loss noise.</span>':patience>20?'<span class="hl3">Long patience. Tolerates noisy val loss, but wastes compute.</span>':'<span class="hl4">Good patience range for most tasks.</span>');
}

function renderESEq(){
  var steps=parseInt(document.getElementById('slSteps').value);
  document.getElementById('vSteps').textContent=steps;
  var r=C('cvESEq');var ctx=r.ctx;var W=r.W;var H=r.H;
  var PAD=40;var plotW=W-PAD-20;var plotH=H-PAD-30;
  axis(ctx,PAD,H-PAD,W-14,H-PAD,'#2d2d40');axis(ctx,PAD,20,PAD,H-PAD,'#2d2d40');
  T(ctx,'Feature eigenvalue strength',W/2,H-10,'#52525b',9,'400');T(ctx,'Weight magnitude',PAD+24,18,'#52525b',9,'400');
  // For different eigenvalues, show what early stopping does vs L2
  var eigVals=[0.1,0.3,0.6,1.0,2.0,4.0];var eta=0.01;
  var colors2=['#4ecdc4','#38bdf8','#4ade80','#fbbf24','#fb923c','#ef4444'];
  var xmax2=eigVals.length-1;
  // Equivalent lambda = 1/(eta*t)
  var equivLam=1/(eta*steps);
  eigVals.forEach(function(ev,i){
    var x2=PAD+i/(eigVals.length-1)*plotW;
    // Weight recovered after t steps of GD from init 0 toward optimal
    // w(t) = w_opt * [1 - (1-eta*ev)^t]
    var wOpt=0.8;// normalise
    var wT=wOpt*(1-Math.pow(Math.max(0,1-eta*ev),steps));
    var wL2=wOpt*(ev/(ev+equivLam));
    var bH=wT*plotH*0.9;var lH=wL2*plotH*0.9;
    var bx2=x2-12;
    fillB(ctx,bx2,H-PAD-bH,12,bH,2,colors2[i]+'40',colors2[i],1.5);
    fillB(ctx,bx2+13,H-PAD-lH,12,lH,2,colors2[i]+'20',colors2[i]+'80',1);
    ctx.fillStyle=colors2[i];ctx.font='8px JetBrains Mono,monospace';ctx.textAlign='center';
    ctx.textBaseline='bottom';ctx.fillText(rnd(ev,1),x2,H-PAD+12);
  });
  // Legend
  fillB(ctx,PAD,14,12,10,2,'#4ecdc440','#4ecdc4',1);T(ctx,'Early stop (t='+steps+')',PAD+60,19,'#4ecdc4',8,'700','left');
  fillB(ctx,PAD+150,14,12,10,2,'#4ecdc420','#4ecdc480',1);T(ctx,'Equiv. L2 (\u03BB=1/\u03B7t='+rnd(equivLam,2)+')',PAD+210,19,'#52525b',8,'700','left');
  document.getElementById('iESEq').innerHTML='Early stopping \u2248 L2 with <span class="hl3">\u03BB = 1/(\u03B7\u00B7t) = '+rnd(equivLam,2)+'</span>  (for linear models, Bishop 1995).<br>Small t (stop early) \u2194 large \u03BB (strong L2). Large t \u2194 weak L2 (converge to OLS).<br>Strong features (high eigenvalue): weight recovers quickly regardless of t.<br><span class="hl4">Weak/noisy features</span> (low eigenvalue): early stopping <span class="hl4">naturally suppresses them</span>.';
}

// ======================== INIT ========================
window.addEventListener('load',function(){renderLoss();renderBV();});
</script>
</body>
</html>"""

REGULARISATION_VISUAL_HEIGHT = 2250