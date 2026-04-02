HP_TUNING_VISUAL_HTML = r"""<!DOCTYPE html>
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
<h2>&#x1F3DB; Hyperparameter Tuning</h2>
<p class="sub">Grid &middot; Random &middot; Bayesian &middot; LR Finder &middot; Hyperband &middot; LR Schedules &middot; Strategy</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Search Methods</button>
  <button class="tab" onclick="showTab(1)">Bayesian Opt</button>
  <button class="tab" onclick="showTab(2)">LR Finder</button>
  <button class="tab" onclick="showTab(3)">Hyperband / PBT</button>
  <button class="tab" onclick="showTab(4)">LR Schedules</button>
  <button class="tab" onclick="showTab(5)">Strategy Guide</button>
</div>

<!-- TAB 0: SEARCH METHODS -->
<div id="tab0" class="panel active">
<div class="g2">
<div class="card">
  <h3>&#9312; Grid vs Random Search</h3>
  <canvas id="cvSearch" width="420" height="280"></canvas>
  <div class="brow">
    <button class="s active" id="srBtn_grid"   onclick="selSearch('grid')">Grid Search</button>
    <button class="s" id="srBtn_random" onclick="selSearch('random')">Random Search</button>
  </div>
  <div class="row"><label>Num trials</label>
    <input type="range" id="slTrials" min="4" max="64" step="4" value="25" oninput="renderSearch()">
    <span class="v" id="vTrials">25</span></div>
  <div class="info" id="iSearch">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Log Scale vs Linear Scale Search</h3>
  <canvas id="cvScale" width="420" height="280"></canvas>
  <div class="brow">
    <button class="s active" id="scBtn_log"    onclick="selScale('log')">Log Scale</button>
    <button class="s" id="scBtn_linear" onclick="selScale('linear')">Linear Scale</button>
  </div>
  <div class="info" id="iScale">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Hyperparameter Sensitivity &mdash; fANOVA Result (typical)</h3>
  <canvas id="cvSens" width="860" height="150"></canvas>
</div>
</div>

<!-- TAB 1: BAYESIAN OPT -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Bayesian Optimisation: Surrogate + Acquisition</h3>
  <canvas id="cvBO" width="420" height="290"></canvas>
  <div class="row"><label>Trial step</label>
    <input type="range" id="slBO" min="2" max="12" step="1" value="4" oninput="renderBO()">
    <span class="v" id="vBO">4</span></div>
  <div class="info" id="iBO">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; BO Loop: Surrogate Model Refinement</h3>
  <canvas id="cvBOLoop" width="420" height="290"></canvas>
  <div class="info" id="iBOLoop">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; When Bayesian Optimisation Helps &mdash; Method Comparison</h3>
  <table class="tbl">
    <thead><tr><th>Method</th><th>Uses history?</th><th>Parallelisable?</th><th>Best for</th><th>Limitation</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">Grid Search</span></td><td>No</td><td>Yes</td><td>&le;2 HP, cheap evals</td><td>Exponential scaling. Wastes budget on unimportant dims.</td></tr>
      <tr><td><span class="hl3">Random Search</span></td><td>No</td><td>Yes (trivial)</td><td>3-10 HP, few evals</td><td>No learning from previous trials. Inefficient at high counts.</td></tr>
      <tr><td><span class="hl2">Bayesian (GP)</span></td><td>Yes (GP)</td><td>Limited</td><td>3-10 HP, 20-200 evals</td><td>Fails above ~20 dimensions. Inherently sequential.</td></tr>
      <tr><td><span class="hl6">Bayesian (TPE)</span></td><td>Yes (KDE)</td><td>Limited</td><td>Up to 20 HP, 50-500 evals</td><td>Less interpretable than GP. Requires ~20 warm-up trials.</td></tr>
      <tr><td><span class="hl4">BOHB</span></td><td>Yes + Hyperband</td><td>Yes</td><td>Many HP, many GPUs</td><td>Complex setup. Needs Ray Tune or SMAC3.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 2: LR FINDER -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; LR Range Test &mdash; Finding the Optimal LR</h3>
  <canvas id="cvLRF" width="420" height="280"></canvas>
  <div class="brow">
    <button class="s active" id="lrfBtn_sgd"  onclick="selLRF('sgd')">SGD</button>
    <button class="s" id="lrfBtn_adam" onclick="selLRF('adam')">Adam</button>
  </div>
  <div class="info" id="iLRF">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Cyclical Learning Rates (CLR)</h3>
  <canvas id="cvCLR" width="420" height="280"></canvas>
  <div class="row"><label>LR max</label>
    <input type="range" id="slLRmax" min="1" max="10" step="1" value="5" oninput="renderCLR()">
    <span class="v" id="vLRmax">0.05</span></div>
  <div class="row"><label>Cycle length</label>
    <input type="range" id="slCycle" min="1" max="5" step="1" value="3" oninput="renderCLR()">
    <span class="v" id="vCycle">3</span></div>
  <div class="info" id="iCLR">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Log Scale Search for Learning Rate</h3>
  <canvas id="cvLRLog" width="860" height="150"></canvas>
</div>
</div>

<!-- TAB 3: HYPERBAND / PBT -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Successive Halving &mdash; Early Stopping Budget</h3>
  <canvas id="cvSH" width="420" height="280"></canvas>
  <div class="row"><label>Configs n</label>
    <input type="range" id="slSHn" min="4" max="16" step="4" value="8" oninput="renderSH()">
    <span class="v" id="vSHn">8</span></div>
  <div class="info" id="iSH">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Population-Based Training (PBT)</h3>
  <canvas id="cvPBT" width="420" height="280"></canvas>
  <div class="row"><label>Population</label>
    <input type="range" id="slPop" min="4" max="12" step="2" value="6" oninput="renderPBT()">
    <span class="v" id="vPop">6</span></div>
  <div class="info" id="iPBT">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Hyperband: Multiple Brackets</h3>
  <canvas id="cvHB" width="860" height="155"></canvas>
</div>
</div>

<!-- TAB 4: LR SCHEDULES -->
<div id="tab4" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Learning Rate Schedule Comparison</h3>
  <canvas id="cvSched" width="420" height="280"></canvas>
  <div class="brow">
    <button class="s active" id="schBtn_warmcos" onclick="selSched('warmcos')">Warmup+Cosine</button>
    <button class="s" id="schBtn_step"    onclick="selSched('step')">Step Decay</button>
    <button class="s" id="schBtn_sgdr"    onclick="selSched('sgdr')">SGDR (restarts)</button>
    <button class="s" id="schBtn_plateau" onclick="selSched('plateau')">Reduce Plateau</button>
  </div>
  <div class="info" id="iSched">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Warmup: Why Transformers Need It</h3>
  <canvas id="cvWarmup" width="420" height="280"></canvas>
  <div class="row"><label>Warmup steps</label>
    <input type="range" id="slWarmup" min="0" max="20" step="1" value="5" oninput="renderWarmup()">
    <span class="v" id="vWarmup">5%</span></div>
  <div class="info" id="iWarmup">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Schedule Selection Guide</h3>
  <table class="tbl">
    <thead><tr><th>Schedule</th><th>Shape</th><th>Best for</th><th>Avoid when</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">Warmup + Cosine</span></td><td>Ramp up, smooth decay</td><td>Transformers, ViT, GPT, BERT. Modern standard.</td><td>Step-based optimisers. When total steps unknown.</td></tr>
      <tr><td><span class="hl2">SGDR (warm restarts)</span></td><td>Cyclic cosine</td><td>Ensemble from snapshots. Escaping sharp minima.</td><td>Very long training where restarts disrupt fine-tuning.</td></tr>
      <tr><td><span class="hl3">Step Decay</span></td><td>Staircase</td><td>ResNets (pre-2019), well-understood tasks.</td><td>Transformers (abrupt changes disrupt Adam's v estimates).</td></tr>
      <tr><td><span class="hl6">ReduceLROnPlateau</span></td><td>Adaptive steps</td><td>When total steps unknown. Classical ML.</td><td>When cosine annealing is applicable (use that instead).</td></tr>
      <tr><td><span class="hl5">Constant LR</span></td><td>Flat</td><td>Debugging only. Quick tests.</td><td>Final production models. Always schedule.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 5: STRATEGY GUIDE -->
<div id="tab5" class="panel">
<div class="card">
  <h3>&#9312; Hyperparameter Priority Tiers</h3>
  <canvas id="cvTiers" width="860" height="200"></canvas>
</div>
<div class="g2">
<div class="card">
  <h3>&#9313; Three-Phase Tuning Protocol</h3>
  <canvas id="cvPhase" width="420" height="240"></canvas>
  <div class="info" id="iPhase">Loading...</div>
</div>
<div class="card">
  <h3>&#9314; Algorithm Selection</h3>
  <table class="tbl">
    <thead><tr><th>Situation</th><th>Algorithm</th></tr></thead>
    <tbody>
      <tr><td>1&ndash;2 HP, cheap evals</td><td><span class="hl5">Grid search</span> (exhaustive)</td></tr>
      <tr><td>3&ndash;10 HP, &lt;20 evals</td><td><span class="hl3">Random search</span> + LR finder</td></tr>
      <tr><td>3&ndash;10 HP, 20&ndash;200 evals</td><td><span class="hl2">Bayesian (Optuna TPE)</span> + ASHA</td></tr>
      <tr><td>&gt;10 HP or noisy objective</td><td><span class="hl3">Random search</span> + ASHA / Hyperband</td></tr>
      <tr><td>Many GPUs, 50&ndash;500 evals</td><td><span class="hl4">BOHB</span> via Ray Tune</td></tr>
      <tr><td>Very expensive evals (&lt;10 trials)</td><td><span class="hl">LR finder</span> + expert priors + scaling laws</td></tr>
      <tr><td>Parallel, adaptive schedule</td><td><span class="hl6">Population-Based Training</span></td></tr>
      <tr><td>LR schedule only</td><td><span class="hl">LR finder</span> + cosine annealing</td></tr>
    </tbody>
  </table>
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

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderSearch();renderScale();renderSens();}if(i===1){renderBO();renderBOLoop();}if(i===2){renderLRF();renderCLR();renderLRLog();}if(i===3){renderSH();renderPBT();renderHB();}if(i===4){renderSched();renderWarmup();}if(i===5){renderTiers();renderPhase();}}

// ============================================================
// TAB 0: SEARCH METHODS
// ============================================================
var curSearch='grid';
function selSearch(k){curSearch=k;['grid','random'].forEach(function(b){document.getElementById('srBtn_'+b).classList.remove('active');});document.getElementById('srBtn_'+k).classList.add('active');renderSearch();}
function renderSearch(){
  var nTrials=parseInt(document.getElementById('slTrials').value);
  document.getElementById('vTrials').textContent=nTrials;
  var ctx=clr('cvSearch',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'LR (important dim)','Weight decay (unimportant)');
  // Simulate: important dim is x, unimportant is y
  // True optimum is around LR=0.003
  function perfFn(lr,wd){var score=Math.exp(-Math.pow(Math.log10(lr)-Math.log10(0.003),2)*4);return score;}
  var bestScore=0,bestX,bestY;
  if(curSearch==='grid'){
    var gridVals=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]; // log-spaced
    var wdVals=[1e-5,1e-4,1e-3,1e-2];
    var shown=0;
    for(var wi=0;wi<wdVals.length&&shown<nTrials;wi++){
      for(var li=0;li<gridVals.length&&shown<nTrials;li++){
        var lr=gridVals[li],wd=wdVals[wi];
        var s=perfFn(lr,wd);
        var px=PAD+(li/5)*pW;
        var py=H-PAD-(wi/3)*pH;
        ctx.fillStyle='#4ecdc4'+Math.round(40+s*200).toString(16).padStart(2,'0');
        ctx.strokeStyle='#4ecdc4';ctx.lineWidth=1;
        ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fill();ctx.stroke();
        if(s>bestScore){bestScore=s;bestX=px;bestY=py;}
        shown++;
      }
    }
    // x-axis labels
    gridVals.slice(0,6).forEach(function(v,i){tx(ctx,v.toExponential(0),PAD+(i/5)*pW,H-PAD+14,'#3f3f46',7,'400');});
    wdVals.forEach(function(v,i){tx(ctx,v.toExponential(0),PAD-6,H-PAD-(i/3)*pH,'#3f3f46',7,'400','right');});
    var uniqueLRs=Math.min(gridVals.length,Math.ceil(nTrials/wdVals.length));
    document.getElementById('iSearch').innerHTML='<span class="hl5">Grid search:</span> '+nTrials+' trials covering <span class="hl5">'+uniqueLRs+' unique LR values</span>.<br>Weight decay is unimportant, so each LR point is tested '+wdVals.length+'x unnecessarily.<br><span class="hl5">Wasted compute:</span> '+Math.round((1-1/wdVals.length)*100)+'% of trials redundant.';
  } else {
    // Random search
    var seed=42;function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
    var lrUnique=new Set();
    for(var i=0;i<nTrials;i++){
      var logLR=-4+rng()*3; // log10(LR) from -4 to -1
      var logWD=-5+rng()*4;
      var lr2=Math.pow(10,logLR),wd2=Math.pow(10,logWD);
      var s2=perfFn(lr2,wd2);
      var px2=PAD+(logLR+4)/3*pW;
      var py2=H-PAD-(logWD+5)/4*pH;
      var alpha=Math.round(40+s2*200).toString(16).padStart(2,'0');
      ctx.fillStyle='#ff6b35'+alpha;ctx.strokeStyle='#ff6b35';ctx.lineWidth=1;
      ctx.beginPath();ctx.arc(px2,py2,4,0,Math.PI*2);ctx.fill();ctx.stroke();
      lrUnique.add(Math.round(logLR*10));
      if(s2>bestScore){bestScore=s2;bestX=px2;bestY=py2;}
    }
    tx(ctx,'1e-4',PAD,H-PAD+14,'#3f3f46',7,'400');
    tx(ctx,'1e-1',W-20,H-PAD+14,'#3f3f46',7,'400');
    document.getElementById('iSearch').innerHTML='<span class="hl4">Random search:</span> '+nTrials+' trials covering ~<span class="hl4">'+Math.min(nTrials,lrUnique.size)+' unique LR regions</span>.<br>Each trial explores a different LR independently. Unimportant WD dim does not waste coverage.<br><span class="hl4">Bergstra-Bengio:</span> with 1 important dim, random search gets '+nTrials+'x more LR coverage vs grid with same budget.';
  }
  // Highlight best
  if(bestX&&bestY){
    ctx.strokeStyle='#fbbf24';ctx.lineWidth=2.5;
    ctx.beginPath();ctx.arc(bestX,bestY,9,0,Math.PI*2);ctx.stroke();
    tx(ctx,'best',bestX,bestY-16,'#fbbf24',8,'800');
  }
}

var curScale='log';
function selScale(k){curScale=k;['log','linear'].forEach(function(b){document.getElementById('scBtn_'+b).classList.remove('active');});document.getElementById('scBtn_'+k).classList.add('active');renderScale();}
function renderScale(){
  var ctx=clr('cvScale',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  // Show density of samples in [1e-4, 1e-1] with 20 points
  var N=20;
  var pts=[];
  if(curScale==='log'){
    for(var i=0;i<N;i++){var lr=Math.pow(10,-4+(i/(N-1))*3);pts.push(lr);}
  } else {
    for(var i=0;i<N;i++){var lr=1e-4+(i/(N-1))*(1e-1-1e-4);pts.push(lr);}
  }
  // Draw number line on log scale
  var logMin=-4,logMax=-1;
  axesPAD(ctx,W,H,PAD,'Learning Rate (log scale)','');
  // Log ticks
  [-4,-3,-2,-1].forEach(function(v){
    var px=PAD+(v-logMin)/(logMax-logMin)*pW;
    ln(ctx,px,H-PAD,px,H-PAD+5,'#3f3f46',1);
    tx(ctx,'1e'+v,px,H-PAD+14,'#3f3f46',7,'400');
    ln(ctx,px,15,px,H-PAD,'#141420',1);
  });
  // Highlight good range around 1e-3 to 3e-3
  var goodMin=Math.log10(8e-4),goodMax=Math.log10(8e-3);
  var gpx1=PAD+(goodMin-logMin)/(logMax-logMin)*pW;
  var gpx2=PAD+(goodMax-logMin)/(logMax-logMin)*pW;
  bx(ctx,gpx1,H-PAD-pH,gpx2-gpx1,pH,0,'#4ade8015',null);
  tx(ctx,'Good LR\nregion',(gpx1+gpx2)/2,H-PAD-pH/2,'#4ade8070',8,'700');

  // Draw sample points
  pts.forEach(function(lr){
    var logLR=Math.log10(lr);
    var px=PAD+(logLR-logMin)/(logMax-logMin)*pW;
    var inGood=lr>=8e-4&&lr<=8e-3;
    ctx.fillStyle=inGood?'#4ade80':'#ff6b35'+50;
    ctx.strokeStyle=inGood?'#4ade80':'#ff6b35';
    ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(px,H-PAD-60,5,0,Math.PI*2);ctx.fill();ctx.stroke();
  });
  var inGoodCount=pts.filter(function(lr){return lr>=8e-4&&lr<=8e-3;}).length;
  var logLabel=curScale==='log'?'Log-uniform samples':'Linear-uniform samples';
  tx(ctx,logLabel,W/2,22,curScale==='log'?'#4ade80':'#ff6b35',10,'800');
  tx(ctx,'Samples in good region: '+inGoodCount+' / '+N+' ('+Math.round(inGoodCount/N*100)+'%)',W/2,H-PAD-40,curScale==='log'?'#4ade80':'#fbbf24',9,'700');
  document.getElementById('iScale').innerHTML=curScale==='log'?
    '<span class="hl4">Log-uniform sampling:</span> equal coverage per order of magnitude.<br>Each decade [1e-4,1e-3], [1e-3,1e-2], [1e-2,1e-1] gets ~equal samples.<br><span class="hl4">'+inGoodCount+'/'+N+'</span> samples fall in the productive range. Always use for LR, weight decay, epsilon.':
    '<span class="hl5">Linear-uniform sampling:</span> most samples cluster at the high end.<br>Range [1e-2, 1e-1] gets ~90% of samples. Range [1e-4, 1e-3] gets ~1%.<br><span class="hl5">'+inGoodCount+'/'+N+'</span> samples in the productive range. Misses most of the interesting space.';
}

function renderSens(){
  var ctx=clr('cvSens',860,150),W=860,H=150;
  var hps=[
    {name:'Learning Rate',pct:62,col:'#ff6b35'},
    {name:'Weight Decay',pct:18,col:'#fbbf24'},
    {name:'LR Schedule',pct:11,col:'#4ecdc4'},
    {name:'Dropout',pct:5,col:'#c084fc'},
    {name:'Beta1 (Adam)',pct:2,col:'#3f3f46'},
    {name:'Beta2 (Adam)',pct:1,col:'#2d2d40'},
    {name:'Epsilon',pct:1,col:'#1e1e2e'},
  ];
  var total=hps.reduce(function(a,b){return a+b.pct;},0);
  var xOff=10;
  hps.forEach(function(hp){
    var bw=(W-20)*hp.pct/total;
    bx(ctx,xOff,20,bw-2,80,4,hp.col+'30',hp.col,bw>60?2:1);
    if(bw>45){
      tx(ctx,hp.name,xOff+bw/2,50,hp.col,bw>80?8:7,'800');
      tx(ctx,hp.pct+'%',xOff+bw/2,66,hp.col,bw>80?10:9,'800');
    } else if(bw>18){
      tx(ctx,hp.pct+'%',xOff+bw/2,60,hp.col,8,'800');
    }
    var tier=hp.pct>20?'Tier 1 - Critical':hp.pct>5?'Tier 2 - Important':'Tier 3 - Fine-tune';
    var tierCol=hp.pct>20?'#ff6b35':hp.pct>5?'#fbbf24':'#3f3f46';
    if(bw>60) tx(ctx,tier,xOff+bw/2,80,tierCol,6,'400');
    xOff+=bw;
  });
  tx(ctx,'fANOVA: % of validation performance variance explained by each HP (typical deep learning result)',W/2,130,'#52525b',8,'400');
  tx(ctx,'Tune these first \u2192',220,10,'#ff6b35',8,'700');
  tx(ctx,'Fix at defaults \u2192',W-150,10,'#3f3f46',8,'400');
}

// ============================================================
// TAB 1: BAYESIAN OPT
// ============================================================
function renderBO(){
  var step=parseInt(document.getElementById('slBO').value);
  document.getElementById('vBO').textContent=step;
  var ctx=clr('cvBO',420,290),W=420,H=290,PAD=44,pW=W-PAD-20,pH=H-PAD-40;
  axesPAD(ctx,W,H,PAD,'Hyperparameter x','Performance f(x)');
  // True underlying function
  function trueFn(x){return 0.5+0.4*Math.sin(x*5)+0.15*Math.cos(x*11);}
  // Draw true function (very faint)
  var truePts=[];
  for(var i=0;i<=200;i++){var x=i/200;truePts.push([PAD+x*pW,H-PAD-trueFn(x)*pH]);}
  ctx.strokeStyle='#ffffff10';ctx.lineWidth=1;ctx.beginPath();truePts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();
  // Observed points up to step
  var obsPoints=[[0.1,trueFn(0.1)],[0.4,trueFn(0.4)],[0.7,trueFn(0.7)],[0.2,trueFn(0.2)],[0.55,trueFn(0.55)],[0.85,trueFn(0.85)],[0.35,trueFn(0.35)],[0.62,trueFn(0.62)],[0.15,trueFn(0.15)],[0.48,trueFn(0.48)],[0.78,trueFn(0.78)],[0.92,trueFn(0.92)]];
  var shown=obsPoints.slice(0,step);
  // Simple surrogate: interpolate GP-like mean and uncertainty
  function surrogateMean(x){
    if(shown.length===0) return 0.5;
    var w=0,sw=0;
    shown.forEach(function(p){var d=Math.abs(x-p[0]);var wi=Math.exp(-d*d*40);w+=wi*p[1];sw+=wi;});
    return sw>0?w/sw:0.5;
  }
  function surrogateStd(x){
    var minD=1;shown.forEach(function(p){minD=Math.min(minD,Math.abs(x-p[0]));});
    return Math.min(0.3,minD*1.5);
  }
  // Draw uncertainty band
  var upperPts=[],lowerPts=[];
  for(var i=0;i<=100;i++){
    var x=i/100;
    var m=surrogateMean(x);var s=surrogateStd(x);
    upperPts.push([PAD+x*pW,H-PAD-(m+s)*pH]);
    lowerPts.push([PAD+x*pW,H-PAD-(m-s)*pH]);
  }
  ctx.fillStyle='#4ecdc415';ctx.beginPath();
  upperPts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});
  lowerPts.slice().reverse().forEach(function(p){ctx.lineTo(p[0],p[1]);});
  ctx.closePath();ctx.fill();
  // Draw surrogate mean
  var meanPts=[];
  for(var i=0;i<=100;i++){var x=i/100;meanPts.push([PAD+x*pW,H-PAD-surrogateMean(x)*pH]);}
  crv(meanPts,ctx,'#4ecdc4',2);
  // Acquisition function (simplified EI)
  var bestSeen=shown.length>0?Math.max.apply(null,shown.map(function(p){return p[1];})):0;
  var acqPts=[];
  for(var i=0;i<=100;i++){
    var x=i/100;var m=surrogateMean(x);var s=surrogateStd(x);
    var ei=Math.max(0,(m-bestSeen))*s;
    acqPts.push([PAD+x*pW,H-PAD-Math.min(ei*pH*4,pH*0.3)]);
  }
  crv(acqPts,ctx,'#fbbf24',1.5);
  // Next point to evaluate (max acquisition)
  var maxAcq=-1,nextX=0;
  for(var i=0;i<=100;i++){var x=i/100;var m=surrogateMean(x);var s=surrogateStd(x);var ei=(m-bestSeen)*s;if(ei>maxAcq){maxAcq=ei;nextX=x;}}
  var nextPX=PAD+nextX*pW;
  ln(ctx,nextPX,15,nextPX,H-PAD,'#fbbf2460',1.5,[3,3]);
  tx(ctx,'Next eval here',nextPX,26,'#fbbf24',8,'800');
  // Observed points
  shown.forEach(function(p,i){
    var px=PAD+p[0]*pW,py=H-PAD-p[1]*pH;
    ctx.fillStyle='#ff6b35';ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fill();
    if(i===shown.length-1){ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;ctx.beginPath();ctx.arc(px,py,9,0,Math.PI*2);ctx.stroke();}
  });
  // Legend
  ln(ctx,W-120,28,W-104,28,'#4ecdc4',2);tx(ctx,'Surrogate mean',W-14,28,'#4ecdc4',7,'700','right');
  ln(ctx,W-120,40,W-104,40,'#fbbf24',1.5);tx(ctx,'Acquisition (EI)',W-14,40,'#fbbf24',7,'700','right');
  ctx.fillStyle='#ff6b35';ctx.beginPath();ctx.arc(W-112,52,4,0,Math.PI*2);ctx.fill();tx(ctx,'Observed eval',W-14,52,'#ff6b35',7,'700','right');
  document.getElementById('iBO').innerHTML=
    'Trial <span class="hl3">'+step+'</span>: surrogate fitted to '+step+' observations.<br>'+
    '<span class="hl2">Surrogate (teal):</span> predicts mean performance across the space.<br>'+
    '<span class="hl3">Acquisition (yellow):</span> balances exploration (uncertain regions) and exploitation (known good regions).<br>'+
    '<span class="hl">Next evaluation:</span> chosen to maximise acquisition \u2014 the most promising unexplored region.';
}

function renderBOLoop(){
  var ctx=clr('cvBOLoop',420,290),W=420,H=290;
  var steps=[
    {label:'Initial random\nsampling (5-10 trials)',col:'#3f3f46',desc:'Explore blindly to bootstrap surrogate'},
    {label:'Fit surrogate\nmodel (GP / TPE)',col:'#4ecdc4',desc:'Learn (config \u2192 performance) relationship'},
    {label:'Maximise\nacquisition fn',col:'#fbbf24',desc:'Find next most promising point to evaluate'},
    {label:'Evaluate\nconfig',col:'#ff6b35',desc:'Train model, measure val metric (expensive)'},
    {label:'Update\nsurrogate',col:'#c084fc',desc:'Add new observation, refit model'},
  ];
  var R=58,cx0=W/2,cy0=H/2,n=steps.length;
  steps.forEach(function(st,i){
    var angle=-Math.PI/2+i*(2*Math.PI/n);
    var bx2=cx0+R*1.55*Math.cos(angle);var by=cy0+R*1.55*Math.sin(angle);
    // Circle node
    ctx.fillStyle=st.col+'25';ctx.strokeStyle=st.col;ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(bx2,by,R,0,Math.PI*2);ctx.fill();ctx.stroke();
    st.label.split('\n').forEach(function(l,li){tx(ctx,l,bx2,by-6+li*14,st.col,8,'800');});
    // Arrow to next
    var nextI=(i+1)%n;
    var na=-Math.PI/2+nextI*(2*Math.PI/n);
    var nx=cx0+R*1.55*Math.cos(na),ny=cy0+R*1.55*Math.sin(na);
    var ang=Math.atan2(ny-by,nx-bx2);
    var sx=bx2+R*Math.cos(ang),sy=by+R*Math.sin(ang);
    var ex=nx-R*Math.cos(ang),ey=ny-R*Math.sin(ang);
    ctx.strokeStyle=st.col+'80';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ex-6*Math.cos(ang-0.4),ey-6*Math.sin(ang-0.4));ctx.stroke();
    ctx.fillStyle=st.col+'80';ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-9*Math.cos(ang-0.4),ey-9*Math.sin(ang-0.4));ctx.lineTo(ex-9*Math.cos(ang+0.4),ey-9*Math.sin(ang+0.4));ctx.closePath();ctx.fill();
  });
  tx(ctx,'Bayesian Optimisation Loop',W/2,cy0,'#fbbf24',10,'800');
  tx(ctx,'(repeat until budget exhausted)',W/2,cy0+16,'#52525b',8,'400');
  document.getElementById('iBOLoop').innerHTML=
    'BO uses each result to improve the next guess.<br>'+
    'After 20+ trials the surrogate is accurate enough to locate the optimum efficiently.<br>'+
    '<span class="hl5">BO limitation:</span> inherently sequential. Each evaluation must finish before the next is chosen.<br>'+
    '<span class="hl4">Parallel BO:</span> batch acquisition (e.g., qEI) can suggest k points simultaneously, but is less efficient per-point than sequential.';
}

// ============================================================
// TAB 2: LR FINDER
// ============================================================
var curLRF='sgd';
function selLRF(k){curLRF=k;['sgd','adam'].forEach(function(b){document.getElementById('lrfBtn_'+b).classList.remove('active');});document.getElementById('lrfBtn_'+k).classList.add('active');renderLRF();}
function renderLRF(){
  var ctx=clr('cvLRF',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Learning Rate (log scale)','Loss');
  // x is log10(LR) from -7 to -1
  var lrMin=-7,lrMax=-1;
  var declineStart=-4.5,minLR=-2.8,divergeStart=-2.0;
  function lossFn(logLR){
    if(logLR<declineStart) return 2.5;
    if(logLR<minLR) return 2.5-2.0*(logLR-declineStart)/(minLR-declineStart);
    if(logLR<divergeStart) return 0.5+1.0*Math.pow((logLR-minLR)/(divergeStart-minLR),2);
    return 0.5+3.0*Math.pow((logLR-divergeStart),1.5)+0.3;
  }
  // Add noise
  var seed=7;function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
  var pts=[];
  for(var i=0;i<=120;i++){
    var logLR=lrMin+i/120*(lrMax-lrMin);
    var raw=lossFn(logLR);
    var noisy=raw+(rng()-0.5)*0.08;
    var px=PAD+(logLR-lrMin)/(lrMax-lrMin)*pW;
    var py=H-PAD-Math.min(noisy,3.5)/3.5*pH;
    pts.push([px,py]);
  }
  // Shade regions
  var steepStart=PAD+(declineStart-lrMin)/(lrMax-lrMin)*pW;
  var minX=PAD+(minLR-lrMin)/(lrMax-lrMin)*pW;
  var divX=PAD+(divergeStart-lrMin)/(lrMax-lrMin)*pW;
  bx(ctx,steepStart,15,minX-steepStart,H-PAD-15,0,'#4ade8015',null);
  bx(ctx,minX,15,divX-minX,H-PAD-15,0,'#fbbf2415',null);
  bx(ctx,divX,15,W-10-divX,H-PAD-15,0,'#ef444415',null);
  crv(pts,ctx,'#4ecdc4',2.5);
  // Vertical markers
  var adamLR=declineStart+0.3; // 1/10 below steep start for Adam
  var sgdLR=minLR-0.5; // just before minimum for SGD
  var pickLR=curLRF==='sgd'?sgdLR:adamLR;
  var pickX=PAD+(pickLR-lrMin)/(lrMax-lrMin)*pW;
  ln(ctx,pickX,15,pickX,H-PAD,'#ff6b35',2,[4,4]);
  ctx.fillStyle='#ff6b35';ctx.beginPath();ctx.arc(pickX,H-PAD-lossFn(pickLR)/3.5*pH,7,0,Math.PI*2);ctx.fill();
  tx(ctx,'Use this LR',pickX,22,'#ff6b35',8,'800');
  tx(ctx,'10^'+rnd(pickLR,1)+' \u2248 '+Math.pow(10,pickLR).toExponential(0),pickX,36,'#fbbf24',8,'700');
  // Log axis labels
  [-7,-6,-5,-4,-3,-2,-1].forEach(function(v){
    var px2=PAD+(v-lrMin)/(lrMax-lrMin)*pW;
    tx(ctx,'1e'+v,px2,H-PAD+14,'#3f3f46',6,'400');
  });
  // Region labels
  tx(ctx,'Too small',PAD+(declineStart-lrMin-0.8)/(lrMax-lrMin)*pW/2+PAD,H-PAD-pH*0.92,'#3f3f46',8,'700');
  tx(ctx,'Good',( steepStart+minX)/2,H-PAD-pH*0.7,'#4ade80',8,'800');
  tx(ctx,'Too large',(divX+W-10)/2,H-PAD-pH*0.5,'#ef4444',8,'700');
  document.getElementById('iLRF').innerHTML=
    (curLRF==='sgd'?'<span class="hl3">SGD:</span> use LR at the beginning of the loss minimum (steepest decline point).':'<span class="hl2">Adam:</span> use LR ~1 order of magnitude below the minimum. Adam already adapts per-parameter LRs, so the raw LR acts as a step-size bound.')+
    '<br>Picked LR: <span class="hl">'+Math.pow(10,pickLR).toExponential(1)+'</span><br>'+
    'LR finder takes only ~100 mini-batches. Collapses LR search from hours to minutes.';
}

function renderCLR(){
  var lrMaxIdx=parseInt(document.getElementById('slLRmax').value);
  var cycleLen=parseInt(document.getElementById('slCycle').value);
  document.getElementById('vLRmax').textContent=(lrMaxIdx/100).toFixed(2);
  document.getElementById('vCycle').textContent=cycleLen;
  var lrMax=lrMaxIdx/100, lrMin=lrMax/10;
  var ctx=clr('cvCLR',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training steps','Learning Rate');
  var N=200;
  var pts=[];
  for(var i=0;i<=N;i++){
    var t=i/N;
    var cyclePos=(t*cycleLen)%1;
    var lr;
    if(cyclePos<0.5){lr=lrMin+(lrMax-lrMin)*cyclePos*2;}
    else{lr=lrMax-(lrMax-lrMin)*(cyclePos-0.5)*2;}
    var cx=PAD+t*pW,cy=H-PAD-(lr/lrMax)*pH;
    pts.push([cx,cy]);
  }
  crv(pts,ctx,'#c084fc',2.5);
  // LR max line
  ln(ctx,PAD,H-PAD-pH,W-10,H-PAD-pH,'#ff6b3530',1,[4,4]);
  tx(ctx,'LR max='+lrMax.toFixed(2),W-14,H-PAD-pH-8,'#ff6b3550',7,'700','right');
  ln(ctx,PAD,H-PAD-pH*0.1,W-10,H-PAD-pH*0.1,'#4ecdc430',1,[4,4]);
  tx(ctx,'LR min='+lrMin.toFixed(3),W-14,H-PAD-pH*0.1-8,'#4ecdc450',7,'700','right');
  tx(ctx,'Triangular CLR \u2014 '+cycleLen+' cycle(s), LR max='+lrMax.toFixed(2),W/2,22,'#c084fc',9,'800');
  document.getElementById('iCLR').innerHTML=
    'CLR oscillates between LR min=<span class="hl3">'+lrMin.toFixed(3)+'</span> and LR max=<span class="hl">'+lrMax.toFixed(2)+'</span>.<br>'+
    'The periodic LR increase helps escape sharp minima. The LR range comes directly from the LR finder.<br>'+
    '<span class="hl4">Benefit:</span> achieves same accuracy as fixed LR with less tuning. Good when the best fixed LR is unknown.';
}

function renderLRLog(){
  var ctx=clr('cvLRLog',860,150),W=860,H=150;
  var half=W/2-20;
  // Linear scale (left)
  tx(ctx,'Linear scale: LR in [1e-4, 1e-1] with 20 samples',half/2,18,'#ef4444',9,'800');
  var N=20,lrArr=[],lrLogArr=[];
  for(var i=0;i<N;i++){lrArr.push(1e-4+(i/(N-1))*(1e-1-1e-4));}
  for(var i=0;i<N;i++){lrLogArr.push(Math.pow(10,-4+(i/(N-1))*3));}
  lrArr.forEach(function(lr){
    var logLR=Math.log10(lr);
    var px=10+(logLR+4)/3*(half-20);
    var inGood=lr>=5e-4&&lr<=5e-3;
    bx(ctx,px-3,30,6,50,0,inGood?'#4ade8050':'#ff6b3530',inGood?'#4ade80':'#ff6b35',1);
  });
  tx(ctx,'1e-4',10,90,'#3f3f46',7,'400','left');tx(ctx,'1e-3',10+(1/3)*(half-20),90,'#3f3f46',7,'400');tx(ctx,'1e-2',10+(2/3)*(half-20),90,'#3f3f46',7,'400');tx(ctx,'1e-1',10+half-20,90,'#3f3f46',7,'400','right');
  var linGood=lrArr.filter(function(lr){return lr>=5e-4&&lr<=5e-3;}).length;
  tx(ctx,linGood+' / '+N+' samples in productive range (1 decade)',half/2,115,'#ef4444',8,'700');
  // Log scale (right)
  var ox=W/2+10;
  tx(ctx,'Log scale: same range, same 20 samples',ox+(W-ox)/2,18,'#4ade80',9,'800');
  lrLogArr.forEach(function(lr){
    var logLR=Math.log10(lr);
    var px=ox+(logLR+4)/3*(half-20);
    var inGood=lr>=5e-4&&lr<=5e-3;
    bx(ctx,px-3,30,6,50,0,inGood?'#4ade8050':'#4ecdc430',inGood?'#4ade80':'#4ecdc4',1);
  });
  tx(ctx,'1e-4',ox,90,'#3f3f46',7,'400','left');tx(ctx,'1e-3',ox+(1/3)*(half-20),90,'#3f3f46',7,'400');tx(ctx,'1e-2',ox+(2/3)*(half-20),90,'#3f3f46',7,'400');tx(ctx,'1e-1',ox+half-20,90,'#3f3f46',7,'400','right');
  var logGood=lrLogArr.filter(function(lr){return lr>=5e-4&&lr<=5e-3;}).length;
  tx(ctx,logGood+' / '+N+' samples in productive range (1 decade out of 3)',ox+(W-ox)/2,115,'#4ade80',8,'700');
}

// ============================================================
// TAB 3: HYPERBAND / PBT
// ============================================================
function renderSH(){
  var n=parseInt(document.getElementById('slSHn').value);
  document.getElementById('vSHn').textContent=n;
  var ctx=clr('cvSH',420,280),W=420,H=280;
  var eta=2; // halving factor
  var rounds=Math.log2(n);
  var configs=[];
  var seed2=17;function rng2(){seed2=(seed2*1664525+1013904223)&0xffffffff;return((seed2>>>0)/4294967296);}
  for(var i=0;i<n;i++){configs.push({perf:rng2(),alive:true});}
  var rowH=(H-60)/(n+1), cellW=(W-60)/rounds;
  // Simulate halving
  var roundResults=[[configs.map(function(c,i){return{idx:i,perf:c.perf,steps:1};})]];
  var alive=configs.map(function(_,i){return i;});
  var steps=1;
  for(var r=1;r<rounds;r++){
    steps*=eta;
    alive.sort(function(a,b){return configs[b].perf-configs[a].perf;});
    alive=alive.slice(0,Math.ceil(alive.length/eta));
    roundResults.push(alive.map(function(i){return{idx:i,perf:configs[i].perf,steps:steps};}));
  }
  // Draw
  var colors=['#4ecdc4','#ff6b35','#fbbf24','#c084fc','#4ade80','#38bdf8','#f472b6','#fb923c','#a78bfa','#34d399','#f59e0b','#60a5fa','#e879f9','#2dd4bf','#84cc16','#facc15'];
  for(var r=0;r<rounds;r++){
    var step2=Math.pow(2,r);
    tx(ctx,'R'+( r+1)+'\n('+step2+'x)',60+r*cellW+cellW/2,H-20,'#52525b',7,'400');
  }
  // Draw rows for each initial config
  configs.forEach(function(c,ci){
    var cy=30+ci*rowH;
    var col=colors[ci%colors.length];
    var lastX=60;
    var isAlive=true;
    roundResults.forEach(function(rnd,ri){
      var found=rnd.find(function(x){return x.idx===ci;});
      if(found&&isAlive){
        var rx=60+ri*cellW;
        ln(ctx,lastX,cy,rx+cellW*0.6,cy,col+'80',1.5);
        ctx.fillStyle=col;ctx.beginPath();ctx.arc(rx+cellW*0.6,cy,4,0,Math.PI*2);ctx.fill();
        lastX=rx+cellW*0.6;
        if(ri===roundResults.length-1){
          ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;ctx.beginPath();ctx.arc(lastX,cy,8,0,Math.PI*2);ctx.stroke();
          tx(ctx,'winner',lastX+14,cy,'#fbbf24',7,'800','left');
        }
      } else if(isAlive){
        isAlive=false;
        ctx.strokeStyle=col+'40';ctx.lineWidth=1;ctx.setLineDash([3,3]);
        ln(ctx,lastX,cy,W-20,cy,col+'30',1,[3,3]);ctx.setLineDash([]);
        tx(ctx,'eliminated',lastX+8,cy-8,col+'50',6,'400','left');
      }
    });
    tx(ctx,'C'+ci,50,cy,'#52525b',7,'400','right');
  });
  var totalSave=n*Math.pow(2,rounds);var actualCost=0;
  for(var r=0;r<rounds;r++){actualCost+=Math.ceil(n/Math.pow(eta,r))*Math.pow(eta,r);}
  tx(ctx,'Successive Halving (n='+n+', \u03B7=2)',W/2,15,'#4ecdc4',9,'800');
  document.getElementById('iSH').innerHTML=
    n+' configs start. Each round: keep top 50%, train '+eta+'\u00D7 longer.<br>'+
    'Total compute: <span class="hl4">'+actualCost+' units</span> vs <span class="hl5">'+totalSave+' units</span> if all run to full budget.<br>'+
    'Compute savings: <span class="hl4">'+rnd(totalSave/actualCost,1)+'x</span>. Winner gets the most compute; losers eliminated early.';
}

function renderPBT(){
  var pop=parseInt(document.getElementById('slPop').value);
  document.getElementById('vPop').textContent=pop;
  var ctx=clr('cvPBT',420,280),W=420,H=280;
  var nCheckpoints=4;
  var cellW=(W-80)/nCheckpoints,rowH=(H-60)/pop;
  var seed3=31;function rng3(){seed3=(seed3*1664525+1013904223)&0xffffffff;return((seed3>>>0)/4294967296);}
  // Simulate: each model has a performance trajectory
  var models=[];
  for(var i=0;i<pop;i++){
    var perf=0.3+rng3()*0.4;
    var lr=Math.pow(10,-2+rng3()*2);
    models.push({perf:perf,lr:lr,trajectory:[perf]});
  }
  var colors2=['#ff6b35','#4ecdc4','#fbbf24','#c084fc','#4ade80','#38bdf8','#f472b6','#fb923c','#a78bfa','#34d399','#f59e0b','#84cc16'];
  for(var cp=1;cp<nCheckpoints;cp++){
    // Sort by perf, bottom 25% gets replaced by top 25%
    var sorted=models.slice().sort(function(a,b){return b.perf-a.perf;});
    var nReplace=Math.max(1,Math.floor(pop*0.25));
    for(var j=0;j<nReplace;j++){
      var worst=models.indexOf(sorted[pop-1-j]);
      var best=models.indexOf(sorted[j]);
      models[worst].perf=models[best].perf*(0.9+rng3()*0.2);
      models[worst].lr=models[best].lr*(0.8+rng3()*0.4);
      models[worst].trajectory.push(models[worst].perf);
    }
    for(var k=0;k<pop-nReplace;k++){
      var idx2=models.indexOf(sorted[k]);
      models[idx2].perf=Math.min(1,models[idx2].perf*(0.95+rng3()*0.12));
      models[idx2].trajectory.push(models[idx2].perf);
    }
  }
  // Draw
  models.forEach(function(m,mi){
    var col=colors2[mi%colors2.length];
    var pts=[];
    m.trajectory.forEach(function(p,ci){
      var px=60+ci*cellW;var py=H-40-(p-0.2)/0.85*( H-70);
      pts.push([px,py]);
    });
    crv(pts,ctx,col,2);
    pts.forEach(function(p,ci){
      ctx.fillStyle=col;ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fill();
    });
  });
  // Checkpoint lines
  for(var cp=0;cp<nCheckpoints;cp++){
    var px=60+cp*cellW;
    ln(ctx,px,10,px,H-30,'#1e1e2e',1,[3,3]);
    tx(ctx,'T='+cp,px,H-20,'#52525b',7,'400');
    if(cp>0){
      bx(ctx,px-28,10,56,20,4,'#ff6b3520','#ff6b3560',1.5);
      tx(ctx,'exploit+explore',px,20,'#ff6b35',6,'800');
    }
  }
  tx(ctx,'Population-Based Training (pop='+pop+')',W/2,14,'#fbbf24',9,'800');
  tx(ctx,'Performance',50,H/2,'#52525b',8,'400','right');
  document.getElementById('iPBT').innerHTML=
    'Population of '+pop+' models train in parallel. Every T steps: bottom 25% ('+Math.floor(pop*0.25)+' models) are replaced by copies of the best performers with perturbed HPs.<br>'+
    '<span class="hl4">Exploit:</span> copy weights from top performers. <span class="hl3">Explore:</span> perturb HPs (multiply LR by 0.8 or 1.2).<br>'+
    '<span class="hl">Result:</span> discovers adaptive HP schedule without manual schedule design. Best for RL and long training with many GPUs.';
}

function renderHB(){
  var ctx=clr('cvHB',860,155),W=860,H=155;
  var brackets=[
    {label:'Bracket 3\n(Explore)',n:27,minBudget:'R/27',col:'#4ecdc4'},
    {label:'Bracket 2',n:9,minBudget:'R/9',col:'#fbbf24'},
    {label:'Bracket 1',n:3,minBudget:'R/3',col:'#ff6b35'},
    {label:'Bracket 0\n(Exploit)',n:1,minBudget:'R',col:'#c084fc'},
  ];
  var bw=(W-20)/brackets.length-8,startX=10;
  brackets.forEach(function(br,i){
    var x=startX+i*(bw+8);
    bx(ctx,x,20,bw,H-30,6,br.col+'15',br.col+'60',1.5);
    br.label.split('\n').forEach(function(l,li){tx(ctx,l,x+bw/2,40+li*14,br.col,9,'800');});
    tx(ctx,br.n+' configs',x+bw/2,72,br.col,9,'800');
    tx(ctx,'min budget: '+br.minBudget+' each',x+bw/2,86,'#52525b',7,'400');
    // Mini successive halving bars
    var halves=Math.log2(br.n);
    for(var h=0;h<=halves;h++){
      var alive2=Math.ceil(br.n/Math.pow(2,h));
      var bh2=Math.min(8,(H-110)/halves);
      bx(ctx,x+8+h*(bw-16)/halves,H-40,Math.max(4,(bw-16)/halves*alive2/br.n*1.5),bh2,2,br.col+'50',br.col,1);
    }
    tx(ctx,'SH races inside',x+bw/2,H-15,'#52525b',7,'400');
  });
  tx(ctx,'Hyperband: runs ALL brackets simultaneously. Total budget \u22485\u00D7 single SH. Robust to whether early perf predicts final perf.',W/2,H-2,'#52525b',7,'400');
}

// ============================================================
// TAB 4: LR SCHEDULES
// ============================================================
var curSched='warmcos';
function selSched(k){curSched=k;['warmcos','step','sgdr','plateau'].forEach(function(b){document.getElementById('schBtn_'+b).classList.remove('active');});document.getElementById('schBtn_'+k).classList.add('active');renderSched();}
function renderSched(){
  var ctx=clr('cvSched',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','Learning Rate');
  var lrMax=0.001,lrMin=1e-6,N2=200,warmup=20;
  var configs={
    warmcos:{col:'#4ade80',label:'Warmup + Cosine',
      fn:function(i){if(i<warmup)return lrMax*(i/warmup);return lrMin+0.5*(lrMax-lrMin)*(1+Math.cos(Math.PI*(i-warmup)/(N2-warmup)));}},
    step:{col:'#fbbf24',label:'Step Decay (\u00D70.1 at 33%, 66%)',
      fn:function(i){var pct=i/N2;if(pct<0.33)return lrMax;if(pct<0.66)return lrMax*0.1;return lrMax*0.01;}},
    sgdr:{col:'#4ecdc4',label:'SGDR (Warm Restarts)',
      fn:function(i){var cycleLen=N2/3;var ci=i%cycleLen;return lrMin+0.5*(lrMax-lrMin)*(1+Math.cos(Math.PI*ci/cycleLen));}},
    plateau:{col:'#c084fc',label:'Reduce on Plateau',
      fn:function(i){var lr=lrMax;var drops=[70,120,150,170];drops.forEach(function(d){if(i>d)lr*=0.3;});return lr;}},
  };
  var cfg=configs[curSched];
  var pts=[];
  for(var i=0;i<=N2;i++){
    var lr2=cfg.fn(i);
    var cx=PAD+(i/N2)*pW;
    var cy=H-PAD-Math.max(0,Math.min(lr2/lrMax,1.05))*pH;
    pts.push([cx,cy]);
  }
  crv(pts,ctx,cfg.col,2.5);
  ln(ctx,PAD,H-PAD-pH,W-10,H-PAD-pH,'#3f3f4620',1,[4,4]);
  tx(ctx,'LR max',PAD-6,H-PAD-pH,'#3f3f46',7,'400','right');
  tx(ctx,cfg.label,W/2,22,cfg.col,10,'800');
  // Warmup annotation
  if(curSched==='warmcos'){
    var wupX=PAD+(warmup/N2)*pW;
    ln(ctx,wupX,15,wupX,H-PAD,'#fbbf2460',1.5,[3,3]);
    tx(ctx,'warmup\nend',wupX+4,38,'#fbbf24',7,'700','left');
  }
  var infoMap={
    warmcos:'<span class="hl4">Warmup+Cosine:</span> linear ramp for first '+warmup+' steps ('+Math.round(warmup/N2*100)+'%), then smooth cosine decay. Standard for all transformers. Starts fast, converges precisely.',
    step:'<span class="hl3">Step decay:</span> staircase, divide by 10 at fixed milestones. Classic for ResNets. Sudden jumps can disrupt Adam\'s variance estimates. Avoid for transformers.',
    sgdr:'<span class="hl2">SGDR:</span> periodic cosine restarts. Each restart escapes the current basin. Snapshot ensembling (average predictions from end of each cycle) often improves performance.',
    plateau:'<span class="hl6">ReduceLROnPlateau:</span> reactive, not proactive. Halves LR after N epochs without improvement. Good when total steps are unknown. Less sample-efficient than cosine annealing.',
  };
  document.getElementById('iSched').innerHTML=infoMap[curSched];
}

function renderWarmup(){
  var wuPct=parseInt(document.getElementById('slWarmup').value);
  document.getElementById('vWarmup').textContent=wuPct+'%';
  var ctx=clr('cvWarmup',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  axesPAD(ctx,W,H,PAD,'Training step','LR / Effective step size');
  var N3=100,warmupSteps=wuPct;
  // Adam effective step size: LR / sqrt(vhat). In early steps vhat is tiny -> huge effective step
  // Show with warmup vs without
  var noWarmPts=[],warmPts=[],effNoWarm=[],effWarm=[];
  for(var i=0;i<=N3;i++){
    var t=i;
    var lrNW=0.001; // constant LR
    var lrW=t<=warmupSteps?0.001*(t/Math.max(1,warmupSteps)):0.001;
    var vhat=Math.min(1,Math.pow((t+1)/10,0.7)); // simplified variance accumulation
    var effNW=Math.min(lrNW/Math.sqrt(vhat+1e-8),0.1); // effective LR
    var effW=Math.min(lrW/Math.sqrt(vhat+1e-8),0.1);
    var cx=PAD+(i/N3)*pW;
    noWarmPts.push([cx,H-PAD-Math.min(lrNW/0.001,1)*pH]);
    warmPts.push([cx,H-PAD-Math.min(lrW/0.001,1)*pH]);
    effNoWarm.push([cx,H-PAD-Math.min(effNW/0.1,1)*pH]);
    effWarm.push([cx,H-PAD-Math.min(effW/0.1,1)*pH]);
  }
  // Shade danger zone
  var wuEndX=PAD+(warmupSteps/N3)*pW;
  bx(ctx,PAD,15,wuEndX-PAD,H-PAD-15,0,'#ef444410',null);
  crv(effNoWarm,ctx,'#ef444490',2); // effective step without warmup
  crv(effWarm,ctx,'#4ade80',2.5);  // effective step with warmup
  crv(noWarmPts,ctx,'#ef444440',1.5); // raw LR no warmup
  crv(warmPts,ctx,'#4ade8060',1.5); // raw LR with warmup
  if(warmupSteps>0){
    ln(ctx,wuEndX,15,wuEndX,H-PAD,'#fbbf2460',1.5,[3,3]);
    tx(ctx,'warmup\nend',wuEndX+4,36,'#fbbf24',7,'700','left');
  }
  tx(ctx,'\u2014 No warmup (effective step)',W-14,28,'#ef4444',8,'700','right');
  tx(ctx,'\u2014 With warmup (effective step)',W-14,42,'#4ade80',8,'700','right');
  tx(ctx,'Danger zone: huge effective step from low v\u0302',PAD+(wuEndX-PAD)/2,H-PAD-pH*0.6,'#ef444450',8,'700');
  document.getElementById('iWarmup').innerHTML=
    'Warmup=<span class="hl3">'+wuPct+'%</span> of total steps.<br>'+
    '<span class="hl5">Without warmup:</span> Adam\'s variance estimate (v\u0302\u209C) is near 0 in early steps \u2192 effective step size = LR/\u221Av\u0302 can be enormous \u2192 chaotic first updates destroy initial representations.<br>'+
    '<span class="hl4">With warmup:</span> global LR is small while v\u0302 accumulates. Effective step size stays bounded. Adam stabilises before full LR is used.';
}

// ============================================================
// TAB 5: STRATEGY
// ============================================================
function renderTiers(){
  var ctx=clr('cvTiers',860,200),W=860,H=200;
  var tiers=[
    {label:'TIER 1 \u2014 CRITICAL (tune first)',col:'#ef4444',hps:[
      {name:'Learning Rate',impact:'62%'},
      {name:'Batch Size',impact:'depends on LR'},
      {name:'Weight Decay',impact:'18%'},
      {name:'Architecture\ndepth/width',impact:'high'},
    ]},
    {label:'TIER 2 \u2014 IMPORTANT',col:'#fbbf24',hps:[
      {name:'LR Schedule',impact:'11%'},
      {name:'Dropout Rate',impact:'5%'},
      {name:'Optimizer\nchoice',impact:'low'},
      {name:'Warmup Steps',impact:'moderate'},
    ]},
    {label:'TIER 3 \u2014 FINE-TUNE LAST',col:'#4ade80',hps:[
      {name:'Epsilon \u03B5',impact:'<1%'},
      {name:'Grad clip\nvalue',impact:'<1%'},
      {name:'Label smooth\n\u03B1',impact:'minor'},
      {name:'Aug\nmagnitude',impact:'minor'},
    ]},
  ];
  var tierW=(W-20)/tiers.length,startX=10;
  tiers.forEach(function(tier,ti){
    var tx2=startX+ti*tierW;
    bx(ctx,tx2,8,tierW-6,H-16,6,tier.col+'12',tier.col+'50',2);
    tx(ctx,tier.label,tx2+( tierW-6)/2,26,tier.col,9,'800');
    var hpW=(tierW-20)/tier.hps.length;
    tier.hps.forEach(function(hp,hi){
      var hx=tx2+8+hi*hpW;
      bx(ctx,hx,44,hpW-4,H-58,4,tier.col+'18',tier.col+'40',1);
      hp.name.split('\n').forEach(function(l,li){tx(ctx,l,hx+(hpW-4)/2,64+li*13,tier.col,8,'800');});
      tx(ctx,hp.impact,hx+(hpW-4)/2,H-28,'#52525b',7,'400');
    });
    startX+=tierW;
  });
}

function renderPhase(){
  var ctx=clr('cvPhase',420,240),W=420,H=240;
  var phases=[
    {label:'Phase 1\nCOARSE',col:'#4ecdc4',x:30,desc:'20-50 trials\n10-20% budget\nWide space\nIdentify LR order'},
    {label:'Phase 2\nMEDIUM',col:'#fbbf24',x:150,desc:'30-100 trials\n50% budget\nNarrowed space\nBayes / random'},
    {label:'Phase 3\nFINE',col:'#4ade80',x:280,desc:'3-5 candidates\nFull budget\n3 seeds each\nSelect best'},
  ];
  phases.forEach(function(ph,i){
    var w=110;
    bx(ctx,ph.x,20,w,H-40,8,ph.col+'20',ph.col,2);
    ph.label.split('\n').forEach(function(l,li){tx(ctx,l,ph.x+w/2,42+li*14,ph.col,9,'800');});
    ph.desc.split('\n').forEach(function(l,li){tx(ctx,l,ph.x+w/2,90+li*18,'#94a3b8',8,'400');});
    if(i<phases.length-1){
      ctx.strokeStyle='#3f3f46';ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(ph.x+w+2,H/2);ctx.lineTo(ph.x+w+36,H/2);ctx.stroke();
      ctx.fillStyle='#3f3f46';ctx.beginPath();ctx.moveTo(ph.x+w+36,H/2);ctx.lineTo(ph.x+w+28,H/2-5);ctx.lineTo(ph.x+w+28,H/2+5);ctx.closePath();ctx.fill();
    }
  });
  tx(ctx,'Total compute \u224810-20\u00D7 one full training run',W/2,H-12,'#52525b',8,'400');
  document.getElementById('iPhase').innerHTML=
    '<span class="hl2">Phase 1:</span> identify the right ORDER OF MAGNITUDE for each Tier-1 HP. Use wide log-scale search, short runs with early stopping.<br>'+
    '<span class="hl3">Phase 2:</span> narrow space, add Tier-2 HPs, use Bayesian search or random with ASHA.<br>'+
    '<span class="hl4">Phase 3:</span> full runs with 3 seeds per candidate. Report mean\u00B1std. Select best, evaluate on test set ONCE.';
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){
  renderSearch(); renderScale(); renderSens();
});
</script>
</body>
</html>"""

HP_TUNING_VISUAL_HEIGHT = 2200