TRAIN_VS_INFERENCE_VISUAL_HTML = r"""<!DOCTYPE html>
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
<h2>&#x1F500; Train vs Inference Mode</h2>
<p class="sub">model.train() &middot; model.eval() &middot; Dropout &middot; BatchNorm &middot; no_grad &middot; Fine-tuning &middot; Deployment</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Three Controls</button>
  <button class="tab" onclick="showTab(1)">Dropout</button>
  <button class="tab" onclick="showTab(2)">BatchNorm</button>
  <button class="tab" onclick="showTab(3)">no_grad &amp; Memory</button>
  <button class="tab" onclick="showTab(4)">Training Loop</button>
  <button class="tab" onclick="showTab(5)">Fine-tuning &amp; Deploy</button>
</div>

<!-- TAB 0: THREE CONTROLS -->
<div id="tab0" class="panel active">
<div class="g2">
<div class="card">
  <h3>&#9312; The Three Independent Controls</h3>
  <canvas id="cvControls" width="420" height="290"></canvas>
  <div class="info" id="iControls">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; model.train() vs model.eval() &mdash; What Changes</h3>
  <canvas id="cvModeComp" width="420" height="290"></canvas>
  <div class="brow">
    <button class="s active" id="mcBtn_train" onclick="selMode('train')">model.train()</button>
    <button class="s" id="mcBtn_eval" onclick="selMode('eval')">model.eval()</button>
  </div>
  <div class="info" id="iMode">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Layer Behaviour Reference by Mode</h3>
  <canvas id="cvLayerRef" width="860" height="155"></canvas>
</div>
</div>

<!-- TAB 1: DROPOUT -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Dropout: Train vs Eval (Inverted Dropout)</h3>
  <canvas id="cvDropout" width="420" height="280"></canvas>
  <div class="row"><label>Dropout p</label>
    <input type="range" id="slDropP" min="0.1" max="0.9" step="0.1" value="0.5" oninput="renderDropout()">
    <span class="v" id="vDropP">0.5</span></div>
  <div class="brow">
    <button class="s active" id="dpBtn_train" onclick="selDropMode('train')">Train mode (stochastic)</button>
    <button class="s" id="dpBtn_eval"  onclick="selDropMode('eval')">Eval mode (deterministic)</button>
  </div>
  <div class="info" id="iDropout">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; MC Dropout &mdash; Uncertainty Estimation</h3>
  <canvas id="cvMCDrop" width="420" height="280"></canvas>
  <div class="row"><label>MC samples T</label>
    <input type="range" id="slMCT" min="5" max="100" step="5" value="30" oninput="renderMCDrop()">
    <span class="v" id="vMCT">30</span></div>
  <div class="info" id="iMCDrop">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Consequences of Forgetting model.eval()</h3>
  <canvas id="cvForgetEval" width="860" height="148"></canvas>
</div>
</div>

<!-- TAB 2: BATCHNORM -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; BatchNorm: Batch Stats vs Running Stats</h3>
  <canvas id="cvBN" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="bnBtn_train" onclick="selBN('train')">Train Mode</button>
    <button class="s" id="bnBtn_eval"  onclick="selBN('eval')">Eval Mode</button>
  </div>
  <div class="info" id="iBN">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Running Stats Convergence</h3>
  <canvas id="cvBNConv" width="420" height="270"></canvas>
  <div class="row"><label>Momentum</label>
    <input type="range" id="slMom" min="1" max="30" step="1" value="10" oninput="renderBNConv()">
    <span class="v" id="vMom">0.10</span></div>
  <div class="info" id="iBNConv">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; BatchNorm at Batch Size 1 &mdash; Why Eval Mode Is Critical</h3>
  <canvas id="cvBNBs1" width="860" height="148"></canvas>
</div>
</div>

<!-- TAB 3: NO_GRAD & MEMORY -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Autograd Graph &mdash; Train vs no_grad</h3>
  <canvas id="cvGraph" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="grBtn_train" onclick="selGraph('train')">With Autograd (Training)</button>
    <button class="s" id="grBtn_nogr"  onclick="selGraph('nogr')">torch.no_grad()</button>
    <button class="s" id="grBtn_infm"  onclick="selGraph('infm')">torch.inference_mode()</button>
  </div>
  <div class="info" id="iGraph">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Memory &amp; Speed Impact</h3>
  <canvas id="cvMemory" width="420" height="270"></canvas>
  <div class="row"><label>Model depth</label>
    <input type="range" id="slDepth" min="4" max="48" step="4" value="12" oninput="renderMemory()">
    <span class="v" id="vDepth">12 layers</span></div>
  <div class="info" id="iMemory">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; no_grad vs inference_mode vs Default &mdash; Comparison</h3>
  <table class="tbl">
    <thead><tr><th>Context</th><th>Graph built?</th><th>requires_grad?</th><th>Speed</th><th>Nestable?</th><th>Use when</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">Default (none)</span></td><td><span class="hl5">YES</span></td><td><span class="hl5">YES</span></td><td>Slowest</td><td>N/A</td><td>Training step only. Never for inference.</td></tr>
      <tr><td><span class="hl3">torch.no_grad()</span></td><td><span class="hl4">NO</span></td><td><span class="hl4">NO</span></td><td>Fast</td><td><span class="hl4">YES</span></td><td>Validation loop. When output may re-enter grad context (MAML).</td></tr>
      <tr><td><span class="hl4">torch.inference_mode()</span></td><td><span class="hl4">NO</span></td><td><span class="hl4">NO (strict)</span></td><td><span class="hl4">Fastest</span></td><td><span class="hl5">NO</span></td><td>Production inference. Pure forward-only deployment.</td></tr>
    </tbody>
  </table>
  <div class="info" style="margin-top:10px;">
    <span class="hl5">CRITICAL misconception:</span> torch.no_grad() does <span class="hl5">NOT</span> switch Dropout off. Dropout is controlled by model.training, not by gradient tracking.<br>
    <span class="hl4">Always combine:</span> model.eval() + torch.no_grad() for safe, efficient inference. Both are required.
  </div>
</div>
</div>

<!-- TAB 4: TRAINING LOOP -->
<div id="tab4" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Canonical Training Loop Structure</h3>
  <canvas id="cvLoop" width="420" height="310"></canvas>
  <div class="info" id="iLoop">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Common Mode-Switch Mistakes</h3>
  <canvas id="cvMistakes" width="420" height="310"></canvas>
  <div class="brow">
    <button class="s active" id="msBtn_1" onclick="selMistake(1)">Mistake 1</button>
    <button class="s" id="msBtn_2" onclick="selMistake(2)">Mistake 2</button>
    <button class="s" id="msBtn_3" onclick="selMistake(3)">Mistake 3</button>
    <button class="s" id="msBtn_4" onclick="selMistake(4)">Mistake 4</button>
    <button class="s" id="msBtn_5" onclick="selMistake(5)">Mistake 5</button>
  </div>
  <div class="info" id="iMistakes">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Mode Switch Checklist</h3>
  <div class="g2">
    <table class="tbl">
      <thead><tr><th>Phase</th><th>Required</th></tr></thead>
      <tbody>
        <tr><td><span class="hl4">Before training</span></td><td>model.train() once before batch loop. No no_grad() in training loop.</td></tr>
        <tr><td><span class="hl2">Before validation</span></td><td>model.eval() + torch.no_grad() wrapping the entire val loop.</td></tr>
        <tr><td><span class="hl3">After validation</span></td><td>model.train() again before next training phase (easy to forget!).</td></tr>
        <tr><td><span class="hl">Before deployment</span></td><td>model.eval() + assert not model.training + torch.inference_mode().</td></tr>
        <tr><td><span class="hl6">Fine-tuning mixed</span></td><td>Apply selective eval() AFTER model.train() (train() overrides all).</td></tr>
      </tbody>
    </table>
    <table class="tbl">
      <thead><tr><th>Debug stochastic inference</th></tr></thead>
      <tbody>
        <tr><td>Check model.training &mdash; must be False</td></tr>
        <tr><td>Scan all submodules: any training=True?</td></tr>
        <tr><td>Check custom layers use self.training flag</td></tr>
        <tr><td>Verify MC Dropout not used unintentionally</td></tr>
        <tr><td>torch.allclose(out1, out2) test for determinism</td></tr>
      </tbody>
    </table>
  </div>
</div>
</div>

<!-- TAB 5: FINE-TUNING & DEPLOY -->
<div id="tab5" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Fine-Tuning Mode Patterns</h3>
  <canvas id="cvFinetune" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="ftBtn_full"   onclick="selFT('full')">Full Fine-tune</button>
    <button class="s" id="ftBtn_linear" onclick="selFT('linear')">Linear Probing</button>
    <button class="s" id="ftBtn_grad"   onclick="selFT('grad')">Gradual Unfreeze</button>
    <button class="s" id="ftBtn_lora"   onclick="selFT('lora')">LoRA</button>
  </div>
  <div class="info" id="iFT">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Deployment Pipeline</h3>
  <canvas id="cvDeploy" width="420" height="270"></canvas>
  <div class="info" id="iDeploy">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Mode Configuration Lookup Table</h3>
  <table class="tbl">
    <thead><tr><th>Situation</th><th>model mode</th><th>gradient context</th><th>requires_grad</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">Standard training step</span></td><td><span class="hl4">.train()</span></td><td>None (default)</td><td>True</td></tr>
      <tr><td><span class="hl2">Validation / evaluation</span></td><td><span class="hl2">.eval()</span></td><td><span class="hl2">no_grad()</span></td><td>True (ok)</td></tr>
      <tr><td><span class="hl4">Production inference</span></td><td><span class="hl4">.eval()</span></td><td><span class="hl4">inference_mode()</span></td><td>False (opt.)</td></tr>
      <tr><td><span class="hl3">MC Dropout uncertainty</span></td><td><span class="hl3">.train() intentional</span></td><td><span class="hl2">no_grad()</span></td><td>False (opt.)</td></tr>
      <tr><td><span class="hl6">Linear probing</span></td><td>backbone: .eval() / head: .train()</td><td>None</td><td>backbone: False / head: True</td></tr>
      <tr><td><span class="hl">Full fine-tuning</span></td><td><span class="hl">.train()</span></td><td>None</td><td>True</td></tr>
      <tr><td><span class="hl5">Feature extraction</span></td><td><span class="hl5">.eval()</span></td><td><span class="hl4">inference_mode()</span></td><td>False</td></tr>
    </tbody>
  </table>
</div>
</div>

<script>
// ============================================================
// UTILS
// ============================================================
function rnd(v,d){return v.toFixed(d===undefined?2:d);}
function clr(id,W,H){var cv=document.getElementById(id),ctx=cv.getContext('2d');ctx.fillStyle='#09090f';ctx.fillRect(0,0,W,H);return ctx;}
function tx(ctx,s,x,y,col,sz,wt,al){ctx.fillStyle=col||'#e4e4e7';ctx.font=(wt||'500')+' '+(sz||11)+'px JetBrains Mono,monospace';ctx.textAlign=al||'center';ctx.textBaseline='middle';ctx.fillText(s,x,y);}
function ln(ctx,x1,y1,x2,y2,col,lw,dash){ctx.strokeStyle=col;ctx.lineWidth=lw||1.5;if(dash)ctx.setLineDash(dash);else ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);}
function bx(ctx,x,y,w,h,r,fill,stroke,sw){ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,w,h,r||4);else ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw||1.5;ctx.stroke();}}
function arr(ctx,x1,y1,x2,y2,col,lw){var a=Math.atan2(y2-y1,x2-x1);ctx.strokeStyle=col;ctx.lineWidth=lw||1.5;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-8*Math.cos(a-0.4),y2-8*Math.sin(a-0.4));ctx.lineTo(x2-8*Math.cos(a+0.4),y2-8*Math.sin(a+0.4));ctx.closePath();ctx.fill();}

var seed=42;function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
function resetSeed(s){seed=s||42;}

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderControls();renderMode();renderLayerRef();}if(i===1){renderDropout();renderMCDrop();renderForgetEval();}if(i===2){renderBN();renderBNConv();renderBNBs1();}if(i===3){renderGraph();renderMemory();}if(i===4){renderLoop();renderMistakes();}if(i===5){renderFT();renderDeploy();}}

// ============================================================
// TAB 0: THREE CONTROLS
// ============================================================
function renderControls(){
  var ctx=clr('cvControls',420,290),W=420,H=290;
  var controls=[
    {label:'CONTROL 1',name:'model.train() / model.eval()',col:'#ff6b35',governs:'Dropout ON/OFF, BatchNorm batch/running stats',scope:'All submodules (recursive)',y:20},
    {label:'CONTROL 2',name:'torch.no_grad() / inference_mode()',col:'#4ecdc4',governs:'Whether autograd builds computation graph',scope:'All ops inside the context',y:110},
    {label:'CONTROL 3',name:'param.requires_grad',col:'#c084fc',governs:'Whether a parameter receives gradient updates',scope:'One specific parameter tensor',y:200},
  ];
  controls.forEach(function(c){
    bx(ctx,12,c.y,W-24,80,8,c.col+'15',c.col,2);
    tx(ctx,c.label,50,c.y+16,c.col,8,'800','left');
    tx(ctx,c.name,50,c.y+30,c.col,9,'800','left');
    tx(ctx,'Governs: '+c.governs,24,c.y+50,'#94a3b8',7,'400','left');
    tx(ctx,'Scope: '+c.scope,24,c.y+65,'#52525b',7,'400','left');
  });
  // Independence arrows
  bx(ctx,W-50,65,36,18,4,'#ef444420','#ef4444',1.5);
  tx(ctx,'INDEPENDENT',W-32,74,'#ef4444',5,'800');
  bx(ctx,W-50,155,36,18,4,'#ef444420','#ef4444',1.5);
  tx(ctx,'INDEPENDENT',W-32,164,'#ef4444',5,'800');
  document.getElementById('iControls').innerHTML=
    '<span class="hl5">These three are INDEPENDENT. Common misconceptions:</span><br>'+
    '\u2022 model.eval() does <span class="hl5">NOT</span> disable gradient computation.<br>'+
    '\u2022 torch.no_grad() does <span class="hl5">NOT</span> switch Dropout off.<br>'+
    '\u2022 requires_grad=False does <span class="hl5">NOT</span> switch BN to running stats.<br>'+
    '<span class="hl4">For safe inference: set ALL THREE correctly together.</span>';
}

var curMode='train';
function selMode(k){curMode=k;['train','eval'].forEach(function(b){document.getElementById('mcBtn_'+b).classList.remove('active');});document.getElementById('mcBtn_'+k).classList.add('active');renderMode();}
function renderMode(){
  var ctx=clr('cvModeComp',420,290),W=420,H=290;
  var isTrain=curMode==='train';
  var col=isTrain?'#4ecdc4':'#4ade80';
  tx(ctx,isTrain?'model.train()  \u2192  training=True':'model.eval()  \u2192  training=False',W/2,22,col,10,'800');

  var layers=[
    {name:'Dropout',
     trainDesc:'Random mask per forward pass',trainSub:'output = m\u2219x / (1-p),  m ~ Bernoulli(1-p)',trainColor:'#fbbf24',trainRisk:null,
     evalDesc:'Identity passthrough',evalSub:'output = x  (all neurons active)',evalColor:'#4ade80',evalRisk:null},
    {name:'BatchNorm',
     trainDesc:'Uses BATCH \u03BC, \u03C3\u00B2 (fresh each step)',trainSub:'Updates running_mean, running_var',trainColor:'#ff6b35',trainRisk:null,
     evalDesc:'Uses RUNNING stats (accumulated)',evalSub:'No updates to running_mean/var',evalColor:'#4ade80',evalRisk:'B=1: TRAIN mode crashes \u2192 \u03C3\u00B2=0!'},
    {name:'LayerNorm / GroupNorm',
     trainDesc:'Per-example stats',trainSub:'Identical in both modes',trainColor:'#4ade80',trainRisk:null,
     evalDesc:'Per-example stats',evalSub:'Identical in both modes',evalColor:'#4ade80',evalRisk:null},
    {name:'nn.Linear / Conv',
     trainDesc:'Same computation',trainSub:'Weights and output identical',trainColor:'#4ade80',trainRisk:null,
     evalDesc:'Same computation',evalSub:'Weights and output identical',evalColor:'#4ade80',evalRisk:null},
  ];
  var bh=48,sy=40;
  layers.forEach(function(l,i){
    var y=sy+i*(bh+8);
    var desc=isTrain?l.trainDesc:l.evalDesc;
    var sub=isTrain?l.trainSub:l.evalSub;
    var lcol=isTrain?l.trainColor:l.evalColor;
    var risk=isTrain?l.trainRisk:l.evalRisk;
    bx(ctx,10,y,W-20,bh,6,lcol+'15',lcol+(risk?'':'50'),risk?2:1.5);
    tx(ctx,l.name,20,y+13,lcol,9,'800','left');
    tx(ctx,desc,20,y+27,'#e4e4e7',8,'400','left');
    tx(ctx,sub,20,y+40,'#52525b',7,'400','left');
    if(risk){bx(ctx,W-165,y+6,150,14,3,'#ef444430',null);tx(ctx,'\u26A0 '+risk,W-90,y+13,'#ef4444',5,'800');}
  });
  document.getElementById('iMode').innerHTML=isTrain?
    '<span class="hl2">model.train():</span> Dropout randomly masks activations (regularisation). BatchNorm computes fresh batch \u03BC/\u03C3\u00B2 and updates running stats. Required before every training phase.':
    '<span class="hl4">model.eval():</span> Dropout passes all activations unchanged. BatchNorm uses accumulated running stats (no update). Essential for deterministic, correct inference. <span class="hl5">Never forget this before validation or deployment.</span>';
}

function renderLayerRef(){
  var ctx=clr('cvLayerRef',860,155),W=860,H=155;
  var layers=[
    {name:'nn.Dropout',trainB:'Random zero + scale',evalB:'Identity (pass-thru)',col:'#fbbf24',sensitive:true},
    {name:'nn.BatchNorm',trainB:'Batch \u03BC,\u03C3\u00B2 + update running',evalB:'Running mean/var (no update)',col:'#ff6b35',sensitive:true},
    {name:'nn.LayerNorm',trainB:'Per-example stats',evalB:'Per-example stats (same)',col:'#4ade80',sensitive:false},
    {name:'nn.GroupNorm',trainB:'Group stats',evalB:'Group stats (same)',col:'#4ade80',sensitive:false},
    {name:'nn.Linear / Conv',trainB:'Same computation',evalB:'Same computation',col:'#4ade80',sensitive:false},
    {name:'MultiheadAttention',trainB:'Attention dropout active',evalB:'Attention dropout off',col:'#fbbf24',sensitive:true},
  ];
  var colW=(W-20)/layers.length,sx=10;
  layers.forEach(function(l,i){
    var x=sx+i*colW;
    bx(ctx,x,8,colW-6,H-16,6,l.col+'15',l.col+(l.sensitive?'80':'30'),l.sensitive?2:1);
    tx(ctx,l.name,x+(colW-6)/2,26,l.col,8,'800');
    if(l.sensitive){bx(ctx,x+4,38,colW-14,14,3,'#ef444430','#ef4444',1);tx(ctx,'MODE-SENSITIVE',x+(colW-6)/2,45,'#ef4444',6,'800');}
    else{bx(ctx,x+4,38,colW-14,14,3,'#4ade8018','#4ade8040',1);tx(ctx,'identical both modes',x+(colW-6)/2,45,'#4ade8060',6,'400');}
    tx(ctx,'TRAIN:',x+8,68,l.col,7,'800','left');
    var tw=l.trainB.split(' ');var tl=[''];
    tw.forEach(function(w){if((tl[tl.length-1]+' '+w).length>colW/7)tl.push(w);else tl[tl.length-1]+=(tl[tl.length-1]?' ':'')+w;});
    tl.forEach(function(line,li){tx(ctx,line,x+8,78+li*11,'#94a3b8',6,'400','left');});
    tx(ctx,'EVAL:',x+8,108,'#4ade80',7,'800','left');
    var ew=l.evalB.split(' ');var el=[''];
    ew.forEach(function(w){if((el[el.length-1]+' '+w).length>colW/7)el.push(w);else el[el.length-1]+=(el[el.length-1]?' ':'')+w;});
    el.forEach(function(line,li){tx(ctx,line,x+8,118+li*11,'#52525b',6,'400','left');});
  });
}

// ============================================================
// TAB 1: DROPOUT
// ============================================================
var curDropMode='train';
function selDropMode(k){curDropMode=k;['train','eval'].forEach(function(b){document.getElementById('dpBtn_'+b).classList.remove('active');});document.getElementById('dpBtn_'+k).classList.add('active');renderDropout();}
function renderDropout(){
  var p=parseFloat(document.getElementById('slDropP').value);
  document.getElementById('vDropP').textContent=rnd(p,1);
  var ctx=clr('cvDropout',420,280),W=420,H=280;
  var N=8,scale=1/(1-p);
  var activations=[0.8,0.5,0.9,0.3,0.7,0.6,0.4,0.85];
  // Deterministic mask based on p for display
  var masks=[[1,0,1,0,1,1,0,1],[0,1,0,1,0,1,1,0],[1,1,0,0,1,0,1,1]];
  var maskIdx=Math.floor(p*10)%3;
  var mask=curDropMode==='eval'?[1,1,1,1,1,1,1,1]:masks[maskIdx];

  var isTrain=curDropMode==='train';
  tx(ctx,isTrain?'TRAIN MODE \u2014 stochastic':'EVAL MODE \u2014 deterministic',W/2,18,isTrain?'#fbbf24':'#4ade80',10,'800');
  if(isTrain)tx(ctx,'(each pass shows a different mask)',W/2,32,'#52525b',7,'400');
  else tx(ctx,'(always identical output)',W/2,32,'#52525b',7,'400');

  var bw=(W-40)/N,sy=50;
  activations.forEach(function(a,i){
    var x=20+i*bw;
    var alive=mask[i]===1;
    var out=isTrain?(alive?a*scale:0):a;
    // Input neuron
    bx(ctx,x+4,sy,bw-8,30,4,'#4ecdc420','#4ecdc4',1.5);
    tx(ctx,rnd(a,1),x+bw/2,sy+15,'#4ecdc4',9,'700');
    // Arrow & mask
    if(isTrain){
      var mc=alive?'#4ade80':'#ef4444';
      ctx.fillStyle=mc;ctx.font='800 11px JetBrains Mono,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(alive?'\u00D7'+rnd(scale,1):'\u00D70',x+bw/2,sy+46);
    } else {
      arr(ctx,x+bw/2,sy+30,x+bw/2,sy+58,'#4ade8060',1.5);
    }
    // Output neuron
    var ocol=alive?'#4ade80':'#1e1e2e';
    var ofill=alive?'#4ade8025':'#09090f';
    bx(ctx,x+4,sy+60,bw-8,30,4,ofill,ocol,alive?1.5:0.5);
    tx(ctx,alive?rnd(out,1):'0',x+bw/2,sy+75,alive?'#4ade80':'#2d2d40',9,'700');
    if(isTrain&&!alive){
      ctx.strokeStyle='#ef444460';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(x+6,sy+62);ctx.lineTo(x+bw-6,sy+88);ctx.stroke();
    }
  });
  // E[output] annotation
  var sumIn=activations.reduce(function(a,b){return a+b;},0);
  var sumOut=mask.reduce(function(s,m,i){return s+(isTrain?(m?activations[i]*scale:0):activations[i]);},0);
  tx(ctx,'Input sum: '+rnd(sumIn,2),W/4,sy+112,'#4ecdc4',8,'700');
  tx(ctx,'Output sum: '+rnd(sumOut,2),3*W/4,sy+112,Math.abs(sumOut-sumIn)<0.5?'#4ade80':'#ef4444',8,'700');
  tx(ctx,'Input neurons',W/2,sy-8,'#52525b',7,'400');
  tx(ctx,'Output (after dropout)',W/2,sy+95,'#52525b',7,'400');
  // Inverted dropout key
  bx(ctx,10,H-56,W-20,20,4,'#fbbf2415','#fbbf2440',1);
  tx(ctx,'Inverted dropout: training scales active neurons by 1/(1-p)='+rnd(scale,2)+'. Eval needs no rescaling \u2014 E[out] matches.',W/2,H-46,'#fbbf24',7,'400');
  var deadN=mask.filter(function(m){return m===0;}).length;
  document.getElementById('iDropout').innerHTML=
    'p=<span class="hl3">'+rnd(p,1)+'</span>. Scale=<span class="hl3">'+rnd(scale,2)+'</span>.<br>'+
    (isTrain?deadN+'/'+N+' neurons dropped this pass. Different mask every forward pass \u2192 <span class="hl3">stochastic regularisation</span>.':'All '+N+' neurons active. <span class="hl4">Deterministic output</span>. No scaling needed (inverted dropout ensures E[out] matches).'+
    '<br>Verify determinism: torch.allclose(model(x), model(x)) should be True.');
}

function renderMCDrop(){
  var T=parseInt(document.getElementById('slMCT').value);
  document.getElementById('vMCT').textContent=T;
  var ctx=clr('cvMCDrop',420,280),W=420,H=280,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Class probability','#52525b',null);tx(ctx,'Class probability',PAD+20,22,'#52525b',8,'400');
  tx(ctx,'MC sample index',W/2,H-5,'#52525b',8,'400');
  // Simulate T predictions (noisy due to dropout)
  resetSeed(77);
  var trueP=0.72;
  var preds=[];
  for(var i=0;i<T;i++){preds.push(trueP+(rng()-0.5)*0.25);}
  var mean=preds.reduce(function(a,b){return a+b;},0)/T;
  var variance=preds.map(function(x){return Math.pow(x-mean,2);}).reduce(function(a,b){return a+b;},0)/T;
  // Plot predictions
  preds.forEach(function(pred,i){
    var cx=PAD+(i/(T-1))*pW;
    var cy=H-PAD-pred*pH;
    ctx.fillStyle='#c084fc40';ctx.beginPath();ctx.arc(cx,Math.max(16,Math.min(H-PAD,cy)),3,0,Math.PI*2);ctx.fill();
  });
  // Mean line
  var meanY=H-PAD-mean*pH;
  ln(ctx,PAD,meanY,W-10,meanY,'#4ade80',2.5);
  tx(ctx,'mean='+rnd(mean,3),W-14,meanY-8,'#4ade80',8,'800','right');
  // Std band
  var stdY1=H-PAD-(mean+Math.sqrt(variance))*pH;
  var stdY2=H-PAD-(mean-Math.sqrt(variance))*pH;
  ctx.fillStyle='#4ade8015';ctx.fillRect(PAD,Math.max(15,stdY1),pW,Math.min(H-PAD,stdY2)-Math.max(15,stdY1));
  tx(ctx,'\u00B1\u03C3='+rnd(Math.sqrt(variance),3),PAD+10,meanY-20,'#4ade8060',7,'700','left');
  tx(ctx,'T='+T+' MC samples (model.train() + no_grad())',W/2,22,'#c084fc',9,'800');
  tx(ctx,'Spread = epistemic uncertainty',W/2,H-PAD+18,'#52525b',8,'400');
  document.getElementById('iMCDrop').innerHTML=
    'T=<span class="hl3">'+T+'</span> forward passes. mean=<span class="hl4">'+rnd(mean,3)+'</span>. uncertainty (var)=<span class="hl6">'+rnd(variance,4)+'</span><br>'+
    '<span class="hl3">MC Dropout:</span> intentionally call model.<span class="hl5">train()</span> during inference. Each pass has a different dropout mask \u2192 different prediction. Variance across T passes = epistemic uncertainty.<br>'+
    '<span class="hl5">NOT a bug.</span> Do not confuse with forgetting eval(). MC Dropout is deliberate and requires design.';
}

function renderForgetEval(){
  var ctx=clr('cvForgetEval',860,148),W=860,H=148;
  var symptoms=[
    {title:'Stochastic output',col:'#ef4444',desc:'Same input, different output each call. torch.allclose(out1,out2)=False.',fix:'model.eval() before inference.'},
    {title:'Lower output magnitude',col:'#fbbf24',desc:'~(1-p)\u00D7 smaller activations if not inverted dropout. Predictions biased toward 0.',fix:'model.eval() ensures no 1/(1-p) scale mismatch.'},
    {title:'Poor calibration',col:'#ff6b35',desc:'Dropout noise inflates logit variance. Model appears less confident than it is.',fix:'model.eval() removes stochasticity from logits.'},
    {title:'Wrong val accuracy',col:'#c084fc',desc:'Val loss is higher and accuracy lower than true performance. Early stopping too early.',fix:'model.eval() before val loop. Never measure with dropout active.'},
  ];
  var bw=(W-20)/symptoms.length-6,sx=10;
  symptoms.forEach(function(s,i){
    var x=sx+i*(bw+6);
    bx(ctx,x,10,bw,H-20,6,s.col+'15',s.col+'60',1.5);
    tx(ctx,s.title,x+bw/2,28,s.col,9,'800');
    var words=s.desc.split(' ');var lines=[''];
    words.forEach(function(w){if((lines[lines.length-1]+' '+w).length<bw/7+2)lines[lines.length-1]+=(lines[lines.length-1]?' ':'')+w;else lines.push(w);});
    lines.forEach(function(l,li){tx(ctx,l,x+bw/2,48+li*12,'#94a3b8',6,'400');});
    bx(ctx,x+4,H-36,bw-8,22,4,'#4ade8018','#4ade8050',1);
    tx(ctx,'Fix: '+s.fix,x+bw/2,H-25,'#4ade80',6,'700');
  });
}

// ============================================================
// TAB 2: BATCHNORM
// ============================================================
var curBN='train';
function selBN(k){curBN=k;['train','eval'].forEach(function(b){document.getElementById('bnBtn_'+b).classList.remove('active');});document.getElementById('bnBtn_'+k).classList.add('active');renderBN();}
function renderBN(){
  var ctx=clr('cvBN',420,270),W=420,H=270;
  var isTrain=curBN==='train';
  tx(ctx,isTrain?'TRAIN MODE \u2014 batch statistics':'EVAL MODE \u2014 running statistics',W/2,20,isTrain?'#fbbf24':'#4ade80',10,'800');
  // Show a batch of 6 values in train mode, and a running_mean in eval mode
  var batchVals=[2.1,3.4,1.8,4.2,2.9,3.1];
  var N=batchVals.length;
  var batchMu=batchVals.reduce(function(a,b){return a+b;},0)/N;
  var batchVar=batchVals.reduce(function(s,v){return s+Math.pow(v-batchMu,2);},0)/N;
  var runMu=3.0,runVar=0.81; // pre-accumulated running stats
  var eps=1e-5;
  var mu=isTrain?batchMu:runMu;
  var vr=isTrain?batchVar:runVar;
  // Draw batch values
  var bh=30,sy=48;
  batchVals.forEach(function(v,i){
    var x=20+i*(W-40)/N;
    bx(ctx,x,sy,56,bh,4,'#4ecdc420','#4ecdc4',1.5);
    tx(ctx,'x'+i+'='+rnd(v,1),x+28,sy+15,'#4ecdc4',9,'700');
  });
  // Stats box
  if(isTrain){
    bx(ctx,20,sy+bh+16,W-40,36,6,'#fbbf2418','#fbbf24',2);
    tx(ctx,'\u03BC_B = '+rnd(batchMu,2)+'   \u03C3\u00B2_B = '+rnd(batchVar,2)+'   (computed from this batch)',W/2,sy+bh+28,'#fbbf24',9,'800');
    tx(ctx,'Updates running_mean += 0.1 \u00D7 (\u03BC_B \u2212 running_mean)',W/2,sy+bh+44,'#52525b',7,'400');
  } else {
    bx(ctx,20,sy+bh+16,W-40,36,6,'#4ade8018','#4ade80',2);
    tx(ctx,'running_mean = '+rnd(runMu,2)+'   running_var = '+rnd(runVar,2)+'   (accumulated over training)',W/2,sy+bh+28,'#4ade80',9,'800');
    tx(ctx,'No update to running stats \u2014 read-only at inference',W/2,sy+bh+44,'#52525b',7,'400');
  }
  // Normalised output
  var normVals=batchVals.map(function(v){return (v-mu)/Math.sqrt(vr+eps);});
  normVals.forEach(function(v,i){
    var x=20+i*(W-40)/N;
    bx(ctx,x,sy+bh+68,56,bh,4,'#c084fc20','#c084fc',1.5);
    tx(ctx,rnd(v,2),x+28,sy+bh+83,'#c084fc',9,'700');
  });
  tx(ctx,'\u0302x_i = (x_i \u2212 '+rnd(mu,2)+') / sqrt('+rnd(vr,2)+'+\u03B5)  \u2192  normalised:',W/2,sy+bh+60,'#52525b',7,'400');
  // Final y = gamma * xhat + beta
  tx(ctx,'y_i = \u03B3 \u00D7 \u0302x_i + \u03B2   (\u03B3,\u03B2 are learned affine params, same in both modes)',W/2,sy+bh+116,'#94a3b8',7,'400');
  document.getElementById('iBN').innerHTML=isTrain?
    '<span class="hl3">Train mode:</span> \u03BC_B and \u03C3\u00B2_B computed fresh from current batch. Running stats updated via EMA. Provides regularisation (stochastic normalisation). BUT: B=1 causes \u03C3\u00B2_B=0 \u2192 division by \u03B5 only \u2192 degenerate output.':
    '<span class="hl4">Eval mode:</span> uses stable running_mean and running_var accumulated over training. No update to running stats (read-only). Safe at any batch size including B=1.';
}

function renderBNConv(){
  var momPct=parseInt(document.getElementById('slMom').value);
  var mom=momPct/100;
  document.getElementById('vMom').textContent=rnd(mom,2);
  var ctx=clr('cvBNConv',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Training step','#52525b',null);tx(ctx,'Training step',W/2,H-5,'#52525b',8,'400');
  tx(ctx,'running_mean value','#52525b',null);tx(ctx,'running_mean',PAD+20,22,'#52525b',8,'400');
  var trueMu=3.0;
  var N2=100;
  var running=0,pts=[];
  resetSeed(55);
  for(var i=0;i<N2;i++){
    var batchMu2=trueMu+(rng()-0.5)*1.5; // noisy batch means
    running=(1-mom)*running+mom*batchMu2;
    pts.push([PAD+(i/N2)*pW,H-PAD-Math.min(Math.max(running,0),4)/4*pH]);
  }
  // True mean line
  ln(ctx,PAD,H-PAD-trueMu/4*pH,W-10,H-PAD-trueMu/4*pH,'#4ade8040',1.5,[4,4]);
  tx(ctx,'True \u03BC='+rnd(trueMu,1),W-14,H-PAD-trueMu/4*pH-8,'#4ade8040',7,'700','right');
  // Convergence annotation
  var convStep=Math.round(1/mom)-1;
  var convX=PAD+(convStep/N2)*pW;
  if(convX<W-10){ln(ctx,convX,15,convX,H-PAD,'#fbbf2450',1,[3,3]);tx(ctx,'~'+convStep+' steps\nto converge',convX+4,H-PAD-pH*0.7,'#fbbf24',7,'700','left');}
  // Running mean curve
  ctx.strokeStyle='#ff6b35';ctx.lineWidth=2.5;ctx.beginPath();
  pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();
  tx(ctx,'running_mean (momentum='+rnd(mom,2)+')',W/2,22,'#ff6b35',9,'800');
  var convSteps=Math.round(1/mom);
  document.getElementById('iBNConv').innerHTML=
    'Momentum=<span class="hl3">'+rnd(mom,2)+'</span>. Convergence in ~<span class="hl3">'+convSteps+' steps</span>.<br>'+
    'running_mean \u2190 (1\u2212'+rnd(mom,2)+') \u00D7 running_mean + '+rnd(mom,2)+' \u00D7 \u03BC_B<br>'+
    '<span class="hl5">Problem:</span> if model.eval() is called before convergence (step &lt;'+convSteps+'), running stats are unreliable \u2192 poor normalisation at inference.';
}

function renderBNBs1(){
  var ctx=clr('cvBNBs1',860,148),W=860,H=148;
  var half=W/2-20;
  // Train mode B=1 (disaster)
  bx(ctx,10,10,half,H-20,8,'#ef444415','#ef4444',2);
  tx(ctx,'TRAIN MODE, Batch Size = 1',10+half/2,28,'#ef4444',10,'800');
  tx(ctx,'x\u2081 = 2.7',10+half/4,50,'#4ecdc4',9,'700');
  tx(ctx,'\u03BC_B = x\u2081 = 2.7   (trivially)',10+half/2,66,'#fbbf24',8,'700');
  tx(ctx,'\u03C3\u00B2_B = 0   (variance of a single point)',10+half/2,80,'#fbbf24',8,'700');
  tx(ctx,'\u0302x = (2.7 - 2.7) / \u221A(0 + \u03B5)  =  0 / \u03B5\u207D \u2248 0',10+half/2,96,'#ef4444',9,'800');
  bx(ctx,14,108,half-8,26,4,'#ef444430','#ef4444',2);
  tx(ctx,'\u2715 OUTPUT IS ALWAYS ZERO \u2014 destroys learned representations!',10+half/2,121,'#ef4444',8,'800');

  // Eval mode B=1 (safe)
  var ox=W/2+10;
  bx(ctx,ox,10,half,H-20,8,'#4ade8015','#4ade80',2);
  tx(ctx,'EVAL MODE, Batch Size = 1',ox+half/2,28,'#4ade80',10,'800');
  tx(ctx,'x\u2081 = 2.7',ox+half/4,50,'#4ecdc4',9,'700');
  tx(ctx,'running_mean = 3.0   running_var = 0.81   (stable)',ox+half/2,66,'#4ade80',8,'700');
  tx(ctx,'\u0302x = (2.7 - 3.0) / \u221A(0.81 + \u03B5)  =  \u22120.33',ox+half/2,82,'#4ade80',9,'800');
  tx(ctx,'y = \u03B3 \u00D7 (\u22120.33) + \u03B2  =  correct normalised output',ox+half/2,98,'#94a3b8',8,'400');
  bx(ctx,ox+4,108,half-8,26,4,'#4ade8030','#4ade80',2);
  tx(ctx,'\u2714 Correct normalisation at any batch size. Always use model.eval() for deployment.',ox+half/2,121,'#4ade80',8,'800');
}

// ============================================================
// TAB 3: NO_GRAD & MEMORY
// ============================================================
var curGraph='train';
function selGraph(k){curGraph=k;['train','nogr','infm'].forEach(function(b){document.getElementById('grBtn_'+b).classList.remove('active');});document.getElementById('grBtn_'+k).classList.add('active');renderGraph();}
function renderGraph(){
  var ctx=clr('cvGraph',420,270),W=420,H=270;
  var isTrain=curGraph==='train';
  var isNoGr=curGraph==='nogr';
  var isInfm=curGraph==='infm';
  var col=isTrain?'#ff6b35':isNoGr?'#fbbf24':'#4ade80';
  tx(ctx,isTrain?'Default (autograd enabled)':isNoGr?'torch.no_grad()':'torch.inference_mode()',W/2,20,col,10,'800');

  var ops=['input','Linear','ReLU','Linear','output'];
  var opW=60,gap=(W-80-(ops.length*opW))/(ops.length-1);
  var nodeY=90,actY=150;
  ops.forEach(function(op,i){
    var x=40+i*(opW+gap);
    bx(ctx,x,nodeY,opW,36,6,'#4ecdc425','#4ecdc4',1.5);
    tx(ctx,op,x+opW/2,nodeY+18,'#4ecdc4',9,'800');
    if(i<ops.length-1){arr(ctx,x+opW,nodeY+18,x+opW+gap,nodeY+18,'#3f3f46',1.5);}
    // Activation stored?
    if(isTrain&&i>0&&i<ops.length-1){
      bx(ctx,x+8,actY,opW-16,22,4,'#ef444430','#ef4444',1.5);
      tx(ctx,'stored',x+opW/2,actY+11,'#ef4444',7,'700');
      arr(ctx,x+opW/2,nodeY+36,x+opW/2,actY,'#ef444460',1);
    } else if(!isTrain&&i>0&&i<ops.length-1){
      bx(ctx,x+8,actY,opW-16,22,4,'#4ade8018','#4ade8040',1);
      tx(ctx,'not stored',x+opW/2,actY+11,'#4ade8060',7,'400');
    }
  });
  // Graph line
  if(isTrain){
    bx(ctx,30,actY+28,W-60,22,6,'#ef444418','#ef4444',1.5);
    tx(ctx,'Computation graph in memory (for backward())',W/2,actY+39,'#ef4444',8,'700');
    bx(ctx,30,actY+56,W-60,22,6,'#ef444410','#ef4444',1);
    tx(ctx,'ResNet-50 @ B=256: ~4 GB activation memory',W/2,actY+67,'#ef444480',7,'400');
  } else {
    bx(ctx,30,actY+28,W-60,22,6,'#4ade8018','#4ade80',2);
    tx(ctx,'No graph. No activations stored. 40-60% less memory.',W/2,actY+39,'#4ade80',8,'700');
    if(isInfm){
      bx(ctx,30,actY+56,W-60,22,6,'#4ade8010','#4ade8050',1);
      tx(ctx,'inference_mode: even stricter (cannot be used in grad computation)',W/2,actY+67,'#4ade8070',7,'400');
    }
  }
  // requires_grad label
  bx(ctx,30,H-28,W-60,20,4,col+'15',col+'50',1);
  tx(ctx,isTrain?'Tensors: requires_grad=True':isInfm?'Tensors: special inference_mode (no grad, strict)':'Tensors: requires_grad=False',W/2,H-18,col,8,'700');
  document.getElementById('iGraph').innerHTML=isTrain?
    '<span class="hl5">With autograd:</span> PyTorch builds a DAG storing all intermediate activations for backward(). At ResNet-50 scale with B=256 this is ~4 GB wasted at inference. Never use this for val or deployment.':
    isNoGr?
    '<span class="hl3">torch.no_grad():</span> no graph built, no activations stored. 40-60% memory saving, 20-30% faster. Output tensors have requires_grad=False. Can be nested with gradient contexts.':
    '<span class="hl4">torch.inference_mode():</span> even stricter than no_grad. Tensors cannot participate in any gradient computation. Slightly faster. Cannot be nested. Use for pure production inference.';
}

function renderMemory(){
  var depth=parseInt(document.getElementById('slDepth').value);
  document.getElementById('vDepth').textContent=depth+' layers';
  var ctx=clr('cvMemory',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Model depth (layers)','#52525b',null);tx(ctx,'Model depth',W/2,H-5,'#52525b',8,'400');
  tx(ctx,'Memory (relative)','#52525b',null);tx(ctx,'Memory',PAD+20,22,'#52525b',8,'400');
  var maxD=48,maxMem=4.5;
  // Training: scales with depth (activations per layer)
  var trainPts=[],nograPts=[],basePts=[];
  for(var d=1;d<=maxD;d++){
    var cx=PAD+(d-1)/(maxD-1)*pW;
    var trainMem=1.0+d*0.07; // grows with depth
    var nograMem=0.4; // just weights and current ops
    var baseMem=0.4;
    trainPts.push([cx,H-PAD-Math.min(trainMem/maxMem,1)*pH]);
    nograPts.push([cx,H-PAD-Math.min(nograMem/maxMem,1)*pH]);
    basePts.push([cx,H-PAD-Math.min(baseMem/maxMem,1)*pH]);
  }
  ctx.fillStyle='#ff6b3515';ctx.beginPath();ctx.moveTo(trainPts[0][0],trainPts[0][1]);
  trainPts.forEach(function(p){ctx.lineTo(p[0],p[1]);});nograPts.slice().reverse().forEach(function(p){ctx.lineTo(p[0],p[1]);});ctx.closePath();ctx.fill();
  ctx.strokeStyle='#ff6b35';ctx.lineWidth=2.5;ctx.beginPath();trainPts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();
  ctx.strokeStyle='#4ade80';ctx.lineWidth=2.5;ctx.beginPath();nograPts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();
  // Current depth marker
  var curX=PAD+(depth-1)/(maxD-1)*pW;
  var trainMemCur=1.0+depth*0.07;
  var noGrMemCur=0.4;
  var saving=Math.round((1-noGrMemCur/trainMemCur)*100);
  ln(ctx,curX,15,curX,H-PAD,'#fbbf2450',1.5,[4,4]);
  tx(ctx,saving+'% saving',curX+4,40,'#fbbf24',9,'800','left');
  tx(ctx,'\u2014 with autograd',W-14,trainPts[maxD-1][1],'#ff6b35',8,'700','right');
  tx(ctx,'\u2014 no_grad / inference_mode',W-14,nograPts[maxD-1][1]-12,'#4ade80',8,'700','right');
  document.getElementById('iMemory').innerHTML=
    depth+' layers. Memory saving with no_grad: <span class="hl4">~'+saving+'%</span>.<br>'+
    'Autograd stores intermediate activations proportional to depth. Deeper model \u2192 more wasted memory at inference.<br>'+
    '<span class="hl3">Rule of thumb:</span> ResNet-50 @ B=256 stores ~4 GB, GPT-2 medium @ B=32 stores ~8 GB. All wasted without no_grad.';
}

// ============================================================
// TAB 4: TRAINING LOOP
// ============================================================
function renderLoop(){
  var ctx=clr('cvLoop',420,310),W=420,H=310;
  var blocks=[
    {label:'for epoch in range(max_epochs):',col:'#3f3f46',y:10,h:20,sub:null},
    {label:'  model.train()',col:'#4ecdc4',y:36,h:24,sub:'[1] Once per epoch, before batch loop',tag:'TRAIN MODE'},
    {label:'  for batch in train_loader:',col:'#3f3f46',y:66,h:20,sub:null},
    {label:'    forward + backward + step',col:'#4ecdc4',y:92,h:24,sub:'Gradients needed \u2014 no no_grad here'},
    {label:'  model.eval()',col:'#4ade80',y:130,h:24,sub:'[2] Once per epoch, before val loop',tag:'EVAL MODE'},
    {label:'  with torch.no_grad():',col:'#4ade80',y:160,h:24,sub:'[3] Wrap entire val loop'},
    {label:'    for batch in val_loader:',col:'#3f3f46',y:190,h:20,sub:null},
    {label:'      outputs = model(inputs)',col:'#4ade80',y:216,h:24,sub:'Deterministic, memory-efficient'},
    {label:'    scheduler / checkpoint',col:'#fbbf24',y:250,h:24,sub:'Decisions based on val metrics'},
    {label:'  # model.train() [4] to restart',col:'#ff6b35',y:280,h:24,sub:'Critical if loop continues',tag:'BACK TO TRAIN'},
  ];
  blocks.forEach(function(b){
    bx(ctx,10,b.y,W-20,b.h,4,b.col+'15',b.col+(b.tag?'':'30'),b.tag?2:1);
    tx(ctx,b.label,18,b.y+b.h/2,b.col,8,'700','left');
    if(b.tag){bx(ctx,W-90,b.y+3,78,b.h-6,3,b.col+'30',b.col+'70',1);tx(ctx,b.tag,W-51,b.y+b.h/2,b.col,6,'800');}
    if(b.sub)tx(ctx,b.sub,18,b.y+b.h+1,'#52525b',6,'400','left');
  });
  document.getElementById('iLoop').innerHTML=
    '<span class="hl2">[1] model.train()</span> once before the training loop \u2014 not inside per-batch loop.<br>'+
    '<span class="hl4">[2] model.eval()</span> once before validation \u2014 must come before the loop, not after.<br>'+
    '<span class="hl4">[3] torch.no_grad()</span> wraps the ENTIRE val loop.<br>'+
    '<span class="hl">[4] model.train()</span> before returning to training if another epoch follows.';
}

var curMistake=2;
var MISTAKES=[
  {n:1,label:'eval() inside batch loop',col:'#fbbf24',
   code:['for batch in val_loader:','  model.eval()   \u2190 harmless but wasteful','  outputs = model(batch)'],
   effect:'Correct output but adds per-batch overhead.',sev:'Low'},
  {n:2,label:'Forget train() after validation',col:'#ef4444',
   code:['evaluate()   \u2190 calls model.eval() inside','for batch in train_loader:','  loss = model(batch)   \u2190 EVAL MODE!'],
   effect:'Dropout OFF during training. BN uses running stats. No regularisation!',sev:'High'},
  {n:3,label:'no_grad() inside training loop',col:'#ef4444',
   code:['for batch in train_loader:','  with torch.no_grad():  \u2190 WRONG','    outputs = model(batch)','  loss.backward()  \u2190 RuntimeError!'],
   effect:'RuntimeError: element 0 of tensors does not require grad.',sev:'Critical'},
  {n:4,label:'No no_grad() in validation',col:'#fbbf24',
   code:['model.eval()','for batch in val_loader:','  outputs = model(batch)  \u2190 graph built','  loss = criterion(...)'],
   effect:'Correct numerics but 2\u00D7 memory. May OOM on large models.',sev:'Medium'},
  {n:5,label:'eval() once, never switching back',col:'#ef4444',
   code:['model.eval()   \u2190 set at script start','for epoch in range(epochs):','  for batch in train_loader:','    loss = model(batch)  \u2190 EVAL MODE!'],
   effect:'Identical to Mistake 2. Training without regularisation for entire run.',sev:'High'},
];
function selMistake(n){curMistake=n;for(var i=1;i<=5;i++){document.getElementById('msBtn_'+i).classList.remove('active');}document.getElementById('msBtn_'+n).classList.add('active');renderMistakes();}
function renderMistakes(){
  var m=MISTAKES[curMistake-1];
  var ctx=clr('cvMistakes',420,310),W=420,H=310;
  var sevCol={'Low':'#4ade80','Medium':'#fbbf24','High':'#ef4444','Critical':'#c084fc'}[m.sev];
  bx(ctx,10,10,W-20,32,6,m.col+'20',m.col,2);
  tx(ctx,'Mistake '+m.n+': '+m.label,W/2,26,m.col,10,'800');
  // Code block
  bx(ctx,10,52,W-20,m.code.length*22+8,6,'#0d0d18','#1e1e2e',1);
  m.code.forEach(function(line,i){
    var isWrong=line.indexOf('\u2190')>=0;
    tx(ctx,line,20,64+i*22,isWrong?m.col:'#94a3b8',8,isWrong?'800':'400','left');
  });
  var codeH=m.code.length*22+16;
  bx(ctx,10,54+codeH,W-20,46,6,'#ef444418','#ef4444',2);
  tx(ctx,'Effect:',W/2,64+codeH,'#ef4444',8,'800');
  tx(ctx,m.effect,W/2,78+codeH,'#94a3b8',8,'400');
  bx(ctx,10,108+codeH,W-20,26,6,sevCol+'20',sevCol,1.5);
  tx(ctx,'Severity: '+m.sev,W/2,121+codeH,sevCol,9,'800');
  // Fix
  var fixes={1:'Move model.eval() outside the batch loop (before it).',2:'Call model.train() at the start of each training phase.',3:'Never use no_grad() in the training loop. Gradients are needed.',4:'Wrap the val loop with torch.no_grad(). Always.',5:'Call model.train() at the start of every training epoch.'};
  bx(ctx,10,142+codeH,W-20,26,6,'#4ade8018','#4ade80',1.5);
  tx(ctx,'Fix: '+fixes[m.n],W/2,155+codeH,'#4ade80',8,'700');
  document.getElementById('iMistakes').innerHTML='<span class="hl">Mistake '+m.n+':</span> <span style="color:'+m.col+'">'+m.label+'</span><br><span class="hl5">Effect:</span> '+m.effect+'<br><span class="hl4">Fix:</span> '+fixes[m.n];
}

// ============================================================
// TAB 5: FINE-TUNING & DEPLOY
// ============================================================
var curFT='full';
function selFT(k){curFT=k;['full','linear','grad','lora'].forEach(function(b){document.getElementById('ftBtn_'+b).classList.remove('active');});document.getElementById('ftBtn_'+k).classList.add('active');renderFT();}
function renderFT(){
  var ctx=clr('cvFinetune',420,270),W=420,H=270;
  var layers=[
    {name:'Input Embedding',idx:0},
    {name:'Block 1',idx:1},
    {name:'Block 2-N',idx:2},
    {name:'Block N',idx:3},
    {name:'Head (new)',idx:4},
  ];
  var configs={
    full:   {modes:['train','train','train','train','train'],grads:[true,true,true,true,true],title:'Full Fine-tuning: all layers train'},
    linear: {modes:['eval','eval','eval','eval','train'],grads:[false,false,false,false,true],title:'Linear Probing: backbone frozen + eval'},
    grad:   {modes:['eval','eval','eval','train','train'],grads:[false,false,false,true,true],title:'Gradual Unfreezing: last blocks active'},
    lora:   {modes:['eval','eval','eval','eval','eval'],grads:[false,false,false,false,false],title:'LoRA: base frozen, adapters only train'},
  };
  var cfg=configs[curFT];
  tx(ctx,cfg.title,W/2,18,'#ff6b35',9,'800');
  var bh=34,sy=36,bw=W-40;
  layers.forEach(function(l,i){
    var mode=cfg.modes[i];
    var grad=cfg.grads[i];
    var hasAdapter=curFT==='lora'&&i>=1&&i<=3;
    var col=mode==='train'?'#4ecdc4':'#3f3f46';
    bx(ctx,20,sy+i*(bh+6),bw,bh,6,col+'18',col,mode==='train'?2:1);
    tx(ctx,l.name,80,sy+i*(bh+6)+17,col,9,'800');
    // Mode badge
    bx(ctx,W-165,sy+i*(bh+6)+6,70,22,4,(mode==='train'?'#4ecdc4':'#3f3f46')+'30',null);
    tx(ctx,mode==='train'?'.train()':'.eval()',W-130,sy+i*(bh+6)+17,mode==='train'?'#4ecdc4':'#52525b',8,'700');
    // Grad badge
    bx(ctx,W-88,sy+i*(bh+6)+6,66,22,4,(grad?'#4ade80':'#ef4444')+'20',null);
    tx(ctx,grad?'grad=True':'frozen',W-55,sy+i*(bh+6)+17,grad?'#4ade80':'#ef4444',8,'700');
    // LoRA adapter
    if(hasAdapter){
      bx(ctx,24,sy+i*(bh+6)+4,40,bh-8,4,'#c084fc30','#c084fc',2);
      tx(ctx,'LoRA',44,sy+i*(bh+6)+bh/2,'#c084fc',7,'800');
    }
  });
  if(curFT==='lora')tx(ctx,'LoRA adapters train separately (not shown)',W/2,H-10,'#c084fc',7,'400');
  var ftInfo={
    full:'<span class="hl4">Full fine-tuning:</span> all layers in train mode, all grads enabled. Highest adaptation but risk of catastrophic forgetting. Use small LR (~2e-5).',
    linear:'<span class="hl2">Linear probing:</span> backbone frozen and in eval mode (preserves BN running stats). Only head trains. Fast, stable, but less flexible.',
    grad:'<span class="hl3">Gradual unfreezing:</span> unfreeze from top down. Each epoch unfreezes more layers. ULMFiT approach. Balance between adaptation and stability.',
    lora:'<span class="hl6">LoRA:</span> base model frozen in eval mode. Only small adapter modules (low-rank matrices) have requires_grad=True. No catastrophic forgetting.'
  };
  document.getElementById('iFT').innerHTML=ftInfo[curFT];
}

function renderDeploy(){
  var ctx=clr('cvDeploy',420,270),W=420,H=270;
  var steps=[
    {n:'1',label:'Load checkpoint',col:'#4ecdc4',sub:'torch.load(path, map_location=device)'},
    {n:'2',label:'model.eval()',col:'#4ade80',sub:'assert not model.training  \u2190 verify!'},
    {n:'3',label:'Move to device',col:'#4ecdc4',sub:'model = model.to(device)'},
    {n:'4',label:'(Optional) FP16',col:'#c084fc',sub:'torch.autocast for mixed precision'},
    {n:'5',label:'@torch.inference_mode()',col:'#4ade80',sub:'Decorator on predict() function'},
    {n:'6',label:'Validate outputs',col:'#fbbf24',sub:'isfinite, shape, metric match checkpoint'},
  ];
  var bh=30,sy=16;
  steps.forEach(function(s,i){
    var y=sy+i*(bh+8);
    bx(ctx,14,y,W-28,bh,6,s.col+'18',s.col+(i===1?'':'50'),i===1?2.5:1.5);
    bx(ctx,14,y,30,bh,6,s.col+'40',null);
    tx(ctx,s.n,29,y+bh/2,s.col,12,'800');
    tx(ctx,s.label,50,y+bh/2-5,s.col,9,'800','left');
    tx(ctx,s.sub,50,y+bh/2+7,'#52525b',7,'400','left');
    if(i===1){bx(ctx,W-115,y+5,100,20,4,'#4ade8030','#4ade80',1.5);tx(ctx,'NEVER forget!',W-65,y+15,'#4ade80',7,'800');}
    if(i<steps.length-1)tx(ctx,'\u2193',W/2,y+bh+4,'#2d2d40',10,'400');
  });
  document.getElementById('iDeploy').innerHTML=
    '<span class="hl5">Most common deployment bug:</span> forgetting model.eval() (Step 2). Output is stochastic and inconsistent.<br>'+
    '<span class="hl4">Steps 2 + 5 must always be paired:</span> model.eval() controls Dropout/BN; torch.inference_mode() controls the autograd engine. Both required.';
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){renderControls();renderMode();renderLayerRef();});
</script>
</body>
</html>"""

TRAIN_VS_INFERENCE_VISUAL_HEIGHT = 2200