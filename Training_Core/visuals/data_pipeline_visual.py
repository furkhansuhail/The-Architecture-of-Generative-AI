DATA_PIPELINE_VISUAL_HTML = r"""<!DOCTYPE html>
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
.v{font-family:'JetBrains Mono',monospace;font-size:0.73em;color:#ff6b35;min-width:44px;text-align:right;}
.info{background:#0d0d18;border:1px solid #1e1e2e;border-radius:8px;padding:10px 14px;font-size:0.76em;color:#94a3b8;line-height:1.8;margin-top:10px;font-family:'JetBrains Mono',monospace;}
.hl{color:#ff6b35;font-weight:700;}.hl2{color:#4ecdc4;font-weight:700;}.hl3{color:#fbbf24;font-weight:700;}.hl4{color:#4ade80;font-weight:700;}.hl5{color:#ef4444;font-weight:700;}.hl6{color:#c084fc;font-weight:700;}
.brow{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
button.s{background:#1e1e2e;color:#a1a1aa;border:1px solid #2d2d40;border-radius:6px;padding:5px 12px;cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:0.73em;font-weight:700;transition:all 0.15s;}
button.s:hover{background:#2d2d40;color:#e4e4e7;}button.s.active{background:#ff6b35;color:#08080f;border-color:#ff6b35;}
.tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.71em;}
.tbl th{padding:6px 9px;text-align:left;color:#52525b;border-bottom:2px solid #1e1e2e;font-weight:800;text-transform:uppercase;letter-spacing:.05em;}
.tbl td{padding:6px 9px;border-bottom:1px solid #0f0f18;color:#94a3b8;vertical-align:middle;line-height:1.5;}
.tbl tr:hover td{background:#0d0d18;}
.pipe{display:flex;align-items:center;flex-wrap:wrap;gap:0;}
.pbox{background:#0d0d18;border-radius:8px;padding:10px 12px;text-align:center;border:1.5px solid;flex:1;min-width:90px;}
.parr{color:#3f3f46;font-size:1.3em;padding:0 4px;flex-shrink:0;}
</style>
</head>
<body>
<h2>&#x1F501; Data Pipeline</h2>
<p class="sub">Batching &middot; Shuffling &middot; Augmentation &middot; DataLoader &middot; Normalisation &middot; Imbalance &middot; Bugs</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Pipeline Flow</button>
  <button class="tab" onclick="showTab(1)">Batching</button>
  <button class="tab" onclick="showTab(2)">Shuffling</button>
  <button class="tab" onclick="showTab(3)">Augmentation</button>
  <button class="tab" onclick="showTab(4)">DataLoader</button>
  <button class="tab" onclick="showTab(5)">Imbalance &amp; Bugs</button>
</div>

<!-- TAB 0: PIPELINE FLOW -->
<div id="tab0" class="panel active">
<div class="card">
  <h3>&#9312; End-to-End Data Pipeline</h3>
  <canvas id="cvPipe" width="860" height="200"></canvas>
  <div class="info" id="iPipe">Loading...</div>
</div>
<div class="g2">
<div class="card">
  <h3>&#9313; GPU Utilisation &mdash; Bottleneck Analyser</h3>
  <canvas id="cvGPU" width="420" height="230"></canvas>
  <div class="row"><label>GPU step (ms)</label>
    <input type="range" id="slGPU" min="10" max="200" step="5" value="50" oninput="renderGPU()">
    <span class="v" id="vGPU">50</span></div>
  <div class="row"><label>DataLoader (ms)</label>
    <input type="range" id="slDL" min="5" max="200" step="5" value="30" oninput="renderGPU()">
    <span class="v" id="vDL">30</span></div>
  <div class="info" id="iGPU">Loading...</div>
</div>
<div class="card">
  <h3>&#9314; CPU vs GPU Work Split</h3>
  <canvas id="cvCPUGPU" width="420" height="230"></canvas>
</div>
</div>
</div>

<!-- TAB 1: BATCHING -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Batch Size Effects</h3>
  <canvas id="cvBatch" width="420" height="260"></canvas>
  <div class="row"><label>Batch size B</label>
    <input type="range" id="slBatch" min="1" max="8" step="1" value="4" oninput="renderBatch()">
    <span class="v" id="vBatch">256</span></div>
  <div class="info" id="iBatch">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Sharp vs Flat Minima &mdash; Generalisation</h3>
  <canvas id="cvMinima" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="mBtn_sharp" onclick="selMinima('sharp')">Sharp (large batch)</button>
    <button class="s" id="mBtn_flat" onclick="selMinima('flat')">Flat (small batch)</button>
  </div>
  <div class="info" id="iMinima">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Sequence Padding vs Packing (NLP)</h3>
  <canvas id="cvPad" width="860" height="150"></canvas>
</div>
</div>

<!-- TAB 2: SHUFFLING -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Shuffled vs Unshuffled Batches</h3>
  <canvas id="cvShuffle" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="shBtn_no"  onclick="selShuffle('no')">No Shuffle</button>
    <button class="s" id="shBtn_yes" onclick="selShuffle('yes')">Shuffled</button>
  </div>
  <div class="info" id="iShuffle">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; BatchNorm Statistics: Impact of Shuffling</h3>
  <canvas id="cvBN" width="420" height="270"></canvas>
  <div class="info" id="iBN">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Epoch-Level vs Step-Level Shuffling &mdash; Coverage per Epoch</h3>
  <canvas id="cvEpoch" width="860" height="140"></canvas>
</div>
</div>

<!-- TAB 3: AUGMENTATION -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Train vs Test Transform Pipeline</h3>
  <canvas id="cvAug" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="augBtn_train" onclick="selAug('train')">Training</button>
    <button class="s" id="augBtn_val" onclick="selAug('val')">Validation / Test</button>
  </div>
  <div class="info" id="iAug">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Augmentation Strength Selector</h3>
  <canvas id="cvAugStrength" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="asBtn_flip"  onclick="selAS('flip')">H-Flip</button>
    <button class="s" id="asBtn_crop"  onclick="selAS('crop')">Rand Crop</button>
    <button class="s" id="asBtn_color" onclick="selAS('color')">ColorJitter</button>
    <button class="s" id="asBtn_mix"   onclick="selAS('mix')">Mixup</button>
    <button class="s" id="asBtn_rand"  onclick="selAS('rand')">RandAugment</button>
  </div>
  <div class="info" id="iAS">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Advanced Augmentation Strategies</h3>
  <table class="tbl">
    <thead><tr><th>Method</th><th>Operation</th><th>Effect</th><th>Typical gain</th><th>Use for</th></tr></thead>
    <tbody>
      <tr><td><span class="hl3">Mixup</span></td><td>x = &lambda;x&sub1; + (1-&lambda;)x&sub2;</td><td>Smooth decision boundaries, linear interpolation</td><td>+0.5&ndash;1.5% ImageNet</td><td>Image classification</td></tr>
      <tr><td><span class="hl2">CutMix</span></td><td>Paste patch from x&sub2; into x&sub1;</td><td>Natural two-region blending, label proportional to area</td><td>+0.5&ndash;2.0%</td><td>ViT, CNN classification</td></tr>
      <tr><td><span class="hl6">RandAugment</span></td><td>N random ops at magnitude M</td><td>Eliminates manual policy design</td><td>+1.0&ndash;2.5%</td><td>Standard for ViT, ResNet</td></tr>
      <tr><td><span class="hl4">TrivialAugment</span></td><td>1 random op at random magnitude</td><td>Simplest, matches RandAugment performance</td><td>\u2248 RandAugment</td><td>Drop-in replacement</td></tr>
      <tr><td><span class="hl">AugMix</span></td><td>Weighted mix of aug chains</td><td>Robustness to distribution shift</td><td>+1&ndash;3% on corruptions</td><td>Robustness benchmarks</td></tr>
      <tr><td><span class="hl5">TTA</span></td><td>Avg predictions over N augmentations</td><td>Inference-time ensemble</td><td>+0.3&ndash;0.5%</td><td>Competition / production</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 4: DATALOADER -->
<div id="tab4" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Prefetch Timeline &mdash; Workers vs GPU</h3>
  <canvas id="cvPrefetch" width="420" height="250"></canvas>
  <div class="row"><label>num_workers</label>
    <input type="range" id="slWorkers" min="0" max="8" step="1" value="4" oninput="renderPrefetch()">
    <span class="v" id="vWorkers">4</span></div>
  <div class="info" id="iPrefetch">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; pin_memory: Transfer Path</h3>
  <canvas id="cvPin" width="420" height="250"></canvas>
  <div class="brow">
    <button class="s active" id="pinBtn_off" onclick="selPin('off')">pin_memory=False</button>
    <button class="s" id="pinBtn_on"  onclick="selPin('on')">pin_memory=True</button>
  </div>
  <div class="info" id="iPin">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; DataLoader Gold Standard Configuration</h3>
  <table class="tbl">
    <thead><tr><th>Parameter</th><th>Recommended</th><th>Why</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">shuffle=True</span></td><td>Training only (False for val/test)</td><td>Epoch-level permutation. Re-shuffled each epoch automatically.</td></tr>
      <tr><td><span class="hl4">num_workers</span></td><td>8&ndash;16 (tune to GPU util)</td><td>Rule: CPU cores / num GPUs. Start at 4, increase until GPU saturates.</td></tr>
      <tr><td><span class="hl4">pin_memory=True</span></td><td>Always for GPU training</td><td>20&ndash;30% faster H2D transfer. Skip if RAM is limited.</td></tr>
      <tr><td><span class="hl4">persistent_workers=True</span></td><td>Almost always</td><td>Avoids 1&ndash;2s worker re-spawn overhead per epoch.</td></tr>
      <tr><td><span class="hl4">prefetch_factor=2</span></td><td>Default; increase to 4 if GPU waits</td><td>Keeps N&times;prefetch_factor batches ready in the buffer.</td></tr>
      <tr><td><span class="hl4">drop_last=True</span></td><td>When using BatchNorm</td><td>Prevents tiny final batch from producing unstable BN statistics.</td></tr>
      <tr><td><span class="hl3">worker_init_fn</span></td><td>Set unique per-worker seed</td><td>Prevents all workers using same random seed \u2192 correlated augmentation.</td></tr>
      <tr><td><span class="hl2">non_blocking=True</span></td><td>In .to(device) call</td><td>Overlaps CPU/GPU: batch.to(device, non_blocking=True)</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 5: IMBALANCE & BUGS -->
<div id="tab5" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Class Imbalance Strategies</h3>
  <canvas id="cvImbal" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="imBtn_none"    onclick="selImbal('none')">No Strategy</button>
    <button class="s" id="imBtn_weight"   onclick="selImbal('weight')">Class Weights</button>
    <button class="s" id="imBtn_over"     onclick="selImbal('over')">Oversampling</button>
    <button class="s" id="imBtn_focal"    onclick="selImbal('focal')">Focal Loss</button>
  </div>
  <div class="info" id="iImbal">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Normalisation: Fit on Train Only</h3>
  <canvas id="cvNorm" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="nBtn_wrong" onclick="selNorm('wrong')">Wrong (leak)</button>
    <button class="s" id="nBtn_right" onclick="selNorm('right')">Correct</button>
  </div>
  <div class="info" id="iNorm">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Common Pipeline Bugs &mdash; Symptoms and Fixes</h3>
  <table class="tbl">
    <thead><tr><th>Bug</th><th>Symptom</th><th>Fix</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">Augment at test time</span></td><td>Val loss stochastic, higher than expected</td><td>Use TWO separate transforms: one with augment (train), one without (val/test)</td></tr>
      <tr><td><span class="hl5">No normalisation at test</span></td><td>Works in dev, fails in deployment</td><td>Apply identical normalisation transform. Save stats alongside model weights.</td></tr>
      <tr><td><span class="hl5">Label index mismatch</span></td><td>Loss decreases but accuracy \u2248 chance</td><td>Verify labels are 0-indexed. Check class ordering matches between splits.</td></tr>
      <tr><td><span class="hl5">num_workers=0</span></td><td>GPU util 30&ndash;50%, slow training</td><td>Increase num_workers. Add pin_memory=True. Profile DataLoader alone.</td></tr>
      <tr><td><span class="hl5">Preprocessing leakage</span></td><td>Val excellent, test significantly worse</td><td>Split FIRST. Fit scaler/normaliser on training split ONLY.</td></tr>
      <tr><td><span class="hl5">Workers share random seed</span></td><td>Augmentation looks repetitive / ordered</td><td>Set unique seed per worker in worker_init_fn.</td></tr>
      <tr><td><span class="hl5">Missing sampler.set_epoch()</span></td><td>Distributed training: same order every epoch</td><td>Call sampler.set_epoch(epoch) at the start of EVERY epoch.</td></tr>
      <tr><td><span class="hl5">Dtype mismatch</span></td><td>RuntimeError: expected Float but got Double</td><td>Add .float() in __getitem__ or collate. Use model.float().</td></tr>
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
function arr(ctx,x1,y1,x2,y2,col,lw){var angle=Math.atan2(y2-y1,x2-x1);ctx.strokeStyle=col;ctx.lineWidth=lw||2;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-9*Math.cos(angle-0.4),y2-9*Math.sin(angle-0.4));ctx.lineTo(x2-9*Math.cos(angle+0.4),y2-9*Math.sin(angle+0.4));ctx.closePath();ctx.fill();}

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderPipe();renderGPU();renderCPUGPU();}if(i===1){renderBatch();renderMinima();renderPad();}if(i===2){renderShuffle();renderBN();renderEpoch();}if(i===3){renderAug();renderAS();}if(i===4){renderPrefetch();renderPin();}if(i===5){renderImbal();renderNorm();}}

// ============================================================
// TAB 0: PIPELINE FLOW
// ============================================================
function renderPipe(){
  var ctx=clr('cvPipe',860,200),W=860,H=200;
  var stages=[
    {label:'Storage\n(disk/S3)',col:'#52525b',sub:'JPEG/HDF5\nWebDataset'},
    {label:'Load &\nDecode',col:'#4ecdc4',sub:'JPEG decode\nfile read'},
    {label:'Pre-\nprocess',col:'#4ecdc4',sub:'Resize\nNormalise'},
    {label:'Batch\nCollate',col:'#4ecdc4',sub:'Stack tensors\nPad sequences'},
    {label:'Augment',col:'#fbbf24',sub:'Flip, Crop\nColorJitter'},
    {label:'H2D\nTransfer',col:'#c084fc',sub:'CPU\u2192GPU\npin_memory'},
    {label:'GPU\nMemory',col:'#c084fc',sub:'CUDA tensors\nprefetched'},
    {label:'Model\nForward',col:'#ff6b35',sub:'Forward+Back\nOptimiser step'},
  ];
  var bw=82,gap=10,startX=10,barH=80,barY=50;
  stages.forEach(function(st,i){
    var x=startX+i*(bw+gap);
    bx(ctx,x,barY,bw,barH,6,st.col+'18',st.col+'80',1.5);
    st.label.split('\n').forEach(function(l,li){tx(ctx,l,x+bw/2,barY+18+li*14,st.col,9,'800');});
    st.sub.split('\n').forEach(function(l,li){tx(ctx,l,x+bw/2,barY+50+li*12,'#52525b',7,'400');});
    if(i<stages.length-1){
      var arx=x+bw+2; var col='#2d2d40';
      arr(ctx,arx,barY+barH/2,arx+gap+2,barY+barH/2,col,1.5);
    }
  });
  // Phase labels
  bx(ctx,10,140,290,28,4,'#4ecdc408','#4ecdc430',1);
  tx(ctx,'CPU Workers (DataLoader)',155,154,'#4ecdc470',8,'700');
  bx(ctx,490,140,270,28,4,'#c084fc08','#c084fc30',1);
  tx(ctx,'CPU\u2192GPU Transfer',625,154,'#c084fc70',8,'700');
  bx(ctx,770,140,90,28,4,'#ff6b3508','#ff6b3530',1);
  tx(ctx,'GPU',815,154,'#ff6b3570',8,'700');
  tx(ctx,'GPU sits idle if any upstream stage is slow \u2014 keep GPU util \u226595%',W/2,185,'#3f3f46',8,'400');
  document.getElementById('iPipe').innerHTML=
    '<span class="hl">Bottleneck detection:</span> if GPU util &lt;80%, the pipeline is starving the GPU.<br>'+
    '1. Time the DataLoader alone (no model). 2. Time one GPU step (no loading).<br>'+
    'If DataLoader &gt; GPU step: pipeline is the bottleneck \u2192 add workers, pin_memory, faster storage.<br>'+
    'If GPU step &gt; DataLoader: <span class="hl4">GPU is the bottleneck \u2192 correct! Pipeline is fine.</span>';
}

function renderGPU(){
  var gpuMs=parseInt(document.getElementById('slGPU').value);
  var dlMs=parseInt(document.getElementById('slDL').value);
  document.getElementById('vGPU').textContent=gpuMs;
  document.getElementById('vDL').textContent=dlMs;
  var ctx=clr('cvGPU',420,230),W=420,H=230;
  // Timeline: show GPU step vs DataLoader step
  var stepMs=Math.max(gpuMs,dlMs); // wall-clock step is max of the two
  var gpuUtil=gpuMs/stepMs;
  var scale=W-80;
  // GPU bar
  var gpuW=scale*gpuMs/stepMs;
  bx(ctx,40,50,gpuW,36,4,'#ff6b3540','#ff6b35',2);
  tx(ctx,'GPU active ('+gpuMs+'ms)',40+gpuW/2,68,'#ff6b35',9,'700');
  if(gpuMs<stepMs){bx(ctx,40+gpuW,50,scale-gpuW,36,4,'#1e1e2e20','#1e1e2820',1);tx(ctx,'idle ('+( stepMs-gpuMs)+'ms)',40+gpuW+(scale-gpuW)/2,68,'#2d2d40',9,'400');}
  // DL bar
  var dlW=scale*dlMs/stepMs;
  bx(ctx,40,100,dlW,36,4,'#4ecdc440','#4ecdc4',2);
  tx(ctx,'DataLoader ('+dlMs+'ms)',40+dlW/2,118,'#4ecdc4',9,'700');
  if(dlMs<stepMs){bx(ctx,40+dlW,100,scale-dlW,36,4,'#1e1e2e20','#1e1e2820',1);tx(ctx,'idle',40+dlW+(scale-dlW)/2,118,'#2d2d40',8,'400');}
  tx(ctx,'GPU',30,68,'#ff6b35',8,'700','right');
  tx(ctx,'DL',30,118,'#4ecdc4',8,'700','right');
  // Wall-clock step
  bx(ctx,40,148,scale,20,4,null,'#1e1e2e',1);
  bx(ctx,40,148,scale,20,4,'#ffffff08',null);
  tx(ctx,'Wall-clock step = '+stepMs+'ms',40+scale/2,158,'#52525b',8,'400');
  // GPU util gauge
  var gaugeCol=gpuUtil>0.9?'#4ade80':gpuUtil>0.7?'#fbbf24':'#ef4444';
  tx(ctx,'GPU Utilisation',W/2,185,gaugeCol,10,'800');
  bx(ctx,60,195,W-120,22,4,'#1e1e2e',null,0);
  bx(ctx,60,195,(W-120)*gpuUtil,22,4,gaugeCol+'40',gaugeCol,1.5);
  tx(ctx,Math.round(gpuUtil*100)+'%',W/2,206,gaugeCol,11,'800');
  var bottleneck=dlMs>gpuMs?'<span class="hl5">Pipeline bottleneck!</span> DataLoader is slower than GPU. Add more workers, pin_memory, faster storage.':gpuMs>dlMs*1.5?'<span class="hl4">GPU bottleneck. Pipeline is not limiting training.</span> Good!':'<span class="hl4">Well balanced.</span> GPU is busy nearly 100% of the time.';
  document.getElementById('iGPU').innerHTML='GPU step: <span class="hl">'+gpuMs+'ms</span>  DataLoader: <span class="hl2">'+dlMs+'ms</span>  Wall-clock: <span class="hl3">'+stepMs+'ms</span><br>'+bottleneck;
}

function renderCPUGPU(){
  var ctx=clr('cvCPUGPU',420,230),W=420,H=230;
  var cpuTasks=['Read files from disk','JPEG / image decode','Data augmentation','Normalisation / cast','Batch collation'];
  var gpuTasks=['Forward pass','Loss computation','Backward pass (grads)','Optimiser step (update)'];
  var half=W/2-10;
  bx(ctx,10,30,half,H-40,6,'#4ecdc410','#4ecdc440',1.5);
  tx(ctx,'CPU (DataLoader Workers)',10+half/2,48,'#4ecdc4',9,'800');
  cpuTasks.forEach(function(t,i){tx(ctx,'\u2022 '+t,20,68+i*24,'#94a3b8',8,'400','left');});
  bx(ctx,W/2+10,30,half,H-40,6,'#ff6b3510','#ff6b3540',1.5);
  tx(ctx,'GPU',W/2+10+half/2,48,'#ff6b35',9,'800');
  gpuTasks.forEach(function(t,i){tx(ctx,'\u2022 '+t,W/2+20,68+i*24,'#94a3b8',8,'400','left');});
  tx(ctx,'Goal: CPU workers are always faster than GPU. GPU waits for no one.',W/2,H-10,'#52525b',7,'400');
}

// ============================================================
// TAB 1: BATCHING
// ============================================================
var BATCH_LABELS=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096];
function renderBatch(){
  var idx=parseInt(document.getElementById('slBatch').value)-1;
  var batchSizes=[8,16,32,64,128,256,512,1024,2048,4096];
  var B=batchSizes[idx]||256;
  document.getElementById('vBatch').textContent=B;
  var ctx=clr('cvBatch',420,260),W=420,H=260,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  // Show gradient variance, throughput, generalisation as slider moves
  var xs=[8,16,32,64,128,256,512,1024,2048,4096];
  var logB=Math.log2(B),logBmax=Math.log2(4096),logBmin=Math.log2(8);
  var t=(logB-logBmin)/(logBmax-logBmin);
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);
  ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  // Three curves vs log batch size
  var N=50;
  // GPU throughput (rises then saturates)
  var tPts=[],vPts=[],gPts=[];
  for(var i=0;i<=N;i++){
    var ti=i/N;
    var cx=PAD+ti*pW;
    var tp=Math.min(1,Math.pow(ti,0.35)*1.3)*pH; // throughput: rises fast
    var vari=(1-Math.pow(ti,0.7))*pH*0.9; // gradient variance: decreases
    var gen=(1-Math.pow(Math.max(0,ti-0.4)*1.4,1.5))*pH*0.8; // generalisation: degrades past midpoint
    tPts.push([cx,H-PAD-tp]);
    vPts.push([cx,H-PAD-vari]);
    gPts.push([cx,H-PAD-Math.max(0,gen)]);
  }
  function drawCurve(pts,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||2;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}
  drawCurve(tPts,'#4ade80',2);
  drawCurve(vPts,'#fbbf24',2);
  drawCurve(gPts,'#ff6b35',2);
  // Current position line
  var cx0=PAD+t*pW;
  ln(ctx,cx0,15,cx0,H-PAD,'#3f3f46',1.5,[4,4]);
  // Dots
  [[tPts,'#4ade80'],[vPts,'#fbbf24'],[gPts,'#ff6b35']].forEach(function(pair){
    var idx2=Math.round(t*N); if(idx2>=pair[0].length)idx2=pair[0].length-1;
    ctx.fillStyle=pair[1]; ctx.beginPath(); ctx.arc(pair[0][idx2][0],pair[0][idx2][1],5,0,Math.PI*2); ctx.fill();
  });
  // Labels
  tx(ctx,'\u2014 GPU Throughput',W-14,28,'#4ade80',8,'700','right');
  tx(ctx,'\u2014 Grad Variance (lower=better)',W-14,42,'#fbbf24',8,'700','right');
  tx(ctx,'\u2014 Generalisation',W-14,56,'#ff6b35',8,'700','right');
  tx(ctx,'Batch size B (log scale)',W/2,H-5,'#52525b',8,'400');
  tx(ctx,'8',PAD,H-PAD+12,'#3f3f46',7,'400');
  tx(ctx,'256',PAD+pW*(Math.log2(256)-3)/(Math.log2(4096)-3),H-PAD+12,'#3f3f46',7,'400');
  tx(ctx,'4096',W-20,H-PAD+12,'#3f3f46',7,'400');
  tx(ctx,'B='+B,cx0,H-PAD+22,'#fbbf24',8,'700');
  var info=B<=32?'<span class="hl5">Very small batch:</span> high gradient noise, slow throughput, good generalisation. Use for quick experiments or when GPU memory is very limited.':B<=256?'<span class="hl4">Small-medium batch (ideal range):</span> good generalisation, decent throughput. Standard for most tasks.':B<=1024?'<span class="hl3">Large batch:</span> apply linear LR scaling. Warmup required. Slight generalisation gap.':'<span class="hl5">Very large batch:</span> apply linear LR scaling + warmup. Monitor for generalisation gap. May need LARS optimiser.';
  document.getElementById('iBatch').innerHTML='B=<span class="hl3">'+B+'</span><br>'+info+'<br><span class="hl2">Linear scaling rule:</span> LR = baseline_LR \u00D7 (B / baseline_B)';
}

var curMinima='sharp';
function selMinima(k){curMinima=k;['sharp','flat'].forEach(function(b){document.getElementById('mBtn_'+b).classList.remove('active');});document.getElementById('mBtn_'+k).classList.add('active');renderMinima();}
function renderMinima(){
  var ctx=clr('cvMinima',420,260),W=420,H=260,PAD=40,pW=W-PAD-20,pH=H-PAD-30;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);
  tx(ctx,'Weight space','#52525b',null);tx(ctx,'Weight space (w)',W/2,H-5,'#52525b',8,'400');
  tx(ctx,'Loss',PAD+18,22,'#52525b',8,'400');
  var N=120;
  function drawLoss(fn,col,lw){var pts=[];for(var i=0;i<=N;i++){var x=i/N,cx=PAD+x*pW,cy=H-PAD-Math.max(0,fn(x))*pH;pts.push([cx,cy]);}ctx.strokeStyle=col;ctx.lineWidth=lw||2.5;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}
  if(curMinima==='sharp'){
    // Sharp minimum: narrow valley
    var cen=0.4;
    drawLoss(function(x){return 0.05+Math.pow(Math.abs(x-cen)*8,2)*0.5+Math.random()*0.005;},'#ff6b35');
    // Train and test points
    var trainX=PAD+cen*pW, trainY=H-PAD-0.06*pH;
    var testX=PAD+(cen+0.04)*pW, testY=H-PAD-(0.06+Math.pow(0.04*8,2)*0.5)*pH;
    ctx.fillStyle='#4ecdc4'; ctx.beginPath(); ctx.arc(trainX,trainY,7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#ff6b3560'; ctx.beginPath(); ctx.arc(testX,testY,7,0,Math.PI*2); ctx.fill();
    tx(ctx,'Train \u25CF',trainX-18,trainY-14,'#4ecdc4',8,'700');
    tx(ctx,'Test \u25CF (drifted OUT)',testX+40,testY-4,'#ff6b35',8,'700');
    tx(ctx,'Sharp minimum \u2014 large batch',W/2,22,'#ff6b35',10,'800');
    tx(ctx,'Test distribution shift \u2192 exits the narrow valley',W/2,H-PAD+22,'#52525b',8,'400');
    document.getElementById('iMinima').innerHTML='<span class="hl5">Sharp minimum:</span> narrow basin with high curvature. A small shift in the test distribution exits the minimum \u2192 poor generalisation.<br>Large-batch training tends to converge to sharp minima (Keskar et al., 2017). Can lose 1\u20132% accuracy vs small batch.';
  } else {
    var cen2=0.55;
    drawLoss(function(x){return 0.05+Math.pow(Math.abs(x-cen2)*2.5,2.5)*0.4;},'#4ade80');
    var trainX2=PAD+cen2*pW, trainY2=H-PAD-0.055*pH;
    var testX2=PAD+(cen2+0.07)*pW, testY2=H-PAD-(0.055+Math.pow(0.07*2.5,2.5)*0.4)*pH;
    ctx.fillStyle='#4ecdc4'; ctx.beginPath(); ctx.arc(trainX2,trainY2,7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#4ade8080'; ctx.beginPath(); ctx.arc(testX2,testY2,7,0,Math.PI*2); ctx.fill();
    tx(ctx,'Train \u25CF',trainX2-18,trainY2-14,'#4ecdc4',8,'700');
    tx(ctx,'Test \u25CF (still in basin)',testX2+48,testY2+4,'#4ade80',8,'700');
    tx(ctx,'Flat minimum \u2014 small batch',W/2,22,'#4ade80',10,'800');
    tx(ctx,'Test shift stays within the wide valley \u2192 good generalisation',W/2,H-PAD+22,'#52525b',8,'400');
    document.getElementById('iMinima').innerHTML='<span class="hl4">Flat minimum:</span> wide basin with low curvature. A shift in the test distribution stays within the basin \u2192 good generalisation.<br>Small-batch training finds flatter minima. The gradient noise acts as an escape mechanism from sharp minima.';
  }
}

function renderPad(){
  var ctx=clr('cvPad',860,150),W=860,H=150;
  // Show 3 sequences with padding vs packing
  var seqLens=[3,7,5];
  var maxLen=7;
  var half=W/2-20;
  // PADDING
  tx(ctx,'PADDING (naive \u2014 wasted compute)',half/2,18,'#ff6b35',9,'800');
  var bh=22,by=35,bw=half/(maxLen+1)-4;
  seqLens.forEach(function(L,i){
    for(var j=0;j<maxLen;j++){
      var isPad=j>=L;
      var x=10+j*(bw+3);
      var by2=by+i*(bh+6);
      bx(ctx,x,by2,bw,bh,3,isPad?'#1e1e2e':'#4ecdc430',isPad?'#2d2d40':'#4ecdc4',1.5);
      tx(ctx,isPad?'PAD':'tok',x+bw/2,by2+bh/2,isPad?'#2d2d40':'#4ecdc4',7,'700');
    }
  });
  var padPct=Math.round((1-seqLens.reduce(function(a,b){return a+b;},0)/(seqLens.length*maxLen))*100);
  tx(ctx,padPct+'% wasted (PAD tokens)',10+half/2,H-12,'#ef4444',8,'700');

  // PACKING
  var ox=W/2+10;
  tx(ctx,'PACKING (\u2248100% compute efficiency)',ox+(W-ox)/2,18,'#4ade80',9,'800');
  var totalToks=seqLens.reduce(function(a,b){return a+b;},0)+seqLens.length-1;
  var pw2=(W-ox-20)/totalToks;
  var posX=ox+10;
  var colors=['#ff6b35','#4ecdc4','#c084fc'];
  seqLens.forEach(function(L,i){
    for(var j=0;j<L;j++){
      bx(ctx,posX,35,pw2-2,66,3,colors[i]+'30',colors[i],1.5);
      tx(ctx,'T',posX+pw2/2-1,68,'#94a3b8',7,'400');
      posX+=pw2;
    }
    if(i<seqLens.length-1){bx(ctx,posX,35,pw2-2,66,3,'#fbbf2420','#fbbf24',1.5);tx(ctx,'|',posX+pw2/2-1,68,'#fbbf24',7,'700');posX+=pw2;}
  });
  tx(ctx,'0% wasted \u2014 all tokens are real',ox+(W-ox)/2,H-12,'#4ade80',8,'700');
}

// ============================================================
// TAB 2: SHUFFLING
// ============================================================
var curShuffle='no';
function selShuffle(k){curShuffle=k;['no','yes'].forEach(function(b){document.getElementById('shBtn_'+b).classList.remove('active');});document.getElementById('shBtn_'+k).classList.add('active');renderShuffle();}
function renderShuffle(){
  var ctx=clr('cvShuffle',420,270),W=420,H=270;
  var classes=['A','B','C'];
  var colors={'A':'#4ecdc4','B':'#ff6b35','C':'#c084fc'};
  var N=24,batchSize=8,nBatches=3;
  var bw=12,bh=28,gap=2;
  var dataset=[];
  for(var i=0;i<8;i++)dataset.push('A');
  for(var i=0;i<8;i++)dataset.push('B');
  for(var i=0;i<8;i++)dataset.push('C');
  var order=curShuffle==='no'?dataset.slice():dataset.slice().sort(function(){return 0.5-Math.random();});
  for(var b=0;b<nBatches;b++){
    var batch=order.slice(b*batchSize,(b+1)*batchSize);
    var by=30+b*82;
    tx(ctx,'Batch '+(b+1),30,by,'#52525b',8,'700','right');
    for(var j=0;j<batch.length;j++){
      var col=colors[batch[j]];
      bx(ctx,40+j*(bw+gap),by-14,bw,bh,3,col+'30',col,1.5);
      tx(ctx,batch[j],40+j*(bw+gap)+bw/2,by,col,8,'800');
    }
    // Class distribution
    var counts={'A':0,'B':0,'C':0};
    batch.forEach(function(c){counts[c]++;});
    classes.forEach(function(c,ci){
      var pct=counts[c]/batchSize;
      var bx2W=50,bx2H=14;
      var bxX=W-130+ci*52;
      bx(ctx,bxX,by-10,bx2W,bx2H,2,colors[c]+'20',colors[c]+'60',1);
      bx(ctx,bxX,by-10,bx2W*pct,bx2H,2,colors[c]+'50',null);
      tx(ctx,c+':'+Math.round(pct*100)+'%',bxX+bx2W/2,by-3,colors[c],6,'700');
    });
  }
  tx(ctx,curShuffle==='no'?'Unshuffled Dataset (class-ordered)':'Shuffled Dataset (random order)',W/2,H-20,'#52525b',8,'400');
  tx(ctx,curShuffle==='no'?'WARNING':'OK',W/2,H-5,curShuffle==='no'?'#ef4444':'#4ade80',9,'800');
  document.getElementById('iShuffle').innerHTML=curShuffle==='no'?
    '<span class="hl5">Unshuffled:</span> all early batches contain class A only. Gradient is biased toward A. BatchNorm statistics computed on non-representative batches. Model trains on a non-stationary objective.':
    '<span class="hl4">Shuffled:</span> each batch contains a random mixture of classes. Gradient accurately estimates the true data distribution. BatchNorm sees representative statistics.';
}

function renderBN(){
  var ctx=clr('cvBN',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);
  ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Batch index (training)','#52525b',null);tx(ctx,'Batch index',W/2,H-5,'#52525b',8,'400');
  tx(ctx,'BN mean',PAD+20,22,'#52525b',8,'400');
  var trueM=0.5;
  ln(ctx,PAD,H-PAD-trueM*pH,W-10,H-PAD-trueM*pH,'#4ade8030',1,[4,4]);
  tx(ctx,'True dataset mean',W-14,H-PAD-trueM*pH-8,'#4ade8040',7,'700','right');
  var N=30;
  // Unshuffled: mean oscillates between 0 and 1 based on class
  var unshPts=[],shPts=[];
  for(var i=0;i<N;i++){
    var cx=PAD+i/(N-1)*pW;
    var batchClass=Math.floor(i/(N/3)); // 0=A, 1=B, 2=C
    var unshM=[0.1,0.5,0.9][batchClass]; // class means diverge
    var shM=0.45+Math.sin(i*0.7)*0.08; // shuffled converges to true mean
    unshPts.push([cx,H-PAD-unshM*pH]);
    shPts.push([cx,H-PAD-shM*pH]);
  }
  ctx.strokeStyle='#ff6b35'; ctx.lineWidth=2; ctx.beginPath(); unshPts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);}); ctx.stroke();
  ctx.strokeStyle='#4ade80'; ctx.lineWidth=2; ctx.beginPath(); shPts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);}); ctx.stroke();
  tx(ctx,'Unshuffled (oscillates wildly)',W-14,unshPts[N-1][1],'#ff6b35',8,'700','right');
  tx(ctx,'Shuffled (converges to true)',W-14,shPts[N-1][1]-12,'#4ade80',8,'700','right');
  // Sudden class switch annotation
  var switchX=PAD+pW/3;
  ln(ctx,switchX,15,switchX,H-PAD,'#fbbf2450',1,[3,3]);
  tx(ctx,'class switch',switchX,28,'#fbbf24',7,'700');
  document.getElementById('iBN').innerHTML=
    '<span class="hl5">Unshuffled + BatchNorm:</span> BN mean oscillates between 0.1, 0.5, 0.9 as class changes. Running mean never converges. At inference, running stats are wrong for most inputs.<br>'+
    '<span class="hl4">Shuffled + BatchNorm:</span> each batch is representative. BN running mean converges smoothly to the true dataset mean. Inference statistics are correct.';
}

function renderEpoch(){
  var ctx=clr('cvEpoch',860,140),W=860,H=140;
  var N=30, batchSize=4;
  var half=W/2-20;
  // Epoch-level: each example appears exactly once
  tx(ctx,'Epoch-Level Shuffling (DataLoader default)',half/2,18,'#4ecdc4',9,'800');
  for(var i=0;i<N;i++){
    var col=['#4ecdc4','#ff6b35','#c084fc','#fbbf24'][i%4];
    bx(ctx,10+i*(half-10)/N,30,half/N-3,40,3,col+'30',col,1.5);
    if(i<4) tx(ctx,i,10+i*(half-10)/N+(half/N-3)/2,50,'#94a3b8',7,'400');
  }
  tx(ctx,'Each example appears exactly once per epoch',10+half/2,82,'#4ade80',8,'700');
  tx(ctx,'Reshuffle \u27F3',10+half/2,96,'#3f3f46',7,'400');
  // Step-level: sampling with replacement
  var ox=W/2+10;
  tx(ctx,'Step-Level Shuffling (sampling with replacement)',ox+(W-ox)/2,18,'#fbbf24',9,'800');
  var seen=new Set();
  var rngS=99;function rng3(){rngS=(rngS*1664525+1013904223)&0xffffffff;return(rngS>>>0)/4294967296;}
  for(var i=0;i<N;i++){
    var idx=Math.floor(rng3()*N);
    seen.add(idx);
    var col=['#4ecdc4','#ff6b35','#c084fc','#fbbf24'][idx%4];
    bx(ctx,ox+10+i*(W-ox-20)/N,30,(W-ox-20)/N-3,40,3,col+'30',col,1.5);
    if(i<4) tx(ctx,idx,ox+10+i*(W-ox-20)/N+(W-ox-20)/N/2,50,'#94a3b8',7,'400');
  }
  var missed=N-seen.size;
  tx(ctx,missed+' / '+N+' examples never seen ('+Math.round(missed/N*100)+'% missed per epoch)',ox+10+(W-ox)/2,82,'#ef4444',8,'700');
  tx(ctx,'Some duplicated \u2014 needs more epochs for full coverage',ox+10+(W-ox)/2,96,'#3f3f46',7,'400');
}

// ============================================================
// TAB 3: AUGMENTATION
// ============================================================
var curAug='train';
function selAug(k){curAug=k;['train','val'].forEach(function(b){document.getElementById('augBtn_'+b).classList.remove('active');});document.getElementById('augBtn_'+k).classList.add('active');renderAug();}
function renderAug(){
  var ctx=clr('cvAug',420,260),W=420,H=260;
  var isTrain=curAug==='train';
  var steps=isTrain?[
    {label:'RandomResizedCrop(224)',col:'#fbbf24',rand:true},
    {label:'RandomHorizontalFlip(p=0.5)',col:'#fbbf24',rand:true},
    {label:'ColorJitter(b,c,s,h)',col:'#fbbf24',rand:true},
    {label:'ToTensor()',col:'#4ecdc4',rand:false},
    {label:'Normalize(mean,std)',col:'#4ecdc4',rand:false},
  ]:[
    {label:'Resize(256)',col:'#4ecdc4',rand:false},
    {label:'CenterCrop(224)',col:'#4ecdc4',rand:false},
    {label:'ToTensor()',col:'#4ecdc4',rand:false},
    {label:'Normalize(mean,std)',col:'#4ecdc4',rand:false},
  ];
  var bh=34, startY=30;
  steps.forEach(function(st,i){
    var y=startY+i*(bh+8);
    bx(ctx,20,y,W-40,bh,6,st.col+'15',st.col+(st.rand?'':'60'),1.5);
    tx(ctx,st.label,W/2,y+bh/2,st.col,10,'700');
    if(st.rand){
      bx(ctx,W-95,y+5,70,24,4,'#fbbf2420','#fbbf2460',1);
      tx(ctx,'RANDOM',W-60,y+17,'#fbbf24',7,'800');
    } else {
      bx(ctx,W-110,y+5,86,24,4,'#4ecdc420','#4ecdc460',1);
      tx(ctx,'DETERMINISTIC',W-67,y+17,'#4ecdc4',7,'800');
    }
  });
  tx(ctx,isTrain?'Training Transform Pipeline':'Validation/Test Transform Pipeline',W/2,H-15,isTrain?'#fbbf24':'#4ecdc4',9,'800');
  document.getElementById('iAug').innerHTML=isTrain?
    '<span class="hl3">Training:</span> random transforms enforce invariances. Model sees diverse views of each image.<br>'+
    '<span class="hl5">Critical:</span> augmentation is ONLY applied during training. Never at val/test time.':
    '<span class="hl2">Validation/Test:</span> only deterministic preprocessing. No randomness.<br>'+
    'Normalisation stats (mean, std) are IDENTICAL to training. Computed on training set only.';
}

var curAS='flip';
var AS_DATA={
  flip:{label:'Horizontal Flip (p=0.5)',col:'#4ecdc4',gain:'+0.5\u20131.5%',complexity:'None',domain:'Most object recognition (not text, digits)',info:'Simplest augmentation. 50% chance of mirror reflection. Teaches orientation invariance. Free regularisation on virtually every image task.'},
  crop:{label:'RandomResizedCrop',col:'#4ade80',gain:'+1\u20132%',complexity:'Low',domain:'Object recognition, detection',info:'Random crop of 8\u2013100% of image area, then resize. Forces recognition at varying scales and positions. Standard for ImageNet training. Reduces reliance on absolute position.'},
  color:{label:'ColorJitter(b=0.4,c=0.4,s=0.4,h=0.1)',col:'#fbbf24',gain:'+0.5\u20131%',complexity:'Low',domain:'Natural images, outdoor scenes',info:'Random brightness, contrast, saturation, hue changes. Teaches colour-independent features. Critical for models deployed in variable lighting conditions.'},
  mix:{label:'Mixup (\u03B1=0.2)',col:'#c084fc',gain:'+0.5\u20131.5%',complexity:'Medium',domain:'Classification, detection',info:'x=\u03BBx\u2081+(1-\u03BB)x\u2082, y=\u03BBy\u2081+(1-\u03BB)y\u2082. Forces smooth linear interpolation in feature space. Reduces overconfidence. One of the most effective regularisers for large-scale classification.'},
  rand:{label:'RandAugment (N=2, M=9)',col:'#ff6b35',gain:'+1\u20132.5%',complexity:'Medium',domain:'ViT, ResNet, any classifier',info:'Randomly sample N=2 augmentation ops from a library of K ops. Apply at magnitude M. Eliminates manual augmentation policy design. Standard for modern vision training (ViT, EfficientNet, Swin).'},
};
function selAS(k){curAS=k;['flip','crop','color','mix','rand'].forEach(function(b){document.getElementById('asBtn_'+b).classList.remove('active');});document.getElementById('asBtn_'+k).classList.add('active');renderAS();}
function renderAS(){
  var d=AS_DATA[curAS];
  var ctx=clr('cvAugStrength',420,260),W=420,H=260;
  // Draw augmentation strength bar
  var strengths={flip:1,crop:2,color:3,mix:4,rand:5};
  var s=strengths[curAS];
  var bw=(W-60)/5;
  for(var i=0;i<5;i++){
    var filled=i<s;
    var col=filled?d.col:'#1e1e2e';
    bx(ctx,30+i*(bw+4),30,bw,36,4,col+(filled?'30':'20'),col+(filled?'':'40'),filled?2:1);
    tx(ctx,i+1,30+i*(bw+4)+bw/2,48,filled?d.col:'#2d2d40',10,'800');
  }
  tx(ctx,'Augmentation Strength: '+s+'/5',W/2,22,d.col,9,'800');
  // Info boxes
  var items=[
    ['Method',d.label,d.col],
    ['Accuracy gain',d.gain,'#4ade80'],
    ['Complexity',d.complexity,'#fbbf24'],
    ['Best for',d.domain,'#94a3b8'],
  ];
  items.forEach(function(it,i){
    var y=90+i*40;
    bx(ctx,20,y,W-40,32,4,'#0d0d18','#1e1e2e',1);
    tx(ctx,it[0],30,y+16,it[2],8,'800','left');
    tx(ctx,it[1],W-30,y+16,'#e4e4e7',8,'400','right');
  });
  document.getElementById('iAS').innerHTML=d.info;
}

// ============================================================
// TAB 4: DATALOADER
// ============================================================
function renderPrefetch(){
  var nw=parseInt(document.getElementById('slWorkers').value);
  document.getElementById('vWorkers').textContent=nw;
  var ctx=clr('cvPrefetch',420,250),W=420,H=250;
  var nBatches=8, cellW=(W-60)/nBatches, cellH=28;
  // GPU row
  tx(ctx,'GPU',30,50,'#ff6b35',8,'700','right');
  for(var i=0;i<nBatches;i++){
    var x=40+i*cellW;
    bx(ctx,x+1,36,cellW-2,cellH,3,'#ff6b3530','#ff6b35',1.5);
    tx(ctx,'B'+i,x+cellW/2,50,'#ff6b35',9,'700');
  }
  // Workers rows
  var workerH=cellH+6;
  var nRows=Math.max(1,nw);
  for(var w=0;w<Math.min(nRows,4);w++){
    var wy=36+cellH+16+w*workerH;
    tx(ctx,'W'+w,30,wy+cellH/2,'#4ecdc4',7,'700','right');
    for(var i=0;i<nBatches;i++){
      var batIdx=i+1; // workers always ahead by 1+
      if(batIdx>=nBatches) continue;
      var off=nw>0?w/nw:0;
      var wx2=40+(i+off)*cellW;
      if(wx2+cellW-2>W-10) continue;
      bx(ctx,wx2+1,wy,cellW-2,cellH,3,'#4ecdc420','#4ecdc4',1);
      tx(ctx,'B'+batIdx,wx2+cellW/2,wy+cellH/2,'#4ecdc480',7,'400');
    }
  }
  if(nw===0){
    bx(ctx,40,36+cellH+16,W-50,30,4,'#ef444420','#ef444450',1.5);
    tx(ctx,'num_workers=0: loading happens in main process (GPU WAITS)',W/2,36+cellH+31,'#ef4444',8,'700');
  }
  var gpuY=36+cellH+16+(Math.min(nRows,4))*workerH+20;
  var util=nw===0?0.3:Math.min(0.97,0.6+nw*0.08);
  var gaugeCol=util>0.9?'#4ade80':util>0.7?'#fbbf24':'#ef4444';
  bx(ctx,40,gpuY,W-80,18,4,'#1e1e2e',null);
  bx(ctx,40,gpuY,(W-80)*util,18,4,gaugeCol+'40',gaugeCol,1.5);
  tx(ctx,'GPU util \u2248'+Math.round(util*100)+'%',W/2,gpuY+9,gaugeCol,9,'800');
  document.getElementById('iPrefetch').innerHTML=
    'num_workers=<span class="hl3">'+nw+'</span><br>'+
    (nw===0?'<span class="hl5">0 workers: synchronous loading in main process. GPU waits for every batch. Only for debugging.</span>':
     nw<4?'<span class="hl3">Few workers: may not keep GPU busy. Increase until GPU util saturates.</span>':
     '<span class="hl4">'+nw+' workers: pipeline runs ahead of GPU. Workers prepare batches while GPU processes.</span>')+
    '<br>Rule of thumb: num_workers = CPU cores / num_GPUs. Start at 4, increase to GPU saturation.';
}

var curPin='off';
function selPin(k){curPin=k;['off','on'].forEach(function(b){document.getElementById('pinBtn_'+b).classList.remove('active');});document.getElementById('pinBtn_'+k).classList.add('active');renderPin();}
function renderPin(){
  var ctx=clr('cvPin',420,250),W=420,H=250;
  var isOn=curPin==='on';
  // Show memory path
  var boxes=isOn?[
    {label:'CPU RAM\n(Pinned)',col:'#4ecdc4',x:30},
    {label:'GPU VRAM',col:'#ff6b35',x:200},
  ]:[
    {label:'CPU RAM\n(Pageable)',col:'#52525b',x:30},
    {label:'Pinned\nBuffer',col:'#fbbf24',x:160},
    {label:'GPU VRAM',col:'#ff6b35',x:290},
  ];
  boxes.forEach(function(b,i){
    bx(ctx,b.x,60,110,80,8,b.col+'20',b.col,2);
    b.label.split('\n').forEach(function(l,li){tx(ctx,l,b.x+55,84+li*16,b.col,10,'800');});
    if(i<boxes.length-1){
      var nx=boxes[i+1].x;
      arr(ctx,b.x+112,100,nx-4,100,b.col,2);
      var ms=isOn?'~20ms':'~35ms';
      tx(ctx,ms,(b.x+112+nx)/2,88,b.col,8,'700');
    }
  });
  tx(ctx,isOn?'Direct DMA (one copy)':'Two copies: pageable \u2192 pinned \u2192 GPU',W/2,160,isOn?'#4ade80':'#fbbf24',9,'700');
  var speedup=isOn?'20\u201330% faster H2D transfer':'Baseline (slower)';
  bx(ctx,60,175,W-120,30,6,(isOn?'#4ade80':'#fbbf24')+'15',(isOn?'#4ade80':'#fbbf24')+'60',1.5);
  tx(ctx,speedup,W/2,190,isOn?'#4ade80':'#fbbf24',10,'800');
  tx(ctx,isOn?'pin_memory=True (recommended)':'pin_memory=False (default)',W/2,215,'#52525b',8,'400');
  tx(ctx,'Use with: batch.to(device, non_blocking=True)',W/2,232,'#3f3f46',7,'400');
  document.getElementById('iPin').innerHTML=isOn?
    '<span class="hl4">pin_memory=True:</span> tensors allocated in page-locked (pinned) RAM. DMA transfer is direct CPU\u2192GPU without an intermediate copy.<br>20\u201330% faster H2D transfer. Use with non_blocking=True for overlapped CPU/GPU work.':
    '<span class="hl5">pin_memory=False:</span> OS must copy pageable\u2192pinned buffer before DMA to GPU. Two copies instead of one. Slower H2D transfer. Only acceptable when RAM is very limited.';
}

// ============================================================
// TAB 5: IMBALANCE & BUGS
// ============================================================
var curImbal='none';
function selImbal(k){curImbal=k;['none','weight','over','focal'].forEach(function(b){document.getElementById('imBtn_'+b).classList.remove('active');});document.getElementById('imBtn_'+k).classList.add('active');renderImbal();}
function renderImbal(){
  var ctx=clr('cvImbal',420,260),W=420,H=260;
  var classCounts={neg:950,pos:50};
  var total=1000;
  // Show class distribution bar
  bx(ctx,20,30,W-40,36,4,'#1e1e2e',null);
  var negW=(W-40)*classCounts.neg/total;
  bx(ctx,20,30,negW,36,4,'#4ecdc440','#4ecdc4',1.5);
  bx(ctx,20+negW,30,W-40-negW,36,4,'#ff6b3540','#ff6b35',1.5);
  tx(ctx,'Negative: 95% ('+classCounts.neg+')',20+negW/2,48,'#4ecdc4',8,'700');
  tx(ctx,'Positive: 5% ('+classCounts.pos+')',20+negW+(W-40-negW)/2,48,'#ff6b35',8,'700');
  // Effective sample distribution per batch (after strategy)
  var effNeg,effPos,lossNeg,lossPos;
  if(curImbal==='none'){effNeg=95;effPos=5;lossNeg=1;lossPos=1;}
  else if(curImbal==='weight'){effNeg=95;effPos=5;lossNeg=0.526;lossPos=10;}
  else if(curImbal==='over'){effNeg=50;effPos=50;lossNeg=1;lossPos=1;}
  else{effNeg=95;effPos=5;lossNeg=0.3;lossPos=1;}

  // Show effective training signal
  tx(ctx,'Effective training signal per class:',W/2,88,'#52525b',8,'700');
  var sigNeg=effNeg*lossNeg, sigPos=effPos*lossPos;
  var sigTotal=sigNeg+sigPos;
  var sigNegW=(W-40)*sigNeg/sigTotal;
  bx(ctx,20,100,sigNegW,32,4,'#4ecdc440','#4ecdc4',1.5);
  bx(ctx,20+sigNegW,100,W-40-sigNegW,32,4,'#ff6b3540','#ff6b35',1.5);
  tx(ctx,'NEG: '+Math.round(sigNeg/sigTotal*100)+'%',20+sigNegW/2,116,'#4ecdc4',8,'700');
  tx(ctx,'POS: '+Math.round(sigPos/sigTotal*100)+'%',20+sigNegW+(W-40-sigNegW)/2,116,'#ff6b35',8,'700');

  // Show what model learns
  var stratMap={none:'<span class="hl5">No strategy:</span> model predicts "negative" for everything. 95% accuracy but 0% recall on positives. Useless for the actual task.',weight:'<span class="hl4">Class weights:</span> loss weight 10\u00D7 for positives. Training signal balanced despite raw counts. Simple and effective.',over:'<span class="hl4">Oversampling:</span> positive examples drawn more often. Each batch has ~50% positives. But repeated minority examples can cause overfitting.',focal:'<span class="hl4">Focal Loss (\u03B3=2):</span> easy negatives down-weighted by (1-p)\u00B2. Model focuses learning on hard positives. Best for extreme imbalance (detection, rare events).'};
  bx(ctx,20,148,W-40,H-158,6,'#0d0d18','#1e1e2e',1);
  var lines=stratMap[curImbal].replace(/<[^>]+>/g,'').split('.');
  document.getElementById('iImbal').innerHTML=stratMap[curImbal];
  tx(ctx,curImbal==='none'?'95% Accuracy = Useless Model':'Signal Balanced',W/2,H-16,curImbal==='none'?'#ef4444':'#4ade80',9,'800');
}

var curNorm='wrong';
function selNorm(k){curNorm=k;['wrong','right'].forEach(function(b){document.getElementById('nBtn_'+b).classList.remove('active');});document.getElementById('nBtn_'+k).classList.add('active');renderNorm();}
function renderNorm(){
  var ctx=clr('cvNorm',420,260),W=420,H=260;
  var steps=curNorm==='wrong'?[
    {label:'1. Load FULL dataset',col:'#4ecdc4'},
    {label:'2. Fit scaler on ALL data',col:'#ef4444',warn:true},
    {label:'3. Transform all data',col:'#ef4444',warn:true},
    {label:'4. Split into train/val/test',col:'#fbbf24'},
  ]:[
    {label:'1. Split into train/val/test FIRST',col:'#4ade80'},
    {label:'2. Fit scaler on TRAIN only',col:'#4ade80'},
    {label:'3. Transform train/val/test (same scaler)',col:'#4ade80'},
    {label:'4. Train model',col:'#4ecdc4'},
  ];
  var bh=36,startY=25;
  steps.forEach(function(st,i){
    var y=startY+i*(bh+8);
    bx(ctx,20,y,W-40,bh,6,st.col+(st.warn?'30':'20'),st.col+(st.warn?'':'60'),st.warn?2.5:1.5);
    tx(ctx,st.label,W/2,y+bh/2,st.col,9,'700');
    if(st.warn){
      bx(ctx,W-95,y+7,70,22,4,'#ef444430','#ef4444',1.5);
      tx(ctx,'LEAK!',W-60,y+18,'#ef4444',8,'800');
    }
  });
  var resultY=startY+4*(bh+8)+10;
  bx(ctx,20,resultY,W-40,36,6,curNorm==='wrong'?'#ef444420':'#4ade8020',curNorm==='wrong'?'#ef444460':'#4ade8060',1.5);
  tx(ctx,curNorm==='wrong'?'Val/test stats leaked into training scaler \u2014 inflated test performance':'No leakage \u2014 test performance is unbiased',W/2,resultY+18,curNorm==='wrong'?'#ef4444':'#4ade80',9,'800');
  document.getElementById('iNorm').innerHTML=curNorm==='wrong'?
    '<span class="hl5">Preprocessing leakage:</span> scaler fitted on full dataset includes val/test statistics. The training set is normalised using information from the test set. Reported test performance is optimistically biased.<br>Impact: 0.1\u20135% inflated performance, harder to detect than it sounds.':
    '<span class="hl4">Correct:</span> split first. Fit scaler on training data only. Apply the same transform to val/test. This applies to: StandardScaler, PCA, TF-IDF, vocabulary, imputation values \u2014 all preprocessing statistics.';
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){
  renderPipe(); renderGPU(); renderCPUGPU();
});
</script>
</body>
</html>"""

DATA_PIPELINE_VISUAL_HEIGHT = 2200