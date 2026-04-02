CHECKPOINTING_VISUAL_HTML = r"""<!DOCTYPE html>
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
<h2>&#x1F4BE; Checkpointing</h2>
<p class="sub">Contents &middot; Save &amp; Resume &middot; Best-Model &middot; Distributed &middot; Formats &middot; Deploy &amp; Debug</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Checkpoint Contents</button>
  <button class="tab" onclick="showTab(1)">Save &amp; Resume</button>
  <button class="tab" onclick="showTab(2)">Best-Model Tracking</button>
  <button class="tab" onclick="showTab(3)">Distributed</button>
  <button class="tab" onclick="showTab(4)">Formats</button>
  <button class="tab" onclick="showTab(5)">Deploy &amp; Debug</button>
</div>

<!-- TAB 0: CHECKPOINT CONTENTS -->
<div id="tab0" class="panel active">
<div class="g2">
<div class="card">
  <h3>&#9312; Complete Checkpoint Components</h3>
  <canvas id="cvContents" width="420" height="300"></canvas>
  <div class="brow">
    <button class="s active" id="ccBtn_all"  onclick="selCC('all')">All Components</button>
    <button class="s" id="ccBtn_no_opt"  onclick="selCC('no_opt')">Missing Optimiser</button>
    <button class="s" id="ccBtn_no_sched" onclick="selCC('no_sched')">Missing Scheduler</button>
    <button class="s" id="ccBtn_no_epoch" onclick="selCC('no_epoch')">Missing Epoch</button>
  </div>
  <div class="info" id="iCC">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; State Dict vs Full Model Serialisation</h3>
  <canvas id="cvStateDict" width="420" height="300"></canvas>
  <div class="brow">
    <button class="s active" id="sdBtn_state" onclick="selSD('state')">State Dict</button>
    <button class="s" id="sdBtn_full"  onclick="selSD('full')">Full Model (pickle)</button>
  </div>
  <div class="info" id="iSD">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; What Each Component Contains</h3>
  <table class="tbl">
    <thead><tr><th>Component</th><th>Contains</th><th>Effect if missing on resume</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">model_state_dict</span></td><td>Weights, biases, BN running mean/var, buffers</td><td>Training restarts from random init &mdash; total loss.</td></tr>
      <tr><td><span class="hl2">optimiser_state_dict</span></td><td>Adam m&circ;, v&circ; per param; SGD momentum; step count</td><td>Moment estimates reset &rarr; 5&ndash;10% performance regression for many steps.</td></tr>
      <tr><td><span class="hl3">scheduler_state_dict</span></td><td>Current step, LR, warmup state, decay history</td><td>LR jumps back to LR_max (cosine) &rarr; instability spike.</td></tr>
      <tr><td><span class="hl6">scaler_state_dict (FP16)</span></td><td>Loss scale S, growth interval counter</td><td>S resets to 65536; likely overflow in first few steps.</td></tr>
      <tr><td><span class="hl">epoch / step</span></td><td>Current epoch number, global gradient step</td><td>Loop restarts from epoch 0; data ordering repeats; same examples seen twice.</td></tr>
      <tr><td><span class="hl5">rng_state</span></td><td>Python, NumPy, PyTorch CPU &amp; CUDA RNG states</td><td>Augmentation and dropout differ from original run &mdash; not exactly reproducible.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 1: SAVE & RESUME -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Atomic Write &mdash; Protecting Against Corruption</h3>
  <canvas id="cvAtomic" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="awBtn_naive"  onclick="selAW('naive')">Non-Atomic (Dangerous)</button>
    <button class="s" id="awBtn_atomic" onclick="selAW('atomic')">Atomic Write (Safe)</button>
  </div>
  <div class="info" id="iAW">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Strict vs Non-Strict Loading</h3>
  <canvas id="cvStrict" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="stBtn_strict" onclick="selStrict('strict')">strict=True</button>
    <button class="s" id="stBtn_loose"  onclick="selStrict('loose')">strict=False</button>
  </div>
  <div class="info" id="iStrict">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; map_location: Loading Across Devices</h3>
  <canvas id="cvMapLoc" width="860" height="148"></canvas>
</div>
</div>

<!-- TAB 2: BEST-MODEL TRACKING -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Best Checkpoint vs Final Checkpoint</h3>
  <canvas id="cvBest" width="420" height="270"></canvas>
  <div class="row"><label>Overfit degree</label>
    <input type="range" id="slOverfit" min="0" max="100" step="5" value="50" oninput="renderBest()">
    <span class="v" id="vOverfit">50%</span></div>
  <div class="info" id="iBest">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Checkpoint Retention Policies</h3>
  <canvas id="cvRetention" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="rpBtn_last3"   onclick="selRP('last3')">Keep Last 3</button>
    <button class="s" id="rpBtn_bestlast" onclick="selRP('bestlast')">Best + Last 3</button>
    <button class="s" id="rpBtn_exp"     onclick="selRP('exp')">Exponential</button>
  </div>
  <div class="info" id="iRP">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Checkpoint Frequency Guide</h3>
  <table class="tbl">
    <thead><tr><th>Training duration</th><th>Frequency</th><th>Rationale</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">&lt; 1 hour</span></td><td>End of training only</td><td>Recovery cost trivial vs checkpoint overhead.</td></tr>
      <tr><td><span class="hl4">1&ndash;8 hours</span></td><td>Every epoch or every N steps (~hourly)</td><td>Balance between overhead and recovery window.</td></tr>
      <tr><td><span class="hl3">Days</span></td><td>Every 30&ndash;60 minutes (step-based)</td><td>+ best-model checkpoint on val improvement.</td></tr>
      <tr><td><span class="hl">Weeks (LLM pre-training)</span></td><td>Every 500&ndash;1000 steps; async</td><td>Async save to avoid blocking the critical path.</td></tr>
      <tr><td><span class="hl5">Cloud preemptible / spot</span></td><td>Every 10&ndash;30 minutes</td><td>~30s kill warning. At $10/hr, 30 min lost = $5. Checkpoint aggressively.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 3: DISTRIBUTED -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; DDP: Only Rank 0 Saves</h3>
  <canvas id="cvDDP" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="ddpBtn_wrong" onclick="selDDP('wrong')">Wrong (all ranks save)</button>
    <button class="s" id="ddpBtn_right" onclick="selDDP('right')">Correct (rank 0 only)</button>
  </div>
  <div class="info" id="iDDP">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; FSDP: Sharded vs Consolidated Saving</h3>
  <canvas id="cvFSDP" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="fsBtn_sharded" onclick="selFSDP('sharded')">Sharded (per-rank)</button>
    <button class="s" id="fsBtn_full"    onclick="selFSDP('full')">Consolidated (rank 0)</button>
    <button class="s" id="fsBtn_async"   onclick="selFSDP('async')">Async Save</button>
  </div>
  <div class="info" id="iFSDP">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Distributed Checkpointing Rules</h3>
  <table class="tbl">
    <thead><tr><th>Scenario</th><th>Rule</th><th>Bug if broken</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">DDP: who saves</span></td><td>Only dist.get_rank()==0 saves</td><td>N identical files written, race conditions.</td></tr>
      <tr><td><span class="hl4">DDP: what to save</span></td><td>model.<span class="hl5">module</span>.state_dict() not model.state_dict()</td><td>'module.' prefix on all keys &rarr; strict load fails.</td></tr>
      <tr><td><span class="hl2">DDP: synchronisation</span></td><td>dist.barrier() before and after save</td><td>Ranks diverge; checkpoint is inconsistent.</td></tr>
      <tr><td><span class="hl3">FSDP: consolidated</span></td><td>FullStateDictConfig(offload_to_cpu=True, rank0_only=True)</td><td>GPU OOM during gather or non-portable sharded ckpt.</td></tr>
      <tr><td><span class="hl6">Async save</span></td><td>Wait for previous save thread before starting next</td><td>Two saves overlap, second overwrites .tmp of first.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 4: FORMATS -->
<div id="tab4" class="panel">
<div class="card">
  <h3>&#9312; Checkpoint Format Comparison</h3>
  <canvas id="cvFormats" width="860" height="220"></canvas>
</div>
<div class="g2">
<div class="card">
  <h3>&#9313; SafeTensors: Security &amp; Speed</h3>
  <canvas id="cvSafeTensors" width="420" height="210"></canvas>
  <div class="info" id="iST">Loading...</div>
</div>
<div class="card">
  <h3>&#9314; ONNX &amp; TorchScript: Cross-Runtime</h3>
  <canvas id="cvONNX" width="420" height="210"></canvas>
  <div class="info" id="iONNX">Loading...</div>
</div>
</div>
</div>

<!-- TAB 5: DEPLOY & DEBUG -->
<div id="tab5" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; train() vs eval(): What Changes</h3>
  <canvas id="cvTrainEval" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="teBtn_train" onclick="selTE('train')">model.train()</button>
    <button class="s" id="teBtn_eval"  onclick="selTE('eval')">model.eval()</button>
  </div>
  <div class="info" id="iTE">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Common Bugs &amp; Fixes</h3>
  <canvas id="cvBugs" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="bgBtn_1" onclick="selBug(0)">DDP prefix</button>
    <button class="s" id="bgBtn_2" onclick="selBug(1)">No eval()</button>
    <button class="s" id="bgBtn_3" onclick="selBug(2)">No opt state</button>
    <button class="s" id="bgBtn_4" onclick="selBug(3)">Wrong device</button>
    <button class="s" id="bgBtn_5" onclick="selBug(4)">Corrupt write</button>
  </div>
  <div class="info" id="iBug">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Deployment Verification Checklist</h3>
  <table class="tbl">
    <thead><tr><th>Step</th><th>Check</th><th>Code</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">1</span></td><td>No missing / unexpected keys</td><td>missing, unexpected = model.load_state_dict(..., strict=True)</td></tr>
      <tr><td><span class="hl4">2</span></td><td>Model in eval mode</td><td>model.eval(); assert not model.training</td></tr>
      <tr><td><span class="hl4">3</span></td><td>Smoke test on known input</td><td>assert torch.isfinite(output).all()</td></tr>
      <tr><td><span class="hl4">4</span></td><td>Val metric matches saved best</td><td>assert abs(val_metric - ckpt['best_val_metric']) &lt; 1e-3</td></tr>
      <tr><td><span class="hl4">5</span></td><td>Memory footprint checked</td><td>sum(p.numel() * p.element_size() for p in model.parameters())</td></tr>
      <tr><td><span class="hl4">6</span></td><td>Latency benchmarked</td><td>10 warmup + 100 timed forward passes; torch.cuda.synchronize()</td></tr>
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
function crv(pts,ctx,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||2.5;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderCC();renderSD();}if(i===1){renderAW();renderStrict();renderMapLoc();}if(i===2){renderBest();renderRP();}if(i===3){renderDDP();renderFSDP();}if(i===4){renderFormats();renderST();renderONNX();}if(i===5){renderTE();renderBugs();}}

// ============================================================
// TAB 0: CHECKPOINT CONTENTS
// ============================================================
var curCC='all';
var CC_INFO={
  all:'<span class="hl4">Complete checkpoint:</span> all 6 components present. Resume is exact &mdash; same LR, same data ordering, same moment estimates, bit-identical continuation.',
  no_opt:'<span class="hl5">Missing optimiser state:</span> Adam m\u0302 and v\u0302 reset to zero. Effective per-parameter LR is wrong. Training temporarily regresses 5\u201310% as moment estimates rebuild over hundreds of steps.',
  no_sched:'<span class="hl5">Missing scheduler state:</span> scheduler restarts from step 0. LR jumps back to LR_max in cosine annealing. Can cause instability spike immediately after resume.',
  no_epoch:'<span class="hl5">Missing epoch/step:</span> training loop restarts from epoch 0. Data sampler re-seeded from start \u2192 same ordering as the first run \u2192 each example seen twice in first epoch. Breaks data coverage guarantee.',
};
var CC_MISSING={all:[],no_opt:['optimiser_state_dict'],no_sched:['scheduler_state_dict'],no_epoch:['epoch','step']};
function selCC(k){curCC=k;['all','no_opt','no_sched','no_epoch'].forEach(function(b){document.getElementById('ccBtn_'+b).classList.remove('active');});document.getElementById('ccBtn_'+k).classList.add('active');renderCC();}
function renderCC(){
  var ctx=clr('cvContents',420,300),W=420,H=300;
  var missing=CC_MISSING[curCC];
  var components=[
    {key:'model_state_dict',label:'Model State Dict',sub:'weights, biases, BN buffers',col:'#4ade80'},
    {key:'optimiser_state_dict',label:'Optimiser State',sub:'Adam m\u0302,v\u0302 per parameter',col:'#4ecdc4'},
    {key:'scheduler_state_dict',label:'LR Scheduler',sub:'current step, LR, warmup',col:'#fbbf24'},
    {key:'scaler_state_dict',label:'GradScaler (FP16)',sub:'loss scale S, growth counter',col:'#c084fc'},
    {key:'epoch',label:'Epoch / Step',sub:'global training position',col:'#ff6b35'},
    {key:'rng_state',label:'RNG States',sub:'Python, NumPy, CUDA rng',col:'#38bdf8'},
  ];
  var bh=36,gap=6,startY=18;
  components.forEach(function(c,i){
    var y=startY+i*(bh+gap);
    var isMissing=missing.indexOf(c.key)>=0;
    var fill=isMissing?'#ef444418':c.col+'18';
    var stroke=isMissing?'#ef444460':c.col+'80';
    bx(ctx,16,y,W-32,bh,6,fill,stroke,isMissing?2.5:1.5);
    if(isMissing){
      ctx.strokeStyle='#ef444460';ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(18,y+2);ctx.lineTo(W-18,y+bh-2);ctx.stroke();
      tx(ctx,'MISSING',W-70,y+bh/2,'#ef4444',8,'800');
    }
    tx(ctx,c.label,80,y+bh/2-6,isMissing?'#3f3f46':c.col,9,'800','left');
    tx(ctx,c.sub,80,y+bh/2+7,'#52525b',7,'400','left');
    // Lock/check icon
    tx(ctx,isMissing?'\u26A0':'\u2713',32,y+bh/2,isMissing?'#ef4444':'#4ade80',12,'800');
  });
  document.getElementById('iCC').innerHTML=CC_INFO[curCC];
}

var curSD='state';
function selSD(k){curSD=k;['state','full'].forEach(function(b){document.getElementById('sdBtn_'+b).classList.remove('active');});document.getElementById('sdBtn_'+k).classList.add('active');renderSD();}
function renderSD(){
  var ctx=clr('cvStateDict',420,300),W=420,H=300;
  if(curSD==='state'){
    // State dict: data only, architecture separate
    bx(ctx,20,20,180,80,8,'#4ade8018','#4ade80',2);
    tx(ctx,'model.state_dict()',110,45,'#4ade80',9,'800');
    tx(ctx,'OrderedDict of tensors',110,62,'#52525b',7,'400');
    tx(ctx,'layer1.weight: Tensor',110,78,'#94a3b8',7,'400');
    bx(ctx,220,20,180,80,8,'#4ecdc418','#4ecdc4',2);
    tx(ctx,'Architecture code',310,50,'#4ecdc4',9,'800');
    tx(ctx,'class MyModel(nn.Module):',310,66,'#94a3b8',7,'400');
    tx(ctx,'def forward(self,x):...',310,80,'#52525b',7,'400');
    // Combine arrow
    arr(ctx,200,60,220,60,'#3f3f46',1.5);
    // Load
    bx(ctx,20,130,380,60,8,'#4ade8018','#4ade80',1.5);
    tx(ctx,'model.load_state_dict(state_dict)',210,155,'#4ade80',9,'800');
    tx(ctx,'strict=True checks key exact match',210,172,'#52525b',7,'400');
    // Pros
    var pros=['\u2713 Portable: load into any matching arch','  \u2713 Stable across code refactors','\u2713 No code execution risk','\u2713 Small: only tensor data stored'];
    pros.forEach(function(p,i){tx(ctx,p,30,210+i*18,'#4ade80',8,'400','left');});
    tx(ctx,'STATE DICT (RECOMMENDED)',W/2,H-12,'#4ade80',9,'800');
  } else {
    // Full model: code + data entangled via pickle
    bx(ctx,20,20,380,80,8,'#ef444418','#ef4444',2);
    tx(ctx,'torch.save(model, path)',210,40,'#ef4444',9,'800');
    tx(ctx,'pickle: architecture + weights entangled',210,58,'#fbbf24',8,'400');
    tx(ctx,'Serialises Python class definition',210,74,'#52525b',7,'400');
    // Issues
    var cons=['\u2715 Brittle: breaks on class rename/refactor','\u2715 Couples checkpoint to codebase version','\u2715 Security risk: arbitrary code on load','\u2715 Fails across Python versions','\u2715 Cannot load in different framework'];
    cons.forEach(function(c,i){tx(ctx,c,30,120+i*22,'#ef4444',8,'400','left');});
    // Warning box
    bx(ctx,20,240,380,44,6,'#ef444420','#ef4444',2);
    tx(ctx,'\u26A0 AVOID for any checkpoint that outlives one session',210,255,'#ef4444',8,'800');
    tx(ctx,'Use state dict always for long-term storage',210,270,'#71717a',7,'400');
    tx(ctx,'FULL PICKLE (AVOID)',W/2,H-12,'#ef4444',9,'800');
  }
  document.getElementById('iSD').innerHTML=curSD==='state'?
    '<span class="hl4">State dict:</span> saves only tensor data in a readable OrderedDict. Architecture code is separate. Stable, portable, safe.<br><span class="hl3">Convention:</span> .pt for all new checkpoints.':
    '<span class="hl5">Full pickle:</span> embeds the Python class definition. Any rename, move, or refactor breaks loading. A malicious .pt file can execute code on load.<br>Only acceptable for throwaway experiments lasting one session.';
}

// ============================================================
// TAB 1: SAVE & RESUME
// ============================================================
var curAW='naive';
function selAW(k){curAW=k;['naive','atomic'].forEach(function(b){document.getElementById('awBtn_'+b).classList.remove('active');});document.getElementById('awBtn_'+k).classList.add('active');renderAW();}
function renderAW(){
  var ctx=clr('cvAtomic',420,260),W=420,H=260;
  if(curAW==='naive'){
    // Show: write in-place, kill mid-write -> both lost
    var steps=[
      {label:'checkpoint_N-1.pt',col:'#4ade80',y:30,sub:'Previous checkpoint (intact)'},
      {label:'torch.save(ckpt, path)',col:'#fbbf24',y:90,sub:'Writing in-place...'},
      {label:'\u26A1 KILLED mid-write',col:'#ef4444',y:150,sub:'Power failure / OOM / SIGKILL'},
      {label:'checkpoint_N.pt = CORRUPT',col:'#ef4444',y:210,sub:'Previous overwritten. Both gone.'},
    ];
    steps.forEach(function(st,i){
      bx(ctx,20,st.y,W-40,46,6,st.col+'18',st.col+(i===2?'':i===3?'':'60'),i===2?2.5:1.5);
      tx(ctx,st.label,W/2,st.y+16,st.col,9,'800');
      tx(ctx,st.sub,W/2,st.y+32,'#52525b',7,'400');
      if(i<steps.length-1&&i!==1){arr(ctx,W/2,st.y+46,W/2,steps[i+1].y,'#3f3f46',1.5);}
    });
    // Red X on checkpoint N-1
    ctx.strokeStyle='#ef444460';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(22,32);ctx.lineTo(W-22,76);ctx.stroke();
  } else {
    var steps2=[
      {label:'torch.save(ckpt, path+\'.tmp\')',col:'#4ecdc4',y:20,sub:'Write to temp file first'},
      {label:'\u26A1 Killed here?',col:'#fbbf24',y:80,sub:'Only .tmp is corrupt \u2014 original safe!'},
      {label:'os.replace(path+\'.tmp\', path)',col:'#4ade80',y:140,sub:'Atomic rename (POSIX guarantee)'},
      {label:'checkpoint_N.pt is valid',col:'#4ade80',y:200,sub:'Either fully written or original preserved'},
    ];
    steps2.forEach(function(st,i){
      bx(ctx,20,st.y,W-40,46,6,st.col+'18',st.col+'60',1.5);
      tx(ctx,st.label,W/2,st.y+16,st.col,9,'800');
      tx(ctx,st.sub,W/2,st.y+32,'#52525b',7,'400');
      if(i<steps2.length-1)arr(ctx,W/2,st.y+46,W/2,steps2[i+1].y,'#3f3f46',1.5);
    });
    tx(ctx,'\u2713 Safe: os.replace() is atomic on Linux/macOS',W/2,H-8,'#4ade80',8,'700');
  }
  document.getElementById('iAW').innerHTML=curAW==='naive'?
    '<span class="hl5">Non-atomic write:</span> torch.save() overwrites the file in-place. If killed mid-write, the ORIGINAL is already gone and the NEW is corrupt. Both checkpoints are lost.<br>This failure mode is silent and unrecoverable.':
    '<span class="hl4">Atomic write:</span> write to .tmp, then os.replace(). On POSIX, rename is atomic &mdash; the kernel guarantees it either completes or the original remains intact.<br>If killed during torch.save(), path.tmp is corrupt but the PREVIOUS checkpoint at path is safe.';
}

var curStrict='strict';
function selStrict(k){curStrict=k;['strict','loose'].forEach(function(b){document.getElementById('stBtn_'+b).classList.remove('active');});document.getElementById('stBtn_'+b.replace('strict','strict').replace('loose','loose'),null);document.getElementById('stBtn_'+k).classList.add('active');renderStrict();}
function renderStrict(){
  var ctx=clr('cvStrict',420,260),W=420,H=260;
  var ckptKeys=['backbone.layer1.weight','backbone.layer2.weight','backbone.layer3.weight','head.fc.weight'];
  var modelKeys=curStrict==='strict'?['backbone.layer1.weight','backbone.layer2.weight','backbone.layer3.weight','head.fc.weight']:['backbone.layer1.weight','backbone.layer2.weight','backbone.layer3.weight','new_head.fc.weight'];
  var bh=28,gap=4,cx2=W/2;
  var leftX=20,rightX=cx2+20,colW=cx2-40;
  tx(ctx,'Checkpoint keys',leftX+colW/2,18,'#4ecdc4',9,'800');
  tx(ctx,'Model parameters',rightX+colW/2,18,'#fbbf24',9,'800');
  ckptKeys.forEach(function(k,i){
    var y=34+i*(bh+gap);
    bx(ctx,leftX,y,colW,bh,4,'#4ecdc420','#4ecdc4',1.5);
    tx(ctx,k,leftX+colW/2,y+bh/2,'#4ecdc4',7,'700');
  });
  modelKeys.forEach(function(k,i){
    var y=34+i*(bh+gap);
    var inCkpt=ckptKeys.indexOf(k)>=0;
    var col=inCkpt?'#fbbf24':'#ef4444';
    bx(ctx,rightX,y,colW,bh,4,col+'20',col,inCkpt?1.5:2.5);
    tx(ctx,k,rightX+colW/2,y+bh/2,col,7,'700');
    if(inCkpt){
      arr(ctx,leftX+colW,y+bh/2,rightX,y+bh/2,'#4ade8060',2);
    } else {
      tx(ctx,'not in ckpt',rightX+colW/2,y+bh/2+12,'#ef4444',6,'700');
    }
  });
  // Divider
  ln(ctx,cx2,10,cx2,H-30,'#1e1e2e',1);
  var status=curStrict==='strict'?(ckptKeys.join()+''===modelKeys.join()+''?'\u2714 Keys match exactly \u2014 load succeeds':'\u2716 Key mismatch \u2014 RuntimeError raised'):'\u2714 strict=False: missing & unexpected keys silently handled';
  var statusCol=curStrict==='strict'?(ckptKeys.join()===modelKeys.join()?'#4ade80':'#ef4444'):'#fbbf24';
  bx(ctx,20,H-36,W-40,28,6,statusCol+'18',statusCol,1.5);
  tx(ctx,status,W/2,H-22,statusCol,8,'700');
  document.getElementById('iStrict').innerHTML=curStrict==='strict'?
    '<span class="hl4">strict=True (default):</span> every checkpoint key must exactly match the model. Missing or unexpected keys raise RuntimeError. Use when resuming an exact checkpoint of the same model.':
    '<span class="hl3">strict=False:</span> missing checkpoint keys leave model parameters at init values; unexpected keys are silently skipped. <span class="hl">Always print</span> missing_keys and unexpected_keys to verify what was and was not loaded. Use for transfer learning, fine-tuning with a new head, or loading a partial pretrained backbone.';
}

function renderMapLoc(){
  var ctx=clr('cvMapLoc',860,148),W=860,H=148;
  var scenarios=[
    {from:'cuda:0',to:'cuda:0',label:'Same GPU',col:'#4ade80',code:'torch.load(path)',warn:null},
    {from:'cuda:0',to:'cuda:1',label:'Different GPU',col:'#fbbf24',code:'torch.load(path, map_location=\'cuda:1\')',warn:'Move opt state to device!'},
    {from:'cuda:0',to:'cpu',   label:'CPU inference',col:'#4ecdc4',code:'torch.load(path, map_location=\'cpu\')',warn:'Move opt state to device!'},
    {from:'cuda:0',to:'any',   label:'Auto detect',col:'#c084fc',code:'torch.load(path, map_location=device)',warn:null},
  ];
  var bw=(W-20)/scenarios.length-6,sx=10;
  scenarios.forEach(function(sc,i){
    var x=sx+i*(bw+6);
    bx(ctx,x,10,bw,H-20,6,sc.col+'15',sc.col+'50',1.5);
    tx(ctx,sc.label,x+bw/2,28,sc.col,9,'800');
    tx(ctx,'Saved on: '+sc.from,x+bw/2,46,sc.col+'80',7,'400');
    tx(ctx,'Load on: '+sc.to,x+bw/2,60,sc.col,8,'700');
    // Code box
    bx(ctx,x+4,70,bw-8,28,4,'#0d0d18','#1e1e2e',1);
    tx(ctx,sc.code,x+bw/2,84,'#e4e4e7',6,'400');
    if(sc.warn){
      bx(ctx,x+4,102,bw-8,22,4,'#fbbf2420','#fbbf2460',1);
      tx(ctx,'\u26A0 '+sc.warn,x+bw/2,113,'#fbbf24',6,'700');
    } else {
      tx(ctx,'\u2714 No extra steps needed',x+bw/2,113,'#4ade8060',6,'700');
    }
  });
}

// ============================================================
// TAB 2: BEST-MODEL TRACKING
// ============================================================
function renderBest(){
  var overfit=parseInt(document.getElementById('slOverfit').value)/100;
  document.getElementById('vOverfit').textContent=Math.round(overfit*100)+'%';
  var ctx=clr('cvBest',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-30;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1);ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Epoch',W/2,H-5,'#52525b',8,'400');tx(ctx,'Val loss',PAD+20,22,'#52525b',8,'400');
  var N=80,seed=42;function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
  var trainPts=[],valPts=[];
  var bestY=Infinity,bestX,finalY;
  for(var i=0;i<=N;i++){
    var t=i/N,cx=PAD+t*pW;
    var tr=2.5*Math.exp(-t*4)+0.3+rng()*0.04-0.02;
    var vl=2.5*Math.exp(-t*3)+0.42+rng()*0.04+(overfit*1.4)*Math.pow(Math.max(0,t-0.4),2.2);
    vl=Math.min(vl,3.0);
    var ty=H-PAD-Math.min(tr,2.8)/2.8*pH;
    var vy=H-PAD-Math.min(vl,2.8)/2.8*pH;
    trainPts.push([cx,ty]);valPts.push([cx,vy]);
    if(vy<bestY){bestY=vy;bestX=cx;}
  }
  finalY=valPts[N][1];
  crv(trainPts,ctx,'#4ecdc4',2);crv(valPts,ctx,'#ff6b35',2.5);
  // Best checkpoint marker
  ln(ctx,bestX,15,bestX,H-PAD,'#4ade8060',1.5,[4,4]);
  ctx.fillStyle='#4ade80';ctx.beginPath();ctx.arc(bestX,bestY,7,0,Math.PI*2);ctx.fill();
  tx(ctx,'Best ckpt',bestX+4,bestY-14,'#4ade80',8,'800','left');
  // Final marker
  ctx.fillStyle='#ef4444';ctx.beginPath();ctx.arc(PAD+pW,finalY,6,0,Math.PI*2);ctx.fill();
  tx(ctx,'Final',PAD+pW+4,finalY,'#ef4444',8,'800','left');
  // Gap annotation
  if(finalY<bestY-4){
    ctx.strokeStyle='#fbbf2460';ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(bestX,bestY);ctx.lineTo(PAD+pW,finalY);ctx.stroke();ctx.setLineDash([]);
    var gap=rnd((finalY-bestY)/pH*2.8,2);
    tx(ctx,'Gap: +'+gap+' val loss',( bestX+PAD+pW)/2,(bestY+finalY)/2-10,'#fbbf24',8,'700');
  }
  tx(ctx,'Train','#4ecdc4',null);tx(ctx,'Train',W-30,trainPts[N][1],'#4ecdc4',8,'700','right');
  tx(ctx,'Val','#ff6b35',null);  tx(ctx,'Val',W-30,valPts[Math.round(N*0.45)][1],'#ff6b35',8,'700','right');
  document.getElementById('iBest').innerHTML=
    'Overfit degree: <span class="hl3">'+Math.round(overfit*100)+'%</span><br>'+
    (overfit>0.3?'<span class="hl5">Significant overfit:</span> final checkpoint is worse than best by ~'+rnd((finalY-bestY)/pH*2.8,2)+' val loss units. Best-model tracking essential.':'<span class="hl4">Mild overfit:</span> gap is small. Best-model tracking still recommended as a safety net.')+
    '<br><span class="hl">Rule:</span> always deploy the BEST checkpoint, not the FINAL. Save them to separate paths.';
}

var curRP='last3';
function selRP(k){curRP=k;['last3','bestlast','exp'].forEach(function(b){document.getElementById('rpBtn_'+b).classList.remove('active');});document.getElementById('rpBtn_'+k).classList.add('active');renderRP();}
function renderRP(){
  var ctx=clr('cvRetention',420,270),W=420,H=270;
  var N=12,bw=(W-60)/N-3,y0=80,bh=50;
  var bestIdx=5; // best model at step 5
  var now=N-1;
  for(var i=0;i<N;i++){
    var x=30+i*(bw+3);
    var isBest=i===bestIdx;
    var kept=false;
    if(curRP==='last3') kept=(i>=N-3);
    else if(curRP==='bestlast') kept=(i>=N-3||i===bestIdx);
    else kept=(i>=N-3||i===bestIdx||i===0||i===2||i===5||i===8); // exponential-ish
    bx(ctx,x,y0,bw,bh,4,kept?'#4ecdc440':'#1e1e2e','#4ecdc4'+(kept?'':'20'),kept?2:0.5);
    if(isBest){bx(ctx,x,y0,bw,bh,4,'#4ade8030','#4ade80',2.5);}
    tx(ctx,'S'+i,x+bw/2,y0+bh/2,kept?(isBest?'#4ade80':'#4ecdc4'):'#2d2d40',8,'800');
    if(!kept){
      ctx.strokeStyle='#2d2d4060';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(x+2,y0+2);ctx.lineTo(x+bw-2,y0+bh-2);ctx.stroke();
    }
    if(isBest)tx(ctx,'BEST',x+bw/2,y0-12,'#4ade80',7,'800');
  }
  tx(ctx,'Checkpoint step index (S0=oldest, S11=newest)',W/2,50,'#52525b',8,'400');
  var kept2=[];
  for(var i=0;i<N;i++){
    if(curRP==='last3'&&i>=N-3)kept2.push('S'+i);
    else if(curRP==='bestlast'&&(i>=N-3||i===bestIdx))kept2.push('S'+i+(i===bestIdx?' (BEST)':''));
    else if(curRP==='exp'&&(i>=N-3||i===bestIdx||i===0||i===2||i===5||i===8))kept2.push('S'+i+(i===bestIdx?' (BEST)':''));
  }
  var rpLabel={'last3':'Keep Last 3','bestlast':'Best + Last 3','exp':'Exponential Retention'};
  tx(ctx,rpLabel[curRP],W/2,155,'#ff6b35',9,'800');
  tx(ctx,'Kept: '+kept2.join(', '),W/2,175,'#4ecdc4',8,'400');
  var size=kept2.length*14; // ~14 GB per checkpoint for 7B model
  tx(ctx,'Storage: ~'+size+' GB (at 14 GB/checkpoint for 7B model)',W/2,196,'#52525b',7,'400');
  var rpInfo={
    last3:'<span class="hl2">Keep Last 3:</span> rolling window. Oldest deleted when new one saved. Minimal storage. Risk: if all 3 are post-best, the best model is gone.',
    bestlast:'<span class="hl4">Best + Last 3 (recommended):</span> best-metric checkpoint is NEVER deleted by the rolling window. Last 3 for fault tolerance. Best for deploy.',
    exp:'<span class="hl3">Exponential:</span> dense recent coverage for debugging + sparse old coverage for reference. Complex but gives the most flexibility.'
  };
  document.getElementById('iRP').innerHTML=rpInfo[curRP];
}

// ============================================================
// TAB 3: DISTRIBUTED
// ============================================================
var curDDP='right';
function selDDP(k){curDDP=k;['wrong','right'].forEach(function(b){document.getElementById('ddpBtn_'+b).classList.remove('active');});document.getElementById('ddpBtn_'+k).classList.add('active');renderDDP();}
function renderDDP(){
  var ctx=clr('cvDDP',420,270),W=420,H=270;
  var ranks=4,bw2=70,gap2=10,startX=(W-(ranks*(bw2+gap2)-gap2))/2,y0=30,bh2=70;
  for(var r=0;r<ranks;r++){
    var x=startX+r*(bw2+gap2);
    var isRank0=r===0;
    var isSaving=curDDP==='wrong'||(curDDP==='right'&&isRank0);
    var col=isRank0?'#4ade80':(curDDP==='wrong'?'#fbbf24':'#3f3f46');
    bx(ctx,x,y0,bw2,bh2,6,col+'18',col,isRank0?2:1);
    tx(ctx,'GPU '+r,x+bw2/2,y0+20,col,9,'800');
    tx(ctx,'Rank '+r,x+bw2/2,y0+34,'#52525b',7,'400');
    if(isSaving){
      bx(ctx,x+4,y0+44,bw2-8,22,4,'#4ade8030','#4ade80',1.5);
      tx(ctx,'\u1F4BE save',x+bw2/2,y0+55,'#4ade80',8,'800');
    } else {
      bx(ctx,x+4,y0+44,bw2-8,22,4,'#1e1e2e','#2d2d40',1);
      tx(ctx,'idle',x+bw2/2,y0+55,'#2d2d40',8,'400');
    }
  }
  // Barrier lines
  var barY=120;
  ln(ctx,10,barY,W-10,barY,'#4ecdc460',1.5,[4,4]);
  tx(ctx,'dist.barrier()  \u2014  all ranks synchronise here',W/2,barY-10,'#4ecdc4',8,'700');
  ln(ctx,10,H-50,W-10,H-50,'#4ecdc460',1.5,[4,4]);
  tx(ctx,'dist.barrier()  \u2014  all wait for rank 0 to finish',W/2,H-60,'#4ecdc4',8,'700');
  // File output
  if(curDDP==='wrong'){
    bx(ctx,20,barY+18,W-40,36,6,'#ef444418','#ef4444',2);
    tx(ctx,'\u2715 4 identical files written = wasted I/O + race conditions',W/2,barY+36,'#ef4444',8,'700');
  } else {
    bx(ctx,20,barY+18,W-40,36,6,'#4ade8018','#4ade80',2);
    tx(ctx,'\u2714 Only rank 0 writes. Save model.module not model',W/2,barY+36,'#4ade80',8,'700');
  }
  // model.module warning
  bx(ctx,20,H-44,W-40,36,6,'#fbbf2418','#fbbf24',1.5);
  tx(ctx,curDDP==='wrong'?'\u26A0 Wrong: model.state_dict() \u2192 \'module.\' prefix added':'\u2714 Correct: model.module.state_dict() \u2192 no prefix',W/2,H-26,'#fbbf24',8,'700');
  document.getElementById('iDDP').innerHTML=curDDP==='wrong'?
    '<span class="hl5">All ranks save:</span> N identical files, race conditions, wasted disk I/O. DDP wraps the model \u2014 model.state_dict() has \'module.\' prefix. Loading into a non-DDP model fails.':
    '<span class="hl4">Rank 0 only, model.module:</span> one file, no races, no prefix. dist.barrier() before and after ensures consistent state across all ranks.';
}

var curFSDP='full';
function selFSDP(k){curFSDP=k;['sharded','full','async'].forEach(function(b){document.getElementById('fsBtn_'+b).classList.remove('active');});document.getElementById('fsBtn_'+k).classList.add('active');renderFSDP();}
function renderFSDP(){
  var ctx=clr('cvFSDP',420,270),W=420,H=270;
  if(curFSDP==='sharded'){
    // Each GPU holds a shard, saves separately
    var shards=4,sw=60,gap=12,sx2=(W-(shards*(sw+gap)-gap))/2;
    for(var g=0;g<shards;g++){
      var gx=sx2+g*(sw+gap);
      bx(ctx,gx,20,sw,60,6,'#ff6b3520','#ff6b35',1.5);
      tx(ctx,'GPU '+g,gx+sw/2,38,'#ff6b35',9,'800');
      tx(ctx,'shard '+g,gx+sw/2,54,'#52525b',7,'400');
      bx(ctx,gx,90,sw,30,4,'#fbbf2430','#fbbf24',1.5);
      tx(ctx,'shard_'+g+'.pt',gx+sw/2,105,'#fbbf24',7,'700');
    }
    bx(ctx,20,135,W-40,40,6,'#ef444420','#ef4444',2);
    tx(ctx,'\u2715 GPU-count dependent: must reload with same N GPUs',W/2,155,'#ef4444',8,'700');
    tx(ctx,'Portable only within same FSDP configuration',W/2,200,'#52525b',8,'400');
    document.getElementById('iFSDP').innerHTML='<span class="hl5">Sharded save:</span> each rank saves its local shard. Fast (no gather communication). But cannot be loaded with a different number of GPUs or FSDP config. Use only when model is too large to gather on one GPU.';
  } else if(curFSDP==='full'){
    // Gather all shards to rank 0, save unified
    var shards2=4,sw2=50,gp2=10,sx3=(W-140-(shards2*(sw2+gp2)-gp2))/2;
    for(var g=0;g<shards2;g++){
      var gx2=sx3+g*(sw2+gp2);
      bx(ctx,gx2,20,sw2,50,6,'#4ecdc420','#4ecdc4',1.5);
      tx(ctx,'GPU '+g,gx2+sw2/2,38,'#4ecdc4',8,'800');
      tx(ctx,'shard',gx2+sw2/2,52,'#52525b',6,'400');
      // Arrow to rank 0 collect box
      arr(ctx,gx2+sw2/2,72,W-130,110,'#4ecdc440',1.5);
    }
    bx(ctx,W-160,90,140,50,6,'#4ade8030','#4ade80',2);
    tx(ctx,'Rank 0',W-90,108,'#4ade80',9,'800');
    tx(ctx,'full gather',W-90,122,'#52525b',7,'400');
    arr(ctx,W-90,142,W-90,162,'#4ade80',1.5);
    bx(ctx,W-160,162,140,40,6,'#4ade8020','#4ade80',1.5);
    tx(ctx,'model.safetensors',W-90,182,'#4ade80',8,'700');
    tx(ctx,'FullStateDictConfig:',20,200,'#c084fc',8,'700','left');
    tx(ctx,'offload_to_cpu=True',20,216,'#94a3b8',7,'400','left');
    tx(ctx,'rank0_only=True',20,230,'#94a3b8',7,'400','left');
    bx(ctx,20,248,W-40,16,4,'#4ade8018',null);
    tx(ctx,'\u2714 Portable: load with any GPU count',W/2,256,'#4ade80',7,'700');
    document.getElementById('iFSDP').innerHTML='<span class="hl4">Consolidated save:</span> gather all shards to rank 0 (offloaded to CPU to avoid OOM). Save a single unified state dict. Portable \u2014 load with any GPU configuration. This is the recommended approach for most FSDP models.';
  } else {
    // Async save: CPU copy while GPU continues
    var steps3=[
      {label:'State dict \u2192 CPU copy',col:'#4ecdc4',y:20,sub:'v.cpu().clone() \u2014 fast GPU\u2192CPU transfer'},
      {label:'Thread: torch.save (background)',col:'#c084fc',y:80,sub:'CPU saves while GPU trains next step'},
      {label:'GPU: continues training',col:'#4ade80',y:80,sub:'No blocking! GPU + CPU overlap'},
      {label:'thread.join() before next save',col:'#fbbf24',y:160,sub:'Prevent two saves overlapping .tmp'},
    ];
    bx(ctx,20,20,W-40,46,6,'#4ecdc418','#4ecdc4',1.5);
    tx(ctx,'State dict \u2192 CPU copy (v.cpu().clone())',W/2,36,'#4ecdc4',9,'800');
    tx(ctx,'Fast GPU\u2192CPU. Negligible blocking.',W/2,52,'#52525b',7,'400');
    arr(ctx,W/2,68,W/4,88,'#c084fc80',1.5);arr(ctx,W/2,68,3*W/4,88,'#4ade8080',1.5);
    bx(ctx,20,88,W/2-30,46,6,'#c084fc18','#c084fc',1.5);
    tx(ctx,'Background thread',W/4,104,'#c084fc',9,'800');tx(ctx,'torch.save(snapshot,path)',W/4,118,'#52525b',7,'400');
    bx(ctx,W/2+10,88,W/2-30,46,6,'#4ade8018','#4ade80',1.5);
    tx(ctx,'GPU continues',3*W/4,104,'#4ade80',9,'800');tx(ctx,'Next training step',3*W/4,118,'#52525b',7,'400');
    bx(ctx,20,155,W-40,40,6,'#fbbf2418','#fbbf24',1.5);
    tx(ctx,'thread.join()  before  next save',W/2,170,'#fbbf24',9,'800');tx(ctx,'Prevents overlapping .tmp files',W/2,186,'#52525b',7,'400');
    tx(ctx,'CPU \u2016 GPU overlap: no training pause',W/2,212,'#4ade80',9,'800');
    document.getElementById('iFSDP').innerHTML='<span class="hl6">Async save:</span> snapshot the state dict to CPU immediately (fast), then save to disk in a background thread. GPU training continues uninterrupted. Critical: thread.join() before starting the next save to prevent two saves racing on the same .tmp file.';
  }
}

// ============================================================
// TAB 4: FORMATS
// ============================================================
function renderFormats(){
  var ctx=clr('cvFormats',860,220),W=860,H=220;
  var formats=[
    {name:'.pt / .pth',col:'#4ecdc4',security:1,speed:2,portable:1,crossFW:1,codeFree:0,use:'Training resume, all PyTorch workflows'},
    {name:'SafeTensors',col:'#4ade80',security:5,speed:5,portable:4,crossFW:4,codeFree:5,use:'Public distribution, HF Hub standard'},
    {name:'ONNX',col:'#fbbf24',security:3,speed:4,portable:5,crossFW:5,codeFree:5,use:'Cross-framework deploy, TensorRT, mobile'},
    {name:'TorchScript',col:'#c084fc',security:3,speed:4,portable:3,crossFW:2,codeFree:5,use:'C++ inference, mobile (LibTorch)'},
    {name:'HF Hub format',col:'#ff6b35',security:4,speed:4,portable:4,crossFW:4,codeFree:5,use:'Transformers ecosystem, public models'},
  ];
  var metrics=['Security','Load speed','Portability','Cross-framework','No code exec'];
  var colW=(W-20)/formats.length,bh2=H-80;
  var maxPad=10;
  formats.forEach(function(f,i){
    var x=10+i*colW;
    bx(ctx,x,10,colW-6,H-20,6,f.col+'12',f.col+'50',1.5);
    tx(ctx,f.name,x+(colW-6)/2,28,f.col,9,'800');
    var scores=[f.security,f.speed,f.portable,f.crossFW,f.codeFree];
    var barH=(bh2-30)/metrics.length-4;
    scores.forEach(function(sc,mi){
      var by=48+mi*(barH+4);
      var maxW=colW-24;
      bx(ctx,x+8,by,maxW,barH,3,'#1e1e2e',null);
      bx(ctx,x+8,by,maxW*sc/5,barH,3,f.col+'50',f.col,1);
      tx(ctx,metrics[mi],x+12,by+barH/2,'#52525b',6,'400','left');
      tx(ctx,sc+'/5',x+colW-16,by+barH/2,f.col,7,'800','right');
    });
    tx(ctx,f.use,x+(colW-6)/2,H-14,'#52525b',6,'400');
  });
}

function renderST(){
  var ctx=clr('cvSafeTensors',420,210),W=420,H=210;
  // Show memory-mapped loading benefit
  bx(ctx,10,10,W-20,40,6,'#ef444418','#ef4444',1.5);
  tx(ctx,'.pt / pickle: 30 seconds to load 13 GB model',W/2,30,'#ef4444',8,'700');
  bx(ctx,10,58,W-20,40,6,'#4ade8018','#4ade80',2);
  tx(ctx,'SafeTensors mmap=True: < 1 second',W/2,78,'#4ade80',9,'800');
  // Feature list
  var features=[
    ['\u1F512 No code execution: binary header + raw data','#4ade80'],
    ['\u26A1 Memory-mapped: load only what you need','#4ecdc4'],
    ['\u2194 Cross-framework: PyTorch / TF / JAX / Flax','#fbbf24'],
    ['\u2702 Lazy loading: load one layer at a time','#c084fc'],
  ];
  features.forEach(function(f,i){
    bx(ctx,10,108+i*24,W-20,20,4,f[1]+'12',f[1]+'40',1);
    tx(ctx,f[0],22,118+i*24,f[1],8,'700','left');
  });
  document.getElementById('iST').innerHTML=
    '<span class="hl4">SafeTensors is now the default on Hugging Face Hub.</span><br>'+
    'Use for any public checkpoint. The mmap loading makes 13 GB models load in &lt;1s vs 30s for pickle.<br>'+
    '<code>from safetensors.torch import save_file, load_file</code>';
}

function renderONNX(){
  var ctx=clr('cvONNX',420,210),W=420,H=210;
  var runtimes=[
    ['PyTorch model','#4ecdc4',20],
    ['torch.onnx.export()','#fbbf24',60],
    ['model.onnx','#ff6b35',100],
  ];
  runtimes.forEach(function(r,i){
    bx(ctx,10,r[2],200,34,6,r[1]+'18',r[1],1.5);
    tx(ctx,r[0],110,r[2]+17,r[1],9,'800');
    if(i<runtimes.length-1){
      ctx.fillStyle='#3f3f46';ctx.beginPath();ctx.moveTo(105,r[2]+34);ctx.lineTo(100,r[2]+44);ctx.lineTo(110,r[2]+44);ctx.closePath();ctx.fill();
    }
  });
  var targets=[['ONNX Runtime','#4ade80',20],['TensorRT','#c084fc',60],['OpenVINO','#4ecdc4',100],['Core ML','#ff6b35',140]];
  targets.forEach(function(t){
    bx(ctx,W/2+10,t[2],180,28,6,t[1]+'18',t[1],1.5);
    tx(ctx,t[0],W/2+100,t[2]+14,t[1],9,'800');
    ln(ctx,220,118,W/2+10,t[2]+14,t[1]+'50',1.5);
  });
  tx(ctx,'model.eval() required before export!',110,170,'#ef4444',7,'700');
  document.getElementById('iONNX').innerHTML=
    '<span class="hl3">ONNX:</span> exports the model graph + weights. One file runs across ONNX Runtime, TensorRT, OpenVINO, Core ML.<br>'+
    '<span class="hl5">Constraint:</span> model must be in eval mode. Dynamic shapes need dynamic_axes. Custom ops require custom ONNX operators.<br>'+
    '<span class="hl6">TorchScript:</span> similar but stays in PyTorch ecosystem. Use for C++ / mobile (LibTorch) without Python.';
}

// ============================================================
// TAB 5: DEPLOY & DEBUG
// ============================================================
var curTE='eval';
function selTE(k){curTE=k;['train','eval'].forEach(function(b){document.getElementById('teBtn_'+b).classList.remove('active');});document.getElementById('teBtn_'+k).classList.add('active');renderTE();}
function renderTE(){
  var ctx=clr('cvTrainEval',420,260),W=420,H=260;
  var isTrain=curTE==='train';
  var col=isTrain?'#4ecdc4':'#4ade80';
  tx(ctx,isTrain?'model.train()':'model.eval()',W/2,22,col,11,'800');
  var layers=[
    {name:'Dropout',trainBehaviour:'Random mask (p=0.3)',evalBehaviour:'Pass-through (no drop)',danger:isTrain?null:'Stochastic output if forgotten'},
    {name:'BatchNorm',trainBehaviour:'Batch stats + update running mean/var',evalBehaviour:'Stored running mean/var (no update)',danger:isTrain?null:'BN(1) \u2192 divide by 0, garbage output'},
    {name:'LayerNorm',trainBehaviour:'Per-example stats (identical)',evalBehaviour:'Per-example stats (identical)',danger:null},
    {name:'model.training',trainBehaviour:'True',evalBehaviour:'False \u2014 check with assert not model.training',danger:null},
  ];
  var bh3=42,sy=42;
  layers.forEach(function(l,i){
    var y=sy+i*(bh3+6);
    var lCol=l.danger?'#ef4444':col;
    bx(ctx,10,y,W-20,bh3,6,lCol+'12',lCol+(l.danger?'':'50'),l.danger?2:1.5);
    tx(ctx,l.name,W/2,y+12,lCol,10,'800');
    tx(ctx,isTrain?l.trainBehaviour:l.evalBehaviour,W/2,y+27,'#94a3b8',7,'400');
    if(!isTrain&&l.danger){
      bx(ctx,W-130,y+5,120,12,3,'#ef444420',null);
      tx(ctx,'\u26A0 '+l.danger,W-70,y+11,'#ef4444',5,'700');
    }
  });
  document.getElementById('iTE').innerHTML=isTrain?
    '<span class="hl2">model.train():</span> Dropout randomly masks activations. BatchNorm computes batch stats and updates running mean/var. Required before every training loop.':
    '<span class="hl4">model.eval():</span> Dropout passes all activations unchanged. BatchNorm uses stored running stats &mdash; no update.<br><span class="hl5">Always call model.eval() + torch.no_grad() for inference.</span> Forgetting eval() makes output stochastic &mdash; a subtle, hard-to-catch bug.';
}

var BUGS=[
  {title:'DDP \'module.\' prefix',col:'#fbbf24',symptom:'RuntimeError: Missing key(s): \'layer1.weight\'',cause:'Saved model.state_dict() (DDP wrapper) instead of model.module.state_dict()',fix:'model.module.state_dict() always in DDP. Or strip prefix: {k[7:]:v for k,v...}'},
  {title:'Forgot model.eval()',col:'#ef4444',symptom:'Inference output stochastic, worse than val',cause:'Dropout still active in train mode. Output changes every forward pass.',fix:'model.eval() immediately after loading checkpoint for inference. assert not model.training.'},
  {title:'No optimiser state',col:'#ff6b35',symptom:'Resumed training slower for first 50+ epochs',cause:'Adam m\u0302,v\u0302 reset to zero. Per-param LR wrong until estimates rebuild.',fix:'Always restore optimiser.load_state_dict(). Only skip for fine-tuning on a new task.'},
  {title:'Wrong map_location',col:'#c084fc',symptom:'Expected all tensors on same device / slow CPU-GPU',cause:'torch.load(path) without map_location. Tensors loaded to original device.',fix:'torch.load(path, map_location=device). Then move opt states: state[k]=v.to(device).'},
  {title:'Corrupt checkpoint',col:'#4ecdc4',symptom:'RuntimeError: PytorchStreamReader failed reading zip',cause:'Process killed during torch.save(). Partial write left corrupt file.',fix:'Atomic write: torch.save(ckpt, path+\'.tmp\'); os.replace(\'.tmp\', path).'},
];
var curBug=0;
function selBug(i){curBug=i;for(var b=0;b<5;b++){document.getElementById('bgBtn_'+(b+1)).classList.remove('active');}document.getElementById('bgBtn_'+(i+1)).classList.add('active');renderBugs();}
function renderBugs(){
  var bg=BUGS[curBug];
  var ctx=clr('cvBugs',420,260),W=420,H=260;
  var rows=[
    {label:'Bug',val:bg.title,col:bg.col},
    {label:'Symptom',val:bg.symptom,col:'#94a3b8'},
    {label:'Cause',val:bg.cause,col:'#fbbf24'},
    {label:'Fix',val:bg.fix,col:'#4ade80'},
  ];
  var bh4=48,sy=14;
  rows.forEach(function(r,i){
    var y=sy+i*(bh4+8);
    bx(ctx,10,y,W-20,bh4,6,r.col+'12',r.col+(i===0?'':'40'),i===0?2:1);
    tx(ctx,r.label,25,y+bh4/2,r.col,8,'800','left');
    var words=r.val.split(' '),lines=[''],li=0;
    words.forEach(function(w){
      if((lines[li]+w).length<52)lines[li]+=(lines[li]?' ':'')+w;
      else{li++;lines.push(w);}
    });
    lines.forEach(function(l,ll){tx(ctx,l,105,y+(bh4-(lines.length*12))/2+ll*12+6,r.col==='#94a3b8'?r.col:'#e4e4e7',7,'400','left');});
  });
  document.getElementById('iBug').innerHTML=
    '<span class="hl">Bug:</span> <span style="color:'+bg.col+'">'+bg.title+'</span><br>'+
    '<span class="hl3">Symptom:</span> '+bg.symptom+'<br>'+
    '<span class="hl5">Cause:</span> '+bg.cause+'<br>'+
    '<span class="hl4">Fix:</span> '+bg.fix;
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){renderCC();renderSD();});
</script>
</body>
</html>"""

CHECKPOINTING_VISUAL_HEIGHT = 2200