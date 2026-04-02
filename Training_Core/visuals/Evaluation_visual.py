EVALUATION_VISUAL_HTML = r"""<!DOCTYPE html>
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
.tab.active{color:#ff6b35;border-bottom-color:#ff6b35;}
.panel{display:none;}
.panel.active{display:block;}
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
button.s:hover{background:#2d2d40;color:#e4e4e7;}
button.s.active{background:#ff6b35;color:#08080f;border-color:#ff6b35;}
.tbl{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.71em;}
.tbl th{padding:6px 9px;text-align:left;color:#52525b;border-bottom:2px solid #1e1e2e;font-weight:800;text-transform:uppercase;letter-spacing:.05em;}
.tbl td{padding:6px 9px;border-bottom:1px solid #0f0f18;color:#94a3b8;vertical-align:middle;line-height:1.5;}
.tbl tr:hover td{background:#0d0d18;}
</style>
</head>
<body>
<h2>&#x1F4CA; Evaluation During Training</h2>
<p class="sub">Data Splits &middot; Val Loop &middot; Classification Metrics &middot; Calibration &middot; Regression &amp; Language &middot; Diagnosis Guide</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Data Splits</button>
  <button class="tab" onclick="showTab(1)">Training Loop</button>
  <button class="tab" onclick="showTab(2)">Classification</button>
  <button class="tab" onclick="showTab(3)">Calibration</button>
  <button class="tab" onclick="showTab(4)">Metrics Guide</button>
  <button class="tab" onclick="showTab(5)">Diagnosis</button>
</div>

<!-- TAB 0: DATA SPLITS -->
<div id="tab0" class="panel active">
<div class="g2">
<div class="card">
  <h3>&#9312; Three-Way Split &mdash; The Only Safe Setup</h3>
  <canvas id="cvSplit" width="420" height="240"></canvas>
  <div class="row"><label>Dataset size</label>
    <input type="range" id="slDS" min="0" max="3" step="1" value="1" oninput="renderSplit()">
    <span class="v" id="vDS">1K-100K</span></div>
  <div class="info" id="iSplit">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; K-Fold Cross-Validation</h3>
  <canvas id="cvKFold" width="420" height="240"></canvas>
  <div class="row"><label>K folds</label>
    <input type="range" id="slK" min="3" max="10" step="1" value="5" oninput="renderKFold()">
    <span class="v" id="vK">5</span></div>
  <div class="info" id="iKFold">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Split Rules &mdash; What Contaminates Your Evaluation</h3>
  <table class="tbl">
    <thead><tr><th>Mistake</th><th>Effect</th><th>Fix</th></tr></thead>
    <tbody>
      <tr><td><span class="hl5">Using test set for tuning</span></td><td>Test accuracy is in-sample. True gen. error is unknown.</td><td>Tune on val set only. Evaluate test ONCE at the very end.</td></tr>
      <tr><td><span class="hl5">Preprocessing on full dataset</span></td><td>Normalisation stats leak val/test info into training.</td><td>Fit preprocessing only on train split. Apply to val/test.</td></tr>
      <tr><td><span class="hl5">Random split on time series</span></td><td>Future data leaks into training. Massive inflation.</td><td>Chronological split: train on past, val/test on future.</td></tr>
      <tr><td><span class="hl5">Random split on grouped data</span></td><td>Same patient/doc in train &amp; test. Model memorises IDs.</td><td>Group split: all examples from one group in ONE split.</td></tr>
      <tr><td><span class="hl5">No stratification</span></td><td>Class distribution varies by luck across splits.</td><td>Stratified split: preserve class ratios in every split.</td></tr>
      <tr><td><span class="hl5">100 HP configs, one val set</span></td><td>Validation overfitting. Reported val &gt; true test perf.</td><td>Nested CV or hold-out test set never used for selection.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 1: TRAINING LOOP -->
<div id="tab1" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; The Validation Loop Structure</h3>
  <canvas id="cvLoop" width="420" height="300"></canvas>
  <div class="info" id="iLoop">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Loss Curve Patterns &mdash; What to Look For</h3>
  <canvas id="cvPatterns" width="420" height="300"></canvas>
  <div class="brow">
    <button class="s active" id="pBtn_good"  onclick="selPat('good')">Good</button>
    <button class="s" id="pBtn_over"  onclick="selPat('over')">Overfitting</button>
    <button class="s" id="pBtn_under" onclick="selPat('under')">Underfitting</button>
    <button class="s" id="pBtn_spike" onclick="selPat('spike')">Loss Spike</button>
    <button class="s" id="pBtn_div"   onclick="selPat('div')">Divergence</button>
  </div>
  <div class="info" id="iPat">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; The Three Critical Mode Switches</h3>
  <table class="tbl">
    <thead><tr><th>Call</th><th>What it activates</th><th>Bug if forgotten</th></tr></thead>
    <tbody>
      <tr><td><span class="hl4">model.train()</span></td><td>Dropout ON &bull; BatchNorm computes batch stats &bull; Running stats updated</td><td>Dropout disabled during training &rarr; no regularisation. BN stats mismatch.</td></tr>
      <tr><td><span class="hl2">model.eval()</span></td><td>Dropout OFF (all neurons active) &bull; BatchNorm uses stored running mean/var</td><td>Val loss computed with random dropout &rarr; stochastic, biased high. Model selection corrupted.</td></tr>
      <tr><td><span class="hl6">torch.no_grad()</span></td><td>No gradient tape. No intermediate activation storage for backward pass.</td><td>Correct results but ~25% slower and may OOM on large models. Always use in eval.</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 2: CLASSIFICATION METRICS -->
<div id="tab2" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Confusion Matrix &mdash; Interactive</h3>
  <canvas id="cvCM" width="420" height="260"></canvas>
  <div class="row"><label>Threshold</label>
    <input type="range" id="slThr" min="0.01" max="0.99" step="0.01" value="0.5" oninput="renderCM()">
    <span class="v" id="vThr">0.50</span></div>
  <div class="info" id="iCM">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; ROC Curve &amp; PR Curve</h3>
  <canvas id="cvROC" width="420" height="260"></canvas>
  <div class="brow">
    <button class="s active" id="rocBtn_roc" onclick="selROC('roc')">ROC Curve</button>
    <button class="s" id="rocBtn_pr" onclick="selROC('pr')">PR Curve</button>
  </div>
  <div class="info" id="iROC">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Precision, Recall, F1 &mdash; When to Use Which</h3>
  <table class="tbl">
    <thead><tr><th>Metric</th><th>Formula</th><th>Optimise when</th><th>Example domain</th></tr></thead>
    <tbody>
      <tr><td><span class="hl2">Precision</span></td><td>TP/(TP+FP)</td><td>False alarms are costly</td><td>Spam filter, recommendation systems</td></tr>
      <tr><td><span class="hl">Recall</span></td><td>TP/(TP+FN)</td><td>Missing positives is costly</td><td>Cancer screening, fraud detection</td></tr>
      <tr><td><span class="hl3">F1</span></td><td>2&sdot;P&sdot;R/(P+R)</td><td>Balance both</td><td>NLP NER, information retrieval</td></tr>
      <tr><td><span class="hl4">F2</span></td><td>(5&sdot;P&sdot;R)/(4P+R)</td><td>Recall 2&times; more important</td><td>Medical diagnosis, safety systems</td></tr>
      <tr><td><span class="hl6">F0.5</span></td><td>(1.25&sdot;P&sdot;R)/(0.25P+R)</td><td>Precision 2&times; more important</td><td>Search results, ad targeting</td></tr>
      <tr><td><span class="hl5">Accuracy</span></td><td>(TP+TN)/N</td><td><span class="hl5">Balanced classes only</span></td><td>99% neg &rarr; 99% acc predicting all neg!</td></tr>
      <tr><td><span class="hl2">ROC-AUC</span></td><td>Area under ROC</td><td>Compare models across thresholds</td><td>Balanced binary clf, ranking</td></tr>
      <tr><td><span class="hl3">PR-AUC</span></td><td>Area under PR</td><td>Rare positive class (&lt;10%)</td><td>Fraud, rare disease, click prediction</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- TAB 3: CALIBRATION -->
<div id="tab3" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Reliability Diagram (Calibration Curve)</h3>
  <canvas id="cvCal" width="420" height="270"></canvas>
  <div class="brow">
    <button class="s active" id="calBtn_over"  onclick="selCal('over')">Overconfident</button>
    <button class="s" id="calBtn_under" onclick="selCal('under')">Underconfident</button>
    <button class="s" id="calBtn_good"  onclick="selCal('good')">Well-calibrated</button>
  </div>
  <div class="info" id="iCal">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Temperature Scaling &mdash; Post-hoc Calibration</h3>
  <canvas id="cvTemp" width="420" height="270"></canvas>
  <div class="row"><label>Temperature T</label>
    <input type="range" id="slT" min="0.2" max="4" step="0.1" value="1" oninput="renderTemp()">
    <span class="v" id="vT">1.0</span></div>
  <div class="info" id="iTemp">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; ECE (Expected Calibration Error) &amp; Calibration Methods</h3>
  <div class="g2">
    <div>
      <div style="font-size:0.72em;font-weight:800;color:#4ecdc4;font-family:'JetBrains Mono',monospace;margin-bottom:8px;text-transform:uppercase;">ECE Interpretation</div>
      <table class="tbl">
        <thead><tr><th>ECE value</th><th>Calibration quality</th></tr></thead>
        <tbody>
          <tr><td><span class="hl4">0.00 &ndash; 0.02</span></td><td>Excellent. Production-ready probabilities.</td></tr>
          <tr><td><span class="hl3">0.02 &ndash; 0.05</span></td><td>Good. Most tasks do not need improvement.</td></tr>
          <tr><td><span class="hl">0.05 &ndash; 0.10</span></td><td>Moderate. Apply temperature scaling.</td></tr>
          <tr><td><span class="hl5">0.10 &ndash; 0.30</span></td><td>Poor. Typical uncalibrated neural network.</td></tr>
          <tr><td><span class="hl5">&gt; 0.30</span></td><td>Severe. Probabilities should not be trusted.</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <div style="font-size:0.72em;font-weight:800;color:#c084fc;font-family:'JetBrains Mono',monospace;margin-bottom:8px;text-transform:uppercase;">Post-hoc Methods</div>
      <table class="tbl">
        <thead><tr><th>Method</th><th>Params</th><th>When to use</th></tr></thead>
        <tbody>
          <tr><td><span class="hl4">Temperature scaling</span></td><td>1 (T)</td><td>Default. Fast, cannot overfit. Works for most NNs.</td></tr>
          <tr><td><span class="hl3">Platt scaling</span></td><td>2 (a, b)</td><td>When T-scaling is insufficient. Needs ~1K val samples.</td></tr>
          <tr><td><span class="hl6">Isotonic regression</span></td><td>Many</td><td>Large val set only. Non-parametric, can overfit.</td></tr>
          <tr><td><span class="hl2">Label smoothing</span></td><td>&alpha;</td><td>During training. Prevents overconfidence from the start.</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
</div>

<!-- TAB 4: METRICS GUIDE -->
<div id="tab4" class="panel">
<div class="card">
  <h3>&#9312; Metric Selection by Task Type</h3>
  <canvas id="cvMG" width="860" height="220"></canvas>
</div>
<div class="g2">
<div class="card">
  <h3>&#9313; Regression Metrics Compared</h3>
  <canvas id="cvReg" width="420" height="220"></canvas>
  <div class="info" id="iReg">Loading...</div>
</div>
<div class="card">
  <h3>&#9314; Multi-class Averaging (Imbalanced Example)</h3>
  <canvas id="cvAvg" width="420" height="220"></canvas>
  <div class="info" id="iAvg">Loading...</div>
</div>
</div>
</div>

<!-- TAB 5: DIAGNOSIS -->
<div id="tab5" class="panel">
<div class="g2">
<div class="card">
  <h3>&#9312; Generalisation Gap Analyser</h3>
  <canvas id="cvGap" width="420" height="270"></canvas>
  <div class="row"><label>Train loss</label>
    <input type="range" id="slTL" min="0.05" max="2.0" step="0.05" value="0.3" oninput="renderGap()">
    <span class="v" id="vTL">0.30</span></div>
  <div class="row"><label>Val loss</label>
    <input type="range" id="slVL" min="0.05" max="2.0" step="0.05" value="0.6" oninput="renderGap()">
    <span class="v" id="vVL">0.60</span></div>
  <div class="info" id="iGap">Loading...</div>
</div>
<div class="card">
  <h3>&#9313; Common Evaluation Mistakes</h3>
  <canvas id="cvMistakes" width="420" height="270"></canvas>
  <div class="info" id="iMistakes">Loading...</div>
</div>
</div>
<div class="card">
  <h3>&#9314; Evaluation Checklist Before Reporting Results</h3>
  <div class="g2">
    <div>
      <div style="font-size:0.72em;font-weight:800;color:#4ade80;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">Data &amp; Splits</div>
      <table class="tbl">
        <tbody>
          <tr><td><span class="hl4">&#9633;</span></td><td>Train/val/test splits are non-overlapping and stratified</td></tr>
          <tr><td><span class="hl4">&#9633;</span></td><td>Preprocessing fitted ONLY on training split</td></tr>
          <tr><td><span class="hl4">&#9633;</span></td><td>No temporal or group leakage in splits</td></tr>
          <tr><td><span class="hl4">&#9633;</span></td><td>&ge;1,000 examples per evaluation set</td></tr>
          <tr><td><span class="hl4">&#9633;</span></td><td>Val and test distributions match</td></tr>
        </tbody>
      </table>
      <div style="font-size:0.72em;font-weight:800;color:#4ecdc4;font-family:'JetBrains Mono',monospace;margin:12px 0 8px;">Model State</div>
      <table class="tbl">
        <tbody>
          <tr><td><span class="hl2">&#9633;</span></td><td>model.eval() called before evaluation</td></tr>
          <tr><td><span class="hl2">&#9633;</span></td><td>torch.no_grad() context active</td></tr>
          <tr><td><span class="hl2">&#9633;</span></td><td>Best checkpoint loaded (not final epoch)</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <div style="font-size:0.72em;font-weight:800;color:#fbbf24;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">Metric Computation</div>
      <table class="tbl">
        <tbody>
          <tr><td><span class="hl3">&#9633;</span></td><td>Loss accumulated weighted by sample count</td></tr>
          <tr><td><span class="hl3">&#9633;</span></td><td>Per-class metrics reported for imbalanced data</td></tr>
          <tr><td><span class="hl3">&#9633;</span></td><td>Baseline comparison included</td></tr>
          <tr><td><span class="hl3">&#9633;</span></td><td>Threshold chosen for the task (not always 0.5)</td></tr>
        </tbody>
      </table>
      <div style="font-size:0.72em;font-weight:800;color:#c084fc;font-family:'JetBrains Mono',monospace;margin:12px 0 8px;">Reporting</div>
      <table class="tbl">
        <tbody>
          <tr><td><span class="hl6">&#9633;</span></td><td>Val metrics from BEST checkpoint, not final</td></tr>
          <tr><td><span class="hl6">&#9633;</span></td><td>Test metrics reported only ONCE, on final model</td></tr>
          <tr><td><span class="hl6">&#9633;</span></td><td>3&ndash;5 seeds run; mean &plusmn; std reported</td></tr>
          <tr><td><span class="hl6">&#9633;</span></td><td>Statistical significance checked for small gaps</td></tr>
        </tbody>
      </table>
    </div>
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
function cv2(pts,ctx,col,lw){ctx.strokeStyle=col;ctx.lineWidth=lw||2.5;ctx.beginPath();pts.forEach(function(p,i){if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();}
function bx(ctx,x,y,w,h,r,fill,stroke,sw){ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,w,h,r||4);else ctx.rect(x,y,w,h);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw||1.5;ctx.stroke();}}

function showTab(i){for(var t=0;t<6;t++){document.getElementById('tab'+t).classList.remove('active');document.querySelectorAll('.tab')[t].classList.remove('active');}document.getElementById('tab'+i).classList.add('active');document.querySelectorAll('.tab')[i].classList.add('active');if(i===0){renderSplit();renderKFold();}if(i===1){renderLoop();renderPat();}if(i===2){renderCM();renderROC();}if(i===3){renderCal();renderTemp();}if(i===4){renderMG();renderReg();renderAvg();}if(i===5){renderGap();renderMistakes();}}

// ============================================================
// TAB 0: DATA SPLITS
// ============================================================
var DS_CONFIGS=[
  {label:'< 1,000',train:60,val:20,test:20,note:'Consider k-fold CV instead of fixed val/test'},
  {label:'1K\u2013100K',train:70,val:15,test:15,note:'Standard split. Stratify classes.'},
  {label:'100K\u20131M',train:80,val:10,test:10,note:'Standard split. Val is large enough.'},
  {label:'> 1M',train:98,val:1,test:1,note:'1% of 10M = 100K samples. More than enough.'}
];
function renderSplit(){
  var idx=parseInt(document.getElementById('slDS').value);
  var cfg=DS_CONFIGS[idx];
  document.getElementById('vDS').textContent=cfg.label;
  var ctx=clr('cvSplit',420,240),W=420,H=240;
  var barY=90,barH=50,barX=20,barW=W-40;

  // Draw three-way bar
  var segments=[
    {pct:cfg.train,col:'#4ecdc4',label:'TRAIN\n'+cfg.train+'%',sub:'Gradient updates'},
    {pct:cfg.val,  col:'#fbbf24',label:'VAL\n'+cfg.val+'%',  sub:'Tuning & selection'},
    {pct:cfg.test, col:'#c084fc',label:'TEST\n'+cfg.test+'%', sub:'Final report only'}
  ];
  var xOff=barX;
  segments.forEach(function(seg){
    var sw=barW*seg.pct/100;
    bx(ctx,xOff,barY,sw,barH,0,seg.col+'30',seg.col,2);
    // label inside if wide enough
    if(sw>50){
      tx(ctx,seg.label.split('\n')[0],xOff+sw/2,barY+16,seg.col,10,'800');
      tx(ctx,seg.label.split('\n')[1],xOff+sw/2,barY+30,'#fbbf24',10,'800');
      tx(ctx,seg.sub,xOff+sw/2,barY+barH+16,seg.col+'80',8,'400');
    } else {
      tx(ctx,seg.pct+'%',xOff+sw/2,barY+barH/2,seg.col,9,'800');
    }
    xOff+=sw;
  });

  // Arrow annotations
  tx(ctx,'Optimiser touches this',barX+barW*cfg.train/200,barY-22,'#4ecdc4',8,'700');
  tx(ctx,'You touch this (researcher)',barX+barW*(cfg.train+cfg.val/2)/100,barY-10,'#fbbf24',8,'700');
  tx(ctx,'Nobody touches this\u2014until the very end',barX+barW*(cfg.train+cfg.val+cfg.test/2)/100,barY-22,'#c084fc',8,'700');

  // "Lock" icon for test set
  tx(ctx,'\uD83D\uDD12',barX+barW*(cfg.train+cfg.val+cfg.test/2)/100,barY+barH+35,'#c084fc',14,'400');
  tx(ctx,cfg.note,W/2,H-12,'#52525b',8,'400');

  document.getElementById('iSplit').innerHTML=
    '<span class="hl">Rule:</span> any data that influences ANY decision is no longer an unbiased test set.<br>'+
    'Train='+cfg.train+'% &bull; Val='+cfg.val+'% &bull; Test='+cfg.test+'%<br>'+
    'Dataset: <span class="hl3">'+cfg.label+'</span>. '+cfg.note;
}

function renderKFold(){
  var K=parseInt(document.getElementById('slK').value);
  document.getElementById('vK').textContent=K;
  var ctx=clr('cvKFold',420,240),W=420,H=240;
  var rowH=Math.min(26,(H-60)/K);
  var barW=W-80,barX=60;

  for(var k=0;k<K;k++){
    var y=30+k*rowH;
    tx(ctx,'Fold '+(k+1),30,y+rowH/2,'#52525b',8,'400','right');
    var segW=barW/K;
    for(var j=0;j<K;j++){
      var isVal=j===k;
      bx(ctx,barX+j*segW,y,segW-2,rowH-3,3,
        isVal?'#ff6b3540':'#4ecdc420',
        isVal?'#ff6b35':'#4ecdc460',1.5);
      if(segW>24) tx(ctx,isVal?'VAL':'TR',barX+j*segW+segW/2,y+rowH/2-1.5,
        isVal?'#ff6b35':'#4ecdc460',7,'700');
    }
  }
  // Arrow & score
  var arX=barX+barW+8;
  for(var k2=0;k2<K;k2++){
    var y2=30+k2*rowH+rowH/2;
    tx(ctx,'\u2192 score '+(k2+1),arX,y2,'#52525b',8,'400','left');
  }
  tx(ctx,'Final: mean \u00B1 std of '+K+' scores',W/2,H-14,'#fbbf24',9,'700');

  var cvStr=K===3?'3-fold: fast, slightly higher variance':K<=5?K+'-fold: standard for deep learning':K<=8?K+'-fold: stable estimate, 8\u00D7 training runs':'10-fold: gold standard, expensive';
  document.getElementById('iKFold').innerHTML=
    '<span class="hl3">'+K+'-fold CV</span>: '+K+' training runs, each using '+Math.round((1-1/K)*100)+'% as train, '+Math.round(100/K)+'% as val.<br>'+
    cvStr+'<br><span class="hl5">CRITICAL:</span> test set is NEVER part of the CV loop. CV selects the model. Test evaluates it.';
}

// ============================================================
// TAB 1: TRAINING LOOP + PATTERNS
// ============================================================
function renderLoop(){
  var ctx=clr('cvLoop',420,300),W=420,H=300;
  var blocks=[
    {y:18, h:28, col:'#4ecdc4', label:'model.train()',      sub:'Dropout ON, BN batch stats'},
    {y:52, h:28, col:'#4ecdc4', label:'for batch in train_loader', sub:'Compute loss, backward, step'},
    {y:90, h:28, col:'#2d2d40', label:'train_loss = accumulate / n', sub:'Weighted sum / total samples'},
    {y:128,h:28, col:'#c084fc', label:'model.eval()',       sub:'Dropout OFF, BN running stats'},
    {y:162,h:28, col:'#c084fc', label:'with torch.no_grad():', sub:'No gradient tape (faster, less mem)'},
    {y:196,h:28, col:'#c084fc', label:'  for batch in val_loader',  sub:'Compute loss only'},
    {y:230,h:28, col:'#2d2d40', label:'val_loss = accumulate / n',   sub:'Weighted by sample count!'},
    {y:264,h:28, col:'#fbbf24', label:'scheduler / checkpoint / early_stop', sub:'Decisions based on val_loss'},
  ];
  var bx2=16,bw2=W-32;
  blocks.forEach(function(b){
    bx(ctx,bx2,b.y,bw2,b.h,4,b.col+'18',b.col+'60',1.5);
    tx(ctx,b.label,bx2+12,b.y+10,b.col,9,'800','left');
    tx(ctx,b.sub,bx2+12,b.y+21,'#52525b',7,'400','left');
  });
  // Phase labels
  bx(ctx,bx2-1,16,4,74,0,'#4ecdc4',null,0);
  tx(ctx,'TRAIN',bx2-8,16+37,'#4ecdc430',8,'800','right');
  bx(ctx,bx2-1,126,4,130,0,'#c084fc',null,0);
  tx(ctx,'EVAL',bx2-8,126+65,'#c084fc30',8,'800','right');

  document.getElementById('iLoop').innerHTML=
    '<span class="hl5">Most common bugs:</span><br>'+
    '1. Forgetting <span class="hl4">model.eval()</span> \u2192 dropout on during validation, stochastic val loss<br>'+
    '2. No <span class="hl6">torch.no_grad()</span> \u2192 correct but slow + OOM risk on large models<br>'+
    '3. Naive batch average \u2192 <span class="hl5">weight batches by sample count</span>, not equally';
}

var curPat='good';
var PAT_DATA={
  good:{train:[1.8,1.4,1.1,0.85,0.65,0.52,0.43,0.37,0.33,0.30],val:[1.9,1.5,1.15,0.92,0.74,0.62,0.55,0.51,0.49,0.48],col:'#4ade80',label:'Good Fit',info:'<span class="hl4">Good convergence:</span> train and val loss both decrease and converge within a small gap. Val loss tracks train. No intervention needed.'},
  over:{train:[1.8,1.3,0.9,0.6,0.38,0.22,0.12,0.06,0.03,0.01],val:[1.9,1.45,1.1,0.9,0.85,0.88,0.95,1.05,1.18,1.32],col:'#ff6b35',label:'Overfitting',info:'<span class="hl5">Overfitting:</span> train loss falls but val loss diverges upward after epoch 4. <span class="hl">Fix:</span> add regularisation (dropout, L2), early stopping, more data, reduce model size.'},
  under:{train:[1.8,1.65,1.55,1.48,1.43,1.40,1.38,1.37,1.36,1.35],val:[1.9,1.75,1.65,1.60,1.57,1.55,1.54,1.53,1.53,1.52],col:'#4ecdc4',label:'Underfitting',info:'<span class="hl5">Underfitting:</span> both losses are high and plateau early. Model lacks capacity or training is too short. <span class="hl">Fix:</span> increase capacity, reduce regularisation, train longer, check learning rate.'},
  spike:{train:[1.8,1.3,0.9,0.6,0.45,4.2,0.55,0.42,0.35,0.30],val:[1.9,1.4,1.0,0.78,0.62,4.5,0.72,0.58,0.50,0.47],col:'#fbbf24',label:'Loss Spike',info:'<span class="hl5">Loss spike:</span> sudden jump at epoch 5 (corrupted batch, gradient explosion, or a cliff in the loss surface). <span class="hl">Fix:</span> gradient clipping (clip_norm=1.0), lower learning rate, check for corrupted data.'},
  div:{train:[1.8,2.2,3.1,4.5,6.2,8.0,10.2,12.8,15.5,18.5],val:[1.9,2.4,3.4,5.0,6.8,8.9,11.2,14.0,17.0,20.5],col:'#ef4444',label:'Divergence',info:'<span class="hl5">Divergence:</span> both losses increase monotonically. Learning rate far too large, or wrong loss function / label mismatch. <span class="hl">Fix:</span> reduce lr by 10\u00D7. Check loss function matches task. Verify label and output shapes.'},
};
function selPat(k){curPat=k;['good','over','under','spike','div'].forEach(function(b){document.getElementById('pBtn_'+b).classList.remove('active');});document.getElementById('pBtn_'+k).classList.add('active');renderPat();}
function renderPat(){
  var d=PAT_DATA[curPat];
  var ctx=clr('cvPatterns',420,300),W=420,H=300,PAD=44;
  var pW=W-PAD-20,pH=H-PAD-30;
  var allV=d.train.concat(d.val).filter(function(v){return v<30;});
  var maxL=Math.min(Math.max.apply(null,allV)*1.1,22);
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1); ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Loss',PAD+20,20,'#52525b',8,'400'); tx(ctx,'Epoch',W/2,H-5,'#52525b',8,'400');
  var N=d.train.length-1;
  [d.train,d.val].forEach(function(arr,ai){
    var col=ai===0?d.col+'90':d.col;
    var pts=arr.map(function(v,i){return[PAD+(i/N)*pW,H-PAD-Math.min(v/maxL,1)*pH];});
    cv2(pts,ctx,col,ai===0?2:2.5);
    if(ai===0)tx(ctx,'Train',pts[N][0]-20,pts[N][1]-10,col,8,'700');
    else tx(ctx,'Val',pts[N][0]+5,pts[N][1],col,8,'700','left');
  });
  tx(ctx,d.label,W/2,22,d.col,10,'800');
  document.getElementById('iPat').innerHTML=d.info;
}

// ============================================================
// TAB 2: CLASSIFICATION METRICS
// ============================================================
// Simulated scores for 100 examples: 30 positive, 70 negative
var SCORES=(function(){
  var s=[];var seed=42;
  function rng(){seed=(seed*1664525+1013904223)&0xffffffff;return((seed>>>0)/4294967296);}
  for(var i=0;i<30;i++) s.push({y:1,p:0.4+rng()*0.6});  // positives
  for(var i=0;i<70;i++) s.push({y:0,p:rng()*0.55});      // negatives
  return s;
})();

function getConfusion(thr){
  var TP=0,TN=0,FP=0,FN=0;
  SCORES.forEach(function(s){var pred=s.p>=thr?1:0;if(pred===1&&s.y===1)TP++;else if(pred===0&&s.y===0)TN++;else if(pred===1&&s.y===0)FP++;else FN++;});
  return{TP:TP,TN:TN,FP:FP,FN:FN};
}

function renderCM(){
  var thr=parseFloat(document.getElementById('slThr').value);
  document.getElementById('vThr').textContent=rnd(thr,2);
  var c=getConfusion(thr);
  var ctx=clr('cvCM',420,260),W=420,H=260;
  var cSize=80,startX=90,startY=60;
  var cells=[
    {r:0,c:0,v:c.TP,label:'TP',col:'#4ade80',sub:'True Positive'},
    {r:0,c:1,v:c.FN,label:'FN',col:'#ef4444',sub:'False Negative'},
    {r:1,c:0,v:c.FP,label:'FP',col:'#fbbf24',sub:'False Positive'},
    {r:1,c:1,v:c.TN,label:'TN',col:'#4ecdc4',sub:'True Negative'},
  ];
  cells.forEach(function(cell){
    var x=startX+cell.c*(cSize+4),y=startY+cell.r*(cSize+4);
    bx(ctx,x,y,cSize,cSize,6,cell.col+'20',cell.col,2);
    tx(ctx,cell.v,x+cSize/2,y+cSize/2-8,cell.col,22,'800');
    tx(ctx,cell.label,x+cSize/2,y+cSize/2+14,cell.col,9,'700');
    tx(ctx,cell.sub,x+cSize/2,y+cSize/2+24,'#52525b',7,'400');
  });
  tx(ctx,'Predicted Positive',startX+cSize/2,startY-18,'#e4e4e7',9,'700');
  tx(ctx,'Predicted Negative',startX+cSize+cSize/2+4,startY-18,'#e4e4e7',9,'700');
  tx(ctx,'Actual\nPositive',startX-12,startY+cSize/2,'#e4e4e7',8,'700','right');
  tx(ctx,'Actual\nNegative',startX-12,startY+cSize+cSize/2+4,'#e4e4e7',8,'700','right');
  // Metrics
  var P=c.TP/(c.TP+c.FP+0.001),R=c.TP/(c.TP+c.FN+0.001),F1=2*P*R/(P+R+0.001);
  var acc=(c.TP+c.TN)/100;
  var metrics=[
    ['Accuracy',rnd(acc,2),'#94a3b8'],['Precision',rnd(P,2),'#4ecdc4'],
    ['Recall',rnd(R,2),'#ff6b35'],['F1',rnd(F1,2),'#fbbf24'],
  ];
  metrics.forEach(function(m,i){
    var mx=startX+cSize*2+4+20+(i%2)*100,my=startY+(Math.floor(i/2))*46+30;
    tx(ctx,m[0],mx,my,'#52525b',8,'400');
    tx(ctx,m[1],mx,my+16,m[2],14,'800');
  });
  document.getElementById('iCM').innerHTML=
    'Threshold=<span class="hl3">'+rnd(thr,2)+'</span> &bull; TP=<span class="hl4">'+c.TP+'</span> TN=<span class="hl2">'+c.TN+'</span> FP=<span class="hl3">'+c.FP+'</span> FN=<span class="hl5">'+c.FN+'</span><br>'+
    'Precision=<span class="hl2">'+rnd(P,2)+'</span> &bull; Recall=<span class="hl">'+rnd(R,2)+'</span> &bull; F1=<span class="hl3">'+rnd(F1,2)+'</span><br>'+
    (thr<0.3?'<span class="hl5">Low threshold:</span> high recall, many FP \u2192 lower precision.':thr>0.7?'<span class="hl5">High threshold:</span> high precision, many FN \u2192 lower recall.':'<span class="hl4">Balanced threshold.</span> Adjust based on your domain\'s cost of FP vs FN.');
}

var curROC='roc';
function selROC(k){curROC=k;['roc','pr'].forEach(function(b){document.getElementById('rocBtn_'+b).classList.remove('active');});document.getElementById('rocBtn_'+k).classList.add('active');renderROC();}
function renderROC(){
  var ctx=clr('cvROC',420,260),W=420,H=260,PAD=44,pW=W-PAD-20,pH=H-PAD-20;
  ln(ctx,PAD,H-PAD,W-10,H-PAD,'#2d2d40',1); ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);

  var thrs=[]; for(var t=0;t<=100;t++) thrs.push(t/100);
  if(curROC==='roc'){
    // ROC: FPR vs TPR
    ln(ctx,PAD,H-PAD,W-10,15,'#2d2d4060',1,[4,4]); // diagonal
    var pts=thrs.map(function(t){
      var thr=1-t/100; var c=getConfusion(thr);
      var fpr=(c.FP/(c.FP+c.TN+0.001)); var tpr=(c.TP/(c.TP+c.FN+0.001));
      return[PAD+fpr*pW,H-PAD-tpr*pH];
    });
    ctx.fillStyle='#4ecdc410'; ctx.beginPath();
    ctx.moveTo(PAD,H-PAD); pts.forEach(function(p){ctx.lineTo(p[0],p[1]);}); ctx.lineTo(W-10,H-PAD); ctx.closePath(); ctx.fill();
    cv2(pts,ctx,'#4ecdc4',2.5);
    // Calc AUC
    var auc=0; for(var i=1;i<pts.length;i++){var dx=(pts[i][0]-pts[i-1][0])/pW;var y1=(H-PAD-pts[i-1][1])/pH;var y2=(H-PAD-pts[i][1])/pH;auc+=dx*(y1+y2)/2;}
    tx(ctx,'AUC='+rnd(auc,2),PAD+pW*0.65,PAD+pH*0.35,'#4ecdc4',11,'800');
    tx(ctx,'FPR (False Positive Rate)',PAD+pW/2,H-5,'#52525b',8,'400');
    tx(ctx,'TPR (Recall)',PAD+16,22,'#52525b',8,'400');
    tx(ctx,'Random (AUC=0.5)',PAD+pW*0.45,H-PAD-pH*0.52,'#2d2d40',8,'400');
    document.getElementById('iROC').innerHTML=
      '<span class="hl2">ROC-AUC = '+rnd(auc,2)+'</span> (30 positives / 70 negatives)<br>'+
      'AUC = probability a random positive ranks higher than a random negative.<br>'+
      '<span class="hl">AUC=1.0:</span> perfect &bull; <span class="hl3">AUC=0.5:</span> random<br>'+
      '<span class="hl5">Limitation:</span> TN inflates AUC for rare positives. Use PR-AUC when positive rate &lt;10%.';
  } else {
    // PR Curve
    var prPts=thrs.map(function(t){
      var thr=1-t/100; var c=getConfusion(thr);
      var prec=c.TP/(c.TP+c.FP+0.001); var rec=c.TP/(c.TP+c.FN+0.001);
      return[PAD+rec*pW,H-PAD-prec*pH];
    });
    // Baseline
    ln(ctx,PAD,H-PAD-(30/100)*pH,W-10,H-PAD-(30/100)*pH,'#2d2d4060',1,[4,4]);
    tx(ctx,'Baseline (random) = 0.30',W-14,H-PAD-(30/100)*pH-8,'#2d2d40',7,'400','right');
    ctx.fillStyle='#ff6b3510'; ctx.beginPath();
    ctx.moveTo(PAD,H-PAD); prPts.forEach(function(p){ctx.lineTo(p[0],p[1]);}); ctx.lineTo(W-10,H-PAD); ctx.closePath(); ctx.fill();
    cv2(prPts,ctx,'#ff6b35',2.5);
    var prauc=0; for(var i=1;i<prPts.length;i++){var dx=(prPts[i][0]-prPts[i-1][0])/pW;var y1=(H-PAD-prPts[i-1][1])/pH;var y2=(H-PAD-prPts[i][1])/pH;prauc+=Math.abs(dx)*(y1+y2)/2;}
    tx(ctx,'PR-AUC='+rnd(prauc,2),PAD+pW*0.5,PAD+pH*0.4,'#ff6b35',11,'800');
    tx(ctx,'Recall',PAD+pW/2,H-5,'#52525b',8,'400');
    tx(ctx,'Precision',PAD+16,22,'#52525b',8,'400');
    document.getElementById('iROC').innerHTML=
      '<span class="hl">PR-AUC = '+rnd(prauc,2)+'</span> vs random baseline = 0.30 (positive rate)<br>'+
      'PR curve does NOT include TN \u2014 only TP, FP, FN matter.<br>'+
      '<span class="hl4">Use PR-AUC when positive rate &lt;10%.</span> ROC-AUC is misleadingly optimistic on rare events because the large TN pool inflates the FPR denominator.';
  }
}

// ============================================================
// TAB 3: CALIBRATION
// ============================================================
var curCal='over';
var CAL_DATA={
  over: {bins:[0.07,0.14,0.22,0.30,0.45,0.58,0.69,0.76,0.87,0.92],col:'#ff6b35',label:'Overconfident (typical neural net)',
    info:'<span class="hl5">Overconfident:</span> model says 90% but only 76% are actually positive. ECE \u2248 0.12.<br>Neural nets trained with cross-entropy tend to be overconfident. <span class="hl">Fix:</span> temperature scaling (T&gt;1) or label smoothing.'},
  under:{bins:[0.12,0.22,0.35,0.44,0.52,0.58,0.63,0.70,0.74,0.78],col:'#4ecdc4',label:'Underconfident',
    info:'<span class="hl5">Underconfident:</span> model says 50% but 70% are positive. ECE \u2248 0.10.<br>Less common. Seen with heavy label smoothing. <span class="hl">Fix:</span> temperature scaling (T&lt;1).'},
  good: {bins:[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],col:'#4ade80',label:'Well-calibrated',
    info:'<span class="hl4">Well-calibrated:</span> predicted probabilities match observed frequencies. ECE &lt;0.03.<br>Points lie close to the diagonal. Probabilities are trustworthy for decision-making.'},
};
function selCal(k){curCal=k;['over','under','good'].forEach(function(b){document.getElementById('calBtn_'+b).classList.remove('active');});document.getElementById('calBtn_'+k).classList.add('active');renderCal();}
function renderCal(){
  var d=CAL_DATA[curCal];
  var ctx=clr('cvCal',420,270),W=420,H=270,PAD=44,pW=W-PAD-20,pH=H-PAD-20;
  // Perfect diagonal
  ln(ctx,PAD,H-PAD,W-20,15,'#4ade8040',1.5,[5,4]);
  tx(ctx,'Perfect calibration',PAD+pW*0.55,H-PAD-pH*0.6,'#4ade8040',8,'700');
  // Axes
  ln(ctx,PAD,H-PAD,W-20,H-PAD,'#2d2d40',1); ln(ctx,PAD,15,PAD,H-PAD,'#2d2d40',1);
  tx(ctx,'Predicted probability',PAD+pW/2,H-5,'#52525b',8,'400');
  tx(ctx,'Actual positive rate',PAD+16,22,'#52525b',8,'400');
  [0,0.2,0.4,0.6,0.8,1.0].forEach(function(v){
    tx(ctx,rnd(v,1),PAD-6,H-PAD-v*pH,'#3f3f46',7,'400','right');
    tx(ctx,rnd(v,1),PAD+v*pW,H-PAD+12,'#3f3f46',7,'400');
  });
  // Calibration points
  var binW=pW/10;
  var ece=0;
  d.bins.forEach(function(actual,i){
    var pred=(i+0.5)/10;
    var px=PAD+pred*pW, py=H-PAD-actual*pH;
    // bar
    bx(ctx,PAD+i*binW+2,H-PAD-actual*pH,binW-4,actual*pH,0,d.col+'20',d.col+'60',1);
    // point
    ctx.fillStyle=d.col; ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
    // error line to diagonal
    var diagY=H-PAD-pred*pH;
    ln(ctx,px,py,px,diagY,d.col+'50',1.5,[3,3]);
    ece+=Math.abs(actual-pred)*0.1;
  });
  // Connect dots
  ctx.strokeStyle=d.col; ctx.lineWidth=2; ctx.beginPath();
  d.bins.forEach(function(actual,i){var pred=(i+0.5)/10;var px=PAD+pred*pW,py=H-PAD-actual*pH;if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);});
  ctx.stroke();
  tx(ctx,d.label,W/2,22,d.col,9,'800');
  tx(ctx,'ECE \u2248 '+rnd(ece,3),W-30,H-PAD-pH*0.15,d.col,10,'800','right');
  document.getElementById('iCal').innerHTML=d.info+'<br>ECE \u2248 <span class="hl3">'+rnd(ece,3)+'</span>';
}

function renderTemp(){
  var T=parseFloat(document.getElementById('slT').value);
  document.getElementById('vT').textContent=rnd(T,1);
  // Show softmax distribution change at various temperatures
  var logits=[3.1,-0.2,0.8,-1.2,0.5];
  function sm(ls,temp){var m=Math.max.apply(null,ls);var exps=ls.map(function(z){return Math.exp((z-m)/temp);});var s=exps.reduce(function(a,b){return a+b;},0);return exps.map(function(e){return e/s;});}
  var ps=sm(logits,T), p1=sm(logits,1);
  var ctx=clr('cvTemp',420,270),W=420,H=270;
  var bw2=44,gap=10,N2=5,startX=(W-(N2*(bw2+gap)))/2;
  var colors=['#ff6b35','#4ecdc4','#fbbf24','#c084fc','#4ade80'];
  var labels=['Cat','Dog','Bird','Fish','Car'];
  var maxH=H-100;
  // Baseline (T=1) ghosted
  p1.forEach(function(p,i){
    var x=startX+i*(bw2+gap), h=p*maxH;
    bx(ctx,x,H-50-h,bw2,h,3,colors[i]+'15',colors[i]+'40',1);
  });
  // Current T bars
  ps.forEach(function(p,i){
    var x=startX+i*(bw2+gap)+4, h=p*maxH;
    bx(ctx,x,H-50-h,bw2-4,h,3,colors[i]+'40',colors[i],2);
    tx(ctx,rnd(p,2),x+(bw2-4)/2,H-50-h-10,colors[i],8,'700');
    tx(ctx,labels[i],x+(bw2-4)/2,H-34,'#52525b',7,'400');
  });
  // Legend
  bx(ctx,20,H-60,14,14,2,colors[0]+'15',colors[0]+'40',1);
  tx(ctx,'T=1 (ghost)',40,H-53,'#52525b',7,'400','left');
  bx(ctx,20,H-44,14,14,2,colors[0]+'40',colors[0],2);
  tx(ctx,'T='+rnd(T,1),40,H-37,colors[0],7,'800','left');
  tx(ctx,'Temperature T='+rnd(T,1)+': Softmax Distribution',W/2,22,'#e4e4e7',10,'800');
  var entropy=-ps.reduce(function(s,p){return s+(p>1e-9?p*Math.log(p):0);},0);
  tx(ctx,'Entropy: '+rnd(entropy,2)+' nats',W-14,22,'#fbbf24',8,'700','right');
  document.getElementById('iTemp').innerHTML=
    'T=<span class="hl3">'+rnd(T,1)+'</span> &rarr; distribution entropy: <span class="hl3">'+rnd(entropy,2)+'</span><br>'+
    (T<1?'<span class="hl5">T&lt;1:</span> sharpens distribution. More confident (rarely needed).' :
     T>1?'<span class="hl2">T&gt;1:</span> softens distribution. Reduces overconfidence. ECE improves.':'<span class="hl3">T=1:</span> original output. No change.')+
    '<br><span class="hl4">Key:</span> temperature scaling does NOT change accuracy (argmax unchanged). Only probability magnitudes shift.';
}

// ============================================================
// TAB 4: METRICS GUIDE
// ============================================================
function renderMG(){
  var ctx=clr('cvMG',860,220),W=860,H=220;
  var tasks=[
    {label:'Binary Clf\n(balanced)',metric:'ROC-AUC',sec:'F1, Precision, Recall',col:'#4ecdc4'},
    {label:'Binary Clf\n(imbalanced)',metric:'PR-AUC',sec:'F1 at threshold',col:'#ff6b35'},
    {label:'Multi-class\n(balanced)',metric:'Macro F1',sec:'Per-class F1',col:'#c084fc'},
    {label:'Multi-class\n(imbalanced)',metric:'Macro F1 +\nper-class',sec:'Confusion matrix',col:'#fbbf24'},
    {label:'Regression\n(general)',metric:'RMSE',sec:'MAE, R\u00B2',col:'#4ade80'},
    {label:'Language\nModelling',metric:'Perplexity',sec:'Val cross-entropy',col:'#38bdf8'},
    {label:'Translation /\nGeneration',metric:'BLEU / ROUGE',sec:'BERTScore',col:'#f472b6'},
    {label:'Object\nDetection',metric:'mAP',sec:'AP per class',col:'#fb923c'},
  ];
  var bw3=(W-20)/tasks.length-4,startX=12;
  tasks.forEach(function(t,i){
    var x=startX+i*(bw3+4);
    bx(ctx,x,8,bw3,H-16,6,t.col+'15',t.col+'50',1.5);
    t.label.split('\n').forEach(function(l,li){tx(ctx,l,x+bw3/2,28+li*13,t.col,8,'800');});
    tx(ctx,'\u2192',x+bw3/2,65,'#52525b',10,'400');
    tx(ctx,'PRIMARY',x+bw3/2,80,'#3f3f46',7,'700');
    t.metric.split('\n').forEach(function(l,li){tx(ctx,l,x+bw3/2,95+li*14,t.col,9,'800');});
    tx(ctx,'Also check:',x+bw3/2,130,'#3f3f46',7,'400');
    tx(ctx,t.sec,x+bw3/2,145,'#52525b',7,'400');
    if(i>0) ln(ctx,x-2,10,x-2,H-10,'#1e1e2e',1);
  });
}

function renderReg(){
  // Scatter plot + compare MSE vs MAE vs R2 sensitivity to outlier
  var ctx=clr('cvReg',420,220),W=420,H=220,PAD=40;
  var trueVals=[1,2,3,4,5,6,7,8,9,10];
  var preds=   [1.2,2.1,2.8,3.9,5.2,5.8,7.2,8.1,8.9,10.1];
  var predsOut=[1.2,2.1,2.8,3.9,5.2,5.8,7.2,8.1,8.9,4.0]; // last point is outlier

  function mse(y,yh){return y.reduce(function(s,v,i){return s+Math.pow(v-yh[i],2);},0)/y.length;}
  function mae(y,yh){return y.reduce(function(s,v,i){return s+Math.abs(v-yh[i]);},0)/y.length;}
  function r2(y,yh){var mn=y.reduce(function(a,b){return a+b;},0)/y.length;var ss_tot=y.reduce(function(s,v){return s+Math.pow(v-mn,2);},0);var ss_res=y.reduce(function(s,v,i){return s+Math.pow(v-yh[i],2);},0);return 1-ss_res/ss_tot;}

  var metrics=[
    ['RMSE',Math.sqrt(mse(trueVals,preds)),Math.sqrt(mse(trueVals,predsOut)),'#4ecdc4'],
    ['MAE', mae(trueVals,preds),            mae(trueVals,predsOut),           '#ff6b35'],
    ['R\u00B2',  r2(trueVals,preds),             r2(trueVals,predsOut),            '#c084fc'],
  ];
  var bw4=60,gp=20,sx=(W-(metrics.length*(bw4*2+gp+20)))/2;
  var maxV=6.5;
  metrics.forEach(function(m,i){
    var x=sx+i*(bw4*2+gp+20);
    var h1=Math.abs(m[1])/maxV*(H-80);
    var h2=Math.abs(m[2])/maxV*(H-80);
    // Normal bar
    bx(ctx,x,H-50-h1,bw4,h1,3,m[3]+'30',m[3],1.5);
    tx(ctx,rnd(m[1],2),x+bw4/2,H-50-h1-10,m[3],8,'700');
    tx(ctx,'normal',x+bw4/2,H-35,'#52525b',7,'400');
    // Outlier bar
    bx(ctx,x+bw4+8,H-50-h2,bw4,h2,3,'#ef444430','#ef4444',1.5);
    tx(ctx,rnd(m[2],2),x+bw4+8+bw4/2,H-50-h2-10,'#ef4444',8,'700');
    tx(ctx,'outlier',x+bw4+8+bw4/2,H-35,'#ef444480',7,'400');
    tx(ctx,m[0],x+bw4+(gp/2),H-20,m[3],9,'800');
  });
  tx(ctx,'Effect of one outlier on regression metrics',W/2,16,'#e4e4e7',9,'800');
  document.getElementById('iReg').innerHTML=
    'One outlier (y=10, pred=4) changes:<br>'+
    '<span class="hl2">RMSE:</span> '+rnd(Math.sqrt(mse(trueVals,preds)),2)+' \u2192 <span class="hl5">'+rnd(Math.sqrt(mse(trueVals,predsOut)),2)+'</span> (+'+rnd(Math.sqrt(mse(trueVals,predsOut))-Math.sqrt(mse(trueVals,preds)),2)+')<br>'+
    '<span class="hl">MAE:</span> '+rnd(mae(trueVals,preds),2)+' \u2192 <span class="hl3">'+rnd(mae(trueVals,predsOut),2)+'</span> (less sensitive)<br>'+
    '<span class="hl6">R\u00B2:</span> '+rnd(r2(trueVals,preds),2)+' \u2192 <span class="hl5">'+rnd(r2(trueVals,predsOut),2)+'</span><br>'+
    'RMSE is most sensitive to outliers (squared error). MAE is robust.';
}

function renderAvg(){
  var ctx=clr('cvAvg',420,220),W=420,H=220;
  // 3-class: A=800, B=150, C=50  F1: A=0.92, B=0.71, C=0.35
  var classes=[{name:'A',n:800,f1:0.92,col:'#4ecdc4'},{name:'B',n:150,f1:0.71,col:'#fbbf24'},{name:'C',n:50,f1:0.35,col:'#ef4444'}];
  var N3=1000;
  var macro=classes.reduce(function(s,c){return s+c.f1;},0)/classes.length;
  var weighted=classes.reduce(function(s,c){return s+c.n/N3*c.f1;},0);
  var micro=0.88; // approximate
  var barH=H-100, bw5=40;
  var xs=[55,140,225];
  classes.forEach(function(cl,i){
    bx(ctx,xs[i],H-50-cl.f1*barH,bw5,cl.f1*barH,3,cl.col+'30',cl.col,2);
    tx(ctx,'Class '+cl.name,xs[i]+bw5/2,H-50-cl.f1*barH-12,cl.col,9,'800');
    tx(ctx,'F1='+rnd(cl.f1,2),xs[i]+bw5/2,H-50-cl.f1*barH-1,cl.col,8,'700');
    tx(ctx,'n='+cl.n,xs[i]+bw5/2,H-34,'#52525b',8,'400');
  });
  // Aggregates
  var aggData=[
    ['Macro',macro,'#c084fc','(0.92+0.71+0.35)/3'],
    ['Weighted',weighted,'#4ade80','n-weighted avg'],
    ['Micro',micro,'#38bdf8','pool all TP/FP/FN'],
  ];
  aggData.forEach(function(a,i){
    var ax=300+i*55;
    bx(ctx,ax,H-50-a[1]*barH,40,a[1]*barH,3,a[2]+'30',a[2],2);
    tx(ctx,a[0],ax+20,H-50-a[1]*barH-12,a[2],8,'800');
    tx(ctx,rnd(a[1],2),ax+20,H-50-a[1]*barH-1,a[2],8,'700');
    tx(ctx,a[3],ax+20,H-34,'#52525b',6,'400');
  });
  tx(ctx,'Imbalanced 3-class: A=800, B=150, C=50',W/2,16,'#e4e4e7',9,'800');
  ln(ctx,280,20,280,H-20,'#1e1e2e',1);
  tx(ctx,'Per-class',130,22,'#52525b',8,'700');
  tx(ctx,'Aggregates',360,22,'#52525b',8,'700');
  document.getElementById('iAvg').innerHTML=
    '<span class="hl6">Macro F1='+rnd(macro,3)+':</span> treats all classes equally. Pulled down by poor C performance.<br>'+
    '<span class="hl4">Weighted F1='+rnd(weighted,3)+':</span> class B and C barely matter (few examples).<br>'+
    '<span class="hl2">Micro F1\u2248'+rnd(micro,2)+':</span> dominated by class A (800 examples).<br>'+
    'Imbalanced data: use <span class="hl6">Macro F1</span> + per-class breakdown. Never just accuracy.';
}

// ============================================================
// TAB 5: DIAGNOSIS
// ============================================================
function renderGap(){
  var tl=parseFloat(document.getElementById('slTL').value);
  var vl=parseFloat(document.getElementById('slVL').value);
  document.getElementById('vTL').textContent=rnd(tl,2);
  document.getElementById('vVL').textContent=rnd(vl,2);
  var gap=vl-tl;
  var ctx=clr('cvGap',420,270),W=420,H=270;
  var cx=W/2, cy=H/2;
  // Gauge-style display
  var maxLoss=2.0;
  var trainPct=tl/maxLoss, valPct=vl/maxLoss;
  var barW=W-80, barH2=36;
  // Train bar
  bx(ctx,40,cy-50,barW*trainPct,barH2,4,'#4ecdc440','#4ecdc4',2);
  bx(ctx,40,cy-50,barW,barH2,4,null,'#1e1e2e',1);
  tx(ctx,'Train loss: '+rnd(tl,2),40+barW/2,cy-50+barH2/2,'#4ecdc4',10,'800');
  // Val bar
  bx(ctx,40,cy+10,barW*valPct,barH2,4,'#ff6b3540','#ff6b35',2);
  bx(ctx,40,cy+10,barW,barH2,4,null,'#1e1e2e',1);
  tx(ctx,'Val loss: '+rnd(vl,2),40+barW/2,cy+10+barH2/2,'#ff6b35',10,'800');
  // Gap display
  var gapColor=gap>0.5?'#ef4444':gap>0.15?'#fbbf24':gap<-0.05?'#c084fc':'#4ade80';
  var gapLabel=gap>0.5?'OVERFITTING':gap>0.15?'Mild overfit':gap<-0.05?'Check splits!':'Good fit';
  bx(ctx,40,cy+65,barW,36,6,gapColor+'15',gapColor,2);
  tx(ctx,'Gap: '+rnd(gap,2)+' \u2192 '+gapLabel,cx,cy+65+18,gapColor,11,'800');
  // Diagnosis
  var diag=gap>0.5?'<span class="hl5">Significant overfitting:</span> add regularisation, more data, reduce capacity, early stopping.':
           gap>0.15?'<span class="hl3">Mild overfitting:</span> acceptable in many settings. Monitor closely.':
           gap<-0.1?'<span class="hl6">Val better than train:</span> check dropout in train mode, augmentation, or distribution shift.':
           tl>1.2&&vl>1.2?'<span class="hl5">Underfitting:</span> both losses high. Increase capacity, reduce regularisation, train longer.':
           '<span class="hl4">Good fit:</span> val tracks train closely. Model is generalising well.';
  var fix=gap>0.5?'Fix: L2/dropout, early stopping, reduce model, get more data':
          gap>0.15?'Fix: slightly increase regularisation, monitor for further divergence':
          gap<-0.1?'Fix: check model.eval(), check train/val split for distribution mismatch':
          tl>1.2?'Fix: increase capacity, reduce regularisation, lower lr, more epochs':
          'No action needed \u2014 consider deploying!';
  document.getElementById('iGap').innerHTML=diag+'<br><span class="hl2">'+fix+'</span>';
}

function renderMistakes(){
  var ctx=clr('cvMistakes',420,270),W=420,H=270;
  var mistakes=[
    {title:'Report final not best',col:'#ef4444',y:30,
     desc:'Final epoch \u2260 best checkpoint. Report best val metric.'},
    {title:'Eval on early-stop val set',col:'#fbbf24',y:80,
     desc:'Selected best by val_loss \u2192 slightly biased. Use test set.'},
    {title:'100 HP configs, one val',col:'#ff6b35',y:130,
     desc:'Implicitly overfit to val. Need nested CV or held-out test.'},
    {title:'Single seed comparison',col:'#c084fc',y:180,
     desc:'84.3% vs 84.5%: run 3-5 seeds, compare mean\u00B1std first.'},
    {title:'Preprocessing data leak',col:'#38bdf8',y:230,
     desc:'Fit scaler on all data BEFORE split \u2192 val/test leak into train.'},
  ];
  mistakes.forEach(function(m){
    bx(ctx,10,m.y-16,W-20,50,6,m.col+'12',m.col+'50',1.5);
    tx(ctx,m.title,20,m.y-4,m.col,9,'800','left');
    tx(ctx,m.desc,20,m.y+11,'#71717a',8,'400','left');
  });
  document.getElementById('iMistakes').innerHTML=
    'These 5 mistakes silently inflate your reported performance.<br>'+
    '<span class="hl5">Most dangerous:</span> preprocessing leakage (hard to detect) and reporting final-epoch metrics (very common).';
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load',function(){
  renderSplit(); renderKFold();
});
</script>
</body>
</html>"""

EVALUATION_VISUAL_HEIGHT = 2200