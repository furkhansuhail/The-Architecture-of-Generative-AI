ADVTRAINING_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: #08080f; color: #e4e4e7; padding: 20px; min-height: 100vh; }
h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.1em; color: #ff6b35; margin-bottom: 3px; }
.subtitle { color: #52525b; font-size: 0.78em; margin-bottom: 18px; }
.tabs { display: flex; gap: 0; border-bottom: 2px solid #1e1e2e; margin-bottom: 18px; overflow-x: auto; }
.tab { padding: 10px 16px; background: none; border: none; border-bottom: 2px solid transparent; color: #71717a; cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 0.72em; font-weight: 700; white-space: nowrap; margin-bottom: -2px; transition: all 0.2s; }
.tab:hover { color: #e4e4e7; }
.tab.active { color: #ff6b35; border-bottom-color: #ff6b35; }
.panel { display: none; }
.panel.active { display: block; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.card { background: #111118; border: 1px solid #1e1e2e; border-radius: 14px; padding: 18px; margin-bottom: 14px; }
.card h3 { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: #ff6b35; margin-bottom: 10px; }
canvas { display: block; border-radius: 8px; background: #09090f; }
.row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.row label { font-family: 'JetBrains Mono', monospace; font-size: 0.73em; color: #71717a; min-width: 90px; }
input[type=range] { flex: 1; height: 4px; border-radius: 2px; -webkit-appearance: none; background: #1e1e2e; outline: none; cursor: pointer; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #ff6b35; cursor: pointer; }
.val { font-family: 'JetBrains Mono', monospace; font-size: 0.73em; color: #ff6b35; min-width: 44px; text-align: right; }
.info { background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px; padding: 10px 14px; font-size: 0.76em; color: #94a3b8; line-height: 1.8; margin-top: 10px; font-family: 'JetBrains Mono', monospace; }
.hl  { color: #ff6b35; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }
.hl4 { color: #4ade80; font-weight: 700; }
.hl5 { color: #ef4444; font-weight: 700; }
.hl6 { color: #c084fc; font-weight: 700; }
.btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
button.sel { background: #1e1e2e; color: #a1a1aa; border: 1px solid #2d2d40; border-radius: 6px; padding: 5px 12px; cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 0.73em; font-weight: 700; transition: all 0.15s; }
button.sel:hover { background: #2d2d40; color: #e4e4e7; }
button.sel.active { background: #ff6b35; color: #08080f; border-color: #ff6b35; }
.tbl { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.71em; }
.tbl th { padding: 6px 9px; text-align: left; color: #52525b; border-bottom: 2px solid #1e1e2e; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }
.tbl td { padding: 6px 9px; border-bottom: 1px solid #0f0f18; color: #94a3b8; vertical-align: middle; }
.tbl tr:hover td { background: #0d0d18; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 0.85em; font-weight: 700; }
.bg { background: #0a1520; color: #38bdf8; border: 1px solid #38bdf840; }
.bo { background: #1a110a; color: #fb923c; border: 1px solid #fb923c40; }
.br { background: #1a0808; color: #ef4444; border: 1px solid #ef444440; }
.bg2 { background: #0a1f0f; color: #4ade80; border: 1px solid #4ade8040; }
.bpu { background: #150a20; color: #c084fc; border: 1px solid #c084fc40; }
.pipeline { display: flex; align-items: stretch; gap: 0; margin: 12px 0; flex-wrap: wrap; }
.pstep { flex: 1; min-width: 80px; background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 0; padding: 10px 8px; text-align: center; font-family: 'JetBrains Mono', monospace; position: relative; }
.pstep:first-child { border-radius: 8px 0 0 8px; }
.pstep:last-child  { border-radius: 0 8px 8px 0; }
.pstep .pnum { font-size: 0.6em; color: #3f3f46; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.pstep .ptitle { font-size: 0.72em; font-weight: 700; margin-bottom: 3px; }
.pstep .pdesc { font-size: 0.62em; color: #52525b; line-height: 1.4; }
.parr { display: flex; align-items: center; padding: 0 2px; color: #3f3f46; font-size: 1.2em; flex-shrink: 0; }
</style>
</head>
<body>

<h2>&#9881; Advanced Training Techniques</h2>
<p class="subtitle">Label Smoothing &middot; Knowledge Distillation &middot; Quantisation &middot; Pruning &middot; Mixed Precision &middot; Production Pipeline</p>

<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Label Smoothing</button>
  <button class="tab" onclick="showTab(1)">Distillation</button>
  <button class="tab" onclick="showTab(2)">Quantisation</button>
  <button class="tab" onclick="showTab(3)">Pruning</button>
  <button class="tab" onclick="showTab(4)">Mixed Precision</button>
  <button class="tab" onclick="showTab(5)">Production Stack</button>
</div>

<!-- ============================================================
     TAB 0: LABEL SMOOTHING
     ============================================================ -->
<div id="tab0" class="panel active">
  <div class="grid2">
    <div class="card">
      <h3>&#9312; Hard vs Smooth Label Targets</h3>
      <canvas id="cvLS" width="420" height="240"></canvas>
      <div class="row"><label>Smoothing &alpha;</label>
        <input type="range" id="slAlpha" min="0" max="0.5" step="0.01" value="0.1" oninput="renderLS()">
        <span class="val" id="valAlpha">0.10</span></div>
      <div class="row"><label>Num classes K</label>
        <input type="range" id="slK" min="2" max="10" step="1" value="5" oninput="renderLS()">
        <span class="val" id="valK">5</span></div>
      <div class="info" id="infoLS">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Logit Magnitude vs Loss</h3>
      <canvas id="cvLogit" width="420" height="240"></canvas>
      <div class="row"><label>Smoothing &alpha;</label>
        <input type="range" id="slAlpha2" min="0" max="0.5" step="0.01" value="0.1" oninput="renderLogit()">
        <span class="val" id="valAlpha2">0.10</span></div>
      <div class="info" id="infoLogit">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; When to Use Label Smoothing</h3>
    <div class="grid2">
      <div>
        <div style="font-size:0.72em;font-weight:800;color:#4ade80;font-family:'JetBrains Mono',monospace;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">Use Label Smoothing</div>
        <table class="tbl">
          <thead><tr><th>Situation</th><th>Why it helps</th></tr></thead>
          <tbody>
            <tr><td>Multi-class classification</td><td>Prevents overconfident logits</td></tr>
            <tr><td>Poor calibration problem</td><td>Forces bounded predictions</td></tr>
            <tr><td>Noisy or ambiguous labels</td><td>Absorbs annotation errors</td></tr>
            <tr><td>Pre-training for transfer</td><td>Better intermediate representations</td></tr>
            <tr><td>&alpha; default</td><td>0.1 (ImageNet, Transformers)</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <div style="font-size:0.72em;font-weight:800;color:#ef4444;font-family:'JetBrains Mono',monospace;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">Do NOT Use</div>
        <table class="tbl">
          <thead><tr><th>Situation</th><th>Why it hurts</th></tr></thead>
          <tbody>
            <tr><td><span class="hl5">Teacher for distillation</span></td><td>Destroys dark knowledge (inter-class info)</td></tr>
            <tr><td>Object detection boxes</td><td>Targets already not one-hot</td></tr>
            <tr><td>Regression tasks</td><td>Meaningless for continuous outputs</td></tr>
            <tr><td>Binary classification</td><td>Effect minimal; use class weights</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ============================================================
     TAB 1: KNOWLEDGE DISTILLATION
     ============================================================ -->
<div id="tab1" class="panel">
  <div class="grid2">
    <div class="card">
      <h3>&#9312; Teacher Soft Targets &mdash; Temperature Effect</h3>
      <canvas id="cvDist" width="420" height="250"></canvas>
      <div class="row"><label>Temperature T</label>
        <input type="range" id="slTemp" min="0.5" max="12" step="0.5" value="1" oninput="renderDist()">
        <span class="val" id="valTemp">1.0</span></div>
      <div class="info" id="infoDist">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Student Loss Composition</h3>
      <canvas id="cvDistLoss" width="420" height="250"></canvas>
      <div class="row"><label>Hard weight &alpha;</label>
        <input type="range" id="slDistA" min="0" max="1" step="0.05" value="0.3" oninput="renderDistLoss()">
        <span class="val" id="valDistA">0.30</span></div>
      <div class="row"><label>Temperature T</label>
        <input type="range" id="slDistT" min="1" max="10" step="0.5" value="4" oninput="renderDistLoss()">
        <span class="val" id="valDistT">4.0</span></div>
      <div class="info" id="infoDistLoss">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Distillation Variants</h3>
    <div class="grid2">
      <table class="tbl">
        <thead><tr><th>Variant</th><th>What is matched</th><th>Best for</th></tr></thead>
        <tbody>
          <tr><td><span class="hl2">Response-based</span></td><td>Final output logits</td><td>General, simple</td></tr>
          <tr><td><span class="hl3">Feature-based</span></td><td>Intermediate activations</td><td>Deeper compression</td></tr>
          <tr><td><span class="hl6">Attention transfer</span></td><td>Attention maps</td><td>CNN/Transformer</td></tr>
          <tr><td><span class="hl4">Relation-based</span></td><td>Pairwise relationships</td><td>Embedding spaces</td></tr>
          <tr><td><span class="hl5">Online / DML</span></td><td>Mutual teaching</td><td>No pre-trained teacher</td></tr>
          <tr><td><span class="hl">Self-distillation</span></td><td>Previous generation</td><td>Born-again networks</td></tr>
        </tbody>
      </table>
      <div>
        <div class="info" style="height:100%;">
          <span class="hl">Dark Knowledge</span> &mdash; the information contained in a teacher's soft predictions beyond the correct class label.<br><br>
          Example: teacher predicts a "3" image as 99% class 3, but 0.9% class 8.<br>The soft target tells the student "3 resembles 8." Hard label tells nothing.<br><br>
          <span class="hl4">Key insight:</span> a student trained ONLY on soft targets (no hard labels) on MNIST achieves near-teacher accuracy. The soft distribution contains all training signal.<br><br>
          <span class="hl5">Gotcha:</span> if the teacher was trained with label smoothing, soft targets have LESS dark knowledge. Train the teacher WITHOUT label smoothing.
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ============================================================
     TAB 2: QUANTISATION
     ============================================================ -->
<div id="tab2" class="panel">
  <div class="grid2">
    <div class="card">
      <h3>&#9312; Number Format Bits</h3>
      <canvas id="cvFmt" width="420" height="220"></canvas>
      <div class="btn-row">
        <button class="sel active" id="fmtBtn_fp32"  onclick="selectFmt('fp32')">FP32</button>
        <button class="sel" id="fmtBtn_fp16"  onclick="selectFmt('fp16')">FP16</button>
        <button class="sel" id="fmtBtn_bf16"  onclick="selectFmt('bf16')">BF16</button>
        <button class="sel" id="fmtBtn_int8"  onclick="selectFmt('int8')">INT8</button>
        <button class="sel" id="fmtBtn_int4"  onclick="selectFmt('int4')">INT4</button>
      </div>
      <div class="info" id="infoFmt">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Memory Calculator: 7B Parameter Model</h3>
      <canvas id="cvMem" width="420" height="220"></canvas>
      <div class="btn-row">
        <button class="sel active" id="memBtn_fp32" onclick="selectMem('fp32')">FP32</button>
        <button class="sel" id="memBtn_fp16" onclick="selectMem('fp16')">FP16</button>
        <button class="sel" id="memBtn_bf16" onclick="selectMem('bf16')">BF16</button>
        <button class="sel" id="memBtn_int8" onclick="selectMem('int8')">INT8</button>
        <button class="sel" id="memBtn_int4" onclick="selectMem('int4')">INT4</button>
      </div>
      <div class="info" id="infoMem">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; PTQ vs QAT &mdash; Quantisation Workflow</h3>
    <div class="grid2">
      <div>
        <div style="font-size:0.72em;font-weight:800;color:#4ecdc4;font-family:'JetBrains Mono',monospace;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">Post-Training Quantisation (PTQ)</div>
        <div class="info" style="min-height:130px;">
          <span class="hl2">No retraining needed.</span> Run a calibration dataset through the model, collect activation statistics, then quantise.<br><br>
          <span class="hl4">Fast:</span> calibration takes minutes.<br>
          <span class="hl4">Works well:</span> INT8 with &lt;1% accuracy drop on most models.<br>
          <span class="hl5">Struggles:</span> INT4 can cause &gt;5% drop on sensitive models.<br><br>
          Methods: GPTQ, AWQ, SmoothQuant &mdash; all are PTQ variants optimised for LLMs.
        </div>
      </div>
      <div>
        <div style="font-size:0.72em;font-weight:800;color:#c084fc;font-family:'JetBrains Mono',monospace;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">Quantisation-Aware Training (QAT)</div>
        <div class="info" style="min-height:130px;">
          <span class="hl6">Simulates quantisation during training</span> using fake-quantise operations. The model learns to work within quantisation constraints.<br><br>
          <span class="hl4">Better accuracy:</span> often &lt;1% drop even at INT4.<br>
          <span class="hl5">Slower:</span> requires retraining 10-30% of original steps.<br>
          <span class="hl5">More complex:</span> straight-through estimator for gradients.<br><br>
          <span class="hl">Rule:</span> PTQ first. Only do QAT if PTQ accuracy is insufficient.
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ============================================================
     TAB 3: PRUNING
     ============================================================ -->
<div id="tab3" class="panel">
  <div class="grid2">
    <div class="card">
      <h3>&#9312; Unstructured vs Structured Pruning</h3>
      <canvas id="cvPrune" width="420" height="250"></canvas>
      <div class="btn-row">
        <button class="sel active" id="prBtn_unstruct" onclick="selectPrune('unstruct')">Unstructured</button>
        <button class="sel" id="prBtn_struct"   onclick="selectPrune('struct')">Structured</button>
        <button class="sel" id="prBtn_24"       onclick="selectPrune('24')">2:4 Sparsity</button>
      </div>
      <div class="row"><label>Sparsity %</label>
        <input type="range" id="slSparsity" min="0" max="90" step="5" value="50" oninput="renderPrune()">
        <span class="val" id="valSparsity">50%</span></div>
      <div class="info" id="infoPrune">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Gradual Pruning Schedule</h3>
      <canvas id="cvSched" width="420" height="250"></canvas>
      <div class="row"><label>Target sparsity</label>
        <input type="range" id="slTarget" min="10" max="95" step="5" value="70" oninput="renderSched()">
        <span class="val" id="valTarget">70%</span></div>
      <div class="info" id="infoSched">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Pruning Methods Compared</h3>
    <table class="tbl">
      <thead><tr><th>Method</th><th>Criterion</th><th>Best for</th><th>Limitation</th></tr></thead>
      <tbody>
        <tr><td><span class="hl2">Magnitude pruning</span></td><td>|w| smallest removed</td><td>Simple baseline; works surprisingly well</td><td>Magnitude &ne; importance; small but critical weights removed</td></tr>
        <tr><td><span class="hl3">Gradual (GMP)</span></td><td>Magnitude + schedule</td><td>High sparsity (&gt;70%) with recovery</td><td>Requires full training run</td></tr>
        <tr><td><span class="hl6">Movement pruning</span></td><td>w &times; &part;L/&part;w</td><td>Fine-tuning pre-trained models</td><td>Gradient computation overhead</td></tr>
        <tr><td><span class="hl">Lottery ticket</span></td><td>Reset to init after pruning</td><td>Finding minimal subnetworks</td><td>Very expensive (multiple full runs)</td></tr>
        <tr><td><span class="hl4">Structured (filter)</span></td><td>L1-norm of filter/channel</td><td>Real speedup without sparse hardware</td><td>Lower sparsity ceiling before accuracy drops</td></tr>
        <tr><td><span class="hl5">2:4 structured</span></td><td>2 largest in every 4</td><td>NVIDIA Ampere/Hopper GPU deployment</td><td>Hardware-specific; fixed 50% sparsity</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ============================================================
     TAB 4: MIXED PRECISION
     ============================================================ -->
<div id="tab4" class="panel">
  <div class="grid2">
    <div class="card">
      <h3>&#9312; FP16 vs BF16 &mdash; Precision Range Trade-off</h3>
      <canvas id="cvPrec" width="420" height="250"></canvas>
      <div class="info" id="infoPrec">Loading...</div>
    </div>
    <div class="card">
      <h3>&#9313; Dynamic Loss Scaling (FP16 Training)</h3>
      <canvas id="cvScale" width="420" height="250"></canvas>
      <div class="info" id="infoScale">Loading...</div>
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Mixed Precision: What Runs in What Precision</h3>
    <div class="grid2">
      <div>
        <div style="font-size:0.72em;font-weight:800;color:#4ade80;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">FP16 / BF16 (Fast Tensor Cores)</div>
        <table class="tbl">
          <thead><tr><th>Operation</th><th>Reason</th></tr></thead>
          <tbody>
            <tr><td>Linear layers (nn.Linear)</td><td>Tensor Core accelerated</td></tr>
            <tr><td>Convolutional layers</td><td>Main compute bottleneck</td></tr>
            <tr><td>BatchNorm (forward)</td><td>Safe with scaling</td></tr>
            <tr><td>Embedding lookups</td><td>Memory bandwidth bound</td></tr>
            <tr><td>Element-wise ops</td><td>Simple, bounded values</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <div style="font-size:0.72em;font-weight:800;color:#ef4444;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">Kept in FP32 (Precision Critical)</div>
        <table class="tbl">
          <thead><tr><th>Operation</th><th>Reason</th></tr></thead>
          <tbody>
            <tr><td>Softmax</td><td>exp() overflow risk</td></tr>
            <tr><td>Log-softmax</td><td>Numerical stability</td></tr>
            <tr><td>Loss computation</td><td>Must accumulate accurately</td></tr>
            <tr><td>Layer / Group Norm</td><td>Variance computation</td></tr>
            <tr><td>Optimiser step (Adam m,v)</td><td>Momentum needs precision</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div class="info" style="margin-top:10px;">
      <span class="hl">BF16 on A100+:</span> no loss scaling needed. Same exponent range as FP32 &rarr; no overflow. Just <code>torch.autocast(dtype=torch.bfloat16)</code>.<br>
      <span class="hl3">FP16 on V100:</span> needs GradScaler for dynamic loss scaling. Scale multiplied before backward, divided before optimizer step. Overflow &rarr; skip step, halve scale.
    </div>
  </div>
</div>

<!-- ============================================================
     TAB 5: PRODUCTION STACK
     ============================================================ -->
<div id="tab5" class="panel">
  <div class="card">
    <h3>&#9312; Accuracy &mdash; Memory &mdash; Speed Trade-off</h3>
    <canvas id="cvTradeoff" width="860" height="240"></canvas>
  </div>
  <div class="card">
    <h3>&#9313; Compression Pipeline: Training a New Model</h3>
    <div class="pipeline">
      <div class="pstep"><div class="pnum">Step 1</div><div class="ptitle" style="color:#4ecdc4;">Architecture</div><div class="pdesc">Design + structured pruning at design time</div></div>
      <div class="parr">&rsaquo;</div>
      <div class="pstep"><div class="pnum">Step 2</div><div class="ptitle" style="color:#c084fc;">Train BF16/FP16</div><div class="pdesc">Mixed precision + gradient accumulation</div></div>
      <div class="parr">&rsaquo;</div>
      <div class="pstep"><div class="pnum">Step 3</div><div class="ptitle" style="color:#ff6b35;">Label Smoothing</div><div class="pdesc">&alpha;=0.1 during training (classification only)</div></div>
      <div class="parr">&rsaquo;</div>
      <div class="pstep"><div class="pnum">Step 4</div><div class="ptitle" style="color:#fbbf24;">Prune</div><div class="pdesc">Gradual magnitude pruning or structured</div></div>
      <div class="parr">&rsaquo;</div>
      <div class="pstep"><div class="pnum">Step 5</div><div class="ptitle" style="color:#4ade80;">Quantise</div><div class="pdesc">PTQ (INT8/INT4) ALWAYS LAST</div></div>
      <div class="parr">&rsaquo;</div>
      <div class="pstep"><div class="pnum">Step 6</div><div class="ptitle" style="color:#38bdf8;">Export</div><div class="pdesc">TensorRT / ONNX / hardware format</div></div>
    </div>
    <div style="font-size:0.72em;font-family:'JetBrains Mono',monospace;color:#ef4444;margin-top:8px;">
      &#x26A0; KEY RULE: Quantisation is ALWAYS last. Fine-tune BEFORE quantising. Pruning after quantisation invalidates calibration.
    </div>
  </div>
  <div class="card">
    <h3>&#9314; Decision Framework &mdash; Which Technique for Which Problem</h3>
    <table class="tbl">
      <thead><tr><th>Problem / Constraint</th><th>Primary Solution</th><th>Impact</th></tr></thead>
      <tbody>
        <tr><td>Training too slow on available GPUs</td><td><span class="hl6">Mixed precision BF16/FP16</span></td><td>2&ndash;8&times; faster compute</td></tr>
        <tr><td>Batch size won&apos;t fit in GPU memory</td><td><span class="hl2">Gradient accumulation</span></td><td>K&times; larger effective batch</td></tr>
        <tr><td>Model overconfident / poor calibration</td><td><span class="hl">Label smoothing (&alpha;=0.1)</span></td><td>Better probs, slight acc. gain</td></tr>
        <tr><td>Need smaller model, similar accuracy</td><td><span class="hl3">Knowledge distillation</span></td><td>&minus;50&ndash;80% params, &minus;1&ndash;5% acc.</td></tr>
        <tr><td>Model too large for inference hardware</td><td><span class="hl4">INT8 quantisation (PTQ)</span></td><td>&minus;75% memory, 2&ndash;4&times; faster</td></tr>
        <tr><td>Inference too slow (latency)</td><td><span class="hl5">Structured pruning + quant.</span></td><td>2&ndash;4&times; speedup</td></tr>
        <tr><td>Consumer GPU deployment (LLM)</td><td><span class="hl6">INT4 (GPTQ/AWQ)</span></td><td>&minus;87.5% memory, &lt;2% ppl drop</td></tr>
        <tr><td>NVIDIA Ampere/Hopper deployment</td><td><span class="hl2">2:4 structured sparsity</span></td><td>1.5&ndash;2&times; speedup, &lt;1% drop</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
// ============================================================
// UTILITIES
// ============================================================
function rnd(v, d) { return v.toFixed(d === undefined ? 3 : d); }
function softmax(logits, T) {
  T = T || 1;
  var m = Math.max.apply(null, logits);
  var exps = logits.map(function(z) { return Math.exp((z - m) / T); });
  var s = exps.reduce(function(a, b) { return a + b; }, 0);
  return exps.map(function(e) { return e / s; });
}

function clearCV(id, W, H) {
  var cv = document.getElementById(id);
  var ctx = cv.getContext('2d');
  ctx.fillStyle = '#09090f'; ctx.fillRect(0, 0, W, H);
  return ctx;
}

function txt(ctx, str, x, y, col, sz, wt, al) {
  ctx.fillStyle = col || '#e4e4e7';
  ctx.font = (wt || '500') + ' ' + (sz || 11) + 'px JetBrains Mono, monospace';
  ctx.textAlign = al || 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(str, x, y);
}

function bar(ctx, x, y, w, h, fill, stroke, sw) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, 3);
  else ctx.rect(x, y, w, h);
  if (fill)   { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw || 1; ctx.stroke(); }
}

// ============================================================
// TAB SWITCHING
// ============================================================
var tabCount = 6;
function showTab(i) {
  for (var t = 0; t < tabCount; t++) {
    document.getElementById('tab' + t).classList.remove('active');
    document.querySelectorAll('.tab')[t].classList.remove('active');
  }
  document.getElementById('tab' + i).classList.add('active');
  document.querySelectorAll('.tab')[i].classList.add('active');
  // Lazy render
  if (i === 0) { renderLS(); renderLogit(); }
  if (i === 1) { renderDist(); renderDistLoss(); }
  if (i === 2) { renderFmt(); renderMem(); }
  if (i === 3) { renderPrune(); renderSched(); }
  if (i === 4) { renderPrec(); renderScale(); }
  if (i === 5) { renderTradeoff(); }
}

// ============================================================
// TAB 0: LABEL SMOOTHING
// ============================================================
function renderLS() {
  var alpha = parseFloat(document.getElementById('slAlpha').value);
  var K = parseInt(document.getElementById('slK').value);
  document.getElementById('valAlpha').textContent = rnd(alpha, 2);
  document.getElementById('valK').textContent = K;

  var trueClass = Math.floor(K / 2);
  var hardLabels = [];
  var smoothLabels = [];
  for (var k = 0; k < K; k++) {
    hardLabels.push(k === trueClass ? 1 : 0);
    smoothLabels.push(k === trueClass ? (1 - alpha + alpha / K) : alpha / K);
  }

  var ctx = clearCV('cvLS', 420, 240);
  var W = 420, H = 240, PAD = 36, bH = H - PAD - 30;
  var bW = Math.min(32, (W - PAD - 20) / (K * 2 + 1) - 4);
  var gap = bW * 0.4;
  var groupW = bW * 2 + gap + 10;
  var startX = PAD + ((W - PAD - 20) - groupW * K) / 2;

  // baseline
  ctx.strokeStyle = '#2d2d40'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();
  // 1.0 line
  ctx.strokeStyle = '#4ade8020'; ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD - bH); ctx.lineTo(W - 10, H - PAD - bH); ctx.stroke();
  ctx.setLineDash([]);
  txt(ctx, '1.0', PAD - 3, H - PAD - bH, '#4ade8050', 8, '400', 'right');

  for (var k2 = 0; k2 < K; k2++) {
    var gx = startX + k2 * groupW;
    // hard label bar
    var hH = hardLabels[k2] * bH;
    bar(ctx, gx, H - PAD - hH, bW, hH, '#4ecdc425', '#4ecdc4', 1.5);
    // smooth label bar
    var sH = smoothLabels[k2] * bH;
    bar(ctx, gx + bW + gap, H - PAD - sH, bW, sH, '#ff6b3530', '#ff6b35', 1.5);
    // class label
    txt(ctx, 'c' + (k2 + 1), gx + bW + gap / 2, H - PAD + 14, '#3f3f46', 8, '400');
    // value
    if (hardLabels[k2] > 0 || k2 === trueClass) {
      txt(ctx, rnd(hardLabels[k2], 1), gx + bW / 2, H - PAD - hH - 8, '#4ecdc4', 8, '700');
      txt(ctx, rnd(smoothLabels[k2], 3), gx + bW + gap + bW / 2, H - PAD - sH - 8, '#ff6b35', 8, '700');
    }
  }

  // legend
  bar(ctx, W - 120, 10, 12, 12, '#4ecdc425', '#4ecdc4', 1);
  txt(ctx, 'Hard', W - 103, 16, '#4ecdc4', 9, '700', 'left');
  bar(ctx, W - 68, 10, 12, 12, '#ff6b3530', '#ff6b35', 1);
  txt(ctx, 'Smooth', W - 51, 16, '#ff6b35', 9, '700', 'left');
  txt(ctx, 'True class: c' + (trueClass + 1), W / 2, 16, '#fbbf24', 9, '700');

  var trueSmooth = smoothLabels[trueClass];
  var otherSmooth = smoothLabels[0];
  document.getElementById('infoLS').innerHTML =
    '<span class="hl">Hard label:</span> [' + hardLabels.map(function(v) { return rnd(v, 0); }).join(', ') + ']<br>' +
    '<span class="hl2">Smooth label:</span> [' + smoothLabels.map(function(v) { return rnd(v, 3); }).join(', ') + ']<br><br>' +
    'True class target: <span class="hl3">' + rnd(trueSmooth, 4) + '</span> = (1\u2212\u03B1) + \u03B1/K = ' + rnd(1-alpha, 2) + ' + ' + rnd(alpha/K, 4) + '<br>' +
    'Other class floor: <span class="hl3">' + rnd(otherSmooth, 4) + '</span> = \u03B1/K = ' + rnd(alpha, 2) + '/' + K + '<br>' +
    'Model can <span class="hl">never</span> reach zero loss &mdash; gradient target for true class is bounded at ' + rnd(trueSmooth, 4) + '.';
}

function renderLogit() {
  var alpha = parseFloat(document.getElementById('slAlpha2').value);
  document.getElementById('valAlpha2').textContent = rnd(alpha, 2);
  var K = 5;

  var ctx = clearCV('cvLogit', 420, 240);
  var W = 420, H = 240, PAD = 38;
  var plotW = W - PAD - 20, plotH = H - PAD - 20;

  // Draw loss curves: L = -[(1-alpha+alpha/K)*log(p_true) + (K-1)*(alpha/K)*log((1-p_true)/(K-1))]
  // As we increase logit gap Δz (true class logit minus others)
  // p_true ≈ softmax (approx)

  var xmin = -2, xmax = 6;
  ctx.strokeStyle = '#1e1e2e'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD, 15); ctx.lineTo(PAD, H - PAD); ctx.stroke();

  // Loss vs logit gap for hard and smooth
  var colors = ['#4ecdc4', '#ff6b35'];
  var alphas2 = [0, alpha];
  var labels2 = ['Hard (α=0)', 'Smooth (α=' + rnd(alpha, 2) + ')'];

  alphas2.forEach(function(a, ai) {
    var pts = [];
    for (var i = 0; i <= 200; i++) {
      var dz = xmin + i / 200 * (xmax - xmin);
      // logits: true class gets dz, others get 0
      var logits = [0, 0, dz, 0, 0];
      var ps = softmax(logits, 1);
      var pt = ps[2];
      var tgt_true = (1 - a + a / K);
      var tgt_other = a / K;
      var loss = -tgt_true * Math.log(pt + 1e-10);
      for (var k = 0; k < K; k++) { if (k !== 2) loss -= tgt_other * Math.log(ps[k] + 1e-10); }
      var lossC = Math.min(loss, 3.5);
      var cx = PAD + (dz - xmin) / (xmax - xmin) * plotW;
      var cy = H - PAD - (lossC / 3.5) * plotH;
      pts.push([cx, cy]);
    }
    ctx.strokeStyle = colors[ai]; ctx.lineWidth = 2.5;
    ctx.beginPath();
    pts.forEach(function(p, pi) { if (pi === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
    ctx.stroke();
    // label
    txt(ctx, labels2[ai], pts[200][0] - 5, pts[200][1] - 10, colors[ai], 9, '700', 'right');
  });

  // Mark where smooth loss bottoms out
  var dzOpt = Math.log(K * (1 - alpha + alpha / K) / ((K - 1) * alpha / K + 1e-10));
  if (dzOpt > xmin && dzOpt < xmax) {
    var optX = PAD + (dzOpt - xmin) / (xmax - xmin) * plotW;
    ctx.strokeStyle = '#ff6b3560'; ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(optX, 15); ctx.lineTo(optX, H - PAD); ctx.stroke();
    ctx.setLineDash([]);
    txt(ctx, 'loss minimum \u0394z=' + rnd(dzOpt, 1), optX, 22, '#ff6b35', 8, '700');
  }

  // axis labels
  txt(ctx, '\u0394z (true class logit gap)', W / 2, H - 5, '#52525b', 9, '400');
  txt(ctx, 'Loss', PAD + 18, 20, '#52525b', 9, '400');
  txt(ctx, '0', PAD - 4, H - PAD, '#3f3f46', 8, '400', 'right');

  document.getElementById('infoLogit').innerHTML =
    '<span class="hl">Hard label (blue):</span> loss keeps decreasing as logit gap grows. Model pushes toward infinite logit gap &mdash; overconfidence.<br>' +
    '<span class="hl2">Smooth label (orange):</span> loss reaches a <span class="hl3">finite minimum</span> at \u0394z \u2248 ' + rnd(Math.max(0, Math.log(K*(1-alpha+alpha/K)/((K-1)*alpha/K+1e-10))), 1) + '.<br>' +
    'The model is trained to stop boosting the logit at a natural bound. This prevents overconfidence and improves calibration.';
}

// ============================================================
// TAB 1: DISTILLATION
// ============================================================
var TEACHER_LOGITS = [3.2, 0.1, -0.4, 0.8, 0.2];
var CLASS_NAMES = ['Cat', 'Dog', 'Bird', 'Fish', 'Car'];

function renderDist() {
  var T = parseFloat(document.getElementById('slTemp').value);
  document.getElementById('valTemp').textContent = rnd(T, 1);

  var probs = softmax(TEACHER_LOGITS, T);
  var probsT1 = softmax(TEACHER_LOGITS, 1);

  var ctx = clearCV('cvDist', 420, 250);
  var W = 420, H = 250, PAD = 36;
  var K = TEACHER_LOGITS.length;
  var bH = H - PAD - 40;
  var bW = 36;
  var startX = PAD + 20;
  var spacing = (W - PAD - 20 - startX) / (K - 1);

  // baseline
  ctx.strokeStyle = '#2d2d40'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();
  txt(ctx, '1.0', PAD - 3, H - PAD - bH, '#4ade8050', 8, '400', 'right');
  ctx.strokeStyle = '#4ade8015'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD - bH); ctx.lineTo(W - 10, H - PAD - bH); ctx.stroke();
  ctx.setLineDash([]);

  var cols = ['#ff6b35', '#4ecdc4', '#c084fc', '#fbbf24', '#4ade80'];
  probs.forEach(function(p, i) {
    var cx = startX + i * spacing;
    var pH = p * bH;
    var p1H = probsT1[i] * bH;
    // T=1 bar (ghost)
    bar(ctx, cx - bW / 2, H - PAD - p1H, bW, p1H, cols[i] + '18', cols[i] + '50', 1);
    // Temperature T bar
    bar(ctx, cx - bW / 2 + 4, H - PAD - pH, bW - 4, pH, cols[i] + '45', cols[i], 2);
    txt(ctx, rnd(p, 3), cx, H - PAD - pH - 10, cols[i], 9, '700');
    txt(ctx, CLASS_NAMES[i], cx, H - PAD + 14, '#52525b', 8, '400');
  });

  // legend
  bar(ctx, W - 140, 8, 12, 12, '#ffffff18', '#ffffff50', 1);
  txt(ctx, 'T=1 (ghost)', W - 122, 14, '#71717a', 8, '700', 'left');
  bar(ctx, W - 140, 24, 12, 12, '#ff6b3545', '#ff6b35', 1);
  txt(ctx, 'T=' + rnd(T, 1), W - 122, 30, '#ff6b35', 8, '700', 'left');
  txt(ctx, 'Teacher logits: [3.2, 0.1, \u22120.4, 0.8, 0.2]', W / 2, 14, '#3f3f46', 8, '400');

  var entropy = -probs.reduce(function(acc, p) { return acc + (p > 1e-9 ? p * Math.log(p) : 0); }, 0);
  document.getElementById('infoDist').innerHTML =
    '<span class="hl">Temperature T=' + rnd(T, 1) + '</span>  &rarr;  probs: [' + probs.map(function(p) { return rnd(p, 3); }).join(', ') + ']<br>' +
    'Distribution entropy: <span class="hl3">' + rnd(entropy, 3) + '</span> nats (higher = more uniform = more dark knowledge)<br><br>' +
    (T < 1.5 ? '<span class="hl5">Very peaked &mdash; student sees little inter-class structure.</span>' :
     T < 3   ? '<span class="hl3">Moderate softening &mdash; inter-class similarities starting to show.</span>' :
                '<span class="hl4">Soft distribution &mdash; rich dark knowledge visible (Bird-Fish-Car relationship).</span>');
}

function renderDistLoss() {
  var alpha = parseFloat(document.getElementById('slDistA').value);
  var T = parseFloat(document.getElementById('slDistT').value);
  document.getElementById('valDistA').textContent = rnd(alpha, 2);
  document.getElementById('valDistT').textContent = rnd(T, 1);

  var ctx = clearCV('cvDistLoss', 420, 250);
  var W = 420, H = 250;

  // Draw the loss composition diagram
  var cxM = W / 2, cyM = H / 2 - 20;
  var R = 75;

  // Two overlapping circles representing loss components
  // Circle 1: Hard label loss (left)
  ctx.globalAlpha = 0.8;
  ctx.fillStyle = '#4ecdc420';
  ctx.beginPath(); ctx.arc(cxM - 55, cyM, R, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#4ecdc460'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cxM - 55, cyM, R, 0, Math.PI * 2); ctx.stroke();

  // Circle 2: Distillation loss (right)
  ctx.fillStyle = '#ff6b3520';
  ctx.beginPath(); ctx.arc(cxM + 55, cyM, R, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#ff6b3560'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cxM + 55, cyM, R, 0, Math.PI * 2); ctx.stroke();
  ctx.globalAlpha = 1;

  txt(ctx, 'Hard Label Loss', cxM - 85, cyM - 28, '#4ecdc4', 9, '700');
  txt(ctx, 'L_CE(y_hard, p_s)', cxM - 85, cyM - 14, '#4ecdc4', 8, '400');
  txt(ctx, 'weight: \u03B1=' + rnd(alpha, 2), cxM - 85, cyM, '#4ecdc4', 9, '800');

  txt(ctx, 'Distill Loss', cxM + 85, cyM - 28, '#ff6b35', 9, '700');
  txt(ctx, 'T\u00B2\u00B7KL(p_T, p_s)', cxM + 85, cyM - 14, '#ff6b35', 8, '400');
  txt(ctx, 'weight: (1\u2212\u03B1)=' + rnd(1 - alpha, 2), cxM + 85, cyM, '#ff6b35', 9, '800');

  // Combined formula
  txt(ctx, 'Combined Student Loss', W / 2, 18, '#e4e4e7', 10, '700');
  txt(ctx, 'L = \u03B1\u00B7L_CE + (1\u2212\u03B1)\u00B7T\u00B2\u00B7L_KL', W / 2, cyM + R + 20, '#fbbf24', 10, '700');
  txt(ctx, '\u03B1=' + rnd(alpha, 2) + '  T=' + rnd(T, 1) + '  T\u00B2=' + rnd(T * T, 1), W / 2, cyM + R + 38, '#fbbf24', 9, '400');

  // Weight visualization bar
  var barY = H - 35, barX = 60, bW2 = W - 120;
  bar(ctx, barX, barY - 12, bW2 * alpha, 18, '#4ecdc440', '#4ecdc4', 1.5);
  bar(ctx, barX + bW2 * alpha, barY - 12, bW2 * (1 - alpha), 18, '#ff6b3540', '#ff6b35', 1.5);
  txt(ctx, '\u03B1=' + rnd(alpha, 2), barX + bW2 * alpha / 2, barY - 3, '#4ecdc4', 8, '700');
  txt(ctx, '1\u2212\u03B1=' + rnd(1 - alpha, 2), barX + bW2 * alpha + bW2 * (1 - alpha) / 2, barY - 3, '#ff6b35', 8, '700');

  document.getElementById('infoDistLoss').innerHTML =
    '<span class="hl">T\u00B2 factor:</span> dividing logits by T reduces gradients by T\u00B2. Multiplying the distillation loss by T\u00B2 restores gradient balance.<br>' +
    '<span class="hl3">Typical:</span> \u03B1=0.1&ndash;0.5, T=2&ndash;8. More T &rarr; softer targets &rarr; more inter-class info but less certainty about which class is correct.<br>' +
    '<span class="hl4">At \u03B1=0:</span> student trained entirely on soft targets (no hard labels). Works well in practice!';
}

// ============================================================
// TAB 2: QUANTISATION
// ============================================================
var FMTS = {
  fp32: { label:'FP32', bits:32, sign:1, exp:8, mant:23, color:'#4ecdc4', range:'±3.4×10³⁸', prec:'7 decimal', bytes:4,
    desc:'Full precision. No quantisation error. Standard for training. 4 bytes/param.' },
  fp16: { label:'FP16', bits:16, sign:1, exp:5, mant:10, color:'#fbbf24', range:'±65,504', prec:'3&ndash;4 decimal', bytes:2,
    desc:'Half precision. Risk of overflow (max 65,504) and underflow (&lt;6e&minus;5). Needs loss scaling for training.' },
  bf16: { label:'BF16', bits:16, sign:1, exp:8, mant:7, color:'#c084fc', range:'±3.4×10³⁸', prec:'2&ndash;3 decimal', bytes:2,
    desc:'Brain float. Same exponent as FP32 &rarr; no overflow. Less mantissa precision. Modern training default on A100+.' },
  int8: { label:'INT8', bits:8, sign:1, exp:0, mant:7, color:'#ff6b35', range:'&minus;128 to 127', prec:'Integer only', bytes:1,
    desc:'Integer quantisation. Requires scale factor. 4× memory reduction vs FP32. &lt;1% accuracy drop for most models (PTQ).' },
  int4: { label:'INT4', bits:4, sign:1, exp:0, mant:3, color:'#ef4444', range:'&minus;8 to 7', prec:'Integer only', bytes:0.5,
    desc:'4-bit quantisation. 8× memory reduction. GPTQ/AWQ enable &lt;2% perplexity increase on LLMs. Consumer GPU friendly.' }
};
var curFmt = 'fp32', curMem = 'fp32';

function selectFmt(k) {
  curFmt = k;
  document.querySelectorAll('[id^="fmtBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('fmtBtn_' + k).classList.add('active');
  renderFmt();
}

function renderFmt() {
  var f = FMTS[curFmt];
  var ctx = clearCV('cvFmt', 420, 220);
  var W = 420, H = 220;

  // Draw bit layout
  var totalBits = f.bits;
  var bW2 = Math.min(20, (W - 40) / totalBits - 1);
  var startX = (W - (bW2 + 1) * totalBits) / 2;
  var bitY = 80;
  var bH = 34;

  // Color regions
  var regions = [
    { start: 0, len: 1, color: '#ef4444', label: 'Sign' },
  ];
  if (f.exp > 0) regions.push({ start: 1, len: f.exp, color: '#4ecdc4', label: 'Exponent' });
  if (f.mant > 0) regions.push({ start: 1 + f.exp, len: f.mant, color: '#ff6b35', label: 'Mantissa' });

  // Int formats
  if (f.exp === 0) {
    regions = [
      { start: 0, len: 1, color: '#ef4444', label: 'Sign' },
      { start: 1, len: f.mant, color: '#ff6b3560', label: 'Value bits' },
    ];
  }

  regions.forEach(function(rg) {
    for (var b = 0; b < rg.len; b++) {
      var bx = startX + (rg.start + b) * (bW2 + 1);
      bar(ctx, bx, bitY, bW2, bH, rg.color + '30', rg.color, 1.5);
      if (bW2 >= 12) txt(ctx, b < 3 ? (rg.start + b === 0 ? 'S' : '') : '', bx + bW2 / 2, bitY + bH / 2, rg.color, 8, '700');
    }
    // bracket label
    var mid = startX + (rg.start + rg.len / 2) * (bW2 + 1) - bW2 / 2;
    txt(ctx, rg.label + ' (' + rg.len + 'b)', mid, bitY + bH + 14, rg.color, 9, '700');
  });

  // Title
  txt(ctx, f.label + ' \u2014 ' + f.bits + ' bits total', W / 2, 20, f.color, 13, '800');
  txt(ctx, f.bytes + ' bytes/param \u00B7 range: ' + f.range + ' \u00B7 precision: ' + f.prec, W / 2, 40, '#52525b', 9, '400');

  // Memory bar comparison
  var baseBytes = 4;
  var mbarY = 160, mbarH = 20, mbarW = W - 80;
  bar(ctx, 40, mbarY, mbarW, mbarH, '#09090f', '#1e1e2e', 1);
  bar(ctx, 40, mbarY, mbarW * (f.bytes / baseBytes), mbarH, f.color + '40', f.color, 1.5);
  txt(ctx, rnd(f.bytes / baseBytes * 100, 0) + '% of FP32 memory', W / 2, mbarY + mbarH / 2, f.color, 9, '700');

  txt(ctx, 'FP32: 4 bytes (100%)', 40, 190, '#3f3f46', 8, '400', 'left');
  txt(ctx, f.label + ': ' + (f.bytes < 1 ? f.bytes : rnd(f.bytes, 0)) + ' bytes', W - 40, 190, f.color, 8, '700', 'right');

  document.getElementById('infoFmt').innerHTML = f.desc;
}

function selectMem(k) {
  curMem = k;
  document.querySelectorAll('[id^="memBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('memBtn_' + k).classList.add('active');
  renderMem();
}

function renderMem() {
  var ctx = clearCV('cvMem', 420, 220);
  var W = 420, H = 220, PAD = 40;
  var params = 7e9;
  var fmtOrder = ['fp32', 'fp16', 'bf16', 'int8', 'int4'];
  var labels = ['FP32', 'FP16', 'BF16', 'INT8', 'INT4'];
  var maxGB = params * 4 / 1e9;
  var bH = H - PAD - 40;
  var bW = (W - PAD - 20) / fmtOrder.length - 6;
  var spacing = (W - PAD - 20) / fmtOrder.length;

  ctx.strokeStyle = '#2d2d40'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();

  // GPU memory reference lines
  var gpuLines = [{ gb: 80, label: 'A100 80GB', col: '#4ade8040' }, { gb: 24, label: 'RTX 3090 24GB', col: '#fbbf2440' }];
  gpuLines.forEach(function(gl) {
    var ly = H - PAD - (gl.gb / maxGB) * bH;
    if (ly > 15) {
      ctx.strokeStyle = gl.col; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(PAD, ly); ctx.lineTo(W - 10, ly); ctx.stroke();
      ctx.setLineDash([]);
      txt(ctx, gl.label, W - 12, ly - 6, gl.col.replace('40', 'ff'), 8, '700', 'right');
    }
  });

  fmtOrder.forEach(function(k, i) {
    var f = FMTS[k];
    var gb = params * f.bytes / 1e9;
    var barH2 = (gb / maxGB) * bH;
    var bx = PAD + i * spacing + (spacing - bW) / 2;
    var active = k === curMem;
    bar(ctx, bx, H - PAD - barH2, bW, barH2, f.color + (active ? '45' : '20'), f.color + (active ? 'ff' : '50'), active ? 2 : 1);
    txt(ctx, rnd(gb, 1) + 'GB', bx + bW / 2, H - PAD - barH2 - 10, f.color, 9, '700');
    txt(ctx, labels[i], bx + bW / 2, H - PAD + 14, active ? f.color : '#52525b', 9, active ? '800' : '400');
  });

  var selF = FMTS[curMem];
  var selGB = params * selF.bytes / 1e9;
  document.getElementById('infoMem').innerHTML =
    '<span class="hl">7B parameter model in ' + selF.label + ':</span> <span class="hl3">' + rnd(selGB, 1) + ' GB</span><br>' +
    'Fits on: ' + (selGB <= 8 ? '<span class="hl4">Consumer GPU (RTX 4090 24GB) \u2713</span>' : selGB <= 24 ? '<span class="hl3">RTX 3090/4090 24GB \u2713</span>' : selGB <= 80 ? '<span class="hl2">A100 80GB \u2713</span>' : '<span class="hl5">Needs multiple A100s \u2717</span>') + '<br>' +
    'vs FP32 (' + rnd(params * 4 / 1e9, 0) + ' GB): <span class="hl4">' + rnd((1 - selF.bytes / 4) * 100, 0) + '% memory saved</span>';
}

// ============================================================
// TAB 3: PRUNING
// ============================================================
var curPrune = 'unstruct';
var GRID = 8;

function selectPrune(k) {
  curPrune = k;
  document.querySelectorAll('[id^="prBtn_"]').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('prBtn_' + k).classList.add('active');
  renderPrune();
}

function renderPrune() {
  var sparsity = parseInt(document.getElementById('slSparsity').value);
  document.getElementById('valSparsity').textContent = sparsity + '%';
  var ctx = clearCV('cvPrune', 420, 250);
  var W = 420, H = 250;
  var cS = 22, gap = 3, N = 8;

  // Dense matrix (left)
  var lx = 30, ly = 60;
  txt(ctx, 'DENSE (original)', lx + N * (cS + gap) / 2, ly - 16, '#4ecdc4', 9, '700');
  for (var r = 0; r < N; r++) {
    for (var c = 0; c < N; c++) {
      bar(ctx, lx + c * (cS + gap), ly + r * (cS + gap), cS, cS, '#4ecdc420', '#4ecdc440', 1);
      txt(ctx, 'w', lx + c * (cS + gap) + cS / 2, ly + r * (cS + gap) + cS / 2, '#4ecdc460', 9, '700');
    }
  }

  // Arrow
  var mx = lx + N * (cS + gap) + 28;
  ctx.strokeStyle = '#3f3f46'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(mx, H / 2); ctx.lineTo(mx + 22, H / 2); ctx.stroke();
  ctx.fillStyle = '#3f3f46';
  ctx.beginPath(); ctx.moveTo(mx + 22, H / 2); ctx.lineTo(mx + 16, H / 2 - 5); ctx.lineTo(mx + 16, H / 2 + 5); ctx.closePath(); ctx.fill();
  txt(ctx, rnd(sparsity, 0) + '%', mx + 11, H / 2 - 14, '#fbbf24', 8, '700');
  txt(ctx, 'sparse', mx + 11, H / 2 + 14, '#52525b', 7, '400');

  // Sparse matrix (right)
  var rx = mx + 32, ry = ly;
  var pruneCount = Math.round(N * N * sparsity / 100);

  var pruned = [];
  if (curPrune === 'unstruct') {
    // Random positions pruned
    var all = [];
    for (var i = 0; i < N * N; i++) all.push(i);
    for (var i2 = 0; i2 < pruneCount; i2++) {
      var swap = i2 + Math.floor(Math.random() * (all.length - i2));
      var tmp = all[i2]; all[i2] = all[swap]; all[swap] = tmp;
    }
    for (var i3 = 0; i3 < pruneCount; i3++) pruned[all[i3]] = true;
  } else if (curPrune === 'struct') {
    // Prune entire rows (neurons/filters)
    var rowsToPrune = Math.round(N * sparsity / 100);
    for (var r2 = N - rowsToPrune; r2 < N; r2++) {
      for (var c2 = 0; c2 < N; c2++) pruned[r2 * N + c2] = true;
    }
  } else {
    // 2:4 pattern: in every group of 4, zero 2
    for (var i4 = 0; i4 < N * N; i4 += 4) {
      // zero the 1st and 3rd in each group
      pruned[i4] = true;
      pruned[i4 + 2] = true;
    }
  }

  var labels3 = { unstruct: 'SPARSE (unstructured)', struct: 'SMALLER (structured)', '24': '2:4 SPARSE' };
  txt(ctx, labels3[curPrune], rx + (curPrune === 'struct' ? (N - Math.round(N * sparsity / 100)) * (cS + gap) / 2 : N * (cS + gap) / 2), ry - 16, '#ff6b35', 9, '700');

  if (curPrune === 'struct') {
    var rowsLeft = N - Math.round(N * sparsity / 100);
    for (var r3 = 0; r3 < rowsLeft; r3++) {
      for (var c3 = 0; c3 < N; c3++) {
        bar(ctx, rx + c3 * (cS + gap), ry + r3 * (cS + gap), cS, cS, '#ff6b3525', '#ff6b3560', 1.5);
        txt(ctx, 'w', rx + c3 * (cS + gap) + cS / 2, ry + r3 * (cS + gap) + cS / 2, '#ff6b3580', 9, '700');
      }
    }
    // removed rows shown faded
    for (var r4 = rowsLeft; r4 < N; r4++) {
      for (var c4 = 0; c4 < N; c4++) {
        bar(ctx, rx + c4 * (cS + gap), ry + r4 * (cS + gap), cS, cS, '#09090f', '#1e1e2820', 0.5);
        txt(ctx, '0', rx + c4 * (cS + gap) + cS / 2, ry + r4 * (cS + gap) + cS / 2, '#1e1e28', 8, '400');
      }
    }
  } else {
    for (var r5 = 0; r5 < N; r5++) {
      for (var c5 = 0; c5 < N; c5++) {
        var idx = r5 * N + c5;
        var isp = !!pruned[idx];
        bar(ctx, rx + c5 * (cS + gap), ry + r5 * (cS + gap), cS, cS, isp ? '#09090f' : '#ff6b3525', isp ? '#1e1e28' : '#ff6b3560', isp ? 0.5 : 1.5);
        txt(ctx, isp ? '0' : 'w', rx + c5 * (cS + gap) + cS / 2, ry + r5 * (cS + gap) + cS / 2, isp ? '#1e1e28' : '#ff6b3580', 9, isp ? '400' : '700');
      }
    }
  }

  var infoMap = {
    unstruct: '<span class="hl5">Hardware speedup: LIMITED.</span> Zero values still processed by dense matmul. Needs >90% sparsity or sparse hardware for actual speedup.<br><span class="hl4">Accuracy: best</span> &mdash; flexible which weights are removed.',
    struct:   '<span class="hl4">Hardware speedup: IMMEDIATE.</span> Fewer rows = smaller matrix = fewer FLOPs with standard dense ops.<br><span class="hl5">Accuracy: lower ceiling</span> &mdash; removing whole neurons is coarser than removing individual weights.',
    '24':     '<span class="hl4">Hardware speedup: 1.5&ndash;2&times; on NVIDIA Ampere+.</span> Dedicated Tensor Core sparse units. Fixed 50% sparsity, hardware-native pattern.<br>NVIDIA\'s ASP: train dense &rarr; sparsify (keep 2 largest in each 4) &rarr; fine-tune.'
  };
  document.getElementById('infoPrune').innerHTML = infoMap[curPrune];
}

function renderSched() {
  var sf = parseInt(document.getElementById('slTarget').value) / 100;
  document.getElementById('valTarget').textContent = (sf * 100).toFixed(0) + '%';

  var ctx = clearCV('cvSched', 420, 250);
  var W = 420, H = 250, PAD = 40;
  var plotW = W - PAD - 20, plotH = H - PAD - 30;

  ctx.strokeStyle = '#2d2d40'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD, 20); ctx.lineTo(PAD, H - PAD); ctx.stroke();

  // Cubic schedule: s(t) = sf * [1 - (1 - t/T)^3]
  var T = 1.0;
  ctx.strokeStyle = '#ff6b35'; ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (var i = 0; i <= 200; i++) {
    var t = i / 200 * T;
    var s = sf * (1 - Math.pow(1 - t / T, 3));
    var cx = PAD + t * plotW;
    var cy = H - PAD - s * plotH;
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // Target line
  ctx.strokeStyle = '#ff6b3550'; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD - sf * plotH); ctx.lineTo(W - 10, H - PAD - sf * plotH); ctx.stroke();
  ctx.setLineDash([]);
  txt(ctx, 's_f = ' + (sf * 100).toFixed(0) + '%', W - 12, H - PAD - sf * plotH - 6, '#ff6b35', 8, '700', 'right');

  // Annotations for slow/fast/slow phases
  txt(ctx, 'slow start', PAD + plotW * 0.1, H - PAD - sf * 0.1 * plotH + 14, '#52525b', 8, '400');
  txt(ctx, 'fast growth', PAD + plotW * 0.45, H - PAD - sf * 0.6 * plotH - 14, '#52525b', 8, '400');
  txt(ctx, 'recovery', PAD + plotW * 0.85, H - PAD - sf * 0.97 * plotH - 14, '#52525b', 8, '400');

  // Axis labels
  txt(ctx, 'Training progress', W / 2, H - 10, '#52525b', 9, '400');
  txt(ctx, 'Sparsity', PAD + 22, 14, '#52525b', 9, '400');
  txt(ctx, '0', PAD - 4, H - PAD, '#3f3f46', 8, '400', 'right');
  txt(ctx, '100%', PAD - 4, 22, '#3f3f46', 8, '400', 'right');

  document.getElementById('infoSched').innerHTML =
    '<span class="hl">Cubic schedule:</span> s(t) = s_f &times; [1 &minus; (1 &minus; t/T)&sup3;]<br>' +
    'Starts slow &rarr; ramps fast &rarr; levels off at target. Gives the model time to adapt.<br><br>' +
    'Target sparsity: <span class="hl3">' + (sf * 100).toFixed(0) + '%</span>. ' +
    (sf >= 0.8 ? '<span class="hl5">Very aggressive. Use movement pruning or multiple cycles.</span>' : sf >= 0.5 ? '<span class="hl3">Moderate. GMP achieves this well with &lt;1% accuracy loss.</span>' : '<span class="hl4">Conservative. One-shot pruning + fine-tune usually sufficient.</span>');
}

// ============================================================
// TAB 4: MIXED PRECISION
// ============================================================
function renderPrec() {
  var ctx = clearCV('cvPrec', 420, 250);
  var W = 420, H = 250;

  var fmts2 = [
    { name: 'FP32', exp: 8, mant: 23, color: '#4ecdc4', range: '±3.4e38', overflow: 'None', underflow: 'None' },
    { name: 'FP16', exp: 5, mant: 10, color: '#fbbf24', range: '±65504', overflow: 'HIGH', underflow: 'HIGH' },
    { name: 'BF16', exp: 8, mant: 7, color: '#c084fc', range: '±3.4e38', overflow: 'None', underflow: 'None' },
  ];

  var bH = H - 100, barY = 50;
  var maxBits = 31;
  var bW = (W - 60) / 3 - 10;

  fmts2.forEach(function(f, i) {
    var cx = 30 + i * ((W - 60) / 3 + 5);
    // exp bar
    var eH = (f.exp / maxBits) * bH;
    bar(ctx, cx, barY + bH - eH, bW / 2 - 2, eH, '#4ecdc430', '#4ecdc4', 1.5);
    // mant bar
    var mH = (f.mant / maxBits) * bH;
    bar(ctx, cx + bW / 2 + 2, barY + bH - mH, bW / 2 - 2, mH, '#ff6b3530', '#ff6b35', 1.5);

    txt(ctx, f.name, cx + bW / 2, barY - 12, f.color, 11, '800');
    txt(ctx, 'Exp:' + f.exp + 'b', cx + bW / 4, barY + bH + 14, '#4ecdc4', 8, '700');
    txt(ctx, 'Mant:' + f.mant + 'b', cx + bW * 3 / 4, barY + bH + 14, '#ff6b35', 8, '700');
    txt(ctx, f.range, cx + bW / 2, barY + bH + 28, '#52525b', 7, '400');

    var oCol = f.overflow === 'None' ? '#4ade80' : '#ef4444';
    var uCol = f.underflow === 'None' ? '#4ade80' : '#ef4444';
    txt(ctx, 'OV:' + f.overflow, cx + bW / 4, barY + bH + 42, oCol, 8, '700');
    txt(ctx, 'UF:' + f.underflow, cx + bW * 3 / 4, barY + bH + 42, uCol, 8, '700');
  });

  // Legend
  bar(ctx, 10, H - 24, 10, 10, '#4ecdc430', '#4ecdc4', 1);
  txt(ctx, 'Exponent (range)', 26, H - 19, '#4ecdc4', 8, '700', 'left');
  bar(ctx, 130, H - 24, 10, 10, '#ff6b3530', '#ff6b35', 1);
  txt(ctx, 'Mantissa (precision)', 146, H - 19, '#ff6b35', 8, '700', 'left');
  txt(ctx, 'OV=Overflow risk  UF=Underflow risk', W - 10, H - 19, '#52525b', 7, '400', 'right');

  document.getElementById('infoPrec').innerHTML =
    '<span class="hl3">FP16 vs BF16 trade-off:</span><br>' +
    'FP16: more mantissa bits (10) &rarr; better precision. Less exponent bits (5) &rarr; overflow risk during training.<br>' +
    'BF16: same exponent as FP32 (8) &rarr; same range, no overflow. Less mantissa (7) &rarr; coarser precision but rarely matters.<br>' +
    '<span class="hl4">Rule:</span> A100/H100/TPU &rarr; BF16. V100/older &rarr; FP16 + loss scaling.';
}

function renderScale() {
  var ctx = clearCV('cvScale', 420, 250);
  var W = 420, H = 250, PAD = 40;
  var plotW = W - PAD - 20, plotH = H - PAD - 30;

  // Simulate dynamic loss scale over training steps
  var N2 = 160;
  var scale = 512;
  var scales = [scale];
  var overflows = [];
  var rng = { v: 42 };
  function rand01(r) { r.v = (r.v * 1664525 + 1013904223) & 0xffffffff; return (r.v >>> 0) / 4294967296; }

  for (var t = 1; t < N2; t++) {
    var overflow = rand01(rng) < (scale > 4096 ? 0.15 : scale > 1024 ? 0.06 : 0.02);
    if (overflow) {
      scale = scale / 2;
      overflows.push(t);
    } else if (t % 20 === 0) {
      scale = Math.min(scale * 2, 65536);
    }
    scales.push(scale);
  }

  var maxS = Math.max.apply(null, scales);
  ctx.strokeStyle = '#2d2d40'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD, 15); ctx.lineTo(PAD, H - PAD); ctx.stroke();

  // Scale curve
  ctx.strokeStyle = '#4ecdc4'; ctx.lineWidth = 2;
  ctx.beginPath();
  scales.forEach(function(s, t2) {
    var cx = PAD + t2 / N2 * plotW;
    var cy = H - PAD - Math.log2(s) / Math.log2(maxS) * plotH;
    if (t2 === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Overflow events
  overflows.forEach(function(t3) {
    var cx = PAD + t3 / N2 * plotW;
    ctx.strokeStyle = '#ef444460'; ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(cx, 15); ctx.lineTo(cx, H - PAD); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.beginPath(); ctx.arc(cx, H - PAD - Math.log2(scales[t3]) / Math.log2(maxS) * plotH, 4, 0, Math.PI * 2); ctx.fill();
  });

  txt(ctx, 'Dynamic Loss Scale Over Training Steps', W / 2, 12, '#e4e4e7', 9, '700');
  txt(ctx, 'Loss Scale', PAD + 24, 20, '#4ecdc4', 9, '400');
  txt(ctx, 'Training Steps', W / 2, H - 10, '#52525b', 9, '400');
  txt(ctx, 'Overflow detected (scale halved)', W - 12, 28, '#ef4444', 8, '700', 'right');
  ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(W - 115, 28, 4, 0, Math.PI * 2); ctx.fill();
  txt(ctx, '&every 2000 steps: scale doubled', W - 12, 44, '#4ecdc460', 8, '400', 'right');

  document.getElementById('infoScale').innerHTML =
    '<span class="hl">Dynamic Loss Scaling:</span> scale grows every K good steps (no Inf/NaN gradients). When overflow detected: skip optimizer step, halve scale immediately.<br>' +
    'This self-regulating loop keeps gradients in FP16\'s representable range automatically.<br>' +
    '<span class="hl4">BF16 users:</span> no loss scaling needed (same exponent range as FP32 &rarr; no overflow possible).';
}

// ============================================================
// TAB 5: PRODUCTION STACK
// ============================================================
function renderTradeoff() {
  var ctx = clearCV('cvTradeoff', 860, 240);
  var W = 860, H = 240;

  var techniques = [
    { label: 'Baseline\nFP32 Dense', mem: 100, speed: 1.0, acc: 0, color: '#4ecdc4' },
    { label: 'Label\nSmoothing', mem: 100, speed: 1.0, acc: 1, color: '#ff6b35' },
    { label: 'BF16\nTraining', mem: 50, speed: 6.0, acc: 0, color: '#c084fc' },
    { label: 'Knowledge\nDistillation', mem: 25, speed: 4.0, acc: -2, color: '#fbbf24' },
    { label: 'INT8 PTQ', mem: 25, speed: 3.0, acc: -0.5, color: '#4ade80' },
    { label: '50% Struct\nPruning', mem: 50, speed: 2.0, acc: -2, color: '#38bdf8' },
    { label: 'INT4\n(GPTQ)', mem: 12.5, speed: 6.0, acc: -2.5, color: '#f472b6' },
    { label: 'Prune+Quant\nINT8', mem: 12, speed: 8.0, acc: -3, color: '#fb923c' },
  ];

  var rowH = H / 3;
  var colW = W / techniques.length;

  // Headers
  txt(ctx, 'Memory (% of FP32)', 10, rowH / 2, '#52525b', 9, '700', 'left');
  txt(ctx, 'Speed vs FP32', 10, rowH + rowH / 2, '#52525b', 9, '700', 'left');
  txt(ctx, 'Accuracy \u0394', 10, rowH * 2 + rowH / 2, '#52525b', 9, '700', 'left');

  var memMax = 100, speedMax = 9, accRange = 5;
  var barPad = 10, colStart = 90;
  var availW = W - colStart;
  var bW2 = availW / techniques.length - barPad;

  techniques.forEach(function(t, i) {
    var cx = colStart + i * (availW / techniques.length) + bW2 / 2;
    var bx = colStart + i * (availW / techniques.length) + barPad / 2;
    var col = t.color;

    // memory bar (smaller = better, bar shows reduction)
    var mH = ((100 - t.mem) / memMax) * (rowH - 24);
    bar(ctx, bx, rowH / 2 - mH / 2, bW2, mH, col + '30', col + '80', 1.5);
    txt(ctx, t.mem + '%', cx, rowH / 2, col, 9, '700');

    // speed bar
    var sH = (t.speed / speedMax) * (rowH - 24);
    bar(ctx, bx, rowH + rowH / 2 - sH / 2, bW2, sH, col + '30', col + '80', 1.5);
    txt(ctx, t.speed.toFixed(1) + 'x', cx, rowH + rowH / 2, col, 9, '700');

    // accuracy delta
    var aH = (Math.abs(t.acc) / accRange) * (rowH - 24);
    var aCol = t.acc > 0 ? '#4ade80' : t.acc === 0 ? '#52525b' : '#ef4444';
    bar(ctx, bx, rowH * 2 + rowH / 2 - aH / 2, bW2, aH, aCol + '30', aCol + '80', 1.5);
    txt(ctx, (t.acc > 0 ? '+' : '') + t.acc + '%', cx, rowH * 2 + rowH / 2, aCol, 9, '700');

    // label at bottom
    var lines = t.label.split('\n');
    lines.forEach(function(l, li) {
      txt(ctx, l, cx, H - 28 + li * 12, col, 8, '700');
    });

    // divider
    if (i > 0) {
      ctx.strokeStyle = '#1e1e2e'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(bx - barPad / 2, 8); ctx.lineTo(bx - barPad / 2, H - 36); ctx.stroke();
    }
  });

  // Row dividers
  [rowH, rowH * 2].forEach(function(y) {
    ctx.strokeStyle = '#1e1e2e'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(colStart - 10, y); ctx.lineTo(W, y); ctx.stroke();
  });
}

// ============================================================
// INIT
// ============================================================
window.addEventListener('load', function() {
  renderLS(); renderLogit();
});
</script>
</body>
</html>"""

ADVTRAINING_VISUAL_HEIGHT = 2200