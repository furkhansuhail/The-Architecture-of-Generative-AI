# ─────────────────────────────────────────────────────────────────────────────
# VISUAL HTML
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TUNING_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Prompt Tuning — PEFT Visual</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Sora:wght@300;400;600;700&display=swap');

*{box-sizing:border-box;margin:0;padding:0;}
body{
  font-family:'Sora',sans-serif;
  background:#07070e;
  color:#e4e4e7;
  padding:22px;
  min-height:100vh;
}
h2{font-family:'JetBrains Mono',monospace;font-size:1.18em;color:#f97316;margin-bottom:3px;letter-spacing:.02em;}
.subtitle{color:#3f3f46;font-size:0.78em;margin-bottom:20px;font-family:'JetBrains Mono',monospace;}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.full{grid-column:1/-1;}
.col3{grid-template-columns:1fr 1fr 1fr;}

.card{
  background:#0f0f1a;
  border:1px solid #1e1e30;
  border-radius:14px;
  padding:18px;
  box-shadow:0 8px 36px rgba(0,0,0,.45);
}
.card h3{
  font-family:'JetBrains Mono',monospace;
  font-size:.76em;font-weight:800;
  text-transform:uppercase;letter-spacing:.1em;
  color:#f97316;margin-bottom:12px;
}
canvas{display:block;border-radius:8px;}

.info-box{
  background:#0a0a14;border:1px solid #1e1e30;border-radius:8px;
  padding:8px 12px;font-size:.76em;color:#94a3b8;
  line-height:1.75;margin-top:8px;font-family:'JetBrains Mono',monospace;
}
.hl {color:#f97316;font-weight:700;}
.hl2{color:#34d399;font-weight:700;}
.hl3{color:#60a5fa;font-weight:700;}
.hl4{color:#facc15;font-weight:700;}

.btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
button{
  background:#1a1a2e;color:#71717a;
  border:1px solid #2d2d44;border-radius:6px;
  padding:5px 13px;cursor:pointer;
  font-family:'JetBrains Mono',monospace;
  font-size:.73em;font-weight:700;
  transition:all .15s;
}
button:hover{background:#252540;color:#e4e4e7;}
button.active{background:#f97316;color:#07070e;border-color:#f97316;}

.row{display:flex;align-items:center;gap:10px;margin:6px 0;}
.row label{font-family:'JetBrains Mono',monospace;font-size:.73em;color:#52525b;min-width:90px;}
input[type=range]{
  flex:1;height:4px;border-radius:2px;
  -webkit-appearance:none;appearance:none;
  background:#1e1e30;outline:none;cursor:pointer;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:#f97316;cursor:pointer;
}
.val{font-family:'JetBrains Mono',monospace;font-size:.73em;color:#f97316;min-width:46px;text-align:right;}

/* family tree */
.tree-wrap{overflow-x:auto;padding-bottom:4px;}

/* quiz */
.quiz-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px;}
.q-card{background:#0a0a14;border:1px solid #1e1e30;border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:7px;}
.q-num{font-family:'JetBrains Mono',monospace;font-size:.65em;color:#2d2d44;text-transform:uppercase;letter-spacing:.1em;}
.q-prompt{font-family:'JetBrains Mono',monospace;font-size:.76em;color:#cbd5e1;line-height:1.55;font-weight:700;}
.q-opt{background:#0f0f1a;border:1px solid #2d2d44;border-radius:6px;padding:6px 10px;font-family:'JetBrains Mono',monospace;font-size:.71em;color:#52525b;cursor:pointer;text-align:left;transition:all .15s;line-height:1.4;width:100%;}
.q-opt:hover:not([disabled]){background:#1a1a2e;color:#e4e4e7;border-color:#3f3f56;}
.q-opt.correct{border-color:#34d399;color:#34d399;background:#051a10!important;}
.q-opt.wrong  {border-color:#f87171;color:#f87171;background:#1a0808!important;}
.q-feedback{font-family:'JetBrains Mono',monospace;font-size:.69em;line-height:1.55;padding:6px 8px;border-radius:6px;margin-top:2px;display:none;}
.q-feedback.show{display:block;}
.q-feedback.ok {background:#051a10;border:1px solid #34d39940;color:#34d399;}
.q-feedback.bad{background:#1a0808;border:1px solid #f8717140;color:#f87171;}
.score-bar{display:flex;align-items:center;gap:14px;margin-top:14px;padding:10px 16px;background:#0a0a14;border:1px solid #1e1e30;border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:.78em;}
.pip{width:11px;height:11px;border-radius:50%;background:#1e1e30;border:1px solid #2d2d44;transition:all .3s;flex-shrink:0;}
.pip.ok {background:#34d399;border-color:#34d399;box-shadow:0 0 6px #34d39966;}
.pip.bad{background:#f87171;border-color:#f87171;}

/* legend chip */
.chip{display:inline-block;padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.68em;font-weight:700;margin:2px;}
.chip-orange{background:#f9731622;border:1px solid #f9731655;color:#f97316;}
.chip-green {background:#34d39922;border:1px solid #34d39955;color:#34d399;}
.chip-blue  {background:#60a5fa22;border:1px solid #60a5fa55;color:#60a5fa;}
.chip-gray  {background:#52525b22;border:1px solid #52525b55;color:#71717a;}
</style>
</head>
<body>

<h2>🔵 Prompt Tuning — Parameter-Efficient Fine-Tuning (PEFT)</h2>
<p class="subtitle">Soft prompts &middot; Frozen model &middot; Embedding space &middot; Family tree &middot; Parameter math &middot; Quiz</p>

<div class="grid">

  <!-- ══════ PANEL 1: Architecture — The Core Mechanism ══════ -->
  <div class="card full">
    <h3>① Architecture — Soft Prompt Prepended to a Frozen LLM</h3>
    <canvas id="cvArch" width="900" height="230"></canvas>
    <div class="btn-row" style="margin-top:10px;">
      <button id="btnHard"  class="active" onclick="setMode('hard')">Hard Prompt (text)</button>
      <button id="btnSoft"  onclick="setMode('soft')">Soft Prompt (learned vectors)</button>
      <button id="btnAnim"  onclick="toggleAnim()">▶ Animate Forward Pass</button>
    </div>
    <div class="info-box" id="archInfo">—</div>
  </div>

  <!-- ══════ PANEL 2: Embedding Space ══════ -->
  <div class="card">
    <h3>② Embedding Space — Hard vs Soft Tokens</h3>
    <canvas id="cvEmbed" width="430" height="230"></canvas>
    <div class="row"><label>soft token k</label><input type="range" id="slK" min="1" max="8" step="1" value="3"><span class="val" id="slKv">3</span></div>
    <div class="info-box" id="embedInfo">—</div>
  </div>

  <!-- ══════ PANEL 3: Parameter Efficiency ══════ -->
  <div class="card">
    <h3>③ Parameter Efficiency — How Few is Few?</h3>
    <canvas id="cvParam" width="430" height="230"></canvas>
    <div class="row"><label>model size</label>
      <input type="range" id="slModel" min="0" max="3" step="1" value="1">
      <span class="val" id="slModelv">770M</span>
    </div>
    <div class="row"><label>soft tokens k</label>
      <input type="range" id="slPK" min="1" max="100" step="1" value="20">
      <span class="val" id="slPKv">20</span>
    </div>
    <div class="info-box" id="paramInfo">—</div>
  </div>

  <!-- ══════ PANEL 4: PEFT Family Tree ══════ -->
  <div class="card full">
    <h3>④ Prompt-Based PEFT Family Tree — WHERE Soft Tokens Are Inserted</h3>
    <canvas id="cvFamily" width="900" height="240"></canvas>
    <div class="btn-row" style="margin-top:10px;">
      <button id="btnFamHard"  class="active"  onclick="setFamily('hard')">Hard Prompting</button>
      <button id="btnFamPT"   onclick="setFamily('pt')">Prompt Tuning</button>
      <button id="btnFamPre"  onclick="setFamily('prefix')">Prefix Tuning</button>
      <button id="btnFamP2"   onclick="setFamily('p2')">P-Tuning v2</button>
    </div>
    <div class="info-box" id="familyInfo">—</div>
  </div>

  <!-- ══════ PANEL 5: Accuracy vs Scale ══════ -->
  <div class="card">
    <h3>⑤ Accuracy vs Model Scale</h3>
    <canvas id="cvScale" width="430" height="230"></canvas>
    <div class="info-box" id="scaleInfo">—</div>
  </div>

  <!-- ══════ PANEL 6: Training Loop ══════ -->
  <div class="card">
    <h3>⑥ Training Loop — Gradient Flow</h3>
    <canvas id="cvGrad" width="430" height="230"></canvas>
    <div class="btn-row">
      <button id="btnGradPlay" onclick="toggleGrad()">▶ Run Training</button>
      <button onclick="resetGrad()">↺ Reset</button>
    </div>
    <div class="info-box" id="gradInfo">—</div>
  </div>

  <!-- ══════ QUIZ ══════ -->
  <div class="card full">
    <h3>⑦ Knowledge Check</h3>
    <div class="quiz-grid" id="quizGrid"></div>
    <div class="score-bar" id="scoreBar"><span id="scoreText">Answer all 6 questions to see your score.</span></div>
  </div>

</div>

<script>
// ─── shared palette ──────────────────────────────────────────────────────────
const C = {
  bg:      '#07070e',
  card:    '#0f0f1a',
  border:  '#1e1e30',
  orange:  '#f97316',
  green:   '#34d399',
  blue:    '#60a5fa',
  yellow:  '#facc15',
  purple:  '#c084fc',
  red:     '#f87171',
  dim:     '#3f3f56',
  dimmer:  '#1e1e30',
  text:    '#e4e4e7',
  muted:   '#52525b',
  mono:    "'JetBrains Mono',monospace",
};

function mono(ctx,sz,bold){ctx.font=(bold?'bold ':'')+sz+'px '+C.mono;}
function fill(ctx,color){ctx.fillStyle=color;}
function stroke(ctx,color,w=1){ctx.strokeStyle=color;ctx.lineWidth=w;}

// ─────────────────────────────────────────────────────────────────────────────
// PANEL 1 — Architecture
// ─────────────────────────────────────────────────────────────────────────────
const cvA = document.getElementById('cvArch');
const ctxA = cvA.getContext('2d');
let archMode = 'hard';
let animT = 0, animRunning = false, animRaf = null;

function setMode(m){
  archMode = m;
  ['Hard','Soft'].forEach(x=>{
    document.getElementById('btn'+x).classList.toggle('active', m===x.toLowerCase());
  });
  animT = 0;
  drawArch();
}

function toggleAnim(){
  if(animRunning){ animRunning=false; cancelAnimationFrame(animRaf); drawArch(); return; }
  document.getElementById('btnAnim').textContent='⏸ Pause';
  animRunning = true;
  function loop(){
    animT = (animT+0.012) % 1;
    drawArch();
    if(animRunning) animRaf = requestAnimationFrame(loop);
    else document.getElementById('btnAnim').textContent='▶ Animate Forward Pass';
  }
  loop();
}

function drawArch(){
  const W=900, H=230;
  const ctx=ctxA;
  ctx.clearRect(0,0,W,H);
  fill(ctx,'#09091500'); ctx.fillRect(0,0,W,H);

  const isSoft = archMode==='soft';
  const PAD=22, blockH=54, blockY=(H-blockH)/2;
  const tokenH=38, tokenY=blockY+(blockH-tokenH)/2;

  // ── section labels ────────────────────────────────────────────────────────
  mono(ctx,9,false); fill(ctx,C.muted); ctx.textAlign='center';

  // Hard prompt tokens
  const hardTokens = isSoft
    ? ['[?]','[?]','[?]']       // soft → 3 soft tokens shown
    : ['Classify','the','sentiment',':','Great','movie','!'];
  const softCount  = isSoft ? 3 : 0;
  const realTokens = ['Great','movie','!'];

  const allTokens   = isSoft ? [...Array(softCount).fill(null), ...realTokens] : hardTokens;
  const totalT      = allTokens.length;
  const tokW        = Math.min(72, (W*0.42 - PAD*2)/totalT - 6);
  const tokGap      = 5;
  const tokenAreaW  = totalT * (tokW+tokGap);
  const tokenAreaX  = PAD;

  // draw tokens
  allTokens.forEach((tok,i)=>{
    const tx = tokenAreaX + i*(tokW+tokGap);
    const isSoftTok = isSoft && i < softCount;
    const pulse = isSoftTok && animRunning
      ? 0.5 + 0.5*Math.sin(animT*Math.PI*2 - i*0.8) : 0;

    ctx.beginPath();
    ctx.roundRect(tx, tokenY, tokW, tokenH, 6);
    const baseColor = isSoftTok ? `rgba(249,115,22,${0.15+pulse*0.25})`
                                : 'rgba(96,165,250,0.1)';
    fill(ctx, baseColor);
    ctx.fill();
    stroke(ctx, isSoftTok ? C.orange : C.blue, isSoftTok?1.5:1);
    ctx.stroke();

    mono(ctx,9,true); ctx.textAlign='center';
    fill(ctx, isSoftTok ? C.orange : C.blue);
    ctx.fillText(isSoftTok ? `p${i+1}` : tok, tx+tokW/2, tokenY+tokenH/2+4);

    // label above
    mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
    ctx.fillText(isSoftTok?'soft':'token', tx+tokW/2, tokenY-5);
  });

  // bracket label
  if(isSoft && softCount>0){
    const bx1=tokenAreaX, bx2=tokenAreaX+softCount*(tokW+tokGap)-tokGap;
    const by = tokenY+tokenH+12;
    ctx.beginPath(); ctx.moveTo(bx1,by); ctx.lineTo(bx1,by+5); ctx.lineTo(bx2,by+5); ctx.lineTo(bx2,by);
    stroke(ctx,C.orange,1.2); ctx.stroke();
    mono(ctx,8.5,true); fill(ctx,C.orange); ctx.textAlign='center';
    ctx.fillText('Trainable soft prompt P ∈ ℝᵏˣᵈ', (bx1+bx2)/2, by+16);
  }
  if(isSoft){
    const bx1=tokenAreaX+softCount*(tokW+tokGap), bx2=tokenAreaX+totalT*(tokW+tokGap)-tokGap;
    const by = tokenY+tokenH+12;
    ctx.beginPath(); ctx.moveTo(bx1,by); ctx.lineTo(bx1,by+5); ctx.lineTo(bx2,by+5); ctx.lineTo(bx2,by);
    stroke(ctx,C.blue,1); ctx.stroke();
    mono(ctx,8.5,false); fill(ctx,C.blue); ctx.textAlign='center';
    ctx.fillText('Frozen token embeddings', (bx1+bx2)/2, by+16);
  }

  // ── Arrow ────────────────────────────────────────────────────────────────
  const arrowX = tokenAreaX + tokenAreaW + 8;
  const midY   = H/2;
  const arrowLen = 30;

  // animated particle along arrow
  if(animRunning){
    const px = arrowX + animT * arrowLen;
    ctx.beginPath(); ctx.arc(px, midY, 4, 0, Math.PI*2);
    fill(ctx,'#f97316aa'); ctx.fill();
  }

  ctx.beginPath(); ctx.moveTo(arrowX,midY); ctx.lineTo(arrowX+arrowLen,midY);
  stroke(ctx,C.dim,1.5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(arrowX+arrowLen,midY-5); ctx.lineTo(arrowX+arrowLen+7,midY); ctx.lineTo(arrowX+arrowLen,midY+5);
  fill(ctx,C.dim); ctx.fill();

  // ── Transformer block ────────────────────────────────────────────────────
  const tbX = arrowX + arrowLen + 7;
  const tbW = 210;
  const tbH = blockH+30;
  const tbY = (H-tbH)/2;

  ctx.beginPath(); ctx.roundRect(tbX,tbY,tbW,tbH,10);
  fill(ctx,'#131326'); ctx.fill();
  stroke(ctx,C.dimmer,1); ctx.stroke();

  // frozen snowflake indicator
  mono(ctx,10,false); fill(ctx,'#60a5fa88'); ctx.textAlign='center';
  ctx.fillText('❄', tbX+tbW-14, tbY+16);

  // inner layers representation
  const layerCount = 5;
  for(let l=0;l<layerCount;l++){
    const lx = tbX+18 + l*(tbW-36)/layerCount;
    const lw = (tbW-36)/layerCount - 4;
    // attention bars
    for(let r=0;r<3;r++){
      const ry = tbY+14+r*16;
      ctx.beginPath(); ctx.roundRect(lx,ry,lw,11,3);
      const glow = animRunning ? 0.4+0.4*Math.sin(animT*Math.PI*2 - l*0.6) : 0.25;
      fill(ctx, `rgba(52,211,153,${glow*0.3})`); ctx.fill();
      stroke(ctx, `rgba(52,211,153,${glow})`,0.8); ctx.stroke();
    }
  }

  mono(ctx,8.5,true); fill(ctx,C.muted); ctx.textAlign='center';
  ctx.fillText('TRANSFORMER LAYERS', tbX+tbW/2, tbY+tbH-8);
  mono(ctx,8,false); fill(ctx,C.dimmer);
  ctx.fillText('ALL WEIGHTS FROZEN  ❄', tbX+tbW/2, tbY+tbH+12);

  // ── Arrow out ────────────────────────────────────────────────────────────
  const outX = tbX+tbW+5;
  if(animRunning){
    const px2 = outX + animT*35;
    ctx.beginPath(); ctx.arc(px2,midY,4,0,Math.PI*2);
    fill(ctx,'#34d39988'); ctx.fill();
  }
  ctx.beginPath(); ctx.moveTo(outX,midY); ctx.lineTo(outX+35,midY);
  stroke(ctx,C.dim,1.5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(outX+35,midY-5); ctx.lineTo(outX+42,midY); ctx.lineTo(outX+35,midY+5);
  fill(ctx,C.dim); ctx.fill();

  // ── Output box ────────────────────────────────────────────────────────────
  const obX=outX+42, obW=140, obH=54;
  const obY=(H-obH)/2;
  ctx.beginPath(); ctx.roundRect(obX,obY,obW,obH,10);
  fill(ctx,'rgba(52,211,153,0.08)'); ctx.fill();
  stroke(ctx,'rgba(52,211,153,0.5)',1.2); ctx.stroke();
  mono(ctx,9,true); fill(ctx,C.green); ctx.textAlign='center';
  ctx.fillText('OUTPUT LOGITS', obX+obW/2, obY+obH/2-2);
  mono(ctx,8,false); fill(ctx,C.muted);
  ctx.fillText('task prediction', obX+obW/2, obY+obH/2+12);

  // ── Gradient feedback arrow (only soft mode) ───────────────────────────
  if(isSoft && animRunning){
    const gAlpha = 0.3+0.7*Math.sin(animT*Math.PI*2+1);
    const startX = obX, endX = tokenAreaX+softCount*(tokW+tokGap)/2;
    const arcY   = H-14;
    ctx.beginPath();
    ctx.moveTo(startX+10,obY+obH);
    ctx.quadraticCurveTo((startX+endX)/2, arcY+20, endX, tokenY+tokenH);
    ctx.setLineDash([4,4]);
    stroke(ctx,`rgba(249,115,22,${gAlpha})`,1.5); ctx.stroke();
    ctx.setLineDash([]);
    // label
    mono(ctx,8,true); fill(ctx,`rgba(249,115,22,${gAlpha})`); ctx.textAlign='center';
    ctx.fillText('∇ gradient ONLY to soft prompt', (startX+endX)/2, arcY+12);
  }

  // ── Title / legend ────────────────────────────────────────────────────────
  mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='left';
  ctx.fillText(isSoft
    ? '⊕ Soft tokens are free-floating vectors in ℝᵈ — not constrained to real words'
    : '⊕ Hard prompt: every token must be a real vocabulary item', PAD, H-6);

  document.getElementById('archInfo').innerHTML = isSoft
    ? `<span class="hl">Soft Prompt P</span> = [p₁,p₂,...,pₖ] each pᵢ ∈ ℝᵈ&emsp;|&emsp;Prepended to every input&emsp;|&emsp;Model weights: <span class="hl2">ALL FROZEN ❄</span><br>Gradients flow <span class="hl">ONLY</span> to P during training. The model learns to interpret these learned vectors as task instructions.`
    : `<span class="hl3">Hard Prompt</span>: carefully written text prefix — constrained to real words, human-intensive, hits a quality ceiling.<br>Every token maps to a fixed embedding vector. You <span class="hl">cannot</span> move those vectors — they are fixed in the embedding table.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PANEL 2 — Embedding Space
// ─────────────────────────────────────────────────────────────────────────────
const cvE = document.getElementById('cvEmbed');
const ctxE = cvE.getContext('2d');
let embedK = 3;

// fixed word embedding positions (2d projection)
const wordEmbeds = [
  {x:0.15,y:0.28,label:'great',color:'#60a5fa'},
  {x:0.22,y:0.60,label:'movie',color:'#60a5fa'},
  {x:0.78,y:0.70,label:'terrible',color:'#60a5fa'},
  {x:0.68,y:0.20,label:'classify',color:'#60a5fa'},
  {x:0.40,y:0.45,label:'review',color:'#60a5fa'},
  {x:0.55,y:0.72,label:'bad',color:'#60a5fa'},
  {x:0.88,y:0.40,label:'good',color:'#60a5fa'},
  {x:0.30,y:0.80,label:'film',color:'#60a5fa'},
];

// soft token positions (learned, NOT constrained to grid)
const softPositions = [
  {x:0.52,y:0.18},
  {x:0.35,y:0.32},
  {x:0.64,y:0.42},
  {x:0.20,y:0.48},
  {x:0.75,y:0.55},
  {x:0.44,y:0.62},
  {x:0.58,y:0.78},
  {x:0.28,y:0.15},
];

document.getElementById('slK').addEventListener('input',function(){
  embedK=+this.value;
  document.getElementById('slKv').textContent=embedK;
  drawEmbed();
});

function drawEmbed(){
  const W=430,H=230;
  const ctx=ctxE;
  ctx.clearRect(0,0,W,H);
  fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);

  const PAD=30;
  const pw=W-2*PAD, ph=H-2*PAD;

  // grid lines
  for(let i=0;i<=4;i++){
    const gx=PAD+i*pw/4, gy=PAD+i*ph/4;
    ctx.beginPath(); ctx.moveTo(gx,PAD); ctx.lineTo(gx,PAD+ph);
    stroke(ctx,C.dimmer,0.5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(PAD+pw,gy);
    stroke(ctx,C.dimmer,0.5); ctx.stroke();
  }

  // axis labels
  mono(ctx,9,false); fill(ctx,C.muted); ctx.textAlign='center';
  ctx.fillText('← Embedding Dimension 1 →', PAD+pw/2, H-4);
  ctx.save(); ctx.translate(10,PAD+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('← Embedding Dimension 2 →',0,0); ctx.restore();

  // word embeddings (blue, fixed grid)
  wordEmbeds.forEach(w=>{
    const wx=PAD+w.x*pw, wy=PAD+w.y*ph;
    ctx.beginPath(); ctx.arc(wx,wy,5,0,Math.PI*2);
    fill(ctx,'#60a5fa22'); ctx.fill();
    stroke(ctx,'#60a5fa',1.5); ctx.stroke();
    mono(ctx,8,false); fill(ctx,'#60a5fa88'); ctx.textAlign='left';
    ctx.fillText(w.label, wx+7, wy+3);
  });

  // vocabulary boundary (conceptual hull)
  ctx.beginPath();
  ctx.ellipse(PAD+pw*0.5,PAD+ph*0.5, pw*0.42, ph*0.4, 0.2, 0, Math.PI*2);
  ctx.setLineDash([4,4]);
  stroke(ctx,'#60a5fa33',1); ctx.stroke();
  ctx.setLineDash([]);
  mono(ctx,8,false); fill(ctx,'#60a5fa44'); ctx.textAlign='center';
  ctx.fillText('vocabulary manifold', PAD+pw*0.5, PAD+ph*0.92);

  // soft prompts (orange, anywhere)
  for(let i=0;i<embedK;i++){
    const sp=softPositions[i];
    const sx=PAD+sp.x*pw, sy=PAD+sp.y*ph;
    // glow
    const grad=ctx.createRadialGradient(sx,sy,0,sx,sy,18);
    grad.addColorStop(0,'rgba(249,115,22,0.35)');
    grad.addColorStop(1,'rgba(249,115,22,0)');
    ctx.beginPath(); ctx.arc(sx,sy,18,0,Math.PI*2);
    fill(ctx,grad); ctx.fill();
    // dot
    ctx.beginPath(); ctx.arc(sx,sy,6,0,Math.PI*2);
    fill(ctx,C.orange); ctx.fill();
    stroke(ctx,'#fff3',1); ctx.stroke();
    mono(ctx,8.5,true); fill(ctx,C.orange); ctx.textAlign='left';
    ctx.fillText(`p${i+1}`, sx+8, sy+3);
  }

  // legend
  mono(ctx,8,false); ctx.textAlign='left';
  fill(ctx,'#60a5fa'); ctx.fillRect(PAD,PAD+3,8,8);
  fill(ctx,'#60a5fa88'); ctx.fillText('Real word embeddings (fixed)',PAD+12,PAD+11);
  fill(ctx,C.orange); ctx.beginPath(); ctx.arc(PAD+4,PAD+22,4,0,Math.PI*2); ctx.fill();
  fill(ctx,'#f9731688'); ctx.fillText('Soft prompt vectors (learnable, free in ℝᵈ)',PAD+12,PAD+25);

  document.getElementById('embedInfo').innerHTML =
    `<span class="hl">${embedK} soft token${embedK>1?'s':''}</span> shown — each lives at an <span class="hl">arbitrary location</span> in ℝᵈ, <span class="hl3">not</span> constrained to the vocabulary manifold.<br>Real word embeddings (blue) are fixed. Soft tokens (orange) are free parameters — gradient descent moves them wherever needed.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PANEL 3 — Parameter Efficiency
// ─────────────────────────────────────────────────────────────────────────────
const cvP = document.getElementById('cvParam');
const ctxP = cvP.getContext('2d');

const MODEL_SIZES = [
  {label:'T5-Small',  params:60e6,  d:512},
  {label:'T5-Base',   params:250e6, d:768},
  {label:'T5-Large',  params:770e6, d:1024},
  {label:'T5-11B',    params:11e9,  d:1024},
];
let pmModel=1, pmK=20;

document.getElementById('slModel').addEventListener('input',function(){
  pmModel=+this.value;
  document.getElementById('slModelv').textContent=MODEL_SIZES[pmModel].label;
  drawParam();
});
document.getElementById('slPK').addEventListener('input',function(){
  pmK=+this.value;
  document.getElementById('slPKv').textContent=pmK;
  drawParam();
});

function drawParam(){
  const W=430,H=230;
  const ctx=ctxP;
  ctx.clearRect(0,0,W,H);
  fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);

  const m=MODEL_SIZES[pmModel];
  const softParams = pmK * m.d;
  const pct = (softParams / m.params * 100);
  const fullH=H-50, barW=30, PAD=30;

  // total params bar
  const tb={x:PAD, w:barW, h:fullH, y:20};
  ctx.beginPath(); ctx.roundRect(tb.x,tb.y,tb.w,tb.h,6);
  fill(ctx,'#1a1a30'); ctx.fill();
  stroke(ctx,C.dimmer,1); ctx.stroke();
  // soft prompt portion inside
  const softH=Math.max(3, tb.h * Math.min(pct/100, 1));
  ctx.beginPath(); ctx.roundRect(tb.x, tb.y+tb.h-softH, tb.w, softH, 3);
  fill(ctx,'#f97316'); ctx.fill();

  mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
  ctx.fillText(m.label, tb.x+tb.w/2, tb.y+tb.h+12);
  ctx.fillText('total', tb.x+tb.w/2, tb.y+tb.h+22);

  mono(ctx,8.5,true); fill(ctx,C.orange);
  ctx.fillText(`${pct<0.001?pct.toFixed(6):pct.toFixed(4)}%`, tb.x+tb.w/2, tb.y+tb.h-softH-5);

  // big number display
  const numX=PAD+60;

  function fmtN(n){
    if(n>=1e9) return (n/1e9).toFixed(1)+'B';
    if(n>=1e6) return (n/1e6).toFixed(0)+'M';
    if(n>=1e3) return (n/1e3).toFixed(0)+'K';
    return n+'';
  }

  mono(ctx,9,false); fill(ctx,C.muted); ctx.textAlign='left';
  ctx.fillText('Model architecture:', numX, 36);
  mono(ctx,11,true); fill(ctx,C.blue);
  ctx.fillText(`${m.label}   d=${m.d}`, numX, 52);

  mono(ctx,9,false); fill(ctx,C.muted);
  ctx.fillText('Total model parameters:', numX, 76);
  mono(ctx,13,true); fill(ctx,C.text);
  ctx.fillText(fmtN(m.params), numX, 95);

  mono(ctx,9,false); fill(ctx,C.muted);
  ctx.fillText(`Soft prompt:  k=${pmK} × d=${m.d}`, numX, 118);
  mono(ctx,13,true); fill(ctx,C.orange);
  ctx.fillText(fmtN(softParams)+' trainable', numX, 137);

  mono(ctx,9,false); fill(ctx,C.muted);
  ctx.fillText('Percentage of total:', numX, 160);
  mono(ctx,16,true); fill(ctx,C.orange);
  const pctStr = pct < 0.001
    ? pct.toExponential(2)+'%'
    : pct.toFixed(4)+'%';
  ctx.fillText(pctStr, numX, 182);

  // bar compare: full FT vs soft prompt
  const compX=numX+185, compW=W-compX-PAD;
  const bars=[
    {label:'Full FT',  val:m.params,  color:C.red},
    {label:'Soft PT',  val:softParams, color:C.orange},
  ];
  const maxV=m.params;
  bars.forEach((b,i)=>{
    const bY=60+i*70, bH=24;
    const bW=Math.max(2,(b.val/maxV)*(compW));
    ctx.beginPath(); ctx.roundRect(compX,bY,compW,bH,4);
    fill(ctx,'#1a1a30'); ctx.fill();
    ctx.beginPath(); ctx.roundRect(compX,bY,bW,bH,4);
    fill(ctx,b.color+'33'); ctx.fill();
    stroke(ctx,b.color,1); ctx.stroke();
    mono(ctx,8.5,true); fill(ctx,b.color); ctx.textAlign='left';
    ctx.fillText(b.label, compX+6, bY+16);
    ctx.textAlign='right';
    ctx.fillText(fmtN(b.val), compX+compW-6, bY+16);
  });

  mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='left';
  ctx.fillText('Parameter comparison', compX, 50);

  document.getElementById('paramInfo').innerHTML=
    `<span class="hl">Soft Prompt</span>: P ∈ ℝ^{${pmK} × ${m.d}} = <span class="hl">${fmtN(softParams)}</span> params &emsp;vs&emsp; `+
    `<span class="hl3">Full fine-tuning</span>: <span class="hl3">${fmtN(m.params)}</span> params<br>`+
    `Only <span class="hl">${pctStr}</span> of model weights are touched. At 11B scale, matches full FT quality.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PANEL 4 — Family Tree
// ─────────────────────────────────────────────────────────────────────────────
const cvF = document.getElementById('cvFamily');
const ctxF = cvF.getContext('2d');
let famMode = 'hard';

const FAM_INFO = {
  hard:   { name:'Hard Prompting', color:'#60a5fa', trainable:0, layers:'input (text only)', note:'No training. Carefully written text prefixes. Constrained to vocabulary.' },
  pt:     { name:'Prompt Tuning',  color:'#f97316', trainable:1, layers:'input embedding only', note:'Soft tokens prepended to input ONLY. Fewest params. Best simplicity-to-quality ratio at ≥1B scale.' },
  prefix: { name:'Prefix Tuning', color:'#c084fc', trainable:2, layers:'K,V of EVERY attention layer', note:'Soft tokens added to Key and Value projections at every layer. More steering power, more params.' },
  p2:     { name:'P-Tuning v2',   color:'#facc15', trainable:3, layers:'all layers (deep prefix)', note:'Layer-specific prefix parameters on every layer. Most powerful prompt-based method.' },
};

function setFamily(m){
  famMode=m;
  ['Hard','PT','Pre','P2'].forEach(x=>{
    document.getElementById('btnFam'+x).classList.toggle('active', m===x.toLowerCase());
  });
  drawFamily();
}
// fix button ids
['hard','pt','prefix','p2'].forEach(m=>{
  const id = m==='hard'?'btnFamHard':m==='pt'?'btnFamPT':m==='prefix'?'btnFamPre':'btnFamP2';
  document.getElementById(id).onclick=()=>setFamily(m);
});

function drawFamily(){
  const W=900,H=240;
  const ctx=ctxF;
  ctx.clearRect(0,0,W,H);
  fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);

  // Show transformer cross-section: embedding + N layers + output
  const PAD=30;
  const layerCount=6;
  const layerW=90, layerH=34, layerGap=10;
  const startX = PAD+60;
  const totalW = layerCount*(layerW+layerGap)+layerW;
  const midY = H/2;

  const fi = FAM_INFO[famMode];

  // INPUT EMBEDDING block
  const embX=PAD, embY=midY-22, embW=55, embH=44;
  ctx.beginPath(); ctx.roundRect(embX,embY,embW,embH,8);
  fill(ctx,'#13132a'); ctx.fill();
  stroke(ctx, famMode==='pt'||famMode==='hard' ? fi.color : C.dim, famMode==='pt'?2:1); ctx.stroke();
  mono(ctx,8,true); fill(ctx, famMode==='pt'||famMode==='hard' ? fi.color : C.muted); ctx.textAlign='center';
  ctx.fillText('EMBED', embX+embW/2, embY+embH/2-4);
  ctx.fillText('LAYER', embX+embW/2, embY+embH/2+7);

  // soft tokens at input (prompt tuning and hard)
  if(famMode==='hard'){
    // text tokens above embed
    ['t₁','t₂','...','tₙ'].forEach((t,i)=>{
      const tx=embX-6+i*14, ty=embY-22;
      ctx.beginPath(); ctx.roundRect(tx,ty,11,14,3);
      fill(ctx,'#60a5fa22'); ctx.fill(); stroke(ctx,C.blue,0.8); ctx.stroke();
      mono(ctx,7,false); fill(ctx,C.blue); ctx.textAlign='center';
      ctx.fillText(t,tx+5.5,ty+10);
    });
  }
  if(famMode==='pt'){
    ['p₁','p₂','p₃'].forEach((p,i)=>{
      const tx=embX-4+i*16, ty=embY-26;
      ctx.beginPath(); ctx.roundRect(tx,ty,13,16,3);
      fill(ctx,'#f9731633'); ctx.fill(); stroke(ctx,C.orange,1.5); ctx.stroke();
      mono(ctx,7.5,true); fill(ctx,C.orange); ctx.textAlign='center';
      ctx.fillText(p,tx+6.5,ty+11);
    });
    mono(ctx,7,false); fill(ctx,C.orange); ctx.textAlign='center';
    ctx.fillText('soft', embX+26, embY-30);
  }

  // Transformer layers
  for(let l=0;l<layerCount;l++){
    const lx=startX+l*(layerW+layerGap);
    const ly=midY-layerH/2-5;

    // outer frame
    ctx.beginPath(); ctx.roundRect(lx,ly,layerW,layerH+10,8);
    fill(ctx,'#13132a'); ctx.fill();
    stroke(ctx,C.dim,0.8); ctx.stroke();

    // Attention sublayer
    const attnY=ly+4, attnH=14;
    ctx.beginPath(); ctx.roundRect(lx+4,attnY,layerW-8,attnH,4);

    const showPrefix = (famMode==='prefix'||famMode==='p2');
    const attnColor = showPrefix ? fi.color : '#34d39933';
    const attnBorder = showPrefix ? fi.color : '#34d39966';
    fill(ctx,attnColor+'22'); ctx.fill(); stroke(ctx,attnBorder,showPrefix?1.5:0.8); ctx.stroke();
    mono(ctx,7,true); fill(ctx,showPrefix?fi.color:C.green); ctx.textAlign='center';
    ctx.fillText('Attention K,V', lx+layerW/2, attnY+attnH/2+3);

    // prefix tokens on K,V
    if(showPrefix && l<(famMode==='p2'?layerCount:layerCount)){
      const px = lx+layerW/2-14;
      const py = attnY-14;
      ['k̃','ṽ'].forEach((s,si)=>{
        const sx=px+si*15, sy2=py;
        ctx.beginPath(); ctx.roundRect(sx,sy2,12,11,2);
        fill(ctx,fi.color+'33'); ctx.fill(); stroke(ctx,fi.color,1); ctx.stroke();
        mono(ctx,7,true); fill(ctx,fi.color); ctx.textAlign='center';
        ctx.fillText(s,sx+6,sy2+8);
      });
    }

    // FFN sublayer
    const ffnY=attnY+attnH+2, ffnH=14;
    ctx.beginPath(); ctx.roundRect(lx+4,ffnY,layerW-8,ffnH,4);
    fill(ctx,'#3f3f5622'); ctx.fill(); stroke(ctx,C.dim,0.5); ctx.stroke();
    mono(ctx,7,false); fill(ctx,C.muted); ctx.textAlign='center';
    ctx.fillText('FFN', lx+layerW/2, ffnY+ffnH/2+3);

    // layer label
    mono(ctx,7.5,false); fill(ctx,C.dim); ctx.textAlign='center';
    ctx.fillText(`L${l+1}`, lx+layerW/2, ly+layerH+18);

    // connections
    if(l>0){
      const px=startX+l*(layerW+layerGap)-layerGap;
      ctx.beginPath(); ctx.moveTo(px,midY); ctx.lineTo(lx,midY);
      stroke(ctx,C.dimmer,1); ctx.stroke();
    }
  }

  // Arrow from embed to L1
  ctx.beginPath(); ctx.moveTo(embX+embW,midY); ctx.lineTo(startX,midY);
  stroke(ctx,C.dim,1.2); ctx.stroke();

  // Output head
  const outX=startX+layerCount*(layerW+layerGap)+4;
  ctx.beginPath(); ctx.moveTo(outX-2,midY); ctx.lineTo(outX+40,midY);
  stroke(ctx,C.dim,1.2); ctx.stroke();
  ctx.beginPath(); ctx.roundRect(outX+40,midY-22,60,44,8);
  fill(ctx,'rgba(52,211,153,0.1)'); ctx.fill();
  stroke(ctx,C.green,1); ctx.stroke();
  mono(ctx,8,true); fill(ctx,C.green); ctx.textAlign='center';
  ctx.fillText('OUTPUT', outX+70, midY-4); ctx.fillText('HEAD', outX+70, midY+10);

  // Bottom labels
  const labs=[
    {x:embX+embW/2,  label:'Embedding',    sub:'input layer'},
    {x:startX+layerCount*(layerW+layerGap)/2-20, label:`${layerCount} Transformer Layers`, sub:'attention + FFN'},
    {x:outX+70,      label:'Output',       sub:'head'},
  ];
  labs.forEach(l=>{
    mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
  });

  // Annotation
  mono(ctx,9,true); fill(ctx,fi.color); ctx.textAlign='left';
  ctx.fillText(`◉ ${fi.name}`, PAD, H-14);
  mono(ctx,8,false); fill(ctx,C.muted);
  ctx.fillText(`  Soft tokens at: ${fi.layers}`, PAD+160, H-14);

  document.getElementById('familyInfo').innerHTML=
    `<span style="color:${fi.color};font-weight:700">${fi.name}</span>: ${fi.note}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PANEL 5 — Accuracy vs Scale
// ─────────────────────────────────────────────────────────────────────────────
const cvS = document.getElementById('cvScale');
const ctxS = cvS.getContext('2d');

const scaleData = [
  // [model_size_M, full_ft_acc, prompt_tune_acc, few_shot_acc]
  [60,   92, 56, 45],
  [250,  94, 72, 62],
  [770,  95, 88, 76],
  [3000, 96, 93, 86],
  [11000,96, 96, 91],
];

function drawScale(){
  const W=430,H=230,ctx=ctxS;
  ctx.clearRect(0,0,W,H);
  fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);

  const PAD=40, PB=35, PR=30;
  const cW=W-PAD-PR, cH=H-PAD-PB;

  // axes
  ctx.beginPath(); ctx.moveTo(PAD,PAD); ctx.lineTo(PAD,PAD+cH); ctx.lineTo(PAD+cW,PAD+cH);
  stroke(ctx,C.dimmer,1.2); ctx.stroke();

  const xVals=scaleData.map(d=>Math.log10(d[0]));
  const xMin=xVals[0], xMax=xVals[xVals.length-1];
  const xS=v=>(PAD + (Math.log10(v)-xMin)/(xMax-xMin)*cW);
  const yS=v=>(PAD+cH - (v-40)/(60)*cH);

  // grid
  [60,70,80,90,100].forEach(y=>{
    const gy=yS(y);
    ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(PAD+cW,gy);
    stroke(ctx,C.dimmer,0.5); ctx.stroke();
    mono(ctx,7.5,false); fill(ctx,C.muted); ctx.textAlign='right';
    ctx.fillText(y+'%',PAD-4,gy+3);
  });

  // x labels
  scaleData.forEach(d=>{
    mono(ctx,7.5,false); fill(ctx,C.muted); ctx.textAlign='center';
    const label = d[0]>=1000?Math.round(d[0]/1000)+'B':d[0]+'M';
    ctx.fillText(label, xS(d[0]), PAD+cH+12);
  });

  mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
  ctx.fillText('Model size (log scale)', PAD+cW/2, H-4);
  ctx.save(); ctx.translate(10,PAD+cH/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Accuracy (%)',0,0); ctx.restore();

  const series=[
    {idx:1, color:C.red,    label:'Full Fine-Tuning'},
    {idx:2, color:C.orange, label:'Prompt Tuning'},
    {idx:3, color:C.blue,   label:'Few-Shot (GPT-3 style)'},
  ];
  series.forEach(s=>{
    ctx.beginPath();
    scaleData.forEach((d,i)=>{
      const px=xS(d[0]),py=yS(d[s.idx]);
      i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
    });
    stroke(ctx,s.color,2); ctx.stroke();
    // dots
    scaleData.forEach(d=>{
      ctx.beginPath(); ctx.arc(xS(d[0]),yS(d[s.idx]),4,0,Math.PI*2);
      fill(ctx,s.color); ctx.fill();
    });
  });

  // legend
  series.forEach((s,i)=>{
    const lx=PAD+5, ly=PAD+8+i*16;
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx+18,ly);
    stroke(ctx,s.color,2); ctx.stroke();
    ctx.beginPath(); ctx.arc(lx+9,ly,3,0,Math.PI*2);
    fill(ctx,s.color); ctx.fill();
    mono(ctx,8,false); fill(ctx,s.color); ctx.textAlign='left';
    ctx.fillText(s.label, lx+22, ly+3);
  });

  // annotation at convergence
  const cx=xS(11000), cy=yS(96);
  ctx.beginPath(); ctx.moveTo(cx,cy-8); ctx.lineTo(cx+30,cy-28);
  stroke(ctx,'#f9731688',1); ctx.stroke();
  mono(ctx,7.5,true); fill(ctx,C.orange); ctx.textAlign='left';
  ctx.fillText('≈ Full FT quality', cx+32, cy-26);
  ctx.fillText('at 11B scale!', cx+32, cy-14);

  document.getElementById('scaleInfo').innerHTML=
    `At small scales, Prompt Tuning <span class="hl">underperforms</span> full fine-tuning. At <span class="hl">≥1B params</span>, the gap closes rapidly.<br>At <span class="hl">11B+</span>, Prompt Tuning matches full FT with <span class="hl">0.001% of parameters</span>.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PANEL 6 — Gradient Flow / Training Loop
// ─────────────────────────────────────────────────────────────────────────────
const cvG = document.getElementById('cvGrad');
const ctxG = cvG.getContext('2d');
let gradRunning=false, gradRaf=null, gradT=0, gradEpoch=0, gradLoss=[];
const INIT_LOSS=2.8;

function resetGrad(){
  gradRunning=false; cancelAnimationFrame(gradRaf);
  gradT=0; gradEpoch=0; gradLoss=[];
  document.getElementById('btnGradPlay').textContent='▶ Run Training';
  drawGrad();
}

function toggleGrad(){
  if(gradRunning){
    gradRunning=false; cancelAnimationFrame(gradRaf);
    document.getElementById('btnGradPlay').textContent='▶ Run Training';
    return;
  }
  gradRunning=true;
  document.getElementById('btnGradPlay').textContent='⏸ Pause';
  function loop(){
    gradT=(gradT+0.02)%1;
    if(gradT<0.02){ gradEpoch++; gradLoss.push(Math.max(0.08, INIT_LOSS*Math.exp(-gradEpoch*0.18)+0.08)); if(gradLoss.length>80) gradLoss.shift(); }
    drawGrad();
    if(gradRunning) gradRaf=requestAnimationFrame(loop);
    else document.getElementById('btnGradPlay').textContent='▶ Run Training';
  }
  loop();
}

function drawGrad(){
  const W=430,H=230,ctx=ctxG;
  ctx.clearRect(0,0,W,H);
  fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);

  const midY=H*0.44;

  // ── Architecture diagram ─────────────────────────────────────────────────
  const blks=[
    {x:20,  w:52, h:38, label:'Soft\nPrompt', color:C.orange, trainable:true},
    {x:82,  w:52, h:38, label:'Embed\nLayer',  color:C.dim,   trainable:false},
    {x:144, w:62, h:38, label:'Transformer\nLayers', color:C.green, trainable:false},
    {x:216, w:52, h:38, label:'Output\nHead',  color:C.blue,  trainable:false},
  ];

  blks.forEach(b=>{
    const by=midY-b.h/2;
    ctx.beginPath(); ctx.roundRect(b.x,by,b.w,b.h,7);
    fill(ctx, b.trainable ? b.color+'22' : '#13132a'); ctx.fill();
    stroke(ctx, b.trainable ? b.color : C.dim, b.trainable?2:0.8); ctx.stroke();
    mono(ctx,7.5, b.trainable); fill(ctx,b.trainable?b.color:C.muted); ctx.textAlign='center';
    const lines=b.label.split('\n');
    lines.forEach((l,i)=>ctx.fillText(l, b.x+b.w/2, by+b.h/2+(i-lines.length/2+0.5)*10));
    if(!b.trainable){
      mono(ctx,7,false); fill(ctx,'#60a5fa55'); ctx.textAlign='center';
      ctx.fillText('❄frozen',b.x+b.w/2,by+b.h+10);
    }
  });

  // arrows (forward pass)
  blks.forEach((b,i)=>{
    if(i===blks.length-1) return;
    const nb=blks[i+1];
    const ax=b.x+b.w, ay=midY;
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(nb.x,ay);
    stroke(ctx,C.dimmer,1); ctx.stroke();
    const pulse = gradRunning && gradT < 0.5 ? gradT*2 : 0;
    if(gradRunning){
      const px=ax+pulse*(nb.x-ax);
      ctx.beginPath(); ctx.arc(px,ay,3.5,0,Math.PI*2);
      fill(ctx,'#34d39988'); ctx.fill();
    }
  });

  // Loss computation
  const lossX=285, lossY=midY-18, lossW=50, lossH=36;
  ctx.beginPath(); ctx.roundRect(lossX,lossY,lossW,lossH,7);
  const lossVal = gradLoss.length>0 ? gradLoss[gradLoss.length-1] : INIT_LOSS;
  const lossColor = lossVal < 0.5 ? C.green : lossVal<1.5?C.yellow:C.red;
  fill(ctx,lossColor+'22'); ctx.fill(); stroke(ctx,lossColor,1.5); ctx.stroke();
  mono(ctx,8,true); fill(ctx,lossColor); ctx.textAlign='center';
  ctx.fillText('LOSS', lossX+lossW/2, lossY+14);
  ctx.fillText(lossVal.toFixed(3), lossX+lossW/2, lossY+28);

  ctx.beginPath(); ctx.moveTo(blks[3].x+blks[3].w,midY); ctx.lineTo(lossX,midY);
  stroke(ctx,C.dimmer,1); ctx.stroke();

  // Gradient back arrow (only to soft prompt!)
  if(gradRunning && gradT > 0.5){
    const gProg=(gradT-0.5)*2;
    const startX=lossX, endX=blks[0].x+blks[0].w/2;
    const arcH=H*0.55;
    ctx.beginPath();
    const cp1x=(startX+endX)/2, cp1y=arcH;
    const px=startX+(endX-startX)*gProg;
    // just draw the gradient path up to progress
    ctx.save();
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    // quadratic to point
    for(let s=0;s<gProg;s+=0.02){
      const qx=Math.pow(1-s,2)*startX+2*(1-s)*s*cp1x+s*s*endX;
      const qy=Math.pow(1-s,2)*midY+2*(1-s)*s*arcH+s*s*(midY+28);
      if(s===0) ctx.moveTo(qx,qy); else ctx.lineTo(qx,qy);
    }
    stroke(ctx,'#f97316bb',1.8); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // ∇ label
    const labProg=Math.min(gProg,0.5);
    const lqx=Math.pow(1-labProg,2)*startX+2*(1-labProg)*labProg*cp1x+labProg*labProg*endX;
    const lqy=Math.pow(1-labProg,2)*midY+2*(1-labProg)*labProg*arcH+labProg*labProg*(midY+28);
    ctx.beginPath(); ctx.arc(lqx,lqy,4,0,Math.PI*2);
    fill(ctx,C.orange+'cc'); ctx.fill();
  }

  mono(ctx,7.5,true); fill(ctx,C.orange); ctx.textAlign='center';
  ctx.fillText('∇ gradient → only here', blks[0].x+blks[0].w/2, midY+28);

  // ── Loss curve ────────────────────────────────────────────────────────────
  const cX=350, cY=18, cW=W-cX-10, cH=H-40;
  fill(ctx,'#0d0d1a'); ctx.fillRect(cX,cY,cW,cH);
  stroke(ctx,C.dimmer,0.8); ctx.strokeRect(cX,cY,cW,cH);

  mono(ctx,7.5,false); fill(ctx,C.muted); ctx.textAlign='center';
  ctx.fillText('Training Loss', cX+cW/2, cY+10);

  if(gradLoss.length>1){
    const maxL=INIT_LOSS, minL=0;
    ctx.beginPath();
    gradLoss.forEach((l,i)=>{
      const px=cX+1+(i/(Math.max(gradLoss.length-1,1)))*(cW-2);
      const py=cY+cH-2-(l-minL)/(maxL-minL)*(cH-4);
      i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
    });
    stroke(ctx,C.orange,2); ctx.stroke();
    gradLoss.forEach((l,i)=>{}); // lineTo done
    ctx.lineTo(cX+1+(gradLoss.length-1)/(Math.max(gradLoss.length-1,1))*(cW-2), cY+cH-2);
    ctx.lineTo(cX+1,cY+cH-2); ctx.closePath();
    fill(ctx,'rgba(249,115,22,0.08)'); ctx.fill();
  }

  mono(ctx,7,false); fill(ctx,C.muted); ctx.textAlign='center';
  ctx.fillText('epoch →', cX+cW/2, cY+cH+10);

  document.getElementById('gradInfo').innerHTML=
    `Epoch <span class="hl">${gradEpoch}</span> &nbsp;|&nbsp; `+
    `Loss: <span class="${lossVal<0.5?'hl2':lossVal<1.5?'hl4':'hl'}">${lossVal.toFixed(4)}</span><br>`+
    `Gradients flow backwards through the frozen transformer but <span class="hl">only update</span> the soft prompt P. All model weights remain <span class="hl2">unchanged ❄</span>.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// QUIZ
// ─────────────────────────────────────────────────────────────────────────────
const QUIZ=[
  {q:'What are "soft prompts" in Prompt Tuning?',
   opts:['Carefully written text instructions','Learnable floating-point vectors prepended to input','Low-rank weight matrices added to attention','Extra layers added on top of the model'],
   ans:1,
   fb:['Incorrect — that describes hard prompting.','✓ Correct! Soft prompts are free-floating vectors pᵢ ∈ ℝᵈ — not constrained to real words.','Incorrect — that describes LoRA.','Incorrect — no new layers are added.']},
  {q:'Which model weights are updated during Prompt Tuning?',
   opts:['All weights (like full fine-tuning)','Only the attention Q,K,V matrices','Only the soft prompt matrix P','The embedding table E only'],
   ans:2,
   fb:['Incorrect — full FT updates all weights.','Incorrect — that is closer to LoRA/Prefix Tuning.','✓ Correct! ONLY P ∈ ℝ^{k × d} receives gradient updates.','Incorrect — the embedding table is frozen.']},
  {q:'At what model scale does Prompt Tuning match full fine-tuning quality?',
   opts:['60M parameters','250M parameters','≥1B parameters','It never matches full fine-tuning'],
   ans:2,
   fb:['Incorrect — at 60M the gap is large.','Incorrect — still a noticeable gap at 250M.','✓ Correct! Lester et al. showed parity at ≥1B, especially ≥11B.','Incorrect — at sufficient scale it matches.']},
  {q:'How does Prefix Tuning differ from Prompt Tuning?',
   opts:['Prefix Tuning trains the full model','Prefix Tuning adds soft tokens to every attention K,V layer','Prefix Tuning uses text tokens instead of vectors','Prefix Tuning requires more data'],
   ans:1,
   fb:['Incorrect.','✓ Correct! Prefix Tuning prepends soft tokens to K and V projections at EVERY transformer layer, not just input.','Incorrect — both use continuous vectors.','Incorrect — this is not the distinction.']},
  {q:'If d=1024 and k=20, how many trainable parameters does Prompt Tuning add?',
   opts:['1,024','20,480','204,800','1,048,576'],
   ans:1,
   fb:['Incorrect — that is d alone.','✓ Correct! P ∈ ℝ^{20 × 1024} = 20 × 1024 = 20,480 parameters.','Incorrect.','Incorrect.']},
  {q:'Why can\'t hard prompts be placed "anywhere" in embedding space?',
   opts:['Hard prompts are too long','Every token must correspond to a real vocabulary item with a fixed embedding','Hard prompts cannot attend to input tokens','Hard prompts update model weights'],
   ans:1,
   fb:['Incorrect — length is not the constraint.','✓ Correct! Tokens map to fixed embedding vectors. You cannot place them at arbitrary ℝᵈ locations.','Incorrect.','Incorrect — hard prompts involve no training.']},
];

let quizAnswers=new Array(QUIZ.length).fill(null);
let quizDone=0;

function buildQuiz(){
  const grid=document.getElementById('quizGrid');
  grid.innerHTML='';
  QUIZ.forEach((q,qi)=>{
    const card=document.createElement('div'); card.className='q-card';
    card.innerHTML=`<div class="q-num">Question ${qi+1}</div><div class="q-prompt">${q.q}</div>`;
    q.opts.forEach((o,oi)=>{
      const btn=document.createElement('button'); btn.className='q-opt';
      btn.textContent=o;
      btn.onclick=()=>answerQ(qi,oi,btn,card);
      card.appendChild(btn);
    });
    const fb=document.createElement('div'); fb.className='q-feedback'; fb.id=`fb${qi}`;
    card.appendChild(fb);
    grid.appendChild(card);
  });
}

function answerQ(qi,oi,btn,card){
  if(quizAnswers[qi]!==null) return;
  quizAnswers[qi]=oi;
  quizDone++;
  const q=QUIZ[qi];
  const correct=oi===q.ans;
  btn.classList.add(correct?'correct':'wrong');
  card.querySelectorAll('.q-opt').forEach((b,i)=>{
    b.disabled=true;
    if(i===q.ans) b.classList.add('correct');
  });
  const fb=document.getElementById(`fb${qi}`);
  fb.textContent=q.fb[oi]; fb.className=`q-feedback show ${correct?'ok':'bad'}`;

  // score
  const sc=quizAnswers.filter(a=>a!==null&&QUIZ[quizAnswers.indexOf(a)]&&a===QUIZ[quizAnswers.indexOf(a)].ans).length;
  const correctCount = QUIZ.filter((q,i)=>quizAnswers[i]===q.ans).length;
  const bar=document.getElementById('scoreBar');
  bar.innerHTML=`<span id="scoreText">Score: <span style="color:${correctCount===QUIZ.length?'#34d399':'#f97316'};font-weight:700">${correctCount}/${QUIZ.length}</span></span>`;
  QUIZ.forEach((_,i)=>{
    const pip=document.createElement('span'); pip.className='pip'+(quizAnswers[i]===null?'':quizAnswers[i]===QUIZ[i].ans?' ok':' bad');
    bar.appendChild(pip);
  });
  if(quizDone===QUIZ.length){
    const t=document.createElement('span');
    t.style.cssText='margin-left:8px;color:#34d399;font-weight:700;';
    t.textContent=correctCount===QUIZ.length?'🎉 Perfect! You understand Prompt Tuning!':correctCount>=4?'👍 Great work!':'Keep reviewing the panels above.';
    bar.appendChild(t);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────
drawArch();
drawEmbed();
document.getElementById('slModelv').textContent = MODEL_SIZES[pmModel].label;
drawParam();
drawFamily();
drawScale();
drawGrad();
buildQuiz();

// Ctrl+Scroll zoom
var ZOOM=1.0, zoomTimer=null;
function applyZoom(){
  document.body.style.zoom=ZOOM;
  clearTimeout(zoomTimer);
  var t=document.getElementById('zt');
  if(!t){t=document.createElement('div');t.id='zt';t.style.cssText='position:fixed;bottom:18px;right:18px;background:#0f0f1a;border:1px solid #f97316;color:#f97316;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;padding:6px 12px;border-radius:7px;z-index:9999;pointer-events:none;transition:opacity .3s;';document.body.appendChild(t);}
  t.textContent='zoom '+Math.round(ZOOM*100)+'%'; t.style.opacity='1';
  zoomTimer=setTimeout(()=>t.style.opacity='0',1200);
}
document.addEventListener('wheel',e=>{
  if(!e.ctrlKey&&!e.metaKey)return; e.preventDefault();
  ZOOM=Math.min(3,Math.max(0.4,Math.round((ZOOM+(e.deltaY>0?-0.05:0.05))*100)/100));
  applyZoom();
},{passive:false});
</script>
</body>
</html>
"""

PROMPT_TUNING_VISUAL_HEIGHT = 1800