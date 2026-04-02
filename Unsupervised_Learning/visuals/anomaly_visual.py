VISUAL_HEIGHT = 1300
VISUAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0;
       padding: 20px; }
h2   { color: #f87171; margin-bottom: 4px; }
.subtitle { color: #64748b; margin-bottom: 22px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #1e2130; border-radius: 12px; padding: 18px;
        border: 1px solid #2d3148; }
.card h3 { color: #f87171; margin: 0 0 10px; font-size: 0.9em;
           text-transform: uppercase; letter-spacing: 0.05em; }
canvas { display: block; }
.params { background: #12141f; padding: 8px 12px; border-radius: 8px;
          font-size: 0.81em; color: #94a3b8; margin: 8px 0; line-height: 1.6; }
.pv { color: #f87171; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 110px; }
input[type=range] { accent-color: #f87171; flex: 1; }
.vb { font-size: 0.8em; color: #f87171; min-width: 40px; }
.btn-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 8px; }
button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
         border-radius: 6px; padding: 5px 12px; cursor: pointer;
         font-size: 0.8em; transition: background 0.15s; }
button:hover { background: #3d4168; }
button.active { background: #f87171; color: #0f1117; }
</style>
</head>
<body>
<h2>&#128680; Anomaly Detection Explorer</h2>
<p class="subtitle">Isolation Forest, Z-Score, LOF — finding the outliers</p>

<div class="grid">

  <!-- Panel 1: Isolation Forest live tree visualiser -->
  <div class="card">
    <h3>Isolation Forest — Path Length</h3>
    <div class="params">
      Each point is coloured by its anomaly score: <span style="color:#f87171">red</span> = high score (anomaly),
      <span style="color:#38bdf8">blue</span> = low score (normal).
      Use sliders to change contamination and see which points get flagged.
    </div>
    <canvas id="cvIF" width="340" height="240" style="cursor:crosshair"></canvas>
    <div class="slider-row">
      <label>Contamination</label>
      <input type="range" id="contSlider" min="1" max="30" step="1" value="8">
      <span class="vb" id="contVal">8%</span>
    </div>
    <div class="params" id="ifInfo">Hover a point to see its anomaly score.</div>
  </div>

  <!-- Panel 2: Score distribution histogram -->
  <div class="card">
    <h3>Anomaly Score Distribution</h3>
    <div class="params">
      Histogram of isolation scores. Drag the <span style="color:#fbbf24">threshold line</span> to
      change the decision boundary. Blue = normal, red = anomaly (true labels shown).
    </div>
    <canvas id="cvHist" width="340" height="240" style="cursor:ew-resize"></canvas>
    <div class="params" id="histInfo">—</div>
  </div>

  <!-- Panel 3: LOF vs Z-score comparison -->
  <div class="card">
    <h3>LOF vs Z-Score on Multi-Density Data</h3>
    <div class="params">
      Dense cluster (left) + sparse cluster (right) + 3 true outliers.
      Z-score incorrectly flags the sparse cluster. LOF only flags true outliers.
    </div>
    <canvas id="cvLOF" width="340" height="240"></canvas>
    <div class="btn-row">
      <button onclick="showLOF('lof')"  id="btnLOF"  class="active">LOF</button>
      <button onclick="showLOF('z')"    id="btnZ">Z-Score</button>
      <button onclick="showLOF('true')" id="btnTrue">True Labels</button>
    </div>
    <div class="params" id="lofInfo">LOF correctly identifies only the 3 isolated outliers.</div>
  </div>

  <!-- Panel 4: Precision-recall tradeoff -->
  <div class="card">
    <h3>Precision vs Recall Tradeoff</h3>
    <div class="params">
      As you flag more points (lower threshold), recall rises but precision falls.
      Move slider to find the F1-optimal operating point.
    </div>
    <canvas id="cvPR" width="340" height="200"></canvas>
    <div class="slider-row">
      <label>Flagged points</label>
      <input type="range" id="prSlider" min="1" max="30" step="1" value="10">
      <span class="vb" id="prVal">10</span>
    </div>
    <div class="params" id="prInfo">—</div>
  </div>

</div>

<script>
let sd = 42;
const rng = () => { sd=(sd*1664525+1013904223)>>>0; return sd/4294967296; };
const gauss = () => { const u=rng()||1e-9,v=rng(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

// ── Generate dataset: normal cluster + outliers ────────────────────────────
function genDataset(){
  sd=42;
  const pts=[], labels=[];
  for(let i=0;i<80;i++){
    pts.push([170+gauss()*30, 120+gauss()*30]); labels.push(0);
  }
  for(let i=0;i<10;i++){
    const angle=rng()*2*Math.PI, r=110+rng()*60;
    pts.push([170+r*Math.cos(angle), 120+r*Math.sin(angle)]); labels.push(1);
  }
  return {pts,labels};
}
const {pts:DATA, labels:LABELS} = genDataset();
const N = DATA.length;

// ── Mini isolation forest (JS) ─────────────────────────────────────────────
function cN(n){ if(n<=1)return 0; if(n==2)return 1;
  return 2*(Math.log(n-1)+0.5772156649)-2*(n-1)/n; }

function buildTree(pts, d, md){
  if(pts.length<=1||d>=md) return {leaf:true,size:pts.length};
  const dims=pts[0].length, f=Math.floor(rng()*dims);
  const col=pts.map(p=>p[f]);
  const mn=Math.min(...col),mx=Math.max(...col);
  if(mn===mx) return {leaf:true,size:pts.length};
  const sv=mn+rng()*(mx-mn);
  const L=pts.filter(p=>p[f]<sv), R=pts.filter(p=>p[f]>=sv);
  if(!L.length||!R.length) return {leaf:true,size:pts.length};
  return {leaf:false,f,sv,L:buildTree(L,d+1,md),R:buildTree(R,d+1,md)};
}

function pathLen(x,node,l){
  if(node.leaf) return l+cN(node.size);
  return x[node.f]<node.sv?pathLen(x,node.L,l+1):pathLen(x,node.R,l+1);
}

// Pre-compute IF scores
sd=99;
const PSI=32, N_TREES=40, MD=Math.ceil(Math.log2(PSI));
const trees=[];
for(let t=0;t<N_TREES;t++){
  const sub=[...DATA].sort(()=>rng()-0.5).slice(0,PSI);
  trees.push(buildTree(sub,0,MD));
}
const IF_SCORES=DATA.map(x=>{
  const avg=trees.reduce((s,t)=>s+pathLen(x,t,0),0)/N_TREES;
  return Math.pow(2,-avg/cN(PSI));
});
const scoreOrder=[...IF_SCORES.keys()].sort((a,b)=>IF_SCORES[b]-IF_SCORES[a]);

// ── Panel 1: Scatter with IF scores ───────────────────────────────────────
const cv1=document.getElementById('cvIF'),ctx1=cv1.getContext('2d');
function scoreToCol(s){
  const t=Math.min((s-0.4)/0.4,1);
  const r=Math.round(56+t*(248-56));
  const g=Math.round(189-t*(189-113));
  const b=Math.round(248-t*(248-113));
  return `rgb(${r},${g},${b})`;
}

function drawIF(){
  const W=340,H=240; ctx1.clearRect(0,0,W,H);
  const cont=parseInt(document.getElementById('contSlider').value)/100;
  const nFlag=Math.max(1,Math.round(cont*N));
  const flagged=new Set(scoreOrder.slice(0,nFlag));

  DATA.forEach((p,i)=>{
    const s=IF_SCORES[i];
    ctx1.beginPath(); ctx1.arc(p[0],p[1],flagged.has(i)?6:4,0,2*Math.PI);
    ctx1.fillStyle=scoreToCol(s); ctx1.fill();
    if(flagged.has(i)){ctx1.strokeStyle='#f87171';ctx1.lineWidth=1.5;ctx1.stroke();}
  });

  // Legend
  ctx1.font='9px sans-serif'; ctx1.textAlign='left';
  ctx1.fillStyle='#38bdf8'; ctx1.fillText('● Low score (normal)',10,H-18);
  ctx1.fillStyle='#f87171'; ctx1.fillText('● High score (anomaly)',10,H-6);
  document.getElementById('contVal').textContent=
    parseInt(document.getElementById('contSlider').value)+'%';
  const tp=scoreOrder.slice(0,nFlag).filter(i=>LABELS[i]===1).length;
  document.getElementById('ifInfo').innerHTML=
    `Flagging <span class="pv">${nFlag}</span> points &nbsp;|&nbsp; ` +
    `True anomalies caught: <span class="pv">${tp}/10</span>`;
}
cv1.addEventListener('mousemove',e=>{
  const r=cv1.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  let best=-1,bd=Infinity;
  DATA.forEach((p,i)=>{const d=Math.hypot(p[0]-mx,p[1]-my); if(d<bd){bd=d;best=i;}});
  if(bd<12) document.getElementById('ifInfo').innerHTML=
    `Point ${best} | score=<span class="pv">${IF_SCORES[best].toFixed(4)}</span> | ` +
    `true label: <span class="pv">${LABELS[best]===1?'ANOMALY':'normal'}</span>`;
});
document.getElementById('contSlider').addEventListener('input',drawIF);
drawIF();

// ── Panel 2: Score histogram ───────────────────────────────────────────────
const cv2=document.getElementById('cvHist'),ctx2=cv2.getContext('2d');
let histThreshX=null;

function drawHist(){
  const W=340,H=240,PAD=30;
  ctx2.clearRect(0,0,W,H);
  const N_BINS=20;
  const bins=Array.from({length:N_BINS},()=>({nor:0,ano:0}));
  const minS=Math.min(...IF_SCORES),maxS=Math.max(...IF_SCORES),rng_=maxS-minS||1;
  IF_SCORES.forEach((s,i)=>{
    const b=Math.min(Math.floor((s-minS)/rng_*N_BINS),N_BINS-1);
    if(LABELS[i]===1) bins[b].ano++; else bins[b].nor++;
  });
  const maxH=Math.max(...bins.map(b=>b.nor+b.ano));
  const bw=(W-2*PAD)/N_BINS;

  bins.forEach((b,bi)=>{
    const x=PAD+bi*bw;
    if(b.nor>0){
      ctx2.fillStyle='rgba(56,189,248,0.7)';
      ctx2.fillRect(x,H-PAD-b.nor/maxH*(H-2*PAD),bw-1,b.nor/maxH*(H-2*PAD));
    }
    if(b.ano>0){
      const y_start=H-PAD-(b.nor+b.ano)/maxH*(H-2*PAD);
      ctx2.fillStyle='rgba(248,113,113,0.85)';
      ctx2.fillRect(x,y_start,bw-1,b.ano/maxH*(H-2*PAD));
    }
  });

  // Threshold line
  const thresh=histThreshX!==null?(histThreshX-PAD)/(W-2*PAD)*rng_+minS:
    IF_SCORES[scoreOrder[Math.round(0.1*N)]];
  const tx=PAD+(thresh-minS)/rng_*(W-2*PAD);
  ctx2.setLineDash([4,2]); ctx2.strokeStyle='#fbbf24'; ctx2.lineWidth=2;
  ctx2.beginPath(); ctx2.moveTo(tx,PAD); ctx2.lineTo(tx,H-PAD); ctx2.stroke();
  ctx2.setLineDash([]);
  ctx2.fillStyle='#fbbf24'; ctx2.font='8px sans-serif'; ctx2.textAlign='left';
  ctx2.fillText(`t=${thresh.toFixed(3)}`,tx+2,PAD+10);

  // Axes labels
  ctx2.fillStyle='#475569'; ctx2.font='8px sans-serif'; ctx2.textAlign='center';
  ctx2.fillText(minS.toFixed(3),PAD,H-PAD+12); ctx2.fillText(maxS.toFixed(3),W-PAD,H-PAD+12);
  ctx2.fillText('Anomaly score',W/2,H-2);

  const flagged=IF_SCORES.filter(s=>s>=thresh).length;
  const tp=IF_SCORES.filter((s,i)=>s>=thresh&&LABELS[i]===1).length;
  const fp=flagged-tp;
  document.getElementById('histInfo').innerHTML=
    `Threshold: <span class="pv">${thresh.toFixed(3)}</span> &nbsp;|&nbsp; ` +
    `Flagged: <span class="pv">${flagged}</span> &nbsp;|&nbsp; ` +
    `TP: <span class="pv">${tp}</span> FP: <span class="pv">${fp}</span>`;
}

cv2.addEventListener('mousemove',e=>{
  histThreshX=e.clientX-cv2.getBoundingClientRect().left; drawHist();
});
cv2.addEventListener('mouseleave',()=>{histThreshX=null; drawHist();});
drawHist();

// ── Panel 3: LOF vs Z-Score ───────────────────────────────────────────────
sd=7;
const lofData=[];
for(let i=0;i<25;i++) lofData.push([50+gauss()*15,  120+gauss()*15,  0]); // dense
for(let i=0;i<25;i++) lofData.push([230+gauss()*50, 120+gauss()*50,  0]); // sparse
lofData.push([310,200,1],[320,60,1],[-10,60,1]); // true outliers

// LOF scores
function computeLOF(data, k){
  const n=data.length;
  const pts=data.map(d=>[d[0],d[1]]);
  function dst(a,b){return Math.hypot(a[0]-b[0],a[1]-b[1]);}
  const nbrs=[],kdists=[];
  for(let i=0;i<n;i++){
    const ds=pts.map((_,j)=>j===i?Infinity:dst(pts[i],pts[j]));
    const sorted=[...ds.keys()].sort((a,b)=>ds[a]-ds[b]);
    nbrs.push(sorted.slice(0,k)); kdists.push(ds[sorted[k-1]]);
  }
  function rd(i,o){return Math.max(kdists[o],dst(pts[i],pts[o]));}
  const lrd=pts.map((_,i)=>1/Math.max(nbrs[i].reduce((s,o)=>s+rd(i,o),0)/k,1e-10));
  return pts.map((_,i)=>nbrs[i].reduce((s,o)=>s+lrd[o],0)/(k*lrd[i]));
}
const LOF_SC=computeLOF(lofData,5);

// Z-scores on x-coordinate
const xs=lofData.map(d=>d[0]);
const xm=xs.reduce((a,b)=>a+b,0)/xs.length;
const xs_=Math.sqrt(xs.reduce((s,x)=>s+(x-xm)**2,0)/xs.length)||1;
const Z_SC=xs.map(x=>Math.abs((x-xm)/xs_));

let lofMode='lof';
const cv3=document.getElementById('cvLOF'),ctx3=cv3.getContext('2d');
function showLOF(mode){
  lofMode=mode;
  ['lof','z','true'].forEach(m=>document.getElementById('btn'+m[0].toUpperCase()+m.slice(1))
    .classList.toggle('active',m===mode));

  let info='';
  if(mode==='lof') info='LOF flags only the 3 truly isolated points. Dense and sparse clusters both appear normal to their own neighbours.';
  else if(mode==='z') info='Z-score flags sparse cluster B as outliers because they are far from the global mean. This is a FALSE POSITIVE — they form a valid cluster.';
  else info='True labels: blue = normal (2 clusters), red = true outliers (isolated points).';
  document.getElementById('lofInfo').textContent=info;
  drawLOF();
}
function drawLOF(){
  const W=340,H=240; ctx3.clearRect(0,0,W,H);
  lofData.forEach((d,i)=>{
    let col;
    if(lofMode==='true'){
      col=d[2]===1?'#f87171':'#38bdf8';
    } else if(lofMode==='lof'){
      const s=LOF_SC[i]; const t=Math.min((s-1)/4,1);
      col=`rgb(${Math.round(56+t*192)},${Math.round(189-t*76)},${Math.round(248-t*135)})`;
    } else {
      const s=Z_SC[i]; const t=Math.min(s/3,1);
      col=`rgb(${Math.round(56+t*192)},${Math.round(189-t*76)},${Math.round(248-t*135)})`;
    }
    ctx3.beginPath(); ctx3.arc(d[0],d[1],d[2]===1?7:4.5,0,2*Math.PI);
    ctx3.fillStyle=col; ctx3.fill();
    if(d[2]===1){ctx3.strokeStyle='#fbbf24';ctx3.lineWidth=1.5;ctx3.stroke();}
  });
  ctx3.fillStyle='#475569'; ctx3.font='9px sans-serif'; ctx3.textAlign='center';
  ctx3.fillText('Dense cluster A',70,H-8); ctx3.fillText('Sparse cluster B',230,H-8);
  ctx3.fillStyle='#fbbf24'; ctx3.fillText('★ outliers',305,60);
}
showLOF('lof');

// ── Panel 4: PR curve ─────────────────────────────────────────────────────
const cv4=document.getElementById('cvPR'),ctx4=cv4.getContext('2d');
function drawPR(){
  const W=340,H=200,PAD=34;
  ctx4.clearRect(0,0,W,H);
  const N_ANO=10;
  const nFlag=parseInt(document.getElementById('prSlider').value);
  document.getElementById('prVal').textContent=nFlag;

  const prPoints=[];
  for(let nf=1;nf<=N;nf++){
    const flagged=new Set(scoreOrder.slice(0,nf));
    const tp=[...flagged].filter(i=>LABELS[i]===1).length;
    const prec=tp/nf, rec=tp/N_ANO;
    prPoints.push({nf,prec,rec});
  }

  // Draw PR curve
  ctx4.beginPath();
  prPoints.forEach((p,i)=>{
    const x=PAD+p.rec*(W-2*PAD), y=H-PAD-p.prec*(H-2*PAD);
    i===0?ctx4.moveTo(x,y):ctx4.lineTo(x,y);
  });
  ctx4.strokeStyle='#f87171'; ctx4.lineWidth=2; ctx4.stroke();

  // Current operating point
  const cp=prPoints[nFlag-1];
  const cx_=PAD+cp.rec*(W-2*PAD), cy_=H-PAD-cp.prec*(H-2*PAD);
  ctx4.beginPath(); ctx4.arc(cx_,cy_,6,0,2*Math.PI);
  ctx4.fillStyle='#fbbf24'; ctx4.fill();

  // Grid
  ctx4.strokeStyle='#2d3148'; ctx4.lineWidth=0.5;
  [0,0.25,0.5,0.75,1].forEach(f=>{
    const y=PAD+f*(H-2*PAD); const x=PAD+f*(W-2*PAD);
    ctx4.beginPath(); ctx4.moveTo(PAD,y); ctx4.lineTo(W-PAD,y); ctx4.stroke();
    ctx4.beginPath(); ctx4.moveTo(x,PAD); ctx4.lineTo(x,H-PAD); ctx4.stroke();
    ctx4.fillStyle='#475569'; ctx4.font='8px sans-serif'; ctx4.textAlign='right';
    ctx4.fillText((1-f).toFixed(2),PAD-2,PAD+f*(H-2*PAD)+3);
    ctx4.textAlign='center'; ctx4.fillText(f.toFixed(2),PAD+f*(W-2*PAD),H-PAD+12);
  });

  ctx4.fillStyle='#64748b'; ctx4.font='9px sans-serif';
  ctx4.textAlign='center'; ctx4.fillText('Recall',W/2,H-2);
  ctx4.save(); ctx4.translate(10,H/2); ctx4.rotate(-Math.PI/2);
  ctx4.fillText('Precision',0,0); ctx4.restore();

  const f1=2*cp.prec*cp.rec/(cp.prec+cp.rec+1e-9);
  document.getElementById('prInfo').innerHTML=
    `Flagging <span class="pv">${nFlag}</span> points &nbsp;|&nbsp; ` +
    `Prec: <span class="pv">${cp.prec.toFixed(3)}</span> &nbsp;|&nbsp; ` +
    `Rec: <span class="pv">${cp.rec.toFixed(3)}</span> &nbsp;|&nbsp; ` +
    `F1: <span class="pv">${f1.toFixed(3)}</span>`;
}
document.getElementById('prSlider').addEventListener('input',drawPR);
drawPR();
</script>
</body>
</html>
"""