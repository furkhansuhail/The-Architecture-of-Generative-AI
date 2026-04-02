
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
h2   { color: #e879f9; margin-bottom: 4px; }
.subtitle { color: #64748b; margin-bottom: 22px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #1e2130; border-radius: 12px; padding: 18px;
        border: 1px solid #2d3148; }
.card h3 { color: #e879f9; margin: 0 0 10px; font-size: 0.9em;
           text-transform: uppercase; letter-spacing: 0.05em; }
canvas { display: block; }
.params { background: #12141f; padding: 8px 12px; border-radius: 8px;
          font-size: 0.81em; color: #94a3b8; margin: 8px 0; line-height: 1.6; }
.pv { color: #e879f9; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 90px; }
input[type=range] { accent-color: #e879f9; flex: 1; }
.vb { font-size: 0.8em; color: #e879f9; min-width: 44px; }
.btn-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 8px; }
button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
         border-radius: 6px; padding: 5px 13px; cursor: pointer;
         font-size: 0.8em; transition: background 0.15s; }
button:hover { background: #3d4168; }
button.active { background: #e879f9; color: #0f1117; }
.legend { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.8em; }
.dot { width: 11px; height: 11px; border-radius: 50%; }
</style>
</head>
<body>
<h2>🗺️ t-SNE & UMAP Visual Explorer</h2>
<p class="subtitle">Non-linear dimensionality reduction — neighbourhood-preserving embeddings</p>

<div class="grid">

  <!-- Panel 1: t-SNE live simulation -->
  <div class="card">
    <h3>t-SNE Optimisation (Live)</h3>
    <div class="params">
      Iteration: <span class="pv" id="iterLabel">0</span> /
      <span class="pv" id="totalIter">250</span> &nbsp;|&nbsp;
      KL divergence: <span class="pv" id="klLabel">—</span><br>
      Watch points attract into clusters. Early exaggeration phase forces initial cluster formation.
    </div>
    <canvas id="cvTsne" width="340" height="250"></canvas>
    <div class="btn-row">
      <button onclick="startTsne()" id="btnRun">▶ Run</button>
      <button onclick="resetTsne()">Reset</button>
    </div>
    <div class="slider-row">
      <label>Perplexity</label>
      <input type="range" id="perpSlider" min="3" max="25" step="1" value="8">
      <span class="vb" id="perpVal">8</span>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#e879f9"></div>Cluster 1</div>
      <div class="legend-item"><div class="dot" style="background:#38bdf8"></div>Cluster 2</div>
      <div class="legend-item"><div class="dot" style="background:#fbbf24"></div>Cluster 3</div>
    </div>
  </div>

  <!-- Panel 2: KL divergence curve -->
  <div class="card">
    <h3>KL Divergence During Optimisation</h3>
    <div class="params">
      KL divergence = information lost representing high-dim P with low-dim Q.<br>
      Lower = better embedding. The sharp drop shows when structure forms.
      <span style="color:#fbbf24">Yellow region</span> = early exaggeration phase.
    </div>
    <canvas id="cvKL" width="340" height="250"></canvas>
    <div class="params" id="klPhaseInfo">Run t-SNE to see the convergence curve.</div>
  </div>

  <!-- Panel 3: Perplexity comparison -->
  <div class="card">
    <h3>Effect of Perplexity on Structure</h3>
    <div class="params">
      Same 3-cluster dataset embedded at 4 perplexity values.<br>
      <span style="color:#ef4444">Low perp</span> → fragmented.
      <span style="color:#34d399">Mid perp</span> → well-separated.
      <span style="color:#94a3b8">High perp</span> → merging.
    </div>
    <canvas id="cvPerp" width="340" height="250"></canvas>
    <div class="params">
      Perplexity ≈ effective number of neighbours. Rule: use 5–50, < n/3.
    </div>
  </div>

  <!-- Panel 4: t-SNE warnings visualiser -->
  <div class="card">
    <h3>⚠️ Interpreting t-SNE Correctly</h3>
    <div class="params" id="warnInfo">
      Select a warning to explore.
    </div>
    <canvas id="cvWarn" width="340" height="220"></canvas>
    <div class="btn-row">
      <button onclick="showWarning(0)" id="wBtn0" class="active">Cluster sizes</button>
      <button onclick="showWarning(1)" id="wBtn1">Gap distances</button>
      <button onclick="showWarning(2)" id="wBtn2">Local only</button>
    </div>
  </div>

</div>

<script>
// ── Seeded RNG ────────────────────────────────────────────────────────────────
let seed = 42;
const rng = () => { seed=(seed*1664525+1013904223)>>>0; return seed/4294967296; };
const gauss = () => {
  const u=rng()||1e-9,v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
};

// ── Colour palette ────────────────────────────────────────────────────────────
const PAL = ['#e879f9','#38bdf8','#fbbf24','#34d399','#fb923c'];

// ── Generate 3-cluster 4D data (embed into 2 for vis distance reference) ──────
function genData3(){
  seed=42;
  const pts=[], lbl=[];
  const cs=[[3,3,0,0],[-3,0,3,0],[0,-3,-3,0]];
  cs.forEach((cx,ci)=>{
    for(let i=0;i<18;i++){
      pts.push(cx.map(c=>c+gauss()*0.5));
      lbl.push(ci);
    }
  });
  return {pts,lbl};
}

// ── Tiny t-SNE for vis (2D input for speed) ────────────────────────────────────
// We project to 2D first for the visual, then run t-SNE on that
// Actually we'll run directly on 2D toy data for the visual panel

function genVis3Clusters(){
  seed=42;
  const pts=[], lbl=[];
  const cs=[[2.5,2],[-2.5,0.5],[0,-2.5]];
  cs.forEach((cx,ci)=>{
    for(let i=0;i<18;i++){
      pts.push([cx[0]+gauss()*0.5, cx[1]+gauss()*0.5]);
      lbl.push(ci);
    }
  });
  return {pts,lbl};
}

const {pts:ORIG_PTS, lbl:ORIG_LBL} = genVis3Clusters();
const N = ORIG_PTS.length;

function dist2(a,b){ return (a[0]-b[0])**2+(a[1]-b[1])**2; }

// Compute P matrix
function computeP(pts, perp){
  const n=pts.length;
  const P=Array.from({length:n},()=>new Float32Array(n));
  for(let i=0;i<n;i++){
    let lo=0,hi=1e10,beta=1;
    for(let iter=0;iter<50;iter++){
      const ed=pts.map((p,j)=>j===i?0:Math.exp(-dist2(pts[i],p)*beta));
      const s=ed.reduce((a,b)=>a+b,1e-10);
      const pc=ed.map(e=>e/s);
      const H=-pc.reduce((a,p)=>p>0?a+p*Math.log(p+1e-10):a,0);
      const diff=Math.exp(H)-perp;
      if(Math.abs(diff)<1e-5) break;
      if(diff>0){lo=beta;beta=hi<1e9?(beta+hi)/2:beta*2;}
      else{hi=beta;beta=(beta+lo)/2;}
    }
    const ed2=pts.map((p,j)=>j===i?0:Math.exp(-dist2(pts[i],p)*beta));
    const s2=ed2.reduce((a,b)=>a+b,1e-10);
    for(let j=0;j<n;j++) P[i][j]=ed2[j]/s2;
  }
  // Symmetrise
  const Ps=Array.from({length:n},()=>new Float32Array(n));
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) Ps[i][j]=(P[i][j]+P[j][i])/(2*n);
  return Ps;
}

// t-SNE state
let Y=[], vel=[], gains=[], Pij=null;
let tsneRunning=false, tsneInterval=null, tsneIter=0;
const TOTAL_ITER=250, EXAG_ITERS=80, LR=200, MOM=0.8;
const klHistory=[];

function resetTsne(){
  seed=7; // reset for init
  if(tsneInterval){clearInterval(tsneInterval);tsneInterval=null;}
  tsneRunning=false; tsneIter=0; klHistory.length=0;
  Y=ORIG_PTS.map(()=>[gauss()*0.01,gauss()*0.01]);
  vel=ORIG_PTS.map(()=>[0,0]);
  gains=ORIG_PTS.map(()=>[1,1]);
  const perp=parseInt(document.getElementById('perpSlider').value);
  Pij=computeP(ORIG_PTS,perp);
  document.getElementById('iterLabel').textContent='0';
  document.getElementById('klLabel').textContent='—';
  drawTsne(); drawKL();
  document.getElementById('btnRun').textContent='▶ Run';
}

function tsneStep(){
  if(tsneIter>=TOTAL_ITER){
    clearInterval(tsneInterval); tsneRunning=false;
    document.getElementById('btnRun').textContent='▶ Run';
    return;
  }
  tsneIter++;
  const exag=tsneIter<=EXAG_ITERS?4:1;
  const n=N;
  // Compute Q
  const q_num=Array.from({length:n},()=>new Float32Array(n));
  let qs=1e-10;
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) if(i!==j){
    const d2=(Y[i][0]-Y[j][0])**2+(Y[i][1]-Y[j][1])**2;
    q_num[i][j]=1/(1+d2); qs+=q_num[i][j];
  }
  // KL
  let kl=0;
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) if(i!==j&&Pij[i][j]>1e-12){
    kl+=Pij[i][j]*exag*Math.log((Pij[i][j]*exag)/(q_num[i][j]/qs+1e-10));
  }
  if(tsneIter%3===0) klHistory.push({t:tsneIter,kl,exag});
  // Gradient
  const grad=ORIG_PTS.map(()=>[0,0]);
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) if(i!==j){
    const f=4*(Pij[i][j]*exag-q_num[i][j]/qs)*q_num[i][j];
    grad[i][0]+=f*(Y[i][0]-Y[j][0]);
    grad[i][1]+=f*(Y[i][1]-Y[j][1]);
  }
  for(let i=0;i<n;i++) for(let d=0;d<2;d++){
    const same=(grad[i][d]>0)===(vel[i][d]>0);
    gains[i][d]=same?gains[i][d]*0.8:gains[i][d]+0.2;
    gains[i][d]=Math.max(gains[i][d],0.01);
    vel[i][d]=MOM*vel[i][d]-LR*gains[i][d]*grad[i][d];
    Y[i][d]+=vel[i][d];
  }
  // Centre
  for(let d=0;d<2;d++){const m=Y.reduce((s,p)=>s+p[d],0)/n; Y.forEach(p=>p[d]-=m);}
  document.getElementById('iterLabel').textContent=tsneIter;
  document.getElementById('klLabel').textContent=kl.toFixed(3);
  drawTsne(); drawKL();
}

function startTsne(){
  if(tsneRunning){
    clearInterval(tsneInterval); tsneRunning=false;
    document.getElementById('btnRun').textContent='▶ Run';
  } else {
    if(tsneIter===0) resetTsne();
    tsneRunning=true;
    document.getElementById('btnRun').textContent='⏸ Pause';
    tsneInterval=setInterval(tsneStep,30);
  }
}

document.getElementById('perpSlider').addEventListener('input',e=>{
  document.getElementById('perpVal').textContent=e.target.value;
  resetTsne();
});

// ── Draw t-SNE scatter ─────────────────────────────────────────────────────────
const cv1=document.getElementById('cvTsne'),ctx1=cv1.getContext('2d');
function drawTsne(){
  const W=340,H=250;
  ctx1.clearRect(0,0,W,H);
  if(!Y.length) return;
  const xs=Y.map(p=>p[0]),ys=Y.map(p=>p[1]);
  const pad=20;
  const mnx=Math.min(...xs)-0.1,mxx=Math.max(...xs)+0.1;
  const mny=Math.min(...ys)-0.1,mxy=Math.max(...ys)+0.1;
  const sx=x=>pad+(x-mnx)/(mxx-mnx)*(W-2*pad);
  const sy=y=>H-pad-(y-mny)/(mxy-mny)*(H-2*pad);
  Y.forEach((p,i)=>{
    ctx1.beginPath(); ctx1.arc(sx(p[0]),sy(p[1]),4.5,0,2*Math.PI);
    ctx1.fillStyle=PAL[ORIG_LBL[i]]; ctx1.fill();
    ctx1.strokeStyle='#0f1117'; ctx1.lineWidth=0.8; ctx1.stroke();
  });
  // Phase label
  const phase=tsneIter<=EXAG_ITERS&&tsneIter>0?'Early exaggeration':'Optimising';
  ctx1.fillStyle='#64748b'; ctx1.font='9px sans-serif'; ctx1.textAlign='right';
  ctx1.fillText(tsneIter>0?phase:'Press Run to start',W-8,H-6);
}

// ── Draw KL curve ──────────────────────────────────────────────────────────────
const cv2=document.getElementById('cvKL'),ctx2=cv2.getContext('2d');
function drawKL(){
  const W=340,H=250,PAD=34;
  ctx2.clearRect(0,0,W,H);
  if(klHistory.length<2){ ctx2.fillStyle='#475569'; ctx2.font='11px sans-serif';
    ctx2.textAlign='center'; ctx2.fillText('Run t-SNE to see KL curve',W/2,H/2); return; }
  const kls=klHistory.map(h=>h.kl);
  const maxKL=Math.max(...kls)||1; const minKL=Math.min(...kls);
  const sy=kl=>H-PAD-(kl-minKL)/(maxKL-minKL+1e-6)*(H-2*PAD);
  const sx=t=>PAD+(t/TOTAL_ITER)*(W-2*PAD);
  // Exaggeration zone
  ctx2.fillStyle='rgba(251,191,36,0.08)';
  ctx2.fillRect(PAD,PAD,sx(EXAG_ITERS)-PAD,H-2*PAD);
  ctx2.fillStyle='#fbbf24'; ctx2.font='8px sans-serif'; ctx2.textAlign='center';
  ctx2.fillText('Early exag.',PAD+(sx(EXAG_ITERS)-PAD)/2,PAD+10);
  // Grid
  ctx2.strokeStyle='#2d3148'; ctx2.lineWidth=0.5;
  for(let g=0;g<=4;g++){
    const y=PAD+g/4*(H-2*PAD);
    ctx2.beginPath(); ctx2.moveTo(PAD,y); ctx2.lineTo(W-PAD,y); ctx2.stroke();
  }
  // KL curve
  ctx2.beginPath();
  klHistory.forEach((h,i)=>{
    const x=sx(h.t),y=sy(h.kl);
    i===0?ctx2.moveTo(x,y):ctx2.lineTo(x,y);
  });
  ctx2.strokeStyle='#e879f9'; ctx2.lineWidth=2; ctx2.stroke();
  // Axes
  ctx2.fillStyle='#64748b'; ctx2.font='9px sans-serif'; ctx2.textAlign='center';
  ctx2.fillText('Iteration',W/2,H-4);
  ctx2.save(); ctx2.translate(10,H/2); ctx2.rotate(-Math.PI/2);
  ctx2.fillText('KL divergence',0,0); ctx2.restore();
  // Current val dot
  const last=klHistory[klHistory.length-1];
  ctx2.beginPath(); ctx2.arc(sx(last.t),sy(last.kl),4,0,2*Math.PI);
  ctx2.fillStyle='#e879f9'; ctx2.fill();

  const pct=tsneIter/TOTAL_ITER*100;
  document.getElementById('klPhaseInfo').innerHTML=
    `Progress: <span class="pv">${pct.toFixed(0)}%</span> &nbsp;|&nbsp;
     KL: <span class="pv">${last.kl.toFixed(3)}</span> &nbsp;|&nbsp;
     Phase: <span class="pv">${tsneIter<=EXAG_ITERS?'Early exaggeration':'Normal optimisation'}</span>`;
}

// ── Panel 3: Perplexity comparison (pre-run, shown as static) ─────────────────
const cv3=document.getElementById('cvPerp'),ctx3=cv3.getContext('2d');

function drawPerpComparison(){
  const W=340,H=250,perps=[3,8,15,25];
  const titles=['perp=3','perp=8','perp=15','perp=25'];
  const cols=['#ef4444','#34d399','#34d399','#94a3b8'];
  const panelW=(W-10)/2, panelH=(H-10)/2;
  ctx3.clearRect(0,0,W,H);

  perps.forEach((perp,pi)=>{
    seed=77;
    const px=(pi%2)*(panelW+5)+3, py=Math.floor(pi/2)*(panelH+5)+3;
    ctx3.fillStyle='#12141f'; ctx3.fillRect(px,py,panelW,panelH);

    // Quick mini t-SNE: just run ~80 iters on small data
    const {pts:mPts,lbl:mLbl}=genVis3Clusters();
    const mn=mPts.length;
    const mP=computeP(mPts,perp);
    let mY=mPts.map(()=>[gauss()*0.01,gauss()*0.01]);
    let mVel=mPts.map(()=>[0,0]),mGains=mPts.map(()=>[1,1]);
    for(let t=1;t<=80;t++){
      const exag=t<=30?4:1;
      const qn=Array.from({length:mn},()=>new Float32Array(mn)); let qs=1e-10;
      for(let i=0;i<mn;i++) for(let j=0;j<mn;j++) if(i!==j){
        const d2=(mY[i][0]-mY[j][0])**2+(mY[i][1]-mY[j][1])**2;
        qn[i][j]=1/(1+d2); qs+=qn[i][j];
      }
      const gr=mPts.map(()=>[0,0]);
      for(let i=0;i<mn;i++) for(let j=0;j<mn;j++) if(i!==j){
        const f=4*(mP[i][j]*exag-qn[i][j]/qs)*qn[i][j];
        gr[i][0]+=f*(mY[i][0]-mY[j][0]); gr[i][1]+=f*(mY[i][1]-mY[j][1]);
      }
      for(let i=0;i<mn;i++) for(let d=0;d<2;d++){
        const same=(gr[i][d]>0)===(mVel[i][d]>0);
        mGains[i][d]=same?mGains[i][d]*0.8:mGains[i][d]+0.2;
        mGains[i][d]=Math.max(mGains[i][d],0.01);
        mVel[i][d]=0.8*mVel[i][d]-150*mGains[i][d]*gr[i][d];
        mY[i][d]+=mVel[i][d];
      }
      for(let d=0;d<2;d++){const m=mY.reduce((s,p)=>s+p[d],0)/mn; mY.forEach(p=>p[d]-=m);}
    }
    const xs=mY.map(p=>p[0]),ys=mY.map(p=>p[1]);
    const pad=8,mnx=Math.min(...xs)-0.1,mxx=Math.max(...xs)+0.1;
    const mny2=Math.min(...ys)-0.1,mxy2=Math.max(...ys)+0.1;
    const ssx=x=>px+pad+(x-mnx)/(mxx-mnx)*(panelW-2*pad);
    const ssy=y=>py+panelH-pad-(y-mny2)/(mxy2-mny2)*(panelH-2*pad);
    mY.forEach((p,i)=>{
      ctx3.beginPath(); ctx3.arc(ssx(p[0]),ssy(p[1]),3,0,2*Math.PI);
      ctx3.fillStyle=PAL[mLbl[i]]; ctx3.fill();
    });
    ctx3.fillStyle=cols[pi]; ctx3.font='bold 9px sans-serif'; ctx3.textAlign='left';
    ctx3.fillText(titles[pi],px+3,py+10);
  });
}

// ── Panel 4: Warnings ─────────────────────────────────────────────────────────
const cv4=document.getElementById('cvWarn'),ctx4=cv4.getContext('2d');
let currentWarning=0;

const warnings=[
  {
    title:"Cluster sizes don't reflect true sizes",
    note:"Left: 1 large + 2 small true clusters. Right: t-SNE makes them appear equal size. Visual size ≠ true cluster size.",
    draw:(ctx,W,H)=>{
      // Left panel: true distribution
      ctx.fillStyle='#12141f'; ctx.fillRect(4,4,W/2-6,H-8);
      ctx.fillStyle='#64748b'; ctx.font='8px sans-serif'; ctx.textAlign='center';
      ctx.fillText('True data',W/4,14);
      // Big cluster
      for(let i=0;i<50;i++){
        seed++; const a=rng()*2*Math.PI,r=rng()*30;
        ctx.beginPath(); ctx.arc(W/4+r*Math.cos(a)+10,H/2+r*Math.sin(a),2.5,0,2*Math.PI);
        ctx.fillStyle='#e879f988'; ctx.fill();
      }
      // 2 small clusters
      for(let i=0;i<8;i++){
        seed++; ctx.beginPath(); ctx.arc(W/4-60+(rng()-0.5)*12,H/2-40+(rng()-0.5)*12,2.5,0,2*Math.PI);
        ctx.fillStyle='#38bdf888'; ctx.fill();
      }
      for(let i=0;i<8;i++){
        seed++; ctx.beginPath(); ctx.arc(W/4+60+(rng()-0.5)*12,H/2+45+(rng()-0.5)*12,2.5,0,2*Math.PI);
        ctx.fillStyle='#fbbf2488'; ctx.fill();
      }
      // Right panel: "t-SNE" (simulated equal blobs)
      ctx.fillStyle='#12141f'; ctx.fillRect(W/2+2,4,W/2-6,H-8);
      ctx.fillStyle='#64748b'; ctx.font='8px sans-serif'; ctx.textAlign='center';
      ctx.fillText('t-SNE output',3*W/4,14);
      const cc3=[[3*W/4-40,H/2-25],[3*W/4+35,H/2],[3*W/4,H/2+40]];
      const cols3=['#e879f9','#38bdf8','#fbbf24'];
      cc3.forEach(([cx,cy],ci)=>{
        for(let i=0;i<18;i++){seed++; ctx.beginPath(); ctx.arc(cx+(rng()-0.5)*22,cy+(rng()-0.5)*22,2.5,0,2*Math.PI); ctx.fillStyle=cols3[ci]+'aa'; ctx.fill();}
      });
      ctx.fillStyle='#ef4444'; ctx.font='bold 9px sans-serif'; ctx.textAlign='center';
      ctx.fillText('⚠ All clusters appear equal size!',W/2,H-6);
    }
  },
  {
    title:"Distances between clusters are meaningless",
    note:"Two datasets: one with clusters 10 units apart, one with 1.5 units apart. t-SNE makes them look identical.",
    draw:(ctx,W,H)=>{
      ctx.fillStyle='#12141f'; ctx.fillRect(4,4,W/2-6,H-8);
      ctx.fillStyle='#64748b'; ctx.font='8px sans-serif'; ctx.textAlign='center';
      ctx.fillText('Far apart in 4D',W/4,14);
      const c1a=[[W/4-55,H/2],[W/4+55,H/2],[W/4,H/2-45]];
      c1a.forEach(([cx,cy],ci)=>{
        for(let i=0;i<10;i++){seed++; ctx.beginPath(); ctx.arc(cx+(rng()-0.5)*14,cy+(rng()-0.5)*14,2.5,0,2*Math.PI); ctx.fillStyle=PAL[ci]+'bb'; ctx.fill();}
      });
      ctx.fillStyle='#12141f'; ctx.fillRect(W/2+2,4,W/2-6,H-8);
      ctx.fillStyle='#64748b'; ctx.font='8px sans-serif'; ctx.textAlign='center';
      ctx.fillText('Close together in 4D',3*W/4,14);
      const c1b=[[3*W/4-52,H/2+2],[3*W/4+50,H/2+2],[3*W/4,H/2-44]];
      c1b.forEach(([cx,cy],ci)=>{
        for(let i=0;i<10;i++){seed++; ctx.beginPath(); ctx.arc(cx+(rng()-0.5)*14,cy+(rng()-0.5)*14,2.5,0,2*Math.PI); ctx.fillStyle=PAL[ci]+'bb'; ctx.fill();}
      });
      ctx.fillStyle='#fbbf24'; ctx.font='9px sans-serif'; ctx.textAlign='center';
      ctx.fillText('← Both t-SNE plots look nearly identical →',W/2,H-6);
    }
  },
  {
    title:"Only local structure is preserved",
    note:"Global ordering may not reflect true structure. A cluster near another in t-SNE may actually be far away in the original space.",
    draw:(ctx,W,H)=>{
      ctx.fillStyle='#12141f'; ctx.fillRect(4,4,W/2-6,H-8);
      ctx.fillStyle='#64748b'; ctx.font='8px sans-serif'; ctx.textAlign='center';
      ctx.fillText('True global order (1D)',W/4,14);
      // Draw a line with 4 clusters in order
      for(let ci=0;ci<4;ci++){
        const cx=W/4-50+ci*34, cy=H/2;
        for(let i=0;i<8;i++){seed++; ctx.beginPath(); ctx.arc(cx+(rng()-0.5)*10,cy+(rng()-0.5)*18,2.5,0,2*Math.PI); ctx.fillStyle=PAL[ci]+'cc'; ctx.fill();}
        ctx.fillStyle=PAL[ci]; ctx.font='7px sans-serif'; ctx.textAlign='center';
        ctx.fillText(['C1','C2','C3','C4'][ci],cx,H/2+28);
      }
      ctx.strokeStyle='#475569'; ctx.setLineDash([2,2]);
      ctx.beginPath(); ctx.moveTo(W/4-50,H/2); ctx.lineTo(W/4+52,H/2); ctx.stroke();
      ctx.setLineDash([]);
      // Right: t-SNE reorders
      ctx.fillStyle='#12141f'; ctx.fillRect(W/2+2,4,W/2-6,H-8);
      ctx.fillStyle='#64748b'; ctx.font='8px sans-serif'; ctx.textAlign='center';
      ctx.fillText('t-SNE (order scrambled)',3*W/4,14);
      const scramble=[[3*W/4-40,H/2-30],[3*W/4+40,H/2+28],[3*W/4+5,H/2-10],[3*W/4-15,H/2+35]];
      scramble.forEach(([cx,cy],ci)=>{
        for(let i=0;i<8;i++){seed++; ctx.beginPath(); ctx.arc(cx+(rng()-0.5)*14,cy+(rng()-0.5)*14,2.5,0,2*Math.PI); ctx.fillStyle=PAL[ci]+'cc'; ctx.fill();}
        ctx.fillStyle=PAL[ci]; ctx.font='7px sans-serif'; ctx.textAlign='center';
        ctx.fillText(['C1','C2','C3','C4'][ci],cx,cy+18);
      });
      ctx.fillStyle='#ef4444'; ctx.font='9px sans-serif'; ctx.textAlign='center';
      ctx.fillText('⚠ Global order not preserved',W/2,H-6);
    }
  }
];

function showWarning(i){
  currentWarning=i;
  [0,1,2].forEach(k=>document.getElementById(`wBtn${k}`).classList.toggle('active',k===i));
  document.getElementById('warnInfo').innerHTML=
    `<strong style="color:#e879f9">${warnings[i].title}</strong><br>${warnings[i].note}`;
  const W=340,H=220; ctx4.clearRect(0,0,W,H);
  warnings[i].draw(ctx4,W,H);
}

// ── Init ──────────────────────────────────────────────────────────────────────
resetTsne();
drawPerpComparison();
showWarning(0);
</script>
</body>
</html>
"""