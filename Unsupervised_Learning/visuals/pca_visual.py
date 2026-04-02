
PCA_VISUAL_HEIGHT = 1300
PCA_VISUAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0;
       padding: 20px; }
h2   { color: #60a5fa; margin-bottom: 4px; }
.subtitle { color: #64748b; margin-bottom: 22px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #1e2130; border-radius: 12px; padding: 18px;
        border: 1px solid #2d3148; }
.card h3 { color: #60a5fa; margin: 0 0 10px; font-size: 0.9em;
           text-transform: uppercase; letter-spacing: 0.05em; }
canvas { display: block; }
.params { background: #12141f; padding: 8px 12px; border-radius: 8px;
          font-size: 0.81em; color: #94a3b8; margin: 8px 0; line-height: 1.6; }
.pv { color: #60a5fa; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 10px; margin-top: 7px; }
.slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 90px; }
input[type=range] { accent-color: #60a5fa; flex: 1; }
.vb { font-size: 0.8em; color: #60a5fa; min-width: 36px; }
.btn-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 9px; }
button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
         border-radius: 6px; padding: 4px 12px; cursor: pointer;
         font-size: 0.8em; transition: background 0.15s; }
button:hover { background: #3d4168; }
button.active { background: #60a5fa; color: #0f1117; }
</style>
</head>
<body>

<h2>📐 PCA Visual Explorer</h2>
<p class="subtitle">Principal Component Analysis — see variance directions, scree plots, and projection in real time</p>

<div class="grid">

  <!-- Panel 1: 2D scatter with PC axes -->
  <div class="card">
    <h3>PC Directions on Raw Data</h3>
    <div class="params">
      PC1 explains <span class="pv" id="evr1">—</span>% of variance &nbsp;|&nbsp;
      PC2 explains <span class="pv" id="evr2">—</span>%<br>
      Arrows show principal component directions. Length ∝ √λ (standard deviation along that axis).
    </div>
    <canvas id="cvScatter" width="340" height="260"></canvas>
    <div class="slider-row">
      <label>Rotation θ</label>
      <input type="range" id="rotSlider" min="0" max="180" step="1" value="35">
      <span class="vb" id="rotVal">35°</span>
    </div>
    <div class="slider-row">
      <label>Noise σ</label>
      <input type="range" id="noiseSlider" min="1" max="60" step="1" value="15">
      <span class="vb" id="noiseVal">0.15</span>
    </div>
    <div class="params">Rotate the data cloud or add noise to see how PCs adapt.</div>
  </div>

  <!-- Panel 2: Scree plot -->
  <div class="card">
    <h3>Scree Plot + Cumulative Variance</h3>
    <div class="params">
      Eigenvalue per component (bars) and cumulative explained variance (line).<br>
      Drag the <span style="color:#fbbf24">threshold line</span> to set a variance target.
      Components right of the elbow add diminishing information.
    </div>
    <canvas id="cvScree" width="340" height="260" style="cursor:crosshair;"></canvas>
    <div class="params" id="screeInfo">—</div>
  </div>

  <!-- Panel 3: Projected 1D distribution -->
  <div class="card">
    <h3>Projection onto PC1 vs PC2</h3>
    <div class="params">
      Distribution of PC scores along each component.<br>
      PC1 has the widest spread (maximum variance); PC2 is narrower and orthogonal.
    </div>
    <canvas id="cvProj" width="340" height="260"></canvas>
    <div class="btn-row">
      <button onclick="setDataset('ellipse')" id="btnEllipse" class="active">Ellipse</button>
      <button onclick="setDataset('clusters')" id="btnClusters">3 Clusters</button>
      <button onclick="setDataset('ring')" id="btnRing">Ring</button>
      <button onclick="setDataset('uniform')" id="btnUniform">Uniform</button>
    </div>
  </div>

  <!-- Panel 4: Reconstruction error vs K -->
  <div class="card">
    <h3>Reconstruction Error vs K</h3>
    <div class="params" id="reconInfo">
      RMSE and retained variance as number of components K increases.
      Moving <span style="color:#60a5fa">K slider</span> shows the reconstructed point cloud above.
    </div>
    <canvas id="cvRecon" width="340" height="200"></canvas>
    <div class="slider-row">
      <label>K components</label>
      <input type="range" id="kSlider" min="1" max="2" step="1" value="1">
      <span class="vb" id="kVal">1</span>
    </div>
    <canvas id="cvReconScatter" width="340" height="120"></canvas>
  </div>

</div>

<script>
// ── Utilities ─────────────────────────────────────────────────────────────────
const rng = (() => {
  let s = 12345;
  return () => { s=(s*1664525+1013904223)>>>0; return s/4294967296; };
})();
const gauss = () => {
  const u=rng()||1e-9, v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
};
const dist2 = (a,b) => Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2);

// ── Dataset generators ────────────────────────────────────────────────────────
function genEllipse(rot_deg, noise_frac){
  const rot=rot_deg*Math.PI/180;
  const pts=[];
  for(let i=0;i<120;i++){
    const t=rng()*2*Math.PI;
    let x=2.5*Math.cos(t)+gauss()*noise_frac;
    let y=0.5*Math.sin(t)+gauss()*noise_frac;
    pts.push([x*Math.cos(rot)-y*Math.sin(rot), x*Math.sin(rot)+y*Math.cos(rot)]);
  }
  return pts;
}

function genClusters(){
  const pts=[];
  const cs=[[2,2],[-2,-1],[0,-2.5]];
  cs.forEach(([cx,cy])=>{
    for(let i=0;i<40;i++) pts.push([cx+gauss()*0.6,cy+gauss()*0.6]);
  });
  return pts;
}

function genRing(){
  const pts=[];
  for(let i=0;i<120;i++){
    const a=rng()*2*Math.PI;
    const r=2+gauss()*0.2;
    pts.push([r*Math.cos(a),r*Math.sin(a)]);
  }
  return pts;
}

function genUniform(){
  const pts=[];
  for(let i=0;i<120;i++) pts.push([(rng()-0.5)*6,(rng()-0.5)*6]);
  return pts;
}

// ── PCA (2D only for vis) ──────────────────────────────────────────────────────
function pca2D(pts){
  const n=pts.length;
  const mx=pts.reduce((s,p)=>s+p[0],0)/n;
  const my=pts.reduce((s,p)=>s+p[1],0)/n;
  const Xc=pts.map(p=>[p[0]-mx,p[1]-my]);
  // Covariance
  const c11=Xc.reduce((s,p)=>s+p[0]*p[0],0)/(n-1);
  const c22=Xc.reduce((s,p)=>s+p[1]*p[1],0)/(n-1);
  const c12=Xc.reduce((s,p)=>s+p[0]*p[1],0)/(n-1);
  // Analytic eigendecomposition for 2×2
  const tr=c11+c22, det=c11*c22-c12*c12;
  const disc=Math.sqrt(Math.max(0,(tr*tr/4)-det));
  const l1=tr/2+disc, l2=tr/2-disc;
  // Eigenvectors
  let v1,v2;
  if(Math.abs(c12)>1e-10){
    const a=[l1-c22, c12]; const an=Math.sqrt(a[0]**2+a[1]**2);
    v1=[a[0]/an,a[1]/an];
    v2=[-v1[1],v1[0]];
  } else {
    v1=c11>=c22?[1,0]:[0,1];
    v2=c11>=c22?[0,1]:[1,0];
  }
  // Scores
  const z1=Xc.map(p=>p[0]*v1[0]+p[1]*v1[1]);
  const z2=Xc.map(p=>p[0]*v2[0]+p[1]*v2[1]);
  const total=l1+l2||1;
  return { l1,l2,v1,v2,z1,z2,mx,my,Xc,total,evr1:l1/total,evr2:l2/total };
}

// ── Canvas scaling helpers ────────────────────────────────────────────────────
function makeScale(pts, W, H, pad=28){
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  const mnx=Math.min(...xs)-0.5, mxx=Math.max(...xs)+0.5;
  const mny=Math.min(...ys)-0.5, mxy=Math.max(...ys)+0.5;
  const sx=x=>pad+(x-mnx)/(mxx-mnx)*(W-2*pad);
  const sy=y=>H-pad-(y-mny)/(mxy-mny)*(H-2*pad);
  return {sx,sy,mnx,mxx,mny,mxy};
}

// ── State ─────────────────────────────────────────────────────────────────────
let currentDataset='ellipse';
let currentRot=35, currentNoise=0.15;
let rawPts=[], pcaResult=null;
let screeThreshold=0.85;
let K=1;

function rebuildData(){
  if(currentDataset==='ellipse') rawPts=genEllipse(currentRot,currentNoise);
  else if(currentDataset==='clusters') rawPts=genClusters();
  else if(currentDataset==='ring') rawPts=genRing();
  else rawPts=genUniform();
  pcaResult=pca2D(rawPts);
  drawAll();
}

function setDataset(name){
  currentDataset=name;
  ['Ellipse','Clusters','Ring','Uniform'].forEach(n=>{
    document.getElementById('btn'+n).classList.toggle('active',n.toLowerCase()===name);
  });
  rebuildData();
}

// ── Panel 1: Scatter + PC arrows ─────────────────────────────────────────────
const cv1=document.getElementById('cvScatter'); const ctx1=cv1.getContext('2d');
function drawScatter(){
  const W=340,H=260;
  ctx1.clearRect(0,0,W,H);
  const r=pcaResult;
  const {sx,sy}=makeScale(rawPts,W,H);

  // Points
  rawPts.forEach(p=>{
    ctx1.beginPath(); ctx1.arc(sx(p[0]),sy(p[1]),3.5,0,2*Math.PI);
    ctx1.fillStyle='rgba(96,165,250,0.6)'; ctx1.fill();
  });

  // Mean point
  const msx=sx(r.mx), msy=sy(r.my);

  // PC arrows: length = sqrt(eigenvalue) in data units × scale factor
  const sc=makeScale(rawPts,W,H);
  const xRange=sc.mxx-sc.mnx, yRange=sc.mxy-sc.mny;
  const dataScale=Math.min((W-56)/xRange,(H-56)/yRange);

  function drawArrow(ctx,x0,y0,dx,dy,color,label){
    const len=Math.sqrt(dx**2+dy**2);
    if(len<1) return;
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x0+dx,y0+dy);
    ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.stroke();
    // Arrowhead
    const angle=Math.atan2(dy,dx);
    ctx.beginPath();
    ctx.moveTo(x0+dx,y0+dy);
    ctx.lineTo(x0+dx-8*Math.cos(angle-0.4),y0+dy-8*Math.sin(angle-0.4));
    ctx.lineTo(x0+dx-8*Math.cos(angle+0.4),y0+dy-8*Math.sin(angle+0.4));
    ctx.closePath(); ctx.fillStyle=color; ctx.fill();
    ctx.fillStyle=color; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(label,x0+dx+14*Math.cos(angle),y0+dy+14*Math.sin(angle));
  }

  const arrowScale = dataScale * 1.2;
  const s1=Math.sqrt(Math.max(r.l1,0)); const s2=Math.sqrt(Math.max(r.l2,0));
  drawArrow(ctx1,msx,msy, r.v1[0]*s1*arrowScale,-r.v1[1]*s1*arrowScale,'#fbbf24','PC1');
  drawArrow(ctx1,msx,msy, r.v2[0]*s2*arrowScale,-r.v2[1]*s2*arrowScale,'#f472b6','PC2');

  document.getElementById('evr1').textContent=(r.evr1*100).toFixed(1);
  document.getElementById('evr2').textContent=(r.evr2*100).toFixed(1);
}

// ── Panel 2: Scree ────────────────────────────────────────────────────────────
const cv2=document.getElementById('cvScree'); const ctx2=cv2.getContext('2d');
let screeY=null;

function drawScree(){
  const W=340,H=260,PAD=36;
  ctx2.clearRect(0,0,W,H);
  const r=pcaResult;
  const evals=[r.l1,r.l2];
  const total=r.total;
  const evrs=[r.l1/total,r.l2/total];
  const cumevrs=[evrs[0],evrs[0]+evrs[1]];

  const barW=(W-2*PAD)/2-10;
  const maxE=Math.max(...evals,0.01);

  // Bars
  evals.forEach((lam,k)=>{
    const x=PAD+k*(barW+10);
    const barH=(lam/maxE)*(H-2*PAD);
    const y=H-PAD-barH;
    ctx2.fillStyle='rgba(96,165,250,0.5)';
    ctx2.fillRect(x,y,barW,barH);
    ctx2.strokeStyle='#60a5fa'; ctx2.lineWidth=1;
    ctx2.strokeRect(x,y,barW,barH);
    ctx2.fillStyle='#94a3b8'; ctx2.font='10px sans-serif'; ctx2.textAlign='center';
    ctx2.fillText(`PC${k+1}`+`\n${(evrs[k]*100).toFixed(1)}%`,x+barW/2,H-PAD+14);
    ctx2.fillText(`λ=${lam.toFixed(2)}`,x+barW/2,y-6);
  });

  // Cumulative line
  ctx2.beginPath();
  ctx2.strokeStyle='#34d399'; ctx2.lineWidth=2; ctx2.setLineDash([]);
  cumevrs.forEach((cev,k)=>{
    const x=PAD+(k+0.5)*(barW+10);
    const y=H-PAD-(cev)*(H-2*PAD);
    k===0?ctx2.moveTo(x,y):ctx2.lineTo(x,y);
  });
  ctx2.stroke();
  cumevrs.forEach((cev,k)=>{
    const x=PAD+(k+0.5)*(barW+10);
    const y=H-PAD-cev*(H-2*PAD);
    ctx2.beginPath(); ctx2.arc(x,y,4,0,2*Math.PI);
    ctx2.fillStyle='#34d399'; ctx2.fill();
  });

  // Threshold line (draggable)
  const threshY=screeY!==null?screeY:H-PAD-screeThreshold*(H-2*PAD);
  ctx2.beginPath(); ctx2.setLineDash([5,3]);
  ctx2.moveTo(PAD,threshY); ctx2.lineTo(W-PAD,threshY);
  ctx2.strokeStyle='#fbbf24'; ctx2.lineWidth=1.5; ctx2.stroke();
  ctx2.setLineDash([]);
  const thrVal=((H-PAD-threshY)/(H-2*PAD));
  ctx2.fillStyle='#fbbf24'; ctx2.font='9px sans-serif'; ctx2.textAlign='right';
  ctx2.fillText(`${(thrVal*100).toFixed(0)}%`,W-PAD-2,threshY-3);

  // How many PCs to reach threshold?
  const needed=cumevrs.findIndex(c=>c>=thrVal)+1 || 2;
  document.getElementById('screeInfo').innerHTML=
    `Threshold: <span class="pv">${(thrVal*100).toFixed(0)}%</span> variance requires
     <span class="pv">${needed}</span> component(s). Drag the yellow line to change.`;
}

cv2.addEventListener('mousemove',e=>{
  const rect=cv2.getBoundingClientRect();
  screeY=e.clientY-rect.top;
  drawScree();
});
cv2.addEventListener('mouseleave',()=>{ screeY=null; drawScree(); });

// ── Panel 3: Projection distributions ────────────────────────────────────────
const cv3=document.getElementById('cvProj'); const ctx3=cv3.getContext('2d');
function drawProjection(){
  const W=340,H=260,PAD=24;
  ctx3.clearRect(0,0,W,H);
  const r=pcaResult;

  function drawHist(scores,color,yOffset,label,sigma){
    const mn=Math.min(...scores), mx=Math.max(...scores);
    const bins=20, binW=(mx-mn)/bins||1;
    const counts=new Array(bins).fill(0);
    scores.forEach(s=>{ const b=Math.min(bins-1,Math.floor((s-mn)/binW)); counts[b]++; });
    const maxC=Math.max(...counts,1);
    const plotH=80, xscale=(W-2*PAD)/(mx-mn||1);
    ctx3.fillStyle='#475569'; ctx3.font='9px sans-serif'; ctx3.textAlign='left';
    ctx3.fillText(label,PAD,yOffset-2);
    counts.forEach((c,i)=>{
      const x=PAD+i*((W-2*PAD)/bins);
      const h=c/maxC*plotH;
      ctx3.fillStyle=color+'99';
      ctx3.fillRect(x,yOffset+plotH-h,(W-2*PAD)/bins-1,h);
      ctx3.strokeStyle=color; ctx3.lineWidth=0.5;
      ctx3.strokeRect(x,yOffset+plotH-h,(W-2*PAD)/bins-1,h);
    });
    // Std dev marker
    const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
    const mx_=Math.max(...scores), mn_=Math.min(...scores);
    const cx=PAD+(mean-mn_)/(mx_-mn_||1)*(W-2*PAD);
    ctx3.beginPath(); ctx3.setLineDash([3,2]);
    ctx3.moveTo(cx,yOffset); ctx3.lineTo(cx,yOffset+plotH);
    ctx3.strokeStyle=color; ctx3.lineWidth=1.5; ctx3.stroke(); ctx3.setLineDash([]);
    ctx3.fillStyle=color; ctx3.font='8px sans-serif'; ctx3.textAlign='center';
    ctx3.fillText(`σ=${sigma.toFixed(2)}`,cx,yOffset+plotH+12);
  }

  drawHist(r.z1,'#fbbf24',30,`PC1 scores  (${(r.evr1*100).toFixed(1)}% variance)`,Math.sqrt(Math.max(r.l1,0)));
  drawHist(r.z2,'#f472b6',148,`PC2 scores  (${(r.evr2*100).toFixed(1)}% variance)`,Math.sqrt(Math.max(r.l2,0)));

  ctx3.fillStyle='#64748b'; ctx3.font='8px sans-serif'; ctx3.textAlign='center';
  ctx3.fillText('PC1 is always the widest distribution — it captures maximum variance.',W/2,250);
}

// ── Panel 4: Reconstruction error ────────────────────────────────────────────
const cv4=document.getElementById('cvRecon'); const ctx4=cv4.getContext('2d');
const cv5=document.getElementById('cvReconScatter'); const ctx5=cv5.getContext('2d');

document.getElementById('kSlider').max=2;
document.getElementById('kSlider').addEventListener('input',e=>{
  K=parseInt(e.target.value);
  document.getElementById('kVal').textContent=K;
  drawRecon();
});

function drawRecon(){
  const W=340,H=200,PAD=36;
  ctx4.clearRect(0,0,W,H);
  const r=pcaResult;
  const n=rawPts.length;

  // Compute RMSE for K=1 and K=2
  function rmseK(K){
    const vecs=K===1?[r.v1]:[r.v1,r.v2];
    const evals_k=K===1?[r.l1]:[r.l1,r.l2];
    let sse=0;
    r.Xc.forEach((p,i)=>{
      const scores=vecs.map(v=>p[0]*v[0]+p[1]*v[1]);
      const rec=[
        scores.reduce((s,sc,k)=>s+sc*vecs[k][0],0),
        scores.reduce((s,sc,k)=>s+sc*vecs[k][1],0),
      ];
      sse+=(p[0]-rec[0])**2+(p[1]-rec[1])**2;
    });
    return Math.sqrt(sse/(n*2));
  }

  const ks=[1,2];
  const rmses=ks.map(k=>rmseK(k));
  const varRet=[r.evr1,r.evr1+r.evr2];
  const maxRmse=Math.max(...rmses,0.01);

  ks.forEach((k,i)=>{
    const x=PAD+i*(W-2*PAD)/2+10;
    const barH=(rmses[i]/maxRmse)*(H-2*PAD)*0.8;
    const barW=(W-2*PAD)/2-20;
    const col=k===K?'#60a5fa':'#334155';
    ctx4.fillStyle=col+'aa';
    ctx4.fillRect(x,H-PAD-barH,barW,barH);
    ctx4.strokeStyle=col; ctx4.lineWidth=k===K?2:1;
    ctx4.strokeRect(x,H-PAD-barH,barW,barH);
    ctx4.fillStyle=k===K?'#60a5fa':'#64748b';
    ctx4.font=`${k===K?'bold ':' '}10px sans-serif`; ctx4.textAlign='center';
    ctx4.fillText(`K=${k}`,x+barW/2,H-PAD+13);
    ctx4.fillText(`RMSE=${rmses[i].toFixed(3)}`,x+barW/2,H-PAD-barH-14);
    ctx4.fillText(`Var=${(varRet[i]*100).toFixed(1)}%`,x+barW/2,H-PAD-barH-4);
  });

  document.getElementById('reconInfo').innerHTML=
    `K=<span class="pv">${K}</span> retains <span class="pv">${(varRet[K-1]*100).toFixed(1)}%</span>
     variance. Reconstruction RMSE = <span class="pv">${rmses[K-1].toFixed(3)}</span>.`;

  // Reconstruction scatter
  const W5=340,H5=120,P5=16;
  ctx5.clearRect(0,0,W5,H5);
  const vecs=K===1?[r.v1]:[r.v1,r.v2];
  const recPts=r.Xc.map(p=>{
    const scores=vecs.map(v=>p[0]*v[0]+p[1]*v[1]);
    return [
      scores.reduce((s,sc,k)=>s+sc*vecs[k][0],0)+r.mx,
      scores.reduce((s,sc,k)=>s+sc*vecs[k][1],0)+r.my,
    ];
  });

  const allPts=[...rawPts,...recPts];
  const {sx,sy}=makeScale(allPts,W5,H5,P5);

  rawPts.forEach((p,i)=>{
    ctx5.beginPath(); ctx5.moveTo(sx(p[0]),sy(p[1])); ctx5.lineTo(sx(recPts[i][0]),sy(recPts[i][1]));
    ctx5.strokeStyle='rgba(248,113,113,0.3)'; ctx5.lineWidth=0.8; ctx5.stroke();
  });
  rawPts.forEach(p=>{
    ctx5.beginPath(); ctx5.arc(sx(p[0]),sy(p[1]),2.5,0,2*Math.PI);
    ctx5.fillStyle='rgba(96,165,250,0.7)'; ctx5.fill();
  });
  recPts.forEach(p=>{
    ctx5.beginPath(); ctx5.arc(sx(p[0]),sy(p[1]),2.5,0,2*Math.PI);
    ctx5.fillStyle='rgba(251,191,36,0.8)'; ctx5.fill();
  });

  ctx5.fillStyle='#64748b'; ctx5.font='8px sans-serif'; ctx5.textAlign='center';
  ctx5.fillText('Blue=original  Gold=reconstructed  Red line=reconstruction error',W5/2,H5-2);
}

// ── Sliders ───────────────────────────────────────────────────────────────────
document.getElementById('rotSlider').addEventListener('input',e=>{
  currentRot=parseInt(e.target.value);
  document.getElementById('rotVal').textContent=currentRot+'°';
  if(currentDataset==='ellipse') rebuildData();
});
document.getElementById('noiseSlider').addEventListener('input',e=>{
  currentNoise=parseInt(e.target.value)/100;
  document.getElementById('noiseVal').textContent=currentNoise.toFixed(2);
  if(currentDataset==='ellipse') rebuildData();
});

function drawAll(){ drawScatter(); drawScree(); drawProjection(); drawRecon(); }

// ── Init ──────────────────────────────────────────────────────────────────────
rebuildData();
</script>
</body>
</html>
"""