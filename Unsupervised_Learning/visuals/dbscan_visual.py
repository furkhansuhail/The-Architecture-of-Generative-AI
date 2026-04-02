DBSCAN_VISUAL_HEIGHT = 1300
DBSCAN_VISUAL_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0;
         margin: 0; padding: 20px; }
  h2   { color: #a78bfa; margin-bottom: 4px; }
  .subtitle { color: #64748b; margin-bottom: 24px; font-size: 0.9em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: #1e2130; border-radius: 12px; padding: 20px;
          border: 1px solid #2d3148; }
  .card h3 { color: #a78bfa; margin: 0 0 12px; font-size: 0.95em;
             text-transform: uppercase; letter-spacing: 0.05em; }
  canvas { display: block; margin: 0 auto; }
  .legend { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.82em; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .params { background: #12141f; padding: 10px 14px; border-radius: 8px;
            font-size: 0.82em; color: #94a3b8; margin-bottom: 12px; }
  .param-val { color: #a78bfa; font-weight: bold; }
  .slider-row { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
  .slider-row label { font-size: 0.82em; color: #94a3b8; min-width: 80px; }
  input[type=range] { accent-color: #a78bfa; flex: 1; }
  .val-badge { font-size: 0.82em; color: #a78bfa; min-width: 36px; }
</style>
</head>
<body>
<h2>🌀 DBSCAN Visual Explorer</h2>
<p class="subtitle">Interactive density-based clustering — adjust ε and MinPts to see point types change</p>

<div class="grid">

  <!-- Panel 1: Point type visualiser -->
  <div class="card" style="grid-column: 1 / -1;">
    <h3>Core / Border / Noise Classification</h3>
    <div class="slider-row">
      <label>ε (radius)</label>
      <input type="range" id="epsSlider" min="0.1" max="1.2" step="0.05" value="0.45">
      <span class="val-badge" id="epsVal">0.45</span>
    </div>
    <div class="slider-row">
      <label>MinPts</label>
      <input type="range" id="mpSlider" min="2" max="8" step="1" value="3">
      <span class="val-badge" id="mpVal">3</span>
    </div>
    <div class="params" id="statsBar">Stats loading...</div>
    <canvas id="cvMain" width="680" height="280"></canvas>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#a78bfa;"></div>Core point</div>
      <div class="legend-item"><div class="dot" style="background:#38bdf8;"></div>Border point</div>
      <div class="legend-item"><div class="dot" style="background:#ef4444; border-radius:0; width:10px;height:10px;transform:rotate(45deg)"></div>Noise point</div>
    </div>
  </div>

  <!-- Panel 2: K-distance curve -->
  <div class="card">
    <h3>k-Distance Graph (ε Selection)</h3>
    <canvas id="cvKdist" width="310" height="200"></canvas>
    <div class="params" style="margin-top:10px;">
      Sorted k-distances (k=MinPts). The <span style="color:#fbbf24">elbow</span>
      suggests the optimal ε. Points left of elbow are in dense regions; right = sparse/noise.
    </div>
  </div>

  <!-- Panel 3: Cluster expansion animation -->
  <div class="card">
    <h3>Seeds Expansion Trace</h3>
    <canvas id="cvExpand" width="310" height="200"></canvas>
    <div style="display:flex; gap:8px; margin-top:10px;">
      <button onclick="stepExpand()" style="background:#a78bfa;color:#fff;border:none;
              border-radius:6px;padding:5px 14px;cursor:pointer;font-size:0.82em;">Step →</button>
      <button onclick="resetExpand()" style="background:#2d3148;color:#e2e8f0;border:none;
              border-radius:6px;padding:5px 14px;cursor:pointer;font-size:0.82em;">Reset</button>
    </div>
    <div class="params" id="expandInfo" style="margin-top:8px;">Press Step to begin seeds expansion from first core point.</div>
  </div>

</div>

<script>
// ── Shared data ──────────────────────────────────────────────────────────────
const seed = 42;
function seededRand(s){ let x = Math.sin(s)*10000; return x - Math.floor(x); }

function genData(){
  const pts = [];
  // 3 clusters
  const centres = [[100,100],[220,170],[160,230]];
  const sig = 22;
  let si = seed;
  for(const [cx,cy] of centres){
    for(let i=0;i<20;i++){
      const u1=seededRand(si++), u2=seededRand(si++);
      const z1=Math.sqrt(-2*Math.log(u1+1e-9))*Math.cos(2*Math.PI*u2);
      const z2=Math.sqrt(-2*Math.log(u1+1e-9))*Math.sin(2*Math.PI*u2);
      pts.push([cx+z1*sig, cy+z2*sig]);
    }
  }
  // noise
  for(let i=0;i<8;i++){
    pts.push([seededRand(si++)*280+20, seededRand(si++)*240+20]);
  }
  return pts;
}

const RAW = genData();

function dist(a,b){ return Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2); }

function classify(pts, eps, mp){
  const nb = pts.map(p => pts.filter(q => dist(p,q) <= eps).length);
  const isCore   = pts.map((_,i) => nb[i] >= mp);
  const isBorder = pts.map((p,i) => !isCore[i] &&
    pts.some((q,j) => isCore[j] && dist(p,q) <= eps));
  return pts.map((_,i) => isCore[i] ? 'core' : isBorder[i] ? 'border' : 'noise');
}

// ── Panel 1: Main scatter ─────────────────────────────────────────────────────
const cv1 = document.getElementById('cvMain');
const ctx1 = cv1.getContext('2d');

function drawMain(){
  const eps = parseFloat(document.getElementById('epsSlider').value);
  const mp  = parseInt(document.getElementById('mpSlider').value);
  document.getElementById('epsVal').textContent = eps.toFixed(2);
  document.getElementById('mpVal').textContent  = mp;

  const types = classify(RAW, eps, mp);
  const nCore   = types.filter(t=>t==='core').length;
  const nBorder = types.filter(t=>t==='border').length;
  const nNoise  = types.filter(t=>t==='noise').length;
  document.getElementById('statsBar').innerHTML =
    `ε = <span class="param-val">${eps.toFixed(2)}</span> &nbsp;|&nbsp;
     MinPts = <span class="param-val">${mp}</span> &nbsp;|&nbsp;
     Core: <span class="param-val">${nCore}</span> &nbsp;
     Border: <span class="param-val">${nBorder}</span> &nbsp;
     Noise: <span class="param-val">${nNoise}</span>`;

  // Scale RAW to canvas
  const W=680, H=280, PAD=30;
  const xs=RAW.map(p=>p[0]), ys=RAW.map(p=>p[1]);
  const mnx=Math.min(...xs)-10, mxx=Math.max(...xs)+10;
  const mny=Math.min(...ys)-10, mxy=Math.max(...ys)+10;
  const sx=p=>PAD+(p[0]-mnx)/(mxx-mnx)*(W-2*PAD);
  const sy=p=>PAD+(p[1]-mny)/(mxy-mny)*(H-2*PAD);

  ctx1.clearRect(0,0,W,H);

  // Draw ε circles for core points (subtle)
  RAW.forEach((p,i)=>{
    if(types[i]==='core'){
      const pxEps = eps/(mxx-mnx)*(W-2*PAD);
      ctx1.beginPath();
      ctx1.arc(sx(p), sy(p), pxEps, 0, 2*Math.PI);
      ctx1.strokeStyle='rgba(167,139,250,0.15)';
      ctx1.lineWidth=1;
      ctx1.stroke();
      ctx1.fillStyle='rgba(167,139,250,0.04)';
      ctx1.fill();
    }
  });

  // Draw points
  const COLS={core:'#a78bfa', border:'#38bdf8', noise:'#ef4444'};
  RAW.forEach((p,i)=>{
    const t=types[i];
    ctx1.beginPath();
    if(t==='noise'){
      // draw X
      const cx=sx(p), cy=sy(p), s=6;
      ctx1.moveTo(cx-s,cy-s); ctx1.lineTo(cx+s,cy+s);
      ctx1.moveTo(cx+s,cy-s); ctx1.lineTo(cx-s,cy+s);
      ctx1.strokeStyle=COLS[t]; ctx1.lineWidth=2; ctx1.stroke();
    } else {
      ctx1.arc(sx(p), sy(p), t==='core'?6:5, 0, 2*Math.PI);
      ctx1.fillStyle=COLS[t]; ctx1.fill();
      if(t==='border'){ctx1.strokeStyle='#1e2130';ctx1.lineWidth=1.5;ctx1.stroke();}
    }
  });
}

document.getElementById('epsSlider').addEventListener('input', drawMain);
document.getElementById('mpSlider').addEventListener('input', drawMain);
drawMain();

// ── Panel 2: k-Distance graph ─────────────────────────────────────────────────
const cv2 = document.getElementById('cvKdist');
const ctx2 = cv2.getContext('2d');

function drawKdist(){
  const mp = parseInt(document.getElementById('mpSlider').value);
  const k  = mp;
  const W=310, H=200, PAD=28;

  const kdists = RAW.map(p=>{
    const d = RAW.map(q=>dist(p,q)).sort((a,b)=>a-b);
    return d[k] || d[d.length-1];
  }).sort((a,b)=>b-a);

  // Find elbow: max second difference
  let elbowIdx=1;
  let maxD2=0;
  for(let i=1;i<kdists.length-1;i++){
    const d2=Math.abs(kdists[i-1]-2*kdists[i]+kdists[i+1]);
    if(d2>maxD2){maxD2=d2;elbowIdx=i;}
  }

  ctx2.clearRect(0,0,W,H);

  const maxK=Math.max(...kdists);
  const sx=i=>PAD+i/(kdists.length-1)*(W-2*PAD);
  const sy=v=>H-PAD-(v/maxK)*(H-2*PAD);

  // Grid
  ctx2.strokeStyle='#2d3148'; ctx2.lineWidth=0.5;
  for(let g=0;g<=4;g++){
    const y=PAD+g/4*(H-2*PAD);
    ctx2.beginPath(); ctx2.moveTo(PAD,y); ctx2.lineTo(W-PAD,y); ctx2.stroke();
  }

  // Curve
  ctx2.beginPath();
  kdists.forEach((v,i)=>{
    i===0?ctx2.moveTo(sx(i),sy(v)):ctx2.lineTo(sx(i),sy(v));
  });
  ctx2.strokeStyle='#38bdf8'; ctx2.lineWidth=2; ctx2.stroke();

  // Elbow marker
  const ex=sx(elbowIdx), ey=sy(kdists[elbowIdx]);
  ctx2.beginPath(); ctx2.arc(ex,ey,5,0,2*Math.PI);
  ctx2.fillStyle='#fbbf24'; ctx2.fill();

  // Elbow label
  ctx2.fillStyle='#fbbf24'; ctx2.font='10px sans-serif';
  ctx2.fillText(`ε≈${kdists[elbowIdx].toFixed(2)}`, ex+6, ey-4);

  // Axes labels
  ctx2.fillStyle='#64748b'; ctx2.font='9px sans-serif';
  ctx2.fillText('Points (sorted)', W/2-20, H-5);
  ctx2.save(); ctx2.translate(10,H/2); ctx2.rotate(-Math.PI/2);
  ctx2.fillText('k-dist', -18, 0); ctx2.restore();
}

document.getElementById('mpSlider').addEventListener('input', drawKdist);
drawKdist();

// ── Panel 3: Expansion trace ──────────────────────────────────────────────────
const cv3     = document.getElementById('cvExpand');
const ctx3    = cv3.getContext('2d');
const W3=310, H3=200, PAD3=20;

// Use a small subset of points for clarity
const EXPAND_PTS = RAW.slice(0,20);
const EXP_EPS    = 0.45;
const EXP_MP     = 3;

// Scale
const xs3=EXPAND_PTS.map(p=>p[0]), ys3=EXPAND_PTS.map(p=>p[1]);
const mnx3=Math.min(...xs3)-5, mxx3=Math.max(...xs3)+5;
const mny3=Math.min(...ys3)-5, mxy3=Math.max(...ys3)+5;
const sx3=p=>PAD3+(p[0]-mnx3)/(mxx3-mnx3)*(W3-2*PAD3);
const sy3=p=>PAD3+(p[1]-mny3)/(mxy3-mny3)*(H3-2*PAD3);

// Precompute expansion steps
let expandSteps=[], expandStep=0;

function buildExpansion(){
  const pts=EXPAND_PTS;
  const n=pts.length;
  const EPS=EXP_EPS*100, MP=EXP_MP;  // scale back
  const scaledEps=EXP_EPS*(mxx3-mnx3);

  const nb=pts.map(p=>pts.map((q,j)=>({j,d:dist(p,q)}))
    .filter(({d})=>d<=scaledEps).map(({j})=>j));
  const isCore=pts.map((_,i)=>nb[i].length>=MP);

  // Find first core point
  const startIdx = isCore.findIndex(Boolean);
  if(startIdx===-1) return [];

  const steps=[];
  steps.push({type:'start', idx:startIdx, visited:new Set([startIdx]),
              cluster: new Set([startIdx]), seeds:[...nb[startIdx]]});

  let visited=new Set([startIdx]);
  let cluster=new Set([startIdx]);
  let seeds=[...nb[startIdx]];
  let si=0;

  while(si<seeds.length && si<30){
    const q=seeds[si];
    const newVisited=new Set(visited);
    const newCluster=new Set(cluster);
    newVisited.add(q);
    newCluster.add(q);
    const newSeeds=[...seeds];
    if(isCore[q]){
      for(const nbq of nb[q]){
        if(!newSeeds.includes(nbq)) newSeeds.push(nbq);
      }
    }
    steps.push({type:'expand', idx:q, processing:si, visited:newVisited,
                cluster:newCluster, seeds:newSeeds, isCore:isCore[q]});
    visited=newVisited; cluster=newCluster; seeds=newSeeds;
    si++;
  }
  return steps;
}

expandSteps = buildExpansion();

function drawExpand(){
  const pts=EXPAND_PTS;
  const s = expandSteps[expandStep] || expandSteps[expandSteps.length-1];
  ctx3.clearRect(0,0,W3,H3);

  const scaledEps=EXP_EPS*(mxx3-mnx3);
  const nb=pts.map(p=>pts.map((q,j)=>({j,d:dist(p,q)}))
    .filter(({d})=>d<=scaledEps).map(({j})=>j));
  const isCore=pts.map((_,i)=>nb[i].length>=EXP_MP);

  // Draw ε circle around current point
  if(s && s.idx!==undefined){
    const pxEps=EXP_EPS*(mxx3-mnx3)/(mxx3-mnx3)*(W3-2*PAD3);
    ctx3.beginPath();
    ctx3.arc(sx3(pts[s.idx]),sy3(pts[s.idx]),pxEps,0,2*Math.PI);
    ctx3.strokeStyle='rgba(251,191,36,0.4)'; ctx3.lineWidth=1.5; ctx3.stroke();
    ctx3.fillStyle='rgba(251,191,36,0.06)'; ctx3.fill();
  }

  // Draw points
  pts.forEach((p,i)=>{
    let col='#334155', r=4;
    if(s){
      if(s.cluster && s.cluster.has(i)){col='#a78bfa';r=isCore[i]?6:4;}
      else if(s.seeds && s.seeds.includes(i)){col='#38bdf8';r=4;}
      if(s.idx===i){col='#fbbf24';r=7;}
    }
    ctx3.beginPath(); ctx3.arc(sx3(p),sy3(p),r,0,2*Math.PI);
    ctx3.fillStyle=col; ctx3.fill();
  });

  const info=document.getElementById('expandInfo');
  if(!s) return;
  if(s.type==='start'){
    info.innerHTML=`<span style="color:#a78bfa">★ Start:</span> Point ${s.idx} is a Core Point. Seeds = [${s.seeds.join(',')}]`;
  } else {
    info.innerHTML=`<span style="color:#fbbf24">→ Step ${expandStep}:</span> Visiting seed point ${s.idx}. 
    ${s.isCore?'<span style="color:#a78bfa">Core point — expanding frontier.</span>':'<span style="color:#38bdf8">Border point — added to cluster.</span>'}
    Cluster size: ${s.cluster.size}`;
  }
}

function stepExpand(){
  if(expandStep < expandSteps.length-1) expandStep++;
  drawExpand();
}
function resetExpand(){ expandStep=0; drawExpand(); }

document.getElementById('epsSlider').addEventListener('input', ()=>{expandSteps=buildExpansion();resetExpand();drawKdist();});
document.getElementById('mpSlider').addEventListener('input', ()=>{expandSteps=buildExpansion();resetExpand();drawKdist();});

drawExpand();
</script>
</body>
</html>
"""