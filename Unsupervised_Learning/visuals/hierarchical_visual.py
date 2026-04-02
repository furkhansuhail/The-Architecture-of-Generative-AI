from xml.dom import HIERARCHY_REQUEST_ERR

from torch.utils.model_dump import hierarchical_pickle

HIERARCHICAL_VISUAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0;
         margin: 0; padding: 20px; }
  h2   { color: #34d399; margin-bottom: 4px; }
  .subtitle { color: #64748b; margin-bottom: 24px; font-size: 0.9em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: #1e2130; border-radius: 12px; padding: 20px;
          border: 1px solid #2d3148; }
  .card h3 { color: #34d399; margin: 0 0 12px; font-size: 0.95em;
             text-transform: uppercase; letter-spacing: 0.05em; }
  canvas { display: block; }
  .params { background: #12141f; padding: 8px 12px; border-radius: 8px;
            font-size: 0.82em; color: #94a3b8; margin-bottom: 10px; }
  .param-val { color: #34d399; font-weight: bold; }
  .slider-row { display: flex; align-items: center; gap: 10px; margin-top: 6px; }
  .slider-row label { font-size: 0.82em; color: #94a3b8; min-width: 80px; }
  input[type=range] { accent-color: #34d399; flex: 1; }
  .val-badge { font-size: 0.82em; color: #34d399; min-width: 40px; }
  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
           border-radius: 6px; padding: 5px 13px; cursor: pointer;
           font-size: 0.82em; transition: background 0.15s; }
  button:hover { background: #3d4168; }
  button.active { background: #34d399; color: #0f1117; }
  .legend { display: flex; gap: 14px; margin-top: 10px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.8em; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
</style>
</head>
<body>
<h2>🌿 Hierarchical Clustering Visual Explorer</h2>
<p class="subtitle">Agglomerative clustering — build a dendrogram step by step and explore linkage methods</p>

<div class="grid">

  <!-- Panel 1: Scatter + step-by-step merging -->
  <div class="card">
    <h3>Merge Step Animator</h3>
    <div class="params">
      Linkage:
      <span id="lnkLabel" class="param-val">WARD</span> &nbsp;|&nbsp;
      Step: <span id="stepLabel" class="param-val">0</span> /
            <span id="totalSteps" class="param-val">-</span> &nbsp;|&nbsp;
      Clusters: <span id="nClusters" class="param-val">-</span>
    </div>
    <canvas id="cvScatter" width="340" height="270"></canvas>
    <div class="btn-row">
      <button onclick="prevStep()">← Back</button>
      <button onclick="nextStep()">Step →</button>
      <button onclick="resetSteps()">Reset</button>
      <button onclick="autoPlay()">▶ Auto</button>
    </div>
    <div class="params" id="stepInfo" style="margin-top:8px;">Press Step to start merging.</div>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#34d399"></div>Active cluster</div>
      <div class="legend-item"><div class="dot" style="background:#fbbf24"></div>Just merged</div>
      <div class="legend-item"><div class="dot" style="background:#475569"></div>Pending</div>
    </div>
  </div>

  <!-- Panel 2: Live dendrogram -->
  <div class="card">
    <h3>Dendrogram</h3>
    <div class="params">
      Drag cut line to extract clusters.
      Current cut: <span id="cutLabel" class="param-val">-</span> →
      <span id="cutClusters" class="param-val">-</span> clusters
    </div>
    <canvas id="cvDendrogram" width="340" height="270" style="cursor:crosshair;"></canvas>
    <div class="btn-row">
      <button onclick="setLinkage('single')" id="btnSingle">Single</button>
      <button onclick="setLinkage('complete')" id="btnComplete">Complete</button>
      <button onclick="setLinkage('average')" id="btnAverage">Average</button>
      <button onclick="setLinkage('ward')" id="btnWard" class="active">Ward</button>
    </div>
    <div class="params" style="margin-top:8px;">
      Switch linkage to see how the dendrogram shape changes. Ward = compact clusters.
      Single = chaining effect.
    </div>
  </div>

  <!-- Panel 3: Linkage distance comparison -->
  <div class="card" style="grid-column: 1 / -1;">
    <h3>Linkage Comparison — Cluster Shapes</h3>
    <canvas id="cvCompare" width="700" height="200"></canvas>
    <div class="params" style="margin-top:8px;">
      Each panel shows the K=3 cluster assignment from a different linkage on the same data.
      Ward produces compact spherical clusters. Single linkage is susceptible to chaining.
    </div>
  </div>

</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────────
const SEED_POINTS = (() => {
  let s = 1;
  const rng = () => { s = (s*1664525+1013904223)&0xffffffff; return (s>>>0)/4294967296; };
  const pts = [];
  const centres = [[60,80],[190,90],[125,195]];
  for(const [cx,cy] of centres){
    for(let i=0;i<7;i++){
      pts.push({ x: cx+(rng()*2-1)*30, y: cy+(rng()*2-1)*30, trueLabel: centres.indexOf([cx,cy]) });
    }
  }
  // fix true labels
  for(let i=0;i<pts.length;i++) pts[i].trueLabel = Math.floor(i/7);
  return pts;
})();

const PALETTE = ['#34d399','#f472b6','#38bdf8','#fbbf24','#a78bfa','#fb923c'];
const N = SEED_POINTS.length;

function dist(a,b){ return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2); }

function clusterDist(ci, cj, linkage, pts){
  const pairs = ci.flatMap(i=>cj.map(j=>dist(pts[i],pts[j])));
  if(linkage==='single')   return Math.min(...pairs);
  if(linkage==='complete') return Math.max(...pairs);
  if(linkage==='average')  return pairs.reduce((a,b)=>a+b,0)/pairs.length;
  if(linkage==='ward'){
    const ni=ci.length, nj=cj.length;
    const mi=[0,1].map(d=>ci.reduce((s,i)=>s+(d===0?pts[i].x:pts[i].y),0)/ni);
    const mj=[0,1].map(d=>cj.reduce((s,i)=>s+(d===0?pts[i].x:pts[i].y),0)/nj);
    return (ni*nj/(ni+nj))*((mi[0]-mj[0])**2+(mi[1]-mj[1])**2);
  }
}

// ── Run agglomerative ─────────────────────────────────────────────────────────
let currentLinkage = 'ward';
let mergeHistory = [];
let mergeStates  = [];  // snapshots for animation

function runAgglom(linkage){
  const pts = SEED_POINTS;
  let clusters = pts.map((_,i)=>([i]));  // array of member arrays
  let clusterIds = pts.map((_,i)=>i);    // parallel ids
  const history = [];
  const states  = [];

  // Initial state
  const initColour = pts.map((_,i)=>i);
  states.push({ colours: [...initColour], mergedPair: null, height: 0, nClusters: pts.length });

  while(clusters.length > 1){
    let best = { i:-1, j:-1, d:Infinity };
    for(let i=0;i<clusters.length;i++){
      for(let j=i+1;j<clusters.length;j++){
        const d = clusterDist(clusters[i],clusters[j],linkage,pts);
        if(d<best.d) best={i,j,d};
      }
    }
    const ci=clusters[best.i], cj=clusters[best.j];
    const newCluster=[...ci,...cj];
    const newId = clusterIds[best.i];  // inherit first cluster's colour
    history.push({ left:ci, right:cj, leftId:clusterIds[best.i],
                   rightId:clusterIds[best.j], height:best.d,
                   members:newCluster });

    // New colour assignment for snapshot
    const newColours = pts.map((_,k)=>{
      const prevCol = states[states.length-1].colours[k];
      if(cj.includes(k)) return newId;  // recolour merged
      return prevCol;
    });
    clusters.splice(best.j,1); clusterIds.splice(best.j,1);
    clusters[best.i]=newCluster;
    states.push({ colours:[...newColours], mergedPair:[...ci,...cj],
                  height:best.d, nClusters:clusters.length,
                  msg:`Merged ${ci.length+cj.length} pts at dist=${best.d.toFixed(1)}` });
  }
  return { history, states };
}

let result = runAgglom(currentLinkage);
mergeHistory = result.history;
mergeStates  = result.states;
let currentStep = 0;
let autoInterval = null;

// ── Canvas 1: Scatter ─────────────────────────────────────────────────────────
const cv1 = document.getElementById('cvScatter');
const cx1 = cv1.getContext('2d');

function drawScatter(){
  const s = mergeStates[currentStep];
  const W=340,H=270;
  cx1.clearRect(0,0,W,H);

  // Draw lines between merged points
  if(s.mergedPair){
    for(let a=0;a<s.mergedPair.length;a++){
      for(let b=a+1;b<s.mergedPair.length;b++){
        const pa=SEED_POINTS[s.mergedPair[a]], pb=SEED_POINTS[s.mergedPair[b]];
        cx1.beginPath(); cx1.moveTo(pa.x,pa.y); cx1.lineTo(pb.x,pb.y);
        cx1.strokeStyle='rgba(251,191,36,0.2)'; cx1.lineWidth=1; cx1.stroke();
      }
    }
  }

  SEED_POINTS.forEach((p,i)=>{
    const cid = s.colours[i];
    const col = s.mergedPair && s.mergedPair.includes(i) ? '#fbbf24' : PALETTE[cid%PALETTE.length];
    cx1.beginPath(); cx1.arc(p.x,p.y,5,0,2*Math.PI);
    cx1.fillStyle=col; cx1.fill();
    cx1.strokeStyle='#1e2130'; cx1.lineWidth=1.5; cx1.stroke();
  });

  document.getElementById('stepLabel').textContent = currentStep;
  document.getElementById('totalSteps').textContent = mergeStates.length-1;
  document.getElementById('nClusters').textContent = s.nClusters;
  document.getElementById('stepInfo').textContent = s.msg || `Initial: ${N} singleton clusters.`;
}

function nextStep(){ if(currentStep<mergeStates.length-1){ currentStep++; drawScatter(); drawDendrogram(); }}
function prevStep(){ if(currentStep>0){ currentStep--; drawScatter(); drawDendrogram(); }}
function resetSteps(){ currentStep=0; if(autoInterval){clearInterval(autoInterval);autoInterval=null;} drawScatter(); drawDendrogram(); }
function autoPlay(){
  if(autoInterval){ clearInterval(autoInterval); autoInterval=null; return; }
  autoInterval=setInterval(()=>{ if(currentStep>=mergeStates.length-1){ clearInterval(autoInterval);autoInterval=null; return; } nextStep(); },600);
}

// ── Canvas 2: Dendrogram ──────────────────────────────────────────────────────
const cv2 = document.getElementById('cvDendrogram');
const cx2 = cv2.getContext('2d');
let cutHeight = null;
let maxMergeHeight = 1;

function buildDendrogramLayout(history, pts){
  if(!history.length) return { leafX:{}, nodes:[] };
  // Assign leaf x positions
  const leafOrder = [...Array(N).keys()];
  const W=340, H=270, PAD=24, BOTTOM=H-PAD, TOP=PAD;
  const leafX = {};
  leafOrder.forEach((i,idx)=>{ leafX[i]=PAD+idx*(W-2*PAD)/(N-1); });

  const maxH = Math.max(...history.map(m=>m.height));
  maxMergeHeight = maxH || 1;
  const scaleH = h => BOTTOM - (h/maxMergeHeight)*(BOTTOM-TOP);

  // For each merge, compute x = midpoint of children
  const nodeX = {};
  history.forEach((m,i)=>{
    const lx = m.left.length===1 ? leafX[m.left[0]] :
               (nodeX[history.slice(0,i).findLastIndex(h=>JSON.stringify(h.members)===JSON.stringify([...m.left].sort((a,b)=>a-b))) ] ?? (m.left.reduce((s,v)=>s+leafX[v],0)/m.left.length));
    const rx = m.right.length===1 ? leafX[m.right[0]] :
               (m.right.reduce((s,v)=>s+(leafX[v]??0),0)/m.right.length);
    nodeX[i] = (lx+rx)/2 || (m.members.reduce((s,v)=>s+(leafX[v]??0),0)/m.members.length);
  });

  return { leafX, nodeX, scaleH, maxH, W, H, PAD, BOTTOM };
}

function drawDendrogram(){
  const W=340,H=270,PAD=24,BOTTOM=H-PAD,TOP=PAD;
  cx2.clearRect(0,0,W,H);

  const history = mergeHistory.slice(0, currentStep);
  if(!history.length){
    // Just draw leaves
    SEED_POINTS.forEach((_,i)=>{
      const x = PAD+i*(W-2*PAD)/(N-1);
      cx2.beginPath(); cx2.arc(x,BOTTOM,3,0,2*Math.PI);
      cx2.fillStyle=PALETTE[i%PALETTE.length]; cx2.fill();
    });
    document.getElementById('cutLabel').textContent='—';
    document.getElementById('cutClusters').textContent=N;
    return;
  }

  const maxH = Math.max(...mergeHistory.map(m=>m.height));
  const scaleH = h => BOTTOM - (h/maxH)*(BOTTOM-TOP);
  const cutY = cutHeight!==null ? scaleH(cutHeight*maxH) : null;

  // Compute leaf x (order by first appearance in merges)
  const visited=new Set(); const order=[];
  mergeHistory.forEach(m=>{ m.members.forEach(i=>{ if(!visited.has(i)){visited.add(i);order.push(i);} }); });
  const leafX={};
  order.forEach((i,idx)=>{ leafX[i]=PAD+idx*(W-2*PAD)/(N>1?N-1:1); });

  // Draw horizontal/vertical lines for each merge
  const getMidX = (members) => members.reduce((s,i)=>s+(leafX[i]||0),0)/members.length;

  mergeHistory.slice(0,currentStep).forEach((m,idx)=>{
    const y  = scaleH(m.height);
    const lx = getMidX(m.left);
    const rx = getMidX(m.right);
    const mx = getMidX(m.members);
    const ly = m.left.length===1  ? BOTTOM : scaleH(mergeHistory.slice(0,idx).find(h=>JSON.stringify(h.members.sort())==JSON.stringify([...m.left].sort()))?.height||0);
    const ry = m.right.length===1 ? BOTTOM : scaleH(mergeHistory.slice(0,idx).find(h=>JSON.stringify(h.members.sort())==JSON.stringify([...m.right].sort()))?.height||0);

    const isCurrent = idx===currentStep-1;
    const col = isCurrent ? '#fbbf24' : '#34d399';
    cx2.strokeStyle=col; cx2.lineWidth=isCurrent?2:1.5;

    // Vertical from children up to merge height
    cx2.beginPath(); cx2.moveTo(lx,ly); cx2.lineTo(lx,y); cx2.stroke();
    cx2.beginPath(); cx2.moveTo(rx,ry); cx2.lineTo(rx,y); cx2.stroke();
    // Horizontal bar
    cx2.beginPath(); cx2.moveTo(lx,y); cx2.lineTo(rx,y); cx2.stroke();
  });

  // Draw leaves
  order.forEach((i,idx)=>{
    cx2.beginPath(); cx2.arc(leafX[i],BOTTOM,3,0,2*Math.PI);
    cx2.fillStyle=PALETTE[Math.floor(idx/7)%PALETTE.length]; cx2.fill();
  });

  // Draw leaf labels
  cx2.fillStyle='#64748b'; cx2.font='8px sans-serif'; cx2.textAlign='center';
  order.forEach((i,idx)=>{ cx2.fillText(i, leafX[i], BOTTOM+12); });

  // Draw cut line
  if(cutY!==null){
    cx2.beginPath(); cx2.setLineDash([4,3]);
    cx2.moveTo(PAD,cutY); cx2.lineTo(W-PAD,cutY);
    cx2.strokeStyle='#f87171'; cx2.lineWidth=1.5; cx2.stroke();
    cx2.setLineDash([]);

    // Count clusters at this cut
    const cutH_val = cutHeight*maxH;
    const nCut = mergeHistory.filter(m=>m.height>cutH_val).length + 1;
    document.getElementById('cutLabel').textContent=(cutHeight*maxH).toFixed(1);
    document.getElementById('cutClusters').textContent=nCut;
    cx2.fillStyle='#f87171'; cx2.font='9px sans-serif'; cx2.textAlign='left';
    cx2.fillText(`h=${(cutHeight*maxH).toFixed(1)}`, W-PAD-30, cutY-4);
  } else {
    document.getElementById('cutLabel').textContent='—';
    document.getElementById('cutClusters').textContent=currentStep<mergeHistory.length?N-currentStep:1;
  }
}

// Drag to set cut height
cv2.addEventListener('mousemove', e=>{
  const rect=cv2.getBoundingClientRect();
  const y=e.clientY-rect.top;
  const H=270,PAD=24,BOTTOM=H-PAD,TOP=PAD;
  cutHeight=Math.max(0,Math.min(1,(BOTTOM-y)/(BOTTOM-TOP)));
  drawDendrogram();
});
cv2.addEventListener('mouseleave',()=>{ cutHeight=null; drawDendrogram(); });

// ── Linkage switcher ──────────────────────────────────────────────────────────
function setLinkage(lnk){
  currentLinkage=lnk;
  result=runAgglom(lnk);
  mergeHistory=result.history;
  mergeStates =result.states;
  currentStep=mergeStates.length-1;  // show full dendrogram
  ['single','complete','average','ward'].forEach(l=>{
    document.getElementById(`btn${l.charAt(0).toUpperCase()+l.slice(1)}`).classList.toggle('active',l===lnk);
  });
  document.getElementById('lnkLabel').textContent=lnk.toUpperCase();
  drawScatter(); drawDendrogram(); drawCompare();
}

// ── Canvas 3: Linkage comparison ──────────────────────────────────────────────
const cv3 = document.getElementById('cvCompare');
const cx3 = cv3.getContext('2d');

function drawCompare(){
  const W=700,H=200,PAD=20;
  cx3.clearRect(0,0,W,H);
  const lnks=['single','complete','average','ward'];
  const panelW=(W-PAD*(lnks.length+1))/lnks.length;

  lnks.forEach((lnk,li)=>{
    const res=runAgglom(lnk);
    const finalState=res.states[res.states.length-1];
    const px=PAD+li*(panelW+PAD);

    // Draw panel bg
    cx3.fillStyle='#12141f'; cx3.fillRect(px,PAD,panelW,H-2*PAD);

    // Scale points to panel
    const xs=SEED_POINTS.map(p=>p.x), ys=SEED_POINTS.map(p=>p.y);
    const mnx=Math.min(...xs),mxx=Math.max(...xs);
    const mny=Math.min(...ys),mxy=Math.max(...ys);
    const sx=x=>px+8+(x-mnx)/(mxx-mnx)*(panelW-16);
    const sy=y=>PAD+8+(y-mny)/(mxy-mny)*(H-2*PAD-20);

    SEED_POINTS.forEach((p,i)=>{
      const cid=finalState.colours[i];
      cx3.beginPath(); cx3.arc(sx(p.x),sy(p.y),4,0,2*Math.PI);
      cx3.fillStyle=PALETTE[cid%PALETTE.length]; cx3.fill();
    });

    // Label
    cx3.fillStyle= lnk===currentLinkage?'#34d399':'#94a3b8';
    cx3.font=`${lnk===currentLinkage?'bold ':' '}10px sans-serif`;
    cx3.textAlign='center';
    cx3.fillText(lnk.toUpperCase(), px+panelW/2, H-PAD/2+4);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
currentStep = mergeStates.length-1;
drawScatter(); drawDendrogram(); drawCompare();
</script>
</body>
</html>
"""

HIERARCHICAL_VISUAL_HEIGHT = 1300