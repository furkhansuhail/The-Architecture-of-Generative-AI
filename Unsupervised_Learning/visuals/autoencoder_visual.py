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
h2   { color: #fb923c; margin-bottom: 4px; }
.subtitle { color: #64748b; margin-bottom: 22px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #1e2130; border-radius: 12px; padding: 18px;
        border: 1px solid #2d3148; }
.card h3 { color: #fb923c; margin: 0 0 10px; font-size: 0.9em;
           text-transform: uppercase; letter-spacing: 0.05em; }
canvas { display: block; }
.params { background: #12141f; padding: 8px 12px; border-radius: 8px;
          font-size: 0.81em; color: #94a3b8; margin: 8px 0; line-height: 1.6; }
.pv { color: #fb923c; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 100px; }
input[type=range] { accent-color: #fb923c; flex: 1; }
.vb { font-size: 0.8em; color: #fb923c; min-width: 44px; }
.btn-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 8px; }
button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
         border-radius: 6px; padding: 5px 12px; cursor: pointer;
         font-size: 0.8em; transition: background 0.15s; }
button:hover { background: #3d4168; }
button.active { background: #fb923c; color: #0f1117; }
</style>
</head>
<body>
<h2>🔄 Autoencoder Visual Explorer</h2>
<p class="subtitle">Architecture, latent space, training dynamics, and anomaly detection</p>

<div class="grid">

  <!-- Panel 1: Architecture diagram -->
  <div class="card">
    <h3>Architecture &amp; Forward Pass</h3>
    <div class="params">
      Visualise how an input flows through the encoder bottleneck and decoder.
      Adjust bottleneck size to see how compression changes. Node brightness = activation.
    </div>
    <canvas id="cvArch" width="340" height="260"></canvas>
    <div class="slider-row">
      <label>Bottleneck dim</label>
      <input type="range" id="bottleneckSlider" min="1" max="6" step="1" value="2">
      <span class="vb" id="bottleneckVal">2</span>
    </div>
    <div class="params" id="archInfo">Input → 8 → <span class="pv">2</span> → 8 → Output</div>
  </div>

  <!-- Panel 2: Latent space scatter -->
  <div class="card">
    <h3>Latent Space Explorer</h3>
    <div class="params">
      2D latent codes for a 3-class dataset. Well-trained autoencoder clusters classes
      without labels. Click a point to decode it.
    </div>
    <canvas id="cvLatent" width="340" height="260" style="cursor:crosshair;"></canvas>
    <div class="params" id="latentInfo">
      Hover over points to inspect their latent coordinates.
    </div>
  </div>

  <!-- Panel 3: Training loss curve (live) -->
  <div class="card">
    <h3>Training Loss (Live)</h3>
    <div class="params">
      Watch reconstruction loss drop as the autoencoder trains.
      <span style="color:#fb923c">Orange</span> = total MSE loss per epoch.
    </div>
    <canvas id="cvLoss" width="340" height="230"></canvas>
    <div class="btn-row">
      <button onclick="startTraining()" id="btnTrain">▶ Train</button>
      <button onclick="resetTraining()">Reset</button>
    </div>
    <div class="params" id="trainInfo">Press Train to begin.</div>
  </div>

  <!-- Panel 4: Anomaly detection threshold -->
  <div class="card">
    <h3>Anomaly Detection</h3>
    <div class="params">
      Reconstruction error for normal (blue) and anomalous (red) samples.
      Drag the <span style="color:#fbbf24">threshold line</span> to set the decision boundary.
    </div>
    <canvas id="cvAnomaly" width="340" height="230" style="cursor:ns-resize;"></canvas>
    <div class="slider-row">
      <label>Noise level</label>
      <input type="range" id="noiseSlider" min="1" max="10" step="1" value="4">
      <span class="vb" id="noiseVal">0.4</span>
    </div>
    <div class="params" id="anomalyInfo">—</div>
  </div>

</div>

<script>
// ── Seeded RNG ────────────────────────────────────────────────────────────────
let sd=42;
const rng=()=>{sd=(sd*1664525+1013904223)>>>0;return sd/4294967296;};
const gauss=()=>{const u=rng()||1e-9,v=rng();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};

// ── PANEL 1: Architecture Diagram ────────────────────────────────────────────
const cvArch=document.getElementById('cvArch');
const ctxArch=cvArch.getContext('2d');

function drawArch(){
  const W=340,H=260;
  ctxArch.clearRect(0,0,W,H);
  const bn=parseInt(document.getElementById('bottleneckSlider').value);

  const layers=[8,5,bn,5,8];
  const layerX=[30,95,170,245,310];
  const PAL=['#60a5fa','#34d399','#fb923c','#34d399','#60a5fa'];
  const labels=['Input','Enc','z','Dec','Output'];
  const maxN=Math.max(...layers);

  // Draw connections (subset for clarity)
  ctxArch.strokeStyle='rgba(100,116,139,0.18)'; ctxArch.lineWidth=0.8;
  for(let l=0;l<layers.length-1;l++){
    const n1=layers[l],n2=layers[l+1];
    const y1s=Array.from({length:n1},(_,i)=>H/2+(i-(n1-1)/2)*Math.min(28,H/(n1+1)));
    const y2s=Array.from({length:n2},(_,i)=>H/2+(i-(n2-1)/2)*Math.min(28,H/(n2+1)));
    for(const y1 of y1s) for(const y2 of y2s){
      ctxArch.beginPath(); ctxArch.moveTo(layerX[l]+10,y1); ctxArch.lineTo(layerX[l+1]-10,y2);
      ctxArch.stroke();
    }
  }

  // Draw nodes with random activations
  layers.forEach((n,l)=>{
    const ys=Array.from({length:n},(_,i)=>H/2+(i-(n-1)/2)*Math.min(28,H/(n+1)));
    ys.forEach((y,i)=>{
      const act=l===0?0.7+rng()*0.3:l===2?(Math.sin(i*2.1+0.5)+1)/2:rng()*0.8+0.1;
      const col=PAL[l];
      ctxArch.beginPath(); ctxArch.arc(layerX[l],y,7,0,2*Math.PI);
      ctxArch.fillStyle=col+(Math.floor(act*200+55)).toString(16).padStart(2,'0');
      ctxArch.fill();
      ctxArch.strokeStyle=col; ctxArch.lineWidth=1.5; ctxArch.stroke();
    });
    ctxArch.fillStyle='#94a3b8'; ctxArch.font='8px sans-serif'; ctxArch.textAlign='center';
    ctxArch.fillText(labels[l],layerX[l],H-6);
    ctxArch.fillText(n,layerX[l],H-16);
  });

  // Bottleneck highlight
  const bnY=H/2; const bnX=layerX[2];
  ctxArch.strokeStyle='#fb923c'; ctxArch.lineWidth=1.5; ctxArch.setLineDash([3,2]);
  ctxArch.strokeRect(bnX-12,bnY-layers[2]*16-8,24,layers[2]*32+16);
  ctxArch.setLineDash([]);
  ctxArch.fillStyle='#fb923c'; ctxArch.font='bold 9px sans-serif'; ctxArch.textAlign='center';
  ctxArch.fillText('bottleneck',bnX,20);

  document.getElementById('bottleneckVal').textContent=bn;
  document.getElementById('archInfo').innerHTML=
    `Input(8) → Enc(5) → <span class="pv">z(${bn})</span> → Dec(5) → Output(8)&nbsp;&nbsp;
     Compression: <span class="pv">${(8/bn).toFixed(1)}×</span>`;
}

document.getElementById('bottleneckSlider').addEventListener('input',drawArch);
drawArch();

// ── PANEL 2: Latent Space ─────────────────────────────────────────────────────
const cvLat=document.getElementById('cvLatent');
const ctxLat=cvLat.getContext('2d');
const PAL=['#fb923c','#34d399','#60a5fa'];

// Generate synthetic latent codes (simulated well-trained AE)
sd=7;
const latentPts=[], latentLbls=[];
const cs2=[[50,120],[-40,-60],[80,-80]];
cs2.forEach(([cx,cy],ci)=>{
  for(let i=0;i<25;i++){
    latentPts.push([cx+gauss()*22,cy+gauss()*22]);
    latentLbls.push(ci);
  }
});

function drawLatent(hoverIdx=-1){
  const W=340,H=260,PAD=20;
  ctxLat.clearRect(0,0,W,H);
  const xs=latentPts.map(p=>p[0]),ys=latentPts.map(p=>p[1]);
  const mnx=Math.min(...xs)-10,mxx=Math.max(...xs)+10;
  const mny=Math.min(...ys)-10,mxy=Math.max(...ys)+10;
  const sx=x=>PAD+(x-mnx)/(mxx-mnx)*(W-2*PAD);
  const sy=y=>H-PAD-(y-mny)/(mxy-mny)*(H-2*PAD);

  // Axes
  ctxLat.strokeStyle='#2d3148'; ctxLat.lineWidth=0.5; ctxLat.setLineDash([3,3]);
  const cx=sx(0),cy=sy(0);
  ctxLat.beginPath(); ctxLat.moveTo(cx,PAD); ctxLat.lineTo(cx,H-PAD); ctxLat.stroke();
  ctxLat.beginPath(); ctxLat.moveTo(PAD,cy); ctxLat.lineTo(W-PAD,cy); ctxLat.stroke();
  ctxLat.setLineDash([]);

  latentPts.forEach((p,i)=>{
    const isHover=i===hoverIdx;
    ctxLat.beginPath(); ctxLat.arc(sx(p[0]),sy(p[1]),isHover?7:4.5,0,2*Math.PI);
    ctxLat.fillStyle=PAL[latentLbls[i]]+(isHover?'ff':'bb');
    ctxLat.fill();
    if(isHover){ctxLat.strokeStyle='#fff';ctxLat.lineWidth=1.5;ctxLat.stroke();}
  });

  ctxLat.fillStyle='#475569'; ctxLat.font='9px sans-serif'; ctxLat.textAlign='center';
  ctxLat.fillText('z₁ (PC1)',W/2,H-2);
  ctxLat.save(); ctxLat.translate(8,H/2); ctxLat.rotate(-Math.PI/2);
  ctxLat.fillText('z₂ (PC2)',0,0); ctxLat.restore();
}

cvLat.addEventListener('mousemove',e=>{
  const rect=cvLat.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  const W=340,H=260,PAD=20;
  const xs=latentPts.map(p=>p[0]),ys=latentPts.map(p=>p[1]);
  const mnx=Math.min(...xs)-10,mxx=Math.max(...xs)+10;
  const mny=Math.min(...ys)-10,mxy=Math.max(...ys)+10;
  const sx=x=>PAD+(x-mnx)/(mxx-mnx)*(W-2*PAD);
  const sy=y=>H-PAD-(y-mny)/(mxy-mny)*(H-2*PAD);
  let best=-1,bestD=Infinity;
  latentPts.forEach((p,i)=>{
    const d=Math.hypot(sx(p[0])-mx,sy(p[1])-my);
    if(d<bestD){bestD=d;best=i;}
  });
  const info=bestD<15?
    `Point ${best} | z=[${latentPts[best][0].toFixed(1)}, ${latentPts[best][1].toFixed(1)}] | Class ${latentLbls[best]+1}`:
    'Hover over a point to inspect its latent coordinates.';
  document.getElementById('latentInfo').textContent=info;
  drawLatent(bestD<15?best:-1);
});

drawLatent();

// ── PANEL 3: Training (Mini t-SNE-like simulation) ────────────────────────────
const cvLoss=document.getElementById('cvLoss');
const ctxLoss=cvLoss.getContext('2d');
let lossHistory=[], trainInterval=null, trainEpoch=0;
const TOTAL_EPOCHS=200;

// Simulate a plausible loss curve
function simulateLoss(epoch){
  const base=0.5*Math.exp(-epoch/60)+0.03+Math.sin(epoch*0.4)*0.003*Math.exp(-epoch/80);
  return Math.max(base,0.025)+Math.abs(gauss()*0.008);
}

function resetTraining(){
  sd=13;
  if(trainInterval){clearInterval(trainInterval);trainInterval=null;}
  trainEpoch=0; lossHistory=[];
  drawLoss();
  document.getElementById('btnTrain').textContent='▶ Train';
  document.getElementById('trainInfo').textContent='Press Train to begin.';
}

function startTraining(){
  if(trainInterval){
    clearInterval(trainInterval); trainInterval=null;
    document.getElementById('btnTrain').textContent='▶ Train';
  } else {
    document.getElementById('btnTrain').textContent='⏸ Pause';
    trainInterval=setInterval(()=>{
      if(trainEpoch>=TOTAL_EPOCHS){ clearInterval(trainInterval); trainInterval=null;
        document.getElementById('btnTrain').textContent='▶ Train'; return; }
      trainEpoch++;
      lossHistory.push(simulateLoss(trainEpoch));
      drawLoss();
    },30);
  }
}

function drawLoss(){
  const W=340,H=230,PAD=32;
  ctxLoss.clearRect(0,0,W,H);
  if(lossHistory.length<2){
    ctxLoss.fillStyle='#475569'; ctxLoss.font='11px sans-serif'; ctxLoss.textAlign='center';
    ctxLoss.fillText('Press Train to see loss curve',W/2,H/2); return;
  }
  const maxL=Math.max(...lossHistory)||1;
  const sx=t=>PAD+t/(TOTAL_EPOCHS)*(W-2*PAD);
  const sy=v=>H-PAD-(v/maxL)*(H-2*PAD);

  // Grid
  ctxLoss.strokeStyle='#2d3148'; ctxLoss.lineWidth=0.5;
  for(let g=0;g<=4;g++){
    const y=PAD+g/4*(H-2*PAD);
    ctxLoss.beginPath(); ctxLoss.moveTo(PAD,y); ctxLoss.lineTo(W-PAD,y); ctxLoss.stroke();
    ctxLoss.fillStyle='#475569'; ctxLoss.font='8px sans-serif'; ctxLoss.textAlign='right';
    ctxLoss.fillText((maxL*(1-g/4)).toFixed(3),PAD-2,y+3);
  }

  // Fill under curve
  ctxLoss.beginPath();
  ctxLoss.moveTo(sx(0),H-PAD);
  lossHistory.forEach((v,t)=>ctxLoss.lineTo(sx(t+1),sy(v)));
  ctxLoss.lineTo(sx(lossHistory.length),H-PAD); ctxLoss.closePath();
  ctxLoss.fillStyle='rgba(251,146,60,0.15)'; ctxLoss.fill();

  // Curve
  ctxLoss.beginPath();
  lossHistory.forEach((v,t)=>{ t===0?ctxLoss.moveTo(sx(t+1),sy(v)):ctxLoss.lineTo(sx(t+1),sy(v)); });
  ctxLoss.strokeStyle='#fb923c'; ctxLoss.lineWidth=2; ctxLoss.stroke();

  // Current dot
  const last=lossHistory[lossHistory.length-1];
  ctxLoss.beginPath(); ctxLoss.arc(sx(lossHistory.length),sy(last),4,0,2*Math.PI);
  ctxLoss.fillStyle='#fb923c'; ctxLoss.fill();

  ctxLoss.fillStyle='#64748b'; ctxLoss.font='9px sans-serif'; ctxLoss.textAlign='center';
  ctxLoss.fillText('Epoch',W/2,H-4);

  const pct=(lossHistory.length/TOTAL_EPOCHS*100).toFixed(0);
  document.getElementById('trainInfo').innerHTML=
    `Epoch <span class="pv">${lossHistory.length}</span> / ${TOTAL_EPOCHS} &nbsp;|&nbsp;
     Loss: <span class="pv">${last.toFixed(4)}</span> &nbsp;|&nbsp;
     ${pct}% complete`;
}

// ── PANEL 4: Anomaly Detection ─────────────────────────────────────────────────
const cvAnom=document.getElementById('cvAnomaly');
const ctxAnom=cvAnom.getContext('2d');
let anomThresholdY=null;

function genAnomalyData(noiseMult){
  sd=33;
  const normal=Array.from({length:30},()=>0.02+Math.abs(gauss()*0.018+gauss()*0.012));
  const anomalous=Array.from({length:15},()=>0.04*noiseMult+Math.abs(gauss()*0.03*noiseMult));
  return {normal,anomalous};
}

function drawAnomaly(){
  const W=340,H=230,PAD=28;
  ctxAnom.clearRect(0,0,W,H);
  const noiseM=parseInt(document.getElementById('noiseSlider').value)/4;
  const {normal,anomalous}=genAnomalyData(noiseM);

  const allVals=[...normal,...anomalous];
  const maxV=Math.max(...allVals)*1.1||0.5;

  const sy=v=>H-PAD-(v/maxV)*(H-2*PAD);

  // Threshold (draggable)
  const threshVal=anomThresholdY!==null?(H-PAD-anomThresholdY)/(H-2*PAD)*maxV:
    (normal.reduce((a,b)=>a+b,0)/normal.length + 3*Math.sqrt(normal.reduce((s,v)=>{
      const m=normal.reduce((a,b)=>a+b,0)/normal.length; return s+(v-m)**2;},0)/normal.length));

  const threshY=sy(threshVal);

  // Draw jittered dots
  const drawDots=(vals,col,xOff)=>{
    vals.forEach((v,i)=>{
      const x=PAD+xOff+(rng()-0.5)*30;
      ctxAnom.beginPath(); ctxAnom.arc(x,sy(v),4.5,0,2*Math.PI);
      ctxAnom.fillStyle=col+(v>threshVal?'ff':'88');
      ctxAnom.fill();
      if(v>threshVal){ctxAnom.strokeStyle='#fbbf24';ctxAnom.lineWidth=1.5;ctxAnom.stroke();}
    });
  };
  sd=44; drawDots(normal,'#60a5fa',80); drawDots(anomalous,'#ef4444',230);

  // Threshold line
  ctxAnom.beginPath(); ctxAnom.setLineDash([5,3]);
  ctxAnom.moveTo(PAD,threshY); ctxAnom.lineTo(W-PAD,threshY);
  ctxAnom.strokeStyle='#fbbf24'; ctxAnom.lineWidth=2; ctxAnom.stroke();
  ctxAnom.setLineDash([]);
  ctxAnom.fillStyle='#fbbf24'; ctxAnom.font='9px sans-serif'; ctxAnom.textAlign='right';
  ctxAnom.fillText(`threshold=${threshVal.toFixed(4)}`,W-PAD-2,threshY-4);

  // Labels
  ctxAnom.fillStyle='#60a5fa'; ctxAnom.font='9px sans-serif'; ctxAnom.textAlign='center';
  ctxAnom.fillText('Normal',PAD+80,H-PAD+14);
  ctxAnom.fillStyle='#ef4444';
  ctxAnom.fillText('Anomalous',PAD+230,H-PAD+14);
  ctxAnom.fillStyle='#94a3b8'; ctxAnom.textAlign='left';
  ctxAnom.fillText('↑ Reconstruction error',PAD+2,PAD+10);

  // Stats
  const tp=anomalous.filter(v=>v>threshVal).length;
  const fp=normal.filter(v=>v>threshVal).length;
  const fn=anomalous.filter(v=>v<=threshVal).length;
  const tn=normal.filter(v=>v<=threshVal).length;
  const prec=tp/(tp+fp+1e-9), rec=tp/(tp+fn+1e-9);
  document.getElementById('anomalyInfo').innerHTML=
    `Threshold: <span class="pv">${threshVal.toFixed(4)}</span> &nbsp;|&nbsp;
     TP:<span class="pv">${tp}</span> FP:<span class="pv">${fp}</span>
     FN:<span class="pv">${fn}</span> TN:<span class="pv">${tn}</span> &nbsp;|&nbsp;
     Precision:<span class="pv">${prec.toFixed(2)}</span>
     Recall:<span class="pv">${rec.toFixed(2)}</span>`;
}

cvAnom.addEventListener('mousemove',e=>{
  const rect=cvAnom.getBoundingClientRect();
  anomThresholdY=e.clientY-rect.top;
  drawAnomaly();
});
cvAnom.addEventListener('mouseleave',()=>{anomThresholdY=null; drawAnomaly();});
document.getElementById('noiseSlider').addEventListener('input',e=>{
  document.getElementById('noiseVal').textContent=(parseInt(e.target.value)/10).toFixed(1);
  drawAnomaly();
});

// ── Init ──────────────────────────────────────────────────────────────────────
drawLoss(); drawAnomaly();
</script>
</body>
</html>
"""