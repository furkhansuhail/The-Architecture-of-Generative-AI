from Unsupervised_Learning.visuals.tSNE_UMAP_visual import VISUAL_HEIGHT

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
h2   { color: #a78bfa; margin-bottom: 4px; }
.subtitle { color: #64748b; margin-bottom: 22px; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #1e2130; border-radius: 12px; padding: 18px;
        border: 1px solid #2d3148; }
.card h3 { color: #a78bfa; margin: 0 0 10px; font-size: 0.9em;
           text-transform: uppercase; letter-spacing: 0.05em; }
canvas { display: block; }
.params { background: #12141f; padding: 8px 12px; border-radius: 8px;
          font-size: 0.81em; color: #94a3b8; margin: 8px 0; line-height: 1.6; }
.pv { color: #a78bfa; font-weight: bold; }
.slider-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
.slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 95px; }
input[type=range] { accent-color: #a78bfa; flex: 1; }
.vb { font-size: 0.8em; color: #a78bfa; min-width: 32px; }
.btn-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 8px; }
button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168;
         border-radius: 6px; padding: 5px 12px; cursor: pointer;
         font-size: 0.8em; transition: background 0.15s; }
button:hover { background: #3d4168; }
button.active { background: #a78bfa; color: #0f1117; }
</style>
</head>
<body>
<h2>&#128202; GMM Visual Explorer</h2>
<p class="subtitle">Gaussian Mixture Models — EM algorithm, soft assignments, density estimation</p>

<div class="grid">

  <div class="card">
    <h3>EM Algorithm (Step-by-Step)</h3>
    <div class="params">
      Ellipses show each Gaussian at the 1&#963; contour. Point colour = blended responsibility.
      E-step updates colours; M-step reshapes the ellipses.
    </div>
    <canvas id="cvEM" width="340" height="260"></canvas>
    <div class="btn-row">
      <button onclick="emStep()">&#9654; EM Step</button>
      <button onclick="autoEM()" id="btnAuto">Auto</button>
      <button onclick="resetEM()">Reset</button>
    </div>
    <div class="params" id="emInfo">Press EM Step to begin.</div>
  </div>

  <div class="card">
    <h3>Soft Responsibility Surface</h3>
    <div class="params">
      Colour at each pixel = component with highest responsibility.
      Brightness = confidence. Faded zones = boundary uncertainty.
    </div>
    <canvas id="cvResp" width="340" height="260"></canvas>
    <div class="params" id="respInfo">Runs after first EM step.</div>
  </div>

  <div class="card">
    <h3>Log-Likelihood Convergence</h3>
    <div class="params">
      EM is guaranteed to increase log-likelihood every iteration.
      Watch it climb monotonically until convergence.
    </div>
    <canvas id="cvLL" width="340" height="230"></canvas>
    <div class="params" id="llInfo">Log-likelihood appears here as EM runs.</div>
  </div>

  <div class="card">
    <h3>AIC / BIC Model Selection</h3>
    <div class="params">
      Penalised likelihood for K = 1 to 7. Lower = better.
      True K = 3 (star). Slide to inspect each model.
    </div>
    <canvas id="cvAIC" width="340" height="200"></canvas>
    <div class="slider-row">
      <label>Highlight K</label>
      <input type="range" id="kSlider" min="1" max="7" step="1" value="3">
      <span class="vb" id="kVal">3</span>
    </div>
    <div class="params" id="aicInfo">—</div>
  </div>

</div>

<script>
let sd = 42;
const rng  = () => { sd = (sd*1664525+1013904223)>>>0; return sd/4294967296; };
const gauss = () => {
  const u=rng()||1e-9, v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
};
const PAL      = ['#a78bfa','#34d399','#f472b6'];
const COLS_RGB = [[167,139,250],[52,211,153],[244,114,182]];

function genData(){
  sd=42; const pts=[],lbl=[];
  [[80,130],[220,100],[150,210]].forEach(([cx,cy],ci)=>{
    for(let i=0;i<22;i++){ pts.push([cx+gauss()*22,cy+gauss()*22]); lbl.push(ci); }
  });
  return {pts,lbl};
}
const {pts:DATA,lbl:TRUE_LBL}=genData();
const N=DATA.length, K=3;

let mus,vars_,pis,R,iter=0,llHistory=[];

function resetEM(){
  sd=9; iter=0; llHistory=[];
  const init=[Math.floor(rng()*N),Math.floor(rng()*N),Math.floor(rng()*N)];
  mus  =init.map(i=>[...DATA[i]]);
  vars_=Array.from({length:K},()=>[500,500]);
  pis  =Array.from({length:K},()=>1/K);
  R    =Array.from({length:N},()=>Array(K).fill(1/K));
  document.getElementById('emInfo').textContent='Initialised. Press EM Step.';
  document.getElementById('llInfo').textContent='Log-likelihood appears here.';
  drawEM(); drawResp(); drawLL();
}

function gpdf(x,mu,v){
  const dx=x[0]-mu[0],dy=x[1]-mu[1],det=v[0]*v[1];
  if(det<1e-12) return 0;
  return Math.exp(-0.5*(dx*dx/v[0]+dy*dy/v[1]))/(2*Math.PI*Math.sqrt(det));
}

function eStep(){
  for(let i=0;i<N;i++){
    const nums=pis.map((pi,k)=>pi*gpdf(DATA[i],mus[k],vars_[k]));
    const d=nums.reduce((a,b)=>a+b,1e-300);
    R[i]=nums.map(v=>v/d);
  }
}
function mStep(){
  const Nk=Array.from({length:K},(_,k)=>Math.max(R.reduce((s,r)=>s+r[k],0),1e-6));
  pis=Nk.map(nk=>nk/N);
  mus=Array.from({length:K},(_,k)=>
    [0,1].map(j=>R.reduce((s,r,i)=>s+r[k]*DATA[i][j],0)/Nk[k]));
  vars_=Array.from({length:K},(_,k)=>
    [0,1].map(j=>Math.max(R.reduce((s,r,i)=>s+r[k]*(DATA[i][j]-mus[k][j])**2,0)/Nk[k],30)));
}
function logLL(){
  return DATA.reduce((s,xi)=>
    s+Math.log(Math.max(pis.reduce((a,pi,k)=>a+pi*gpdf(xi,mus[k],vars_[k]),0),1e-300)),0);
}

function emStep(){
  eStep(); mStep(); iter++;
  const ll=logLL(); llHistory.push(ll);
  document.getElementById('emInfo').innerHTML=
    `Iteration <span class="pv">${iter}</span> &nbsp;|&nbsp; LL = <span class="pv">${ll.toFixed(2)}</span>`;
  drawEM(); drawResp(); drawLL();
}

let autoInt=null;
function autoEM(){
  if(autoInt){clearInterval(autoInt);autoInt=null;
    document.getElementById('btnAuto').textContent='Auto';return;}
  document.getElementById('btnAuto').textContent='Stop';
  autoInt=setInterval(()=>{if(iter>=40){clearInterval(autoInt);autoInt=null;
    document.getElementById('btnAuto').textContent='Auto';return;} emStep();},180);
}

const cvEM=document.getElementById('cvEM'),ctxEM=cvEM.getContext('2d');
function drawEM(){
  const W=340,H=260; ctxEM.clearRect(0,0,W,H);
  for(let k=0;k<K;k++){
    const [mx,my]=mus[k],[vx,vy]=vars_[k];
    ctxEM.beginPath(); ctxEM.ellipse(mx,my,Math.sqrt(vx),Math.sqrt(vy),0,0,2*Math.PI);
    ctxEM.strokeStyle=PAL[k]+'bb'; ctxEM.lineWidth=2.5; ctxEM.stroke();
    ctxEM.fillStyle=PAL[k]+'14'; ctxEM.fill();
    ctxEM.beginPath(); ctxEM.arc(mx,my,5,0,2*Math.PI);
    ctxEM.fillStyle=PAL[k]; ctxEM.fill();
    ctxEM.fillStyle=PAL[k]; ctxEM.font='bold 9px sans-serif'; ctxEM.textAlign='center';
    ctxEM.fillText('C'+(k+1),mx,my-9);
  }
  DATA.forEach((p,i)=>{
    let cr=0,cg=0,cb=0;
    COLS_RGB.forEach(([r,g,b],k)=>{cr+=R[i][k]*r;cg+=R[i][k]*g;cb+=R[i][k]*b;});
    ctxEM.beginPath(); ctxEM.arc(p[0],p[1],4,0,2*Math.PI);
    ctxEM.fillStyle=`rgb(${Math.round(cr)},${Math.round(cg)},${Math.round(cb)})`;
    ctxEM.fill(); ctxEM.strokeStyle='#0f1117'; ctxEM.lineWidth=0.8; ctxEM.stroke();
  });
  ctxEM.fillStyle='#475569'; ctxEM.font='9px sans-serif'; ctxEM.textAlign='right';
  ctxEM.fillText(iter===0?'Random init':`iter ${iter} | pi=[${pis.map(p=>p.toFixed(2)).join(',')}]`,W-6,H-5);
}

const cvResp=document.getElementById('cvResp'),ctxResp=cvResp.getContext('2d');
function drawResp(){
  if(iter===0){ctxResp.clearRect(0,0,340,260);
    ctxResp.fillStyle='#475569';ctxResp.font='11px sans-serif';ctxResp.textAlign='center';
    ctxResp.fillText('Run EM to see responsibility surface',170,130);return;}
  const W=340,H=260,STEP=5;
  const img=ctxResp.createImageData(W,H);
  for(let py=0;py<H;py+=STEP) for(let px=0;px<W;px+=STEP){
    const nums=pis.map((pi,k)=>pi*gpdf([px,py],mus[k],vars_[k]));
    const d=nums.reduce((a,b)=>a+b,1e-300);
    const r=nums.map(v=>v/d); const conf=Math.max(...r);
    let cr=0,cg=0,cb=0;
    COLS_RGB.forEach(([ri,gi,bi],k)=>{cr+=r[k]*ri;cg+=r[k]*gi;cb+=r[k]*bi;});
    const alpha=Math.round(conf*200+30);
    for(let dy=0;dy<STEP&&py+dy<H;dy++) for(let dx=0;dx<STEP&&px+dx<W;dx++){
      const idx=((py+dy)*W+(px+dx))*4;
      img.data[idx]=cr;img.data[idx+1]=cg;img.data[idx+2]=cb;img.data[idx+3]=alpha;
    }
  }
  ctxResp.clearRect(0,0,W,H); ctxResp.putImageData(img,0,0);
  DATA.forEach((p,i)=>{
    ctxResp.beginPath(); ctxResp.arc(p[0],p[1],3,0,2*Math.PI);
    ctxResp.fillStyle=PAL[R[i].indexOf(Math.max(...R[i]))]; ctxResp.fill();
  });
  const mc=DATA.reduce((s,_,i)=>s+Math.max(...R[i]),0)/N;
  document.getElementById('respInfo').innerHTML=
    `Mean max-responsibility: <span class="pv">${mc.toFixed(3)}</span> (1.0 = fully certain, 0.33 = maximum uncertainty)`;
}

const cvLL=document.getElementById('cvLL'),ctxLL=cvLL.getContext('2d');
function drawLL(){
  const W=340,H=230,PAD=36; ctxLL.clearRect(0,0,W,H);
  if(llHistory.length<1){ctxLL.fillStyle='#475569';ctxLL.font='11px sans-serif';ctxLL.textAlign='center';
    ctxLL.fillText('Log-likelihood will appear here',W/2,H/2);return;}
  const maxLL=Math.max(...llHistory),minLL=Math.min(...llHistory),range=maxLL-minLL||1;
  const sx=t=>PAD+t/Math.max(llHistory.length-1,1)*(W-2*PAD);
  const sy=v=>H-PAD-(v-minLL)/range*(H-2*PAD);
  ctxLL.strokeStyle='#2d3148'; ctxLL.lineWidth=0.5;
  [0,0.25,0.5,0.75,1].forEach(f=>{
    const y=PAD+f*(H-2*PAD); ctxLL.beginPath(); ctxLL.moveTo(PAD,y); ctxLL.lineTo(W-PAD,y); ctxLL.stroke();
    ctxLL.fillStyle='#475569'; ctxLL.font='8px sans-serif'; ctxLL.textAlign='right';
    ctxLL.fillText((maxLL-(maxLL-minLL)*f).toFixed(0),PAD-2,y+3);
  });
  ctxLL.beginPath(); ctxLL.moveTo(sx(0),H-PAD);
  llHistory.forEach((v,t)=>ctxLL.lineTo(sx(t),sy(v)));
  ctxLL.lineTo(sx(llHistory.length-1),H-PAD); ctxLL.closePath();
  ctxLL.fillStyle='rgba(167,139,250,0.12)'; ctxLL.fill();
  ctxLL.beginPath(); llHistory.forEach((v,t)=>t===0?ctxLL.moveTo(sx(t),sy(v)):ctxLL.lineTo(sx(t),sy(v)));
  ctxLL.strokeStyle='#a78bfa'; ctxLL.lineWidth=2; ctxLL.stroke();
  const last=llHistory[llHistory.length-1];
  ctxLL.beginPath(); ctxLL.arc(sx(llHistory.length-1),sy(last),4,0,2*Math.PI);
  ctxLL.fillStyle='#a78bfa'; ctxLL.fill();
  ctxLL.fillStyle='#64748b'; ctxLL.font='9px sans-serif'; ctxLL.textAlign='center';
  ctxLL.fillText('Iteration',W/2,H-4);
  const delta=llHistory.length>1?last-llHistory[llHistory.length-2]:0;
  document.getElementById('llInfo').innerHTML=
    `LL = <span class="pv">${last.toFixed(2)}</span> &nbsp;|&nbsp; delta = <span class="pv">${delta.toFixed(4)}</span> &nbsp;|&nbsp; iter <span class="pv">${iter}</span>`;
}

const cvAIC=document.getElementById('cvAIC'),ctxAIC=cvAIC.getContext('2d');
const SIM_LLS=[-420,-360,-305,-278,-274,-272,-271];
const SIM_NP=[3,7,11,15,19,23,27];
const SIM_N=66;
const AICS=SIM_LLS.map((ll,i)=>-2*ll+2*SIM_NP[i]);
const BICS=SIM_LLS.map((ll,i)=>-2*ll+SIM_NP[i]*Math.log(SIM_N));
const BEST_AIC=AICS.indexOf(Math.min(...AICS))+1;
const BEST_BIC=BICS.indexOf(Math.min(...BICS))+1;

function drawAICBIC(){
  const W=340,H=200,PAD=32; ctxAIC.clearRect(0,0,W,H);
  const K_=parseInt(document.getElementById('kSlider').value);
  const allV=[...AICS,...BICS];
  const minV=Math.min(...allV)*0.99,maxV=Math.max(...allV)*1.01;
  const sx=k=>PAD+(k-1)/6*(W-2*PAD);
  const sy=v=>H-PAD-(v-minV)/(maxV-minV)*(H-2*PAD);
  ctxAIC.strokeStyle='#2d3148'; ctxAIC.lineWidth=0.5;
  [0,0.5,1].forEach(f=>{const y=PAD+f*(H-2*PAD);ctxAIC.beginPath();ctxAIC.moveTo(PAD,y);ctxAIC.lineTo(W-PAD,y);ctxAIC.stroke();});
  ctxAIC.fillStyle='rgba(167,139,250,0.07)'; ctxAIC.fillRect(sx(K_)-12,PAD,24,H-2*PAD);
  ctxAIC.beginPath(); AICS.forEach((v,i)=>i===0?ctxAIC.moveTo(sx(i+1),sy(v)):ctxAIC.lineTo(sx(i+1),sy(v)));
  ctxAIC.strokeStyle='#f472b6'; ctxAIC.lineWidth=2; ctxAIC.stroke();
  AICS.forEach((v,i)=>{ctxAIC.beginPath();ctxAIC.arc(sx(i+1),sy(v),i+1===K_?6:3.5,0,2*Math.PI);ctxAIC.fillStyle='#f472b6';ctxAIC.fill();});
  ctxAIC.beginPath(); BICS.forEach((v,i)=>i===0?ctxAIC.moveTo(sx(i+1),sy(v)):ctxAIC.lineTo(sx(i+1),sy(v)));
  ctxAIC.strokeStyle='#38bdf8'; ctxAIC.lineWidth=2; ctxAIC.stroke();
  BICS.forEach((v,i)=>{ctxAIC.beginPath();ctxAIC.arc(sx(i+1),sy(v),i+1===K_?6:3.5,0,2*Math.PI);ctxAIC.fillStyle='#38bdf8';ctxAIC.fill();});
  for(let k=1;k<=7;k++){
    ctxAIC.fillStyle=k===K_?'#a78bfa':'#64748b';
    ctxAIC.font=k===K_?'bold 9px sans-serif':'9px sans-serif'; ctxAIC.textAlign='center';
    ctxAIC.fillText('K='+k,sx(k),H-PAD+13);
    if(k===3){ctxAIC.fillStyle='#fbbf24';ctxAIC.fillText('★',sx(k),H-PAD+23);}
  }
  ctxAIC.fillStyle='#f472b6';ctxAIC.font='9px sans-serif';ctxAIC.textAlign='left';ctxAIC.fillText('— AIC',PAD,PAD+8);
  ctxAIC.fillStyle='#38bdf8';ctxAIC.fillText('— BIC',PAD+50,PAD+8);
  ctxAIC.fillStyle='#fbbf24';ctxAIC.fillText('★ True K',PAD+100,PAD+8);
  document.getElementById('aicInfo').innerHTML=
    `K = <span class="pv">${K_}</span> &nbsp;|&nbsp; AIC min at K=<span class="pv">${BEST_AIC}</span> &nbsp;|&nbsp; BIC min at K=<span class="pv">${BEST_BIC}</span>`;
}

document.getElementById('kSlider').addEventListener('input',e=>{
  document.getElementById('kVal').textContent=e.target.value; drawAICBIC();
});
resetEM(); drawAICBIC();
</script>
</body>
</html>
"""