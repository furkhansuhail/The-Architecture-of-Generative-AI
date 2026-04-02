KUBERNETES_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', sans-serif;
  background: #08080f;
  color: #e4e4e7;
  padding: 20px;
  min-height: 100vh;
}

h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.15em; color: #326ce5; margin-bottom: 3px; }
.subtitle { color: #52525b; font-size: 0.8em; margin-bottom: 20px; }

.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.full { grid-column: 1 / -1; }

.card {
  background: #111118;
  border: 1px solid #1e1e2e;
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.card h3 {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78em;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #326ce5;
  margin-bottom: 12px;
}

canvas { display: block; border-radius: 8px; }

.row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.row label { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #71717a; min-width: 100px; }
input[type=range] {
  flex: 1; height: 4px; border-radius: 2px;
  -webkit-appearance: none; appearance: none;
  background: #1e1e2e; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px;
  border-radius: 50%; background: #326ce5; cursor: pointer;
}
.val { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #326ce5; min-width: 46px; text-align: right; }

.info-box {
  background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px;
  padding: 8px 12px; font-size: 0.78em; color: #94a3b8;
  line-height: 1.7; margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
}
.hl  { color: #326ce5; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }
.hlg { color: #4ade80; font-weight: 700; }
.hlr { color: #f87171; font-weight: 700; }
.hlp { color: #c084fc; font-weight: 700; }

.btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
button {
  background: #1e1e2e; color: #a1a1aa;
  border: 1px solid #2d2d40; border-radius: 6px;
  padding: 5px 12px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75em; font-weight: 700;
  transition: all 0.15s;
}
button:hover { background: #2d2d40; color: #e4e4e7; }
button.active { background: #326ce5; color: #08080f; border-color: #326ce5; }

/* Quiz */
.quiz-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 4px;
}
.q-card {
  background: #0d0d18;
  border: 1px solid #1e1e2e;
  border-radius: 10px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.q-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68em;
  color: #3f3f46;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.q-prompt {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78em;
  color: #cbd5e1;
  line-height: 1.55;
  font-weight: 700;
}
.q-opt {
  background: #111118;
  border: 1px solid #2d2d40;
  border-radius: 6px;
  padding: 6px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72em;
  color: #71717a;
  cursor: pointer;
  text-align: left;
  transition: all 0.15s;
  line-height: 1.4;
  width: 100%;
}
.q-opt:hover:not([disabled]) { background: #1e1e2e; color: #e4e4e7; border-color: #3f3f46; }
.q-opt.correct  { border-color: #4ade80; color: #4ade80; background: #0a1f0f !important; }
.q-opt.wrong    { border-color: #ef4444; color: #ef4444; background: #1a0808 !important; }
.q-feedback {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.71em;
  line-height: 1.55;
  padding: 6px 8px;
  border-radius: 6px;
  margin-top: 2px;
  display: none;
}
.q-feedback.show { display: block; }
.q-feedback.ok  { background: #0a1f0f; border: 1px solid #4ade8040; color: #4ade80; }
.q-feedback.bad { background: #1a0808; border: 1px solid #ef444440; color: #ef4444; }

.score-bar {
  display: flex; align-items: center; gap: 14px;
  margin-top: 14px; padding: 10px 16px;
  background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.8em;
}
.pip {
  width: 11px; height: 11px; border-radius: 50%;
  background: #1e1e2e; border: 1px solid #2d2d40;
  transition: all 0.3s; flex-shrink: 0;
}
.pip.ok  { background: #4ade80; border-color: #4ade80; box-shadow: 0 0 6px #4ade8066; }
.pip.bad { background: #ef4444; border-color: #ef4444; }
</style>
</head>
<body>

<h2>&#9096; Kubernetes &#8212; Container Orchestration at Scale</h2>
<p class="subtitle">Cluster architecture &middot; Pods &middot; Deployments &middot; Services &middot; Scheduling &middot; Quiz</p>

<div class="grid">

<!-- ══ PANEL 1: Cluster Architecture ══ -->
<div class="card full">
  <h3>&#9312; Cluster Architecture &#8212; Control Plane &amp; Worker Nodes</h3>
  <canvas id="cvCluster" width="860" height="290"></canvas>
  <div class="btn-row" style="margin-top:10px;">
    <button id="btnCP"  class="active" onclick="setClusterFocus('cp')">Control Plane</button>
    <button id="btnNode" onclick="setClusterFocus('node')">Worker Node</button>
    <button id="btnFlow" onclick="setClusterFocus('flow')">kubectl Flow</button>
    <button id="btnEtcd" onclick="setClusterFocus('etcd')">etcd</button>
    <button id="btnAll"  onclick="setClusterFocus('all')">Full Cluster</button>
  </div>
  <div class="info-box" id="clusterInfo">
    The <span class="hl">Control Plane</span> is the brain of Kubernetes — it makes all scheduling and orchestration decisions. It runs the API Server, Scheduler, Controller Manager, and etcd.
  </div>
</div>

<!-- ══ PANEL 2: Pod Anatomy ══ -->
<div class="card">
  <h3>&#9313; Pod Anatomy</h3>
  <canvas id="cvPod" width="420" height="280"></canvas>
  <div class="btn-row" style="margin-top:10px;">
    <button id="btnPodSingle" class="active" onclick="setPodView('single')">Single Container</button>
    <button id="btnPodSidecar" onclick="setPodView('sidecar')">Sidecar Pattern</button>
    <button id="btnPodInit"   onclick="setPodView('init')">Init Container</button>
    <button id="btnPodNet"    onclick="setPodView('net')">Shared Network</button>
  </div>
  <div class="info-box" id="podInfo">
    A <span class="hl">Pod</span> is the smallest deployable unit in Kubernetes. It wraps one or more containers that share a network namespace, IP address, and volumes.
  </div>
</div>

<!-- ══ PANEL 3: Deployments & ReplicaSets ══ -->
<div class="card">
  <h3>&#9314; Deployments &amp; Rolling Updates</h3>
  <canvas id="cvDeploy" width="420" height="280"></canvas>
  <div class="row">
    <label>Replicas</label>
    <input type="range" id="replicaSlider" min="1" max="6" step="1" value="3" oninput="setReplicas(this.value)">
    <span class="val" id="replicaVal">3</span>
  </div>
  <div class="btn-row">
    <button id="btnRolling" onclick="startRollingUpdate()">Rolling Update</button>
    <button id="btnRollback" onclick="startRollback()">Rollback</button>
    <button id="btnRecreate" onclick="startRecreate()">Recreate</button>
  </div>
  <div class="info-box" id="deployInfo">
    A <span class="hl">Deployment</span> manages a <span class="hl2">ReplicaSet</span> which ensures the desired number of Pod replicas are always running. Adjust replicas above or trigger a <span class="hlg">rolling update</span>.
  </div>
</div>

<!-- ══ PANEL 4: Services & Networking ══ -->
<div class="card">
  <h3>&#9315; Services &amp; Networking</h3>
  <canvas id="cvSvc" width="420" height="280"></canvas>
  <div class="btn-row" style="margin-top:10px;">
    <button id="btnClusterIP"   class="active" onclick="setSvcType('clusterip')">ClusterIP</button>
    <button id="btnNodePort"    onclick="setSvcType('nodeport')">NodePort</button>
    <button id="btnLB"          onclick="setSvcType('lb')">LoadBalancer</button>
    <button id="btnIngress"     onclick="setSvcType('ingress')">Ingress</button>
  </div>
  <div class="info-box" id="svcInfo">
    <span class="hl">ClusterIP</span> (default) creates a stable virtual IP only reachable inside the cluster. Kube-proxy routes traffic from that VIP to the actual Pod IPs using iptables/ipvs.
  </div>
</div>

<!-- ══ PANEL 5: Scheduler & Resource Management ══ -->
<div class="card">
  <h3>&#9316; Scheduler &amp; Resource Requests</h3>
  <canvas id="cvSched" width="420" height="270"></canvas>
  <div class="row">
    <label>CPU request</label>
    <input type="range" id="cpuReq" min="100" max="1000" step="100" value="300" oninput="updateSched()">
    <span class="val" id="cpuReqVal">300m</span>
  </div>
  <div class="row">
    <label>Mem request</label>
    <input type="range" id="memReq" min="64" max="512" step="64" value="128" oninput="updateSched()">
    <span class="val" id="memReqVal">128Mi</span>
  </div>
  <div class="info-box" id="schedInfo">
    The <span class="hl">Scheduler</span> watches for unscheduled Pods and finds the best Node. It filters nodes that don't meet <span class="hl2">resource requests</span>, then scores the remainder by available capacity.
  </div>
</div>

<!-- ══ PANEL 6: Quiz ══ -->
<div class="card full">
  <h3>&#9317; Quiz &#8212; Test Your Kubernetes Knowledge</h3>
  <div id="quiz-root"></div>
</div>

</div>

<!-- ═══════════════════════ SCRIPTS ═══════════════════════ -->
<script>
const K = {
  bg:'#08080f', bg2:'#111118', bg3:'#0d0d18', border:'#1e1e2e',
  blue:'#326ce5', teal:'#4ecdc4', amber:'#fbbf24', green:'#4ade80',
  red:'#f87171', purple:'#c084fc', gray:'#71717a', light:'#94a3b8',
  text:'#e4e4e7', pink:'#f472b6'
};

function rr(ctx, x, y, w, h, r, fill, stroke, sw=1) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r); ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h); ctx.arcTo(x,y+h,x,y+h-r,r);
  ctx.lineTo(x,y+r); ctx.arcTo(x,y,x+r,y,r);
  ctx.closePath();
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=sw;ctx.stroke();}
}

function txt(ctx,t,x,y,color=K.light,size=10,align='center',bold=false){
  ctx.fillStyle=color;
  ctx.font=`${bold?'700':'400'} ${size}px 'JetBrains Mono',monospace`;
  ctx.textAlign=align; ctx.textBaseline='middle';
  ctx.fillText(t,x,y);
}

function arr(ctx,x1,y1,x2,y2,color=K.border,sw=1.5){
  const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy);
  if(len<1)return;
  const ux=dx/len,uy=dy/len;
  ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
  ctx.strokeStyle=color;ctx.lineWidth=sw;ctx.stroke();
  const hx=x2-ux*8,hy=y2-uy*8;
  ctx.beginPath();ctx.moveTo(x2,y2);
  ctx.lineTo(hx-uy*4,hy+ux*4);ctx.lineTo(hx+uy*4,hy-ux*4);
  ctx.closePath();ctx.fillStyle=color;ctx.fill();
}

function animDot(ctx,x1,y1,x2,y2,color,t,speed=0.4){
  const frac=((t*speed*0.001)%1+1)%1;
  const px=x1+frac*(x2-x1),py=y1+frac*(y2-y1);
  ctx.beginPath();ctx.arc(px,py,3.5,0,Math.PI*2);
  ctx.fillStyle=color;ctx.fill();
}

// ──────────────────────────────────────────────
// PANEL 1 — Cluster Architecture
// ──────────────────────────────────────────────
let clusterFocus='cp';
const clusterInfos={
  cp:'The <span class="hl">Control Plane</span> is the brain of Kubernetes — it makes all scheduling and orchestration decisions. It runs the API Server, Scheduler, Controller Manager, and etcd.',
  node:'A <span class="hl">Worker Node</span> runs the actual application Pods. Each node runs <span class="hl2">kubelet</span> (communicates with API server), <span class="hl3">kube-proxy</span> (handles networking), and a container runtime (containerd).',
  flow:'<span class="hl">kubectl apply</span> → API Server validates &amp; stores in <span class="hl2">etcd</span> → Scheduler assigns the Pod to a Node → <span class="hlg">kubelet</span> on that Node pulls the image and starts the container.',
  etcd:'<span class="hl">etcd</span> is a distributed key-value store — the single source of truth for the entire cluster state. All API Server writes go here. If etcd is lost, the cluster loses its memory.',
  all:'A <span class="hl">Kubernetes Cluster</span> = 1+ Control Plane nodes + N Worker Nodes. The control plane never runs user workloads (by default). Nodes are added via <span class="hl2">kubeadm join</span> or managed by your cloud provider.',
};

function setClusterFocus(f){
  clusterFocus=f;
  document.getElementById('clusterInfo').innerHTML=clusterInfos[f];
  ['CP','Node','Flow','Etcd','All'].forEach(k=>document.getElementById('btn'+k).classList.remove('active'));
  document.getElementById('btn'+{cp:'CP',node:'Node',flow:'Flow',etcd:'Etcd',all:'All'}[f]).classList.add('active');
}

let clusterT=0;
function drawCluster(ts){
  clusterT=ts;
  const cv=document.getElementById('cvCluster');
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  rr(ctx,0,0,cv.width,cv.height,8,K.bg3,K.border,1);

  const f=clusterFocus;
  const hi=(group)=>{
    if(f==='all'||f==='flow') return 1;
    const map={cp:['cp'],node:['node'],etcd:['etcd'],flow:['cp','node']};
    return (map[f]||[]).includes(group)?1:0.2;
  };

  // ── Control Plane box ──
  ctx.globalAlpha=hi('cp');
  rr(ctx,10,10,340,265,10,K.blue+'0a',K.blue+'55',1);
  txt(ctx,'CONTROL PLANE',185,24,K.blue,9,'center',true);

  // API Server
  rr(ctx,24,38,140,52,7,K.blue+'22',K.blue,1.5);
  txt(ctx,'API Server',94,58,K.blue,10,'center',true);
  txt(ctx,'REST · Auth · Validation',94,72,K.light+'88',8,'center');

  // etcd
  ctx.globalAlpha=(f==='etcd'||f==='all'||f==='flow')?1:0.2;
  rr(ctx,176,38,160,52,7,K.amber+'22',K.amber+(f==='etcd'?'':f==='all'||f==='flow'?'':'44'),f==='etcd'?2:1);
  txt(ctx,'etcd',256,58,K.amber,10,'center',true);
  txt(ctx,'distributed key-value store',256,72,K.amber+'88',7.5,'center');
  ctx.globalAlpha=hi('cp');

  // Scheduler
  rr(ctx,24,102,140,48,7,K.green+'18',K.green+'66',1);
  txt(ctx,'Scheduler',94,122,K.green,10,'center',true);
  txt(ctx,'watches · filters · scores',94,136,K.light+'88',8,'center');

  // Controller Manager
  rr(ctx,176,102,160,48,7,K.purple+'18',K.purple+'66',1);
  txt(ctx,'Controller Manager',256,118,K.purple,9,'center',true);
  txt(ctx,'Deployment · ReplicaSet',256,132,K.purple+'88',8,'center');
  txt(ctx,'Node · Job controllers',256,144,K.purple+'88',8,'center');

  // Cloud Controller (optional)
  rr(ctx,24,162,312,40,7,K.teal+'10',K.teal+'33',1);
  txt(ctx,'Cloud Controller Manager (optional — managed K8s)',180,182,K.teal+'88',8,'center');

  ctx.globalAlpha=1;

  // ── Worker Node boxes ──
  const nodes=[
    {x:370,label:'Node 1',pods:['nginx','redis'],color:K.teal},
    {x:560,label:'Node 2',pods:['api','worker'],color:K.teal},
    {x:750,label:'Node 3',pods:['db'],color:K.teal},
  ];

  nodes.forEach(nd=>{
    ctx.globalAlpha=hi('node');
    rr(ctx,nd.x,10,80+20,265,10,K.teal+'08',K.teal+(f==='node'?'':f==='all'||f==='flow'?'':'33'),f==='node'||f==='all'||f==='flow'?1.5:1);
    txt(ctx,nd.label,nd.x+50,24,K.teal,9,'center',true);

    // Node components
    rr(ctx,nd.x+8,36,84,30,5,K.green+'15',K.green+'44',1);
    txt(ctx,'kubelet',nd.x+50,51,K.green,8,'center',true);
    rr(ctx,nd.x+8,72,84,24,5,K.amber+'15',K.amber+'44',1);
    txt(ctx,'kube-proxy',nd.x+50,84,K.amber,8,'center',true);
    rr(ctx,nd.x+8,102,84,20,5,K.gray+'20',K.gray+'44',1);
    txt(ctx,'containerd',nd.x+50,112,K.gray,8,'center');

    // Pods
    nd.pods.forEach((p,pi)=>{
      const py=130+pi*56;
      rr(ctx,nd.x+10,py,80,48,6,K.blue+'18',K.blue+(f==='node'||f==='all'||f==='flow'?'':K.blue),1);
      txt(ctx,'Pod',nd.x+50,py+12,K.blue,8,'center',true);
      rr(ctx,nd.x+16,py+18,68,22,4,K.blue+'22',K.blue+'55',1);
      txt(ctx,p,nd.x+50,py+29,K.teal,8,'center');
    });
    ctx.globalAlpha=1;
  });

  // ── Arrows ──
  // API Server → etcd
  ctx.globalAlpha=(f==='etcd'||f==='all'||f==='flow')?1:0.1;
  arr(ctx,164,62,174,62,K.amber,1.5);
  if(f==='flow'||f==='all') animDot(ctx,164,62,174,62,K.amber,ts,0.5);
  ctx.globalAlpha=1;

  // API Server → Scheduler
  ctx.globalAlpha=(f==='all'||f==='flow'||f==='cp')?0.6:0.1;
  arr(ctx,94,90,94,100,K.green,1);
  ctx.globalAlpha=1;

  // API Server → ControllerMgr
  ctx.globalAlpha=(f==='all'||f==='flow'||f==='cp')?0.6:0.1;
  arr(ctx,164,68,174,120,K.purple,1);
  ctx.globalAlpha=1;

  // API Server → kubelet (each node)
  nodes.forEach(nd=>{
    ctx.globalAlpha=(f==='flow'||f==='all'||f==='node')?1:0.1;
    const midX=(352+nd.x)/2;
    ctx.beginPath();ctx.moveTo(352,64);ctx.lineTo(midX,64);ctx.lineTo(midX,270);ctx.lineTo(nd.x+50,270);ctx.lineTo(nd.x+50,260);
    ctx.strokeStyle=K.blue+(f==='flow'||f==='all'?'':f==='node'?'':'55');ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
    if(f==='flow'||f==='all') animDot(ctx,352,64,nd.x+50,64,K.blue,ts+nd.x*2,0.25);
    ctx.globalAlpha=1;
  });

  // kubectl label
  ctx.globalAlpha=(f==='flow'||f==='all')?1:0.3;
  rr(ctx,24,220,100,30,6,K.blue+'15',K.blue+'44',1);
  txt(ctx,'kubectl',74,235,K.blue,10,'center',true);
  if(f==='flow'||f==='all') arr(ctx,124,235,154,64,K.blue,1.5);
  if(f==='flow'||f==='all') animDot(ctx,124,235,154,64,K.blue,ts,0.6);
  ctx.globalAlpha=1;

  requestAnimationFrame(drawCluster);
}
requestAnimationFrame(drawCluster);
setClusterFocus('cp');


// ──────────────────────────────────────────────
// PANEL 2 — Pod Anatomy
// ──────────────────────────────────────────────
let podView='single';
const podInfos={
  single:'A <span class="hl">Pod</span> wraps one container in the simplest case. It gets a unique cluster IP, can mount volumes, and is scheduled as one unit. Pods are <span class="hlr">ephemeral</span> — never rely on a fixed IP.',
  sidecar:'The <span class="hl">Sidecar pattern</span> runs a helper container alongside the main app in the same Pod. Common uses: log shipping (<span class="hl2">Fluentd</span>), service mesh proxy (<span class="hl2">Envoy</span>), secret syncing.',
  init:'<span class="hl">Init containers</span> run to completion <span class="hlg">before</span> the main container starts. Used for DB migrations, config fetching, or waiting for a dependency to be ready.',
  net:'All containers in a Pod share the same <span class="hl">network namespace</span> — same IP, same port space. They talk via <span class="hl2">localhost</span>. Volumes declared at Pod level are mounted into each container individually.',
};

function setPodView(v){
  podView=v;
  document.getElementById('podInfo').innerHTML=podInfos[v];
  ['PodSingle','PodSidecar','PodInit','PodNet'].forEach(k=>document.getElementById('btn'+k).classList.remove('active'));
  document.getElementById('btn'+{single:'PodSingle',sidecar:'PodSidecar',init:'PodInit',net:'PodNet'}[v]).classList.add('active');
  drawPod();
}

let podT=0;
function drawPod(ts){
  podT=ts||podT;
  const cv=document.getElementById('cvPod');
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  rr(ctx,0,0,cv.width,cv.height,8,K.bg3,K.border,1);

  const v=podView;

  // Pod outer shell
  rr(ctx,20,30,cv.width-40,cv.height-50,12,K.blue+'0d',K.blue+'55',1.5);
  txt(ctx,'POD  ·  ip: 10.1.0.42',cv.width/2,48,K.blue,9,'center',true);

  if(v==='single'){
    // single main container
    rr(ctx,60,70,300,130,8,K.teal+'18',K.teal,2);
    txt(ctx,'Main Container',210,118,K.teal,12,'center',true);
    txt(ctx,'nginx:latest',210,138,K.teal+'88',10,'center');
    // volume
    rr(ctx,60,215,130,40,6,K.amber+'18',K.amber+'66',1);
    txt(ctx,'Volume: /data',125,235,K.amber,9,'center',true);
    rr(ctx,250,215,110,40,6,K.gray+'18',K.gray+'44',1);
    txt(ctx,'ConfigMap',305,235,K.gray,9,'center');
    // arrows
    arr(ctx,210,200,125,214,K.amber+'88',1);
    arr(ctx,210,200,305,214,K.gray+'88',1);
  } else if(v==='sidecar'){
    rr(ctx,38,70,158,130,8,K.teal+'18',K.teal,2);
    txt(ctx,'Main',117,118,K.teal,11,'center',true);
    txt(ctx,'api-server',117,134,K.teal+'88',9,'center');
    rr(ctx,224,70,158,130,8,K.purple+'18',K.purple,1.5);
    txt(ctx,'Sidecar',303,118,K.purple,11,'center',true);
    txt(ctx,'envoy proxy',303,134,K.purple+'88',9,'center');
    // shared volume between them
    rr(ctx,130,215,160,40,6,K.amber+'18',K.amber+'66',1);
    txt(ctx,'Shared Volume',210,235,K.amber,9,'center',true);
    arr(ctx,117,200,170,214,K.amber+'88',1);
    arr(ctx,303,200,250,214,K.amber+'88',1);
    // localhost arrow
    ctx.beginPath();ctx.moveTo(196,135);ctx.lineTo(224,135);
    ctx.strokeStyle=K.green;ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.stroke();ctx.setLineDash([]);
    txt(ctx,'localhost',210,125,K.green,8,'center');
    const frac=((podT*0.0005)%1+1)%1;
    const px=196+frac*(224-196);
    ctx.beginPath();ctx.arc(px,135,3,0,Math.PI*2);ctx.fillStyle=K.green;ctx.fill();
  } else if(v==='init'){
    // init containers run first
    rr(ctx,38,70,100,55,7,K.red+'18',K.red+'66',1.5);
    txt(ctx,'Init 1',88,90,K.red,10,'center',true);
    txt(ctx,'wait-for-db',88,104,K.red+'88',8,'center');
    rr(ctx,152,70,100,55,7,K.red+'18',K.red+'66',1.5);
    txt(ctx,'Init 2',202,90,K.red,10,'center',true);
    txt(ctx,'run-migration',202,104,K.red+'88',8,'center');
    // arrow then main
    const frac=((podT*0.0003)%1+1)%1;
    const stage=Math.floor(frac*3);
    arr(ctx,138,97,150,97,K.red,1.5);
    arr(ctx,252,97,278,130,K.green,1.5);
    txt(ctx,'completed ✓',195,155,stage>0?K.green:K.gray,8,'center');
    rr(ctx,278,110,120,80,8,K.green+'18',K.green,stage>1?2:1);
    txt(ctx,'Main Container',338,148,stage>1?K.green:K.gray,10,'center',true);
    txt(ctx,'app:v2',338,164,stage>1?K.green+'88':K.gray,9,'center');
    txt(ctx,stage===0?'⏳ waiting for init':'✓ init done → app starts',cv.width/2,230,stage===0?K.amber:K.green,9,'center');
  } else {
    // net view — shared namespace
    rr(ctx,30,65,175,130,8,K.teal+'18',K.teal,1.5);
    txt(ctx,'Container A',117,103,K.teal,10,'center',true);
    txt(ctx,'port 8080',117,119,K.teal+'88',9,'center');
    rr(ctx,215,65,175,130,8,K.purple+'18',K.purple,1.5);
    txt(ctx,'Container B',302,103,K.purple,10,'center',true);
    txt(ctx,'port 9090',302,119,K.purple+'88',9,'center');
    // shared net namespace
    rr(ctx,30,208,360,45,8,K.blue+'10',K.blue+'55',1);
    txt(ctx,'Shared Network Namespace  ·  Pod IP: 10.1.0.42',210,231,K.blue,8.5,'center');
    // localhost connection
    ctx.beginPath();ctx.moveTo(205,130);ctx.lineTo(215,130);
    ctx.strokeStyle=K.green;ctx.lineWidth=2;ctx.setLineDash([5,3]);ctx.stroke();ctx.setLineDash([]);
    txt(ctx,'localhost',210,122,K.green,8,'center');
    const frac2=((podT*0.0006)%1+1)%1;
    ctx.beginPath();ctx.arc(205+frac2*10,130,3,0,Math.PI*2);ctx.fillStyle=K.green;ctx.fill();
    // arrows to net namespace
    arr(ctx,117,195,117,206,K.blue+'66',1);
    arr(ctx,302,195,302,206,K.blue+'66',1);
  }

  requestAnimationFrame(drawPod);
}
requestAnimationFrame(ts=>{podT=ts;drawPod(ts);});
setPodView('single');


// ──────────────────────────────────────────────
// PANEL 3 — Deployments & ReplicaSets
// ──────────────────────────────────────────────
let replicaCount=3;
let podStates=[]; // 'running','updating','ready','old'
let deployInfo='A <span class="hl">Deployment</span> manages a <span class="hl2">ReplicaSet</span> which ensures the desired number of Pod replicas are always running. Adjust replicas above or trigger a <span class="hlg">rolling update</span>.';
let deployVersion=1; // 1 = v1, 2 = v2
let deployAnim=null; // null | 'rolling' | 'rollback' | 'recreate'
let deployAnimT=0;

function setReplicas(n){
  replicaCount=parseInt(n);
  document.getElementById('replicaVal').textContent=n;
  podStates=Array(replicaCount).fill('running');
  document.getElementById('deployInfo').innerHTML=`<span class="hl">ReplicaSet</span> now targeting <span class="hlg">${n} replicas</span>. The controller continuously reconciles actual vs desired count — if a Pod crashes, it spins up a replacement instantly.`;
  drawDeploy();
}

function startRollingUpdate(){
  deployAnim='rolling'; deployAnimT=0; deployVersion=2;
  podStates=Array(replicaCount).fill('old');
  document.getElementById('deployInfo').innerHTML='<span class="hlg">Rolling update</span> in progress — Kubernetes replaces Pods one-by-one. Old version (v1) Pods stay running until new version (v2) passes readiness checks. Zero downtime.';
}
function startRollback(){
  deployAnim='rollback'; deployAnimT=0; deployVersion=1;
  podStates=Array(replicaCount).fill('running');
  document.getElementById('deployInfo').innerHTML='<span class="hlr">Rollback</span> triggered via <span class="hl2">kubectl rollout undo</span>. Kubernetes rolls back to the previous ReplicaSet. Deployment history stores the last 10 revisions by default.';
}
function startRecreate(){
  deployAnim='recreate'; deployAnimT=0;
  podStates=Array(replicaCount).fill('old');
  document.getElementById('deployInfo').innerHTML='<span class="hl3">Recreate strategy</span>: terminates ALL old Pods first, then creates new ones. Causes downtime — only suitable for dev or stateful apps that cannot run two versions simultaneously.';
}

function drawDeploy(ts){
  const cv=document.getElementById('cvDeploy');
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  rr(ctx,0,0,cv.width,cv.height,8,K.bg3,K.border,1);

  if(deployAnim&&ts) deployAnimT=ts;

  const n=replicaCount;
  const podW=54, podH=64, gap=8;
  const totalW=n*(podW+gap)-gap;
  const ox=(cv.width-totalW)/2;
  const oy=100;

  // Deployment label
  rr(ctx,20,14,200,36,7,K.blue+'18',K.blue+'55',1);
  txt(ctx,'Deployment',100+10,32,K.blue,10,'center',true);
  txt(ctx,'nginx  rev:'+(deployAnim==='rollback'?'1':deployVersion),100+10,46,K.blue+'88',8,'center');

  // ReplicaSet label
  rr(ctx,230,14,170,36,7,K.purple+'18',K.purple+'55',1);
  txt(ctx,'ReplicaSet',315,32,K.purple,10,'center',true);
  txt(ctx,'desired: '+n+'  actual: '+n,315,46,K.purple+'88',8,'center');
  arr(ctx,220,32,228,32,K.blue+'88',1);

  // animate rolling update
  let states=[...podStates];
  if(deployAnim==='rolling'&&deployAnimT){
    const progress=(deployAnimT%6000)/6000;
    const updated=Math.floor(progress*n);
    for(let i=0;i<n;i++) states[i]=i<updated?'ready':'old';
    if(updated>=n){deployAnim=null;podStates=Array(n).fill('running');}
  } else if(deployAnim==='recreate'&&deployAnimT){
    const progress=(deployAnimT%5000)/5000;
    if(progress<0.5) states=Array(n).fill('terminated');
    else {
      const pct=(progress-0.5)*2;
      const newCount=Math.floor(pct*n);
      states=Array(n).fill('terminated');
      for(let i=0;i<newCount;i++) states[i]='ready';
      if(pct>=1){deployAnim=null;podStates=Array(n).fill('running');}
    }
  } else if(deployAnim==='rollback'&&deployAnimT){
    const progress=(deployAnimT%4000)/4000;
    const updated=Math.floor(progress*n);
    for(let i=0;i<n;i++) states[i]=i<updated?'running':'ready';
    if(updated>=n){deployAnim=null;podStates=Array(n).fill('running');}
  }

  const podColors={
    running:{fill:K.green+'22',stroke:K.green,label:K.green,text:'v1 ✓'},
    old:    {fill:K.amber+'18',stroke:K.amber,label:K.amber,text:'v1 old'},
    ready:  {fill:K.blue+'22', stroke:K.blue, label:K.blue, text:'v2 ✓'},
    terminated:{fill:K.red+'18',stroke:K.red+'55',label:K.red+'88',text:'killed'},
  };

  for(let i=0;i<n;i++){
    const x=ox+i*(podW+gap), y=oy;
    const s=states[i]||'running';
    const c=podColors[s]||podColors.running;
    rr(ctx,x,y,podW,podH,7,c.fill,c.stroke,1.5);
    txt(ctx,'Pod '+(i+1),x+podW/2,y+18,c.label,8,'center',true);
    rr(ctx,x+6,y+28,podW-12,20,4,c.stroke+'22',c.stroke+'44',1);
    txt(ctx,c.text,x+podW/2,y+38,c.stroke,7.5,'center');
    // pulse on updating
    if(s==='ready'&&deployAnim){
      ctx.beginPath();ctx.arc(x+podW/2,y+58,4,0,Math.PI*2);
      ctx.fillStyle=K.green;ctx.fill();
    }
  }

  // Service VIP
  rr(ctx,cv.width/2-70,200,140,36,7,K.teal+'18',K.teal+'66',1);
  txt(ctx,'Service (ClusterIP)',cv.width/2,218,K.teal,9,'center',true);
  txt(ctx,'stable VIP load-balances pods',cv.width/2,231,K.teal+'77',7.5,'center');
  // arrow from service to pods
  if(n>0){
    arr(ctx,cv.width/2,198,ox+((n-1)/2)*(podW+gap)+podW/2,oy+podH,K.teal+'66',1);
  }

  // HPA
  rr(ctx,20,200,110,36,7,K.pink+'15',K.pink+'44',1);
  txt(ctx,'HPA',75,216,K.pink,9,'center',true);
  txt(ctx,'auto-scales replicas',75,230,K.pink+'77',7.5,'center');
  arr(ctx,130,218,180+20,218,K.pink+'66',1);

  requestAnimationFrame(drawDeploy);
}
requestAnimationFrame(drawDeploy);
setReplicas(3);


// ──────────────────────────────────────────────
// PANEL 4 — Services & Networking
// ──────────────────────────────────────────────
let svcType='clusterip';
const svcInfos={
  clusterip:'<span class="hl">ClusterIP</span> (default) creates a stable virtual IP only reachable inside the cluster. Kube-proxy routes traffic from that VIP to the actual Pod IPs using iptables/ipvs.',
  nodeport:'<span class="hl">NodePort</span> exposes a port (30000–32767) on every Node. External traffic hits <span class="hl2">NodeIP:NodePort</span> → kube-proxy forwards to the Service VIP → then to a Pod.',
  lb:'<span class="hl">LoadBalancer</span> provisions a cloud load balancer (AWS ALB, GCP LB) that routes external traffic to NodePorts. The most common way to expose production services.',
  ingress:'<span class="hl">Ingress</span> is an L7 (HTTP/HTTPS) router. One Ingress controller (nginx, Traefik) handles many services via hostname/path rules — far cheaper than one LoadBalancer per service.',
};

function setSvcType(t){
  svcType=t;
  document.getElementById('svcInfo').innerHTML=svcInfos[t];
  ['ClusterIP','NodePort','LB','Ingress'].forEach(k=>document.getElementById('btn'+k).classList.remove('active'));
  document.getElementById('btn'+{clusterip:'ClusterIP',nodeport:'NodePort',lb:'LB',ingress:'Ingress'}[t]).classList.add('active');
}

let svcT=0;
function drawSvc(ts){
  svcT=ts;
  const cv=document.getElementById('cvSvc');
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  rr(ctx,0,0,cv.width,cv.height,8,K.bg3,K.border,1);

  const t=svcType;

  // Draw 3 pods on the right
  const pods=[{y:60,name:'pod-1',ip:'10.1.0.10'},{y:120,name:'pod-2',ip:'10.1.0.11'},{y:180,name:'pod-3',ip:'10.1.0.12'}];
  pods.forEach(p=>{
    rr(ctx,290,p.y,120,50,7,K.teal+'18',K.teal,1.5);
    txt(ctx,p.name,350,p.y+16,K.teal,9,'center',true);
    txt(ctx,p.ip,350,p.y+30,K.teal+'88',8,'center');
    rr(ctx,300,p.y+36,100,10,3,K.teal+'15',K.teal+'33',1);
    txt(ctx,'container',350,p.y+41,K.teal+'66',7,'center');
  });

  // Service box
  rr(ctx,150,110,110,60,8,K.blue+'18',K.blue,2);
  txt(ctx,'Service',205,133,K.blue,10,'center',true);
  txt(ctx,t==='clusterip'?'ClusterIP':t==='nodeport'?'NodePort':t==='lb'?'LoadBalancer':'ClusterIP',205,148,K.blue+'88',8,'center');
  txt(ctx,'10.96.0.1',205,160,K.blue+'66',7.5,'center');

  // arrows service → pods
  pods.forEach(p=>{
    arr(ctx,262,140,288,p.y+25,K.teal+'77',1);
    animDot(ctx,262,140,288,p.y+25,K.teal,ts,0.3);
  });

  // External traffic sources
  if(t==='clusterip'){
    rr(ctx,14,90,100,50,7,K.purple+'18',K.purple+'66',1);
    txt(ctx,'Internal',64,108,K.purple,9,'center',true);
    txt(ctx,'Pod/Service',64,122,K.purple+'88',8,'center');
    arr(ctx,114,115,148,130,K.purple,1.5);
    animDot(ctx,114,115,148,130,K.purple,ts,0.5);
    txt(ctx,'cluster-internal only',215,248,K.gray,8,'center');
    rr(ctx,60,230,290,28,6,K.red+'10',K.red+'33',1);
    txt(ctx,'✗  not reachable from outside the cluster',205,244,K.red+'88',8,'center');
  } else if(t==='nodeport'){
    rr(ctx,14,60,100,42,7,K.green+'18',K.green+'66',1.5);
    txt(ctx,'External',64,75,K.green,9,'center',true);
    txt(ctx,'Client',64,89,K.green+'88',8,'center');
    arr(ctx,114,80,148,130,K.green,1.5);
    animDot(ctx,114,80,148,130,K.green,ts,0.5);
    rr(ctx,14,118,100,42,7,K.amber+'18',K.amber+'66',1);
    txt(ctx,'Node',64,133,K.amber,9,'center',true);
    txt(ctx,':31080',64,147,K.amber+'88',8.5,'center');
    arr(ctx,114,139,148,139,K.amber,1);
    txt(ctx,'NodeIP:31080  →  Service VIP  →  Pod',215,250,K.amber+'88',8,'center');
  } else if(t==='lb'){
    rr(ctx,10,50,110,46,7,K.green+'18',K.green+'66',1.5);
    txt(ctx,'Internet',65,68,K.green,9,'center',true);
    txt(ctx,'external traffic',65,82,K.green+'88',8,'center');
    rr(ctx,10,110,110,46,7,K.pink+'18',K.pink+'66',1.5);
    txt(ctx,'Cloud LB',65,128,K.pink,9,'center',true);
    txt(ctx,'AWS ALB / GCP',65,142,K.pink+'88',8,'center');
    arr(ctx,65,96,65,108,K.green,1);
    animDot(ctx,65,96,65,108,K.green,ts,0.8);
    arr(ctx,120,133,148,133,K.pink,1.5);
    animDot(ctx,120,133,148,133,K.pink,ts,0.5);
    txt(ctx,'cloud LB  →  NodePort  →  Service  →  Pod',215,250,K.pink+'88',8,'center');
  } else {
    // ingress
    rr(ctx,10,40,110,36,7,K.green+'18',K.green+'66',1.5);
    txt(ctx,'Client',65,58,K.green,9,'center',true);
    rr(ctx,10,90,110,42,7,K.purple+'18',K.purple,1.5);
    txt(ctx,'Ingress',65,107,K.purple,10,'center',true);
    txt(ctx,'nginx/Traefik',65,121,K.purple+'88',8,'center');
    arr(ctx,65,76,65,88,K.green,1);
    animDot(ctx,65,76,65,88,K.green,ts,0.8);
    arr(ctx,120,111,148,130,K.purple,1.5);
    animDot(ctx,120,111,148,130,K.purple,ts,0.5);
    txt(ctx,'Ingress rules: /api → svc-a,  /web → svc-b',215,247,K.purple+'88',8,'center');
    // path rules
    txt(ctx,'/api → this svc',65,150,K.purple+'88',8,'center');
    txt(ctx,'/web → other svc',65,165,K.gray,8,'center');
  }

  requestAnimationFrame(drawSvc);
}
requestAnimationFrame(drawSvc);
setSvcType('clusterip');


// ──────────────────────────────────────────────
// PANEL 5 — Scheduler & Resource Requests
// ──────────────────────────────────────────────
let schedT=0;

function updateSched(){
  const cpu=parseInt(document.getElementById('cpuReq').value);
  const mem=parseInt(document.getElementById('memReq').value);
  document.getElementById('cpuReqVal').textContent=cpu+'m';
  document.getElementById('memReqVal').textContent=mem+'Mi';
  drawSched();
}

function drawSched(ts){
  schedT=ts||schedT;
  const cv=document.getElementById('cvSched');
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  rr(ctx,0,0,cv.width,cv.height,8,K.bg3,K.border,1);

  const cpu=parseInt(document.getElementById('cpuReq').value);
  const mem=parseInt(document.getElementById('memReq').value);

  // Nodes with capacity
  const nodes=[
    {name:'Node 1',cpuCap:1000,memCap:512,cpuUsed:700,memUsed:320,x:20},
    {name:'Node 2',cpuCap:1000,memCap:512,cpuUsed:200,memUsed:128,x:160},
    {name:'Node 3',cpuCap:1000,memCap:512,cpuUsed:600,memUsed:400,x:300},
  ];

  const canFit=(n)=>(n.cpuUsed+cpu<=n.cpuCap)&&(n.memUsed+mem<=n.memCap);
  const scores=nodes.map(n=>canFit(n)?Math.round(((n.cpuCap-n.cpuUsed-cpu)/n.cpuCap*50)+((n.memCap-n.memUsed-mem)/n.memCap*50)):0);
  const best=scores.indexOf(Math.max(...scores));

  nodes.forEach((n,i)=>{
    const fits=canFit(n);
    const isBest=fits&&i===best;
    const borderColor=isBest?K.green:fits?K.blue+'66':K.red+'44';
    const x=n.x, y=20;

    rr(ctx,x,y,130,220,8,K.bg2,borderColor,isBest?2:1);
    txt(ctx,n.name,x+65,y+14,isBest?K.green:fits?K.blue:K.gray,9,'center',true);

    if(isBest){
      txt(ctx,'✓ BEST FIT',x+65,y+26,K.green,7.5,'center',true);
    } else if(!fits){
      txt(ctx,'✗ INSUFFICIENT',x+65,y+26,K.red+'77',7.5,'center');
    } else {
      txt(ctx,'viable',x+65,y+26,K.blue+'77',7.5,'center');
    }

    // CPU bar
    txt(ctx,'CPU',x+14,y+42,K.light,7.5,'left');
    rr(ctx,x+8,y+50,114,12,3,K.bg3,K.border,1);
    const cpuFrac=Math.min(n.cpuUsed/n.cpuCap,1);
    rr(ctx,x+8,y+50,Math.round(114*cpuFrac),12,3,K.amber+'44',null);
    if(fits){
      const reqFrac=Math.min((n.cpuUsed+cpu)/n.cpuCap,1);
      rr(ctx,x+8+Math.round(114*cpuFrac),y+50,Math.round(114*(reqFrac-cpuFrac)),12,0,K.blue+'44',null);
    }
    txt(ctx,n.cpuUsed+'m used',x+65,y+57,K.amber+'99',6.5,'center');

    // Mem bar
    txt(ctx,'Mem',x+14,y+70,K.light,7.5,'left');
    rr(ctx,x+8,y+78,114,12,3,K.bg3,K.border,1);
    const memFrac=Math.min(n.memUsed/n.memCap,1);
    rr(ctx,x+8,y+78,Math.round(114*memFrac),12,3,K.purple+'44',null);
    if(fits){
      const rmf=Math.min((n.memUsed+mem)/n.memCap,1);
      rr(ctx,x+8+Math.round(114*memFrac),y+78,Math.round(114*(rmf-memFrac)),12,0,K.blue+'44',null);
    }
    txt(ctx,n.memUsed+'Mi used',x+65,y+85,K.purple+'99',6.5,'center');

    // Score
    txt(ctx,'Score: '+scores[i],x+65,y+104,isBest?K.green:fits?K.blue:K.red+'66',8,'center',isBest);

    // Pods already running
    const runningPods=Math.floor(n.cpuUsed/200);
    txt(ctx,'Pods:',x+14,y+118,K.gray,7.5,'left');
    for(let p=0;p<runningPods&&p<4;p++){
      rr(ctx,x+46+p*18,y+110,14,14,3,K.teal+'22',K.teal+'55',1);
      txt(ctx,'P',x+53+p*18,y+117,K.teal,7,'center');
    }

    // New pod if fits
    if(fits){
      rr(ctx,x+8,y+132,114,50,6,K.blue+'15',K.blue+(isBest?'':'+55'),isBest?2:1);
      txt(ctx,'New Pod',x+65,y+150,isBest?K.green:K.blue,9,'center',true);
      txt(ctx,'cpu:'+cpu+'m mem:'+mem+'Mi',x+65,y+165,K.blue+'88',7,'center');
    } else {
      rr(ctx,x+8,y+132,114,50,6,K.red+'10',K.red+'33',1);
      txt(ctx,'Cannot schedule',x+65,y+150,K.red+'88',8,'center');
      txt(ctx,'Insufficient resources',x+65,y+165,K.red+'55',7,'center');
    }
  });

  // Scheduler label
  rr(ctx,20,248,380,20,5,K.blue+'10',K.blue+'33',1);
  txt(ctx,'Scheduler: filter → score → bind  →  kubelet starts Pod on winning Node',210,258,K.blue+'88',8,'center');

  requestAnimationFrame(drawSched);
}
requestAnimationFrame(drawSched);
</script>


<!-- ════════════════ REACT QUIZ ════════════════ -->
<script type="text/babel">
(() => {
  const { useState } = React;

  const questions = [
    {
      q: 'What is the role of etcd in a Kubernetes cluster?',
      opts: ['Runs application containers', 'Distributed store for all cluster state', 'Routes network traffic to Pods', 'Schedules Pods to Nodes'],
      ans: 1,
      fb: 'etcd is the single source of truth for the entire cluster. Every object (Pods, Services, ConfigMaps) is persisted there. Loss of etcd = loss of cluster memory.',
    },
    {
      q: 'Which resource ensures a specified number of Pod replicas are always running?',
      opts: ['DaemonSet', 'StatefulSet', 'ReplicaSet', 'Job'],
      ans: 2,
      fb: 'A ReplicaSet continuously reconciles desired vs. actual replica count. A Deployment wraps a ReplicaSet to add rolling update and rollback capabilities.',
    },
    {
      q: 'What is the smallest deployable unit in Kubernetes?',
      opts: ['Container', 'Pod', 'Node', 'Namespace'],
      ans: 1,
      fb: 'A Pod can hold one or more containers that share a network namespace and volumes. You never schedule bare containers — only Pods.',
    },
    {
      q: 'Which Service type exposes a workload via a cloud-provider load balancer?',
      opts: ['ClusterIP', 'NodePort', 'ExternalName', 'LoadBalancer'],
      ans: 3,
      fb: 'LoadBalancer provisions a cloud LB (AWS ALB, GCP LB) automatically. It is the standard way to expose production traffic, but costs one LB per service — use Ingress to share one LB.',
    },
    {
      q: 'What does the Kubernetes Scheduler do?',
      opts: ['Stores desired state in etcd', 'Watches for Pods with no Node assigned and picks the best Node', 'Restarts crashed containers', 'Manages DNS resolution inside the cluster'],
      ans: 1,
      fb: 'The Scheduler watches for unbound Pods, runs filter (removes nodes that cannot fit the Pod) then score (ranks viable nodes by available resources), and binds the Pod to the winner.',
    },
    {
      q: 'What agent runs on every Worker Node and communicates with the API Server?',
      opts: ['kube-proxy', 'kubelet', 'etcd', 'Controller Manager'],
      ans: 1,
      fb: 'kubelet is the node agent. It receives PodSpec from the API Server, ensures containers are running via the container runtime (containerd/cri-o), and reports Node/Pod status back.',
    },
  ];

  function Quiz() {
    const [answers, setAnswers] = useState({});
    const score = Object.values(answers).filter(Boolean).length;
    const done  = Object.keys(answers).length;

    function pick(qi, oi) {
      if (qi in answers) return;
      setAnswers(prev => ({ ...prev, [qi]: oi === questions[qi].ans }));
    }

    return (
      <div>
        <div className="quiz-grid">
          {questions.map((q, qi) => {
            const answered = qi in answers;
            const correct  = answers[qi];
            return (
              <div className="q-card" key={qi}>
                <div className="q-num">Question {qi + 1}</div>
                <div className="q-prompt">{q.q}</div>
                {q.opts.map((opt, oi) => {
                  let cls = 'q-opt';
                  if (answered) {
                    if (oi === q.ans) cls += ' correct';
                    else if (answers[qi] === false && oi !== q.ans) {
                      // find which was selected — we need to mark wrong selection
                    }
                  }
                  return (
                    <button key={oi} className={cls} onClick={() => pick(qi, oi)} disabled={answered}>
                      {opt}
                    </button>
                  );
                })}
                <div className={`q-feedback ${answered ? 'show' : ''} ${answers[qi] ? 'ok' : 'bad'}`}>
                  {answered ? (answers[qi] ? '✓ Correct — ' : '✗ Wrong — ') + q.fb : ''}
                </div>
              </div>
            );
          })}
        </div>
        <div className="score-bar">
          <span style={{color:'#52525b', fontSize:'0.78em', flex:1}}>SCORE</span>
          {questions.map((_, i) => (
            <div key={i} className={`pip ${i in answers ? (answers[i] ? 'ok' : 'bad') : ''}`}/>
          ))}
          <span style={{color: score === questions.length ? '#4ade80' : '#326ce5', fontWeight: 700}}>
            {done === 0 ? 'Answer to begin' : `${score} / ${questions.length}`}
          </span>
          {score === questions.length && done === questions.length &&
            <span style={{color:'#4ade80', marginLeft:8}}>⎈ K8s certified!</span>}
        </div>
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById('quiz-root')).render(<Quiz/>);
})();
</script>

</body>
</html>"""

KUBERNETES_VISUAL_HEIGHT = 2100