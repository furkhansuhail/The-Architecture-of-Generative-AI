DOCKER_VISUAL_HTML = r"""<!DOCTYPE html>
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

h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.15em; color: #2496ed; margin-bottom: 3px; }
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
  color: #2496ed;
  margin-bottom: 12px;
}

canvas { display: block; border-radius: 8px; }

.row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.row label { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #71717a; min-width: 90px; }
input[type=range] {
  flex: 1; height: 4px; border-radius: 2px;
  -webkit-appearance: none; appearance: none;
  background: #1e1e2e; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px;
  border-radius: 50%; background: #2496ed; cursor: pointer;
}
.val { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #2496ed; min-width: 46px; text-align: right; }

.info-box {
  background: #0d0d18; border: 1px solid #1e1e2e; border-radius: 8px;
  padding: 8px 12px; font-size: 0.78em; color: #94a3b8;
  line-height: 1.7; margin-top: 8px;
  font-family: 'JetBrains Mono', monospace;
}
.hl  { color: #2496ed; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }
.hlg { color: #4ade80; font-weight: 700; }
.hlr { color: #f87171; font-weight: 700; }

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
button.active { background: #2496ed; color: #08080f; border-color: #2496ed; }
button.active-green { background: #4ade80; color: #08080f; border-color: #4ade80; }
button.active-amber { background: #fbbf24; color: #08080f; border-color: #fbbf24; }
button.active-red   { background: #f87171; color: #08080f; border-color: #f87171; }

/* Dockerfile layer panel */
.df-line {
  display: flex; align-items: flex-start; gap: 8px;
  padding: 7px 10px; border-radius: 6px; cursor: pointer;
  margin-bottom: 4px; border: 1px solid transparent;
  transition: all 0.15s;
}
.df-line:hover { background: #1a1a28; border-color: #2d2d40; }
.df-line.selected { background: #0d1a2e; border-color: #2496ed44; }
.df-kw { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; font-weight: 800; min-width: 48px; }
.df-arg { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #94a3b8; flex: 1; word-break: break-all; }

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

<h2>&#x1F433; Docker &#8212; Containers for Everyone</h2>
<p class="subtitle">VMs vs. Containers &middot; Architecture &middot; Dockerfile layers &middot; Lifecycle &middot; Compose &middot; Quiz</p>

<div class="grid">

  <!-- ══════════════════ PANEL 1: VMs vs Containers ══════════════════ -->
  <div class="card">
    <h3>&#9312; VMs vs. Containers</h3>
    <canvas id="cvComp" width="420" height="260"></canvas>
    <div class="btn-row" style="margin-top:10px;">
      <button id="btnVM"  class="active" onclick="setCompMode('vm')">Virtual Machines</button>
      <button id="btnCT"  onclick="setCompMode('containers')">Containers</button>
      <button id="btnBoth" onclick="setCompMode('both')">Side by Side</button>
    </div>
    <div class="info-box" id="compInfo">
      <span class="hl">Virtual Machines</span> run a full OS per app. Each VM needs its own kernel, binaries, and libraries &#8212; taking GBs of RAM and minutes to boot.
    </div>
  </div>

  <!-- ══════════════════ PANEL 2: Docker Architecture ══════════════════ -->
  <div class="card">
    <h3>&#9313; Docker Architecture</h3>
    <canvas id="cvArch" width="420" height="260"></canvas>
    <div class="btn-row" style="margin-top:10px;">
      <button id="btnClient" class="active" onclick="setArchFocus('client')">Client</button>
      <button id="btnDaemon" onclick="setArchFocus('daemon')">Daemon</button>
      <button id="btnRegistry" onclick="setArchFocus('registry')">Registry</button>
      <button id="btnAll" onclick="setArchFocus('all')">Full Flow</button>
    </div>
    <div class="info-box" id="archInfo">
      The <span class="hl">Docker CLI</span> is the client you type into. Commands like <span class="hl2">docker run</span> are sent over a socket to the Docker Daemon, which does all the real work.
    </div>
  </div>

  <!-- ══════════════════ PANEL 3: Dockerfile + Layers ══════════════════ -->
  <div class="card">
    <h3>&#9314; Dockerfile &amp; Image Layers</h3>
    <div style="display:flex; gap:12px; align-items:flex-start;">
      <div style="flex:1;" id="dfLines">
        <!-- filled by JS -->
      </div>
      <canvas id="cvLayers" width="160" height="260"></canvas>
    </div>
    <div class="info-box" id="layerInfo">
      Click any Dockerfile instruction to see its layer. Each instruction creates a <span class="hl">read-only layer</span>. Only the topmost <span class="hl2">container layer</span> is writable at runtime.
    </div>
  </div>

  <!-- ══════════════════ PANEL 4: Container Lifecycle ══════════════════ -->
  <div class="card">
    <h3>&#9315; Container Lifecycle</h3>
    <canvas id="cvLife" width="420" height="200"></canvas>
    <div class="btn-row" style="margin-top:10px;">
      <button onclick="lcTransition('create')">docker create</button>
      <button onclick="lcTransition('start')">docker start</button>
      <button onclick="lcTransition('pause')">docker pause</button>
      <button onclick="lcTransition('unpause')">docker unpause</button>
      <button onclick="lcTransition('stop')">docker stop</button>
      <button onclick="lcTransition('remove')">docker rm</button>
    </div>
    <div class="info-box" id="lcInfo">
      A container begins as <span class="hl">non-existent</span>. Use <span class="hl2">docker create</span> to allocate it, <span class="hlg">docker start</span> to run it, and <span class="hl3">docker stop</span> to freeze it gracefully.
    </div>
  </div>

  <!-- ══════════════════ PANEL 5: Docker Compose (full) ══════════════════ -->
  <div class="card full">
    <h3>&#9316; Docker Compose &#8212; Multi-Container App</h3>
    <canvas id="cvCompose" width="860" height="220"></canvas>
    <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
      <button id="btnReq" onclick="animateCompose('request')">Send Request</button>
      <button id="btnDB"  onclick="animateCompose('db')">DB Query</button>
      <button id="btnScale" onclick="animateCompose('scale')">Scale App</button>
      <button id="btnNet"  onclick="animateCompose('net')">Show Network</button>
    </div>
    <div class="info-box" id="composeInfo" style="margin-top:8px;">
      <span class="hl">docker-compose.yml</span> defines every service, its image, environment, ports, and network links &#8212; letting you spin up an entire multi-container stack with a single <span class="hl2">docker compose up</span>.
    </div>
  </div>

  <!-- ══════════════════ PANEL 6: QUIZ (full, React) ══════════════════ -->
  <div class="card full">
    <h3>&#9316; Quiz &#8212; Test Your Docker Knowledge</h3>
    <div id="quiz-root"></div>
  </div>

</div>

<!-- ══════════════════════════════════════════════════════════════════
     SCRIPTS
══════════════════════════════════════════════════════════════════ -->
<script>
// ─── colours ───────────────────────────────────────────────────────
const C = {
  bg:    '#08080f',
  bg2:   '#111118',
  bg3:   '#0d0d18',
  border:'#1e1e2e',
  blue:  '#2496ed',
  teal:  '#4ecdc4',
  amber: '#fbbf24',
  green: '#4ade80',
  red:   '#f87171',
  purple:'#c084fc',
  gray:  '#71717a',
  light: '#94a3b8',
  text:  '#e4e4e7',
};

function roundRect(ctx, x, y, w, h, r, fill, stroke, sw=1) {
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.lineTo(x+w-r, y); ctx.arcTo(x+w, y, x+w, y+r, r);
  ctx.lineTo(x+w, y+h-r); ctx.arcTo(x+w, y+h, x+w-r, y+h, r);
  ctx.lineTo(x+r, y+h); ctx.arcTo(x, y+h, x, y+h-r, r);
  ctx.lineTo(x, y+r); ctx.arcTo(x, y, x+r, y, r);
  ctx.closePath();
  if (fill)  { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke){ ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke(); }
}

function label(ctx, text, x, y, color='#94a3b8', size=11, align='center', bold=false) {
  ctx.fillStyle = color;
  ctx.font = `${bold?'700':'400'} ${size}px 'JetBrains Mono', monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y);
}

function arrow(ctx, x1, y1, x2, y2, color='#2d2d40', animated=false, t=0) {
  const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx,dy);
  const ux = dx/len, uy = dy/len;
  if (animated) {
    const px = x1 + (((t*0.003)%1)) * dx;
    const py = y1 + (((t*0.003)%1)) * dy;
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2);
    ctx.fillStyle = color; ctx.fill();
  }
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
  const hx = x2 - ux*8, hy = y2 - uy*8;
  ctx.beginPath(); ctx.moveTo(x2,y2);
  ctx.lineTo(hx - uy*4, hy + ux*4);
  ctx.lineTo(hx + uy*4, hy - ux*4);
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}


// ─────────────────────────────────────────────────────────────────────
// PANEL 1 — VMs vs Containers
// ─────────────────────────────────────────────────────────────────────
let compMode = 'vm';
const compInfos = {
  vm: '<span class="hl">Virtual Machines</span> run a full OS per app. Each VM needs its own kernel, binaries, and libraries — taking GBs of RAM and minutes to boot.',
  containers: '<span class="hl2">Containers</span> share the host OS kernel. They package only the app + its deps. They start in <span class="hlg">milliseconds</span> and use <span class="hlg">MBs</span> instead of GBs.',
  both: 'The key difference: VMs virtualise <span class="hl">hardware</span>, containers virtualise the <span class="hl2">OS</span>. Containers are lighter, faster, and more portable — but share the host kernel.'
};

function setCompMode(m) {
  compMode = m;
  document.getElementById('compInfo').innerHTML = compInfos[m];
  ['VM','CT','Both'].forEach(k => document.getElementById('btn'+k).classList.remove('active'));
  document.getElementById('btn'+{vm:'VM',containers:'CT',both:'Both'}[m]).classList.add('active');
  drawComp();
}

function drawVMStack(ctx, ox, oy, w, title, apps) {
  const layers = [
    { label: 'App', color: C.blue,   h: 28 },
    { label: 'Bins/Libs', color: C.teal, h: 22 },
    { label: 'Guest OS', color: C.purple, h: 22 },
    { label: 'Hypervisor', color: '#3f3f70', h: 22 },
    { label: 'Host OS', color: '#2d2d50', h: 22 },
    { label: 'Hardware', color: '#1e1e38', h: 22 },
  ];
  let y = oy;
  if (title) { label(ctx, title, ox + w/2, y - 12, C.light, 10, 'center', true); }
  layers.forEach(l => {
    roundRect(ctx, ox, y, w, l.h-2, 4, l.color+'22', l.color+'66', 1);
    label(ctx, l.label, ox+w/2, y+l.h/2-2, l.color, 9, 'center', true);
    y += l.h;
  });
}

function drawContainerStack(ctx, ox, oy, w, apps) {
  // multiple containers
  const appW = (w - (apps.length-1)*6) / apps.length;
  apps.forEach((app, i) => {
    const ax = ox + i*(appW+6);
    roundRect(ctx, ax, oy, appW, 26, 4, C.blue+'22', C.blue+'66', 1);
    label(ctx, app, ax+appW/2, oy+13, C.blue, 9, 'center', true);
    roundRect(ctx, ax, oy+30, appW, 20, 4, C.teal+'22', C.teal+'66', 1);
    label(ctx, 'Bins/Libs', ax+appW/2, oy+40, C.teal, 8.5, 'center');
  });
  const bottom = oy + 54;
  roundRect(ctx, ox, bottom+4, w, 20, 4, C.amber+'22', C.amber+'66', 1);
  label(ctx, 'Docker Engine', ox+w/2, bottom+14, C.amber, 9, 'center', true);
  roundRect(ctx, ox, bottom+28, w, 20, 4, '#2d2d50', '#555588', 1);
  label(ctx, 'Host OS', ox+w/2, bottom+38, C.light, 9, 'center');
  roundRect(ctx, ox, bottom+52, w, 20, 4, '#1e1e38', '#333360', 1);
  label(ctx, 'Hardware', ox+w/2, bottom+62, C.gray, 9, 'center');
}

function drawComp() {
  const cv = document.getElementById('cvComp');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  roundRect(ctx, 0, 0, cv.width, cv.height, 8, C.bg3, C.border, 1);

  if (compMode === 'vm') {
    label(ctx, 'VIRTUAL MACHINE MODEL', cv.width/2, 18, C.blue, 10, 'center', true);
    // 3 VMs side by side
    const w = 110, gap = 16, total = 3*w + 2*gap;
    const ox = (cv.width - total)/2;
    for (let i=0; i<3; i++) {
      drawVMStack(ctx, ox + i*(w+gap), 36, w, `VM ${i+1}`, ['App '+(i+1)]);
    }
  } else if (compMode === 'containers') {
    label(ctx, 'CONTAINER MODEL', cv.width/2, 18, C.teal, 10, 'center', true);
    const ox = 20, w = cv.width - 40;
    drawContainerStack(ctx, ox, 36, w, ['App 1','App 2','App 3']);
  } else {
    // both side by side
    label(ctx, 'VM', 105, 18, C.blue, 10, 'center', true);
    label(ctx, 'CONTAINER', 315, 18, C.teal, 10, 'center', true);
    drawVMStack(ctx, 10, 30, 190, null, ['App']);
    drawContainerStack(ctx, 220, 30, 200, ['App 1','App 2']);
    // divider
    ctx.beginPath(); ctx.moveTo(210, 20); ctx.lineTo(210, 250);
    ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.setLineDash([4,4]); ctx.stroke();
    ctx.setLineDash([]);
    label(ctx, 'vs', 210, 135, C.gray, 10, 'center', true);
  }
}
drawComp();


// ─────────────────────────────────────────────────────────────────────
// PANEL 2 — Docker Architecture
// ─────────────────────────────────────────────────────────────────────
let archFocus = 'client';
const archInfos = {
  client:   'The <span class="hl">Docker CLI</span> (or any REST client) sends commands to the daemon via a Unix socket or TCP. You type <span class="hl2">docker run</span> — the client forwards it.',
  daemon:   'The <span class="hl">dockerd</span> daemon manages images, containers, networks, and volumes. It exposes a REST API and calls <span class="hl2">containerd</span> to actually run containers.',
  registry: 'A <span class="hl">Registry</span> stores Docker images. <span class="hl2">Docker Hub</span> is the default public registry. You can self-host with <span class="hlg">registry:2</span> or use AWS ECR / GCR.',
  all:      '<span class="hl">docker pull</span> → registry sends image → daemon stores it → <span class="hl2">docker run</span> → daemon creates container from image layers.'
};

function setArchFocus(f) {
  archFocus = f;
  document.getElementById('archInfo').innerHTML = archInfos[f];
  ['Client','Daemon','Registry','All'].forEach(k => document.getElementById('btn'+k).classList.remove('active'));
  document.getElementById('btn'+{client:'Client',daemon:'Daemon',registry:'Registry',all:'All'}[f]).classList.add('active');
  archAnimT = 0;
  drawArch();
}

let archAnimT = 0;
let archRaf = null;

function drawArch(t=0) {
  const cv = document.getElementById('cvArch');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  roundRect(ctx,0,0,cv.width,cv.height,8,C.bg3,C.border,1);

  const hi = (thing) => {
    if (archFocus==='all') return 1;
    const map = { client:['cli'], daemon:['daemon','img','cnt','vol','net'], registry:['registry'] };
    return (map[archFocus]||[]).includes(thing) ? 1 : 0.25;
  };

  // CLI box
  const cliAlpha = hi('cli');
  ctx.globalAlpha = cliAlpha;
  roundRect(ctx, 14, 90, 100, 80, 8, C.blue+'22', C.blue, 1.5);
  label(ctx, 'Docker CLI', 64, 118, C.blue, 10, 'center', true);
  label(ctx, 'docker run', 64, 134, C.teal, 9, 'center');
  label(ctx, 'docker build', 64, 148, C.teal, 9, 'center');
  label(ctx, 'docker pull', 64, 162, C.teal, 9, 'center');
  ctx.globalAlpha = 1;

  // Daemon box
  const daemonAlpha = hi('daemon');
  ctx.globalAlpha = daemonAlpha;
  roundRect(ctx, 140, 20, 180, 220, 10, C.amber+'11', C.amber+'55', 1);
  label(ctx, 'Docker Daemon (dockerd)', 230, 38, C.amber, 9, 'center', true);
  // sub-boxes
  [[160,56,60,36,'Images',C.teal],[230,56,70,36,'Containers',C.blue],[160,108,60,36,'Volumes',C.purple],[230,108,70,36,'Networks',C.green]].forEach(([x,y,w,h,t,c])=>{
    roundRect(ctx,x,y,w,h,6,c+'18',c+'55',1);
    label(ctx,t,x+w/2,y+h/2,c,9,'center',true);
  });
  label(ctx,'containerd / runc', 230, 192, C.gray, 8.5, 'center');
  roundRect(ctx,170,180,120,26,6,'#1a1a2e','#333355',1);
  label(ctx,'low-level runtime', 230,193,C.gray,8.5,'center');
  ctx.globalAlpha = 1;

  // Arrow CLI→Daemon
  ctx.globalAlpha = (archFocus==='all'||archFocus==='client'||archFocus==='daemon') ? 1 : 0.15;
  arrow(ctx, 115, 130, 138, 130, archFocus==='all' ? C.blue : C.border, archFocus==='all', t);
  label(ctx,'REST/socket', 126, 120, C.gray, 8, 'center');
  ctx.globalAlpha = 1;

  // Registry box
  ctx.globalAlpha = hi('registry');
  roundRect(ctx,342,60,72,100,8,C.purple+'22',C.purple,1.5);
  label(ctx,'Registry', 378, 80, C.purple, 10, 'center', true);
  label(ctx,'Docker Hub', 378, 96, C.light, 8.5, 'center');
  label(ctx,'ECR / GCR', 378, 110, C.light, 8.5, 'center');
  label(ctx,'Private reg', 378, 124, C.light, 8.5, 'center');
  ctx.globalAlpha = 1;

  // Arrow Daemon→Registry
  ctx.globalAlpha = (archFocus==='all'||archFocus==='registry') ? 1 : 0.15;
  arrow(ctx, 322, 90, 340, 90, archFocus==='all' ? C.purple : C.border, archFocus==='all', t);
  label(ctx,'push/pull', 331, 80, C.gray, 8, 'center');
  ctx.globalAlpha = 1;

  // Host label
  ctx.globalAlpha = 0.4;
  roundRect(ctx, 130, 10, 300, 240, 10, null, '#333355', 1);
  label(ctx,'HOST MACHINE', 280, 242, C.gray, 8, 'center');
  ctx.globalAlpha = 1;
}

function archLoop(ts) {
  archAnimT = ts;
  drawArch(ts);
  archRaf = requestAnimationFrame(archLoop);
}
archRaf = requestAnimationFrame(archLoop);
setArchFocus('client');


// ─────────────────────────────────────────────────────────────────────
// PANEL 3 — Dockerfile + Layers
// ─────────────────────────────────────────────────────────────────────
const dfInstructions = [
  { kw:'FROM',    arg:'python:3.11-slim',         color:C.purple, layer:true,  desc:'Base layer: pulls <span class="hl">python:3.11-slim</span> from Docker Hub. Sets the OS + Python runtime. This is a read-only layer from the registry.' },
  { kw:'WORKDIR', arg:'/app',                     color:C.teal,   layer:true,  desc:'Sets the working directory inside the container. Creates <span class="hl2">/app</span> if it does not exist. Adds a thin metadata layer.' },
  { kw:'COPY',    arg:'requirements.txt .',       color:C.amber,  layer:true,  desc:'Copies <span class="hl3">requirements.txt</span> from your host into /app. Only this file — so the next RUN layer is cached unless deps change.' },
  { kw:'RUN',     arg:'pip install -r requirements.txt', color:C.blue, layer:true, desc:'Executes pip install and <span class="hl">commits the result as a layer</span>. Cached if requirements.txt is unchanged — saves rebuild time.' },
  { kw:'COPY',    arg:'. .',                      color:C.amber,  layer:true,  desc:'Copies <span class="hl3">your source code</span> into /app. This layer changes every time code changes — intentionally placed after deps.' },
  { kw:'EXPOSE',  arg:'8000',                     color:C.green,  layer:false, desc:'Documents that the container listens on port 8000. <span class="hlg">Does not actually publish</span> — you still need <span class="hl2">-p 8000:8000</span> at run time.' },
  { kw:'CMD',     arg:'["python","app.py"]',      color:C.red,    layer:false, desc:'Default command run when the container starts. Not a layer — just metadata. Override with <span class="hlr">docker run ... python manage.py</span>.' },
];

let selectedLine = 0;

function buildDockerfileUI() {
  const el = document.getElementById('dfLines');
  el.innerHTML = '';
  dfInstructions.forEach((d, i) => {
    const div = document.createElement('div');
    div.className = 'df-line' + (i===selectedLine?' selected':'');
    div.onclick = () => { selectedLine=i; buildDockerfileUI(); drawLayers(); document.getElementById('layerInfo').innerHTML = d.desc; };
    div.innerHTML = `<span class="df-kw" style="color:${d.color}">${d.kw}</span><span class="df-arg">${d.arg}</span>`;
    el.appendChild(div);
  });
}

function drawLayers() {
  const cv = document.getElementById('cvLayers');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  roundRect(ctx,0,0,cv.width,cv.height,8,C.bg3,C.border,1);

  const layers = dfInstructions.filter(d=>d.layer);
  const totalH = cv.height - 20;
  const lh = Math.floor(totalH / (layers.length+1)) - 4;
  const pad = 8;

  // Container writable layer at top
  roundRect(ctx, pad, 8, cv.width-2*pad, lh, 6, C.green+'22', C.green+'66', 1.5);
  label(ctx,'container layer', cv.width/2, 8+lh/2, C.green, 8, 'center', true);
  label(ctx,'(writable)', cv.width/2, 8+lh/2+11, C.green+'99', 7, 'center');

  layers.forEach((d, i) => {
    const y = 8 + (lh+4) + i*(lh+4);
    const isSelected = dfInstructions.indexOf(d) === selectedLine;
    roundRect(ctx, pad, y, cv.width-2*pad, lh, 6,
      isSelected ? d.color+'33' : d.color+'11',
      isSelected ? d.color : d.color+'44',
      isSelected ? 1.5 : 1);
    label(ctx, d.kw, cv.width/2, y+lh/2-4, d.color, 9, 'center', true);
    // layer index
    ctx.fillStyle = d.color+'77';
    ctx.font = '400 7px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('layer '+(layers.length - i), cv.width/2, y+lh/2+8);
  });

  // stack indicator
  ctx.fillStyle = C.gray;
  ctx.font = '400 7px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText('▲ stacked R/O layers', cv.width/2, cv.height-4);
}

buildDockerfileUI();
drawLayers();
document.getElementById('layerInfo').innerHTML = dfInstructions[0].desc;


// ─────────────────────────────────────────────────────────────────────
// PANEL 4 — Container Lifecycle
// ─────────────────────────────────────────────────────────────────────
const lcStates = ['non-existent','created','running','paused','stopped','removed'];
const lcColors = { 'non-existent':C.gray, 'created':C.amber, 'running':C.green, 'paused':C.blue, 'stopped':C.red, 'removed':C.gray };
const lcTransitions = {
  create:   { from:['non-existent'],       to:'created',      cmd:'docker create IMAGE' },
  start:    { from:['created','stopped'],  to:'running',      cmd:'docker start CONTAINER' },
  pause:    { from:['running'],            to:'paused',       cmd:'docker pause CONTAINER' },
  unpause:  { from:['paused'],             to:'running',      cmd:'docker unpause CONTAINER' },
  stop:     { from:['running','paused'],   to:'stopped',      cmd:'docker stop CONTAINER' },
  remove:   { from:['created','stopped'],  to:'removed',      cmd:'docker rm CONTAINER' },
};
const lcInfoTexts = {
  'non-existent': 'The container <span class="hl">does not exist yet</span>. No resources allocated. Use <span class="hl2">docker create</span> or <span class="hl2">docker run</span> to bring it into existence.',
  'created':  '<span class="hl3">Created</span> — resources allocated (filesystem, network), but the process has not started. <span class="hl2">docker run</span> skips this step and jumps straight to running.',
  'running':  '<span class="hlg">Running</span> — the main process (PID 1) is executing. Use <span class="hl2">docker logs</span>, <span class="hl2">docker exec</span>, or <span class="hl2">docker stats</span> to inspect it.',
  'paused':   '<span class="hl">Paused</span> — processes are frozen via SIGSTOP (cgroups freezer). Memory is preserved. CPU is <span class="hl2">not consumed</span>. Useful for snapshots.',
  'stopped':  '<span class="hlr">Stopped</span> — the process exited. The container filesystem still exists; you can <span class="hlg">restart</span> it or <span class="hl2">commit</span> its state to a new image.',
  'removed':  '<span class="hlr">Removed</span> — container and its writable layer are deleted. Use <span class="hl2">-v</span> flag to also remove named volumes.',
};

let lcState = 'non-existent';

function lcTransition(cmd) {
  const t = lcTransitions[cmd];
  if (t && t.from.includes(lcState)) {
    lcState = t.to;
    document.getElementById('lcInfo').innerHTML = lcInfoTexts[lcState];
  } else {
    document.getElementById('lcInfo').innerHTML = `<span class="hlr">Invalid transition:</span> cannot <span class="hl2">${lcTransitions[cmd].cmd}</span> from state <span class="hl">${lcState}</span>.`;
  }
  drawLifecycle();
}

function drawLifecycle() {
  const cv = document.getElementById('cvLife');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  roundRect(ctx,0,0,cv.width,cv.height,8,C.bg3,C.border,1);

  const states = ['non-existent','created','running','paused','stopped','removed'];
  const positions = [
    [50,100], [140,50], [230,100], [320,50], [320,150], [410,100]
  ];

  // edges
  const edges = [
    [0,1,'create', C.amber],
    [1,2,'start',  C.green],
    [2,3,'pause',  C.blue],
    [3,2,'unpause',C.green],
    [2,4,'stop',   C.red],
    [3,4,'stop',   C.red],
    [1,5,'rm',     C.gray],
    [4,5,'rm',     C.gray],
    [4,2,'start',  C.green],
  ];

  edges.forEach(([f,t,lbl,c]) => {
    const [x1,y1]=positions[f], [x2,y2]=positions[t];
    const mx = (x1+x2)/2, my = (y1+y2)/2;
    const isActive = lcState===states[f];
    ctx.globalAlpha = isActive ? 1 : 0.25;
    arrow(ctx, x1+20, y1, x2-20, y2, isActive?c:C.gray);
    label(ctx, lbl, mx, my-8, isActive?c:C.gray+'44', 8, 'center');
    ctx.globalAlpha = 1;
  });

  // nodes
  states.forEach((s, i) => {
    const [x,y] = positions[i];
    const isCurrent = lcState===s;
    const c = lcColors[s];
    roundRect(ctx, x-20, y-14, 40, 28, 6, isCurrent?c+'33':c+'11', isCurrent?c:c+'33', isCurrent?2:1);
    label(ctx, s==='non-existent'?'N/A':s, x, y+1, isCurrent?c:c+'66', 8.5, 'center', isCurrent);
  });

  // current state badge
  const c = lcColors[lcState];
  roundRect(ctx, cv.width-140, cv.height-36, 130, 26, 6, c+'22', c, 1.5);
  label(ctx,'STATE: '+lcState.toUpperCase(), cv.width-75, cv.height-23, c, 8.5, 'center', true);
}
drawLifecycle();


// ─────────────────────────────────────────────────────────────────────
// PANEL 5 — Docker Compose
// ─────────────────────────────────────────────────────────────────────
const composeInfoTexts = {
  request: 'A <span class="hl">HTTP request</span> enters via the published port → nginx <span class="hl2">reverse-proxies</span> it to the FastAPI app → app queries postgres → response flows back.',
  db:      '<span class="hl">postgres</span> stores persistent data in a named <span class="hl2">volume</span> (docker volume). It is only reachable from within the <span class="hlg">app-network</span>, not from the public internet.',
  scale:   'Run <span class="hl2">docker compose up --scale app=3</span> to spin up 3 app replicas. Nginx load-balances across them automatically on the internal network.',
  net:     'All services live on <span class="hl">app-network</span> (a Docker bridge network). Only <span class="hl2">nginx</span> exposes port 80 to the host. The others are internal-only.'
};

let composeAnim = 'none';
let composeT = 0;

function animateCompose(mode) {
  composeAnim = mode;
  composeT = 0;
  document.getElementById('composeInfo').innerHTML = composeInfoTexts[mode];
  ['Req','DB','Scale','Net'].forEach(k=>document.getElementById('btn'+k).classList.remove('active'));
  document.getElementById('btn'+{request:'Req',db:'DB',scale:'Scale',net:'Net'}[mode]).classList.add('active');
}

function drawCompose(ts) {
  composeT = ts;
  const cv = document.getElementById('cvCompose');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  roundRect(ctx,0,0,cv.width,cv.height,8,C.bg3,C.border,1);

  // Network background
  if (composeAnim==='net') {
    roundRect(ctx,80,20,700,180,12,C.teal+'08',C.teal+'33',1);
    label(ctx,'app-network (bridge)', 430, 36, C.teal+'66', 9, 'center');
  }

  // Services
  const services = [
    { name:'nginx',   port:':80',  color:C.green,  x:120, note:'reverse proxy' },
    { name:'app',     port:':8000',color:C.blue,   x:310, note:'FastAPI' },
    { name:'app-2',   port:':8000',color:C.blue,   x:430, note:'replica (scaled)' },
    { name:'postgres',port:':5432',color:C.amber,  x:620, note:'database' },
  ];

  const showScale = composeAnim==='scale';
  const svcList = showScale ? services : services.filter(s=>s.name!=='app-2');

  svcList.forEach(svc => {
    const isHighlighted = (
      composeAnim==='request' ||
      (composeAnim==='db' && (svc.name==='postgres'||svc.name==='app')) ||
      (composeAnim==='scale' && svc.name.startsWith('app')) ||
      composeAnim==='net'
    );
    ctx.globalAlpha = isHighlighted ? 1 : 0.4;
    roundRect(ctx, svc.x-50, 60, 100, 100, 10, svc.color+'18', svc.color, isHighlighted?2:1);
    label(ctx, svc.name, svc.x, 92, svc.color, 11, 'center', true);
    label(ctx, svc.note, svc.x, 110, svc.light||C.light, 8.5, 'center');
    label(ctx, svc.port, svc.x, 128, svc.color+'99', 8, 'center');
    // volume for postgres
    if (svc.name==='postgres') {
      roundRect(ctx, svc.x-30, 168, 60, 22, 5, C.amber+'15', C.amber+'44', 1);
      label(ctx, 'volume', svc.x, 179, C.amber+'88', 8, 'center');
      arrow(ctx, svc.x, 162, svc.x, 166, C.amber+'66');
    }
    ctx.globalAlpha = 1;
  });

  // Arrows / animated packets
  const t = ts * 0.001;
  const anim = composeAnim;

  function animPacket(x1,y,x2,color) {
    const frac = (t*0.5) % 1;
    const px = x1 + frac*(x2-x1);
    ctx.beginPath(); ctx.arc(px, y, 4, 0, Math.PI*2);
    ctx.fillStyle = color; ctx.fill();
  }

  // nginx→app
  const appX = showScale ? 310 : 310;
  ctx.globalAlpha = (anim==='request'||anim==='scale'||anim==='net') ? 1 : 0.25;
  arrow(ctx, 172, 110, showScale ? 258 : 258, 110, C.blue);
  if (anim==='request') animPacket(172,110,260,C.blue);
  ctx.globalAlpha = 1;

  // app→app-2 (scale only)
  if (showScale) {
    ctx.globalAlpha = 1;
    arrow(ctx, 362, 110, 378, 110, C.blue+'88');
    if (anim==='scale') animPacket(362,110,378,C.teal);
  }

  // app→postgres
  const pgSrc = showScale ? 482 : 362;
  ctx.globalAlpha = (anim==='request'||anim==='db'||anim==='net') ? 1 : 0.25;
  arrow(ctx, pgSrc, 110, 568, 110, C.amber);
  if (anim==='request'||anim==='db') animPacket(pgSrc,110,568,C.amber);
  ctx.globalAlpha = 1;

  // host→nginx
  ctx.globalAlpha = (anim==='request'||anim==='net') ? 1 : 0.25;
  arrow(ctx, 26, 110, 68, 110, C.green);
  label(ctx,'host:80', 14, 98, C.green+'88', 8, 'center');
  if (anim==='request') animPacket(26,110,68,C.green);
  ctx.globalAlpha = 1;

  // compose.yml label
  label(ctx,'docker-compose.yml', cv.width/2, cv.height-8, C.gray, 9, 'center');

  requestAnimationFrame(drawCompose);
}
requestAnimationFrame(drawCompose);
animateCompose('request');

</script>


<!-- ══════════════════════════════════════════════════════════════════
     REACT QUIZ
══════════════════════════════════════════════════════════════════ -->
<script type="text/babel">
(() => {
  const { useState } = React;

  const questions = [
    {
      q: 'What do containers share with the host that VMs do not?',
      opts: ['CPU', 'The OS kernel', 'RAM', 'Storage'],
      ans: 1,
      fb: 'Containers share the host OS kernel (via namespaces + cgroups). VMs each carry a full guest OS — hence the size and boot-time difference.',
    },
    {
      q: 'Which component receives CLI commands and orchestrates all Docker operations?',
      opts: ['containerd', 'Docker Registry', 'runc', 'Docker Daemon (dockerd)'],
      ans: 3,
      fb: 'dockerd is the long-running background service. The CLI (docker) just sends REST calls to it over a Unix socket.',
    },
    {
      q: 'Why should you COPY requirements.txt before copying source code?',
      opts: ['Security best practice', 'Layer caching — dep layer rarely changes', 'Smaller image', 'Required by Docker'],
      ans: 1,
      fb: 'Docker caches each layer. If requirements.txt is unchanged, the pip install layer is reused — dramatically speeding up rebuilds after code edits.',
    },
    {
      q: 'What does "docker pause" do to a running container?',
      opts: ['Stops and removes it', 'Saves its state to disk', 'Freezes processes via SIGSTOP (no CPU usage)', 'Commits it to an image'],
      ans: 2,
      fb: 'docker pause uses cgroups freezer to suspend all processes. Memory is preserved. CPU drops to zero. docker unpause resumes from exactly that point.',
    },
    {
      q: 'What is the writable layer in a container?',
      opts: ['The base FROM layer', 'A Docker volume', 'The thin read/write layer created on container start', 'The CMD instruction'],
      ans: 2,
      fb: 'All image layers are read-only. Docker adds a thin R/W layer on top per container. Any file writes go here — and are lost when the container is removed (unless a volume is used).',
    },
    {
      q: 'What does "docker compose up --scale app=3" do?',
      opts: ['Increases container memory 3×', 'Runs 3 replicas of the app service', 'Builds the image 3 times', 'Opens port 3 on the host'],
      ans: 1,
      fb: 'Compose starts 3 instances of the app service. Combined with a load balancer (e.g. nginx) in the same compose file, traffic is distributed across all replicas.',
    },
  ];

  function Quiz() {
    const [answers, setAnswers]  = useState({});
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
                <div className="q-num">Question {qi+1}</div>
                <div className="q-prompt">{q.q}</div>
                {q.opts.map((opt, oi) => {
                  let cls = 'q-opt';
                  if (answered) {
                    if (oi === q.ans) cls += ' correct';
                    else if (oi === answers[qi] && !correct) cls += ' wrong'; // only if wrong answer selected
                  }
                  return (
                    <button key={oi} className={cls} onClick={() => pick(qi, oi)} disabled={answered}>
                      {opt}
                    </button>
                  );
                })}
                <div className={`q-feedback ${answered?'show':''} ${answers[qi]?'ok':'bad'}`}>
                  {answered ? (answers[qi] ? '✓ Correct — ' : '✗ Wrong — ') + q.fb : ''}
                </div>
              </div>
            );
          })}
        </div>
        <div className="score-bar">
          <span style={{color:'#52525b',fontSize:'0.78em',flex:1}}>SCORE</span>
          {questions.map((_, i) => (
            <div key={i} className={`pip ${i in answers ? (answers[i]?'ok':'bad') : ''}`}/>
          ))}
          <span style={{color: score===questions.length?'#4ade80':'#2496ed', fontWeight:700}}>
            {done===0 ? 'Answer to begin' : `${score} / ${questions.length}`}
          </span>
          {score===questions.length && done===questions.length &&
            <span style={{color:'#4ade80',marginLeft:8}}>🐳 Docker ready!</span>}
        </div>
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById('quiz-root')).render(<Quiz/>);
})();
</script>

</body>
</html>"""

DOCKER_VISUAL_HEIGHT = 1800