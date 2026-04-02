"""
Self-contained HTML visual for Unsupervised Learning Core Idea.
5 interactive tabs:
  0 — The Core Idea (paradigm comparison, 3 goals, no ground truth)
  1 — K-Means Step-by-Step (Lloyd's algorithm, elbow method)
  2 — DBSCAN & Clustering Shapes (core/border/noise, vs K-Means)
  3 — PCA & Dimensionality Reduction (walkthrough, t-SNE vs UMAP)
  4 — Generative Models & Anomaly Detection (autoencoder, GMM, isolation forest)
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(UL_VISUAL_HTML, height=UL_VISUAL_HEIGHT, scrolling=True)
"""

UL_VISUAL_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
button{cursor:pointer;font-family:inherit;}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#4ecdc4;cursor:pointer;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.fade{animation:fadeIn .3s ease both;}
.card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
.tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
.tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
.tab-btn.active{border-bottom-color:#4ecdc4;color:#4ecdc4;}
.section-title{text-align:center;margin-bottom:20px;}
.section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
.section-title p{font-size:11px;color:#71717a;}
.insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(78,205,196,.06);border-radius:10px;border:1px solid rgba(78,205,196,.2);}
.ins-title{font-size:11px;font-weight:700;color:#4ecdc4;margin-bottom:6px;}
.ins-body{font-size:11px;color:#71717a;line-height:1.8;}
</style>
</head>
<body>
<div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
<script>
/* ─── PALETTE ─── */
var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
  accent:"#4ecdc4",blue:"#38bdf8",purple:"#a78bfa",
  yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
  dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
  orange:"#fb923c"};

/* ─── HELPERS ─── */
function hex(c,a){
  var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
  return 'rgba('+r+','+g+','+b+','+a+')';}
function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
function card(inner,extra){
  return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';}
function sectionTitle(t,s){
  return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';}
function insight(icon,title,body){
  return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';}
function btnSel(idx,cur,color,label,action){
  var on=idx===cur;
  return '<button data-action="'+action+'" data-idx="'+idx
    +'" style="padding:8px 14px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
    +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';}
function sliderRow(action,val,min,max,step,label,fmt){
  var dv=fmt?fmt(val):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:90px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
    +'</div>';}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+(color||C.accent)+';font-weight:700;">'+val+'</span></div>';}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';}

/* ─── PLOT HELPERS ─── */
function mkAxes(svg,PL,PR,PT,PB,W,H,xl,yl,xmin,xmax,ymin,ymax,xtks,ytks){
  var PW=W-PL-PR, PH=H-PT-PB;
  function fx(x){return PL+(x-xmin)/(xmax-xmin)*PW;}
  function fy(y){return PT+PH-(y-ymin)/(ymax-ymin)*PH;}
  var o='';
  (xtks||[]).forEach(function(v){
    o+='<line x1="'+fx(v).toFixed(1)+'" y1="'+PT+'" x2="'+fx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<text x="'+fx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">'+v+'</text>';});
  (ytks||[]).forEach(function(v){
    o+='<line x1="'+PL+'" y1="'+fy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+fy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<text x="'+(PL-5)+'" y="'+(fy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8.5" font-family="monospace">'+v+'</text>';});
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  if(xl) o+='<text x="'+(PL+PW/2)+'" y="'+(H-2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
  if(yl) o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return {svg:o,fx:fx,fy:fy,PW:PW,PH:PH,PL:PL,PT:PT};}

/* ─── MATH ─── */
function dist(a,b){return Math.sqrt(Math.pow(a[0]-b[0],2)+Math.pow(a[1]-b[1],2));}
function mean(pts){
  var mx=0,my=0;
  pts.forEach(function(p){mx+=p[0];my+=p[1];});
  return [mx/pts.length,my/pts.length];}

/* ─── STATE ─── */
var S={
  tab:0,
  goalTab:0,        /* 0=paradigms,1=3goals,2=3layers,3=no-ground-truth */
  showLabels:false,
  kmStep:0,         /* 0..4: init,assign1,update1,assign2,converged */
  kmK:2,
  elbowK:3,
  dbMode:0,         /* 0=dbscan,1=kmeans */
  dbEps:1.8,
  pcaStep:0,        /* 0..3: raw,centered,pc1,projected */
  drMode:0,         /* 0=PCA,1=tSNE,2=UMAP */
  genMode:0,        /* 0=autoencoder,1=gmm,2=anomaly */
  anomalyPoint:0    /* 0=normal,1=anomaly */
};

/* ══════════════════════════════════════════════════════════
   TAB 0 — THE CORE IDEA
══════════════════════════════════════════════════════════ */
function renderCoreIdea(){
  var gt=S.goalTab;
  var showL=S.showLabels;

  /* ── scatter: labeled vs unlabeled ── */
  var pts=[
    [2,2],[2.5,2.8],[1.8,3.2],[3,2.5],
    [7,7],[7.5,7.8],[6.8,7.2],[7.2,6.5],
    [4,8],[4.8,8.5],[3.8,7.8],[5,8.2],
    [4.5,1.5],[5,2],[3.8,1.8],[5.5,1.2]
  ];
  var trueClusters=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3];
  var clusterCols=[C.blue,C.orange,C.purple,C.green];
  var clusterLabels=['Class A','Class B','Class C','Class D'];

  var SW=440,SH=220;
  var ax=mkAxes('',44,16,14,28,SW,SH,'x\u2082','x\u2081',0,10,0,10,
    [0,2,4,6,8,10],[0,2,4,6,8,10]);

  var sv=ax.svg;
  /* draw points */
  pts.forEach(function(p,i){
    var col=showL?clusterCols[trueClusters[i]]:C.muted;
    var r=showL?6:5;
    sv+='<circle cx="'+ax.fx(p[0]).toFixed(1)+'" cy="'+ax.fy(p[1]).toFixed(1)+'" r="'+r
      +'" fill="'+hex(col,0.75)+'" stroke="'+col+'" stroke-width="1.2"/>';
  });
  /* labels */
  if(showL){
    [[2.3,2.5],[7,7.2],[4.4,8.2],[4.8,1.7]].forEach(function(lp,i){
      sv+='<text x="'+ax.fx(lp[0]).toFixed(1)+'" y="'+ax.fy(lp[1]).toFixed(1)+'" fill="'+clusterCols[i]
        +'" font-size="9" font-weight="700">'+clusterLabels[i]+'</text>';});
    sv+='<rect x="'+(SW-90)+'" y="8" width="80" height="18" rx="4" fill="'+hex(C.green,0.1)+'" stroke="'+C.green+'" stroke-width="0.8"/>';
    sv+='<text x="'+(SW-50)+'" y="20" text-anchor="middle" fill="'+C.green+'" font-size="8.5" font-weight="700">Supervised</text>';
  } else {
    sv+='<rect x="'+(SW-100)+'" y="8" width="92" height="18" rx="4" fill="'+hex(C.orange,0.1)+'" stroke="'+C.orange+'" stroke-width="0.8"/>';
    sv+='<text x="'+(SW-54)+'" y="20" text-anchor="middle" fill="'+C.orange+'" font-size="8.5" font-weight="700">Unsupervised</text>';
    sv+='<text x="'+(SW/2)+'" y="'+(SH-8)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8.5">No labels \u2014 structure must be discovered</text>';}

  /* ── 3 goals panel ── */
  var goals=[
    {icon:'&#128209;',name:'Clustering',     col:C.blue,   desc:'Group similar points into discrete clusters',
     obj:'Minimise within-cluster distance', ex:'K-Means, DBSCAN, GMM, Hierarchical'},
    {icon:'&#128200;',name:'Dim. Reduction', col:C.purple, desc:'Compress data to a lower-dimensional space',
     obj:'Maximise explained variance / preserve structure', ex:'PCA, t-SNE, UMAP, Autoencoder'},
    {icon:'&#127922;',name:'Density Est.',   col:C.orange,  desc:'Learn the probability distribution of the data',
     obj:'Maximise data likelihood (ELBO for VAE)', ex:'GMM, KDE, VAE, GAN, Diffusion Models'},
  ];

  var out=sectionTitle('Unsupervised Learning \u2014 The Core Idea',
    'No labels, no supervisor \u2014 discover hidden structure from data alone');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ['&#127757; Paradigms','&#127919; Three Goals','&#129529; Three Layers','&#10067; No Ground Truth'].forEach(function(lbl,i){
    out+=btnSel(i,gt,C.accent,lbl,'goalTab');});
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';

  if(gt===0){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',
        (showL?'Supervised: each point has a label':'Unsupervised: same data, no labels'))
      +svgBox(sv,SW,SH)
      +'<div style="margin-top:10px;">'
      +'<button data-action="toggleLabels" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
      +'background:'+hex(showL?C.green:C.orange,0.12)+';border:1.5px solid '+(showL?C.green:C.orange)+';'
      +'color:'+(showL?C.green:C.orange)+';cursor:pointer;">'
      +(showL?'\u2714 Showing labels \u2014 click to hide':'Show labels (supervised view)')+'</button></div>'
    );
  } else if(gt===1){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Three Structural Goals')
      +goals.map(function(g){
        return '<div style="padding:10px 12px;margin-bottom:8px;border-radius:8px;background:'+hex(g.col,0.07)+';border:1px solid '+hex(g.col,0.25)+';">'
          +'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
          +'<span style="font-size:16px;">'+g.icon+'</span>'
          +'<span style="font-size:11px;font-weight:700;color:'+g.col+';">'+g.name+'</span></div>'
          +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:3px;">'+g.desc+'</div>'
          +'<div style="font-size:8.5px;color:'+C.dim+';">Objective: '+g.obj+'</div></div>';
      }).join('')
    );
  } else if(gt===2){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Three Layers to Keep Separate')
      +[
        {n:'Learning Paradigm',  col:C.accent, d:'Absence of labels; goal is to find latent structure. This is the category of problem.', ex:'Unsupervised learning (vs supervised, RL)'},
        {n:'Structural Objective',col:C.yellow,d:'What the algorithm actually optimises. Defines what \u201cgood structure\u201d means.', ex:'K-Means: inertia \u2014 PCA: variance \u2014 AE: recon error'},
        {n:'Model Class',        col:C.purple, d:'The type of function used. Each makes different assumptions about what structure looks like.', ex:'Centroid, probabilistic, neural, graph-based'},
      ].map(function(r){
        return '<div style="padding:10px 12px;margin-bottom:8px;border-radius:8px;border-left:3px solid '+r.col+';">'
          +'<div style="font-size:10px;font-weight:700;color:'+r.col+';margin-bottom:4px;">'+r.n+'</div>'
          +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:3px;">'+r.d+'</div>'
          +'<div style="font-size:8.5px;color:'+C.dim+';">e.g. '+r.ex+'</div></div>';
      }).join('')
      +div('font-size:8.5px;color:'+C.dim+';margin-top:6px;padding:6px 8px;border-radius:5px;background:#0a0a0f;',
        'Confusion: thinking unsupervised = K-Means, or that dim. reduction = clustering. They are different layers.')
    );
  } else {
    /* no ground truth */
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','The Core Challenge: No Ground Truth')
      +div('padding:10px 12px;border-radius:8px;margin-bottom:10px;background:'+hex(C.green,0.06)+';border:1px solid '+hex(C.green,0.2)+';',
        div('font-size:9.5px;font-weight:700;color:'+C.green+';margin-bottom:5px;','\u2714 Supervised Learning')
        +div('font-size:9px;color:'+C.muted+';','Ground truth exists. Always ask: does prediction match label? '
          +'Train loss, val loss, test accuracy \u2014 all are objective.'))
      +div('padding:10px 12px;border-radius:8px;margin-bottom:10px;background:'+hex(C.red,0.06)+';border:1px solid '+hex(C.red,0.2)+';',
        div('font-size:9.5px;font-weight:700;color:'+C.red+';margin-bottom:5px;','\u2718 Unsupervised Learning')
        +div('font-size:9px;color:'+C.muted+';','No labels to compare against. Internal quality (tight clusters?) \u2260 meaningful structure. '
          +'Human judgement is essential.'))
      +div('padding:10px 12px;border-radius:8px;background:'+hex(C.yellow,0.06)+';border:1px solid '+hex(C.yellow,0.2)+';',
        div('font-size:9.5px;font-weight:700;color:'+C.yellow+';margin-bottom:5px;','Evaluation Strategy')
        +[
          {m:'Internal',  d:'Silhouette score, inertia, reconstruction error',c:C.accent},
          {m:'External',  d:'ARI, NMI \u2014 requires ground truth labels (benchmarking only)',c:C.blue},
          {m:'Downstream',d:'Train supervised model on learned repr. \u2014 the gold standard',c:C.green},
        ].map(function(r){
          return '<div style="display:flex;gap:6px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
            +'<span style="font-size:9px;color:'+r.c+';width:72px;flex-shrink:0;font-weight:700;">'+r.m+'</span>'
            +'<span style="font-size:8.5px;color:'+C.muted+';">'+r.d+'</span></div>';}).join('')
      )
    );
  }
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  if(gt===0){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PARADIGM COMPARISON');
    [
      {n:'Supervised',   lbl:'Has labels',       sig:'Prediction error',      ex:'Classification, regression', c:C.green},
      {n:'Unsupervised', lbl:'No labels',         sig:'Structural objective',  ex:'Clustering, dim. reduction', c:C.orange},
      {n:'Reinforcement',lbl:'No labels',         sig:'Reward signal over time',ex:'Games, robotics, control',  c:C.purple},
      {n:'Self-supervised',lbl:'Self-generated',  sig:'Predict masked input',  ex:'BERT, GPT pre-training',     c:C.blue},
    ].forEach(function(r){
      out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
        +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.n+'</div>'
        +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.lbl+' \u00b7 Signal: '+r.sig+'</div>'
        +'<div style="font-size:8px;color:'+C.muted+';">'+r.ex+'</div></div>';});
    out+='</div>';
  }

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THE 4-STEP PROCESS');
  [
    {n:'1. Collect unlabeled data', d:'Inputs x only \u2014 no targets y'},
    {n:'2. Define structural obj.', d:'Inertia, likelihood, recon error'},
    {n:'3. Discover structure',     d:'Iterate until objective satisfied'},
    {n:'4. Interpret & apply',      d:'Visualise, detect anomalies, compress'},
  ].forEach(function(r){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:2px solid '+C.accent+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+C.accent+';">'+r.n+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.d+'</div></div>';});
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','CHOOSE METHOD BY GOAL');
  [
    {g:'Clustering',    a:'K-Means, DBSCAN, GMM',           c:C.blue},
    {g:'Visualisation', a:'t-SNE, UMAP, PCA',               c:C.purple},
    {g:'Compression',   a:'PCA, Autoencoder',                c:C.green},
    {g:'Generation',    a:'VAE, GAN, Diffusion',             c:C.orange},
    {g:'Anomaly detect',a:'Isolation Forest, AE, LOF',       c:C.red},
  ].forEach(function(r){
    out+=statRow(r.g,r.a,r.c);});
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128269;','The Explorer Analogy',
    'Unsupervised learning is like an explorer dropped into an unknown city with no map and no guide. '
    +'They notice which buildings cluster together, which streets feel similar, and build an internal map '
    +'<span style="color:'+C.accent+';font-weight:700;">entirely from observation</span>. '
    +'No one tells them what they\'re looking at. The structure they discover comes from the geometry of the city itself \u2014 '
    +'just as the structure unsupervised models find comes from the geometry and statistics of the data.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 1 — K-MEANS STEP-BY-STEP
══════════════════════════════════════════════════════════ */
/* 6-point example from the theory */
var KM_PTS=[
  {p:[1,1],lbl:'A'},{p:[1,2],lbl:'B'},{p:[2,1],lbl:'C'},
  {p:[8,8],lbl:'D'},{p:[8,9],lbl:'E'},{p:[9,8],lbl:'F'}];
var KM_STEPS=[
  {title:'Init \u2014 random centroids',                c1:[1,1],c2:[8,8],  assign:[-1,-1,-1,-1,-1,-1]},
  {title:'Assignment Step 1 \u2014 assign to nearest',  c1:[1,1],c2:[8,8],  assign:[0,0,0,1,1,1]},
  {title:'Update Step 1 \u2014 recompute centroids',    c1:[4/3,4/3],c2:[25/3,25/3],assign:[0,0,0,1,1,1]},
  {title:'Assignment Step 2 \u2014 check for changes',  c1:[4/3,4/3],c2:[25/3,25/3],assign:[0,0,0,1,1,1]},
  {title:'Converged \u2014 no assignments changed',     c1:[4/3,4/3],c2:[25/3,25/3],assign:[0,0,0,1,1,1]},
];

function kmInertia(c1,c2,assign){
  var tot=0;
  KM_PTS.forEach(function(pt,i){
    if(assign[i]<0) return;
    var c=assign[i]===0?c1:c2;
    tot+=Math.pow(dist(pt.p,c),2);});
  return tot;}

/* elbow curve data */
function elbowInertia(k){
  /* hand-tuned for the 6-pt dataset */
  if(k===1) return 108;
  if(k===2) return 3.1;
  if(k===3) return 1.4;
  if(k===4) return 0.9;
  if(k===5) return 0.5;
  return 0.2;}

function renderKMeans(){
  var step=S.kmStep;
  var ek=S.elbowK;
  var ks=KM_STEPS[step];

  var SW=440,SH=260;
  var ax=mkAxes('',48,24,18,36,SW,SH,'x\u2081','x\u2082',0,11,0,11,
    [0,2,4,6,8,10],[0,2,4,6,8,10]);
  var sv=ax.svg;
  var c1=ks.c1,c2=ks.c2;

  /* distance lines from points to centroids (step 1 and 3) */
  if(step===1||step===3){
    KM_PTS.forEach(function(pt,i){
      var c=ks.assign[i]===0?c1:c2;
      var col=ks.assign[i]===0?C.blue:C.orange;
      sv+='<line x1="'+ax.fx(pt.p[0]).toFixed(1)+'" y1="'+ax.fy(pt.p[1]).toFixed(1)
        +'" x2="'+ax.fx(c[0]).toFixed(1)+'" y2="'+ax.fy(c[1]).toFixed(1)
        +'" stroke="'+col+'" stroke-width="0.8" stroke-dasharray="4,3" opacity="0.5"/>';});
  }

  /* voronoi boundary (vertical-ish midpoint) at step>=1 */
  if(step>=1){
    /* midpoint between centroids, draw a line perpendicular */
    var mx=(c1[0]+c2[0])/2, my=(c1[1]+c2[1])/2;
    sv+='<line x1="'+ax.fx(mx).toFixed(1)+'" y1="'+ax.fy(0).toFixed(1)
      +'" x2="'+ax.fx(mx).toFixed(1)+'" y2="'+ax.fy(11).toFixed(1)
      +'" stroke="'+C.dim+'" stroke-width="1" stroke-dasharray="6,4" opacity="0.5"/>';
    sv+='<text x="'+ax.fx(mx).toFixed(1)+'" y="'+(ax.fy(10)+2)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8">boundary</text>';}

  /* data points */
  KM_PTS.forEach(function(pt,i){
    var col=ks.assign[i]===0?C.blue:ks.assign[i]===1?C.orange:C.muted;
    sv+='<circle cx="'+ax.fx(pt.p[0]).toFixed(1)+'" cy="'+ax.fy(pt.p[1]).toFixed(1)+'" r="8"'
      +' fill="'+hex(col,0.8)+'" stroke="'+col+'" stroke-width="1.5"/>';
    sv+='<text x="'+ax.fx(pt.p[0]).toFixed(1)+'" y="'+(ax.fy(pt.p[1])+3.5).toFixed(1)+'" text-anchor="middle" fill="#0a0a0f" font-size="9" font-weight="700">'+pt.lbl+'</text>';});

  /* centroids */
  [[c1,C.blue,'C\u2081'],[c2,C.orange,'C\u2082']].forEach(function(cd){
    var x=ax.fx(cd[0][0]),y=ax.fy(cd[0][1]);
    svg_diamond(sv,x,y,12,cd[1]);
    sv+='<text x="'+(x+14)+'" y="'+(y-8)+'" fill="'+cd[1]+'" font-size="9" font-weight="700">'
      +cd[2]+'=('+cd[0][0].toFixed(2)+','+cd[0][1].toFixed(2)+')</text>';});
  function svg_diamond(s,x,y,r,col){
    sv+='<polygon points="'+x+','+(y-r)+' '+(x+r)+','+y+' '+x+','+(y+r)+' '+(x-r)+','+y+'"'
      +' fill="'+hex(col,0.9)+'" stroke="#0a0a0f" stroke-width="2"/>';}

  /* step indicator strip */
  sv+='<rect x="4" y="4" width="432" height="20" rx="4" fill="'+hex(step===4?C.green:C.accent,0.1)+'" stroke="'+(step===4?C.green:C.accent)+'" stroke-width="0.8"/>';
  sv+='<text x="220" y="17" text-anchor="middle" fill="'+(step===4?C.green:C.accent)+'" font-size="9" font-weight="700">Step '+(step+1)+'/5: '+ks.title+'</text>';

  /* ── elbow chart ── */
  var EW=440,EH=160;
  var eax=mkAxes('',48,16,14,30,EW,EH,'K (clusters)','Inertia',0.5,6.5,0,120,
    [1,2,3,4,5,6],[0,30,60,90,120]);
  var ev=eax.svg;
  /* highlight elbow zone */
  var elbowX1=eax.fx(1.7),elbowX2=eax.fx(2.5);
  ev+='<rect x="'+elbowX1.toFixed(1)+'" y="'+(eax.PT)+'" width="'+(elbowX2-elbowX1).toFixed(1)+'" height="'+eax.PH+'" fill="'+hex(C.yellow,0.06)+'"/>';
  ev+='<text x="'+((elbowX1+elbowX2)/2).toFixed(1)+'" y="'+(eax.PT+12)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="8">elbow</text>';
  /* inertia line */
  var pts2=[];
  for(var k2=1;k2<=6;k2++) pts2.push([eax.fx(k2),eax.fy(elbowInertia(k2))]);
  ev+='<path d="'+pts2.map(function(p,i){return (i===0?'M':'L')+p[0].toFixed(1)+','+p[1].toFixed(1);}).join(' ')+'" fill="none" stroke="'+C.accent+'" stroke-width="2.5"/>';
  /* dots */
  for(var k3=1;k3<=6;k3++){
    var kx=eax.fx(k3),ky=eax.fy(elbowInertia(k3));
    var isEl=k3===ek;
    ev+='<circle cx="'+kx.toFixed(1)+'" cy="'+ky.toFixed(1)+'" r="'+(isEl?7:5)+'"'
      +' fill="'+(isEl?C.yellow:C.accent)+'" stroke="#0a0a0f" stroke-width="1.5"/>';
    ev+='<text x="'+kx.toFixed(1)+'" y="'+(ky-10)+'" text-anchor="middle" fill="'+(isEl?C.yellow:C.accent)+'" font-size="8.5">'+elbowInertia(k3).toFixed(1)+'</text>';}
  /* selected K cursor */
  ev+='<text x="'+eax.fx(ek).toFixed(1)+'" y="'+(eax.PT+eax.PH+18)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="8.5" font-weight="700">K='+ek+'</text>';

  var curInertia=kmInertia(ks.c1,ks.c2,ks.assign);

  var out=sectionTitle('K-Means: Lloyd\'s Algorithm','6-point walkthrough \u2014 iterate assignment + update until convergence');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','K=2 on 6 points: A=(1,1) B=(1,2) C=(2,1) D=(8,8) E=(8,9) F=(9,8)')
    +svgBox(sv,SW,SH)
    +'<div style="margin-top:10px;display:flex;justify-content:center;gap:8px;">'
    +'<button data-action="kmPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step>0?0.12:0.04)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
    +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Back</button>'
    +'<button data-action="kmNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step<4?0.12:0.04)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
    +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Next \u2192</button>'
    +'</div>'
    +'<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;margin-top:8px;">'
    +KM_STEPS.map(function(s,i){
      var col=i<step?C.dim:i===step?C.accent:C.border;
      return '<div style="width:32px;height:6px;border-radius:3px;background:'+col+'"></div>';}).join('')
    +'</div>'
  );
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Elbow Method \u2014 Choosing K')
    +svgBox(ev,EW,EH)
    +sliderRow('elbowK',ek,1,6,1,'Select K',function(v){return 'K = '+v;})
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT STATE');
  out+=statRow('Centroid C\u2081','('+ks.c1[0].toFixed(2)+', '+ks.c1[1].toFixed(2)+')',C.blue);
  out+=statRow('Centroid C\u2082','('+ks.c2[0].toFixed(2)+', '+ks.c2[1].toFixed(2)+')',C.orange);
  out+=statRow('Cluster 1',ks.assign.map(function(a,i){return a===0?KM_PTS[i].lbl:'';}).filter(Boolean).join(', ')||'—',C.blue);
  out+=statRow('Cluster 2',ks.assign.map(function(a,i){return a===1?KM_PTS[i].lbl:'';}).filter(Boolean).join(', ')||'—',C.orange);
  out+=statRow('Inertia (WCSS)',step>0?curInertia.toFixed(2):'—',step===4?C.green:C.accent);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','ALGORITHM');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
    '1. <span style="color:'+(step===0?C.yellow:C.dim)+';">Initialise K centroids</span><br>'
    +'2. <span style="color:'+(step===1||step===3?C.blue:C.dim)+';">Assign each point to nearest \u03bc</span><br>'
    +'   cluster(x) = argmin\u2096 ||x \u2212 \u03bc\u2096||&#178;<br>'
    +'3. <span style="color:'+(step===2?C.orange:C.dim)+';">Update: \u03bc\u2096 = mean of cluster k</span><br>'
    +'4. <span style="color:'+(step===4?C.green:C.dim)+';">Repeat until no change \u2192 done</span>'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PROPERTIES');
  [
    {t:'Convergence',    v:'Guaranteed',                c:C.green},
    {t:'Global optimum', v:'Not guaranteed',            c:C.red},
    {t:'Cluster shape',  v:'Spherical only',            c:C.orange},
    {t:'Outlier robust', v:'No \u2014 pulls centroids', c:C.red},
    {t:'K required',     v:'Yes \u2014 set manually',   c:C.yellow},
    {t:'Complexity',     v:'O(n\u00b7k\u00b7d\u00b7I)', c:C.dim},
  ].forEach(function(r){out+=statRow(r.t,r.v,r.c);});
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:6px;','ELBOW METHOD \u2014 K='+ek);
  out+=statRow('Inertia at K='+ek,elbowInertia(ek).toFixed(1),C.accent);
  out+=statRow('Natural clusters','2 (A-B-C and D-E-F)',C.green);
  out+=statRow('Elbow location','K = 2 \u2014 sharpest drop',C.yellow);
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9881;&#65039;','K-Means Finds Structure Without Labels',
    'With K=2, starting from random centroids at A=(1,1) and D=(8,8), Lloyd\'s algorithm converges in just 2 iterations. '
    +'The two natural groups \u2014 {A,B,C} and {D,E,F} \u2014 are recovered with inertia = 3.12. '
    +'The <span style="color:'+C.yellow+';font-weight:700;">elbow at K=2</span> is unmistakable: '
    +'inertia drops from 108 to 3.1 when going from K=1 to K=2, but barely changes thereafter. '
    +'This is how you discover K without labels.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 2 — DBSCAN & CLUSTERING SHAPES
══════════════════════════════════════════════════════════ */
/* Two crescent-shaped clusters + outliers */
function mkCrescents(){
  var pts=[];
  /* upper crescent: class 0 */
  for(var i=0;i<22;i++){
    var a=Math.PI*(0.1+0.8*i/21);
    pts.push({x:5+3.5*Math.cos(a),y:5+3.5*Math.sin(a),cls:0});}
  /* lower crescent: class 1 */
  for(var i=0;i<22;i++){
    var a=Math.PI*(1.1+0.8*i/21);
    pts.push({x:5.5+3.5*Math.cos(a),y:4.8+3.5*Math.sin(a),cls:1});}
  /* outliers */
  [{x:0.8,y:9},{x:9.5,y:1},{x:0.5,y:0.5}].forEach(function(o){
    pts.push({x:o.x,y:o.y,cls:-1});});
  return pts;}

var CRESCENT_PTS=mkCrescents();

function dbscanClassify(pts,eps,minPts){
  /* count neighbors */
  var neighbors=pts.map(function(p,i){
    return pts.filter(function(q,j){return j!==i&&dist([p.x,p.y],[q.x,q.y])<=eps;}).length;});
  var isCore=neighbors.map(function(n){return n>=minPts;});
  /* assign cluster labels via BFS */
  var labels=pts.map(function(){return -1;});
  var cid=0;
  for(var i=0;i<pts.length;i++){
    if(!isCore[i]||labels[i]>=0) continue;
    labels[i]=cid;
    var queue=[i];
    while(queue.length){
      var cur=queue.shift();
      pts.forEach(function(p,j){
        if(labels[j]>=0) return;
        if(dist([pts[cur].x,pts[cur].y],[p.x,p.y])<=eps){
          labels[j]=cid;
          if(isCore[j]) queue.push(j);}});}
    cid++;}
  return {labels:labels,isCore:isCore};}

function kmeansCrescents(pts){
  /* run K-means with K=2 */
  var c=[{x:3,y:7},{x:7,y:3}];
  for(var iter=0;iter<20;iter++){
    var asgn=pts.map(function(p){
      return dist([p.x,p.y],[c[0].x,c[0].y])<dist([p.x,p.y],[c[1].x,c[1].y])?0:1;});
    [0,1].forEach(function(k){
      var grp=pts.filter(function(p,i){return asgn[i]===k;});
      if(!grp.length) return;
      c[k]={x:grp.reduce(function(s,p){return s+p.x;},0)/grp.length,
            y:grp.reduce(function(s,p){return s+p.y;},0)/grp.length};});}
  return {asgn:pts.map(function(p){
    return dist([p.x,p.y],[c[0].x,c[0].y])<dist([p.x,p.y],[c[1].x,c[1].y])?0:1;}),cents:c};}

function renderDBSCAN(){
  var mode=S.dbMode;
  var eps=S.dbEps;
  var minPts=3;
  var pts=CRESCENT_PTS;

  var SW=440,SH=280;
  var ax=mkAxes('',44,20,18,32,SW,SH,'x\u2081','x\u2082',0,11,0,11,
    [0,2,4,6,8,10],[0,2,4,6,8,10]);
  var sv=ax.svg;

  if(mode===0){
    /* DBSCAN */
    var db=dbscanClassify(pts,eps,minPts);
    var clCols=[C.blue,C.orange,C.purple,C.green];
    /* eps rings for core points */
    pts.forEach(function(p,i){
      if(db.isCore[i]){
        sv+='<circle cx="'+ax.fx(p.x).toFixed(1)+'" cy="'+ax.fy(p.y).toFixed(1)+'" r="'+(eps/(11)*ax.PW).toFixed(1)+'"'
          +' fill="none" stroke="'+hex(clCols[db.labels[i]]||C.dim,0.15)+'" stroke-width="0.8"/>';}});
    /* points */
    pts.forEach(function(p,i){
      var lbl=db.labels[i];
      var col=lbl<0?C.dim:(clCols[lbl]||C.muted);
      var isCore=db.isCore[i];
      var isBorder=!isCore&&lbl>=0;
      sv+='<circle cx="'+ax.fx(p.x).toFixed(1)+'" cy="'+ax.fy(p.y).toFixed(1)+'" r="'+(isCore?7:isBorder?6:5)+'"'
        +' fill="'+hex(col,isCore?0.85:isBorder?0.5:0.2)+'" stroke="'+col+'" stroke-width="'+(isCore?2:1)+'"'
        +(lbl<0?' stroke-dasharray="3,2"':'')+'/>';});
    /* legend */
    sv+='<rect x="'+(SW-118)+'" y="6" width="110" height="54" rx="4" fill="#0a0a0f" opacity="0.92"/>';
    [[C.blue,'Core point (filled)'],[C.blue,'Border (faint)'],[C.dim,'Noise (grey \u25c7']].forEach(function(r,ri){
      sv+='<circle cx="'+(SW-108)+'" cy="'+(20+ri*14)+'" r="5" fill="'+hex(r[0],ri===1?0.4:0.8)+'" stroke="'+r[0]+'" stroke-width="1"/>';
      sv+='<text x="'+(SW-98)+'" y="'+(24+ri*14)+'" fill="'+C.muted+'" font-size="8.5">'+r[1]+'</text>';});
    sv+='<text x="'+(SW-118+55)+'" y="'+(6+54+10)+'" text-anchor="middle" fill="'+C.accent+'" font-size="8.5">\u03b5='+eps.toFixed(1)+', MinPts='+minPts+'</text>';
  } else {
    /* K-Means K=2 on crescents */
    var km=kmeansCrescents(pts);
    pts.forEach(function(p,i){
      var col=km.asgn[i]===0?C.blue:C.orange;
      sv+='<circle cx="'+ax.fx(p.x).toFixed(1)+'" cy="'+ax.fy(p.y).toFixed(1)+'" r="6"'
        +' fill="'+hex(col,0.75)+'" stroke="'+col+'" stroke-width="1.2"/>';});
    km.cents.forEach(function(c,ci){
      var col=ci===0?C.blue:C.orange;
      var x=ax.fx(c.x),y=ax.fy(c.y);
      sv+='<polygon points="'+x+','+(y-10)+' '+(x+10)+','+y+' '+x+','+(y+10)+' '+(x-10)+','+y+'"'
        +' fill="'+hex(col,0.9)+'" stroke="#0a0a0f" stroke-width="2"/>';
      sv+='<text x="'+(x+12)+'" y="'+(y+4)+'" fill="'+col+'" font-size="8.5" font-weight="700">C'+(ci+1)+'</text>';});
    sv+='<text x="220" y="'+(SH-6)+'" text-anchor="middle" fill="'+C.red+'" font-size="9" font-weight="700">K-Means fails on non-spherical clusters!</text>';}

  var db2=dbscanClassify(pts,eps,minPts);
  var nCore=db2.isCore.filter(Boolean).length;
  var nNoise=db2.labels.filter(function(l){return l<0;}).length;
  var nClusters=db2.labels.reduce(function(mx,l){return l>mx?l:mx;},-1)+1;

  var out=sectionTitle('DBSCAN vs K-Means \u2014 Clustering Shapes',
    'DBSCAN finds arbitrarily shaped clusters and flags outliers \u2014 no K required');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  out+=btnSel(0,mode,C.accent,'&#128209; DBSCAN','dbMode');
  out+=btnSel(1,mode,C.red,'&#9940; K-Means (K=2)','dbMode');
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',
      mode===0?'DBSCAN on crescent + outlier dataset':'K-Means (K=2) on same dataset')
    +svgBox(sv,SW,SH)
    +(mode===0?sliderRow('dbEps',eps,0.5,3.0,0.1,'\u03b5 radius',function(v){return v.toFixed(1);}):'')
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  if(mode===0){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','DBSCAN RESULTS (\u03b5='+eps.toFixed(1)+', MinPts=3)');
    out+=statRow('Clusters found',nClusters,nClusters===2?C.green:C.yellow);
    out+=statRow('Core points',nCore,C.blue);
    out+=statRow('Noise points',nNoise,nNoise<=3?C.green:C.red);
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THREE POINT TYPES');
    [
      {t:'Core',    col:C.blue,   d:'Has \u2265 MinPts neighbours within radius \u03b5. Solid filled.'},
      {t:'Border',  col:C.accent, d:'Within \u03b5 of a core point, but fewer than MinPts neighbours itself.'},
      {t:'Noise',   col:C.dim,    d:'Not within \u03b5 of any core point. Treated as outliers.'},
    ].forEach(function(r){
      out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.col+';">'
        +'<div style="font-size:9px;font-weight:700;color:'+r.col+';">'+r.t+'</div>'
        +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.d+'</div></div>';});
    out+='</div>';
  } else {
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.red+';margin-bottom:8px;font-weight:700;','WHY K-MEANS FAILS HERE');
    out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
      'K-Means assumes <span style="color:'+C.red+';font-weight:700;">spherical clusters</span> of similar size. '
      +'Its decision boundary is always the perpendicular bisector between centroids \u2014 a straight line. '
      +'The crescent shapes are non-convex: points near each crescent\'s tips are mis-assigned. '
      +'<span style="color:'+C.accent+';font-weight:700;">DBSCAN</span> finds the correct boundary by following density.');
    out+='</div>';
  }
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','DBSCAN vs K-MEANS');
  [
    {feat:'Specifying K',        km:'Required',        db:'Not needed',        dbg:true},
    {feat:'Cluster shape',       km:'Spherical only',  db:'Any shape',         dbg:true},
    {feat:'Outlier handling',    km:'Distorts centroids',db:'Explicitly labels',dbg:true},
    {feat:'Variable density',    km:'Handles OK',      db:'Struggles',         dbg:false},
    {feat:'Scalability',         km:'O(nkdI)',         db:'O(n\u00b7log n)',   dbg:true},
  ].forEach(function(r){
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="flex:1.1;color:'+C.dim+';">'+r.feat+'</div>'
      +'<div style="flex:1;color:'+C.red+';">\u2716 '+r.km+'</div>'
      +'<div style="flex:1;color:'+(r.dbg?C.green:C.orange)+'">'+(r.dbg?'\u2714':'\u26a0')+' '+r.db+'</div></div>';});
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128209;','Density as the Definition of a Cluster',
    'DBSCAN\'s insight: a cluster is a <span style="color:'+C.accent+';font-weight:700;">region of high density</span>. '
    +'Instead of minimising distance to centroids, it asks: how many neighbours does each point have within radius \u03b5? '
    +'Core points have enough neighbours to anchor a cluster; border points are absorbed; noise points are isolated. '
    +'Drag the \u03b5 slider to see how the neighbourhood radius changes cluster boundaries and noise classification in real time.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 3 — PCA & DIMENSIONALITY REDUCTION
══════════════════════════════════════════════════════════ */
/* 4-point dataset from theory */
var PCA_DATA=[[2,4],[3,5],[5,7],[8,9]];
var PCA_MEAN=[4.5,6.25];
var PCA_CENTERED=[[-2.5,-2.25],[-1.5,-1.25],[0.5,0.75],[3.5,2.75]];
var PCA_V1=[0.766,0.643]; /* first principal component */
var PCA_PROJ=[-3.362,-1.953,0.865,4.449];
var PCA_COLS=[C.blue,C.orange,C.purple,C.green];
var PCA_LBLS=['P1','P2','P3','P4'];
var PC_VAR=[99.7,0.3]; /* % variance explained */

function renderPCA(){
  var step=S.pcaStep;
  var drMode=S.drMode;

  var SW=440,SH=260;
  var ax=mkAxes('',48,20,18,32,SW,SH,'x\u2081','x\u2082',-5,11,-5,11,
    [-4,-2,0,2,4,6,8,10],[-4,-2,0,2,4,6,8,10]);
  var sv=ax.svg;

  /* raw data */
  var rawPts=step===0?PCA_DATA:PCA_CENTERED;
  var origin=step===0?[0,0]:[0,0];

  /* PC1 direction line */
  if(step>=2){
    /* draw PC1 axis through origin */
    var len=8;
    var v=PCA_V1;
    var x1=ax.fx(-v[0]*len),y1=ax.fy(-v[1]*len);
    var x2=ax.fx(v[0]*len),y2=ax.fy(v[1]*len);
    sv+='<line x1="'+x1.toFixed(1)+'" y1="'+y1.toFixed(1)+'" x2="'+x2.toFixed(1)+'" y2="'+y2.toFixed(1)+'" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="6,3"/>';
    sv+='<text x="'+(x2+6).toFixed(1)+'" y="'+(y2+4).toFixed(1)+'" fill="'+C.yellow+'" font-size="9" font-weight="700">PC1 (99.7%)</text>';}

  /* projection lines */
  if(step>=3){
    PCA_CENTERED.forEach(function(p,i){
      /* projection = (p . v1) * v1 */
      var t=p[0]*PCA_V1[0]+p[1]*PCA_V1[1];
      var projPt=[t*PCA_V1[0],t*PCA_V1[1]];
      sv+='<line x1="'+ax.fx(p[0]).toFixed(1)+'" y1="'+ax.fy(p[1]).toFixed(1)
        +'" x2="'+ax.fx(projPt[0]).toFixed(1)+'" y2="'+ax.fy(projPt[1]).toFixed(1)
        +'" stroke="'+PCA_COLS[i]+'" stroke-width="1" stroke-dasharray="3,2" opacity="0.6"/>';
      /* projection point on PC1 */
      sv+='<circle cx="'+ax.fx(projPt[0]).toFixed(1)+'" cy="'+ax.fy(projPt[1]).toFixed(1)+'" r="5"'
        +' fill="'+hex(PCA_COLS[i],0.9)+'" stroke="#0a0a0f" stroke-width="1.5"/>';});}

  /* origin cross for centered data */
  if(step>=1){
    sv+='<line x1="'+ax.fx(-0.4)+'" y1="'+ax.fy(0).toFixed(1)+'" x2="'+ax.fx(0.4)+'" y2="'+ax.fy(0).toFixed(1)+'" stroke="'+C.dim+'" stroke-width="1"/>';
    sv+='<line x1="'+ax.fx(0)+'" y1="'+ax.fy(-0.4).toFixed(1)+'" x2="'+ax.fx(0)+'" y2="'+ax.fy(0.4).toFixed(1)+'" stroke="'+C.dim+'" stroke-width="1"/>';
    sv+='<text x="'+ax.fx(0.4)+'" y="'+ax.fy(-0.6).toFixed(1)+'" fill="'+C.dim+'" font-size="8.5">origin</text>';}

  /* data points */
  rawPts.forEach(function(p,i){
    sv+='<circle cx="'+ax.fx(p[0]).toFixed(1)+'" cy="'+ax.fy(p[1]).toFixed(1)+'" r="8"'
      +' fill="'+hex(PCA_COLS[i],0.8)+'" stroke="'+PCA_COLS[i]+'" stroke-width="1.5"/>';
    sv+='<text x="'+ax.fx(p[0]).toFixed(1)+'" y="'+(ax.fy(p[1])+3.5).toFixed(1)+'" text-anchor="middle" fill="#0a0a0f" font-size="8.5" font-weight="700">'+PCA_LBLS[i]+'</text>';});

  /* mean cross for step 0 */
  if(step===0){
    sv+='<circle cx="'+ax.fx(PCA_MEAN[0]).toFixed(1)+'" cy="'+ax.fy(PCA_MEAN[1]).toFixed(1)+'" r="5" fill="none" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="3,2"/>';
    sv+='<text x="'+ax.fx(PCA_MEAN[0]).toFixed(1)+'" y="'+(ax.fy(PCA_MEAN[1])-8).toFixed(1)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="8.5">\u03bc=(4.5, 6.25)</text>';}

  /* step header */
  var stepLabels=['Step 0: Raw data','Step 1: Centred data (subtract mean)','Step 2: PC1 direction (max variance)','Step 3: Project onto PC1 (1D)'];
  var stepCols=[C.muted,C.blue,C.yellow,C.accent];
  sv+='<rect x="4" y="4" width="432" height="20" rx="4" fill="'+hex(stepCols[step],0.1)+'" stroke="'+stepCols[step]+'" stroke-width="0.8"/>';
  sv+='<text x="220" y="17" text-anchor="middle" fill="'+stepCols[step]+'" font-size="9" font-weight="700">'+stepLabels[step]+'</text>';

  /* ── DR method comparison panel ── */
  var drData=[
    {name:'PCA',label:'Linear',strength:'Global structure, fast, deterministic, re-applicable', weakness:'Misses non-linear structure', use:'Compression, preprocessing, visualisation', col:C.blue},
    {name:'t-SNE',label:'Non-linear',strength:'Reveals cluster structure invisible to PCA', weakness:'Slow O(N\u00b2), non-deterministic, no new-point projection', use:'Visualisation only', col:C.orange},
    {name:'UMAP',label:'Non-linear',strength:'Fast, preserves global structure, deterministic, re-applicable', weakness:'Hyperparameter sensitive (n_neighbors, min_dist)', use:'Visualisation + feature extraction', col:C.purple},
  ];

  var out=sectionTitle('PCA & Dimensionality Reduction',
    'Find the directions of maximum variance \u2014 project to fewer dimensions without losing structure');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','PCA Walkthrough: 4 points, 2D \u2192 1D')
    +svgBox(sv,SW,SH)
    +'<div style="margin-top:10px;display:flex;justify-content:center;gap:8px;">'
    +'<button data-action="pcaPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step>0?0.12:0.04)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
    +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Back</button>'
    +'<button data-action="pcaNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step<3?0.12:0.04)+';border:1.5px solid '+(step<3?C.accent:C.border)+';'
    +'color:'+(step<3?C.accent:C.dim)+';cursor:'+(step<3?'pointer':'default')+';">Next \u2192</button>'
    +'</div>'
    +'<div style="display:flex;gap:6px;justify-content:center;margin-top:8px;">'
    +['Raw','Centre','PC1','Project'].map(function(lbl,i){
      return '<div style="text-align:center;">'
        +'<div style="width:38px;height:6px;border-radius:3px;background:'+(i<=step?stepCols[i]:C.dim)+';margin:0 auto 2px;"></div>'
        +'<div style="font-size:8px;color:'+(i===step?stepCols[i]:C.dim)+';">'+lbl+'</div></div>';}).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','STEP VALUES');
  if(step===0){
    PCA_DATA.forEach(function(p,i){out+=statRow(PCA_LBLS[i],'('+p[0]+', '+p[1]+')',PCA_COLS[i]);});
    out+=statRow('Mean','(4.5, 6.25)',C.yellow);
  } else if(step===1){
    PCA_CENTERED.forEach(function(p,i){out+=statRow(PCA_LBLS[i]+' centred','('+p[0]+', '+p[1]+')',PCA_COLS[i]);});
    out+=statRow('Cov[x1,x1]','5.250',C.accent);
    out+=statRow('Cov[x2,x2]','3.688',C.accent);
    out+=statRow('Cov[x1,x2]','4.375',C.accent);
  } else if(step===2){
    out+=statRow('\u03bb\u2081 (PC1)','8.913 \u2192 99.7%',C.yellow);
    out+=statRow('\u03bb\u2082 (PC2)','0.024 \u2192  0.3%',C.dim);
    out+=statRow('v\u2081 direction','(0.766, 0.643)',C.yellow);
    out+=statRow('Interpretation','Diagonal: bottom-left \u2192 top-right',C.muted);
  } else {
    PCA_PROJ.forEach(function(z,i){out+=statRow(PCA_LBLS[i]+' \u2192 1D',z.toFixed(3),PCA_COLS[i]);});
    out+=statRow('Dim. lost','-1 (2D \u2192 1D)',C.red);
    out+=statRow('Variance kept','99.7%',C.green);
  }
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PCA ALGORITHM');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:1.9;font-family:monospace;',
    '1. <span style="color:'+(step===0?C.yellow:C.dim)+';">Centre: x \u2190 x \u2212 mean(x)</span><br>'
    +'2. <span style="color:'+(step===1?C.blue:C.dim)+';">Covariance: C = (1/N) X\u1d40X</span><br>'
    +'3. <span style="color:'+(step===2?C.yellow:C.dim)+';">Eigendecomp: Cv = \u03bbv</span><br>'
    +'4. <span style="color:'+(step===2?C.purple:C.dim)+';">Sort by \u03bb, select top k</span><br>'
    +'5. <span style="color:'+(step===3?C.accent:C.dim)+';">Project: Z = X_c \u00b7 V_k</span>'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PCA vs t-SNE vs UMAP');
  drData.forEach(function(d){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+d.col+';">'
      +'<div style="display:flex;justify-content:space-between;">'
      +'<span style="font-size:9px;font-weight:700;color:'+d.col+';">'+d.name+'</span>'
      +'<span style="font-size:8px;color:'+C.dim+';">'+d.label+'</span></div>'
      +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+d.strength.slice(0,40)+(d.strength.length>40?'\u2026':'')+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';">Use: '+d.use+'</div></div>';});
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128200;','PC1 Captures 99.7% of All Variance',
    'The four points P1\u2013P4 lie almost perfectly along a single diagonal. '
    +'PCA finds this direction automatically through eigendecomposition of the covariance matrix. '
    +'The first principal component v\u2081=(0.766, 0.643) explains 99.7% of all variance (\u03bb\u2081=8.91 vs \u03bb\u2082=0.02). '
    +'Compressing from 2D to 1D discards only <span style="color:'+C.accent+';font-weight:700;">0.3% of information</span>. '
    +'In practice, PCA is used to remove noise, reduce computation, and reveal the dominant structure in data.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 4 — GENERATIVE MODELS & ANOMALY DETECTION
══════════════════════════════════════════════════════════ */
function renderGenerative(){
  var mode=S.genMode;

  /* ── autoencoder flow SVG ── */
  function autoencoderSVG(){
    var AW=440,AH=220;
    var av='';
    /* layers: input 784 → 256 → 64 → 16 (bottleneck) → 64 → 256 → 784 */
    var layers=[784,256,64,16,64,256,784];
    var lx=[28,88,148,220,292,352,412];
    var maxH=180, baseY=20;
    var cols=[C.blue,C.blue,C.blue,C.accent,C.orange,C.orange,C.orange];
    var lbls=['784','256','64','16','64','256','784'];
    var rects=layers.map(function(l,i){
      var h=Math.max(8,Math.min(maxH,l/784*maxH));
      return {x:lx[i],h:h,y:baseY+(maxH-h)/2};});

    /* arrows */
    for(var i=0;i<rects.length-1;i++){
      var r1=rects[i],r2=rects[i+1];
      av+='<line x1="'+(r1.x+12)+'" y1="'+(baseY+maxH/2)+'" x2="'+(r2.x-1)+'" y2="'+(baseY+maxH/2)+'"'
        +' stroke="'+C.border+'" stroke-width="1.2" opacity="0.6"/>';}

    /* rects */
    rects.forEach(function(r,i){
      av+='<rect x="'+(r.x-10)+'" y="'+r.y+'" width="20" height="'+r.h+'" rx="3"'
        +' fill="'+hex(cols[i],0.7)+'" stroke="'+cols[i]+'" stroke-width="1"/>';
      av+='<text x="'+r.x+'" y="'+(baseY+maxH+14)+'" text-anchor="middle" fill="'+cols[i]+'" font-size="8">'+lbls[i]+'</text>';});

    /* labels */
    av+='<text x="80" y="14" text-anchor="middle" fill="'+C.blue+'" font-size="9" font-weight="700">ENCODER</text>';
    av+='<text x="362" y="14" text-anchor="middle" fill="'+C.orange+'" font-size="9" font-weight="700">DECODER</text>';
    av+='<text x="220" y="14" text-anchor="middle" fill="'+C.accent+'" font-size="9" font-weight="700">BOTTLENECK z</text>';
    /* bottleneck highlight */
    av+='<rect x="205" y="'+(baseY-4)+'" width="30" height="'+(maxH+8)+'" rx="4" fill="none" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="5,3"/>';

    /* x and xhat labels */
    av+='<text x="28" y="'+(baseY+maxH/2+4)+'" text-anchor="middle" fill="'+C.blue+'" font-size="10" font-weight="700">x</text>';
    av+='<text x="412" y="'+(baseY+maxH/2+4)+'" text-anchor="middle" fill="'+C.orange+'" font-size="10" font-weight="700">x&#770;</text>';

    /* loss bar */
    var normalLoss=0.06, anomLoss=0.78;
    var isAnom=S.anomalyPoint===1;
    var displayLoss=isAnom?anomLoss:normalLoss;
    var lossBarW=displayLoss*(AW-60);
    av+='<rect x="30" y="'+(baseY+maxH+28)+'" width="'+(AW-60)+'" height="12" rx="3" fill="'+C.border+'"/>';
    av+='<rect x="30" y="'+(baseY+maxH+28)+'" width="'+lossBarW.toFixed(0)+'" height="12" rx="3" fill="'+hex(isAnom?C.red:C.green,0.8)+'"/>';
    av+='<text x="30" y="'+(baseY+maxH+50)+'" fill="'+C.muted+'" font-size="8.5">Reconstruction error: </text>';
    av+='<text x="170" y="'+(baseY+maxH+50)+'" fill="'+(isAnom?C.red:C.green)+'" font-size="8.5" font-weight="700">'+(displayLoss*100).toFixed(0)+'% \u2014 '+(isAnom?'ANOMALY DETECTED':'Normal')+'</text>';
    return svgBox(av,AW,AH);}

  /* ── GMM soft assignment SVG ── */
  function gmmSVG(){
    var GW=440, GH=220;
    var gv='';
    /* show two Gaussian blobs with soft colouring */
    var pts=[[2,5],[2.8,5.5],[1.8,4.8],[3,5.2],[3.2,4.5],
              [7,5],[7.5,5.8],[6.8,4.7],[7.2,5.5],[6.5,5.2],
              [4.5,5],[5,5.2]]; /* ambiguous points in middle */
    var ax=mkAxes('',44,16,18,28,GW,GH,'x\u2081','x\u2082',0,10,0,10,
      [0,2,4,6,8,10],[0,2,4,6,8,10]);
    gv+=ax.svg;

    /* draw Gaussian contours */
    function gaussContour(cx,cy,sx,sy,col,r){
      var cx2=ax.fx(cx),cy2=ax.fy(cy);
      var rw=ax.PW/(10)*r*sx, rh=ax.PH/(10)*r*sy;
      gv+='<ellipse cx="'+cx2.toFixed(1)+'" cy="'+cy2.toFixed(1)+'" rx="'+rw.toFixed(1)+'" ry="'+rh.toFixed(1)+'"'
        +' fill="'+hex(col,0.07)+'" stroke="'+hex(col,0.25)+'" stroke-width="1.2"/>';}
    gaussContour(2.7,5,1.8,1.2,C.blue,1);
    gaussContour(2.7,5,1.8,1.2,C.blue,1.8);
    gaussContour(7,5.1,1.8,1.2,C.orange,1);
    gaussContour(7,5.1,1.8,1.2,C.orange,1.8);

    pts.forEach(function(p,i){
      /* soft assignment: how close to each cluster centre */
      var d1=dist(p,[2.7,5]), d2=dist(p,[7,5.1]);
      var p1=Math.exp(-d1)/( Math.exp(-d1)+Math.exp(-d2));
      /* blend colour */
      var r1=parseInt(C.blue.slice(1,3),16),g1=parseInt(C.blue.slice(3,5),16),b1=parseInt(C.blue.slice(5,7),16);
      var r2=parseInt(C.orange.slice(1,3),16),g2=parseInt(C.orange.slice(3,5),16),b2=parseInt(C.orange.slice(5,7),16);
      var rb=Math.round(r1*p1+r2*(1-p1)).toString(16).padStart(2,'0');
      var gb=Math.round(g1*p1+g2*(1-p1)).toString(16).padStart(2,'0');
      var bb=Math.round(b1*p1+b2*(1-p1)).toString(16).padStart(2,'0');
      var blendCol='#'+rb+gb+bb;
      gv+='<circle cx="'+ax.fx(p[0]).toFixed(1)+'" cy="'+ax.fy(p[1]).toFixed(1)+'" r="7"'
        +' fill="'+blendCol+'" stroke="none" opacity="0.85"/>';
      /* show probability for ambiguous points */
      if(Math.abs(p1-0.5)<0.18){
        gv+='<text x="'+ax.fx(p[0]).toFixed(1)+'" y="'+(ax.fy(p[1])-9).toFixed(1)+'" text-anchor="middle" fill="'+C.text+'" font-size="7.5">'
          +(p1*100).toFixed(0)+'%</text>';}});

    /* centroids */
    [[2.7,5,C.blue,'&#956;\u2081'],[7,5.1,C.orange,'&#956;\u2082']].forEach(function(c){
      gv+='<circle cx="'+ax.fx(c[0]).toFixed(1)+'" cy="'+ax.fy(c[1]).toFixed(1)+'" r="9"'
        +' fill="'+hex(c[2],0.9)+'" stroke="#0a0a0f" stroke-width="2"/>';
      gv+='<text x="'+ax.fx(c[0]).toFixed(1)+'" y="'+(ax.fy(c[1])+3.5).toFixed(1)+'" text-anchor="middle" fill="#0a0a0f" font-size="9" font-weight="700">'+c[3]+'</text>';});
    gv+='<text x="220" y="'+(GH-8)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5">Point colour = blended soft probability P(cluster|x)</text>';
    return svgBox(gv,GW,GH);}

  /* ── Isolation Forest SVG ── */
  function isolationForestSVG(){
    var IW=440,IH=220;
    var iv='';
    var ax=mkAxes('',44,20,14,28,IW,IH,'x\u2081','x\u2082',0,10,0,10,
      [0,2,4,6,8,10],[0,2,4,6,8,10]);
    iv+=ax.svg;

    /* normal cluster + 2 anomalies */
    var normalPts=[[3,5],[3.5,5.5],[2.8,4.8],[4,5.2],[3.2,4.5],[3.8,4.2],[2.5,5.8],[4.2,5.8]];
    var anomPts=[[8.5,9],[1,1]];

    /* illustrative isolation split lines */
    /* for anomaly at (8.5, 9) - few splits needed */
    iv+='<line x1="'+ax.fx(7)+'" y1="'+ax.fy(0)+'" x2="'+ax.fx(7)+'" y2="'+ax.fy(10)+'" stroke="'+hex(C.red,0.4)+'" stroke-width="1" stroke-dasharray="4,3"/>';
    iv+='<line x1="'+ax.fx(7)+'" y1="'+ax.fy(7)+'" x2="'+ax.fx(10)+'" y2="'+ax.fy(7)+'" stroke="'+hex(C.red,0.4)+'" stroke-width="1" stroke-dasharray="4,3"/>';
    iv+='<text x="'+ax.fx(8.5)+'" y="'+ax.fy(7.8)+'" text-anchor="middle" fill="'+C.red+'" font-size="8">2 splits!</text>';

    /* many splits for normal region */
    iv+='<line x1="'+ax.fx(2)+'" y1="'+ax.fy(0)+'" x2="'+ax.fx(2)+'" y2="'+ax.fy(10)+'" stroke="'+hex(C.blue,0.25)+'" stroke-width="0.8" stroke-dasharray="3,3"/>';
    iv+='<line x1="'+ax.fx(5)+'" y1="'+ax.fy(0)+'" x2="'+ax.fx(5)+'" y2="'+ax.fy(10)+'" stroke="'+hex(C.blue,0.25)+'" stroke-width="0.8" stroke-dasharray="3,3"/>';
    iv+='<line x1="'+ax.fx(2)+'" y1="'+ax.fy(4)+'" x2="'+ax.fx(5)+'" y2="'+ax.fy(4)+'" stroke="'+hex(C.blue,0.25)+'" stroke-width="0.8" stroke-dasharray="3,3"/>';
    iv+='<line x1="'+ax.fx(2)+'" y1="'+ax.fy(6.5)+'" x2="'+ax.fx(5)+'" y2="'+ax.fy(6.5)+'" stroke="'+hex(C.blue,0.25)+'" stroke-width="0.8" stroke-dasharray="3,3"/>';
    iv+='<text x="'+ax.fx(3.5)+'" y="'+ax.fy(3.6)+'" text-anchor="middle" fill="'+C.blue+'" font-size="8">many splits</text>';

    normalPts.forEach(function(p){
      iv+='<circle cx="'+ax.fx(p[0]).toFixed(1)+'" cy="'+ax.fy(p[1]).toFixed(1)+'" r="6"'
        +' fill="'+hex(C.blue,0.7)+'" stroke="'+C.blue+'" stroke-width="1"/>';});
    anomPts.forEach(function(p){
      iv+='<circle cx="'+ax.fx(p[0]).toFixed(1)+'" cy="'+ax.fy(p[1]).toFixed(1)+'" r="8"'
        +' fill="'+hex(C.red,0.8)+'" stroke="'+C.red+'" stroke-width="2"/>';
      iv+='<text x="'+ax.fx(p[0]).toFixed(1)+'" y="'+(ax.fy(p[1])-11).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="8.5" font-weight="700">ANOMALY</text>';});
    return svgBox(iv,IW,IH);}

  var out=sectionTitle('Generative Models & Anomaly Detection',
    'Learn the underlying data distribution \u2014 then generate, compress, or detect the unusual');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ['&#128257; Autoencoder','&#127922; GMM','&#128683; Anomaly Detect.'].forEach(function(lbl,i){
    out+=btnSel(i,mode,[C.blue,C.orange,C.red][i],lbl,'genMode');});
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';

  if(mode===0){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Autoencoder: 784\u2192256\u219264\u219216\u219264\u2192256\u2192784')
      +autoencoderSVG()
      +'<div style="margin-top:10px;display:flex;gap:8px;justify-content:center;">'
      +btnSel(0,S.anomalyPoint,C.green,'Normal input','anomalyPoint')
      +btnSel(1,S.anomalyPoint,C.red,'Anomalous input','anomalyPoint')
      +'</div>'
    );
  } else if(mode===1){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','GMM: Soft Probabilistic Cluster Assignments')
      +gmmSVG()
    );
  } else {
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Isolation Forest: Anomalies are Isolated Faster')
      +isolationForestSVG()
    );
  }
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  if(mode===0){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','AUTOENCODER vs VAE');
    [
      {n:'Autoencoder',  d:'Encodes to a point z. Compresses, reconstructs, detects anomalies.', c:C.blue,  extra:'Cannot generate new data'},
      {n:'VAE',          d:'Encodes to a distribution z~N(\u03bc,\u03c3). Smooth latent space. Can sample and generate.', c:C.purple, extra:'ELBO = recon loss + KL divergence'},
    ].forEach(function(r){
      out+='<div style="padding:6px 10px;margin:3px 0;border-radius:6px;border-left:3px solid '+r.c+';">'
        +'<div style="font-size:9.5px;font-weight:700;color:'+r.c+';margin-bottom:3px;">'+r.n+'</div>'
        +'<div style="font-size:8.5px;color:'+C.muted+';">'+r.d+'</div>'
        +'<div style="font-size:8px;color:'+C.dim+';margin-top:2px;">'+r.extra+'</div></div>';});
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHAT THE BOTTLENECK DOES');
    [
      {t:'Compression',       d:'Forces network to keep only essential features'},
      {t:'Feature extraction',d:'z is a learned repr. for downstream tasks'},
      {t:'Anomaly detection', d:'Unusual input \u2192 high recon error \u2192 flag it'},
    ].forEach(function(r){out+=statRow(r.t,r.d,C.accent);});
    out+='</div>';
  } else if(mode===1){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GMM vs K-MEANS');
    [
      {feat:'Assignments',    km:'Hard (binary)',          gmm:'Soft probabilities',         gmg:true},
      {feat:'Cluster shape',  km:'Spherical',              gmm:'Elliptical (full covariance)',gmg:true},
      {feat:'Model type',     km:'Geometric',              gmm:'Probabilistic',               gmg:true},
      {feat:'Likelihood',     km:'Not available',          gmm:'Exact log-likelihood',        gmg:true},
      {feat:'Training',       km:'Assignment + update',    gmm:'EM algorithm',                gmg:false},
      {feat:'K required',     km:'Yes',                    gmm:'Yes',                         gmg:false},
    ].forEach(function(r){
      out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
        +'<div style="flex:1;color:'+C.dim+';">'+r.feat+'</div>'
        +'<div style="flex:1;color:'+C.red+';">\u2716 '+r.km+'</div>'
        +'<div style="flex:1;color:'+(r.gmg?C.green:C.yellow)+';">'+(r.gmg?'\u2714':'\u2248')+' '+r.gmm+'</div></div>';});
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','EM ALGORITHM');
    out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:1.9;font-family:monospace;',
      '<span style="color:'+C.blue+';">E-step</span>: compute responsibility<br>'
      +'  r\u1d62\u2096 = \u03c0\u2096 N(x\u1d62|\u03bc\u2096,\u03a3\u2096) / \u03a3\u2c7c \u03c0\u2c7c N(x\u1d62|\u03bc\u2c7c,\u03a3\u2c7c)<br>'
      +'<span style="color:'+C.orange+';">M-step</span>: update parameters<br>'
      +'  \u03bc\u2096 \u2190 \u03a3 r\u1d62\u2096 x\u1d62 / \u03a3 r\u1d62\u2096<br>'
      +'Repeat until log-likelihood converges.'
    );
    out+='</div>';
  } else {
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','ANOMALY DETECTION METHODS');
    [
      {n:'Isolation Forest', d:'Random splits isolate anomalies faster (fewer splits in sparse regions)', c:C.red},
      {n:'One-Class SVM',    d:'Learns a boundary around normal data; points outside are anomalies', c:C.orange},
      {n:'Autoencoder',      d:'Train on normal data; high recon error = anomalous at inference', c:C.blue},
      {n:'Z-score / IQR',    d:'Statistical: flag points > N std. devs from mean. Simple, fast.', c:C.dim},
      {n:'LOF',              d:'Local Outlier Factor: compares density to neighbourhood density', c:C.purple},
    ].forEach(function(r){
      out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
        +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.n+'</div>'
        +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.d+'</div></div>';});
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY UNSUPERVISED?');
    out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
      'Anomalies are <span style="color:'+C.red+';font-weight:700;">rare, novel, and unlabeled</span>. '
      +'In fraud detection, a new attack pattern has never been seen before. '
      +'In industrial monitoring, a novel failure mode has no historical labels. '
      +'You can\'t use supervised learning when you have no examples of what you\'re trying to find.'
    );
    out+='</div>';
  }
  out+='</div></div>';

  out+=insight('&#128257;','Autoencoders: Compression Reveals Anomalies',
    'Train the autoencoder on <span style="color:'+C.green+';font-weight:700;">normal data only</span>. '
    +'It learns to encode and decode normal patterns efficiently. '
    +'At inference, a normal input reconstructs cleanly (low MSE). '
    +'An anomalous input \u2014 with structure the autoencoder has never seen \u2014 '
    +'<span style="color:'+C.red+';font-weight:700;">reconstructs poorly (high MSE)</span>. '
    +'The reconstruction error becomes the anomaly score. '
    +'Toggle "Normal / Anomalous input" to see the reconstruction error bar flip from green to red. '
    +'The GMM tab shows how soft probability assignments blend at cluster boundaries.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════════ */
var TABS=[
  '&#128269; Core Idea',
  '&#9711; K-Means',
  '&#128209; DBSCAN',
  '&#128200; PCA',
  '&#127922; Generative & Anomaly'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.blue+','+C.purple+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Unsupervised Learning</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Discover hidden structure in unlabeled data \u2014 clustering, dimensionality reduction, and generative models')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';});
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderCoreIdea();
  else if(S.tab===1) html+=renderKMeans();
  else if(S.tab===2) html+=renderDBSCAN();
  else if(S.tab===3) html+=renderPCA();
  else if(S.tab===4) html+=renderGenerative();
  html+='</div></div>';
  return html;}

function render(){
  document.getElementById('app').innerHTML=renderApp();
  bindEvents();}

function bindEvents(){
  document.querySelectorAll('[data-action]').forEach(function(el){
    var action=el.getAttribute('data-action');
    var idx=parseInt(el.getAttribute('data-idx'));
    var tag=el.tagName.toLowerCase();
    if(tag==='button'){
      el.addEventListener('click',function(){
        if(action==='tab')          {S.tab=idx;              render();}
        else if(action==='goalTab') {S.goalTab=idx;          render();}
        else if(action==='toggleLabels'){S.showLabels=!S.showLabels; render();}
        else if(action==='kmNext')  {if(S.kmStep<4){S.kmStep++;render();}}
        else if(action==='kmPrev')  {if(S.kmStep>0){S.kmStep--;render();}}
        else if(action==='dbMode')  {S.dbMode=idx;           render();}
        else if(action==='pcaNext') {if(S.pcaStep<3){S.pcaStep++;render();}}
        else if(action==='pcaPrev') {if(S.pcaStep>0){S.pcaStep--;render();}}
        else if(action==='drMode')  {S.drMode=idx;           render();}
        else if(action==='genMode') {S.genMode=idx;          render();}
        else if(action==='anomalyPoint'){S.anomalyPoint=idx; render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='elbowK')  {S.elbowK=Math.round(val);  render();}
        else if(action==='dbEps'){S.dbEps=val;              render();}
      });
    }
  });}

render();
</script>
</body>
</html>"""

UL_VISUAL_HEIGHT = 1100